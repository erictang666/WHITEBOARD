
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from scorer_hyperparameters import get_scorer_hyperparameter
from word_norms2_norms import WordNorms2Norms
from typed_axis_aggregation import (
    build_propconj_item_subtype_contributions,
    mean_subtype_contributions,
)
from t1_assoc_v3 import (
    T1_ASSOC_VERSION,
    effective_diversity,
    get_component_params,
    propconj_item_quality,
    top_mean,
    transform_common_answer_rarity,
)


DATA_DIR = Path(__file__).resolve().parent / "data"

_FANTASY_OR_IMPOSSIBLE_MARKERS_FALLBACK = [
    "imaginary",
    "fictional",
    "made up",
    "mythical",
    "magic",
    "magical",
    "dragon",
    "unicorn",
    "miniature sun",
    "pocket sun",
    "tiny sun",
    "micro sun",
    "personal star",
    "alien",
    "teleport",
]
FANTASY_OR_IMPOSSIBLE_MARKERS = set(get_scorer_hyperparameter(
    "propconj",
    "FANTASY_OR_IMPOSSIBLE_MARKERS",
    default=_FANTASY_OR_IMPOSSIBLE_MARKERS_FALLBACK,
))
FANTASY_MARKER_SIGMOID_PARAMS = get_scorer_hyperparameter(
    "propconj",
    "fantasy_marker_sigmoid",
    default={
        "midpoint": 0.50,
        "temperature": 0.22,
        "exact_hit_score": 0.95,
        "token_hit_score": 0.82,
        "substring_hit_score": 0.70,
        "cap": 1.0,
    },
)


def clip01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    return sum(filtered) / len(filtered) if filtered else None


def _normalize_text(text: str) -> str:
    text = (text or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> List[str]:
    return [token for token in _normalize_text(text).split() if token]


def _sigmoid_soft_score(raw_score: float, params: Dict[str, object]) -> float:
    raw_score = clip01(raw_score)
    if raw_score <= 0.0:
        return 0.0
    midpoint = float(params.get("midpoint", 0.50) or 0.50)
    temperature = max(1e-6, float(params.get("temperature", 0.22) or 0.22))
    cap = clip01(params.get("cap", 1.0))
    return clip01(cap * (1.0 / (1.0 + math.exp(-(raw_score - midpoint) / temperature))))


def _fantasy_marker_record(item_text: str) -> Dict[str, object]:
    item_norm = _normalize_text(item_text)
    item_tokens = set(item_norm.split())
    best_raw = 0.0
    hits: List[Dict[str, object]] = []
    for marker in sorted(FANTASY_OR_IMPOSSIBLE_MARKERS):
        marker_norm = _normalize_text(str(marker))
        if not marker_norm:
            continue
        marker_tokens = set(marker_norm.split())
        mode = None
        raw = 0.0
        if item_norm == marker_norm:
            mode = "exact"
            raw = float(FANTASY_MARKER_SIGMOID_PARAMS.get("exact_hit_score", 0.95) or 0.95)
        elif marker_tokens and marker_tokens.issubset(item_tokens):
            mode = "token"
            raw = float(FANTASY_MARKER_SIGMOID_PARAMS.get("token_hit_score", 0.82) or 0.82)
        elif marker_norm in item_norm:
            mode = "substring"
            raw = float(FANTASY_MARKER_SIGMOID_PARAMS.get("substring_hit_score", 0.70) or 0.70)
        if mode is None:
            continue
        best_raw = max(best_raw, raw)
        hits.append({
            "marker": marker,
            "mode": mode,
            "raw_score": round(clip01(raw), 4),
        })
    score = _sigmoid_soft_score(best_raw, FANTASY_MARKER_SIGMOID_PARAMS)
    return {
        "score": round(score, 4),
        "raw_score": round(clip01(best_raw), 4),
        "hits": hits,
        "transform": "sigmoid_temperature",
    }


def _keyword_hit(text: str, keywords: Sequence[str], strength: float) -> Tuple[float, List[str]]:
    normalized = f" {_normalize_text(text)} "
    token_set = set(normalized.split())
    hits: List[str] = []
    for keyword in keywords or []:
        keyword_norm = _normalize_text(str(keyword))
        if not keyword_norm:
            continue
        keyword_tokens = keyword_norm.split()
        if f" {keyword_norm} " in normalized or (keyword_tokens and set(keyword_tokens).issubset(token_set)):
            hits.append(str(keyword))
    return (float(strength) if hits else 0.0), sorted(set(hits))


def _weighted_geomean(scores: Sequence[float], weights: Sequence[float]) -> float:
    usable = [
        (clip01(score), max(0.0, float(weight)))
        for score, weight in zip(scores, weights)
        if weight is not None and float(weight) > 0.0
    ]
    if not usable:
        return 0.0
    total_weight = sum(weight for _, weight in usable)
    if total_weight <= 0.0:
        return 0.0
    eps = 1e-6
    return clip01(math.exp(sum(weight * math.log(max(score, eps)) for score, weight in usable) / total_weight))


def _max_predicate_strength(source_map: Dict[str, float], predicates: Sequence[str]) -> Tuple[float, Optional[str]]:
    best_strength = 0.0
    best_predicate = None
    for predicate in predicates or []:
        try:
            strength = float(source_map.get(predicate, 0.0))
        except Exception:
            strength = 0.0
        if strength > best_strength:
            best_strength = strength
            best_predicate = predicate
    return best_strength, best_predicate


def _extract_evidence_for_property(evidence_map: Dict[str, object], prop: Dict[str, object]) -> str:
    if not isinstance(evidence_map, dict):
        return ""
    candidates = {
        _normalize_text(str(prop.get("id") or "")),
        _normalize_text(str(prop.get("label") or "")),
    }
    for key, value in evidence_map.items():
        key_norm = _normalize_text(str(key))
        if key_norm in candidates:
            return str(value or "")
    return ""


class PropConjScorer:
    """Scores PropConj items with frozen predicates, lexicons, and evidence fields."""

    def __init__(self, word_norms2: Optional[WordNorms2Norms] = None, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.word_norms2 = word_norms2 or WordNorms2Norms(data_dir=str(self.data_dir))

    def _concept_features(self, item_text: str) -> Tuple[Dict[str, float], Dict[str, object]]:
        if not self.word_norms2 or not getattr(self.word_norms2, "available", False):
            return {}, {"concept": None, "matched_alias": None, "confidence": 0.0}
        match = self.word_norms2.match_concept(item_text)
        features = self.word_norms2.get_concept_features(item_text)
        return features, {
            "concept": match.concept,
            "matched_alias": match.matched_alias,
            "confidence": round(float(match.confidence), 4),
        }

    def _score_property(
        self,
        prop: Dict[str, object],
        *,
        item_text: str,
        evidence_text: str,
        concept_features: Dict[str, float],
    ) -> Dict[str, object]:
        positive_predicates = list(prop.get("positive_predicates") or [])
        negative_predicates = list(prop.get("negative_predicates") or [])
        positive_keywords = list(prop.get("positive_keywords") or [])
        negative_keywords = list(prop.get("negative_keywords") or [])
        evidence_keywords = list(prop.get("evidence_keywords") or []) + positive_keywords

        pos_feature, pos_predicate = _max_predicate_strength(concept_features, positive_predicates)
        neg_feature, neg_predicate = _max_predicate_strength(concept_features, negative_predicates)
        pos_item, pos_item_hits = _keyword_hit(item_text, positive_keywords, 0.92)
        neg_item, neg_item_hits = _keyword_hit(item_text, negative_keywords, 1.00)
        pos_evidence, pos_evidence_hits = _keyword_hit(evidence_text, evidence_keywords, 0.72)
        neg_evidence, neg_evidence_hits = _keyword_hit(evidence_text, negative_keywords, 0.90)

        positive_strength = max(pos_feature, pos_item, pos_evidence)
        negative_strength = max(neg_feature, neg_item, neg_evidence)
        support = clip01(positive_strength - 0.85 * negative_strength)

        if not evidence_text:
            evidence_status = "missing"
        elif support < 0.35 or negative_strength >= 0.55:
            evidence_status = "mismatch"
        else:
            evidence_status = "consistent"

        return {
            "id": prop.get("id"),
            "label": prop.get("label"),
            "weight": float(prop.get("weight", 1.0) or 1.0),
            "support": round(support, 4),
            "positive_strength": round(positive_strength, 4),
            "negative_strength": round(negative_strength, 4),
            "positive_predicate": pos_predicate,
            "negative_predicate": neg_predicate,
            "positive_item_hits": pos_item_hits,
            "negative_item_hits": neg_item_hits,
            "positive_evidence_hits": pos_evidence_hits,
            "negative_evidence_hits": neg_evidence_hits,
            "evidence_text": evidence_text or None,
            "evidence_status": evidence_status,
        }

    def _unresolved_entity_score(
        self,
        item_text: str,
        *,
        concept_match: Dict[str, object],
        property_records: Sequence[Dict[str, object]],
        fantasy_marker_score: float = 0.0,
    ) -> float:
        fantasy_marker_score = clip01(fantasy_marker_score)
        if fantasy_marker_score > 0.0:
            return max(0.65, fantasy_marker_score)
        if concept_match.get("concept"):
            return 0.0
        if any(float(record.get("positive_strength", 0.0)) >= 0.72 for record in property_records):
            return 0.0
        if len(_tokens(item_text)) <= 3 and any(float(record.get("support", 0.0)) >= 0.55 for record in property_records):
            return 0.15
        return 0.65

    @staticmethod
    def _novelty_score(novelty_score=None, legacy_score=None, bank_score=None) -> Dict[str, float]:
        legacy = clip01(legacy_score if legacy_score is not None else novelty_score if novelty_score is not None else 0.0)
        bank = clip01(bank_score if bank_score is not None else novelty_score if novelty_score is not None else legacy)
        frequency_rarity = bank if bank_score is not None else legacy
        novelty = clip01(0.60 * bank + 0.40 * frequency_rarity)
        return {
            "novelty": round(novelty, 4),
            "embedding_distance_to_common_bank": round(bank, 4),
            "frequency_rarity": round(frequency_rarity, 4),
        }

    def score_item(
        self,
        *,
        task_metadata: Dict[str, object],
        parsed_item: Optional[Dict[str, object]] = None,
        item_text: Optional[str] = None,
        novelty_score=None,
        legacy_score=None,
        bank_score=None,
    ) -> Dict[str, object]:
        parsed_item = parsed_item or {}
        item_text = (
            item_text
            or parsed_item.get("propconj_item")
            or parsed_item.get("noun_phrase")
            or parsed_item.get("display_text")
            or ""
        )
        evidence_map = parsed_item.get("evidence_for_each_property") or {}
        properties = list(task_metadata.get("properties") or [])

        concept_features, concept_match = self._concept_features(item_text)
        property_records = []
        for prop in properties:
            evidence_text = _extract_evidence_for_property(evidence_map, prop)
            property_records.append(self._score_property(
                prop,
                item_text=item_text,
                evidence_text=evidence_text,
                concept_features=concept_features,
            ))

        supports = [float(record.get("support", 0.0)) for record in property_records]
        weights = [float(record.get("weight", 1.0) or 1.0) for record in property_records]
        conjunction_geomean = _weighted_geomean(supports, weights)
        conjunction_soft = max(0.15 * mean_or_none(supports), conjunction_geomean)
        min_property_support = min(supports) if supports else 0.0
        covered_property_count = sum(1 for support in supports if support >= 0.55)
        intent_coverage = covered_property_count / len(supports) if supports else 0.0
        fantasy_marker = _fantasy_marker_record(item_text)
        unresolved = self._unresolved_entity_score(
            item_text,
            concept_match=concept_match,
            property_records=property_records,
            fantasy_marker_score=float(fantasy_marker.get("score", 0.0) or 0.0),
        )
        contradiction = clip01(max((float(record.get("negative_strength", 0.0)) for record in property_records), default=0.0))

        if property_records:
            mismatch_mass = 0.0
            for record in property_records:
                if record.get("evidence_status") == "missing":
                    mismatch_mass += 0.50
                elif record.get("evidence_status") == "mismatch":
                    mismatch_mass += 1.00
            evidence_mismatch = clip01(mismatch_mass / len(property_records))
        else:
            evidence_mismatch = 1.0

        grounding = clip01(
            0.45 * conjunction_soft +
            0.20 * min_property_support +
            0.20 * (1.0 - unresolved) +
            0.15 * intent_coverage
        )
        grounding_for_i = grounding
        grounding_for_h = clip01(0.70 * contradiction + 0.30 * evidence_mismatch)
        appropriateness_gate = clip01(
            0.35 * conjunction_soft +
            0.25 * min_property_support +
            0.25 * intent_coverage +
            0.15 * (1.0 - evidence_mismatch)
        )
        appropriateness_gate = clip01(
            appropriateness_gate *
            (1.0 - 0.25 * contradiction - 0.20 * unresolved)
        )
        if intent_coverage < 1.0:
            appropriateness_gate = min(appropriateness_gate, clip01(0.10 + 0.75 * intent_coverage))
        if unresolved >= 0.85:
            appropriateness_gate = min(appropriateness_gate, clip01(0.25 + 0.25 * (1.0 - unresolved)))
        if contradiction >= 0.85:
            appropriateness_gate = min(appropriateness_gate, 0.25)

        hallucination = clip01(
            0.50 * (1.0 - grounding) +
            0.20 * contradiction +
            0.10 * unresolved +
            0.10 * evidence_mismatch +
            0.10 * (1.0 - intent_coverage)
        )
        fantasy_score = float(fantasy_marker.get("score", 0.0) or 0.0)
        if fantasy_score > 0.0:
            hallucination = max(hallucination, clip01(0.55 + 0.32 * fantasy_score))
        elif unresolved >= 0.95:
            hallucination = max(hallucination, 0.85)
        elif contradiction >= 0.85 or evidence_mismatch >= 0.95:
            hallucination = max(hallucination, 0.70)
        novelty_parts = self._novelty_score(
            novelty_score=novelty_score,
            legacy_score=legacy_score,
            bank_score=bank_score,
        )
        rarity_v3 = transform_common_answer_rarity(
            novelty_parts.get("embedding_distance_to_common_bank", novelty_parts["novelty"])
        )
        imagination_contribution = propconj_item_quality(
            rarity=rarity_v3,
            grounding=grounding,
            gate=appropriateness_gate,
            intent_coverage=intent_coverage,
        )
        valid = grounding >= 0.70 and hallucination <= 0.30 and intent_coverage >= 0.999
        soft_valid = bool(grounding >= 0.55 and hallucination <= 0.45 and intent_coverage >= 0.60)
        hard_valid = bool(
            valid and
            min_property_support >= 0.55 and
            contradiction <= 0.20 and
            evidence_mismatch <= 0.20 and
            unresolved <= 0.20
        )
        conjunction_difficulty_bonus = 0.15 if (len(property_records) >= 4 and hard_valid) else 0.0

        result = {
            "version": "propconj_item_v2",
            "t1_assoc_version": T1_ASSOC_VERSION,
            "item": item_text,
            "property_count": len(property_records),
            "covered_property_count": covered_property_count,
            "intent_coverage": round(intent_coverage, 4),
            "property_support": property_records,
            "conjunction_geomean": round(conjunction_geomean, 4),
            "conjunction_soft": round(conjunction_soft, 4),
            "min_property_support": round(min_property_support, 4),
            "appropriateness_gate": round(appropriateness_gate, 4),
            "grounding": round(grounding, 4),
            "grounding_for_I": round(grounding_for_i, 4),
            "grounding_for_H": round(grounding_for_h, 4),
            "unresolved_entity": round(unresolved, 4),
            "fantasy_marker_score": fantasy_marker["score"],
            "fantasy_marker_raw_score": fantasy_marker["raw_score"],
            "fantasy_marker_hits": fantasy_marker["hits"],
            "contradiction": round(contradiction, 4),
            "evidence_mismatch": round(evidence_mismatch, 4),
            "hallucination": round(hallucination, 4),
            "novelty": novelty_parts["novelty"],
            "rarity_v3": round(rarity_v3, 4),
            "embedding_distance_to_common_bank": novelty_parts["embedding_distance_to_common_bank"],
            "frequency_rarity": novelty_parts["frequency_rarity"],
            "imagination_contribution": round(imagination_contribution, 4),
            "imagination_contribution_v3": round(imagination_contribution, 4),
            "valid": bool(valid),
            "soft_valid": soft_valid,
            "hard_valid": hard_valid,
            "conjunction_difficulty_bonus": round(conjunction_difficulty_bonus, 4),
            "word_norms2_match": concept_match,
            "formula": {
                "intent_coverage": "covered_required_properties / total_required_properties, support>=0.55",
                "appropriateness_gate": "clip(0.35*A+0.25*M+0.25*K+0.15*(1-E)) with contradiction/unresolved discount",
                "grounding": "G=clip(0.45*max(gmean,0.15*mean_props)+0.20*M_i+0.20*(1-u_i)+0.15*K_i)",
                "grounding_for_I": "property-bank conjunction validity used only by I_assoc",
                "grounding_for_H": "soft contradiction/evidence-mismatch burden used by H_context after atom decoupling",
                "hallucination": "H=clip(0.50*(1-G)+0.20*C+0.10*u+0.10*E+0.10*(1-K))",
                "imagination_contribution": "T1 I_i=rarity^1.3*grounding^1.4*appropriateness_gate*intent_coverage",
            },
        }
        result["subtype_contributions"] = build_propconj_item_subtype_contributions(result)
        result["atom_signals"] = result["subtype_contributions"].get("atom_signals", {})
        return result


def compute_propconj_diversity(embedding_flexibility=None, ontological_flexibility=None) -> float:
    embedding_flexibility = embedding_flexibility or {}
    ontological_flexibility = ontological_flexibility or {}
    return clip01(
        0.50 * float(embedding_flexibility.get("mean_pairwise_distance") or 0.0) +
        0.20 * float(embedding_flexibility.get("cluster_entropy") or 0.0) +
        0.30 * float(ontological_flexibility.get("category_diversity_index") or 0.0)
    )


def compute_propconj_task_scores(
    task_details: Sequence[Dict[str, object]],
    diversity_score: float,
    *,
    expected_output_count: int = 12,
    beta_ih: float = 0.20,
    beta_hi: float = 0.10,
) -> Optional[Dict[str, object]]:
    item_scores = [
        detail.get("propconj_scores")
        for detail in task_details
        if isinstance(detail.get("propconj_scores"), dict)
    ]
    if not item_scores:
        return None

    contributions = sorted(
        [float(item.get("imagination_contribution_v3", item.get("imagination_contribution", 0.0)) or 0.0) for item in item_scores],
        reverse=True,
    )
    quality_mass_top6 = top_mean(contributions, 6)
    elite_tail_top3 = top_mean(contributions, 3)
    valid_count = sum(1 for item in item_scores if item.get("valid"))
    soft_valid_count = sum(1 for item in item_scores if item.get("soft_valid"))
    hard_valid_count = sum(1 for item in item_scores if item.get("hard_valid"))
    valid_ratio = min(1.0, valid_count / max(1, expected_output_count))
    soft_valid_ratio = min(1.0, soft_valid_count / max(1, expected_output_count))
    hard_valid_ratio = min(1.0, hard_valid_count / max(1, expected_output_count))
    hallucination_raw = mean_or_none([float(item.get("hallucination", 0.0) or 0.0) for item in item_scores]) or 0.0
    diversity_score = clip01(diversity_score)
    diversity_eff = effective_diversity(diversity_score, hard_valid_ratio)
    params = get_component_params("PropConj")
    weights = params.get("task_weights", {}) if isinstance(params.get("task_weights"), dict) else {}
    conjunction_difficulty_bonus = min(
        float(params.get("conjunction_difficulty_bonus", 0.15)),
        mean_or_none([float(item.get("conjunction_difficulty_bonus", 0.0) or 0.0) for item in item_scores]) or 0.0,
    )
    imagination_raw = clip01(
        float(weights.get("quality_mass_top6", 0.40)) * quality_mass_top6 +
        float(weights.get("elite_tail_top3", 0.30)) * elite_tail_top3 +
        float(weights.get("diversity_eff", 0.08)) * diversity_eff +
        float(weights.get("hard_valid_ratio", 0.10)) * hard_valid_ratio +
        float(weights.get("soft_valid_ratio", 0.12)) * soft_valid_ratio +
        conjunction_difficulty_bonus
    )
    imagination = clip01(imagination_raw - beta_ih * max(0.0, hallucination_raw))
    hallucination = clip01(hallucination_raw - beta_hi * max(0.0, imagination_raw))

    primitive_fields = [
        "grounding",
        "grounding_for_I",
        "grounding_for_H",
        "conjunction_geomean",
        "conjunction_soft",
        "min_property_support",
        "intent_coverage",
        "appropriateness_gate",
        "novelty",
        "embedding_distance_to_common_bank",
        "frequency_rarity",
        "rarity_v3",
        "imagination_contribution",
        "imagination_contribution_v3",
        "conjunction_difficulty_bonus",
        "hallucination",
        "contradiction",
        "unresolved_entity",
        "fantasy_marker_score",
        "evidence_mismatch",
    ]
    primitive_means = {}
    for field in primitive_fields:
        value = mean_or_none([
            float(item.get(field))
            for item in item_scores
            if item.get(field) is not None
        ])
        if value is not None:
            primitive_means[field] = round(value, 4)
    subtype_contributions = mean_subtype_contributions(
        item.get("subtype_contributions")
        for item in item_scores
        if isinstance(item.get("subtype_contributions"), dict)
    )

    return {
        "version": "propconj_dual_axis_v2_t1",
        "t1_assoc_version": T1_ASSOC_VERSION,
        "score": round(imagination, 4),
        "imagination": round(imagination, 4),
        "hallucination": round(hallucination, 4),
        "imagination_raw": round(imagination_raw, 4),
        "hallucination_raw": round(hallucination_raw, 4),
        "mean_top8_imagination_contribution": round(quality_mass_top6, 4),
        "quality_mass_top6": round(quality_mass_top6, 4),
        "elite_tail_top3": round(elite_tail_top3, 4),
        "diversity": round(diversity_score, 4),
        "diversity_eff": round(diversity_eff, 4),
        "valid_ratio": round(valid_ratio, 4),
        "soft_valid_ratio": round(soft_valid_ratio, 4),
        "hard_valid_ratio": round(hard_valid_ratio, 4),
        "valid_count": valid_count,
        "soft_valid_count": soft_valid_count,
        "hard_valid_count": hard_valid_count,
        "conjunction_difficulty_bonus": round(conjunction_difficulty_bonus, 4),
        "scored_ideas": len(item_scores),
        "primitive_means": primitive_means,
        "subtype_contributions": subtype_contributions,
        "formula": {
            "imagination_raw": "T1  0.40*top6(item_I)+0.30*top3(item_I)+0.08*diversity_eff+0.10*hard_valid_ratio+0.12*soft_valid_ratio+difficulty_bonus",
            "hallucination_raw": "mean_i(H_i)",
            "residual": "fallback task residual: I=clip(I_raw-beta*H_raw); H=clip(H_raw-gamma*I_raw)",
        },
        "residualization": {
            "beta_IH": beta_ih,
            "beta_HI": beta_hi,
            "source": "task_default",
        },
    }


def aggregate_propconj_model_axes(
    task_scores: Sequence[Dict[str, object]],
    *,
    gate_pass: bool,
    beta_ih: float = 0.20,
    beta_hi: float = 0.10,
) -> Dict[str, object]:
    raw_i = mean_or_none([score.get("imagination_raw") for score in task_scores])
    raw_h = mean_or_none([score.get("hallucination_raw") for score in task_scores])
    quality_mass_top6 = mean_or_none([score.get("quality_mass_top6") for score in task_scores])
    elite_tail_top3 = mean_or_none([score.get("elite_tail_top3") for score in task_scores])
    diversity_eff = mean_or_none([score.get("diversity_eff") for score in task_scores])
    soft_valid_ratio = mean_or_none([score.get("soft_valid_ratio") for score in task_scores])
    hard_valid_ratio = mean_or_none([score.get("hard_valid_ratio") for score in task_scores])
    conjunction_difficulty_bonus = mean_or_none([score.get("conjunction_difficulty_bonus") for score in task_scores])

    imagination_score = clip01((raw_i or 0.0) - beta_ih * max(0.0, raw_h or 0.0)) if gate_pass else None
    hallucination_score = clip01((raw_h or 0.0) - beta_hi * max(0.0, raw_i or 0.0)) if gate_pass else None
    residual_source = "task_default"
    standardization = "none"

    primitive_means = {}
    fields = set()
    for score in task_scores:
        if isinstance(score.get("primitive_means"), dict):
            fields.update(score["primitive_means"].keys())
    for field in sorted(fields):
        value = mean_or_none([
            score.get("primitive_means", {}).get(field)
            for score in task_scores
            if isinstance(score.get("primitive_means"), dict)
        ])
        if value is not None:
            primitive_means[field] = round(value, 4)
    subtype_contributions = mean_subtype_contributions(
        score.get("subtype_contributions")
        for score in task_scores
        if isinstance(score.get("subtype_contributions"), dict)
    )

    return {
        "score": round(imagination_score, 4) if imagination_score is not None else None,
        "imagination": round(imagination_score, 4) if imagination_score is not None else None,
        "hallucination": round(hallucination_score, 4) if hallucination_score is not None else None,
        "imagination_raw": round(raw_i, 4) if raw_i is not None else None,
        "hallucination_raw": round(raw_h, 4) if raw_h is not None else None,
        "t1_assoc_version": T1_ASSOC_VERSION,
        "quality_mass_top6": round(quality_mass_top6, 4) if quality_mass_top6 is not None else None,
        "elite_tail_top3": round(elite_tail_top3, 4) if elite_tail_top3 is not None else None,
        "diversity_eff": round(diversity_eff, 4) if diversity_eff is not None else None,
        "soft_valid_ratio": round(soft_valid_ratio, 4) if soft_valid_ratio is not None else None,
        "hard_valid_ratio": round(hard_valid_ratio, 4) if hard_valid_ratio is not None else None,
        "conjunction_difficulty_bonus": round(conjunction_difficulty_bonus, 4) if conjunction_difficulty_bonus is not None else None,
        "coverage_gate_pass": bool(gate_pass),
        "task_scores": list(task_scores),
        "primitive_means": primitive_means,
        "subtype_contributions": subtype_contributions,
        "scored_tasks": len(task_scores),
        "formula": {
            "item_grounding": "G_i=clip(0.45*max(gmean,0.15*mean_props)+0.20*M_i+0.20*(1-u_i)+0.15*K_i)",
            "item_hallucination": "H_i=clip(0.50*(1-G_i)+0.20*C_i+0.10*u_i+0.10*E_i+0.10*(1-K_i))",
            "task_imagination_raw": "T1 I_raw(q)=0.40*top6(item_I)+0.30*top3(item_I)+0.08*diversity_eff+0.10*hard_valid_ratio+0.12*soft_valid_ratio+difficulty_bonus",
            "model_residual": "clip(mean(I_raw)-beta*mean(H_raw))",
        },
        "residualization": {
            "beta_IH": beta_ih,
            "beta_HI": beta_hi,
            "source": residual_source,
            "standardization": standardization,
        },
    }
