
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from scorer_hyperparameters import get_scorer_hyperparameter
from typed_axis_aggregation import build_cjst_task_subtype_contributions, mean_subtype_contributions
from world_state_scorer import WorldStateScorer


DATA_DIR = Path(__file__).resolve().parent / "data"
CJST_CARDS_PATH = DATA_DIR / "cjst_scenario_cards.json"
CJST_WORLD_CARDS_V2_PATH = DATA_DIR / "cjst_world_cards_v2.json"
CJST_ANCHOR_BANK_PATH = DATA_DIR / "cjst_anchor_bank.json"
CJST_FORBIDDEN_FOILS_PATH = DATA_DIR / "cjst_forbidden_foils.json"
CJST_SCORING_CONFIG_PATH = DATA_DIR / "dual_axis_scoring_config.json"
CJST_COMMON_CONSEQUENCE_BANK_V3_PATH = DATA_DIR / "cjst_common_consequence_bank_v3.json"
CJST_TASK_SCORING_CONFIG_PATH = DATA_DIR / "cjst_scoring_config.json"
CJST_VERSION = "cjst_dual_axis"
CJST_V3_CALIBRATION_POLICY = "benchmark_default"
CJST_V3_RUNTIME_SCORING_POLICY = (
    "fixed output-only parameters"
)
_DEFAULT_CJST_V3_PARAMS_FALLBACK = {
    "rarity_gamma": 1.35,
    "grounding_gamma": 1.25,
    "hard_zero_threshold": 0.42,
    "broad_common_threshold": 0.38,
    "broad_common_floor": 0.25,
    "supported_rare_floor": 0.82,
    "min_intervention_for_uncapped": 0.55,
    "min_causal_edge_for_uncapped": 0.45,
    "task_weights": {
        "quality_mass_top6": 0.35,
        "elite_tail_top3": 0.30,
        "tier_balanced_depth": 0.15,
        "second_order_chain_score": 0.20,
    },
    "top_quality_n": 6,
    "elite_tail_n": 3,
}
DEFAULT_CJST_V3_PARAMS = get_scorer_hyperparameter(
    "counterfactual",
    "DEFAULT_CJST_V3_PARAMS",
    default=_DEFAULT_CJST_V3_PARAMS_FALLBACK,
)
VALID_TIERS = {"immediate", "adaptive", "second_order"}
TIER_ALIASES = {
    "immediate": "immediate",
    "near_term": "immediate",
    "near term": "immediate",
    "nearterm": "immediate",
    "adaptive": "adaptive",
    "adaptation": "adaptive",
    "adapted": "adaptive",
    "second_order": "second_order",
    "second-order": "second_order",
    "second order": "second_order",
    "secondorder": "second_order",
    "second": "second_order",
    "long_term": "second_order",
    "long term": "second_order",
    "longterm": "second_order",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for", "with",
    "in", "on", "at", "by", "is", "are", "was", "were", "be", "being",
    "been", "would", "could", "should", "might", "may", "can", "will",
    "all", "every", "some", "any", "this", "that", "these", "those",
    "people", "person", "someone", "something", "thing", "things", "fact",
    "nearby", "briefly", "slowly", "quietly", "only", "just",
}

MECHANISM_HINTS = {
    "causes", "cause", "enables", "enable", "allows", "allow", "lets", "let",
    "turns", "turn", "forces", "force", "reveals", "reveal", "shows", "show",
    "provides", "provide", "creates", "create", "uses", "use", "because",
    "therefore", "so", "when", "after", "before", "due", "from", "through",
}

_COMMON_CONSEQUENCE_BANK_CACHE = None
_CJST_V3_PARAMS_CACHE = None


def clip01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    return sum(filtered) / len(filtered) if filtered else None


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if payload is not None else default
    except Exception:
        return default


def load_cjst_common_consequence_bank(path: Optional[Path] = None) -> Dict[str, object]:
    global _COMMON_CONSEQUENCE_BANK_CACHE
    if path is None and _COMMON_CONSEQUENCE_BANK_CACHE is not None:
        return _COMMON_CONSEQUENCE_BANK_CACHE
    bank_path = path or CJST_COMMON_CONSEQUENCE_BANK_V3_PATH
    try:
        with bank_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        payload = {"schema": "missing_cjst_common_consequence_bank_v3", "tasks": {}}
    if path is None:
        _COMMON_CONSEQUENCE_BANK_CACHE = payload
    return payload


def load_cjst_v3_calibration_params(path: Optional[Path] = None) -> Dict[str, object]:
    global _CJST_V3_PARAMS_CACHE
    if path is None and _CJST_V3_PARAMS_CACHE is not None:
        return dict(_CJST_V3_PARAMS_CACHE)
    params = dict(DEFAULT_CJST_V3_PARAMS)
    params["task_weights"] = dict(DEFAULT_CJST_V3_PARAMS["task_weights"])
    calibration_path = path or CJST_TASK_SCORING_CONFIG_PATH
    try:
        with calibration_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        payload = {}
    frozen = payload.get("final_scoring_params") if isinstance(payload, dict) else None
    if isinstance(frozen, dict):
        params.update({key: value for key, value in frozen.items() if key != "task_weights"})
        if isinstance(frozen.get("task_weights"), dict):
            weights = dict(DEFAULT_CJST_V3_PARAMS["task_weights"])
            weights.update(frozen["task_weights"])
            params["task_weights"] = weights
    if path is None:
        _CJST_V3_PARAMS_CACHE = dict(params)
    return dict(params)


def get_cjst_common_consequence_bank_coverage(task_ids: Sequence[str]) -> Dict[str, object]:
    bank = load_cjst_common_consequence_bank()
    tasks = bank.get("tasks") if isinstance(bank, dict) else {}
    covered = []
    missing = []
    for task_id in task_ids:
        record = tasks.get(task_id) if isinstance(tasks, dict) else None
        has_required = (
            isinstance(record, dict)
            and bool(record.get("hard_zero_consequence_families"))
            and bool(record.get("broad_common_consequence_families"))
        )
        (covered if has_required else missing).append(task_id)
    return {
        "covered": covered,
        "missing": missing,
        "coverage": round(len(covered) / max(1, len(task_ids)), 4),
    }


def _normalize_text(text: str) -> str:
    text = (text or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_cjst_tier(value) -> str:
    """Map prompt/schema tier spellings to the canonical CJST tier id."""
    raw = str(value or "").strip().lower()
    raw_underscore = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    raw_space = _normalize_text(raw)
    raw_compact = re.sub(r"[^a-z0-9]+", "", raw)
    for candidate in (raw, raw_underscore, raw_space, raw_compact):
        if candidate in TIER_ALIASES:
            return TIER_ALIASES[candidate]
    return raw_underscore or raw_space


def _tokens(text: str) -> List[str]:
    return [
        token for token in _normalize_text(text).split()
        if token and token not in STOPWORDS and len(token) > 1
    ]


def _phrase_hit(text_norm: str, phrase: str) -> bool:
    phrase_norm = _normalize_text(str(phrase))
    if not phrase_norm:
        return False
    phrase_tokens = phrase_norm.split()
    if f" {phrase_norm} " in f" {text_norm} ":
        return True
    token_set = set(text_norm.split())
    return bool(phrase_tokens) and set(phrase_tokens).issubset(token_set)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    left = set(_tokens(text_a))
    right = set(_tokens(text_b))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _cosine(vec_a, vec_b) -> float:
    try:
        left = [float(value) for value in vec_a]
        right = [float(value) for value in vec_b]
        numerator = sum(a * b for a, b in zip(left, right))
        denom = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
        if denom <= 1e-12:
            return 0.0
        return max(-1.0, min(1.0, numerator / denom))
    except Exception:
        return 0.0


def _semantic_similarities(semantic_scorer, query: str, references: Sequence[str]) -> List[float]:
    model = getattr(semantic_scorer, "model", None)
    if model is None or not query or not references:
        return [_jaccard_similarity(query, ref) for ref in references]
    try:
        vectors = model.encode([query] + list(references))
        query_vec = vectors[0]
        return [max(0.0, _cosine(query_vec, ref_vec)) for ref_vec in vectors[1:]]
    except Exception:
        return [_jaccard_similarity(query, ref) for ref in references]


def _pairwise_semantic_distance(semantic_scorer, texts: Sequence[str]) -> float:
    texts = [text for text in texts if text]
    if len(texts) < 2:
        return 0.0
    model = getattr(semantic_scorer, "model", None)
    distances = []
    if model is not None:
        try:
            vectors = model.encode(list(texts))
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    distances.append(clip01(1.0 - max(0.0, _cosine(vectors[i], vectors[j]))))
        except Exception:
            distances = []
    if not distances:
        for i, left in enumerate(texts):
            for right in texts[i + 1:]:
                distances.append(clip01(1.0 - _jaccard_similarity(left, right)))
    return mean_or_none(distances) or 0.0


def _entropy_ratio(values: Sequence[str]) -> float:
    normalized = [_normalize_text(value) for value in values if _normalize_text(value)]
    if not normalized:
        return 0.0
    counts = Counter(normalized)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log2(probability)
    max_entropy = math.log2(max(2, len(counts)))
    return clip01(entropy / max_entropy if max_entropy > 0 else 0.0)


def _as_list(value) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _flatten_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten_text(inner)}" for key, inner in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(inner) for inner in value)
    return str(value)


def _geometric_mean_weighted(components: Sequence[Tuple[float, float]]) -> float:
    if not components:
        return 0.0
    if any(clip01(value) <= 0.0 for value, _ in components):
        return 0.0
    product = 1.0
    total_weight = 0.0
    for value, weight in components:
        clean_weight = max(0.0, float(weight))
        if clean_weight <= 0.0:
            continue
        product *= clip01(value) ** clean_weight
        total_weight += clean_weight
    if total_weight <= 0.0:
        return 0.0
    return clip01(product ** (1.0 / total_weight))


def _top_mean(values: Sequence[float], count: int) -> float:
    usable = sorted([clip01(value) for value in values if value is not None], reverse=True)
    if not usable:
        return 0.0
    selected = usable[:max(1, int(count))]
    return sum(selected) / len(selected)


class CounterfactualScorer:
    def __init__(
        self,
        *,
        data_dir: Optional[str] = None,
        beta_ih: Optional[float] = None,
        beta_hi: Optional[float] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.cards_payload = _load_json(self.data_dir / "cjst_scenario_cards.json", {})
        self.cards = self.cards_payload.get("cards") or {}
        self.world_cards_payload = _load_json(self.data_dir / "cjst_world_cards_v2.json", {})
        self.world_cards = self.world_cards_payload.get("cards") or {}
        self.world_state_scorer = WorldStateScorer()
        self.anchor_bank = (_load_json(self.data_dir / "cjst_anchor_bank.json", {}).get("anchors") or {})
        self.forbidden = _load_json(self.data_dir / "cjst_forbidden_foils.json", {})
        self.common_consequence_bank = load_cjst_common_consequence_bank()
        self.v3_params = load_cjst_v3_calibration_params()
        calibration = _load_json(self.data_dir / "dual_axis_scoring_config.json", {})
        if isinstance(calibration, dict):
            tasks_cal = calibration.get("tasks") or {}
            cjst_cal = (tasks_cal or {}).get("CJST") or calibration.get("CJST") or {}
        else:
            cjst_cal = {}
        self.beta_ih = float(beta_ih if beta_ih is not None else (cjst_cal or {}).get("beta_IH", 0.80))
        self.beta_hi = float(beta_hi if beta_hi is not None else (cjst_cal or {}).get("beta_HI", 0.12))
        self.calibration_source = (
            (cjst_cal or {}).get("source")
            or (calibration.get("source") if isinstance(calibration, dict) else None)
            or "benchmark_default"
        )

    def get_card(self, task_id: str, task: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        if task and isinstance(task.get("scenario_card"), dict):
            return dict(task["scenario_card"])
        return dict(self.cards.get(task_id) or {})

    def _support_terms(self, card: Dict[str, object], anchors: Sequence[str]) -> List[str]:
        chunks: List[str] = []
        for field in ["premise", "allowed_mutation"]:
            value = card.get(field)
            if isinstance(value, str):
                chunks.append(value)
        chunks.extend(str(item) for item in card.get("allowed_mechanisms") or [])
        impact_channels = card.get("impact_channels") or {}
        if isinstance(impact_channels, dict):
            for key, values in impact_channels.items():
                chunks.append(str(key))
                chunks.extend(str(value) for value in values or [])
        chunks.extend(str(anchor) for anchor in anchors)
        terms = []
        seen = set()
        for token in _tokens(" ".join(chunks)):
            if token not in seen:
                seen.add(token)
                terms.append(token)
        return terms

    def _forbidden_hits(self, text: str, card: Dict[str, object]) -> Dict[str, object]:
        text_norm = _normalize_text(text)
        global_hits = []
        severity_sum = 0.0
        for category, payload in (self.forbidden.get("global") or {}).items():
            severity = float(payload.get("severity", 0.7) or 0.7)
            for term in payload.get("terms") or []:
                if _phrase_hit(text_norm, str(term)):
                    global_hits.append({"category": category, "term": term, "severity": severity})
                    severity_sum += severity

        card_hits = []
        for term in card.get("forbidden_foils") or []:
            if _phrase_hit(text_norm, str(term)):
                card_hits.append(str(term))
                severity_sum += 0.90

        return {
            "global_hits": global_hits,
            "card_hits": card_hits,
            "score": clip01(severity_sum / 1.6),
        }

    def _task_bank_record(self, task_id: str) -> Dict[str, object]:
        tasks = self.common_consequence_bank.get("tasks") if isinstance(self.common_consequence_bank, dict) else {}
        record = tasks.get(task_id) if isinstance(tasks, dict) else None
        return record if isinstance(record, dict) else {}

    def _family_match_score(
        self,
        text: str,
        family: Dict[str, object],
        mechanism_tags: set,
        tier: str,
    ) -> Dict[str, object]:
        text_norm = _normalize_text(text)
        text_tokens = set(_tokens(text))
        phrase_hits = [
            str(phrase)
            for phrase in family.get("phrases") or []
            if _phrase_hit(text_norm, str(phrase))
        ]
        keywords = [str(item) for item in family.get("keywords") or [] if str(item)]
        keyword_tokens = []
        seen = set()
        for keyword in keywords:
            for token in _tokens(keyword):
                if token not in seen:
                    seen.add(token)
                    keyword_tokens.append(token)
        keyword_hits = [token for token in keyword_tokens if token in text_tokens]
        mechanism_tokens = {
            token
            for item in family.get("mechanism_tags") or []
            for token in _tokens(str(item))
        }
        mechanism_hits = sorted(mechanism_tokens & mechanism_tags)
        tier_score = 1.0
        if family.get("tier"):
            tier_score = 1.0 if canonicalize_cjst_tier(family.get("tier")) == tier else 0.0
        keyword_score = (
            len(keyword_hits) / min(max(3.0, 1.0), max(1, len(keyword_tokens)))
            if keyword_tokens else 0.0
        )
        phrase_score = 1.0 if phrase_hits else 0.0
        mechanism_score = len(mechanism_hits) / max(1, len(mechanism_tokens)) if mechanism_tokens else 0.0
        score = clip01(max(phrase_score, 0.75 * keyword_score + 0.25 * mechanism_score) * tier_score)
        return {
            "id": family.get("id"),
            "score": round(score, 4),
            "phrase_hits": phrase_hits,
            "keyword_hits": keyword_hits,
            "mechanism_hits": mechanism_hits,
        }

    def _best_family_match(
        self,
        families: Sequence[Dict[str, object]],
        text: str,
        mechanism_tags: set,
        tier: str,
    ) -> Optional[Dict[str, object]]:
        matches = [
            self._family_match_score(text, family, mechanism_tags, tier)
            for family in families
            if isinstance(family, dict)
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: float(item.get("score") or 0.0))

    def _mechanism_tags(self, card: Dict[str, object], item_score_context: Dict[str, object]) -> set:
        tags = set()
        for field in ("support_hits", "bridge_hits"):
            tags.update(str(item) for item in item_score_context.get(field) or [])
        for mechanism in card.get("allowed_mechanisms") or []:
            mechanism_tokens = set(_tokens(str(mechanism)))
            if mechanism_tokens & tags:
                tags.update(mechanism_tokens)
        return {tag for tag in tags if tag}

    def _consequence_rarity(
        self,
        task_id: str,
        *,
        display_text: str,
        fallback_novelty: float,
        mechanism_tags: set,
        tier: str,
    ) -> Dict[str, object]:
        bank = self._task_bank_record(task_id)
        hard_match = self._best_family_match(
            bank.get("hard_zero_consequence_families") or [],
            display_text,
            mechanism_tags,
            tier,
        )
        broad_match = self._best_family_match(
            bank.get("broad_common_consequence_families") or [],
            display_text,
            mechanism_tags,
            tier,
        )
        rare_match = self._best_family_match(
            bank.get("supported_rare_mechanism_families") or [],
            display_text,
            mechanism_tags,
            tier,
        )
        hard_threshold = float(self.v3_params.get("hard_zero_threshold", 0.42))
        broad_threshold = float(self.v3_params.get("broad_common_threshold", 0.38))
        broad_floor = float(self.v3_params.get("broad_common_floor", 0.25))
        rare_floor = float(self.v3_params.get("supported_rare_floor", 0.82))
        rarity = clip01(fallback_novelty)
        source = "anchor_distance_fallback"
        if rare_match and float(rare_match.get("score") or 0.0) >= broad_threshold:
            rarity = max(rarity, rare_floor)
            source = "supported_rare_mechanism_family"
        if broad_match and float(broad_match.get("score") or 0.0) >= broad_threshold:
            similarity = clip01(float(broad_match.get("score") or 0.0))
            rarity = min(rarity, broad_floor + (1.0 - similarity) * (1.0 - broad_floor))
            source = "broad_common_family_cap"
        hard_match_specific = bool(hard_match and (
            hard_match.get("phrase_hits") or
            len(hard_match.get("keyword_hits") or []) >= 3
        ))
        if hard_match and hard_match_specific and float(hard_match.get("score") or 0.0) >= hard_threshold:
            rarity = 0.0
            source = "hard_zero_family"
        return {
            "score": round(clip01(rarity), 4),
            "source": source,
            "bank_available": bool(bank),
            "hard_zero_match": hard_match,
            "broad_common_match": broad_match,
            "supported_rare_match": rare_match,
        }

    def _legacy_jst_groundedness(
        self,
        *,
        card: Dict[str, object],
        display_text: str,
        parsed_item: Dict[str, object],
        semantic_scorer=None,
        groundedness_scorer=None,
    ) -> Optional[float]:
        legacy_id = card.get("legacy_jst_id")
        if not legacy_id or groundedness_scorer is None:
            return None
        try:
            result = groundedness_scorer.score_idea(
                task_type="JST",
                task_id=legacy_id,
                target_concept=card.get("premise") or "",
                idea_text=display_text,
                raw_originality=0.0,
                parsed_item={
                    "consequence_clause": parsed_item.get("consequence_clause") or display_text,
                    "display_text": display_text,
                },
                semantic_scorer=semantic_scorer,
            )
            return float(result.get("groundedness_score"))
        except Exception:
            return None

    def _mechanism_depth(
        self,
        *,
        parsed_item: Dict[str, object],
        bridge_relevance: float,
        world_state: Optional[Dict[str, object]],
    ) -> float:
        causal_chain = [
            str(item)
            for item in _as_list(parsed_item.get("causal_chain"))
            if _normalize_text(str(item))
        ]
        chain_text = " ".join(causal_chain)
        chain_len_score = clip01(len(causal_chain) / 3.0)
        bridge_tokens = _tokens(parsed_item.get("causal_bridge") or "")
        bridge_len_score = clip01(len(bridge_tokens) / 8.0)
        causal_edge_support = clip01((world_state or {}).get("causal_edge_support"))
        update_present = 1.0 if _flatten_text(parsed_item.get("world_state_update")).strip() else 0.0
        chain_has_causal_marker = 1.0 if any(
            marker in set(_tokens(chain_text))
            for marker in MECHANISM_HINTS
        ) else 0.0
        score = clip01(
            0.32 * causal_edge_support +
            0.22 * bridge_relevance +
            0.20 * chain_len_score +
            0.16 * update_present +
            0.10 * max(bridge_len_score, chain_has_causal_marker)
        )
        if not causal_chain:
            score = min(score, 0.35)
        return score

    def _tier_depth(self, tier: str, parsed_item: Dict[str, object]) -> float:
        if tier not in VALID_TIERS:
            return 0.0
        causal_chain = [
            str(item)
            for item in _as_list(parsed_item.get("causal_chain"))
            if _normalize_text(str(item))
        ]
        chain_text = _normalize_text(" ".join(causal_chain + [str(parsed_item.get("causal_bridge") or "")]))
        base = {"immediate": 0.75, "adaptive": 0.85, "second_order": 1.0}.get(tier, 0.0)
        if len(causal_chain) < 2:
            base = min(base, 0.55)
        if tier == "adaptive" and not any(token in chain_text for token in ["adapt", "use", "avoid", "schedule", "install", "design", "change"]):
            base = min(base, 0.75)
        if tier == "second_order" and not any(token in chain_text for token in ["policy", "market", "industry", "habit", "protocol", "over", "later", "repeated", "norm"]):
            base = min(base, 0.70)
        return clip01(base)

    def _world_state_update_score(self, parsed_item: Dict[str, object], world_state: Optional[Dict[str, object]]) -> float:
        update = parsed_item.get("world_state_update")
        if isinstance(update, dict):
            fields = [
                update.get("variable"),
                update.get("old_state") or update.get("old"),
                update.get("new_state") or update.get("new"),
                update.get("licensed_by") or update.get("evidence") or update.get("premise_link"),
            ]
            field_score = sum(1 for field in fields if _normalize_text(str(field or ""))) / 4.0
            text = _flatten_text(update)
        else:
            text = _flatten_text(update)
            field_score = min(0.65, len(_tokens(text)) / 18.0) if text.strip() else 0.0
        world_state = world_state or {}
        formal_support = clip01(
            0.40 * clip01(world_state.get("intervention_used")) +
            0.40 * clip01(world_state.get("causal_edge_support")) +
            0.20 * clip01(world_state.get("world_consistency"))
        ) if world_state else 0.0
        specificity = clip01(len([token for token in _tokens(text) if len(token) >= 5]) / 8.0)
        return clip01(0.45 * field_score + 0.35 * formal_support + 0.20 * specificity)

    def _hard_gate(
        self,
        *,
        parsed_item: Dict[str, object],
        premise_relevance: float,
        premise_lock: float,
        contradiction: float,
        world_state: Optional[Dict[str, object]],
    ) -> float:
        world_state = world_state or {}
        protected_violation = clip01(world_state.get("protected_variable_violation"))
        forbidden_update = clip01(world_state.get("forbidden_update_rate"))
        extra_miracle = clip01(world_state.get("extra_miracle_rate"))
        intervention_used = clip01(world_state.get("intervention_used"))
        causal_edge_support = clip01(world_state.get("causal_edge_support"))
        gate = 1.0
        if (
            contradiction >= 0.75 or
            protected_violation >= 0.75 or
            forbidden_update >= 0.75 or
            extra_miracle >= 0.75
        ):
            return 0.0
        if max(contradiction, protected_violation, forbidden_update, extra_miracle) >= 0.35:
            gate = min(gate, 0.25)
        if intervention_used < float(self.v3_params.get("min_intervention_for_uncapped", 0.55)):
            gate = min(gate, 0.30)
        if causal_edge_support < float(self.v3_params.get("min_causal_edge_for_uncapped", 0.45)):
            gate = min(gate, 0.30)
        if not _as_list(parsed_item.get("causal_chain")):
            gate = min(gate, 0.40)
        if not _flatten_text(parsed_item.get("world_state_update")).strip():
            gate = min(gate, 0.50)
        if premise_relevance < 0.25:
            gate = min(gate, 0.35)
        if premise_lock < 0.30:
            gate = min(gate, 0.45)
        return clip01(gate)

    def score_item(
        self,
        *,
        task_id: str,
        card: Dict[str, object],
        parsed_item: Dict[str, object],
        semantic_scorer=None,
        groundedness_scorer=None,
    ) -> Dict[str, object]:
        consequence = parsed_item.get("consequence_clause") or parsed_item.get("display_text") or ""
        causal_bridge = parsed_item.get("causal_bridge") or ""
        causal_chain = [
            str(item)
            for item in _as_list(parsed_item.get("causal_chain"))
            if _normalize_text(str(item))
        ]
        causal_chain_text = " ".join(causal_chain)
        display_text = parsed_item.get("display_text") or (
            f"{consequence} because {causal_bridge}" if consequence and causal_bridge else consequence or causal_bridge
        )
        anchor_terms = parsed_item.get("anchor_terms") or []
        domain = _normalize_text(parsed_item.get("domain") or "unspecified") or "unspecified"
        tier = canonicalize_cjst_tier(parsed_item.get("tier") or "unspecified")
        anchors = list(self.anchor_bank.get(task_id) or card.get("anchor_consequences") or [])
        premise = card.get("premise") or ""
        support_terms = set(self._support_terms(card, anchors))
        evidence_text = " ".join([display_text, causal_chain_text, " ".join(anchor_terms)])
        text_tokens = set(_tokens(evidence_text))
        bridge_tokens = set(_tokens(" ".join([causal_bridge, causal_chain_text])))

        anchor_sims = _semantic_similarities(semantic_scorer, display_text, anchors)
        max_anchor_similarity = max(anchor_sims) if anchor_sims else 0.0
        premise_similarity = max(_semantic_similarities(semantic_scorer, display_text, [premise]) or [0.0])
        novelty = clip01(0.70 * (1.0 - max_anchor_similarity) + 0.30 * (1.0 - premise_similarity))

        support_hits = sorted(text_tokens & support_terms)
        bridge_hits = sorted(bridge_tokens & support_terms)
        keyword_relevance = clip01(len(support_hits) / max(3.0, min(8.0, len(support_terms) * 0.18)))
        bridge_relevance = clip01(len(bridge_hits) / max(2.0, min(5.0, len(support_terms) * 0.10)))
        legacy_ground = self._legacy_jst_groundedness(
            card=card,
            display_text=display_text,
            parsed_item=parsed_item,
            semantic_scorer=semantic_scorer,
            groundedness_scorer=groundedness_scorer,
        )
        premise_relevance = clip01(
            0.45 * keyword_relevance +
            0.35 * max_anchor_similarity +
            0.20 * bridge_relevance
        )
        if legacy_ground is not None:
            premise_relevance = max(premise_relevance, clip01(0.85 * legacy_ground))

        forbidden = self._forbidden_hits(display_text, card)
        content_token_count = max(1, len(text_tokens))
        support_fraction = clip01(len(support_hits) / content_token_count)
        premise_lock = clip01(0.70 * support_fraction + 0.30 * (1.0 - float(forbidden.get("score", 0.0))))

        bridge_len = len(_tokens(causal_bridge))
        length_score = clip01(bridge_len / 6.0)
        mechanism_hint_score = 1.0 if any(hint in _normalize_text(causal_bridge).split() for hint in MECHANISM_HINTS) else 0.0
        mechanism_completeness = clip01(
            0.55 * length_score +
            0.25 * mechanism_hint_score +
            0.20 * bridge_relevance
        )

        contradiction = clip01(float(forbidden.get("score", 0.0)))
        missing_anchor_terms = 1.0 if not anchor_terms else 0.0
        unsupported_extra_claim_rate = clip01(
            0.50 * contradiction +
            0.30 * max(0.0, 0.55 - premise_relevance) / 0.55 +
            0.20 * missing_anchor_terms
        )

        legacy_item_hallucination_raw = clip01(
            0.35 * (1.0 - premise_relevance) +
            0.25 * (1.0 - premise_lock) +
            0.25 * contradiction +
            0.15 * unsupported_extra_claim_rate
        )
        if contradiction >= 0.75:
            legacy_item_hallucination_raw = max(legacy_item_hallucination_raw, 0.70)

        world_card = self.world_cards.get(task_id) or {}
        world_state = None
        item_hallucination_raw = legacy_item_hallucination_raw
        if isinstance(world_card, dict) and world_card:
            world_state = self.world_state_scorer.score_item(
                world_card,
                parsed_item,
                display_text=display_text,
                causal_bridge=" ".join([causal_bridge, causal_chain_text]),
            )
            formal_h = clip01(world_state.get("formal_hallucination_raw"))
            h_logic = clip01(max(
                contradiction,
                clip01(1.15 * clip01(world_state.get("protected_variable_violation"))),
                clip01(world_state.get("extra_miracle_rate")),
                clip01(world_state.get("forbidden_update_rate")),
                0.75 * (1.0 - clip01(world_state.get("causal_edge_support"))),
            ))
            h_context = clip01(max(
                unsupported_extra_claim_rate,
                0.70 * (1.0 - premise_relevance),
                0.60 * (1.0 - clip01(world_state.get("intervention_used"))),
            ))
            h_drift = clip01(max(
                clip01(0.70 * clip01(world_state.get("forbidden_update_rate")) + 0.30 * formal_h),
                0.65 * (1.0 - premise_lock),
                0.55 * (1.0 - clip01(world_state.get("world_consistency"))),
            ))
            protected_penalty = 0.15 * clip01(world_state.get("protected_variable_violation"))
            item_hallucination_raw = clip01(
                0.18 * legacy_item_hallucination_raw +
                0.32 * formal_h +
                0.16 * h_logic +
                0.17 * h_context +
                0.17 * h_drift +
                protected_penalty
            )
            if (
                clip01(world_state.get("protected_variable_violation")) >= 0.75 or
                clip01(world_state.get("forbidden_update_rate")) >= 0.75 or
                clip01(world_state.get("extra_miracle_rate")) >= 0.75
            ):
                item_hallucination_raw = max(item_hallucination_raw, 0.70)
        else:
            h_logic = contradiction
            h_context = unsupported_extra_claim_rate
            h_drift = clip01(1.0 - premise_lock)

        context_for_tags = {
            "support_hits": support_hits,
            "bridge_hits": bridge_hits,
        }
        mechanism_tags = self._mechanism_tags(card, context_for_tags)
        rarity_record = self._consequence_rarity(
            task_id,
            display_text=display_text,
            fallback_novelty=novelty,
            mechanism_tags=mechanism_tags,
            tier=tier,
        )
        rarity = clip01(rarity_record.get("score"))
        intervention_used = clip01((world_state or {}).get("intervention_used")) if world_state is not None else premise_relevance
        causal_edge_support = clip01((world_state or {}).get("causal_edge_support")) if world_state is not None else bridge_relevance
        world_consistency = clip01((world_state or {}).get("world_consistency")) if world_state is not None else premise_lock
        grounding_gmean = _geometric_mean_weighted([
            (intervention_used, 0.22),
            (causal_edge_support, 0.22),
            (premise_relevance, 0.20),
            (premise_lock, 0.18),
            (world_consistency, 0.18),
        ])
        mechanism_depth = self._mechanism_depth(
            parsed_item=parsed_item,
            bridge_relevance=bridge_relevance,
            world_state=world_state,
        )
        tier_depth = self._tier_depth(tier, parsed_item)
        world_state_update_score = self._world_state_update_score(parsed_item, world_state)
        hard_gate = self._hard_gate(
            parsed_item=parsed_item,
            premise_relevance=premise_relevance,
            premise_lock=premise_lock,
            contradiction=contradiction,
            world_state=world_state,
        )
        item_imagination_raw = clip01(
            (rarity ** float(self.v3_params.get("rarity_gamma", 1.35))) *
            (grounding_gmean ** float(self.v3_params.get("grounding_gamma", 1.25))) *
            hard_gate *
            clip01(0.40 + 0.25 * mechanism_depth + 0.15 * tier_depth + 0.20 * world_state_update_score)
        )
        hard_valid = bool(
            hard_gate >= 0.999 and
            grounding_gmean >= 0.55 and
            mechanism_depth >= 0.45 and
            tier in VALID_TIERS and
            bool(causal_chain) and
            bool(_flatten_text(parsed_item.get("world_state_update")).strip())
        )

        result = {
            "version": "cjst_item_v3",
            "tier": tier,
            "domain": domain,
            "consequence": consequence,
            "causal_bridge": causal_bridge,
            "causal_chain": causal_chain,
            "novelty": round(novelty, 4),
            "rarity_v3": round(rarity, 4),
            "premise_relevance": round(premise_relevance, 4),
            "premise_lock": round(premise_lock, 4),
            "mechanism_completeness": round(mechanism_completeness, 4),
            "mechanism_depth": round(mechanism_depth, 4),
            "tier_depth": round(tier_depth, 4),
            "world_state_update_score": round(world_state_update_score, 4),
            "grounding_gmean": round(grounding_gmean, 4),
            "hard_gate": round(hard_gate, 4),
            "hard_valid": hard_valid,
            "imagination_raw": round(item_imagination_raw, 4),
            "unsupported_extra_claim_rate": round(unsupported_extra_claim_rate, 4),
            "contradiction": round(contradiction, 4),
            "legacy_item_hallucination_raw": round(legacy_item_hallucination_raw, 4),
            "item_hallucination_raw": round(item_hallucination_raw, 4),
            "H_logic": round(h_logic, 4),
            "H_context": round(h_context, 4),
            "H_drift": round(h_drift, 4),
            "legacy_jst_groundedness": round(legacy_ground, 4) if legacy_ground is not None else None,
            "rarity_record": rarity_record,
            "evidence": {
                "support_hits": support_hits,
                "bridge_hits": bridge_hits,
                "mechanism_tags": sorted(mechanism_tags),
                "max_anchor_similarity": round(max_anchor_similarity, 4),
                "premise_similarity": round(premise_similarity, 4),
                "forbidden_hits": forbidden,
                "anchor_terms": anchor_terms,
            },
        }
        if world_state is not None:
            result["world_state"] = world_state
            for field in [
                "intervention_used",
                "causal_edge_support",
                "protected_variable_violation",
                "forbidden_update_rate",
                "extra_miracle_rate",
                "world_consistency",
                "formal_hallucination_raw",
            ]:
                result[field] = world_state.get(field)
            result["evidence"]["world_state"] = world_state.get("evidence")
        return result

    def tier_counts(self, parsed_items: Sequence[Dict[str, object]]) -> Dict[str, int]:
        counts = {tier: 0 for tier in sorted(VALID_TIERS)}
        for item in parsed_items:
            tier = canonicalize_cjst_tier(item.get("tier") or "")
            if tier in counts:
                counts[tier] += 1
        return counts

    def score_task(
        self,
        task: Dict[str, object],
        parsed_items: Sequence[Dict[str, object]],
        *,
        semantic_scorer=None,
        groundedness_scorer=None,
        expected_output_count: int = 12,
    ) -> Dict[str, object]:
        task_id = str(task.get("id") or "")
        card = self.get_card(task_id, task=task)
        item_scores = [
            self.score_item(
                task_id=task_id,
                card=card,
                parsed_item=item,
                semantic_scorer=semantic_scorer,
                groundedness_scorer=groundedness_scorer,
            )
            for item in parsed_items
        ]
        if not item_scores:
            return {
                "version": CJST_VERSION,
                "task_id": task_id,
                "score": None,
                "imagination": None,
                "hallucination": None,
                "imagination_raw": None,
                "hallucination_raw": None,
                "details": [],
            }

        diversity = _pairwise_semantic_distance(semantic_scorer, [item.get("display_text") or "" for item in parsed_items])
        domain_entropy = _entropy_ratio([score.get("domain") for score in item_scores])
        tier_counts = self.tier_counts(parsed_items)
        tier_coverage = len([tier for tier, count in tier_counts.items() if count > 0]) / max(1, len(VALID_TIERS))
        tier_balance = min(1.0, min(tier_counts.values()) / 4.0) if tier_counts else 0.0
        diversity_task = clip01(0.55 * diversity + 0.25 * domain_entropy + 0.20 * tier_coverage)
        tier_domain_coverage = clip01(0.60 * tier_balance + 0.40 * domain_entropy)

        mean_novelty = mean_or_none([score.get("novelty") for score in item_scores]) or 0.0
        mean_mechanism = mean_or_none([score.get("mechanism_completeness") for score in item_scores]) or 0.0
        mean_world_consistency = mean_or_none([score.get("world_consistency") for score in item_scores])
        hallucination_raw = mean_or_none([score.get("item_hallucination_raw") for score in item_scores]) or 0.0
        novelty_ranks = self._percentile_ranks([float(score.get("novelty", 0.0) or 0.0) for score in item_scores])
        for score, rank in zip(item_scores, novelty_ranks):
            score["novelty_percentile_rank"] = round(rank, 4)
        item_i = [
            clip01(score.get("imagination_raw")) * (0.65 + 0.35 * clip01(score.get("novelty_percentile_rank")))
            for score in item_scores
        ]
        quality_mass_top6 = _top_mean(item_i, int(self.v3_params.get("top_quality_n", 6)))
        elite_tail_top3 = _top_mean(item_i, int(self.v3_params.get("elite_tail_n", 3)))
        tier_balanced_depth = self._tier_balanced_depth(item_scores, tier_counts)
        mechanism_diversity_eff = self._mechanism_diversity_eff(item_scores)
        second_order_chain_score = self._second_order_chain_score(item_scores)
        hard_valid_ratio = mean_or_none([1.0 if score.get("hard_valid") else 0.0 for score in item_scores]) or 0.0
        common_bank_coverage = mean_or_none([
            1.0 if (score.get("rarity_record") or {}).get("bank_available") else 0.0
            for score in item_scores
        ]) or 0.0
        task_weights = self.v3_params.get("task_weights") if isinstance(self.v3_params.get("task_weights"), dict) else {}
        imagination_raw = clip01(
            float(task_weights.get("quality_mass_top6", 0.35)) * quality_mass_top6 +
            float(task_weights.get("elite_tail_top3", 0.30)) * elite_tail_top3 +
            float(task_weights.get("tier_balanced_depth", 0.15)) * tier_balanced_depth +
            float(task_weights.get("second_order_chain_score", 0.20)) * second_order_chain_score
        )
        imagination = clip01(imagination_raw - self.beta_ih * hallucination_raw)
        hallucination = clip01(hallucination_raw - self.beta_hi * imagination_raw)

        primitive_means = {}
        for field in [
            "novelty",
            "rarity_v3",
            "premise_relevance",
            "premise_lock",
            "mechanism_completeness",
            "mechanism_depth",
            "tier_depth",
            "world_state_update_score",
            "novelty_percentile_rank",
            "grounding_gmean",
            "hard_gate",
            "unsupported_extra_claim_rate",
            "contradiction",
            "item_hallucination_raw",
            "legacy_item_hallucination_raw",
            "imagination_raw",
            "H_logic",
            "H_context",
            "H_drift",
            "intervention_used",
            "causal_edge_support",
            "protected_variable_violation",
            "forbidden_update_rate",
            "extra_miracle_rate",
            "world_consistency",
            "formal_hallucination_raw",
        ]:
            value = mean_or_none([score.get(field) for score in item_scores])
            if value is not None:
                primitive_means[field] = round(value, 4)
        primitive_means.update({
            "diversity": round(diversity_task, 4),
            "semantic_diversity": round(diversity, 4),
            "domain_entropy": round(domain_entropy, 4),
            "tier_coverage": round(tier_coverage, 4),
            "tier_domain_coverage": round(tier_domain_coverage, 4),
            "quality_mass_top6": round(quality_mass_top6, 4),
            "elite_tail_top3": round(elite_tail_top3, 4),
            "tier_balanced_depth": round(tier_balanced_depth, 4),
            "mechanism_diversity_eff": round(mechanism_diversity_eff, 4),
            "second_order_chain_score": round(second_order_chain_score, 4),
            "hard_valid_ratio": round(hard_valid_ratio, 4),
            "common_bank_coverage": round(common_bank_coverage, 4),
            "false_forbidden_match_rate": 0.0,
        })

        result = {
            "version": CJST_VERSION,
            "task_id": task_id,
            "score": round(imagination, 4),
            "imagination": round(imagination, 4),
            "hallucination": round(hallucination, 4),
            "imagination_raw": round(imagination_raw, 4),
            "hallucination_raw": round(hallucination_raw, 4),
            "diversity": round(diversity_task, 4),
            "quality_mass_top6": round(quality_mass_top6, 4),
            "elite_tail_top3": round(elite_tail_top3, 4),
            "tier_balanced_depth": round(tier_balanced_depth, 4),
            "mechanism_diversity_eff": round(mechanism_diversity_eff, 4),
            "second_order_chain_score": round(second_order_chain_score, 4),
            "hard_valid_ratio": round(hard_valid_ratio, 4),
            "common_bank_coverage": round(common_bank_coverage, 4),
            "semantic_diversity": round(diversity, 4),
            "domain_entropy": round(domain_entropy, 4),
            "tier_counts": tier_counts,
            "tier_coverage": round(tier_coverage, 4),
            "tier_domain_coverage": round(tier_domain_coverage, 4),
            "scored_ideas": len(item_scores),
            "expected_output_count": expected_output_count,
            "primitive_means": primitive_means,
            "details": item_scores,
            "formula": {
                "item_imagination_raw": "v3  rarity^1.35 * grounding_gmean^1.25 * hard_gate * (0.40+0.25*mechanism_depth+0.15*tier_depth+0.20*world_state_update)",
                "imagination_raw": "v3  0.35*top6_quality_mass+0.30*top3_elite_tail+0.15*tier_balanced_depth+0.20*second_order_chain_score; item_i uses per-task novelty percentile stretch",
                "hallucination_raw": " mean white-box context/logic/drift burden with protected/forbidden/extra-miracle floor",
                "residual": "I=clip01(I_raw-beta_IH*H_raw); H=clip01(H_raw-beta_HI*I_raw)",
            },
            "residualization": {
                "beta_IH": self.beta_ih,
                "beta_HI": self.beta_hi,
                "source": CJST_V3_CALIBRATION_POLICY,
            },
            "calibration_policy": CJST_V3_CALIBRATION_POLICY,
            "runtime_scoring_policy": CJST_V3_RUNTIME_SCORING_POLICY,
            "common_bank_version": self.common_consequence_bank.get("version") if isinstance(self.common_consequence_bank, dict) else None,
        }
        result["subtype_contributions"] = build_cjst_task_subtype_contributions(
            result,
            beta_ih=self.beta_ih,
            beta_hi=self.beta_hi,
        )
        result["atom_signals"] = result["subtype_contributions"].get("atom_signals", {})
        return result

    def _tier_balanced_depth(self, item_scores: Sequence[Dict[str, object]], tier_counts: Dict[str, int]) -> float:
        tier_tops = []
        for tier in sorted(VALID_TIERS):
            values = [
                clip01(score.get("imagination_raw"))
                for score in item_scores
                if score.get("tier") == tier and score.get("hard_valid")
            ]
            tier_tops.append(max(values) if values else 0.0)
        coverage = sum(1 for value in tier_tops if value > 0.0) / max(1, len(VALID_TIERS))
        balance = min(1.0, min(tier_counts.values()) / 4.0) if tier_counts else 0.0
        return clip01((sum(tier_tops) / max(1, len(tier_tops))) * coverage * (0.50 + 0.50 * balance))

    def _item_strategy_signature(self, item_score: Dict[str, object]) -> set:
        tags = {str(item_score.get("tier") or ""), str(item_score.get("domain") or "")}
        rarity_record = item_score.get("rarity_record") or {}
        for key in ("hard_zero_match", "broad_common_match", "supported_rare_match"):
            match = rarity_record.get(key)
            if isinstance(match, dict) and match.get("id") and float(match.get("score") or 0.0) > 0.0:
                tags.add(str(match.get("id")))
        evidence = item_score.get("evidence") or {}
        tags.update(str(item) for item in evidence.get("mechanism_tags") or [])
        if not any(tag for tag in tags if tag):
            tags.add(_normalize_text(item_score.get("consequence") or "")[:60])
        return {tag for tag in tags if tag}

    def _mechanism_diversity_eff(self, item_scores: Sequence[Dict[str, object]]) -> float:
        valid_scores = [score for score in item_scores if score.get("hard_valid")]
        if len(valid_scores) < 2:
            return 0.0
        distances = []
        for i in range(len(valid_scores)):
            left = self._item_strategy_signature(valid_scores[i])
            for j in range(i + 1, len(valid_scores)):
                right = self._item_strategy_signature(valid_scores[j])
                if not left and not right:
                    distance = 0.0
                elif not left or not right:
                    distance = 1.0
                else:
                    distance = 1.0 - len(left & right) / len(left | right)
                distances.append(clip01(distance))
        diversity = mean_or_none(distances) or 0.0
        hard_valid_ratio = len(valid_scores) / max(1, len(item_scores))
        return clip01(diversity * (0.50 + 0.50 * hard_valid_ratio))

    def _percentile_ranks(self, values: Sequence[float]) -> List[float]:
        if not values:
            return []
        ranked = sorted((float(value), index) for index, value in enumerate(values))
        result = [0.0 for _ in values]
        if len(ranked) == 1:
            result[ranked[0][1]] = 1.0
            return result
        for rank, (_, index) in enumerate(ranked):
            result[index] = rank / max(1, len(ranked) - 1)
        return [clip01(value) for value in result]

    def _second_order_chain_score(self, item_scores: Sequence[Dict[str, object]]) -> float:
        upstream_tokens = set()
        for score in item_scores:
            if score.get("tier") not in {"immediate", "adaptive"}:
                continue
            upstream_tokens.update(_tokens(score.get("consequence") or ""))
            upstream_tokens.update(_tokens(score.get("causal_bridge") or ""))
            for step in score.get("causal_chain") or []:
                upstream_tokens.update(_tokens(step))
            evidence = score.get("evidence") or {}
            upstream_tokens.update(str(token) for token in evidence.get("support_hits") or [])
            upstream_tokens.update(str(token) for token in evidence.get("bridge_hits") or [])
        upstream_tokens = {token for token in upstream_tokens if len(token) >= 4 and token not in STOPWORDS}
        second_scores = []
        for score in item_scores:
            if score.get("tier") != "second_order":
                continue
            chain_tokens = set(_tokens(" ".join([
                str(score.get("consequence") or ""),
                str(score.get("causal_bridge") or ""),
                " ".join(str(step) for step in score.get("causal_chain") or []),
            ])))
            chain_tokens = {token for token in chain_tokens if len(token) >= 4 and token not in STOPWORDS}
            overlap = len(chain_tokens & upstream_tokens) / max(1, min(8, len(chain_tokens)))
            hard_valid = 1.0 if score.get("hard_valid") else 0.0
            mechanism = clip01(score.get("mechanism_depth"))
            world_update = clip01(score.get("world_state_update_score"))
            second_scores.append(clip01(
                0.45 * clip01(overlap) +
                0.25 * mechanism +
                0.20 * world_update +
                0.10 * hard_valid
            ))
        return max(second_scores) if second_scores else 0.0


def aggregate_cjst_model_axes(
    task_scores: Sequence[Dict[str, object]],
    *,
    gate_pass: bool,
    beta_ih: float = 0.80,
    beta_hi: float = 0.12,
) -> Dict[str, object]:
    raw_i = mean_or_none([score.get("imagination_raw") for score in task_scores])
    raw_h = mean_or_none([score.get("hallucination_raw") for score in task_scores])
    imagination = clip01((raw_i or 0.0) - beta_ih * max(0.0, raw_h or 0.0)) if gate_pass else None
    hallucination = clip01((raw_h or 0.0) - beta_hi * max(0.0, raw_i or 0.0)) if gate_pass else None

    fields = set()
    for score in task_scores:
        if isinstance(score.get("primitive_means"), dict):
            fields.update(score["primitive_means"].keys())
    primitive_means = {}
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
    diagnostic_fields = {
        "quality_mass_top6",
        "elite_tail_top3",
        "tier_balanced_depth",
        "second_order_chain_score",
        "mechanism_diversity_eff",
        "hard_valid_ratio",
        "common_bank_coverage",
    }
    diagnostic_means = {}
    for field in sorted(diagnostic_fields):
        value = mean_or_none([
            score.get(field)
            for score in task_scores
            if score.get(field) is not None
        ])
        if value is not None:
            diagnostic_means[field] = round(value, 4)

    return {
        "version": CJST_VERSION,
        "calibration_policy": CJST_V3_CALIBRATION_POLICY,
        "runtime_scoring_policy": CJST_V3_RUNTIME_SCORING_POLICY,
        "score": round(imagination, 4) if imagination is not None else None,
        "imagination": round(imagination, 4) if imagination is not None else None,
        "hallucination": round(hallucination, 4) if hallucination is not None else None,
        "imagination_raw": round(raw_i, 4) if raw_i is not None else None,
        "hallucination_raw": round(raw_h, 4) if raw_h is not None else None,
        "coverage_gate_pass": bool(gate_pass),
        "task_scores": list(task_scores),
        "primitive_means": primitive_means,
        "subtype_contributions": subtype_contributions,
        **diagnostic_means,
        "scored_tasks": len(task_scores),
        "formula": {
            "item_imagination_raw": "v3  rarity^1.35 * grounding_gmean^1.25 * hard_gate * (0.40+0.25*mechanism_depth+0.15*tier_depth+0.20*world_state_update)",
            "task_imagination_raw": "v3  0.35*top6_quality_mass+0.30*top3_elite_tail+0.15*tier_balanced_depth+0.20*second_order_chain_score",
            "task_hallucination_raw": "v3 white-box context/logic/drift burden",
            "model_residual": "clip(mean(raw)-beta*mean(other_raw))",
        },
        "residualization": {
            "beta_IH": beta_ih,
            "beta_HI": beta_hi,
            "source": CJST_V3_CALIBRATION_POLICY,
            "standardization": "none",
        },
    }


__all__ = [
    "CJST_VERSION",
    "CJST_V3_CALIBRATION_POLICY",
    "CJST_V3_RUNTIME_SCORING_POLICY",
    "VALID_TIERS",
    "canonicalize_cjst_tier",
    "CounterfactualScorer",
    "aggregate_cjst_model_axes",
    "get_cjst_common_consequence_bank_coverage",
    "load_cjst_common_consequence_bank",
    "load_cjst_v3_calibration_params",
    "clip01",
    "mean_or_none",
]
