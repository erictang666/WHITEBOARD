
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from json_repair_utils import compact_prose_paragraphs, parse_jsonish_payload
from support_ledger_scorer import SupportLedgerScorer
from typed_axis_aggregation import (
    build_gcw_task_subtype_contributions,
    mean_subtype_contributions,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
GCW_VERSION = "gcw_dual_axis"
DEFAULT_GCW_BETA_IH = 0.80
DEFAULT_GCW_BETA_HI = 0.12
DEFAULT_BEAT_COUNT = 6
GCW_COMMON_STORY_BANK_V3_PATH = DATA_DIR / "gcw_common_story_bank_v3.json"
GCW_ENTITY_ALIASES_V3_PATH = DATA_DIR / "gcw_entity_aliases_v3.json"
GCW_SCORING_CONFIG_PATH = DATA_DIR / "gcw_scoring_config.json"
GCW_V3_CALIBRATION_POLICY = "benchmark_default"
GCW_V3_RUNTIME_SCORING_POLICY = (
    "fixed output-only parameters"
)
DEFAULT_GCW_V3_PARAMS = {
    "rarity_gamma": 1.35,
    "support_gamma": 1.35,
    "hard_zero_threshold": 0.42,
    "broad_common_threshold": 0.38,
    "broad_common_floor": 0.25,
    "supported_rare_floor": 0.84,
    "min_support_gmean_uncapped": 0.45,
    "task_weights": {
        "grounded_turn_quality": 0.40,
        "causal_payoff": 0.20,
        "top3_scene_specificity": 0.15,
        "arc_diversity_eff": 0.15,
        "hard_valid_ledger_ratio": 0.10,
    },
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for", "with",
    "in", "on", "at", "by", "is", "are", "was", "were", "be", "being",
    "been", "would", "could", "should", "might", "may", "can", "will",
    "all", "every", "some", "any", "this", "that", "these", "those",
    "one", "two", "three", "before", "after", "when", "then", "there",
    "their", "them", "they", "she", "he", "her", "his", "into", "from",
}

CAUSAL_TERMS = {
    "because", "so", "therefore", "turns", "turn", "lets", "allows",
    "uses", "use", "through", "instead", "without", "prevents", "makes",
    "causes", "becomes", "marks", "guides", "signals", "calibrates",
}

_COMMON_STORY_BANK_CACHE = None
_ENTITY_ALIASES_CACHE = None
_GCW_V3_PARAMS_CACHE = None


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


def load_gcw_common_story_bank(path: Optional[Path] = None) -> Dict[str, object]:
    global _COMMON_STORY_BANK_CACHE
    if path is None and _COMMON_STORY_BANK_CACHE is not None:
        return _COMMON_STORY_BANK_CACHE
    bank_path = path or GCW_COMMON_STORY_BANK_V3_PATH
    try:
        with bank_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        payload = {"schema": "missing_gcw_common_story_bank_v3", "tasks": {}}
    if path is None:
        _COMMON_STORY_BANK_CACHE = payload
    return payload


def load_gcw_entity_aliases(path: Optional[Path] = None) -> Dict[str, object]:
    global _ENTITY_ALIASES_CACHE
    if path is None and _ENTITY_ALIASES_CACHE is not None:
        return _ENTITY_ALIASES_CACHE
    alias_path = path or GCW_ENTITY_ALIASES_V3_PATH
    try:
        with alias_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        payload = {"schema": "missing_gcw_entity_aliases_v3", "tasks": {}}
    if path is None:
        _ENTITY_ALIASES_CACHE = payload
    return payload


def load_gcw_v3_calibration_params(path: Optional[Path] = None) -> Dict[str, object]:
    global _GCW_V3_PARAMS_CACHE
    if path is None and _GCW_V3_PARAMS_CACHE is not None:
        return dict(_GCW_V3_PARAMS_CACHE)
    params = dict(DEFAULT_GCW_V3_PARAMS)
    params["task_weights"] = dict(DEFAULT_GCW_V3_PARAMS["task_weights"])
    calibration_path = path or GCW_SCORING_CONFIG_PATH
    try:
        with calibration_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        payload = {}
    frozen = payload.get("final_scoring_params") if isinstance(payload, dict) else None
    if isinstance(frozen, dict):
        params.update({key: value for key, value in frozen.items() if key != "task_weights"})
        if isinstance(frozen.get("task_weights"), dict):
            weights = dict(DEFAULT_GCW_V3_PARAMS["task_weights"])
            weights.update(frozen["task_weights"])
            params["task_weights"] = weights
    if path is None:
        _GCW_V3_PARAMS_CACHE = dict(params)
    return dict(params)


def get_gcw_common_story_bank_coverage(task_ids: Sequence[str]) -> Dict[str, object]:
    bank = load_gcw_common_story_bank()
    tasks = bank.get("tasks") if isinstance(bank, dict) else {}
    covered = []
    missing = []
    for task_id in task_ids:
        record = tasks.get(task_id) if isinstance(tasks, dict) else None
        has_required = (
            isinstance(record, dict)
            and bool(record.get("hard_zero_plot_families"))
            and bool(record.get("broad_common_plot_families"))
            and bool(record.get("supported_rare_turn_families"))
        )
        (covered if has_required else missing).append(task_id)
    return {
        "covered": covered,
        "missing": missing,
        "coverage": round(len(covered) / max(1, len(task_ids)), 4),
    }


def get_gcw_entity_alias_coverage(task_ids: Sequence[str]) -> Dict[str, object]:
    payload = load_gcw_entity_aliases()
    tasks = payload.get("tasks") if isinstance(payload, dict) else {}
    covered = []
    missing = []
    for task_id in task_ids:
        record = tasks.get(task_id) if isinstance(tasks, dict) else None
        has_required = isinstance(record, dict) and bool(record.get("aliases"))
        (covered if has_required else missing).append(task_id)
    return {
        "covered": covered,
        "missing": missing,
        "coverage": round(len(covered) / max(1, len(task_ids)), 4),
    }
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if payload is not None else default
    except Exception:
        return default


def _normalize_text(text: str) -> str:
    text = (text or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> List[str]:
    return [
        token for token in _normalize_text(text).split()
        if token and token not in STOPWORDS and len(token) > 1
    ]


def _clean_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", str(value)).strip()


def _as_list(value) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _dedupe_strings(values: Iterable[object]) -> List[str]:
    seen = set()
    results = []
    for value in values:
        cleaned = _clean_string(value)
        if not cleaned:
            continue
        key = _normalize_text(cleaned)
        if key in seen:
            continue
        seen.add(key)
        results.append(cleaned)
    return results


def _extract_json_payload(raw_text):
    return parse_jsonish_payload(raw_text)


def _phrase_hit(text_norm: str, phrase: str) -> bool:
    phrase_norm = _normalize_text(str(phrase))
    if not phrase_norm:
        return False
    phrase_tokens = phrase_norm.split()
    if f" {phrase_norm} " in f" {text_norm} ":
        return True
    token_set = set(text_norm.split())
    return bool(phrase_tokens) and set(phrase_tokens).issubset(token_set)


def _keyword_hits(text: str, keywords: Sequence[str]) -> List[str]:
    text_norm = _normalize_text(text)
    hits = []
    for keyword in keywords or []:
        if _phrase_hit(text_norm, str(keyword)):
            hits.append(str(keyword))
    return sorted(set(hits))


def _geometric_mean(values: Sequence[float], *, floor: float = 1e-6) -> float:
    filtered = [clip01(value) for value in values]
    if not filtered:
        return 0.0
    product = 1.0
    for value in filtered:
        product *= max(float(floor), value)
    return clip01(product ** (1.0 / len(filtered)))


def _top_mean(values: Sequence[float], n: int) -> float:
    clean = sorted((clip01(value) for value in values), reverse=True)
    if not clean:
        return 0.0
    return sum(clean[:max(1, int(n))]) / min(len(clean), max(1, int(n)))


def _alias_variants(text: str) -> List[str]:
    base = _normalize_text(text)
    if not base:
        return []
    variants = {base}
    if base.endswith("s") and len(base) > 3:
        variants.add(base[:-1])
    else:
        variants.add(base + "s")
    if base.endswith("ies") and len(base) > 4:
        variants.add(base[:-3] + "y")
    if base.endswith("y") and len(base) > 2:
        variants.add(base[:-1] + "ies")
    return [item for item in variants if item]


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
    references = [ref for ref in references if ref]
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


def _sigmoid(value: float) -> float:
    try:
        value = max(-30.0, min(30.0, float(value)))
        return 1.0 / (1.0 + math.exp(-value))
    except Exception:
        return 0.5


def _robust_z(value: float, median: float, mad: float) -> float:
    sigma = max(1e-6, 1.4826 * float(mad or 0.0))
    return (float(value) - float(median)) / sigma


def _entity_names_from_facts(task: Dict[str, object], fact_type: str) -> List[str]:
    names = []
    for fact in task.get("fact_sheet") or []:
        if fact.get("type") != fact_type:
            continue
        keywords = fact.get("keywords") or []
        if keywords:
            names.append(str(keywords[0]))
    return _dedupe_strings(names)


class GroundedCreativeWritingScorer:
    """Scores one GCW story against a fact/constraint card."""

    def __init__(
        self,
        *,
        data_dir: Optional[str] = None,
        beta_ih: Optional[float] = None,
        beta_hi: Optional[float] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.common_bank = _load_json(self.data_dir / "gcw_common_story_bank.json", {})
        self.common_story_bank_v3 = load_gcw_common_story_bank(
            self.data_dir / "gcw_common_story_bank_v3.json"
        )
        self.entity_aliases_v3 = load_gcw_entity_aliases(
            self.data_dir / "gcw_entity_aliases_v3.json"
        )
        self.v3_params = load_gcw_v3_calibration_params(
            self.data_dir / "gcw_scoring_config.json"
        )
        self.patterns = _load_json(self.data_dir / "gcw_constraint_patterns.json", {})
        self.constraint_ladders = _load_json(self.data_dir / "gcw_constraint_ladders.json", {})
        self.support_ledger = SupportLedgerScorer(patterns=self.patterns)
        calibration = _load_json(self.data_dir / "dual_axis_scoring_config.json", {})
        if isinstance(calibration, dict):
            tasks_cal = calibration.get("tasks") or {}
            gcw_cal = (tasks_cal or {}).get("GCW") or calibration.get("GCW") or {}
        else:
            gcw_cal = {}
        self.beta_ih = float(beta_ih if beta_ih is not None else (gcw_cal or {}).get("beta_IH", DEFAULT_GCW_BETA_IH))
        self.beta_hi = float(beta_hi if beta_hi is not None else (gcw_cal or {}).get("beta_HI", DEFAULT_GCW_BETA_HI))
        self.i_median = float((gcw_cal or {}).get("imagination_raw_median", 0.55))
        self.i_mad = float((gcw_cal or {}).get("imagination_raw_mad", 0.16))
        self.h_median = float((gcw_cal or {}).get("hallucination_raw_median", 0.16))
        self.h_mad = float((gcw_cal or {}).get("hallucination_raw_mad", 0.10))
        self.per_task_standardization = (
            self.v3_params.get("per_task_standardization")
            if isinstance(self.v3_params.get("per_task_standardization"), dict) else
            {}
        )
        self.calibration_source = (
            (gcw_cal or {}).get("source")
            or (calibration.get("source") if isinstance(calibration, dict) else None)
            or "benchmark_default"
        )

    def _standardization_stats(self, task_id: str) -> Dict[str, object]:
        stats = (
            self.per_task_standardization.get(task_id)
            if isinstance(self.per_task_standardization, dict) else None
        )
        if not isinstance(stats, dict):
            return {
                "scope": "global_fallback",
                "imagination_raw_median": self.i_median,
                "imagination_raw_mad": self.i_mad,
                "hallucination_raw_median": self.h_median,
                "hallucination_raw_mad": self.h_mad,
            }
        return {
            "scope": "per_task",
            "imagination_raw_median": float(stats.get("imagination_raw_median", self.i_median)),
            "imagination_raw_mad": float(stats.get("imagination_raw_mad", self.i_mad)),
            "hallucination_raw_median": float(stats.get("hallucination_raw_median", self.h_median)),
            "hallucination_raw_mad": float(stats.get("hallucination_raw_mad", self.h_mad)),
        }

    
    
    

    def parse_response(self, raw_text: str) -> Dict[str, object]:
        payload = _extract_json_payload(raw_text)
        if payload is None:
            prose_beats = compact_prose_paragraphs(raw_text, max_paragraphs=DEFAULT_BEAT_COUNT)
            if prose_beats:
                payload = {
                    "title": "",
                    "grounded_turn": "",
                    "constraint_strategy": "",
                    "payoff_ledger": [],
                    "beats": prose_beats,
                    "claims": [],
                    "ending_callback": "",
                    "style_devices": [],
                }
        if isinstance(payload, list):
            payload = {
                "title": "",
                "grounded_turn": "",
                "constraint_strategy": "",
                "payoff_ledger": [],
                "beats": payload,
                "claims": [],
                "ending_callback": "",
                "style_devices": [],
            }
        if not isinstance(payload, dict):
            return {
                "parse_valid": False,
                "title": "",
                "grounded_turn": "",
                "constraint_strategy": "",
                "payoff_ledger": [],
                "beats": [],
                "claims": [],
                "ending_callback": "",
                "style_devices": [],
                "raw_payload": None,
                "parse_error": "no_json_object",
            }

        raw_beats = (
            payload.get("beats")
            or payload.get("story_beats")
            or payload.get("paragraphs")
            or payload.get("scenes")
        )
        if not raw_beats:
            story_text = (
                payload.get("story")
                or payload.get("microfiction")
                or payload.get("draft")
                or payload.get("text")
                or payload.get("paragraph")
            )
            raw_beats = compact_prose_paragraphs(story_text, max_paragraphs=DEFAULT_BEAT_COUNT)

        beats = []
        for index, raw_beat in enumerate(_as_list(raw_beats), start=1):
            beat = self._parse_beat(raw_beat, default_id=index)
            if beat is not None:
                beats.append(beat)

        claims = []
        for index, raw_claim in enumerate(_as_list(payload.get("claims") or payload.get("claim_ledger") or payload.get("story_claims")), start=1):
            claim = self._parse_claim(raw_claim, default_id=index)
            if claim is not None:
                claims.append(claim)

        return {
            "parse_valid": True,
            "title": _clean_string(payload.get("title") or payload.get("story_title")),
            "grounded_turn": _clean_string(payload.get("grounded_turn") or payload.get("turn")),
            "constraint_strategy": _clean_string(payload.get("constraint_strategy") or payload.get("strategy")),
            "payoff_ledger": self._parse_payoff_ledger(payload.get("payoff_ledger") or payload.get("payoffs")),
            "beats": beats,
            "claims": claims,
            "ending_callback": _clean_string(payload.get("ending_callback")),
            "style_devices": _dedupe_strings(_as_list(payload.get("style_devices"))),
            "raw_payload": payload,
            "parse_error": None,
        }

    def _parse_beat(self, raw_beat, *, default_id: int) -> Optional[Dict[str, object]]:
        if isinstance(raw_beat, str):
            paragraph = _clean_string(raw_beat)
            if not paragraph:
                return None
            return {
                "beat_id": default_id,
                "beat_role": "",
                "causal_function": "",
                "paragraph": paragraph,
                "used_fact_ids": [],
                "characters": [],
                "places": [],
                "objects": [],
                "claimed_new_facts": [],
                "raw_beat": raw_beat,
            }
        if not isinstance(raw_beat, dict):
            return None
        paragraph = _clean_string(
            raw_beat.get("paragraph")
            or raw_beat.get("text")
            or raw_beat.get("beat")
            or raw_beat.get("content")
            or raw_beat.get("scene")
        )
        if not paragraph:
            return None
        try:
            beat_id = int(raw_beat.get("beat_id") or raw_beat.get("id") or default_id)
        except Exception:
            beat_id = default_id
        return {
            "beat_id": beat_id,
            "beat_role": _clean_string(raw_beat.get("beat_role") or raw_beat.get("role")),
            "causal_function": _clean_string(raw_beat.get("causal_function") or raw_beat.get("function")),
            "paragraph": paragraph,
            "used_fact_ids": _dedupe_strings(_as_list(raw_beat.get("used_fact_ids") or raw_beat.get("facts"))),
            "characters": _dedupe_strings(_as_list(raw_beat.get("characters"))),
            "places": _dedupe_strings(_as_list(raw_beat.get("places"))),
            "objects": _dedupe_strings(_as_list(raw_beat.get("objects"))),
            "claimed_new_facts": _dedupe_strings(_as_list(raw_beat.get("claimed_new_facts") or raw_beat.get("new_facts"))),
            "raw_beat": raw_beat,
        }

    def _parse_payoff_ledger(self, raw_ledger) -> List[Dict[str, object]]:
        records = []
        for index, raw_record in enumerate(_as_list(raw_ledger), start=1):
            if isinstance(raw_record, str):
                payoff = _clean_string(raw_record)
                if payoff:
                    records.append({
                        "payoff_id": f"P{index}",
                        "payoff": payoff,
                        "evidence_ids": [],
                        "beat_ids": [],
                    })
                continue
            if not isinstance(raw_record, dict):
                continue
            payoff = _clean_string(raw_record.get("payoff") or raw_record.get("text") or raw_record.get("claim"))
            if not payoff:
                continue
            beat_ids = []
            for value in _as_list(raw_record.get("beat_ids") or raw_record.get("beats")):
                try:
                    beat_ids.append(int(value))
                except Exception:
                    continue
            records.append({
                "payoff_id": _clean_string(raw_record.get("payoff_id") or raw_record.get("id") or f"P{index}"),
                "payoff": payoff,
                "evidence_ids": _dedupe_strings(
                    list(_as_list(raw_record.get("evidence_ids"))) +
                    list(_as_list(raw_record.get("support_ids"))) +
                    list(_as_list(raw_record.get("citation_ids")))
                ),
                "beat_ids": beat_ids,
            })
        return records

    def _parse_claim(self, raw_claim, *, default_id: int) -> Optional[Dict[str, object]]:
        if isinstance(raw_claim, str):
            text = _clean_string(raw_claim)
            if not text:
                return None
            return {
                "claim_id": f"CL{default_id}",
                "beat_id": None,
                "text": text,
                "claim_type": "claim",
                "support_ids": [],
                "evidence_ids": [],
                "raw_claim": raw_claim,
            }
        if not isinstance(raw_claim, dict):
            return None
        text = _clean_string(raw_claim.get("text") or raw_claim.get("claim") or raw_claim.get("statement"))
        if not text:
            return None
        try:
            beat_id = int(raw_claim.get("beat_id")) if raw_claim.get("beat_id") is not None else None
        except Exception:
            beat_id = None
        support_ids = _dedupe_strings(
            list(_as_list(raw_claim.get("support_ids"))) +
            list(_as_list(raw_claim.get("evidence_ids"))) +
            list(_as_list(raw_claim.get("citation_ids")))
        )
        evidence_ids = _dedupe_strings(_as_list(raw_claim.get("evidence_ids") or support_ids))
        return {
            "claim_id": _clean_string(raw_claim.get("claim_id") or raw_claim.get("id") or f"CL{default_id}"),
            "beat_id": beat_id,
            "text": text,
            "claim_type": _clean_string(raw_claim.get("claim_type") or raw_claim.get("type") or "claim"),
            "support_ids": support_ids,
            "evidence_ids": evidence_ids,
            "raw_claim": raw_claim,
        }

    
    
    

    def _constraint_profile(self, task: Dict[str, object]) -> Dict[str, object]:
        if isinstance(task.get("constraint_profile"), dict):
            profile = dict(task.get("constraint_profile") or {})
        else:
            ladders = self.constraint_ladders if isinstance(self.constraint_ladders, dict) else {}
            card_profile = (ladders.get("cards") or {}).get(task.get("id")) or {}
            selected_level = (
                task.get("constraint_level")
                or card_profile.get("selected_level")
                or ladders.get("default_selected_level")
                or "gcw_l3"
            )
            profile = dict((ladders.get("level_defaults") or {}).get(selected_level) or {})
            profile.update((card_profile.get("levels") or {}).get(selected_level) or {})
            profile["level_id"] = selected_level
            profile["selected_level"] = selected_level
        profile.setdefault("level_id", profile.get("selected_level") or task.get("constraint_level") or "gcw_l1")
        profile.setdefault("selected_level", profile.get("level_id"))
        profile["required_fact_ids"] = list(profile.get("required_fact_ids") or task.get("required_facts") or [])
        profile["evidence_ids"] = [
            str(fact.get("id"))
            for fact in task.get("fact_sheet") or []
            if fact.get("id")
        ]
        profile["constraint_ids"] = [
            str(constraint.get("id"))
            for constraint in task.get("constraint_sheet") or []
            if constraint.get("id")
        ]
        if "required_causal_callback_terms" not in profile:
            profile["required_causal_callback_terms"] = list(task.get("motifs") or [])
        return profile

    def _constraint_level_score(self, profile: Dict[str, object]) -> float:
        level = profile.get("constraint_level")
        if level is None:
            match = re.search(r"(\d+)$", str(profile.get("level_id") or ""))
            level = int(match.group(1)) if match else 1
        return clip01((float(level) - 1.0) / 2.0)

    def _task_story_bank(self, task: Dict[str, object]) -> Dict[str, object]:
        tasks = self.common_story_bank_v3.get("tasks") if isinstance(self.common_story_bank_v3, dict) else {}
        record = (tasks or {}).get(task.get("id"))
        return record if isinstance(record, dict) else {}

    def _task_entity_aliases(self, task: Dict[str, object]) -> List[str]:
        aliases = []
        tasks = self.entity_aliases_v3.get("tasks") if isinstance(self.entity_aliases_v3, dict) else {}
        record = (tasks or {}).get(task.get("id")) if isinstance(tasks, dict) else None
        if isinstance(record, dict):
            for group in record.get("aliases") or []:
                aliases.extend(_as_list(group))
        aliases.extend(task.get("allowed_entities") or [])
        for fact in task.get("fact_sheet") or []:
            aliases.extend((fact or {}).get("keywords") or [])
        expanded = []
        for alias in aliases:
            expanded.extend(_alias_variants(str(alias)))
        return _dedupe_strings(expanded)

    def _task_with_entity_aliases(self, task: Dict[str, object]) -> Dict[str, object]:
        task_copy = dict(task)
        aliases = self._task_entity_aliases(task)
        task_copy["entity_aliases_v3"] = aliases
        task_copy["allowed_entities"] = _dedupe_strings(list(task.get("allowed_entities") or []) + aliases)
        return task_copy

    def _family_terms(self, family: Dict[str, object]) -> List[str]:
        terms = []
        terms.extend(family.get("phrases") or [])
        terms.extend(family.get("keywords") or [])
        return _dedupe_strings(terms)

    def _family_similarity(
        self,
        text: str,
        family: Dict[str, object],
        *,
        keyword_cap: Optional[float] = None,
    ) -> float:
        text_norm = _normalize_text(text)
        if not text_norm:
            return 0.0
        phrases = family.get("phrases") or []
        keywords = family.get("keywords") or []
        phrase_hit = False
        for phrase in phrases:
            phrase_norm = _normalize_text(str(phrase))
            if not phrase_norm:
                continue
            padded = f" {text_norm} "
            needle = f" {phrase_norm} "
            index = padded.find(needle)
            if index < 0:
                continue
            before_tokens = padded[:index].split()[-4:]
            if any(token in {"no", "not", "never", "without", "nobody", "nothing"} for token in before_tokens):
                continue
            phrase_hit = True
            break
        keyword_hits = _keyword_hits(text_norm, keywords)
        keyword_score = clip01(len(keyword_hits) / max(1, min(4, len(keywords))))
        if keyword_cap is not None and not phrase_hit:
            keyword_score = min(keyword_score, float(keyword_cap))
        phrase_score = 1.0 if phrase_hit else 0.0
        if not phrases and not keywords:
            return 0.0
        return clip01(max(phrase_score, keyword_score))

    def _best_family_match(
        self,
        text: str,
        families: Sequence[Dict[str, object]],
        *,
        keyword_cap: Optional[float] = None,
    ) -> Tuple[float, Optional[str], List[str]]:
        best_score = 0.0
        best_id = None
        best_hits = []
        for family in families or []:
            if not isinstance(family, dict):
                continue
            score = self._family_similarity(text, family, keyword_cap=keyword_cap)
            if score > best_score:
                best_score = score
                best_id = family.get("id")
                best_hits = _keyword_hits(text, self._family_terms(family))
        return clip01(best_score), best_id, best_hits

    def _story_rarity_v3(
        self,
        task: Dict[str, object],
        full_text: str,
        originality: Dict[str, object],
    ) -> Dict[str, object]:
        bank = self._task_story_bank(task)
        hard_score, hard_id, hard_hits = self._best_family_match(
            full_text, bank.get("hard_zero_plot_families") or [], keyword_cap=0.35
        )
        broad_score, broad_id, broad_hits = self._best_family_match(
            full_text, bank.get("broad_common_plot_families") or [], keyword_cap=0.35
        )
        rare_score, rare_id, rare_hits = self._best_family_match(
            full_text, bank.get("supported_rare_turn_families") or []
        )
        hard_threshold = float(self.v3_params.get("hard_zero_threshold", 0.42))
        broad_threshold = float(self.v3_params.get("broad_common_threshold", 0.38))
        if hard_score >= hard_threshold:
            rarity = 0.0
            family_kind = "hard_zero"
            family_id = hard_id
        elif broad_score >= broad_threshold:
            rarity = min(float(self.v3_params.get("broad_common_floor", 0.25)), 0.38 * (1.0 - 0.35 * broad_score))
            family_kind = "broad_common"
            family_id = broad_id
        else:
            fallback = clip01(
                0.60 * float(originality.get("reference_bank_distance", 0.0)) +
                0.40 * float(originality.get("twist_distance", 0.0))
            )
            rare_floor = float(self.v3_params.get("supported_rare_floor", 0.84))
            rarity = max(fallback, rare_floor * rare_score if rare_score > 0 else 0.0)
            family_kind = "supported_rare" if rare_score >= 0.30 else "fallback"
            family_id = rare_id if rare_score >= 0.30 else None
        return {
            "rarity": round(clip01(rarity), 4),
            "family_kind": family_kind,
            "family_id": family_id,
            "hard_zero_similarity": round(hard_score, 4),
            "broad_common_similarity": round(broad_score, 4),
            "supported_rare_similarity": round(rare_score, 4),
            "hard_zero_hits": hard_hits,
            "broad_common_hits": broad_hits,
            "supported_rare_hits": rare_hits,
            "bank_coverage": 1.0 if bank else 0.0,
        }

    def _payoff_evidence_coverage(
        self,
        task: Dict[str, object],
        parsed_response: Dict[str, object],
    ) -> Tuple[float, int, int]:
        valid_ids = {
            str(item.get("id"))
            for item in list(task.get("fact_sheet") or []) + list(task.get("constraint_sheet") or [])
            if isinstance(item, dict) and item.get("id")
        }
        records = [item for item in parsed_response.get("payoff_ledger") or [] if isinstance(item, dict)]
        if not records:
            return 0.0, 0, 0
        supported = 0
        cited = 0
        for record in records:
            ids = [str(item) for item in record.get("evidence_ids") or []]
            cited += len(ids)
            if ids and all(item in valid_ids for item in ids):
                supported += 1
        return clip01(supported / max(1, len(records))), supported, cited

    def _scene_specificity_scores(
        self,
        task: Dict[str, object],
        beats: Sequence[Dict[str, object]],
    ) -> List[float]:
        fact_keywords = []
        for fact in task.get("fact_sheet") or []:
            fact_keywords.extend((fact or {}).get("keywords") or [])
        allowed = list(task.get("allowed_entities") or [])
        action_terms = self.common_bank.get("action_terms") or []
        sensory_terms = self.common_bank.get("sensory_terms") or []
        scores = []
        for beat in beats:
            text = " ".join([
                beat.get("paragraph") or "",
                beat.get("causal_function") or "",
                " ".join(beat.get("characters") or []),
                " ".join(beat.get("places") or []),
                " ".join(beat.get("objects") or []),
            ])
            entity_score = clip01(len(_keyword_hits(text, allowed)) / 2.0)
            fact_score = clip01(len(_keyword_hits(text, fact_keywords)) / 3.0)
            action_score = clip01(len(_keyword_hits(text, action_terms)) / 2.0)
            state_score = clip01(len(_keyword_hits(text, sensory_terms + list(CAUSAL_TERMS))) / 3.0)
            role_score = 1.0 if beat.get("beat_role") and beat.get("causal_function") else 0.55 if beat.get("causal_function") else 0.0
            scores.append(clip01(0.25 * entity_score + 0.30 * fact_score + 0.20 * action_score + 0.15 * state_score + 0.10 * role_score))
        return scores

    def _gcw_v3_story_scores(
        self,
        task: Dict[str, object],
        parsed_response: Dict[str, object],
        beats: Sequence[Dict[str, object]],
        full_text: str,
        originality: Dict[str, object],
        flexibility: Dict[str, object],
        elaboration: Dict[str, object],
        hallucination_parts: Dict[str, object],
        support_ledger: Dict[str, object],
        constraint_profile: Dict[str, object],
        callback_failure: float,
    ) -> Dict[str, object]:
        turn_chunks = [
            parsed_response.get("grounded_turn") or "",
            parsed_response.get("constraint_strategy") or "",
            json.dumps(parsed_response.get("payoff_ledger") or [], ensure_ascii=False),
            parsed_response.get("ending_callback") or "",
            " ".join(beat.get("causal_function") or "" for beat in beats),
        ]
        turn_text = " ".join(chunk for chunk in turn_chunks if chunk)
        rarity_info = self._story_rarity_v3(task, f"{turn_text} {full_text}", originality)
        payoff_coverage, supported_payoffs, cited_payoff_ids = self._payoff_evidence_coverage(task, parsed_response)
        used_fact_ids = {
            str(fact_id)
            for beat in beats
            for fact_id in (beat.get("used_fact_ids") or [])
        }
        required_fact_ids = {str(item) for item in constraint_profile.get("required_fact_ids") or task.get("required_facts") or []}
        required_fact_coverage = (
            len(used_fact_ids & required_fact_ids) / len(required_fact_ids)
            if required_fact_ids else 1.0
        )
        constraint_respect = clip01(1.0 - max(
            float(hallucination_parts.get("hard_constraint_violation_rate", 0.0)),
            float(support_ledger.get("hard_no_drift_violation", 0.0)),
            float(support_ledger.get("forbidden_motif_rate", 0.0)),
        ))
        entity_persistence = clip01(1.0 - float(support_ledger.get("entity_persistence_failure", 0.0)))
        support_gmean = _geometric_mean([
            float(support_ledger.get("claim_support_precision", 0.0)),
            max(float(support_ledger.get("claim_support_recall", 0.0)), required_fact_coverage),
            payoff_coverage if parsed_response.get("payoff_ledger") else 0.45,
            constraint_respect,
            entity_persistence,
        ])

        turn_norm = _normalize_text(turn_text)
        constraint_strategy = parsed_response.get("constraint_strategy") or ""
        avoidance_hits = _keyword_hits(turn_text + " " + constraint_strategy, ["avoid", "without", "instead", "cannot", "forbidden", "wrong", "missing", "no"])
        constraint_ids = [
            str(constraint.get("id"))
            for constraint in task.get("constraint_sheet") or []
            if constraint.get("id")
        ]
        cited_constraint_ids = [cid for cid in constraint_ids if _phrase_hit(turn_norm, cid)]
        constraint_transform = clip01(
            0.35 * (1.0 if parsed_response.get("constraint_strategy") else 0.0) +
            0.30 * min(1.0, len(avoidance_hits) / 2.0) +
            0.20 * min(1.0, len(cited_constraint_ids) / max(1, min(2, len(constraint_ids)))) +
            0.15 * (1.0 if rarity_info["family_kind"] == "supported_rare" else 0.0)
        )
        fact_id_text = " ".join([turn_text, full_text])
        turn_fact_hits = [fid for fid in used_fact_ids if _phrase_hit(_normalize_text(fact_id_text), fid)]
        causal_hits = _keyword_hits(turn_text + " " + full_text, list(CAUSAL_TERMS))
        multi_fact_synthesis = clip01(
            0.50 * min(1.0, len(used_fact_ids & (required_fact_ids or used_fact_ids)) / max(1, min(3, len(required_fact_ids or used_fact_ids)))) +
            0.30 * min(1.0, len(turn_fact_hits) / 2.0) +
            0.20 * min(1.0, len(causal_hits) / 3.0)
        )
        style_specificity = clip01(
            0.45 * float(elaboration.get("sensory_specificity", 0.0)) +
            0.30 * float(elaboration.get("setting_anchoring", 0.0)) +
            0.25 * float(originality.get("twist_distance", 0.0))
        )

        hard_gate = 1.0
        if rarity_info["family_kind"] == "hard_zero":
            hard_gate = 0.0
        if (
            hallucination_parts.get("hard_constraint_violation_rate", 0.0) >= 1.0
            or support_ledger.get("hard_no_drift_violation", 0.0) >= 1.0
            or support_ledger.get("forbidden_motif_rate", 0.0) >= 0.35
        ):
            hard_gate = min(hard_gate, 0.12)
        if support_ledger.get("unsupported_span_rate", 0.0) >= 0.50:
            hard_gate = min(hard_gate, 0.35)
        if support_gmean < float(self.v3_params.get("min_support_gmean_uncapped", 0.45)):
            hard_gate = min(hard_gate, 0.45 + 0.70 * support_gmean)
        if rarity_info["family_kind"] == "broad_common":
            hard_gate = min(hard_gate, 0.65)
        if not parsed_response.get("grounded_turn"):
            hard_gate = min(hard_gate, 0.75)
        if not parsed_response.get("payoff_ledger"):
            hard_gate = min(hard_gate, 0.80)
        if required_fact_coverage < 1.0:
            hard_gate = min(hard_gate, 0.55 + 0.40 * required_fact_coverage)

        rarity_gamma = float(self.v3_params.get("rarity_gamma", 1.35))
        support_gamma = float(self.v3_params.get("support_gamma", 1.35))
        grounded_turn_quality = clip01(
            (float(rarity_info["rarity"]) ** rarity_gamma) *
            (support_gmean ** support_gamma) *
            hard_gate *
            (
                0.45 +
                0.25 * constraint_transform +
                0.20 * multi_fact_synthesis +
                0.10 * style_specificity
            )
        )

        role_values = [_normalize_text(beat.get("beat_role") or "") for beat in beats if beat.get("beat_role")]
        role_diversity = clip01(len(set(role_values)) / max(1, min(5, len(beats))))
        causal_functions = [beat.get("causal_function") or "" for beat in beats if beat.get("causal_function")]
        causal_function_diversity = _pairwise_semantic_distance(None, causal_functions)
        arc_diversity_eff = clip01(
            0.45 * float(flexibility.get("semantic_beat_diversity", 0.0)) +
            0.35 * role_diversity +
            0.20 * causal_function_diversity
        )
        scene_scores = self._scene_specificity_scores(task, beats)
        top3_scene_specificity = _top_mean(scene_scores, 3)
        causal_payoff = clip01(
            0.45 * payoff_coverage +
            0.25 * (1.0 if parsed_response.get("ending_callback") else 0.0) +
            0.20 * min(1.0, len(causal_functions) / max(1, min(4, len(beats)))) +
            0.10 * clip01(1.0 - callback_failure)
        )
        hard_valid_ledger_ratio = clip01(
            0.30 * float(support_ledger.get("claim_support_precision", 0.0)) +
            0.20 * float(support_ledger.get("claim_support_recall", 0.0)) +
            0.20 * (1.0 - float(support_ledger.get("citation_mismatch_rate", 0.0))) +
            0.15 * (1.0 - float(support_ledger.get("entity_drift_rate", 0.0))) +
            0.15 * constraint_respect
        )
        permitted_scene_claims = [
            record for record in support_ledger.get("claim_records") or []
            if isinstance(record, dict) and record.get("permitted_scene_invention")
        ]
        permitted_scene_invention_rate = clip01(
            len(permitted_scene_claims) / max(1, int(support_ledger.get("checked_claims") or 0))
        )
        unsupported_physical_claim_rate = clip01(
            max(
                float(support_ledger.get("unsupported_span_rate", 0.0)),
                float(hallucination_parts.get("unsupported_claim_rate", 0.0)),
            )
        )
        entity_alias_false_drift_rate = 0.0 if task.get("entity_aliases_v3") else float(support_ledger.get("entity_drift_rate", 0.0))
        task_weights = self.v3_params.get("task_weights") or DEFAULT_GCW_V3_PARAMS["task_weights"]
        imagination_raw = clip01(
            float(task_weights.get("grounded_turn_quality", 0.40)) * grounded_turn_quality +
            float(task_weights.get("causal_payoff", 0.20)) * causal_payoff +
            float(task_weights.get("top3_scene_specificity", 0.15)) * top3_scene_specificity +
            float(task_weights.get("arc_diversity_eff", 0.15)) * arc_diversity_eff +
            float(task_weights.get("hard_valid_ledger_ratio", 0.10)) * hard_valid_ledger_ratio
        )
        return {
            "imagination_raw": round(imagination_raw, 4),
            "grounded_turn_quality": round(grounded_turn_quality, 4),
            "turn_rarity_v3": round(float(rarity_info["rarity"]), 4),
            "turn_support_gmean": round(support_gmean, 4),
            "turn_hard_gate": round(hard_gate, 4),
            "constraint_transform": round(constraint_transform, 4),
            "multi_fact_synthesis": round(multi_fact_synthesis, 4),
            "style_specificity_v3": round(style_specificity, 4),
            "causal_payoff": round(causal_payoff, 4),
            "top3_scene_specificity": round(top3_scene_specificity, 4),
            "arc_diversity_eff": round(arc_diversity_eff, 4),
            "hard_valid_ledger_ratio": round(hard_valid_ledger_ratio, 4),
            "common_bank_coverage": round(float(rarity_info["bank_coverage"]), 4),
            "claim_support_precision": round(float(support_ledger.get("claim_support_precision", 0.0)), 4),
            "payoff_evidence_coverage": round(payoff_coverage, 4),
            "supported_payoff_count": supported_payoffs,
            "cited_payoff_id_count": cited_payoff_ids,
            "required_fact_coverage": round(required_fact_coverage, 4),
            "entity_alias_false_drift_rate": round(entity_alias_false_drift_rate, 4),
            "unsupported_physical_claim_rate": round(unsupported_physical_claim_rate, 4),
            "permitted_scene_invention_rate": round(permitted_scene_invention_rate, 4),
            "rarity_family_kind": rarity_info["family_kind"],
            "rarity_family_id": rarity_info["family_id"],
            "hard_zero_similarity": rarity_info["hard_zero_similarity"],
            "broad_common_similarity": rarity_info["broad_common_similarity"],
            "supported_rare_similarity": rarity_info["supported_rare_similarity"],
            "rarity_hits": {
                "hard_zero": rarity_info["hard_zero_hits"],
                "broad_common": rarity_info["broad_common_hits"],
                "supported_rare": rarity_info["supported_rare_hits"],
            },
        }

    def score_task(
        self,
        task: Dict[str, object],
        parsed_response: Dict[str, object],
        *,
        semantic_scorer=None,
        expected_beat_count: Optional[int] = None,
    ) -> Dict[str, object]:
        expected_beat_count = int(expected_beat_count or task.get("beat_count") or DEFAULT_BEAT_COUNT)
        beats = (parsed_response.get("beats") or [])[:expected_beat_count]
        paragraphs = [beat.get("paragraph") or "" for beat in beats]
        full_text = self._full_story_text(parsed_response, beats)
        task_for_support = self._task_with_entity_aliases(task)
        constraint_profile = self._constraint_profile(task_for_support)

        fluency = self._fluency_scores(parsed_response, beats, expected_beat_count)
        flexibility = self._flexibility_scores(task_for_support, beats, paragraphs, semantic_scorer=semantic_scorer)
        originality = self._originality_scores(task_for_support, parsed_response, paragraphs, full_text, semantic_scorer=semantic_scorer)
        elaboration = self._elaboration_scores(task_for_support, parsed_response, beats, paragraphs, full_text)
        hallucination_parts = self._hallucination_scores(task_for_support, parsed_response, beats, full_text)
        support_ledger = self.support_ledger.score_response(
            task_for_support,
            parsed_response,
            constraint_profile=constraint_profile,
            full_text=full_text,
        )

        f_story = clip01(mean_or_none([
            fluency["schema_validity"],
            fluency["beat_completeness"],
            fluency["pacing_balance"],
            fluency["ending_callback"],
        ]) or 0.0)
        x_story = clip01(
            0.45 * flexibility["semantic_beat_diversity"] +
            0.30 * flexibility["entity_coverage"] +
            0.25 * flexibility["action_interiority_balance"]
        )
        o_story = clip01(
            0.40 * originality["reference_bank_distance"] +
            0.30 * originality["non_cliche_motif_score"] +
            0.30 * originality["twist_distance"]
        )
        e_story = clip01(
            0.30 * elaboration["sensory_specificity"] +
            0.25 * elaboration["setting_anchoring"] +
            0.25 * elaboration["character_state_change"] +
            0.20 * elaboration["subtext_marker"]
        )
        constraint_level_score = self._constraint_level_score(constraint_profile)
        closed_h_raw = clip01(
            0.30 * hallucination_parts["unsupported_claim_rate"] +
            0.40 * hallucination_parts["contradiction_rate"] +
            0.20 * hallucination_parts["hard_constraint_violation_rate"] +
            0.10 * hallucination_parts["missing_required_fact_rate"]
        )
        ledger_h_raw = clip01(
            0.24 * support_ledger["unsupported_span_rate"] +
            0.24 * support_ledger["contradicted_claim_rate"] +
            0.18 * support_ledger["citation_mismatch_rate"] +
            0.14 * support_ledger["entity_drift_rate"] +
            0.10 * support_ledger["claim_without_evidence_rate"] +
            0.10 * support_ledger["hard_no_drift_violation"]
        )
        h_raw = clip01(0.58 * closed_h_raw + 0.42 * ledger_h_raw)
        if (
            hallucination_parts["hard_constraint_violation_rate"] >= 1.0
            or hallucination_parts["contradiction_rate"] >= 0.35
            or support_ledger["contradicted_claim_rate"] >= 0.35
            or support_ledger["hard_no_drift_violation"] >= 1.0
        ):
            h_raw = max(h_raw, 0.70)

        callback_hits = _keyword_hits(full_text, constraint_profile.get("required_causal_callback_terms") or [])
        callback_required = bool(constraint_profile.get("required_causal_callback"))
        callback_rate = clip01(len(callback_hits) / max(1, min(2, len(constraint_profile.get("required_causal_callback_terms") or []))))
        callback_failure = clip01(1.0 - callback_rate) if callback_required else 0.0
        v3_story = self._gcw_v3_story_scores(
            task_for_support,
            parsed_response,
            beats,
            full_text,
            originality,
            flexibility,
            elaboration,
            hallucination_parts,
            support_ledger,
            constraint_profile,
            callback_failure,
        )
        imagination_raw = clip01(v3_story["imagination_raw"])
        support_gate = clip01(
            0.55 * support_ledger["claim_support_precision"] +
            0.25 * support_ledger["claim_support_recall"] +
            0.20 * (1.0 - support_ledger["citation_mismatch_rate"])
        )
        constraint_gate = clip01(1.0 - max(
            hallucination_parts["hard_constraint_violation_rate"],
            support_ledger["hard_no_drift_violation"],
            support_ledger["forbidden_motif_rate"],
            0.50 * hallucination_parts["missing_required_fact_rate"],
            0.40 * callback_failure,
        ))
        imagination_gated = clip01(imagination_raw * (0.40 + 0.30 * support_gate + 0.30 * constraint_gate))
        claim_evidence_mismatch = clip01(max(
            support_ledger["unsupported_span_rate"],
            hallucination_parts["unsupported_claim_rate"],
            support_ledger["claim_without_evidence_rate"],
            1.0 - support_ledger["claim_support_precision"],
        ))
        constraint_degradation = clip01(
            (0.35 + 0.65 * constraint_level_score) *
            max(
                hallucination_parts["hard_constraint_violation_rate"],
                hallucination_parts["missing_required_fact_rate"],
                support_ledger["unsupported_span_rate"],
                support_ledger["citation_mismatch_rate"],
                support_ledger["entity_persistence_failure"],
                callback_failure,
            )
        )

        standardization = self._standardization_stats(str(task.get("id") or ""))
        z_i = _robust_z(
            imagination_gated,
            standardization["imagination_raw_median"],
            standardization["imagination_raw_mad"],
        )
        z_h = _robust_z(
            h_raw,
            standardization["hallucination_raw_median"],
            standardization["hallucination_raw_mad"],
        )
        i_resid_z = z_i - self.beta_ih * z_h
        h_resid_z = z_h - self.beta_hi * z_i
        imagination = clip01(0.50 + 0.20 * i_resid_z)
        hallucination = clip01(0.50 + 0.20 * h_resid_z)
        imagination_simple = clip01(imagination_gated - self.beta_ih * h_raw)
        hallucination_simple = clip01(h_raw - self.beta_hi * imagination_gated)

        primitive_means = {
            "F_story": round(f_story, 4),
            "X_story": round(x_story, 4),
            "O_story": round(o_story, 4),
            "E_story": round(e_story, 4),
            "schema_validity": round(fluency["schema_validity"], 4),
            "beat_completeness": round(fluency["beat_completeness"], 4),
            "pacing_balance": round(fluency["pacing_balance"], 4),
            "ending_callback": round(fluency["ending_callback"], 4),
            "semantic_beat_diversity": round(flexibility["semantic_beat_diversity"], 4),
            "entity_coverage": round(flexibility["entity_coverage"], 4),
            "action_interiority_balance": round(flexibility["action_interiority_balance"], 4),
            "reference_bank_distance": round(originality["reference_bank_distance"], 4),
            "non_cliche_motif_score": round(originality["non_cliche_motif_score"], 4),
            "twist_distance": round(originality["twist_distance"], 4),
            "sensory_specificity": round(elaboration["sensory_specificity"], 4),
            "setting_anchoring": round(elaboration["setting_anchoring"], 4),
            "character_state_change": round(elaboration["character_state_change"], 4),
            "subtext_marker": round(elaboration["subtext_marker"], 4),
            "unsupported_claim_rate": round(hallucination_parts["unsupported_claim_rate"], 4),
            "contradiction_rate": round(hallucination_parts["contradiction_rate"], 4),
            "hard_constraint_violation_rate": round(hallucination_parts["hard_constraint_violation_rate"], 4),
            "missing_required_fact_rate": round(hallucination_parts["missing_required_fact_rate"], 4),
            "closed_world_hallucination_raw": round(closed_h_raw, 4),
            "claim_support_precision": round(support_ledger["claim_support_precision"], 4),
            "claim_support_recall": round(support_ledger["claim_support_recall"], 4),
            "unsupported_span_rate": round(support_ledger["unsupported_span_rate"], 4),
            "citation_mismatch_rate": round(support_ledger["citation_mismatch_rate"], 4),
            "contradicted_claim_rate": round(support_ledger["contradicted_claim_rate"], 4),
            "claim_without_evidence_rate": round(support_ledger["claim_without_evidence_rate"], 4),
            "unknown_evidence_rate": round(support_ledger["unknown_evidence_rate"], 4),
            "entity_drift_rate": round(support_ledger["entity_drift_rate"], 4),
            "entity_persistence_failure": round(support_ledger["entity_persistence_failure"], 4),
            "forbidden_motif_rate": round(support_ledger["forbidden_motif_rate"], 4),
            "hard_no_drift_violation": round(support_ledger["hard_no_drift_violation"], 4),
            "support_ledger_hallucination_raw": round(ledger_h_raw, 4),
            "support_gate": round(support_gate, 4),
            "narrative_grounding": round(support_gate, 4),
            "claim_evidence_mismatch": round(claim_evidence_mismatch, 4),
            "constraint_gate": round(constraint_gate, 4),
            "constraint_level": round(constraint_level_score, 4),
            "constraint_degradation": round(constraint_degradation, 4),
            "causal_callback_rate": round(callback_rate, 4),
            "causal_callback_failure": round(callback_failure, 4),
            "fact_grounding": round(1.0 - h_raw, 4),
            "grounded_turn_quality": v3_story["grounded_turn_quality"],
            "turn_rarity_v3": v3_story["turn_rarity_v3"],
            "turn_support_gmean": v3_story["turn_support_gmean"],
            "turn_hard_gate": v3_story["turn_hard_gate"],
            "constraint_transform": v3_story["constraint_transform"],
            "multi_fact_synthesis": v3_story["multi_fact_synthesis"],
            "style_specificity_v3": v3_story["style_specificity_v3"],
            "causal_payoff": v3_story["causal_payoff"],
            "top3_scene_specificity": v3_story["top3_scene_specificity"],
            "arc_diversity_eff": v3_story["arc_diversity_eff"],
            "hard_valid_ledger_ratio": v3_story["hard_valid_ledger_ratio"],
            "common_bank_coverage": v3_story["common_bank_coverage"],
            "payoff_evidence_coverage": v3_story["payoff_evidence_coverage"],
            "required_fact_coverage": v3_story["required_fact_coverage"],
            "entity_alias_false_drift_rate": v3_story["entity_alias_false_drift_rate"],
            "unsupported_physical_claim_rate": v3_story["unsupported_physical_claim_rate"],
            "permitted_scene_invention_rate": v3_story["permitted_scene_invention_rate"],
        }

        task_result = {
            "version": GCW_VERSION,
            "task_id": task.get("id"),
            "calibration_policy": GCW_V3_CALIBRATION_POLICY,
            "runtime_scoring_policy": GCW_V3_RUNTIME_SCORING_POLICY,
            "common_story_bank_version": (
                self.common_story_bank_v3.get("version") if isinstance(self.common_story_bank_v3, dict) else None
            ),
            "entity_alias_bank_version": (
                self.entity_aliases_v3.get("version") if isinstance(self.entity_aliases_v3, dict) else None
            ),
            "score": round(imagination, 4),
            "imagination": round(imagination, 4),
            "hallucination": round(hallucination, 4),
            "imagination_raw": round(imagination_raw, 4),
            "imagination_gated": round(imagination_gated, 4),
            "hallucination_raw": round(h_raw, 4),
            "imagination_simple": round(imagination_simple, 4),
            "hallucination_simple": round(hallucination_simple, 4),
            "gcw_ttcw_proxy": {
                "F_story": round(f_story, 4),
                "X_story": round(x_story, 4),
                "O_story": round(o_story, 4),
                "E_story": round(e_story, 4),
            },
            "gcw_fact_grounding": round(1.0 - h_raw, 4),
            "primitive_means": primitive_means,
            "constraint_level": constraint_profile.get("level_id"),
            "constraint_profile": {
                key: value for key, value in constraint_profile.items()
                if key in {
                    "level_id",
                    "selected_level",
                    "constraint_level",
                    "label",
                    "required_fact_coverage",
                    "entity_persistence",
                    "require_evidence_ids",
                    "require_claim_ledger",
                    "required_causal_callback",
                    "hard_no_drift",
                    "required_causal_callback_terms",
                }
            },
            "support_ledger": support_ledger,
            "details": {
                "fluency": fluency,
                "flexibility": flexibility,
                "originality": originality,
                "elaboration": elaboration,
                "hallucination": hallucination_parts,
                "support_ledger": support_ledger,
                "gcw_v3": v3_story,
                "constraint": {
                "support_gate": round(support_gate, 4),
                "narrative_grounding": round(support_gate, 4),
                "claim_evidence_mismatch": round(claim_evidence_mismatch, 4),
                "constraint_gate": round(constraint_gate, 4),
                    "constraint_degradation": round(constraint_degradation, 4),
                    "causal_callback_hits": callback_hits,
                },
            },
            "residualization": {
                "zI": round(z_i, 6),
                "zH": round(z_h, 6),
                "I_resid_z": round(i_resid_z, 6),
                "H_resid_z": round(h_resid_z, 6),
                "beta_IH": self.beta_ih,
                "beta_HI": self.beta_hi,
                "standardization": {
                    **standardization,
                },
                "source": GCW_V3_CALIBRATION_POLICY,
            },
            "formula": {
                "F_story": "mean(schema_validity, beat_completeness, pacing_balance, ending_callback)",
                "X_story": "0.45*semantic_beat_diversity + 0.30*entity_coverage + 0.25*action_interiority_balance",
                "O_story": "0.40*reference_bank_distance + 0.30*non_cliche_motif_score + 0.30*twist_distance",
                "E_story": "0.30*sensory_specificity + 0.25*setting_anchoring + 0.25*character_state_change + 0.20*subtext_marker",
                "imagination_raw": "T4- 0.40*grounded_turn_quality + 0.20*causal_payoff + 0.15*top3_scene_specificity + 0.15*arc_diversity_eff + 0.10*hard_valid_ledger_ratio",
                "grounded_turn_quality": "rarity^1.35 * support_gmean^1.35 * hard_gate * (0.45 + 0.25*constraint_transform + 0.20*multi_fact_synthesis + 0.10*style_specificity)",
                "imagination_gated": "imagination_raw * (0.40+0.30*support_gate+0.30*constraint_gate)",
                "hallucination_raw": "0.58*closed_world_h + 0.42*support_ledger_h",
                "support_ledger_h": "0.24*unsupported_span + 0.24*contradicted_claim + 0.18*citation_mismatch + 0.14*entity_drift + 0.10*missing_evidence + 0.10*hard_no_drift",
                "residual": "I=clip01(0.50+0.20*(zI_gated-beta_IH*zH)); H=clip01(0.50+0.20*(zH-beta_HI*zI_gated))",
            },
            "scored_beats": len(beats),
            "expected_beats": expected_beat_count,
        }
        task_result["subtype_contributions"] = build_gcw_task_subtype_contributions(task_result)
        task_result["atom_signals"] = task_result["subtype_contributions"].get("atom_signals", {})

        return task_result

    def _full_story_text(self, parsed_response: Dict[str, object], beats: Sequence[Dict[str, object]]) -> str:
        chunks = [parsed_response.get("title") or ""]
        chunks.append(parsed_response.get("grounded_turn") or "")
        chunks.append(parsed_response.get("constraint_strategy") or "")
        for payoff in parsed_response.get("payoff_ledger") or []:
            if isinstance(payoff, dict):
                chunks.append(payoff.get("payoff") or "")
        chunks.extend(beat.get("paragraph") or "" for beat in beats)
        chunks.extend(beat.get("causal_function") or "" for beat in beats)
        chunks.append(parsed_response.get("ending_callback") or "")
        chunks.extend(parsed_response.get("style_devices") or [])
        for beat in beats:
            chunks.extend(beat.get("claimed_new_facts") or [])
        return " ".join(chunk for chunk in chunks if chunk)

    def _fluency_scores(self, parsed_response: Dict[str, object], beats: Sequence[Dict[str, object]], expected_beat_count: int) -> Dict[str, float]:
        parse_score = 1.0 if parsed_response.get("parse_valid") else 0.0
        title_score = 1.0 if parsed_response.get("title") else 0.0
        beat_count_score = clip01(len(beats) / max(1, expected_beat_count))
        beat_field_scores = []
        lengths = []
        for beat in beats:
            paragraph = beat.get("paragraph") or ""
            word_count = len(paragraph.split())
            lengths.append(word_count)
            paragraph_score = 1.0 if 25 <= word_count <= 90 else 0.65 if 12 <= word_count <= 120 else 0.25
            fact_score = 1.0 if beat.get("used_fact_ids") else 0.35
            entity_score = 1.0 if (beat.get("characters") or beat.get("places") or beat.get("objects")) else 0.35
            beat_field_scores.append((paragraph_score + fact_score + entity_score) / 3.0)
        schema_validity = clip01((parse_score + title_score + beat_count_score) / 3.0)
        beat_completeness = clip01(0.45 * beat_count_score + 0.55 * (mean_or_none(beat_field_scores) or 0.0))
        if lengths:
            mean_len = sum(lengths) / len(lengths)
            variance = sum((value - mean_len) ** 2 for value in lengths) / len(lengths)
            balance = 1.0 - min(1.0, math.sqrt(variance) / max(1.0, mean_len))
            pacing_balance = clip01(0.65 * balance + 0.35 * beat_count_score)
        else:
            pacing_balance = 0.0
        ending_text = parsed_response.get("ending_callback") or ""
        last_paragraph = beats[-1].get("paragraph") if beats else ""
        ending_overlap = _jaccard_similarity(ending_text, last_paragraph) if ending_text and last_paragraph else 0.0
        ending_callback = clip01((1.0 if ending_text else 0.0) * (0.65 + 0.35 * min(1.0, ending_overlap * 4.0)))
        return {
            "schema_validity": round(schema_validity, 4),
            "beat_completeness": round(beat_completeness, 4),
            "pacing_balance": round(pacing_balance, 4),
            "ending_callback": round(ending_callback, 4),
            "beat_lengths": lengths,
        }

    def _flexibility_scores(self, task, beats, paragraphs, *, semantic_scorer=None) -> Dict[str, float]:
        semantic_diversity = _pairwise_semantic_distance(semantic_scorer, paragraphs)
        declared_entities = []
        for beat in beats:
            declared_entities.extend(beat.get("characters") or [])
            declared_entities.extend(beat.get("places") or [])
            declared_entities.extend(beat.get("objects") or [])
        allowed = task.get("allowed_entities") or []
        entity_coverage = len({_normalize_text(item) for item in declared_entities if item}) / max(1, min(len(allowed), 8))
        entity_coverage = clip01(entity_coverage)
        full_text = " ".join(paragraphs)
        action_hits = _keyword_hits(full_text, self.common_bank.get("action_terms") or [])
        emotion_hits = _keyword_hits(full_text, self.common_bank.get("emotion_terms") or [])
        action_density = clip01(len(action_hits) / max(3, len(beats)))
        emotion_density = clip01(len(emotion_hits) / max(3, len(beats)))
        action_interiority_balance = clip01(0.50 * min(action_density, emotion_density) + 0.25 * action_density + 0.25 * emotion_density)
        return {
            "semantic_beat_diversity": round(semantic_diversity, 4),
            "entity_coverage": round(entity_coverage, 4),
            "action_interiority_balance": round(action_interiority_balance, 4),
            "action_hits": action_hits,
            "emotion_hits": emotion_hits,
        }

    def _originality_scores(self, task, parsed_response, paragraphs, full_text, *, semantic_scorer=None) -> Dict[str, float]:
        references = []
        references.extend(task.get("reference_bank") or [])
        references.extend(self.common_bank.get("global_reference_plots") or [])
        similarities = _semantic_similarities(semantic_scorer, full_text, references)
        max_similarity = max(similarities) if similarities else 0.0
        reference_distance = clip01(1.0 - max_similarity)
        cliche_hits = _keyword_hits(full_text, self.common_bank.get("cliche_markers") or [])
        non_cliche = clip01(1.0 - len(cliche_hits) / 3.0)
        first_half = " ".join(paragraphs[:max(1, len(paragraphs) // 2)])
        final_text = " ".join(paragraphs[-2:]) + " " + (parsed_response.get("ending_callback") or "")
        turn_markers = _keyword_hits(final_text, ["instead", "but", "however", "because", "realizes", "not", "without"])
        final_distance = clip01(1.0 - _jaccard_similarity(first_half, final_text))
        twist_distance = clip01(0.75 * final_distance + 0.25 * min(1.0, len(turn_markers) / 2.0))
        return {
            "reference_bank_distance": round(reference_distance, 4),
            "non_cliche_motif_score": round(non_cliche, 4),
            "twist_distance": round(twist_distance, 4),
            "nearest_reference_similarity": round(max_similarity, 4),
            "cliche_hits": cliche_hits,
            "turn_markers": turn_markers,
        }

    def _elaboration_scores(self, task, parsed_response, beats, paragraphs, full_text) -> Dict[str, float]:
        sensory_hits = _keyword_hits(full_text, self.common_bank.get("sensory_terms") or [])
        sensory_specificity = clip01(len(sensory_hits) / max(4, len(beats)))
        place_entities = _entity_names_from_facts(task, "setting")
        declared_places = []
        for beat in beats:
            declared_places.extend(beat.get("places") or [])
        setting_hits = _keyword_hits(full_text + " " + " ".join(declared_places), place_entities + [item for item in task.get("allowed_entities") or [] if item])
        setting_anchoring = clip01(len(setting_hits) / max(2, min(5, len(task.get("allowed_entities") or []))))
        character_names = _entity_names_from_facts(task, "character")
        emotion_hits = _keyword_hits(full_text, self.common_bank.get("emotion_terms") or [])
        character_hits = _keyword_hits(full_text, character_names)
        claimed_fact_count = sum(len(beat.get("claimed_new_facts") or []) for beat in beats)
        character_state_change = clip01(0.45 * min(1.0, len(character_hits) / max(1, len(character_names))) + 0.35 * min(1.0, len(emotion_hits) / 3.0) + 0.20 * min(1.0, claimed_fact_count / max(1, len(beats))))
        subtext_hits = _keyword_hits(full_text, self.common_bank.get("subtext_markers") or [])
        style_devices = [device.lower() for device in parsed_response.get("style_devices") or []]
        style_hit = any(device in {"metaphor", "subtext", "symbol", "motif", "image", "imagery"} for device in style_devices)
        subtext_marker = clip01(0.65 * min(1.0, len(subtext_hits) / 2.0) + 0.35 * (1.0 if style_hit else 0.0))
        return {
            "sensory_specificity": round(sensory_specificity, 4),
            "setting_anchoring": round(setting_anchoring, 4),
            "character_state_change": round(character_state_change, 4),
            "subtext_marker": round(subtext_marker, 4),
            "sensory_hits": sensory_hits,
            "setting_hits": setting_hits,
            "emotion_hits": emotion_hits,
            "subtext_hits": subtext_hits,
        }

    def _hallucination_scores(self, task, parsed_response, beats, full_text) -> Dict[str, object]:
        fact_ids = {str(fact.get("id")) for fact in task.get("fact_sheet") or [] if fact.get("id")}
        required = {str(item) for item in task.get("required_facts") or []}
        allowed = {_normalize_text(item) for item in task.get("allowed_entities") or [] if item}
        supported = 0
        unsupported = 0
        unsupported_records = []
        used_fact_ids = set()

        for beat in beats:
            for fact_id in beat.get("used_fact_ids") or []:
                fact_id = str(fact_id)
                if fact_id in fact_ids:
                    supported += 1
                    used_fact_ids.add(fact_id)
                else:
                    unsupported += 1
                    unsupported_records.append({"kind": "unknown_fact_id", "value": fact_id, "beat_id": beat.get("beat_id")})
            for field in ["characters", "places", "objects"]:
                for entity in beat.get(field) or []:
                    entity_norm = _normalize_text(entity)
                    if entity_norm in allowed or any(entity_norm and (entity_norm in item or item in entity_norm) for item in allowed):
                        supported += 1
                    else:
                        unsupported += 1
                        unsupported_records.append({"kind": f"unsupported_{field}", "value": entity, "beat_id": beat.get("beat_id")})

        full_norm = _normalize_text(full_text)
        global_forbidden = list(self.patterns.get("global_forbidden_terms") or [])
        hard_patterns = list(self.patterns.get("hard_contradiction_patterns") or [])
        unsupported_major = list(self.patterns.get("unsupported_major_entity_markers") or [])
        card_forbidden = []
        constraint_count = 0
        violated_constraints = 0
        constraint_hits = []
        for constraint in task.get("constraint_sheet") or []:
            if constraint.get("severity") == "hard":
                constraint_count += 1
            terms = list(constraint.get("forbidden_terms") or [])
            card_forbidden.extend(terms)
            hits = [term for term in terms if _phrase_hit(full_norm, term)]
            if hits and constraint.get("severity") == "hard":
                violated_constraints += 1
            for hit in hits:
                constraint_hits.append({"constraint_id": constraint.get("id"), "term": hit})

        global_hits = [term for term in global_forbidden if _phrase_hit(full_norm, term)]
        pattern_hits = [term for term in hard_patterns if _phrase_hit(full_norm, term)]
        unsupported_text_hits = [term for term in unsupported_major if _phrase_hit(full_norm, term)]
        contradiction_hits = sorted(set(global_hits + pattern_hits + [hit["term"] for hit in constraint_hits]))
        contradiction_count = len(contradiction_hits)
        if unsupported_text_hits:
            unsupported += len(unsupported_text_hits)
            for hit in unsupported_text_hits:
                unsupported_records.append({"kind": "unsupported_major_text_marker", "value": hit, "beat_id": None})

        missing_required = sorted(required - used_fact_ids)
        checked_claims = max(1, supported + unsupported + contradiction_count)
        unsupported_rate = clip01(unsupported / checked_claims)
        contradiction_rate = clip01(contradiction_count / checked_claims)
        violation_rate = clip01(violated_constraints / max(1, constraint_count))
        missing_rate = clip01(len(missing_required) / max(1, len(required)))
        return {
            "supported_closed_claims": supported,
            "unsupported_closed_claims": unsupported,
            "contradictory_claims": contradiction_count,
            "checked_closed_claims": checked_claims,
            "unsupported_claim_rate": round(unsupported_rate, 4),
            "contradiction_rate": round(contradiction_rate, 4),
            "hard_constraint_violation_rate": round(violation_rate, 4),
            "missing_required_fact_rate": round(missing_rate, 4),
            "missing_required_facts": missing_required,
            "used_fact_ids": sorted(used_fact_ids),
            "unsupported_records": unsupported_records,
            "constraint_hits": constraint_hits,
            "global_forbidden_hits": global_hits,
            "hard_pattern_hits": pattern_hits,
            "unsupported_text_hits": unsupported_text_hits,
        }


def aggregate_gcw_model_axes(
    task_scores: Sequence[Dict[str, object]],
    *,
    gate_pass: bool = True,
) -> Dict[str, object]:
    if not task_scores:
        return {
            "version": GCW_VERSION,
            "calibration_policy": GCW_V3_CALIBRATION_POLICY,
            "runtime_scoring_policy": GCW_V3_RUNTIME_SCORING_POLICY,
            "score": None,
            "imagination": None,
            "hallucination": None,
            "imagination_raw": None,
            "imagination_gated": None,
            "hallucination_raw": None,
            "primitive_means": {},
            "subtype_contributions": mean_subtype_contributions([]),
            "task_count": 0,
            "coverage_gate_pass": False,
        }

    imagination_raw = mean_or_none([item.get("imagination_raw") for item in task_scores])
    imagination_gated = mean_or_none([item.get("imagination_gated") for item in task_scores])
    hallucination_raw = mean_or_none([item.get("hallucination_raw") for item in task_scores])
    imagination = mean_or_none([item.get("imagination") for item in task_scores]) if gate_pass else None
    hallucination = mean_or_none([item.get("hallucination") for item in task_scores]) if gate_pass else None

    primitive_fields = set()
    for score in task_scores:
        if isinstance(score.get("primitive_means"), dict):
            primitive_fields.update(score["primitive_means"].keys())
    primitive_means = {}
    for field in sorted(primitive_fields):
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
    diagnostic_fields = [
        "grounded_turn_quality",
        "causal_payoff",
        "top3_scene_specificity",
        "arc_diversity_eff",
        "hard_valid_ledger_ratio",
        "common_bank_coverage",
        "claim_support_precision",
        "citation_mismatch_rate",
        "hard_constraint_violation_rate",
        "unsupported_physical_claim_rate",
        "entity_alias_false_drift_rate",
        "permitted_scene_invention_rate",
    ]
    diagnostics = {}
    for field in diagnostic_fields:
        value = mean_or_none([
            score.get("primitive_means", {}).get(field)
            for score in task_scores
            if isinstance(score.get("primitive_means"), dict)
        ])
        if value is not None:
            diagnostics[field] = round(value, 4)

    return {
        "version": GCW_VERSION,
        "calibration_policy": GCW_V3_CALIBRATION_POLICY,
        "runtime_scoring_policy": GCW_V3_RUNTIME_SCORING_POLICY,
        "common_story_bank_version": next(
            (
                score.get("common_story_bank_version")
                for score in task_scores
                if isinstance(score, dict) and score.get("common_story_bank_version")
            ),
            None,
        ),
        "entity_alias_bank_version": next(
            (
                score.get("entity_alias_bank_version")
                for score in task_scores
                if isinstance(score, dict) and score.get("entity_alias_bank_version")
            ),
            None,
        ),
        "score": round(imagination, 4) if imagination is not None else None,
        "imagination": round(imagination, 4) if imagination is not None else None,
        "hallucination": round(hallucination, 4) if hallucination is not None else None,
        "imagination_raw": round(imagination_raw, 4) if imagination_raw is not None else None,
        "imagination_gated": round(imagination_gated, 4) if imagination_gated is not None else None,
        "hallucination_raw": round(hallucination_raw, 4) if hallucination_raw is not None else None,
        "primitive_means": primitive_means,
        **diagnostics,
        "subtype_contributions": subtype_contributions,
        "task_count": len(task_scores),
        "coverage_gate_pass": bool(gate_pass),
        "residualization": (task_scores[0] or {}).get("residualization"),
        "formula": (task_scores[0] or {}).get("formula"),
    }


__all__ = [
    "GCW_VERSION",
    "GCW_V3_CALIBRATION_POLICY",
    "GCW_V3_RUNTIME_SCORING_POLICY",
    "GroundedCreativeWritingScorer",
    "aggregate_gcw_model_axes",
    "load_gcw_common_story_bank",
    "load_gcw_entity_aliases",
    "load_gcw_v3_calibration_params",
    "get_gcw_common_story_bank_coverage",
    "get_gcw_entity_alias_coverage",
]
