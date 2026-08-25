
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from typed_axis_aggregation import (
    build_analogy_transfer_challenge_contributions,
    mean_subtype_contributions,
)


ANALOGY_TRANSFER_VERSION = "analogy_transfer_dual_axis"
DEFAULT_ANALOGY_TRANSFER_BETA_IH = 0.90
DEFAULT_ANALOGY_TRANSFER_BETA_HI = 0.30
ANALOGY_TRANSFER_V3_CALIBRATION_POLICY = "benchmark_default"
ANALOGY_TRANSFER_V3_RUNTIME_SCORING_POLICY = "fixed output-only parameters"
ANALOGY_TRANSFER_V3_TEST_VISIBILITY_POLICY = "hidden_gold_mappings"
ANALOGY_TRANSFER_V3_TASK_OVERLAY_VERSION = "analogy_transfer_task_overlay"
ANALOGY_COMMON_MAPPING_BANK_VERSION = "analogy_common_mapping_bank"
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_ANALOGY_TRANSFER_V3_PARAMS = {
    "rarity_gamma": 1.30,
    "structural_gamma": 1.45,
    "evidence_gamma": 1.25,
    "mapping_multiplier_weights": {
        "base": 0.35,
        "abstraction_depth": 0.15,
        "cross_domain_transform": 0.15,
        "relational_depth": 0.20,
        "cross_domain_distance": 0.05,
        "boundary_awareness": 0.10,
    },
    "task_aggregation_weights": {
        "top3_mapping_quality": 0.25,
        "elite_tail_top1": 0.25,
        "relational_depth": 0.20,
        "licensed_inference_quality": 0.15,
        "abstraction_diversity_eff": 0.05,
        "boundary_aware_valid_ratio": 0.10,
    },
    "rarity": {
        "hard_zero": 0.0,
        "broad_common_cap": 0.35,
        "supported_rare_floor": 0.72,
        "default_floor": 0.42,
    },
}

COMMON_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "can", "do",
    "does", "for", "from", "has", "have", "in", "into", "is", "it", "like",
    "not", "of", "on", "or", "only", "the", "their", "them", "this", "to",
    "with", "without",
}


def clip01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _json_load(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_analogy_transfer_v3_task_overlay() -> Dict[str, object]:
    return _json_load(DATA_DIR / "analogy_transfer_v3_task_overlay.json")


def load_analogy_common_mapping_bank() -> Dict[str, object]:
    return _json_load(DATA_DIR / "analogy_common_mapping_bank_v3.json")


def load_analogy_transfer_v3_calibration_params() -> Dict[str, object]:
    payload = _json_load(DATA_DIR / "analogy_transfer_scoring_config.json")
    params = copy.deepcopy(DEFAULT_ANALOGY_TRANSFER_V3_PARAMS)
    final_params = payload.get("final_params") if isinstance(payload, Mapping) else None
    if isinstance(final_params, Mapping):
        for key, value in final_params.items():
            if isinstance(value, dict) and isinstance(params.get(key), dict):
                merged = copy.deepcopy(params[key])
                merged.update(value)
                params[key] = merged
            else:
                params[key] = value
    return params


def _coverage_for_task_ids(payload: Mapping[str, object], task_ids: Sequence[object]) -> Dict[str, object]:
    tasks = payload.get("tasks") if isinstance(payload, Mapping) else None
    if not isinstance(tasks, Mapping):
        tasks = {}
    ids = [str(task_id) for task_id in task_ids if task_id]
    covered = [task_id for task_id in ids if isinstance(tasks.get(task_id), Mapping)]
    missing = [task_id for task_id in ids if task_id not in set(covered)]
    return {
        "version": payload.get("version") if isinstance(payload, Mapping) else None,
        "requested": len(ids),
        "covered": len(covered),
        "coverage": round(len(covered) / len(ids), 4) if ids else None,
        "missing_task_ids": missing,
    }


def get_analogy_transfer_task_overlay_coverage(task_ids: Sequence[object]) -> Dict[str, object]:
    return _coverage_for_task_ids(load_analogy_transfer_v3_task_overlay(), task_ids)


def get_analogy_common_mapping_bank_coverage(task_ids: Sequence[object]) -> Dict[str, object]:
    return _coverage_for_task_ids(load_analogy_common_mapping_bank(), task_ids)


def get_analogy_evidence_alias_coverage(task_ids: Sequence[object]) -> Dict[str, object]:
    payload = load_analogy_transfer_v3_task_overlay()
    tasks = payload.get("tasks") if isinstance(payload, Mapping) else {}
    ids = [str(task_id) for task_id in task_ids if task_id]
    covered = [
        task_id for task_id in ids
        if isinstance(tasks.get(task_id), Mapping) and isinstance(tasks[task_id].get("evidence_aliases"), Mapping)
    ]
    return {
        "version": payload.get("version") if isinstance(payload, Mapping) else None,
        "requested": len(ids),
        "covered": len(covered),
        "coverage": round(len(covered) / len(ids), 4) if ids else None,
        "missing_task_ids": [task_id for task_id in ids if task_id not in set(covered)],
    }


def _clean_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_text(value) -> str:
    text = _clean_string(value).lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value) -> List[str]:
    return [
        token
        for token in _normalize_text(value).split()
        if token and token not in COMMON_WORDS
    ]


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
    result = []
    for value in values:
        text = _clean_string(value)
        if not text:
            continue
        key = _normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _phrase_hit(text: object, phrase: object) -> bool:
    text_norm = f" {_normalize_text(text)} "
    phrase_norm = _normalize_text(phrase)
    if not phrase_norm:
        return False
    if f" {phrase_norm} " in text_norm:
        return True
    phrase_tokens = set(phrase_norm.split())
    text_tokens = set(text_norm.split())
    return bool(phrase_tokens) and phrase_tokens.issubset(text_tokens)


def _keyword_fraction(text: object, keywords: Sequence[object]) -> float:
    usable = [keyword for keyword in keywords if _clean_string(keyword)]
    if not usable:
        return 0.0
    hits = sum(1 for keyword in usable if _phrase_hit(text, keyword))
    return hits / len(usable)


def _gmean(values: Sequence[float], *, floor: float = 1e-6) -> float:
    usable = [clip01(value) for value in values]
    if not usable:
        return 0.0
    if any(value <= 0.0 for value in usable):
        return 0.0
    return clip01(math.exp(sum(math.log(max(floor, value)) for value in usable) / len(usable)))


def _top_mean(values: Sequence[float], n: int) -> float:
    usable = sorted([clip01(value) for value in values], reverse=True)
    if not usable:
        return 0.0
    return sum(usable[:n]) / min(n, len(usable))


def _strip_json_code_fence(raw_text: str) -> str:
    text = (raw_text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return text


def _extract_json_payload(raw_text: str):
    text = _strip_json_code_fence(raw_text)
    candidates = []
    if text:
        candidates.append(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start:end + 1]
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


class AnalogyTransferScorer:
    """Scores one AnalogyTransfer response against a closed analogy card."""

    def __init__(
        self,
        *,
        beta_ih: float = DEFAULT_ANALOGY_TRANSFER_BETA_IH,
        beta_hi: float = DEFAULT_ANALOGY_TRANSFER_BETA_HI,
    ):
        self.beta_ih = beta_ih
        self.beta_hi = beta_hi
        self.task_overlay_payload = load_analogy_transfer_v3_task_overlay()
        self.task_overlay = self.task_overlay_payload.get("tasks") if isinstance(self.task_overlay_payload.get("tasks"), dict) else {}
        self.common_mapping_bank_payload = load_analogy_common_mapping_bank()
        self.common_mapping_bank = self.common_mapping_bank_payload.get("tasks") if isinstance(self.common_mapping_bank_payload.get("tasks"), dict) else {}
        self.params = load_analogy_transfer_v3_calibration_params()

    def parse_response(self, raw_text: str) -> Dict[str, object]:
        payload = _extract_json_payload(raw_text)
        if isinstance(payload, dict):
            return {
                "parse_valid": True,
                "analogy_summary": _clean_string(payload.get("analogy_summary")),
                "candidate_abstractions": [
                    _clean_string(item)
                    for item in _as_list(payload.get("candidate_abstractions") or [])
                    if _clean_string(item)
                ],
                "mapping_ledger": [
                    self._normalize_mapping(item)
                    for item in _as_list(payload.get("mapping_ledger") or [])
                ],
                "mapping_chain": [
                    _clean_string(item)
                    for item in _as_list(payload.get("mapping_chain") or [])
                    if _clean_string(item)
                ],
                "transfer_tests": [
                    self._normalize_inference(item)
                    for item in _as_list(payload.get("transfer_tests") or [])
                ],
                "negative_transfer_tests": [
                    self._normalize_warning(item)
                    for item in _as_list(payload.get("negative_transfer_tests") or [])
                ],
                "transferred_inferences": [
                    self._normalize_inference(item)
                    for item in _as_list(payload.get("transferred_inferences") or [])
                ],
                "limits_of_analogy": [
                    self._normalize_limit(item)
                    for item in _as_list(payload.get("limits_of_analogy") or [])
                ],
                "unsupported_transfer_warnings": [
                    self._normalize_warning(item)
                    for item in _as_list(payload.get("unsupported_transfer_warnings") or [])
                ],
                "boundary_rationale": _clean_string(payload.get("boundary_rationale")),
                "confidence": payload.get("confidence"),
                "raw_payload": payload,
                "parse_error": None,
                "legacy_fallback": False,
            }
        fallback = _clean_string(raw_text)
        return {
            "parse_valid": bool(fallback),
            "analogy_summary": fallback,
            "candidate_abstractions": [],
            "mapping_ledger": [],
            "mapping_chain": [],
            "transfer_tests": [],
            "negative_transfer_tests": [],
            "transferred_inferences": [],
            "limits_of_analogy": [],
            "unsupported_transfer_warnings": [],
            "boundary_rationale": "",
            "confidence": None,
            "raw_payload": payload,
            "parse_error": None if fallback else "empty_response",
            "legacy_fallback": True,
        }

    def _normalize_mapping(self, item) -> Dict[str, object]:
        if isinstance(item, Mapping):
            return {
                "mapping_id": _clean_string(item.get("mapping_id") or item.get("id")),
                "source_evidence_ids": _dedupe_strings(item.get("source_evidence_ids") or []),
                "target_evidence_ids": _dedupe_strings(item.get("target_evidence_ids") or []),
                "evidence_ids": _dedupe_strings(item.get("evidence_ids") or []),
                "dimension": _clean_string(item.get("dimension")),
                "abstraction": _clean_string(item.get("abstraction")),
                "mapped_relation": _clean_string(item.get("mapped_relation") or item.get("relation")),
                "role_alignment": _clean_string(item.get("role_alignment")),
                "structural_bridge": _clean_string(item.get("structural_bridge") or item.get("causal_bridge")),
                "text": _clean_string(item.get("text") or item.get("rationale") or item),
            }
        return {
            "mapping_id": "",
            "source_evidence_ids": [],
            "target_evidence_ids": [],
            "evidence_ids": [],
            "dimension": "",
            "abstraction": "",
            "mapped_relation": "",
            "role_alignment": "",
            "structural_bridge": "",
            "text": _clean_string(item),
        }

    def _normalize_inference(self, item) -> Dict[str, object]:
        if isinstance(item, Mapping):
            return {
                "text": _clean_string(item.get("text") or item.get("inference") or item),
                "evidence_ids": _dedupe_strings(item.get("evidence_ids") or item.get("support_ids") or []),
            }
        return {"text": _clean_string(item), "evidence_ids": []}

    def _normalize_limit(self, item) -> Dict[str, object]:
        if isinstance(item, Mapping):
            return {
                "limit_id": _clean_string(item.get("limit_id") or item.get("id")),
                "text": _clean_string(item.get("text") or item.get("limit") or item),
                "evidence_ids": _dedupe_strings(item.get("evidence_ids") or []),
            }
        return {"limit_id": "", "text": _clean_string(item), "evidence_ids": []}

    def _normalize_warning(self, item) -> Dict[str, object]:
        if isinstance(item, Mapping):
            return {
                "transfer_id": _clean_string(item.get("transfer_id") or item.get("id")),
                "text": _clean_string(item.get("text") or item.get("warning") or item),
                "evidence_ids": _dedupe_strings(item.get("evidence_ids") or []),
            }
        return {"transfer_id": "", "text": _clean_string(item), "evidence_ids": []}

    def score_task(self, task: Mapping[str, object], parsed_response: Mapping[str, object]) -> Dict[str, object]:
        task = self._task_with_overlay(task)
        cluster = task.get("cluster") or task
        mapping_record = self._score_mappings(task, cluster, parsed_response)
        evidence_record = self._score_evidence(task, cluster, parsed_response)
        limit_record = self._score_limits(cluster, parsed_response)
        transfer_record = self._score_false_transfer(cluster, parsed_response)

        mapping_coverage = mapping_record["mapping_coverage"]
        relational_consistency = mapping_record["relational_consistency"]
        dimension_coverage = mapping_record["dimension_coverage"]
        abstraction_support = mapping_record["abstraction_support"]
        limit_awareness = limit_record["limit_awareness"]
        evidence_precision = evidence_record["evidence_precision"]
        evidence_recall = evidence_record["evidence_recall"]
        false_transfer_rate = transfer_record["false_transfer_rate"]
        unsupported_inference_rate = transfer_record["unsupported_inference_rate"]
        surface_stereotype_rate = self._surface_stereotype_rate(
            cluster,
            parsed_response,
            mapping_coverage,
            evidence_recall,
            abstraction_support,
        )
        unknown_evidence_penalty = 0.35 if evidence_record["unknown_evidence_rate"] > 0.0 else 0.0
        boundary_overclaim_rate = clip01(max(
            1.0 - limit_awareness,
            transfer_record["boundary_overclaim_rate"],
        ))
        invented_fact_rate = clip01(max(
            evidence_record["unknown_evidence_rate"],
            unknown_evidence_penalty,
            transfer_record["invented_fact_rate"],
            0.5 * false_transfer_rate,
        ))
        source_fact_error = clip01(max(
            evidence_record["unknown_source_evidence_rate"],
            unknown_evidence_penalty if evidence_record["unknown_source_evidence_rate"] > 0.0 else 0.0,
            transfer_record["source_fact_error"],
            0.5 * transfer_record["source_forbidden_rate"],
        ))
        target_fact_error = clip01(max(
            evidence_record["unknown_target_evidence_rate"],
            unknown_evidence_penalty if evidence_record["unknown_target_evidence_rate"] > 0.0 else 0.0,
            transfer_record["target_fact_error"],
            0.5 * transfer_record["target_forbidden_rate"],
        ))
        source_fact_accuracy = clip01(1.0 - source_fact_error)
        target_fact_accuracy = clip01(1.0 - target_fact_error)

        source_fact_gate = source_fact_accuracy
        target_fact_gate = target_fact_accuracy
        transfer_gate = clip01(1.0 - max(
            false_transfer_rate,
            unsupported_inference_rate,
            0.65 * surface_stereotype_rate,
        ))
        legacy_imagination_raw = clip01(
            0.30 * mapping_coverage +
            0.25 * relational_consistency +
            0.20 * dimension_coverage +
            0.15 * abstraction_support +
            0.10 * limit_awareness
        )
        legacy_imagination_gated = clip01(legacy_imagination_raw * source_fact_gate * target_fact_gate * transfer_gate)
        v3_record = self._score_v3_imagination(
            task,
            cluster,
            parsed_response,
            mapping_record=mapping_record,
            evidence_record=evidence_record,
            limit_record=limit_record,
            transfer_record=transfer_record,
            surface_stereotype_rate=surface_stereotype_rate,
        )
        if task.get("task_overlay_version") == ANALOGY_TRANSFER_V3_TASK_OVERLAY_VERSION:
            imagination_raw = v3_record["imagination_raw"]
            imagination_gated = clip01(imagination_raw * transfer_gate * source_fact_gate * target_fact_gate)
        else:
            imagination_raw = legacy_imagination_raw
            imagination_gated = legacy_imagination_gated

        h_false_transfer = clip01(max(false_transfer_rate, unsupported_inference_rate))
        h_fact = clip01(max(
            invented_fact_rate,
            1.0 - source_fact_accuracy,
            1.0 - target_fact_accuracy,
            evidence_record["unknown_evidence_rate"],
        ))
        h_logic = clip01(max(
            1.0 - (
                v3_record["structural_match_gmean"]
                if task.get("task_overlay_version") == ANALOGY_TRANSFER_V3_TASK_OVERLAY_VERSION else
                relational_consistency
            ),
            mapping_record["wrong_mapping_direction_rate"],
            mapping_record["dimension_inconsistency_rate"],
        ))
        h_context = clip01(max(
            1.0 - (
                v3_record["boundary_aware_valid_ratio"]
                if task.get("task_overlay_version") == ANALOGY_TRANSFER_V3_TASK_OVERLAY_VERSION else
                limit_awareness
            ),
            boundary_overclaim_rate,
            surface_stereotype_rate,
            transfer_record.get("warning_assertion_confusion_rate", 0.0),
        ))
        hallucination_raw = clip01(
            0.25 * h_false_transfer +
            0.30 * h_fact +
            0.25 * h_logic +
            0.20 * h_context
        )
        imagination = clip01(imagination_gated - self.beta_ih * hallucination_raw)
        hallucination = clip01(hallucination_raw - self.beta_hi * imagination_gated)

        primitive_means = {
            "mapping_coverage": round(mapping_coverage, 4),
            "relational_consistency": round(relational_consistency, 4),
            "dimension_coverage": round(dimension_coverage, 4),
            "abstraction_support": round(abstraction_support, 4),
            "limit_awareness": round(limit_awareness, 4),
            "source_fact_accuracy": round(source_fact_accuracy, 4),
            "target_fact_accuracy": round(target_fact_accuracy, 4),
            "evidence_precision": round(evidence_precision, 4),
            "evidence_recall": round(evidence_recall, 4),
            "false_transfer_rate": round(false_transfer_rate, 4),
            "invented_fact_rate": round(invented_fact_rate, 4),
            "unsupported_inference_rate": round(unsupported_inference_rate, 4),
            "surface_stereotype_rate": round(surface_stereotype_rate, 4),
            "unknown_evidence_rate": round(evidence_record["unknown_evidence_rate"], 4),
            "wrong_mapping_direction_rate": round(mapping_record["wrong_mapping_direction_rate"], 4),
            "dimension_inconsistency_rate": round(mapping_record["dimension_inconsistency_rate"], 4),
            "boundary_overclaim_rate": round(boundary_overclaim_rate, 4),
            "top3_mapping_quality": round(v3_record["top3_mapping_quality"], 4),
            "elite_tail_top1": round(v3_record["elite_tail_top1"], 4),
            "licensed_inference_quality": round(v3_record["licensed_inference_quality"], 4),
            "abstraction_diversity_eff": round(v3_record["abstraction_diversity_eff"], 4),
            "relational_depth": round(v3_record["relational_depth"], 4),
            "cross_domain_distance": round(v3_record["cross_domain_distance"], 4),
            "boundary_aware_valid_ratio": round(v3_record["boundary_aware_valid_ratio"], 4),
            "structural_match_gmean": round(v3_record["structural_match_gmean"], 4),
            "evidence_grounding": round(v3_record["evidence_grounding"], 4),
            "mapping_rarity": round(v3_record["mapping_rarity"], 4),
            "warning_assertion_confusion_rate": round(transfer_record.get("warning_assertion_confusion_rate", 0.0), 4),
            "H_false_transfer": round(h_false_transfer, 4),
            "H_fact": round(h_fact, 4),
            "H_logic": round(h_logic, 4),
            "H_context": round(h_context, 4),
        }
        result = {
            "version": ANALOGY_TRANSFER_VERSION,
            "task_id": task.get("id"),
            "cluster_id": cluster.get("cluster_id"),
            "variant": task.get("variant"),
            "calibration_policy": ANALOGY_TRANSFER_V3_CALIBRATION_POLICY,
            "runtime_scoring_policy": ANALOGY_TRANSFER_V3_RUNTIME_SCORING_POLICY,
            "test_visibility_policy": ANALOGY_TRANSFER_V3_TEST_VISIBILITY_POLICY,
            "score": round(imagination, 4),
            "imagination": round(imagination, 4),
            "hallucination": round(hallucination, 4),
            "imagination_raw": round(imagination_raw, 4),
            "imagination_gated": round(imagination_gated, 4),
            "hallucination_raw": round(hallucination_raw, 4),
            "I_analogy_raw": round(imagination_raw, 4),
            "I_analogy_gated": round(imagination_gated, 4),
            "H_false_transfer": round(h_false_transfer, 4),
            "H_fact": round(h_fact, 4),
            "H_logic": round(h_logic, 4),
            "H_context": round(h_context, 4),
            "top3_mapping_quality": round(v3_record["top3_mapping_quality"], 4),
            "elite_tail_top1": round(v3_record["elite_tail_top1"], 4),
            "licensed_inference_quality": round(v3_record["licensed_inference_quality"], 4),
            "abstraction_diversity_eff": round(v3_record["abstraction_diversity_eff"], 4),
            "boundary_aware_valid_ratio": round(v3_record["boundary_aware_valid_ratio"], 4),
            "structural_match_gmean": round(v3_record["structural_match_gmean"], 4),
            "evidence_grounding": round(v3_record["evidence_grounding"], 4),
            "mapping_rarity": round(v3_record["mapping_rarity"], 4),
            "primitive_means": primitive_means,
            "v3_record": v3_record,
            "mapping_record": mapping_record,
            "evidence_record": evidence_record,
            "limit_record": limit_record,
            "transfer_record": transfer_record,
            "parsed_response": {
                key: value
                for key, value in parsed_response.items()
                if key != "raw_payload"
            },
            "formula": {
                "imagination_raw": "T8  I_raw=0.25*top3_mapping_quality+0.25*elite_tail_top1+0.20*relational_depth+0.15*licensed_inference_quality+0.05*abstraction_diversity_eff+0.10*boundary_aware_valid_ratio",
                "mapping_quality": "mapping_I=rarity^1.30*structural_match_gmean^1.45*evidence_grounding^1.25*hard_gate*(0.35+0.15*abstraction_depth+0.15*cross_domain_transform+0.20*relational_depth+0.05*cross_domain_distance+0.10*boundary_awareness)",
                "imagination_gated": "I_gated=I_raw*source_fact_gate*target_fact_gate*transfer_gate",
                "hallucination_raw": "H_raw=0.25*H_false_transfer+0.30*H_fact+0.25*H_logic+0.20*H_context",
                "residual": "I=clip01(I_gated-beta_IH*H_raw); H=clip01(H_raw-beta_HI*I_gated)",
            },
            "residualization": {
                "beta_IH": self.beta_ih,
                "beta_HI": self.beta_hi,
                "source": ANALOGY_TRANSFER_V3_CALIBRATION_POLICY,
                "standardization": "clip01_raw_v1",
            },
        }
        result["subtype_contributions"] = build_analogy_transfer_challenge_contributions(
            result,
            beta_ih=self.beta_ih,
            beta_hi=self.beta_hi,
        )
        result["atom_signals"] = result["subtype_contributions"].get("atom_signals", {})
        return result

    def _task_with_overlay(self, task: Mapping[str, object]) -> Dict[str, object]:
        task_copy = copy.deepcopy(dict(task))
        task_id = str(task_copy.get("id") or "")
        overlay = self.task_overlay.get(task_id)
        if isinstance(overlay, Mapping):
            task_copy["analogy_v3_overlay"] = copy.deepcopy(dict(overlay))
            task_copy["task_overlay_version"] = ANALOGY_TRANSFER_V3_TASK_OVERLAY_VERSION
            task_copy["test_visibility_policy"] = ANALOGY_TRANSFER_V3_TEST_VISIBILITY_POLICY
        return task_copy

    def _score_v3_imagination(
        self,
        task: Mapping[str, object],
        cluster: Mapping[str, object],
        parsed_response: Mapping[str, object],
        *,
        mapping_record: Mapping[str, object],
        evidence_record: Mapping[str, object],
        limit_record: Mapping[str, object],
        transfer_record: Mapping[str, object],
        surface_stereotype_rate: float,
    ) -> Dict[str, object]:
        mapping_quality_records = self._mapping_quality_records(
            task,
            cluster,
            parsed_response,
            mapping_record=mapping_record,
            evidence_record=evidence_record,
            limit_record=limit_record,
            transfer_record=transfer_record,
            surface_stereotype_rate=surface_stereotype_rate,
        )
        qualities = [record["mapping_quality"] for record in mapping_quality_records]
        top3_mapping_quality = _top_mean(qualities, 3)
        elite_tail_top1 = max(qualities) if qualities else 0.0
        licensed_inference_quality = self._licensed_inference_quality(task, parsed_response)
        abstraction_diversity_eff = self._abstraction_diversity_eff(parsed_response, mapping_quality_records)
        relational_depth = mean_or_none(record.get("relational_depth") for record in mapping_quality_records) or 0.0
        cross_domain_distance = mean_or_none(record.get("cross_domain_distance") for record in mapping_quality_records) or 0.0
        boundary_aware_valid_ratio = clip01(_gmean([
            limit_record.get("limit_awareness", 0.0),
            1.0 - transfer_record.get("false_transfer_rate", 0.0),
            1.0 - transfer_record.get("warning_assertion_confusion_rate", 0.0),
        ]))
        weights = self.params.get("task_aggregation_weights") or {}
        imagination_raw = clip01(
            float(weights.get("top3_mapping_quality", 0.25)) * top3_mapping_quality +
            float(weights.get("elite_tail_top1", 0.25)) * elite_tail_top1 +
            float(weights.get("relational_depth", 0.20)) * relational_depth +
            float(weights.get("licensed_inference_quality", 0.15)) * licensed_inference_quality +
            float(weights.get("abstraction_diversity_eff", 0.05)) * abstraction_diversity_eff +
            float(weights.get("boundary_aware_valid_ratio", 0.10)) * boundary_aware_valid_ratio
        )
        return {
            "imagination_raw": imagination_raw,
            "top3_mapping_quality": top3_mapping_quality,
            "elite_tail_top1": elite_tail_top1,
            "licensed_inference_quality": licensed_inference_quality,
            "abstraction_diversity_eff": abstraction_diversity_eff,
            "relational_depth": relational_depth,
            "cross_domain_distance": cross_domain_distance,
            "boundary_aware_valid_ratio": boundary_aware_valid_ratio,
            "structural_match_gmean": mean_or_none(record["structural_match_gmean"] for record in mapping_quality_records) or 0.0,
            "evidence_grounding": mean_or_none(record["evidence_grounding"] for record in mapping_quality_records) or 0.0,
            "mapping_rarity": mean_or_none(record["rarity"] for record in mapping_quality_records) or 0.0,
            "mapping_quality_records": mapping_quality_records,
            "common_mapping_bank_version": ANALOGY_COMMON_MAPPING_BANK_VERSION,
        }

    def _mapping_quality_records(
        self,
        task: Mapping[str, object],
        cluster: Mapping[str, object],
        parsed_response: Mapping[str, object],
        *,
        mapping_record: Mapping[str, object],
        evidence_record: Mapping[str, object],
        limit_record: Mapping[str, object],
        transfer_record: Mapping[str, object],
        surface_stereotype_rate: float,
    ) -> List[Dict[str, object]]:
        required_ids = set(task.get("required_mapping_ids") or [])
        gold_mappings = [
            mapping for mapping in cluster.get("gold_mappings") or []
            if not required_ids or mapping.get("mapping_id") in required_ids
        ]
        matches = mapping_record.get("matches") if isinstance(mapping_record.get("matches"), Mapping) else {}
        records = []
        for gold in gold_mappings:
            mapping_id = gold.get("mapping_id")
            match = matches.get(mapping_id) if isinstance(matches, Mapping) else None
            if not isinstance(match, Mapping):
                records.append({
                    "mapping_id": mapping_id,
                    "mapping_quality": 0.0,
                    "structural_match_gmean": 0.0,
                    "evidence_grounding": 0.0,
                    "rarity": 0.0,
                    "hard_gate": 0.0,
                    "matched": False,
                })
                continue
            returned = match.get("returned") if isinstance(match.get("returned"), Mapping) else {}
            record = match.get("record") if isinstance(match.get("record"), Mapping) else {}
            relation_text = " ".join([
                _clean_string(returned.get("mapped_relation")),
                _clean_string(returned.get("text")),
                _clean_string(returned.get("abstraction")),
                _clean_string(returned.get("role_alignment")),
                _clean_string(returned.get("structural_bridge")),
                " ".join(_clean_string(item) for item in parsed_response.get("mapping_chain") or []),
            ])
            source_relation_match = max(
                clip01(record.get("source_relation_hit", 0.0)),
                _keyword_fraction(relation_text, _tokens(gold.get("source_relation"))),
            )
            target_relation_match = max(
                clip01(record.get("target_relation_hit", 0.0)),
                _keyword_fraction(relation_text, _tokens(gold.get("target_relation"))),
            )
            role_alignment = clip01(max(
                record.get("dimension_hit", 0.0),
                _keyword_fraction(relation_text, _tokens(gold.get("dimension"))),
                0.5 * (record.get("source_overlap", 0.0) + record.get("target_overlap", 0.0)),
            ))
            causal_functional_alignment = max(
                clip01(record.get("abstraction_support", 0.0)),
                _keyword_fraction(relation_text, _tokens(gold.get("abstraction"))),
            )
            direction_correctness = 0.0 if record.get("wrong_direction") else 1.0
            structural_match_gmean = _gmean([
                source_relation_match,
                target_relation_match,
                role_alignment,
                causal_functional_alignment,
                direction_correctness,
            ])
            evidence_grounding = _gmean([
                record.get("source_overlap", 0.0),
                record.get("target_overlap", 0.0),
                evidence_record.get("evidence_precision", 0.0),
                evidence_record.get("evidence_recall", 0.0),
            ])
            rarity, rarity_record = self._mapping_rarity(task, parsed_response, returned)
            abstraction_depth = self._abstraction_depth(returned, parsed_response)
            cross_domain_transform = clip01(max(
                0.5 * source_relation_match + 0.5 * target_relation_match,
                _keyword_fraction(relation_text, _tokens(gold.get("abstraction"))),
            ))
            relational_depth = self._relational_depth(returned, relation_text, gold)
            cross_domain_distance = self._cross_domain_distance(task, returned, relation_text)
            boundary_awareness = clip01(limit_record.get("limit_awareness", 0.0))
            hard_gate = 1.0
            if record.get("wrong_direction"):
                hard_gate = 0.0
            if evidence_record.get("unknown_evidence_rate", 0.0) > 0.0:
                hard_gate = min(hard_gate, 0.25)
            if transfer_record.get("false_transfer_rate", 0.0) > 0.0:
                hard_gate = min(hard_gate, 0.25)
            if surface_stereotype_rate >= 0.65:
                hard_gate = min(hard_gate, 0.35)
            weights = self.params.get("mapping_multiplier_weights") or {}
            multiplier = clip01(
                float(weights.get("base", 0.35)) +
                float(weights.get("abstraction_depth", 0.15)) * abstraction_depth +
                float(weights.get("cross_domain_transform", 0.15)) * cross_domain_transform +
                float(weights.get("relational_depth", 0.20)) * relational_depth +
                float(weights.get("cross_domain_distance", 0.05)) * cross_domain_distance +
                float(weights.get("boundary_awareness", 0.10)) * boundary_awareness
            )
            mapping_quality = clip01(
                math.pow(clip01(rarity), float(self.params.get("rarity_gamma", 1.30))) *
                math.pow(clip01(structural_match_gmean), float(self.params.get("structural_gamma", 1.45))) *
                math.pow(clip01(evidence_grounding), float(self.params.get("evidence_gamma", 1.25))) *
                hard_gate *
                multiplier
            )
            records.append({
                "mapping_id": mapping_id,
                "mapping_quality": mapping_quality,
                "structural_match_gmean": structural_match_gmean,
                "evidence_grounding": evidence_grounding,
                "rarity": rarity,
                "rarity_record": rarity_record,
                "abstraction_depth": abstraction_depth,
                "cross_domain_transform": cross_domain_transform,
                "relational_depth": relational_depth,
                "cross_domain_distance": cross_domain_distance,
                "boundary_awareness": boundary_awareness,
                "hard_gate": hard_gate,
                "matched": True,
            })
        return records

    def _bank_for_task(self, task: Mapping[str, object]) -> Mapping[str, object]:
        bank = self.common_mapping_bank.get(str(task.get("id") or ""))
        return bank if isinstance(bank, Mapping) else {}

    def _family_hit(self, family: Mapping[str, object], text: str) -> bool:
        keywords = [str(item).lower() for item in _as_list(family.get("keywords")) if str(item).strip()]
        return bool(keywords) and any(keyword in text for keyword in keywords)

    def _mapping_rarity(
        self,
        task: Mapping[str, object],
        parsed_response: Mapping[str, object],
        returned_mapping: Mapping[str, object],
    ) -> Tuple[float, Dict[str, object]]:
        rarity_cfg = self.params.get("rarity") or {}
        text = " ".join([
            _clean_string(returned_mapping.get("mapped_relation")),
            _clean_string(returned_mapping.get("abstraction")),
            _clean_string(returned_mapping.get("role_alignment")),
            _clean_string(returned_mapping.get("structural_bridge")),
            " ".join(_clean_string(item) for item in parsed_response.get("candidate_abstractions") or []),
        ]).lower()
        bank = self._bank_for_task(task)
        hard_hits = [
            str(family.get("id") or "hard_zero")
            for family in bank.get("hard_zero_mapping_families") or []
            if isinstance(family, Mapping) and self._family_hit(family, text)
        ]
        if hard_hits:
            base_rarity = clip01(rarity_cfg.get("hard_zero", 0.0))
            rarity = self._phase_a_rarity_transform(base_rarity)
            return rarity, {
                "rarity_class": "hard_zero",
                "base_rarity": round(base_rarity, 4),
                "phase_a_smoothed": True,
                "matched_families": sorted(set(hard_hits)),
            }
        rare_hits = [
            str(family.get("id") or "supported_rare")
            for family in bank.get("supported_rare_abstraction_families") or []
            if isinstance(family, Mapping) and self._family_hit(family, text)
        ]
        if rare_hits:
            base_rarity = clip01(max(float(rarity_cfg.get("supported_rare_floor", 0.72)), 0.86))
            rarity = self._phase_a_rarity_transform(base_rarity)
            return rarity, {
                "rarity_class": "supported_rare",
                "base_rarity": round(base_rarity, 4),
                "phase_a_smoothed": True,
                "matched_families": sorted(set(rare_hits)),
            }
        broad_hits = [
            str(family.get("id") or "broad_common")
            for family in bank.get("broad_common_mapping_families") or []
            if isinstance(family, Mapping) and self._family_hit(family, text)
        ]
        if broad_hits:
            base_rarity = clip01(float(rarity_cfg.get("broad_common_cap", 0.35)))
            rarity = self._phase_a_rarity_transform(base_rarity)
            return rarity, {
                "rarity_class": "broad_common",
                "base_rarity": round(base_rarity, 4),
                "phase_a_smoothed": True,
                "matched_families": sorted(set(broad_hits)),
            }
        abstraction_tokens = [
            token for token in _tokens(returned_mapping.get("abstraction"))
            if len(token) >= 5
        ]
        base_rarity = clip01(float(rarity_cfg.get("default_floor", 0.42)) + 0.08 * min(3, len(set(abstraction_tokens))))
        rarity = self._phase_a_rarity_transform(base_rarity)
        return rarity, {
            "rarity_class": "unmatched_default",
            "base_rarity": round(base_rarity, 4),
            "phase_a_smoothed": True,
            "matched_families": [],
        }

    def _phase_a_rarity_transform(self, value: float) -> float:
        base = clip01(value)
        return clip01(0.10 + 1.20 * base - 0.30 * base * base)

    def _abstraction_depth(self, returned_mapping: Mapping[str, object], parsed_response: Mapping[str, object]) -> float:
        text = " ".join([
            _clean_string(returned_mapping.get("abstraction")),
            _clean_string(returned_mapping.get("mapped_relation")),
            _clean_string(returned_mapping.get("role_alignment")),
            _clean_string(returned_mapping.get("structural_bridge")),
            " ".join(_clean_string(item) for item in parsed_response.get("candidate_abstractions") or []),
            " ".join(_clean_string(item) for item in parsed_response.get("mapping_chain") or []),
        ])
        tokens = [token for token in _tokens(text) if len(token) >= 5]
        generic = {"same", "similar", "both", "thing", "system", "works"}
        rich = [token for token in tokens if token not in generic]
        return clip01(len(set(rich)) / 8.0)

    def _relational_depth(
        self,
        returned_mapping: Mapping[str, object],
        relation_text: str,
        gold_mapping: Mapping[str, object],
    ) -> float:
        text = _normalize_text(" ".join([
            relation_text,
            _clean_string(returned_mapping.get("structural_bridge")),
            _clean_string(returned_mapping.get("role_alignment")),
        ]))
        relation_terms = {
            "causes", "enables", "prevents", "controls", "regulates", "preserves",
            "releases", "stores", "indexes", "maps", "aligns", "transfers",
            "constrains", "routes", "signals", "supports", "responds", "feedback",
            "function", "role", "process", "relation", "because", "therefore",
        }
        surface_terms = {"looks", "shape", "color", "size", "both", "similar", "same", "thing", "object"}
        tokens = set(_tokens(text))
        relation_score = clip01(len(tokens & relation_terms) / 4.0)
        surface_penalty = clip01(len(tokens & surface_terms) / 4.0)
        gold_relation_score = clip01(
            0.5 * _keyword_fraction(text, _tokens(gold_mapping.get("source_relation"))) +
            0.5 * _keyword_fraction(text, _tokens(gold_mapping.get("target_relation")))
        )
        role_score = 1.0 if _clean_string(returned_mapping.get("role_alignment")) else 0.0
        return clip01(0.35 * relation_score + 0.35 * gold_relation_score + 0.20 * role_score + 0.10 * (1.0 - surface_penalty))

    def _cross_domain_distance(
        self,
        task: Mapping[str, object],
        returned_mapping: Mapping[str, object],
        relation_text: str,
    ) -> float:
        source_tokens = set(_tokens(task.get("source_domain") or ""))
        target_tokens = set(_tokens(task.get("target_domain") or ""))
        if not source_tokens and isinstance(task.get("cluster"), Mapping):
            source_tokens = set(_tokens((task.get("cluster") or {}).get("source_domain") or ""))
        if not target_tokens and isinstance(task.get("cluster"), Mapping):
            target_tokens = set(_tokens((task.get("cluster") or {}).get("target_domain") or ""))
        if not source_tokens or not target_tokens:
            domain_distance = 0.5
        else:
            domain_distance = 1.0 - (len(source_tokens & target_tokens) / max(1, len(source_tokens | target_tokens)))
        text_tokens = set(_tokens(relation_text))
        mentions_both = 1.0 if (text_tokens & source_tokens and text_tokens & target_tokens) else 0.0
        abstraction_tokens = {
            token for token in _tokens(returned_mapping.get("abstraction"))
            if len(token) >= 5 and token not in source_tokens and token not in target_tokens
        }
        abstraction_score = clip01(len(abstraction_tokens) / 5.0)
        return clip01(0.40 * domain_distance + 0.30 * mentions_both + 0.30 * abstraction_score)

    def _licensed_inference_quality(self, task: Mapping[str, object], parsed_response: Mapping[str, object]) -> float:
        overlay = task.get("analogy_v3_overlay") if isinstance(task.get("analogy_v3_overlay"), Mapping) else {}
        targets = overlay.get("licensed_inference_targets") if isinstance(overlay, Mapping) else None
        if not targets:
            return 0.0
        inferences = [
            item for item in list(parsed_response.get("transferred_inferences") or []) + list(parsed_response.get("transfer_tests") or [])
            if isinstance(item, Mapping)
        ]
        scores = []
        for target in targets:
            required_ids = {str(item) for item in target.get("target_evidence_ids") or []}
            keywords = target.get("keywords") or []
            best = 0.0
            for inference in inferences:
                text = _clean_string(inference.get("text"))
                ids = set(str(item) for item in inference.get("evidence_ids") or [])
                evidence_score = len(required_ids & ids) / max(1, len(required_ids))
                keyword_score = _keyword_fraction(text, keywords)
                best = max(best, clip01(0.60 * evidence_score + 0.40 * keyword_score))
            scores.append(best)
        return sum(scores) / len(scores) if scores else 0.0

    def _abstraction_diversity_eff(
        self,
        parsed_response: Mapping[str, object],
        mapping_quality_records: Sequence[Mapping[str, object]],
    ) -> float:
        abstractions = []
        for item in parsed_response.get("mapping_ledger") or []:
            if isinstance(item, Mapping) and item.get("abstraction"):
                abstractions.append(_normalize_text(item.get("abstraction")))
        for item in parsed_response.get("candidate_abstractions") or []:
            abstractions.append(_normalize_text(item))
        unique = {item for item in abstractions if item}
        hard_valid = sum(1 for record in mapping_quality_records if record.get("hard_gate", 0.0) >= 1.0 and record.get("mapping_quality", 0.0) > 0.0)
        return clip01(min(len(unique), hard_valid or len(unique)) / 3.0)

    def _full_response_text(self, parsed_response: Mapping[str, object]) -> str:
        parts = [_clean_string(parsed_response.get("analogy_summary"))]
        parts.extend(_clean_string(item) for item in parsed_response.get("candidate_abstractions") or [])
        parts.extend(_clean_string(item) for item in parsed_response.get("mapping_chain") or [])
        for item in parsed_response.get("mapping_ledger") or []:
            if isinstance(item, Mapping):
                parts.extend([
                    _clean_string(item.get("text")),
                    _clean_string(item.get("abstraction")),
                    _clean_string(item.get("mapped_relation")),
                    _clean_string(item.get("dimension")),
                    _clean_string(item.get("role_alignment")),
                    _clean_string(item.get("structural_bridge")),
                ])
        for item in parsed_response.get("transfer_tests") or []:
            if isinstance(item, Mapping):
                parts.append(_clean_string(item.get("text")))
        for item in parsed_response.get("transferred_inferences") or []:
            if isinstance(item, Mapping):
                parts.append(_clean_string(item.get("text")))
        for item in parsed_response.get("limits_of_analogy") or []:
            if isinstance(item, Mapping):
                parts.append(_clean_string(item.get("text")))
        for item in parsed_response.get("negative_transfer_tests") or []:
            if isinstance(item, Mapping):
                parts.append(_clean_string(item.get("text")))
        for item in parsed_response.get("unsupported_transfer_warnings") or []:
            if isinstance(item, Mapping):
                parts.append(_clean_string(item.get("text")))
        parts.append(_clean_string(parsed_response.get("boundary_rationale")))
        return " ".join(part for part in parts if part)

    def _asserted_response_text(self, parsed_response: Mapping[str, object]) -> str:
        parts = [_clean_string(parsed_response.get("analogy_summary"))]
        for item in parsed_response.get("mapping_ledger") or []:
            if isinstance(item, Mapping):
                parts.extend([
                    _clean_string(item.get("text")),
                    _clean_string(item.get("abstraction")),
                    _clean_string(item.get("mapped_relation")),
                    _clean_string(item.get("role_alignment")),
                    _clean_string(item.get("structural_bridge")),
                ])
        for item in list(parsed_response.get("transferred_inferences") or []) + list(parsed_response.get("transfer_tests") or []):
            if isinstance(item, Mapping):
                parts.append(_clean_string(item.get("text")))
        return " ".join(part for part in parts if part)

    def _warning_response_text(self, parsed_response: Mapping[str, object]) -> str:
        parts = []
        for key in ("limits_of_analogy", "unsupported_transfer_warnings", "negative_transfer_tests"):
            for item in parsed_response.get(key) or []:
                if isinstance(item, Mapping):
                    parts.append(_clean_string(item.get("text")))
                else:
                    parts.append(_clean_string(item))
        parts.append(_clean_string(parsed_response.get("boundary_rationale")))
        return " ".join(part for part in parts if part)

    def _score_mappings(
        self,
        task: Mapping[str, object],
        cluster: Mapping[str, object],
        parsed_response: Mapping[str, object],
    ) -> Dict[str, object]:
        required_ids = set(task.get("required_mapping_ids") or [])
        gold = [
            mapping for mapping in cluster.get("gold_mappings") or []
            if not required_ids or mapping.get("mapping_id") in required_ids
        ]
        returned = [
            item for item in parsed_response.get("mapping_ledger") or []
            if isinstance(item, Mapping)
        ]
        if not gold:
            return {
                "required_mapping_ids": [],
                "matched_mapping_ids": [],
                "mapping_coverage": 1.0,
                "relational_consistency": 1.0,
                "dimension_coverage": 1.0,
                "abstraction_support": 1.0,
                "wrong_mapping_direction_rate": 0.0,
                "dimension_inconsistency_rate": 0.0,
            }

        matches = {}
        consistency_scores = []
        abstraction_scores = []
        wrong_direction = 0
        dimension_mismatch = 0
        used_returned = set()
        for gold_mapping in gold:
            best_index = None
            best_score = 0.0
            best_record = None
            for idx, item in enumerate(returned):
                if idx in used_returned:
                    continue
                score, record = self._mapping_match_score(gold_mapping, item)
                if score > best_score:
                    best_score = score
                    best_index = idx
                    best_record = record
            if best_index is not None and best_score >= 0.50:
                used_returned.add(best_index)
                matches[gold_mapping.get("mapping_id")] = {
                    "score": best_score,
                    "record": best_record,
                    "returned": returned[best_index],
                }
                consistency_scores.append(best_record["relational_consistency"])
                abstraction_scores.append(best_record["abstraction_support"])
                wrong_direction += 1 if best_record["wrong_direction"] else 0
                dimension_mismatch += 1 if best_record["dimension_mismatch"] else 0

        required_dimensions = {
            mapping.get("dimension")
            for mapping in gold
            if mapping.get("dimension")
        }
        matched_dimensions = {
            mapping.get("dimension")
            for mapping in gold
            if mapping.get("mapping_id") in matches and mapping.get("dimension")
        }
        mapping_coverage = len(matches) / len(gold)
        dimension_coverage = (
            len(matched_dimensions) / len(required_dimensions)
            if required_dimensions else mapping_coverage
        )
        relational_consistency = (
            sum(consistency_scores) / len(gold)
            if consistency_scores else 0.0
        )
        abstraction_support = (
            sum(abstraction_scores) / len(gold)
            if abstraction_scores else 0.0
        )
        unmatched_returned = max(0, len(returned) - len(used_returned))
        dimension_inconsistency = clip01(
            (dimension_mismatch + 0.5 * unmatched_returned) / max(1, len(returned))
        )
        return {
            "required_mapping_ids": [mapping.get("mapping_id") for mapping in gold],
            "matched_mapping_ids": sorted(matches.keys()),
            "mapping_coverage": clip01(mapping_coverage),
            "relational_consistency": clip01(relational_consistency),
            "dimension_coverage": clip01(dimension_coverage),
            "abstraction_support": clip01(abstraction_support),
            "wrong_mapping_direction_rate": clip01(wrong_direction / max(1, len(returned))),
            "dimension_inconsistency_rate": dimension_inconsistency,
            "matches": matches,
        }

    def _mapping_match_score(self, gold_mapping: Mapping[str, object], item: Mapping[str, object]) -> Tuple[float, Dict[str, object]]:
        gold_source = set(str(eid) for eid in gold_mapping.get("source_evidence_ids") or [])
        gold_target = set(str(eid) for eid in gold_mapping.get("target_evidence_ids") or [])
        returned_source = set(str(eid) for eid in item.get("source_evidence_ids") or [])
        returned_target = set(str(eid) for eid in item.get("target_evidence_ids") or [])
        returned_any = set(str(eid) for eid in item.get("evidence_ids") or [])
        returned_source |= {eid for eid in returned_any if "_S" in eid}
        returned_target |= {eid for eid in returned_any if "_T" in eid}
        source_overlap = len(gold_source & returned_source) / max(1, len(gold_source))
        target_overlap = len(gold_target & returned_target) / max(1, len(gold_target))
        relation_text = " ".join([
            _clean_string(item.get("mapped_relation")),
            _clean_string(item.get("text")),
            _clean_string(item.get("abstraction")),
            _clean_string(item.get("role_alignment")),
            _clean_string(item.get("structural_bridge")),
        ])
        source_relation_hit = 1.0 if _phrase_hit(relation_text, gold_mapping.get("source_relation")) else 0.0
        target_relation_hit = 1.0 if _phrase_hit(relation_text, gold_mapping.get("target_relation")) else 0.0
        dimension_hit = _normalize_text(item.get("dimension")) == _normalize_text(gold_mapping.get("dimension"))
        id_hit = bool(item.get("mapping_id") and item.get("mapping_id") == gold_mapping.get("mapping_id"))
        wrong_direction = bool((gold_source & returned_target) or (gold_target & returned_source))
        dimension_mismatch = bool(item.get("dimension")) and not dimension_hit
        relation_score = clip01(
            0.32 * source_overlap +
            0.32 * target_overlap +
            0.18 * source_relation_hit +
            0.18 * target_relation_hit
        )
        if id_hit:
            relation_score = max(relation_score, 0.85)
        if dimension_hit:
            relation_score = min(1.0, relation_score + 0.08)
        abstraction_tokens = [
            token for token in _tokens(gold_mapping.get("abstraction"))
            if len(token) >= 4
        ]
        abstraction_support = _keyword_fraction(relation_text, abstraction_tokens)
        score = clip01(0.80 * relation_score + 0.20 * abstraction_support)
        return score, {
            "source_overlap": clip01(source_overlap),
            "target_overlap": clip01(target_overlap),
            "source_relation_hit": source_relation_hit,
            "target_relation_hit": target_relation_hit,
            "dimension_hit": 1.0 if dimension_hit else 0.0,
            "id_hit": 1.0 if id_hit else 0.0,
            "relational_consistency": relation_score,
            "abstraction_support": clip01(abstraction_support),
            "wrong_direction": wrong_direction,
            "dimension_mismatch": dimension_mismatch,
        }

    def _score_evidence(
        self,
        task: Mapping[str, object],
        cluster: Mapping[str, object],
        parsed_response: Mapping[str, object],
    ) -> Dict[str, object]:
        support_ids = set(
            str(item)
            for item in ((cluster.get("support_boundary") or {}).get("evidence_ids") or [])
            if item
        )
        if not support_ids:
            support_ids = {
                str(fact.get("id"))
                for fact in list(cluster.get("source_facts") or []) + list(cluster.get("target_facts") or [])
                if fact.get("id")
            }
        required_ids = set()
        required_mapping_ids = set(task.get("required_mapping_ids") or [])
        for mapping in cluster.get("gold_mappings") or []:
            if required_mapping_ids and mapping.get("mapping_id") not in required_mapping_ids:
                continue
            required_ids.update(str(item) for item in mapping.get("source_evidence_ids") or [])
            required_ids.update(str(item) for item in mapping.get("target_evidence_ids") or [])
        returned_ids = self._returned_evidence_ids(parsed_response, task=task)
        known_returned = [item for item in returned_ids if item in support_ids]
        unknown = [item for item in returned_ids if item not in support_ids]
        overlap = [item for item in known_returned if item in required_ids]
        precision = len(known_returned) / len(returned_ids) if returned_ids else (1.0 if not required_ids else 0.0)
        recall = len(set(overlap)) / len(required_ids) if required_ids else 1.0
        unknown_source = [item for item in unknown if "_S" in item]
        unknown_target = [item for item in unknown if "_T" in item]
        return {
            "required_evidence_ids": sorted(required_ids),
            "returned_evidence_ids": returned_ids,
            "unknown_evidence_ids": unknown,
            "evidence_precision": clip01(precision),
            "evidence_recall": clip01(recall),
            "unknown_evidence_rate": clip01(len(unknown) / max(1, len(returned_ids))),
            "unknown_source_evidence_rate": clip01(len(unknown_source) / max(1, len(returned_ids))),
            "unknown_target_evidence_rate": clip01(len(unknown_target) / max(1, len(returned_ids))),
        }

    def _returned_evidence_ids(self, parsed_response: Mapping[str, object], *, task: Optional[Mapping[str, object]] = None) -> List[str]:
        values = []
        for mapping in parsed_response.get("mapping_ledger") or []:
            if isinstance(mapping, Mapping):
                values.extend(mapping.get("source_evidence_ids") or [])
                values.extend(mapping.get("target_evidence_ids") or [])
                values.extend(mapping.get("evidence_ids") or [])
        for inference in parsed_response.get("transferred_inferences") or []:
            if isinstance(inference, Mapping):
                values.extend(inference.get("evidence_ids") or [])
        for item in parsed_response.get("limits_of_analogy") or []:
            if isinstance(item, Mapping):
                values.extend(item.get("evidence_ids") or [])
        for item in parsed_response.get("unsupported_transfer_warnings") or []:
            if isinstance(item, Mapping):
                values.extend(item.get("evidence_ids") or [])
        for item in parsed_response.get("transfer_tests") or []:
            if isinstance(item, Mapping):
                values.extend(item.get("evidence_ids") or [])
        for item in parsed_response.get("negative_transfer_tests") or []:
            if isinstance(item, Mapping):
                values.extend(item.get("evidence_ids") or [])
        return _dedupe_strings(self._canonical_evidence_id(value, task=task) for value in values)

    def _canonical_evidence_id(self, value: object, *, task: Optional[Mapping[str, object]] = None) -> str:
        text = _clean_string(value)
        if not task:
            return text
        overlay = task.get("analogy_v3_overlay") if isinstance(task.get("analogy_v3_overlay"), Mapping) else {}
        aliases = overlay.get("evidence_aliases") if isinstance(overlay, Mapping) else {}
        if isinstance(aliases, Mapping):
            key = _normalize_text(text)
            for alias, canonical in aliases.items():
                if key == _normalize_text(alias):
                    return _clean_string(canonical)
        return text

    def _score_limits(self, cluster: Mapping[str, object], parsed_response: Mapping[str, object]) -> Dict[str, object]:
        required_limits = cluster.get("required_limits") or []
        if not required_limits:
            return {"limit_awareness": 1.0, "matched_limit_ids": []}
        limit_text = " ".join(
            _clean_string(item.get("text")) if isinstance(item, Mapping) else _clean_string(item)
            for item in parsed_response.get("limits_of_analogy") or []
        )
        warning_text = " ".join(
            _clean_string(item.get("text")) if isinstance(item, Mapping) else _clean_string(item)
            for item in parsed_response.get("unsupported_transfer_warnings") or []
        )
        combined = f"{limit_text} {warning_text}"
        matched = []
        for limit in required_limits:
            limit_id = limit.get("limit_id")
            id_hit = any(
                isinstance(item, Mapping) and item.get("limit_id") == limit_id
                for item in parsed_response.get("limits_of_analogy") or []
            )
            keyword_hit = _keyword_fraction(combined, limit.get("keywords") or []) >= 0.50
            if id_hit or keyword_hit:
                matched.append(limit_id)
        return {
            "limit_awareness": clip01(len(matched) / len(required_limits)),
            "matched_limit_ids": [item for item in matched if item],
        }

    def _score_false_transfer(self, cluster: Mapping[str, object], parsed_response: Mapping[str, object]) -> Dict[str, object]:
        full_text = self._asserted_response_text(parsed_response)
        warning_text = self._warning_response_text(parsed_response)
        forbidden_hits = []
        forbidden_warning_hits = []
        for forbidden in cluster.get("forbidden_transfers") or []:
            keywords = forbidden.get("keywords") or []
            if _phrase_hit(full_text, forbidden.get("text")) or _keyword_fraction(full_text, keywords) >= 0.67:
                forbidden_hits.append(forbidden.get("id"))
            if _phrase_hit(warning_text, forbidden.get("text")) or _keyword_fraction(warning_text, keywords) >= 0.67:
                forbidden_warning_hits.append(forbidden.get("id"))
        asserted_forbidden = list(forbidden_hits)
        false_transfer_rate = clip01(len(asserted_forbidden) / max(1, len(cluster.get("forbidden_transfers") or [])))

        support_ids = set(str(item) for item in ((cluster.get("support_boundary") or {}).get("evidence_ids") or []))
        unsupported_inferences = 0
        inference_count = 0
        inference_forbidden = 0
        for inference in list(parsed_response.get("transferred_inferences") or []) + list(parsed_response.get("transfer_tests") or []):
            if not isinstance(inference, Mapping):
                continue
            inference_count += 1
            evidence_ids = set(str(item) for item in inference.get("evidence_ids") or [])
            text = _clean_string(inference.get("text"))
            if not evidence_ids or any(eid not in support_ids for eid in evidence_ids):
                unsupported_inferences += 1
            if any(
                _phrase_hit(text, forbidden.get("text")) or _keyword_fraction(text, forbidden.get("keywords") or []) >= 0.67
                for forbidden in cluster.get("forbidden_transfers") or []
            ):
                inference_forbidden += 1
        unsupported_inference_rate = clip01(
            (unsupported_inferences + inference_forbidden) / max(1, inference_count)
        )
        target_limit_facts = " ".join(
            fact.get("text", "")
            for fact in cluster.get("target_facts") or []
            if str(fact.get("relation") or "").endswith("limit") or "do not state" in _normalize_text(fact.get("text"))
        )
        boundary_overclaim = 1.0 if target_limit_facts and any(
            _phrase_hit(full_text, keyword)
            for forbidden in cluster.get("forbidden_transfers") or []
            for keyword in forbidden.get("keywords") or []
            if _phrase_hit(target_limit_facts, keyword)
        ) and false_transfer_rate > 0.0 else 0.0
        warning_assertion_confusion_rate = clip01(
            len(set(forbidden_hits) & set(forbidden_warning_hits)) / max(1, len(forbidden_warning_hits))
            if forbidden_warning_hits and forbidden_hits else 0.0
        )
        invented_fact_rate = clip01(max(false_transfer_rate, unsupported_inference_rate))
        source_forbidden_rate = clip01(sum(1 for item in asserted_forbidden if str(item).endswith("FT1")) / max(1, len(asserted_forbidden)))
        target_forbidden_rate = false_transfer_rate
        return {
            "forbidden_transfer_hits": asserted_forbidden,
            "warning_hits": forbidden_warning_hits,
            "false_transfer_rate": false_transfer_rate,
            "unsupported_inference_rate": unsupported_inference_rate,
            "invented_fact_rate": invented_fact_rate,
            "boundary_overclaim_rate": boundary_overclaim,
            "warning_assertion_confusion_rate": warning_assertion_confusion_rate,
            "source_fact_error": 0.0,
            "target_fact_error": false_transfer_rate,
            "source_forbidden_rate": source_forbidden_rate,
            "target_forbidden_rate": target_forbidden_rate,
        }

    def _surface_stereotype_rate(
        self,
        cluster: Mapping[str, object],
        parsed_response: Mapping[str, object],
        mapping_coverage: float,
        evidence_recall: float,
        abstraction_support: float,
    ) -> float:
        full_text = self._full_response_text(parsed_response)
        source_domain = cluster.get("source_domain")
        target_domain = cluster.get("target_domain")
        domain_mentions = _phrase_hit(full_text, source_domain) and _phrase_hit(full_text, target_domain)
        has_low_structure = (
            mapping_coverage < 0.50 or
            evidence_recall < 0.50 or
            abstraction_support < 0.35
        )
        if parsed_response.get("legacy_fallback"):
            return 1.0 if has_low_structure else 0.5
        if domain_mentions and has_low_structure:
            return 0.75
        if not parsed_response.get("mapping_ledger"):
            return 0.65
        return 0.0


def aggregate_analogy_transfer_challenge_axes(
    task_scores: Sequence[Dict[str, object]],
    *,
    gate_pass: bool = True,
    beta_ih: float = DEFAULT_ANALOGY_TRANSFER_BETA_IH,
    beta_hi: float = DEFAULT_ANALOGY_TRANSFER_BETA_HI,
) -> Dict[str, object]:
    if not task_scores:
        return {
            "version": ANALOGY_TRANSFER_VERSION,
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
            "calibration_policy": ANALOGY_TRANSFER_V3_CALIBRATION_POLICY,
            "runtime_scoring_policy": ANALOGY_TRANSFER_V3_RUNTIME_SCORING_POLICY,
            "test_visibility_policy": ANALOGY_TRANSFER_V3_TEST_VISIBILITY_POLICY,
        }
    imagination_raw = mean_or_none(score.get("imagination_raw") for score in task_scores)
    imagination_gated = mean_or_none(score.get("imagination_gated") for score in task_scores)
    hallucination_raw = mean_or_none(score.get("hallucination_raw") for score in task_scores)
    if gate_pass and imagination_gated is not None and hallucination_raw is not None:
        imagination = clip01(imagination_gated - beta_ih * hallucination_raw)
        hallucination = clip01(hallucination_raw - beta_hi * imagination_gated)
    else:
        imagination = None
        hallucination = None

    primitive_fields: Set[str] = set()
    for score in task_scores:
        if isinstance(score.get("primitive_means"), Mapping):
            primitive_fields.update(score["primitive_means"].keys())
    primitive_means = {}
    for field in sorted(primitive_fields):
        value = mean_or_none(
            score.get("primitive_means", {}).get(field)
            for score in task_scores
            if isinstance(score.get("primitive_means"), Mapping)
        )
        if value is not None:
            primitive_means[field] = round(value, 4)
    subtype_contributions = mean_subtype_contributions(
        score.get("subtype_contributions")
        for score in task_scores
        if isinstance(score.get("subtype_contributions"), Mapping)
    )
    return {
        "version": ANALOGY_TRANSFER_VERSION,
        "score": round(imagination, 4) if imagination is not None else None,
        "imagination": round(imagination, 4) if imagination is not None else None,
        "hallucination": round(hallucination, 4) if hallucination is not None else None,
        "imagination_raw": round(imagination_raw, 4) if imagination_raw is not None else None,
        "imagination_gated": round(imagination_gated, 4) if imagination_gated is not None else None,
        "hallucination_raw": round(hallucination_raw, 4) if hallucination_raw is not None else None,
        "primitive_means": primitive_means,
        "subtype_contributions": subtype_contributions,
        "task_count": len(task_scores),
        "coverage_gate_pass": bool(gate_pass),
        "residualization": {
            "beta_IH": beta_ih,
            "beta_HI": beta_hi,
            "source": ANALOGY_TRANSFER_V3_CALIBRATION_POLICY,
            "standardization": "clip01_raw_v1",
        },
        "formula": {
            "task_imagination_raw": "T8  I_raw=0.25*top3_mapping_quality+0.25*elite_tail_top1+0.20*relational_depth+0.15*licensed_inference_quality+0.05*abstraction_diversity_eff+0.10*boundary_aware_valid_ratio",
            "task_mapping_quality": "mapping_I=rarity^1.30*structural_match_gmean^1.45*evidence_grounding^1.25*hard_gate*(0.35+0.15*abstraction_depth+0.15*cross_domain_transform+0.20*relational_depth+0.05*cross_domain_distance+0.10*boundary_awareness)",
            "task_imagination_gated": "I_gated=I_raw*source_fact_gate*target_fact_gate*transfer_gate",
            "task_hallucination_raw": "H_raw=0.25*H_false_transfer+0.30*H_fact+0.25*H_logic+0.20*H_context",
            "model_residual": "I=clip01(mean(I_gated)-beta_IH*mean(H_raw)); H=clip01(mean(H_raw)-beta_HI*mean(I_gated))",
        },
        "calibration_policy": ANALOGY_TRANSFER_V3_CALIBRATION_POLICY,
        "runtime_scoring_policy": ANALOGY_TRANSFER_V3_RUNTIME_SCORING_POLICY,
        "test_visibility_policy": ANALOGY_TRANSFER_V3_TEST_VISIBILITY_POLICY,
        "top3_mapping_quality": primitive_means.get("top3_mapping_quality"),
        "elite_tail_top1": primitive_means.get("elite_tail_top1"),
        "licensed_inference_quality": primitive_means.get("licensed_inference_quality"),
        "abstraction_diversity_eff": primitive_means.get("abstraction_diversity_eff"),
        "relational_depth": primitive_means.get("relational_depth"),
        "cross_domain_distance": primitive_means.get("cross_domain_distance"),
        "boundary_aware_valid_ratio": primitive_means.get("boundary_aware_valid_ratio"),
        "structural_match_gmean": primitive_means.get("structural_match_gmean"),
        "evidence_grounding": primitive_means.get("evidence_grounding"),
        "mapping_rarity": primitive_means.get("mapping_rarity"),
    }


__all__ = [
    "ANALOGY_TRANSFER_VERSION",
    "ANALOGY_TRANSFER_V3_CALIBRATION_POLICY",
    "ANALOGY_TRANSFER_V3_RUNTIME_SCORING_POLICY",
    "ANALOGY_TRANSFER_V3_TEST_VISIBILITY_POLICY",
    "ANALOGY_TRANSFER_V3_TASK_OVERLAY_VERSION",
    "ANALOGY_COMMON_MAPPING_BANK_VERSION",
    "AnalogyTransferScorer",
    "aggregate_analogy_transfer_challenge_axes",
    "get_analogy_transfer_task_overlay_coverage",
    "get_analogy_common_mapping_bank_coverage",
    "get_analogy_evidence_alias_coverage",
]
