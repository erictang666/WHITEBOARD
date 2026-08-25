
from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from typed_axis_aggregation import build_cross_task_fact_consistency_contributions


CROSS_TASK_FACT_CONSISTENCY_VERSION = "cross_task_fact_consistency"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for", "with",
    "in", "on", "at", "by", "is", "are", "was", "were", "be", "being",
    "been", "would", "could", "should", "might", "may", "can", "will",
    "all", "every", "some", "any", "this", "that", "these", "those",
    "one", "two", "three", "before", "after", "when", "then", "there",
    "their", "them", "they", "she", "he", "her", "his", "into", "from",
    "as", "if", "it", "its", "only", "not", "no",
}

NEGATION_TERMS = {
    "not", "no", "never", "cannot", "can't", "without", "false", "impossible",
    "unsupported", "unavailable", "forbidden",
}


def clip01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    usable = [float(value) for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def _clean_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_text(text: str) -> str:
    text = (text or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> List[str]:
    return [
        token for token in _normalize_text(text).split()
        if token and token not in STOPWORDS and len(token) > 1
    ]


def _dedupe_strings(values: Iterable[object]) -> List[str]:
    seen = set()
    results = []
    for value in values or []:
        cleaned = _clean_string(value)
        if not cleaned:
            continue
        key = _normalize_text(cleaned)
        if key in seen:
            continue
        seen.add(key)
        results.append(cleaned)
    return results


def _fact_key(raw_id: object, *, component: str = "", task_id: str = "") -> str:
    """Normalize evidence ids while avoiding accidental local-id merges."""

    text = _normalize_text(str(raw_id or ""))
    text = re.sub(r"\s+", "_", text)
    if re.fullmatch(r"[fc]_\d+|[fc]\d+", text):
        return f"{_normalize_text(component)}:{_normalize_text(task_id)}:{text}"
    return text


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _polarity(text: str) -> int:
    token_set = set(_normalize_text(text).split())
    return -1 if token_set & NEGATION_TERMS else 1


def _as_mapping(value) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _claim_observations_from_support_ledger(
    *,
    component: str,
    task_id: str,
    support_ledger: Mapping[str, object],
) -> List[Dict[str, object]]:
    observations = []
    for record in support_ledger.get("claim_records") or []:
        if not isinstance(record, Mapping):
            continue
        text = _clean_string(record.get("text"))
        if not text:
            continue
        support_ids = _dedupe_strings(record.get("support_ids") or [])
        for support_id in support_ids:
            observations.append({
                "component": component,
                "task_id": task_id,
                "fact_id": support_id,
                "fact_key": _fact_key(support_id, component=component, task_id=task_id),
                "text": text,
                "supported": bool(record.get("supported")),
                "contradicted": bool(record.get("contradicted")),
                "citation_mismatch": bool(record.get("citation_mismatch")),
                "unknown_evidence": bool(record.get("unknown_evidence_ids")),
            })
    return observations


def _gcw_observations(task_scores: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    observations = []
    for score in task_scores or []:
        if not isinstance(score, Mapping):
            continue
        support_ledger = _as_mapping(score.get("support_ledger") or (score.get("details") or {}).get("support_ledger"))
        observations.extend(_claim_observations_from_support_ledger(
            component="GCW",
            task_id=_clean_string(score.get("task_id")),
            support_ledger=support_ledger,
        ))
    return observations


def _hypospace_observations(task_scores: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    observations = []
    for score in task_scores or []:
        if not isinstance(score, Mapping):
            continue
        task_id = _clean_string(score.get("task_id"))
        support_ledger = _as_mapping(score.get("support_ledger"))
        observations.extend(_claim_observations_from_support_ledger(
            component="HypoUseSpace",
            task_id=task_id,
            support_ledger=support_ledger,
        ))
        for hyp in score.get("hypothesis_scores") or []:
            if not isinstance(hyp, Mapping):
                continue
            text = _clean_string(hyp.get("hypothesis") or hyp.get("core_mechanism") or hyp.get("full_text"))
            if not text:
                text = _clean_string(hyp)
            for evidence_id in _dedupe_strings(hyp.get("cited_evidence_ids") or hyp.get("evidence_ids") or []):
                observations.append({
                    "component": "HypoUseSpace",
                    "task_id": task_id,
                    "fact_id": evidence_id,
                    "fact_key": _fact_key(evidence_id, component="HypoUseSpace", task_id=task_id),
                    "text": text,
                    "supported": clip01(hyp.get("evidence_support")) >= 0.45,
                    "contradicted": clip01(hyp.get("explicit_contradiction_or_forbidden_foil")) >= 0.50,
                    "citation_mismatch": clip01(hyp.get("citation_mismatch")) >= 0.50,
                    "unknown_evidence": bool(hyp.get("unknown_evidence_ids")),
                })
    return observations


def _closed_world_observations(task_scores: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    observations = []
    for score in task_scores or []:
        if not isinstance(score, Mapping):
            continue
        task_id = _clean_string(score.get("task_id"))
        parsed = _as_mapping(score.get("parsed_response"))
        answer_text = " ".join(
            _clean_string(parsed.get(field))
            for field in ("answer", "rationale", "rationale_summary")
            if parsed.get(field) is not None
        )
        if not answer_text:
            answer_text = _clean_string(score.get("answer_record"))
        evidence_record = _as_mapping(score.get("evidence_record"))
        evidence_ids = _dedupe_strings(
            list(evidence_record.get("returned_evidence_ids") or []) +
            list(evidence_record.get("required_evidence_ids") or [])
        )
        for evidence_id in evidence_ids:
            observations.append({
                "component": "ClosedWorldFact",
                "task_id": task_id,
                "fact_id": evidence_id,
                "fact_key": _fact_key(evidence_id, component="ClosedWorldFact", task_id=task_id),
                "text": answer_text,
                "supported": clip01(evidence_record.get("evidence_precision")) >= 0.50,
                "contradicted": clip01(_as_mapping(score.get("primitive_means")).get("contradicted_fact")) >= 0.50,
                "citation_mismatch": clip01(evidence_record.get("unknown_evidence_rate")) >= 0.50,
                "unknown_evidence": bool(evidence_record.get("unknown_evidence_ids")),
            })
    return observations


def _pair_inconsistency(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    text_left = _clean_string(left.get("text"))
    text_right = _clean_string(right.get("text"))
    similarity = _jaccard(text_left, text_right)
    polarity_conflict = 1.0 if _polarity(text_left) != _polarity(text_right) else 0.0
    support_conflict = 1.0 if bool(left.get("supported")) != bool(right.get("supported")) else 0.0
    contradiction_conflict = 1.0 if bool(left.get("contradicted")) != bool(right.get("contradicted")) else 0.0
    citation_issue = 1.0 if left.get("citation_mismatch") or right.get("citation_mismatch") or left.get("unknown_evidence") or right.get("unknown_evidence") else 0.0
    semantic_conflict = max(0.0, 0.28 - similarity) / 0.28
    return clip01(
        0.45 * semantic_conflict +
        0.25 * polarity_conflict +
        0.15 * support_conflict +
        0.10 * contradiction_conflict +
        0.05 * citation_issue
    )


def score_cross_task_fact_consistency(
    *,
    gcw_task_scores: Sequence[Mapping[str, object]] = (),
    hypospace_task_scores: Sequence[Mapping[str, object]] = (),
    closed_world_fact_task_scores: Sequence[Mapping[str, object]] = (),
) -> Dict[str, object]:
    """Compute cross-task fact consistency from repeated evidence mentions."""

    observations = []
    observations.extend(_gcw_observations(gcw_task_scores))
    observations.extend(_hypospace_observations(hypospace_task_scores))
    observations.extend(_closed_world_observations(closed_world_fact_task_scores))

    by_fact: Dict[str, List[Dict[str, object]]] = {}
    for observation in observations:
        fact_key = observation.get("fact_key")
        if not fact_key:
            continue
        by_fact.setdefault(str(fact_key), []).append(observation)

    pair_records = []
    group_records = []
    for fact_key, fact_observations in sorted(by_fact.items()):
        task_ids = {item.get("task_id") for item in fact_observations if item.get("task_id")}
        if len(task_ids) < 2:
            continue
        pair_scores = []
        for left, right in combinations(fact_observations, 2):
            if left.get("task_id") == right.get("task_id"):
                continue
            score = _pair_inconsistency(left, right)
            pair_scores.append(score)
            pair_records.append({
                "fact_key": fact_key,
                "left": {"component": left.get("component"), "task_id": left.get("task_id"), "fact_id": left.get("fact_id")},
                "right": {"component": right.get("component"), "task_id": right.get("task_id"), "fact_id": right.get("fact_id")},
                "inconsistency": round(score, 4),
            })
        if pair_scores:
            group_records.append({
                "fact_key": fact_key,
                "fact_ids": sorted({str(item.get("fact_id")) for item in fact_observations if item.get("fact_id")}),
                "task_ids": sorted(str(item) for item in task_ids),
                "observation_count": len(fact_observations),
                "pair_count": len(pair_scores),
                "inconsistency": round(max(pair_scores), 4),
            })

    hallucination_raw = mean_or_none(record["inconsistency"] for record in group_records)
    ran = hallucination_raw is not None
    axis = {
        "version": CROSS_TASK_FACT_CONSISTENCY_VERSION,
        "role": "diagnostic",
        "score": round(1.0 - hallucination_raw, 4) if hallucination_raw is not None else None,
        "hallucination": round(hallucination_raw, 4) if hallucination_raw is not None else None,
        "hallucination_raw": round(hallucination_raw, 4) if hallucination_raw is not None else None,
        "imagination": None,
        "imagination_raw": None,
        "ran": bool(ran),
        "coverage_gate_pass": bool(ran),
        "primitive_means": {
            "observation_count": len(observations),
            "repeated_fact_group_count": len(group_records),
            "compared_pair_count": len(pair_records),
            "consistency_inconsistency_rate": round(hallucination_raw, 4) if hallucination_raw is not None else None,
        },
        "fact_groups": group_records[:25],
        "pair_records": pair_records[:50],
        "formula": {
            "hallucination_raw": "mean over repeated fact/evidence groups of max pair inconsistency",
            "pair_inconsistency": "0.45*semantic_conflict + 0.25*polarity_conflict + 0.15*support_conflict + 0.10*contradiction_conflict + 0.05*citation_issue",
            "scope": "diagnostic only; does not change primary I/H aggregation",
        },
    }
    axis["subtype_contributions"] = build_cross_task_fact_consistency_contributions(axis)
    axis["atom_signals"] = axis["subtype_contributions"].get("atom_signals", {})
    return axis


def aggregate_cross_task_fact_consistency_axes(
    axes: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    usable = [axis for axis in axes if isinstance(axis, Mapping)]
    h_values = [axis.get("hallucination_raw") for axis in usable if axis.get("hallucination_raw") is not None]
    h_mean = mean_or_none(h_values)
    primitive_fields = set()
    for axis in usable:
        primitive = axis.get("primitive_means")
        if isinstance(primitive, Mapping):
            primitive_fields.update(primitive.keys())
    primitive_means = {}
    for field in sorted(primitive_fields):
        value = mean_or_none(
            axis.get("primitive_means", {}).get(field)
            for axis in usable
            if isinstance(axis.get("primitive_means"), Mapping)
        )
        if value is not None:
            primitive_means[field] = round(value, 4)
    result = {
        "version": CROSS_TASK_FACT_CONSISTENCY_VERSION,
        "role": "diagnostic",
        "score": round(1.0 - h_mean, 4) if h_mean is not None else None,
        "hallucination": round(h_mean, 4) if h_mean is not None else None,
        "hallucination_raw": round(h_mean, 4) if h_mean is not None else None,
        "imagination": None,
        "imagination_raw": None,
        "ran": h_mean is not None,
        "coverage_gate_pass": h_mean is not None,
        "primitive_means": primitive_means,
        "replicate_count": len(usable),
        "eligible_repeat_count": len(h_values),
        "formula": {
            "hallucination_raw": "mean repeat H_consistency",
            "scope": "diagnostic only; does not change primary I/H aggregation",
        },
    }
    result["subtype_contributions"] = build_cross_task_fact_consistency_contributions(result)
    result["atom_signals"] = result["subtype_contributions"].get("atom_signals", {})
    return result


__all__ = [
    "CROSS_TASK_FACT_CONSISTENCY_VERSION",
    "aggregate_cross_task_fact_consistency_axes",
    "score_cross_task_fact_consistency",
]
