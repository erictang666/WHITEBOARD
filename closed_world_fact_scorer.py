
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from typed_axis_aggregation import (
    build_closed_world_fact_calibration_contributions,
    mean_subtype_contributions,
)


CLOSED_WORLD_FACT_VERSION = "closed_world_fact"
DEFAULT_CLOSED_WORLD_FACT_BETA_HI = 0.0

DATA_DIR = Path(__file__).resolve().parent / "data"

REFUSAL_MARKERS = (
    "unknown",
    "unanswerable",
    "cannot determine",
    "can't determine",
    "not listed",
    "not in the closed world",
    "not supported",
    "no listed",
    "no replacement",
    "insufficient evidence",
)


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


def _normalize_answer_items(answer) -> List[str]:
    if isinstance(answer, list):
        return [_normalize_text(item) for item in answer if _clean_string(item)]
    if isinstance(answer, tuple):
        return [_normalize_text(item) for item in answer if _clean_string(item)]
    text = _clean_string(answer)
    if not text:
        return []
    chunks = re.split(r"\s*,\s*|\s+;\s+|\s+\band\b\s+", text)
    return [_normalize_text(chunk) for chunk in chunks if _normalize_text(chunk)]


def _parse_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = _normalize_text(value)
    if text in {"true", "yes", "y"}:
        return True
    if text in {"false", "no", "n"}:
        return False
    return None


def _parse_number(value) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = _clean_string(value)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


class ClosedWorldFactScorer:
    """Scores one ClosedWorldFact response against a closed fact card."""

    def parse_response(self, raw_text: str) -> Dict[str, object]:
        payload = _extract_json_payload(raw_text)
        if isinstance(payload, dict):
            answer = payload.get("answer")
            return {
                "parse_valid": True,
                "answerable": payload.get("answerable"),
                "answer": answer,
                "evidence_ids": _dedupe_strings(payload.get("evidence_ids") or []),
                "rationale_steps": [
                    self._normalize_rationale_step(step)
                    for step in _as_list(payload.get("rationale_steps") or [])
                ],
                "unknown_or_conflict_reason": _clean_string(payload.get("unknown_or_conflict_reason") or ""),
                "confidence": payload.get("confidence"),
                "raw_payload": payload,
                "parse_error": None,
                "legacy_fallback": False,
            }
        fallback = _clean_string(raw_text)
        return {
            "parse_valid": bool(fallback),
            "answerable": None,
            "answer": fallback,
            "evidence_ids": [],
            "rationale_steps": [],
            "unknown_or_conflict_reason": "",
            "confidence": None,
            "raw_payload": payload,
            "parse_error": None if fallback else "empty_response",
            "legacy_fallback": True,
        }

    def _normalize_rationale_step(self, raw) -> Dict[str, object]:
        if isinstance(raw, Mapping):
            return {
                "text": _clean_string(raw.get("text") or raw.get("rationale") or raw.get("step") or raw),
                "evidence_ids": _dedupe_strings(raw.get("evidence_ids") or raw.get("support_ids") or []),
            }
        return {"text": _clean_string(raw), "evidence_ids": []}

    def score_task(self, task: Mapping[str, object], parsed_response: Mapping[str, object]) -> Dict[str, object]:
        answer_mode = str(task.get("answer_mode") or "string")
        unanswerable_task = answer_mode in {"unanswerable", "false_presupposition"}
        answer = parsed_response.get("answer")
        answer_text = _clean_string(answer)
        rationale_text = self._combined_rationale_text(parsed_response)
        full_text = " ".join(
            part for part in [
                answer_text,
                rationale_text,
                _clean_string(parsed_response.get("unknown_or_conflict_reason")),
            ]
            if part
        )

        answer_score, answer_record = self._score_answer(task, answer)
        answer_correct = answer_score >= 0.999
        refusal_like = self._is_refusal_like(parsed_response, full_text)
        false_acceptance = 1.0 if unanswerable_task and not refusal_like else 0.0
        unanswerable_false_answer = 1.0 if (
            unanswerable_task
            and (
                parsed_response.get("answerable") is True
                or (answer_text and not refusal_like and _normalize_text(answer_text) not in {"null", "none"})
            )
        ) else 0.0
        if unanswerable_task and refusal_like:
            answer_score = 1.0
            answer_correct = True

        evidence_record = self._score_evidence(task, parsed_response)
        rationale_record = self._score_rationale(task, parsed_response, full_text)
        forbidden_hit = self._forbidden_answer_hit(task, answer_text)
        unknown_entity_overclaim = self._unknown_entity_overclaim(task, parsed_response, full_text, refusal_like)
        closed_boundary_failure = self._closed_boundary_failure(task, parsed_response, refusal_like)
        unsupported_fact_claim = clip01(max(
            1.0 - evidence_record["evidence_precision"],
            evidence_record["unknown_evidence_rate"],
        ) * (0.65 if not unanswerable_task and answer_text else 0.0))
        wrong_answer = clip01(1.0 - answer_score)
        contradicted_fact = clip01(max(wrong_answer, forbidden_hit))
        comparison_or_set_failure = 0.0
        if answer_mode in {"comparison", "list", "set"}:
            comparison_or_set_failure = wrong_answer

        h_fact = clip01(max(
            wrong_answer,
            contradicted_fact,
            forbidden_hit,
            unsupported_fact_claim,
            0.65 * unanswerable_false_answer,
        ))
        h_logic = clip01(max(
            1.0 - evidence_record["evidence_recall"],
            1.0 - rationale_record["rationale_keyword_support"],
            evidence_record["chain_order_failure"],
            comparison_or_set_failure,
        ))
        h_boundary = clip01(max(
            false_acceptance,
            unanswerable_false_answer,
            unknown_entity_overclaim,
            closed_boundary_failure,
        ))
        hallucination_raw = clip01(0.45 * h_fact + 0.30 * h_logic + 0.25 * h_boundary)
        score = clip01(1.0 - hallucination_raw)

        primitive_means = {
            "answer_correct": round(1.0 if answer_correct else 0.0, 4),
            "answer_score": round(answer_score, 4),
            "wrong_answer": round(wrong_answer, 4),
            "evidence_precision": round(evidence_record["evidence_precision"], 4),
            "evidence_recall": round(evidence_record["evidence_recall"], 4),
            "unknown_evidence_rate": round(evidence_record["unknown_evidence_rate"], 4),
            "chain_order_failure": round(evidence_record["chain_order_failure"], 4),
            "rationale_keyword_support": round(rationale_record["rationale_keyword_support"], 4),
            "rationale_keyword_failure": round(1.0 - rationale_record["rationale_keyword_support"], 4),
            "forbidden_answer_hit": round(forbidden_hit, 4),
            "contradicted_fact": round(contradicted_fact, 4),
            "unsupported_fact_claim": round(unsupported_fact_claim, 4),
            "comparison_or_set_failure": round(comparison_or_set_failure, 4),
            "false_presupposition_acceptance": round(false_acceptance, 4),
            "unanswerable_false_answer": round(unanswerable_false_answer, 4),
            "unknown_entity_overclaim": round(unknown_entity_overclaim, 4),
            "closed_boundary_failure": round(closed_boundary_failure, 4),
            "H_fact": round(h_fact, 4),
            "H_logic": round(h_logic, 4),
            "H_boundary": round(h_boundary, 4),
        }
        result = {
            "version": CLOSED_WORLD_FACT_VERSION,
            "task_id": task.get("id"),
            "question_type": task.get("question_type"),
            "answer_mode": answer_mode,
            "score": round(score, 4),
            "hallucination": round(hallucination_raw, 4),
            "hallucination_raw": round(hallucination_raw, 4),
            "imagination": None,
            "imagination_raw": None,
            "H_fact": round(h_fact, 4),
            "H_logic": round(h_logic, 4),
            "H_boundary": round(h_boundary, 4),
            "primitive_means": primitive_means,
            "answer_record": answer_record,
            "evidence_record": evidence_record,
            "rationale_record": rationale_record,
            "parsed_response": {
                key: value
                for key, value in parsed_response.items()
                if key != "raw_payload"
            },
            "formula": {
                "hallucination_raw": "H_raw=0.45*H_fact+0.30*H_logic+0.25*H_boundary",
                "score": "calibration_score=1-H_raw",
            },
        }
        result["subtype_contributions"] = build_closed_world_fact_calibration_contributions(result)
        result["atom_signals"] = result["subtype_contributions"].get("atom_signals", {})
        return result

    def _combined_rationale_text(self, parsed_response: Mapping[str, object]) -> str:
        steps = []
        for step in parsed_response.get("rationale_steps") or []:
            if isinstance(step, Mapping):
                steps.append(_clean_string(step.get("text")))
            else:
                steps.append(_clean_string(step))
        return " ".join(step for step in steps if step)

    def _is_refusal_like(self, parsed_response: Mapping[str, object], full_text: str) -> bool:
        if parsed_response.get("answerable") is False:
            return True
        norm = _normalize_text(full_text)
        return any(marker in norm for marker in REFUSAL_MARKERS)

    def _score_answer(self, task: Mapping[str, object], answer) -> Tuple[float, Dict[str, object]]:
        mode = str(task.get("answer_mode") or "string")
        expected = task.get("expected_answer")
        acceptable = task.get("acceptable_answers") or []
        answer_text = _clean_string(answer)
        record = {"mode": mode, "expected": expected, "answer": answer, "list_f1": None}

        if mode in {"unanswerable", "false_presupposition"}:
            return 0.0, record
        if mode in {"number", "numeric"}:
            expected_num = _parse_number(expected)
            answer_num = _parse_number(answer)
            correct = expected_num is not None and answer_num is not None and abs(expected_num - answer_num) < 1e-9
            return (1.0 if correct else 0.0), {**record, "expected_number": expected_num, "answer_number": answer_num}
        if mode == "boolean":
            expected_bool = _parse_bool(expected)
            answer_bool = _parse_bool(answer)
            correct = expected_bool is not None and answer_bool is not None and expected_bool == answer_bool
            return (1.0 if correct else 0.0), {**record, "expected_bool": expected_bool, "answer_bool": answer_bool}
        if mode in {"list", "set"}:
            expected_items = {
                _normalize_text(item)
                for item in _as_list(expected)
                if _normalize_text(item)
            }
            answer_items = {
                item for item in _normalize_answer_items(answer)
                if item
            }
            if not expected_items:
                score = 1.0 if not answer_items else 0.0
            else:
                overlap = len(expected_items & answer_items)
                precision = overlap / len(answer_items) if answer_items else 0.0
                recall = overlap / len(expected_items)
                score = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
            return clip01(score), {**record, "expected_items": sorted(expected_items), "answer_items": sorted(answer_items), "list_f1": round(clip01(score), 4)}

        candidates = list(acceptable)
        if expected is not None:
            candidates.append(expected)
        for candidate in candidates:
            if _phrase_hit(answer_text, candidate) or _normalize_text(answer_text) == _normalize_text(candidate):
                return 1.0, record
        return 0.0, record

    def _score_evidence(self, task: Mapping[str, object], parsed_response: Mapping[str, object]) -> Dict[str, object]:
        required = [str(item) for item in task.get("required_evidence_ids") or [] if item]
        returned = _dedupe_strings(parsed_response.get("evidence_ids") or [])
        step_ids = []
        for step in parsed_response.get("rationale_steps") or []:
            if isinstance(step, Mapping):
                step_ids.extend(_dedupe_strings(step.get("evidence_ids") or []))
        returned_all = _dedupe_strings(list(returned) + step_ids)
        support_ids = set((task.get("support_boundary") or {}).get("evidence_ids") or required)
        unknown = [item for item in returned_all if item not in support_ids and item not in required]
        overlap = [item for item in returned_all if item in set(required)]
        precision = len(overlap) / len(returned_all) if returned_all else (1.0 if not required else 0.0)
        recall = len(set(overlap)) / len(set(required)) if required else 1.0
        chain_failure = self._chain_order_failure(required, returned_all)
        return {
            "required_evidence_ids": required,
            "returned_evidence_ids": returned_all,
            "unknown_evidence_ids": unknown,
            "evidence_precision": clip01(precision),
            "evidence_recall": clip01(recall),
            "unknown_evidence_rate": clip01(len(unknown) / max(1, len(returned_all))),
            "chain_order_failure": chain_failure,
        }

    def _chain_order_failure(self, required: Sequence[str], returned: Sequence[str]) -> float:
        if len(required) <= 1:
            return 0.0
        index = 0
        for item in returned:
            if index < len(required) and item == required[index]:
                index += 1
        return 0.0 if index == len(required) else 1.0

    def _score_rationale(
        self,
        task: Mapping[str, object],
        parsed_response: Mapping[str, object],
        full_text: str,
    ) -> Dict[str, object]:
        keywords = [str(item) for item in task.get("required_rationale_keywords") or [] if str(item).strip()]
        if not keywords:
            support = 1.0
            hits = []
        else:
            hits = [keyword for keyword in keywords if _phrase_hit(full_text, keyword)]
            support = len(hits) / len(keywords)
        return {
            "required_rationale_keywords": keywords,
            "matched_rationale_keywords": hits,
            "rationale_keyword_support": clip01(support),
        }

    def _forbidden_answer_hit(self, task: Mapping[str, object], answer_text: str) -> float:
        for forbidden in task.get("forbidden_answers") or []:
            if _phrase_hit(answer_text, forbidden):
                return 1.0
        return 0.0

    def _unknown_entity_overclaim(
        self,
        task: Mapping[str, object],
        parsed_response: Mapping[str, object],
        full_text: str,
        refusal_like: bool,
    ) -> float:
        if refusal_like:
            return 0.0
        unknown_entities = [str(item) for item in task.get("unknown_entities") or [] if str(item).strip()]
        if any(_phrase_hit(full_text, entity) for entity in unknown_entities):
            return 1.0
        return 0.0

    def _closed_boundary_failure(
        self,
        task: Mapping[str, object],
        parsed_response: Mapping[str, object],
        refusal_like: bool,
    ) -> float:
        mode = str(task.get("answer_mode") or "")
        if mode not in {"unanswerable", "false_presupposition"}:
            return 0.0
        reason = _clean_string(parsed_response.get("unknown_or_conflict_reason"))
        if refusal_like and reason:
            return 0.0
        if refusal_like:
            return 0.35
        return 1.0


def aggregate_closed_world_fact_calibration_axes(
    task_scores: Sequence[Dict[str, object]],
    *,
    gate_pass: bool = True,
) -> Dict[str, object]:
    if not task_scores:
        return {
            "version": CLOSED_WORLD_FACT_VERSION,
            "score": None,
            "hallucination": None,
            "hallucination_raw": None,
            "primitive_means": {},
            "subtype_contributions": mean_subtype_contributions([]),
            "task_count": 0,
            "coverage_gate_pass": False,
        }
    score = mean_or_none(item.get("score") for item in task_scores)
    hallucination_raw = mean_or_none(item.get("hallucination_raw") for item in task_scores)
    primitive_fields: Set[str] = set()
    for task_score in task_scores:
        if isinstance(task_score.get("primitive_means"), Mapping):
            primitive_fields.update(task_score["primitive_means"].keys())
    primitive_means = {}
    for field in sorted(primitive_fields):
        value = mean_or_none(
            task_score.get("primitive_means", {}).get(field)
            for task_score in task_scores
            if isinstance(task_score.get("primitive_means"), Mapping)
        )
        if value is not None:
            primitive_means[field] = round(value, 4)
    subtype_contributions = mean_subtype_contributions(
        score_item.get("subtype_contributions")
        for score_item in task_scores
        if isinstance(score_item.get("subtype_contributions"), Mapping)
    )
    return {
        "version": CLOSED_WORLD_FACT_VERSION,
        "score": round(score, 4) if gate_pass and score is not None else None,
        "hallucination": round(hallucination_raw, 4) if gate_pass and hallucination_raw is not None else None,
        "hallucination_raw": round(hallucination_raw, 4) if hallucination_raw is not None else None,
        "primitive_means": primitive_means,
        "subtype_contributions": subtype_contributions,
        "task_count": len(task_scores),
        "coverage_gate_pass": bool(gate_pass),
        "formula": {
            "hallucination_raw": "H_raw=0.45*H_fact+0.30*H_logic+0.25*H_boundary",
            "score": "calibration_score=1-H_raw",
        },
    }


__all__ = [
    "CLOSED_WORLD_FACT_VERSION",
    "ClosedWorldFactScorer",
    "aggregate_closed_world_fact_calibration_axes",
]
