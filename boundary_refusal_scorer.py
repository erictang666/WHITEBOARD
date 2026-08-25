
from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Optional, Sequence


BOUNDARY_REFUSAL_VERSION = "boundary_refusal"


def clip01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _normalize_text(text: object) -> str:
    text = ("" if text is None else str(text)).lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: object) -> Sequence[str]:
    return [token for token in _normalize_text(text).split() if token]


def _keyword_hits(text: object, keywords: Iterable[object]) -> Sequence[str]:
    normalized = f" {_normalize_text(text)} "
    token_set = set(normalized.split())
    hits = []
    for keyword in keywords or []:
        keyword_norm = _normalize_text(keyword)
        if not keyword_norm:
            continue
        keyword_tokens = keyword_norm.split()
        if f" {keyword_norm} " in normalized or (keyword_tokens and set(keyword_tokens).issubset(token_set)):
            hits.append(str(keyword))
    return sorted(set(hits))


def expected_response_mode(task: Mapping[str, object]) -> str:
    raw = _normalize_text(task.get("expected_response_mode"))
    if raw in {"needs clarification", "need clarification", "clarify", "clarification"}:
        return "needs_clarification"
    if raw in {"unsolvable", "impossible", "no solution"} or bool(task.get("unsolvable")):
        return "unsolvable"
    return "solvable"


def _response_mode(parsed_response: Mapping[str, object]) -> str:
    raw = _normalize_text(parsed_response.get("solvability"))
    if raw in {"needs clarification", "need clarification", "clarify", "clarification"}:
        return "needs_clarification"
    if "clarification" in raw or "clarify" in raw:
        return "needs_clarification"
    if "unsolvable" in raw or "impossible" in raw or raw == "no solution":
        return "unsolvable"
    return "solvable"


def _task_support_text(task: Mapping[str, object]) -> str:
    chunks = [
        task.get("scene", ""),
        task.get("goal", ""),
        task.get("title", ""),
        " ".join(str(field) for field in task.get("clarification_fields") or []),
    ]
    expectation = task.get("boundary_expectation") or {}
    if isinstance(expectation, Mapping):
        chunks.extend(str(item) for item in expectation.get("supported_reason_keywords") or [])
        chunks.extend(str(item) for item in expectation.get("required_clarification_keywords") or [])
        chunks.append(str(expectation.get("reason") or ""))
    for constraint in task.get("constraints") or []:
        if isinstance(constraint, Mapping):
            chunks.append(str(constraint.get("description") or ""))
            chunks.extend(str(item) for item in constraint.get("forbidden_keywords") or [])
            chunks.extend(str(item) for item in constraint.get("forbidden_tool_keywords") or [])
    for tool in task.get("tools") or []:
        if isinstance(tool, Mapping):
            chunks.extend([str(tool.get("id") or ""), str(tool.get("name") or "")])
            chunks.extend(str(item) for item in tool.get("aliases") or [])
    return " ".join(chunks)


def _unsupported_reason_score(task: Mapping[str, object], parsed_response: Mapping[str, object]) -> float:
    reason = parsed_response.get("impossibility_reason") or parsed_response.get("reason") or ""
    if not reason:
        return 0.05
    reason_tokens = set(_tokens(reason))
    if not reason_tokens:
        return 0.05
    support_tokens = set(_tokens(_task_support_text(task)))
    unsupported = len(reason_tokens - support_tokens) / max(1, len(reason_tokens))
    expectation = task.get("boundary_expectation") or {}
    keyword_hits = []
    if isinstance(expectation, Mapping):
        keyword_hits = _keyword_hits(reason, expectation.get("supported_reason_keywords") or [])
    keyword_credit = 0.30 if keyword_hits else 0.0
    return clip01(0.05 + 0.55 * unsupported - keyword_credit)


def _clarification_coverage(task: Mapping[str, object], parsed_response: Mapping[str, object]) -> Dict[str, object]:
    questions = [
        str(item)
        for item in parsed_response.get("clarification_questions") or []
        if str(item).strip()
    ]
    text = " ".join(questions)
    expectation = task.get("boundary_expectation") or {}
    required = []
    if isinstance(expectation, Mapping):
        required = list(expectation.get("required_clarification_keywords") or [])
    if not required:
        required = list(task.get("clarification_fields") or [])
    hits = _keyword_hits(text, required)
    coverage = len(hits) / max(1, len(required)) if required else (1.0 if questions else 0.0)
    return {
        "questions": questions,
        "required_keywords": [str(item) for item in required],
        "matched_keywords": list(hits),
        "coverage": round(clip01(coverage), 4),
    }


class BoundaryRefusalScorer:
    """Scores closed-world false acceptance/refusal and clarification behavior."""

    def score(
        self,
        task: Mapping[str, object],
        parsed_response: Mapping[str, object],
        *,
        plan_scores: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> Dict[str, object]:
        expected = expected_response_mode(task)
        response = _response_mode(parsed_response)
        plans = list(parsed_response.get("plans") or [])
        has_plans = bool(plans)
        clarification = _clarification_coverage(task, parsed_response)
        has_clarification = bool(clarification["questions"])

        false_acceptance = 0.0
        false_refusal = 0.0
        clarification_miss = 0.0
        unsupported_refusal_reason = 0.0

        if expected == "unsolvable":
            false_acceptance = 1.0 if has_plans or response == "solvable" else 0.0
            if response == "unsolvable" and not has_plans:
                unsupported_refusal_reason = _unsupported_reason_score(task, parsed_response)
        elif expected == "needs_clarification":
            clarification_miss = 0.0 if response == "needs_clarification" and has_clarification else 1.0
            false_acceptance = 1.0 if has_plans or response == "solvable" else 0.0
            if response == "unsolvable":
                unsupported_refusal_reason = _unsupported_reason_score(task, parsed_response)
        else:
            false_refusal = 1.0 if response in {"unsolvable", "needs_clarification"} and not has_plans else 0.0
            if response == "unsolvable":
                unsupported_refusal_reason = _unsupported_reason_score(task, parsed_response)

        distractor_violation = max(
            (clip01(score.get("D_distractor_tool_violation")) for score in plan_scores or []),
            default=0.0,
        )
        correct_boundary = (
            (expected == "solvable" and response == "solvable" and has_plans) or
            (expected == "unsolvable" and response == "unsolvable" and not has_plans and false_acceptance == 0.0) or
            (expected == "needs_clarification" and response == "needs_clarification" and has_clarification and not has_plans)
        )
        boundary_hallucination = clip01(
            0.40 * false_acceptance +
            0.35 * false_refusal +
            0.30 * clarification_miss +
            0.20 * unsupported_refusal_reason +
            0.20 * distractor_violation
        )
        if false_acceptance:
            boundary_hallucination = max(boundary_hallucination, 0.75)
        if false_refusal:
            boundary_hallucination = max(boundary_hallucination, 0.60)
        if clarification_miss:
            boundary_hallucination = max(boundary_hallucination, 0.55)

        return {
            "version": BOUNDARY_REFUSAL_VERSION,
            "expected_response_mode": expected,
            "response_mode": response,
            "false_acceptance": round(false_acceptance, 4),
            "false_refusal": round(false_refusal, 4),
            "unsupported_refusal_reason": round(unsupported_refusal_reason, 4),
            "clarification_miss": round(clarification_miss, 4),
            "distractor_tool_violation": round(distractor_violation, 4),
            "correct_boundary_response": 1.0 if correct_boundary else 0.0,
            "boundary_hallucination": round(boundary_hallucination, 4),
            "clarification_record": clarification,
            "formula": "0.40*false_acceptance + 0.35*false_refusal + 0.30*clarification_miss + 0.20*unsupported_reason + 0.20*distractor",
        }


__all__ = [
    "BOUNDARY_REFUSAL_VERSION",
    "BoundaryRefusalScorer",
    "clip01",
    "expected_response_mode",
]
