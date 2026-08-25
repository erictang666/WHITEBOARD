
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from json_repair_utils import parse_jsonish_payload
from support_ledger_scorer import SupportLedgerScorer
from typed_axis_aggregation import (
    build_hypospace_task_subtype_contributions,
    mean_subtype_contributions,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
HYPOUSESPACE_VERSION = "hypouse_space_dual_axis"
HYPOUSESPACE_V3_CALIBRATION_POLICY = "benchmark_default"
HYPOUSESPACE_V3_RUNTIME_SCORING_POLICY = "fixed output-only parameters"
DEFAULT_HYPOUSESPACE_BETA_IH = 1.00
DEFAULT_HYPOUSESPACE_BETA_HI = 0.10
DEFAULT_HYPOUSESPACE_OUTPUT_COUNT = 6
HYPOUSESPACE_COMMON_BANK_PATH = DATA_DIR / "hypospace_common_hypothesis_bank_v3.json"
HYPOUSESPACE_ALIAS_PATH = DATA_DIR / "hypospace_valid_match_aliases_v3.json"
HYPOUSESPACE_SCORING_CONFIG_PATH = DATA_DIR / "hypospace_scoring_config.json"
DEFAULT_HYPOUSESPACE_V3_PARAMS = {
    "rarity_gamma": 1.35,
    "match_quality_gamma": 1.35,
    "soft_match_threshold": 0.50,
    "hard_valid_soft_threshold": 0.88,
    "evidence_support_floor": 0.45,
    "broad_common_rarity_cap": 0.50,
    "supported_rare_rarity_floor": 0.82,
    "valid_rarity_lift_base": 0.0,
    "task_aggregation_weights": {
        "quality_mass_top3": 0.40,
        "elite_tail_top2": 0.20,
        "mechanism_diversity_eff": 0.15,
        "evidence_synthesis_coverage": 0.10,
        "evidence_synthesis_depth": 0.10,
        "hard_valid_ratio": 0.05,
    },
    "hypothesis_multiplier_weights": {
        "base": 0.45,
        "evidence_synthesis": 0.25,
        "mechanism_depth": 0.20,
        "testability_boundary_awareness": 0.10,
    },
}
_COMMON_BANK_CACHE = None
_ALIAS_CACHE = None
_CALIBRATION_PARAMS_CACHE = None

COMMON_UNAVAILABLE_ENTITY_MARKERS = {
    "3d printer",
    "3d printed holder",
    "battery",
    "book",
    "candle",
    "cardboard",
    "flashlight",
    "glow stick",
    "glue",
    "grabber",
    "hammer",
    "knife",
    "lighter",
    "matches",
    "phone stand",
    "saw",
    "scissors",
    "spoon",
    "superglue",
    "tripod",
    "vacuum cleaner",
    "wire hook",
    "wood shim",
}


def clip01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    return sum(filtered) / len(filtered) if filtered else None


def geometric_mean(values: Iterable[float]) -> float:
    clipped = [max(0.0, min(1.0, float(value))) for value in values]
    if not clipped:
        return 0.0
    if any(value <= 0.0 for value in clipped):
        return 0.0
    return clip01(math.exp(sum(math.log(value) for value in clipped) / len(clipped)))


def top_quality_mass(values: Iterable[float], k: int, *, denominator: Optional[int] = None) -> float:
    values = sorted((clip01(value) for value in values), reverse=True)
    denom = int(denominator or k or 1)
    denom = max(1, min(k, denom))
    return clip01(sum(values[:k]) / denom)


def _normalize_text(text: str) -> str:
    text = (text or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def _extract_json_payload(raw_text):
    return parse_jsonish_payload(raw_text)


def _phrase_hit(text_norm: str, phrase: str) -> bool:
    phrase_norm = _normalize_text(str(phrase))
    if not phrase_norm:
        return False
    phrase_tokens = phrase_norm.split()
    token_set = set(text_norm.split())
    return f" {phrase_norm} " in f" {text_norm} " or bool(phrase_tokens and set(phrase_tokens).issubset(token_set))


def _keyword_hits(text: str, keywords: Sequence[str]) -> List[str]:
    text_norm = _normalize_text(text)
    hits = []
    for keyword in keywords or []:
        if _phrase_hit(text_norm, str(keyword)):
            hits.append(str(keyword))
    return sorted(set(hits))


def _forbidden_keyword_hits(text: str, keywords: Sequence[str]) -> List[str]:
    text_norm = _normalize_text(text)
    hits = []
    negators = ("not", "no", "without", "avoid", "avoids", "do not", "does not")
    for keyword in keywords or []:
        keyword_norm = _normalize_text(str(keyword))
        if not keyword_norm:
            continue
        keyword_tokens = keyword_norm.split()
        if len(keyword_tokens) == 1:
            present = keyword_norm in set(text_norm.split())
        else:
            pattern = r"\b" + r"\s+(?:the\s+|a\s+|an\s+)?".join(re.escape(token) for token in keyword_tokens) + r"\b"
            present = re.search(pattern, text_norm) is not None
        if not present:
            continue
        negated = any(f"{negator} {keyword_norm}" in text_norm for negator in negators)
        if not negated:
            flexible_keyword = r"\s+(?:the\s+|a\s+|an\s+)?".join(re.escape(token) for token in keyword_tokens)
            negated = re.search(
                r"\b(?:not|no|without|avoid|avoids|do\s+not|does\s+not)\b(?:\s+\w+){0,4}\s+" + flexible_keyword + r"\b",
                text_norm,
            ) is not None
        if not negated:
            hits.append(str(keyword))
    return sorted(set(hits))


def _canonical_key(
    entities: Iterable[str],
    operations: Iterable[str],
    mechanisms: Iterable[str],
    effects: Iterable[str],
) -> str:
    return "|".join([
        "E=" + ",".join(sorted(set(entities))),
        "O=" + ",".join(sorted(set(operations))),
        "M=" + ",".join(sorted(set(mechanisms))),
        "X=" + ",".join(sorted(set(effects))),
    ])


def _entropy_ratio(values: Sequence[str], possible_count: int) -> float:
    values = [str(value) for value in values if str(value)]
    if len(values) < 2:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    denom = math.log(max(2, min(possible_count or len(counts), total)))
    return clip01(entropy / denom) if denom > 0 else 0.0


def _load_json_file(path: Path, default):
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if payload is not None else default
    except Exception:
        return default


def load_hypospace_common_hypothesis_bank(path: Optional[Path] = None) -> Dict[str, object]:
    global _COMMON_BANK_CACHE
    if path is None and _COMMON_BANK_CACHE is not None:
        return _COMMON_BANK_CACHE
    payload = _load_json_file(path or HYPOUSESPACE_COMMON_BANK_PATH, {"schema": "missing_hypospace_common_hypothesis_bank_v3", "tasks": {}})
    if path is None:
        _COMMON_BANK_CACHE = payload
    return payload


def load_hypospace_valid_match_aliases(path: Optional[Path] = None) -> Dict[str, object]:
    global _ALIAS_CACHE
    if path is None and _ALIAS_CACHE is not None:
        return _ALIAS_CACHE
    payload = _load_json_file(path or HYPOUSESPACE_ALIAS_PATH, {"schema": "missing_hypospace_valid_match_aliases_v3", "global": {}, "tasks": {}})
    if path is None:
        _ALIAS_CACHE = payload
    return payload


def load_hypospace_v3_calibration_params(path: Optional[Path] = None) -> Dict[str, object]:
    global _CALIBRATION_PARAMS_CACHE
    if path is None and _CALIBRATION_PARAMS_CACHE is not None:
        return dict(_CALIBRATION_PARAMS_CACHE)
    params = dict(DEFAULT_HYPOUSESPACE_V3_PARAMS)
    params["task_aggregation_weights"] = dict(DEFAULT_HYPOUSESPACE_V3_PARAMS["task_aggregation_weights"])
    params["hypothesis_multiplier_weights"] = dict(DEFAULT_HYPOUSESPACE_V3_PARAMS["hypothesis_multiplier_weights"])
    payload = _load_json_file(path or HYPOUSESPACE_SCORING_CONFIG_PATH, {})
    frozen = payload.get("final_params") if isinstance(payload, dict) else None
    if isinstance(frozen, dict):
        for key, value in frozen.items():
            if key in {"task_aggregation_weights", "hypothesis_multiplier_weights"}:
                continue
            params[key] = value
        if isinstance(frozen.get("task_aggregation_weights"), dict):
            weights = dict(DEFAULT_HYPOUSESPACE_V3_PARAMS["task_aggregation_weights"])
            weights.update(frozen["task_aggregation_weights"])
            params["task_aggregation_weights"] = weights
        if isinstance(frozen.get("hypothesis_multiplier_weights"), dict):
            weights = dict(DEFAULT_HYPOUSESPACE_V3_PARAMS["hypothesis_multiplier_weights"])
            weights.update(frozen["hypothesis_multiplier_weights"])
            params["hypothesis_multiplier_weights"] = weights
    if path is None:
        _CALIBRATION_PARAMS_CACHE = dict(params)
    return dict(params)


def get_hypospace_common_hypothesis_bank_coverage(task_ids: Sequence[str]) -> Dict[str, object]:
    bank = load_hypospace_common_hypothesis_bank()
    tasks = bank.get("tasks") if isinstance(bank, dict) else {}
    covered = []
    missing = []
    for task_id in task_ids:
        record = tasks.get(task_id) if isinstance(tasks, dict) else None
        has_required = (
            isinstance(record, dict)
            and bool(record.get("hard_zero_hypothesis_families"))
            and bool(record.get("broad_common_hypothesis_families"))
            and bool(record.get("supported_rare_hypothesis_families"))
        )
        (covered if has_required else missing).append(str(task_id))
    return {
        "covered": covered,
        "missing": missing,
        "coverage": round(len(covered) / max(1, len(task_ids)), 4),
    }


def get_hypospace_valid_match_alias_coverage(task_ids: Sequence[str]) -> Dict[str, object]:
    payload = load_hypospace_valid_match_aliases()
    tasks = payload.get("tasks") if isinstance(payload, dict) else {}
    global_aliases = payload.get("global") if isinstance(payload, dict) else {}
    covered = []
    missing = []
    for task_id in task_ids:
        record = tasks.get(task_id) if isinstance(tasks, dict) else None
        has_required = isinstance(record, dict) and all(
            bool((global_aliases or {}).get(kind))
            for kind in ("entities", "operations", "mechanisms", "effects", "evidence_keywords")
        )
        (covered if has_required else missing).append(str(task_id))
    return {
        "covered": covered,
        "missing": missing,
        "coverage": round(len(covered) / max(1, len(task_ids)), 4),
    }


class HypoUseSpaceScorer:
    """Scores one HypoUseSpace response against a closed finite hypothesis set."""

    def __init__(
        self,
        *,
        beta_ih: float = DEFAULT_HYPOUSESPACE_BETA_IH,
        beta_hi: float = DEFAULT_HYPOUSESPACE_BETA_HI,
        expected_output_count: int = DEFAULT_HYPOUSESPACE_OUTPUT_COUNT,
    ):
        self.beta_ih = float(beta_ih)
        self.beta_hi = float(beta_hi)
        self.expected_output_count = int(expected_output_count)
        self.support_ledger = SupportLedgerScorer()
        self.common_bank = load_hypospace_common_hypothesis_bank()
        self.alias_overlay = load_hypospace_valid_match_aliases()
        self.v3_params = load_hypospace_v3_calibration_params()

    
    
    

    def parse_response(self, raw_text: str) -> Dict[str, object]:
        payload = _extract_json_payload(raw_text)
        if payload is None:
            return {
                "parse_valid": False,
                "no_valid_hypothesis": False,
                "hypotheses": [],
                "claim_ledger": [],
                "reason": "",
                "raw_payload": None,
                "parse_error": "no_json_payload",
            }

        no_valid_hypothesis = False
        reason = ""
        raw_hypotheses = []
        raw_claims = []

        if isinstance(payload, list):
            raw_hypotheses = payload
        elif isinstance(payload, dict):
            no_valid_hypothesis = bool(
                payload.get("no_valid_hypothesis")
                or payload.get("no_valid_hypotheses")
                or payload.get("unsolvable")
            )
            reason = _clean_string(
                payload.get("reason")
                or payload.get("impossibility_reason")
                or payload.get("explanation")
            )
            raw_hypotheses = (
                payload.get("hypotheses")
                or payload.get("hypothesis_set")
                or payload.get("candidate_hypotheses")
                or payload.get("valid_hypotheses")
                or payload.get("ideas")
                or payload.get("uses")
                or payload.get("solutions")
                or payload.get("proposals")
                or payload.get("answers")
                or []
            )
            raw_claims = payload.get("claim_ledger") or payload.get("claims") or []
            if not raw_hypotheses and any(
                key in payload
                for key in [
                    "hypothesis",
                    "idea",
                    "use",
                    "proposal",
                    "answer",
                    "entities",
                    "entity_ids",
                    "operation_tags",
                    "operation_ids",
                    "mechanism_tags",
                    "mechanism_ids",
                    "expected_effects",
                    "effect_ids",
                ]
            ):
                raw_hypotheses = [payload]
        else:
            return {
                "parse_valid": False,
                "no_valid_hypothesis": False,
                "hypotheses": [],
                "claim_ledger": [],
                "reason": "",
                "raw_payload": payload,
                "parse_error": "json_payload_not_object_or_array",
            }

        hypotheses = []
        for raw_hypothesis in _as_list(raw_hypotheses):
            parsed = self._parse_hypothesis(raw_hypothesis)
            if parsed is not None:
                hypotheses.append(parsed)
        claim_ledger = []
        for index, raw_claim in enumerate(_as_list(raw_claims), start=1):
            parsed_claim = self._parse_claim(raw_claim, default_id=index)
            if parsed_claim is not None:
                claim_ledger.append(parsed_claim)

        return {
            "parse_valid": True,
            "no_valid_hypothesis": bool(no_valid_hypothesis),
            "hypotheses": hypotheses,
            "claim_ledger": claim_ledger,
            "reason": reason,
            "raw_payload": payload,
            "parse_error": None,
        }

    def _parse_hypothesis(self, raw_hypothesis) -> Optional[Dict[str, object]]:
        if isinstance(raw_hypothesis, str):
            text = _clean_string(raw_hypothesis)
            if not text:
                return None
            return {
                "hypothesis": text,
                "entities": [],
                "operation_tags": [],
                "mechanism_tags": [],
                "expected_effects": [],
                "evidence": [],
                "evidence_ids": [],
                "claim_ids": [],
                "core_mechanism": "",
                "evidence_chain": [],
                "why_distinct": "",
                "boundary_note": "",
                "testable_prediction": "",
                "raw_hypothesis": raw_hypothesis,
            }
        if not isinstance(raw_hypothesis, dict):
            return None

        hypothesis_text = _clean_string(
            raw_hypothesis.get("hypothesis")
            or raw_hypothesis.get("idea")
            or raw_hypothesis.get("use")
            or raw_hypothesis.get("proposal")
            or raw_hypothesis.get("answer")
            or raw_hypothesis.get("description")
            or raw_hypothesis.get("mechanism")
            or raw_hypothesis.get("core_mechanism")
        )
        return {
            "hypothesis": hypothesis_text,
            "entities": [
                _clean_string(item)
                for item in _as_list(raw_hypothesis.get("entities") or raw_hypothesis.get("entity_ids") or raw_hypothesis.get("objects"))
                if _clean_string(item)
            ],
            "operation_tags": [
                _clean_string(item)
                for item in _as_list(
                    raw_hypothesis.get("operation_tags")
                    or raw_hypothesis.get("operation_ids")
                    or raw_hypothesis.get("operations")
                )
                if _clean_string(item)
            ],
            "mechanism_tags": [
                _clean_string(item)
                for item in _as_list(
                    raw_hypothesis.get("mechanism_tags")
                    or raw_hypothesis.get("mechanism_ids")
                    or raw_hypothesis.get("mechanisms")
                    or raw_hypothesis.get("affordances")
                )
                if _clean_string(item)
            ],
            "expected_effects": [
                _clean_string(item)
                for item in _as_list(
                    raw_hypothesis.get("expected_effects")
                    or raw_hypothesis.get("effect_ids")
                    or raw_hypothesis.get("effects")
                    or raw_hypothesis.get("goal_predicates")
                )
                if _clean_string(item)
            ],
            "evidence": [
                _clean_string(item)
                for item in _as_list(
                    raw_hypothesis.get("evidence")
                    or raw_hypothesis.get("support")
                    or raw_hypothesis.get("supporting_evidence")
                )
                if _clean_string(item)
            ],
            "evidence_ids": [
                _clean_string(item)
                for item in _as_list(
                    raw_hypothesis.get("evidence_ids")
                    or raw_hypothesis.get("support_ids")
                    or raw_hypothesis.get("citations")
                    or raw_hypothesis.get("citation_ids")
                )
                if _clean_string(item)
            ],
            "claim_ids": [
                _clean_string(item)
                for item in _as_list(raw_hypothesis.get("claim_ids") or raw_hypothesis.get("claims"))
                if _clean_string(item)
            ],
            "core_mechanism": _clean_string(
                raw_hypothesis.get("core_mechanism")
                or raw_hypothesis.get("central_mechanism")
                or raw_hypothesis.get("mechanism_summary")
            ),
            "evidence_chain": [
                _clean_string(item)
                for item in _as_list(raw_hypothesis.get("evidence_chain") or raw_hypothesis.get("support_chain"))
                if _clean_string(item)
            ],
            "why_distinct": _clean_string(
                raw_hypothesis.get("why_distinct")
                or raw_hypothesis.get("distinctiveness")
                or raw_hypothesis.get("difference")
            ),
            "boundary_note": _clean_string(
                raw_hypothesis.get("boundary_note")
                or raw_hypothesis.get("closed_world_boundary")
                or raw_hypothesis.get("boundary")
            ),
            "testable_prediction": _clean_string(
                raw_hypothesis.get("testable_prediction")
                or raw_hypothesis.get("prediction")
                or raw_hypothesis.get("test")
            ),
            "raw_hypothesis": raw_hypothesis,
        }

    def _parse_claim(self, raw_claim, *, default_id: int) -> Optional[Dict[str, object]]:
        if isinstance(raw_claim, str):
            text = _clean_string(raw_claim)
            if not text:
                return None
            return {
                "claim_id": f"HCL{default_id}",
                "hypothesis_id": None,
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
        support_ids = [
            _clean_string(item)
            for item in _as_list(raw_claim.get("support_ids") or raw_claim.get("evidence_ids") or raw_claim.get("citation_ids"))
            if _clean_string(item)
        ]
        evidence_ids = [
            _clean_string(item)
            for item in _as_list(raw_claim.get("evidence_ids") or support_ids)
            if _clean_string(item)
        ]
        return {
            "claim_id": _clean_string(raw_claim.get("claim_id") or raw_claim.get("id") or f"HCL{default_id}"),
            "hypothesis_id": _clean_string(raw_claim.get("hypothesis_id") or raw_claim.get("hypothesis") or ""),
            "text": text,
            "claim_type": _clean_string(raw_claim.get("claim_type") or raw_claim.get("type") or "claim"),
            "support_ids": support_ids,
            "evidence_ids": evidence_ids,
            "raw_claim": raw_claim,
        }

    
    
    

    def score_task(
        self,
        task: Dict[str, object],
        parsed_response: Dict[str, object],
        *,
        expected_output_count: Optional[int] = None,
    ) -> Dict[str, object]:
        valid_hypotheses = list(task.get("valid_hypotheses") or [])
        task_no_valid = bool(task.get("no_valid_hypothesis")) or not valid_hypotheses
        expected = int(expected_output_count or task.get("output_count") or self.expected_output_count)
        budget = min(expected, len(valid_hypotheses)) if valid_hypotheses else 0
        hypotheses = list(parsed_response.get("hypotheses") or [])[:expected]
        response_no_valid = bool(parsed_response.get("no_valid_hypothesis"))

        if task_no_valid and response_no_valid and not hypotheses:
            return self._empty_task_score(
                task,
                budget=budget,
                imagination_excluded=True,
                hallucination_raw=0.0,
                no_valid_correct=True,
                note="correct_no_valid_hypothesis",
                parsed_response=parsed_response,
            )

        if task_no_valid:
            hypothesis_scores = [self.score_hypothesis(task, hypothesis) for hypothesis in hypotheses]
            hallucination_raw = max(0.65, mean_or_none([item["hallucination_raw"] for item in hypothesis_scores]) or 0.0)
            if not hypotheses:
                hallucination_raw = 0.20
            return self._compose_task_score(
                task,
                hypothesis_scores,
                budget=budget,
                imagination_raw_override=0.0,
                hallucination_raw_override=hallucination_raw,
                imagination_excluded=True,
                no_valid_correct=False,
                note="no_valid_task_but_hypotheses" if hypotheses else "no_valid_task_without_abstention_flag",
                parsed_response=parsed_response,
            )

        if response_no_valid and not hypotheses:
            return self._empty_task_score(
                task,
                budget=budget,
                imagination_excluded=False,
                hallucination_raw=self._no_valid_reason_hallucination(task, parsed_response.get("reason") or ""),
                no_valid_correct=False,
                note="solvable_but_no_valid_claim",
                parsed_response=parsed_response,
            )

        hypothesis_scores = [self.score_hypothesis(task, hypothesis) for hypothesis in hypotheses]
        return self._compose_task_score(
            task,
            hypothesis_scores,
            budget=budget,
            imagination_excluded=False,
            no_valid_correct=not task_no_valid,
            note="scored_hypotheses",
            parsed_response=parsed_response,
        )

    def score_hypothesis(self, task: Dict[str, object], hypothesis: Dict[str, object]) -> Dict[str, object]:
        entity_index = self._entity_index(task)
        operation_index = self._operation_index(task)
        mechanism_index = self._mechanism_index(task)
        effect_index = self._effect_index(task)
        valid_index = self._valid_hypothesis_index(task)

        entities, unknown_entities = self._canonicalize_terms(hypothesis.get("entities") or [], entity_index)
        operations, unknown_operations = self._canonicalize_terms(hypothesis.get("operation_tags") or [], operation_index)
        mechanisms, unknown_mechanisms = self._canonicalize_terms(hypothesis.get("mechanism_tags") or [], mechanism_index)
        effects, unknown_effects = self._canonicalize_terms(hypothesis.get("expected_effects") or [], effect_index)
        canonical_key = _canonical_key(entities, operations, mechanisms, effects)
        matched = valid_index.get(canonical_key)

        full_text = self._hypothesis_text(hypothesis)
        schema_unverifiable = self._schema_unverifiable(hypothesis)
        implicit_unavailable = self._implicit_unavailable_entities(task, full_text, entity_index)
        unavailable_entity = clip01(
            (len(unknown_entities) + len(implicit_unavailable)) /
            max(1, len(hypothesis.get("entities") or []) + len(implicit_unavailable))
        )
        constraint_record = self._constraint_violations(task, full_text)
        contradiction_record = self._contradictions(task, full_text, entities)
        support_record = self._mechanism_support(
            task,
            full_text,
            entities,
            mechanisms,
            matched_valid_hypothesis=matched,
        )
        evidence_record = self._evidence_alignment(task, hypothesis, full_text, matched_valid_hypothesis=matched)
        soft_match = self._soft_valid_match(
            task,
            full_text,
            entities=entities,
            operations=operations,
            mechanisms=mechanisms,
            effects=effects,
            evidence_support=evidence_record["evidence_support"],
        )
        matched_for_quality = matched or soft_match.get("matched_valid_hypothesis")
        if matched is None and matched_for_quality:
            support_record = self._mechanism_support(
                task,
                full_text,
                entities,
                mechanisms,
                matched_valid_hypothesis=matched_for_quality,
            )
            evidence_record = self._evidence_alignment(
                task,
                hypothesis,
                full_text,
                matched_valid_hypothesis=matched_for_quality,
            )
            soft_match = self._soft_valid_match(
                task,
                full_text,
                entities=entities,
                operations=operations,
                mechanisms=mechanisms,
                effects=effects,
                evidence_support=evidence_record["evidence_support"],
            )
            matched_for_quality = matched or soft_match.get("matched_valid_hypothesis")
        valid = (
            matched is not None
            and unavailable_entity == 0.0
            and constraint_record["score"] == 0.0
            and contradiction_record["score"] == 0.0
        )
        unknown_support_ratio = (
            (len(unknown_operations) + len(unknown_mechanisms) + len(unknown_effects)) /
            max(1, len(hypothesis.get("operation_tags") or []) + len(hypothesis.get("mechanism_tags") or []) + len(hypothesis.get("expected_effects") or []))
        )
        if valid:
            unsupported = 0.0
        else:
            structured_but_not_in_space = bool(entities or operations or mechanisms or effects)
            unsupported = max(
                clip01(unknown_support_ratio),
                clip01(1.0 - support_record["affordance_support"]) if mechanisms else 0.0,
                0.35 if structured_but_not_in_space else 0.0,
            )

        explicit_contradiction_or_forbidden_foil = contradiction_record["score"]
        constraint_or_observation_violation = constraint_record["score"]
        false_feasibility_claim = self._false_feasibility_claim(
            full_text,
            valid=valid,
            unsupported=unsupported,
            unavailable_entity=unavailable_entity,
            constraint_violation=constraint_or_observation_violation,
            contradiction=explicit_contradiction_or_forbidden_foil,
        )
        legacy_hallucination_raw = clip01(
            0.30 * schema_unverifiable +
            0.25 * unavailable_entity +
            0.20 * unsupported +
            0.15 * constraint_or_observation_violation +
            0.10 * explicit_contradiction_or_forbidden_foil
        )
        evidence_hallucination_raw = clip01(
            0.30 * evidence_record["citation_mismatch"] +
            0.25 * evidence_record["missing_required_citation"] +
            0.25 * evidence_record["evidence_boundary_violation"] +
            0.20 * false_feasibility_claim
        )
        hallucination_raw = clip01(
            0.70 * legacy_hallucination_raw +
            0.30 * evidence_hallucination_raw
        )
        if (
            unavailable_entity >= 1.0
            or constraint_or_observation_violation >= 1.0
            or explicit_contradiction_or_forbidden_foil >= 1.0
            or evidence_record["evidence_boundary_violation"] >= 1.0
        ):
            hallucination_raw = max(hallucination_raw, 0.70)

        common_record = self._common_hypothesis_record(task, full_text, matched_for_quality)
        rarity_v3 = self._rarity_v3(matched_for_quality, common_record)
        valid_match_quality = 1.0 if matched else clip01(soft_match.get("valid_match_quality"))
        evidence_synthesis = self._evidence_synthesis_score(hypothesis, evidence_record, support_record, matched_for_quality)
        mechanism_depth = self._mechanism_depth_score(hypothesis, support_record, valid_match_quality)
        testability_boundary_awareness = self._testability_boundary_awareness(task, hypothesis, full_text)
        hard_gate = self._hypothesis_hard_gate(
            evidence_required=bool(task.get("evidence_pack") or task.get("requires_claim_ledger")),
            unavailable_entity=unavailable_entity,
            constraint_violation=constraint_or_observation_violation,
            contradiction=explicit_contradiction_or_forbidden_foil,
            evidence_record=evidence_record,
            valid_match_quality=valid_match_quality,
            common_record=common_record,
        )
        multiplier_weights = self.v3_params.get("hypothesis_multiplier_weights") or DEFAULT_HYPOUSESPACE_V3_PARAMS["hypothesis_multiplier_weights"]
        multiplier = clip01(
            float(multiplier_weights.get("base", 0.45)) +
            float(multiplier_weights.get("evidence_synthesis", 0.25)) * evidence_synthesis +
            float(multiplier_weights.get("mechanism_depth", 0.20)) * mechanism_depth +
            float(multiplier_weights.get("testability_boundary_awareness", 0.10)) * testability_boundary_awareness
        )
        hypothesis_i = clip01(
            (rarity_v3 ** float(self.v3_params.get("rarity_gamma", 1.35))) *
            (valid_match_quality ** float(self.v3_params.get("match_quality_gamma", 1.35))) *
            hard_gate *
            multiplier
        )

        return {
            "version": "hypouse_space_hypothesis_v3",
            "hypothesis": hypothesis.get("hypothesis"),
            "canonical_key": canonical_key,
            "canonical_space_match_id": matched.get("id") if matched else None,
            "canonical": {
                "entities": sorted(entities),
                "operation_tags": sorted(operations),
                "mechanism_tags": sorted(mechanisms),
                "expected_effects": sorted(effects),
            },
            "valid": bool(valid),
            "matched_valid_hypothesis_id": matched.get("id") if (matched and valid) else None,
            "matched_rarity": round(float(matched.get("rarity", 0.5)), 4) if matched else None,
            "soft_valid_match_id": (
                (soft_match.get("matched_valid_hypothesis") or {}).get("id")
                if soft_match.get("matched_valid_hypothesis") else None
            ),
            "soft_valid_match_quality": round(valid_match_quality, 4),
            "soft_valid_match_components": soft_match.get("components") or {},
            "rarity_v3": round(rarity_v3, 4),
            "common_family_kind": common_record.get("kind"),
            "common_family_id": common_record.get("id"),
            "hard_gate": round(hard_gate, 4),
            "hypothesis_I": round(hypothesis_i, 4),
            "evidence_synthesis": round(evidence_synthesis, 4),
            "mechanism_depth": round(mechanism_depth, 4),
            "testability_boundary_awareness": round(testability_boundary_awareness, 4),
            "schema_unverifiable": round(schema_unverifiable, 4),
            "unavailable_entity": round(unavailable_entity, 4),
            "unsupported_affordance_or_mechanism": round(clip01(unsupported), 4),
            "constraint_or_observation_violation": round(constraint_or_observation_violation, 4),
            "explicit_contradiction_or_forbidden_foil": round(explicit_contradiction_or_forbidden_foil, 4),
            "mechanism_support": round(support_record["mechanism_support"], 4),
            "affordance_support": round(support_record["affordance_support"], 4),
            "evidence_support": round(evidence_record["evidence_support"], 4),
            "citation_mismatch": round(evidence_record["citation_mismatch"], 4),
            "missing_required_citation": round(evidence_record["missing_required_citation"], 4),
            "evidence_boundary_violation": round(evidence_record["evidence_boundary_violation"], 4),
            "false_feasibility_claim": round(false_feasibility_claim, 4),
            "hallucination_raw": round(hallucination_raw, 4),
            "legacy_hallucination_raw": round(legacy_hallucination_raw, 4),
            "evidence_hallucination_raw": round(evidence_hallucination_raw, 4),
            "unknown_entities": sorted(unknown_entities),
            "unknown_operations": sorted(unknown_operations),
            "unknown_mechanisms": sorted(unknown_mechanisms),
            "unknown_effects": sorted(unknown_effects),
            "unknown_evidence_ids": evidence_record["unknown_evidence_ids"],
            "cited_evidence_ids": evidence_record["cited_evidence_ids"],
            "implicit_unavailable_entities": sorted(implicit_unavailable),
            "support_hits": support_record["support_hits"],
            "evidence_hits": evidence_record["evidence_hits"],
            "constraint_hits": constraint_record["hits"],
            "forbidden_foil_hits": contradiction_record["forbidden_foil_hits"],
            "negative_affordance_hits": contradiction_record["negative_affordance_hits"],
            "formula": {
                "imagination": "I_i=rarity^1.35*valid_match_quality^1.35*hard_gate*(0.45+0.25*evidence_synthesis+0.20*mechanism_depth+0.10*testability_boundary_awareness)",
                "hallucination": "h_i=0.70*legacy_closed_world_h+0.30*evidence_h",
            },
        }

    def _alias_map(self, kind: str) -> Dict[str, List[str]]:
        global_aliases = self.alias_overlay.get("global") if isinstance(self.alias_overlay, dict) else {}
        values = (global_aliases or {}).get(kind) or {}
        return values if isinstance(values, dict) else {}

    def _aliases_for(self, kind: str, canonical_id: str) -> List[str]:
        aliases = self._alias_map(kind).get(str(canonical_id)) or []
        return [str(canonical_id), *[str(alias) for alias in aliases if str(alias)]]

    def _match_set_quality(self, predicted: Iterable[str], valid_values: Sequence[str], text: str, kind: str) -> float:
        predicted = {str(value) for value in predicted if str(value)}
        valid_values = [str(value) for value in valid_values if str(value)]
        if not valid_values:
            return 1.0
        hits = 0
        text_norm = _normalize_text(text)
        for value in valid_values:
            if value in predicted:
                hits += 1
                continue
            if any(_phrase_hit(text_norm, alias) for alias in self._aliases_for(kind, value)):
                hits += 1
        return clip01(hits / max(1, len(valid_values)))

    def _soft_valid_match(
        self,
        task: Dict[str, object],
        text: str,
        *,
        entities: Iterable[str],
        operations: Iterable[str],
        mechanisms: Iterable[str],
        effects: Iterable[str],
        evidence_support: float,
    ) -> Dict[str, object]:
        best = None
        best_quality = 0.0
        best_components = {}
        text_norm = _normalize_text(text)
        for valid_hypothesis in task.get("valid_hypotheses") or []:
            entity_match = self._match_set_quality(entities, valid_hypothesis.get("entities") or [], text, "entities")
            operation_match = self._match_set_quality(operations, valid_hypothesis.get("operation_tags") or [], text, "operations")
            mechanism_match = self._match_set_quality(mechanisms, valid_hypothesis.get("mechanism_tags") or [], text, "mechanisms")
            effect_match = self._match_set_quality(effects, valid_hypothesis.get("expected_effects") or [], text, "effects")
            evidence_keywords = list(valid_hypothesis.get("evidence_keywords") or [])
            keyword_hit_ratio = (
                len(_keyword_hits(text, evidence_keywords)) / max(1, min(4, len(evidence_keywords)))
                if evidence_keywords else 0.75
            )
            evidence_match = max(clip01(evidence_support), clip01(keyword_hit_ratio))
            quality = geometric_mean([
                entity_match,
                operation_match,
                mechanism_match,
                effect_match,
                evidence_match,
            ])
            if quality > best_quality:
                best = valid_hypothesis
                best_quality = quality
                best_components = {
                    "entity_match": round(entity_match, 4),
                    "operation_match": round(operation_match, 4),
                    "mechanism_match": round(mechanism_match, 4),
                    "effect_match": round(effect_match, 4),
                    "evidence_match": round(evidence_match, 4),
                }
        if best_quality < float(self.v3_params.get("soft_match_threshold", 0.50)):
            best = None
        return {
            "matched_valid_hypothesis": best,
            "valid_match_quality": clip01(best_quality),
            "components": best_components,
        }

    def _task_common_bank(self, task: Dict[str, object]) -> Dict[str, object]:
        tasks = self.common_bank.get("tasks") if isinstance(self.common_bank, dict) else {}
        record = tasks.get(str(task.get("id"))) if isinstance(tasks, dict) else None
        return record if isinstance(record, dict) else {}

    def _family_keyword_score(self, text: str, family: Dict[str, object]) -> float:
        keywords = [str(keyword) for keyword in family.get("keywords") or [] if str(keyword)]
        if not keywords:
            return 0.0
        hits = _keyword_hits(text, keywords)
        return clip01(len(hits) / max(1, min(3, len(keywords))))

    def _best_family_match(
        self,
        text: str,
        families: Sequence[Dict[str, object]],
        matched_valid_hypothesis: Optional[Dict[str, object]],
        *,
        forbidden: bool = False,
        require_valid_id: bool = False,
    ) -> Dict[str, object]:
        matched_id = str((matched_valid_hypothesis or {}).get("id") or "")
        best_family = None
        best_score = 0.0
        for family in families or []:
            if not isinstance(family, dict):
                continue
            family_ids = {str(item) for item in family.get("valid_ids") or []}
            if forbidden:
                hits = _forbidden_keyword_hits(text, family.get("keywords") or [])
                score = 1.0 if hits else 0.0
            else:
                if matched_id and matched_id in family_ids:
                    score = 1.0
                elif require_valid_id and family_ids:
                    score = 0.0
                else:
                    score = self._family_keyword_score(text, family)
            if score > best_score:
                best_family = family
                best_score = score
        if not best_family:
            return {"score": 0.0}
        return {
            "id": best_family.get("id"),
            "score": clip01(best_score),
            "family": best_family,
        }

    def _common_hypothesis_record(
        self,
        task: Dict[str, object],
        text: str,
        matched_valid_hypothesis: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        bank = self._task_common_bank(task)
        hard = self._best_family_match(
            text,
            bank.get("hard_zero_hypothesis_families") or [],
            matched_valid_hypothesis,
            forbidden=True,
        )
        if hard.get("score", 0.0) > 0.0:
            return {"kind": "hard_zero", **hard}
        broad = self._best_family_match(text, bank.get("broad_common_hypothesis_families") or [], matched_valid_hypothesis)
        supported = self._best_family_match(
            text,
            bank.get("supported_rare_hypothesis_families") or [],
            matched_valid_hypothesis,
            require_valid_id=True,
        )
        if supported.get("score", 0.0) >= 0.45 and supported.get("score", 0.0) >= broad.get("score", 0.0):
            return {"kind": "supported_rare", **supported}
        if broad.get("score", 0.0) >= 0.45:
            return {"kind": "broad_common", **broad}
        return {"kind": "unbanked", "score": 0.0, "id": None, "family": None}

    def _rarity_v3(
        self,
        matched_valid_hypothesis: Optional[Dict[str, object]],
        common_record: Dict[str, object],
    ) -> float:
        if common_record.get("kind") == "hard_zero":
            return 0.0
        raw_rarity = float((matched_valid_hypothesis or {}).get("rarity", 0.45) or 0.45)
        lift_base = float(self.v3_params.get("valid_rarity_lift_base", 0.0))
        rarity = clip01(raw_rarity)
        if lift_base > 0.0:
            rarity = lift_base + (1.0 - lift_base) * rarity
        if common_record.get("kind") == "broad_common":
            family = common_record.get("family") or {}
            cap = float(family.get("rarity_cap", self.v3_params.get("broad_common_rarity_cap", 0.50)))
            rarity = min(rarity, cap)
        elif common_record.get("kind") == "supported_rare":
            family = common_record.get("family") or {}
            floor = float(family.get("rarity_floor", self.v3_params.get("supported_rare_rarity_floor", 0.82)))
            rarity = max(rarity, floor)
        return clip01(rarity)

    def _evidence_synthesis_score(
        self,
        hypothesis: Dict[str, object],
        evidence_record: Dict[str, object],
        support_record: Dict[str, object],
        matched_valid_hypothesis: Optional[Dict[str, object]],
    ) -> float:
        chain = hypothesis.get("evidence_chain") or []
        chain_presence = clip01(len(chain) / 2.0)
        cited = 1.0 if evidence_record.get("cited_evidence_ids") else 0.0
        matched_keywords = 0.0
        if matched_valid_hypothesis:
            matched_keywords = clip01(
                len(_keyword_hits(self._hypothesis_text(hypothesis), matched_valid_hypothesis.get("evidence_keywords") or [])) /
                max(1, min(4, len(matched_valid_hypothesis.get("evidence_keywords") or [])))
            )
        return clip01(
            0.45 * clip01(evidence_record.get("evidence_support")) +
            0.20 * cited +
            0.20 * chain_presence +
            0.15 * max(clip01(support_record.get("mechanism_support")), matched_keywords)
        )

    def _mechanism_depth_score(
        self,
        hypothesis: Dict[str, object],
        support_record: Dict[str, object],
        valid_match_quality: float,
    ) -> float:
        core = _clean_string(hypothesis.get("core_mechanism"))
        core_tokens = [token for token in _normalize_text(core).split() if len(token) > 3]
        tag_specificity = clip01(len(hypothesis.get("mechanism_tags") or []) / 2.0)
        support_specificity = max(
            clip01(support_record.get("mechanism_support")),
            clip01(support_record.get("affordance_support")),
            clip01(valid_match_quality),
        )
        chain_depth = clip01(len(hypothesis.get("evidence_chain") or []) / 3.0)
        return clip01(
            0.25 * (1.0 if core_tokens else 0.0) +
            0.25 * tag_specificity +
            0.30 * support_specificity +
            0.20 * chain_depth
        )

    def _testability_boundary_awareness(self, task: Dict[str, object], hypothesis: Dict[str, object], text: str) -> float:
        boundary_note = _clean_string(hypothesis.get("boundary_note"))
        prediction = _clean_string(hypothesis.get("testable_prediction"))
        has_prediction = 1.0 if prediction else 0.0
        boundary_terms = []
        for constraint in task.get("constraints") or []:
            boundary_terms.append(constraint.get("description"))
            boundary_terms.extend(constraint.get("forbidden_keywords") or [])
            boundary_terms.extend(constraint.get("forbidden_entity_keywords") or [])
        for effect in task.get("goal_predicates") or []:
            boundary_terms.append(effect.get("id"))
            boundary_terms.append(effect.get("name"))
        boundary_hit = 1.0 if boundary_note and _keyword_hits(boundary_note + " " + text, [term for term in boundary_terms if term]) else 0.0
        return clip01(0.55 * has_prediction + 0.45 * boundary_hit)

    def _hypothesis_hard_gate(
        self,
        *,
        evidence_required: bool,
        unavailable_entity: float,
        constraint_violation: float,
        contradiction: float,
        evidence_record: Dict[str, object],
        valid_match_quality: float,
        common_record: Dict[str, object],
    ) -> float:
        if common_record.get("kind") == "hard_zero":
            return 0.0
        if max(unavailable_entity, constraint_violation, contradiction) > 0.0:
            return 0.0
        if evidence_record.get("evidence_boundary_violation", 0.0) >= 1.0:
            return 0.0
        if evidence_record.get("citation_mismatch", 0.0) >= 1.0:
            return 0.0
        gate = 1.0
        if valid_match_quality < float(self.v3_params.get("soft_match_threshold", 0.50)):
            gate = min(gate, 0.25)
        if evidence_required and evidence_record.get("evidence_support", 0.0) < float(self.v3_params.get("evidence_support_floor", 0.45)):
            gate = min(gate, 0.35)
        if max(unavailable_entity, constraint_violation, contradiction, evidence_record.get("citation_mismatch", 0.0)) > 0.0:
            gate = min(gate, 0.55)
        return clip01(gate)

    def _evidence_synthesis_depth(
        self,
        parsed_response: Dict[str, object],
        hypothesis_scores: Sequence[Dict[str, object]],
    ) -> float:
        claims = [claim for claim in parsed_response.get("claim_ledger") or [] if isinstance(claim, dict)]
        if not claims:
            chain_depth = mean_or_none([
                clip01(len(score.get("evidence_hits") or {}) / 3.0)
                for score in hypothesis_scores
            ]) or 0.0
            return clip01(0.35 * chain_depth)
        multi_evidence_claims = 0
        synthesis_claims = 0
        cited_ids = set()
        for claim in claims:
            evidence_ids = {
                _clean_string(item)
                for item in _as_list(claim.get("evidence_ids") or claim.get("support_ids"))
                if _clean_string(item)
            }
            cited_ids.update(evidence_ids)
            if len(evidence_ids) >= 2:
                multi_evidence_claims += 1
                text = _normalize_text(claim.get("text") or "")
                if any(marker in text for marker in ("combine", "together", "because", "therefore", "while", "so", "chain", "supports")):
                    synthesis_claims += 1
        multi_rate = clip01(multi_evidence_claims / max(1, len(claims)))
        synthesis_rate = clip01(synthesis_claims / max(1, len(claims)))
        evidence_span = clip01(len(cited_ids) / 4.0)
        item_depth = mean_or_none([
            clip01(score.get("evidence_synthesis"))
            for score in hypothesis_scores
        ]) or 0.0
        return clip01(0.35 * multi_rate + 0.25 * synthesis_rate + 0.20 * evidence_span + 0.20 * item_depth)

    def _compose_task_score(
        self,
        task: Dict[str, object],
        hypothesis_scores: Sequence[Dict[str, object]],
        *,
        budget: int,
        imagination_excluded: bool,
        no_valid_correct: bool,
        note: str,
        imagination_raw_override: Optional[float] = None,
        hallucination_raw_override: Optional[float] = None,
        parsed_response: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        parsed_response = dict(parsed_response or {})
        valid_index_by_id = {
            str(hypothesis.get("id")): hypothesis
            for hypothesis in task.get("valid_hypotheses") or []
            if hypothesis.get("id")
        }
        unique_valid = {}
        duplicate_valid_count = 0
        for score in hypothesis_scores:
            valid_id = score.get("matched_valid_hypothesis_id")
            if not valid_id:
                continue
            if valid_id in unique_valid:
                duplicate_valid_count += 1
            else:
                unique_valid[valid_id] = score

        budget = max(0, int(budget))
        brr = (len(unique_valid) / budget) if budget > 0 else 0.0
        denominator_weights = [
            float(item.get("rarity", 0.5) or 0.5)
            for item in valid_index_by_id.values()
        ]
        denominator = sum(denominator_weights)
        numerator = sum(float(valid_index_by_id[valid_id].get("rarity", 0.5) or 0.5) for valid_id in unique_valid)
        wrr = (numerator / denominator) if denominator > 0 else 0.0
        diversity = self._valid_hypothesis_diversity(task, unique_valid.keys())
        mechanism_support = mean_or_none([
            score.get("mechanism_support")
            for score in unique_valid.values()
        ]) or 0.0
        support_ledger = self._support_ledger_record(
            task,
            parsed_response,
            hypothesis_scores,
            no_valid_correct=no_valid_correct,
        )
        evidence_required = bool(task.get("evidence_pack") or task.get("requires_claim_ledger"))
        evidence_support_gate = 1.0
        if evidence_required and not (no_valid_correct and not hypothesis_scores):
            evidence_support_gate = clip01(
                0.45 * clip01(support_ledger.get("claim_support_precision")) +
                0.20 * clip01(support_ledger.get("claim_support_recall")) +
                0.20 * (1.0 - clip01(support_ledger.get("citation_mismatch_rate"))) +
                0.15 * (
                    mean_or_none([score.get("evidence_support") for score in hypothesis_scores]) or 0.0
                )
            )
        boundary_gate = clip01(
            1.0 - max(
                mean_or_none([score.get("unavailable_entity") for score in hypothesis_scores]) or 0.0,
                mean_or_none([score.get("constraint_or_observation_violation") for score in hypothesis_scores]) or 0.0,
                mean_or_none([score.get("evidence_boundary_violation") for score in hypothesis_scores]) or 0.0,
                clip01(support_ledger.get("entity_drift_rate")),
            )
        )
        quality_by_key = {}
        family_values = []
        for index, score in enumerate(hypothesis_scores):
            quality = clip01(score.get("hypothesis_I"))
            key = (
                score.get("matched_valid_hypothesis_id")
                or score.get("soft_valid_match_id")
                or (
                    f"common:{score.get('common_family_kind')}:{score.get('common_family_id')}"
                    if score.get("common_family_id") else None
                )
                or f"unmatched:{index}"
            )
            quality_by_key[key] = max(quality_by_key.get(key, 0.0), quality)
            if quality > 0.0:
                family_values.append(str(score.get("common_family_id") or score.get("soft_valid_match_id") or key))
        unique_quality_values = list(quality_by_key.values())
        quality_denominator = max(1, budget or min(self.expected_output_count, len(hypothesis_scores)) or len(hypothesis_scores) or 1)
        quality_mass_top3 = top_quality_mass(unique_quality_values, 3, denominator=quality_denominator)
        elite_tail_top2 = top_quality_mass(unique_quality_values, 2, denominator=quality_denominator)
        mechanism_diversity_eff = _entropy_ratio(
            family_values,
            max(2, len(task.get("valid_hypotheses") or []), len(set(family_values))),
        )
        evidence_synthesis_coverage = mean_or_none([
            score.get("evidence_synthesis") for score in hypothesis_scores
        ]) or 0.0
        evidence_synthesis_for_scoring = (
            evidence_synthesis_coverage
            if any(clip01(score.get("hypothesis_I")) > 0.0 for score in hypothesis_scores) else
            0.0
        )
        evidence_synthesis_depth = self._evidence_synthesis_depth(parsed_response, hypothesis_scores)
        evidence_synthesis_depth_for_scoring = (
            evidence_synthesis_depth
            if any(clip01(score.get("hypothesis_I")) > 0.0 for score in hypothesis_scores) else
            0.0
        )
        hard_valid_ids = {
            str(score.get("matched_valid_hypothesis_id"))
            for score in hypothesis_scores
            if score.get("matched_valid_hypothesis_id") and clip01(score.get("hard_gate")) >= 0.999
        }
        hard_valid_ratio = len(hard_valid_ids) / max(1, budget) if budget > 0 else 0.0
        soft_match_quality = mean_or_none([
            score.get("soft_valid_match_quality") for score in hypothesis_scores
        ]) or 0.0
        common_bank_record = self._task_common_bank(task)
        common_bank_coverage = 1.0 if (
            common_bank_record.get("hard_zero_hypothesis_families")
            and common_bank_record.get("broad_common_hypothesis_families")
            and common_bank_record.get("supported_rare_hypothesis_families")
        ) else 0.0
        alias_coverage = 1.0 if self.alias_overlay.get("global") else 0.0

        legacy_imagination_raw = clip01(0.40 * wrr + 0.25 * brr + 0.20 * diversity + 0.15 * mechanism_support)
        v3_imagination_raw = clip01(
            float((self.v3_params.get("task_aggregation_weights") or {}).get("quality_mass_top3", 0.40)) * quality_mass_top3 +
            float((self.v3_params.get("task_aggregation_weights") or {}).get("elite_tail_top2", 0.20)) * elite_tail_top2 +
            float((self.v3_params.get("task_aggregation_weights") or {}).get("mechanism_diversity_eff", 0.15)) * mechanism_diversity_eff +
            float((self.v3_params.get("task_aggregation_weights") or {}).get("evidence_synthesis_coverage", 0.10)) * evidence_synthesis_for_scoring +
            float((self.v3_params.get("task_aggregation_weights") or {}).get("evidence_synthesis_depth", 0.10)) * evidence_synthesis_depth_for_scoring +
            float((self.v3_params.get("task_aggregation_weights") or {}).get("hard_valid_ratio", 0.05)) * hard_valid_ratio
        )
        if (
            common_bank_coverage <= 0.0
            and any(score.get("valid") for score in hypothesis_scores)
            and clip01(support_ledger.get("citation_mismatch_rate")) <= 0.0
            and clip01(support_ledger.get("contradicted_claim_rate")) <= 0.0
            and not any(clip01(score.get("evidence_boundary_violation")) > 0.0 for score in hypothesis_scores)
        ):
            v3_imagination_raw = max(v3_imagination_raw, legacy_imagination_raw)
        imagination_raw = (
            float(imagination_raw_override)
            if imagination_raw_override is not None else
            v3_imagination_raw
        )
        imagination_gated = (
            0.0 if imagination_excluded else
            clip01(imagination_raw * (0.50 + 0.25 * evidence_support_gate + 0.25 * boundary_gate))
        )
        support_ledger_hallucination = clip01(
            0.30 * clip01(support_ledger.get("unsupported_span_rate")) +
            0.25 * clip01(support_ledger.get("citation_mismatch_rate")) +
            0.20 * clip01(support_ledger.get("contradicted_claim_rate")) +
            0.15 * clip01(support_ledger.get("claim_without_evidence_rate")) +
            0.10 * clip01(support_ledger.get("entity_drift_rate"))
        )
        hypothesis_hallucination_raw = mean_or_none([score.get("hallucination_raw") for score in hypothesis_scores]) or 0.0
        hallucination_raw = (
            float(hallucination_raw_override)
            if hallucination_raw_override is not None else
            clip01(0.65 * hypothesis_hallucination_raw + 0.35 * support_ledger_hallucination)
        )
        imagination = clip01(imagination_gated - self.beta_ih * hallucination_raw)
        hallucination = clip01(hallucination_raw - self.beta_hi * imagination_gated)
        if imagination_excluded:
            imagination = None

        primitive_means = self._primitive_means(hypothesis_scores)
        primitive_means.update({
            "BRR": round(clip01(brr), 4),
            "WRR": round(clip01(wrr), 4),
            "D_diversity": round(clip01(diversity), 4),
            "M_mechanism_support": round(clip01(mechanism_support), 4),
            "validity_rate": round(
                sum(1 for item in hypothesis_scores if item.get("valid")) / max(1, len(hypothesis_scores)),
                4,
            ) if hypothesis_scores else 0.0,
            "claim_support_precision": support_ledger.get("claim_support_precision", 1.0 if not evidence_required else 0.0),
            "claim_support_recall": support_ledger.get("claim_support_recall", 1.0 if not evidence_required else 0.0),
            "unsupported_span_rate": support_ledger.get("unsupported_span_rate", 0.0),
            "citation_mismatch_rate": support_ledger.get("citation_mismatch_rate", 0.0),
            "contradicted_claim_rate": support_ledger.get("contradicted_claim_rate", 0.0),
            "claim_without_evidence_rate": support_ledger.get("claim_without_evidence_rate", 0.0),
            "unknown_evidence_rate": support_ledger.get("unknown_evidence_rate", 0.0),
            "evidence_support_gate": round(clip01(evidence_support_gate), 4),
            "boundary_gate": round(clip01(boundary_gate), 4),
            "support_ledger_hallucination": round(clip01(support_ledger_hallucination), 4),
            "false_feasibility_claim": round(
                mean_or_none([score.get("false_feasibility_claim") for score in hypothesis_scores]) or 0.0,
                4,
            ),
            "quality_mass_top3": round(clip01(quality_mass_top3), 4),
            "elite_tail_top2": round(clip01(elite_tail_top2), 4),
            "mechanism_diversity_eff": round(clip01(mechanism_diversity_eff), 4),
            "evidence_synthesis_coverage": round(clip01(evidence_synthesis_coverage), 4),
            "evidence_synthesis_depth": round(clip01(evidence_synthesis_depth), 4),
            "hard_valid_ratio": round(clip01(hard_valid_ratio), 4),
            "soft_match_quality": round(clip01(soft_match_quality), 4),
            "common_bank_coverage": round(clip01(common_bank_coverage), 4),
            "alias_coverage": round(clip01(alias_coverage), 4),
            "hypothesis_I": round(
                mean_or_none([score.get("hypothesis_I") for score in hypothesis_scores]) or 0.0,
                4,
            ),
            "evidence_synthesis": round(clip01(evidence_synthesis_coverage), 4),
            "mechanism_depth": round(
                mean_or_none([score.get("mechanism_depth") for score in hypothesis_scores]) or 0.0,
                4,
            ),
        })
        task_no_valid = bool(task.get("no_valid_hypothesis")) or not bool(valid_index_by_id)
        response_no_valid = bool(parsed_response.get("no_valid_hypothesis"))
        false_acceptance = 1.0 if task_no_valid and bool(hypothesis_scores) else 0.0
        false_refusal = 1.0 if (not task_no_valid and response_no_valid and not hypothesis_scores) else 0.0
        correct_abstention = 1.0 if task_no_valid and no_valid_correct else 0.0
        boundary_accuracy = 1.0 if (
            (task_no_valid and no_valid_correct) or
            (not task_no_valid and not false_refusal)
        ) else 0.0
        boundary_record = {
            "boundary_accuracy": round(clip01(boundary_accuracy), 4),
            "correct_abstention": round(clip01(correct_abstention), 4),
            "false_acceptance": round(clip01(false_acceptance), 4),
            "false_refusal": round(clip01(false_refusal), 4),
        }

        task_result = {
            "version": HYPOUSESPACE_VERSION,
            "task_id": task.get("id"),
            "task_subtype": task.get("task_subtype"),
            "score": round(imagination, 4) if imagination is not None else None,
            "imagination": round(imagination, 4) if imagination is not None else None,
            "hallucination": round(hallucination, 4),
            "imagination_raw": round(clip01(imagination_raw), 4),
            "imagination_gated": round(clip01(imagination_gated), 4) if not imagination_excluded else None,
            "hallucination_raw": round(clip01(hallucination_raw), 4),
            "BRR": round(clip01(brr), 4),
            "WRR": round(clip01(wrr), 4),
            "D_diversity": round(clip01(diversity), 4),
            "M_mechanism_support": round(clip01(mechanism_support), 4),
            "quality_mass_top3": round(clip01(quality_mass_top3), 4),
            "elite_tail_top2": round(clip01(elite_tail_top2), 4),
            "mechanism_diversity_eff": round(clip01(mechanism_diversity_eff), 4),
            "evidence_synthesis_coverage": round(clip01(evidence_synthesis_coverage), 4),
            "evidence_synthesis_depth": round(clip01(evidence_synthesis_depth), 4),
            "hard_valid_ratio": round(clip01(hard_valid_ratio), 4),
            "soft_match_quality": round(clip01(soft_match_quality), 4),
            "common_bank_coverage": round(clip01(common_bank_coverage), 4),
            "alias_coverage": round(clip01(alias_coverage), 4),
            "boundary_record": boundary_record,
            "budget": budget,
            "valid_space_size": len(valid_index_by_id),
            "recovered_unique_count": len(unique_valid),
            "recovered_valid_hypotheses": sorted(unique_valid.keys()),
            "duplicate_valid_count": duplicate_valid_count,
            "scored_hypotheses": len(hypothesis_scores),
            "hypothesis_scores": list(hypothesis_scores),
            "primitive_means": primitive_means,
            "support_ledger": support_ledger,
            "imagination_excluded": bool(imagination_excluded),
            "no_valid_hypothesis": task_no_valid,
            "no_valid_correct": bool(no_valid_correct),
            "note": note,
            "formula": {
                "BRR": "BRR_q=|unique valid recovered|/min(N_q,|H_q|)",
                "WRR": "WRR_q=sum(recovered rarity weights)/sum(all rarity weights in H_q)",
                "imagination_raw": "v3 I_raw=0.40*top3_quality_mass+0.20*elite_tail_top2+0.15*mechanism_diversity_eff+0.10*evidence_synthesis_coverage+0.10*evidence_synthesis_depth+0.05*hard_valid_ratio",
                "imagination_gated": "I_gated=I_raw*(0.50+0.25*evidence_support_gate+0.25*boundary_gate)",
                "hallucination_raw": "H_raw=0.65*mean_i(h_i)+0.35*support_ledger_h",
                "residual": "I=clip01(I_gated-beta_IH*H_raw); H=clip01(H_raw-beta_HI*I_gated)",
            },
            "residualization": {
                "beta_IH": self.beta_ih,
                "beta_HI": self.beta_hi,
                "source": "benchmark_default",
                "standardization": "clip01_raw_v1",
            },
        }
        task_result["subtype_contributions"] = build_hypospace_task_subtype_contributions(
            task_result,
            beta_ih=self.beta_ih,
            beta_hi=self.beta_hi,
        )
        task_result["atom_signals"] = task_result["subtype_contributions"].get("atom_signals", {})
        return task_result

    def _empty_task_score(
        self,
        task: Dict[str, object],
        *,
        budget: int,
        imagination_excluded: bool,
        hallucination_raw: float,
        no_valid_correct: bool,
        note: str,
        parsed_response: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        return self._compose_task_score(
            task,
            [],
            budget=budget,
            imagination_raw_override=0.0,
            hallucination_raw_override=hallucination_raw,
            imagination_excluded=imagination_excluded,
            no_valid_correct=no_valid_correct,
            note=note,
            parsed_response=parsed_response,
        )

    
    
    

    def _entity_index(self, task: Dict[str, object]) -> Dict[str, str]:
        index = {}
        for entity in task.get("available_entities") or []:
            entity_id = str(entity.get("id") or "").strip()
            if not entity_id:
                continue
            aliases = [entity_id, entity.get("name"), *list(entity.get("aliases") or []), *self._aliases_for("entities", entity_id)]
            for alias in aliases:
                alias_norm = _normalize_text(str(alias or ""))
                if alias_norm:
                    index.setdefault(alias_norm, entity_id)
        return index

    def _operation_index(self, task: Dict[str, object]) -> Dict[str, str]:
        index = {}
        for operation in task.get("allowed_operations") or []:
            operation_id = str(operation.get("id") or "").strip()
            if not operation_id:
                continue
            aliases = [operation_id, operation.get("name"), *list(operation.get("aliases") or []), *self._aliases_for("operations", operation_id)]
            for alias in aliases:
                alias_norm = _normalize_text(str(alias or ""))
                if alias_norm:
                    index.setdefault(alias_norm, operation_id)
        return index

    def _mechanism_index(self, task: Dict[str, object]) -> Dict[str, str]:
        index = {}
        tags = set()
        for entity in task.get("available_entities") or []:
            affordances = entity.get("affordances") or {}
            if isinstance(affordances, dict):
                tags.update(str(tag) for tag in affordances.keys())
        for hypothesis in task.get("valid_hypotheses") or []:
            tags.update(str(tag) for tag in hypothesis.get("mechanism_tags") or [])
        for tag in tags:
            for alias in self._aliases_for("mechanisms", tag):
                tag_norm = _normalize_text(alias)
                if tag_norm:
                    index.setdefault(tag_norm, tag)
        return index

    def _effect_index(self, task: Dict[str, object]) -> Dict[str, str]:
        index = {}
        for effect in task.get("goal_predicates") or []:
            effect_id = str(effect.get("id") or "").strip()
            if not effect_id:
                continue
            aliases = [effect_id, effect.get("name"), *list(effect.get("keywords") or []), *self._aliases_for("effects", effect_id)]
            for alias in aliases:
                alias_norm = _normalize_text(str(alias or ""))
                if alias_norm:
                    index.setdefault(alias_norm, effect_id)
        return index

    def _valid_hypothesis_index(self, task: Dict[str, object]) -> Dict[str, Dict[str, object]]:
        index = {}
        for hypothesis in task.get("valid_hypotheses") or []:
            key = _canonical_key(
                hypothesis.get("entities") or [],
                hypothesis.get("operation_tags") or [],
                hypothesis.get("mechanism_tags") or [],
                hypothesis.get("expected_effects") or [],
            )
            index[key] = hypothesis
        return index

    def _canonicalize_terms(self, values: Sequence[object], index: Dict[str, str]) -> Tuple[set, set]:
        canonical = set()
        unknown = set()
        for raw in values or []:
            raw_norm = _normalize_text(_clean_string(raw))
            if not raw_norm:
                continue
            if raw_norm in index:
                canonical.add(index[raw_norm])
            else:
                unknown.add(raw_norm)
        return canonical, unknown

    def _hypothesis_text(self, hypothesis: Dict[str, object]) -> str:
        pieces = [
            hypothesis.get("hypothesis"),
            " ".join(str(item) for item in hypothesis.get("entities") or []),
            " ".join(str(item) for item in hypothesis.get("operation_tags") or []),
            " ".join(str(item) for item in hypothesis.get("mechanism_tags") or []),
            " ".join(str(item) for item in hypothesis.get("expected_effects") or []),
            " ".join(str(item) for item in hypothesis.get("evidence") or []),
            " ".join(str(item) for item in hypothesis.get("evidence_ids") or []),
            hypothesis.get("core_mechanism"),
            " ".join(str(item) for item in hypothesis.get("evidence_chain") or []),
            hypothesis.get("why_distinct"),
            hypothesis.get("boundary_note"),
            hypothesis.get("testable_prediction"),
        ]
        return " ".join(_clean_string(piece) for piece in pieces if _clean_string(piece))

    def _schema_unverifiable(self, hypothesis: Dict[str, object]) -> float:
        required_checks = [
            bool(_clean_string(hypothesis.get("hypothesis"))),
            bool(hypothesis.get("entities")),
            bool(hypothesis.get("operation_tags")),
            bool(hypothesis.get("mechanism_tags")),
            bool(hypothesis.get("expected_effects")),
            bool(hypothesis.get("evidence")),
        ]
        return clip01(1.0 - (sum(1 for item in required_checks if item) / len(required_checks)))

    def _implicit_unavailable_entities(self, task: Dict[str, object], text: str, entity_index: Dict[str, str]) -> set:
        allowed_aliases = set(entity_index.keys())
        markers = set(COMMON_UNAVAILABLE_ENTITY_MARKERS)
        for constraint in task.get("constraints") or []:
            markers.update(str(item) for item in constraint.get("forbidden_entity_keywords") or [])
        hits = set()
        for marker in markers:
            marker_norm = _normalize_text(marker)
            if marker_norm and marker_norm not in allowed_aliases and _forbidden_keyword_hits(text, [marker_norm]):
                hits.add(marker_norm)
        return hits

    def _constraint_violations(self, task: Dict[str, object], text: str) -> Dict[str, object]:
        constraints = list(task.get("constraints") or [])
        if not constraints:
            return {"score": 0.0, "hits": []}
        violated = []
        for constraint in constraints:
            keywords = list(constraint.get("forbidden_keywords") or []) + list(constraint.get("forbidden_entity_keywords") or [])
            hits = _forbidden_keyword_hits(text, keywords)
            if hits:
                violated.append({"id": constraint.get("id"), "hits": hits})
        return {
            "score": clip01(len(violated) / max(1, len(constraints))),
            "hits": violated,
        }

    def _contradictions(self, task: Dict[str, object], text: str, entities: Iterable[str]) -> Dict[str, object]:
        foil_hits = _forbidden_keyword_hits(text, task.get("forbidden_foils") or [])
        entity_by_id = {
            str(entity.get("id")): entity
            for entity in task.get("available_entities") or []
            if entity.get("id")
        }
        negative_hits = []
        for entity_id in entities:
            entity = entity_by_id.get(entity_id)
            if not entity:
                continue
            negatives = entity.get("negative_affordances") or {}
            if isinstance(negatives, dict):
                iterable = negatives.items()
            else:
                iterable = [(str(item), {"keywords": [str(item)]}) for item in negatives]
            for tag, spec in iterable:
                keywords = list(spec.get("keywords") or []) if isinstance(spec, dict) else [str(spec)]
                aliases = [entity_id, entity.get("name"), *list(entity.get("aliases") or [])]
                hits = self._local_negative_hits(text, aliases, keywords)
                if hits:
                    negative_hits.append({"entity": entity_id, "tag": str(tag), "hits": hits})
        score = clip01((len(foil_hits) + len(negative_hits)) / 2.0)
        return {
            "score": score,
            "forbidden_foil_hits": foil_hits,
            "negative_affordance_hits": negative_hits,
        }

    def _local_negative_hits(self, text: str, aliases: Sequence[object], keywords: Sequence[str]) -> List[str]:
        text_norm = _normalize_text(text)
        alias_norms = [_normalize_text(str(alias or "")) for alias in aliases if _normalize_text(str(alias or ""))]
        hits = []
        for keyword in keywords or []:
            keyword_norm = _normalize_text(str(keyword))
            if not keyword_norm or not _forbidden_keyword_hits(text, [keyword_norm]):
                continue
            keyword_tokens = keyword_norm.split()
            if len(keyword_tokens) > 1 and any(entity_word in keyword_tokens for alias in alias_norms for entity_word in alias.split()):
                hits.append(str(keyword))
                continue
            local_hit = any(
                f"{alias} {keyword_norm}" in text_norm or f"{keyword_norm} {alias}" in text_norm
                for alias in alias_norms
            )
            if local_hit:
                hits.append(str(keyword))
        return sorted(set(hits))

    def _mechanism_support(
        self,
        task: Dict[str, object],
        text: str,
        entities: Iterable[str],
        mechanisms: Iterable[str],
        *,
        matched_valid_hypothesis: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        entities = set(entities)
        mechanisms = set(mechanisms)
        entity_by_id = {
            str(entity.get("id")): entity
            for entity in task.get("available_entities") or []
            if entity.get("id")
        }
        supported_mechanisms = set()
        support_hits = {}
        for entity_id in entities:
            entity = entity_by_id.get(entity_id)
            if not entity:
                continue
            affordances = entity.get("affordances") or {}
            if isinstance(affordances, dict):
                iterable = affordances.items()
            else:
                iterable = [(str(item), {"keywords": [str(item)]}) for item in affordances]
            for tag, spec in iterable:
                if tag not in mechanisms:
                    continue
                keywords = list(spec.get("keywords") or []) + [str(tag)] if isinstance(spec, dict) else [str(tag), str(spec)]
                hits = _keyword_hits(text, keywords)
                supported_mechanisms.add(tag)
                if hits:
                    support_hits.setdefault(entity_id, []).extend([str(tag), *hits])

        affordance_support = (
            len(supported_mechanisms & mechanisms) / max(1, len(mechanisms))
            if mechanisms else 0.0
        )
        mechanism_support = 0.0
        if matched_valid_hypothesis:
            evidence_keywords = list(matched_valid_hypothesis.get("evidence_keywords") or [])
            evidence_hits = _keyword_hits(text, evidence_keywords)
            if evidence_keywords:
                mechanism_support = max(0.55, len(evidence_hits) / max(1, min(4, len(evidence_keywords))))
            else:
                mechanism_support = 0.75
            if evidence_hits:
                support_hits.setdefault("valid_evidence", []).extend(evidence_hits)
            affordance_support = max(affordance_support, 0.75)
        else:
            mechanism_support = affordance_support

        return {
            "affordance_support": clip01(affordance_support),
            "mechanism_support": clip01(mechanism_support),
            "support_hits": {
                key: sorted(set(values))
                for key, values in support_hits.items()
            },
        }

    def _evidence_index(self, task: Dict[str, object]) -> Dict[str, Dict[str, object]]:
        evidence_pack = task.get("evidence_pack") or {}
        if not isinstance(evidence_pack, dict):
            return {}
        index = {}
        doc_id = _clean_string(evidence_pack.get("doc_id"))
        evidence_aliases = self._alias_map("evidence_keywords")
        if doc_id:
            index[doc_id] = {
                "kind": "doc",
                "id": doc_id,
                "text": _clean_string(evidence_pack.get("topic") or doc_id),
                "keywords": [doc_id, _clean_string(evidence_pack.get("topic")), *list(evidence_aliases.get(doc_id) or [])],
                "forbidden": False,
            }
        for claim in evidence_pack.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            claim_id = _clean_string(claim.get("claim_id") or claim.get("id"))
            if not claim_id:
                continue
            index[claim_id] = {
                "kind": "claim",
                "id": claim_id,
                "text": _clean_string(claim.get("text")),
                "keywords": [*list(claim.get("keywords") or []), *list(evidence_aliases.get(claim_id) or [])],
                "forbidden": False,
            }
        for claim in evidence_pack.get("forbidden_claims") or []:
            if not isinstance(claim, dict):
                continue
            claim_id = _clean_string(claim.get("claim_id") or claim.get("id"))
            if not claim_id:
                continue
            index[claim_id] = {
                "kind": "forbidden_claim",
                "id": claim_id,
                "text": _clean_string(claim.get("text")),
                "keywords": [*list(claim.get("keywords") or []), *list(evidence_aliases.get(claim_id) or [])],
                "forbidden": True,
            }
        return index

    def _evidence_alignment(
        self,
        task: Dict[str, object],
        hypothesis: Dict[str, object],
        text: str,
        *,
        matched_valid_hypothesis: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        evidence_index = self._evidence_index(task)
        if not evidence_index:
            return {
                "requires_evidence": False,
                "evidence_support": 1.0,
                "citation_mismatch": 0.0,
                "missing_required_citation": 0.0,
                "evidence_boundary_violation": 0.0,
                "unknown_evidence_ids": [],
                "cited_evidence_ids": [],
                "evidence_hits": {},
            }

        cited_ids = []
        for value in list(hypothesis.get("evidence_ids") or []):
            cleaned = _clean_string(value)
            if cleaned and cleaned not in cited_ids:
                cited_ids.append(cleaned)
        unknown_ids = [item for item in cited_ids if item not in evidence_index]
        known_refs = [evidence_index[item] for item in cited_ids if item in evidence_index]
        missing_required = 1.0 if not cited_ids else 0.0

        forbidden_hits = []
        support_hits = {}
        support_scores = []
        for ref in known_refs:
            keywords = list(ref.get("keywords") or [])
            text_keywords = [
                token for token in _normalize_text(ref.get("text") or "").split()
                if len(token) > 3
            ][:8]
            hits = _keyword_hits(text, [*keywords, *text_keywords])
            if ref.get("forbidden"):
                if hits or _phrase_hit(_normalize_text(text), ref.get("text") or ""):
                    forbidden_hits.append(ref["id"])
                support_scores.append(0.0)
                continue
            if ref.get("kind") == "doc":
                doc_hit = 0.0
                for evidence_ref in evidence_index.values():
                    if evidence_ref.get("forbidden") or evidence_ref.get("kind") != "claim":
                        continue
                    if _keyword_hits(text, list(evidence_ref.get("keywords") or [])):
                        doc_hit = max(doc_hit, 0.55)
                support_scores.append(doc_hit)
            else:
                score = 1.0 if hits else 0.0
                if not score and matched_valid_hypothesis:
                    valid_keywords = matched_valid_hypothesis.get("evidence_keywords") or []
                    score = 0.70 if _keyword_hits(" ".join(valid_keywords), list(ref.get("keywords") or [])) else 0.0
                support_scores.append(score)
            if hits:
                support_hits[ref["id"]] = hits

        if cited_ids:
            evidence_support = sum(support_scores) / max(1, len(known_refs)) if known_refs else 0.0
        else:
            evidence_support = 0.0
        evidence_support = clip01(evidence_support)
        boundary_terms = []
        evidence_pack = task.get("evidence_pack") or {}
        if isinstance(evidence_pack, dict):
            for claim in evidence_pack.get("forbidden_claims") or []:
                boundary_terms.extend(claim.get("keywords") or [])
                boundary_terms.append(_clean_string(claim.get("text")))
        boundary_hits = _forbidden_keyword_hits(text, boundary_terms)
        evidence_boundary = clip01(max(
            1.0 if unknown_ids else 0.0,
            1.0 if forbidden_hits or boundary_hits else 0.0,
        ))
        citation_mismatch = clip01(max(
            1.0 if unknown_ids else 0.0,
            1.0 - evidence_support if cited_ids else 0.0,
            1.0 if forbidden_hits else 0.0,
        ))
        return {
            "requires_evidence": True,
            "evidence_support": evidence_support,
            "citation_mismatch": citation_mismatch,
            "missing_required_citation": missing_required,
            "evidence_boundary_violation": evidence_boundary,
            "unknown_evidence_ids": unknown_ids,
            "cited_evidence_ids": cited_ids,
            "evidence_hits": {
                **{key: sorted(set(values)) for key, values in support_hits.items()},
                **({"forbidden_evidence": sorted(set(forbidden_hits))} if forbidden_hits else {}),
                **({"forbidden_terms": sorted(set(boundary_hits))} if boundary_hits else {}),
            },
        }

    def _false_feasibility_claim(
        self,
        text: str,
        *,
        valid: bool,
        unsupported: float,
        unavailable_entity: float,
        constraint_violation: float,
        contradiction: float,
    ) -> float:
        if valid:
            return 0.0
        problem_signal = max(
            clip01(unsupported),
            clip01(unavailable_entity),
            clip01(constraint_violation),
            clip01(contradiction),
        )
        if problem_signal <= 0.0:
            return 0.0
        feasibility_terms = [
            "works", "will work", "can", "can use", "stable", "supports",
            "creates", "achieves", "solves", "valid", "feasible", "produces",
            "generates", "emits", "holds", "retrieves",
        ]
        if _keyword_hits(text, feasibility_terms):
            return 1.0 if problem_signal >= 0.35 else 0.5
        return 0.5 if problem_signal >= 0.70 else 0.0

    def _support_ledger_record(
        self,
        task: Dict[str, object],
        parsed_response: Dict[str, object],
        hypothesis_scores: Sequence[Dict[str, object]],
        *,
        no_valid_correct: bool,
    ) -> Dict[str, object]:
        evidence_required = bool(task.get("evidence_pack") or task.get("requires_claim_ledger"))
        if not evidence_required and not parsed_response.get("claim_ledger"):
            return {
                "version": "support_ledger_neutral",
                "checked_claims": 0,
                "claim_support_precision": 1.0,
                "claim_support_recall": 1.0,
                "unsupported_span_rate": 0.0,
                "citation_mismatch_rate": 0.0,
                "contradicted_claim_rate": 0.0,
                "claim_without_evidence_rate": 0.0,
                "unknown_evidence_rate": 0.0,
                "entity_drift_rate": 0.0,
            }
        if no_valid_correct and not hypothesis_scores:
            return {
                "version": "support_ledger_correct_abstention",
                "checked_claims": 0,
                "claim_support_precision": 1.0,
                "claim_support_recall": 1.0,
                "unsupported_span_rate": 0.0,
                "citation_mismatch_rate": 0.0,
                "contradicted_claim_rate": 0.0,
                "claim_without_evidence_rate": 0.0,
                "unknown_evidence_rate": 0.0,
                "entity_drift_rate": 0.0,
            }

        full_text = " ".join([
            " ".join(self._hypothesis_text(item) for item in parsed_response.get("hypotheses") or []),
            " ".join(_clean_string((claim or {}).get("text") or (claim or {}).get("claim")) for claim in parsed_response.get("claim_ledger") or [] if isinstance(claim, dict)),
            _clean_string(parsed_response.get("reason")),
        ])
        ledger = self.support_ledger.score_response(
            task,
            {
                "claim_ledger": parsed_response.get("claim_ledger") or [],
                "hypotheses": parsed_response.get("hypotheses") or [],
            },
            constraint_profile={
                "require_evidence_ids": evidence_required,
                "require_claim_ledger": bool(task.get("requires_claim_ledger")),
            },
            full_text=full_text,
        )
        if evidence_required and not (parsed_response.get("claim_ledger") or []):
            ledger = dict(ledger)
            ledger["claim_support_precision"] = min(clip01(ledger.get("claim_support_precision")), 0.35)
            ledger["claim_support_recall"] = min(clip01(ledger.get("claim_support_recall")), 0.35)
            ledger["claim_without_evidence_rate"] = max(clip01(ledger.get("claim_without_evidence_rate")), 1.0)
            ledger["citation_mismatch_rate"] = max(clip01(ledger.get("citation_mismatch_rate")), 1.0)
        return ledger

    def _valid_hypothesis_diversity(self, task: Dict[str, object], valid_ids: Iterable[str]) -> float:
        valid_ids = set(valid_ids)
        if len(valid_ids) < 2:
            return 0.0
        valid_by_id = {
            str(hypothesis.get("id")): hypothesis
            for hypothesis in task.get("valid_hypotheses") or []
            if hypothesis.get("id")
        }
        family_values = {"entity": [], "operation": [], "effect": []}
        possible_values = {"entity": set(), "operation": set(), "effect": set()}
        for hypothesis in valid_by_id.values():
            families = hypothesis.get("families") or {}
            for field in family_values:
                value = str(families.get(field) or "")
                if value:
                    possible_values[field].add(value)
        for valid_id in valid_ids:
            families = (valid_by_id.get(valid_id) or {}).get("families") or {}
            for field in family_values:
                value = str(families.get(field) or "")
                if value:
                    family_values[field].append(value)
        entropies = [
            _entropy_ratio(values, len(possible_values[field]))
            for field, values in family_values.items()
            if values
        ]
        return mean_or_none(entropies) or 0.0

    def _primitive_means(self, hypothesis_scores: Sequence[Dict[str, object]]) -> Dict[str, float]:
        fields = [
            "schema_unverifiable",
            "unavailable_entity",
            "unsupported_affordance_or_mechanism",
            "constraint_or_observation_violation",
            "explicit_contradiction_or_forbidden_foil",
            "mechanism_support",
            "affordance_support",
            "evidence_support",
            "citation_mismatch",
            "missing_required_citation",
            "evidence_boundary_violation",
            "false_feasibility_claim",
            "hypothesis_I",
            "soft_valid_match_quality",
            "rarity_v3",
            "hard_gate",
            "evidence_synthesis",
            "mechanism_depth",
            "testability_boundary_awareness",
            "legacy_hallucination_raw",
            "evidence_hallucination_raw",
            "hallucination_raw",
        ]
        primitives = {}
        for field in fields:
            value = mean_or_none([score.get(field) for score in hypothesis_scores])
            if value is not None:
                primitives[field] = round(value, 4)
        return primitives

    def _no_valid_reason_hallucination(self, task: Dict[str, object], reason: str) -> float:
        reason_norm = _normalize_text(reason)
        if not reason_norm:
            return 0.10
        task_text = _normalize_text(" ".join([
            task.get("scene", ""),
            task.get("goal", ""),
            " ".join(str(constraint.get("description", "")) for constraint in task.get("constraints") or []),
        ]))
        reason_tokens = set(reason_norm.split())
        task_tokens = set(task_text.split())
        unsupported = len(reason_tokens - task_tokens) / max(1, len(reason_tokens))
        return clip01(0.10 + 0.30 * unsupported)


def aggregate_hypospace_model_axes(
    task_scores: Sequence[Dict[str, object]],
    *,
    gate_pass: bool = True,
    beta_ih: float = DEFAULT_HYPOUSESPACE_BETA_IH,
    beta_hi: float = DEFAULT_HYPOUSESPACE_BETA_HI,
) -> Dict[str, object]:
    if not task_scores:
        return {
            "version": HYPOUSESPACE_VERSION,
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

    imagination_task_scores = [
        score for score in task_scores
        if not score.get("imagination_excluded")
    ]
    imagination_raw = mean_or_none([score.get("imagination_raw") for score in imagination_task_scores])
    imagination_gated = mean_or_none([
        score.get("imagination_gated")
        for score in imagination_task_scores
        if score.get("imagination_gated") is not None
    ])
    imagination_basis = imagination_gated if imagination_gated is not None else imagination_raw
    hallucination_raw = mean_or_none([score.get("hallucination_raw") for score in task_scores])
    if gate_pass and imagination_basis is not None and hallucination_raw is not None:
        imagination = clip01(imagination_basis - beta_ih * hallucination_raw)
        hallucination = clip01(hallucination_raw - beta_hi * imagination_basis)
    else:
        imagination = None
        hallucination = None

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

    no_valid_items = [score for score in task_scores if score.get("no_valid_hypothesis")]
    no_valid_accuracy = mean_or_none([
        1.0 if score.get("no_valid_correct") else 0.0
        for score in no_valid_items
    ])
    subtype_contributions = mean_subtype_contributions(
        score.get("subtype_contributions")
        for score in task_scores
        if isinstance(score.get("subtype_contributions"), dict)
    )

    return {
        "version": HYPOUSESPACE_VERSION,
        "score": round(imagination, 4) if imagination is not None else None,
        "imagination": round(imagination, 4) if imagination is not None else None,
        "hallucination": round(hallucination, 4) if hallucination is not None else None,
        "imagination_raw": round(imagination_raw, 4) if imagination_raw is not None else None,
        "imagination_gated": round(imagination_gated, 4) if imagination_gated is not None else None,
        "hallucination_raw": round(hallucination_raw, 4) if hallucination_raw is not None else None,
        "primitive_means": primitive_means,
        "subtype_contributions": subtype_contributions,
        "task_count": len(task_scores),
        "imagination_task_count": len(imagination_task_scores),
        "quality_mass_top3": primitive_means.get("quality_mass_top3"),
        "elite_tail_top2": primitive_means.get("elite_tail_top2"),
        "mechanism_diversity_eff": primitive_means.get("mechanism_diversity_eff"),
        "evidence_synthesis_coverage": primitive_means.get("evidence_synthesis_coverage"),
        "evidence_synthesis_depth": primitive_means.get("evidence_synthesis_depth"),
        "hard_valid_ratio": primitive_means.get("hard_valid_ratio"),
        "soft_match_quality": primitive_means.get("soft_match_quality"),
        "common_bank_coverage": primitive_means.get("common_bank_coverage"),
        "alias_coverage": primitive_means.get("alias_coverage"),
        "coverage_gate_pass": bool(gate_pass),
        "no_valid_accuracy": round(no_valid_accuracy, 4) if no_valid_accuracy is not None else None,
        "residualization": {
            "beta_IH": beta_ih,
            "beta_HI": beta_hi,
            "source": HYPOUSESPACE_V3_CALIBRATION_POLICY,
            "standardization": "clip01_raw_v1",
        },
        "formula": {
            "task_imagination_raw": "v3 I_raw=0.40*top3_quality_mass+0.20*elite_tail_top2+0.15*mechanism_diversity_eff+0.10*evidence_synthesis_coverage+0.10*evidence_synthesis_depth+0.05*hard_valid_ratio",
            "task_imagination_gated": "I_gated=I_raw*(0.50+0.25*evidence_support_gate+0.25*boundary_gate)",
            "task_hallucination_raw": "H_raw=0.65*mean_i(closed_world/evidence_h)+0.35*support_ledger_h",
            "model_residual": "I=clip01(mean(I_gated)-beta_IH*mean(H_raw)); H=clip01(mean(H_raw)-beta_HI*mean(I_gated))",
        },
    }


def aggregate_hypospace_boundary_diagnostics(
    task_scores: Sequence[Dict[str, object]],
    *,
    beta_hi: float = DEFAULT_HYPOUSESPACE_BETA_HI,
) -> Dict[str, object]:
    if not task_scores:
        return {
            "version": HYPOUSESPACE_VERSION,
            "hallucination": None,
            "hallucination_raw": None,
            "task_count": 0,
            "boundary_accuracy": None,
            "boundary_record_means": {},
        }
    records = [score.get("boundary_record") or {} for score in task_scores]
    record_fields = sorted({field for record in records for field in record.keys()})
    record_means = {}
    for field in record_fields:
        value = mean_or_none(record.get(field) for record in records)
        if value is not None:
            record_means[field] = round(value, 4)
    hallucination_raw = mean_or_none([score.get("hallucination_raw") for score in task_scores]) or 0.0
    hallucination = clip01(hallucination_raw - beta_hi * 0.0)
    return {
        "version": HYPOUSESPACE_VERSION,
        "hallucination": round(hallucination, 4),
        "hallucination_raw": round(hallucination_raw, 4),
        "task_count": len(task_scores),
        "boundary_accuracy": record_means.get("boundary_accuracy"),
        "false_acceptance_rate": record_means.get("false_acceptance"),
        "false_refusal_rate": record_means.get("false_refusal"),
        "boundary_record_means": record_means,
        "formula": {
            "boundary_accuracy": "mean(correct abstention on no-valid tasks plus non-refusal on solvable boundary tasks)",
            "hallucination": "white-box boundary burden only; no external strength calibration",
        },
    }


__all__ = [
    "HYPOUSESPACE_VERSION",
    "DEFAULT_HYPOUSESPACE_BETA_IH",
    "DEFAULT_HYPOUSESPACE_BETA_HI",
    "DEFAULT_HYPOUSESPACE_OUTPUT_COUNT",
    "HYPOUSESPACE_V3_CALIBRATION_POLICY",
    "HYPOUSESPACE_V3_RUNTIME_SCORING_POLICY",
    "load_hypospace_common_hypothesis_bank",
    "load_hypospace_valid_match_aliases",
    "load_hypospace_v3_calibration_params",
    "get_hypospace_common_hypothesis_bank_coverage",
    "get_hypospace_valid_match_alias_coverage",
    "HypoUseSpaceScorer",
    "aggregate_hypospace_model_axes",
    "aggregate_hypospace_boundary_diagnostics",
]
