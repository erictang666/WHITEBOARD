
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from boundary_refusal_scorer import BoundaryRefusalScorer, expected_response_mode
from json_repair_utils import parse_jsonish_payload
from scorer_hyperparameters import get_scorer_hyperparameter
from typed_axis_aggregation import build_macgyver_task_subtype_contributions, mean_subtype_contributions


MACGYVER_DUAL_AXIS_VERSION = "macgyver_dual_axis"
DEFAULT_MACGYVER_BETA_IH = 0.00
DEFAULT_MACGYVER_BETA_HI = 0.90
DEFAULT_PLAN_COUNT = 5
MACGYVER_V3_CALIBRATION_POLICY = "benchmark_default"
MACGYVER_V3_RUNTIME_SCORING_POLICY = (
    "fixed output-only parameters"
)
MACGYVER_COMMON_PLAN_BANK_PATH = Path(__file__).resolve().parent / "data" / "macgyver_common_plan_bank_v3.json"
MACGYVER_SCORING_CONFIG_PATH = Path(__file__).resolve().parent / "data" / "macgyver_scoring_config.json"
DEFAULT_MACGYVER_V3_PARAMS = {
    "rarity_gamma": 1.35,
    "feasibility_gamma": 1.60,
    "hard_zero_threshold": 0.42,
    "broad_common_threshold": 0.38,
    "broad_common_floor": 0.25,
    "supported_rare_floor": 0.82,
    "task_weights": {
        "quality_mass": 0.35,
        "elite_tail": 0.25,
        "mechanism_chain_depth": 0.20,
        "constraint_juggling_score": 0.10,
        "strategy_diversity_eff": 0.05,
        "hard_valid_ratio": 0.05,
    },
    "top_quality_n": 3,
    "elite_tail_n": 2,
}

_COMMON_UNAVAILABLE_TOOL_MARKERS_FALLBACK = [
    "lighter",
    "matches",
    "match",
    "knife",
    "scissors",
    "saw",
    "drill",
    "screwdriver",
    "hammer",
    "nail",
    "screw",
    "glue",
    "superglue",
    "heat gun",
    "torch",
    "wrench",
    "jack",
    "battery",
    "motor",
    "plastic sheet",
    "wire",
    "rope",
    "string",
    "water",
]
COMMON_UNAVAILABLE_TOOL_MARKERS = set(get_scorer_hyperparameter(
    "macgyver",
    "COMMON_UNAVAILABLE_TOOL_MARKERS",
    default=_COMMON_UNAVAILABLE_TOOL_MARKERS_FALLBACK,
))

CAUSAL_MARKERS = {
    "because",
    "so",
    "therefore",
    "by",
    "using",
    "through",
    "allows",
    "prevents",
    "adds",
    "increases",
    "reduces",
    "holds",
    "supports",
    "secures",
}

SUCCESS_WORDS = {
    "stable",
    "stabilize",
    "fixed",
    "works",
    "solved",
    "secure",
    "safe",
    "complete",
    "achieve",
}


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


def _keyword_hits(text: str, keywords: Sequence[str]) -> List[str]:
    normalized = f" {_normalize_text(text)} "
    token_set = set(normalized.split())
    hits = []
    for keyword in keywords or []:
        keyword_norm = _normalize_text(str(keyword))
        if not keyword_norm:
            continue
        keyword_tokens = keyword_norm.split()
        if f" {keyword_norm} " in normalized or (keyword_tokens and set(keyword_tokens).issubset(token_set)):
            hits.append(str(keyword))
    return sorted(set(hits))


def _jaccard_distance(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = {str(item) for item in a if str(item)}
    set_b = {str(item) for item in b if str(item)}
    if not set_a and not set_b:
        return 0.0
    if not set_a or not set_b:
        return 1.0
    return 1.0 - (len(set_a & set_b) / len(set_a | set_b))


def _as_list(value) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", str(value)).strip()


def _extract_json_payload(raw_text):
    return parse_jsonish_payload(raw_text)


_COMMON_PLAN_BANK_CACHE = None
_CALIBRATION_PARAMS_CACHE = None


def load_macgyver_common_plan_bank(path: Optional[Path] = None) -> Dict[str, object]:
    global _COMMON_PLAN_BANK_CACHE
    if path is None and _COMMON_PLAN_BANK_CACHE is not None:
        return _COMMON_PLAN_BANK_CACHE
    bank_path = path or MACGYVER_COMMON_PLAN_BANK_PATH
    try:
        with bank_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        payload = {"schema": "missing_macgyver_common_plan_bank_v3", "tasks": {}}
    if path is None:
        _COMMON_PLAN_BANK_CACHE = payload
    return payload


def load_macgyver_v3_calibration_params(path: Optional[Path] = None) -> Dict[str, object]:
    global _CALIBRATION_PARAMS_CACHE
    if path is None and _CALIBRATION_PARAMS_CACHE is not None:
        return dict(_CALIBRATION_PARAMS_CACHE)
    params = dict(DEFAULT_MACGYVER_V3_PARAMS)
    calibration_path = path or MACGYVER_SCORING_CONFIG_PATH
    try:
        with calibration_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        payload = {}
    frozen = payload.get("final_scoring_params") if isinstance(payload, dict) else None
    if isinstance(frozen, dict):
        params.update({key: value for key, value in frozen.items() if key != "task_weights"})
        if isinstance(frozen.get("task_weights"), dict):
            weights = dict(DEFAULT_MACGYVER_V3_PARAMS["task_weights"])
            weights.update(frozen["task_weights"])
            params["task_weights"] = weights
    if path is None:
        _CALIBRATION_PARAMS_CACHE = dict(params)
    return dict(params)


def get_macgyver_common_plan_bank_coverage(task_ids: Sequence[str]) -> Dict[str, object]:
    bank = load_macgyver_common_plan_bank()
    tasks = bank.get("tasks") if isinstance(bank, dict) else {}
    covered = []
    missing = []
    for task_id in task_ids:
        record = tasks.get(task_id) if isinstance(tasks, dict) else None
        has_required = (
            isinstance(record, dict)
            and bool(record.get("hard_zero_plan_families"))
            and bool(record.get("broad_common_plan_families"))
        )
        (covered if has_required else missing).append(task_id)
    return {
        "covered": covered,
        "missing": missing,
        "coverage": round(len(covered) / max(1, len(task_ids)), 4),
    }


class MacGyverScorer:
    """Scores structured MacGyver responses against closed task metadata."""

    def __init__(
        self,
        *,
        beta_ih: float = DEFAULT_MACGYVER_BETA_IH,
        beta_hi: float = DEFAULT_MACGYVER_BETA_HI,
        expected_plan_count: int = DEFAULT_PLAN_COUNT,
        common_plan_bank: Optional[Dict[str, object]] = None,
        calibration_params: Optional[Dict[str, object]] = None,
    ):
        self.beta_ih = float(beta_ih)
        self.beta_hi = float(beta_hi)
        self.expected_plan_count = int(expected_plan_count)
        self.common_plan_bank = common_plan_bank or load_macgyver_common_plan_bank()
        self.v3_params = calibration_params or load_macgyver_v3_calibration_params()
        self.boundary_scorer = BoundaryRefusalScorer()

    
    
    

    def parse_response(self, raw_text: str) -> Dict[str, object]:
        payload = _extract_json_payload(raw_text)
        if isinstance(payload, list):
            payload = {"solvability": "solvable", "plans": payload, "impossibility_reason": None}
        if not isinstance(payload, dict):
            return {
                "parse_valid": False,
                "solvability": None,
                "plans": [],
                "impossibility_reason": None,
                "raw_payload": None,
                "parse_error": "no_json_object",
            }

        solvability = _normalize_text(str(payload.get("solvability") or payload.get("status") or ""))
        if "clarification" in solvability or "clarify" in solvability:
            solvability = "needs_clarification"
        elif "unsolvable" in solvability or "impossible" in solvability or solvability == "no solution":
            solvability = "unsolvable"
        elif "solvable" in solvability or "solution" in solvability or not solvability:
            solvability = "solvable"

        raw_plans = (
            payload.get("plans")
            or payload.get("solutions")
            or payload.get("ideas")
            or payload.get("strategies")
            or payload.get("approaches")
            or payload.get("answer")
        )
        if not raw_plans and any(
            key in payload
            for key in ("plan_name", "name", "title", "steps", "core_trick", "strategy", "solution")
        ):
            raw_plans = [payload]

        parsed_plans = []
        for raw_plan in _as_list(raw_plans):
            plan = self._parse_plan(raw_plan)
            if plan is not None:
                parsed_plans.append(plan)

        return {
            "parse_valid": True,
            "solvability": solvability,
            "plans": parsed_plans,
            "impossibility_reason": _clean_string(payload.get("impossibility_reason") or payload.get("reason")),
            "clarification_questions": [
                _clean_string(item)
                for item in _as_list(payload.get("clarification_questions") or payload.get("questions"))
                if _clean_string(item)
            ],
            "raw_payload": payload,
            "parse_error": None,
        }

    def _parse_plan(self, raw_plan) -> Optional[Dict[str, object]]:
        if isinstance(raw_plan, str):
            text = _clean_string(raw_plan)
            if not text:
                return None
            return {
                "plan_name": text[:80],
                "core_trick": "",
                "used_tools": [],
                "tool_chain": [],
                "steps": [{"action": text, "tools": [], "mechanism": "", "target_effect": ""}],
                "final_state": "",
                "failure_mode": "",
                "why_distinct": "",
                "risk_note": "",
                "raw_plan": raw_plan,
            }
        if not isinstance(raw_plan, dict):
            return None
        steps = []
        raw_steps = raw_plan.get("steps") or raw_plan.get("tool_steps") or raw_plan.get("actions")
        if not raw_steps and raw_plan.get("solution"):
            raw_steps = [raw_plan.get("solution")]
        for raw_step in _as_list(raw_steps):
            if isinstance(raw_step, dict):
                steps.append({
                    "action": _clean_string(raw_step.get("action") or raw_step.get("step") or raw_step.get("text")),
                    "tools": [_clean_string(item) for item in _as_list(raw_step.get("tools") or raw_step.get("used_tools")) if _clean_string(item)],
                    "mechanism": _clean_string(raw_step.get("mechanism") or raw_step.get("physics_rationale") or raw_step.get("rationale")),
                    "target_effect": _clean_string(raw_step.get("target_effect") or raw_step.get("effect") or raw_step.get("goal")),
                })
            else:
                text = _clean_string(raw_step)
                if text:
                    steps.append({"action": text, "tools": [], "mechanism": "", "target_effect": ""})
        if not steps:
            steps = [{
                "action": _clean_string(
                    raw_plan.get("action")
                    or raw_plan.get("idea")
                    or raw_plan.get("description")
                    or raw_plan.get("solution")
                ),
                "tools": [],
                "mechanism": "",
                "target_effect": "",
            }]
        if not any(step.get("action") for step in steps):
            return None
        return {
            "plan_name": _clean_string(
                raw_plan.get("plan_name")
                or raw_plan.get("name")
                or raw_plan.get("title")
                or raw_plan.get("idea")
                or raw_plan.get("solution")
            ),
            "core_trick": _clean_string(
                raw_plan.get("core_trick")
                or raw_plan.get("trick")
                or raw_plan.get("strategy")
                or raw_plan.get("mechanism")
            ),
            "used_tools": [_clean_string(item) for item in _as_list(raw_plan.get("used_tools") or raw_plan.get("tools") or raw_plan.get("materials")) if _clean_string(item)],
            "tool_chain": [_clean_string(item) for item in _as_list(raw_plan.get("tool_chain") or raw_plan.get("chain")) if _clean_string(item)],
            "steps": steps,
            "final_state": _clean_string(raw_plan.get("final_state") or raw_plan.get("result") or raw_plan.get("outcome")),
            "failure_mode": _clean_string(raw_plan.get("failure_mode") or raw_plan.get("limit") or raw_plan.get("risk")),
            "why_distinct": _clean_string(raw_plan.get("why_distinct") or raw_plan.get("distinctness") or raw_plan.get("difference")),
            "risk_note": _clean_string(raw_plan.get("risk_note") or raw_plan.get("safety_note") or raw_plan.get("constraint_note")),
            "raw_plan": raw_plan,
        }

    
    
    

    def score_task(self, task: Dict[str, object], parsed_response: Dict[str, object]) -> Dict[str, object]:
        expected_mode = expected_response_mode(task)
        task_unsolvable = expected_mode == "unsolvable"
        task_needs_clarification = expected_mode == "needs_clarification"
        response_says_unsolvable = parsed_response.get("solvability") == "unsolvable"
        response_says_clarification = parsed_response.get("solvability") == "needs_clarification"
        plans = self._dedupe_plans(parsed_response.get("plans") or [])

        if task_needs_clarification and response_says_clarification and not plans:
            boundary_record = self.boundary_scorer.score(task, parsed_response, plan_scores=[])
            return self._empty_task_score(
                task,
                solvability_correct=True,
                imagination_excluded=True,
                hallucination_raw=boundary_record["boundary_hallucination"],
                note="correct_needs_clarification",
                boundary_record=boundary_record,
            )

        if task_needs_clarification and plans:
            plan_scores = [self.score_plan(task, plan) for plan in plans[:self.expected_plan_count]]
            boundary_record = self.boundary_scorer.score(task, parsed_response, plan_scores=plan_scores)
            hallucination_raw = max(
                0.65,
                boundary_record["boundary_hallucination"],
                self._mean_first_n([item["hallucination_raw"] for item in plan_scores], fill=0.0),
            )
            return self._compose_task_score(
                task,
                plan_scores,
                diversity=0.0,
                imagination_raw_override=0.0,
                hallucination_raw_override=hallucination_raw,
                solvability_correct=False,
                imagination_excluded=True,
                note="clarification_needed_but_planned",
                boundary_record=boundary_record,
            )

        if task_needs_clarification:
            boundary_record = self.boundary_scorer.score(task, parsed_response, plan_scores=[])
            return self._empty_task_score(
                task,
                solvability_correct=False,
                imagination_excluded=True,
                hallucination_raw=max(0.55, boundary_record["boundary_hallucination"]),
                note="clarification_needed_but_not_asked",
                boundary_record=boundary_record,
            )

        if task_unsolvable and response_says_unsolvable and not plans:
            boundary_record = self.boundary_scorer.score(task, parsed_response, plan_scores=[])
            return self._empty_task_score(
                task,
                solvability_correct=True,
                imagination_excluded=True,
                hallucination_raw=boundary_record["boundary_hallucination"],
                note="correct_unsolvable",
                boundary_record=boundary_record,
            )

        if task_unsolvable and plans:
            plan_scores = [self.score_plan(task, plan) for plan in plans[:self.expected_plan_count]]
            boundary_record = self.boundary_scorer.score(task, parsed_response, plan_scores=plan_scores)
            hallucination_raw = max(
                0.65,
                boundary_record["boundary_hallucination"],
                self._mean_first_n([item["hallucination_raw"] for item in plan_scores], fill=0.0),
            )
            return self._compose_task_score(
                task,
                plan_scores,
                diversity=0.0,
                imagination_raw_override=0.0,
                hallucination_raw_override=hallucination_raw,
                solvability_correct=False,
                imagination_excluded=True,
                note="unsolvable_but_planned",
                boundary_record=boundary_record,
            )

        if not task_unsolvable and (response_says_unsolvable or response_says_clarification):
            boundary_record = self.boundary_scorer.score(task, parsed_response, plan_scores=[])
            reason_h = self._impossibility_reason_hallucination(task, parsed_response.get("impossibility_reason") or "")
            return self._empty_task_score(
                task,
                solvability_correct=False,
                imagination_excluded=False,
                hallucination_raw=max(reason_h, boundary_record["boundary_hallucination"]),
                note="solvable_but_refused",
                boundary_record=boundary_record,
            )

        plan_scores = [self.score_plan(task, plan) for plan in plans[:self.expected_plan_count]]
        diversity = self._plan_diversity(plan_scores)
        boundary_record = self.boundary_scorer.score(task, parsed_response, plan_scores=plan_scores)
        return self._compose_task_score(
            task,
            plan_scores,
            diversity=diversity,
            solvability_correct=not task_unsolvable,
            imagination_excluded=False,
            note="scored_plans",
            boundary_record=boundary_record,
        )

    def score_plan(self, task: Dict[str, object], plan: Dict[str, object]) -> Dict[str, object]:
        tool_index = self._tool_index(task)
        all_text = self._plan_text(plan)
        declared_tools, undeclared_tools = self._canonical_tools(plan, tool_index)
        implicit_unavailable = self._implicit_unavailable_tools(task, all_text, tool_index)
        distractor_tools = self._distractor_tool_ids(task)
        distractor_hits = sorted(declared_tools & distractor_tools)
        distractor_violation = clip01(len(distractor_hits) / max(1, len(declared_tools))) if declared_tools else 0.0

        tool_mentions = len(declared_tools) + len(undeclared_tools)
        if tool_mentions == 0:
            tool_legality = 0.0
        else:
            tool_legality = clip01(1.0 - ((len(undeclared_tools) + len(implicit_unavailable)) / max(1, tool_mentions + len(implicit_unavailable))))

        step_records = []
        for step in plan.get("steps") or []:
            step_records.append(self._score_step(task, step, declared_tools, tool_index))
        physical_support = mean_or_none([record["support"] for record in step_records]) or 0.0

        goal_record = self._goal_achievement(task, plan, step_records)
        goal_achievement = goal_record["score"]
        constraint_record = self._constraint_satisfaction(task, all_text)
        constraint_satisfaction = constraint_record["score"]
        mechanism_completeness = self._mechanism_completeness(plan, step_records)
        contradiction = clip01(0.60 * (mean_or_none([record["contradiction"] for record in step_records]) or 0.0) + 0.40 * constraint_record["violation_rate"])
        unsupported_success = self._unsupported_success_assertion(plan, goal_achievement, physical_support)
        novelty_record = self._plan_novelty(task, plan, declared_tools, step_records)
        novelty = novelty_record["score"]
        rarity_record = self._plan_rarity(task, plan, declared_tools, step_records, novelty)
        rarity = rarity_record["score"]
        mechanism_specificity = self._mechanism_specificity(plan, step_records)
        implementation_depth = self._implementation_depth(plan, step_records)
        mechanism_chain_depth = self._mechanism_chain_depth(plan, step_records)
        constraint_juggling_score = self._constraint_juggling_score(
            task,
            plan,
            tool_legality=tool_legality,
            constraint_satisfaction=constraint_satisfaction,
        )

        feasibility = clip01(
            0.25 * tool_legality +
            0.25 * physical_support +
            0.30 * goal_achievement +
            0.20 * constraint_satisfaction
        )
        feasibility_gmean = self._feasibility_gmean(
            tool_legality,
            physical_support,
            goal_achievement,
            constraint_satisfaction,
        )
        hard_gate = self._hard_gate(
            tool_legality=tool_legality,
            physical_support=physical_support,
            goal_achievement=goal_achievement,
            constraint_satisfaction=constraint_satisfaction,
            contradiction=contradiction,
            distractor_violation=distractor_violation,
        )
        discriminative_multiplier = clip01(
            0.55 + 0.25 * mechanism_specificity + 0.20 * implementation_depth
        )
        imagination_raw = clip01(
            (rarity ** float(self.v3_params.get("rarity_gamma", 1.35))) *
            ((0.15 + 0.85 * feasibility_gmean) ** float(self.v3_params.get("feasibility_gamma", 1.60))) *
            hard_gate *
            discriminative_multiplier
        )
        hallucination_raw = clip01(
            0.25 * (1.0 - tool_legality) +
            0.20 * (1.0 - physical_support) +
            0.18 * (1.0 - constraint_satisfaction) +
            0.12 * contradiction +
            0.10 * unsupported_success +
            0.15 * distractor_violation
        )
        if tool_legality <= 0.0 or constraint_satisfaction <= 0.0 or contradiction >= 0.85:
            hallucination_raw = max(hallucination_raw, 0.70)

        return {
            "version": "macgyver_plan_v1",
            "plan_name": plan.get("plan_name"),
            "used_tools": sorted(declared_tools),
            "undeclared_tools": sorted(undeclared_tools),
            "implicit_unavailable_tools": sorted(implicit_unavailable),
            "distractor_tools": distractor_hits,
            "D_distractor_tool_violation": round(distractor_violation, 4),
            "T_tool_legality": round(tool_legality, 4),
            "P_physical_affordance_support": round(physical_support, 4),
            "G_goal_achievement": round(goal_achievement, 4),
            "K_constraint_satisfaction": round(constraint_satisfaction, 4),
            "M_mechanism_completeness": round(mechanism_completeness, 4),
            "U_unavailable_entity_rate": round(1.0 - tool_legality, 4),
            "C_contradiction_rate": round(contradiction, 4),
            "A_unsupported_success_assertion": round(unsupported_success, 4),
            "N_plan_novelty": round(novelty, 4),
            "R_common_bank_rarity": round(rarity, 4),
            "F_feasibility": round(feasibility, 4),
            "F_feasibility_gmean": round(feasibility_gmean, 4),
            "G_hard_gate": round(hard_gate, 4),
            "M_mechanism_specificity": round(mechanism_specificity, 4),
            "D_implementation_depth": round(implementation_depth, 4),
            "C_mechanism_chain_depth": round(mechanism_chain_depth, 4),
            "J_constraint_juggling_score": round(constraint_juggling_score, 4),
            "hard_valid": bool(hard_gate >= 0.999 and tool_legality >= 0.95 and physical_support >= 0.60 and goal_achievement >= 0.60 and constraint_satisfaction >= 0.95),
            "imagination_raw": round(imagination_raw, 4),
            "hallucination_raw": round(hallucination_raw, 4),
            "step_records": step_records,
            "goal_record": goal_record,
            "constraint_record": constraint_record,
            "novelty_record": novelty_record,
            "rarity_record": rarity_record,
            "formula": {
                "feasibility": "F_i=0.25*T_i+0.25*P_i+0.30*G_i+0.20*K_i",
                "feasibility_gmean": "Fv3_i=geometric_mean(T_i,P_i,G_i,K_i)",
                "imagination": "I_i=rarity^1.35 * (0.15+0.85*Fv3_i)^1.60 * soft_gate * (0.55+0.25*mechanism_specificity+0.20*implementation_depth)",
                "hallucination": "H_i=0.25*U_i+0.20*(1-P_i)+0.18*(1-K_i)+0.12*C_i+0.10*A_i+0.15*D_i",
            },
        }

    def _compose_task_score(
        self,
        task: Dict[str, object],
        plan_scores: Sequence[Dict[str, object]],
        *,
        diversity: float,
        solvability_correct: bool,
        imagination_excluded: bool,
        note: str,
        imagination_raw_override: Optional[float] = None,
        hallucination_raw_override: Optional[float] = None,
        boundary_record: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        plan_i = [float(item.get("imagination_raw", 0.0) or 0.0) for item in plan_scores]
        plan_h = [float(item.get("hallucination_raw", 0.0) or 0.0) for item in plan_scores]
        quality_mass = self._top_mean(plan_i, int(self.v3_params.get("top_quality_n", 3)))
        elite_tail = self._top_mean(plan_i, int(self.v3_params.get("elite_tail_n", 2)))
        strategy_diversity_eff = self._strategy_diversity_eff(plan_scores)
        hard_valid_ratio = mean_or_none([1.0 if item.get("hard_valid") else 0.0 for item in plan_scores]) or 0.0
        if boundary_record and boundary_record.get("correct_boundary_response"):
            hard_valid_ratio = max(hard_valid_ratio, clip01(boundary_record.get("correct_boundary_response")))
        mechanism_chain_depth = self._top_mean(
            [float(item.get("C_mechanism_chain_depth", 0.0) or 0.0) for item in plan_scores],
            3,
        )
        constraint_juggling_score = self._top_mean(
            [float(item.get("J_constraint_juggling_score", 0.0) or 0.0) for item in plan_scores],
            3,
        )
        bank_coverage = mean_or_none([
            1.0 if (item.get("rarity_record") or {}).get("bank_available") else 0.0
            for item in plan_scores
        ]) or 0.0
        task_weights = self.v3_params.get("task_weights") if isinstance(self.v3_params.get("task_weights"), dict) else {}
        imagination_raw = (
            float(imagination_raw_override)
            if imagination_raw_override is not None else
            clip01(
                float(task_weights.get("quality_mass", 0.35)) * quality_mass +
                float(task_weights.get("elite_tail", 0.25)) * elite_tail +
                float(task_weights.get("mechanism_chain_depth", 0.20)) * mechanism_chain_depth +
                float(task_weights.get("constraint_juggling_score", 0.10)) * constraint_juggling_score +
                float(task_weights.get("strategy_diversity_eff", 0.05)) * strategy_diversity_eff +
                float(task_weights.get("hard_valid_ratio", 0.05)) * hard_valid_ratio
            )
        )
        hallucination_raw = (
            float(hallucination_raw_override)
            if hallucination_raw_override is not None else
            self._mean_first_n(plan_h, fill=0.0)
        )
        imagination = self._robust_norm(max(0.0, imagination_raw - self.beta_ih * hallucination_raw), task, "imagination")
        hallucination = self._robust_norm(max(0.0, hallucination_raw - self.beta_hi * imagination_raw), task, "hallucination")
        if imagination_excluded:
            imagination = None

        primitive_means = self._primitive_means(plan_scores)
        boundary_record = boundary_record or self.boundary_scorer.score(task, {"solvability": "solvable", "plans": []}, plan_scores=plan_scores)
        result = {
            "version": MACGYVER_DUAL_AXIS_VERSION,
            "task_id": task.get("id"),
            "task_subtype": task.get("task_subtype") or ("MacGyverUnsolvable" if task.get("unsolvable") else "MacGyverSolvable"),
            "expected_response_mode": expected_response_mode(task),
            "score": round(imagination, 4) if imagination is not None else None,
            "imagination": round(imagination, 4) if imagination is not None else None,
            "hallucination": round(hallucination, 4),
            "imagination_raw": round(imagination_raw, 4),
            "hallucination_raw": round(hallucination_raw, 4),
            "diversity": round(clip01(diversity), 4),
            "quality_mass_top3": round(quality_mass, 4),
            "elite_tail": round(elite_tail, 4),
            "strategy_diversity_eff": round(strategy_diversity_eff, 4),
            "mechanism_chain_depth": round(mechanism_chain_depth, 4),
            "constraint_juggling_score": round(constraint_juggling_score, 4),
            "hard_valid_ratio": round(hard_valid_ratio, 4),
            "common_bank_coverage": round(bank_coverage, 4),
            "plan_scores": list(plan_scores),
            "primitive_means": primitive_means,
            "boundary_record": boundary_record,
            "scored_plans": len(plan_scores),
            "expected_plan_count": self.expected_plan_count,
            "solvability_correct": bool(solvability_correct),
            "imagination_excluded": bool(imagination_excluded),
            "note": note,
            "formula": {
                "task_imagination_raw": "I_raw_t=0.35*top3_quality_mass+0.25*elite_tail+0.20*mechanism_chain_depth+0.10*constraint_juggling_score+0.05*strategy_diversity_eff+0.05*hard_valid_ratio",
                "task_hallucination_raw": "H_raw_t=mean_firstN(H_i)",
                "residual": "I=norm(max(0,I_raw-beta_IH*H_raw)); H=norm(max(0,H_raw-beta_HI*I_raw))",
            },
            "residualization": {
                "beta_IH": self.beta_ih,
                "beta_HI": self.beta_hi,
                "normalization": "task_reference_q05_q95_if_present_else_clip01",
            },
            "calibration_policy": MACGYVER_V3_CALIBRATION_POLICY,
            "runtime_scoring_policy": MACGYVER_V3_RUNTIME_SCORING_POLICY,
        }
        result["subtype_contributions"] = build_macgyver_task_subtype_contributions(
            result,
            beta_ih=self.beta_ih,
            beta_hi=self.beta_hi,
        )
        result["atom_signals"] = result["subtype_contributions"].get("atom_signals", {})
        return result

    def _empty_task_score(
        self,
        task: Dict[str, object],
        *,
        solvability_correct: bool,
        imagination_excluded: bool,
        hallucination_raw: float,
        note: str,
        boundary_record: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        return self._compose_task_score(
            task,
            [],
            diversity=0.0,
            imagination_raw_override=0.0,
            hallucination_raw_override=hallucination_raw,
            solvability_correct=solvability_correct,
            imagination_excluded=imagination_excluded,
            note=note,
            boundary_record=boundary_record,
        )

    
    
    

    def _tool_index(self, task: Dict[str, object]) -> Dict[str, str]:
        index = {}
        for tool in task.get("tools") or []:
            tool_id = str(tool.get("id") or "").strip()
            if not tool_id:
                continue
            aliases = [tool_id, tool.get("name"), *list(tool.get("aliases") or [])]
            for alias in aliases:
                alias_norm = _normalize_text(str(alias or ""))
                if alias_norm:
                    index[alias_norm] = tool_id
        return index

    def _canonical_tools(self, plan: Dict[str, object], tool_index: Dict[str, str]) -> Tuple[set, set]:
        declared = []
        declared.extend(plan.get("used_tools") or [])
        for step in plan.get("steps") or []:
            declared.extend(step.get("tools") or [])
        canonical = set()
        unknown = set()
        for raw in declared:
            raw_norm = _normalize_text(str(raw))
            if not raw_norm:
                continue
            if raw_norm in tool_index:
                canonical.add(tool_index[raw_norm])
            else:
                unknown.add(raw_norm)
        return canonical, unknown

    def _implicit_unavailable_tools(self, task: Dict[str, object], text: str, tool_index: Dict[str, str]) -> set:
        allowed_aliases = set(tool_index.keys())
        markers = set(COMMON_UNAVAILABLE_TOOL_MARKERS)
        for constraint in task.get("constraints") or []:
            markers.update(str(item) for item in constraint.get("forbidden_tool_keywords") or [])
        hits = set()
        for marker in markers:
            marker_norm = _normalize_text(marker)
            if marker_norm and marker_norm not in allowed_aliases and _keyword_hits(text, [marker_norm]):
                hits.add(marker_norm)
        return hits

    def _distractor_tool_ids(self, task: Dict[str, object]) -> set:
        explicit = {str(item) for item in task.get("distractor_tool_ids") or [] if str(item)}
        tagged = {
            str(tool.get("id"))
            for tool in task.get("tools") or []
            if str(tool.get("role") or "").strip().lower() == "distractor" and str(tool.get("id") or "")
        }
        return explicit | tagged

    def _score_step(self, task: Dict[str, object], step: Dict[str, object], declared_tools: set, tool_index: Dict[str, str]) -> Dict[str, object]:
        step_text = " ".join([
            _clean_string(step.get("action")),
            _clean_string(step.get("mechanism")),
            _clean_string(step.get("target_effect")),
        ])
        step_tools, step_unknown = self._canonical_tools({"used_tools": step.get("tools") or [], "steps": []}, tool_index)
        if not step_tools:
            step_tools = set(declared_tools)
        support_values = []
        support_hits = {}
        contradiction_values = []
        contradiction_hits = {}
        for tool in task.get("tools") or []:
            tool_id = str(tool.get("id") or "")
            if step_tools and tool_id not in step_tools:
                continue
            support, hits = self._tool_affordance_support(tool, step_text)
            contradiction, neg_hits = self._tool_negative_support(tool, step_text)
            support_values.append(support)
            contradiction_values.append(contradiction)
            if hits:
                support_hits[tool_id] = hits
            if neg_hits:
                contradiction_hits[tool_id] = neg_hits
        support = max(support_values) if support_values else 0.0
        contradiction = max(contradiction_values) if contradiction_values else 0.0
        if step_unknown:
            support *= 0.5
            contradiction = max(contradiction, 0.5)
        return {
            "action": _clean_string(step.get("action")),
            "target_effect": _clean_string(step.get("target_effect")),
            "tools": sorted(step_tools),
            "unknown_tools": sorted(step_unknown),
            "support": round(clip01(support), 4),
            "contradiction": round(clip01(contradiction), 4),
            "support_hits": support_hits,
            "contradiction_hits": contradiction_hits,
        }

    def _tool_affordance_support(self, tool: Dict[str, object], text: str) -> Tuple[float, List[str]]:
        best = 0.0
        hits = []
        affordances = tool.get("affordances") or {}
        if isinstance(affordances, dict):
            iterable = affordances.items()
        else:
            iterable = [(str(item), {"keywords": [str(item)], "strength": 0.75}) for item in affordances]
        for tag, spec in iterable:
            if isinstance(spec, dict):
                keywords = list(spec.get("keywords") or []) + [str(tag)]
                strength = float(spec.get("strength", 0.85) or 0.85)
            else:
                keywords = [str(tag), str(spec)]
                strength = 0.75
            current_hits = _keyword_hits(text, keywords)
            if current_hits:
                best = max(best, strength)
                hits.extend([str(tag), *current_hits])
        return clip01(best), sorted(set(hits))

    def _tool_negative_support(self, tool: Dict[str, object], text: str) -> Tuple[float, List[str]]:
        best = 0.0
        hits = []
        negatives = tool.get("negative_affordances") or {}
        if isinstance(negatives, dict):
            iterable = negatives.items()
        else:
            iterable = [(str(item), {"keywords": [str(item)], "strength": 0.75}) for item in negatives]
        for tag, spec in iterable:
            if isinstance(spec, dict):
                keywords = list(spec.get("keywords") or []) + [str(tag)]
                strength = float(spec.get("strength", 0.85) or 0.85)
            else:
                keywords = [str(tag), str(spec)]
                strength = 0.75
            current_hits = _keyword_hits(text, keywords)
            if current_hits:
                best = max(best, strength)
                hits.extend([str(tag), *current_hits])
        return clip01(best), sorted(set(hits))

    def _goal_achievement(self, task: Dict[str, object], plan: Dict[str, object], step_records: Sequence[Dict[str, object]]) -> Dict[str, object]:
        del step_records
        all_text = self._plan_text(plan)
        covered = []
        missing = []
        for goal in task.get("goal_predicates") or []:
            goal_id = str(goal.get("id") or "")
            keywords = list(goal.get("keywords") or []) + [goal_id]
            target_hit = any(_normalize_text(goal_id) == _normalize_text(step.get("target_effect") or "") for step in plan.get("steps") or [])
            keyword_hit = bool(_keyword_hits(all_text, keywords))
            if target_hit or keyword_hit:
                covered.append(goal_id)
            else:
                missing.append(goal_id)
        total = len(covered) + len(missing)
        return {
            "score": round((len(covered) / total) if total else 0.0, 4),
            "covered": covered,
            "missing": missing,
        }

    def _constraint_satisfaction(self, task: Dict[str, object], text: str) -> Dict[str, object]:
        constraints = list(task.get("constraints") or [])
        if not constraints:
            return {"score": 1.0, "violated": [], "violation_rate": 0.0}
        violated = []
        for constraint in constraints:
            keywords = list(constraint.get("forbidden_keywords") or []) + list(constraint.get("forbidden_tool_keywords") or [])
            hits = _keyword_hits(text, keywords)
            if hits:
                violated.append({"id": constraint.get("id"), "hits": hits})
        violation_rate = len(violated) / len(constraints)
        return {
            "score": round(clip01(1.0 - violation_rate), 4),
            "violated": violated,
            "violation_rate": round(clip01(violation_rate), 4),
        }

    def _mechanism_completeness(self, plan: Dict[str, object], step_records: Sequence[Dict[str, object]]) -> float:
        step_scores = []
        for step, record in zip(plan.get("steps") or [], step_records):
            action = bool(_clean_string(step.get("action")))
            tools = bool(record.get("tools") or record.get("unknown_tools"))
            mechanism = _clean_string(step.get("mechanism"))
            effect = bool(_clean_string(step.get("target_effect")))
            causal = bool(_keyword_hits(mechanism, CAUSAL_MARKERS)) or bool(record.get("support_hits"))
            step_scores.append(clip01(0.25 * action + 0.25 * tools + 0.25 * bool(mechanism) + 0.15 * effect + 0.10 * causal))
        return mean_or_none(step_scores) or 0.0

    def _unsupported_success_assertion(self, plan: Dict[str, object], goal: float, support: float) -> float:
        text = " ".join([_clean_string(plan.get("final_state")), _clean_string(plan.get("risk_note"))])
        if not _keyword_hits(text, SUCCESS_WORDS):
            return 0.0
        if goal >= 0.5 and support >= 0.4:
            return 0.0
        return clip01(1.0 - min(goal, support))

    def _plan_novelty(self, task: Dict[str, object], plan: Dict[str, object], tools: set, step_records: Sequence[Dict[str, object]]) -> Dict[str, object]:
        references = list(task.get("reference_plans") or [])
        plan_actions = self._plan_action_tokens(plan)
        mechanism_tags = self._mechanism_tags(step_records)
        if not references:
            return {
                "score": 0.5,
                "tool_set_distance": 0.5,
                "action_sequence_distance": 0.5,
                "mechanism_tag_distance": 0.5,
                "nearest_reference": None,
            }
        best_similarity = None
        best_record = None
        for ref in references:
            ref_tools = set(ref.get("used_tools") or [])
            ref_actions = set(_tokens(" ".join(str(item) for item in ref.get("action_sequence") or [])))
            ref_mechanisms = set(str(item) for item in ref.get("mechanism_tags") or [])
            tool_distance = _jaccard_distance(tools, ref_tools)
            action_distance = _jaccard_distance(plan_actions, ref_actions)
            mechanism_distance = _jaccard_distance(mechanism_tags, ref_mechanisms)
            similarity = 1.0 - (0.35 * tool_distance + 0.35 * action_distance + 0.30 * mechanism_distance)
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_record = {
                    "reference_id": ref.get("id"),
                    "tool_set_distance": tool_distance,
                    "action_sequence_distance": action_distance,
                    "mechanism_tag_distance": mechanism_distance,
                }
        score = clip01(
            0.35 * best_record["tool_set_distance"] +
            0.35 * best_record["action_sequence_distance"] +
            0.30 * best_record["mechanism_tag_distance"]
        )
        best_record["score"] = round(score, 4)
        return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in best_record.items()}

    def _task_bank_record(self, task: Dict[str, object]) -> Dict[str, object]:
        payload = self.common_plan_bank if isinstance(self.common_plan_bank, dict) else {}
        tasks = payload.get("tasks") if isinstance(payload.get("tasks"), dict) else {}
        record = tasks.get(str(task.get("id"))) if isinstance(tasks, dict) else None
        return record if isinstance(record, dict) else {}

    def _family_match_score(self, text: str, family: Dict[str, object], tools: set, mechanism_tags: set) -> Dict[str, object]:
        keywords = [str(item) for item in family.get("keywords") or [] if str(item)]
        keyword_hits = _keyword_hits(text, keywords)
        keyword_score = len(keyword_hits) / max(1, min(len(keywords), 4))
        family_tools = {str(item) for item in family.get("tool_ids") or [] if str(item)}
        if family_tools:
            tool_score = len(family_tools & tools) / len(family_tools)
        else:
            tool_score = 0.0
        family_mechanisms = {str(item) for item in family.get("mechanism_tags") or [] if str(item)}
        if family_mechanisms:
            mechanism_score = len(family_mechanisms & mechanism_tags) / len(family_mechanisms)
        else:
            mechanism_score = 0.0
        score = clip01(0.62 * clip01(keyword_score) + 0.23 * tool_score + 0.15 * mechanism_score)
        return {
            "id": family.get("id"),
            "score": round(score, 4),
            "keyword_hits": keyword_hits,
            "tool_overlap": sorted(family_tools & tools),
            "mechanism_overlap": sorted(family_mechanisms & mechanism_tags),
        }

    def _best_family_match(
        self,
        families: Sequence[Dict[str, object]],
        text: str,
        tools: set,
        mechanism_tags: set,
    ) -> Optional[Dict[str, object]]:
        matches = [
            self._family_match_score(text, family, tools, mechanism_tags)
            for family in families
            if isinstance(family, dict)
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: float(item.get("score") or 0.0))

    def _plan_rarity(
        self,
        task: Dict[str, object],
        plan: Dict[str, object],
        tools: set,
        step_records: Sequence[Dict[str, object]],
        fallback_novelty: float,
    ) -> Dict[str, object]:
        bank = self._task_bank_record(task)
        text = self._plan_text(plan)
        mechanism_tags = self._mechanism_tags(step_records)
        hard_match = self._best_family_match(
            bank.get("hard_zero_plan_families") or [],
            text,
            tools,
            mechanism_tags,
        )
        broad_match = self._best_family_match(
            bank.get("broad_common_plan_families") or [],
            text,
            tools,
            mechanism_tags,
        )
        rare_match = self._best_family_match(
            bank.get("supported_rare_affordance_families") or [],
            text,
            tools,
            mechanism_tags,
        )
        hard_threshold = float(self.v3_params.get("hard_zero_threshold", 0.42))
        broad_threshold = float(self.v3_params.get("broad_common_threshold", 0.38))
        broad_floor = float(self.v3_params.get("broad_common_floor", 0.25))
        rare_floor = float(self.v3_params.get("supported_rare_floor", 0.82))
        rarity = clip01(fallback_novelty)
        source = "reference_distance_fallback"
        if rare_match and float(rare_match.get("score") or 0.0) >= broad_threshold:
            rarity = max(rarity, rare_floor)
            source = "supported_rare_affordance_family"
        if broad_match and float(broad_match.get("score") or 0.0) >= broad_threshold:
            similarity = clip01(float(broad_match.get("score") or 0.0))
            rarity = min(rarity, broad_floor + (1.0 - similarity) * (1.0 - broad_floor))
            source = "broad_common_family_cap"
        if hard_match and float(hard_match.get("score") or 0.0) >= hard_threshold:
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

    def _feasibility_gmean(self, tool_legality: float, physical_support: float, goal_achievement: float, constraint_satisfaction: float) -> float:
        components = [
            (clip01(tool_legality), 0.25),
            (clip01(physical_support), 0.25),
            (clip01(goal_achievement), 0.30),
            (clip01(constraint_satisfaction), 0.20),
        ]
        product = 1.0
        for value, weight in components:
            product *= value ** weight
        return clip01(product)

    def _hard_gate(
        self,
        *,
        tool_legality: float,
        physical_support: float,
        goal_achievement: float,
        constraint_satisfaction: float,
        contradiction: float,
        distractor_violation: float,
    ) -> float:
        if tool_legality <= 0.0 or constraint_satisfaction <= 0.0 or contradiction >= 0.85:
            return 0.0
        gate = clip01(0.30 + 0.875 * clip01(tool_legality))
        gate *= clip01(1.0 - 0.70 * clip01(distractor_violation))
        if physical_support < 0.40 or goal_achievement < 0.40:
            gate = min(gate, 0.35)
        if constraint_satisfaction < 0.80:
            gate *= clip01(0.30 + 0.70 * (clip01(constraint_satisfaction) / 0.80))
        return clip01(gate)

    def _mechanism_specificity(self, plan: Dict[str, object], step_records: Sequence[Dict[str, object]]) -> float:
        scores = []
        for step, record in zip(plan.get("steps") or [], step_records):
            mechanism = _clean_string(step.get("mechanism"))
            mechanism_tokens = _tokens(mechanism)
            support_hits = []
            for hits in (record.get("support_hits") or {}).values():
                support_hits.extend(hits)
            has_affordance = bool(support_hits)
            has_causal_marker = bool(_keyword_hits(mechanism, CAUSAL_MARKERS))
            effect_tokens = set(_tokens(_clean_string(step.get("target_effect"))))
            mechanism_effect_overlap = bool(effect_tokens & set(mechanism_tokens))
            specific_length = 1.0 if len(mechanism_tokens) >= 6 else len(mechanism_tokens) / 6.0
            score = (
                0.45 * has_affordance +
                0.20 * has_causal_marker +
                0.20 * specific_length +
                0.15 * mechanism_effect_overlap
            )
            if not has_affordance:
                score = min(score, 0.35)
            scores.append(clip01(score))
        return mean_or_none(scores) or 0.0

    def _implementation_depth(self, plan: Dict[str, object], step_records: Sequence[Dict[str, object]]) -> float:
        nonempty_steps = [step for step in plan.get("steps") or [] if _clean_string(step.get("action"))]
        step_score = min(len(nonempty_steps), 3) / 3.0
        field_score = (
            0.25 * bool(_clean_string(plan.get("core_trick"))) +
            0.25 * bool(plan.get("tool_chain")) +
            0.25 * bool(_clean_string(plan.get("failure_mode"))) +
            0.25 * bool(_clean_string(plan.get("why_distinct")))
        )
        mechanism_tag_score = min(len(self._mechanism_tags(step_records)), 3) / 3.0
        return clip01(0.45 * step_score + 0.35 * field_score + 0.20 * mechanism_tag_score)

    def _mechanism_chain_depth(self, plan: Dict[str, object], step_records: Sequence[Dict[str, object]]) -> float:
        steps = [step for step in plan.get("steps") or [] if _clean_string(step.get("action"))]
        if not steps:
            return 0.0
        mechanized_steps = 0
        causal_steps = 0
        effect_steps = 0
        continuity_hits = 0
        previous_effect_tokens = set()
        for step, record in zip(steps, step_records):
            mechanism_text = _clean_string(step.get("mechanism"))
            action_text = _clean_string(step.get("action"))
            effect_text = _clean_string(step.get("target_effect"))
            support_hits = []
            for hits_by_tool in (record.get("support_hits") or {}).values():
                support_hits.extend(hits_by_tool)
            has_mechanism = bool(mechanism_text) and bool(support_hits)
            mechanized_steps += 1 if has_mechanism else 0
            causal_steps += 1 if _keyword_hits(" ".join([mechanism_text, action_text]), CAUSAL_MARKERS) else 0
            current_effect_tokens = set(_tokens(effect_text))
            effect_steps += 1 if current_effect_tokens else 0
            current_text_tokens = set(_tokens(" ".join([action_text, mechanism_text, effect_text])))
            if previous_effect_tokens and previous_effect_tokens & current_text_tokens:
                continuity_hits += 1
            previous_effect_tokens = current_effect_tokens or previous_effect_tokens
        depth_score = clip01(len(steps) / 5.0)
        mechanism_score = mechanized_steps / max(1, len(steps))
        causal_score = causal_steps / max(1, len(steps))
        effect_score = effect_steps / max(1, len(steps))
        continuity_score = continuity_hits / max(1, len(steps) - 1)
        return clip01(
            0.25 * depth_score +
            0.30 * mechanism_score +
            0.20 * causal_score +
            0.15 * effect_score +
            0.10 * continuity_score
        )

    def _constraint_juggling_score(
        self,
        task: Dict[str, object],
        plan: Dict[str, object],
        *,
        tool_legality: float,
        constraint_satisfaction: float,
    ) -> float:
        constraints = [item for item in task.get("constraints") or [] if isinstance(item, dict)]
        if not constraints:
            return clip01(0.50 * tool_legality + 0.50 * constraint_satisfaction)
        text = _normalize_text(self._plan_text(plan))
        respected = 0
        relevant = 0
        for constraint in constraints:
            forbidden_terms = list(constraint.get("forbidden_keywords") or [])
            forbidden_terms.extend(constraint.get("forbidden_tool_keywords") or [])
            if not forbidden_terms:
                continue
            relevant += 1
            if not _keyword_hits(text, forbidden_terms):
                respected += 1
        if relevant == 0:
            respect_score = constraint_satisfaction
        else:
            respect_score = respected / max(1, relevant)
        multi_constraint_bonus = clip01(relevant / 2.0)
        return clip01(
            0.45 * clip01(tool_legality) +
            0.35 * clip01(constraint_satisfaction) +
            0.20 * clip01(respect_score * multi_constraint_bonus)
        )

    def _plan_strategy_signature(self, plan_score: Dict[str, object]) -> set:
        tags = set()
        tags.update(str(tool) for tool in plan_score.get("used_tools") or [])
        for record in plan_score.get("step_records") or []:
            for hits_by_tool in (record.get("support_hits") or {}).values():
                tags.update(str(hit) for hit in hits_by_tool)
        rarity_record = plan_score.get("rarity_record") or {}
        for key in ("hard_zero_match", "broad_common_match", "supported_rare_match"):
            match = rarity_record.get(key)
            if isinstance(match, dict) and match.get("id"):
                tags.add(str(match.get("id")))
        return {tag for tag in tags if tag}

    def _strategy_diversity_eff(self, plan_scores: Sequence[Dict[str, object]]) -> float:
        if len(plan_scores) < 2:
            return 0.0
        distances = []
        for i in range(len(plan_scores)):
            for j in range(i + 1, len(plan_scores)):
                distances.append(_jaccard_distance(
                    self._plan_strategy_signature(plan_scores[i]),
                    self._plan_strategy_signature(plan_scores[j]),
                ))
        diversity = mean_or_none(distances) or 0.0
        hard_valid_ratio = mean_or_none([1.0 if score.get("hard_valid") else 0.0 for score in plan_scores]) or 0.0
        return clip01(diversity * (0.50 + 0.50 * hard_valid_ratio))

    def _plan_diversity(self, plan_scores: Sequence[Dict[str, object]]) -> float:
        if len(plan_scores) < 2:
            return 0.0
        distances = []
        for i in range(len(plan_scores)):
            for j in range(i + 1, len(plan_scores)):
                a = plan_scores[i].get("novelty_record", {})
                b = plan_scores[j].get("novelty_record", {})
                vec_a = {
                    str(plan_scores[i].get("used_tools")),
                    str(round(float(a.get("tool_set_distance", 0.0)), 1)),
                    str(round(float(a.get("action_sequence_distance", 0.0)), 1)),
                    str(round(float(a.get("mechanism_tag_distance", 0.0)), 1)),
                }
                vec_b = {
                    str(plan_scores[j].get("used_tools")),
                    str(round(float(b.get("tool_set_distance", 0.0)), 1)),
                    str(round(float(b.get("action_sequence_distance", 0.0)), 1)),
                    str(round(float(b.get("mechanism_tag_distance", 0.0)), 1)),
                }
                distances.append(_jaccard_distance(vec_a, vec_b))
        return mean_or_none(distances) or 0.0

    def _impossibility_reason_hallucination(self, task: Dict[str, object], reason: str) -> float:
        reason_norm = _normalize_text(reason)
        if not reason_norm:
            return 0.05
        task_text = _normalize_text(" ".join([
            task.get("scene", ""),
            task.get("goal", ""),
            " ".join(str(constraint.get("description", "")) for constraint in task.get("constraints") or []),
        ]))
        reason_tokens = set(_tokens(reason_norm))
        task_tokens = set(_tokens(task_text))
        unsupported = len(reason_tokens - task_tokens) / max(1, len(reason_tokens))
        marker_hit = bool(_keyword_hits(reason_norm, COMMON_UNAVAILABLE_TOOL_MARKERS))
        return clip01(0.05 + 0.35 * unsupported + (0.25 if marker_hit else 0.0))

    def _plan_text(self, plan: Dict[str, object]) -> str:
        pieces = [
            plan.get("plan_name"),
            plan.get("core_trick"),
            plan.get("final_state"),
            plan.get("failure_mode"),
            plan.get("why_distinct"),
            plan.get("risk_note"),
        ]
        for step in plan.get("steps") or []:
            pieces.extend([step.get("action"), step.get("mechanism"), step.get("target_effect"), " ".join(step.get("tools") or [])])
        pieces.extend(plan.get("used_tools") or [])
        pieces.extend(plan.get("tool_chain") or [])
        return " ".join(_clean_string(piece) for piece in pieces if _clean_string(piece))

    def _plan_action_tokens(self, plan: Dict[str, object]) -> set:
        text = " ".join(_clean_string(step.get("action")) for step in plan.get("steps") or [])
        text += " " + " ".join(_clean_string(step.get("target_effect")) for step in plan.get("steps") or [])
        return set(_tokens(text))

    def _mechanism_tags(self, step_records: Sequence[Dict[str, object]]) -> set:
        tags = set()
        for record in step_records:
            for hits_by_tool in (record.get("support_hits") or {}).values():
                for hit in hits_by_tool:
                    tags.add(_normalize_text(hit))
        return {tag for tag in tags if tag}

    def _dedupe_plans(self, plans: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        deduped = []
        seen = set()
        for plan in plans:
            key = _normalize_text(" ".join([
                str(plan.get("plan_name") or ""),
                " ".join(str(tool) for tool in plan.get("used_tools") or []),
                " ".join(str(step.get("action") or "") for step in plan.get("steps") or []),
            ]))
            if key and key in seen:
                continue
            seen.add(key)
            deduped.append(plan)
        return deduped

    def _mean_first_n(self, values: Sequence[float], *, fill: float) -> float:
        padded = [float(value) for value in values[:self.expected_plan_count]]
        while len(padded) < self.expected_plan_count:
            padded.append(float(fill))
        return sum(padded) / max(1, self.expected_plan_count)

    def _top_mean(self, values: Sequence[float], n: int) -> float:
        if not values or n <= 0:
            return 0.0
        ranked = sorted((float(value) for value in values), reverse=True)
        return sum(ranked[:n]) / min(n, len(ranked))

    def _robust_norm(self, value: float, task: Dict[str, object], metric: str) -> float:
        scales = task.get("score_reference") or {}
        metric_scale = scales.get(metric) if isinstance(scales, dict) else None
        if isinstance(metric_scale, dict) and metric_scale.get("q95") is not None and metric_scale.get("q05") is not None:
            q05 = float(metric_scale.get("q05"))
            q95 = float(metric_scale.get("q95"))
            if q95 > q05:
                return clip01((float(value) - q05) / (q95 - q05))
        return clip01(value)

    def _primitive_means(self, plan_scores: Sequence[Dict[str, object]]) -> Dict[str, float]:
        fields = [
            "T_tool_legality",
            "P_physical_affordance_support",
            "G_goal_achievement",
            "K_constraint_satisfaction",
            "M_mechanism_completeness",
            "U_unavailable_entity_rate",
            "C_contradiction_rate",
            "A_unsupported_success_assertion",
            "N_plan_novelty",
            "R_common_bank_rarity",
            "F_feasibility",
            "F_feasibility_gmean",
            "G_hard_gate",
            "M_mechanism_specificity",
            "D_implementation_depth",
            "C_mechanism_chain_depth",
            "J_constraint_juggling_score",
            "D_distractor_tool_violation",
            "imagination_raw",
            "hallucination_raw",
        ]
        result = {}
        for field in fields:
            value = mean_or_none([score.get(field) for score in plan_scores if score.get(field) is not None])
            if value is not None:
                result[field] = round(value, 4)
        return result


def aggregate_macgyver_model_axes(
    task_scores: Sequence[Dict[str, object]],
    *,
    gate_pass: bool,
    beta_ih: float = DEFAULT_MACGYVER_BETA_IH,
    beta_hi: float = DEFAULT_MACGYVER_BETA_HI,
) -> Dict[str, object]:
    imagination_scores = [
        score.get("imagination_raw")
        for score in task_scores
        if not score.get("imagination_excluded") and score.get("imagination_raw") is not None
    ]
    hallucination_scores = [
        score.get("hallucination_raw")
        for score in task_scores
        if score.get("hallucination_raw") is not None
    ]
    raw_i = mean_or_none(imagination_scores)
    raw_h = mean_or_none(hallucination_scores)
    imagination = clip01((raw_i or 0.0) - beta_ih * max(0.0, raw_h or 0.0)) if gate_pass and raw_i is not None else None
    hallucination = clip01((raw_h or 0.0) - beta_hi * max(0.0, raw_i or 0.0)) if gate_pass and raw_h is not None else None

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
    boundary_record_means = {}
    boundary_fields = {
        "false_acceptance",
        "false_refusal",
        "unsupported_refusal_reason",
        "clarification_miss",
        "distractor_tool_violation",
        "correct_boundary_response",
        "boundary_hallucination",
    }
    for field in sorted(boundary_fields):
        value = mean_or_none([
            score.get("boundary_record", {}).get(field)
            for score in task_scores
            if isinstance(score.get("boundary_record"), dict)
        ])
        if value is not None:
            boundary_record_means[field] = round(value, 4)
    subtype_contributions = mean_subtype_contributions(
        score.get("subtype_contributions")
        for score in task_scores
        if isinstance(score.get("subtype_contributions"), dict)
    )
    diagnostic_fields = {
        "quality_mass_top3",
        "elite_tail",
        "strategy_diversity_eff",
        "mechanism_chain_depth",
        "constraint_juggling_score",
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
        "version": MACGYVER_DUAL_AXIS_VERSION,
        "calibration_policy": MACGYVER_V3_CALIBRATION_POLICY,
        "runtime_scoring_policy": MACGYVER_V3_RUNTIME_SCORING_POLICY,
        "score": round(imagination, 4) if imagination is not None else None,
        "imagination": round(imagination, 4) if imagination is not None else None,
        "hallucination": round(hallucination, 4) if hallucination is not None else None,
        "imagination_raw": round(raw_i, 4) if raw_i is not None else None,
        "hallucination_raw": round(raw_h, 4) if raw_h is not None else None,
        "coverage_gate_pass": bool(gate_pass),
        "task_scores": list(task_scores),
        "primitive_means": primitive_means,
        "boundary_record_means": boundary_record_means,
        "subtype_contributions": subtype_contributions,
        **diagnostic_means,
        "scored_tasks": len(task_scores),
        "solvability_accuracy": (
            round(sum(1 for score in task_scores if score.get("solvability_correct")) / len(task_scores), 4)
            if task_scores else None
        ),
        "formula": {
            "plan_feasibility": "F_i=0.25*T_i+0.25*P_i+0.30*G_i+0.20*K_i",
            "plan_imagination": "I_i=rarity^1.35 * (0.15+0.85*feasibility_gmean)^1.60 * soft_gate * (0.55+0.25*mechanism_specificity+0.20*implementation_depth)",
            "plan_hallucination": "H_i=0.25*U_i+0.20*(1-P_i)+0.18*(1-K_i)+0.12*C_i+0.10*A_i+0.15*D_i",
            "task_imagination_raw": "I_raw_t=0.35*top3_quality_mass+0.25*elite_tail+0.20*mechanism_chain_depth+0.10*constraint_juggling_score+0.05*strategy_diversity_eff+0.05*hard_valid_ratio",
            "task_hallucination_raw": "H_raw_t=mean_firstN(H_i)",
            "model_residual": "I=clip(mean(I_raw)-beta_IH*mean(H_raw)); H=clip(mean(H_raw)-beta_HI*mean(I_raw))",
        },
        "residualization": {
            "beta_IH": beta_ih,
            "beta_HI": beta_hi,
            "source": "task_default",
        },
    }


def aggregate_macgyver_boundary_diagnostics(
    task_scores: Sequence[Dict[str, object]],
    *,
    beta_hi: float = DEFAULT_MACGYVER_BETA_HI,
) -> Dict[str, object]:
    hallucination_scores = [
        score.get("hallucination_raw")
        for score in task_scores
        if score.get("hallucination_raw") is not None
    ]
    raw_h = mean_or_none(hallucination_scores)
    boundary_record_means = {}
    boundary_fields = {
        "false_acceptance",
        "false_refusal",
        "unsupported_refusal_reason",
        "clarification_miss",
        "distractor_tool_violation",
        "correct_boundary_response",
        "boundary_hallucination",
    }
    for field in sorted(boundary_fields):
        value = mean_or_none([
            score.get("boundary_record", {}).get(field)
            for score in task_scores
            if isinstance(score.get("boundary_record"), dict)
        ])
        if value is not None:
            boundary_record_means[field] = round(value, 4)
    subtype_contributions = mean_subtype_contributions(
        score.get("subtype_contributions")
        for score in task_scores
        if isinstance(score.get("subtype_contributions"), dict)
    )
    return {
        "version": MACGYVER_DUAL_AXIS_VERSION,
        "calibration_policy": "not_strength_calibrated",
        "score": round(raw_h, 4) if raw_h is not None else None,
        "hallucination": round(raw_h, 4) if raw_h is not None else None,
        "hallucination_raw": round(raw_h, 4) if raw_h is not None else None,
        "imagination": None,
        "imagination_raw": None,
        "task_scores": list(task_scores),
        "boundary_record_means": boundary_record_means,
        "subtype_contributions": subtype_contributions,
        "scored_tasks": len(task_scores),
        "solvability_accuracy": (
            round(sum(1 for score in task_scores if score.get("solvability_correct")) / len(task_scores), 4)
            if task_scores else None
        ),
        "residualization": {
            "beta_HI": beta_hi,
            "source": "boundary_diagnostic_only_no_strength_calibration",
        },
    }


__all__ = [
    "MACGYVER_DUAL_AXIS_VERSION",
    "MACGYVER_V3_CALIBRATION_POLICY",
    "MACGYVER_V3_RUNTIME_SCORING_POLICY",
    "DEFAULT_MACGYVER_BETA_IH",
    "DEFAULT_MACGYVER_BETA_HI",
    "DEFAULT_PLAN_COUNT",
    "load_macgyver_common_plan_bank",
    "load_macgyver_v3_calibration_params",
    "get_macgyver_common_plan_bank_coverage",
    "MacGyverScorer",
    "aggregate_macgyver_model_axes",
    "aggregate_macgyver_boundary_diagnostics",
    "clip01",
    "mean_or_none",
]
