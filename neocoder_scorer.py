
from __future__ import annotations

import ast
import copy
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

from scorer_hyperparameters import get_scorer_hyperparameter
from typed_axis_aggregation import (
    build_neocoder_task_subtype_contributions,
    mean_subtype_contributions,
)


NEOCODER_VERSION = "neocoder_dual_axis"
DEFAULT_NEOCODER_BETA_IH = 0.25
DEFAULT_NEOCODER_BETA_HI = 0.10
NEOCODER_V3_CALIBRATION_POLICY = "benchmark_default"
NEOCODER_V3_RUNTIME_SCORING_POLICY = "fixed output-only parameters"
NEOCODER_V3_TEST_VISIBILITY_POLICY = "public_examples_hidden_scoring_tests"

DATA_DIR = Path(__file__).resolve().parent / "data"
NEOCODER_V3_TASK_OVERLAY_VERSION = "neocoder_task_overlay"
NEOCODER_V3_TECHNIQUE_ALIAS_VERSION = "neocoder_technique_aliases"
NEOCODER_V3_COMMON_SOLUTION_BANK_VERSION = "neocoder_common_solution_bank"

DEFAULT_NEOCODER_V3_PARAMS = {
    "rarity_gamma": 1.35,
    "functional_gamma": 1.45,
    "constraint_gamma": 1.25,
    "quality_multiplier_weights": {
        "base": 0.35,
        "implementation_depth": 0.20,
        "denial_adaptation": 0.15,
        "algorithmic_pattern_diversity": 0.15,
        "denial_adaptation_quality": 0.10,
        "ledger_consistency": 0.05,
    },
    "functional_quality_weights": {
        "public_pass": 0.30,
        "hidden_pass": 0.45,
        "metamorphic_pass": 0.25,
    },
    "rarity": {
        "hard_zero": 0.0,
        "broad_common_cap": 0.35,
        "supported_rare_floor": 0.72,
        "default_floor": 0.42,
    },
}

_DANGEROUS_IMPORTS_FALLBACK = [
    "builtins",
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "threading",
    "urllib",
]
DANGEROUS_IMPORTS = set(get_scorer_hyperparameter(
    "neocoder",
    "DANGEROUS_IMPORTS",
    default=_DANGEROUS_IMPORTS_FALLBACK,
))

_DANGEROUS_CALLS_FALLBACK = [
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "globals",
    "input",
    "locals",
    "open",
    "quit",
    "vars",
]
DANGEROUS_CALLS = set(get_scorer_hyperparameter(
    "neocoder",
    "DANGEROUS_CALLS",
    default=_DANGEROUS_CALLS_FALLBACK,
))

ALLOWED_BUILTIN_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
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


def load_neocoder_v3_task_overlay() -> Dict[str, object]:
    return _json_load(DATA_DIR / "neocoder_v3_task_overlay.json")


def load_neocoder_technique_aliases() -> Dict[str, object]:
    return _json_load(DATA_DIR / "neocoder_technique_aliases_v3.json")


def load_neocoder_common_solution_bank() -> Dict[str, object]:
    return _json_load(DATA_DIR / "neocoder_common_solution_bank_v3.json")


def load_neocoder_v3_calibration_params() -> Dict[str, object]:
    payload = _json_load(DATA_DIR / "neocoder_scoring_config.json")
    params = copy.deepcopy(DEFAULT_NEOCODER_V3_PARAMS)
    final_params = payload.get("final_params") if isinstance(payload, dict) else None
    if isinstance(final_params, dict):
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


def get_neocoder_task_overlay_coverage(task_ids: Sequence[object]) -> Dict[str, object]:
    return _coverage_for_task_ids(load_neocoder_v3_task_overlay(), task_ids)


def get_neocoder_common_solution_bank_coverage(task_ids: Sequence[object]) -> Dict[str, object]:
    return _coverage_for_task_ids(load_neocoder_common_solution_bank(), task_ids)


def get_neocoder_technique_alias_coverage() -> Dict[str, object]:
    payload = load_neocoder_technique_aliases()
    aliases = payload.get("denied_aliases") if isinstance(payload, Mapping) else None
    alias_count = len(aliases) if isinstance(aliases, Mapping) else 0
    return {
        "version": payload.get("version") if isinstance(payload, Mapping) else None,
        "denied_alias_family_count": alias_count,
        "coverage": 1.0 if alias_count else 0.0,
    }


def _clean_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", str(value)).strip()


def _clean_code(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_list(value) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_label(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _normalize_complexity(value: object) -> str:
    return re.sub(r"[^a-z0-9*^]+", "", str(value or "").lower())


def _dedupe(values: Iterable[object]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = _clean_string(value)
        if not cleaned:
            continue
        key = _normalize_label(cleaned)
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


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


def _extract_python_code(raw_text: str) -> str:
    text = (raw_text or "").strip()
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return text if "def " in text else ""


class _TechniqueVisitor(ast.NodeVisitor):
    def __init__(self, *, function_name: str):
        self.function_name = function_name
        self.techniques: Set[str] = set()
        self.imports: List[str] = []
        self.import_aliases: Dict[str, str] = {}
        self.calls: List[str] = []
        self.undefined_calls: Set[str] = set()
        self.dangerous_calls: Set[str] = set()
        self.function_defs: Set[str] = set()
        self.loop_depth = 0
        self.assignment_count = 0
        self.branch_count = 0
        self.return_count = 0
        self.max_loop_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_defs.add(node.name)
        if node.name != self.function_name:
            self.techniques.add("helper_function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.function_defs.add(node.name)
        self.techniques.add("async_function")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            top = (alias.name or "").split(".")[0]
            self.imports.append(top)
            self.import_aliases[alias.asname or top] = top
            if top == "itertools":
                self.techniques.add("itertools_groupby")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        top = (node.module or "").split(".")[0]
        if top:
            self.imports.append(top)
            for alias in node.names:
                self.import_aliases[alias.asname or alias.name] = top
                if top == "itertools" and alias.name == "groupby":
                    self.techniques.add("itertools_groupby")
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.techniques.add("for_loop")
        if self.loop_depth > 0:
            self.techniques.add("nested_loop")
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self.techniques.add("for_loop")
        self.techniques.add("async_function")
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self.techniques.add("while_loop")
        if self.loop_depth > 0:
            self.techniques.add("nested_loop")
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_If(self, node: ast.If):
        self.techniques.add("if_statement")
        self.branch_count += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp):
        self.techniques.add("if_statement")
        self.branch_count += 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp):
        self.techniques.add("list_comprehension")
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp):
        self.techniques.add("list_comprehension")
        self.techniques.add("set_comprehension")
        self.techniques.add("set_usage")
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp):
        self.techniques.add("list_comprehension")
        self.techniques.add("dict_comprehension")
        self.techniques.add("dict_usage")
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        self.techniques.add("list_comprehension")
        self.techniques.add("generator_expression")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict):
        self.techniques.add("dict_literal")
        self.techniques.add("dict_usage")
        self.generic_visit(node)

    def visit_Set(self, node: ast.Set):
        self.techniques.add("set_literal")
        self.techniques.add("set_usage")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        self.assignment_count += 1
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self.assignment_count += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self.assignment_count += 1
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        self.return_count += 1
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            self.techniques.add("slicing")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in {"queue", "stack", "frontier"}:
            self.techniques.add("queue")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        call_name = self._call_name(node.func)
        if call_name:
            self.calls.append(call_name)
            base_name = call_name.split(".", 1)[0]
            if call_name in DANGEROUS_CALLS or base_name in DANGEROUS_CALLS:
                self.dangerous_calls.add(call_name)
            if base_name == self.function_name:
                self.techniques.add("recursion")
                self.techniques.add("self_call")
            if base_name == "set":
                self.techniques.add("set_usage")
                self.techniques.add("set_call")
            if base_name == "dict":
                self.techniques.add("dict_usage")
                self.techniques.add("dict_call")
            if base_name == "enumerate":
                self.techniques.add("enumerate")
            if base_name == "zip":
                self.techniques.add("zip_usage")
            if base_name == "sum":
                self.techniques.add("sum_call")
            if base_name in {"sorted"} or call_name.endswith(".sort"):
                self.techniques.add("sort_call")
            if base_name == "sorted":
                self.techniques.add("sorted_call")
            if call_name.endswith(".sort"):
                self.techniques.add("list_sort")
            if base_name == "groupby" or call_name.endswith(".groupby"):
                self.techniques.add("itertools_groupby")
                self.techniques.add("groupby_call")
        self.generic_visit(node)

    def finalize_undefined_calls(self):
        for call_name in self.calls:
            base = call_name.split(".", 1)[0]
            if "." in call_name:
                module = self.import_aliases.get(base)
                if module:
                    continue
                
                continue
            if base in ALLOWED_BUILTIN_CALLS or base in self.function_defs or base in self.import_aliases:
                continue
            if base in DANGEROUS_CALLS:
                continue
            self.undefined_calls.add(base)

    @staticmethod
    def _call_name(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = _TechniqueVisitor._call_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""


class NeoCoderScorer:
    """Scores one NeoCoder response against executable Python unit tests."""

    def __init__(
        self,
        *,
        beta_ih: float = DEFAULT_NEOCODER_BETA_IH,
        beta_hi: float = DEFAULT_NEOCODER_BETA_HI,
    ):
        self.beta_ih = float(beta_ih)
        self.beta_hi = float(beta_hi)
        self.task_overlay_payload = load_neocoder_v3_task_overlay()
        self.task_overlay = self.task_overlay_payload.get("tasks") if isinstance(self.task_overlay_payload.get("tasks"), dict) else {}
        self.technique_alias_payload = load_neocoder_technique_aliases()
        aliases = self.technique_alias_payload.get("denied_aliases") if isinstance(self.technique_alias_payload, dict) else {}
        self.denied_aliases = aliases if isinstance(aliases, dict) else {}
        self.common_solution_bank_payload = load_neocoder_common_solution_bank()
        self.common_solution_bank = self.common_solution_bank_payload.get("tasks") if isinstance(self.common_solution_bank_payload.get("tasks"), dict) else {}
        self.params = load_neocoder_v3_calibration_params()

    def parse_response(self, raw_text: str) -> Dict[str, object]:
        payload = _extract_json_payload(raw_text)
        if isinstance(payload, dict):
            code = _clean_code(
                payload.get("code")
                or payload.get("solution")
                or payload.get("python_code")
                or ""
            )
            if not code:
                code = _extract_python_code(raw_text)
            return {
                "parse_valid": bool(code),
                "code": code,
                "technique_ledger": self._normalize_ledger(payload.get("technique_ledger") or payload.get("techniques") or []),
                "constraint_ledger": self._normalize_ledger(payload.get("constraint_ledger") or payload.get("constraints") or []),
                "complexity_claim": _clean_string(payload.get("complexity_claim") or payload.get("complexity") or ""),
                "strategy_summary": _clean_string(payload.get("strategy_summary") or ""),
                "denial_adaptation": _clean_string(payload.get("denial_adaptation") or ""),
                "invariant_notes": _clean_string(payload.get("invariant_notes") or ""),
                "edge_case_notes": _clean_string(payload.get("edge_case_notes") or ""),
                "raw_payload": payload,
                "parse_error": None if code else "missing_code",
                "legacy_fallback": False,
            }

        code = _extract_python_code(raw_text)
        return {
            "parse_valid": bool(code),
            "code": code,
            "technique_ledger": [],
            "constraint_ledger": [],
            "complexity_claim": "",
            "strategy_summary": "",
            "denial_adaptation": "",
            "invariant_notes": "",
            "edge_case_notes": "",
            "raw_payload": payload,
            "parse_error": None if code else "no_json_or_python_code",
            "legacy_fallback": True,
        }

    def _normalize_ledger(self, value) -> List[Dict[str, object]]:
        records = []
        for index, raw in enumerate(_as_list(value), start=1):
            if isinstance(raw, str):
                text = _clean_string(raw)
                if text:
                    records.append({"id": f"L{index}", "text": text})
                continue
            if isinstance(raw, Mapping):
                record = {
                    "id": _clean_string(raw.get("id") or raw.get("technique") or raw.get("constraint") or f"L{index}"),
                    "text": _clean_string(raw.get("text") or raw.get("description") or raw.get("claim") or ""),
                }
                record.update({
                    key: raw.get(key)
                    for key in ("used", "satisfied", "evidence", "notes")
                    if key in raw
                })
                records.append(record)
        return records

    def score_task(self, task: Dict[str, object], parsed_response: Dict[str, object]) -> Dict[str, object]:
        task = self._task_with_overlay(task)
        code = _clean_code(parsed_response.get("code") or "")
        static_record = self._static_verify(task, code)
        should_execute = (
            bool(code)
            and not static_record["syntax_error"]
            and not static_record["missing_entrypoint"]
            and not static_record["dangerous_call_violation"]
            and not static_record["import_violation"]
        )
        execution_record = (
            self._execute_unit_tests(task, code)
            if should_execute else
            self._skipped_execution_record(task, reason=static_record["skip_reason"])
        )

        total_test_count = max(1, execution_record.get("total_tests") or len(task.get("unit_tests") or []))
        pass_rate = clip01(execution_record["passed"] / total_test_count)
        public_pass_rate = clip01(execution_record.get("group_pass_rates", {}).get("public", pass_rate))
        hidden_pass_rate = clip01(execution_record.get("group_pass_rates", {}).get("hidden", pass_rate))
        metamorphic_pass_rate = clip01(execution_record.get("group_pass_rates", {}).get("metamorphic", pass_rate))
        runtime_error_rate = clip01(execution_record["runtime_errors"] / total_test_count)
        timeout_rate = clip01(execution_record.get("timeout_rate", 0.0))
        detected = {_normalize_label(item) for item in static_record["detected_techniques"]}
        denied_roots = [_normalize_label(item) for item in task.get("denied_techniques") or [] if _normalize_label(item)]
        denied_alias_map = self._expanded_denied_aliases(denied_roots)
        denied_hits = sorted(
            alias for aliases in denied_alias_map.values() for alias in aliases
            if alias in detected
        )
        denied_root_hits = sorted(root for root, aliases in denied_alias_map.items() if detected & set(aliases))
        banned_technique_violation_rate = (
            clip01(len(denied_root_hits) / max(1, len(denied_roots)))
            if denied_roots else 0.0
        )
        import_violation = clip01(static_record["import_violation"])
        dangerous_call_violation = clip01(static_record["dangerous_call_violation"])
        missing_entrypoint = clip01(static_record["missing_entrypoint"])
        syntax_error = clip01(static_record["syntax_error"])
        literal_overfit_rate = self._literal_overfit_rate(task, code)
        mutation_violation_rate = clip01(execution_record.get("mutation_violation_rate", 0.0))
        constraint_following = clip01(1.0 - max(
            banned_technique_violation_rate,
            import_violation,
            dangerous_call_violation,
            missing_entrypoint,
            syntax_error,
        ))
        technique_novelty = self._technique_novelty(task, detected)
        strategy_shift = self._strategy_shift(task, detected, denied_hits)
        safety_gate = 0.0 if (
            static_record["syntax_error"]
            or static_record["missing_entrypoint"]
            or static_record["dangerous_call_violation"]
            or static_record["import_violation"]
        ) else 1.0
        missing_complexity_claim, unsupported_complexity_claim = self._complexity_claim_record(
            task,
            parsed_response.get("complexity_claim") or "",
        )
        hallucinated_import_rate = static_record["hallucinated_import_rate"]
        unsupported_api_call_rate = static_record["unsupported_api_call_rate"]
        ledger_mismatch_rate = self._ledger_mismatch_rate(
            parsed_response,
            detected=detected,
            denied_hits=denied_hits,
            static_record=static_record,
        )
        strategy_rarity, rarity_record = self._strategy_rarity(
            task,
            code=code,
            parsed_response=parsed_response,
            detected=detected,
            denied_hits=denied_hits,
            literal_overfit_rate=literal_overfit_rate,
            mutation_violation_rate=mutation_violation_rate,
            static_record=static_record,
        )
        functional_weights = self.params.get("functional_quality_weights") or {}
        functional_quality = clip01(
            float(functional_weights.get("public_pass", 0.25)) * public_pass_rate +
            float(functional_weights.get("hidden_pass", 0.55)) * hidden_pass_rate +
            float(functional_weights.get("metamorphic_pass", 0.20)) * metamorphic_pass_rate
        )
        constraint_quality = clip01(1.0 - max(
            syntax_error,
            missing_entrypoint,
            import_violation,
            dangerous_call_violation,
            banned_technique_violation_rate,
        ))
        anti_overfit_gate = clip01(1.0 - max(literal_overfit_rate, mutation_violation_rate))
        if static_record["syntax_error"] or static_record["missing_entrypoint"] or static_record["dangerous_call_violation"] or static_record["import_violation"]:
            anti_overfit_gate = 0.0
        implementation_depth = self._implementation_depth(static_record, code, detected)
        denial_adaptation = self._denial_adaptation_score(task, detected, denied_root_hits, parsed_response)
        algorithmic_pattern_diversity = self._algorithmic_pattern_diversity(static_record, detected)
        denial_adaptation_quality = self._denial_adaptation_quality(
            task,
            denied_root_hits=denied_root_hits,
            functional_quality=functional_quality,
            constraint_quality=constraint_quality,
            parsed_response=parsed_response,
        )
        ledger_consistency = clip01(1.0 - ledger_mismatch_rate)
        multiplier_weights = self.params.get("quality_multiplier_weights") or {}
        quality_multiplier = clip01(
            float(multiplier_weights.get("base", 0.35)) +
            float(multiplier_weights.get("implementation_depth", 0.20)) * implementation_depth +
            float(multiplier_weights.get("denial_adaptation", 0.15)) * denial_adaptation +
            float(multiplier_weights.get("algorithmic_pattern_diversity", 0.15)) * algorithmic_pattern_diversity +
            float(multiplier_weights.get("denial_adaptation_quality", 0.10)) * denial_adaptation_quality +
            float(multiplier_weights.get("ledger_consistency", 0.05)) * ledger_consistency
        )
        rarity_gamma = float(self.params.get("rarity_gamma", 1.35))
        functional_gamma = float(self.params.get("functional_gamma", 1.45))
        constraint_gamma = float(self.params.get("constraint_gamma", 1.25))
        v3_imagination_raw = clip01(
            math.pow(clip01(strategy_rarity), rarity_gamma) *
            math.pow(clip01(functional_quality), functional_gamma) *
            math.pow(clip01(constraint_quality), constraint_gamma) *
            anti_overfit_gate *
            quality_multiplier
        )
        if banned_technique_violation_rate > 0:
            v3_imagination_raw *= (0.40 + 0.60 * functional_quality)
        if mutation_violation_rate > 0:
            v3_imagination_raw *= (0.40 + 0.60 * functional_quality)
        v3_imagination_raw = clip01(v3_imagination_raw)

        h_logic = clip01(max(
            1.0 - functional_quality,
            static_record["syntax_error"],
            runtime_error_rate,
            timeout_rate,
            mutation_violation_rate,
        ))
        h_intent = clip01(max(
            banned_technique_violation_rate,
            import_violation,
            dangerous_call_violation,
            missing_entrypoint,
            mutation_violation_rate,
            literal_overfit_rate,
            0.25 * missing_complexity_claim,
        ))
        h_fact = clip01(max(
            unsupported_complexity_claim,
            hallucinated_import_rate,
            unsupported_api_call_rate,
            ledger_mismatch_rate,
        ))
        hallucination_raw = clip01(0.45 * h_logic + 0.35 * h_intent + 0.20 * h_fact)
        fallback_imagination_raw = clip01(
            0.35 * pass_rate +
            0.25 * constraint_following +
            0.25 * technique_novelty +
            0.15 * strategy_shift
        )
        
        
        if task.get("task_overlay_version") == NEOCODER_V3_TASK_OVERLAY_VERSION:
            imagination_raw = v3_imagination_raw
            execution_gate = clip01(0.30 + 0.35 * pass_rate + 0.35 * constraint_following)
            imagination_gated = clip01(imagination_raw * execution_gate * safety_gate)
        else:
            imagination_raw = max(v3_imagination_raw, fallback_imagination_raw)
            execution_gate = clip01(pass_rate * constraint_following)
            imagination_gated = clip01(imagination_raw * pass_rate * constraint_following * safety_gate)
        imagination = clip01(imagination_gated - self.beta_ih * hallucination_raw)
        hallucination = clip01(hallucination_raw - self.beta_hi * imagination_gated)

        primitive_means = {
            "pass_rate": round(pass_rate, 4),
            "public_pass_rate": round(public_pass_rate, 4),
            "hidden_pass_rate": round(hidden_pass_rate, 4),
            "metamorphic_pass_rate": round(metamorphic_pass_rate, 4),
            "functional_quality": round(functional_quality, 4),
            "constraint_following": round(constraint_following, 4),
            "constraint_quality": round(constraint_quality, 4),
            "technique_novelty": round(technique_novelty, 4),
            "strategy_shift": round(strategy_shift, 4),
            "strategy_rarity": round(strategy_rarity, 4),
            "implementation_depth": round(implementation_depth, 4),
            "denial_adaptation": round(denial_adaptation, 4),
            "algorithmic_pattern_diversity": round(algorithmic_pattern_diversity, 4),
            "denial_adaptation_quality": round(denial_adaptation_quality, 4),
            "ledger_consistency": round(ledger_consistency, 4),
            "anti_overfit_gate": round(anti_overfit_gate, 4),
            "literal_overfit_rate": round(literal_overfit_rate, 4),
            "mutation_violation_rate": round(mutation_violation_rate, 4),
            "banned_technique_violation_rate": round(banned_technique_violation_rate, 4),
            "denied_alias_violation_rate": round(banned_technique_violation_rate, 4),
            "hallucinated_import_rate": round(hallucinated_import_rate, 4),
            "unsupported_complexity_claim": round(unsupported_complexity_claim, 4),
            "missing_complexity_claim": round(missing_complexity_claim, 4),
            "unsupported_api_call_rate": round(unsupported_api_call_rate, 4),
            "ledger_mismatch_rate": round(ledger_mismatch_rate, 4),
            "timeout_rate": round(timeout_rate, 4),
            "runtime_error_rate": round(runtime_error_rate, 4),
            "syntax_error": round(static_record["syntax_error"], 4),
            "missing_entrypoint": round(missing_entrypoint, 4),
            "import_violation": round(import_violation, 4),
            "dangerous_call_violation": round(dangerous_call_violation, 4),
            "H_logic": round(h_logic, 4),
            "H_intent": round(h_intent, 4),
            "H_fact": round(h_fact, 4),
        }
        task_result = {
            "version": NEOCODER_VERSION,
            "task_id": task.get("id"),
            "base_task_id": task.get("base_task_id"),
            "denial_state": int(task.get("denial_state") or 0),
            "test_visibility_policy": NEOCODER_V3_TEST_VISIBILITY_POLICY,
            "calibration_policy": NEOCODER_V3_CALIBRATION_POLICY,
            "score": round(imagination, 4),
            "imagination": round(imagination, 4),
            "hallucination": round(hallucination, 4),
            "imagination_raw": round(imagination_raw, 4),
            "imagination_gated": round(imagination_gated, 4),
            "hallucination_raw": round(hallucination_raw, 4),
            "execution_gate": round(execution_gate, 4),
            "pass_rate": round(pass_rate, 4),
            "public_pass_rate": round(public_pass_rate, 4),
            "hidden_pass_rate": round(hidden_pass_rate, 4),
            "metamorphic_pass_rate": round(metamorphic_pass_rate, 4),
            "functional_quality": round(functional_quality, 4),
            "strategy_rarity": round(strategy_rarity, 4),
            "implementation_depth": round(implementation_depth, 4),
            "constraint_quality": round(constraint_quality, 4),
            "denial_adaptation": round(denial_adaptation, 4),
            "algorithmic_pattern_diversity": round(algorithmic_pattern_diversity, 4),
            "denial_adaptation_quality": round(denial_adaptation_quality, 4),
            "anti_overfit_gate": round(anti_overfit_gate, 4),
            "literal_overfit_rate": round(literal_overfit_rate, 4),
            "mutation_violation_rate": round(mutation_violation_rate, 4),
            "ledger_mismatch_rate": round(ledger_mismatch_rate, 4),
            "constraint_following": round(constraint_following, 4),
            "technique_novelty": round(technique_novelty, 4),
            "strategy_shift": round(strategy_shift, 4),
            "H_logic": round(h_logic, 4),
            "H_intent": round(h_intent, 4),
            "H_fact": round(h_fact, 4),
            "primitive_means": primitive_means,
            "static_record": static_record,
            "execution_record": execution_record,
            "rarity_record": rarity_record,
            "detected_techniques": sorted(detected),
            "denied_technique_hits": denied_hits,
            "denied_technique_roots": denied_roots,
            "denied_technique_root_hits": denied_root_hits,
            "parsed_response": {
                key: value
                for key, value in parsed_response.items()
                if key != "raw_payload"
            },
            "formula": {
                "imagination_raw": "T6  I=rarity^1.35*functional_quality^1.45*constraint_quality^1.25*anti_overfit_gate*(0.35+0.20*implementation_depth+0.15*denial_adaptation+0.15*algorithmic_pattern_diversity+0.10*denial_adaptation_quality+0.05*ledger_consistency)",
                "functional_quality": "0.30*public_pass+0.45*hidden_pass+0.25*metamorphic_pass",
                "imagination_gated": "I_gated=I_raw*(0.30+0.35*pass_rate+0.35*constraint_following)*safety_gate; denied/mutation violations use soft functional penalties",
                "hallucination_raw": "H_raw=0.45*H_logic+0.35*H_intent+0.20*H_fact",
                "residual": "I=clip01(I_gated-beta_IH*H_raw); H=clip01(H_raw-beta_HI*I_gated)",
            },
            "residualization": {
                "beta_IH": self.beta_ih,
                "beta_HI": self.beta_hi,
                "source": NEOCODER_V3_CALIBRATION_POLICY,
                "standardization": "clip01_raw_v1",
            },
        }
        task_result["subtype_contributions"] = build_neocoder_task_subtype_contributions(
            task_result,
            beta_ih=self.beta_ih,
            beta_hi=self.beta_hi,
        )
        task_result["atom_signals"] = task_result["subtype_contributions"].get("atom_signals", {})
        return task_result

    def _task_with_overlay(self, task: Mapping[str, object]) -> Dict[str, object]:
        task_copy = copy.deepcopy(dict(task))
        task_id = str(task_copy.get("id") or "")
        overlay = self.task_overlay.get(task_id)
        if isinstance(overlay, dict):
            task_copy.update(copy.deepcopy(overlay))
            task_copy["task_overlay_version"] = NEOCODER_V3_TASK_OVERLAY_VERSION
            task_copy["test_visibility_policy"] = NEOCODER_V3_TEST_VISIBILITY_POLICY
        return task_copy

    def _expanded_denied_aliases(self, denied_roots: Sequence[str]) -> Dict[str, Set[str]]:
        expanded = {}
        for root in denied_roots:
            aliases = self.denied_aliases.get(root) if isinstance(self.denied_aliases, Mapping) else None
            if aliases is None:
                aliases = [root]
            expanded[root] = {
                _normalize_label(alias)
                for alias in _as_list(aliases)
                if _normalize_label(alias)
            }
            expanded[root].add(root)
        return expanded

    def _scoring_tests(self, task: Mapping[str, object]) -> List[Dict[str, object]]:
        tests = []
        public = task.get("public_examples")
        if not isinstance(public, list) or not public:
            public = task.get("unit_tests") or []
        for test in public:
            record = copy.deepcopy(test)
            record["group"] = "public"
            tests.append(record)
        hidden = task.get("hidden_unit_tests") or []
        for test in hidden:
            record = copy.deepcopy(test)
            record["group"] = "hidden"
            tests.append(record)
        metamorphic = task.get("metamorphic_tests") or []
        for test in metamorphic:
            record = copy.deepcopy(test)
            record["group"] = "metamorphic"
            tests.append(record)
        mutation_checks = task.get("input_mutation_checks") or []
        for test in mutation_checks:
            record = copy.deepcopy(test)
            record["group"] = "hidden"
            record["mutation_check"] = True
            tests.append(record)
        return tests

    def _literal_overfit_rate(self, task: Mapping[str, object], code: str) -> float:
        if not code:
            return 0.0
        compact_code = re.sub(r"\s+", "", code)
        hits = 0
        checks = 0
        public_examples = task.get("public_examples") or task.get("unit_tests") or []
        for example in public_examples:
            if not isinstance(example, Mapping):
                continue
            expected = example.get("expected")
            if expected is None or expected == [] or expected == {}:
                continue
            checks += 1
            expected_json = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
            expected_py = repr(expected)
            if re.sub(r"\s+", "", expected_json) in compact_code or re.sub(r"\s+", "", expected_py) in compact_code:
                hits += 1
        if "if len(" in code and len(public_examples) >= 2:
            lengths = []
            for example in public_examples:
                args = example.get("args") if isinstance(example, Mapping) else None
                if isinstance(args, list) and args:
                    try:
                        lengths.append(str(len(args[0])))
                    except Exception:
                        pass
            if lengths and all(length in code for length in set(lengths)):
                hits += 1
                checks += 1
        return clip01(hits / max(1, checks))

    def _bank_for_task(self, task: Mapping[str, object]) -> Mapping[str, object]:
        task_id = str(task.get("id") or "")
        bank = self.common_solution_bank.get(task_id)
        return bank if isinstance(bank, Mapping) else {}

    def _family_hit(self, family: Mapping[str, object], *, text: str, detected: Set[str]) -> bool:
        keywords = [
            str(item).lower()
            for item in _as_list(family.get("keywords"))
            if str(item).strip()
        ]
        techniques = {
            _normalize_label(item)
            for item in _as_list(family.get("techniques"))
            if _normalize_label(item)
        }
        text_hit = bool(keywords) and any(keyword in text for keyword in keywords)
        technique_hit = bool(techniques) and bool(techniques & detected)
        family_id = str(family.get("id") or "")
        if family_id.startswith("denied_"):
            return technique_hit
        if keywords and techniques:
            return text_hit and technique_hit
        return text_hit or technique_hit

    def _strategy_rarity(
        self,
        task: Mapping[str, object],
        *,
        code: str,
        parsed_response: Mapping[str, object],
        detected: Set[str],
        denied_hits: Sequence[str],
        literal_overfit_rate: float,
        mutation_violation_rate: float,
        static_record: Mapping[str, object],
    ) -> tuple[float, Dict[str, object]]:
        rarity_cfg = self.params.get("rarity") or {}
        hard_zero = float(rarity_cfg.get("hard_zero", 0.0))
        broad_cap = float(rarity_cfg.get("broad_common_cap", 0.35))
        rare_floor = float(rarity_cfg.get("supported_rare_floor", 0.72))
        default_floor = float(rarity_cfg.get("default_floor", 0.42))
        text = " ".join([
            code or "",
            _clean_string(parsed_response.get("strategy_summary")),
            _clean_string(parsed_response.get("denial_adaptation")),
            _clean_string(parsed_response.get("invariant_notes")),
            _clean_string(parsed_response.get("edge_case_notes")),
            json.dumps(parsed_response.get("technique_ledger") or [], ensure_ascii=False),
        ]).lower()
        bank = self._bank_for_task(task)
        hard_reasons = []
        if denied_hits:
            hard_reasons.append("denied_alias")
        if literal_overfit_rate > 0:
            hard_reasons.append("literal_overfit")
        if mutation_violation_rate > 0:
            hard_reasons.append("input_mutation")
        if static_record.get("dangerous_call_violation") or static_record.get("import_violation"):
            hard_reasons.append("unsafe_or_import")
        for family in bank.get("hard_zero_code_families") or []:
            if isinstance(family, Mapping) and self._family_hit(family, text=text, detected=detected):
                hard_reasons.append(str(family.get("id") or "hard_zero_family"))
        if hard_reasons:
            return clip01(hard_zero), {
                "rarity_class": "hard_zero",
                "matched_families": sorted(set(hard_reasons)),
                "common_bank_version": NEOCODER_V3_COMMON_SOLUTION_BANK_VERSION,
            }

        broad_hits = []
        for family in bank.get("broad_common_code_families") or []:
            if isinstance(family, Mapping) and self._family_hit(family, text=text, detected=detected):
                broad_hits.append(str(family.get("id") or "broad_common_family"))
        rare_hits = []
        for family in bank.get("supported_rare_strategy_families") or []:
            if isinstance(family, Mapping) and self._family_hit(family, text=text, detected=detected):
                rare_hits.append(str(family.get("id") or "supported_rare_family"))
        if rare_hits:
            novelty = self._technique_novelty(task, detected)
            return clip01(max(rare_floor, 0.70 + 0.25 * novelty)), {
                "rarity_class": "supported_rare",
                "matched_families": sorted(set(rare_hits)),
                "common_bank_version": NEOCODER_V3_COMMON_SOLUTION_BANK_VERSION,
            }
        if broad_hits:
            return clip01(min(broad_cap, 0.20 + 0.20 * self._technique_novelty(task, detected))), {
                "rarity_class": "broad_common",
                "matched_families": sorted(set(broad_hits)),
                "common_bank_version": NEOCODER_V3_COMMON_SOLUTION_BANK_VERSION,
            }
        fallback = clip01(default_floor + 0.35 * self._technique_novelty(task, detected))
        return fallback, {
            "rarity_class": "unmatched_default",
            "matched_families": [],
            "common_bank_version": NEOCODER_V3_COMMON_SOLUTION_BANK_VERSION,
        }

    def _implementation_depth(self, static_record: Mapping[str, object], code: str, detected: Set[str]) -> float:
        lines = [line for line in (code or "").splitlines() if line.strip()]
        control = clip01(0.25 * static_record.get("max_loop_depth", 0) + 0.15 * len({"while_loop", "recursion", "helper_function", "queue"} & detected))
        state = clip01((float(static_record.get("assignment_count", 0)) + 0.5 * float(static_record.get("return_count", 0))) / 8.0)
        branching = clip01(float(static_record.get("branch_count", 0)) / 4.0)
        size = clip01(len(lines) / 22.0)
        return clip01(0.35 * control + 0.30 * state + 0.20 * branching + 0.15 * size)

    def _denial_adaptation_score(
        self,
        task: Mapping[str, object],
        detected: Set[str],
        denied_root_hits: Sequence[str],
        parsed_response: Mapping[str, object],
    ) -> float:
        denied = [_normalize_label(item) for item in task.get("denied_techniques") or [] if _normalize_label(item)]
        if denied_root_hits:
            return 0.0
        non_common = [
            item for item in detected
            if item not in {_normalize_label(x) for x in task.get("common_techniques") or []}
            and item not in {"if_statement"}
        ]
        notes = " ".join([
            _clean_string(parsed_response.get("denial_adaptation")),
            _clean_string(parsed_response.get("strategy_summary")),
        ]).lower()
        explanation = 0.2 if any(word in notes for word in ("avoid", "without", "denied", "instead", "state")) else 0.0
        base = 0.55 if denied else 0.45
        return clip01(base + 0.25 * min(1.0, len(non_common) / max(1, len(denied) or 1)) + explanation)

    def _algorithmic_pattern_diversity(self, static_record: Mapping[str, object], detected: Set[str]) -> float:
        patterns = set()
        if "nested_loop" in detected:
            patterns.add("nested_scan")
        if "while_loop" in detected and "for_loop" not in detected:
            patterns.add("manual_state_machine")
        if "recursion" in detected or "self_call" in detected:
            patterns.add("recursive_decomposition")
        if "queue" in detected:
            patterns.add("frontier_search")
        if {"dict_usage", "dict_call", "dict_literal"} & detected:
            patterns.add("associative_index")
        if {"set_usage", "set_call", "set_literal"} & detected:
            patterns.add("membership_set")
        if {"sort_call", "sorted_call", "list_sort"} & detected:
            patterns.add("order_transform")
        if {"list_comprehension", "generator_expression", "dict_comprehension", "set_comprehension"} & detected:
            patterns.add("declarative_comprehension")
        if static_record.get("max_loop_depth", 0) >= 2 and static_record.get("branch_count", 0) >= 2:
            patterns.add("multi_state_iteration")
        if static_record.get("assignment_count", 0) >= 5 and static_record.get("return_count", 0) >= 1:
            patterns.add("state_accumulation")
        return clip01(len(patterns) / 4.0)

    def _denial_adaptation_quality(
        self,
        task: Mapping[str, object],
        *,
        denied_root_hits: Sequence[str],
        functional_quality: float,
        constraint_quality: float,
        parsed_response: Mapping[str, object],
    ) -> float:
        denied_count = len([item for item in task.get("denied_techniques") or [] if _normalize_label(item)])
        if denied_count <= 0:
            return 0.0
        if denied_root_hits:
            return 0.0
        notes = " ".join([
            _clean_string(parsed_response.get("denial_adaptation")),
            _clean_string(parsed_response.get("invariant_notes")),
            _clean_string(parsed_response.get("edge_case_notes")),
        ]).lower()
        explanation = 1.0 if any(term in notes for term in ("without", "avoid", "instead", "invariant", "edge", "denied")) else 0.35
        pressure = clip01(denied_count / 3.0)
        performance = clip01(0.65 * functional_quality + 0.35 * constraint_quality)
        return clip01(0.45 * performance + 0.30 * pressure + 0.25 * explanation)

    def _ledger_mismatch_rate(
        self,
        parsed_response: Mapping[str, object],
        *,
        detected: Set[str],
        denied_hits: Sequence[str],
        static_record: Mapping[str, object],
    ) -> float:
        mismatches = 0
        checks = 0
        ledger_text = " ".join(
            _clean_string(item.get("text") or item.get("id"))
            for item in parsed_response.get("technique_ledger") or []
            if isinstance(item, Mapping)
        ).lower()
        for label in [
            "for_loop", "set_usage", "dict_usage", "sort_call", "recursion",
            "while_loop", "list_comprehension", "itertools_groupby",
        ]:
            label_words = label.replace("_", " ")
            if label in ledger_text or label_words in ledger_text:
                checks += 1
                aliases = {
                    _normalize_label(alias)
                    for alias in _as_list(self.denied_aliases.get(label, []))
                    if _normalize_label(alias)
                }
                if label not in detected and not aliases & detected:
                    mismatches += 1
        for item in parsed_response.get("constraint_ledger") or []:
            if not isinstance(item, Mapping):
                continue
            text = _clean_string(item.get("text") or item.get("id")).lower()
            satisfied = item.get("satisfied")
            if satisfied is True and any(word in text for word in ("denied", "import", "entrypoint", "safe")):
                checks += 1
                if denied_hits or static_record.get("import_violation") or static_record.get("missing_entrypoint"):
                    mismatches += 1
        if not parsed_response.get("technique_ledger") or not parsed_response.get("constraint_ledger"):
            checks += 1
            mismatches += 1
        return clip01(mismatches / max(1, checks))

    def _static_verify(self, task: Dict[str, object], code: str) -> Dict[str, object]:
        if not code:
            return {
                "syntax_error": 1.0,
                "syntax_error_message": "missing_code",
                "missing_entrypoint": 1.0,
                "detected_techniques": [],
                "imports": [],
                "unknown_imports": [],
                "dangerous_calls": [],
                "unsupported_api_calls": [],
                "import_violation": 0.0,
                "dangerous_call_violation": 0.0,
                "hallucinated_import_rate": 0.0,
                "unsupported_api_call_rate": 0.0,
                "assignment_count": 0,
                "branch_count": 0,
                "return_count": 0,
                "max_loop_depth": 0,
                "skip_reason": "missing_code",
            }
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return {
                "syntax_error": 1.0,
                "syntax_error_message": f"{exc.__class__.__name__}: {exc}",
                "missing_entrypoint": 1.0,
                "detected_techniques": [],
                "imports": [],
                "unknown_imports": [],
                "dangerous_calls": [],
                "unsupported_api_calls": [],
                "import_violation": 0.0,
                "dangerous_call_violation": 0.0,
                "hallucinated_import_rate": 0.0,
                "unsupported_api_call_rate": 0.0,
                "assignment_count": 0,
                "branch_count": 0,
                "return_count": 0,
                "max_loop_depth": 0,
                "skip_reason": "syntax_error",
            }

        function_name = str(task.get("function_name") or "")
        visitor = _TechniqueVisitor(function_name=function_name)
        visitor.visit(tree)
        visitor.finalize_undefined_calls()
        allowed_imports = {str(item) for item in task.get("allowed_imports") or []}
        imports = sorted(set(visitor.imports))
        unknown_imports = sorted(
            item for item in imports
            if item not in allowed_imports
        )
        dangerous_imports = sorted(set(imports) & DANGEROUS_IMPORTS)
        missing_entrypoint = 0.0 if function_name in visitor.function_defs else 1.0
        import_violation = 1.0 if unknown_imports or dangerous_imports else 0.0
        dangerous_calls = sorted(visitor.dangerous_calls)
        unsupported_api_calls = sorted(visitor.undefined_calls)
        skip_reason = ""
        if missing_entrypoint:
            skip_reason = "missing_entrypoint"
        if import_violation:
            skip_reason = "import_violation"
        if dangerous_calls:
            skip_reason = "dangerous_call_violation"
        return {
            "syntax_error": 0.0,
            "syntax_error_message": "",
            "missing_entrypoint": missing_entrypoint,
            "detected_techniques": sorted(visitor.techniques),
            "imports": imports,
            "unknown_imports": unknown_imports,
            "dangerous_imports": dangerous_imports,
            "dangerous_calls": dangerous_calls,
            "unsupported_api_calls": unsupported_api_calls,
            "import_violation": import_violation,
            "dangerous_call_violation": 1.0 if dangerous_calls else 0.0,
            "hallucinated_import_rate": clip01(len(unknown_imports) / max(1, len(imports))),
            "unsupported_api_call_rate": clip01(len(unsupported_api_calls) / max(1, len(visitor.calls))),
            "assignment_count": visitor.assignment_count,
            "branch_count": visitor.branch_count,
            "return_count": visitor.return_count,
            "max_loop_depth": visitor.max_loop_depth,
            "skip_reason": skip_reason,
        }

    def _execute_unit_tests(self, task: Dict[str, object], code: str) -> Dict[str, object]:
        scoring_tests = self._scoring_tests(task)
        payload = {
            "code": code,
            "function_name": task.get("function_name"),
            "unit_tests": scoring_tests,
            "allowed_imports": task.get("allowed_imports") or [],
            "memory_limit_mb": task.get("memory_limit_mb") or 128,
        }
        wrapper = _EXECUTION_WRAPPER
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix="_neocoder_runner.py", delete=False, encoding="utf-8") as handle:
                temp_path = handle.name
                handle.write(wrapper)
            completed = subprocess.run(
                [sys.executable, "-I", "-S", temp_path],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=max(0.25, float(task.get("time_limit_s") or 1.0) + 0.75),
            )
            if completed.returncode != 0 and not completed.stdout.strip():
                return self._failed_execution_record(
                    task,
                    reason="runner_error",
                    stderr=completed.stderr[-1000:],
                )
            try:
                record = json.loads(completed.stdout.strip() or "{}")
            except Exception:
                return self._failed_execution_record(
                    task,
                    reason="runner_json_error",
                    stderr=(completed.stderr + completed.stdout)[-1000:],
                )
            record.setdefault("version", "neocoder_execution_v1")
            record.setdefault("execution_skipped", False)
            record.setdefault("timeout_rate", 0.0)
            record.setdefault("total_tests", len(scoring_tests))
            record.setdefault("group_pass_rates", self._group_pass_rates(record.get("test_records") or []))
            record.setdefault("mutation_violation_rate", self._mutation_violation_rate(record.get("test_records") or []))
            return record
        except subprocess.TimeoutExpired:
            return self._failed_execution_record(task, reason="timeout", timeout_rate=1.0)
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _skipped_execution_record(self, task: Dict[str, object], *, reason: str) -> Dict[str, object]:
        return {
            "version": "neocoder_execution_v1",
            "execution_skipped": True,
            "skip_reason": reason or "static_check_failed",
            "passed": 0,
            "failed": len(self._scoring_tests(task)),
            "runtime_errors": 0,
            "timeout_rate": 0.0,
            "total_tests": len(self._scoring_tests(task)),
            "group_pass_rates": {"public": 0.0, "hidden": 0.0, "metamorphic": 0.0},
            "mutation_violation_rate": 0.0,
            "test_records": [],
        }

    def _failed_execution_record(
        self,
        task: Dict[str, object],
        *,
        reason: str,
        stderr: str = "",
        timeout_rate: float = 0.0,
    ) -> Dict[str, object]:
        return {
            "version": "neocoder_execution_v1",
            "execution_skipped": False,
            "runner_failure": reason,
            "stderr": stderr,
            "passed": 0,
            "failed": len(self._scoring_tests(task)),
            "runtime_errors": len(self._scoring_tests(task)),
            "timeout_rate": clip01(timeout_rate),
            "total_tests": len(self._scoring_tests(task)),
            "group_pass_rates": {"public": 0.0, "hidden": 0.0, "metamorphic": 0.0},
            "mutation_violation_rate": 0.0,
            "test_records": [],
        }

    def _group_pass_rates(self, records: Sequence[Mapping[str, object]]) -> Dict[str, float]:
        rates = {}
        for group in ("public", "hidden", "metamorphic"):
            group_records = [record for record in records if record.get("group") == group]
            if group_records:
                rates[group] = clip01(sum(1 for record in group_records if record.get("passed")) / len(group_records))
        return rates

    def _mutation_violation_rate(self, records: Sequence[Mapping[str, object]]) -> float:
        mutation_records = [
            record for record in records
            if record.get("mutation_check") or record.get("input_mutation_violation") is not None
        ]
        if not mutation_records:
            return 0.0
        return clip01(sum(1 for record in mutation_records if record.get("input_mutation_violation")) / len(mutation_records))

    def _technique_novelty(self, task: Mapping[str, object], detected: Set[str]) -> float:
        weights = {
            _normalize_label(key): clip01(value)
            for key, value in (task.get("technique_weights") or {}).items()
        }
        relevant = [
            weights.get(item, 0.55)
            for item in detected
            if item not in {"if_statement"}
        ]
        if not relevant:
            return 0.25
        return clip01(sum(relevant) / len(relevant))

    def _strategy_shift(self, task: Mapping[str, object], detected: Set[str], denied_hits: Sequence[str]) -> float:
        denial_state = int(task.get("denial_state") or 0)
        common = {_normalize_label(item) for item in task.get("common_techniques") or []}
        non_common = [item for item in detected if item not in common and item != "if_statement"]
        avoided = 1.0
        denied = [_normalize_label(item) for item in task.get("denied_techniques") or [] if _normalize_label(item)]
        if denied:
            avoided = clip01(1.0 - len(denied_hits) / max(1, len(denied)))
        if denial_state <= 0:
            return clip01(0.45 + 0.10 * min(1, len(non_common)))
        return clip01(0.25 + 0.50 * avoided + 0.25 * min(1.0, len(non_common) / max(1, denial_state)))

    def _complexity_claim_record(self, task: Mapping[str, object], claim: object) -> tuple[float, float]:
        claim_text = _clean_string(claim)
        if not claim_text:
            return 1.0, 0.0
        claim_norm = _normalize_complexity(claim_text)
        accepted = [
            _normalize_complexity(item)
            for item in task.get("accepted_complexity_claims") or []
            if _normalize_complexity(item)
        ]
        if not accepted:
            return 0.0, 0.0
        supported = any(item in claim_norm or claim_norm in item for item in accepted)
        return 0.0, 0.0 if supported else 1.0


def aggregate_neocoder_model_axes(
    task_scores: Sequence[Dict[str, object]],
    *,
    gate_pass: bool = True,
    beta_ih: float = DEFAULT_NEOCODER_BETA_IH,
    beta_hi: float = DEFAULT_NEOCODER_BETA_HI,
) -> Dict[str, object]:
    if not task_scores:
        return {
            "version": NEOCODER_VERSION,
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
            "calibration_policy": NEOCODER_V3_CALIBRATION_POLICY,
            "runtime_scoring_policy": NEOCODER_V3_RUNTIME_SCORING_POLICY,
            "test_visibility_policy": NEOCODER_V3_TEST_VISIBILITY_POLICY,
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

    primitive_fields = set()
    for score in task_scores:
        if isinstance(score.get("primitive_means"), dict):
            primitive_fields.update(score["primitive_means"].keys())
    primitive_means = {}
    for field in sorted(primitive_fields):
        value = mean_or_none(
            score.get("primitive_means", {}).get(field)
            for score in task_scores
            if isinstance(score.get("primitive_means"), dict)
        )
        if value is not None:
            primitive_means[field] = round(value, 4)
    subtype_contributions = mean_subtype_contributions(
        score.get("subtype_contributions")
        for score in task_scores
        if isinstance(score.get("subtype_contributions"), dict)
    )
    return {
        "version": NEOCODER_VERSION,
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
            "source": NEOCODER_V3_CALIBRATION_POLICY,
            "standardization": "clip01_raw_v1",
        },
        "formula": {
            "task_imagination_raw": "T6  I=rarity^1.35*functional_quality^1.45*constraint_quality^1.25*anti_overfit_gate*(0.35+0.20*implementation_depth+0.15*denial_adaptation+0.15*algorithmic_pattern_diversity+0.10*denial_adaptation_quality+0.05*ledger_consistency)",
            "task_functional_quality": "0.30*public_pass+0.45*hidden_pass+0.25*metamorphic_pass",
            "task_imagination_gated": "I_gated=I_raw*(0.30+0.35*pass_rate+0.35*constraint_following)*safety_gate; denied/mutation violations use soft functional penalties",
            "task_hallucination_raw": "H_raw=0.45*H_logic+0.35*H_intent+0.20*H_fact",
            "model_residual": "I=clip01(mean(I_gated)-beta_IH*mean(H_raw)); H=clip01(mean(H_raw)-beta_HI*mean(I_gated))",
        },
        "calibration_policy": NEOCODER_V3_CALIBRATION_POLICY,
        "runtime_scoring_policy": NEOCODER_V3_RUNTIME_SCORING_POLICY,
        "test_visibility_policy": NEOCODER_V3_TEST_VISIBILITY_POLICY,
        "functional_quality": primitive_means.get("functional_quality"),
        "public_pass_rate": primitive_means.get("public_pass_rate"),
        "hidden_pass_rate": primitive_means.get("hidden_pass_rate"),
        "metamorphic_pass_rate": primitive_means.get("metamorphic_pass_rate"),
        "strategy_rarity": primitive_means.get("strategy_rarity"),
        "implementation_depth": primitive_means.get("implementation_depth"),
        "constraint_quality": primitive_means.get("constraint_quality"),
        "denial_adaptation": primitive_means.get("denial_adaptation"),
        "algorithmic_pattern_diversity": primitive_means.get("algorithmic_pattern_diversity"),
        "denial_adaptation_quality": primitive_means.get("denial_adaptation_quality"),
        "anti_overfit_gate": primitive_means.get("anti_overfit_gate"),
        "literal_overfit_rate": primitive_means.get("literal_overfit_rate"),
        "mutation_violation_rate": primitive_means.get("mutation_violation_rate"),
        "denied_alias_violation_rate": primitive_means.get("denied_alias_violation_rate"),
        "ledger_mismatch_rate": primitive_means.get("ledger_mismatch_rate"),
    }


_EXECUTION_WRAPPER = r'''
import builtins
import copy
import json
import sys
import traceback

try:
    import resource
except Exception:
    resource = None


def normalize(value):
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(val) for key, val in value.items()}
    return value


def main():
    payload = json.loads(sys.stdin.read())
    memory_mb = int(payload.get("memory_limit_mb") or 128)
    if resource is not None:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (2, 3))
        except Exception:
            pass
        try:
            memory = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        except Exception:
            pass

    allowed_imports = set(payload.get("allowed_imports") or [])
    real_import = builtins.__import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        top = (name or "").split(".")[0]
        if top not in allowed_imports:
            raise ImportError("Import is not allowed: " + str(top))
        return real_import(name, globals, locals, fromlist, level)

    safe_builtins = {
        "abs": builtins.abs,
        "all": builtins.all,
        "any": builtins.any,
        "bool": builtins.bool,
        "dict": builtins.dict,
        "enumerate": builtins.enumerate,
        "Exception": builtins.Exception,
        "float": builtins.float,
        "int": builtins.int,
        "isinstance": builtins.isinstance,
        "len": builtins.len,
        "list": builtins.list,
        "max": builtins.max,
        "min": builtins.min,
        "range": builtins.range,
        "reversed": builtins.reversed,
        "round": builtins.round,
        "set": builtins.set,
        "sorted": builtins.sorted,
        "str": builtins.str,
        "sum": builtins.sum,
        "tuple": builtins.tuple,
        "ValueError": builtins.ValueError,
        "zip": builtins.zip,
        "__import__": safe_import,
    }
    namespace = {"__builtins__": safe_builtins}
    records = []
    try:
        exec(payload["code"], namespace, namespace)
        func = namespace[payload["function_name"]]
    except Exception as exc:
        total = len(payload.get("unit_tests") or [])
        print(json.dumps({
            "passed": 0,
            "failed": total,
            "runtime_errors": total,
            "timeout_rate": 0.0,
            "setup_error": repr(exc),
            "test_records": [],
        }))
        return

    passed = 0
    runtime_errors = 0
    for index, test in enumerate(payload.get("unit_tests") or [], start=1):
        try:
            args = copy.deepcopy(test.get("args") or [])
            kwargs = copy.deepcopy(test.get("kwargs") or {})
            original_args = normalize(copy.deepcopy(args))
            original_kwargs = normalize(copy.deepcopy(kwargs))
            expected = normalize(test.get("expected"))
            actual = normalize(func(*args, **kwargs))
            ok = actual == expected
            mutated = normalize(args) != original_args or normalize(kwargs) != original_kwargs
            passed += int(ok)
            records.append({
                "index": index,
                "group": test.get("group") or "public",
                "mutation_check": bool(test.get("mutation_check")),
                "input_mutation_violation": bool(mutated),
                "passed": bool(ok),
                "expected": expected,
                "actual": actual,
            })
        except Exception as exc:
            runtime_errors += 1
            records.append({
                "index": index,
                "group": test.get("group") or "public",
                "mutation_check": bool(test.get("mutation_check")),
                "input_mutation_violation": None,
                "passed": False,
                "runtime_error": repr(exc),
                "traceback_tail": traceback.format_exc()[-500:],
            })

    total = len(payload.get("unit_tests") or [])
    groups = {}
    for record in records:
        group = record.get("group") or "public"
        groups.setdefault(group, [0, 0])
        groups[group][1] += 1
        groups[group][0] += int(bool(record.get("passed")))
    group_pass_rates = {
        group: (counts[0] / counts[1] if counts[1] else 0.0)
        for group, counts in groups.items()
    }
    mutation_records = [record for record in records if record.get("mutation_check")]
    mutation_violation_rate = (
        sum(1 for record in mutation_records if record.get("input_mutation_violation")) / len(mutation_records)
        if mutation_records else 0.0
    )
    print(json.dumps({
        "passed": passed,
        "failed": total - passed,
        "runtime_errors": runtime_errors,
        "timeout_rate": 0.0,
        "total_tests": total,
        "group_pass_rates": group_pass_rates,
        "mutation_violation_rate": mutation_violation_rate,
        "test_records": records,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


__all__ = [
    "NEOCODER_VERSION",
    "DEFAULT_NEOCODER_BETA_IH",
    "DEFAULT_NEOCODER_BETA_HI",
    "NEOCODER_V3_CALIBRATION_POLICY",
    "NEOCODER_V3_RUNTIME_SCORING_POLICY",
    "NEOCODER_V3_TEST_VISIBILITY_POLICY",
    "NEOCODER_V3_TASK_OVERLAY_VERSION",
    "NEOCODER_V3_TECHNIQUE_ALIAS_VERSION",
    "NEOCODER_V3_COMMON_SOLUTION_BANK_VERSION",
    "NeoCoderScorer",
    "aggregate_neocoder_model_axes",
    "get_neocoder_task_overlay_coverage",
    "get_neocoder_common_solution_bank_coverage",
    "get_neocoder_technique_alias_coverage",
]
