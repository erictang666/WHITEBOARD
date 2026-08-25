
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

from t1_assoc_v3 import T1_ASSOC_VERSION


T1_TYPED_AXIS_VERSION = "typed_axis_t1"
T2_TYPED_AXIS_VERSION = "typed_axis_t2"
T3_TYPED_AXIS_VERSION = "typed_axis_t3"
T4_TYPED_AXIS_VERSION = "typed_axis_t4"
T5_TYPED_AXIS_VERSION = "typed_axis_t5"
T6_TYPED_AXIS_VERSION = "typed_axis_t6"
T7_TYPED_AXIS_VERSION = "typed_axis_t7"
T8_TYPED_AXIS_VERSION = "typed_axis_t8"
TYPED_AXIS_VERSION = "typed_axis"
ATOM_SIGNAL_VERSION = "atom_signals"
T1_IMAGINATION_SUBTYPES = ("assoc",)
T1_HALLUCINATION_SUBTYPES = ("context", "intent", "drift", "logic")
T2_IMAGINATION_SUBTYPES = ("constraint",)
T2_HALLUCINATION_SUBTYPES = ("logic", "boundary", "intent")
T3_IMAGINATION_SUBTYPES = ("cf",)
T3_HALLUCINATION_SUBTYPES = ("logic", "context", "drift")
T4_IMAGINATION_SUBTYPES = ("narrative",)
T4_HALLUCINATION_SUBTYPES = ("detail", "context", "drift", "citation")
T5_IMAGINATION_SUBTYPES = ("hypothesis",)
T5_HALLUCINATION_SUBTYPES = ("fact", "citation", "boundary")
T6_IMAGINATION_SUBTYPES = ("code",)
T6_HALLUCINATION_SUBTYPES = ("logic", "intent", "fact")
T7_HALLUCINATION_SUBTYPES = ("fact", "logic", "boundary")
T8_IMAGINATION_SUBTYPES = ("analogy",)
T8_HALLUCINATION_SUBTYPES = ("false_transfer", "fact", "logic", "context")
CROSS_TASK_HALLUCINATION_SUBTYPES = ("consistency",)
IMAGINATION_SUBTYPES = ("assoc", "constraint", "cf", "narrative", "hypothesis", "code", "analogy")
HALLUCINATION_SUBTYPES = (
    "context",
    "intent",
    "drift",
    "logic",
    "boundary",
    "detail",
    "citation",
    "fact",
    "false_transfer",
    "consistency",
)





ATOM_SUBTYPE_DEPENDENCIES = {
    "UUT.novelty": {"I.assoc": "+"},
    "UUT.supported_affordance_ratio": {"I.assoc": "+"},
    "UUT.mechanism_completeness": {"I.assoc": "+"},
    "UUT.diversity": {"I.assoc": "+"},
    "UUT.appropriateness_gate": {"I.assoc": "+"},
    "UUT.unsupported_claim_ratio": {"H.context": "+"},
    "UUT.contradiction_ratio": {"H.context": "+"},
    "UUT.extra_tool_violation": {"H.intent": "+"},
    "UUT.cue_drift": {"H.drift": "+"},
    "UUT.physical_drift": {"H.drift": "+", "H.logic": "+"},
    "PropConj.novelty": {"I.assoc": "+"},
    "PropConj.grounding_for_I": {"I.assoc": "+"},
    "PropConj.appropriateness_gate": {"I.assoc": "+"},
    "PropConj.intent_coverage": {"I.assoc": "+", "H.intent": "-"},
    "PropConj.grounding_for_H": {"H.context": "+"},
    "PropConj.contradiction": {"H.context": "+", "H.logic": "+"},
    "PropConj.unresolved_entity": {"H.drift": "+"},
    "PropConj.evidence_mismatch": {"H.context": "+"},
    "MacGyver.imagination_raw": {"I.constraint": "+"},
    "MacGyver.constraint_gate": {"I.constraint": "+"},
    "MacGyver.F_feasibility": {"I.constraint": "+"},
    "MacGyver.M_mechanism_completeness": {"I.constraint": "+"},
    "MacGyver.N_plan_novelty": {"I.constraint": "+"},
    "MacGyver.C_contradiction_rate": {"H.logic": "+"},
    "MacGyver.A_unsupported_success_assertion": {"H.logic": "+"},
    "MacGyver.P_physical_affordance_support": {"H.logic": "-"},
    "MacGyver.boundary_hallucination": {"H.boundary": "+"},
    "MacGyver.U_unavailable_entity_rate": {"H.intent": "+"},
    "MacGyver.K_constraint_satisfaction": {"H.intent": "-"},
    "CJST.imagination_raw": {"I.cf": "+"},
    "CJST.consistency_gate": {"I.cf": "+"},
    "CJST.hard_gate": {"I.cf": "+"},
    "CJST.premise_relevance": {"I.cf": "+", "H.context": "-"},
    "CJST.premise_lock": {"I.cf": "+", "H.drift": "-"},
    "CJST.world_consistency": {"I.cf": "+", "H.drift": "-"},
    "CJST.formal_hallucination_raw": {"I.cf": "-", "H.context": "+", "H.drift": "+"},
    "CJST.contradiction": {"H.logic": "+"},
    "CJST.protected_variable_violation": {"H.logic": "+"},
    "CJST.forbidden_update_rate": {"H.logic": "+", "H.drift": "+"},
    "CJST.extra_miracle_rate": {"H.logic": "+", "H.drift": "+"},
    "CJST.causal_edge_support": {"H.logic": "-"},
    "CJST.unsupported_extra_claim_rate": {"H.context": "+"},
    "GCW.imagination_raw": {"I.narrative": "+"},
    "GCW.narrative_grounding": {"I.narrative": "+"},
    "GCW.constraint_gate": {"I.narrative": "+"},
    "GCW.claim_evidence_mismatch": {"H.detail": "+"},
    "GCW.contradiction_rate": {"H.context": "+"},
    "GCW.contradicted_claim_rate": {"H.context": "+"},
    "GCW.missing_required_fact_rate": {"H.context": "+"},
    "GCW.hard_constraint_violation_rate": {"H.drift": "+"},
    "GCW.hard_no_drift_violation": {"H.drift": "+"},
    "GCW.entity_persistence_failure": {"H.drift": "+"},
    "GCW.forbidden_motif_rate": {"H.drift": "+"},
    "GCW.entity_drift_rate": {"H.detail": "+", "H.drift": "+"},
    "GCW.citation_mismatch_rate": {"H.citation": "+"},
    "GCW.claim_without_evidence_rate": {"H.detail": "+", "H.citation": "+"},
    "HypoUseSpace.imagination_raw": {"I.hypothesis": "+"},
    "HypoUseSpace.evidence_support_gate": {"I.hypothesis": "+"},
    "HypoUseSpace.boundary_gate": {"I.hypothesis": "+"},
    "HypoUseSpace.unsupported_affordance_or_mechanism": {"H.fact": "+"},
    "HypoUseSpace.explicit_contradiction_or_forbidden_foil": {"H.fact": "+"},
    "HypoUseSpace.contradicted_claim_rate": {"H.fact": "+"},
    "HypoUseSpace.false_feasibility_claim": {"H.fact": "+"},
    "HypoUseSpace.unsupported_span_rate": {"H.fact": "+"},
    "HypoUseSpace.citation_mismatch_rate": {"H.citation": "+"},
    "HypoUseSpace.unknown_evidence_rate": {"H.citation": "+"},
    "HypoUseSpace.claim_without_evidence_rate": {"H.citation": "+"},
    "HypoUseSpace.unavailable_entity": {"H.boundary": "+"},
    "HypoUseSpace.constraint_or_observation_violation": {"H.boundary": "+"},
    "HypoUseSpace.evidence_boundary_violation": {"H.boundary": "+"},
    "NeoCoder.imagination_raw": {"I.code": "+"},
    "NeoCoder.pass_rate": {"I.code": "+", "H.logic": "-"},
    "NeoCoder.constraint_following": {"I.code": "+"},
    "NeoCoder.syntax_error": {"H.logic": "+"},
    "NeoCoder.runtime_error_rate": {"H.logic": "+"},
    "NeoCoder.timeout_rate": {"H.logic": "+"},
    "NeoCoder.banned_technique_violation_rate": {"H.intent": "+"},
    "NeoCoder.import_violation": {"H.intent": "+"},
    "NeoCoder.dangerous_call_violation": {"H.intent": "+"},
    "NeoCoder.missing_entrypoint": {"H.intent": "+"},
    "NeoCoder.unsupported_complexity_claim": {"H.fact": "+"},
    "NeoCoder.hallucinated_import_rate": {"H.fact": "+"},
    "NeoCoder.unsupported_api_call_rate": {"H.fact": "+"},
    "ClosedWorldFact.wrong_answer": {"H.fact": "+"},
    "ClosedWorldFact.contradicted_fact": {"H.fact": "+"},
    "ClosedWorldFact.forbidden_answer_hit": {"H.fact": "+"},
    "ClosedWorldFact.unsupported_fact_claim": {"H.fact": "+"},
    "ClosedWorldFact.rationale_keyword_failure": {"H.logic": "+"},
    "ClosedWorldFact.evidence_recall": {"H.logic": "-"},
    "ClosedWorldFact.false_presupposition_acceptance": {"H.boundary": "+"},
    "ClosedWorldFact.unanswerable_false_answer": {"H.boundary": "+"},
    "AnalogyTransfer.imagination_raw": {"I.analogy": "+"},
    "AnalogyTransfer.false_transfer_rate": {"H.false_transfer": "+"},
    "AnalogyTransfer.unsupported_inference_rate": {"H.false_transfer": "+"},
    "AnalogyTransfer.invented_fact_rate": {"H.fact": "+"},
    "AnalogyTransfer.unknown_evidence_rate": {"H.fact": "+"},
    "AnalogyTransfer.source_fact_accuracy": {"H.fact": "-"},
    "AnalogyTransfer.target_fact_accuracy": {"H.fact": "-"},
    "AnalogyTransfer.relational_consistency": {"I.analogy": "+", "H.logic": "-"},
    "AnalogyTransfer.wrong_mapping_direction_rate": {"H.logic": "+"},
    "AnalogyTransfer.dimension_inconsistency_rate": {"H.logic": "+"},
    "AnalogyTransfer.limit_awareness": {"H.context": "-"},
    "AnalogyTransfer.surface_stereotype_rate": {"H.context": "+"},
    "AnalogyTransfer.boundary_overclaim_rate": {"H.context": "+"},
    "CrossTaskConsistency.pair_inconsistency": {"H.consistency": "+"},
}
T1_COMPONENT_BASE_WEIGHTS = {
    "UUT": (1.0 / 7.0) * (0.18 / (0.18 + 0.14)),
    "PropConj": (1.0 / 7.0) * (0.14 / (0.18 + 0.14)),
}
TYPED_COMPONENT_BASE_WEIGHTS = {
    **T1_COMPONENT_BASE_WEIGHTS,
    "MacGyver": 1.0 / 7.0,
    "CJST": 1.0 / 7.0,
    "GCW": 1.0 / 7.0,
    "HypoUseSpace": 1.0 / 7.0,
    "NeoCoder": 1.0 / 7.0,
    "ClosedWorldFact": 0.0,
    "AnalogyTransfer": 1.0 / 7.0,
    "CrossTaskConsistency": 0.0,
}
COMPONENT_IMAGINATION_SUBTYPES = {
    "UUT": T1_IMAGINATION_SUBTYPES,
    "PropConj": T1_IMAGINATION_SUBTYPES,
    "MacGyver": T2_IMAGINATION_SUBTYPES,
    "CJST": T3_IMAGINATION_SUBTYPES,
    "GCW": T4_IMAGINATION_SUBTYPES,
    "HypoUseSpace": T5_IMAGINATION_SUBTYPES,
    "NeoCoder": T6_IMAGINATION_SUBTYPES,
    "ClosedWorldFact": (),
    "AnalogyTransfer": T8_IMAGINATION_SUBTYPES,
    "CrossTaskConsistency": (),
}
COMPONENT_HALLUCINATION_SUBTYPES = {
    "UUT": T1_HALLUCINATION_SUBTYPES,
    "PropConj": T1_HALLUCINATION_SUBTYPES,
    "MacGyver": T2_HALLUCINATION_SUBTYPES,
    "CJST": T3_HALLUCINATION_SUBTYPES,
    "GCW": T4_HALLUCINATION_SUBTYPES,
    "HypoUseSpace": T5_HALLUCINATION_SUBTYPES,
    "NeoCoder": T6_HALLUCINATION_SUBTYPES,
    "ClosedWorldFact": T7_HALLUCINATION_SUBTYPES,
    "AnalogyTransfer": T8_HALLUCINATION_SUBTYPES,
    "CrossTaskConsistency": CROSS_TASK_HALLUCINATION_SUBTYPES,
}


def clip01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _round_nested(payload):
    if isinstance(payload, dict):
        return {key: _round_nested(value) for key, value in payload.items()}
    if isinstance(payload, float):
        return round(payload, 4)
    return payload


def _atom_key(component: str, name: str) -> str:
    return f"{component}.{name}"


def _collect_atom_signals(component: str, source: Mapping[str, object], fields: Sequence[str]) -> Dict[str, float]:
    atoms: Dict[str, float] = {}
    for field in fields:
        value = optional_float(source.get(field))
        if value is not None:
            atoms[_atom_key(component, field)] = round(clip01(value), 4)
    return atoms


def _mean_atom_signals(contributions: Iterable[Optional[Mapping[str, object]]]) -> Dict[str, float]:
    values_by_atom: Dict[str, list[float]] = {}
    for contribution in contributions:
        if not isinstance(contribution, Mapping):
            continue
        atoms = contribution.get("atom_signals")
        if not isinstance(atoms, Mapping):
            continue
        for atom, value in atoms.items():
            numeric = optional_float(value)
            if numeric is None:
                continue
            values_by_atom.setdefault(str(atom), []).append(clip01(numeric))
    return {
        atom: round(sum(values) / len(values), 4)
        for atom, values in sorted(values_by_atom.items())
        if values
    }


def _with_atom_signals(payload: Mapping[str, object], atom_signals: Mapping[str, object]) -> Dict[str, object]:
    result = dict(payload)
    result["atom_signal_version"] = ATOM_SIGNAL_VERSION
    result["atom_signals"] = {
        str(atom): round(clip01(value), 4)
        for atom, value in sorted((atom_signals or {}).items())
        if optional_float(value) is not None
    }
    return _round_nested(result)


def empty_subtype_contributions(version: str = T1_TYPED_AXIS_VERSION) -> Dict[str, object]:
    return {
        "version": version,
        "atom_signal_version": ATOM_SIGNAL_VERSION,
        "atom_signals": {},
        "raw": {
            "imagination": {subtype: None for subtype in IMAGINATION_SUBTYPES},
            "hallucination": {subtype: None for subtype in HALLUCINATION_SUBTYPES},
        },
        "gated": {
            "imagination": {subtype: None for subtype in IMAGINATION_SUBTYPES},
            "hallucination": {subtype: None for subtype in HALLUCINATION_SUBTYPES},
        },
        "residual": {
            "imagination": {subtype: None for subtype in IMAGINATION_SUBTYPES},
            "hallucination": {subtype: None for subtype in HALLUCINATION_SUBTYPES},
        },
    }


def _hallucination_mean(hallucination_scores: Mapping[str, object]) -> float:
    return mean_or_none(clip01(hallucination_scores.get(key)) for key in hallucination_scores.keys()) or 0.0


def build_uut_idea_subtype_contributions(
    primitives: Mapping[str, object],
    *,
    beta_ih: float = 0.28,
    beta_hi: float = 0.10,
) -> Dict[str, object]:
    """Map one UUT idea primitive ledger to T1 subtype contributions."""

    novelty = clip01(primitives.get("novelty"))
    supported = clip01(primitives.get("supported_affordance_ratio"))
    mechanism = clip01(primitives.get("mechanism_completeness"))
    diversity = clip01(primitives.get("diversity"))
    gate = clip01(primitives.get("appropriateness_gate"))
    unsupported = clip01(primitives.get("unsupported_claim_ratio"))
    contradiction = clip01(primitives.get("contradiction_ratio"))
    extra_tool = clip01(primitives.get("extra_tool_violation"))
    cue_drift = clip01(primitives.get("cue_drift"))
    physical_drift = clip01(primitives.get("physical_drift"))

    assoc_raw = optional_float(primitives.get("imagination_contribution_v3"))
    if assoc_raw is None:
        assoc_raw = clip01(
            0.68 * novelty * supported +
            0.17 * novelty * mechanism * supported +
            0.08 * diversity +
            0.07 * novelty
        )
    else:
        assoc_raw = clip01(assoc_raw)
    assoc_gated = clip01(assoc_raw * gate)
    hallucination_raw = {
        "context": clip01((0.50 / 0.70) * unsupported + (0.20 / 0.70) * contradiction),
        "intent": extra_tool,
        "drift": clip01(0.70 * cue_drift + 0.30 * physical_drift),
        "logic": physical_drift,
    }
    h_mean = _hallucination_mean(hallucination_raw)
    assoc_residual = clip01(assoc_gated - beta_ih * h_mean)
    hallucination_residual = {
        key: clip01(value - beta_hi * assoc_gated)
        for key, value in hallucination_raw.items()
    }

    atom_signals = _collect_atom_signals("UUT", {
        "novelty": novelty,
        "supported_affordance_ratio": supported,
        "mechanism_completeness": mechanism,
        "diversity": diversity,
        "appropriateness_gate": gate,
        "unsupported_claim_ratio": unsupported,
        "contradiction_ratio": contradiction,
        "extra_tool_violation": extra_tool,
        "cue_drift": cue_drift,
        "physical_drift": physical_drift,
        "imagination_contribution_v3": assoc_raw,
    }, [
        "novelty",
        "supported_affordance_ratio",
        "mechanism_completeness",
        "diversity",
        "appropriateness_gate",
        "unsupported_claim_ratio",
        "contradiction_ratio",
        "extra_tool_violation",
        "cue_drift",
        "physical_drift",
        "imagination_contribution_v3",
    ])

    return _with_atom_signals({
        "version": T1_TYPED_AXIS_VERSION,
        "t1_assoc_version": T1_ASSOC_VERSION,
        "component": "UUT",
        "raw": {
            "imagination": {"assoc": assoc_raw},
            "hallucination": hallucination_raw,
        },
        "gated": {
            "imagination": {"assoc": assoc_gated},
            "hallucination": hallucination_raw,
        },
        "residual": {
            "imagination": {"assoc": assoc_residual},
            "hallucination": hallucination_residual,
        },
    }, atom_signals)


def build_propconj_item_subtype_contributions(
    item_score: Mapping[str, object],
    *,
    beta_ih: float = 0.20,
    beta_hi: float = 0.10,
) -> Dict[str, object]:
    """Map one PropConj item score to T1 subtype contributions."""

    novelty = clip01(item_score.get("novelty"))
    grounding = clip01(item_score.get("grounding"))
    grounding_for_i = clip01(item_score.get("grounding_for_I")) if item_score.get("grounding_for_I") is not None else grounding
    gate = clip01(item_score.get("appropriateness_gate"))
    intent_coverage = clip01(item_score.get("intent_coverage"))
    contradiction = clip01(item_score.get("contradiction"))
    unresolved = clip01(item_score.get("unresolved_entity"))
    evidence_mismatch = clip01(item_score.get("evidence_mismatch"))
    grounding_for_h = (
        clip01(item_score.get("grounding_for_H"))
        if item_score.get("grounding_for_H") is not None
        else clip01(0.70 * contradiction + 0.30 * evidence_mismatch)
    )

    assoc_raw = optional_float(item_score.get("imagination_contribution_v3"))
    if assoc_raw is None:
        assoc_raw = clip01(novelty * grounding_for_i)
    else:
        assoc_raw = clip01(assoc_raw)
    assoc_gated = clip01(assoc_raw * gate)
    hallucination_raw = {
        "context": clip01(0.625 * grounding_for_h + 0.25 * contradiction + 0.125 * evidence_mismatch),
        "intent": clip01(1.0 - intent_coverage),
        "drift": unresolved,
        "logic": contradiction,
    }
    h_mean = _hallucination_mean(hallucination_raw)
    assoc_residual = clip01(assoc_gated - beta_ih * h_mean)
    hallucination_residual = {
        key: clip01(value - beta_hi * assoc_gated)
        for key, value in hallucination_raw.items()
    }

    atom_signals = _collect_atom_signals("PropConj", {
        "novelty": novelty,
        "grounding_for_I": grounding_for_i,
        "appropriateness_gate": gate,
        "intent_coverage": intent_coverage,
        "grounding_for_H": grounding_for_h,
        "contradiction": contradiction,
        "unresolved_entity": unresolved,
        "evidence_mismatch": evidence_mismatch,
        "imagination_contribution_v3": assoc_raw,
    }, [
        "novelty",
        "grounding_for_I",
        "appropriateness_gate",
        "intent_coverage",
        "grounding_for_H",
        "contradiction",
        "unresolved_entity",
        "evidence_mismatch",
        "imagination_contribution_v3",
    ])

    return _with_atom_signals({
        "version": T1_TYPED_AXIS_VERSION,
        "t1_assoc_version": T1_ASSOC_VERSION,
        "component": "PropConj",
        "raw": {
            "imagination": {"assoc": assoc_raw},
            "hallucination": hallucination_raw,
        },
        "gated": {
            "imagination": {"assoc": assoc_gated},
            "hallucination": hallucination_raw,
        },
        "residual": {
            "imagination": {"assoc": assoc_residual},
            "hallucination": hallucination_residual,
        },
    }, atom_signals)


def build_macgyver_task_subtype_contributions(
    task_score: Mapping[str, object],
    *,
    beta_ih: float = 0.00,
    beta_hi: float = 0.90,
) -> Dict[str, object]:
    """Map one MacGyver task score to T2 subtype contributions."""

    primitive_means = task_score.get("primitive_means") or {}
    boundary_record = task_score.get("boundary_record") or {}
    imagination_raw = 0.0 if task_score.get("imagination_excluded") else clip01(task_score.get("imagination_raw"))
    boundary_h = clip01(boundary_record.get("boundary_hallucination"))
    feasibility = clip01(primitive_means.get("F_feasibility"))
    mechanism = clip01(primitive_means.get("M_mechanism_completeness"))
    novelty = clip01(primitive_means.get("N_plan_novelty"))
    scored_plans = int(task_score.get("scored_plans") or 0)
    constraint_gate = clip01(
        0.45 * feasibility +
        0.25 * mechanism +
        0.20 * (1.0 - boundary_h) +
        0.10 * novelty
    )
    if boundary_h >= 0.70 or task_score.get("imagination_excluded"):
        constraint_gate = min(constraint_gate, 0.20)

    logic_raw = 0.0
    if scored_plans > 0:
        logic_raw = clip01(max(
            clip01(primitive_means.get("C_contradiction_rate")),
            clip01(primitive_means.get("A_unsupported_success_assertion")),
            0.70 * clip01(1.0 - clip01(primitive_means.get("P_physical_affordance_support"))),
        ))
    if clip01(boundary_record.get("false_acceptance")) >= 1.0 and boundary_record.get("expected_response_mode") == "unsolvable":
        logic_raw = max(logic_raw, 0.80)
    plan_intent = 0.0
    if scored_plans > 0:
        plan_intent = max(
            clip01(primitive_means.get("U_unavailable_entity_rate")),
            clip01(1.0 - clip01(primitive_means.get("K_constraint_satisfaction"))),
            clip01(boundary_record.get("distractor_tool_violation")),
        )
    intent_raw = clip01(max(plan_intent, clip01(boundary_record.get("clarification_miss"))))
    hallucination_raw = {
        "logic": logic_raw,
        "boundary": boundary_h,
        "intent": intent_raw,
    }
    constraint_raw = clip01(imagination_raw)
    constraint_gated = clip01(constraint_raw * constraint_gate)
    h_mean = _hallucination_mean(hallucination_raw)
    constraint_residual = clip01(constraint_gated - beta_ih * h_mean)
    hallucination_residual = {
        key: clip01(value - beta_hi * constraint_gated)
        for key, value in hallucination_raw.items()
    }

    atom_signals = _collect_atom_signals("MacGyver", {
        "imagination_raw": constraint_raw,
        "constraint_gate": constraint_gate,
        "boundary_hallucination": boundary_h,
        **primitive_means,
    }, [
        "imagination_raw",
        "constraint_gate",
        "boundary_hallucination",
        "F_feasibility",
        "M_mechanism_completeness",
        "N_plan_novelty",
        "P_physical_affordance_support",
        "K_constraint_satisfaction",
        "C_contradiction_rate",
        "A_unsupported_success_assertion",
        "U_unavailable_entity_rate",
    ])

    return _with_atom_signals({
        "version": T2_TYPED_AXIS_VERSION,
        "component": "MacGyver",
        "task_subtype": task_score.get("task_subtype"),
        "raw": {
            "imagination": {"constraint": constraint_raw},
            "hallucination": hallucination_raw,
        },
        "gated": {
            "imagination": {"constraint": constraint_gated},
            "hallucination": hallucination_raw,
        },
        "residual": {
            "imagination": {"constraint": constraint_residual},
            "hallucination": hallucination_residual,
        },
    }, atom_signals)


def build_cjst_task_subtype_contributions(
    task_score: Mapping[str, object],
    *,
    beta_ih: float = 0.80,
    beta_hi: float = 0.12,
) -> Dict[str, object]:
    """Map one CJST task score to T3 subtype contributions."""

    primitive_means = task_score.get("primitive_means") or {}
    cf_raw = clip01(task_score.get("imagination_raw"))
    premise_relevance = clip01(primitive_means.get("premise_relevance"))
    premise_lock = clip01(primitive_means.get("premise_lock"))
    unsupported = clip01(primitive_means.get("unsupported_extra_claim_rate"))
    contradiction = clip01(primitive_means.get("contradiction"))
    world_consistency = primitive_means.get("world_consistency")
    world_consistency = clip01(world_consistency) if world_consistency is not None else clip01(
        0.50 * premise_relevance + 0.50 * premise_lock
    )
    formal_h = clip01(primitive_means.get("formal_hallucination_raw"))
    intervention_used = primitive_means.get("intervention_used")
    intervention_used = clip01(intervention_used) if intervention_used is not None else premise_relevance
    causal_edge_support = primitive_means.get("causal_edge_support")
    causal_edge_support = clip01(causal_edge_support) if causal_edge_support is not None else clip01(
        primitive_means.get("mechanism_completeness")
    )
    protected_violation = clip01(primitive_means.get("protected_variable_violation"))
    forbidden_update = clip01(primitive_means.get("forbidden_update_rate"))
    extra_miracle = clip01(primitive_means.get("extra_miracle_rate"))
    hard_gate = clip01(primitive_means.get("hard_gate"))

    consistency_gate = clip01(
        0.50 * world_consistency +
        0.20 * premise_lock +
        0.15 * premise_relevance +
        0.15 * (1.0 - formal_h)
    )
    cf_gated = clip01(cf_raw * consistency_gate)
    hallucination_raw = {
        "logic": clip01(max(
            contradiction,
            clip01(1.15 * protected_violation),
            extra_miracle,
            forbidden_update,
            0.75 * (1.0 - causal_edge_support),
        )),
        "context": clip01(max(
            unsupported,
            0.70 * (1.0 - premise_relevance),
            0.55 * (1.0 - intervention_used),
            0.60 * formal_h,
        )),
        "drift": clip01(max(
            0.70 * (1.0 - premise_lock),
            clip01(0.70 * forbidden_update + 0.30 * formal_h),
            extra_miracle,
            0.55 * (1.0 - world_consistency),
        )),
    }
    h_mean = _hallucination_mean(hallucination_raw)
    cf_residual = clip01(cf_gated - beta_ih * h_mean)
    hallucination_residual = {
        key: clip01(value - beta_hi * cf_gated)
        for key, value in hallucination_raw.items()
    }

    atom_signals = _collect_atom_signals("CJST", {
        "imagination_raw": cf_raw,
        "consistency_gate": consistency_gate,
        "hard_gate": primitive_means.get("hard_gate"),
        "premise_relevance": premise_relevance,
        "premise_lock": premise_lock,
        "unsupported_extra_claim_rate": unsupported,
        "contradiction": contradiction,
        "world_consistency": world_consistency,
        "formal_hallucination_raw": formal_h,
        "intervention_used": intervention_used,
        "causal_edge_support": causal_edge_support,
        "protected_variable_violation": protected_violation,
        "forbidden_update_rate": forbidden_update,
        "extra_miracle_rate": extra_miracle,
    }, [
        "imagination_raw",
        "consistency_gate",
        "hard_gate",
        "premise_relevance",
        "premise_lock",
        "unsupported_extra_claim_rate",
        "contradiction",
        "world_consistency",
        "formal_hallucination_raw",
        "intervention_used",
        "causal_edge_support",
        "protected_variable_violation",
        "forbidden_update_rate",
        "extra_miracle_rate",
    ])

    return _with_atom_signals({
        "version": T3_TYPED_AXIS_VERSION,
        "component": "CJST",
        "raw": {
            "imagination": {"cf": cf_raw},
            "hallucination": hallucination_raw,
        },
        "gated": {
            "imagination": {"cf": cf_gated},
            "hallucination": hallucination_raw,
        },
        "residual": {
            "imagination": {"cf": cf_residual},
            "hallucination": hallucination_residual,
        },
    }, atom_signals)


def build_gcw_task_subtype_contributions(
    task_score: Mapping[str, object],
    *,
    beta_ih: float = 0.80,
    beta_hi: float = 0.12,
) -> Dict[str, object]:
    """Map one GCW task score to T4 subtype contributions."""

    primitive_means = task_score.get("primitive_means") or {}
    narrative_raw = clip01(task_score.get("imagination_raw"))
    support_gate = primitive_means.get("narrative_grounding")
    if support_gate is None:
        support_gate = primitive_means.get("support_gate")
    support_gate = clip01(support_gate) if support_gate is not None else clip01(
        0.55 * clip01(primitive_means.get("claim_support_precision")) +
        0.25 * clip01(primitive_means.get("claim_support_recall")) +
        0.20 * (1.0 - clip01(primitive_means.get("citation_mismatch_rate")))
    )
    constraint_gate = primitive_means.get("constraint_gate")
    constraint_gate = clip01(constraint_gate) if constraint_gate is not None else clip01(
        1.0 - max(
            clip01(primitive_means.get("hard_constraint_violation_rate")),
            clip01(primitive_means.get("hard_no_drift_violation")),
            clip01(primitive_means.get("forbidden_motif_rate")),
            0.50 * clip01(primitive_means.get("missing_required_fact_rate")),
        )
    )
    narrative_gated = task_score.get("imagination_gated")
    narrative_gated = clip01(narrative_gated) if narrative_gated is not None else clip01(
        narrative_raw * support_gate * constraint_gate
    )

    unsupported_claim = clip01(primitive_means.get("unsupported_claim_rate"))
    unsupported_span = clip01(primitive_means.get("unsupported_span_rate"))
    unsupported_entity = clip01(primitive_means.get("entity_drift_rate"))
    contradicted = clip01(primitive_means.get("contradicted_claim_rate"))
    contradiction = clip01(primitive_means.get("contradiction_rate"))
    missing_required = clip01(primitive_means.get("missing_required_fact_rate"))
    claim_precision_failure = clip01(1.0 - clip01(primitive_means.get("claim_support_precision")))
    citation_mismatch = clip01(primitive_means.get("citation_mismatch_rate"))
    missing_evidence = clip01(primitive_means.get("claim_without_evidence_rate"))
    hard_constraint = clip01(primitive_means.get("hard_constraint_violation_rate"))
    no_drift = clip01(primitive_means.get("hard_no_drift_violation"))
    entity_persistence = clip01(primitive_means.get("entity_persistence_failure"))
    forbidden_motif = clip01(primitive_means.get("forbidden_motif_rate"))
    claim_evidence_mismatch = (
        clip01(primitive_means.get("claim_evidence_mismatch"))
        if primitive_means.get("claim_evidence_mismatch") is not None
        else clip01(max(unsupported_span, unsupported_claim, missing_evidence, claim_precision_failure))
    )

    hallucination_raw = {
        "detail": clip01(max(claim_evidence_mismatch, unsupported_entity)),
        "context": clip01(max(
            contradiction,
            contradicted,
            missing_required,
            claim_precision_failure,
        )),
        "drift": clip01(max(
            hard_constraint,
            no_drift,
            entity_persistence,
            forbidden_motif,
            unsupported_entity,
        )),
        "citation": clip01(max(citation_mismatch, missing_evidence)),
    }
    h_mean = _hallucination_mean(hallucination_raw)
    narrative_residual = clip01(narrative_gated - beta_ih * h_mean)
    hallucination_residual = {
        key: clip01(value - beta_hi * narrative_gated)
        for key, value in hallucination_raw.items()
    }

    atom_signals = _collect_atom_signals("GCW", {
        "imagination_raw": narrative_raw,
        "narrative_grounding": support_gate,
        "constraint_gate": constraint_gate,
        "claim_evidence_mismatch": claim_evidence_mismatch,
        "unsupported_claim_rate": unsupported_claim,
        "unsupported_span_rate": unsupported_span,
        "entity_drift_rate": unsupported_entity,
        "contradicted_claim_rate": contradicted,
        "contradiction_rate": contradiction,
        "missing_required_fact_rate": missing_required,
        "citation_mismatch_rate": citation_mismatch,
        "claim_without_evidence_rate": missing_evidence,
        "hard_constraint_violation_rate": hard_constraint,
        "hard_no_drift_violation": no_drift,
        "entity_persistence_failure": entity_persistence,
        "forbidden_motif_rate": forbidden_motif,
    }, [
        "imagination_raw",
        "narrative_grounding",
        "constraint_gate",
        "claim_evidence_mismatch",
        "unsupported_claim_rate",
        "unsupported_span_rate",
        "entity_drift_rate",
        "contradicted_claim_rate",
        "contradiction_rate",
        "missing_required_fact_rate",
        "citation_mismatch_rate",
        "claim_without_evidence_rate",
        "hard_constraint_violation_rate",
        "hard_no_drift_violation",
        "entity_persistence_failure",
        "forbidden_motif_rate",
    ])

    return _with_atom_signals({
        "version": T4_TYPED_AXIS_VERSION,
        "component": "GCW",
        "constraint_level": task_score.get("constraint_level"),
        "raw": {
            "imagination": {"narrative": narrative_raw},
            "hallucination": hallucination_raw,
        },
        "gated": {
            "imagination": {"narrative": narrative_gated},
            "hallucination": hallucination_raw,
        },
        "residual": {
            "imagination": {"narrative": narrative_residual},
            "hallucination": hallucination_residual,
        },
    }, atom_signals)


def build_hypospace_task_subtype_contributions(
    task_score: Mapping[str, object],
    *,
    beta_ih: float = 1.00,
    beta_hi: float = 0.10,
) -> Dict[str, object]:
    """Map one HypoUseSpace task score to T5 subtype contributions."""

    primitive_means = task_score.get("primitive_means") or {}
    if task_score.get("imagination_excluded"):
        hypothesis_raw = 0.0
    else:
        hypothesis_raw = clip01(task_score.get("imagination_raw"))
    evidence_gate = primitive_means.get("evidence_support_gate")
    evidence_gate = clip01(evidence_gate) if evidence_gate is not None else clip01(
        0.45 * clip01(primitive_means.get("claim_support_precision")) +
        0.20 * clip01(primitive_means.get("claim_support_recall")) +
        0.20 * (1.0 - clip01(primitive_means.get("citation_mismatch_rate"))) +
        0.15 * clip01(primitive_means.get("evidence_support"))
    )
    boundary_gate = primitive_means.get("boundary_gate")
    boundary_gate = clip01(boundary_gate) if boundary_gate is not None else clip01(
        1.0 - max(
            clip01(primitive_means.get("unavailable_entity")),
            clip01(primitive_means.get("constraint_or_observation_violation")),
            clip01(primitive_means.get("evidence_boundary_violation")),
        )
    )
    hypothesis_gated = task_score.get("imagination_gated")
    hypothesis_gated = clip01(hypothesis_gated) if hypothesis_gated is not None else clip01(
        hypothesis_raw * evidence_gate * boundary_gate
    )

    unsupported_mechanism = clip01(primitive_means.get("unsupported_affordance_or_mechanism"))
    contradiction = clip01(primitive_means.get("explicit_contradiction_or_forbidden_foil"))
    contradicted_claim = clip01(primitive_means.get("contradicted_claim_rate"))
    false_feasibility = clip01(primitive_means.get("false_feasibility_claim"))
    unsupported_span = clip01(primitive_means.get("unsupported_span_rate"))
    citation_mismatch = clip01(primitive_means.get("citation_mismatch_rate"))
    unknown_evidence = clip01(primitive_means.get("unknown_evidence_rate"))
    missing_citation = clip01(max(
        primitive_means.get("missing_required_citation") or 0.0,
        primitive_means.get("claim_without_evidence_rate") or 0.0,
    ))
    unavailable_entity = clip01(primitive_means.get("unavailable_entity"))
    constraint_violation = clip01(primitive_means.get("constraint_or_observation_violation"))
    evidence_boundary = clip01(primitive_means.get("evidence_boundary_violation"))
    no_valid_false_acceptance = 1.0 if (
        task_score.get("no_valid_hypothesis")
        and not task_score.get("no_valid_correct")
        and int(task_score.get("scored_hypotheses") or 0) > 0
    ) else 0.0

    hallucination_raw = {
        "fact": clip01(max(
            unsupported_mechanism,
            contradiction,
            contradicted_claim,
            false_feasibility,
            0.70 * unsupported_span,
        )),
        "citation": clip01(max(citation_mismatch, unknown_evidence, missing_citation)),
        "boundary": clip01(max(
            unavailable_entity,
            constraint_violation,
            evidence_boundary,
            no_valid_false_acceptance,
        )),
    }
    h_mean = _hallucination_mean(hallucination_raw)
    hypothesis_residual = clip01(hypothesis_gated - beta_ih * h_mean)
    hallucination_residual = {
        key: clip01(value - beta_hi * hypothesis_gated)
        for key, value in hallucination_raw.items()
    }

    atom_signals = _collect_atom_signals("HypoUseSpace", {
        "imagination_raw": hypothesis_raw,
        "evidence_support_gate": evidence_gate,
        "boundary_gate": boundary_gate,
        "unsupported_affordance_or_mechanism": unsupported_mechanism,
        "explicit_contradiction_or_forbidden_foil": contradiction,
        "contradicted_claim_rate": contradicted_claim,
        "false_feasibility_claim": false_feasibility,
        "unsupported_span_rate": unsupported_span,
        "citation_mismatch_rate": citation_mismatch,
        "unknown_evidence_rate": unknown_evidence,
        "claim_without_evidence_rate": missing_citation,
        "unavailable_entity": unavailable_entity,
        "constraint_or_observation_violation": constraint_violation,
        "evidence_boundary_violation": evidence_boundary,
    }, [
        "imagination_raw",
        "evidence_support_gate",
        "boundary_gate",
        "unsupported_affordance_or_mechanism",
        "explicit_contradiction_or_forbidden_foil",
        "contradicted_claim_rate",
        "false_feasibility_claim",
        "unsupported_span_rate",
        "citation_mismatch_rate",
        "unknown_evidence_rate",
        "claim_without_evidence_rate",
        "unavailable_entity",
        "constraint_or_observation_violation",
        "evidence_boundary_violation",
    ])

    return _with_atom_signals({
        "version": T5_TYPED_AXIS_VERSION,
        "component": "HypoUseSpace",
        "task_subtype": task_score.get("task_subtype"),
        "raw": {
            "imagination": {"hypothesis": hypothesis_raw},
            "hallucination": hallucination_raw,
        },
        "gated": {
            "imagination": {"hypothesis": hypothesis_gated},
            "hallucination": hallucination_raw,
        },
        "residual": {
            "imagination": {"hypothesis": hypothesis_residual},
            "hallucination": hallucination_residual,
        },
    }, atom_signals)


def build_neocoder_task_subtype_contributions(
    task_score: Mapping[str, object],
    *,
    beta_ih: float = 0.25,
    beta_hi: float = 0.10,
) -> Dict[str, object]:
    """Map one NeoCoder task score to optional T6 subtype contributions."""

    primitive_means = task_score.get("primitive_means") or {}
    code_raw = clip01(task_score.get("imagination_raw"))
    code_gated = task_score.get("imagination_gated")
    code_gated = clip01(code_gated) if code_gated is not None else clip01(
        code_raw *
        clip01(primitive_means.get("pass_rate")) *
        clip01(primitive_means.get("constraint_following"))
    )

    h_logic = primitive_means.get("H_logic")
    h_intent = primitive_means.get("H_intent")
    h_fact = primitive_means.get("H_fact")
    hallucination_raw = {
        "logic": clip01(h_logic) if h_logic is not None else clip01(max(
            1.0 - clip01(primitive_means.get("pass_rate")),
            clip01(primitive_means.get("syntax_error")),
            clip01(primitive_means.get("runtime_error_rate")),
            clip01(primitive_means.get("timeout_rate")),
        )),
        "intent": clip01(h_intent) if h_intent is not None else clip01(max(
            clip01(primitive_means.get("banned_technique_violation_rate")),
            clip01(primitive_means.get("import_violation")),
            clip01(primitive_means.get("dangerous_call_violation")),
            clip01(primitive_means.get("missing_entrypoint")),
            0.25 * clip01(primitive_means.get("missing_complexity_claim")),
        )),
        "fact": clip01(h_fact) if h_fact is not None else clip01(max(
            clip01(primitive_means.get("unsupported_complexity_claim")),
            clip01(primitive_means.get("hallucinated_import_rate")),
            clip01(primitive_means.get("unsupported_api_call_rate")),
        )),
    }
    h_mean = _hallucination_mean(hallucination_raw)
    code_residual = clip01(code_gated - beta_ih * h_mean)
    hallucination_residual = {
        key: clip01(value - beta_hi * code_gated)
        for key, value in hallucination_raw.items()
    }

    atom_signals = _collect_atom_signals("NeoCoder", {
        "imagination_raw": code_raw,
        **primitive_means,
    }, [
        "imagination_raw",
        "pass_rate",
        "constraint_following",
        "syntax_error",
        "runtime_error_rate",
        "timeout_rate",
        "banned_technique_violation_rate",
        "import_violation",
        "dangerous_call_violation",
        "missing_entrypoint",
        "unsupported_complexity_claim",
        "hallucinated_import_rate",
        "unsupported_api_call_rate",
    ])

    return _with_atom_signals({
        "version": T6_TYPED_AXIS_VERSION,
        "component": "NeoCoder",
        "base_task_id": task_score.get("base_task_id"),
        "denial_state": task_score.get("denial_state"),
        "raw": {
            "imagination": {"code": code_raw},
            "hallucination": hallucination_raw,
        },
        "gated": {
            "imagination": {"code": code_gated},
            "hallucination": hallucination_raw,
        },
        "residual": {
            "imagination": {"code": code_residual},
            "hallucination": hallucination_residual,
        },
    }, atom_signals)


def build_closed_world_fact_calibration_contributions(
    task_score: Mapping[str, object],
) -> Dict[str, object]:
    """Map one ClosedWorldFact score to hallucination subtypes."""

    primitive_means = task_score.get("primitive_means") or {}
    h_fact = primitive_means.get("H_fact")
    h_logic = primitive_means.get("H_logic")
    h_boundary = primitive_means.get("H_boundary")
    hallucination_raw = {
        "fact": clip01(h_fact) if h_fact is not None else clip01(max(
            clip01(primitive_means.get("wrong_answer")),
            clip01(primitive_means.get("contradicted_fact")),
            clip01(primitive_means.get("forbidden_answer_hit")),
            clip01(primitive_means.get("unsupported_fact_claim")),
        )),
        "logic": clip01(h_logic) if h_logic is not None else clip01(max(
            clip01(primitive_means.get("rationale_keyword_failure")),
            clip01(1.0 - clip01(primitive_means.get("evidence_recall"))),
            clip01(primitive_means.get("chain_order_failure")),
            clip01(primitive_means.get("comparison_or_set_failure")),
        )),
        "boundary": clip01(h_boundary) if h_boundary is not None else clip01(max(
            clip01(primitive_means.get("false_presupposition_acceptance")),
            clip01(primitive_means.get("unanswerable_false_answer")),
            clip01(primitive_means.get("unknown_entity_overclaim")),
            clip01(primitive_means.get("closed_boundary_failure")),
        )),
    }

    atom_signals = _collect_atom_signals("ClosedWorldFact", primitive_means, [
        "wrong_answer",
        "contradicted_fact",
        "forbidden_answer_hit",
        "unsupported_fact_claim",
        "rationale_keyword_failure",
        "evidence_recall",
        "chain_order_failure",
        "comparison_or_set_failure",
        "false_presupposition_acceptance",
        "unanswerable_false_answer",
        "unknown_entity_overclaim",
        "closed_boundary_failure",
    ])

    return _with_atom_signals({
        "version": "typed_axis_t7",
        "component": "ClosedWorldFact",
        "question_type": task_score.get("question_type"),
        "raw": {
            "imagination": {},
            "hallucination": hallucination_raw,
        },
        "gated": {
            "imagination": {},
            "hallucination": hallucination_raw,
        },
        "residual": {
            "imagination": {},
            "hallucination": hallucination_raw,
        },
    }, atom_signals)


def build_cross_task_fact_consistency_contributions(
    axis_score: Mapping[str, object],
) -> Dict[str, object]:
    """Map cross-task fact consistency to its hallucination subtype."""

    h_consistency = axis_score.get("hallucination")
    if h_consistency is None:
        h_consistency = axis_score.get("hallucination_raw")
    h_consistency = clip01(h_consistency) if h_consistency is not None else None
    hallucination_raw = {"consistency": h_consistency}

    primitive_means = axis_score.get("primitive_means") if isinstance(axis_score.get("primitive_means"), Mapping) else {}
    atom_signals = _collect_atom_signals("CrossTaskConsistency", {
        "pair_inconsistency": h_consistency,
        **primitive_means,
    }, [
        "pair_inconsistency",
        "semantic_conflict",
        "polarity_conflict",
        "support_conflict",
        "contradiction_conflict",
        "citation_issue",
    ])

    return _with_atom_signals({
        "version": "typed_axis_cross_task_consistency_v1",
        "component": "CrossTaskConsistency",
        "raw": {
            "imagination": {},
            "hallucination": hallucination_raw,
        },
        "gated": {
            "imagination": {},
            "hallucination": hallucination_raw,
        },
        "residual": {
            "imagination": {},
            "hallucination": hallucination_raw,
        },
    }, atom_signals)


def build_analogy_transfer_challenge_contributions(
    task_score: Mapping[str, object],
    *,
    beta_ih: float = 0.90,
    beta_hi: float = 0.30,
) -> Dict[str, object]:
    """Map one AnalogyTransfer challenge score to T8 subtype contributions."""

    primitive_means = task_score.get("primitive_means") or {}
    analogy_raw = clip01(task_score.get("imagination_raw"))
    analogy_gated = clip01(
        task_score.get("imagination_gated")
        if task_score.get("imagination_gated") is not None
        else analogy_raw
    )
    h_false_transfer = primitive_means.get("H_false_transfer")
    h_fact = primitive_means.get("H_fact")
    h_logic = primitive_means.get("H_logic")
    h_context = primitive_means.get("H_context")
    hallucination_raw = {
        "false_transfer": clip01(h_false_transfer) if h_false_transfer is not None else clip01(max(
            clip01(primitive_means.get("false_transfer_rate")),
            clip01(primitive_means.get("unsupported_inference_rate")),
        )),
        "fact": clip01(h_fact) if h_fact is not None else clip01(max(
            clip01(primitive_means.get("invented_fact_rate")),
            clip01(primitive_means.get("unknown_evidence_rate")),
            clip01(1.0 - clip01(primitive_means.get("source_fact_accuracy"))),
            clip01(1.0 - clip01(primitive_means.get("target_fact_accuracy"))),
        )),
        "logic": clip01(h_logic) if h_logic is not None else clip01(max(
            clip01(1.0 - clip01(primitive_means.get("relational_consistency"))),
            clip01(primitive_means.get("wrong_mapping_direction_rate")),
            clip01(primitive_means.get("dimension_inconsistency_rate")),
        )),
        "context": clip01(h_context) if h_context is not None else clip01(max(
            clip01(1.0 - clip01(primitive_means.get("limit_awareness"))),
            clip01(primitive_means.get("surface_stereotype_rate")),
            clip01(primitive_means.get("boundary_overclaim_rate")),
        )),
    }
    h_mean = _hallucination_mean(hallucination_raw)
    analogy_residual = clip01(analogy_gated - beta_ih * h_mean)
    hallucination_residual = {
        key: clip01(value - beta_hi * analogy_gated)
        for key, value in hallucination_raw.items()
    }

    atom_signals = _collect_atom_signals("AnalogyTransfer", {
        "imagination_raw": analogy_raw,
        **primitive_means,
    }, [
        "imagination_raw",
        "mapping_quality",
        "structural_match_gmean",
        "evidence_grounding",
        "relational_consistency",
        "false_transfer_rate",
        "unsupported_inference_rate",
        "invented_fact_rate",
        "unknown_evidence_rate",
        "source_fact_accuracy",
        "target_fact_accuracy",
        "wrong_mapping_direction_rate",
        "dimension_inconsistency_rate",
        "limit_awareness",
        "surface_stereotype_rate",
        "boundary_overclaim_rate",
    ])

    return _with_atom_signals({
        "version": T8_TYPED_AXIS_VERSION,
        "component": "AnalogyTransfer",
        "cluster_id": task_score.get("cluster_id"),
        "variant": task_score.get("variant"),
        "raw": {
            "imagination": {"analogy": analogy_raw},
            "hallucination": hallucination_raw,
        },
        "gated": {
            "imagination": {"analogy": analogy_gated},
            "hallucination": hallucination_raw,
        },
        "residual": {
            "imagination": {"analogy": analogy_residual},
            "hallucination": hallucination_residual,
        },
    }, atom_signals)


def mean_subtype_contributions(contributions: Iterable[Optional[Mapping[str, object]]]) -> Dict[str, object]:
    """Average a collection of subtype contribution ledgers."""

    usable = [contrib for contrib in contributions if isinstance(contrib, Mapping)]
    if not usable:
        return empty_subtype_contributions(TYPED_AXIS_VERSION)

    result = empty_subtype_contributions(TYPED_AXIS_VERSION)
    result["atom_signals"] = _mean_atom_signals(usable)
    for view in ("raw", "gated", "residual"):
        for subtype in IMAGINATION_SUBTYPES:
            value = mean_or_none(
                optional_float(((contrib.get(view) or {}).get("imagination") or {}).get(subtype))
                for contrib in usable
            )
            result[view]["imagination"][subtype] = round(value, 4) if value is not None else None
        for subtype in HALLUCINATION_SUBTYPES:
            value = mean_or_none(
                optional_float(((contrib.get(view) or {}).get("hallucination") or {}).get(subtype))
                for contrib in usable
            )
            result[view]["hallucination"][subtype] = round(value, 4) if value is not None else None
    return result


def _component_weights(component_names: Iterable[str]) -> Dict[str, float]:
    usable = {
        component: TYPED_COMPONENT_BASE_WEIGHTS[component]
        for component in component_names
        if component in TYPED_COMPONENT_BASE_WEIGHTS
    }
    total = sum(usable.values())
    if total <= 0.0:
        return {component: 0.0 for component in usable}
    return {component: weight / total for component, weight in usable.items()}


def _component_subtypes(component: str, axis: str) -> tuple[str, ...]:
    if axis == "imagination":
        return tuple(COMPONENT_IMAGINATION_SUBTYPES.get(component, ()))
    if axis == "hallucination":
        return tuple(COMPONENT_HALLUCINATION_SUBTYPES.get(component, ()))
    return ()


def aggregate_t1_subtype_scores(
    component_contributions: Mapping[str, Optional[Mapping[str, object]]],
    *,
    component_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    """Aggregate UUT and PropConj subtype ledgers into the report schema."""

    usable_components = {
        component: contribution
        for component, contribution in (component_contributions or {}).items()
        if component in TYPED_COMPONENT_BASE_WEIGHTS and isinstance(contribution, Mapping)
    }
    weights = dict(component_weights or {})
    if not weights:
        weights = _component_weights(usable_components.keys())
    else:
        weights = {
            component: clip01(weight)
            for component, weight in weights.items()
            if component in usable_components
        }
        total = sum(weights.values())
        weights = {component: weight / total for component, weight in weights.items()} if total > 0.0 else {}

    result = empty_subtype_contributions(TYPED_AXIS_VERSION)
    result.update({
        "version": TYPED_AXIS_VERSION,
        "scope": "T1 Anchored Divergent Association + T2 Constrained Problem Solving + T3 Counterfactual World Modeling + T4 Grounded Creative Writing + T5 Evidence-Constrained Ideation + T6 Executable Code Creativity + optional T7 hallucination calibration + T8 Analogy False Transfer",
        "component_weights": {component: round(weight, 6) for component, weight in sorted(weights.items())},
        "atom_signal_version": ATOM_SIGNAL_VERSION,
        "atom_signals": _mean_atom_signals(usable_components.values()),
        "component_contributions": {
            component: _round_nested(dict(contribution))
            for component, contribution in sorted(usable_components.items())
        },
    })

    def _apply_h_only_consistency_diagnostic() -> None:
        for view_name in ("raw", "gated", "residual"):
            consistency_values = [
                optional_float(((contribution.get(view_name) or {}).get("hallucination") or {}).get("consistency"))
                for component, contribution in usable_components.items()
                if component == "CrossTaskConsistency"
            ]
            consistency_mean = mean_or_none(value for value in consistency_values if value is not None)
            if consistency_mean is not None:
                result[view_name]["hallucination"]["consistency"] = round(clip01(consistency_mean), 4)

    if not weights:
        _apply_h_only_consistency_diagnostic()
        return result

    for view in ("raw", "gated", "residual"):
        for subtype in IMAGINATION_SUBTYPES:
            total = 0.0
            has_value = False
            for component in weights:
                if subtype not in _component_subtypes(component, "imagination"):
                    continue
                value = optional_float(
                    ((usable_components[component].get(view) or {}).get("imagination") or {}).get(subtype)
                )
                if value is None:
                    continue
                total += weights[component] * clip01(value)
                has_value = True
            result[view]["imagination"][subtype] = round(total, 4) if has_value else None
        for subtype in HALLUCINATION_SUBTYPES:
            total = 0.0
            has_value = False
            for component in weights:
                if subtype not in _component_subtypes(component, "hallucination"):
                    continue
                value = optional_float(
                    ((usable_components[component].get(view) or {}).get("hallucination") or {}).get(subtype)
                )
                if value is None:
                    continue
                total += weights[component] * clip01(value)
                has_value = True
            result[view]["hallucination"][subtype] = round(total, 4) if has_value else None
    _apply_h_only_consistency_diagnostic()
    return result


def aggregate_repeat_subtype_scores(repeat_reports: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    """Average repeat-level `overall_summary.axes.subtype_scores` ledgers."""

    ledgers = []
    for report in repeat_reports:
        axes = ((report.get("overall_summary") or {}).get("axes") or {})
        ledger = axes.get("subtype_scores")
        if isinstance(ledger, Mapping):
            ledgers.append(ledger)
    if not ledgers:
        return aggregate_t1_subtype_scores({})

    component_names = set()
    for ledger in ledgers:
        contributions = ledger.get("component_contributions")
        if isinstance(contributions, Mapping):
            component_names.update(contributions.keys())

    averaged_components = {}
    for component in sorted(component_names):
        component_ledgers = []
        for ledger in ledgers:
            contributions = ledger.get("component_contributions")
            if isinstance(contributions, Mapping) and isinstance(contributions.get(component), Mapping):
                component_ledgers.append(contributions[component])
        if component_ledgers:
            averaged_components[component] = mean_subtype_contributions(component_ledgers)
    return aggregate_t1_subtype_scores(averaged_components)
