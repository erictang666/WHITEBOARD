
import copy
import math
from collections import Counter

from ttct_zero_originality import (
    analyze_zero_originality,
    get_zero_originality_runtime_context,
)
from common_answer_bank import (
    build_common_answer_bank_context,
    blend_creative_novelty,
    get_common_answer_bank_hybrid_formula,
    has_common_answer_reference_bank,
    score_common_answer_bank_novelty,
)

from groundedness_scorer import WHITE_BOX_GROUNDEDNESS_VERSION
from propconj_scorer import (
    PropConjScorer,
    aggregate_propconj_model_axes,
    compute_propconj_diversity,
    compute_propconj_task_scores,
)
from macgyver_scorer import (
    MACGYVER_V3_CALIBRATION_POLICY,
    MACGYVER_V3_RUNTIME_SCORING_POLICY,
    MacGyverScorer,
    aggregate_macgyver_boundary_diagnostics,
    aggregate_macgyver_model_axes,
    get_macgyver_common_plan_bank_coverage,
)
from counterfactual_scorer import (
    CJST_V3_CALIBRATION_POLICY,
    CJST_V3_RUNTIME_SCORING_POLICY,
    CounterfactualScorer,
    aggregate_cjst_model_axes,
    get_cjst_common_consequence_bank_coverage,
)
from grounded_creative_writing_scorer import (
    GCW_V3_CALIBRATION_POLICY,
    GCW_V3_RUNTIME_SCORING_POLICY,
    GroundedCreativeWritingScorer,
    aggregate_gcw_model_axes,
    get_gcw_common_story_bank_coverage,
    get_gcw_entity_alias_coverage,
)
from hypospace_scorer import (
    HYPOUSESPACE_V3_CALIBRATION_POLICY,
    HYPOUSESPACE_V3_RUNTIME_SCORING_POLICY,
    HypoUseSpaceScorer,
    aggregate_hypospace_boundary_diagnostics,
    aggregate_hypospace_model_axes,
    get_hypospace_common_hypothesis_bank_coverage,
    get_hypospace_valid_match_alias_coverage,
)
from neocoder_scorer import (
    NEOCODER_V3_CALIBRATION_POLICY,
    NEOCODER_V3_COMMON_SOLUTION_BANK_VERSION,
    NEOCODER_V3_RUNTIME_SCORING_POLICY,
    NEOCODER_V3_TASK_OVERLAY_VERSION,
    NEOCODER_V3_TECHNIQUE_ALIAS_VERSION,
    NEOCODER_V3_TEST_VISIBILITY_POLICY,
    NeoCoderScorer,
    aggregate_neocoder_model_axes,
    get_neocoder_common_solution_bank_coverage,
    get_neocoder_task_overlay_coverage,
    get_neocoder_technique_alias_coverage,
)
from closed_world_fact_scorer import (
    CLOSED_WORLD_FACT_VERSION,
    ClosedWorldFactScorer,
    aggregate_closed_world_fact_calibration_axes,
)
from analogy_scorer import (
    ANALOGY_COMMON_MAPPING_BANK_VERSION,
    ANALOGY_TRANSFER_VERSION,
    ANALOGY_TRANSFER_V3_CALIBRATION_POLICY,
    ANALOGY_TRANSFER_V3_RUNTIME_SCORING_POLICY,
    ANALOGY_TRANSFER_V3_TASK_OVERLAY_VERSION,
    ANALOGY_TRANSFER_V3_TEST_VISIBILITY_POLICY,
    AnalogyTransferScorer,
    aggregate_analogy_transfer_challenge_axes,
    get_analogy_common_mapping_bank_coverage,
    get_analogy_evidence_alias_coverage,
    get_analogy_transfer_task_overlay_coverage,
)
from cross_task_fact_consistency_scorer import (
    CROSS_TASK_FACT_CONSISTENCY_VERSION,
    aggregate_cross_task_fact_consistency_axes,
    score_cross_task_fact_consistency,
)
from typed_axis_aggregation import (
    aggregate_repeat_subtype_scores,
    aggregate_t1_subtype_scores,
    build_uut_idea_subtype_contributions,
    mean_subtype_contributions,
)
from t1_assoc_v3 import (
    T1_ASSOC_VERSION,
    effective_diversity,
    get_component_params,
    mechanism_elaboration_score,
    top_mean,
    transform_common_answer_rarity,
    uut_item_quality,
)
from task_registry import load_task_registry_v2
from taxonomy_registry import load_taxonomy_v2
from benchmark_core import *


def get_v2_registry_metadata():
    try:
        taxonomy = load_taxonomy_v2()
        registry = load_task_registry_v2(taxonomy=taxonomy)
        return {
            "taxonomy_version": taxonomy.get("version"),
            "task_registry_version": registry.get("version"),
        }
    except Exception as exc:
        return {
            "taxonomy_version": None,
            "task_registry_version": None,
            "registry_metadata_error": str(exc),
        }


def subtype_scores_ready_for_correlation(subtype_scores):
    if not isinstance(subtype_scores, dict):
        return False
    contributions = subtype_scores.get("component_contributions")
    if not isinstance(contributions, dict) or not contributions:
        return False
    for view in ("raw", "gated", "residual"):
        view_payload = subtype_scores.get(view) or {}
        for axis in ("imagination", "hallucination"):
            axis_payload = view_payload.get(axis) if isinstance(view_payload, dict) else {}
            if isinstance(axis_payload, dict) and any(value is not None for value in axis_payload.values()):
                return True
    return False

def extract_atom_signals_from_score(score):
    if not isinstance(score, dict):
        return None
    atoms = score.get("atom_signals")
    if isinstance(atoms, dict):
        return atoms
    subtype_contributions = score.get("subtype_contributions")
    if isinstance(subtype_contributions, dict) and isinstance(subtype_contributions.get("atom_signals"), dict):
        return subtype_contributions.get("atom_signals")
    return None

def attach_task_atom_signals(task_result):
    if not isinstance(task_result, dict) or isinstance(task_result.get("atom_signals"), dict):
        return task_result
    for key in ("dual_axis", "calibration", "challenge"):
        atoms = extract_atom_signals_from_score(task_result.get(key))
        if atoms:
            task_result["atom_signals"] = atoms
            return task_result
    return task_result

def get_non_model_skip_reason(llm_result):
    status = llm_result.get("status")
    if status == "infra_error":
        return llm_result.get("error_type") or "infra_error"
    if status == "harness_error":
        return llm_result.get("error_type") or "harness_error"
    if status == "error":
        return llm_result.get("error_type") or "runtime_error"
    return None

def select_valid_dat_words(dat_scorer_obj, candidates, max_words=None):
    if max_words is None:
        max_words = dat_scorer_obj.N_SCORING_WORDS

    valid = []
    seen_lemmas = set()
    for word in candidates:
        is_valid, normalized = dat_scorer_obj.validate_word(word)
        if not is_valid or normalized is None:
            continue

        lemma = normalized
        if dat_scorer_obj.lemmatizer:
            try:
                lemma = dat_scorer_obj.lemmatizer.lemmatize(normalized, pos='n')
            except Exception:
                lemma = normalized

        if lemma in seen_lemmas or normalized in seen_lemmas:
            continue

        valid.append(normalized)
        seen_lemmas.add(lemma)
        seen_lemmas.add(normalized)
        if len(valid) >= max_words:
            break
    return valid

def get_nested_value(payload, *path):
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current

def aggregate_nested_numeric_dicts(reports, *path):
    dict_values = [get_nested_value(report, *path) for report in reports]
    keys = set()
    for value in dict_values:
        if isinstance(value, dict):
            keys.update(value.keys())

    means = {}
    stats_by_key = {}
    for key in sorted(keys):
        stats = summarize_numeric_samples([
            value.get(key) if isinstance(value, dict) else None
            for value in dict_values
        ])
        stats_by_key[key] = stats
        if stats["mean"] is not None:
            means[key] = stats["mean"]
    return means, stats_by_key

def summarize_report_metric(reports, *path):
    return summarize_numeric_samples([get_nested_value(report, *path) for report in reports])

def summarize_repeat_report(report):
    axes = report.get("overall_summary", {}).get("axes", {})
    validity = report.get("run_validity", {})
    return {
        "repeat_index": report.get("repeat_index"),
        "ranking_eligible": validity.get("ranking_eligible"),
        "eligibility_failures": validity.get("eligibility_failures", []),
        "dt_total": axes.get("dt_total", {}).get("score"),
        "novelty": axes.get("novelty", {}).get("score"),
        "flexibility": axes.get("flexibility", {}).get("score"),
        "groundedness": axes.get("groundedness", {}).get("score"),
        "imagination": axes.get("imagination", {}).get("score"),
        "hallucination": axes.get("hallucination", {}).get("score"),
        "creative_total_coverage": validity.get("creative_total_coverage"),
        "creative_total_availability": validity.get("creative_total_availability"),
        "dat_coverage": validity.get("dat_coverage"),
        "dat_availability": validity.get("dat_availability"),
        "cdat_coverage": validity.get("cdat_coverage"),
        "cdat_availability": validity.get("cdat_availability"),
        "macgyver_coverage": validity.get("macgyver_coverage"),
        "macgyver_availability": validity.get("macgyver_availability"),
        "cjst_coverage": validity.get("cjst_coverage"),
        "cjst_availability": validity.get("cjst_availability"),
        "hypospace_coverage": validity.get("hypospace_coverage"),
        "hypospace_availability": validity.get("hypospace_availability"),
        "gcw_coverage": validity.get("gcw_coverage"),
        "gcw_availability": validity.get("gcw_availability"),
        "neocoder_coverage": validity.get("neocoder_coverage"),
        "neocoder_availability": validity.get("neocoder_availability"),
        "closed_world_fact_coverage": validity.get("closed_world_fact_coverage"),
        "closed_world_fact_availability": validity.get("closed_world_fact_availability"),
        "analogy_transfer_coverage": validity.get("analogy_transfer_coverage"),
        "analogy_transfer_availability": validity.get("analogy_transfer_availability"),
        "non_model_skip_counts": validity.get("non_model_skip_counts", {}),
        "invalid_run_counts": validity.get("invalid_run_counts", {}),
    }

def format_percent_or_na(value):
    return f"{value:.2%}" if value is not None else "N/A"

def collect_repeat_task_results(repeat_reports):
    return [
        copy.deepcopy(task_result)
        for report in repeat_reports
        for task_result in (report.get("task_results") or [])
    ]

def collect_repeat_section_details(repeat_reports, section_name):
    details = []
    for report in repeat_reports:
        section = (report.get("dat_cdat_ff_results") or {}).get(section_name)
        if not isinstance(section, dict):
            continue
        for detail in section.get("details") or []:
            details.append(copy.deepcopy(detail))
    return details

def build_repeat_level_summary(repeat_reports):
    audit_rows = []
    for report in repeat_reports:
        axes = report.get("overall_summary", {}).get("axes", {})
        validity = report.get("run_validity", {})
        dat_cdat_ff = report.get("dat_cdat_ff_results") or {}
        audit_rows.append({
            "repeat_index": report.get("repeat_index"),
            "ranking_eligible": validity.get("ranking_eligible"),
            "eligibility_failures": validity.get("eligibility_failures", []),
            "axes": {
                "dt_total": get_nested_value(report, "overall_summary", "axes", "dt_total", "score"),
                "novelty": get_nested_value(report, "overall_summary", "axes", "novelty", "score"),
                "flexibility": get_nested_value(report, "overall_summary", "axes", "flexibility", "score"),
                "groundedness": get_nested_value(report, "overall_summary", "axes", "groundedness", "score"),
                "imagination": get_nested_value(report, "overall_summary", "axes", "imagination", "score"),
                "hallucination": get_nested_value(report, "overall_summary", "axes", "hallucination", "score"),
                "macgyver_imagination": get_nested_value(report, "overall_summary", "axes", "macgyver_dual_axis", "imagination"),
                "macgyver_hallucination": get_nested_value(report, "overall_summary", "axes", "macgyver_dual_axis", "hallucination"),
                "cjst_imagination": get_nested_value(report, "overall_summary", "axes", "cjst_dual_axis", "imagination"),
                "cjst_hallucination": get_nested_value(report, "overall_summary", "axes", "cjst_dual_axis", "hallucination"),
                "hypospace_imagination": get_nested_value(report, "overall_summary", "axes", "hypospace_dual_axis", "imagination"),
                "hypospace_hallucination": get_nested_value(report, "overall_summary", "axes", "hypospace_dual_axis", "hallucination"),
        "gcw_imagination": get_nested_value(report, "overall_summary", "axes", "gcw_dual_axis", "imagination"),
        "gcw_hallucination": get_nested_value(report, "overall_summary", "axes", "gcw_dual_axis", "hallucination"),
        "neocoder_imagination": get_nested_value(report, "overall_summary", "axes", "neocoder_dual_axis", "imagination"),
        "neocoder_hallucination": get_nested_value(report, "overall_summary", "axes", "neocoder_dual_axis", "hallucination"),
        "closed_world_fact_score": get_nested_value(report, "overall_summary", "axes", "closed_world_fact_calibration", "score"),
        "closed_world_fact_hallucination": get_nested_value(report, "overall_summary", "axes", "closed_world_fact_calibration", "hallucination"),
            },
            "component_scores": get_nested_value(report, "overall_summary", "axes", "novelty", "component_scores") or {},
            "component_coverage": get_nested_value(report, "overall_summary", "axes", "novelty", "component_coverage") or {},
            "dat": {
                "mean_score": get_nested_value(dat_cdat_ff, "dat", "mean_score"),
                "scored_trials": len(get_nested_value(dat_cdat_ff, "dat", "scores") or []),
            },
            "cdat": {
                "cdat_score": get_nested_value(dat_cdat_ff, "cdat", "cdat_score"),
                "mean_novelty": get_nested_value(dat_cdat_ff, "cdat", "mean_novelty"),
                "mean_appropriateness": get_nested_value(dat_cdat_ff, "cdat", "mean_appropriateness"),
                "gate_pass_rate": get_nested_value(dat_cdat_ff, "cdat", "gate_pass_rate"),
            },
            "ff": {
                "mean_score": get_nested_value(dat_cdat_ff, "ff", "mean_score"),
                "mean_trajectory_slope": get_nested_value(dat_cdat_ff, "ff", "mean_trajectory_slope"),
            },
            "coverage": {
                "creative_total": validity.get("creative_total_coverage"),
                "creative_availability": validity.get("creative_total_availability"),
                "dat": validity.get("dat_coverage"),
                "dat_availability": validity.get("dat_availability"),
                "cdat": validity.get("cdat_coverage"),
                "cdat_availability": validity.get("cdat_availability"),
                "macgyver": validity.get("macgyver_coverage"),
                "macgyver_availability": validity.get("macgyver_availability"),
                "cjst": validity.get("cjst_coverage"),
                "cjst_availability": validity.get("cjst_availability"),
                "hypospace": validity.get("hypospace_coverage"),
                "hypospace_availability": validity.get("hypospace_availability"),
                "gcw": validity.get("gcw_coverage"),
                "gcw_availability": validity.get("gcw_availability"),
                "neocoder": validity.get("neocoder_coverage"),
                "neocoder_availability": validity.get("neocoder_availability"),
                "closed_world_fact": validity.get("closed_world_fact_coverage"),
                "closed_world_fact_availability": validity.get("closed_world_fact_availability"),
                "analogy_transfer": validity.get("analogy_transfer_coverage"),
                "analogy_transfer_availability": validity.get("analogy_transfer_availability"),
            },
        })
    return audit_rows

def clip01(value):
    return max(0.0, min(1.0, float(value)))

def compute_uut_dual_axis_diversity(embedding_flexibility, ontological_flexibility):
    embedding_flexibility = embedding_flexibility or {}
    ontological_flexibility = ontological_flexibility or {}
    return clip01(
        0.45 * float(embedding_flexibility.get("mean_pairwise_distance") or 0.0) +
        0.25 * float(embedding_flexibility.get("cluster_entropy") or 0.0) +
        0.30 * float(ontological_flexibility.get("category_diversity_index") or 0.0)
    )

def compute_uut_dual_axis_task_scores(task_details, diversity_score, *, expected_output_count=None):
    primitive_details = [
        (detail, detail.get("dual_axis_primitives"))
        for detail in task_details
        if isinstance(detail.get("dual_axis_primitives"), dict)
    ]
    primitives = [item for _, item in primitive_details]
    if not primitives:
        return None

    novelty_grounded = mean_or_none([
        float(item.get("novelty_times_affordance_support", item.get("novelty_times_groundedness")) or 0.0)
        for item in primitives
    ]) or 0.0
    mechanism_grounded = mean_or_none([
        float(item.get("mechanism_times_affordance_support", item.get("mechanism_times_groundedness")) or 0.0)
        for item in primitives
    ]) or 0.0
    expected_output_count = int(expected_output_count or UUT_OUTPUT_COUNT)
    params = get_component_params("UUT")
    item_quality_values = []
    item_rarity_values = []
    item_elaboration_values = []
    valid_count = 0
    bank_available_count = 0
    for detail, item in primitive_details:
        bank_value = detail.get("bank_originality")
        if bank_value is not None:
            bank_available_count += 1
        rarity_source = bank_value if bank_value is not None else item.get("novelty")
        rarity = transform_common_answer_rarity(rarity_source)
        support = clip01(float(item.get("supported_affordance_ratio") or 0.0))
        gate = clip01(float(item.get("appropriateness_gate") or 0.0))
        mechanism = clip01(float(item.get("mechanism_completeness") or 0.0))
        hallucination_item = clip01(float(item.get("idea_hallucination_raw") or 0.0))
        idea_text = " ".join(
            str(detail.get(key) or item.get(key) or "")
            for key in ("idea", "use", "text", "response", "description", "mechanism", "raw_text")
        )
        elaboration = mechanism_elaboration_score(
            idea_text or json.dumps(detail, ensure_ascii=False),
            support=support,
            mechanism=mechanism,
            rarity=rarity,
        )
        quality = uut_item_quality(
            rarity=rarity,
            support=support,
            gate=gate,
            mechanism=mechanism,
            params=params,
        )
        item["t1_assoc_version"] = T1_ASSOC_VERSION
        item["rarity_v3"] = round(rarity, 4)
        item["mechanism_elaboration_v3"] = round(elaboration, 4)
        item["imagination_contribution_v3"] = round(quality, 4)
        item_quality_values.append(quality)
        item_rarity_values.append(rarity)
        item_elaboration_values.append(elaboration)
        if support >= 0.45 and gate >= 0.35 and hallucination_item <= 0.45:
            valid_count += 1

    valid_ratio = min(1.0, valid_count / max(1, expected_output_count))
    hallucination_items = [
        float(item.get("idea_hallucination_raw") or 0.0)
        for item in primitives
    ]
    hallucination_raw = mean_or_none(hallucination_items) or 0.0
    diversity_score = clip01(diversity_score)
    diversity_eff = effective_diversity(diversity_score, valid_ratio)
    dynamic_quality_n = max(1, min(8, valid_count or len(item_quality_values)))
    dynamic_elite_n = max(1, min(3, valid_count or len(item_quality_values)))
    quality_mass = top_mean(item_quality_values, dynamic_quality_n)
    elite_tail = top_mean(item_quality_values, dynamic_elite_n)
    mechanism_elaboration = top_mean(item_elaboration_values, dynamic_elite_n)
    weights = params.get("task_weights", {}) if isinstance(params.get("task_weights"), dict) else {}
    imagination_raw = clip01(
        float(weights.get("quality_mass_top8", 0.40)) * quality_mass +
        float(weights.get("elite_tail_top3", 0.30)) * elite_tail +
        float(weights.get("diversity_eff", 0.05)) * diversity_eff +
        float(weights.get("valid_ratio", 0.15)) * valid_ratio +
        float(weights.get("mechanism_elaboration", 0.10)) * mechanism_elaboration
    )

    imagination = clip01(imagination_raw - UUT_DUAL_AXIS_BETA_IH * hallucination_raw)
    hallucination = clip01(hallucination_raw - UUT_DUAL_AXIS_BETA_HI * imagination_raw)

    primitive_means = {}
    primitive_fields = [
        "novelty",
        "appropriateness_gate",
        "cue_drift",
        "semantic_anchor",
        "cue_support_failure",
        "supported_affordance_ratio",
        "unsupported_claim_ratio",
        "contradiction_ratio",
        "extra_tool_violation",
        "mechanism_completeness",
        "physical_drift",
        "novelty_times_affordance_support",
        "novelty_times_groundedness",
        "appropriateness_gated_novelty_affordance_support",
        "mechanism_times_affordance_support",
        "mechanism_times_groundedness",
        "appropriateness_gated_mechanism_affordance_support",
        "idea_hallucination_raw",
        "rarity_v3",
        "imagination_contribution_v3",
        "mechanism_elaboration_v3",
    ]
    for field in primitive_fields:
        value = mean_or_none([
            float(item.get(field))
            for item in primitives
            if item.get(field) is not None
        ])
        if value is not None:
            primitive_means[field] = round(value, 4)
    subtype_contributions = mean_subtype_contributions(
        build_uut_idea_subtype_contributions({**item, "diversity": diversity_score})
        for item in primitives
    )

    result = {
        "version": UUT_DUAL_AXIS_VERSION,
        "t1_assoc_version": T1_ASSOC_VERSION,
        "score": round(imagination, 4),
        "imagination": round(imagination, 4),
        "hallucination": round(hallucination, 4),
        "imagination_raw": round(imagination_raw, 4),
        "hallucination_raw": round(hallucination_raw, 4),
        "diversity": round(diversity_score, 4),
        "diversity_eff": round(diversity_eff, 4),
        "quality_mass_top8": round(quality_mass, 4),
        "elite_tail_top3": round(elite_tail, 4),
        "mechanism_elaboration": round(mechanism_elaboration, 4),
        "dynamic_quality_top_n": dynamic_quality_n,
        "dynamic_elite_top_n": dynamic_elite_n,
        "valid_ratio": round(valid_ratio, 4),
        "valid_count": valid_count,
        "bank_coverage": round(bank_available_count / max(1, len(primitives)), 4),
        "mean_rarity_v3": round(mean_or_none(item_rarity_values) or 0.0, 4),
        "novelty_times_affordance_support_mean": round(novelty_grounded, 4),
        "novelty_times_groundedness_mean": round(novelty_grounded, 4),
        "mechanism_times_affordance_support_mean": round(mechanism_grounded, 4),
        "mechanism_times_groundedness_mean": round(mechanism_grounded, 4),
        "primitive_means": primitive_means,
        "subtype_contributions": subtype_contributions,
        "scored_ideas": len(primitives),
        "formula": {
            "imagination_raw": "T1  0.40*dynamic_top8(item_I)+0.30*dynamic_top3(item_I)+0.05*diversity_eff+0.15*valid_ratio+0.10*mechanism_elaboration",
            "item_I": "rarity^1.8 * support^1.2 * appropriateness_gate * (0.5+0.5*mechanism)",
            "hallucination_raw": "mean(0.35*U + 0.30*C + 0.20*E + 0.15*P)",
            "subtypes": "T1: I_assoc=appropriateness-gated anchored association; H=context/intent/drift/logic from affordance, tool-role, cue, and physical-drift ledgers",
            "residual": "I=clip01(I_raw - beta_IH*H_raw); H=clip01(H_raw - beta_HI*I_raw)",
        },
        "residualization": {
            "beta_IH": UUT_DUAL_AXIS_BETA_IH,
            "beta_HI": UUT_DUAL_AXIS_BETA_HI,
            "source": "benchmark_default",
        },
    }
    result["atom_signals"] = result["subtype_contributions"].get("atom_signals", {})
    return result

def run_model_assessment_once(client, model_name, dataset, scorer, cog_baseline,
                              wn_analyzer, groundedness_scorer, dat_scorer_obj=None,
                              ff_scorer_obj=None, model_catalog_entry=None, replicate_index=0):
    repeat_label = f"{replicate_index + 1}/{MODEL_SAMPLE_REPEATS}"
    print("\n" + "#" * 80)
    print(f"### Evaluating model: {model_name} [repeat {repeat_label}]")
    print("#" * 80)

    resolved_models_seen = set()
    task_results = []

    total_fluency_raw_all = 0
    total_fluency_deduped_all = 0
    task_count = 0
    sum_wn_distance = 0.0
    sum_wn_ic = 0.0
    all_task_cluster_ratios = []
    all_task_emb_switch_rates = []
    all_task_emb_pairwise_distances = []
    all_task_emb_adjacent_distances = []
    all_task_emb_cluster_entropies = []
    global_wn_categories_set = set()
    global_wn_category_counter = Counter()
    all_task_wn_switch_rates = []
    total_zero_orig_count = 0
    total_scored_ideas = 0
    propconj_scorer = PropConjScorer(
        word_norms2=getattr(groundedness_scorer, "word_norms2", None)
    )
    macgyver_scorer = MacGyverScorer(
        beta_ih=MACGYVER_DUAL_AXIS_BETA_IH,
        beta_hi=MACGYVER_DUAL_AXIS_BETA_HI,
        expected_plan_count=MACGYVER_OUTPUT_COUNT,
    )
    cjst_scorer = CounterfactualScorer(
        beta_ih=CJST_DUAL_AXIS_BETA_IH,
        beta_hi=CJST_DUAL_AXIS_BETA_HI,
    )
    hypospace_scorer = HypoUseSpaceScorer(
        beta_ih=HYPOUSESPACE_DUAL_AXIS_BETA_IH,
        beta_hi=HYPOUSESPACE_DUAL_AXIS_BETA_HI,
        expected_output_count=HYPOUSESPACE_OUTPUT_COUNT,
    )
    gcw_scorer = GroundedCreativeWritingScorer(
        beta_ih=GCW_DUAL_AXIS_BETA_IH,
        beta_hi=GCW_DUAL_AXIS_BETA_HI,
    )
    neocoder_scorer = NeoCoderScorer(
        beta_ih=NEOCODER_DUAL_AXIS_BETA_IH,
        beta_hi=NEOCODER_DUAL_AXIS_BETA_HI,
    )
    closed_world_fact_scorer = ClosedWorldFactScorer()
    analogy_transfer_scorer = AnalogyTransferScorer()

    invalid_run_counts = {
        "creative_tasks": 0,
        "cjst": 0,
        "macgyver": 0,
        "macgyver_boundary": 0,
        "hypospace": 0,
        "hypospace_boundary": 0,
        "gcw": 0,
        "neocoder": 0,
        "closed_world_fact": 0,
        "analogy_transfer": 0,
        "dat": 0,
        "cdat": 0,
        "ff": 0,
    }
    non_model_skip_counts = {
        "creative_tasks": 0,
        "cjst": 0,
        "macgyver": 0,
        "macgyver_boundary": 0,
        "hypospace": 0,
        "hypospace_boundary": 0,
        "gcw": 0,
        "neocoder": 0,
        "closed_world_fact": 0,
        "analogy_transfer": 0,
        "dat": 0,
        "cdat": 0,
        "ff": 0,
    }
    non_model_skip_reasons = Counter()

    category_originality_sums = {task_type: 0.0 for task_type in CREATIVE_TASK_TYPES}
    category_originality_counts = {task_type: 0 for task_type in CREATIVE_TASK_TYPES}
    task_type_ground_scores = {task_type: [] for task_type in CREATIVE_TASK_TYPES}
    task_type_ground_scores_novel = {task_type: [] for task_type in CREATIVE_TASK_TYPES}
    uut_dual_axis_task_scores = []
    uut_dual_axis_task_scores_raw = []
    propconj_dual_axis_task_scores = []
    cjst_results = []
    cjst_dual_axis_task_scores = []
    macgyver_results = []
    macgyver_dual_axis_task_scores = []
    macgyver_boundary_results = []
    macgyver_boundary_task_scores = []
    macgyver_boundary_effective_prompts = 0
    macgyver_boundary_scorable_count = 0
    macgyver_boundary_excluded_count = 0
    hypospace_results = []
    hypospace_dual_axis_task_scores = []
    hypospace_boundary_results = []
    hypospace_boundary_task_scores = []
    hypospace_boundary_effective_prompts = 0
    hypospace_boundary_scorable_count = 0
    hypospace_boundary_excluded_count = 0
    gcw_results = []
    gcw_dual_axis_task_scores = []
    neocoder_results = []
    neocoder_dual_axis_task_scores = []
    closed_world_fact_results = []
    closed_world_fact_task_scores = []
    analogy_transfer_results = []
    analogy_transfer_task_scores = []

    total_groundedness_weighted = 0.0
    total_groundedness_weight = 0.0
    total_groundedness_weighted_novel = 0.0
    total_groundedness_weight_novel = 0.0
    total_groundedness_score_count = 0
    total_groundedness_novel_count = 0
    total_groundedness_confidence = 0.0
    total_penalty = 0.0
    total_penalty_positive = 0
    total_low_groundedness = 0
    total_low_groundedness_novel = 0
    groundedness_reference_cohort = (
        groundedness_scorer.get_reference_cohort()
        if hasattr(groundedness_scorer, "get_reference_cohort")
        else None
    )

    creative_task_totals = {task_type: len(dataset.get(task_type, [])) for task_type in CREATIVE_TASK_TYPES}
    creative_task_valid_counts = {task_type: 0 for task_type in CREATIVE_TASK_TYPES}
    creative_task_effective_totals = {task_type: 0 for task_type in CREATIVE_TASK_TYPES}
    creative_task_excluded_counts = {task_type: 0 for task_type in CREATIVE_TASK_TYPES}
    total_creative_prompts = sum(creative_task_totals.values())

    macgyver_total_prompts = sum(
        1 for task in dataset.get("MacGyver", [])
        if not task.get("macgyver_boundary_diagnostic")
    )
    macgyver_boundary_total_prompts = sum(
        1 for task in dataset.get("MacGyver", [])
        if task.get("macgyver_boundary_diagnostic")
    )
    macgyver_effective_prompts = 0
    macgyver_scorable_count = 0
    macgyver_excluded_count = 0

    cjst_total_prompts = len(dataset.get("CJST", []))
    cjst_effective_prompts = 0
    cjst_scorable_count = 0
    cjst_excluded_count = 0

    hypospace_total_prompts = sum(
        1 for task in dataset.get("HypoUseSpace", [])
        if not task.get("hypospace_boundary_diagnostic")
    )
    hypospace_boundary_total_prompts = sum(
        1 for task in dataset.get("HypoUseSpace", [])
        if task.get("hypospace_boundary_diagnostic")
    )
    hypospace_effective_prompts = 0
    hypospace_scorable_count = 0
    hypospace_excluded_count = 0

    gcw_total_prompts = len(dataset.get("GCW", []))
    gcw_effective_prompts = 0
    gcw_scorable_count = 0
    gcw_excluded_count = 0

    neocoder_total_prompts = len(dataset.get("NeoCoder", []))
    neocoder_effective_prompts = 0
    neocoder_scorable_count = 0
    neocoder_excluded_count = 0

    closed_world_fact_total_prompts = len(dataset.get("ClosedWorldFact", []))
    closed_world_fact_effective_prompts = 0
    closed_world_fact_scorable_count = 0
    closed_world_fact_excluded_count = 0

    analogy_transfer_total_prompts = len(dataset.get("AnalogyTransfer", []))
    analogy_transfer_effective_prompts = 0
    analogy_transfer_scorable_count = 0
    analogy_transfer_excluded_count = 0

    dat_total_prompts = len(dataset.get("DAT", []))
    cdat_total_prompts = len(dataset.get("CDAT", []))
    ff_total_prompts = len(dataset.get("FF", []))
    dat_effective_prompts = 0
    cdat_effective_prompts = 0
    ff_effective_prompts = 0
    dat_scorable_count = 0
    cdat_scorable_count = 0
    ff_scorable_count = 0
    dat_excluded_count = 0
    cdat_excluded_count = 0
    ff_excluded_count = 0

    for task_type in CREATIVE_TASK_TYPES:
        if task_type not in dataset:
            continue

        print(f"\n{'=' * 60}\n  Test Module: {task_type} [{model_name}] [repeat {repeat_label}]\n{'=' * 60}")
        expected_output_count = get_expected_output_count(task_type) or CREATIVE_OUTPUT_COUNT
        min_required_items = (
            MIN_PROP_CONJ_ITEMS_PER_TASK if task_type == "PropConj" else MIN_CREATIVE_ITEMS_PER_TASK
        )

        for task in dataset[task_type]:
            task_id = task["id"]
            target_concept = task.get("item") or task.get("scenario") or task.get("trait")
            prompt_text = task.get("prompt")
            prompt_metadata = {k: v for k, v in task.items() if k not in {"id", "prompt"}}
            print(f"\n[Prompt Target]: {target_concept}")

            zero_orig_context = get_zero_originality_runtime_context(
                item_id=task_id,
                task_type=task_type,
                target_concept=target_concept,
                task_metadata=prompt_metadata,
            )
            dynamic_context = zero_orig_context.get("dynamic_context") or target_concept
            dynamic_baseline = cog_baseline.get_dynamic_baseline(dynamic_context)
            if dynamic_baseline:
                preview = dynamic_baseline[:15]
                suffix = "..." if len(dynamic_baseline) > 15 else ""
                context_preview = dynamic_context[:120] + ("..." if len(dynamic_context) > 120 else "")
                print(f"[Dynamic Baseline ({len(dynamic_baseline)} words)]: {preview}{suffix}")
                print(f"[Zero-Orig Context]: {context_preview}")

            common_answer_bank_context = build_common_answer_bank_context(
                item_id=task_id,
                task_type=task_type,
                target_concept=target_concept,
                task_metadata=prompt_metadata,
                cognitive_baseline=cog_baseline,
                swow_graph=getattr(groundedness_scorer, "swow", None),
                word_norms2=getattr(groundedness_scorer, "word_norms2", None),
            )
            common_bank_total = common_answer_bank_context.get("combined_bank_size", 0)
            if common_bank_total > 0:
                print(
                    "[Common-Answer Bank] "
                    f"static={common_answer_bank_context['static_bank']['size']}, "
                    f"dynamic={common_answer_bank_context['dynamic_bank']['size']}, "
                    f"total={common_bank_total}: "
                    f"{common_answer_bank_context.get('combined_bank_preview', [])[:12]}"
                )

            llm_result = call_llm(
                client,
                task["prompt"],
                model_name,
                task_label=task_type,
                max_tokens_override=get_task_max_tokens(task_type if task_type in {"UUT", "PropConj"} else "creative"),
                seed=stable_seed(model_name, task_type, task_id, replicate_index),
            )
            resolved_model = llm_result["resolved_model"]
            resolved_models_seen.add(resolved_model)
            raw_output = llm_result["content"]
            parsed_items_all = parse_creative_items(raw_output, task_type=task_type)
            parsed_item_count = len(parsed_items_all)
            truncated_item_count = max(0, parsed_item_count - expected_output_count)
            parsed_items_raw = parsed_items_all[:expected_output_count]
            ideas_raw = [item.get("display_text") for item in parsed_items_raw if item.get("display_text")]
            raw_fluency = len(ideas_raw)
            print(f"Model generated {parsed_item_count} parsed ideas (using first {raw_fluency}).")

            non_model_skip_reason = get_non_model_skip_reason(llm_result)
            invalid_reason = None
            if non_model_skip_reason is not None:
                creative_task_excluded_counts[task_type] += 1
                non_model_skip_counts["creative_tasks"] += 1
                non_model_skip_reasons[non_model_skip_reason] += 1
                print(f"[Coverage Excluded] {non_model_skip_reason}")
                task_results.append({
                    "task_id": task_id,
                    "task_type": task_type,
                    "prompt": prompt_text,
                    "prompt_metadata": prompt_metadata,
                    "target_concept": target_concept,
                    "repeat_index": replicate_index,
                    "coverage_eligible": False,
                    "excluded_from_coverage": True,
                    "non_model_failure": True,
                    "non_model_skip_reason": non_model_skip_reason,
                    "valid_run": False,
                    "invalid_reason": None,
                    "generation": build_generation_record(llm_result),
                    "expected_output_count": expected_output_count,
                    "parsed_item_count": parsed_item_count,
                    "raw_items_truncated": truncated_item_count,
                    "raw_fluency": raw_fluency,
                    "fluency_after_dedup": None,
                    "duplicates_removed": None,
                    "zero_originality_count": None,
                    "average_originality": None,
                    "average_originality_legacy": None,
                    "average_originality_bank": None,
                    "average_adjusted_originality": None,
                    "average_penalty": None,
                    "average_groundedness": None,
                    "average_groundedness_novel_only": None,
                    "embedding_flexibility": None,
                    "ontological_flexibility": None,
                    "groundedness": None,
                    "dynamic_baseline_size": len(dynamic_baseline),
                    "common_answer_bank": {
                        "static_bank_size": common_answer_bank_context["static_bank"]["size"],
                        "dynamic_bank_size": common_answer_bank_context["dynamic_bank"]["size"],
                        "combined_bank_size": common_answer_bank_context["combined_bank_size"],
                        "combined_bank_preview": common_answer_bank_context["combined_bank_preview"],
                    },
                    "details": [],
                })
                continue

            creative_task_effective_totals[task_type] += 1
            if not raw_output.strip():
                invalid_reason = "empty_response"
            elif raw_fluency == 0:
                invalid_reason = "parsed_zero_items"
            elif raw_fluency < min_required_items:
                invalid_reason = "insufficient_items"

            if invalid_reason is not None:
                invalid_run_counts["creative_tasks"] += 1
                print(f"[Invalid Run] {invalid_reason}")
                task_results.append({
                    "task_id": task_id,
                    "task_type": task_type,
                    "prompt": prompt_text,
                    "prompt_metadata": prompt_metadata,
                    "target_concept": target_concept,
                    "repeat_index": replicate_index,
                    "coverage_eligible": True,
                    "excluded_from_coverage": False,
                    "non_model_failure": False,
                    "non_model_skip_reason": None,
                    "valid_run": False,
                    "invalid_reason": invalid_reason,
                    "generation": build_generation_record(llm_result),
                    "expected_output_count": expected_output_count,
                    "parsed_item_count": parsed_item_count,
                    "raw_items_truncated": truncated_item_count,
                    "raw_fluency": raw_fluency,
                    "fluency_after_dedup": None,
                    "duplicates_removed": None,
                    "zero_originality_count": None,
                    "average_originality": None,
                    "average_originality_legacy": None,
                    "average_originality_bank": None,
                    "average_adjusted_originality": None,
                    "average_penalty": None,
                    "average_groundedness": None,
                    "average_groundedness_novel_only": None,
                    "embedding_flexibility": None,
                    "ontological_flexibility": None,
                    "groundedness": None,
                    "dynamic_baseline_size": len(dynamic_baseline),
                    "common_answer_bank": {
                        "static_bank_size": common_answer_bank_context["static_bank"]["size"],
                        "dynamic_bank_size": common_answer_bank_context["dynamic_bank"]["size"],
                        "combined_bank_size": common_answer_bank_context["combined_bank_size"],
                        "combined_bank_preview": common_answer_bank_context["combined_bank_preview"],
                    },
                    "details": [],
                })
                continue

            cleaned_ideas_raw = [
                preprocess_for_semantic_scoring(idea, target_concept)
                for idea in ideas_raw
            ]
            dedup_result = scorer.deduplicate_ideas(
                cleaned_ideas_raw,
                similarity_threshold=DEDUP_SIMILARITY_THRESHOLD,
            )
            unique_indices = dedup_result["unique_indices"]
            num_removed = dedup_result["num_removed"]
            ideas = [ideas_raw[index] for index in unique_indices]
            parsed_items = [parsed_items_raw[index] for index in unique_indices]
            cleaned_ideas_list = [cleaned_ideas_raw[index] for index in unique_indices]
            fluency = len(ideas)

            if num_removed > 0:
                print(f"[Deduplication] Removed {num_removed}. Unique: {fluency}/{raw_fluency}")

            total_originality = 0.0
            total_originality_legacy = 0.0
            task_total_penalty = 0.0
            task_total_confidence = 0.0
            task_zero_orig = 0
            task_details = []
            task_ground_values = []
            task_ground_values_novel = []
            task_penalties = []
            task_bank_values = []
            task_low_ground_count = 0
            task_low_ground_count_novel = 0

            print(f"Scoring {fluency} unique ideas:")
            for parsed_item, idea, cleaned_idea in zip(parsed_items, ideas, cleaned_ideas_list):
                zero_orig_trace = analyze_zero_originality(
                    task_id,
                    response_text=idea,
                    scorer=scorer,
                    cognitive_baseline=cog_baseline,
                    target_concept=target_concept,
                    task_type=task_type,
                    task_metadata=prompt_metadata,
                    parsed_item=parsed_item,
                )
                zero_orig = bool(zero_orig_trace.get("zero_orig_final"))
                if zero_orig:
                    legacy_score = 0.0
                    task_zero_orig += 1
                else:
                    legacy_score = scorer.calculate_originality(target_concept, cleaned_idea)

                bank_trace = score_common_answer_bank_novelty(
                    task_id,
                    task_type=task_type,
                    response_text=idea,
                    parsed_item=parsed_item,
                    scorer=scorer,
                    bank_context=common_answer_bank_context,
                    zero_orig_trace=zero_orig_trace,
                )
                bank_score = bank_trace.get("bank_score")
                novelty_blend = blend_creative_novelty(task_type, legacy_score, bank_score)
                score = novelty_blend.get("blended_score")
                if score is None:
                    score = legacy_score if legacy_score is not None else 0.0

                propconj_item_score = None
                if task_type == "PropConj":
                    propconj_item_score = propconj_scorer.score_item(
                        task_metadata=prompt_metadata,
                        parsed_item=parsed_item,
                        item_text=idea,
                        novelty_score=score,
                        legacy_score=legacy_score,
                        bank_score=bank_score,
                    )
                    grounding = float(propconj_item_score.get("grounding", 0.0) or 0.0)
                    penalty = score * (1.0 - grounding) if score > 0 else 0.0
                    ground_result = {
                        "groundedness_score": grounding,
                        "groundedness_confidence": 0.85,
                        "groundedness_penalty": penalty,
                        "score_version": PROPCONJ_DUAL_AXIS_VERSION,
                        "formula": "propconj_grounding_as_conjunction_validity",
                        "subscores": {
                            "conjunction_geomean": propconj_item_score.get("conjunction_geomean"),
                            "min_property_support": propconj_item_score.get("min_property_support"),
                            "unresolved_entity": propconj_item_score.get("unresolved_entity"),
                            "contradiction": propconj_item_score.get("contradiction"),
                            "evidence_mismatch": propconj_item_score.get("evidence_mismatch"),
                        },
                        "dual_axis_primitives": propconj_item_score,
                        "evidence": propconj_item_score,
                        "task_threshold": 0.70,
                        "low_groundedness": grounding < 0.70,
                    }
                else:
                    ground_result = groundedness_scorer.score_idea(
                        task_type=task_type,
                        task_id=task_id,
                        target_concept=target_concept,
                        idea_text=idea,
                        raw_originality=score,
                        parsed_item=parsed_item,
                        semantic_scorer=scorer,
                        common_answer_bank_trace=bank_trace,
                        common_answer_bank_context=common_answer_bank_context,
                        cohort_stats=groundedness_reference_cohort,
                    )
                    penalty = ground_result["groundedness_penalty"] if score > 0 else 0.0
                adjusted_score = max(0.0, score - penalty)

                groundedness_score = ground_result["groundedness_score"]
                groundedness_confidence = ground_result["groundedness_confidence"]
                task_threshold = ground_result.get("task_threshold")

                if score == 0.0:
                    print(f"  [0.0000 | g={groundedness_score:.4f}] (Zero Orig) -> {idea[:72]}")
                elif penalty > 0:
                    bank_display = f"{bank_score:.4f}" if bank_score is not None else "N/A"
                    print(
                        f"  [old={legacy_score:.4f} | bank={bank_display}"
                        f" | mix={score:.4f}->{adjusted_score:.4f} | g={groundedness_score:.4f} | -{penalty:.4f}] "
                        f"-> {idea[:65]}"
                    )
                else:
                    bank_display = f"{bank_score:.4f}" if bank_score is not None else "N/A"
                    print(
                        f"  [old={legacy_score:.4f} | bank={bank_display} | mix={score:.4f} | g={groundedness_score:.4f}] "
                        f"-> {idea[:72]}"
                    )

                wn_detail = wn_analyzer.get_idea_detail(idea)
                total_originality += score
                total_originality_legacy += legacy_score
                if bank_score is not None:
                    task_bank_values.append(bank_score)
                task_total_penalty += penalty
                task_total_confidence += groundedness_confidence
                task_ground_values.append(groundedness_score)
                task_penalties.append(penalty)
                if score > 0:
                    task_ground_values_novel.append(groundedness_score)
                if task_threshold is not None and groundedness_score < task_threshold:
                    task_low_ground_count += 1
                    if score > 0:
                        task_low_ground_count_novel += 1

                total_groundedness_weighted += groundedness_score * groundedness_confidence
                total_groundedness_weight += groundedness_confidence
                total_groundedness_confidence += groundedness_confidence
                total_groundedness_score_count += 1
                total_penalty += penalty
                if penalty > 0:
                    total_penalty_positive += 1
                if task_threshold is not None and groundedness_score < task_threshold:
                    total_low_groundedness += 1
                if score > 0:
                    total_groundedness_weighted_novel += groundedness_score * groundedness_confidence
                    total_groundedness_weight_novel += groundedness_confidence
                    total_groundedness_novel_count += 1
                    if task_threshold is not None and groundedness_score < task_threshold:
                        total_low_groundedness_novel += 1

                task_details.append({
                    "idea": idea,
                    "parsed_item": parsed_item,
                    "cleaned_idea": cleaned_idea,
                    "score": round(score, 4),
                    "legacy_originality": round(legacy_score, 4),
                    "bank_originality": round(bank_score, 4) if bank_score is not None else None,
                    "raw_originality": round(score, 4),
                    "adjusted_originality": round(adjusted_score, 4),
                    "novelty_blend_formula": novelty_blend.get("formula"),
                    "novelty_blend_weights": novelty_blend.get("effective_weights"),
                    "zero_originality": bool(zero_orig),
                    "zero_orig_static": bool(zero_orig_trace.get("zero_orig_static")),
                    "zero_orig_dynamic": bool(zero_orig_trace.get("zero_orig_dynamic")),
                    "zero_orig_core_form": zero_orig_trace.get("zero_orig_core_form"),
                    "zero_orig_match_field": zero_orig_trace.get("zero_orig_match_field"),
                    "zero_orig_static_family": zero_orig_trace.get("zero_orig_static_family"),
                    "zero_orig_static_alias": zero_orig_trace.get("zero_orig_static_alias"),
                    "zero_orig_broad_common_family": zero_orig_trace.get("zero_orig_broad_common_family"),
                    "zero_orig_dynamic_evidence": zero_orig_trace.get("zero_orig_dynamic_evidence"),
                    "common_answer_bank": {
                        "core_text": bank_trace.get("core_text"),
                        "core_norm": bank_trace.get("core_norm"),
                        "match_field": bank_trace.get("match_field"),
                        "task_basic_validity": bank_trace.get("task_basic_validity"),
                        "forced_zero_due_to_zero_originality": bank_trace.get("forced_zero_due_to_zero_originality"),
                        "nearest_static_entry": bank_trace.get("nearest_static_entry"),
                        "nearest_static_distance": bank_trace.get("nearest_static_distance"),
                        "nearest_dynamic_entry": bank_trace.get("nearest_dynamic_entry"),
                        "nearest_dynamic_distance": bank_trace.get("nearest_dynamic_distance"),
                        "nearest_overall_entry": bank_trace.get("nearest_overall_entry"),
                        "nearest_overall_distance": bank_trace.get("nearest_overall_distance"),
                        "static_bank_size": bank_trace.get("static_bank_size"),
                        "dynamic_bank_size": bank_trace.get("dynamic_bank_size"),
                        "combined_bank_size": bank_trace.get("combined_bank_size"),
                        "formula": bank_trace.get("formula"),
                    },
                    "groundedness_score": groundedness_score,
                    "groundedness_confidence": groundedness_confidence,
                    "groundedness_penalty": round(penalty, 4),
                    "groundedness_version": ground_result.get("score_version"),
                    "groundedness_formula": ground_result.get("formula"),
                    "groundedness_subscores": ground_result.get("subscores", {}),
                    "g_new": ground_result.get("g_new"),
                    "g_subscores_v5": ground_result.get("g_subscores_v5"),
                    "dual_axis_primitives": ground_result.get("dual_axis_primitives"),
                    "propconj_scores": propconj_item_score,
                    "cohort_z": ground_result.get("cohort_z"),
                    "cohort_stats": ground_result.get("cohort_stats"),
                    "groundedness_penalty_trace": ground_result.get("penalty_trace_v5"),
                    "anti_cliche_score": ground_result.get("anti_cliche_score"),
                    "mech_score": ground_result.get("mech_score"),
                    "groundedness_evidence": ground_result.get("evidence", {}),
                    "low_groundedness": ground_result.get("low_groundedness"),
                    "task_threshold": task_threshold,
                    "wordnet_category": wn_detail["wordnet_category"],
                    "wordnet_nouns": wn_detail["wordnet_nouns"],
                    "wordnet_ic": wn_detail["wordnet_ic"],
                })

            total_zero_orig_count += task_zero_orig
            total_scored_ideas += fluency
            total_fluency_raw_all += raw_fluency
            total_fluency_deduped_all += fluency

            flexibility_metrics = scorer.calculate_flexibility(cleaned_ideas_list)
            num_clusters = flexibility_metrics["num_clusters"]
            category_switches = flexibility_metrics["category_switches"]
            cluster_labels = flexibility_metrics["labels"]
            for index, detail in enumerate(task_details):
                if index < len(cluster_labels):
                    detail["cluster_label"] = cluster_labels[index]

            wn_flexibility = wn_analyzer.calculate_ontological_flexibility(
                ideas,
                target_concept=target_concept,
            )
            for index, detail in enumerate(task_details):
                if index < len(wn_flexibility.get("idea_categories", [])):
                    detail["wordnet_category"] = wn_flexibility["idea_categories"][index]

            avg_orig_legacy = total_originality_legacy / fluency if fluency > 0 else 0.0
            avg_orig_bank = mean_or_none(task_bank_values)
            avg_orig = total_originality / fluency if fluency > 0 else 0.0
            avg_adj = sum(detail["adjusted_originality"] for detail in task_details) / fluency if fluency > 0 else 0.0
            avg_penalty = task_total_penalty / fluency if fluency > 0 else 0.0
            avg_ground = mean_or_none(task_ground_values)
            avg_ground_novel = mean_or_none(task_ground_values_novel)
            ground_conf_mean = task_total_confidence / fluency if fluency > 0 else 0.0
            penalty_rate = (sum(1 for value in task_penalties if value > 0) / fluency) if fluency > 0 else 0.0
            low_ground_rate = (task_low_ground_count / fluency) if fluency > 0 else 0.0
            low_ground_rate_novel = (
                task_low_ground_count_novel / len(task_ground_values_novel)
                if task_ground_values_novel else None
            )
            cluster_ratio = flexibility_metrics.get("cluster_ratio", num_clusters / fluency if fluency > 0 else 0.0)
            emb_switch_rate = flexibility_metrics.get("switch_rate", category_switches / (fluency - 1) if fluency > 1 else 0.0)
            emb_pairwise_distance = flexibility_metrics.get("mean_pairwise_distance", 0.0)
            emb_adjacent_distance = flexibility_metrics.get("mean_adjacent_distance", 0.0)
            emb_cluster_entropy = flexibility_metrics.get("cluster_entropy", 0.0)
            wn_switch_rate = (
                wn_flexibility["category_switches"] / (fluency - 1)
                if fluency > 1 else 0.0
            )
            zero_orig_rate = task_zero_orig / fluency if fluency > 0 else 0.0
            scored_coverage = len(task_ground_values) / fluency if fluency > 0 else None
            dual_axis_result = None
            if task_type == "UUT":
                dual_axis_diversity = compute_uut_dual_axis_diversity(
                    {
                        "mean_pairwise_distance": emb_pairwise_distance,
                        "cluster_entropy": emb_cluster_entropy,
                    },
                    wn_flexibility,
                )
                dual_axis_result = compute_uut_dual_axis_task_scores(
                    task_details,
                    dual_axis_diversity,
                    expected_output_count=expected_output_count,
                )
                if dual_axis_result:
                    for detail in task_details:
                        primitives = detail.get("dual_axis_primitives")
                        if isinstance(primitives, dict):
                            primitives["diversity"] = dual_axis_result["diversity"]
                    uut_dual_axis_task_scores.append(dual_axis_result)
                    uut_dual_axis_task_scores_raw.append({
                        "imagination_raw": dual_axis_result["imagination_raw"],
                        "hallucination_raw": dual_axis_result["hallucination_raw"],
                    })
            elif task_type == "PropConj":
                dual_axis_diversity = compute_propconj_diversity(
                    {
                        "mean_pairwise_distance": emb_pairwise_distance,
                        "cluster_entropy": emb_cluster_entropy,
                    },
                    wn_flexibility,
                )
                dual_axis_result = compute_propconj_task_scores(
                    task_details,
                    dual_axis_diversity,
                    expected_output_count=expected_output_count,
                    beta_ih=PROPCONJ_DUAL_AXIS_BETA_IH,
                    beta_hi=PROPCONJ_DUAL_AXIS_BETA_HI,
                )
                if dual_axis_result:
                    dual_axis_result["task_id"] = task_id
                    dual_axis_result["property_ids"] = list(prompt_metadata.get("property_ids") or [])
                    for detail in task_details:
                        prop_scores = detail.get("propconj_scores")
                        if isinstance(prop_scores, dict):
                            prop_scores["diversity"] = dual_axis_result["diversity"]
                    propconj_dual_axis_task_scores.append(dual_axis_result)

            print(f"\n>>> Fluency: Raw={raw_fluency}, Unique={fluency}")
            print(f">>> Zero Orig: {task_zero_orig}/{fluency} ({zero_orig_rate:.1%})")
            print(f">>> Avg Legacy Originality: {avg_orig_legacy:.4f}")
            if avg_orig_bank is not None:
                print(f">>> Avg Bank Originality: {avg_orig_bank:.4f}")
            print(f">>> Avg Mixed Originality: {avg_orig:.4f}")
            print(f">>> Avg Adjusted: {avg_adj:.4f}")
            print(f">>> Avg Groundedness: {avg_ground:.4f}" if avg_ground is not None else ">>> Avg Groundedness: N/A")
            if avg_ground_novel is not None:
                print(f">>> Avg Groundedness (Novel Only): {avg_ground_novel:.4f}")
            print(f">>> Avg Penalty: {avg_penalty:.4f}")
            if dual_axis_result:
                axis_label = "UUT Dual Axis" if task_type == "UUT" else "PropConj Dual Axis"
                print(
                    f">>> {axis_label}: "
                    f"I={dual_axis_result['imagination']:.4f} "
                    f"(raw={dual_axis_result['imagination_raw']:.4f}), "
                    f"H={dual_axis_result['hallucination']:.4f} "
                    f"(raw={dual_axis_result['hallucination_raw']:.4f})"
                )
            print(
                f">>> Embedding Flex: Clusters={num_clusters}, Switches={category_switches}, "
                f"PairwiseDist={emb_pairwise_distance:.4f}, Entropy={emb_cluster_entropy:.4f}"
            )
            print(
                f">>> Ontological Flex: Cats={wn_flexibility['unique_categories']}/"
                f"{wn_flexibility.get('num_categories', wn_analyzer.num_categories)}, "
                f"Switches={wn_flexibility['category_switches']} "
                f"[{wn_flexibility.get('analysis_source', wn_analyzer.get_source_label())}]"
            )

            task_count += 1
            creative_task_valid_counts[task_type] += 1
            sum_wn_distance += wn_flexibility["avg_pairwise_wn_distance"]
            sum_wn_ic += wn_flexibility["avg_information_content"]
            all_task_cluster_ratios.append(cluster_ratio)
            all_task_emb_switch_rates.append(emb_switch_rate)
            if emb_pairwise_distance is not None:
                all_task_emb_pairwise_distances.append(emb_pairwise_distance)
            if emb_adjacent_distance is not None:
                all_task_emb_adjacent_distances.append(emb_adjacent_distance)
            if emb_cluster_entropy is not None:
                all_task_emb_cluster_entropies.append(emb_cluster_entropy)
            all_task_wn_switch_rates.append(wn_switch_rate)
            for category in wn_flexibility.get("idea_categories", []):
                if category != "Unknown":
                    global_wn_categories_set.add(category)
                    global_wn_category_counter[category] += 1

            category_originality_sums[task_type] += avg_adj
            category_originality_counts[task_type] += 1
            if avg_ground is not None:
                task_type_ground_scores[task_type].append(avg_ground)
            if avg_ground_novel is not None:
                task_type_ground_scores_novel[task_type].append(avg_ground_novel)

            task_results.append({
                "task_id": task_id,
                "task_type": task_type,
                "prompt": prompt_text,
                "prompt_metadata": prompt_metadata,
                "target_concept": target_concept,
                "repeat_index": replicate_index,
                "coverage_eligible": True,
                "excluded_from_coverage": False,
                "non_model_failure": False,
                "non_model_skip_reason": None,
                "valid_run": True,
                "invalid_reason": None,
                "generation": build_generation_record(llm_result),
                "expected_output_count": expected_output_count,
                "parsed_item_count": parsed_item_count,
                "raw_items_truncated": truncated_item_count,
                "raw_fluency": raw_fluency,
                "fluency_after_dedup": fluency,
                "duplicates_removed": num_removed,
                "zero_originality_count": task_zero_orig,
                "average_originality": round(avg_orig, 4),
                "average_originality_legacy": round(avg_orig_legacy, 4),
                "average_originality_bank": round(avg_orig_bank, 4) if avg_orig_bank is not None else None,
                "average_adjusted_originality": round(avg_adj, 4),
                "average_penalty": round(avg_penalty, 4),
                "average_groundedness": round(avg_ground, 4) if avg_ground is not None else None,
                "average_groundedness_novel_only": (
                    round(avg_ground_novel, 4) if avg_ground_novel is not None else None
                ),
                "embedding_flexibility": {
                    "clusters": num_clusters,
                    "switches": category_switches,
                    "cluster_ratio": round(cluster_ratio, 4),
                    "switch_rate": round(emb_switch_rate, 4),
                    "mean_pairwise_distance": round(emb_pairwise_distance, 4),
                    "mean_adjacent_distance": round(emb_adjacent_distance, 4),
                    "cluster_entropy": round(emb_cluster_entropy, 4),
                    "distance_threshold": 0.55,
                },
                "ontological_flexibility": {
                    "unique_categories": wn_flexibility["unique_categories"],
                    "category_switches": wn_flexibility["category_switches"],
                    "switch_rate": round(wn_switch_rate, 4),
                    "avg_pairwise_wn_distance": wn_flexibility["avg_pairwise_wn_distance"],
                    "avg_information_content": wn_flexibility["avg_information_content"],
                    "category_diversity_index": wn_flexibility["category_diversity_index"],
                    "category_distribution": wn_flexibility["category_distribution"],
                    "wordnet_coverage": wn_flexibility["wordnet_coverage"],
                    "analysis_source": wn_flexibility.get("analysis_source"),
                    "num_categories": wn_flexibility.get("num_categories"),
                },
                "groundedness": {
                    "version": WHITE_BOX_GROUNDEDNESS_VERSION,
                    "formula": WHITE_BOX_GROUNDEDNESS_VERSION,
                    "average_groundedness": round(avg_ground, 4) if avg_ground is not None else None,
                    "average_groundedness_novel_only": (
                        round(avg_ground_novel, 4)
                        if avg_ground_novel is not None else None
                    ),
                    "mean_penalty": round(avg_penalty, 4),
                    "penalty_rate": round(penalty_rate, 4),
                    "groundedness_confidence_mean": round(ground_conf_mean, 4),
                    "low_groundedness_rate": round(low_ground_rate, 4),
                    "low_groundedness_rate_novel_only": (
                        round(low_ground_rate_novel, 4)
                        if low_ground_rate_novel is not None else None
                    ),
                    "scored_coverage": round(scored_coverage, 4) if scored_coverage is not None else None,
                    "groundedness_scored_ideas": len(task_ground_values),
                },
                "dual_axis": dual_axis_result,
                "dynamic_baseline_size": len(dynamic_baseline),
                "common_answer_bank": {
                    "blend_weights": common_answer_bank_context.get("blend_weights"),
                    "static_bank_size": common_answer_bank_context["static_bank"]["size"],
                    "dynamic_bank_size": common_answer_bank_context["dynamic_bank"]["size"],
                    "combined_bank_size": common_answer_bank_context["combined_bank_size"],
                    "static_bank_preview": common_answer_bank_context["static_bank"]["preview"],
                    "dynamic_bank_preview": common_answer_bank_context["dynamic_bank"]["preview"],
                    "combined_bank_preview": common_answer_bank_context["combined_bank_preview"],
                    "static_bank_trace": common_answer_bank_context["static_bank"]["trace"],
                    "dynamic_bank_trace": common_answer_bank_context["dynamic_bank"]["trace"],
                    "formula": "legacy + common-answer-bank rarity blend before groundedness penalty",
                },
                "details": task_details,
            })

    if "CJST" in dataset:
        print(f"\n{'=' * 60}\n  Counterfactual Just-Suppose Dual Axis (CJST) [{model_name}] [repeat {repeat_label}]\n{'=' * 60}")
        expected_output_count = get_expected_output_count("CJST") or CJST_OUTPUT_COUNT
        for task in dataset["CJST"]:
            task_id = task["id"]
            prompt_text = task.get("prompt")
            prompt_metadata = {k: v for k, v in task.items() if k not in {"id", "prompt"}}
            target_concept = task.get("scenario") or task.get("scenario_text") or task_id
            print(f"\n[CJST Target]: {target_concept}")

            llm_result = call_llm(
                client,
                prompt_text,
                model_name,
                task_label="CJST",
                max_tokens_override=get_task_max_tokens("CJST"),
                seed=stable_seed(model_name, "CJST", task_id, replicate_index),
            )
            resolved_model = llm_result["resolved_model"]
            resolved_models_seen.add(resolved_model)
            raw_output = llm_result["content"]
            parsed_items_all = parse_creative_items(raw_output, task_type="CJST")
            parsed_items = parsed_items_all[:expected_output_count]
            truncated_item_count = max(0, len(parsed_items_all) - expected_output_count)
            parsed_item_count = len(parsed_items)
            tier_counts = cjst_scorer.tier_counts(parsed_items)
            non_model_skip_reason = get_non_model_skip_reason(llm_result)
            invalid_reason = None
            score_result = None

            if non_model_skip_reason is not None:
                cjst_excluded_count += 1
                non_model_skip_counts["cjst"] += 1
                non_model_skip_reasons[non_model_skip_reason] += 1
                print(f"[Coverage Excluded] {non_model_skip_reason}")
            else:
                cjst_effective_prompts += 1
                if not raw_output.strip():
                    invalid_reason = "empty_response"
                elif parsed_item_count == 0:
                    invalid_reason = "parsed_zero_items"
                elif parsed_item_count < MIN_CJST_ITEMS_PER_TASK:
                    invalid_reason = "insufficient_cjst_items"
                elif any(tier_counts.get(tier, 0) < MIN_CJST_ITEMS_PER_TIER for tier in sorted(tier_counts)):
                    invalid_reason = "insufficient_cjst_tier_coverage"

                if parsed_item_count > 0:
                    score_result = cjst_scorer.score_task(
                        task,
                        parsed_items,
                        semantic_scorer=scorer,
                        groundedness_scorer=groundedness_scorer,
                        expected_output_count=expected_output_count,
                    )

                if invalid_reason is None:
                    cjst_scorable_count += 1
                    if score_result is not None:
                        cjst_dual_axis_task_scores.append(score_result)
                else:
                    invalid_run_counts["cjst"] += 1
                    print(f"[Invalid Run] {invalid_reason}")

            if score_result is not None and score_result.get("imagination") is not None:
                print(
                    f">>> CJST: items={parsed_item_count}, tiers={tier_counts}, "
                    f"I={score_result['imagination']:.4f} "
                    f"(raw={score_result['imagination_raw']:.4f}), "
                    f"H={score_result['hallucination']:.4f} "
                    f"(raw={score_result['hallucination_raw']:.4f})"
                )
            elif non_model_skip_reason is None:
                print(f">>> CJST: items={parsed_item_count}, tiers={tier_counts}, score=N/A")

            cjst_task_result = {
                "task_id": task_id,
                "task_type": "CJST",
                "prompt": prompt_text,
                "prompt_metadata": prompt_metadata,
                "target_concept": target_concept,
                "repeat_index": replicate_index,
                "coverage_eligible": non_model_skip_reason is None,
                "excluded_from_coverage": non_model_skip_reason is not None,
                "non_model_failure": non_model_skip_reason is not None,
                "non_model_skip_reason": non_model_skip_reason,
                "valid_run": invalid_reason is None and non_model_skip_reason is None,
                "invalid_reason": invalid_reason,
                "scorable": invalid_reason is None and non_model_skip_reason is None and score_result is not None,
                "generation": build_generation_record(llm_result),
                "expected_output_count": expected_output_count,
                "parsed_item_count": parsed_item_count,
                "raw_items_truncated": truncated_item_count,
                "tier_counts": tier_counts,
                "dual_axis": score_result,
                "details": [
                    {
                        "parsed_item": parsed_item,
                        "score": score_detail,
                    }
                    for parsed_item, score_detail in zip(
                        parsed_items,
                        (score_result or {}).get("details") or []
                    )
                ],
            }
            cjst_results.append(cjst_task_result)
            task_results.append(cjst_task_result)

    if "MacGyver" in dataset:
        print(f"\n{'=' * 60}\n  MacGyver Constrained Problem Solving [{model_name}] [repeat {repeat_label}]\n{'=' * 60}")
        expected_output_count = get_expected_output_count("MacGyver") or MACGYVER_OUTPUT_COUNT
        for task in dataset["MacGyver"]:
            task_id = task["id"]
            is_boundary_diagnostic = bool(task.get("macgyver_boundary_diagnostic"))
            prompt_text = task.get("prompt")
            prompt_metadata = {k: v for k, v in task.items() if k not in {"id", "prompt"}}
            target_concept = task.get("goal") or task.get("scene") or task_id
            diagnostic_suffix = " [boundary diagnostic]" if is_boundary_diagnostic else ""
            print(f"\n[MacGyver Target{diagnostic_suffix}]: {target_concept}")

            llm_result = call_llm(
                client,
                prompt_text,
                model_name,
                task_label="MacGyver",
                max_tokens_override=get_task_max_tokens("MacGyver"),
                seed=stable_seed(model_name, "MacGyver", task_id, replicate_index),
            )
            resolved_model = llm_result["resolved_model"]
            resolved_models_seen.add(resolved_model)
            raw_output = llm_result["content"]
            parsed_response = macgyver_scorer.parse_response(raw_output)
            returned_plan_count = len(parsed_response.get("plans") or [])
            non_model_skip_reason = get_non_model_skip_reason(llm_result)
            invalid_reason = None
            score_result = None

            if non_model_skip_reason is not None:
                if is_boundary_diagnostic:
                    macgyver_boundary_excluded_count += 1
                    non_model_skip_counts["macgyver_boundary"] += 1
                else:
                    macgyver_excluded_count += 1
                    non_model_skip_counts["macgyver"] += 1
                non_model_skip_reasons[non_model_skip_reason] += 1
                print(f"[Coverage Excluded] {non_model_skip_reason}")
            else:
                if is_boundary_diagnostic:
                    macgyver_boundary_effective_prompts += 1
                else:
                    macgyver_effective_prompts += 1
                if not raw_output.strip():
                    invalid_reason = "empty_response"
                elif not parsed_response.get("parse_valid"):
                    invalid_reason = parsed_response.get("parse_error") or "parsed_no_json_object"
                else:
                    response_says_unsolvable = parsed_response.get("solvability") == "unsolvable"
                    response_says_boundary = parsed_response.get("solvability") in {"unsolvable", "needs_clarification"}
                    task_unsolvable = bool(task.get("unsolvable"))
                    task_expected_mode = str(task.get("expected_response_mode") or ("unsolvable" if task_unsolvable else "solvable"))
                    if (
                        task_expected_mode != "needs_clarification"
                        and not response_says_boundary
                        and not task_unsolvable
                        and returned_plan_count < MIN_MACGYVER_PLANS_PER_TASK
                    ):
                        invalid_reason = "insufficient_plans"
                    elif not response_says_unsolvable and task_unsolvable and returned_plan_count == 0:
                        invalid_reason = "solvable_claim_without_plans"

                if parsed_response.get("parse_valid"):
                    score_result = macgyver_scorer.score_task(task, parsed_response)

                if invalid_reason is None:
                    if is_boundary_diagnostic:
                        macgyver_boundary_scorable_count += 1
                    else:
                        macgyver_scorable_count += 1
                    if score_result is not None:
                        if is_boundary_diagnostic:
                            macgyver_boundary_task_scores.append(score_result)
                        else:
                            macgyver_dual_axis_task_scores.append(score_result)
                else:
                    invalid_run_counts["macgyver_boundary" if is_boundary_diagnostic else "macgyver"] += 1
                    print(f"[Invalid Run] {invalid_reason}")

            if score_result is not None:
                imag_text = (
                    f"{score_result['imagination']:.4f}"
                    if score_result.get("imagination") is not None else "excluded"
                )
                print(
                    f">>> MacGyver: plans={returned_plan_count}, "
                    f"I={imag_text} (raw={score_result['imagination_raw']:.4f}), "
                    f"H={score_result['hallucination']:.4f} "
                    f"(raw={score_result['hallucination_raw']:.4f}), "
                    f"solvability_correct={score_result.get('solvability_correct')}"
                )
            elif non_model_skip_reason is None:
                print(">>> MacGyver: N/A")

            macgyver_task_result = {
                "task_id": task_id,
                "task_type": "MacGyver",
                "macgyver_boundary_diagnostic": is_boundary_diagnostic,
                "prompt": prompt_text,
                "prompt_metadata": prompt_metadata,
                "target_concept": target_concept,
                "repeat_index": replicate_index,
                "coverage_eligible": non_model_skip_reason is None,
                "excluded_from_coverage": non_model_skip_reason is not None,
                "non_model_failure": non_model_skip_reason is not None,
                "non_model_skip_reason": non_model_skip_reason,
                "valid_run": invalid_reason is None and non_model_skip_reason is None,
                "invalid_reason": invalid_reason,
                "scorable": invalid_reason is None and non_model_skip_reason is None and score_result is not None,
                "generation": build_generation_record(llm_result),
                "expected_output_count": expected_output_count,
                "parsed_item_count": returned_plan_count,
                "returned_plan_count": returned_plan_count,
                "solvability": parsed_response.get("solvability"),
                "impossibility_reason": parsed_response.get("impossibility_reason"),
                "parsed_response": {
                    key: value
                    for key, value in parsed_response.items()
                    if key != "raw_payload"
                },
                "dual_axis": score_result,
            }
            if is_boundary_diagnostic:
                macgyver_boundary_results.append(macgyver_task_result)
            else:
                macgyver_results.append(macgyver_task_result)
            task_results.append(macgyver_task_result)

    if "HypoUseSpace" in dataset:
        print(f"\n{'=' * 60}\n  HypoUseSpace Closed-World Hypothesis Set [{model_name}] [repeat {repeat_label}]\n{'=' * 60}")
        expected_output_count = get_expected_output_count("HypoUseSpace") or HYPOUSESPACE_OUTPUT_COUNT
        for task in dataset["HypoUseSpace"]:
            task_id = task["id"]
            prompt_text = task.get("prompt")
            prompt_metadata = {k: v for k, v in task.items() if k not in {"id", "prompt"}}
            target_concept = task.get("title") or task.get("goal") or task_id
            valid_space_size = len(task.get("valid_hypotheses") or [])
            task_no_valid = bool(task.get("no_valid_hypothesis")) or valid_space_size == 0
            is_boundary_diagnostic = bool(task.get("hypospace_boundary_diagnostic"))
            counter_key = "hypospace_boundary" if is_boundary_diagnostic else "hypospace"
            task_expected_count = min(expected_output_count, valid_space_size) if valid_space_size else expected_output_count
            minimum_valid_hypotheses = min(
                MIN_HYPOUSESPACE_ITEMS_PER_TASK,
                task_expected_count,
                max(1, valid_space_size),
            )
            print(f"\n[HypoUseSpace Target]: {target_concept}")

            llm_result = call_llm(
                client,
                prompt_text,
                model_name,
                task_label="HypoUseSpace",
                max_tokens_override=get_task_max_tokens("HypoUseSpace"),
                seed=stable_seed(model_name, "HypoUseSpace", task_id, replicate_index),
            )
            resolved_model = llm_result["resolved_model"]
            resolved_models_seen.add(resolved_model)
            raw_output = llm_result["content"]
            parsed_response = hypospace_scorer.parse_response(raw_output)
            returned_hypothesis_count = len(parsed_response.get("hypotheses") or [])
            truncated_hypothesis_count = max(0, returned_hypothesis_count - task_expected_count)
            non_model_skip_reason = get_non_model_skip_reason(llm_result)
            invalid_reason = None
            score_result = None

            if non_model_skip_reason is not None:
                if is_boundary_diagnostic:
                    hypospace_boundary_excluded_count += 1
                else:
                    hypospace_excluded_count += 1
                non_model_skip_counts[counter_key] += 1
                non_model_skip_reasons[non_model_skip_reason] += 1
                print(f"[Coverage Excluded] {non_model_skip_reason}")
            else:
                if is_boundary_diagnostic:
                    hypospace_boundary_effective_prompts += 1
                else:
                    hypospace_effective_prompts += 1
                if not raw_output.strip():
                    invalid_reason = "empty_response"
                elif not parsed_response.get("parse_valid"):
                    invalid_reason = parsed_response.get("parse_error") or "parsed_no_json_payload"
                else:
                    response_no_valid = bool(parsed_response.get("no_valid_hypothesis"))
                    if (
                        not task_no_valid
                        and not response_no_valid
                        and returned_hypothesis_count < minimum_valid_hypotheses
                    ):
                        invalid_reason = "insufficient_hypotheses"

                if parsed_response.get("parse_valid"):
                    score_result = hypospace_scorer.score_task(
                        task,
                        parsed_response,
                        expected_output_count=task_expected_count,
                    )

                if invalid_reason is None:
                    if is_boundary_diagnostic:
                        hypospace_boundary_scorable_count += 1
                    else:
                        hypospace_scorable_count += 1
                    if score_result is not None:
                        if is_boundary_diagnostic:
                            hypospace_boundary_task_scores.append(score_result)
                        else:
                            hypospace_dual_axis_task_scores.append(score_result)
                else:
                    invalid_run_counts[counter_key] += 1
                    print(f"[Invalid Run] {invalid_reason}")

            if score_result is not None:
                imag_text = (
                    f"{score_result['imagination']:.4f}"
                    if score_result.get("imagination") is not None else "excluded"
                )
                print(
                    f">>> HypoUseSpace: hypotheses={returned_hypothesis_count}, "
                    f"recovered={score_result.get('recovered_unique_count')}/{score_result.get('budget')}, "
                    f"I={imag_text} (raw={score_result['imagination_raw']:.4f}), "
                    f"H={score_result['hallucination']:.4f} "
                    f"(raw={score_result['hallucination_raw']:.4f})"
                )
            elif non_model_skip_reason is None:
                print(f">>> HypoUseSpace: hypotheses={returned_hypothesis_count}, score=N/A")

            hypospace_task_result = {
                "task_id": task_id,
                "task_type": "HypoUseSpace",
                "hypospace_boundary_diagnostic": is_boundary_diagnostic,
                "prompt": prompt_text,
                "prompt_metadata": prompt_metadata,
                "target_concept": target_concept,
                "repeat_index": replicate_index,
                "coverage_eligible": non_model_skip_reason is None,
                "excluded_from_coverage": non_model_skip_reason is not None,
                "non_model_failure": non_model_skip_reason is not None,
                "non_model_skip_reason": non_model_skip_reason,
                "valid_run": invalid_reason is None and non_model_skip_reason is None,
                "invalid_reason": invalid_reason,
                "scorable": invalid_reason is None and non_model_skip_reason is None and score_result is not None,
                "generation": build_generation_record(llm_result),
                "expected_output_count": task_expected_count,
                "minimum_valid_hypotheses": minimum_valid_hypotheses if not task_no_valid else 0,
                "valid_space_size": valid_space_size,
                "no_valid_task": task_no_valid,
                "parsed_item_count": returned_hypothesis_count,
                "raw_items_truncated": truncated_hypothesis_count,
                "no_valid_hypothesis": parsed_response.get("no_valid_hypothesis"),
                "reason": parsed_response.get("reason"),
                "parsed_response": {
                    key: value
                    for key, value in parsed_response.items()
                    if key != "raw_payload"
                },
                "dual_axis": score_result,
            }
            if is_boundary_diagnostic:
                hypospace_boundary_results.append(hypospace_task_result)
            else:
                hypospace_results.append(hypospace_task_result)
            task_results.append(hypospace_task_result)

    if "GCW" in dataset:
        print(f"\n{'=' * 60}\n  Grounded Creative Writing Dual Axis (GCW) [{model_name}] [repeat {repeat_label}]\n{'=' * 60}")
        expected_output_count = get_expected_output_count("GCW") or GCW_BEAT_COUNT
        minimum_valid_beats = max(MIN_GCW_BEATS_PER_TASK, int(expected_output_count * 0.67))
        for task in dataset["GCW"]:
            task_id = task["id"]
            prompt_text = task.get("prompt")
            prompt_metadata = {k: v for k, v in task.items() if k not in {"id", "prompt"}}
            target_concept = task.get("title") or task_id
            print(f"\n[GCW Card]: {target_concept}")

            llm_result = call_llm(
                client,
                prompt_text,
                model_name,
                task_label="GCW",
                max_tokens_override=get_task_max_tokens("GCW"),
                seed=stable_seed(model_name, "GCW", task_id, replicate_index),
            )
            resolved_model = llm_result["resolved_model"]
            resolved_models_seen.add(resolved_model)
            raw_output = llm_result["content"]
            parsed_response = gcw_scorer.parse_response(raw_output)
            returned_beat_count = len(parsed_response.get("beats") or [])
            truncated_beat_count = max(0, returned_beat_count - expected_output_count)
            non_model_skip_reason = get_non_model_skip_reason(llm_result)
            invalid_reason = None
            score_result = None

            if non_model_skip_reason is not None:
                gcw_excluded_count += 1
                non_model_skip_counts["gcw"] += 1
                non_model_skip_reasons[non_model_skip_reason] += 1
                print(f"[Coverage Excluded] {non_model_skip_reason}")
            else:
                gcw_effective_prompts += 1
                if not raw_output.strip():
                    invalid_reason = "empty_response"
                elif not parsed_response.get("parse_valid"):
                    invalid_reason = parsed_response.get("parse_error") or "parsed_no_json_object"
                elif returned_beat_count < minimum_valid_beats:
                    invalid_reason = "insufficient_gcw_beats"

                if parsed_response.get("parse_valid"):
                    score_result = gcw_scorer.score_task(
                        task,
                        parsed_response,
                        semantic_scorer=scorer,
                        expected_beat_count=expected_output_count,
                    )

                if invalid_reason is None:
                    gcw_scorable_count += 1
                    if score_result is not None:
                        gcw_dual_axis_task_scores.append(score_result)
                else:
                    invalid_run_counts["gcw"] += 1
                    print(f"[Invalid Run] {invalid_reason}")

            if score_result is not None:
                imag_text = (
                    f"{score_result['imagination']:.4f}"
                    if score_result.get("imagination") is not None else "excluded"
                )
                print(
                    f">>> GCW: beats={returned_beat_count}, "
                    f"I={imag_text} (raw={score_result['imagination_raw']:.4f}), "
                    f"H={score_result['hallucination']:.4f} "
                    f"(raw={score_result['hallucination_raw']:.4f})"
                )
            elif non_model_skip_reason is None:
                print(f">>> GCW: beats={returned_beat_count}, score=N/A")

            gcw_task_result = {
                "task_id": task_id,
                "task_type": "GCW",
                "prompt": prompt_text,
                "prompt_metadata": prompt_metadata,
                "target_concept": target_concept,
                "repeat_index": replicate_index,
                "coverage_eligible": non_model_skip_reason is None,
                "excluded_from_coverage": non_model_skip_reason is not None,
                "non_model_failure": non_model_skip_reason is not None,
                "non_model_skip_reason": non_model_skip_reason,
                "valid_run": invalid_reason is None and non_model_skip_reason is None,
                "invalid_reason": invalid_reason,
                "scorable": invalid_reason is None and non_model_skip_reason is None and score_result is not None,
                "generation": build_generation_record(llm_result),
                "expected_output_count": expected_output_count,
                "minimum_valid_beats": minimum_valid_beats,
                "parsed_item_count": returned_beat_count,
                "raw_items_truncated": truncated_beat_count,
                "parsed_response": {
                    key: value
                    for key, value in parsed_response.items()
                    if key != "raw_payload"
                },
                "dual_axis": score_result,
            }
            gcw_results.append(gcw_task_result)
            task_results.append(gcw_task_result)

    if "NeoCoder" in dataset:
        print(f"\n{'=' * 60}\n  NeoCoder Executable Code Creativity (enhanced) [{model_name}] [repeat {repeat_label}]\n{'=' * 60}")
        expected_output_count = get_expected_output_count("NeoCoder") or NEOCODER_OUTPUT_COUNT
        for task in dataset["NeoCoder"]:
            task_id = task["id"]
            prompt_text = task.get("prompt")
            prompt_metadata = {k: v for k, v in task.items() if k not in {"id", "prompt"}}
            target_concept = task.get("title") or task.get("function_name") or task_id
            print(f"\n[NeoCoder Task]: {target_concept}")

            llm_result = call_llm(
                client,
                prompt_text,
                model_name,
                task_label="NeoCoder",
                max_tokens_override=get_task_max_tokens("NeoCoder"),
                seed=stable_seed(model_name, "NeoCoder", task_id, replicate_index),
            )
            resolved_model = llm_result["resolved_model"]
            resolved_models_seen.add(resolved_model)
            raw_output = llm_result["content"]
            parsed_response = neocoder_scorer.parse_response(raw_output)
            returned_code_chars = len(parsed_response.get("code") or "")
            non_model_skip_reason = get_non_model_skip_reason(llm_result)
            invalid_reason = None
            score_result = None

            if non_model_skip_reason is not None:
                neocoder_excluded_count += 1
                non_model_skip_counts["neocoder"] += 1
                non_model_skip_reasons[non_model_skip_reason] += 1
                print(f"[Coverage Excluded] {non_model_skip_reason}")
            else:
                neocoder_effective_prompts += 1
                if not raw_output.strip():
                    invalid_reason = "empty_response"
                elif not parsed_response.get("parse_valid"):
                    invalid_reason = parsed_response.get("parse_error") or "parsed_no_code"

                if parsed_response.get("parse_valid"):
                    score_result = neocoder_scorer.score_task(task, parsed_response)

                if invalid_reason is None:
                    neocoder_scorable_count += 1
                    if score_result is not None:
                        neocoder_dual_axis_task_scores.append(score_result)
                else:
                    invalid_run_counts["neocoder"] += 1
                    print(f"[Invalid Run] {invalid_reason}")

            if score_result is not None:
                print(
                    f">>> NeoCoder: code_chars={returned_code_chars}, "
                    f"pass_rate={score_result['pass_rate']:.4f}, "
                    f"I={score_result['imagination']:.4f} "
                    f"(raw={score_result['imagination_raw']:.4f}, gated={score_result['imagination_gated']:.4f}), "
                    f"H={score_result['hallucination']:.4f} "
                    f"(raw={score_result['hallucination_raw']:.4f})"
                )
            elif non_model_skip_reason is None:
                print(f">>> NeoCoder: code_chars={returned_code_chars}, score=N/A")

            neocoder_task_result = {
                "task_id": task_id,
                "task_type": "NeoCoder",
                "prompt": prompt_text,
                "prompt_metadata": prompt_metadata,
                "target_concept": target_concept,
                "repeat_index": replicate_index,
                "coverage_eligible": non_model_skip_reason is None,
                "excluded_from_coverage": non_model_skip_reason is not None,
                "non_model_failure": non_model_skip_reason is not None,
                "non_model_skip_reason": non_model_skip_reason,
                "valid_run": invalid_reason is None and non_model_skip_reason is None,
                "invalid_reason": invalid_reason,
                "scorable": invalid_reason is None and non_model_skip_reason is None and score_result is not None,
                "generation": build_generation_record(llm_result),
                "expected_output_count": expected_output_count,
                "parsed_item_count": 1 if parsed_response.get("parse_valid") else 0,
                "returned_code_chars": returned_code_chars,
                "parsed_response": {
                    key: value
                    for key, value in parsed_response.items()
                    if key != "raw_payload"
                },
                "dual_axis": score_result,
            }
            neocoder_results.append(neocoder_task_result)
            task_results.append(neocoder_task_result)

    if "ClosedWorldFact" in dataset:
        print(f"\n{'=' * 60}\n  ClosedWorldFact Hallucination Calibration [{model_name}] [repeat {repeat_label}]\n{'=' * 60}")
        expected_output_count = get_expected_output_count("ClosedWorldFact") or CLOSED_WORLD_FACT_OUTPUT_COUNT
        for task in dataset["ClosedWorldFact"]:
            task_id = task["id"]
            prompt_text = task.get("prompt")
            prompt_metadata = {k: v for k, v in task.items() if k not in {"id", "prompt", "database"}}
            target_concept = task.get("question") or task_id
            print(f"\n[ClosedWorldFact Question]: {target_concept}")

            llm_result = call_llm(
                client,
                prompt_text,
                model_name,
                task_label="ClosedWorldFact",
                max_tokens_override=get_task_max_tokens("ClosedWorldFact"),
                seed=stable_seed(model_name, "ClosedWorldFact", task_id, replicate_index),
            )
            resolved_model = llm_result["resolved_model"]
            resolved_models_seen.add(resolved_model)
            raw_output = llm_result["content"]
            parsed_response = closed_world_fact_scorer.parse_response(raw_output)
            non_model_skip_reason = get_non_model_skip_reason(llm_result)
            invalid_reason = None
            score_result = None

            if non_model_skip_reason is not None:
                closed_world_fact_excluded_count += 1
                non_model_skip_counts["closed_world_fact"] += 1
                non_model_skip_reasons[non_model_skip_reason] += 1
                print(f"[Coverage Excluded] {non_model_skip_reason}")
            else:
                closed_world_fact_effective_prompts += 1
                if not raw_output.strip():
                    invalid_reason = "empty_response"
                elif not parsed_response.get("parse_valid"):
                    invalid_reason = parsed_response.get("parse_error") or "parsed_no_answer"

                if parsed_response.get("parse_valid"):
                    score_result = closed_world_fact_scorer.score_task(task, parsed_response)

                if invalid_reason is None:
                    closed_world_fact_scorable_count += 1
                    if score_result is not None:
                        closed_world_fact_task_scores.append(score_result)
                else:
                    invalid_run_counts["closed_world_fact"] += 1
                    print(f"[Invalid Run] {invalid_reason}")

            if score_result is not None:
                print(
                    f">>> ClosedWorldFact: H={score_result['hallucination']:.4f} "
                    f"(fact={score_result['H_fact']:.4f}, logic={score_result['H_logic']:.4f}, "
                    f"boundary={score_result['H_boundary']:.4f}), score={score_result['score']:.4f}"
                )
            elif non_model_skip_reason is None:
                print(">>> ClosedWorldFact: score=N/A")

            closed_world_fact_task_result = {
                "task_id": task_id,
                "task_type": "ClosedWorldFact",
                "prompt": prompt_text,
                "prompt_metadata": prompt_metadata,
                "target_concept": target_concept,
                "repeat_index": replicate_index,
                "coverage_eligible": non_model_skip_reason is None,
                "excluded_from_coverage": non_model_skip_reason is not None,
                "non_model_failure": non_model_skip_reason is not None,
                "non_model_skip_reason": non_model_skip_reason,
                "valid_run": invalid_reason is None and non_model_skip_reason is None,
                "invalid_reason": invalid_reason,
                "scorable": invalid_reason is None and non_model_skip_reason is None and score_result is not None,
                "generation": build_generation_record(llm_result),
                "expected_output_count": expected_output_count,
                "parsed_item_count": 1 if parsed_response.get("parse_valid") else 0,
                "parsed_response": {
                    key: value
                    for key, value in parsed_response.items()
                    if key != "raw_payload"
                },
                "calibration": score_result,
            }
            closed_world_fact_results.append(closed_world_fact_task_result)
            task_results.append(closed_world_fact_task_result)

    if "AnalogyTransfer" in dataset:
        print(f"\n{'=' * 60}\n  AnalogyTransfer False-Transfer Challenge [{model_name}] [repeat {repeat_label}]\n{'=' * 60}")
        expected_output_count = get_expected_output_count("AnalogyTransfer") or ANALOGY_TRANSFER_OUTPUT_COUNT
        for task in dataset["AnalogyTransfer"]:
            task_id = task["id"]
            prompt_text = task.get("prompt")
            prompt_metadata = {k: v for k, v in task.items() if k not in {"id", "prompt", "cluster"}}
            target_concept = f"{task.get('source_domain')} -> {task.get('target_domain')} ({task.get('variant')})"
            print(f"\n[AnalogyTransfer Task]: {target_concept}")

            llm_result = call_llm(
                client,
                prompt_text,
                model_name,
                task_label="AnalogyTransfer",
                max_tokens_override=get_task_max_tokens("AnalogyTransfer"),
                seed=stable_seed(model_name, "AnalogyTransfer", task_id, replicate_index),
            )
            resolved_model = llm_result["resolved_model"]
            resolved_models_seen.add(resolved_model)
            raw_output = llm_result["content"]
            parsed_response = analogy_transfer_scorer.parse_response(raw_output)
            non_model_skip_reason = get_non_model_skip_reason(llm_result)
            invalid_reason = None
            score_result = None

            if non_model_skip_reason is not None:
                analogy_transfer_excluded_count += 1
                non_model_skip_counts["analogy_transfer"] += 1
                non_model_skip_reasons[non_model_skip_reason] += 1
                print(f"[Coverage Excluded] {non_model_skip_reason}")
            else:
                analogy_transfer_effective_prompts += 1
                if not raw_output.strip():
                    invalid_reason = "empty_response"
                elif not parsed_response.get("parse_valid"):
                    invalid_reason = parsed_response.get("parse_error") or "parsed_no_analogy"

                if parsed_response.get("parse_valid"):
                    score_result = analogy_transfer_scorer.score_task(task, parsed_response)

                if invalid_reason is None:
                    analogy_transfer_scorable_count += 1
                    if score_result is not None:
                        analogy_transfer_task_scores.append(score_result)
                else:
                    invalid_run_counts["analogy_transfer"] += 1
                    print(f"[Invalid Run] {invalid_reason}")

            if score_result is not None:
                print(
                    f">>> AnalogyTransfer: I={score_result['imagination']:.4f} "
                    f"(raw={score_result['imagination_raw']:.4f}, gated={score_result['imagination_gated']:.4f}), "
                    f"H={score_result['hallucination']:.4f} "
                    f"(false_transfer={score_result['H_false_transfer']:.4f}, fact={score_result['H_fact']:.4f}, "
                    f"logic={score_result['H_logic']:.4f}, context={score_result['H_context']:.4f})"
                )
            elif non_model_skip_reason is None:
                print(">>> AnalogyTransfer: score=N/A")

            analogy_transfer_task_result = {
                "task_id": task_id,
                "task_type": "AnalogyTransfer",
                "prompt": prompt_text,
                "prompt_metadata": prompt_metadata,
                "target_concept": target_concept,
                "repeat_index": replicate_index,
                "coverage_eligible": non_model_skip_reason is None,
                "excluded_from_coverage": non_model_skip_reason is not None,
                "non_model_failure": non_model_skip_reason is not None,
                "non_model_skip_reason": non_model_skip_reason,
                "valid_run": invalid_reason is None and non_model_skip_reason is None,
                "invalid_reason": invalid_reason,
                "scorable": invalid_reason is None and non_model_skip_reason is None and score_result is not None,
                "generation": build_generation_record(llm_result),
                "expected_output_count": expected_output_count,
                "parsed_item_count": 1 if parsed_response.get("parse_valid") else 0,
                "parsed_response": {
                    key: value
                    for key, value in parsed_response.items()
                    if key != "raw_payload"
                },
                "challenge": score_result,
            }
            analogy_transfer_results.append(analogy_transfer_task_result)
            task_results.append(analogy_transfer_task_result)

    dat_results = []
    cdat_results = []
    ff_results = []
    dat_mean = None
    cdat_score = None
    mean_nov = None
    mean_app = None
    gate_pass_rate = None
    mean_app_gain = None
    mean_multiplier = None
    ff_mean = None

    if "DAT" in dataset:
        print(f"\n{'=' * 60}\n  Divergent Association Task (DAT) [{model_name}] [repeat {repeat_label}]\n{'=' * 60}")
        dat_scores = []
        raw_dat_scores = []
        expected_output_count = get_expected_output_count("DAT") or DAT_OUTPUT_COUNT
        for task in dataset["DAT"]:
            task_id = task["id"]
            llm_result = call_llm(
                client,
                task["prompt"],
                model_name,
                task_label="DAT",
                max_tokens_override=get_task_max_tokens("DAT"),
                seed=stable_seed(model_name, "DAT", task_id, replicate_index),
            )
            resolved_model = llm_result["resolved_model"]
            resolved_models_seen.add(resolved_model)
            raw_output = llm_result["content"]
            parsed_candidates_all = dat_scorer_obj.parse_word_list(raw_output)
            parsed_candidates = parsed_candidates_all[:expected_output_count]
            truncated_item_count = max(0, len(parsed_candidates_all) - expected_output_count)
            valid_words = []
            invalid_reason = None
            non_model_skip_reason = get_non_model_skip_reason(llm_result)

            if non_model_skip_reason is not None:
                dat_excluded_count += 1
                non_model_skip_counts["dat"] += 1
                non_model_skip_reasons[non_model_skip_reason] += 1
                print(f"  [{task_id}] Coverage excluded: {non_model_skip_reason}")
            else:
                dat_effective_prompts += 1
                if not raw_output.strip():
                    invalid_reason = "empty_response"
                elif len(parsed_candidates) == 0:
                    invalid_reason = "parsed_zero_items"
                else:
                    valid_words = select_valid_dat_words(
                        dat_scorer_obj,
                        parsed_candidates,
                        max_words=dat_scorer_obj.N_SCORING_WORDS,
                    )
                    if len(valid_words) == 0:
                        invalid_reason = "valid_word_count_zero"
                    elif len(valid_words) < dat_scorer_obj.MIN_REQUIRED_WORDS:
                        invalid_reason = "insufficient_valid_words"

            raw_dat_score = None
            dat_enhancement = {
                "raw_dat_score": None,
                "supersense_count": None,
                "supersense_coverage": None,
                "enhanced_dat_score": None,
                "formula": None,
            }
            score = None
            if invalid_reason is None and non_model_skip_reason is None:
                raw_dat_score = dat_scorer_obj.compute_dat_score(valid_words)
                dat_enhancement = dat_scorer_obj.compute_enhanced_dat_score(
                    valid_words,
                    raw_dat=raw_dat_score,
                )
                score = dat_enhancement.get("enhanced_dat_score")
                if score is None:
                    score = raw_dat_score
            print(f"\n  [{task_id}] Parsed {len(parsed_candidates_all)} -> using {len(parsed_candidates)} candidates -> {len(valid_words)} valid: {valid_words}")
            if invalid_reason is not None:
                invalid_run_counts["dat"] += 1
                print(f"  [{task_id}] Invalid run: {invalid_reason}")
            if score is not None:
                dat_scores.append(score)
                if raw_dat_score is not None:
                    raw_dat_scores.append(raw_dat_score)
                dat_scorable_count += 1
                print(
                    f"  [{task_id}] DAT Score: {score:.2f} "
                    f"(raw={raw_dat_score:.2f}, "
                    f"supersense={dat_enhancement.get('supersense_count')}/"
                    f"{dat_scorer_obj.NOUN_SUPERSENSE_COUNT})"
                )
            elif non_model_skip_reason is None:
                print(f"  [{task_id}] DAT Score: N/A")

            dat_results.append({
                "task_id": task_id,
                "prompt": task.get("prompt"),
                "repeat_index": replicate_index,
                "coverage_eligible": non_model_skip_reason is None,
                "excluded_from_coverage": non_model_skip_reason is not None,
                "non_model_failure": non_model_skip_reason is not None,
                "non_model_skip_reason": non_model_skip_reason,
                "valid_run": invalid_reason is None and non_model_skip_reason is None,
                "invalid_reason": invalid_reason,
                "scorable": score is not None,
                "generation": build_generation_record(llm_result),
                "expected_output_count": expected_output_count,
                "parsed_item_count": len(parsed_candidates_all),
                "raw_items_truncated": truncated_item_count,
                "returned_item_count": len(parsed_candidates),
                "valid_words": valid_words,
                "valid_word_count": len(valid_words),
                "raw_dat_score": raw_dat_score,
                "supersense_count": dat_enhancement.get("supersense_count"),
                "supersense_coverage": dat_enhancement.get("supersense_coverage"),
                "enhanced_dat_score": dat_enhancement.get("enhanced_dat_score"),
                "dat_score_formula": dat_enhancement.get("formula"),
                "dat_score": score,
            })
        if dat_scores:
            dat_mean = sum(dat_scores) / len(dat_scores)
            print(f"\n  >>> DAT Mean: {dat_mean:.2f} (LLM-internal benchmark only)")

    if "CDAT" in dataset:
        print(f"\n{'=' * 60}\n  Conditional DAT (CDAT) [{model_name}] [repeat {repeat_label}]\n{'=' * 60}")
        cdat_novelties = []
        cdat_apps = []
        cdat_gates = []
        cdat_app_gains = []
        cdat_multipliers = []
        cdat_continuous_scores = []
        expected_output_count = get_expected_output_count("CDAT") or DAT_OUTPUT_COUNT
        for task in dataset["CDAT"]:
            task_id = task["id"]
            cue = task["cue"]
            print(f"\n  [{task_id}] Cue: \"{cue}\"")
            llm_result = call_llm(
                client,
                task["prompt"],
                model_name,
                task_label="CDAT",
                max_tokens_override=get_task_max_tokens("CDAT"),
                seed=stable_seed(model_name, "CDAT", task_id, replicate_index),
            )
            resolved_model = llm_result["resolved_model"]
            resolved_models_seen.add(resolved_model)
            raw_output = llm_result["content"]
            parsed_candidates_all = dat_scorer_obj.parse_word_list(raw_output)
            parsed_candidates = parsed_candidates_all[:expected_output_count]
            truncated_item_count = max(0, len(parsed_candidates_all) - expected_output_count)
            valid_words = []
            invalid_reason = None
            non_model_skip_reason = get_non_model_skip_reason(llm_result)
            if non_model_skip_reason is not None:
                cdat_excluded_count += 1
                non_model_skip_counts["cdat"] += 1
                non_model_skip_reasons[non_model_skip_reason] += 1
                print(f"  [{task_id}] Coverage excluded: {non_model_skip_reason}")
            else:
                cdat_effective_prompts += 1
                if not raw_output.strip():
                    invalid_reason = "empty_response"
                elif len(parsed_candidates) == 0:
                    invalid_reason = "parsed_zero_items"
                else:
                    valid_words = select_valid_dat_words(
                        dat_scorer_obj,
                        parsed_candidates,
                        max_words=dat_scorer_obj.N_SCORING_WORDS,
                    )
                    if len(valid_words) == 0:
                        invalid_reason = "valid_word_count_zero"
                    elif len(valid_words) < dat_scorer_obj.MIN_REQUIRED_WORDS:
                        invalid_reason = "insufficient_valid_words"

            print(f"  [{task_id}] Parsed {len(parsed_candidates_all)} -> using {len(parsed_candidates)} candidates -> {len(valid_words)} valid: {valid_words}")
            if invalid_reason is not None:
                invalid_run_counts["cdat"] += 1
                print(f"  [{task_id}] Invalid run: {invalid_reason}")

            novelty, appropriateness = (
                (None, None)
                if invalid_reason is not None or non_model_skip_reason is not None else
                dat_scorer_obj.compute_cdat_scores(valid_words, cue)
            )
            baseline_seed = stable_seed("cdat_random_baseline", cue, task_id)
            baseline = (
                dat_scorer_obj.compute_random_baseline(cue, n_samples=30, seed=baseline_seed)
                if non_model_skip_reason is None else None
            )
            gate_pass = (
                dat_scorer_obj.check_appropriateness_gate(appropriateness, baseline)
                if novelty is not None and appropriateness is not None and baseline is not None else None
            )
            cdat_blend = (
                dat_scorer_obj.combine_cdat_novelty_and_appropriateness(novelty, appropriateness, baseline)
                if novelty is not None and appropriateness is not None and baseline is not None else
                {"app_gain": None, "multiplier": None, "cdat_score": None}
            )
            if novelty is not None:
                cdat_novelties.append(novelty)
                cdat_apps.append(appropriateness)
                cdat_gates.append(gate_pass)
                if cdat_blend.get("app_gain") is not None:
                    cdat_app_gains.append(cdat_blend["app_gain"])
                if cdat_blend.get("multiplier") is not None:
                    cdat_multipliers.append(cdat_blend["multiplier"])
                if cdat_blend.get("cdat_score") is not None:
                    cdat_continuous_scores.append(cdat_blend["cdat_score"])
                cdat_scorable_count += 1
                gate_str = "PASS" if gate_pass else "FAIL"
                print(
                    f"  [{task_id}] Nov={novelty:.2f} App={appropriateness:.2f} "
                    f"Gate={gate_str} AppGain={cdat_blend.get('app_gain', 0.0):.2f} "
                    f"×{cdat_blend.get('multiplier', 0.0):.2f} => {cdat_blend.get('cdat_score')}"
                )
            cdat_results.append({
                "task_id": task_id,
                "cue": cue,
                "prompt": task.get("prompt"),
                "repeat_index": replicate_index,
                "coverage_eligible": non_model_skip_reason is None,
                "excluded_from_coverage": non_model_skip_reason is not None,
                "non_model_failure": non_model_skip_reason is not None,
                "non_model_skip_reason": non_model_skip_reason,
                "valid_run": invalid_reason is None and non_model_skip_reason is None,
                "invalid_reason": invalid_reason,
                "scorable": novelty is not None,
                "generation": build_generation_record(llm_result),
                "expected_output_count": expected_output_count,
                "parsed_item_count": len(parsed_candidates_all),
                "raw_items_truncated": truncated_item_count,
                "returned_item_count": len(parsed_candidates),
                "valid_words": valid_words,
                "novelty": novelty,
                "appropriateness": appropriateness,
                "gate_pass": gate_pass,
                "app_gain": cdat_blend.get("app_gain"),
                "appropriateness_multiplier": cdat_blend.get("multiplier"),
                "continuous_cdat_score": cdat_blend.get("cdat_score"),
                "random_baseline": baseline,
            })
        if cdat_novelties:
            mean_nov = sum(cdat_novelties) / len(cdat_novelties)
            mean_app = sum(cdat_apps) / len(cdat_apps)
            gate_pass_rate = sum(1 for gate in cdat_gates if gate) / len(cdat_gates)
            mean_app_gain = sum(cdat_app_gains) / len(cdat_app_gains) if cdat_app_gains else None
            mean_multiplier = sum(cdat_multipliers) / len(cdat_multipliers) if cdat_multipliers else None
            cdat_score = (
                sum(cdat_continuous_scores) / len(cdat_continuous_scores)
                if cdat_continuous_scores else None
            )
            if cdat_score is not None:
                print(
                    f"\n  >>> CDAT Mean Nov: {mean_nov:.2f}, "
                    f"Gate: {gate_pass_rate:.0%}, "
                    f"Mean AppGain: {(mean_app_gain or 0.0):.2f}, "
                    f"Score: {cdat_score:.2f}"
                )
            else:
                print(
                    f"\n  >>> CDAT Mean Nov: {mean_nov:.2f}, "
                    f"Gate: {gate_pass_rate:.0%}, Score: N/A"
                )

    if "FF" in dataset:
        print(f"\n{'=' * 60}")
        print(f"  Forward Flow (diagnostic) [{model_name}] [repeat {repeat_label}]")
        print(f"  Chain free association — unconstrained associative cognition")
        print(f"  Note: For relative LLM comparison only")
        print(f"{'=' * 60}")

        ff_scores = []
        ff_slopes = []
        expected_output_count = get_expected_output_count("FF") or FF_OUTPUT_COUNT

        for task in dataset["FF"]:
            task_id = task["id"]
            seed_word = task["seed"]
            print(f"\n  [{task_id}] Seed: \"{seed_word}\"")

            llm_result = call_llm(
                client,
                task["prompt"],
                model_name,
                task_label="FF",
                max_tokens_override=get_task_max_tokens("FF"),
                seed=stable_seed(model_name, "FF", task_id, replicate_index),
            )
            resolved_model = llm_result["resolved_model"]
            resolved_models_seen.add(resolved_model)
            raw_output = llm_result["content"]
            parsed_words_all = ff_scorer_obj.parse_chain(raw_output, seed_word=seed_word)
            parsed_words = parsed_words_all[:expected_output_count]
            truncated_item_count = max(0, len(parsed_words_all) - expected_output_count)
            valid_words = []
            invalid_reason = None
            non_model_skip_reason = get_non_model_skip_reason(llm_result)

            if non_model_skip_reason is not None:
                ff_excluded_count += 1
                non_model_skip_counts["ff"] += 1
                non_model_skip_reasons[non_model_skip_reason] += 1
                print(f"  [{task_id}] Coverage excluded: {non_model_skip_reason}")
            else:
                ff_effective_prompts += 1
                if not raw_output.strip():
                    invalid_reason = "empty_response"
                elif len(parsed_words) == 0:
                    invalid_reason = "parsed_zero_items"
                else:
                    valid_words = ff_scorer_obj.validate_chain(parsed_words, max_words=expected_output_count)
                    if len(valid_words) == 0:
                        invalid_reason = "chain_length_zero"
                    elif len(valid_words) < MIN_FF_WORDS_PER_TASK:
                        invalid_reason = "insufficient_chain_length"

            print(f"  [{task_id}] Parsed {len(parsed_words_all)} -> using {len(parsed_words)} -> {len(valid_words)} valid words")
            if invalid_reason is not None:
                invalid_run_counts["ff"] += 1
                print(f"  [{task_id}] Invalid run: {invalid_reason}")

            ff_result = (ff_scorer_obj.compute_forward_flow(valid_words)
                         if invalid_reason is None and non_model_skip_reason is None else ff_scorer_obj._empty_result(valid_words))
            if ff_result["valid"] and ff_result["dynamic_forward_flow"] is not None:
                ff_val = ff_result["dynamic_forward_flow"]
                ff_scores.append(ff_val)
                ff_scorable_count += 1
                slope = ff_result["trajectory_slope"]
                if slope is not None:
                    ff_slopes.append(slope)

                print(
                    f"  [{task_id}] Chain: {' -> '.join(valid_words[:10])}"
                    f"{'...' if len(valid_words) > 10 else ''}"
                )
                print(
                    f"  [{task_id}] Forward Flow: {ff_val:.4f} "
                    f"(chain={ff_result['chain_length']}, "
                    f"adj_dist={ff_result['mean_adjacent_distance']:.4f})"
                )
                if slope is not None:
                    if slope > 0.005:
                        trend_label = "exploration ↑"
                    elif slope < -0.005:
                        trend_label = "rumination ↓"
                    else:
                        trend_label = "stable →"
                    print(f"  [{task_id}] Trajectory slope: {slope:+.6f} ({trend_label})")
                if ff_result["repetition_rate"] > 0:
                    print(f"  [{task_id}] Repetition rate: {ff_result['repetition_rate']:.1%}")
            elif non_model_skip_reason is None:
                print(
                    f"  [{task_id}] Forward Flow: N/A "
                    f"(chain too short: {ff_result['chain_length']} words)"
                )

            ff_results.append({
                "task_id": task_id,
                "seed": seed_word,
                "prompt": task.get("prompt"),
                "repeat_index": replicate_index,
                "coverage_eligible": non_model_skip_reason is None,
                "excluded_from_coverage": non_model_skip_reason is not None,
                "non_model_failure": non_model_skip_reason is not None,
                "non_model_skip_reason": non_model_skip_reason,
                "valid_run": invalid_reason is None and non_model_skip_reason is None,
                "invalid_reason": invalid_reason,
                "scorable": ff_result["valid"] and ff_result["dynamic_forward_flow"] is not None,
                "generation": build_generation_record(llm_result),
                "expected_output_count": expected_output_count,
                "parsed_item_count": len(parsed_words_all),
                "raw_items_truncated": truncated_item_count,
                "returned_item_count": len(parsed_words),
                "parsed_word_count": len(parsed_words),
                "chain_length": ff_result["chain_length"],
                "valid": ff_result["valid"],
                "words": ff_result["words"],
                "dynamic_forward_flow": ff_result["dynamic_forward_flow"],
                "trajectory_slope": ff_result["trajectory_slope"],
                "mean_adjacent_distance": ff_result["mean_adjacent_distance"],
                "repetition_rate": ff_result["repetition_rate"],
                "instantaneous_flows": ff_result["instantaneous_flows"],
            })

        if ff_scores:
            ff_mean = sum(ff_scores) / len(ff_scores)
            ff_min = min(ff_scores)
            ff_max = max(ff_scores)
            mean_slope = sum(ff_slopes) / len(ff_slopes) if ff_slopes else None
            print(f"\n  >>> FF Summary ({len(ff_scores)} trials):")
            print(f"      Mean FF:    {ff_mean:.4f}  Range: [{ff_min:.4f}, {ff_max:.4f}]")
            if mean_slope is not None:
                trend_label = "exploration" if mean_slope > 0.005 else "rumination" if mean_slope < -0.005 else "stable"
                print(f"      Mean slope: {mean_slope:+.6f} ({trend_label})")
            human_reference = ff_scorer_obj.get_human_reference()
            print(
                f"      LSA literature reference: {human_reference['reference_mean']:.2f} "
                f"(SD={human_reference['reference_sd']:.2f})"
            )
            print(f"      Note: For relative LLM comparison only")
        else:
            print("\n  >>> Forward Flow: No valid scores computed.")

    dat_cdat_ff = {
        "dat": {
            "trials": len(dat_results),
            "scores": [result["dat_score"] for result in dat_results if result["dat_score"] is not None],
            "raw_scores": [result["raw_dat_score"] for result in dat_results if result["raw_dat_score"] is not None],
            "mean_score": round(dat_mean, 2) if dat_mean is not None else None,
            "mean_raw_score": round(sum(raw_dat_scores) / len(raw_dat_scores), 2) if raw_dat_scores else None,
            "score_formula": "0.70 * raw_dat + 0.30 * supersense_coverage_100",
            "details": dat_results,
        } if dat_results else None,
        "cdat": {
            "cues_evaluated": len(cdat_results),
            "mean_novelty": round(mean_nov, 2) if mean_nov is not None else None,
            "mean_appropriateness": round(mean_app, 2) if mean_app is not None else None,
            "gate_pass_rate": round(gate_pass_rate, 4) if gate_pass_rate is not None else None,
            "mean_app_gain": round(mean_app_gain, 4) if mean_app_gain is not None else None,
            "mean_multiplier": round(mean_multiplier, 4) if mean_multiplier is not None else None,
            "cdat_score": round(cdat_score, 2) if cdat_score is not None else None,
            "score_formula": "novelty * app_gain; app_gain = clip((appropriateness - (baseline_mean + 0.5 * baseline_std)) / max(baseline_std, 2.0), 0, 1)",
            "details": cdat_results,
        } if cdat_results else None,
        "ff": {
            "trials": len(ff_results),
            "scores": [result["dynamic_forward_flow"] for result in ff_results if result["dynamic_forward_flow"] is not None],
            "mean_score": round(ff_mean, 4) if ff_mean is not None else None,
            "mean_trajectory_slope": round(sum(ff_s.get("trajectory_slope") for ff_s in ff_results if ff_s.get("trajectory_slope") is not None) / len([ff_s for ff_s in ff_results if ff_s.get("trajectory_slope") is not None]), 6) if any(ff_s.get("trajectory_slope") is not None for ff_s in ff_results) else None,
            "literature_reference_mean": ff_scorer_obj.get_human_reference()["reference_mean"] if ff_results else None,
            "literature_reference_sd": ff_scorer_obj.get_human_reference()["reference_sd"] if ff_results else None,
            "comparison_note": "FF remains diagnostic and is excluded from DT-total aggregation.",
            "details": ff_results,
        } if ff_results else None,
    }
    category_originality_scores_raw = {}
    for category in CREATIVE_TASK_TYPES:
        if category_originality_counts[category] > 0:
            category_originality_scores_raw[category] = round(
                category_originality_sums[category] / category_originality_counts[category],
                4,
            )
    if dat_mean is not None:
        category_originality_scores_raw["DAT"] = round(dat_mean / 100.0, 4)
    if cdat_score is not None:
        category_originality_scores_raw["CDAT"] = round(cdat_score / 100.0, 4)

    creative_task_coverages = {
        task_type: (
            creative_task_valid_counts[task_type] / creative_task_effective_totals[task_type]
            if creative_task_effective_totals.get(task_type) else None
        )
        for task_type in CREATIVE_TASK_TYPES
    }
    creative_task_availability = {
        task_type: (
            creative_task_effective_totals[task_type] / creative_task_totals[task_type]
            if creative_task_totals.get(task_type) else None
        )
        for task_type in CREATIVE_TASK_TYPES
    }
    total_creative_effective_prompts = sum(creative_task_effective_totals.values())
    creative_coverage = (
        task_count / total_creative_effective_prompts
        if total_creative_effective_prompts > 0 else None
    )
    creative_availability = (
        total_creative_effective_prompts / total_creative_prompts
        if total_creative_prompts > 0 else None
    )
    dat_coverage = (dat_scorable_count / dat_effective_prompts) if dat_effective_prompts > 0 else None
    dat_availability = (dat_effective_prompts / dat_total_prompts) if dat_total_prompts > 0 else None
    cdat_coverage = (cdat_scorable_count / cdat_effective_prompts) if cdat_effective_prompts > 0 else None
    cdat_availability = (cdat_effective_prompts / cdat_total_prompts) if cdat_total_prompts > 0 else None
    ff_coverage = (ff_scorable_count / ff_effective_prompts) if ff_effective_prompts > 0 else None
    ff_availability = (ff_effective_prompts / ff_total_prompts) if ff_total_prompts > 0 else None
    macgyver_coverage = (
        macgyver_scorable_count / macgyver_effective_prompts
        if macgyver_effective_prompts > 0 else None
    )
    macgyver_availability = (
        macgyver_effective_prompts / macgyver_total_prompts
        if macgyver_total_prompts > 0 else None
    )
    macgyver_boundary_coverage = (
        macgyver_boundary_scorable_count / macgyver_boundary_effective_prompts
        if macgyver_boundary_effective_prompts > 0 else None
    )
    macgyver_boundary_availability = (
        macgyver_boundary_effective_prompts / macgyver_boundary_total_prompts
        if macgyver_boundary_total_prompts > 0 else None
    )
    cjst_coverage = (
        cjst_scorable_count / cjst_effective_prompts
        if cjst_effective_prompts > 0 else None
    )
    cjst_availability = (
        cjst_effective_prompts / cjst_total_prompts
        if cjst_total_prompts > 0 else None
    )
    hypospace_coverage = (
        hypospace_scorable_count / hypospace_effective_prompts
        if hypospace_effective_prompts > 0 else None
    )
    hypospace_availability = (
        hypospace_effective_prompts / hypospace_total_prompts
        if hypospace_total_prompts > 0 else None
    )
    hypospace_boundary_coverage = (
        hypospace_boundary_scorable_count / hypospace_boundary_effective_prompts
        if hypospace_boundary_effective_prompts > 0 else None
    )
    hypospace_boundary_availability = (
        hypospace_boundary_effective_prompts / hypospace_boundary_total_prompts
        if hypospace_boundary_total_prompts > 0 else None
    )
    gcw_coverage = (
        gcw_scorable_count / gcw_effective_prompts
        if gcw_effective_prompts > 0 else None
    )
    gcw_availability = (
        gcw_effective_prompts / gcw_total_prompts
        if gcw_total_prompts > 0 else None
    )
    neocoder_coverage = (
        neocoder_scorable_count / neocoder_effective_prompts
        if neocoder_effective_prompts > 0 else None
    )
    neocoder_availability = (
        neocoder_effective_prompts / neocoder_total_prompts
        if neocoder_total_prompts > 0 else None
    )
    closed_world_fact_coverage = (
        closed_world_fact_scorable_count / closed_world_fact_effective_prompts
        if closed_world_fact_effective_prompts > 0 else None
    )
    closed_world_fact_availability = (
        closed_world_fact_effective_prompts / closed_world_fact_total_prompts
        if closed_world_fact_total_prompts > 0 else None
    )
    analogy_transfer_coverage = (
        analogy_transfer_scorable_count / analogy_transfer_effective_prompts
        if analogy_transfer_effective_prompts > 0 else None
    )
    analogy_transfer_availability = (
        analogy_transfer_effective_prompts / analogy_transfer_total_prompts
        if analogy_transfer_total_prompts > 0 else None
    )

    creative_overall_output_pass = (
        creative_coverage is not None and creative_coverage >= MIN_CREATIVE_COVERAGE
    )
    creative_tasktype_output_pass = all(
        coverage is not None and coverage >= MIN_CREATIVE_TASKTYPE_COVERAGE
        for task_type, coverage in creative_task_coverages.items()
        if creative_task_effective_totals.get(task_type)
    )
    creative_overall_availability_pass = (
        creative_availability is not None and creative_availability >= MIN_CREATIVE_COVERAGE
    )
    creative_tasktype_availability_pass = all(
        availability is not None and availability >= MIN_CREATIVE_TASKTYPE_COVERAGE
        for task_type, availability in creative_task_availability.items()
        if creative_task_totals.get(task_type)
    )
    creative_axis_gate_pass = (
        creative_overall_output_pass and creative_tasktype_output_pass and
        creative_overall_availability_pass and creative_tasktype_availability_pass
    )

    dat_gate_pass = (
        dat_total_prompts == 0 or (
            dat_availability is not None and dat_availability >= MIN_DAT_COVERAGE and
            dat_coverage is not None and dat_coverage >= MIN_DAT_COVERAGE and
            dat_mean is not None
        )
    )
    cdat_gate_pass = (
        cdat_total_prompts == 0 or (
            cdat_availability is not None and cdat_availability >= MIN_CDAT_COVERAGE and
            cdat_coverage is not None and cdat_coverage >= MIN_CDAT_COVERAGE and
            cdat_score is not None
        )
    )
    macgyver_gate_pass = (
        macgyver_total_prompts == 0 or (
            macgyver_availability is not None and macgyver_availability >= MIN_MACGYVER_COVERAGE and
            macgyver_coverage is not None and macgyver_coverage >= MIN_MACGYVER_COVERAGE and
            bool(macgyver_dual_axis_task_scores)
        )
    )
    cjst_gate_pass = (
        cjst_total_prompts == 0 or (
            cjst_availability is not None and cjst_availability >= MIN_CJST_COVERAGE and
            cjst_coverage is not None and cjst_coverage >= MIN_CJST_COVERAGE and
            bool(cjst_dual_axis_task_scores)
        )
    )
    hypospace_gate_pass = (
        hypospace_total_prompts == 0 or (
            hypospace_availability is not None and hypospace_availability >= MIN_HYPOUSESPACE_COVERAGE and
            hypospace_coverage is not None and hypospace_coverage >= MIN_HYPOUSESPACE_COVERAGE and
            bool(hypospace_dual_axis_task_scores)
        )
    )
    gcw_gate_pass = (
        gcw_total_prompts == 0 or (
            gcw_availability is not None and gcw_availability >= MIN_GCW_COVERAGE and
            gcw_coverage is not None and gcw_coverage >= MIN_GCW_COVERAGE and
            bool(gcw_dual_axis_task_scores)
        )
    )
    neocoder_gate_pass = (
        neocoder_total_prompts == 0 or (
            neocoder_availability is not None and neocoder_availability >= MIN_NEOCODER_COVERAGE and
            neocoder_coverage is not None and neocoder_coverage >= MIN_NEOCODER_COVERAGE and
            bool(neocoder_dual_axis_task_scores)
        )
    )
    closed_world_fact_gate_pass = (
        closed_world_fact_total_prompts == 0 or (
            closed_world_fact_availability is not None and closed_world_fact_availability >= MIN_CLOSED_WORLD_FACT_COVERAGE and
            closed_world_fact_coverage is not None and closed_world_fact_coverage >= MIN_CLOSED_WORLD_FACT_COVERAGE and
            bool(closed_world_fact_task_scores)
        )
    )
    analogy_transfer_gate_pass = (
        analogy_transfer_total_prompts == 0 or (
            analogy_transfer_availability is not None and analogy_transfer_availability >= MIN_ANALOGY_TRANSFER_COVERAGE and
            analogy_transfer_coverage is not None and analogy_transfer_coverage >= MIN_ANALOGY_TRANSFER_COVERAGE and
            bool(analogy_transfer_task_scores)
        )
    )
    macgyver_summary = {
        "trials": len(macgyver_results),
        "scorable_trials": macgyver_scorable_count,
        "effective_prompts": macgyver_effective_prompts,
        "excluded_prompts": macgyver_excluded_count,
        "coverage": round(macgyver_coverage, 4) if macgyver_coverage is not None else None,
        "availability": round(macgyver_availability, 4) if macgyver_availability is not None else None,
        "coverage_gate_pass": macgyver_gate_pass,
        "expected_plan_count": MACGYVER_OUTPUT_COUNT,
        "minimum_valid_plans": MIN_MACGYVER_PLANS_PER_TASK,
        "details": macgyver_results,
    } if macgyver_results else None
    macgyver_boundary_summary = {
        "trials": len(macgyver_boundary_results),
        "scorable_trials": macgyver_boundary_scorable_count,
        "effective_prompts": macgyver_boundary_effective_prompts,
        "excluded_prompts": macgyver_boundary_excluded_count,
        "coverage": round(macgyver_boundary_coverage, 4) if macgyver_boundary_coverage is not None else None,
        "availability": round(macgyver_boundary_availability, 4) if macgyver_boundary_availability is not None else None,
        "expected_plan_count": MACGYVER_OUTPUT_COUNT,
        "minimum_valid_plans": MIN_MACGYVER_PLANS_PER_TASK,
        "details": macgyver_boundary_results,
    } if macgyver_boundary_results else None
    cjst_summary = {
        "trials": len(cjst_results),
        "scorable_trials": cjst_scorable_count,
        "effective_prompts": cjst_effective_prompts,
        "excluded_prompts": cjst_excluded_count,
        "coverage": round(cjst_coverage, 4) if cjst_coverage is not None else None,
        "availability": round(cjst_availability, 4) if cjst_availability is not None else None,
        "coverage_gate_pass": cjst_gate_pass,
        "expected_output_count": CJST_OUTPUT_COUNT,
        "minimum_valid_items": MIN_CJST_ITEMS_PER_TASK,
        "minimum_valid_items_per_tier": MIN_CJST_ITEMS_PER_TIER,
        "details": cjst_results,
    } if cjst_results else None
    hypospace_summary = {
        "trials": len(hypospace_results),
        "scorable_trials": hypospace_scorable_count,
        "effective_prompts": hypospace_effective_prompts,
        "excluded_prompts": hypospace_excluded_count,
        "coverage": round(hypospace_coverage, 4) if hypospace_coverage is not None else None,
        "availability": round(hypospace_availability, 4) if hypospace_availability is not None else None,
        "coverage_gate_pass": hypospace_gate_pass,
        "expected_output_count": HYPOUSESPACE_OUTPUT_COUNT,
        "minimum_valid_hypotheses": MIN_HYPOUSESPACE_ITEMS_PER_TASK,
        "details": hypospace_results,
    } if hypospace_results else None
    hypospace_boundary_summary = {
        "trials": len(hypospace_boundary_results),
        "scorable_trials": hypospace_boundary_scorable_count,
        "effective_prompts": hypospace_boundary_effective_prompts,
        "excluded_prompts": hypospace_boundary_excluded_count,
        "coverage": round(hypospace_boundary_coverage, 4) if hypospace_boundary_coverage is not None else None,
        "availability": round(hypospace_boundary_availability, 4) if hypospace_boundary_availability is not None else None,
        "expected_output_count": HYPOUSESPACE_OUTPUT_COUNT,
        "minimum_valid_hypotheses": 0,
        "details": hypospace_boundary_results,
    } if hypospace_boundary_results else None
    gcw_summary = {
        "trials": len(gcw_results),
        "scorable_trials": gcw_scorable_count,
        "effective_prompts": gcw_effective_prompts,
        "excluded_prompts": gcw_excluded_count,
        "coverage": round(gcw_coverage, 4) if gcw_coverage is not None else None,
        "availability": round(gcw_availability, 4) if gcw_availability is not None else None,
        "coverage_gate_pass": gcw_gate_pass,
        "expected_beat_count": GCW_BEAT_COUNT,
        "minimum_valid_beats": max(MIN_GCW_BEATS_PER_TASK, int(GCW_BEAT_COUNT * 0.67)),
        "details": gcw_results,
    } if gcw_results else None
    neocoder_summary = {
        "trials": len(neocoder_results),
        "scorable_trials": neocoder_scorable_count,
        "effective_prompts": neocoder_effective_prompts,
        "excluded_prompts": neocoder_excluded_count,
        "coverage": round(neocoder_coverage, 4) if neocoder_coverage is not None else None,
        "availability": round(neocoder_availability, 4) if neocoder_availability is not None else None,
        "coverage_gate_pass": neocoder_gate_pass,
        "expected_output_count": NEOCODER_OUTPUT_COUNT,
        "minimum_valid_code_objects": MIN_NEOCODER_ITEMS_PER_TASK,
        "details": neocoder_results,
    } if neocoder_results else None
    closed_world_fact_summary = {
        "trials": len(closed_world_fact_results),
        "scorable_trials": closed_world_fact_scorable_count,
        "effective_prompts": closed_world_fact_effective_prompts,
        "excluded_prompts": closed_world_fact_excluded_count,
        "coverage": round(closed_world_fact_coverage, 4) if closed_world_fact_coverage is not None else None,
        "availability": round(closed_world_fact_availability, 4) if closed_world_fact_availability is not None else None,
        "coverage_gate_pass": closed_world_fact_gate_pass,
        "expected_output_count": CLOSED_WORLD_FACT_OUTPUT_COUNT,
        "minimum_valid_answers": MIN_CLOSED_WORLD_FACT_ITEMS_PER_TASK,
        "details": closed_world_fact_results,
    } if closed_world_fact_results else None
    analogy_transfer_summary = {
        "trials": len(analogy_transfer_results),
        "scorable_trials": analogy_transfer_scorable_count,
        "effective_prompts": analogy_transfer_effective_prompts,
        "excluded_prompts": analogy_transfer_excluded_count,
        "coverage": round(analogy_transfer_coverage, 4) if analogy_transfer_coverage is not None else None,
        "availability": round(analogy_transfer_availability, 4) if analogy_transfer_availability is not None else None,
        "coverage_gate_pass": analogy_transfer_gate_pass,
        "expected_output_count": ANALOGY_TRANSFER_OUTPUT_COUNT,
        "minimum_valid_analogy_objects": MIN_ANALOGY_TRANSFER_ITEMS_PER_TASK,
        "details": analogy_transfer_results,
    } if analogy_transfer_results else None

    category_originality_scores = {}
    for category in CREATIVE_TASK_TYPES:
        coverage = creative_task_coverages.get(category)
        availability = creative_task_availability.get(category)
        if (
            coverage is not None and coverage >= MIN_CREATIVE_TASKTYPE_COVERAGE and
            availability is not None and availability >= MIN_CREATIVE_TASKTYPE_COVERAGE and
            category in category_originality_scores_raw
        ):
            category_originality_scores[category] = category_originality_scores_raw[category]
    if dat_gate_pass and dat_mean is not None:
        category_originality_scores["DAT"] = round(dat_mean / 100.0, 4)
    if cdat_gate_pass and cdat_score is not None:
        category_originality_scores["CDAT"] = round(cdat_score / 100.0, 4)

    novelty_required_components = {
        task_type for task_type in CREATIVE_TASK_TYPES
        if creative_task_totals.get(task_type)
    }
    if dat_total_prompts > 0:
        novelty_required_components.add("DAT")
    if cdat_total_prompts > 0:
        novelty_required_components.add("CDAT")
    effective_novelty_weights = get_effective_novelty_component_weights(novelty_required_components)
    effective_novelty_weights_report = {
        key: round(value, 6)
        for key, value in effective_novelty_weights.items()
    }
    base_novelty_weights_report = {
        key: round(NOVELTY_COMPONENT_BASE_WEIGHTS[key], 6)
        for key in NOVELTY_COMPONENT_ORDER
        if key in novelty_required_components and key in NOVELTY_COMPONENT_BASE_WEIGHTS
    }
    novelty_formula = format_novelty_component_formula(effective_novelty_weights)
    novelty_axis_gate_pass = (
        creative_axis_gate_pass and dat_gate_pass and cdat_gate_pass
        and set(category_originality_scores.keys()) == novelty_required_components
    )
    novelty_axis_score = (
        round(
            sum(
                category_originality_scores[key] * effective_novelty_weights.get(key, 0.0)
                for key in category_originality_scores
            ),
            4,
        )
        if novelty_axis_gate_pass and category_originality_scores and effective_novelty_weights else None
    )
    num_categories_scored = len(category_originality_scores)

    if task_count > 0:
        tc = task_count
        overall_avg_wn_distance = sum_wn_distance / tc
        overall_avg_wn_ic = sum_wn_ic / tc

        model_avg_cluster_ratio = sum(all_task_cluster_ratios) / tc if all_task_cluster_ratios else 0.0
        model_avg_emb_switch_rate = sum(all_task_emb_switch_rates) / tc if all_task_emb_switch_rates else 0.0
        model_avg_emb_pairwise_distance = (
            sum(all_task_emb_pairwise_distances) / len(all_task_emb_pairwise_distances)
            if all_task_emb_pairwise_distances else 0.0
        )
        model_avg_emb_adjacent_distance = (
            sum(all_task_emb_adjacent_distances) / len(all_task_emb_adjacent_distances)
            if all_task_emb_adjacent_distances else 0.0
        )
        model_avg_emb_cluster_entropy = (
            sum(all_task_emb_cluster_entropies) / len(all_task_emb_cluster_entropies)
            if all_task_emb_cluster_entropies else 0.0
        )
        model_embedding_flexibility_score = (
            0.20 * model_avg_cluster_ratio +
            0.15 * model_avg_emb_switch_rate +
            0.30 * model_avg_emb_pairwise_distance +
            0.15 * model_avg_emb_adjacent_distance +
            0.20 * model_avg_emb_cluster_entropy
        )
        embedding_is_degenerate = (
            model_avg_cluster_ratio > 0.97 and model_avg_emb_cluster_entropy < 0.35
        )

        model_global_unique_categories = len(global_wn_categories_set)
        model_global_category_coverage = model_global_unique_categories / max(1, wn_analyzer.num_categories)
        model_avg_wn_switch_rate = sum(all_task_wn_switch_rates) / tc if all_task_wn_switch_rates else 0.0

        total_categorized_global = sum(global_wn_category_counter.values())
        global_entropy = 0.0
        if total_categorized_global > 0:
            for count in global_wn_category_counter.values():
                probability = count / total_categorized_global
                if probability > 0:
                    global_entropy -= probability * math.log2(probability)
        max_possible_entropy = math.log2(max(2, wn_analyzer.num_categories))
        model_global_diversity_index = global_entropy / max_possible_entropy if max_possible_entropy > 0 else 0.0

        model_ontological_flexibility_score = (
            0.25 * model_global_category_coverage +
            0.20 * model_avg_wn_switch_rate +
            0.20 * model_global_diversity_index +
            0.20 * overall_avg_wn_distance +
            0.15 * overall_avg_wn_ic
        )

        if embedding_is_degenerate:
            w_emb = FLEX_WEIGHT_EMBEDDING_DEGENERATE
            w_ont = FLEX_WEIGHT_ONTOLOGICAL_DEGENERATE
        else:
            w_emb = FLEX_WEIGHT_EMBEDDING
            w_ont = FLEX_WEIGHT_ONTOLOGICAL

        flexibility_axis_score = round(
            w_emb * model_embedding_flexibility_score + w_ont * model_ontological_flexibility_score,
            4,
        ) if creative_axis_gate_pass else None
        flex_formula_str = (
            f"{w_emb:.2f}*Emb({model_embedding_flexibility_score:.4f}) + "
            f"{w_ont:.2f}*Ont({model_ontological_flexibility_score:.4f})"
        )
    else:
        overall_avg_wn_distance = None
        overall_avg_wn_ic = None
        model_embedding_flexibility_score = None
        model_ontological_flexibility_score = None
        model_avg_emb_pairwise_distance = None
        model_avg_emb_adjacent_distance = None
        model_avg_emb_cluster_entropy = None
        embedding_is_degenerate = None
        w_emb = None
        w_ont = None
        flexibility_axis_score = None
        flex_formula_str = None
    flexibility_axis_gate_pass = task_count > 0 and creative_axis_gate_pass

    model_ground_task_scores_raw = {
        task_type: round(mean_or_none(values), 4)
        for task_type, values in task_type_ground_scores.items()
        if mean_or_none(values) is not None
    }
    model_ground_task_scores_novel_raw = {
        task_type: round(mean_or_none(values), 4)
        for task_type, values in task_type_ground_scores_novel.items()
        if mean_or_none(values) is not None
    }
    model_ground_task_scores = {
        task_type: score
        for task_type, score in model_ground_task_scores_raw.items()
        if creative_task_coverages.get(task_type) is not None
        and creative_task_coverages[task_type] >= MIN_CREATIVE_TASKTYPE_COVERAGE
        and creative_task_availability.get(task_type) is not None
        and creative_task_availability[task_type] >= MIN_CREATIVE_TASKTYPE_COVERAGE
    }
    model_ground_task_scores_novel = {
        task_type: score
        for task_type, score in model_ground_task_scores_novel_raw.items()
        if creative_task_coverages.get(task_type) is not None
        and creative_task_coverages[task_type] >= MIN_CREATIVE_TASKTYPE_COVERAGE
        and creative_task_availability.get(task_type) is not None
        and creative_task_availability[task_type] >= MIN_CREATIVE_TASKTYPE_COVERAGE
    }
    groundedness_axis_gate_pass = (
        creative_axis_gate_pass
        and len(model_ground_task_scores) == len([task_type for task_type in CREATIVE_TASK_TYPES if creative_task_totals.get(task_type)])
    )
    groundedness_axis_score_raw = mean_or_none(list(model_ground_task_scores_raw.values()))
    groundedness_axis_score_novel_raw = mean_or_none(list(model_ground_task_scores_novel_raw.values()))
    groundedness_axis_score = (
        mean_or_none(list(model_ground_task_scores.values()))
        if groundedness_axis_gate_pass else None
    )
    groundedness_axis_score_novel = (
        mean_or_none(list(model_ground_task_scores_novel.values()))
        if groundedness_axis_gate_pass else None
    )
    mean_penalty = total_penalty / total_groundedness_score_count if total_groundedness_score_count > 0 else None
    penalty_rate = total_penalty_positive / total_groundedness_score_count if total_groundedness_score_count > 0 else None
    low_groundedness_rate = total_low_groundedness / total_groundedness_score_count if total_groundedness_score_count > 0 else None
    low_groundedness_rate_novel = total_low_groundedness_novel / total_groundedness_novel_count if total_groundedness_novel_count > 0 else None
    mean_confidence = total_groundedness_confidence / total_groundedness_score_count if total_groundedness_score_count > 0 else None
    scored_coverage = total_groundedness_score_count / total_scored_ideas if total_scored_ideas > 0 else None
    confidence_weighted_groundedness_raw = (
        total_groundedness_weighted / total_groundedness_weight
        if total_groundedness_weight > 0 else groundedness_axis_score_raw
    )
    confidence_weighted_groundedness_novel_raw = (
        total_groundedness_weighted_novel / total_groundedness_weight_novel
        if total_groundedness_weight_novel > 0 else groundedness_axis_score_novel_raw
    )
    confidence_weighted_groundedness = (
        confidence_weighted_groundedness_raw if groundedness_axis_gate_pass else None
    )
    confidence_weighted_groundedness_novel = (
        confidence_weighted_groundedness_novel_raw if groundedness_axis_gate_pass else None
    )
    uut_dual_axis_gate_pass = (
        creative_task_coverages.get("UUT") is not None
        and creative_task_coverages["UUT"] >= MIN_CREATIVE_TASKTYPE_COVERAGE
        and creative_task_availability.get("UUT") is not None
        and creative_task_availability["UUT"] >= MIN_CREATIVE_TASKTYPE_COVERAGE
        and bool(uut_dual_axis_task_scores)
    )
    uut_imagination_raw = mean_or_none([item.get("imagination_raw") for item in uut_dual_axis_task_scores])
    uut_hallucination_raw = mean_or_none([item.get("hallucination_raw") for item in uut_dual_axis_task_scores])
    uut_quality_mass_top8 = mean_or_none([item.get("quality_mass_top8") for item in uut_dual_axis_task_scores])
    uut_elite_tail_top3 = mean_or_none([item.get("elite_tail_top3") for item in uut_dual_axis_task_scores])
    uut_diversity_eff = mean_or_none([item.get("diversity_eff") for item in uut_dual_axis_task_scores])
    uut_valid_ratio = mean_or_none([item.get("valid_ratio") for item in uut_dual_axis_task_scores])
    uut_bank_coverage = mean_or_none([item.get("bank_coverage") for item in uut_dual_axis_task_scores])
    uut_imagination_score = (
        mean_or_none([item.get("imagination") for item in uut_dual_axis_task_scores])
        if uut_dual_axis_gate_pass else None
    )
    uut_hallucination_score = (
        mean_or_none([item.get("hallucination") for item in uut_dual_axis_task_scores])
        if uut_dual_axis_gate_pass else None
    )
    uut_dual_axis_primitive_means = {}
    for field in [
        "novelty",
        "appropriateness_gate",
        "cue_drift",
        "semantic_anchor",
        "cue_support_failure",
        "supported_affordance_ratio",
        "unsupported_claim_ratio",
        "contradiction_ratio",
        "extra_tool_violation",
        "mechanism_completeness",
        "physical_drift",
        "novelty_times_affordance_support",
        "novelty_times_groundedness",
        "appropriateness_gated_novelty_affordance_support",
        "mechanism_times_affordance_support",
        "mechanism_times_groundedness",
        "appropriateness_gated_mechanism_affordance_support",
        "idea_hallucination_raw",
        "rarity_v3",
        "imagination_contribution_v3",
    ]:
        value = mean_or_none([
            task_score.get("primitive_means", {}).get(field)
            for task_score in uut_dual_axis_task_scores
            if isinstance(task_score.get("primitive_means"), dict)
        ])
        if value is not None:
            uut_dual_axis_primitive_means[field] = round(value, 4)
    uut_subtype_contributions = mean_subtype_contributions(
        task_score.get("subtype_contributions")
        for task_score in uut_dual_axis_task_scores
        if isinstance(task_score.get("subtype_contributions"), dict)
    ) if uut_dual_axis_task_scores else None

    propconj_dual_axis_gate_pass = (
        creative_task_coverages.get("PropConj") is not None
        and creative_task_coverages["PropConj"] >= MIN_CREATIVE_TASKTYPE_COVERAGE
        and creative_task_availability.get("PropConj") is not None
        and creative_task_availability["PropConj"] >= MIN_CREATIVE_TASKTYPE_COVERAGE
        and bool(propconj_dual_axis_task_scores)
    )
    propconj_axis = aggregate_propconj_model_axes(
        propconj_dual_axis_task_scores,
        gate_pass=propconj_dual_axis_gate_pass,
        beta_ih=PROPCONJ_DUAL_AXIS_BETA_IH,
        beta_hi=PROPCONJ_DUAL_AXIS_BETA_HI,
    )
    propconj_imagination_score = propconj_axis.get("imagination")
    propconj_hallucination_score = propconj_axis.get("hallucination")
    propconj_imagination_raw = propconj_axis.get("imagination_raw")
    propconj_hallucination_raw = propconj_axis.get("hallucination_raw")
    propconj_primitive_means = propconj_axis.get("primitive_means") or {}
    propconj_subtype_contributions = (
        propconj_axis.get("subtype_contributions")
        if isinstance(propconj_axis.get("subtype_contributions"), dict)
        else None
    )

    macgyver_axis = aggregate_macgyver_model_axes(
        macgyver_dual_axis_task_scores,
        gate_pass=macgyver_gate_pass,
        beta_ih=MACGYVER_DUAL_AXIS_BETA_IH,
        beta_hi=MACGYVER_DUAL_AXIS_BETA_HI,
    )
    macgyver_imagination_score = macgyver_axis.get("imagination")
    macgyver_hallucination_score = macgyver_axis.get("hallucination")
    macgyver_imagination_raw = macgyver_axis.get("imagination_raw")
    macgyver_hallucination_raw = macgyver_axis.get("hallucination_raw")
    macgyver_primitive_means = macgyver_axis.get("primitive_means") or {}
    macgyver_boundary_record_means = macgyver_axis.get("boundary_record_means") or {}
    macgyver_subtype_contributions = (
        macgyver_axis.get("subtype_contributions")
        if isinstance(macgyver_axis.get("subtype_contributions"), dict)
        else None
    )
    macgyver_boundary_axis = aggregate_macgyver_boundary_diagnostics(
        macgyver_boundary_task_scores,
        beta_hi=MACGYVER_DUAL_AXIS_BETA_HI,
    )

    cjst_axis = aggregate_cjst_model_axes(
        cjst_dual_axis_task_scores,
        gate_pass=(cjst_gate_pass and bool(cjst_dual_axis_task_scores)),
        beta_ih=CJST_DUAL_AXIS_BETA_IH,
        beta_hi=CJST_DUAL_AXIS_BETA_HI,
    )
    cjst_imagination_score = cjst_axis.get("imagination")
    cjst_hallucination_score = cjst_axis.get("hallucination")
    cjst_imagination_raw = cjst_axis.get("imagination_raw")
    cjst_hallucination_raw = cjst_axis.get("hallucination_raw")
    cjst_primitive_means = cjst_axis.get("primitive_means") or {}
    cjst_subtype_contributions = (
        cjst_axis.get("subtype_contributions")
        if isinstance(cjst_axis.get("subtype_contributions"), dict)
        else None
    )

    hypospace_axis = aggregate_hypospace_model_axes(
        hypospace_dual_axis_task_scores,
        gate_pass=(hypospace_gate_pass and bool(hypospace_dual_axis_task_scores)),
        beta_ih=HYPOUSESPACE_DUAL_AXIS_BETA_IH,
        beta_hi=HYPOUSESPACE_DUAL_AXIS_BETA_HI,
    )
    hypospace_imagination_score = hypospace_axis.get("imagination")
    hypospace_hallucination_score = hypospace_axis.get("hallucination")
    hypospace_imagination_raw = hypospace_axis.get("imagination_raw")
    hypospace_hallucination_raw = hypospace_axis.get("hallucination_raw")
    hypospace_primitive_means = hypospace_axis.get("primitive_means") or {}
    hypospace_subtype_contributions = (
        hypospace_axis.get("subtype_contributions")
        if isinstance(hypospace_axis.get("subtype_contributions"), dict)
        else None
    )
    hypospace_boundary_axis = aggregate_hypospace_boundary_diagnostics(
        hypospace_boundary_task_scores,
        beta_hi=HYPOUSESPACE_DUAL_AXIS_BETA_HI,
    )

    gcw_axis = aggregate_gcw_model_axes(
        gcw_dual_axis_task_scores,
        gate_pass=(gcw_gate_pass and bool(gcw_dual_axis_task_scores)),
    )
    gcw_imagination_score = gcw_axis.get("imagination")
    gcw_hallucination_score = gcw_axis.get("hallucination")
    gcw_imagination_raw = gcw_axis.get("imagination_raw")
    gcw_hallucination_raw = gcw_axis.get("hallucination_raw")
    gcw_primitive_means = gcw_axis.get("primitive_means") or {}
    gcw_subtype_contributions = (
        gcw_axis.get("subtype_contributions")
        if isinstance(gcw_axis.get("subtype_contributions"), dict)
        else None
    )
    neocoder_axis = aggregate_neocoder_model_axes(
        neocoder_dual_axis_task_scores,
        gate_pass=(neocoder_gate_pass and bool(neocoder_dual_axis_task_scores)),
        beta_ih=NEOCODER_DUAL_AXIS_BETA_IH,
        beta_hi=NEOCODER_DUAL_AXIS_BETA_HI,
    )
    neocoder_imagination_score = neocoder_axis.get("imagination")
    neocoder_hallucination_score = neocoder_axis.get("hallucination")
    neocoder_imagination_raw = neocoder_axis.get("imagination_raw")
    neocoder_imagination_gated = neocoder_axis.get("imagination_gated")
    neocoder_hallucination_raw = neocoder_axis.get("hallucination_raw")
    neocoder_primitive_means = neocoder_axis.get("primitive_means") or {}
    neocoder_subtype_contributions = (
        neocoder_axis.get("subtype_contributions")
        if isinstance(neocoder_axis.get("subtype_contributions"), dict)
        else None
    )
    closed_world_fact_axis = aggregate_closed_world_fact_calibration_axes(
        closed_world_fact_task_scores,
        gate_pass=(closed_world_fact_gate_pass and bool(closed_world_fact_task_scores)),
    )
    closed_world_fact_score = closed_world_fact_axis.get("score")
    closed_world_fact_hallucination = closed_world_fact_axis.get("hallucination")
    closed_world_fact_hallucination_raw = closed_world_fact_axis.get("hallucination_raw")
    closed_world_fact_primitive_means = closed_world_fact_axis.get("primitive_means") or {}
    closed_world_fact_subtype_contributions = (
        closed_world_fact_axis.get("subtype_contributions")
        if isinstance(closed_world_fact_axis.get("subtype_contributions"), dict)
        else None
    )
    analogy_transfer_axis = aggregate_analogy_transfer_challenge_axes(
        analogy_transfer_task_scores,
        gate_pass=(analogy_transfer_gate_pass and bool(analogy_transfer_task_scores)),
    )
    analogy_transfer_imagination_score = analogy_transfer_axis.get("imagination")
    analogy_transfer_hallucination_score = analogy_transfer_axis.get("hallucination")
    analogy_transfer_imagination_raw = analogy_transfer_axis.get("imagination_raw")
    analogy_transfer_imagination_gated = analogy_transfer_axis.get("imagination_gated")
    analogy_transfer_hallucination_raw = analogy_transfer_axis.get("hallucination_raw")
    analogy_transfer_primitive_means = analogy_transfer_axis.get("primitive_means") or {}
    analogy_transfer_subtype_contributions = (
        analogy_transfer_axis.get("subtype_contributions")
        if isinstance(analogy_transfer_axis.get("subtype_contributions"), dict)
        else None
    )
    cross_task_fact_consistency_axis = score_cross_task_fact_consistency(
        gcw_task_scores=gcw_dual_axis_task_scores,
        hypospace_task_scores=hypospace_dual_axis_task_scores,
        closed_world_fact_task_scores=closed_world_fact_task_scores,
    )
    cross_task_fact_consistency_subtype_contributions = (
        cross_task_fact_consistency_axis.get("subtype_contributions")
        if isinstance(cross_task_fact_consistency_axis.get("subtype_contributions"), dict)
        else None
    )

    component_imagination_scores = {
        "UUT": uut_imagination_score,
        "PropConj": propconj_imagination_score,
        "MacGyver": macgyver_imagination_score,
        "CJST": cjst_imagination_score,
        "HypoUseSpace": hypospace_imagination_score,
        "GCW": gcw_imagination_score,
        "NeoCoder": neocoder_imagination_score,
        "AnalogyTransfer": analogy_transfer_imagination_score,
    }
    component_hallucination_scores = {
        "UUT": uut_hallucination_score,
        "PropConj": propconj_hallucination_score,
        "MacGyver": macgyver_hallucination_score,
        "CJST": cjst_hallucination_score,
        "HypoUseSpace": hypospace_hallucination_score,
        "GCW": gcw_hallucination_score,
        "NeoCoder": neocoder_hallucination_score,
        "AnalogyTransfer": analogy_transfer_hallucination_score,
    }
    component_imagination_raw = {
        "UUT": uut_imagination_raw,
        "PropConj": propconj_imagination_raw,
        "MacGyver": macgyver_imagination_raw,
        "CJST": cjst_imagination_raw,
        "HypoUseSpace": hypospace_imagination_raw,
        "GCW": gcw_imagination_raw,
        "NeoCoder": neocoder_imagination_raw,
        "AnalogyTransfer": analogy_transfer_imagination_raw,
    }
    component_hallucination_raw = {
        "UUT": uut_hallucination_raw,
        "PropConj": propconj_hallucination_raw,
        "MacGyver": macgyver_hallucination_raw,
        "CJST": cjst_hallucination_raw,
        "HypoUseSpace": hypospace_hallucination_raw,
        "GCW": gcw_hallucination_raw,
        "NeoCoder": neocoder_hallucination_raw,
        "AnalogyTransfer": analogy_transfer_hallucination_raw,
    }
    active_dual_components = {
        component for component in PRIMARY_DUAL_AXIS_COMPONENTS
        if (
            component_imagination_scores.get(component) is not None and
            component_hallucination_scores.get(component) is not None
        )
    }
    optional_active_dual_components = {
        component for component in OPTIONAL_DUAL_AXIS_COMPONENTS
        if (
            component_imagination_scores.get(component) is not None and
            component_hallucination_scores.get(component) is not None
        )
    }
    extended_dual_components = active_dual_components | optional_active_dual_components
    missing_primary_dual_components = [
        component for component in PRIMARY_DUAL_AXIS_COMPONENTS
        if component not in active_dual_components
    ]
    primary_dual_axis_gate_pass = not missing_primary_dual_components
    effective_imagination_weights = get_effective_dual_axis_component_weights_for_axis(
        PRIMARY_DUAL_AXIS_COMPONENTS,
        axis="imagination",
    )
    effective_hallucination_weights = get_effective_dual_axis_component_weights_for_axis(
        PRIMARY_DUAL_AXIS_COMPONENTS,
        axis="hallucination",
    )
    effective_dual_axis_weights = effective_imagination_weights
    effective_imagination_weights_report = {
        key: round(value, 6)
        for key, value in effective_imagination_weights.items()
    }
    effective_hallucination_weights_report = {
        key: round(value, 6)
        for key, value in effective_hallucination_weights.items()
    }
    effective_dual_axis_weights_report = effective_imagination_weights_report
    imagination_base_weights = get_dual_axis_component_base_weights_for_axis("imagination")
    hallucination_base_weights = get_dual_axis_component_base_weights_for_axis("hallucination")
    base_imagination_weights_report = {
        key: round(imagination_base_weights[key], 6)
        for key in DUAL_AXIS_COMPONENT_ORDER
        if key in PRIMARY_DUAL_AXIS_COMPONENTS and key in imagination_base_weights
    }
    base_hallucination_weights_report = {
        key: round(hallucination_base_weights[key], 6)
        for key in DUAL_AXIS_COMPONENT_ORDER
        if key in PRIMARY_DUAL_AXIS_COMPONENTS and key in hallucination_base_weights
    }
    base_dual_axis_weights_report = base_imagination_weights_report
    optional_extended_imagination_weights = get_effective_dual_axis_component_weights_for_axis(
        extended_dual_components,
        axis="imagination",
    )
    optional_extended_hallucination_weights = get_effective_dual_axis_component_weights_for_axis(
        extended_dual_components,
        axis="hallucination",
    )
    optional_extended_dual_axis_weights = optional_extended_imagination_weights
    optional_extended_dual_axis_weights_report = {
        key: round(value, 6)
        for key, value in optional_extended_imagination_weights.items()
    }
    imagination_axis_gate_pass = primary_dual_axis_gate_pass
    imagination_aggregation = aggregate_dual_axis_component_scores(
        component_imagination_scores,
        PRIMARY_DUAL_AXIS_COMPONENTS,
        axis="imagination",
    )
    hallucination_aggregation = aggregate_dual_axis_component_scores(
        component_hallucination_scores,
        PRIMARY_DUAL_AXIS_COMPONENTS,
        axis="hallucination",
    )
    imagination_score = (
        imagination_aggregation.get("score")
        if imagination_axis_gate_pass else None
    )
    hallucination_score = (
        hallucination_aggregation.get("score")
        if imagination_axis_gate_pass else None
    )
    optional_extended_imagination_score = (
        sum(
            component_imagination_scores[key] * optional_extended_imagination_weights[key]
            for key in extended_dual_components
        )
        if imagination_axis_gate_pass and optional_active_dual_components else None
    )
    optional_extended_hallucination_score = (
        sum(
            component_hallucination_scores[key] * optional_extended_hallucination_weights[key]
            for key in extended_dual_components
        )
        if imagination_axis_gate_pass and optional_active_dual_components else None
    )
    raw_active_dual_components = {
        component for component in PRIMARY_DUAL_AXIS_COMPONENTS
        if (
            component_imagination_raw.get(component) is not None and
            component_hallucination_raw.get(component) is not None
        )
    }
    raw_missing_primary_dual_components = [
        component for component in PRIMARY_DUAL_AXIS_COMPONENTS
        if component not in raw_active_dual_components
    ]
    raw_imagination_weights = get_effective_dual_axis_component_weights_for_axis(
        PRIMARY_DUAL_AXIS_COMPONENTS,
        axis="imagination",
    )
    raw_hallucination_weights = get_effective_dual_axis_component_weights_for_axis(
        PRIMARY_DUAL_AXIS_COMPONENTS,
        axis="hallucination",
    )
    raw_dual_axis_weights = raw_imagination_weights
    imagination_raw = (
        sum(component_imagination_raw[key] * raw_imagination_weights[key] for key in PRIMARY_DUAL_AXIS_COMPONENTS)
        if not raw_missing_primary_dual_components else None
    )
    hallucination_raw = (
        sum(component_hallucination_raw[key] * raw_hallucination_weights[key] for key in PRIMARY_DUAL_AXIS_COMPONENTS)
        if not raw_missing_primary_dual_components else None
    )
    optional_extended_raw_missing_components = [
        component for component in extended_dual_components
        if (
            component_imagination_raw.get(component) is None or
            component_hallucination_raw.get(component) is None
        )
    ]
    optional_extended_raw_imagination_weights = get_effective_dual_axis_component_weights_for_axis(
        extended_dual_components,
        axis="imagination",
    )
    optional_extended_raw_hallucination_weights = get_effective_dual_axis_component_weights_for_axis(
        extended_dual_components,
        axis="hallucination",
    )
    optional_extended_raw_weights = optional_extended_raw_imagination_weights
    optional_extended_imagination_raw = (
        sum(
            component_imagination_raw[key] * optional_extended_raw_imagination_weights[key]
            for key in extended_dual_components
        )
        if imagination_axis_gate_pass and optional_active_dual_components and not optional_extended_raw_missing_components else None
    )
    optional_extended_hallucination_raw = (
        sum(
            component_hallucination_raw[key] * optional_extended_raw_hallucination_weights[key]
            for key in extended_dual_components
        )
        if imagination_axis_gate_pass and optional_active_dual_components and not optional_extended_raw_missing_components else None
    )
    imagination_task_type_scores = {}
    imagination_task_type_scores_raw = {}
    hallucination_task_type_scores = {}
    hallucination_task_type_scores_raw = {}
    optional_imagination_task_type_scores = {}
    optional_imagination_task_type_scores_raw = {}
    optional_hallucination_task_type_scores = {}
    optional_hallucination_task_type_scores_raw = {}
    def add_dual_axis_task_type_score(component):
        is_primary = component in PRIMARY_DUAL_AXIS_COMPONENTS
        i_target = imagination_task_type_scores if is_primary else optional_imagination_task_type_scores
        i_raw_target = imagination_task_type_scores_raw if is_primary else optional_imagination_task_type_scores_raw
        h_target = hallucination_task_type_scores if is_primary else optional_hallucination_task_type_scores
        h_raw_target = hallucination_task_type_scores_raw if is_primary else optional_hallucination_task_type_scores_raw
        if component_imagination_scores.get(component) is not None:
            i_target[component] = round(component_imagination_scores[component], 4)
        if component_imagination_raw.get(component) is not None:
            i_raw_target[component] = round(component_imagination_raw[component], 4)
        if component_hallucination_scores.get(component) is not None:
            h_target[component] = round(component_hallucination_scores[component], 4)
        if component_hallucination_raw.get(component) is not None:
            h_raw_target[component] = round(component_hallucination_raw[component], 4)

    for component in DUAL_AXIS_COMPONENT_ORDER:
        add_dual_axis_task_type_score(component)
    imagination_primitive_means = {}
    for prefix, primitives in [
        ("UUT", uut_dual_axis_primitive_means),
        ("PropConj", propconj_primitive_means),
        ("MacGyver", macgyver_primitive_means),
        ("CJST", cjst_primitive_means),
        ("HypoUseSpace", hypospace_primitive_means),
        ("GCW", gcw_primitive_means),
        ("NeoCoder", neocoder_primitive_means),
        ("ClosedWorldFact", closed_world_fact_primitive_means),
        ("AnalogyTransfer", analogy_transfer_primitive_means),
    ]:
        if isinstance(primitives, dict):
            for field, value in primitives.items():
                imagination_primitive_means[f"{prefix}.{field}"] = value

    subtype_scores = aggregate_t1_subtype_scores({
        "UUT": uut_subtype_contributions,
        "PropConj": propconj_subtype_contributions,
        "MacGyver": macgyver_subtype_contributions,
        "CJST": cjst_subtype_contributions,
        "GCW": gcw_subtype_contributions,
        "HypoUseSpace": hypospace_subtype_contributions,
        "NeoCoder": neocoder_subtype_contributions,
        "ClosedWorldFact": closed_world_fact_subtype_contributions,
        "AnalogyTransfer": analogy_transfer_subtype_contributions,
        "CrossTaskConsistency": cross_task_fact_consistency_subtype_contributions,
    })
    dual_axis_task_counts = {
        "UUT": len(uut_dual_axis_task_scores),
        "PropConj": len(propconj_dual_axis_task_scores),
        "MacGyver": len(macgyver_dual_axis_task_scores),
        "CJST": len(cjst_dual_axis_task_scores),
        "HypoUseSpace": len(hypospace_dual_axis_task_scores),
        "GCW": len(gcw_dual_axis_task_scores),
        "NeoCoder": len(neocoder_dual_axis_task_scores),
        "AnalogyTransfer": len(analogy_transfer_task_scores),
    }
    primary_dual_axis_scored_tasks = sum(
        dual_axis_task_counts.get(component, 0)
        for component in PRIMARY_DUAL_AXIS_COMPONENTS
    )
    optional_dual_axis_scored_tasks = sum(
        dual_axis_task_counts.get(component, 0)
        for component in OPTIONAL_DUAL_AXIS_COMPONENTS
    )
    component_gate_pass_report = {
        "primary_dual_axis": primary_dual_axis_gate_pass,
        "UUT": uut_dual_axis_gate_pass,
        "PropConj": propconj_dual_axis_gate_pass,
        "MacGyver": macgyver_gate_pass,
        "CJST": cjst_gate_pass,
        "HypoUseSpace": hypospace_gate_pass,
        "GCW": gcw_gate_pass,
        "NeoCoder": neocoder_gate_pass,
        "AnalogyTransfer": analogy_transfer_gate_pass,
    }
    registry_metadata = get_v2_registry_metadata()
    subtype_schema_version = subtype_scores.get("version") if isinstance(subtype_scores, dict) else None
    typed_correlation_ready = subtype_scores_ready_for_correlation(subtype_scores)
    scoring_configuration = {
        "policy": "deterministic_output_only_scoring",
        "t1_assoc_version": T1_ASSOC_VERSION,
        "t1_calibration_policy": "benchmark_default",
        "t1_runtime_scoring_policy": "fixed output-only parameters",
        "macgyver_scoring_version": MACGYVER_DUAL_AXIS_VERSION,
        "macgyver_calibration_policy": MACGYVER_V3_CALIBRATION_POLICY,
        "macgyver_runtime_scoring_policy": MACGYVER_V3_RUNTIME_SCORING_POLICY,
        "cjst_scoring_version": CJST_DUAL_AXIS_VERSION,
        "cjst_calibration_policy": CJST_V3_CALIBRATION_POLICY,
        "cjst_runtime_scoring_policy": CJST_V3_RUNTIME_SCORING_POLICY,
        "disclosure": (
            "Fixed scoring parameters are applied directly to model outputs."
        ),
        "judge_policy": "no_external_llm_judge_in_primary_scoring",
    }

    eligibility_failures = []
    auxiliary_diagnostic_issues = []
    optional_dual_axis_issues = []
    if not creative_overall_availability_pass:
        eligibility_failures.append(
            f"creative availability {creative_availability:.2%}" if creative_availability is not None else "creative availability unavailable"
        )
    if not creative_overall_output_pass:
        eligibility_failures.append(
            f"creative coverage {creative_coverage:.2%}" if creative_coverage is not None else "creative coverage unavailable"
        )
    for task_type in CREATIVE_TASK_TYPES:
        availability = creative_task_availability.get(task_type)
        coverage = creative_task_coverages.get(task_type)
        if availability is not None and availability < MIN_CREATIVE_TASKTYPE_COVERAGE:
            eligibility_failures.append(f"{task_type} availability {availability:.2%}")
        if coverage is not None and coverage < MIN_CREATIVE_TASKTYPE_COVERAGE:
            eligibility_failures.append(f"{task_type} coverage {coverage:.2%}")
    if dat_total_prompts > 0 and (dat_availability is not None and dat_availability < MIN_DAT_COVERAGE):
        auxiliary_diagnostic_issues.append(f"DAT availability {dat_availability:.2%}")
    if dat_total_prompts > 0 and not dat_gate_pass:
        auxiliary_diagnostic_issues.append(
            f"DAT coverage {dat_coverage:.2%}" if dat_coverage is not None else "DAT coverage unavailable"
        )
    if cdat_total_prompts > 0 and (cdat_availability is not None and cdat_availability < MIN_CDAT_COVERAGE):
        auxiliary_diagnostic_issues.append(f"CDAT availability {cdat_availability:.2%}")
    if cdat_total_prompts > 0 and not cdat_gate_pass:
        auxiliary_diagnostic_issues.append(
            f"CDAT coverage {cdat_coverage:.2%}" if cdat_coverage is not None else "CDAT coverage unavailable"
        )
    if macgyver_total_prompts > 0 and (macgyver_availability is not None and macgyver_availability < MIN_MACGYVER_COVERAGE):
        eligibility_failures.append(f"MacGyver availability {macgyver_availability:.2%}")
    if macgyver_total_prompts > 0 and not macgyver_gate_pass:
        eligibility_failures.append(
            f"MacGyver coverage {macgyver_coverage:.2%}" if macgyver_coverage is not None else "MacGyver coverage unavailable"
        )
    if cjst_total_prompts > 0 and (cjst_availability is not None and cjst_availability < MIN_CJST_COVERAGE):
        eligibility_failures.append(f"CJST availability {cjst_availability:.2%}")
    if cjst_total_prompts > 0 and not cjst_gate_pass:
        eligibility_failures.append(
            f"CJST coverage {cjst_coverage:.2%}" if cjst_coverage is not None else "CJST coverage unavailable"
        )
    hypospace_issues = eligibility_failures if "HypoUseSpace" in PRIMARY_DUAL_AXIS_COMPONENTS else optional_dual_axis_issues
    gcw_issues = eligibility_failures if "GCW" in PRIMARY_DUAL_AXIS_COMPONENTS else optional_dual_axis_issues
    if hypospace_total_prompts > 0 and (hypospace_availability is not None and hypospace_availability < MIN_HYPOUSESPACE_COVERAGE):
        hypospace_issues.append(f"HypoUseSpace availability {hypospace_availability:.2%}")
    if hypospace_total_prompts > 0 and not hypospace_gate_pass:
        hypospace_issues.append(
            f"HypoUseSpace coverage {hypospace_coverage:.2%}" if hypospace_coverage is not None else "HypoUseSpace coverage unavailable"
        )
    if gcw_total_prompts > 0 and (gcw_availability is not None and gcw_availability < MIN_GCW_COVERAGE):
        gcw_issues.append(f"GCW availability {gcw_availability:.2%}")
    if gcw_total_prompts > 0 and not gcw_gate_pass:
        gcw_issues.append(
            f"GCW coverage {gcw_coverage:.2%}" if gcw_coverage is not None else "GCW coverage unavailable"
        )
    enhanced_diagnostic_issues = []
    neocoder_issues = eligibility_failures if "NeoCoder" in PRIMARY_DUAL_AXIS_COMPONENTS else enhanced_diagnostic_issues
    if neocoder_total_prompts > 0 and (neocoder_availability is not None and neocoder_availability < MIN_NEOCODER_COVERAGE):
        neocoder_issues.append(f"NeoCoder availability {neocoder_availability:.2%}")
    if neocoder_total_prompts > 0 and not neocoder_gate_pass:
        neocoder_issues.append(
            f"NeoCoder coverage {neocoder_coverage:.2%}" if neocoder_coverage is not None else "NeoCoder coverage unavailable"
        )
    calibration_diagnostic_issues = []
    if (
        closed_world_fact_total_prompts > 0
        and closed_world_fact_availability is not None
        and closed_world_fact_availability < MIN_CLOSED_WORLD_FACT_COVERAGE
    ):
        calibration_diagnostic_issues.append(
            f"ClosedWorldFact availability {closed_world_fact_availability:.2%}"
        )
    if closed_world_fact_total_prompts > 0 and not closed_world_fact_gate_pass:
        calibration_diagnostic_issues.append(
            f"ClosedWorldFact coverage {closed_world_fact_coverage:.2%}"
            if closed_world_fact_coverage is not None else
            "ClosedWorldFact coverage unavailable"
        )
    challenge_diagnostic_issues = []
    analogy_transfer_issues = (
        eligibility_failures
        if "AnalogyTransfer" in PRIMARY_DUAL_AXIS_COMPONENTS else
        challenge_diagnostic_issues
    )
    if (
        analogy_transfer_total_prompts > 0
        and analogy_transfer_availability is not None
        and analogy_transfer_availability < MIN_ANALOGY_TRANSFER_COVERAGE
    ):
        analogy_transfer_issues.append(
            f"AnalogyTransfer availability {analogy_transfer_availability:.2%}"
        )
    if analogy_transfer_total_prompts > 0 and not analogy_transfer_gate_pass:
        analogy_transfer_issues.append(
            f"AnalogyTransfer coverage {analogy_transfer_coverage:.2%}"
            if analogy_transfer_coverage is not None else
            "AnalogyTransfer coverage unavailable"
        )
    for component in missing_primary_dual_components:
        eligibility_failures.append(f"{component} primary dual-axis score unavailable")

    ranking_eligible = (
        primary_dual_axis_gate_pass and
        imagination_score is not None and
        hallucination_score is not None and
        not eligibility_failures
    )

    dt_total_score = None
    if novelty_axis_score is not None and flexibility_axis_score is not None:
        dt_total_score = round(
            DT_TOTAL_NOVELTY_WEIGHT * novelty_axis_score +
            DT_TOTAL_FLEXIBILITY_WEIGHT * flexibility_axis_score,
            4,
        )

    for task_result in task_results:
        attach_task_atom_signals(task_result)

    overall_summary = {
        "axes": {
            "dt_total": {
                "score": dt_total_score,
                "role": "supporting",
                "coverage_gate_pass": novelty_axis_score is not None and flexibility_axis_score is not None,
                "formula": f"{DT_TOTAL_NOVELTY_WEIGHT:.2f}*Novelty + {DT_TOTAL_FLEXIBILITY_WEIGHT:.2f}*Flexibility",
            },
            "novelty": {
                "role": "supporting",
                "score": novelty_axis_score,
                "coverage_gate_pass": novelty_axis_gate_pass,
                "num_components": num_categories_scored,
                "component_scores": category_originality_scores,
                "component_scores_raw": category_originality_scores_raw,
                "component_weights": effective_novelty_weights_report,
                "base_component_weights": base_novelty_weights_report,
                "formula": novelty_formula,
                "creative_hybrid_formula": {
                    task_name: get_common_answer_bank_hybrid_formula(task_name)
                    for task_name in CREATIVE_TASK_TYPES
                },
                "weighting_note": "Supporting scorer output: UUT/PropConj are grounded creative tasks; DAT/CDAT are auxiliary lexical diagnostics.",
                "component_coverage": {
                    **creative_task_coverages,
                    "DAT": round(dat_coverage, 4) if dat_coverage is not None else None,
                    "CDAT": round(cdat_coverage, 4) if cdat_coverage is not None else None,
                },
                "component_availability": {
                    **{task_type: (round(value, 4) if value is not None else None) for task_type, value in creative_task_availability.items()},
                    "DAT": round(dat_availability, 4) if dat_availability is not None else None,
                    "CDAT": round(cdat_availability, 4) if cdat_availability is not None else None,
                },
                "scale": "Supporting scorer output: UUT/PropConj = hybrid raw novelty after white-box groundedness/property-validity penalty; DAT/CDAT = normalized auxiliary lexical scores; FF excluded.",
            },
            "flexibility": {
                "role": "supporting",
                "score": flexibility_axis_score,
                "coverage_gate_pass": flexibility_axis_gate_pass,
                "embedding_composite": (round(model_embedding_flexibility_score, 4)
                                         if model_embedding_flexibility_score is not None else None),
                "ontological_composite": (round(model_ontological_flexibility_score, 4)
                                           if model_ontological_flexibility_score is not None else None),
                "embedding_pairwise_distance": (
                    round(model_avg_emb_pairwise_distance, 4)
                    if model_avg_emb_pairwise_distance is not None else None
                ),
                "embedding_adjacent_distance": (
                    round(model_avg_emb_adjacent_distance, 4)
                    if model_avg_emb_adjacent_distance is not None else None
                ),
                "embedding_cluster_entropy": (
                    round(model_avg_emb_cluster_entropy, 4)
                    if model_avg_emb_cluster_entropy is not None else None
                ),
                "embedding_weight": w_emb,
                "ontological_weight": w_ont,
                "embedding_is_degenerate": embedding_is_degenerate,
                "ontological_source": wn_analyzer.get_source_label(),
                "formula": flex_formula_str,
            },
            "groundedness": {
                "role": "supporting",
                "version": WHITE_BOX_GROUNDEDNESS_VERSION,
                "score": round(groundedness_axis_score, 4) if groundedness_axis_score is not None else None,
                "score_novel_only": round(groundedness_axis_score_novel, 4) if groundedness_axis_score_novel is not None else None,
                "coverage_gate_pass": groundedness_axis_gate_pass,
                "formula": "mean_tasktype_groundedness_v5p0_signal",
                "penalty_formula": "cohort_relative_sigmoid_deficit_v5",
                "reference_cohort": {
                    "schema": groundedness_reference_cohort.get("schema") if isinstance(groundedness_reference_cohort, dict) else None,
                    "source": groundedness_reference_cohort.get("source") if isinstance(groundedness_reference_cohort, dict) else None,
                    "fallback": groundedness_reference_cohort.get("fallback") if isinstance(groundedness_reference_cohort, dict) else None,
                    "per_task": groundedness_reference_cohort.get("per_task") if isinstance(groundedness_reference_cohort, dict) else None,
                },
                "confidence_weighted_mean": (
                    round(confidence_weighted_groundedness, 4)
                    if confidence_weighted_groundedness is not None else None
                ),
                "confidence_weighted_mean_novel_only": (
                    round(confidence_weighted_groundedness_novel, 4)
                    if confidence_weighted_groundedness_novel is not None else None
                ),
                "mean_penalty": round(mean_penalty, 4) if mean_penalty is not None else None,
                "penalty_rate": round(penalty_rate, 4) if penalty_rate is not None else None,
                "low_groundedness_rate": round(low_groundedness_rate, 4) if low_groundedness_rate is not None else None,
                "low_groundedness_rate_novel_only": (
                    round(low_groundedness_rate_novel, 4)
                    if low_groundedness_rate_novel is not None else None
                ),
                "mean_confidence": round(mean_confidence, 4) if mean_confidence is not None else None,
                "scored_coverage": round(scored_coverage, 4) if scored_coverage is not None else None,
                "groundedness_scored_ideas": total_groundedness_score_count,
                "task_type_scores": model_ground_task_scores,
                "task_type_scores_novel_only": model_ground_task_scores_novel,
                "task_type_scores_raw": model_ground_task_scores_raw,
                "task_type_scores_novel_only_raw": model_ground_task_scores_novel_raw,
            },
            "imagination": {
                "role": "primary",
                "version": DUAL_AXIS_REPORT_VERSION,
                "score": round(imagination_score, 4) if imagination_score is not None else None,
                "raw_score": round(imagination_raw, 4) if imagination_raw is not None else None,
                "coverage_gate_pass": imagination_axis_gate_pass,
                "formula": (
                    imagination_aggregation.get("formula")
                    or format_dual_axis_component_formula(effective_imagination_weights, "I")
                    or "weighted residualized imagination across available dual-axis task families"
                ),
                "task_type_scores": imagination_task_type_scores,
                "task_type_scores_raw": imagination_task_type_scores_raw,
                "optional_task_type_scores": optional_imagination_task_type_scores,
                "optional_task_type_scores_raw": optional_imagination_task_type_scores_raw,
                "component_weights": effective_imagination_weights_report,
                "base_component_weights": base_imagination_weights_report,
                "component_gate_pass": component_gate_pass_report,
                "aggregation_policy": imagination_aggregation.get("aggregation_policy"),
                "residualization": {
                    "aggregation": "weighted mean of component residualized imagination scores",
                    "mandatory_components": list(PRIMARY_DUAL_AXIS_COMPONENTS),
                    "optional_components": list(OPTIONAL_DUAL_AXIS_COMPONENTS),
                    "UUT": {"beta_IH": UUT_DUAL_AXIS_BETA_IH, "beta_HI": UUT_DUAL_AXIS_BETA_HI},
                    "PropConj": {"beta_IH": PROPCONJ_DUAL_AXIS_BETA_IH, "beta_HI": PROPCONJ_DUAL_AXIS_BETA_HI},
                    "MacGyver": {"beta_IH": MACGYVER_DUAL_AXIS_BETA_IH, "beta_HI": MACGYVER_DUAL_AXIS_BETA_HI},
                    "CJST": {"beta_IH": CJST_DUAL_AXIS_BETA_IH, "beta_HI": CJST_DUAL_AXIS_BETA_HI},
                    "HypoUseSpace": {"beta_IH": HYPOUSESPACE_DUAL_AXIS_BETA_IH, "beta_HI": HYPOUSESPACE_DUAL_AXIS_BETA_HI},
                    "GCW": {"beta_IH": GCW_DUAL_AXIS_BETA_IH, "beta_HI": GCW_DUAL_AXIS_BETA_HI},
                    "NeoCoder": {"beta_IH": NEOCODER_DUAL_AXIS_BETA_IH, "beta_HI": NEOCODER_DUAL_AXIS_BETA_HI},
                    "AnalogyTransfer": analogy_transfer_axis.get("residualization"),
                },
                "primitive_means": imagination_primitive_means,
                "scored_tasks": primary_dual_axis_scored_tasks,
                "optional_scored_tasks": optional_dual_axis_scored_tasks,
            },
            "hallucination": {
                "role": "primary",
                "version": DUAL_AXIS_REPORT_VERSION,
                "score": round(hallucination_score, 4) if hallucination_score is not None else None,
                "raw_score": round(hallucination_raw, 4) if hallucination_raw is not None else None,
                "coverage_gate_pass": imagination_axis_gate_pass,
                "direction": "lower_is_better",
                "formula": (
                    hallucination_aggregation.get("formula")
                    or format_dual_axis_component_formula(effective_hallucination_weights, "H")
                    or "weighted residualized hallucination across available dual-axis task families"
                ),
                "task_type_scores": hallucination_task_type_scores,
                "task_type_scores_raw": hallucination_task_type_scores_raw,
                "optional_task_type_scores": optional_hallucination_task_type_scores,
                "optional_task_type_scores_raw": optional_hallucination_task_type_scores_raw,
                "component_weights": effective_hallucination_weights_report,
                "base_component_weights": base_hallucination_weights_report,
                "component_gate_pass": component_gate_pass_report,
                "aggregation_policy": hallucination_aggregation.get("aggregation_policy"),
                "residualization": {
                    "aggregation": "weighted mean of component residualized hallucination scores",
                    "mandatory_components": list(PRIMARY_DUAL_AXIS_COMPONENTS),
                    "optional_components": list(OPTIONAL_DUAL_AXIS_COMPONENTS),
                    "UUT": {"beta_IH": UUT_DUAL_AXIS_BETA_IH, "beta_HI": UUT_DUAL_AXIS_BETA_HI},
                    "PropConj": {"beta_IH": PROPCONJ_DUAL_AXIS_BETA_IH, "beta_HI": PROPCONJ_DUAL_AXIS_BETA_HI},
                    "MacGyver": {"beta_IH": MACGYVER_DUAL_AXIS_BETA_IH, "beta_HI": MACGYVER_DUAL_AXIS_BETA_HI},
                    "CJST": {"beta_IH": CJST_DUAL_AXIS_BETA_IH, "beta_HI": CJST_DUAL_AXIS_BETA_HI},
                    "HypoUseSpace": {"beta_IH": HYPOUSESPACE_DUAL_AXIS_BETA_IH, "beta_HI": HYPOUSESPACE_DUAL_AXIS_BETA_HI},
                    "GCW": {"beta_IH": GCW_DUAL_AXIS_BETA_IH, "beta_HI": GCW_DUAL_AXIS_BETA_HI},
                    "NeoCoder": {"beta_IH": NEOCODER_DUAL_AXIS_BETA_IH, "beta_HI": NEOCODER_DUAL_AXIS_BETA_HI},
                    "AnalogyTransfer": analogy_transfer_axis.get("residualization"),
                },
                "primitive_means": imagination_primitive_means,
                "scored_tasks": primary_dual_axis_scored_tasks,
                "optional_scored_tasks": optional_dual_axis_scored_tasks,
            },
            "subtype_scores": subtype_scores,
            "atom_signals": (subtype_scores or {}).get("atom_signals", {}) if isinstance(subtype_scores, dict) else {},
            "cross_task_fact_consistency": cross_task_fact_consistency_axis,
            "dual_axis": {
                "role": "primary",
                "version": DUAL_AXIS_REPORT_VERSION,
                "formula_version": DUAL_AXIS_REPORT_VERSION,
                "score": round(imagination_score, 4) if imagination_score is not None else None,
                "imagination": round(imagination_score, 4) if imagination_score is not None else None,
                "hallucination": round(hallucination_score, 4) if hallucination_score is not None else None,
                "imagination_raw": round(imagination_raw, 4) if imagination_raw is not None else None,
                "hallucination_raw": round(hallucination_raw, 4) if hallucination_raw is not None else None,
                "coverage_gate_pass": imagination_axis_gate_pass,
                "task_type_scores": {
                    "imagination": imagination_task_type_scores,
                    "hallucination": hallucination_task_type_scores,
                    "imagination_raw": imagination_task_type_scores_raw,
                    "hallucination_raw": hallucination_task_type_scores_raw,
                    "optional_imagination": optional_imagination_task_type_scores,
                    "optional_hallucination": optional_hallucination_task_type_scores,
                    "optional_imagination_raw": optional_imagination_task_type_scores_raw,
                    "optional_hallucination_raw": optional_hallucination_task_type_scores_raw,
                },
                "component_weights": effective_imagination_weights_report,
                "hallucination_component_weights": effective_hallucination_weights_report,
                "base_component_weights": base_imagination_weights_report,
                "base_hallucination_component_weights": base_hallucination_weights_report,
                "component_gate_pass": component_gate_pass_report,
                "aggregation": {
                    "imagination": imagination_aggregation.get("aggregation_policy"),
                    "hallucination": hallucination_aggregation.get("aggregation_policy"),
                },
                "calibration": {
                    "mandatory_components": list(PRIMARY_DUAL_AXIS_COMPONENTS),
                    "optional_components": list(OPTIONAL_DUAL_AXIS_COMPONENTS),
                    "aggregation": {
                        "imagination": imagination_aggregation.get("aggregation_policy"),
                        "hallucination": hallucination_aggregation.get("aggregation_policy"),
                    },
                    "UUT": {"beta_IH": UUT_DUAL_AXIS_BETA_IH, "beta_HI": UUT_DUAL_AXIS_BETA_HI},
                    "PropConj": {"beta_IH": PROPCONJ_DUAL_AXIS_BETA_IH, "beta_HI": PROPCONJ_DUAL_AXIS_BETA_HI},
                    "MacGyver": {"beta_IH": MACGYVER_DUAL_AXIS_BETA_IH, "beta_HI": MACGYVER_DUAL_AXIS_BETA_HI},
                    "CJST": {"beta_IH": CJST_DUAL_AXIS_BETA_IH, "beta_HI": CJST_DUAL_AXIS_BETA_HI},
                    "HypoUseSpace": {"beta_IH": HYPOUSESPACE_DUAL_AXIS_BETA_IH, "beta_HI": HYPOUSESPACE_DUAL_AXIS_BETA_HI},
                    "GCW": {"beta_IH": GCW_DUAL_AXIS_BETA_IH, "beta_HI": GCW_DUAL_AXIS_BETA_HI},
                    "NeoCoder": {"beta_IH": NEOCODER_DUAL_AXIS_BETA_IH, "beta_HI": NEOCODER_DUAL_AXIS_BETA_HI},
                    "AnalogyTransfer": analogy_transfer_axis.get("residualization"),
                },
            },
            "dual_axis_optional_extended": {
                "role": "diagnostic",
                "version": DUAL_AXIS_REPORT_VERSION,
                "formula_version": DUAL_AXIS_REPORT_VERSION,
                "score": (
                    round(optional_extended_imagination_score, 4)
                    if optional_extended_imagination_score is not None else None
                ),
                "imagination": (
                    round(optional_extended_imagination_score, 4)
                    if optional_extended_imagination_score is not None else None
                ),
                "hallucination": (
                    round(optional_extended_hallucination_score, 4)
                    if optional_extended_hallucination_score is not None else None
                ),
                "imagination_raw": (
                    round(optional_extended_imagination_raw, 4)
                    if optional_extended_imagination_raw is not None else None
                ),
                "hallucination_raw": (
                    round(optional_extended_hallucination_raw, 4)
                    if optional_extended_hallucination_raw is not None else None
                ),
                "coverage_gate_pass": bool(imagination_axis_gate_pass and optional_active_dual_components),
                "included_optional_components": sorted(optional_active_dual_components),
                "component_weights": optional_extended_dual_axis_weights_report,
                "note": "Diagnostic only: optional dual-axis task families are excluded from the main benchmark score.",
            },
            "propconj_dual_axis": {
                "version": PROPCONJ_DUAL_AXIS_VERSION,
                "t1_assoc_version": T1_ASSOC_VERSION,
                "quality_mass_top6": propconj_axis.get("quality_mass_top6"),
                "diversity_eff": propconj_axis.get("diversity_eff"),
                "hard_valid_ratio": propconj_axis.get("hard_valid_ratio"),
                "coverage_gate_pass": propconj_dual_axis_gate_pass,
                "score": round(propconj_imagination_score, 4) if propconj_imagination_score is not None else None,
                "imagination": round(propconj_imagination_score, 4) if propconj_imagination_score is not None else None,
                "hallucination": round(propconj_hallucination_score, 4) if propconj_hallucination_score is not None else None,
                "imagination_raw": round(propconj_imagination_raw, 4) if propconj_imagination_raw is not None else None,
                "hallucination_raw": round(propconj_hallucination_raw, 4) if propconj_hallucination_raw is not None else None,
                "primitive_means": propconj_primitive_means,
                "subtype_contributions": propconj_subtype_contributions,
                "task_scores": propconj_dual_axis_task_scores,
                "residualization": propconj_axis.get("residualization"),
                "formula": propconj_axis.get("formula"),
            },
            "macgyver_dual_axis": {
                "version": MACGYVER_DUAL_AXIS_VERSION,
                "calibration_policy": MACGYVER_V3_CALIBRATION_POLICY,
                "runtime_scoring_policy": MACGYVER_V3_RUNTIME_SCORING_POLICY,
                "primary_task_ids": [
                    task.get("task_id") for task in macgyver_results
                    if isinstance(task, dict) and task.get("task_id")
                ],
                "boundary_diagnostic_task_ids": [
                    task.get("task_id") for task in macgyver_boundary_results
                    if isinstance(task, dict) and task.get("task_id")
                ],
                "plan_count": MACGYVER_OUTPUT_COUNT,
                "quality_mass_top3": macgyver_axis.get("quality_mass_top3"),
                "elite_tail": macgyver_axis.get("elite_tail"),
                "strategy_diversity_eff": macgyver_axis.get("strategy_diversity_eff"),
                "hard_valid_ratio": macgyver_axis.get("hard_valid_ratio"),
                "common_bank_coverage": macgyver_axis.get("common_bank_coverage"),
                "coverage_gate_pass": macgyver_gate_pass,
                "score": round(macgyver_imagination_score, 4) if macgyver_imagination_score is not None else None,
                "imagination": round(macgyver_imagination_score, 4) if macgyver_imagination_score is not None else None,
                "hallucination": round(macgyver_hallucination_score, 4) if macgyver_hallucination_score is not None else None,
                "imagination_raw": round(macgyver_imagination_raw, 4) if macgyver_imagination_raw is not None else None,
                "hallucination_raw": round(macgyver_hallucination_raw, 4) if macgyver_hallucination_raw is not None else None,
                "primitive_means": macgyver_primitive_means,
                "boundary_record_means": macgyver_boundary_record_means,
                "subtype_contributions": macgyver_subtype_contributions,
                "task_scores": macgyver_dual_axis_task_scores,
                "residualization": macgyver_axis.get("residualization"),
                "formula": macgyver_axis.get("formula"),
                "solvability_accuracy": macgyver_axis.get("solvability_accuracy"),
            },
            "macgyver_boundary_diagnostic": {
                "version": MACGYVER_DUAL_AXIS_VERSION,
                "calibration_policy": "not_strength_calibrated",
                "ran": bool(macgyver_boundary_results),
                "task_ids": [
                    task.get("task_id") for task in macgyver_boundary_results
                    if isinstance(task, dict) and task.get("task_id")
                ],
                "hallucination": macgyver_boundary_axis.get("hallucination"),
                "hallucination_raw": macgyver_boundary_axis.get("hallucination_raw"),
                "boundary_record_means": macgyver_boundary_axis.get("boundary_record_means"),
                "subtype_contributions": macgyver_boundary_axis.get("subtype_contributions"),
                "task_scores": macgyver_boundary_task_scores,
                "solvability_accuracy": macgyver_boundary_axis.get("solvability_accuracy"),
            },
            "cjst_dual_axis": {
                "version": CJST_DUAL_AXIS_VERSION,
                "calibration_policy": CJST_V3_CALIBRATION_POLICY,
                "runtime_scoring_policy": CJST_V3_RUNTIME_SCORING_POLICY,
                "primary_task_ids": [
                    task.get("task_id") for task in cjst_results
                    if isinstance(task, dict) and task.get("task_id")
                ],
                "output_count": CJST_OUTPUT_COUNT,
                "max_tokens": get_task_max_tokens("CJST"),
                "common_consequence_bank_coverage": get_cjst_common_consequence_bank_coverage([
                    task.get("task_id") for task in cjst_results
                    if isinstance(task, dict) and task.get("task_id")
                ]),
                "coverage_gate_pass": cjst_gate_pass,
                "score": round(cjst_imagination_score, 4) if cjst_imagination_score is not None else None,
                "imagination": round(cjst_imagination_score, 4) if cjst_imagination_score is not None else None,
                "hallucination": round(cjst_hallucination_score, 4) if cjst_hallucination_score is not None else None,
                "imagination_raw": round(cjst_imagination_raw, 4) if cjst_imagination_raw is not None else None,
                "hallucination_raw": round(cjst_hallucination_raw, 4) if cjst_hallucination_raw is not None else None,
                "quality_mass_top6": cjst_axis.get("quality_mass_top6"),
                "elite_tail_top3": cjst_axis.get("elite_tail_top3"),
                "tier_balanced_depth": cjst_axis.get("tier_balanced_depth"),
                "mechanism_diversity_eff": cjst_axis.get("mechanism_diversity_eff"),
                "hard_valid_ratio": cjst_axis.get("hard_valid_ratio"),
                "common_bank_coverage": cjst_axis.get("common_bank_coverage"),
                "coverage": round(cjst_coverage, 4) if cjst_coverage is not None else None,
                "availability": round(cjst_availability, 4) if cjst_availability is not None else None,
                "primitive_means": cjst_primitive_means,
                "subtype_contributions": cjst_subtype_contributions,
                "task_scores": cjst_dual_axis_task_scores,
                "residualization": cjst_axis.get("residualization"),
                "formula": cjst_axis.get("formula"),
            },
            "hypospace_dual_axis": {
                "version": HYPOUSESPACE_DUAL_AXIS_VERSION,
                "calibration_policy": HYPOUSESPACE_V3_CALIBRATION_POLICY,
                "runtime_scoring_policy": HYPOUSESPACE_V3_RUNTIME_SCORING_POLICY,
                "primary_task_ids": [
                    task.get("task_id") for task in hypospace_results
                    if isinstance(task, dict) and task.get("task_id")
                ],
                "boundary_diagnostic_task_ids": list(HYPOUSESPACE_BOUNDARY_DIAGNOSTIC_TASK_IDS),
                "output_count": HYPOUSESPACE_OUTPUT_COUNT,
                "max_tokens": get_task_max_tokens("HypoUseSpace"),
                "common_hypothesis_bank_version": "hypospace_common_hypothesis_bank_v3",
                "valid_match_alias_version": "hypospace_valid_match_aliases_v3",
                "common_hypothesis_bank_coverage": get_hypospace_common_hypothesis_bank_coverage([
                    task.get("task_id") for task in hypospace_results
                    if isinstance(task, dict) and task.get("task_id")
                ]),
                "valid_match_alias_coverage": get_hypospace_valid_match_alias_coverage([
                    task.get("task_id") for task in hypospace_results
                    if isinstance(task, dict) and task.get("task_id")
                ]),
                "coverage_gate_pass": hypospace_gate_pass,
                "score": round(hypospace_imagination_score, 4) if hypospace_imagination_score is not None else None,
                "imagination": round(hypospace_imagination_score, 4) if hypospace_imagination_score is not None else None,
                "hallucination": round(hypospace_hallucination_score, 4) if hypospace_hallucination_score is not None else None,
                "imagination_raw": round(hypospace_imagination_raw, 4) if hypospace_imagination_raw is not None else None,
                "hallucination_raw": round(hypospace_hallucination_raw, 4) if hypospace_hallucination_raw is not None else None,
                "quality_mass_top3": hypospace_axis.get("quality_mass_top3"),
                "elite_tail_top2": hypospace_axis.get("elite_tail_top2"),
                "mechanism_diversity_eff": hypospace_axis.get("mechanism_diversity_eff"),
                "evidence_synthesis_coverage": hypospace_axis.get("evidence_synthesis_coverage"),
                "hard_valid_ratio": hypospace_axis.get("hard_valid_ratio"),
                "soft_match_quality": hypospace_axis.get("soft_match_quality"),
                "common_bank_coverage": hypospace_axis.get("common_bank_coverage"),
                "alias_coverage": hypospace_axis.get("alias_coverage"),
                "coverage": round(hypospace_coverage, 4) if hypospace_coverage is not None else None,
                "availability": round(hypospace_availability, 4) if hypospace_availability is not None else None,
                "primitive_means": hypospace_primitive_means,
                "subtype_contributions": hypospace_subtype_contributions,
                "task_scores": hypospace_dual_axis_task_scores,
                "residualization": hypospace_axis.get("residualization"),
                "formula": hypospace_axis.get("formula"),
                "no_valid_accuracy": hypospace_axis.get("no_valid_accuracy"),
            },
            "hypospace_boundary_diagnostic": hypospace_boundary_axis,
            "gcw_dual_axis": {
                "version": GCW_DUAL_AXIS_VERSION,
                "calibration_policy": GCW_V3_CALIBRATION_POLICY,
                "runtime_scoring_policy": GCW_V3_RUNTIME_SCORING_POLICY,
                "primary_task_ids": [
                    task.get("task_id") for task in gcw_results
                    if isinstance(task, dict) and task.get("task_id")
                ],
                "output_beat_count": GCW_BEAT_COUNT,
                "max_tokens": get_task_max_tokens("GCW"),
                "common_story_bank_coverage": get_gcw_common_story_bank_coverage([
                    task.get("task_id") for task in gcw_results
                    if isinstance(task, dict) and task.get("task_id")
                ]),
                "entity_alias_coverage": get_gcw_entity_alias_coverage([
                    task.get("task_id") for task in gcw_results
                    if isinstance(task, dict) and task.get("task_id")
                ]),
                "common_story_bank_version": gcw_axis.get("common_story_bank_version"),
                "entity_alias_bank_version": gcw_axis.get("entity_alias_bank_version"),
                "coverage_gate_pass": gcw_gate_pass,
                "score": round(gcw_imagination_score, 4) if gcw_imagination_score is not None else None,
                "imagination": round(gcw_imagination_score, 4) if gcw_imagination_score is not None else None,
                "hallucination": round(gcw_hallucination_score, 4) if gcw_hallucination_score is not None else None,
                "imagination_raw": round(gcw_imagination_raw, 4) if gcw_imagination_raw is not None else None,
                "imagination_gated": gcw_axis.get("imagination_gated"),
                "hallucination_raw": round(gcw_hallucination_raw, 4) if gcw_hallucination_raw is not None else None,
                "coverage": round(gcw_coverage, 4) if gcw_coverage is not None else None,
                "availability": round(gcw_availability, 4) if gcw_availability is not None else None,
                "primitive_means": gcw_primitive_means,
                "grounded_turn_quality": gcw_axis.get("grounded_turn_quality"),
                "causal_payoff": gcw_axis.get("causal_payoff"),
                "top3_scene_specificity": gcw_axis.get("top3_scene_specificity"),
                "arc_diversity_eff": gcw_axis.get("arc_diversity_eff"),
                "hard_valid_ledger_ratio": gcw_axis.get("hard_valid_ledger_ratio"),
                "common_bank_coverage": gcw_axis.get("common_bank_coverage"),
                "subtype_contributions": gcw_subtype_contributions,
                "task_scores": gcw_dual_axis_task_scores,
                "residualization": gcw_axis.get("residualization"),
                "formula": gcw_axis.get("formula"),
                "gcw_ttcw_proxy": {
                    key: gcw_primitive_means.get(key)
                    for key in ["F_story", "X_story", "O_story", "E_story"]
                    if key in gcw_primitive_means
                },
                "gcw_fact_grounding": gcw_primitive_means.get("fact_grounding"),
            },
            "neocoder_dual_axis": {
                "role": "primary" if "NeoCoder" in PRIMARY_DUAL_AXIS_COMPONENTS else "enhanced_diagnostic",
                "version": NEOCODER_DUAL_AXIS_VERSION,
                "calibration_policy": NEOCODER_V3_CALIBRATION_POLICY,
                "runtime_scoring_policy": NEOCODER_V3_RUNTIME_SCORING_POLICY,
                "test_visibility_policy": NEOCODER_V3_TEST_VISIBILITY_POLICY,
                "primary_task_ids": [
                    task.get("task_id") for task in neocoder_results
                    if isinstance(task, dict) and task.get("task_id")
                ],
                "output_count": NEOCODER_OUTPUT_COUNT,
                "max_tokens": get_task_max_tokens("NeoCoder"),
                "common_solution_bank_version": NEOCODER_V3_COMMON_SOLUTION_BANK_VERSION,
                "task_overlay_version": NEOCODER_V3_TASK_OVERLAY_VERSION,
                "technique_alias_version": NEOCODER_V3_TECHNIQUE_ALIAS_VERSION,
                "common_solution_bank_coverage": get_neocoder_common_solution_bank_coverage([
                    task.get("task_id") for task in neocoder_results
                    if isinstance(task, dict) and task.get("task_id")
                ]),
                "task_overlay_coverage": get_neocoder_task_overlay_coverage([
                    task.get("task_id") for task in neocoder_results
                    if isinstance(task, dict) and task.get("task_id")
                ]),
                "technique_alias_coverage": get_neocoder_technique_alias_coverage(),
                "coverage_gate_pass": neocoder_gate_pass,
                "score": round(neocoder_imagination_score, 4) if neocoder_imagination_score is not None else None,
                "imagination": round(neocoder_imagination_score, 4) if neocoder_imagination_score is not None else None,
                "hallucination": round(neocoder_hallucination_score, 4) if neocoder_hallucination_score is not None else None,
                "imagination_raw": round(neocoder_imagination_raw, 4) if neocoder_imagination_raw is not None else None,
                "imagination_gated": round(neocoder_imagination_gated, 4) if neocoder_imagination_gated is not None else None,
                "hallucination_raw": round(neocoder_hallucination_raw, 4) if neocoder_hallucination_raw is not None else None,
                "coverage": round(neocoder_coverage, 4) if neocoder_coverage is not None else None,
                "availability": round(neocoder_availability, 4) if neocoder_availability is not None else None,
                "primitive_means": neocoder_primitive_means,
                "functional_quality": neocoder_axis.get("functional_quality"),
                "public_pass_rate": neocoder_axis.get("public_pass_rate"),
                "hidden_pass_rate": neocoder_axis.get("hidden_pass_rate"),
                "metamorphic_pass_rate": neocoder_axis.get("metamorphic_pass_rate"),
                "strategy_rarity": neocoder_axis.get("strategy_rarity"),
                "implementation_depth": neocoder_axis.get("implementation_depth"),
                "constraint_quality": neocoder_axis.get("constraint_quality"),
                "denial_adaptation": neocoder_axis.get("denial_adaptation"),
                "anti_overfit_gate": neocoder_axis.get("anti_overfit_gate"),
                "subtype_contributions": neocoder_subtype_contributions,
                "task_scores": neocoder_dual_axis_task_scores,
                "residualization": neocoder_axis.get("residualization"),
                "formula": neocoder_axis.get("formula"),
                "note": (
                    "Primary T6 component in the active experiment profile; executes model-generated Python in a restricted subprocess."
                    if "NeoCoder" in PRIMARY_DUAL_AXIS_COMPONENTS else
                    "Enhanced diagnostic only: NeoCoder executes model-generated Python in a restricted subprocess and is excluded from primary rankings."
                ),
            },
            "closed_world_fact_calibration": {
                "role": "hallucination_calibration",
                "version": closed_world_fact_axis.get("version"),
                "coverage_gate_pass": bool(closed_world_fact_gate_pass and closed_world_fact_task_scores),
                "score": round(closed_world_fact_score, 4) if closed_world_fact_score is not None else None,
                "hallucination": (
                    round(closed_world_fact_hallucination, 4)
                    if closed_world_fact_hallucination is not None else None
                ),
                "hallucination_raw": (
                    round(closed_world_fact_hallucination_raw, 4)
                    if closed_world_fact_hallucination_raw is not None else None
                ),
                "imagination": None,
                "imagination_raw": None,
                "coverage": (
                    round(closed_world_fact_coverage, 4)
                    if closed_world_fact_coverage is not None else None
                ),
                "availability": (
                    round(closed_world_fact_availability, 4)
                    if closed_world_fact_availability is not None else None
                ),
                "primitive_means": closed_world_fact_primitive_means,
                "subtype_contributions": closed_world_fact_subtype_contributions,
                "task_scores": closed_world_fact_task_scores,
                "formula": closed_world_fact_axis.get("formula"),
                "note": "Calibration diagnostic only: ClosedWorldFact is excluded from imagination, primary dual-axis, optional dual-axis, and ranking eligibility.",
            },
            "analogy_transfer_challenge": {
                "role": "primary" if "AnalogyTransfer" in PRIMARY_DUAL_AXIS_COMPONENTS else "challenge_diagnostic",
                "version": analogy_transfer_axis.get("version"),
                "calibration_policy": ANALOGY_TRANSFER_V3_CALIBRATION_POLICY,
                "runtime_scoring_policy": ANALOGY_TRANSFER_V3_RUNTIME_SCORING_POLICY,
                "test_visibility_policy": ANALOGY_TRANSFER_V3_TEST_VISIBILITY_POLICY,
                "primary_task_ids": [
                    task.get("task_id") for task in analogy_transfer_results
                    if isinstance(task, dict) and task.get("task_id")
                ],
                "output_count": ANALOGY_TRANSFER_OUTPUT_COUNT,
                "max_tokens": get_task_max_tokens("AnalogyTransfer"),
                "common_mapping_bank_version": ANALOGY_COMMON_MAPPING_BANK_VERSION,
                "task_overlay_version": ANALOGY_TRANSFER_V3_TASK_OVERLAY_VERSION,
                "common_mapping_bank_coverage": get_analogy_common_mapping_bank_coverage([
                    task.get("task_id") for task in analogy_transfer_results
                    if isinstance(task, dict) and task.get("task_id")
                ]),
                "task_overlay_coverage": get_analogy_transfer_task_overlay_coverage([
                    task.get("task_id") for task in analogy_transfer_results
                    if isinstance(task, dict) and task.get("task_id")
                ]),
                "evidence_alias_coverage": get_analogy_evidence_alias_coverage([
                    task.get("task_id") for task in analogy_transfer_results
                    if isinstance(task, dict) and task.get("task_id")
                ]),
                "coverage_gate_pass": bool(analogy_transfer_gate_pass and analogy_transfer_task_scores),
                "score": round(analogy_transfer_imagination_score, 4) if analogy_transfer_imagination_score is not None else None,
                "imagination": (
                    round(analogy_transfer_imagination_score, 4)
                    if analogy_transfer_imagination_score is not None else None
                ),
                "hallucination": (
                    round(analogy_transfer_hallucination_score, 4)
                    if analogy_transfer_hallucination_score is not None else None
                ),
                "imagination_raw": (
                    round(analogy_transfer_imagination_raw, 4)
                    if analogy_transfer_imagination_raw is not None else None
                ),
                "imagination_gated": (
                    round(analogy_transfer_imagination_gated, 4)
                    if analogy_transfer_imagination_gated is not None else None
                ),
                "hallucination_raw": (
                    round(analogy_transfer_hallucination_raw, 4)
                    if analogy_transfer_hallucination_raw is not None else None
                ),
                "coverage": (
                    round(analogy_transfer_coverage, 4)
                    if analogy_transfer_coverage is not None else None
                ),
                "availability": (
                    round(analogy_transfer_availability, 4)
                    if analogy_transfer_availability is not None else None
                ),
                "primitive_means": analogy_transfer_primitive_means,
                "top3_mapping_quality": analogy_transfer_axis.get("top3_mapping_quality"),
                "elite_tail_top1": analogy_transfer_axis.get("elite_tail_top1"),
                "licensed_inference_quality": analogy_transfer_axis.get("licensed_inference_quality"),
                "abstraction_diversity_eff": analogy_transfer_axis.get("abstraction_diversity_eff"),
                "boundary_aware_valid_ratio": analogy_transfer_axis.get("boundary_aware_valid_ratio"),
                "structural_match_gmean": analogy_transfer_axis.get("structural_match_gmean"),
                "evidence_grounding": analogy_transfer_axis.get("evidence_grounding"),
                "mapping_rarity": analogy_transfer_axis.get("mapping_rarity"),
                "subtype_contributions": analogy_transfer_subtype_contributions,
                "task_scores": analogy_transfer_task_scores,
                "residualization": analogy_transfer_axis.get("residualization"),
                "formula": analogy_transfer_axis.get("formula"),
                "note": (
                    "Primary T8 component in the active experiment profile; evaluates analogy quality and false-transfer burden over closed source/target facts."
                    if "AnalogyTransfer" in PRIMARY_DUAL_AXIS_COMPONENTS else
                    "Challenge diagnostic only: AnalogyTransfer is excluded from primary rankings."
                ),
            },
            "uut_affordance_dual_axis": {
                "version": UUT_DUAL_AXIS_VERSION,
                "t1_assoc_version": T1_ASSOC_VERSION,
                "coverage_gate_pass": uut_dual_axis_gate_pass,
                "score": round(uut_imagination_score, 4) if uut_imagination_score is not None else None,
                "imagination": round(uut_imagination_score, 4) if uut_imagination_score is not None else None,
                "hallucination": round(uut_hallucination_score, 4) if uut_hallucination_score is not None else None,
                "imagination_raw": round(uut_imagination_raw, 4) if uut_imagination_raw is not None else None,
                "hallucination_raw": round(uut_hallucination_raw, 4) if uut_hallucination_raw is not None else None,
                "quality_mass_top8": round(uut_quality_mass_top8, 4) if uut_quality_mass_top8 is not None else None,
                "elite_tail_top3": round(uut_elite_tail_top3, 4) if uut_elite_tail_top3 is not None else None,
                "diversity_eff": round(uut_diversity_eff, 4) if uut_diversity_eff is not None else None,
                "valid_ratio": round(uut_valid_ratio, 4) if uut_valid_ratio is not None else None,
                "bank_coverage": round(uut_bank_coverage, 4) if uut_bank_coverage is not None else None,
                "primitive_means": uut_dual_axis_primitive_means,
                "subtype_contributions": uut_subtype_contributions,
                "task_scores": uut_dual_axis_task_scores,
            },
        },
    }

    model_report = {
        "model_id": model_name,
        "repeat_index": replicate_index,
        "scoring_schema": {
            "report_schema_version": WHITE_BOX_REPORT_SCHEMA_VERSION,
            "embedding_model": getattr(scorer, "model_name", None),
            "embedding_model_note": getattr(
                scorer,
                "model_note",
                "Embedding model and immutable revision are recorded in resources.lock.json.",
            ),
            "primary_score": "Imagination and hallucination are the primary benchmark axes. DT total, novelty, flexibility, and groundedness are supporting scorer outputs.",
            "typed_correlation_ready": typed_correlation_ready,
            "subtype_schema_version": subtype_schema_version,
            "taxonomy_version": registry_metadata.get("taxonomy_version"),
            "task_registry_version": registry_metadata.get("task_registry_version"),
            "scoring_configuration": scoring_configuration,
            "t1_assoc_version": T1_ASSOC_VERSION,
            "t1_calibration_policy": "benchmark_default",
            "macgyver_scoring_version": MACGYVER_DUAL_AXIS_VERSION,
            "macgyver_calibration_policy": MACGYVER_V3_CALIBRATION_POLICY,
            "macgyver_runtime_scoring_policy": MACGYVER_V3_RUNTIME_SCORING_POLICY,
            "cjst_scoring_version": CJST_DUAL_AXIS_VERSION,
            "cjst_calibration_policy": CJST_V3_CALIBRATION_POLICY,
            "cjst_runtime_scoring_policy": CJST_V3_RUNTIME_SCORING_POLICY,
            "hypospace_scoring_version": HYPOUSESPACE_DUAL_AXIS_VERSION,
            "hypospace_calibration_policy": HYPOUSESPACE_V3_CALIBRATION_POLICY,
            "hypospace_runtime_scoring_policy": HYPOUSESPACE_V3_RUNTIME_SCORING_POLICY,
            "gcw_scoring_version": GCW_DUAL_AXIS_VERSION,
            "gcw_calibration_policy": GCW_V3_CALIBRATION_POLICY,
            "gcw_runtime_scoring_policy": GCW_V3_RUNTIME_SCORING_POLICY,
            "neocoder_scoring_version": NEOCODER_DUAL_AXIS_VERSION,
            "neocoder_calibration_policy": NEOCODER_V3_CALIBRATION_POLICY,
            "neocoder_runtime_scoring_policy": NEOCODER_V3_RUNTIME_SCORING_POLICY,
            "neocoder_test_visibility_policy": NEOCODER_V3_TEST_VISIBILITY_POLICY,
            "analogy_transfer_scoring_version": ANALOGY_TRANSFER_VERSION,
            "analogy_transfer_calibration_policy": ANALOGY_TRANSFER_V3_CALIBRATION_POLICY,
            "analogy_transfer_runtime_scoring_policy": ANALOGY_TRANSFER_V3_RUNTIME_SCORING_POLICY,
            "analogy_transfer_test_visibility_policy": ANALOGY_TRANSFER_V3_TEST_VISIBILITY_POLICY,
            "cross_task_fact_consistency_version": CROSS_TASK_FACT_CONSISTENCY_VERSION,
            "cjst_common_consequence_bank_coverage": get_cjst_common_consequence_bank_coverage([
                task.get("task_id") for task in cjst_results
                if isinstance(task, dict) and task.get("task_id")
            ]),
            "macgyver_common_plan_bank_coverage": get_macgyver_common_plan_bank_coverage([
                task.get("task_id") for task in macgyver_results
                if isinstance(task, dict) and task.get("task_id")
            ]),
            "gcw_common_story_bank_coverage": get_gcw_common_story_bank_coverage([
                task.get("task_id") for task in gcw_results
                if isinstance(task, dict) and task.get("task_id")
            ]),
            "gcw_entity_alias_coverage": get_gcw_entity_alias_coverage([
                task.get("task_id") for task in gcw_results
                if isinstance(task, dict) and task.get("task_id")
            ]),
            "hypospace_common_hypothesis_bank_coverage": get_hypospace_common_hypothesis_bank_coverage([
                task.get("task_id") for task in hypospace_results
                if isinstance(task, dict) and task.get("task_id")
            ]),
            "hypospace_valid_match_alias_coverage": get_hypospace_valid_match_alias_coverage([
                task.get("task_id") for task in hypospace_results
                if isinstance(task, dict) and task.get("task_id")
            ]),
            "neocoder_common_solution_bank_coverage": get_neocoder_common_solution_bank_coverage([
                task.get("task_id") for task in neocoder_results
                if isinstance(task, dict) and task.get("task_id")
            ]),
            "neocoder_task_overlay_coverage": get_neocoder_task_overlay_coverage([
                task.get("task_id") for task in neocoder_results
                if isinstance(task, dict) and task.get("task_id")
            ]),
            "neocoder_technique_alias_coverage": get_neocoder_technique_alias_coverage(),
            "analogy_common_mapping_bank_coverage": get_analogy_common_mapping_bank_coverage([
                task.get("task_id") for task in analogy_transfer_results
                if isinstance(task, dict) and task.get("task_id")
            ]),
            "analogy_transfer_task_overlay_coverage": get_analogy_transfer_task_overlay_coverage([
                task.get("task_id") for task in analogy_transfer_results
                if isinstance(task, dict) and task.get("task_id")
            ]),
            "analogy_evidence_alias_coverage": get_analogy_evidence_alias_coverage([
                task.get("task_id") for task in analogy_transfer_results
                if isinstance(task, dict) and task.get("task_id")
            ]),
            "creative_novelty": "supporting scorer output: UUT/PropConj novelty = hybrid prompt-distance/common-answer rarity after white-box groundedness/property-validity penalty; DAT/CDAT are auxiliary lexical diagnostics",
            "groundedness": "audit primitive only: white-box v5p0 SWOW + word-norms2/manual templates + embedding anchors + mechanism consistency + anti-cliche signal; no longer a ranking gate",
            "uut_dual_axis": "UUT-Affordance T1- high-end rarity/support/mechanism quality mass split from unsupported/contradictory/tool-drift hallucination; no external judge API",
            "propconj_dual_axis": "PropConj T1- property-conjunction validators use hard-valid quality mass while hallucination remains unresolved/contradictory/evidence-mismatched burden; no external judge API",
            "macgyver_dual_axis": "MacGyver T2- discriminative closed-tool planning with common-plan rarity bank, feasibility hard gates, high-end quality mass, and separate boundary diagnostics; no external judge API",
            "cjst_dual_axis": "CJST T3- high-end counterfactual consequence scoring with common-consequence rarity bank, causal-chain grounding gates, and context/logic/drift hallucination burdens; no external judge API",
            "hypospace_dual_axis": "HypoUseSpace T5- evidence-constrained mechanism-hypothesis imagination with soft-valid alias matching, common-hypothesis rarity bank, high-end quality mass, and separate no-valid boundary diagnostics; no external judge API",
            "gcw_dual_axis": "GCW T4- grounded narrative-turn scoring with common-story rarity bank, payoff ledger support gates, entity-alias drift repair, and detail/context/drift/citation hallucination burdens; no external judge API",
            "neocoder_dual_axis": "NeoCoder T6- denial-state code imagination scored with public examples plus hidden/metamorphic scoring tests, denied-technique alias checks, anti-overfit gates, and logic/intent/fact hallucination burdens; no external judge API",
            "closed_world_fact_calibration": "ClosedWorldFact v1: optional closed-world hallucination calibration over relational facts, evidence chains, comparisons, sets, and unanswerable boundary cases; no imagination score and no external judge API",
            "analogy_transfer_challenge": "AnalogyTransfer T8- closed-world structural analogy discovery with hidden gold mappings, common-mapping rarity bank, licensed transfer tests, and false-transfer/fact/logic/context hallucination burdens; no external judge API",
            "auxiliary_imagination_diagnostics": "DAT/CDAT/FF are reported as auxiliary imagination diagnostics only and never define hallucination.",
        },
        "runtime_policy": {
            "reasoning_enabled": OPENROUTER_ENABLE_REASONING,
            "task_max_tokens": {key: TASK_MAX_TOKENS[key] for key in PRIMARY_DUAL_AXIS_COMPONENTS},
            "task_temperatures": {key: TASK_TEMPERATURES[key] for key in PRIMARY_DUAL_AXIS_COMPONENTS},
            "task_system_prompts": {key: TASK_SYSTEM_PROMPTS[key] for key in PRIMARY_DUAL_AXIS_COMPONENTS},
            "output_targets": {
                "creative": CREATIVE_OUTPUT_COUNT,
                "PropConj": PROP_CONJ_OUTPUT_COUNT,
                "MacGyver": MACGYVER_OUTPUT_COUNT,
                "CJST": CJST_OUTPUT_COUNT,
                "HypoUseSpace": HYPOUSESPACE_OUTPUT_COUNT,
                "GCW": GCW_BEAT_COUNT,
                "NeoCoder": NEOCODER_OUTPUT_COUNT,
                "AnalogyTransfer": ANALOGY_TRANSFER_OUTPUT_COUNT,
            },
            "minimum_valid_outputs": {
                "creative": MIN_CREATIVE_ITEMS_PER_TASK,
                "PropConj": MIN_PROP_CONJ_ITEMS_PER_TASK,
                "MacGyver": MIN_MACGYVER_PLANS_PER_TASK,
                "CJST": MIN_CJST_ITEMS_PER_TASK,
                "CJST_per_tier": MIN_CJST_ITEMS_PER_TIER,
                "HypoUseSpace": MIN_HYPOUSESPACE_ITEMS_PER_TASK,
                "GCW": MIN_GCW_BEATS_PER_TASK,
                "NeoCoder": MIN_NEOCODER_ITEMS_PER_TASK,
                "AnalogyTransfer": MIN_ANALOGY_TRANSFER_ITEMS_PER_TASK,
            },
            "sampling": {
                "repeat_index": replicate_index,
                "requested_repeats": MODEL_SAMPLE_REPEATS,
                "seed_base": SAMPLING_SEED_BASE,
            },
            "coverage_gates": {
                "creative": MIN_CREATIVE_COVERAGE,
                "creative_task_type": MIN_CREATIVE_TASKTYPE_COVERAGE,
                "MacGyver": MIN_MACGYVER_COVERAGE,
                "CJST": MIN_CJST_COVERAGE,
                "HypoUseSpace": MIN_HYPOUSESPACE_COVERAGE,
                "GCW": MIN_GCW_COVERAGE,
                "NeoCoder": MIN_NEOCODER_COVERAGE,
                "AnalogyTransfer": MIN_ANALOGY_TRANSFER_COVERAGE,
            },
            "task_family_filter": {
                **get_runtime_task_policy(),
                "prompt_counts": {task_type: len(tasks) for task_type, tasks in dataset.items()},
                "total_prompts_per_repeat": sum(len(tasks) for tasks in dataset.values()),
            },
        },
        "prompt_manifest": build_prompt_manifest(dataset),
        "overall_summary": overall_summary,
        "task_results": task_results,
        "cjst_results": cjst_summary,
        "macgyver_results": macgyver_summary,
        "hypospace_results": hypospace_summary,
        "gcw_results": gcw_summary,
        "neocoder_results": neocoder_summary,
        "analogy_transfer_results": analogy_transfer_summary,
        "data_sources": {
            "swow": "Active" if cog_baseline.swow_available else "Inactive",
            "groundedness": groundedness_scorer.get_data_source_label(),
            "common_answer_bank": {
                "static_builtin": "active",
                "reference_bank": "active" if has_common_answer_reference_bank() else "inactive",
                "dynamic_swow": (
                    "active" if getattr(getattr(groundedness_scorer, "swow", None), "available", False)
                    else "inactive"
                ),
                "dynamic_word_norms2": (
                    "active" if getattr(getattr(groundedness_scorer, "word_norms2", None), "available", False)
                    else "inactive"
                ),
            },
            "ontological_flexibility": wn_analyzer.get_source_label(),
            "macgyver": "Static closed-tool task manifest + white-box lexical/affordance rules",
            "cjst": "Static counterfactual scenario cards + anchor banks + white-box premise-lock/forbidden-foil rules",
            "hypospace": "Static closed-world hypothesis spaces + deterministic canonicalizer, soft-valid alias overlay, common-hypothesis bank, and evidence-boundary ledger",
            "gcw": "Static fact/constraint sheets + white-box TTCW proxy and closed-world claim rules",
            "neocoder": "Static Python function tasks + denied-technique manifests + AST verifier + restricted subprocess unit tests",
            "closed_world_fact": "Static closed relational/world database + evidence-chain, comparison, set, and unanswerable calibration checks",
            "analogy_transfer": "Static closed source/target analogy clusters + structural mapping, limit, evidence, and forbidden-transfer checks",
            "cross_task_fact_consistency": "Diagnostic-only consistency check over repeated fact/evidence ids across GCW, HypoUseSpace, and optional ClosedWorldFact outputs",
            "scoring_configuration": scoring_configuration,
            "openrouter_base_url": OPENROUTER_BASE_URL,
        },
        "run_validity": {
            "reports_generated": True,
            "ranking_eligible": ranking_eligible,
            "eligibility_failures": eligibility_failures,
            "primary_axis": "dual_axis",
            "experiment_profile": OPENROUTER_EXPERIMENT_PROFILE,
            "profile_task_manifest_path": PROFILE_TASK_MANIFEST_PATH or None,
            "typed_correlation_ready": typed_correlation_ready,
            "subtype_schema_version": subtype_schema_version,
            "primary_dual_axis_components": list(PRIMARY_DUAL_AXIS_COMPONENTS),
            "optional_dual_axis_components": list(OPTIONAL_DUAL_AXIS_COMPONENTS),
            "auxiliary_imagination_diagnostics": list(AUXILIARY_IMAGINATION_DIAGNOSTICS),
            "enhanced_dual_axis_diagnostics": list(ENHANCED_DUAL_AXIS_DIAGNOSTICS),
            "calibration_diagnostics": list(CALIBRATION_DIAGNOSTICS),
            "challenge_diagnostics": list(CHALLENGE_DIAGNOSTICS),
            "optional_dual_axis_issues": optional_dual_axis_issues,
            "auxiliary_diagnostic_issues": auxiliary_diagnostic_issues,
            "enhanced_diagnostic_issues": enhanced_diagnostic_issues,
            "calibration_diagnostic_issues": calibration_diagnostic_issues,
            "challenge_diagnostic_issues": challenge_diagnostic_issues,
            "coverage_thresholds": {
                "creative": MIN_CREATIVE_COVERAGE,
                "creative_task_type": MIN_CREATIVE_TASKTYPE_COVERAGE,
                "MacGyver": MIN_MACGYVER_COVERAGE,
                "CJST": MIN_CJST_COVERAGE,
                "HypoUseSpace": MIN_HYPOUSESPACE_COVERAGE,
                "GCW": MIN_GCW_COVERAGE,
                "NeoCoder": MIN_NEOCODER_COVERAGE,
                "AnalogyTransfer": MIN_ANALOGY_TRANSFER_COVERAGE,
            },
            "creative_prompt_totals": creative_task_totals,
            "creative_effective_totals": creative_task_effective_totals,
            "creative_excluded_counts": creative_task_excluded_counts,
            "creative_valid_counts": creative_task_valid_counts,
            "creative_coverages": {
                task_type: (round(value, 4) if value is not None else None)
                for task_type, value in creative_task_coverages.items()
            },
            "creative_availability": {
                task_type: (round(value, 4) if value is not None else None)
                for task_type, value in creative_task_availability.items()
            },
            "creative_total_prompts": total_creative_prompts,
            "creative_effective_prompts": total_creative_effective_prompts,
            "creative_total_coverage": round(creative_coverage, 4) if creative_coverage is not None else None,
            "creative_total_availability": round(creative_availability, 4) if creative_availability is not None else None,
            "dat_total_prompts": dat_total_prompts,
            "dat_effective_prompts": dat_effective_prompts,
            "dat_excluded_prompts": dat_excluded_count,
            "dat_scorable_prompts": dat_scorable_count,
            "dat_coverage": round(dat_coverage, 4) if dat_coverage is not None else None,
            "dat_availability": round(dat_availability, 4) if dat_availability is not None else None,
            "cdat_total_prompts": cdat_total_prompts,
            "cdat_effective_prompts": cdat_effective_prompts,
            "cdat_excluded_prompts": cdat_excluded_count,
            "cdat_scorable_prompts": cdat_scorable_count,
            "cdat_coverage": round(cdat_coverage, 4) if cdat_coverage is not None else None,
            "cdat_availability": round(cdat_availability, 4) if cdat_availability is not None else None,
            "macgyver_total_prompts": macgyver_total_prompts,
            "macgyver_effective_prompts": macgyver_effective_prompts,
            "macgyver_excluded_prompts": macgyver_excluded_count,
            "macgyver_scorable_prompts": macgyver_scorable_count,
            "macgyver_coverage": round(macgyver_coverage, 4) if macgyver_coverage is not None else None,
            "macgyver_availability": round(macgyver_availability, 4) if macgyver_availability is not None else None,
            "macgyver_boundary_total_prompts": macgyver_boundary_total_prompts,
            "macgyver_boundary_effective_prompts": macgyver_boundary_effective_prompts,
            "macgyver_boundary_excluded_prompts": macgyver_boundary_excluded_count,
            "macgyver_boundary_scorable_prompts": macgyver_boundary_scorable_count,
            "macgyver_boundary_coverage": round(macgyver_boundary_coverage, 4) if macgyver_boundary_coverage is not None else None,
            "macgyver_boundary_availability": round(macgyver_boundary_availability, 4) if macgyver_boundary_availability is not None else None,
            "cjst_total_prompts": cjst_total_prompts,
            "cjst_effective_prompts": cjst_effective_prompts,
            "cjst_excluded_prompts": cjst_excluded_count,
            "cjst_scorable_prompts": cjst_scorable_count,
            "cjst_coverage": round(cjst_coverage, 4) if cjst_coverage is not None else None,
            "cjst_availability": round(cjst_availability, 4) if cjst_availability is not None else None,
            "hypospace_total_prompts": hypospace_total_prompts,
            "hypospace_effective_prompts": hypospace_effective_prompts,
            "hypospace_excluded_prompts": hypospace_excluded_count,
            "hypospace_scorable_prompts": hypospace_scorable_count,
            "hypospace_coverage": round(hypospace_coverage, 4) if hypospace_coverage is not None else None,
            "hypospace_availability": round(hypospace_availability, 4) if hypospace_availability is not None else None,
            "hypospace_boundary_total_prompts": hypospace_boundary_total_prompts,
            "hypospace_boundary_effective_prompts": hypospace_boundary_effective_prompts,
            "hypospace_boundary_excluded_prompts": hypospace_boundary_excluded_count,
            "hypospace_boundary_scorable_prompts": hypospace_boundary_scorable_count,
            "hypospace_boundary_coverage": round(hypospace_boundary_coverage, 4) if hypospace_boundary_coverage is not None else None,
            "hypospace_boundary_availability": round(hypospace_boundary_availability, 4) if hypospace_boundary_availability is not None else None,
            "gcw_total_prompts": gcw_total_prompts,
            "gcw_effective_prompts": gcw_effective_prompts,
            "gcw_excluded_prompts": gcw_excluded_count,
            "gcw_scorable_prompts": gcw_scorable_count,
            "gcw_coverage": round(gcw_coverage, 4) if gcw_coverage is not None else None,
            "gcw_availability": round(gcw_availability, 4) if gcw_availability is not None else None,
            "neocoder_total_prompts": neocoder_total_prompts,
            "neocoder_effective_prompts": neocoder_effective_prompts,
            "neocoder_excluded_prompts": neocoder_excluded_count,
            "neocoder_scorable_prompts": neocoder_scorable_count,
            "neocoder_coverage": round(neocoder_coverage, 4) if neocoder_coverage is not None else None,
            "neocoder_availability": round(neocoder_availability, 4) if neocoder_availability is not None else None,
            "closed_world_fact_total_prompts": closed_world_fact_total_prompts,
            "closed_world_fact_effective_prompts": closed_world_fact_effective_prompts,
            "closed_world_fact_excluded_prompts": closed_world_fact_excluded_count,
            "closed_world_fact_scorable_prompts": closed_world_fact_scorable_count,
            "closed_world_fact_coverage": (
                round(closed_world_fact_coverage, 4)
                if closed_world_fact_coverage is not None else None
            ),
            "closed_world_fact_availability": (
                round(closed_world_fact_availability, 4)
                if closed_world_fact_availability is not None else None
            ),
            "analogy_transfer_total_prompts": analogy_transfer_total_prompts,
            "analogy_transfer_effective_prompts": analogy_transfer_effective_prompts,
            "analogy_transfer_excluded_prompts": analogy_transfer_excluded_count,
            "analogy_transfer_scorable_prompts": analogy_transfer_scorable_count,
            "analogy_transfer_coverage": (
                round(analogy_transfer_coverage, 4)
                if analogy_transfer_coverage is not None else None
            ),
            "analogy_transfer_availability": (
                round(analogy_transfer_availability, 4)
                if analogy_transfer_availability is not None else None
            ),
            "ff_total_prompts": ff_total_prompts,
            "ff_effective_prompts": ff_effective_prompts,
            "ff_excluded_prompts": ff_excluded_count,
            "ff_scorable_prompts": ff_scorable_count,
            "ff_coverage": round(ff_coverage, 4) if ff_coverage is not None else None,
            "ff_availability": round(ff_availability, 4) if ff_availability is not None else None,
            "axis_validity": {
                "primary_dual_axis": primary_dual_axis_gate_pass,
                "novelty": novelty_axis_gate_pass,
                "flexibility": flexibility_axis_gate_pass,
                "groundedness": groundedness_axis_gate_pass,
                "imagination": imagination_axis_gate_pass,
                "hallucination": imagination_axis_gate_pass,
                "uut_affordance_dual_axis": uut_dual_axis_gate_pass,
                "propconj_dual_axis": propconj_dual_axis_gate_pass,
                "macgyver_dual_axis": macgyver_gate_pass,
                "cjst_dual_axis": cjst_gate_pass,
                "hypospace_dual_axis": hypospace_gate_pass,
                "gcw_dual_axis": gcw_gate_pass,
                "neocoder_dual_axis": neocoder_gate_pass,
                "closed_world_fact_calibration": bool(
                    closed_world_fact_gate_pass and closed_world_fact_task_scores
                ),
                "analogy_transfer_challenge": bool(
                    analogy_transfer_gate_pass and analogy_transfer_task_scores
                ),
                "dt_total": dt_total_score is not None,
            },
            "valid_creative_tasks": task_count,
            "invalid_run_counts": invalid_run_counts,
            "non_model_skip_counts": non_model_skip_counts,
            "non_model_skip_reasons": dict(non_model_skip_reasons),
            "raw_fluency_total": total_fluency_raw_all,
            "deduped_fluency_total": total_fluency_deduped_all,
            "zero_originality_total": total_zero_orig_count,
        },
    }

    print("\n" + "=" * 60)
    print(f"  FINAL MODEL REPORT: {model_name} [repeat {repeat_label}]")
    print("=" * 60)
    print(f"  Primary dual-axis ranking eligible: {'YES' if ranking_eligible else 'NO'}")
    if eligibility_failures:
        print(f"  Eligibility issues: {', '.join(eligibility_failures)}")
    if optional_dual_axis_issues:
        print(f"  Optional dual-axis issues: {', '.join(optional_dual_axis_issues)}")
    if auxiliary_diagnostic_issues:
        print(f"  Auxiliary diagnostic issues: {', '.join(auxiliary_diagnostic_issues)}")
    if enhanced_diagnostic_issues:
        print(f"  Enhanced diagnostic issues: {', '.join(enhanced_diagnostic_issues)}")
    if calibration_diagnostic_issues:
        print(f"  Calibration diagnostic issues: {', '.join(calibration_diagnostic_issues)}")
    print("\n  [1] Legacy DT Total (audit)")
    if dt_total_score is not None:
        print(f"      Score:   {dt_total_score:.4f}")
        print(f"      Formula: {DT_TOTAL_NOVELTY_WEIGHT:.2f}*Novelty + {DT_TOTAL_FLEXIBILITY_WEIGHT:.2f}*Flexibility")
    else:
        print("      N/A")

    print(f"\n  [2] Novelty Axis ({num_categories_scored}/{len(novelty_required_components)} components)")
    for label in list(CREATIVE_TASK_TYPES) + ["DAT", "CDAT"]:
        if label in category_originality_scores:
            value = category_originality_scores[label]
            weight = effective_novelty_weights.get(label)
            weight_text = f"  (w={weight:.3f})" if weight is not None else ""
            extra = ""
            if label == "DAT" and dat_mean is not None:
                extra = f"  (raw: {dat_mean:.2f})"
            elif label == "CDAT" and cdat_score is not None:
                extra = f"  (continuous score: {cdat_score:.2f})"
            print(f"      {label:12s}  {value:.4f}{weight_text}{extra}")
        else:
            print(f"      {label:12s}  N/A")
    if novelty_formula:
        print(f"      Formula: {novelty_formula}")
    print(
        f"      Coverage: UUT={format_percent_or_na(creative_task_coverages.get('UUT'))}, "
        f"PropConj={format_percent_or_na(creative_task_coverages.get('PropConj'))}, "
        f"DAT={format_percent_or_na(dat_coverage)}, CDAT={format_percent_or_na(cdat_coverage)}"
    )
    print(
        f"      Availability: UUT={format_percent_or_na(creative_task_availability.get('UUT'))}, "
        f"PropConj={format_percent_or_na(creative_task_availability.get('PropConj'))}, "
        f"DAT={format_percent_or_na(dat_availability)}, CDAT={format_percent_or_na(cdat_availability)}"
    )
    if novelty_axis_score is not None:
        print(f"      Axis Score: {novelty_axis_score:.4f}")
    else:
        print("      Axis Score: N/A (coverage gate not met)")

    print("\n  [3] Flexibility Axis")
    if flexibility_axis_score is not None:
        print(f"      Embedding:    {model_embedding_flexibility_score:.4f} (w={w_emb:.2f})")
        print(f"      Ontological:  {model_ontological_flexibility_score:.4f} (w={w_ont:.2f})")
        print(f"      Emb Pairwise: {model_avg_emb_pairwise_distance:.4f}")
        print(f"      Emb Entropy:  {model_avg_emb_cluster_entropy:.4f}")
        print(f"      Formula:      {flex_formula_str}")
        print(f"      Axis Score:   {flexibility_axis_score:.4f}")
    else:
        print("      N/A (creative coverage gate not met)")

    print("\n  [4] Groundedness audit primitive")
    if groundedness_axis_score is not None:
        print(f"      Mean score:               {groundedness_axis_score:.4f}")
        if groundedness_axis_score_novel is not None:
            print(f"      Novel-only mean:          {groundedness_axis_score_novel:.4f}")
        if confidence_weighted_groundedness is not None:
            print(f"      Confidence-weighted mean: {confidence_weighted_groundedness:.4f}")
        print(f"      Mean penalty:             {mean_penalty:.4f}" if mean_penalty is not None else "      Mean penalty:             N/A")
        print(f"      Penalty rate:             {penalty_rate:.4f}" if penalty_rate is not None else "      Penalty rate:             N/A")
        print(f"      Low-ground rate:          {low_groundedness_rate:.4f}" if low_groundedness_rate is not None else "      Low-ground rate:          N/A")
        print(f"      Mean confidence:          {mean_confidence:.4f}" if mean_confidence is not None else "      Mean confidence:          N/A")
    else:
        print("      N/A (creative coverage gate not met)")

    print("\n  [4b] UUT-Affordance Dual Axis")
    if uut_dual_axis_gate_pass:
        print(f"      Imagination:   {uut_imagination_score:.4f} (raw={uut_imagination_raw:.4f})")
        print(f"      Hallucination: {uut_hallucination_score:.4f} (raw={uut_hallucination_raw:.4f}; lower is better)")
        print(f"      Tasks scored:  {len(uut_dual_axis_task_scores)}")
    else:
        print("      N/A (UUT coverage gate not met)")

    print("\n  [4c] PropConj Dual Axis")
    if propconj_dual_axis_gate_pass:
        print(f"      Imagination:   {propconj_imagination_score:.4f} (raw={propconj_imagination_raw:.4f})")
        print(f"      Hallucination: {propconj_hallucination_score:.4f} (raw={propconj_hallucination_raw:.4f}; lower is better)")
        print(f"      Tasks scored:  {len(propconj_dual_axis_task_scores)}")
    else:
        print("      N/A (PropConj coverage gate not met)")

    print("\n  [4d] MacGyver Dual Axis")
    if macgyver_gate_pass and macgyver_imagination_score is not None:
        print(f"      Imagination:   {macgyver_imagination_score:.4f} (raw={macgyver_imagination_raw:.4f})")
        print(f"      Hallucination: {macgyver_hallucination_score:.4f} (raw={macgyver_hallucination_raw:.4f}; lower is better)")
        print(f"      Tasks scored:  {len(macgyver_dual_axis_task_scores)}")
    else:
        print("      N/A (MacGyver coverage gate not met)")

    print("\n  [4e] CJST Dual Axis")
    if cjst_gate_pass and cjst_imagination_score is not None:
        print(f"      Imagination:   {cjst_imagination_score:.4f} (raw={cjst_imagination_raw:.4f})")
        print(f"      Hallucination: {cjst_hallucination_score:.4f} (raw={cjst_hallucination_raw:.4f}; lower is better)")
        print(f"      Tasks scored:  {len(cjst_dual_axis_task_scores)}")
        print(f"      Coverage:      {format_percent_or_na(cjst_coverage)}; availability={format_percent_or_na(cjst_availability)}")
    else:
        print("      N/A (CJST coverage gate not met)")

    print("\n  [4f] GCW Dual Axis")
    if gcw_gate_pass and gcw_imagination_score is not None:
        print(f"      Imagination:   {gcw_imagination_score:.4f} (raw={gcw_imagination_raw:.4f})")
        print(f"      Hallucination: {gcw_hallucination_score:.4f} (raw={gcw_hallucination_raw:.4f}; lower is better)")
        print(f"      Tasks scored:  {len(gcw_dual_axis_task_scores)}")
        print(f"      Coverage:      {format_percent_or_na(gcw_coverage)}; availability={format_percent_or_na(gcw_availability)}")
    else:
        print("      N/A (GCW coverage gate not met)")

    print("\n  [4g] HypoUseSpace Dual Axis")
    if hypospace_gate_pass and hypospace_imagination_score is not None:
        print(f"      Imagination:   {hypospace_imagination_score:.4f} (raw={hypospace_imagination_raw:.4f})")
        print(f"      Hallucination: {hypospace_hallucination_score:.4f} (raw={hypospace_hallucination_raw:.4f}; lower is better)")
        print(f"      Tasks scored:  {len(hypospace_dual_axis_task_scores)}")
        print(f"      Coverage:      {format_percent_or_na(hypospace_coverage)}; availability={format_percent_or_na(hypospace_availability)}")
    else:
        print("      N/A (HypoUseSpace coverage gate not met)")

    print("\n  [4h] NeoCoder Enhanced Dual Axis")
    if neocoder_gate_pass and neocoder_imagination_score is not None:
        print(f"      Imagination:   {neocoder_imagination_score:.4f} (raw={neocoder_imagination_raw:.4f}, gated={neocoder_imagination_gated:.4f})")
        print(f"      Hallucination: {neocoder_hallucination_score:.4f} (raw={neocoder_hallucination_raw:.4f}; lower is better)")
        print(f"      Tasks scored:  {len(neocoder_dual_axis_task_scores)}")
        print(f"      Coverage:      {format_percent_or_na(neocoder_coverage)}; availability={format_percent_or_na(neocoder_availability)}")
    else:
        print("      N/A (NeoCoder enhanced diagnostics not run or coverage gate not met)")

    print("\n  [4i] ClosedWorldFact Calibration")
    if closed_world_fact_gate_pass and closed_world_fact_score is not None:
        print(f"      Score:         {closed_world_fact_score:.4f}")
        print(f"      Hallucination: {closed_world_fact_hallucination:.4f} (lower is better)")
        print(f"      Tasks scored:  {len(closed_world_fact_task_scores)}")
        print(f"      Coverage:      {format_percent_or_na(closed_world_fact_coverage)}; availability={format_percent_or_na(closed_world_fact_availability)}")
    else:
        print("      N/A (ClosedWorldFact calibration not run or coverage gate not met)")

    print("\n  [4j] AnalogyTransfer Challenge")
    if analogy_transfer_gate_pass and analogy_transfer_imagination_score is not None:
        print(f"      Imagination:   {analogy_transfer_imagination_score:.4f} (raw={analogy_transfer_imagination_raw:.4f}, gated={analogy_transfer_imagination_gated:.4f})")
        print(f"      Hallucination: {analogy_transfer_hallucination_score:.4f} (raw={analogy_transfer_hallucination_raw:.4f}; lower is better)")
        print(f"      Tasks scored:  {len(analogy_transfer_task_scores)}")
        print(f"      Coverage:      {format_percent_or_na(analogy_transfer_coverage)}; availability={format_percent_or_na(analogy_transfer_availability)}")
    else:
        print("      N/A (AnalogyTransfer challenge not run or coverage gate not met)")

    if dat_cdat_ff.get("ff") and dat_cdat_ff["ff"]["mean_score"] is not None:
        ff_summary = dat_cdat_ff["ff"]
        print("\n  [5] Forward Flow (diagnostic)")
        print(f"      Trials: {ff_summary['trials']}")
        print(f"      Per-seed: {ff_summary['scores']}")
        print(f"      Mean FF:  {ff_summary['mean_score']:.4f}")
        if ff_summary.get("mean_trajectory_slope") is not None:
            slope = ff_summary["mean_trajectory_slope"]
            trend = "exploration ↑" if slope > 0.005 else "rumination ↓" if slope < -0.005 else "stable →"
            print(f"      Mean trajectory slope: {slope:+.6f} ({trend})")
        print(
            f"      LSA literature reference: {ff_summary['literature_reference_mean']:.2f} "
            f"(SD={ff_summary['literature_reference_sd']:.2f})"
        )
        print(f"      Note: {ff_summary['comparison_note']}")

    print("\n  Data Sources:")
    print(f"    SWOW baseline: {'Active' if cog_baseline.swow_available else 'Inactive'}")
    print(f"    Groundedness: {groundedness_scorer.get_data_source_label()}")
    print(f"    Ontological flexibility: {wn_analyzer.get_source_label()}")
    print(f"    OpenRouter resolved models: {sorted(resolved_models_seen)}")
    print(f"    Runtime: {runtime_seconds:.2f}s")
    print("=" * 60)

    return model_report

def aggregate_model_assessment_reports(model_name, repeat_reports, model_catalog_entry=None):
    if not repeat_reports:
        raise RuntimeError(f"No repeat reports were produced for model {model_name}")

    min_required = minimum_eligible_repeats(MODEL_SAMPLE_REPEATS)
    repeat_summaries = [summarize_repeat_report(report) for report in repeat_reports]
    dt_reports = [
        report for report in repeat_reports
        if get_nested_value(report, "overall_summary", "axes", "dt_total", "score") is not None
    ]
    novelty_reports = [
        report for report in repeat_reports
        if get_nested_value(report, "overall_summary", "axes", "novelty", "score") is not None
    ]
    flexibility_reports = [
        report for report in repeat_reports
        if get_nested_value(report, "overall_summary", "axes", "flexibility", "score") is not None
    ]
    groundedness_reports = [
        report for report in repeat_reports
        if get_nested_value(report, "overall_summary", "axes", "groundedness", "score") is not None
    ]
    imagination_reports = [
        report for report in repeat_reports
        if get_nested_value(report, "overall_summary", "axes", "imagination", "score") is not None
    ]
    hallucination_reports = [
        report for report in repeat_reports
        if get_nested_value(report, "overall_summary", "axes", "hallucination", "score") is not None
    ]

    exemplar_report = dt_reports[0] if dt_reports else novelty_reports[0] if novelty_reports else repeat_reports[0]
    final_report = copy.deepcopy(exemplar_report)
    final_report.pop("repeat_index", None)
    final_report["resolved_models_seen"] = sorted({
        resolved_model
        for report in repeat_reports
        for resolved_model in report.get("resolved_models_seen", [])
    })

    dt_stats = summarize_report_metric(dt_reports, "overall_summary", "axes", "dt_total", "score")
    novelty_stats = summarize_report_metric(novelty_reports, "overall_summary", "axes", "novelty", "score")
    flexibility_stats = summarize_report_metric(flexibility_reports, "overall_summary", "axes", "flexibility", "score")
    groundedness_stats = summarize_report_metric(groundedness_reports, "overall_summary", "axes", "groundedness", "score")
    imagination_stats = summarize_report_metric(imagination_reports, "overall_summary", "axes", "imagination", "score")
    hallucination_stats = summarize_report_metric(hallucination_reports, "overall_summary", "axes", "hallucination", "score")
    uut_dual_stats = summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "score")
    propconj_dual_stats = summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "score")
    macgyver_dual_stats = summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "score")
    cjst_dual_stats = summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "score")
    hypospace_dual_stats = summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "score")
    gcw_dual_stats = summarize_report_metric(repeat_reports, "overall_summary", "axes", "gcw_dual_axis", "score")
    neocoder_dual_stats = summarize_report_metric(repeat_reports, "overall_summary", "axes", "neocoder_dual_axis", "score")
    closed_world_fact_stats = summarize_report_metric(
        repeat_reports,
        "overall_summary", "axes", "closed_world_fact_calibration", "score",
    )
    analogy_transfer_stats = summarize_report_metric(
        repeat_reports,
        "overall_summary", "axes", "analogy_transfer_challenge", "score",
    )
    novelty_component_scores, novelty_component_score_stats = aggregate_nested_numeric_dicts(
        novelty_reports,
        "overall_summary", "axes", "novelty", "component_scores",
    )
    novelty_component_scores_raw, novelty_component_score_raw_stats = aggregate_nested_numeric_dicts(
        novelty_reports,
        "overall_summary", "axes", "novelty", "component_scores_raw",
    )
    novelty_component_coverage, novelty_component_coverage_stats = aggregate_nested_numeric_dicts(
        novelty_reports,
        "overall_summary", "axes", "novelty", "component_coverage",
    )
    novelty_component_availability, novelty_component_availability_stats = aggregate_nested_numeric_dicts(
        novelty_reports,
        "overall_summary", "axes", "novelty", "component_availability",
    )
    exemplar_novelty_axis = exemplar_report.get("overall_summary", {}).get("axes", {}).get("novelty", {})
    novelty_component_weights = exemplar_novelty_axis.get("component_weights")
    if not novelty_component_weights:
        novelty_component_weights = {
            key: round(value, 6)
            for key, value in get_effective_novelty_component_weights(novelty_component_scores.keys()).items()
        }
    novelty_base_component_weights = exemplar_novelty_axis.get("base_component_weights")
    if not novelty_base_component_weights:
        novelty_base_component_weights = {
            key: round(NOVELTY_COMPONENT_BASE_WEIGHTS[key], 6)
            for key in NOVELTY_COMPONENT_ORDER
            if key in novelty_component_weights and key in NOVELTY_COMPONENT_BASE_WEIGHTS
        }
    novelty_formula = exemplar_novelty_axis.get("formula")
    if not novelty_formula:
        novelty_formula = format_novelty_component_formula(novelty_component_weights)

    flex_axis = final_report.setdefault("overall_summary", {}).setdefault("axes", {}).setdefault("flexibility", {})
    novelty_axis = final_report["overall_summary"]["axes"].setdefault("novelty", {})
    groundedness_axis = final_report["overall_summary"]["axes"].setdefault("groundedness", {})
    imagination_axis = final_report["overall_summary"]["axes"].setdefault("imagination", {})
    hallucination_axis = final_report["overall_summary"]["axes"].setdefault("hallucination", {})
    subtype_scores_axis = final_report["overall_summary"]["axes"].setdefault("subtype_scores", {})
    dual_axis = final_report["overall_summary"]["axes"].setdefault("dual_axis", {})
    optional_extended_dual_axis = final_report["overall_summary"]["axes"].setdefault("dual_axis_optional_extended", {})
    uut_dual_axis = final_report["overall_summary"]["axes"].setdefault("uut_affordance_dual_axis", {})
    propconj_dual_axis = final_report["overall_summary"]["axes"].setdefault("propconj_dual_axis", {})
    macgyver_dual_axis = final_report["overall_summary"]["axes"].setdefault("macgyver_dual_axis", {})
    cjst_dual_axis = final_report["overall_summary"]["axes"].setdefault("cjst_dual_axis", {})
    hypospace_dual_axis = final_report["overall_summary"]["axes"].setdefault("hypospace_dual_axis", {})
    gcw_dual_axis = final_report["overall_summary"]["axes"].setdefault("gcw_dual_axis", {})
    neocoder_dual_axis = final_report["overall_summary"]["axes"].setdefault("neocoder_dual_axis", {})
    closed_world_fact_calibration = final_report["overall_summary"]["axes"].setdefault("closed_world_fact_calibration", {})
    analogy_transfer_challenge = final_report["overall_summary"]["axes"].setdefault("analogy_transfer_challenge", {})
    cross_task_fact_consistency_axis = final_report["overall_summary"]["axes"].setdefault("cross_task_fact_consistency", {})
    dt_axis = final_report["overall_summary"]["axes"].setdefault("dt_total", {})

    dual_component_repeat_counts = {
        "UUT": uut_dual_stats["n"],
        "PropConj": propconj_dual_stats["n"],
        "MacGyver": macgyver_dual_stats["n"],
        "CJST": cjst_dual_stats["n"],
        "HypoUseSpace": hypospace_dual_stats["n"],
        "GCW": gcw_dual_stats["n"],
        "NeoCoder": neocoder_dual_stats["n"],
        "AnalogyTransfer": analogy_transfer_stats["n"],
    }
    primary_dual_repeat_counts = {
        component: dual_component_repeat_counts.get(component, 0)
        for component in PRIMARY_DUAL_AXIS_COMPONENTS
    }
    final_primary_dual_gate_pass = all(
        primary_dual_repeat_counts.get(component, 0) >= min_required
        for component in PRIMARY_DUAL_AXIS_COMPONENTS
    )
    final_ranking_eligible = (
        final_primary_dual_gate_pass and
        imagination_stats["n"] >= min_required and
        hallucination_stats["n"] >= min_required and
        imagination_stats["mean"] is not None and
        hallucination_stats["mean"] is not None
    )
    subtype_scores_aggregate = aggregate_repeat_subtype_scores(repeat_reports)

    dt_axis.update({
        "score": dt_stats["mean"],
        "role": "supporting",
        "coverage_gate_pass": dt_stats["n"] >= min_required and dt_stats["mean"] is not None,
        "replicate_stats": dt_stats,
        "eligible_repeat_count": dt_stats["n"],
        "minimum_required_repeats": min_required,
        "formula": f"{DT_TOTAL_NOVELTY_WEIGHT:.2f}*Novelty + {DT_TOTAL_FLEXIBILITY_WEIGHT:.2f}*Flexibility",
    })

    novelty_axis.update({
        "role": "supporting",
        "score": novelty_stats["mean"],
        "coverage_gate_pass": novelty_stats["n"] >= min_required and novelty_stats["mean"] is not None,
        "replicate_stats": novelty_stats,
        "eligible_repeat_count": novelty_stats["n"],
        "minimum_required_repeats": min_required,
        "num_components": len(novelty_component_scores),
        "component_scores": novelty_component_scores,
        "component_score_stats": novelty_component_score_stats,
        "component_scores_raw": novelty_component_scores_raw,
        "component_score_raw_stats": novelty_component_score_raw_stats,
        "component_weights": novelty_component_weights,
        "base_component_weights": novelty_base_component_weights,
        "formula": novelty_formula,
        "creative_hybrid_formula": {
            task_name: get_common_answer_bank_hybrid_formula(task_name)
            for task_name in CREATIVE_TASK_TYPES
        },
        "weighting_note": "Supporting scorer output: UUT/PropConj are grounded creative tasks; DAT/CDAT are auxiliary lexical diagnostics.",
        "component_coverage": novelty_component_coverage,
        "component_coverage_stats": novelty_component_coverage_stats,
        "component_availability": novelty_component_availability,
        "component_availability_stats": novelty_component_availability_stats,
        "scale": "Supporting scorer output: UUT/PropConj = hybrid raw novelty after white-box groundedness/property-validity penalty; DAT/CDAT = normalized auxiliary lexical scores; FF excluded.",
    })

    flex_axis.update({
        "role": "supporting",
        "score": flexibility_stats["mean"],
        "coverage_gate_pass": flexibility_stats["n"] >= min_required and flexibility_stats["mean"] is not None,
        "replicate_stats": flexibility_stats,
        "eligible_repeat_count": flexibility_stats["n"],
        "minimum_required_repeats": min_required,
        "embedding_composite": summarize_report_metric(flexibility_reports, "overall_summary", "axes", "flexibility", "embedding_composite")["mean"],
        "ontological_composite": summarize_report_metric(flexibility_reports, "overall_summary", "axes", "flexibility", "ontological_composite")["mean"],
        "embedding_pairwise_distance": summarize_report_metric(flexibility_reports, "overall_summary", "axes", "flexibility", "embedding_pairwise_distance")["mean"],
        "embedding_adjacent_distance": summarize_report_metric(flexibility_reports, "overall_summary", "axes", "flexibility", "embedding_adjacent_distance")["mean"],
        "embedding_cluster_entropy": summarize_report_metric(flexibility_reports, "overall_summary", "axes", "flexibility", "embedding_cluster_entropy")["mean"],
    })

    grounded_task_scores, grounded_task_score_stats = aggregate_nested_numeric_dicts(
        groundedness_reports,
        "overall_summary", "axes", "groundedness", "task_type_scores",
    )
    grounded_task_scores_novel, grounded_task_score_novel_stats = aggregate_nested_numeric_dicts(
        groundedness_reports,
        "overall_summary", "axes", "groundedness", "task_type_scores_novel_only",
    )
    grounded_task_scores_raw, grounded_task_score_raw_stats = aggregate_nested_numeric_dicts(
        groundedness_reports,
        "overall_summary", "axes", "groundedness", "task_type_scores_raw",
    )
    grounded_task_scores_novel_raw, grounded_task_score_novel_raw_stats = aggregate_nested_numeric_dicts(
        groundedness_reports,
        "overall_summary", "axes", "groundedness", "task_type_scores_novel_only_raw",
    )
    groundedness_axis.update({
        "role": "supporting",
        "version": WHITE_BOX_GROUNDEDNESS_VERSION,
        "score": groundedness_stats["mean"],
        "coverage_gate_pass": groundedness_stats["n"] >= min_required and groundedness_stats["mean"] is not None,
        "replicate_stats": groundedness_stats,
        "eligible_repeat_count": groundedness_stats["n"],
        "minimum_required_repeats": min_required,
        "score_novel_only": summarize_report_metric(groundedness_reports, "overall_summary", "axes", "groundedness", "score_novel_only")["mean"],
        "confidence_weighted_mean": summarize_report_metric(groundedness_reports, "overall_summary", "axes", "groundedness", "confidence_weighted_mean")["mean"],
        "confidence_weighted_mean_novel_only": summarize_report_metric(groundedness_reports, "overall_summary", "axes", "groundedness", "confidence_weighted_mean_novel_only")["mean"],
        "mean_penalty": summarize_report_metric(groundedness_reports, "overall_summary", "axes", "groundedness", "mean_penalty")["mean"],
        "penalty_rate": summarize_report_metric(groundedness_reports, "overall_summary", "axes", "groundedness", "penalty_rate")["mean"],
        "low_groundedness_rate": summarize_report_metric(groundedness_reports, "overall_summary", "axes", "groundedness", "low_groundedness_rate")["mean"],
        "low_groundedness_rate_novel_only": summarize_report_metric(groundedness_reports, "overall_summary", "axes", "groundedness", "low_groundedness_rate_novel_only")["mean"],
        "mean_confidence": summarize_report_metric(groundedness_reports, "overall_summary", "axes", "groundedness", "mean_confidence")["mean"],
        "scored_coverage": summarize_report_metric(groundedness_reports, "overall_summary", "axes", "groundedness", "scored_coverage")["mean"],
        "groundedness_scored_ideas": summarize_report_metric(groundedness_reports, "overall_summary", "axes", "groundedness", "groundedness_scored_ideas")["mean"],
        "task_type_scores": grounded_task_scores,
        "task_type_score_stats": grounded_task_score_stats,
        "task_type_scores_novel_only": grounded_task_scores_novel,
        "task_type_score_novel_stats": grounded_task_score_novel_stats,
        "task_type_scores_raw": grounded_task_scores_raw,
        "task_type_score_raw_stats": grounded_task_score_raw_stats,
        "task_type_scores_novel_only_raw": grounded_task_scores_novel_raw,
        "task_type_score_novel_raw_stats": grounded_task_score_novel_raw_stats,
    })

    imagination_task_scores, imagination_task_score_stats = aggregate_nested_numeric_dicts(
        imagination_reports,
        "overall_summary", "axes", "imagination", "task_type_scores",
    )
    imagination_task_scores_raw, imagination_task_score_raw_stats = aggregate_nested_numeric_dicts(
        imagination_reports,
        "overall_summary", "axes", "imagination", "task_type_scores_raw",
    )
    optional_imagination_task_scores, optional_imagination_task_score_stats = aggregate_nested_numeric_dicts(
        imagination_reports,
        "overall_summary", "axes", "imagination", "optional_task_type_scores",
    )
    optional_imagination_task_scores_raw, optional_imagination_task_score_raw_stats = aggregate_nested_numeric_dicts(
        imagination_reports,
        "overall_summary", "axes", "imagination", "optional_task_type_scores_raw",
    )
    hallucination_task_scores, hallucination_task_score_stats = aggregate_nested_numeric_dicts(
        hallucination_reports,
        "overall_summary", "axes", "hallucination", "task_type_scores",
    )
    hallucination_task_scores_raw, hallucination_task_score_raw_stats = aggregate_nested_numeric_dicts(
        hallucination_reports,
        "overall_summary", "axes", "hallucination", "task_type_scores_raw",
    )
    optional_hallucination_task_scores, optional_hallucination_task_score_stats = aggregate_nested_numeric_dicts(
        hallucination_reports,
        "overall_summary", "axes", "hallucination", "optional_task_type_scores",
    )
    optional_hallucination_task_scores_raw, optional_hallucination_task_score_raw_stats = aggregate_nested_numeric_dicts(
        hallucination_reports,
        "overall_summary", "axes", "hallucination", "optional_task_type_scores_raw",
    )
    uut_primitive_means, uut_primitive_mean_stats = aggregate_nested_numeric_dicts(
        imagination_reports,
        "overall_summary", "axes", "imagination", "primitive_means",
    )
    uut_axis_primitive_means, uut_axis_primitive_mean_stats = aggregate_nested_numeric_dicts(
        repeat_reports,
        "overall_summary", "axes", "uut_affordance_dual_axis", "primitive_means",
    )
    exemplar_imagination_axis = (
        imagination_reports[0].get("overall_summary", {}).get("axes", {}).get("imagination", {})
        if imagination_reports else {}
    )
    exemplar_hallucination_axis = (
        hallucination_reports[0].get("overall_summary", {}).get("axes", {}).get("hallucination", {})
        if hallucination_reports else {}
    )
    imagination_axis.update({
        "role": "primary",
        "version": exemplar_imagination_axis.get("version") or DUAL_AXIS_REPORT_VERSION,
        "score": imagination_stats["mean"],
        "raw_score": summarize_report_metric(imagination_reports, "overall_summary", "axes", "imagination", "raw_score")["mean"],
        "coverage_gate_pass": imagination_stats["n"] >= min_required and imagination_stats["mean"] is not None,
        "replicate_stats": imagination_stats,
        "eligible_repeat_count": imagination_stats["n"],
        "minimum_required_repeats": min_required,
        "formula": exemplar_imagination_axis.get("formula") or "PropConj v1: residualized property-valid novelty",
        "residualization": exemplar_imagination_axis.get("residualization") or {
            "beta_IH": PROPCONJ_DUAL_AXIS_BETA_IH,
            "beta_HI": PROPCONJ_DUAL_AXIS_BETA_HI,
            "source": "task_default",
        },
        "component_weights": exemplar_imagination_axis.get("component_weights"),
        "base_component_weights": exemplar_imagination_axis.get("base_component_weights"),
        "component_gate_pass": exemplar_imagination_axis.get("component_gate_pass"),
        "aggregation_policy": exemplar_imagination_axis.get("aggregation_policy"),
        "task_type_scores": imagination_task_scores,
        "task_type_score_stats": imagination_task_score_stats,
        "task_type_scores_raw": imagination_task_scores_raw,
        "task_type_score_raw_stats": imagination_task_score_raw_stats,
        "optional_task_type_scores": optional_imagination_task_scores,
        "optional_task_type_score_stats": optional_imagination_task_score_stats,
        "optional_task_type_scores_raw": optional_imagination_task_scores_raw,
        "optional_task_type_score_raw_stats": optional_imagination_task_score_raw_stats,
        "primitive_means": uut_primitive_means,
        "primitive_mean_stats": uut_primitive_mean_stats,
    })
    hallucination_axis.update({
        "role": "primary",
        "version": exemplar_hallucination_axis.get("version") or DUAL_AXIS_REPORT_VERSION,
        "score": hallucination_stats["mean"],
        "raw_score": summarize_report_metric(hallucination_reports, "overall_summary", "axes", "hallucination", "raw_score")["mean"],
        "coverage_gate_pass": hallucination_stats["n"] >= min_required and hallucination_stats["mean"] is not None,
        "replicate_stats": hallucination_stats,
        "eligible_repeat_count": hallucination_stats["n"],
        "minimum_required_repeats": min_required,
        "direction": "lower_is_better",
        "formula": exemplar_hallucination_axis.get("formula") or "PropConj v1: residualized unsupported/contradictory/evidence-mismatched rate",
        "residualization": exemplar_hallucination_axis.get("residualization") or {
            "beta_IH": PROPCONJ_DUAL_AXIS_BETA_IH,
            "beta_HI": PROPCONJ_DUAL_AXIS_BETA_HI,
            "source": "task_default",
        },
        "component_weights": exemplar_hallucination_axis.get("component_weights"),
        "base_component_weights": exemplar_hallucination_axis.get("base_component_weights"),
        "component_gate_pass": exemplar_hallucination_axis.get("component_gate_pass"),
        "aggregation_policy": exemplar_hallucination_axis.get("aggregation_policy"),
        "task_type_scores": hallucination_task_scores,
        "task_type_score_stats": hallucination_task_score_stats,
        "task_type_scores_raw": hallucination_task_scores_raw,
        "task_type_score_raw_stats": hallucination_task_score_raw_stats,
        "optional_task_type_scores": optional_hallucination_task_scores,
        "optional_task_type_score_stats": optional_hallucination_task_score_stats,
        "optional_task_type_scores_raw": optional_hallucination_task_scores_raw,
        "optional_task_type_score_raw_stats": optional_hallucination_task_score_raw_stats,
        "primitive_means": uut_primitive_means,
        "primitive_mean_stats": uut_primitive_mean_stats,
    })
    subtype_scores_axis.clear()
    subtype_scores_axis.update(subtype_scores_aggregate)
    final_report["overall_summary"]["axes"]["atom_signals"] = subtype_scores_axis.get("atom_signals", {})
    registry_metadata = get_v2_registry_metadata()
    subtype_schema_version = (
        subtype_scores_axis.get("version")
        if isinstance(subtype_scores_axis, dict) else None
    )
    typed_correlation_ready = subtype_scores_ready_for_correlation(subtype_scores_axis)
    scoring_configuration = {
        "policy": "deterministic_output_only_scoring",
        "t1_assoc_version": T1_ASSOC_VERSION,
        "t1_calibration_policy": "benchmark_default",
        "t1_runtime_scoring_policy": "fixed output-only parameters",
        "macgyver_scoring_version": MACGYVER_DUAL_AXIS_VERSION,
        "macgyver_calibration_policy": MACGYVER_V3_CALIBRATION_POLICY,
        "macgyver_runtime_scoring_policy": MACGYVER_V3_RUNTIME_SCORING_POLICY,
        "cjst_scoring_version": CJST_DUAL_AXIS_VERSION,
        "cjst_calibration_policy": CJST_V3_CALIBRATION_POLICY,
        "cjst_runtime_scoring_policy": CJST_V3_RUNTIME_SCORING_POLICY,
        "disclosure": (
            "Fixed scoring parameters are applied directly to model outputs."
        ),
        "judge_policy": "no_external_llm_judge_in_primary_scoring",
    }
    final_report.setdefault("scoring_schema", {}).update({
        "typed_correlation_ready": typed_correlation_ready,
        "subtype_schema_version": subtype_schema_version,
        "taxonomy_version": registry_metadata.get("taxonomy_version"),
        "task_registry_version": registry_metadata.get("task_registry_version"),
        "scoring_configuration": scoring_configuration,
        "t1_assoc_version": T1_ASSOC_VERSION,
        "t1_calibration_policy": "benchmark_default",
        "macgyver_scoring_version": MACGYVER_DUAL_AXIS_VERSION,
        "macgyver_calibration_policy": MACGYVER_V3_CALIBRATION_POLICY,
        "macgyver_runtime_scoring_policy": MACGYVER_V3_RUNTIME_SCORING_POLICY,
        "cjst_scoring_version": CJST_DUAL_AXIS_VERSION,
        "cjst_calibration_policy": CJST_V3_CALIBRATION_POLICY,
        "cjst_runtime_scoring_policy": CJST_V3_RUNTIME_SCORING_POLICY,
        "hypospace_scoring_version": HYPOUSESPACE_DUAL_AXIS_VERSION,
        "hypospace_calibration_policy": HYPOUSESPACE_V3_CALIBRATION_POLICY,
        "hypospace_runtime_scoring_policy": HYPOUSESPACE_V3_RUNTIME_SCORING_POLICY,
        "cross_task_fact_consistency_version": CROSS_TASK_FACT_CONSISTENCY_VERSION,
    })
    dual_axis_task_scores = {
        "imagination": imagination_task_scores,
        "hallucination": hallucination_task_scores,
        "imagination_raw": imagination_task_scores_raw,
        "hallucination_raw": hallucination_task_scores_raw,
        "optional_imagination": optional_imagination_task_scores,
        "optional_hallucination": optional_hallucination_task_scores,
        "optional_imagination_raw": optional_imagination_task_scores_raw,
        "optional_hallucination_raw": optional_hallucination_task_scores_raw,
    }
    dual_axis.update({
        "role": "primary",
        "version": exemplar_imagination_axis.get("version") or DUAL_AXIS_REPORT_VERSION,
        "formula_version": DUAL_AXIS_REPORT_VERSION,
        "score": imagination_stats["mean"],
        "imagination": imagination_stats["mean"],
        "hallucination": hallucination_stats["mean"],
        "imagination_raw": summarize_report_metric(imagination_reports, "overall_summary", "axes", "imagination", "raw_score")["mean"],
        "hallucination_raw": summarize_report_metric(hallucination_reports, "overall_summary", "axes", "hallucination", "raw_score")["mean"],
        "coverage_gate_pass": (
            imagination_stats["n"] >= min_required and imagination_stats["mean"] is not None and
            hallucination_stats["n"] >= min_required and hallucination_stats["mean"] is not None
        ),
        "replicate_stats": {
            "imagination": imagination_stats,
            "hallucination": hallucination_stats,
        },
        "task_type_scores": dual_axis_task_scores,
        "component_weights": exemplar_imagination_axis.get("component_weights"),
        "hallucination_component_weights": exemplar_hallucination_axis.get("component_weights"),
        "base_component_weights": exemplar_imagination_axis.get("base_component_weights"),
        "base_hallucination_component_weights": exemplar_hallucination_axis.get("base_component_weights"),
        "component_gate_pass": exemplar_imagination_axis.get("component_gate_pass"),
        "aggregation": {
            "imagination": exemplar_imagination_axis.get("aggregation_policy"),
            "hallucination": exemplar_hallucination_axis.get("aggregation_policy"),
        },
        "calibration": exemplar_imagination_axis.get("residualization"),
    })
    optional_extended_dual_axis.update({
        "role": "diagnostic",
        "version": DUAL_AXIS_REPORT_VERSION,
        "formula_version": DUAL_AXIS_REPORT_VERSION,
        "score": summarize_report_metric(
            repeat_reports, "overall_summary", "axes", "dual_axis_optional_extended", "score",
        )["mean"],
        "imagination": summarize_report_metric(
            repeat_reports, "overall_summary", "axes", "dual_axis_optional_extended", "imagination",
        )["mean"],
        "hallucination": summarize_report_metric(
            repeat_reports, "overall_summary", "axes", "dual_axis_optional_extended", "hallucination",
        )["mean"],
        "imagination_raw": summarize_report_metric(
            repeat_reports, "overall_summary", "axes", "dual_axis_optional_extended", "imagination_raw",
        )["mean"],
        "hallucination_raw": summarize_report_metric(
            repeat_reports, "overall_summary", "axes", "dual_axis_optional_extended", "hallucination_raw",
        )["mean"],
        "coverage_gate_pass": summarize_report_metric(
            repeat_reports, "overall_summary", "axes", "dual_axis_optional_extended", "score",
        )["n"] >= min_required,
        "component_weights": get_nested_value(
            repeat_reports[0] if repeat_reports else {},
            "overall_summary", "axes", "dual_axis_optional_extended", "component_weights",
        ),
        "note": "Diagnostic only: optional dual-axis task families are excluded from the main benchmark score.",
    })
    uut_dual_axis.update({
        "version": UUT_DUAL_AXIS_VERSION,
        "t1_assoc_version": T1_ASSOC_VERSION,
        "score": summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "score")["mean"],
        "imagination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "imagination")["mean"],
        "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "hallucination")["mean"],
        "imagination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "imagination_raw")["mean"],
        "hallucination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "hallucination_raw")["mean"],
        "quality_mass_top8": summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "quality_mass_top8")["mean"],
        "elite_tail_top3": summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "elite_tail_top3")["mean"],
        "diversity_eff": summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "diversity_eff")["mean"],
        "valid_ratio": summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "valid_ratio")["mean"],
        "mechanism_elaboration": summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "mechanism_elaboration")["mean"],
        "bank_coverage": summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "bank_coverage")["mean"],
        "coverage_gate_pass": (
            summarize_report_metric(repeat_reports, "overall_summary", "axes", "uut_affordance_dual_axis", "score")["n"] >= min_required
        ),
        "residualization": {
            "beta_IH": UUT_DUAL_AXIS_BETA_IH,
            "beta_HI": UUT_DUAL_AXIS_BETA_HI,
            "source": "benchmark_default",
        },
        "task_score_stats": {
            "imagination": imagination_task_score_stats,
            "hallucination": hallucination_task_score_stats,
        },
        "primitive_means": uut_axis_primitive_means,
        "primitive_mean_stats": uut_axis_primitive_mean_stats,
        "subtype_contributions": (
            subtype_scores_axis.get("component_contributions", {}).get("UUT")
            if isinstance(subtype_scores_axis.get("component_contributions"), dict) else None
        ),
    })
    propconj_task_scores, propconj_task_score_stats = aggregate_nested_numeric_dicts(
        repeat_reports,
        "overall_summary", "axes", "propconj_dual_axis", "primitive_means",
    )
    propconj_dual_axis.update({
        "version": PROPCONJ_DUAL_AXIS_VERSION,
        "t1_assoc_version": T1_ASSOC_VERSION,
        "score": summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "score")["mean"],
        "imagination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "imagination")["mean"],
        "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "hallucination")["mean"],
        "imagination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "imagination_raw")["mean"],
        "hallucination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "hallucination_raw")["mean"],
        "quality_mass_top6": summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "quality_mass_top6")["mean"],
        "elite_tail_top3": summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "elite_tail_top3")["mean"],
        "diversity_eff": summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "diversity_eff")["mean"],
        "soft_valid_ratio": summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "soft_valid_ratio")["mean"],
        "hard_valid_ratio": summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "hard_valid_ratio")["mean"],
        "conjunction_difficulty_bonus": summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "conjunction_difficulty_bonus")["mean"],
        "coverage_gate_pass": (
            summarize_report_metric(repeat_reports, "overall_summary", "axes", "propconj_dual_axis", "score")["n"] >= min_required
        ),
        "primitive_means": propconj_task_scores,
        "primitive_mean_stats": propconj_task_score_stats,
        "subtype_contributions": (
            subtype_scores_axis.get("component_contributions", {}).get("PropConj")
            if isinstance(subtype_scores_axis.get("component_contributions"), dict) else None
        ),
    })
    macgyver_task_scores, macgyver_task_score_stats = aggregate_nested_numeric_dicts(
        repeat_reports,
        "overall_summary", "axes", "macgyver_dual_axis", "primitive_means",
    )
    macgyver_boundary_record_means, macgyver_boundary_record_mean_stats = aggregate_nested_numeric_dicts(
        repeat_reports,
        "overall_summary", "axes", "macgyver_dual_axis", "boundary_record_means",
    )
    macgyver_dual_axis.update({
        "version": MACGYVER_DUAL_AXIS_VERSION,
        "calibration_policy": MACGYVER_V3_CALIBRATION_POLICY,
        "runtime_scoring_policy": MACGYVER_V3_RUNTIME_SCORING_POLICY,
        "score": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "score")["mean"],
        "imagination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "imagination")["mean"],
        "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "hallucination")["mean"],
        "imagination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "imagination_raw")["mean"],
        "hallucination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "hallucination_raw")["mean"],
        "quality_mass_top3": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "quality_mass_top3")["mean"],
        "elite_tail": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "elite_tail")["mean"],
        "mechanism_chain_depth": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "mechanism_chain_depth")["mean"],
        "constraint_juggling_score": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "constraint_juggling_score")["mean"],
        "strategy_diversity_eff": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "strategy_diversity_eff")["mean"],
        "hard_valid_ratio": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "hard_valid_ratio")["mean"],
        "common_bank_coverage": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "common_bank_coverage")["mean"],
        "coverage_gate_pass": (
            summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "score")["n"] >= min_required
        ),
        "solvability_accuracy": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "solvability_accuracy")["mean"],
        "primitive_means": macgyver_task_scores,
        "primitive_mean_stats": macgyver_task_score_stats,
        "boundary_record_means": macgyver_boundary_record_means,
        "boundary_record_mean_stats": macgyver_boundary_record_mean_stats,
        "subtype_contributions": (
            subtype_scores_axis.get("component_contributions", {}).get("MacGyver")
            if isinstance(subtype_scores_axis.get("component_contributions"), dict) else None
        ),
    })
    macgyver_boundary_axis = final_report["overall_summary"]["axes"].setdefault("macgyver_boundary_diagnostic", {})
    macgyver_boundary_record_means, macgyver_boundary_record_mean_stats = aggregate_nested_numeric_dicts(
        repeat_reports,
        "overall_summary", "axes", "macgyver_boundary_diagnostic", "boundary_record_means",
    )
    macgyver_boundary_axis.update({
        "version": MACGYVER_DUAL_AXIS_VERSION,
        "calibration_policy": "not_strength_calibrated",
        "ran": any(
            bool(report.get("overall_summary", {}).get("axes", {}).get("macgyver_boundary_diagnostic", {}).get("ran"))
            for report in repeat_reports
        ),
        "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_boundary_diagnostic", "hallucination")["mean"],
        "hallucination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_boundary_diagnostic", "hallucination_raw")["mean"],
        "boundary_record_means": macgyver_boundary_record_means,
        "boundary_record_mean_stats": macgyver_boundary_record_mean_stats,
        "subtype_contributions": (
            subtype_scores_axis.get("component_contributions", {}).get("MacGyver")
            if isinstance(subtype_scores_axis.get("component_contributions"), dict) else None
        ),
        "solvability_accuracy": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_boundary_diagnostic", "solvability_accuracy")["mean"],
    })
    cjst_task_scores, cjst_task_score_stats = aggregate_nested_numeric_dicts(
        repeat_reports,
        "overall_summary", "axes", "cjst_dual_axis", "primitive_means",
    )
    cjst_dual_axis.update({
        "version": CJST_DUAL_AXIS_VERSION,
        "calibration_policy": CJST_V3_CALIBRATION_POLICY,
        "runtime_scoring_policy": CJST_V3_RUNTIME_SCORING_POLICY,
        "score": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "score")["mean"],
        "imagination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "imagination")["mean"],
        "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "hallucination")["mean"],
        "imagination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "imagination_raw")["mean"],
        "hallucination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "hallucination_raw")["mean"],
        "quality_mass_top6": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "quality_mass_top6")["mean"],
        "elite_tail_top3": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "elite_tail_top3")["mean"],
        "tier_balanced_depth": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "tier_balanced_depth")["mean"],
        "second_order_chain_score": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "second_order_chain_score")["mean"],
        "world_state_update_score": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "world_state_update_score")["mean"],
        "mechanism_diversity_eff": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "mechanism_diversity_eff")["mean"],
        "hard_valid_ratio": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "hard_valid_ratio")["mean"],
        "common_bank_coverage": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "common_bank_coverage")["mean"],
        "coverage_gate_pass": (
            summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "score")["n"] >= min_required
        ),
        "coverage": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "coverage")["mean"],
        "availability": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "availability")["mean"],
        "primitive_means": cjst_task_scores,
        "primitive_mean_stats": cjst_task_score_stats,
        "subtype_contributions": (
            subtype_scores_axis.get("component_contributions", {}).get("CJST")
            if isinstance(subtype_scores_axis.get("component_contributions"), dict) else None
        ),
        "residualization": {
            "beta_IH": CJST_DUAL_AXIS_BETA_IH,
            "beta_HI": CJST_DUAL_AXIS_BETA_HI,
            "source": CJST_V3_CALIBRATION_POLICY,
            "standardization": "none",
        },
        "formula": {
            "item_imagination_raw": "v3  rarity^1.35 * grounding_gmean^1.25 * hard_gate * world-state/mechanism/tier multiplier, then within-task novelty percentile stretch",
            "task_imagination_raw": "v3  0.35*top6_quality_mass+0.30*top3_elite_tail+0.15*tier_balanced_depth+0.20*second_order_chain_score",
            "task_hallucination_raw": "v3 white-box context/logic/drift burden",
            "model_residual": "clip(mean(raw)-beta*mean(other_raw))",
        },
    })
    hypospace_task_scores, hypospace_task_score_stats = aggregate_nested_numeric_dicts(
        repeat_reports,
        "overall_summary", "axes", "hypospace_dual_axis", "primitive_means",
    )
    hypospace_dual_axis.update({
        "version": HYPOUSESPACE_DUAL_AXIS_VERSION,
        "calibration_policy": HYPOUSESPACE_V3_CALIBRATION_POLICY,
        "runtime_scoring_policy": HYPOUSESPACE_V3_RUNTIME_SCORING_POLICY,
        "score": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "score")["mean"],
        "imagination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "imagination")["mean"],
        "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "hallucination")["mean"],
        "imagination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "imagination_raw")["mean"],
        "hallucination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "hallucination_raw")["mean"],
        "quality_mass_top3": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "quality_mass_top3")["mean"],
        "elite_tail_top2": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "elite_tail_top2")["mean"],
        "mechanism_diversity_eff": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "mechanism_diversity_eff")["mean"],
        "evidence_synthesis_coverage": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "evidence_synthesis_coverage")["mean"],
        "evidence_synthesis_depth": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "evidence_synthesis_depth")["mean"],
        "hard_valid_ratio": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "hard_valid_ratio")["mean"],
        "soft_match_quality": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "soft_match_quality")["mean"],
        "common_bank_coverage": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "common_bank_coverage")["mean"],
        "alias_coverage": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "alias_coverage")["mean"],
        "coverage_gate_pass": (
            summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "score")["n"] >= min_required
        ),
        "coverage": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "coverage")["mean"],
        "availability": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "availability")["mean"],
        "primitive_means": hypospace_task_scores,
        "primitive_mean_stats": hypospace_task_score_stats,
        "subtype_contributions": (
            subtype_scores_axis.get("component_contributions", {}).get("HypoUseSpace")
            if isinstance(subtype_scores_axis.get("component_contributions"), dict) else None
        ),
        "no_valid_accuracy": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "no_valid_accuracy")["mean"],
        "residualization": {
            "beta_IH": HYPOUSESPACE_DUAL_AXIS_BETA_IH,
            "beta_HI": HYPOUSESPACE_DUAL_AXIS_BETA_HI,
            "source": HYPOUSESPACE_V3_CALIBRATION_POLICY,
            "standardization": "clip01_raw_v1",
        },
        "formula": {
            "task_imagination_raw": "v3 I_raw=0.40*top3_quality_mass+0.20*elite_tail_top2+0.15*mechanism_diversity_eff+0.10*evidence_synthesis_coverage+0.10*evidence_synthesis_depth+0.05*hard_valid_ratio",
            "task_imagination_gated": "I_gated=I_raw*evidence_support_gate*boundary_gate",
            "task_hallucination_raw": "H_raw=0.65*mean_i(closed_world/evidence_h)+0.35*support_ledger_h",
            "model_residual": "clip(mean(gated)-beta*mean(other_raw))",
        },
    })
    gcw_task_scores, gcw_task_score_stats = aggregate_nested_numeric_dicts(
        repeat_reports,
        "overall_summary", "axes", "gcw_dual_axis", "primitive_means",
    )
    gcw_dual_axis.update({
        "version": GCW_DUAL_AXIS_VERSION,
        "calibration_policy": GCW_V3_CALIBRATION_POLICY,
        "runtime_scoring_policy": GCW_V3_RUNTIME_SCORING_POLICY,
        "primary_task_ids": next(
            (
                get_nested_value(report, "overall_summary", "axes", "gcw_dual_axis", "primary_task_ids")
                for report in repeat_reports
                if get_nested_value(report, "overall_summary", "axes", "gcw_dual_axis", "primary_task_ids")
            ),
            [],
        ),
        "output_beat_count": GCW_BEAT_COUNT,
        "max_tokens": get_task_max_tokens("GCW"),
        "common_story_bank_coverage": next(
            (
                get_nested_value(report, "overall_summary", "axes", "gcw_dual_axis", "common_story_bank_coverage")
                for report in repeat_reports
                if get_nested_value(report, "overall_summary", "axes", "gcw_dual_axis", "common_story_bank_coverage")
            ),
            None,
        ),
        "entity_alias_coverage": next(
            (
                get_nested_value(report, "overall_summary", "axes", "gcw_dual_axis", "entity_alias_coverage")
                for report in repeat_reports
                if get_nested_value(report, "overall_summary", "axes", "gcw_dual_axis", "entity_alias_coverage")
            ),
            None,
        ),
        "common_story_bank_version": next(
            (
                get_nested_value(report, "overall_summary", "axes", "gcw_dual_axis", "common_story_bank_version")
                for report in repeat_reports
                if get_nested_value(report, "overall_summary", "axes", "gcw_dual_axis", "common_story_bank_version")
            ),
            None,
        ),
        "entity_alias_bank_version": next(
            (
                get_nested_value(report, "overall_summary", "axes", "gcw_dual_axis", "entity_alias_bank_version")
                for report in repeat_reports
                if get_nested_value(report, "overall_summary", "axes", "gcw_dual_axis", "entity_alias_bank_version")
            ),
            None,
        ),
        "score": summarize_report_metric(repeat_reports, "overall_summary", "axes", "gcw_dual_axis", "score")["mean"],
        "imagination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "gcw_dual_axis", "imagination")["mean"],
        "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "gcw_dual_axis", "hallucination")["mean"],
        "imagination_gated": summarize_report_metric(repeat_reports, "overall_summary", "axes", "gcw_dual_axis", "imagination_gated")["mean"],
        "imagination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "gcw_dual_axis", "imagination_raw")["mean"],
        "hallucination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "gcw_dual_axis", "hallucination_raw")["mean"],
        "coverage_gate_pass": (
            summarize_report_metric(repeat_reports, "overall_summary", "axes", "gcw_dual_axis", "score")["n"] >= min_required
        ),
        "coverage": summarize_report_metric(repeat_reports, "overall_summary", "axes", "gcw_dual_axis", "coverage")["mean"],
        "availability": summarize_report_metric(repeat_reports, "overall_summary", "axes", "gcw_dual_axis", "availability")["mean"],
        "primitive_means": gcw_task_scores,
        "primitive_mean_stats": gcw_task_score_stats,
        "grounded_turn_quality": gcw_task_scores.get("grounded_turn_quality"),
        "causal_payoff": gcw_task_scores.get("causal_payoff"),
        "top3_scene_specificity": gcw_task_scores.get("top3_scene_specificity"),
        "arc_diversity_eff": gcw_task_scores.get("arc_diversity_eff"),
        "hard_valid_ledger_ratio": gcw_task_scores.get("hard_valid_ledger_ratio"),
        "common_bank_coverage": gcw_task_scores.get("common_bank_coverage"),
        "subtype_contributions": (
            subtype_scores_axis.get("component_contributions", {}).get("GCW")
            if isinstance(subtype_scores_axis.get("component_contributions"), dict) else None
        ),
        "gcw_ttcw_proxy": {
            key: gcw_task_scores.get(key)
            for key in ["F_story", "X_story", "O_story", "E_story"]
            if key in gcw_task_scores
        },
        "gcw_fact_grounding": gcw_task_scores.get("fact_grounding"),
        "residualization": {
            "beta_IH": GCW_DUAL_AXIS_BETA_IH,
            "beta_HI": GCW_DUAL_AXIS_BETA_HI,
            "source": "benchmark_default",
            "standardization": "robust_z",
        },
        "formula": {
            "task_imagination_raw": "T4- I_raw=0.40*grounded_turn_quality+0.20*causal_payoff+0.15*top3_scene_specificity+0.15*arc_diversity_eff+0.10*hard_valid_ledger_ratio",
            "task_imagination_gated": "I_gated=I_raw*support_gate*constraint_gate",
            "task_hallucination_raw": "H_raw=0.58*closed_world_h+0.42*support_ledger_h",
            "task_residual": "I=sigmoid(zI_gated-beta_IH*zH); H=sigmoid(zH-beta_HI*zI_gated)",
        },
    })
    neocoder_task_scores, neocoder_task_score_stats = aggregate_nested_numeric_dicts(
        repeat_reports,
        "overall_summary", "axes", "neocoder_dual_axis", "primitive_means",
    )
    neocoder_dual_axis.update({
        "role": "primary" if "NeoCoder" in PRIMARY_DUAL_AXIS_COMPONENTS else "enhanced_diagnostic",
        "version": NEOCODER_DUAL_AXIS_VERSION,
        "calibration_policy": NEOCODER_V3_CALIBRATION_POLICY,
        "runtime_scoring_policy": NEOCODER_V3_RUNTIME_SCORING_POLICY,
        "test_visibility_policy": NEOCODER_V3_TEST_VISIBILITY_POLICY,
        "output_count": NEOCODER_OUTPUT_COUNT,
        "max_tokens": get_task_max_tokens("NeoCoder"),
        "common_solution_bank_version": NEOCODER_V3_COMMON_SOLUTION_BANK_VERSION,
        "task_overlay_version": NEOCODER_V3_TASK_OVERLAY_VERSION,
        "technique_alias_version": NEOCODER_V3_TECHNIQUE_ALIAS_VERSION,
        "primary_task_ids": sorted({
            detail.get("task_id")
            for report in repeat_reports
            for detail in (((report.get("neocoder_results") or {}).get("details")) or [])
            if isinstance(detail, dict) and detail.get("task_id")
        }),
        "common_solution_bank_coverage": get_neocoder_common_solution_bank_coverage(sorted({
            detail.get("task_id")
            for report in repeat_reports
            for detail in (((report.get("neocoder_results") or {}).get("details")) or [])
            if isinstance(detail, dict) and detail.get("task_id")
        })),
        "task_overlay_coverage": get_neocoder_task_overlay_coverage(sorted({
            detail.get("task_id")
            for report in repeat_reports
            for detail in (((report.get("neocoder_results") or {}).get("details")) or [])
            if isinstance(detail, dict) and detail.get("task_id")
        })),
        "technique_alias_coverage": get_neocoder_technique_alias_coverage(),
        "score": summarize_report_metric(repeat_reports, "overall_summary", "axes", "neocoder_dual_axis", "score")["mean"],
        "imagination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "neocoder_dual_axis", "imagination")["mean"],
        "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "neocoder_dual_axis", "hallucination")["mean"],
        "imagination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "neocoder_dual_axis", "imagination_raw")["mean"],
        "imagination_gated": summarize_report_metric(repeat_reports, "overall_summary", "axes", "neocoder_dual_axis", "imagination_gated")["mean"],
        "hallucination_raw": summarize_report_metric(repeat_reports, "overall_summary", "axes", "neocoder_dual_axis", "hallucination_raw")["mean"],
        "coverage_gate_pass": (
            summarize_report_metric(repeat_reports, "overall_summary", "axes", "neocoder_dual_axis", "score")["n"] >= min_required
        ),
        "coverage": summarize_report_metric(repeat_reports, "overall_summary", "axes", "neocoder_dual_axis", "coverage")["mean"],
        "availability": summarize_report_metric(repeat_reports, "overall_summary", "axes", "neocoder_dual_axis", "availability")["mean"],
        "primitive_means": neocoder_task_scores,
        "functional_quality": neocoder_task_scores.get("functional_quality"),
        "public_pass_rate": neocoder_task_scores.get("public_pass_rate"),
        "hidden_pass_rate": neocoder_task_scores.get("hidden_pass_rate"),
        "metamorphic_pass_rate": neocoder_task_scores.get("metamorphic_pass_rate"),
        "strategy_rarity": neocoder_task_scores.get("strategy_rarity"),
        "implementation_depth": neocoder_task_scores.get("implementation_depth"),
        "constraint_quality": neocoder_task_scores.get("constraint_quality"),
        "denial_adaptation": neocoder_task_scores.get("denial_adaptation"),
        "algorithmic_pattern_diversity": neocoder_task_scores.get("algorithmic_pattern_diversity"),
        "denial_adaptation_quality": neocoder_task_scores.get("denial_adaptation_quality"),
        "anti_overfit_gate": neocoder_task_scores.get("anti_overfit_gate"),
        "primitive_mean_stats": neocoder_task_score_stats,
        "subtype_contributions": (
            subtype_scores_axis.get("component_contributions", {}).get("NeoCoder")
            if isinstance(subtype_scores_axis.get("component_contributions"), dict) else None
        ),
        "residualization": {
            "beta_IH": NEOCODER_DUAL_AXIS_BETA_IH,
            "beta_HI": NEOCODER_DUAL_AXIS_BETA_HI,
            "source": NEOCODER_V3_CALIBRATION_POLICY,
            "standardization": "clip01_raw_v1",
        },
        "formula": {
            "task_imagination_raw": "T6  I=rarity^1.35*functional_quality^1.45*constraint_quality^1.25*anti_overfit_gate*(0.35+0.20*implementation_depth+0.15*denial_adaptation+0.15*algorithmic_pattern_diversity+0.10*denial_adaptation_quality+0.05*ledger_consistency)",
            "task_functional_quality": "0.25*public_pass+0.55*hidden_pass+0.20*metamorphic_pass",
            "task_imagination_gated": "I_gated=I_raw*safety_gate with denied/mutation caps",
            "task_hallucination_raw": "H_raw=0.45*H_logic+0.35*H_intent+0.20*H_fact",
            "model_residual": "I=clip01(mean(I_gated)-beta_IH*mean(H_raw)); H=clip01(mean(H_raw)-beta_HI*mean(I_gated))",
        },
        "note": (
            "Primary T6 component in the active experiment profile."
            if "NeoCoder" in PRIMARY_DUAL_AXIS_COMPONENTS else
            "Enhanced diagnostic only; excluded from primary rankings."
        ),
    })
    closed_world_fact_task_scores_mean, closed_world_fact_task_score_stats = aggregate_nested_numeric_dicts(
        repeat_reports,
        "overall_summary", "axes", "closed_world_fact_calibration", "primitive_means",
    )
    closed_world_fact_calibration.update({
        "role": "hallucination_calibration",
        "version": CLOSED_WORLD_FACT_VERSION,
        "score": closed_world_fact_stats["mean"],
        "hallucination": summarize_report_metric(
            repeat_reports,
            "overall_summary", "axes", "closed_world_fact_calibration", "hallucination",
        )["mean"],
        "hallucination_raw": summarize_report_metric(
            repeat_reports,
            "overall_summary", "axes", "closed_world_fact_calibration", "hallucination_raw",
        )["mean"],
        "imagination": None,
        "imagination_raw": None,
        "coverage_gate_pass": closed_world_fact_stats["n"] >= min_required,
        "coverage": summarize_report_metric(
            repeat_reports,
            "overall_summary", "axes", "closed_world_fact_calibration", "coverage",
        )["mean"],
        "availability": summarize_report_metric(
            repeat_reports,
            "overall_summary", "axes", "closed_world_fact_calibration", "availability",
        )["mean"],
        "primitive_means": closed_world_fact_task_scores_mean,
        "primitive_mean_stats": closed_world_fact_task_score_stats,
        "subtype_contributions": (
            subtype_scores_axis.get("component_contributions", {}).get("ClosedWorldFact")
            if isinstance(subtype_scores_axis.get("component_contributions"), dict) else None
        ),
        "replicate_stats": closed_world_fact_stats,
        "formula": {
            "hallucination_raw": "H_raw=0.45*H_fact+0.30*H_logic+0.25*H_boundary",
            "score": "calibration_score=1-H_raw",
        },
        "note": "Calibration diagnostic only; excluded from imagination, primary dual-axis, optional dual-axis, and ranking eligibility.",
    })
    analogy_transfer_task_scores_mean, analogy_transfer_task_score_stats = aggregate_nested_numeric_dicts(
        repeat_reports,
        "overall_summary", "axes", "analogy_transfer_challenge", "primitive_means",
    )
    analogy_transfer_challenge.update({
        "role": "primary" if "AnalogyTransfer" in PRIMARY_DUAL_AXIS_COMPONENTS else "challenge_diagnostic",
        "version": ANALOGY_TRANSFER_VERSION,
        "calibration_policy": ANALOGY_TRANSFER_V3_CALIBRATION_POLICY,
        "runtime_scoring_policy": ANALOGY_TRANSFER_V3_RUNTIME_SCORING_POLICY,
        "test_visibility_policy": ANALOGY_TRANSFER_V3_TEST_VISIBILITY_POLICY,
        "output_count": ANALOGY_TRANSFER_OUTPUT_COUNT,
        "max_tokens": get_task_max_tokens("AnalogyTransfer"),
        "common_mapping_bank_version": ANALOGY_COMMON_MAPPING_BANK_VERSION,
        "task_overlay_version": ANALOGY_TRANSFER_V3_TASK_OVERLAY_VERSION,
        "primary_task_ids": sorted({
            detail.get("task_id")
            for report in repeat_reports
            for detail in (((report.get("analogy_transfer_results") or {}).get("details")) or [])
            if isinstance(detail, dict) and detail.get("task_id")
        }),
        "common_mapping_bank_coverage": get_analogy_common_mapping_bank_coverage(sorted({
            detail.get("task_id")
            for report in repeat_reports
            for detail in (((report.get("analogy_transfer_results") or {}).get("details")) or [])
            if isinstance(detail, dict) and detail.get("task_id")
        })),
        "task_overlay_coverage": get_analogy_transfer_task_overlay_coverage(sorted({
            detail.get("task_id")
            for report in repeat_reports
            for detail in (((report.get("analogy_transfer_results") or {}).get("details")) or [])
            if isinstance(detail, dict) and detail.get("task_id")
        })),
        "evidence_alias_coverage": get_analogy_evidence_alias_coverage(sorted({
            detail.get("task_id")
            for report in repeat_reports
            for detail in (((report.get("analogy_transfer_results") or {}).get("details")) or [])
            if isinstance(detail, dict) and detail.get("task_id")
        })),
        "score": analogy_transfer_stats["mean"],
        "imagination": summarize_report_metric(
            repeat_reports,
            "overall_summary", "axes", "analogy_transfer_challenge", "imagination",
        )["mean"],
        "hallucination": summarize_report_metric(
            repeat_reports,
            "overall_summary", "axes", "analogy_transfer_challenge", "hallucination",
        )["mean"],
        "imagination_raw": summarize_report_metric(
            repeat_reports,
            "overall_summary", "axes", "analogy_transfer_challenge", "imagination_raw",
        )["mean"],
        "imagination_gated": summarize_report_metric(
            repeat_reports,
            "overall_summary", "axes", "analogy_transfer_challenge", "imagination_gated",
        )["mean"],
        "hallucination_raw": summarize_report_metric(
            repeat_reports,
            "overall_summary", "axes", "analogy_transfer_challenge", "hallucination_raw",
        )["mean"],
        "coverage_gate_pass": analogy_transfer_stats["n"] >= min_required,
        "coverage": summarize_report_metric(
            repeat_reports,
            "overall_summary", "axes", "analogy_transfer_challenge", "coverage",
        )["mean"],
        "availability": summarize_report_metric(
            repeat_reports,
            "overall_summary", "axes", "analogy_transfer_challenge", "availability",
        )["mean"],
        "primitive_means": analogy_transfer_task_scores_mean,
        "top3_mapping_quality": analogy_transfer_task_scores_mean.get("top3_mapping_quality"),
        "elite_tail_top1": analogy_transfer_task_scores_mean.get("elite_tail_top1"),
        "licensed_inference_quality": analogy_transfer_task_scores_mean.get("licensed_inference_quality"),
        "abstraction_diversity_eff": analogy_transfer_task_scores_mean.get("abstraction_diversity_eff"),
        "relational_depth": analogy_transfer_task_scores_mean.get("relational_depth"),
        "cross_domain_distance": analogy_transfer_task_scores_mean.get("cross_domain_distance"),
        "boundary_aware_valid_ratio": analogy_transfer_task_scores_mean.get("boundary_aware_valid_ratio"),
        "structural_match_gmean": analogy_transfer_task_scores_mean.get("structural_match_gmean"),
        "evidence_grounding": analogy_transfer_task_scores_mean.get("evidence_grounding"),
        "mapping_rarity": analogy_transfer_task_scores_mean.get("mapping_rarity"),
        "primitive_mean_stats": analogy_transfer_task_score_stats,
        "subtype_contributions": (
            subtype_scores_axis.get("component_contributions", {}).get("AnalogyTransfer")
            if isinstance(subtype_scores_axis.get("component_contributions"), dict) else None
        ),
        "replicate_stats": analogy_transfer_stats,
        "formula": {
            "task_imagination_raw": "T8  I_raw=0.25*top3_mapping_quality+0.25*elite_tail_top1+0.20*relational_depth+0.15*licensed_inference_quality+0.05*abstraction_diversity_eff+0.10*boundary_aware_valid_ratio",
            "task_mapping_quality": "mapping_I=rarity^1.30*structural_match_gmean^1.45*evidence_grounding^1.25*hard_gate*(0.35+0.15*abstraction_depth+0.15*cross_domain_transform+0.20*relational_depth+0.05*cross_domain_distance+0.10*boundary_awareness)",
            "task_imagination_gated": "I_gated=I_raw*source_fact_gate*target_fact_gate*transfer_gate",
            "task_hallucination_raw": "H_raw=0.40*H_false_transfer+0.25*H_fact+0.20*H_logic+0.15*H_context",
            "model_residual": "I=clip01(mean(I_gated)-beta_IH*mean(H_raw)); H=clip01(mean(H_raw)-beta_HI*mean(I_gated))",
        },
        "note": (
            "Primary T8 component in the active experiment profile."
            if "AnalogyTransfer" in PRIMARY_DUAL_AXIS_COMPONENTS else
            "Challenge diagnostic only; excluded from primary rankings."
        ),
    })
    cross_task_fact_consistency_aggregate = aggregate_cross_task_fact_consistency_axes([
        get_nested_value(report, "overall_summary", "axes", "cross_task_fact_consistency")
        for report in repeat_reports
        if isinstance(get_nested_value(report, "overall_summary", "axes", "cross_task_fact_consistency"), dict)
    ])
    cross_task_fact_consistency_axis.clear()
    cross_task_fact_consistency_axis.update(cross_task_fact_consistency_aggregate)


    run_validity = final_report.setdefault("run_validity", {})
    creative_coverages_mean, creative_coverages_stats = aggregate_nested_numeric_dicts(repeat_reports, "run_validity", "creative_coverages")
    creative_availability_mean, creative_availability_stats = aggregate_nested_numeric_dicts(repeat_reports, "run_validity", "creative_availability")
    invalid_run_total = Counter()
    non_model_skip_total = Counter()
    non_model_skip_reasons_total = Counter()
    for report in repeat_reports:
        invalid_run_total.update(get_nested_value(report, "run_validity", "invalid_run_counts") or {})
        non_model_skip_total.update(get_nested_value(report, "run_validity", "non_model_skip_counts") or {})
        non_model_skip_reasons_total.update(get_nested_value(report, "run_validity", "non_model_skip_reasons") or {})
    optional_dual_axis_issues = sorted({
        issue
        for report in repeat_reports
        for issue in (get_nested_value(report, "run_validity", "optional_dual_axis_issues") or [])
    })
    auxiliary_diagnostic_issues = sorted({
        issue
        for report in repeat_reports
        for issue in (get_nested_value(report, "run_validity", "auxiliary_diagnostic_issues") or [])
    })
    enhanced_diagnostic_issues = sorted({
        issue
        for report in repeat_reports
        for issue in (get_nested_value(report, "run_validity", "enhanced_diagnostic_issues") or [])
    })
    calibration_diagnostic_issues = sorted({
        issue
        for report in repeat_reports
        for issue in (get_nested_value(report, "run_validity", "calibration_diagnostic_issues") or [])
    })
    challenge_diagnostic_issues = sorted({
        issue
        for report in repeat_reports
        for issue in (get_nested_value(report, "run_validity", "challenge_diagnostic_issues") or [])
    })

    eligibility_failures = []
    if imagination_stats["n"] < min_required:
        eligibility_failures.append(f"imagination eligible repeats {imagination_stats['n']}/{MODEL_SAMPLE_REPEATS}; need {min_required}")
    if hallucination_stats["n"] < min_required:
        eligibility_failures.append(f"hallucination eligible repeats {hallucination_stats['n']}/{MODEL_SAMPLE_REPEATS}; need {min_required}")
    for component in PRIMARY_DUAL_AXIS_COMPONENTS:
        observed = primary_dual_repeat_counts.get(component, 0)
        if observed < min_required:
            eligibility_failures.append(f"{component} primary dual-axis eligible repeats {observed}/{MODEL_SAMPLE_REPEATS}; need {min_required}")
    if not final_ranking_eligible:
        ineligible_summaries = [summary for summary in repeat_summaries if not summary.get("ranking_eligible")]
        if ineligible_summaries:
            compact = []
            for summary in ineligible_summaries[:3]:
                reasons = summary.get("eligibility_failures") or []
                if reasons:
                    compact.append(f"r{summary['repeat_index'] + 1}: {reasons[0]}")
            if compact:
                eligibility_failures.append("repeat failures: " + "; ".join(compact))

    run_validity.update({
        "reports_generated": True,
        "ranking_eligible": final_ranking_eligible,
        "eligibility_failures": eligibility_failures,
        "primary_axis": "dual_axis",
        "experiment_profile": OPENROUTER_EXPERIMENT_PROFILE,
        "profile_task_manifest_path": PROFILE_TASK_MANIFEST_PATH or None,
        "typed_correlation_ready": typed_correlation_ready,
        "subtype_schema_version": subtype_schema_version,
        "primary_dual_axis_components": list(PRIMARY_DUAL_AXIS_COMPONENTS),
        "optional_dual_axis_components": list(OPTIONAL_DUAL_AXIS_COMPONENTS),
        "auxiliary_imagination_diagnostics": list(AUXILIARY_IMAGINATION_DIAGNOSTICS),
        "enhanced_dual_axis_diagnostics": list(ENHANCED_DUAL_AXIS_DIAGNOSTICS),
        "calibration_diagnostics": list(CALIBRATION_DIAGNOSTICS),
        "challenge_diagnostics": list(CHALLENGE_DIAGNOSTICS),
        "optional_dual_axis_issues": optional_dual_axis_issues,
        "auxiliary_diagnostic_issues": auxiliary_diagnostic_issues,
        "enhanced_diagnostic_issues": enhanced_diagnostic_issues,
        "calibration_diagnostic_issues": calibration_diagnostic_issues,
        "challenge_diagnostic_issues": challenge_diagnostic_issues,
        "repeat_policy": {
            "requested_repeats": MODEL_SAMPLE_REPEATS,
            "repeat_eligible_fraction": REPEAT_ELIGIBLE_FRACTION,
            "minimum_eligible_repeats": min_required,
            "dt_total_eligible_repeats": dt_stats["n"],
            "novelty_eligible_repeats": novelty_stats["n"],
            "flexibility_eligible_repeats": flexibility_stats["n"],
            "groundedness_eligible_repeats": groundedness_stats["n"],
            "imagination_eligible_repeats": imagination_stats["n"],
            "hallucination_eligible_repeats": hallucination_stats["n"],
            "uut_eligible_repeats": uut_dual_stats["n"],
            "propconj_eligible_repeats": propconj_dual_stats["n"],
            "macgyver_eligible_repeats": macgyver_dual_stats["n"],
            "cjst_eligible_repeats": cjst_dual_stats["n"],
            "hypospace_eligible_repeats": hypospace_dual_stats["n"],
            "gcw_eligible_repeats": gcw_dual_stats["n"],
            "neocoder_eligible_repeats": neocoder_dual_stats["n"],
            "closed_world_fact_eligible_repeats": closed_world_fact_stats["n"],
            "analogy_transfer_eligible_repeats": analogy_transfer_stats["n"],
            "dt_total_repeat_indices": [report.get("repeat_index") for report in dt_reports],
            "groundedness_repeat_indices": [report.get("repeat_index") for report in groundedness_reports],
            "imagination_repeat_indices": [report.get("repeat_index") for report in imagination_reports],
            "hallucination_repeat_indices": [report.get("repeat_index") for report in hallucination_reports],
        },
        "repeat_summaries": repeat_summaries,
        "creative_coverages": creative_coverages_mean,
        "creative_coverages_stats": creative_coverages_stats,
        "creative_availability": creative_availability_mean,
        "creative_availability_stats": creative_availability_stats,
        "creative_total_coverage": summarize_report_metric(repeat_reports, "run_validity", "creative_total_coverage")["mean"],
        "creative_total_coverage_stats": summarize_report_metric(repeat_reports, "run_validity", "creative_total_coverage"),
        "creative_total_availability": summarize_report_metric(repeat_reports, "run_validity", "creative_total_availability")["mean"],
        "creative_total_availability_stats": summarize_report_metric(repeat_reports, "run_validity", "creative_total_availability"),
        "dat_coverage": summarize_report_metric(repeat_reports, "run_validity", "dat_coverage")["mean"],
        "dat_coverage_stats": summarize_report_metric(repeat_reports, "run_validity", "dat_coverage"),
        "dat_availability": summarize_report_metric(repeat_reports, "run_validity", "dat_availability")["mean"],
        "dat_availability_stats": summarize_report_metric(repeat_reports, "run_validity", "dat_availability"),
        "cdat_coverage": summarize_report_metric(repeat_reports, "run_validity", "cdat_coverage")["mean"],
        "cdat_coverage_stats": summarize_report_metric(repeat_reports, "run_validity", "cdat_coverage"),
        "cdat_availability": summarize_report_metric(repeat_reports, "run_validity", "cdat_availability")["mean"],
        "cdat_availability_stats": summarize_report_metric(repeat_reports, "run_validity", "cdat_availability"),
        "macgyver_coverage": summarize_report_metric(repeat_reports, "run_validity", "macgyver_coverage")["mean"],
        "macgyver_coverage_stats": summarize_report_metric(repeat_reports, "run_validity", "macgyver_coverage"),
        "macgyver_availability": summarize_report_metric(repeat_reports, "run_validity", "macgyver_availability")["mean"],
        "macgyver_availability_stats": summarize_report_metric(repeat_reports, "run_validity", "macgyver_availability"),
        "macgyver_boundary_coverage": summarize_report_metric(repeat_reports, "run_validity", "macgyver_boundary_coverage")["mean"],
        "macgyver_boundary_coverage_stats": summarize_report_metric(repeat_reports, "run_validity", "macgyver_boundary_coverage"),
        "macgyver_boundary_availability": summarize_report_metric(repeat_reports, "run_validity", "macgyver_boundary_availability")["mean"],
        "macgyver_boundary_availability_stats": summarize_report_metric(repeat_reports, "run_validity", "macgyver_boundary_availability"),
        "cjst_coverage": summarize_report_metric(repeat_reports, "run_validity", "cjst_coverage")["mean"],
        "cjst_coverage_stats": summarize_report_metric(repeat_reports, "run_validity", "cjst_coverage"),
        "cjst_availability": summarize_report_metric(repeat_reports, "run_validity", "cjst_availability")["mean"],
        "cjst_availability_stats": summarize_report_metric(repeat_reports, "run_validity", "cjst_availability"),
        "hypospace_coverage": summarize_report_metric(repeat_reports, "run_validity", "hypospace_coverage")["mean"],
        "hypospace_coverage_stats": summarize_report_metric(repeat_reports, "run_validity", "hypospace_coverage"),
        "hypospace_availability": summarize_report_metric(repeat_reports, "run_validity", "hypospace_availability")["mean"],
        "hypospace_availability_stats": summarize_report_metric(repeat_reports, "run_validity", "hypospace_availability"),
        "gcw_coverage": summarize_report_metric(repeat_reports, "run_validity", "gcw_coverage")["mean"],
        "gcw_coverage_stats": summarize_report_metric(repeat_reports, "run_validity", "gcw_coverage"),
        "gcw_availability": summarize_report_metric(repeat_reports, "run_validity", "gcw_availability")["mean"],
        "gcw_availability_stats": summarize_report_metric(repeat_reports, "run_validity", "gcw_availability"),
        "neocoder_coverage": summarize_report_metric(repeat_reports, "run_validity", "neocoder_coverage")["mean"],
        "neocoder_coverage_stats": summarize_report_metric(repeat_reports, "run_validity", "neocoder_coverage"),
        "neocoder_availability": summarize_report_metric(repeat_reports, "run_validity", "neocoder_availability")["mean"],
        "neocoder_availability_stats": summarize_report_metric(repeat_reports, "run_validity", "neocoder_availability"),
        "closed_world_fact_total_prompts": summarize_report_metric(repeat_reports, "run_validity", "closed_world_fact_total_prompts")["mean"],
        "closed_world_fact_effective_prompts": summarize_report_metric(repeat_reports, "run_validity", "closed_world_fact_effective_prompts")["mean"],
        "closed_world_fact_excluded_prompts": summarize_report_metric(repeat_reports, "run_validity", "closed_world_fact_excluded_prompts")["mean"],
        "closed_world_fact_scorable_prompts": summarize_report_metric(repeat_reports, "run_validity", "closed_world_fact_scorable_prompts")["mean"],
        "closed_world_fact_coverage": summarize_report_metric(repeat_reports, "run_validity", "closed_world_fact_coverage")["mean"],
        "closed_world_fact_coverage_stats": summarize_report_metric(repeat_reports, "run_validity", "closed_world_fact_coverage"),
        "closed_world_fact_availability": summarize_report_metric(repeat_reports, "run_validity", "closed_world_fact_availability")["mean"],
        "closed_world_fact_availability_stats": summarize_report_metric(repeat_reports, "run_validity", "closed_world_fact_availability"),
        "analogy_transfer_total_prompts": summarize_report_metric(repeat_reports, "run_validity", "analogy_transfer_total_prompts")["mean"],
        "analogy_transfer_effective_prompts": summarize_report_metric(repeat_reports, "run_validity", "analogy_transfer_effective_prompts")["mean"],
        "analogy_transfer_excluded_prompts": summarize_report_metric(repeat_reports, "run_validity", "analogy_transfer_excluded_prompts")["mean"],
        "analogy_transfer_scorable_prompts": summarize_report_metric(repeat_reports, "run_validity", "analogy_transfer_scorable_prompts")["mean"],
        "analogy_transfer_coverage": summarize_report_metric(repeat_reports, "run_validity", "analogy_transfer_coverage")["mean"],
        "analogy_transfer_coverage_stats": summarize_report_metric(repeat_reports, "run_validity", "analogy_transfer_coverage"),
        "analogy_transfer_availability": summarize_report_metric(repeat_reports, "run_validity", "analogy_transfer_availability")["mean"],
        "analogy_transfer_availability_stats": summarize_report_metric(repeat_reports, "run_validity", "analogy_transfer_availability"),
        "ff_coverage": summarize_report_metric(repeat_reports, "run_validity", "ff_coverage")["mean"],
        "ff_coverage_stats": summarize_report_metric(repeat_reports, "run_validity", "ff_coverage"),
        "ff_availability": summarize_report_metric(repeat_reports, "run_validity", "ff_availability")["mean"],
        "ff_availability_stats": summarize_report_metric(repeat_reports, "run_validity", "ff_availability"),
        "invalid_run_counts": dict(invalid_run_total),
        "non_model_skip_counts": dict(non_model_skip_total),
        "non_model_skip_reasons": dict(non_model_skip_reasons_total),
        "axis_validity": {
            "primary_dual_axis": final_primary_dual_gate_pass,
            "novelty": novelty_stats["n"] >= min_required and novelty_stats["mean"] is not None,
            "flexibility": flexibility_stats["n"] >= min_required and flexibility_stats["mean"] is not None,
            "groundedness": groundedness_stats["n"] >= min_required and groundedness_stats["mean"] is not None,
            "imagination": imagination_stats["n"] >= min_required and imagination_stats["mean"] is not None,
            "hallucination": hallucination_stats["n"] >= min_required and hallucination_stats["mean"] is not None,
            "uut_affordance_dual_axis": uut_dual_stats["n"] >= min_required,
            "propconj_dual_axis": propconj_dual_stats["n"] >= min_required,
            "macgyver_dual_axis": (
                macgyver_dual_stats["n"] >= min_required
            ),
            "cjst_dual_axis": (
                cjst_dual_stats["n"] >= min_required
            ),
            "hypospace_dual_axis": (
                hypospace_dual_stats["n"] >= min_required
            ),
            "gcw_dual_axis": (
                gcw_dual_stats["n"] >= min_required
            ),
            "neocoder_dual_axis": (
                neocoder_dual_stats["n"] >= min_required
            ),
            "closed_world_fact_calibration": (
                closed_world_fact_stats["n"] >= min_required
            ),
            "analogy_transfer_challenge": (
                analogy_transfer_stats["n"] >= min_required
            ),
            "dt_total": dt_stats["n"] >= min_required and dt_stats["mean"] is not None,
        },
    })

    if isinstance(final_report.get("dat_cdat_ff_results"), dict):
        dat_section = final_report["dat_cdat_ff_results"].get("dat")
        if isinstance(dat_section, dict):
            dat_details = collect_repeat_section_details(repeat_reports, "dat")
            dat_section["mean_score"] = summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "dat", "mean_score")["mean"]
            dat_section["replicate_stats"] = summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "dat", "mean_score")
            dat_section["details"] = dat_details
            dat_section["trials"] = len(dat_details)
            dat_section["scores"] = [
                detail.get("dat_score")
                for detail in dat_details
                if detail.get("dat_score") is not None
            ]
        cdat_section = final_report["dat_cdat_ff_results"].get("cdat")
        if isinstance(cdat_section, dict):
            cdat_details = collect_repeat_section_details(repeat_reports, "cdat")
            cdat_section["mean_novelty"] = summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "cdat", "mean_novelty")["mean"]
            cdat_section["mean_appropriateness"] = summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "cdat", "mean_appropriateness")["mean"]
            cdat_section["gate_pass_rate"] = summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "cdat", "gate_pass_rate")["mean"]
            cdat_section["mean_app_gain"] = summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "cdat", "mean_app_gain")["mean"]
            cdat_section["mean_multiplier"] = summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "cdat", "mean_multiplier")["mean"]
            cdat_section["cdat_score"] = summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "cdat", "cdat_score")["mean"]
            cdat_section["details"] = cdat_details
            cdat_section["cues_evaluated"] = len(cdat_details)
            cdat_section["replicate_stats"] = {
                "mean_novelty": summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "cdat", "mean_novelty"),
                "mean_appropriateness": summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "cdat", "mean_appropriateness"),
                "gate_pass_rate": summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "cdat", "gate_pass_rate"),
                "mean_app_gain": summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "cdat", "mean_app_gain"),
                "mean_multiplier": summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "cdat", "mean_multiplier"),
                "cdat_score": summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "cdat", "cdat_score"),
            }
        ff_section = final_report["dat_cdat_ff_results"].get("ff")
        if isinstance(ff_section, dict):
            ff_details = collect_repeat_section_details(repeat_reports, "ff")
            ff_section["mean_score"] = summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "ff", "mean_score")["mean"]
            ff_section["mean_trajectory_slope"] = summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "ff", "mean_trajectory_slope")["mean"]
            ff_section["details"] = ff_details
            ff_section["trials"] = len(ff_details)
            ff_section["scores"] = [
                detail.get("dynamic_forward_flow")
                for detail in ff_details
                if detail.get("dynamic_forward_flow") is not None
            ]
            ff_section["replicate_stats"] = {
                "mean_score": summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "ff", "mean_score"),
                "mean_trajectory_slope": summarize_report_metric(repeat_reports, "dat_cdat_ff_results", "ff", "mean_trajectory_slope"),
            }

    if isinstance(final_report.get("macgyver_results"), dict):
        macgyver_details = []
        for report in repeat_reports:
            section = report.get("macgyver_results")
            if not isinstance(section, dict):
                continue
            for detail in section.get("details") or []:
                macgyver_details.append(copy.deepcopy(detail))
        final_report["macgyver_results"].update({
            "details": macgyver_details,
            "trials": len(macgyver_details),
            "scorable_trials": summarize_report_metric(repeat_reports, "macgyver_results", "scorable_trials")["mean"],
            "coverage": summarize_report_metric(repeat_reports, "macgyver_results", "coverage")["mean"],
            "availability": summarize_report_metric(repeat_reports, "macgyver_results", "availability")["mean"],
            "replicate_stats": {
                "coverage": summarize_report_metric(repeat_reports, "macgyver_results", "coverage"),
                "availability": summarize_report_metric(repeat_reports, "macgyver_results", "availability"),
                "imagination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "imagination"),
                "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_dual_axis", "hallucination"),
            },
        })

    if isinstance(final_report.get("macgyver_boundary_results"), dict):
        boundary_details = []
        for report in repeat_reports:
            section = report.get("macgyver_boundary_results")
            if not isinstance(section, dict):
                continue
            for detail in section.get("details") or []:
                boundary_details.append(copy.deepcopy(detail))
        final_report["macgyver_boundary_results"].update({
            "details": boundary_details,
            "trials": len(boundary_details),
            "scorable_trials": summarize_report_metric(repeat_reports, "macgyver_boundary_results", "scorable_trials")["mean"],
            "coverage": summarize_report_metric(repeat_reports, "macgyver_boundary_results", "coverage")["mean"],
            "availability": summarize_report_metric(repeat_reports, "macgyver_boundary_results", "availability")["mean"],
            "replicate_stats": {
                "coverage": summarize_report_metric(repeat_reports, "macgyver_boundary_results", "coverage"),
                "availability": summarize_report_metric(repeat_reports, "macgyver_boundary_results", "availability"),
                "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "macgyver_boundary_diagnostic", "hallucination"),
            },
        })

    if isinstance(final_report.get("cjst_results"), dict):
        cjst_details = []
        for report in repeat_reports:
            section = report.get("cjst_results")
            if not isinstance(section, dict):
                continue
            for detail in section.get("details") or []:
                cjst_details.append(copy.deepcopy(detail))
        final_report["cjst_results"].update({
            "details": cjst_details,
            "trials": len(cjst_details),
            "scorable_trials": summarize_report_metric(repeat_reports, "cjst_results", "scorable_trials")["mean"],
            "coverage": summarize_report_metric(repeat_reports, "cjst_results", "coverage")["mean"],
            "availability": summarize_report_metric(repeat_reports, "cjst_results", "availability")["mean"],
            "replicate_stats": {
                "coverage": summarize_report_metric(repeat_reports, "cjst_results", "coverage"),
                "availability": summarize_report_metric(repeat_reports, "cjst_results", "availability"),
                "imagination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "imagination"),
                "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "cjst_dual_axis", "hallucination"),
            },
        })

    if isinstance(final_report.get("hypospace_results"), dict):
        hypospace_details = []
        for report in repeat_reports:
            section = report.get("hypospace_results")
            if not isinstance(section, dict):
                continue
            for detail in section.get("details") or []:
                hypospace_details.append(copy.deepcopy(detail))
        final_report["hypospace_results"].update({
            "details": hypospace_details,
            "trials": len(hypospace_details),
            "scorable_trials": summarize_report_metric(repeat_reports, "hypospace_results", "scorable_trials")["mean"],
            "coverage": summarize_report_metric(repeat_reports, "hypospace_results", "coverage")["mean"],
            "availability": summarize_report_metric(repeat_reports, "hypospace_results", "availability")["mean"],
            "replicate_stats": {
                "coverage": summarize_report_metric(repeat_reports, "hypospace_results", "coverage"),
                "availability": summarize_report_metric(repeat_reports, "hypospace_results", "availability"),
                "imagination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "imagination"),
                "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_dual_axis", "hallucination"),
            },
        })

    if isinstance(final_report.get("hypospace_boundary_results"), dict):
        hypospace_boundary_details = []
        for report in repeat_reports:
            section = report.get("hypospace_boundary_results")
            if not isinstance(section, dict):
                continue
            for detail in section.get("details") or []:
                hypospace_boundary_details.append(copy.deepcopy(detail))
        final_report["hypospace_boundary_results"].update({
            "details": hypospace_boundary_details,
            "trials": len(hypospace_boundary_details),
            "scorable_trials": summarize_report_metric(repeat_reports, "hypospace_boundary_results", "scorable_trials")["mean"],
            "coverage": summarize_report_metric(repeat_reports, "hypospace_boundary_results", "coverage")["mean"],
            "availability": summarize_report_metric(repeat_reports, "hypospace_boundary_results", "availability")["mean"],
            "replicate_stats": {
                "coverage": summarize_report_metric(repeat_reports, "hypospace_boundary_results", "coverage"),
                "availability": summarize_report_metric(repeat_reports, "hypospace_boundary_results", "availability"),
                "boundary_accuracy": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_boundary_diagnostic", "boundary_accuracy"),
                "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "hypospace_boundary_diagnostic", "hallucination"),
            },
        })

    if isinstance(final_report.get("gcw_results"), dict):
        gcw_details = []
        for report in repeat_reports:
            section = report.get("gcw_results")
            if not isinstance(section, dict):
                continue
            for detail in section.get("details") or []:
                gcw_details.append(copy.deepcopy(detail))
        final_report["gcw_results"].update({
            "details": gcw_details,
            "trials": len(gcw_details),
            "scorable_trials": summarize_report_metric(repeat_reports, "gcw_results", "scorable_trials")["mean"],
            "coverage": summarize_report_metric(repeat_reports, "gcw_results", "coverage")["mean"],
            "availability": summarize_report_metric(repeat_reports, "gcw_results", "availability")["mean"],
            "replicate_stats": {
                "coverage": summarize_report_metric(repeat_reports, "gcw_results", "coverage"),
                "availability": summarize_report_metric(repeat_reports, "gcw_results", "availability"),
                "imagination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "gcw_dual_axis", "imagination"),
                "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "gcw_dual_axis", "hallucination"),
            },
        })

    if isinstance(final_report.get("neocoder_results"), dict):
        neocoder_details = []
        for report in repeat_reports:
            section = report.get("neocoder_results")
            if not isinstance(section, dict):
                continue
            for detail in section.get("details") or []:
                neocoder_details.append(copy.deepcopy(detail))
        final_report["neocoder_results"].update({
            "details": neocoder_details,
            "trials": len(neocoder_details),
            "scorable_trials": summarize_report_metric(repeat_reports, "neocoder_results", "scorable_trials")["mean"],
            "coverage": summarize_report_metric(repeat_reports, "neocoder_results", "coverage")["mean"],
            "availability": summarize_report_metric(repeat_reports, "neocoder_results", "availability")["mean"],
            "replicate_stats": {
                "coverage": summarize_report_metric(repeat_reports, "neocoder_results", "coverage"),
                "availability": summarize_report_metric(repeat_reports, "neocoder_results", "availability"),
                "imagination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "neocoder_dual_axis", "imagination"),
                "hallucination": summarize_report_metric(repeat_reports, "overall_summary", "axes", "neocoder_dual_axis", "hallucination"),
            },
        })

    if isinstance(final_report.get("closed_world_fact_results"), dict):
        closed_world_fact_details = []
        for report in repeat_reports:
            section = report.get("closed_world_fact_results")
            if not isinstance(section, dict):
                continue
            for detail in section.get("details") or []:
                closed_world_fact_details.append(copy.deepcopy(detail))
        final_report["closed_world_fact_results"].update({
            "details": closed_world_fact_details,
            "trials": len(closed_world_fact_details),
            "scorable_trials": summarize_report_metric(
                repeat_reports,
                "closed_world_fact_results", "scorable_trials",
            )["mean"],
            "coverage": summarize_report_metric(
                repeat_reports,
                "closed_world_fact_results", "coverage",
            )["mean"],
            "availability": summarize_report_metric(
                repeat_reports,
                "closed_world_fact_results", "availability",
            )["mean"],
            "replicate_stats": {
                "coverage": summarize_report_metric(repeat_reports, "closed_world_fact_results", "coverage"),
                "availability": summarize_report_metric(repeat_reports, "closed_world_fact_results", "availability"),
                "score": summarize_report_metric(
                    repeat_reports,
                    "overall_summary", "axes", "closed_world_fact_calibration", "score",
                ),
                "hallucination": summarize_report_metric(
                    repeat_reports,
                    "overall_summary", "axes", "closed_world_fact_calibration", "hallucination",
                ),
            },
        })

    if isinstance(final_report.get("analogy_transfer_results"), dict):
        analogy_transfer_details = []
        for report in repeat_reports:
            section = report.get("analogy_transfer_results")
            if not isinstance(section, dict):
                continue
            for detail in section.get("details") or []:
                analogy_transfer_details.append(copy.deepcopy(detail))
        final_report["analogy_transfer_results"].update({
            "details": analogy_transfer_details,
            "trials": len(analogy_transfer_details),
            "scorable_trials": summarize_report_metric(
                repeat_reports,
                "analogy_transfer_results", "scorable_trials",
            )["mean"],
            "coverage": summarize_report_metric(
                repeat_reports,
                "analogy_transfer_results", "coverage",
            )["mean"],
            "availability": summarize_report_metric(
                repeat_reports,
                "analogy_transfer_results", "availability",
            )["mean"],
            "replicate_stats": {
                "coverage": summarize_report_metric(repeat_reports, "analogy_transfer_results", "coverage"),
                "availability": summarize_report_metric(repeat_reports, "analogy_transfer_results", "availability"),
                "imagination": summarize_report_metric(
                    repeat_reports,
                    "overall_summary", "axes", "analogy_transfer_challenge", "imagination",
                ),
                "hallucination": summarize_report_metric(
                    repeat_reports,
                    "overall_summary", "axes", "analogy_transfer_challenge", "hallucination",
                ),
            },
        })

    final_report["task_results"] = collect_repeat_task_results(repeat_reports)
    final_report["repeat_aggregation"] = {
        "aggregation_unit": "repeat",
        "score_policy": "Model-level axis scores are means over eligible repeat-level axis scores, not pooled item-level means.",
        "detail_policy": "task_results and DAT/CDAT/FF details are flattened across all repeats and retain repeat_index.",
        "repeat_level_scores": build_repeat_level_summary(repeat_reports),
    }

    final_report["sampling_summary"] = {
        "requested_repeats": MODEL_SAMPLE_REPEATS,
        "minimum_eligible_repeats": min_required,
        "repeat_eligible_fraction": REPEAT_ELIGIBLE_FRACTION,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed_base": SAMPLING_SEED_BASE,
        "exemplar_repeat_index": exemplar_report.get("repeat_index"),
        "repeat_summaries": repeat_summaries,
    }
    final_report.setdefault("data_sources", {})["scoring_configuration"] = scoring_configuration

    final_report["runtime_policy"]["sampling"] = {
        "requested_repeats": MODEL_SAMPLE_REPEATS,
        "minimum_eligible_repeats": min_required,
        "repeat_eligible_fraction": REPEAT_ELIGIBLE_FRACTION,
        "seed_base": SAMPLING_SEED_BASE,
    }
    if model_catalog_entry is not None:
        final_report["openrouter_model_info"] = {
            "id": model_catalog_entry.get("id") if model_catalog_entry else model_name,
            "name": model_catalog_entry.get("name") if model_catalog_entry else model_name,
            "context_length": model_catalog_entry.get("context_length") if model_catalog_entry else None,
            "pricing": model_catalog_entry.get("pricing") if model_catalog_entry else None,
            "supported_parameters": model_catalog_entry.get("supported_parameters") if model_catalog_entry else None,
        }

    print("\n" + "=" * 60)
    print(f"  AGGREGATED MODEL REPORT: {model_name}")
    print("=" * 60)
    print(
        f"  Eligible repeats: DT={dt_stats['n']}/{MODEL_SAMPLE_REPEATS}, "
        f"Ground={groundedness_stats['n']}/{MODEL_SAMPLE_REPEATS}, "
        f"Imag={imagination_stats['n']}/{MODEL_SAMPLE_REPEATS}, "
        f"Halluc={hallucination_stats['n']}/{MODEL_SAMPLE_REPEATS} (need {min_required})"
    )
    print(f"  Primary dual-axis ranking eligible: {'YES' if final_ranking_eligible else 'NO'}")
    if eligibility_failures:
        print(f"  Eligibility issues: {', '.join(eligibility_failures)}")
    if dt_stats["mean"] is not None:
        print(f"  DT Total audit mean±std: {dt_stats['mean']:.4f} ± {dt_stats['std']:.4f}  [CI {dt_stats['ci_low']:.4f}, {dt_stats['ci_high']:.4f}]")
    if novelty_stats["mean"] is not None:
        print(f"  Novelty mean±std: {novelty_stats['mean']:.4f} ± {novelty_stats['std']:.4f}")
    if flexibility_stats["mean"] is not None:
        print(f"  Flex mean±std:    {flexibility_stats['mean']:.4f} ± {flexibility_stats['std']:.4f}")
    if groundedness_stats["mean"] is not None:
        print(f"  Ground mean±std:  {groundedness_stats['mean']:.4f} ± {groundedness_stats['std']:.4f}")
    if imagination_stats["mean"] is not None:
        print(f"  Imagin mean±std:  {imagination_stats['mean']:.4f} ± {imagination_stats['std']:.4f}")
    if hallucination_stats["mean"] is not None:
        print(f"  Halluc mean±std:  {hallucination_stats['mean']:.4f} ± {hallucination_stats['std']:.4f}")
    print("=" * 60)

    return final_report

def run_model_assessment(client, model_name, dataset, scorer, cog_baseline,
                         wn_analyzer, groundedness_scorer, dat_scorer_obj=None,
                         ff_scorer_obj=None, model_catalog_entry=None):
    repeat_reports = []
    for replicate_index in range(MODEL_SAMPLE_REPEATS):
        repeat_report = run_model_assessment_once(
            client=client,
            model_name=model_name,
            dataset=dataset,
            scorer=scorer,
            cog_baseline=cog_baseline,
            wn_analyzer=wn_analyzer,
            groundedness_scorer=groundedness_scorer,
            dat_scorer_obj=dat_scorer_obj,
            ff_scorer_obj=ff_scorer_obj,
            model_catalog_entry=model_catalog_entry,
            replicate_index=replicate_index,
        )
        repeat_reports.append(repeat_report)
    return aggregate_model_assessment_reports(
        model_name=model_name,
        repeat_reports=repeat_reports,
        model_catalog_entry=model_catalog_entry,
    )

__all__ = [
    'get_non_model_skip_reason',
    'select_valid_dat_words',
    'get_nested_value',
    'aggregate_nested_numeric_dicts',
    'summarize_report_metric',
    'summarize_repeat_report',
    'format_percent_or_na',
    'collect_repeat_task_results',
    'collect_repeat_section_details',
    'build_repeat_level_summary',
    'clip01',
    'compute_uut_dual_axis_diversity',
    'compute_uut_dual_axis_task_scores',
    'PropConjScorer',
    'aggregate_propconj_model_axes',
    'compute_propconj_diversity',
    'compute_propconj_task_scores',
    'MacGyverScorer',
    'aggregate_macgyver_model_axes',
    'CounterfactualScorer',
    'aggregate_cjst_model_axes',
    'GroundedCreativeWritingScorer',
    'aggregate_gcw_model_axes',
    'HypoUseSpaceScorer',
    'aggregate_hypospace_model_axes',
    'aggregate_hypospace_boundary_diagnostics',
    'NeoCoderScorer',
    'aggregate_neocoder_model_axes',
    'ClosedWorldFactScorer',
    'aggregate_closed_world_fact_calibration_axes',
    'AnalogyTransferScorer',
    'aggregate_analogy_transfer_challenge_axes',
    'run_model_assessment_once',
    'aggregate_model_assessment_reports',
    'run_model_assessment',
]
