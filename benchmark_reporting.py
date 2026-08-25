
import json
import math
import os
import sys

from groundedness_scorer import WHITE_BOX_GROUNDEDNESS_VERSION
from benchmark_core import *

try:
    import numpy as np
except ImportError:
    np = None

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = np is not None
except ImportError:
    plt = None
    _HAS_MATPLOTLIB = False

def strip_legacy_conceptnet_fields(task_result):
    if not isinstance(task_result, dict):
        return task_result

    task_result.pop("average_insightfulness", None)

    for detail in task_result.get("details", []):
        if not isinstance(detail, dict):
            continue
        for key in list(detail.keys()):
            if key == "insightfulness_score" or key.startswith("cn_"):
                detail.pop(key, None)
    return task_result

def report_has_white_box_groundedness(report):
    schema_version = report.get("scoring_schema", {}).get("report_schema_version")
    if schema_version == WHITE_BOX_REPORT_SCHEMA_VERSION:
        return True

    creative_results = [
        task_result
        for task_result in report.get("task_results", [])
        if task_result.get("task_type") in CREATIVE_TASK_TYPES and task_result.get("valid_run")
    ]
    if not creative_results:
        return False

    saw_white_box_detail = False
    for task_result in creative_results:
        for detail in task_result.get("details", []):
            if not isinstance(detail, dict):
                continue
            if "groundedness_score" in detail and "groundedness_penalty" in detail:
                saw_white_box_detail = True
            if "insightfulness_score" in detail or any(key.startswith("cn_") for key in detail):
                return False
    return saw_white_box_detail

def report_matches_current_schema(report):
    if report.get("scoring_schema", {}).get("report_schema_version") != WHITE_BOX_REPORT_SCHEMA_VERSION:
        return False
    axes = report.get("overall_summary", {}).get("axes", {})
    for axis_name in ("imagination", "hallucination", "dual_axis"):
        axis = axes.get(axis_name)
        if not isinstance(axis, dict) or axis.get("score") is None:
            return False
    imagination_tasks = axes.get("imagination", {}).get("task_type_scores") or {}
    hallucination_tasks = axes.get("hallucination", {}).get("task_type_scores") or {}
    for component in PRIMARY_DUAL_AXIS_COMPONENTS:
        if imagination_tasks.get(component) is None or hallucination_tasks.get(component) is None:
            return False
    validity = report.get("run_validity", {})
    if validity.get("primary_axis") not in {None, "dual_axis"}:
        return False
    return True

def report_has_primary_dual_axis(report):
    return report_matches_current_schema(report)

def summarize_saved_task_groundedness(task_result):
    strip_legacy_conceptnet_fields(task_result)
    details = task_result.get("details", [])
    task_type = task_result.get("task_type")
    threshold = None
    scored = []
    novel = []

    for detail in details:
        groundedness = detail.get("groundedness_score")
        if groundedness is None:
            continue
        scored.append(detail)
        raw_originality = detail.get("raw_originality")
        if raw_originality is None:
            raw_originality = detail.get("score")
        raw_originality = raw_originality or 0.0
        if raw_originality > 0:
            novel.append(detail)
        if threshold is None:
            threshold = detail.get("task_threshold")

    groundedness_values = [detail.get("groundedness_score") for detail in scored]
    groundedness_novel_values = [detail.get("groundedness_score") for detail in novel]
    penalty_values = [detail.get("groundedness_penalty", 0.0) for detail in scored]
    confidence_values = [detail.get("groundedness_confidence", 0.0) for detail in scored]
    confidence_values_novel = [detail.get("groundedness_confidence", 0.0) for detail in novel]

    if threshold is None and task_type in CREATIVE_TASK_TYPES:
        threshold = {"UUT": 0.55, "PropConj": 0.70}.get(task_type)

    low_count = 0
    low_count_novel = 0
    if threshold is not None:
        low_count = sum(1 for value in groundedness_values if value < threshold)
        low_count_novel = sum(1 for value in groundedness_novel_values if value < threshold)

    mean_groundedness = mean_or_none(groundedness_values)
    mean_groundedness_novel = mean_or_none(groundedness_novel_values)
    mean_penalty = mean_or_none(penalty_values)
    mean_confidence = mean_or_none(confidence_values)
    penalty_rate = (
        sum(1 for value in penalty_values if value and value > 0) / len(penalty_values)
        if penalty_values else None
    )
    low_groundedness_rate = (
        low_count / len(groundedness_values)
        if groundedness_values and threshold is not None else None
    )
    low_groundedness_rate_novel = (
        low_count_novel / len(groundedness_novel_values)
        if groundedness_novel_values and threshold is not None else None
    )
    scored_coverage = len(scored) / len(details) if details else None

    task_result["groundedness"] = {
        "version": WHITE_BOX_GROUNDEDNESS_VERSION,
        "formula": WHITE_BOX_GROUNDEDNESS_VERSION,
        "average_groundedness": round(mean_groundedness, 4) if mean_groundedness is not None else None,
        "average_groundedness_novel_only": (
            round(mean_groundedness_novel, 4)
            if mean_groundedness_novel is not None else None
        ),
        "mean_penalty": round(mean_penalty, 4) if mean_penalty is not None else None,
        "penalty_rate": round(penalty_rate, 4) if penalty_rate is not None else None,
        "groundedness_confidence_mean": (
            round(mean_confidence, 4) if mean_confidence is not None else None
        ),
        "low_groundedness_rate": (
            round(low_groundedness_rate, 4)
            if low_groundedness_rate is not None else None
        ),
        "low_groundedness_rate_novel_only": (
            round(low_groundedness_rate_novel, 4)
            if low_groundedness_rate_novel is not None else None
        ),
        "scored_coverage": round(scored_coverage, 4) if scored_coverage is not None else None,
        "groundedness_scored_ideas": len(scored),
        "threshold": threshold,
    }

    return {
        "avg_groundedness": mean_groundedness,
        "avg_groundedness_novel": mean_groundedness_novel,
        "penalties": penalty_values,
        "confidences": confidence_values,
        "confidences_novel": confidence_values_novel,
        "groundedness_values": groundedness_values,
        "groundedness_novel_values": groundedness_novel_values,
        "low_count": low_count,
        "low_count_novel": low_count_novel,
        "threshold": threshold,
        "scored_count": len(scored),
        "detail_count": len(details),
    }

def recompute_saved_report_groundedness(report, _unused=None):
    for task_result in report.get("task_results", []):
        strip_legacy_conceptnet_fields(task_result)

    if isinstance(report.get("data_sources"), dict):
        report["data_sources"].pop("conceptnet_validator", None)

    task_type_scores = {task_type: [] for task_type in CREATIVE_TASK_TYPES}
    task_type_novel_scores = {task_type: [] for task_type in CREATIVE_TASK_TYPES}
    all_penalties = []
    all_confidences = []
    all_confidences_novel = []
    all_groundedness = []
    all_groundedness_novel = []
    low_total = 0
    low_total_novel = 0
    scored_count = 0
    detail_count = 0

    for task_result in report.get("task_results", []):
        if not task_result.get("valid_run"):
            continue
        task_type = task_result.get("task_type")
        if task_type not in CREATIVE_TASK_TYPES:
            continue

        summary = summarize_saved_task_groundedness(task_result)
        if summary["avg_groundedness"] is not None:
            task_type_scores[task_type].append(summary["avg_groundedness"])
        if summary["avg_groundedness_novel"] is not None:
            task_type_novel_scores[task_type].append(summary["avg_groundedness_novel"])

        all_penalties.extend(summary["penalties"])
        all_confidences.extend(summary["confidences"])
        all_confidences_novel.extend(summary["confidences_novel"])
        all_groundedness.extend(summary["groundedness_values"])
        all_groundedness_novel.extend(summary["groundedness_novel_values"])
        low_total += summary["low_count"]
        low_total_novel += summary["low_count_novel"]
        scored_count += summary["scored_count"]
        detail_count += summary["detail_count"]

    task_type_score_summary = {
        task_type: round(mean_or_none(values), 4)
        for task_type, values in task_type_scores.items()
        if mean_or_none(values) is not None
    }
    task_type_novel_score_summary = {
        task_type: round(mean_or_none(values), 4)
        for task_type, values in task_type_novel_scores.items()
        if mean_or_none(values) is not None
    }

    groundedness_score = mean_or_none(list(task_type_score_summary.values()))
    groundedness_score_novel = mean_or_none(list(task_type_novel_score_summary.values()))
    mean_penalty = mean_or_none(all_penalties)
    mean_confidence = mean_or_none(all_confidences)
    penalty_rate = (
        sum(1 for value in all_penalties if value and value > 0) / len(all_penalties)
        if all_penalties else None
    )
    low_groundedness_rate = low_total / len(all_groundedness) if all_groundedness else None
    low_groundedness_rate_novel = (
        low_total_novel / len(all_groundedness_novel)
        if all_groundedness_novel else None
    )
    scored_coverage = scored_count / detail_count if detail_count > 0 else None
    confidence_weighted = (
        sum(value * confidence for value, confidence in zip(all_groundedness, all_confidences)) / sum(all_confidences)
        if all_confidences and sum(all_confidences) > 0 else groundedness_score
    )
    confidence_weighted_novel = (
        sum(value * confidence for value, confidence in zip(all_groundedness_novel, all_confidences_novel)) / sum(all_confidences_novel)
        if all_groundedness_novel and sum(all_confidences_novel) > 0 else groundedness_score_novel
    )

    axes = report.setdefault("overall_summary", {}).setdefault("axes", {})
    axes["groundedness"] = {
        "version": WHITE_BOX_GROUNDEDNESS_VERSION,
        "score": round(groundedness_score, 4) if groundedness_score is not None else None,
        "score_novel_only": round(groundedness_score_novel, 4) if groundedness_score_novel is not None else None,
        "formula": "mean_tasktype_groundedness_v1",
        "confidence_weighted_mean": (
            round(confidence_weighted, 4) if confidence_weighted is not None else None
        ),
        "confidence_weighted_mean_novel_only": (
            round(confidence_weighted_novel, 4)
            if confidence_weighted_novel is not None else None
        ),
        "mean_penalty": round(mean_penalty, 4) if mean_penalty is not None else None,
        "penalty_rate": round(penalty_rate, 4) if penalty_rate is not None else None,
        "low_groundedness_rate": (
            round(low_groundedness_rate, 4)
            if low_groundedness_rate is not None else None
        ),
        "low_groundedness_rate_novel_only": (
            round(low_groundedness_rate_novel, 4)
            if low_groundedness_rate_novel is not None else None
        ),
        "mean_confidence": round(mean_confidence, 4) if mean_confidence is not None else None,
        "scored_coverage": round(scored_coverage, 4) if scored_coverage is not None else None,
        "groundedness_scored_ideas": scored_count,
        "task_type_scores": task_type_score_summary,
        "task_type_scores_novel_only": task_type_novel_score_summary,
    }

    report["scoring_schema"] = {
        "report_schema_version": WHITE_BOX_REPORT_SCHEMA_VERSION,
        "primary_score": "Imagination and hallucination are the primary benchmark axes. DT total, novelty, flexibility, and groundedness are supporting scorer outputs.",
        "typed_correlation_ready": isinstance(axes.get("subtype_scores"), dict),
        "subtype_schema_version": (axes.get("subtype_scores") or {}).get("version") if isinstance(axes.get("subtype_scores"), dict) else None,
        "creative_novelty": "supporting scorer output: UUT/PropConj novelty uses hybrid prompt-distance/common-answer rarity after white-box grounding/property checks; DAT/CDAT are auxiliary diagnostics",
        "groundedness": "audit primitive only; ConceptNet hop removed from runtime",
        "uut_dual_axis": "UUT-Affordance v1 reports white-box imagination and hallucination axes when current-schema UUT primitives are present",
        "propconj_dual_axis": "PropConj v1 reports property-conjunction imagination and hallucination axes when current-schema PropConj details are present",
        "macgyver_dual_axis": "MacGyver T2-v3 reports discriminative closed-tool plan imagination with common-plan rarity, hard feasibility gates, and separate boundary diagnostics when current-schema MacGyver details are present",
        "cjst_dual_axis": "CJST v1 reports structured counterfactual imagination and hallucination axes when current-schema CJST details are present",
        "hypospace_dual_axis": "HypoUseSpace v1 reports finite closed-world hypothesis recovery imagination and hallucination axes when current-schema HypoUseSpace details are present",
        "gcw_dual_axis": "GCW v1 reports fact-grounded microfiction imagination and hallucination axes when current-schema GCW details are present",
        "closed_world_fact_calibration": "ClosedWorldFact reports optional calibration-only closed-world factual hallucination checks when present",
        "analogy_transfer_challenge": "AnalogyTransfer reports default-off challenge-only analogy imagination and false-transfer hallucination checks when present",
    }

    novelty = axes.get("novelty", {}).get("score")
    flexibility = axes.get("flexibility", {}).get("score")
    if novelty is not None and flexibility is not None:
        dt_total = DT_TOTAL_NOVELTY_WEIGHT * novelty + DT_TOTAL_FLEXIBILITY_WEIGHT * flexibility
        axes["dt_total"] = {
            "score": round(dt_total, 4),
            "formula": f"{DT_TOTAL_NOVELTY_WEIGHT:.2f}*Novelty + {DT_TOTAL_FLEXIBILITY_WEIGHT:.2f}*Flexibility",
        }
    else:
        axes["dt_total"] = {
            "score": None,
            "formula": f"{DT_TOTAL_NOVELTY_WEIGHT:.2f}*Novelty + {DT_TOTAL_FLEXIBILITY_WEIGHT:.2f}*Flexibility",
        }

    return report

def extract_comparison_rows(model_reports):
    rows = [
        ("Imagination", lambda report: report.get("overall_summary", {}).get("axes", {}).get("imagination", {}).get("score"), "higher"),
        ("Hallucination", lambda report: report.get("overall_summary", {}).get("axes", {}).get("hallucination", {}).get("score"), "lower"),
        ("I Raw", lambda report: report.get("overall_summary", {}).get("axes", {}).get("imagination", {}).get("raw_score"), "higher"),
        ("H Raw", lambda report: report.get("overall_summary", {}).get("axes", {}).get("hallucination", {}).get("raw_score"), "lower"),
        ("DT Total", lambda report: report.get("overall_summary", {}).get("axes", {}).get("dt_total", {}).get("score"), "higher"),
        ("Novelty", lambda report: report.get("overall_summary", {}).get("axes", {}).get("novelty", {}).get("score"), "higher"),
        ("Flexibility", lambda report: report.get("overall_summary", {}).get("axes", {}).get("flexibility", {}).get("score"), "higher"),
        ("Groundedness", lambda report: report.get("overall_summary", {}).get("axes", {}).get("groundedness", {}).get("score"), "higher"),
        ("Novel-only Ground", lambda report: extract_groundedness_metric(report, "score_novel_only"), "higher"),
        ("Confidence-weighted Ground", lambda report: extract_groundedness_metric(report, "confidence_weighted_mean"), "higher"),
        ("Ground UUT", lambda report: extract_groundedness_task_type_score(report, "UUT"), "higher"),
        ("Ground PropConj", lambda report: extract_groundedness_task_type_score(report, "PropConj"), "higher"),
        ("MacGyver I", lambda report: report.get("overall_summary", {}).get("axes", {}).get("macgyver_dual_axis", {}).get("imagination"), "higher"),
        ("MacGyver H", lambda report: report.get("overall_summary", {}).get("axes", {}).get("macgyver_dual_axis", {}).get("hallucination"), "lower"),
        ("CJST I", lambda report: report.get("overall_summary", {}).get("axes", {}).get("cjst_dual_axis", {}).get("imagination"), "higher"),
        ("CJST H", lambda report: report.get("overall_summary", {}).get("axes", {}).get("cjst_dual_axis", {}).get("hallucination"), "lower"),
        ("HypoUse I", lambda report: report.get("overall_summary", {}).get("axes", {}).get("hypospace_dual_axis", {}).get("imagination"), "higher"),
        ("HypoUse H", lambda report: report.get("overall_summary", {}).get("axes", {}).get("hypospace_dual_axis", {}).get("hallucination"), "lower"),
        ("GCW I", lambda report: report.get("overall_summary", {}).get("axes", {}).get("gcw_dual_axis", {}).get("imagination"), "higher"),
        ("GCW H", lambda report: report.get("overall_summary", {}).get("axes", {}).get("gcw_dual_axis", {}).get("hallucination"), "lower"),
        ("CWF Score", lambda report: report.get("overall_summary", {}).get("axes", {}).get("closed_world_fact_calibration", {}).get("score"), "higher"),
        ("CWF H", lambda report: report.get("overall_summary", {}).get("axes", {}).get("closed_world_fact_calibration", {}).get("hallucination"), "lower"),
        ("Analogy I", lambda report: report.get("overall_summary", {}).get("axes", {}).get("analogy_transfer_challenge", {}).get("imagination"), "higher"),
        ("Analogy H", lambda report: report.get("overall_summary", {}).get("axes", {}).get("analogy_transfer_challenge", {}).get("hallucination"), "lower"),
        ("UUT", lambda report: extract_component_score(report.get("overall_summary", {}).get("axes", {}).get("novelty", {}).get("component_scores", {}), "UUT"), "higher"),
        ("PropConj", lambda report: extract_component_score(report.get("overall_summary", {}).get("axes", {}).get("novelty", {}).get("component_scores", {}), "PropConj"), "higher"),
        ("DAT", lambda report: extract_component_score(report.get("overall_summary", {}).get("axes", {}).get("novelty", {}).get("component_scores", {}), "DAT"), "higher"),
        ("CDAT", lambda report: extract_component_score(report.get("overall_summary", {}).get("axes", {}).get("novelty", {}).get("component_scores", {}), "CDAT"), "higher"),
        ("Embedding Flex", lambda report: report.get("overall_summary", {}).get("axes", {}).get("flexibility", {}).get("embedding_composite"), "higher"),
        ("Ontological Flex", lambda report: report.get("overall_summary", {}).get("axes", {}).get("flexibility", {}).get("ontological_composite"), "higher"),
        ("Mean Penalty", lambda report: extract_groundedness_metric(report, "mean_penalty"), "lower"),
        ("Penalty Rate", lambda report: extract_groundedness_metric(report, "penalty_rate"), "lower"),
        ("Low-ground Rate", lambda report: extract_groundedness_metric(report, "low_groundedness_rate"), "lower"),
        ("Mean Confidence", lambda report: extract_groundedness_metric(report, "mean_confidence"), "higher"),
        ("Runtime (min)", lambda report: (report.get("overall_summary", {}).get("runtime_seconds") or 0.0) / 60.0, "lower"),
    ]

    comparison_rows = []
    for label, getter, direction in rows:
        values = []
        for report in model_reports:
            value = getter(report)
            values.append(float("nan") if value is None else float(value))
        comparison_rows.append({
            "label": label,
            "values": values,
            "direction": direction,
        })
    return comparison_rows

def generate_comparison_chart(model_reports, output_path):
    if not _HAS_MATPLOTLIB:
        print("[Plot] matplotlib or numpy is unavailable; skipping chart generation.")
        return None

    comparison_rows = extract_comparison_rows(model_reports)
    if not comparison_rows:
        return None

    labels = [row["label"] for row in comparison_rows]
    values = np.array([row["values"] for row in comparison_rows], dtype=float)
    normalized = np.zeros_like(values)

    for row_index, row in enumerate(comparison_rows):
        row_values = values[row_index]
        finite_mask = np.isfinite(row_values)
        if not np.any(finite_mask):
            normalized[row_index, :] = np.nan
            continue

        finite_values = row_values[finite_mask]
        min_value = np.min(finite_values)
        max_value = np.max(finite_values)
        if abs(max_value - min_value) < 1e-12:
            normalized[row_index, finite_mask] = 0.5
        elif row["direction"] == "higher":
            normalized[row_index, finite_mask] = (finite_values - min_value) / (max_value - min_value)
        else:
            normalized[row_index, finite_mask] = (max_value - finite_values) / (max_value - min_value)

    fig_width = max(12, len(model_reports) * 2.2)
    fig_height = max(8, len(comparison_rows) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(normalized, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(model_reports)))
    ax.set_xticklabels([report["model_id"] for report in model_reports], rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Multi-Provider Model Comparison\nRow-wise normalized colors, raw values shown in cells")

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isnan(value):
                text = "N/A"
            else:
                text = f"{value:.3f}"
            cell_color = normalized[row_index, col_index]
            text_color = "white" if np.isfinite(cell_color) and cell_color > 0.55 else "black"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Relative performance within each metric row")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path

def build_model_summary_row(report):
    axes = report.get("overall_summary", {}).get("axes", {})
    validity = report.get("run_validity", {})
    scoring_schema = report.get("scoring_schema", {})
    sampling = report.get("sampling_summary", {})
    dt_stats = axes.get("dt_total", {}).get("replicate_stats", {})
    novelty_stats = axes.get("novelty", {}).get("replicate_stats", {})
    flexibility_stats = axes.get("flexibility", {}).get("replicate_stats", {})
    groundedness_stats = axes.get("groundedness", {}).get("replicate_stats", {})
    imagination_stats = axes.get("imagination", {}).get("replicate_stats", {})
    hallucination_stats = axes.get("hallucination", {}).get("replicate_stats", {})
    runtime_stats = report.get("overall_summary", {}).get("runtime_seconds_stats", {})
    repeat_policy = validity.get("repeat_policy", {})
    return {
        "model_id": report.get("model_id"),
        "ranking_eligible": validity.get("ranking_eligible"),
        "typed_correlation_ready": scoring_schema.get("typed_correlation_ready") or validity.get("typed_correlation_ready"),
        "subtype_schema_version": scoring_schema.get("subtype_schema_version") or validity.get("subtype_schema_version"),
        "task_registry_version": scoring_schema.get("task_registry_version"),
        "requested_repeats": sampling.get("requested_repeats", MODEL_SAMPLE_REPEATS),
        "minimum_eligible_repeats": sampling.get("minimum_eligible_repeats", minimum_eligible_repeats(MODEL_SAMPLE_REPEATS)),
        "dt_total_eligible_repeats": repeat_policy.get("dt_total_eligible_repeats", dt_stats.get("n")),
        "groundedness_eligible_repeats": repeat_policy.get("groundedness_eligible_repeats", groundedness_stats.get("n")),
        "imagination_eligible_repeats": repeat_policy.get("imagination_eligible_repeats", imagination_stats.get("n")),
        "hallucination_eligible_repeats": repeat_policy.get("hallucination_eligible_repeats", hallucination_stats.get("n")),
        "uut_eligible_repeats": repeat_policy.get("uut_eligible_repeats"),
        "propconj_eligible_repeats": repeat_policy.get("propconj_eligible_repeats"),
        "macgyver_eligible_repeats": repeat_policy.get("macgyver_eligible_repeats"),
        "cjst_eligible_repeats": repeat_policy.get("cjst_eligible_repeats"),
        "hypospace_eligible_repeats": repeat_policy.get("hypospace_eligible_repeats"),
        "gcw_eligible_repeats": repeat_policy.get("gcw_eligible_repeats"),
        "dt_total": axes.get("dt_total", {}).get("score"),
        "dt_total_std": dt_stats.get("std"),
        "dt_total_ci_low": dt_stats.get("ci_low"),
        "dt_total_ci_high": dt_stats.get("ci_high"),
        "novelty": axes.get("novelty", {}).get("score"),
        "novelty_std": novelty_stats.get("std"),
        "flexibility": axes.get("flexibility", {}).get("score"),
        "flexibility_std": flexibility_stats.get("std"),
        "groundedness": axes.get("groundedness", {}).get("score"),
        "groundedness_std": groundedness_stats.get("std"),
        "imagination": axes.get("imagination", {}).get("score"),
        "imagination_std": imagination_stats.get("std"),
        "imagination_raw": axes.get("imagination", {}).get("raw_score"),
        "imagination_total_pure": axes.get("imagination", {}).get("score_pure"),
        "imagination_pure_rank": axes.get("imagination", {}).get("pure_I_rank"),
        "imagination_raw_subtype_rank": axes.get("imagination", {}).get("raw_I_rank"),
        "pure_minus_raw_rank_delta": axes.get("imagination", {}).get("pure_minus_raw_rank_delta"),
        "hallucination": axes.get("hallucination", {}).get("score"),
        "hallucination_std": hallucination_stats.get("std"),
        "hallucination_raw": axes.get("hallucination", {}).get("raw_score"),
        "creative_coverage": validity.get("creative_total_coverage"),
        "creative_availability": validity.get("creative_total_availability"),
        "dat_coverage": validity.get("dat_coverage"),
        "dat_availability": validity.get("dat_availability"),
        "cdat_coverage": validity.get("cdat_coverage"),
        "cdat_availability": validity.get("cdat_availability"),
        "cjst_coverage": validity.get("cjst_coverage"),
        "cjst_availability": validity.get("cjst_availability"),
        "hypospace_coverage": validity.get("hypospace_coverage"),
        "hypospace_availability": validity.get("hypospace_availability"),
        "gcw_coverage": validity.get("gcw_coverage"),
        "gcw_availability": validity.get("gcw_availability"),
        "closed_world_fact_coverage": validity.get("closed_world_fact_coverage"),
        "closed_world_fact_availability": validity.get("closed_world_fact_availability"),
        "closed_world_fact_score": axes.get("closed_world_fact_calibration", {}).get("score"),
        "closed_world_fact_hallucination": axes.get("closed_world_fact_calibration", {}).get("hallucination"),
        "analogy_transfer_imagination": axes.get("analogy_transfer_challenge", {}).get("imagination"),
        "analogy_transfer_hallucination": axes.get("analogy_transfer_challenge", {}).get("hallucination"),
        "runtime_seconds": report.get("overall_summary", {}).get("runtime_seconds"),
        "runtime_seconds_mean": runtime_stats.get("mean"),
        "runtime_seconds_std": runtime_stats.get("std"),
    }

def rank_models(model_reports, getter, reverse=True):
    sortable = []
    for report in model_reports:
        value = getter(report)
        if value is not None:
            sortable.append((report["model_id"], value))
    sortable.sort(key=lambda item: item[1], reverse=reverse)
    return sortable

def build_combined_report(model_reports, model_errors, model_catalog_index, total_runtime_seconds, chart_path):
    shared_prompt_manifest = None
    if model_reports:
        shared_prompt_manifest = model_reports[0].get("prompt_manifest")
    ranking_eligible_reports = [
        report for report in model_reports
        if report.get("run_validity", {}).get("ranking_eligible")
    ]
    typed_ready_reports = [
        report for report in model_reports
        if (
            report.get("scoring_schema", {}).get("typed_correlation_ready") or
            report.get("run_validity", {}).get("typed_correlation_ready")
        )
    ]

    comparison_summary = {
        "rows": [build_model_summary_row(report) for report in model_reports],
        "rankings": {
            "dt_total": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("dt_total", {}).get("score")),
            "novelty": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("novelty", {}).get("score")),
            "flexibility": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("flexibility", {}).get("score")),
            "groundedness": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("groundedness", {}).get("score")),
            "imagination": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("imagination", {}).get("score")),
            "imagination_pure": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("imagination", {}).get("score_pure")),
            "hallucination": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("hallucination", {}).get("score"), reverse=False),
            "cjst_imagination": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("cjst_dual_axis", {}).get("imagination")),
            "cjst_hallucination": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("cjst_dual_axis", {}).get("hallucination"), reverse=False),
            "hypospace_imagination": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("hypospace_dual_axis", {}).get("imagination")),
            "hypospace_hallucination": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("hypospace_dual_axis", {}).get("hallucination"), reverse=False),
            "gcw_imagination": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("gcw_dual_axis", {}).get("imagination")),
            "gcw_hallucination": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("gcw_dual_axis", {}).get("hallucination"), reverse=False),
            "closed_world_fact_calibration": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("closed_world_fact_calibration", {}).get("score")),
            "closed_world_fact_hallucination": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("closed_world_fact_calibration", {}).get("hallucination"), reverse=False),
            "analogy_transfer_imagination": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("analogy_transfer_challenge", {}).get("imagination")),
            "analogy_transfer_hallucination": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("axes", {}).get("analogy_transfer_challenge", {}).get("hallucination"), reverse=False),
            "runtime_seconds": rank_models(model_reports, lambda report: report.get("overall_summary", {}).get("runtime_seconds"), reverse=False),
        },
    }

    manifest_task_ids = get_profile_manifest_task_ids()
    return {
        "experiment_profile": OPENROUTER_EXPERIMENT_PROFILE,
        "primary_dual_axis_components": list(PRIMARY_DUAL_AXIS_COMPONENTS),
        "profile_task_manifest_path": PROFILE_TASK_MANIFEST_PATH or None,
        "macgyver_scoring": {
            "version": MACGYVER_DUAL_AXIS_VERSION,
            "calibration_policy": "benchmark_default",
            "runtime_scoring_policy": "fixed output-only parameters",
            "primary_task_ids": manifest_task_ids.get("MacGyver", []),
            "boundary_diagnostic_task_ids": manifest_task_ids.get("MacGyverBoundary", list(MACGYVER_BOUNDARY_DIAGNOSTIC_TASK_IDS)),
            "plan_count": MACGYVER_OUTPUT_COUNT,
        },
        "openrouter": {
            "base_url": OPENROUTER_BASE_URL,
            "requested_models": SELECTED_MODELS,
            "http_referer": OPENROUTER_HTTP_REFERER,
            "app_title": OPENROUTER_APP_TITLE,
            "catalog_snapshot": {
                model_id: {
                    "name": model_catalog_index[model_id].get("name"),
                    "context_length": model_catalog_index[model_id].get("context_length"),
                    "pricing": model_catalog_index[model_id].get("pricing"),
                }
                for model_id in SELECTED_MODELS if model_id in model_catalog_index
            },
        },
        "sampling_policy": {
            "requested_repeats": MODEL_SAMPLE_REPEATS,
            "minimum_eligible_repeats": minimum_eligible_repeats(MODEL_SAMPLE_REPEATS),
            "repeat_eligible_fraction": REPEAT_ELIGIBLE_FRACTION,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "seed_base": SAMPLING_SEED_BASE,
        },
        "runtime": {
            "total_seconds": round(total_runtime_seconds, 2),
            "model_count": len(model_reports),
            "ranking_eligible_count": len(ranking_eligible_reports),
            "typed_correlation_ready_count": len(typed_ready_reports),
        },
        "prompt_manifest": shared_prompt_manifest,
        "comparison_summary": comparison_summary,
        "model_reports": model_reports,
        "model_errors": model_errors,
        "artifacts": {
            "combined_json": MULTI_MODEL_REPORT_JSON,
            "combined_markdown": MULTI_MODEL_REPORT_MD,
            "comparison_chart": chart_path,
            "typed_vectors_by_model": os.path.join(REPORTS_DIR, "typed_vectors_by_model.csv"),
            "typed_correlation_matrix_raw": os.path.join(REPORTS_DIR, "typed_correlation_matrix_raw.csv"),
            "typed_correlation_matrix_gated": os.path.join(REPORTS_DIR, "typed_correlation_matrix_gated.csv"),
            "typed_correlation_matrix_residual": os.path.join(REPORTS_DIR, "typed_correlation_matrix_residual.csv"),
            "typed_correlation_pair_coverage": os.path.join(REPORTS_DIR, "typed_correlation_pair_coverage.csv"),
            "typed_correlation_fdr": os.path.join(REPORTS_DIR, "typed_correlation_fdr.csv"),
            "typed_correlation_partial": os.path.join(REPORTS_DIR, "typed_correlation_partial.csv"),
            "typed_atom_signals_by_model": os.path.join(REPORTS_DIR, "typed_atom_signals_by_model.csv"),
            "imagination_purification_by_model": os.path.join(REPORTS_DIR, "imagination_purification_by_model.csv"),
            "per_model_json": [
                os.path.join(REPORTS_DIR, f"{sanitize_filename(report['model_id'])}_report.json")
                for report in model_reports
            ],
        },
    }

def build_markdown_report(combined_report):
    def format_stat(mean_value, std_value=None, ci_low=None, ci_high=None, digits=4):
        if mean_value is None:
            return "N/A"
        text = f"{mean_value:.{digits}f}"
        if std_value is not None:
            text += f" ± {std_value:.{digits}f}"
        if ci_low is not None and ci_high is not None:
            text += f" [{ci_low:.{digits}f}, {ci_high:.{digits}f}]"
        return text

    def format_ratio(coverage, availability):
        if coverage is None and availability is None:
            return "N/A"
        cov_text = f"{coverage:.0%}" if coverage is not None else "N/A"
        avail_text = f"{availability:.0%}" if availability is not None else "N/A"
        return f"{cov_text} / {avail_text}"

    lines = []
    lines.append("# Multi-Provider Imagination/Hallucination Benchmark")
    lines.append("")
    lines.append(f"Total runtime: {combined_report['runtime']['total_seconds']:.2f}s")
    sampling_policy = combined_report.get("sampling_policy", {})
    lines.append(
        "Sampling policy: "
        f"{sampling_policy.get('requested_repeats', MODEL_SAMPLE_REPEATS)} repeats per model, "
        f"minimum eligible repeats = {sampling_policy.get('minimum_eligible_repeats', minimum_eligible_repeats(MODEL_SAMPLE_REPEATS))}, "
        f"bootstrap samples = {sampling_policy.get('bootstrap_samples', BOOTSTRAP_SAMPLES)}"
    )
    lines.append(
        "Typed correlation readiness: "
        f"{combined_report.get('runtime', {}).get('typed_correlation_ready_count', 0)}/"
        f"{combined_report.get('runtime', {}).get('model_count', 0)} per-model reports expose subtype vectors."
    )
    lines.append("")
    lines.append("## Models")
    for model_id in combined_report["openrouter"]["requested_models"]:
        lines.append(f"- {model_id}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("Coverage/Availability is shown as output-valid coverage over eligible prompts / prompt availability after infra-harness exclusions.")
    lines.append("")
    lines.append("| Model | Eligible | Repeats | Creative | CJST | HypoUse | GCW | Imagination | Hallucination | Runtime (s) |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|")
    for row in combined_report["comparison_summary"]["rows"]:
        eligible_text = "Yes" if row.get("ranking_eligible") else "No"
        repeats_text = (
            f"{row.get('imagination_eligible_repeats', 0)}/{row.get('requested_repeats', MODEL_SAMPLE_REPEATS)}"
        )
        creative_text = format_ratio(row.get("creative_coverage"), row.get("creative_availability"))
        cjst_text = format_ratio(row.get("cjst_coverage"), row.get("cjst_availability"))
        hypospace_text = format_ratio(row.get("hypospace_coverage"), row.get("hypospace_availability"))
        gcw_text = format_ratio(row.get("gcw_coverage"), row.get("gcw_availability"))
        imagination_text = format_stat(row.get("imagination"), row.get("imagination_std"))
        hallucination_text = format_stat(row.get("hallucination"), row.get("hallucination_std"))
        runtime_text = format_stat(row.get("runtime_seconds_mean"), row.get("runtime_seconds_std"), digits=2)
        lines.append(
            f"| {row['model_id']} | {eligible_text} | {repeats_text} | {creative_text} | {cjst_text} | {hypospace_text} | {gcw_text} | "
            f"{imagination_text} | {hallucination_text} | {runtime_text} |"
        )
    lines.append("")
    lines.append("## Rankings")
    for axis_name, ranking in combined_report["comparison_summary"]["rankings"].items():
        lines.append(f"### {axis_name.replace('_', ' ').title()}")
        if ranking:
            for model_id, value in ranking:
                lines.append(f"- {model_id}: {value:.4f}")
        else:
            lines.append("- No valid results")
        lines.append("")
    lines.append("## Per-model Results")
    for report in combined_report.get("model_reports", []):
        lines.append(f"### {report.get('model_id')}")
        sampling = report.get("sampling_summary", {})
        validity = report.get("run_validity", {})
        axes = report.get("overall_summary", {}).get("axes", {})
        repeat_policy = validity.get("repeat_policy", {})
        lines.append(
            f"- Eligible repeats: Imagination={repeat_policy.get('imagination_eligible_repeats', 0)}/{sampling.get('requested_repeats', MODEL_SAMPLE_REPEATS)}, "
            f"Hallucination={repeat_policy.get('hallucination_eligible_repeats', 0)}/{sampling.get('requested_repeats', MODEL_SAMPLE_REPEATS)}, "
            f"UUT={repeat_policy.get('uut_eligible_repeats', 0)}/{sampling.get('requested_repeats', MODEL_SAMPLE_REPEATS)}, "
            f"PropConj={repeat_policy.get('propconj_eligible_repeats', 0)}/{sampling.get('requested_repeats', MODEL_SAMPLE_REPEATS)}, "
            f"MacGyver={repeat_policy.get('macgyver_eligible_repeats', 0)}/{sampling.get('requested_repeats', MODEL_SAMPLE_REPEATS)}, "
            f"CJST={repeat_policy.get('cjst_eligible_repeats', 0)}/{sampling.get('requested_repeats', MODEL_SAMPLE_REPEATS)}"
        )
        lines.append(f"- Ranking eligible: {'Yes' if validity.get('ranking_eligible') else 'No'}")
        failures = validity.get("eligibility_failures") or []
        if failures:
            lines.append(f"- Eligibility issues: {'; '.join(failures)}")
        lines.append(f"- Imagination: {format_stat(axes.get('imagination', {}).get('score'), axes.get('imagination', {}).get('replicate_stats', {}).get('std'))}")
        lines.append(f"- Hallucination: {format_stat(axes.get('hallucination', {}).get('score'), axes.get('hallucination', {}).get('replicate_stats', {}).get('std'))} (lower is better)")
        if axes.get("analogy_transfer_challenge", {}).get("imagination") is not None:
            lines.append(
                "- AnalogyTransfer: "
                f"I={format_stat(axes.get('analogy_transfer_challenge', {}).get('imagination'))}; "
                f"H={format_stat(axes.get('analogy_transfer_challenge', {}).get('hallucination'))} (lower is better)"
            )
        lines.append(f"- Non-model skips (total across repeats): {validity.get('non_model_skip_counts', {})}")
        lines.append(f"- Model-output invalid runs (total across repeats): {validity.get('invalid_run_counts', {})}")
        repeat_summaries = sampling.get("repeat_summaries") or []
        if repeat_summaries:
            lines.append("- Repeat summaries:")
            for item in repeat_summaries:
                issues = item.get("eligibility_failures") or []
                issue_text = f" | fail: {issues[0]}" if issues else ""
                lines.append(
                    f"  - r{item.get('repeat_index', 0) + 1}: eligible={item.get('ranking_eligible')} "
                    f"Imag={item.get('imagination')} Halluc={item.get('hallucination')}{issue_text}"
                )
        lines.append("")
    lines.append("## Artifacts")
    for key, value in combined_report["artifacts"].items():
        if isinstance(value, list):
            lines.append(f"- {key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    if combined_report.get("prompt_manifest"):
        lines.append("## Prompt Set")
        for task_type, tasks in combined_report["prompt_manifest"].items():
            lines.append(f"### {task_type}")
            for task in tasks:
                metadata = task.get("metadata") or {}
                meta_text = ", ".join(f"{key}={value}" for key, value in metadata.items())
                suffix = f" ({meta_text})" if meta_text else ""
                lines.append(f"- {task.get('id')}: {task.get('prompt')}{suffix}")
            lines.append("")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Primary ranking uses Imagination (higher is better) and Hallucination (lower is better) from the same dual-axis outputs.")
    excluded_models = combined_report.get("openrouter", {}).get("excluded_models") or []
    if excluded_models:
        excluded_ids = ", ".join(item.get("id", "") for item in excluded_models if item.get("id"))
        lines.append(
            "- Excluded OpenRouter model APIs: "
            f"{excluded_ids}. Reason: {combined_report.get('openrouter', {}).get('excluded_model_reason')}"
        )
    lines.append("- The main benchmark components are UUT, PropConj, MacGyver, CJST, HypoUseSpace, GCW, NeoCoder, and AnalogyTransfer.")
    lines.append("- Subtype analyses consume overall_summary.axes.subtype_scores.")
    lines.append("- Fixed output-only scoring is used.")
    lines.append("- PropConj dual axis reports Imagination and Hallucination from the same property-conjunction outputs using white-box predicates; Hallucination is lower-is-better")
    lines.append("- MacGyver dual axis reports Imagination and Hallucination from the same closed-tool constrained plans using white-box affordance, constraint, and solvability checks; Hallucination is lower-is-better")
    lines.append("- CJST dual axis reports Imagination and Hallucination from the same structured counterfactual consequences using white-box premise-lock, causal-bridge, and forbidden-foil checks; Hallucination is lower-is-better")
    lines.append("- HypoUseSpace dual axis reports Imagination and Hallucination from the same finite closed-world hypothesis set using deterministic canonicalization, valid-set recovery, and local constraint checks; Hallucination is lower-is-better")
    lines.append("- GCW dual axis reports Imagination and Hallucination from the same fact-grounded microfiction outputs using white-box TTCW proxies and closed-world fact/constraint checks; Hallucination is lower-is-better")
    lines.append("- Reasoning/thinking is explicitly disabled for benchmark generation; reasoning-only empty visible outputs are treated as harness failures, not model failures")
    lines.append("- Provider/privacy 404 and similar infra failures are excluded from prompt coverage denominators but repeats still need enough available prompts to be eligible")
    lines.append("- Creative prompts require fixed-length JSON outputs; scoring truncates extras and requires at least the configured minimum valid items")
    lines.append(f"- Task token caps: creative={get_task_max_tokens('creative')}, PropConj={get_task_max_tokens('PropConj')}, CJST={get_task_max_tokens('CJST')}, MacGyver={get_task_max_tokens('MacGyver')}, HypoUseSpace={get_task_max_tokens('HypoUseSpace')}, GCW={get_task_max_tokens('GCW')}, NeoCoder={get_task_max_tokens('NeoCoder')}, AnalogyTransfer={get_task_max_tokens('AnalogyTransfer')}")
    return "\n".join(lines) + "\n"

__all__ = [
    'strip_legacy_conceptnet_fields',
    'report_has_white_box_groundedness',
    'report_matches_current_schema',
    'report_has_primary_dual_axis',
    'summarize_saved_task_groundedness',
    'recompute_saved_report_groundedness',
    'extract_comparison_rows',
    'generate_comparison_chart',
    'build_model_summary_row',
    'rank_models',
    'build_combined_report',
    'build_markdown_report',
]
