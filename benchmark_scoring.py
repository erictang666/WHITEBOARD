#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "data" / "dual_axis_scoring_config.json"
VIEWS = ("raw", "gated", "residual")


def load_scoring_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "boundary_settings",
        "tasks",
        "imagination_multipliers",
        "hallucination_multipliers",
        "component_weights",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Benchmark scoring configuration is missing fields: {missing}")
    payload["betas"] = {
        str(family): {
            "beta_ih": float(values["beta_IH"]),
            "beta_hi": float(values["beta_HI"]),
        }
        for family, values in payload["tasks"].items()
    }
    payload["family_imagination_subtype"] = {
        str(family): str(values["imagination_subtype"])
        for family, values in payload["tasks"].items()
    }
    payload["family_hallucination_subtypes"] = {
        str(family): list(values["hallucination_subtypes"])
        for family, values in payload["tasks"].items()
    }
    return payload


def boundary_transform(value: float, *, floor: float, gamma: float) -> float:
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"Boundary floor must be in [0,1): {floor}")
    scaled = min(1.0, max(0.0, (float(value) - floor) / (1.0 - floor)))
    return scaled ** float(gamma)


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _iter_report_outputs(report: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    seen: set[tuple[object, object, object]] = set()
    for entry in report.get("task_results") or []:
        if not isinstance(entry, Mapping):
            continue
        key = (entry.get("task_type"), entry.get("task_id"), entry.get("repeat_index"))
        seen.add(key)
        yield entry
    for name, block in report.items():
        if not str(name).endswith("_results") or not isinstance(block, Mapping):
            continue
        for entry in block.get("details") or []:
            if not isinstance(entry, Mapping):
                continue
            key = (entry.get("task_type"), entry.get("task_id"), entry.get("repeat_index"))
            if key not in seen:
                seen.add(key)
                yield entry


def _subtype_contributions(entry: Mapping[str, object]) -> Mapping[str, object] | None:
    for holder_name in ("dual_axis", "challenge"):
        holder = entry.get(holder_name)
        if isinstance(holder, Mapping) and isinstance(holder.get("subtype_contributions"), Mapping):
            return holder["subtype_contributions"]
    return None


def extract_long_records(report_dir: str | Path, config: Mapping[str, object]) -> list[dict[str, object]]:
    directory = Path(report_dir)
    families = set(config["betas"])
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        model_id = str(report.get("model_id") or path.stem.removesuffix("_report"))
        for entry in _iter_report_outputs(report):
            family = str(entry.get("task_type") or "")
            if family not in families or not _truthy(entry.get("valid_run")):
                continue
            contribution = _subtype_contributions(entry)
            if contribution is None:
                continue
            base = {
                "model_id": model_id,
                "task_type": family,
                "task_id": str(entry.get("task_id") or ""),
                "repeat_index": entry.get("repeat_index") if entry.get("repeat_index") is not None else 0,
            }
            for view in VIEWS:
                view_block = contribution.get(view)
                if not isinstance(view_block, Mapping):
                    continue
                for axis in ("imagination", "hallucination"):
                    scores = view_block.get(axis)
                    if not isinstance(scores, Mapping):
                        continue
                    for subtype, value in scores.items():
                        if subtype == "consistency" or isinstance(value, bool):
                            continue
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            continue
                        rows.append({
                            **base,
                            "view": view,
                            "axis": axis,
                            "subtype": str(subtype),
                            "score": numeric,
                        })
    if not rows:
        raise ValueError(f"No valid benchmark subtype contributions found in {directory}")
    return rows


def compute_benchmark_scores(records: Iterable[Mapping[str, object]], config: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
    families = tuple(config["betas"])
    family_set = set(families)
    i_subtype_by_family = {str(k): str(v) for k, v in config["family_imagination_subtype"].items()}
    h_subtypes_by_family = {
        str(k): tuple(str(item) for item in values)
        for k, values in config["family_hallucination_subtypes"].items()
    }
    i_boundaries = config["boundary_settings"]["I"]
    h_boundaries = config["boundary_settings"]["H"]

    gated_i: dict[tuple[object, ...], dict[str, object]] = {}
    raw_h: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in records:
        family = str(row["task_type"])
        if family not in family_set:
            continue
        key = (row["model_id"], family, row["task_id"], row["repeat_index"])
        subtype = str(row["subtype"])
        score = float(row["score"])
        if row["view"] == "gated" and row["axis"] == "imagination" and subtype == i_subtype_by_family[family]:
            if key in gated_i:
                raise ValueError(f"Duplicate gated imagination output key: {key}")
            settings = i_boundaries[subtype]
            gated_i[key] = {
                "subtype": subtype,
                "input_score": score,
                "boundary_score": boundary_transform(
                    score, floor=float(settings["floor"]), gamma=float(settings["gamma"])
                ),
            }
        elif row["view"] == "raw" and row["axis"] == "hallucination" and subtype in h_subtypes_by_family[family]:
            if subtype in raw_h[key]:
                raise ValueError(f"Duplicate raw hallucination output/subtype key: {key + (subtype,)}")
            settings = h_boundaries[subtype]
            raw_h[key][subtype] = {
                "input_score": score,
                "boundary_score": boundary_transform(
                    score, floor=float(settings["floor"]), gamma=float(settings["gamma"])
                ),
            }

    common_keys = sorted(set(gated_i) & set(raw_h))
    if not common_keys:
        raise ValueError("No outputs contain both gated imagination and raw hallucination contributions")

    task_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    i_rows: list[dict[str, object]] = []
    h_rows: list[dict[str, object]] = []
    for key in common_keys:
        model_id, family, task_id, repeat_index = key
        observed_h = set(raw_h[key])
        expected_h = set(h_subtypes_by_family[family])
        if observed_h != expected_h:
            raise ValueError(
                f"Hallucination subtype mismatch for {key}: observed={sorted(observed_h)}, expected={sorted(expected_h)}"
            )
        i_item = gated_i[key]
        h_mean = sum(float(item["boundary_score"]) for item in raw_h[key].values()) / len(raw_h[key])
        beta = config["betas"][family]
        i_residual = min(1.0, max(0.0, float(i_item["boundary_score"]) - float(beta["beta_ih"]) * h_mean))
        h_values = []
        i_row = {
            "model_id": model_id,
            "task_type": family,
            "task_id": task_id,
            "repeat_index": repeat_index,
            "subtype": i_item["subtype"],
            "input_score": i_item["input_score"],
            "boundary_score": i_item["boundary_score"],
            "residual_score": i_residual,
        }
        i_rows.append(i_row)
        long_rows.append({**i_row, "axis": "I"})
        for subtype in h_subtypes_by_family[family]:
            h_item = raw_h[key][subtype]
            h_residual = min(
                1.0,
                max(0.0, float(h_item["boundary_score"]) - float(beta["beta_hi"]) * float(i_item["boundary_score"])),
            )
            h_values.append(h_residual)
            h_row = {
                "model_id": model_id,
                "task_type": family,
                "task_id": task_id,
                "repeat_index": repeat_index,
                "subtype": subtype,
                "input_score": h_item["input_score"],
                "boundary_score": h_item["boundary_score"],
                "residual_score": h_residual,
            }
            h_rows.append(h_row)
            long_rows.append({**h_row, "axis": "H"})
        task_rows.append({
            "model_id": model_id,
            "task_type": family,
            "task_id": task_id,
            "repeat_index": repeat_index,
            "I_subtype": i_item["subtype"],
            "I_pure": i_residual,
            "H_dest": sum(h_values) / len(h_values),
        })

    i_multipliers = {str(k): float(v) for k, v in config["imagination_multipliers"].items()}
    h_multipliers = {str(k): float(v) for k, v in config["hallucination_multipliers"].items()}
    i_total = sum(i_multipliers.values())
    h_total = sum(h_multipliers.values())
    i_weights = {key: value / i_total for key, value in i_multipliers.items()}
    h_weights = {key: value / h_total for key, value in h_multipliers.items()}
    component_weights = {str(k): float(v) for k, v in config["component_weights"].items()}

    i_counts: dict[tuple[object, ...], int] = defaultdict(int)
    h_counts: dict[tuple[object, ...], int] = defaultdict(int)
    for row in i_rows:
        i_counts[(row["model_id"], row["task_type"], row["subtype"])] += 1
    for row in h_rows:
        h_counts[(row["model_id"], row["task_type"], row["subtype"])] += 1

    model_scores: dict[str, dict[str, float]] = defaultdict(lambda: {
        "I_pure": 0.0,
        "H_dest": 0.0,
    })
    profiles: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in i_rows:
        model_id = str(row["model_id"])
        key = (row["model_id"], row["task_type"], row["subtype"])
        base = component_weights[str(row["task_type"])] * float(row["residual_score"]) / i_counts[key]
        model_scores[model_id]["I_pure"] += base * i_weights[str(row["subtype"])]
        profiles[model_id][f"residual_I_{row['subtype']}"] += base
    for row in h_rows:
        model_id = str(row["model_id"])
        key = (row["model_id"], row["task_type"], row["subtype"])
        base = component_weights[str(row["task_type"])] * float(row["residual_score"]) / h_counts[key]
        model_scores[model_id]["H_dest"] += base * h_weights[str(row["subtype"])]
        profiles[model_id][f"residual_H_{row['subtype']}"] += base

    model_rows = [
        {"model_id": model_id, **values}
        for model_id, values in sorted(model_scores.items())
    ]
    i_profile_names = [f"residual_I_{name}" for name in i_multipliers]
    h_profile_names = [f"residual_H_{name}" for name in h_multipliers]
    profile_rows = []
    for model_id in sorted(profiles):
        profile_rows.append({
            "model_id": model_id,
            **{name: profiles[model_id].get(name, 0.0) for name in i_profile_names + h_profile_names},
        })

    return {
        "task_scores": sorted(task_rows, key=lambda row: (str(row["model_id"]), str(row["task_type"]), str(row["task_id"]), int(row["repeat_index"]))),
        "model_scores": model_rows,
        "profiles": profile_rows,
        "output_scores": sorted(long_rows, key=lambda row: (str(row["model_id"]), str(row["task_type"]), str(row["task_id"]), int(row["repeat_index"]), str(row["axis"]), str(row["subtype"]))),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty benchmark output: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_benchmark_scores(
    report_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    report_dir = Path(report_dir)
    output_dir = Path(output_dir) if output_dir is not None else report_dir
    config = load_scoring_config(config_path)
    records = extract_long_records(report_dir, config)
    results = compute_benchmark_scores(records, config)
    outputs = {
        "task_scores": output_dir / "benchmark_task_scores.csv",
        "model_scores": output_dir / "benchmark_model_scores.csv",
        "profiles": output_dir / "benchmark_subtype_profiles.csv",
        "output_scores": output_dir / "benchmark_output_scores.csv",
    }
    for name, path in outputs.items():
        _write_csv(path, results[name])
    summary = {
        "schema": "benchmark_score_summary",
        "models": len(results["model_scores"]),
        "task_outputs": len(results["task_scores"]),
        "subtype_rows": len(results["output_scores"]),
        "files": {name: path.name for name, path in outputs.items()},
    }
    summary_path = output_dir / "benchmark_score_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**summary, "summary": summary_path, "paths": outputs}
