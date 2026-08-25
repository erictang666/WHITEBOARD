#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIMARY_FAMILIES = (
    "UUT",
    "PropConj",
    "MacGyver",
    "CJST",
    "GCW",
    "HypoUseSpace",
    "NeoCoder",
    "AnalogyTransfer",
)
ANCHOR_PROMPT_COUNT = 80
FULL_MAIN_PROMPT_COUNT = 1210


def _validate_required_resources() -> None:
    lock = json.loads((ROOT / "resources.lock.json").read_text(encoding="utf-8"))
    for resource_name in ("word_norms_2", "swow_en"):
        spec = lock[resource_name]
        rel = spec.get("bundled_file") or spec.get("required_file")
        path = ROOT / rel
        if not path.is_file():
            if resource_name == "swow_en":
                raise SystemExit(
                    "SWOW-EN data is required but cannot be redistributed by this package. "
                    "Download it from the source in third_party/THIRD_PARTY_NOTICES.md, "
                    "and save it as data/swow_strength.csv."
                )
            raise SystemExit(f"Missing required resource: {rel}")


def _set_runtime_environment(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    full_run = bool(getattr(args, "full", False))
    env = {
        "BENCHMARK_PROVIDER": args.provider,
        "BENCHMARK_RUN_MODE": "full" if full_run else "anchor",
        "OPENROUTER_MODEL_IDS": args.model_id,
        "OPENROUTER_EXPERIMENT_PROFILE": "benchmark",
        "OPENROUTER_REPORTS_DIR": os.path.relpath(output_dir, ROOT),
        "OPENROUTER_TASK_MANIFEST_PATH": "" if full_run else "data/anchor_manifest.json",
        "OPENROUTER_TASK_FAMILIES": ",".join(PRIMARY_FAMILIES),
        "OPENROUTER_MODEL_SAMPLE_REPEATS": "1",
        "OPENROUTER_PROMPT_EXPANSION_FACTOR": "5",
        "OPENROUTER_UUT_OUTPUT_COUNT": "8",
        "OPENROUTER_PROP_CONJ_OUTPUT_COUNT": "6",
        "OPENROUTER_CJST_OUTPUT_COUNT": "6",
        "OPENROUTER_MACGYVER_OUTPUT_COUNT": "3",
        "OPENROUTER_HYPOUSESPACE_OUTPUT_COUNT": "3",
        "OPENROUTER_GCW_BEAT_COUNT": "1",
        "OPENROUTER_NEOCODER_OUTPUT_COUNT": "1",
        "OPENROUTER_ANALOGY_TRANSFER_OUTPUT_COUNT": "1",
        "OPENROUTER_MIN_CREATIVE_ITEMS_PER_TASK": "1",
        "OPENROUTER_MIN_PROP_CONJ_ITEMS_PER_TASK": "1",
        "OPENROUTER_MIN_CJST_ITEMS_PER_TASK": "1",
        "OPENROUTER_MIN_CJST_ITEMS_PER_TIER": "0",
        "OPENROUTER_MIN_MACGYVER_PLANS_PER_TASK": "1",
        "OPENROUTER_MIN_HYPOUSESPACE_ITEMS_PER_TASK": "1",
        "OPENROUTER_MIN_GCW_BEATS_PER_TASK": "1",
        "OPENROUTER_MIN_NEOCODER_ITEMS_PER_TASK": "1",
        "OPENROUTER_MIN_ANALOGY_TRANSFER_ITEMS_PER_TASK": "1",
        "OPENROUTER_MIN_CREATIVE_COVERAGE": "0",
        "OPENROUTER_MIN_CREATIVE_TASKTYPE_COVERAGE": "0",
        "OPENROUTER_MIN_MACGYVER_COVERAGE": "0",
        "OPENROUTER_MIN_CJST_COVERAGE": "0",
        "OPENROUTER_MIN_HYPOUSESPACE_COVERAGE": "0",
        "OPENROUTER_MIN_GCW_COVERAGE": "0",
        "OPENROUTER_MIN_NEOCODER_COVERAGE": "0",
        "OPENROUTER_MIN_ANALOGY_TRANSFER_COVERAGE": "0",
        "OPENROUTER_STREAM": "false",
        "OPENROUTER_ENABLE_REASONING": "false",
        "OPENROUTER_USE_GENERATION_CACHE": "false",
        "OPENROUTER_SAVE_GENERATION_CACHE": "false",
        "OPENROUTER_GENERATION_CACHE_ONLY": "false",
        "OPENROUTER_GENERATION_CACHE_MISS_FATAL": "false",
        "OPENROUTER_EMBEDDING_DEVICE": args.embedding_device,
        "OPENROUTER_APP_TITLE": "Dual-Axis Imagination Benchmark",
    }
    os.environ.update(env)
    os.environ.pop("OPENROUTER_TASK_LIMITS", None)
    if args.smoke:
        os.environ["OPENROUTER_MAX_TASKS_PER_FAMILY"] = "1"
    else:
        os.environ.pop("OPENROUTER_MAX_TASKS_PER_FAMILY", None)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the main-paper benchmark for one model")
    parser.add_argument("--provider", choices=("openrouter", "poe"), default="openrouter")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--embedding-device", default="cpu")
    run_size = parser.add_mutually_exclusive_group()
    run_size.add_argument("--smoke", action="store_true", help="Run one prompt per selected family")
    run_size.add_argument("--full", action="store_true", help="Run all 1,210 main-paper prompts instead of the 80-item anchor")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the selected model's existing report")
    args = parser.parse_args()
    _validate_required_resources()
    if args.smoke:
        mode_name, prompt_count = "smoke check", len(PRIMARY_FAMILIES)
    elif args.full:
        mode_name, prompt_count = "full main-paper set", FULL_MAIN_PROMPT_COUNT
    else:
        mode_name, prompt_count = "main-paper anchor", ANCHOR_PROMPT_COUNT
    print(f"[Benchmark Mode] {mode_name}: {prompt_count} prompts")

    key_name = "OPENROUTER_API_KEY" if args.provider == "openrouter" else "POE_API_KEY"
    if not os.getenv(key_name):
        raise SystemExit(f"Set {key_name} in the environment before running the benchmark")

    os.chdir(ROOT)
    output_dir = _set_runtime_environment(args)
    report_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_id) + "_report.json"
    report_path = output_dir / report_name
    if report_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing report: {report_path.name}; pass --overwrite to replace it")

    import main as benchmark_main
    benchmark_main.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
