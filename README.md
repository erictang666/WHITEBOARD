# Benchmark runtime package

This package contains the fixed main-paper anchor, model-calling runtime, deterministic white-box scorers, and standard dual-axis aggregation.

## Included experiment scope

The broader non-legacy benchmark inventory contains 1,660 items. The public entry point covers the main-paper benchmark evaluation only:

- the shared 80-item anchor across UUT, PropConj, MacGyver, CJST, GCW, HypoUseSpace, NeoCoder, and AnalogyTransfer;

The runtime reads its fixed scoring values directly from `data/dual_axis_scoring_config.json` and the task-specific parameter files.

`data/groundedness_reference_cohort.json` contains fixed aggregate reference statistics used to standardize groundedness and novelty. It contains no model identifiers or raw responses and is required by the scoring runtime.

## Third-party resources

Read `third_party/THIRD_PARTY_NOTICES.md` before running or redistributing the package.

Word Norms 2 derived data is bundled with its source notice and GPL license. SWOW-EN is not bundled because its official CC BY-NC-ND terms do not permit unrestricted redistribution or modification. Obtain the English association-strength file from the official SWOW project and save it as `data/swow_strength.csv`.

The embedding model is fixed to:

    repository: sentence-transformers/all-mpnet-base-v2
    revision: e8c3b32edf5434bc2275fc9bab85f82640a19130

The model is downloaded from Hugging Face on first use and is not bundled.

## Setup

Use Python 3.10 or newer:

    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt

Set only the provider key required for the selected run:

    export OPENROUTER_API_KEY=<your_key>

or:

    export POE_API_KEY=<your_key>

Do not store real keys in this directory.

## Run

Main-paper anchor:

    python run_benchmark.py --provider openrouter --model-id <provider/model-id>

All 1,210 prompts in the eight main-paper task families:

    python run_benchmark.py --provider openrouter --model-id <provider/model-id> --full

Poe:

    python run_benchmark.py --provider poe --model-id <model-id>

One prompt per main-paper family:

    python run_benchmark.py --provider openrouter --model-id <provider/model-id> --smoke

The default output directory is `outputs/`. The launcher does not create a log or generation cache and refuses to replace a model report unless `--overwrite` is supplied.

## Standard score outputs

The runtime applies:

    B(x; floor, gamma) = clip((x-floor)/(1-floor), 0, 1)^gamma
    I_pure = clip(B(I_gated) - beta_IH * mean(B(H_raw)), 0, 1)
    H_dest = clip(B(H_raw) - beta_HI * B(I_gated), 0, 1)

Generated tables:

- benchmark_model_scores.csv
- benchmark_task_scores.csv
- benchmark_subtype_profiles.csv
- benchmark_output_scores.csv
- benchmark_score_summary.json

## Verify the package

Run the included tests before scoring:

    python -m unittest discover -s tests

