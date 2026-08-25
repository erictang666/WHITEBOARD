
import os

from prompts_dataset import get_all_prompts

from semantic_scorer import SemanticScorer
from cognitive_baseline import CognitiveBaseline
from wordnet_analyzer import WordNetAnalyzer
from groundedness_scorer import GroundednessScorer, WHITE_BOX_GROUNDEDNESS_VERSION

import benchmark_core as _benchmark_core
import benchmark_assessment as _benchmark_assessment
import benchmark_scoring as _benchmark_scoring
import common_answer_bank as _common_answer_bank

from benchmark_core import *  
from benchmark_assessment import *  
from common_answer_bank import *  


DEPENDENCY_EXPORTS = [
    'get_all_prompts',
    'SemanticScorer',
    'CognitiveBaseline',
    'WordNetAnalyzer',
    'GroundednessScorer',
    'WHITE_BOX_GROUNDEDNESS_VERSION',
]

__all__ = sorted(set(
    _benchmark_core.__all__
    + _benchmark_assessment.__all__
    + _common_answer_bank.__all__
    + DEPENDENCY_EXPORTS
    + ['main']
))


def main():
    print("=== Starting Imagination/Hallucination Benchmark ===\n")
    if ENV_FILE_LOADED:
        print("[Config] Environment file loaded")
    print(
        "[API Runtime Config] "
        f"stream={OPENROUTER_STREAM}, reasoning={OPENROUTER_ENABLE_REASONING}, "
        f"connect_timeout={OPENROUTER_CONNECT_TIMEOUT:.0f}s, read_timeout={OPENROUTER_REQUEST_TIMEOUT:.0f}s, "
        f"max_tokens={OPENROUTER_MAX_TOKENS}, retries={OPENROUTER_MAX_RETRIES}"
    )
    print(f"[Model Set] active={ACTIVE_MODEL_SET}, count={len(SELECTED_MODELS)}")
    print(
        "[Token Policy] "
        f"creative={get_task_max_tokens('creative')}, PropConj={get_task_max_tokens('PropConj')}, "
        f"CJST={get_task_max_tokens('CJST')}, "
        f"MacGyver={get_task_max_tokens('MacGyver')}, "
        f"HypoUseSpace={get_task_max_tokens('HypoUseSpace')}, "
        f"GCW={get_task_max_tokens('GCW')}, "
        f"NeoCoder={get_task_max_tokens('NeoCoder')}, "
        f"AnalogyTransfer={get_task_max_tokens('AnalogyTransfer')}"
    )
    print(
        "[Sampling Policy] "
        f"repeats={MODEL_SAMPLE_REPEATS}, minimum_eligible_repeats={minimum_eligible_repeats(MODEL_SAMPLE_REPEATS)}, "
        f"seed_base={SAMPLING_SEED_BASE}, bootstrap_samples={BOOTSTRAP_SAMPLES}"
    )
    print(
        "[Output Targets] "
        f"creative={CREATIVE_OUTPUT_COUNT}, PropConj={PROP_CONJ_OUTPUT_COUNT}, "
        f"CJST={CJST_OUTPUT_COUNT}, "
        f"MacGyver={MACGYVER_OUTPUT_COUNT}, "
        f"HypoUseSpace={HYPOUSESPACE_OUTPUT_COUNT}, "
        f"GCW={GCW_BEAT_COUNT}, "
        f"NeoCoder={NEOCODER_OUTPUT_COUNT}, "
        f"AnalogyTransfer={ANALOGY_TRANSFER_OUTPUT_COUNT}; "
        f"minimum_valid creative={MIN_CREATIVE_ITEMS_PER_TASK}, "
        f"PropConj={MIN_PROP_CONJ_ITEMS_PER_TASK}, "
        f"CJST={MIN_CJST_ITEMS_PER_TASK} ({MIN_CJST_ITEMS_PER_TIER}/tier), "
        f"MacGyver={MIN_MACGYVER_PLANS_PER_TASK}, "
        f"HypoUseSpace={MIN_HYPOUSESPACE_ITEMS_PER_TASK}, "
        f"GCW={MIN_GCW_BEATS_PER_TASK}, "
        f"NeoCoder={MIN_NEOCODER_ITEMS_PER_TASK}, "
        f"AnalogyTransfer={MIN_ANALOGY_TRANSFER_ITEMS_PER_TASK}"
    )
    for model_config in MODEL_CONFIGS:
        model_id = model_config["id"]
        print(
            f"[Model Config] {model_id}: "
            f"provider={get_model_provider(model_id)}, "
            f"reasoning={bool(OPENROUTER_ENABLE_REASONING and model_config.get('reasoning'))}, "
            f"stream={should_stream_for_model(model_id)}"
        )

    dataset_all = get_all_prompts()
    dataset, task_policy = filter_dataset_for_runtime(dataset_all)
    print(
        "[Task Runtime Policy] "
        f"included={task_policy['included_task_families']}, "
        f"prompts_per_repeat={task_policy['total_prompts_per_repeat']}"
    )
    if task_policy["limited_task_families"]:
        print(f"[Task Runtime Policy] limited={task_policy['limited_task_families']}")
    if task_policy["skipped_task_families"]:
        print(f"[Task Runtime Policy] skipped={task_policy['skipped_task_families']}")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    model_catalog = fetch_selected_model_catalog()
    model_catalog_index = build_model_catalog_index(model_catalog)
    validate_selected_models(model_catalog_index)

    client = create_provider_clients()
    scorer = SemanticScorer()
    print(f"[Embedding] SentenceTransformer model: {scorer.model_name}")
    print(f"[Embedding] Note: {scorer.model_note}")

    print("-" * 60)
    cog_baseline = CognitiveBaseline(swow_path=SWOW_DATA_PATH)
    print("-" * 60)
    wn_analyzer = WordNetAnalyzer()
    print("-" * 60)
    groundedness_scorer = GroundednessScorer(wn_analyzer=wn_analyzer)
    print("-" * 60 + "\n")

    model_reports = []
    model_errors = []

    for model_name in SELECTED_MODELS:
        try:
            model_report = run_model_assessment(
                client=client,
                model_name=model_name,
                dataset=dataset,
                scorer=scorer,
                cog_baseline=cog_baseline,
                wn_analyzer=wn_analyzer,
                groundedness_scorer=groundedness_scorer,
                model_catalog_entry=model_catalog_index.get(model_name),
            )
            model_reports.append(model_report)
            model_report_path = os.path.join(REPORTS_DIR, f"{sanitize_filename(model_name)}_report.json")
            save_json(model_report_path, model_report)
            print(f"[Saved] {model_report_path}")
        except Exception as exc:
            model_errors.append({
                "model_id": model_name,
                "error": str(exc),
            })
            print(f"[ERROR] Model {model_name} failed: {exc}")

    print(f"Completed model reports: {len(model_reports)} / {len(SELECTED_MODELS)}")
    if model_errors:
        print("Models without a completed report:")
        for item in model_errors:
            print(f"  - {item['model_id']}: {item['error']}")
    if model_reports:
        score_result = _benchmark_scoring.export_benchmark_scores(REPORTS_DIR)
        print(f"Benchmark model scores: {score_result['files']['model_scores']}")
    print("=== Assessment Complete ===")

if __name__ == "__main__":
    main()
