
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
import traceback
import urllib.error
import urllib.request

import httpx
from openai import OpenAI

try:
    import numpy as np
except ImportError:
    np = None

def load_dotenv_if_present():
    candidate_paths = []
    seen_paths = set()

    for base_dir in [os.getcwd(), os.path.dirname(__file__), os.path.dirname(os.path.dirname(__file__))]:
        env_path = os.path.abspath(os.path.join(base_dir, ".env"))
        if env_path not in seen_paths:
            candidate_paths.append(env_path)
            seen_paths.add(env_path)

    for env_path in candidate_paths:
        if not os.path.isfile(env_path):
            continue

        try:
            with open(env_path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    stripped = raw_line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue

                    if stripped.startswith("export "):
                        stripped = stripped[len("export "):].strip()

                    if "=" not in stripped:
                        continue

                    key, value = stripped.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if not key or key in os.environ:
                        continue

                    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                        value = value[1:-1]

                    os.environ[key] = value
            return True
        except OSError:
            continue

    return None

def env_flag(name, default=False):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}

ENV_FILE_LOADED = load_dotenv_if_present()


OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODELS_ENDPOINT = f"{OPENROUTER_BASE_URL.rstrip('/')}/models"
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "Dual-Axis Imagination Benchmark")
POE_API_KEY_ENV = "POE_API_KEY"
POE_BASE_URL = os.getenv("POE_BASE_URL", "https://api.poe.com/v1")
POE_MODELS_ENDPOINT = f"{POE_BASE_URL.rstrip('/')}/models"
MODEL_PROVIDER_OPENROUTER = "openrouter"
MODEL_PROVIDER_POE = "poe"
BENCHMARK_PROVIDER_OVERRIDE = os.getenv("BENCHMARK_PROVIDER", "").strip().lower()
if BENCHMARK_PROVIDER_OVERRIDE and BENCHMARK_PROVIDER_OVERRIDE not in {
    MODEL_PROVIDER_OPENROUTER,
    MODEL_PROVIDER_POE,
}:
    raise RuntimeError("BENCHMARK_PROVIDER must be either 'openrouter' or 'poe'")
OPENROUTER_REQUEST_TIMEOUT = float(os.getenv("OPENROUTER_REQUEST_TIMEOUT", "90"))
OPENROUTER_CONNECT_TIMEOUT = float(os.getenv("OPENROUTER_CONNECT_TIMEOUT", "10"))
OPENROUTER_MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "2"))
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "768"))
OPENROUTER_GENERATION_CACHE_DIR = os.getenv("OPENROUTER_GENERATION_CACHE_DIR", "").strip()




OPENROUTER_USE_GENERATION_CACHE = env_flag(
    "OPENROUTER_USE_GENERATION_CACHE",
    default=bool(OPENROUTER_GENERATION_CACHE_DIR),
)
OPENROUTER_SAVE_GENERATION_CACHE = env_flag(
    "OPENROUTER_SAVE_GENERATION_CACHE",
    default=bool(OPENROUTER_GENERATION_CACHE_DIR),
)
OPENROUTER_GENERATION_CACHE_ONLY = env_flag("OPENROUTER_GENERATION_CACHE_ONLY", default=False)
OPENROUTER_GENERATION_CACHE_MISS_FATAL = env_flag(
    "OPENROUTER_GENERATION_CACHE_MISS_FATAL",
    default=False,
)


OPENROUTER_STREAM = env_flag("OPENROUTER_STREAM", default=True)

OPENROUTER_ENABLE_REASONING = False


def poe_model_config(model_id):
    return {"id": model_id, "provider": MODEL_PROVIDER_POE, "reasoning": False, "stream": False}


def openrouter_model_config(model_id):
    return {"id": model_id, "provider": MODEL_PROVIDER_OPENROUTER, "reasoning": False, "stream": False}




DEFAULT_MODEL_CONFIGS = [
    poe_model_config("deepseek-v3.2"),
    poe_model_config("deepseek-v3-di"),
    poe_model_config("grok-4.20-multi-agent"),
    poe_model_config("grok-3"),
    poe_model_config("llama-3.1-8b-di"),
    openrouter_model_config("bytedance-seed/seed-1.6"),
]


DEFAULT_MODEL_CONFIGS += [
    openrouter_model_config("tencent/hy3-preview:free"),
    poe_model_config("mimo-v2.5"),
    poe_model_config("kimi-k2"),
    poe_model_config("kimi-k2.6"),
    poe_model_config("mimo-v2-omni"),
    poe_model_config("glm-5"),
    poe_model_config("kimi-k2.5"),
]

DEFAULT_MODEL_CONFIGS += [
    poe_model_config("mimo-v2-flash"),
    poe_model_config("gemma-4-31b"),
    poe_model_config("gemma-3-27b"),
    openrouter_model_config("google/gemma-2-27b-it"),
    poe_model_config("qwen3.5-flash"),
    poe_model_config("qwen3.5-397b-a17b"),
    poe_model_config("qwen3.5-plus"),
    poe_model_config("qwen3.6-plus"),
    poe_model_config("deepseek-v3.1"),
    poe_model_config("deepseek-v4-flash-el"),
]

DEFAULT_MODEL_CONFIGS += [
    poe_model_config("claude-opus-4.7"),
    poe_model_config("gpt-5.5"),
    poe_model_config("gpt-5.4"),
    poe_model_config("gpt-5.4-mini"),
    poe_model_config("gemini-3.1-flash-lite"),
    poe_model_config("gemma-4-31b"),
    poe_model_config("claude-haiku-3"),
    poe_model_config("claude-opus-4.6"),
    poe_model_config("claude-haiku-4.5"),
    poe_model_config("claude-sonnet-4.6"),
    poe_model_config("gpt-5.2"),
    poe_model_config("claude-sonnet-4.5"),
    poe_model_config("gpt-4.1-nano"),
    poe_model_config("gpt-5.1"),
    poe_model_config("gpt-4o-mini"),
    poe_model_config("gpt-5-mini"),
    poe_model_config("claude-sonnet-4"),
    poe_model_config("gemini-2.0-flash-lite"),
    poe_model_config("gpt-4.1"),
    poe_model_config("gpt-4.1-mini"),
    poe_model_config("claude-sonnet-3.7"),
    poe_model_config("claude-haiku-3.5"),
    poe_model_config("gemini-2.0-flash"),
    poe_model_config("gpt-3.5-turbo-instruct"),
    poe_model_config("gemini-3.1-pro"),
    poe_model_config("gemini-2.5-pro"),
    poe_model_config("grok-4.3"),
    poe_model_config("llama-3.1-8b-fp16"),
    poe_model_config("mistral-small-3"),
    poe_model_config("llama-3.3-70b"),
]

def dedupe_model_configs(model_configs):
    seen = set()
    unique = []
    for config in model_configs:
        model_id = config.get("id")
        if not model_id or model_id in seen:
            continue
        unique.append(config)
        seen.add(model_id)
    return unique


ALL_MODEL_CONFIGS = dedupe_model_configs(DEFAULT_MODEL_CONFIGS)
ALL_MODEL_CONFIG_INDEX = {item["id"]: item for item in ALL_MODEL_CONFIGS}

MODEL_IDS_OVERRIDE = [
    item.strip()
    for item in os.getenv("OPENROUTER_MODEL_IDS", "").split(",")
    if item.strip()
]
ACTIVE_MODEL_SET = "selected" if MODEL_IDS_OVERRIDE else "default"
if MODEL_IDS_OVERRIDE:
    MODEL_CONFIGS = [
        dict(ALL_MODEL_CONFIG_INDEX[item]) if item in ALL_MODEL_CONFIG_INDEX else {
            "id": item,
            "provider": BENCHMARK_PROVIDER_OVERRIDE or MODEL_PROVIDER_OPENROUTER,
            "reasoning": False,
            "stream": False,
        }
        for item in MODEL_IDS_OVERRIDE
    ]
    if BENCHMARK_PROVIDER_OVERRIDE:
        for model_config in MODEL_CONFIGS:
            model_config["provider"] = BENCHMARK_PROVIDER_OVERRIDE
else:
    MODEL_CONFIGS = list(ALL_MODEL_CONFIGS)


SELECTED_MODELS = [item["id"] for item in MODEL_CONFIGS]
MODEL_CONFIG_INDEX = {item["id"]: item for item in MODEL_CONFIGS}

BENCHMARK_PROFILE = "benchmark"
OPENROUTER_EXPERIMENT_PROFILE = os.getenv(
    "OPENROUTER_EXPERIMENT_PROFILE",
    BENCHMARK_PROFILE,
).strip().lower()
if OPENROUTER_EXPERIMENT_PROFILE != BENCHMARK_PROFILE:
    raise RuntimeError("OPENROUTER_EXPERIMENT_PROFILE must be benchmark")

REPORTS_DIR = os.getenv("OPENROUTER_REPORTS_DIR", "outputs").strip() or "outputs"
def make_file_tag(value):
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    tag = re.sub(r"_+", "_", tag).strip("._")
    return tag or "run"

model_tag_source = SELECTED_MODELS[0] if len(SELECTED_MODELS) == 1 else "benchmark"
MODEL_SET_FILE_TAG = make_file_tag(model_tag_source)
MULTI_MODEL_REPORT_JSON = os.path.join(REPORTS_DIR, f"openrouter_multi_model_report_{MODEL_SET_FILE_TAG}.json")
MULTI_MODEL_REPORT_MD = os.path.join(REPORTS_DIR, f"openrouter_multi_model_report_{MODEL_SET_FILE_TAG}.md")
MULTI_MODEL_CHART = os.path.join(REPORTS_DIR, f"openrouter_model_comparison_heatmap_{MODEL_SET_FILE_TAG}.png")

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for",
    "with", "in", "on", "at", "by", "is", "are", "it", "this", "that"
}

SWOW_DATA_PATH = "data"
DEDUP_SIMILARITY_THRESHOLD = 0.80
UUT_OUTPUT_COUNT = int(os.getenv("OPENROUTER_UUT_OUTPUT_COUNT", os.getenv("OPENROUTER_CREATIVE_OUTPUT_COUNT", "20")))
CREATIVE_OUTPUT_COUNT = UUT_OUTPUT_COUNT
PROP_CONJ_OUTPUT_COUNT = int(os.getenv("OPENROUTER_PROP_CONJ_OUTPUT_COUNT", "12"))
CJST_OUTPUT_COUNT = int(os.getenv("OPENROUTER_CJST_OUTPUT_COUNT", "12"))
MACGYVER_OUTPUT_COUNT = int(os.getenv("OPENROUTER_MACGYVER_OUTPUT_COUNT", "5"))
HYPOUSESPACE_OUTPUT_COUNT = int(os.getenv("OPENROUTER_HYPOUSESPACE_OUTPUT_COUNT", "6"))
GCW_BEAT_COUNT = int(os.getenv("OPENROUTER_GCW_BEAT_COUNT", "6"))
NEOCODER_OUTPUT_COUNT = int(os.getenv("OPENROUTER_NEOCODER_OUTPUT_COUNT", "1"))
CLOSED_WORLD_FACT_OUTPUT_COUNT = int(os.getenv("OPENROUTER_CLOSED_WORLD_FACT_OUTPUT_COUNT", "1"))
ANALOGY_TRANSFER_OUTPUT_COUNT = int(os.getenv("OPENROUTER_ANALOGY_TRANSFER_OUTPUT_COUNT", "1"))
DAT_OUTPUT_COUNT = int(os.getenv("OPENROUTER_DAT_OUTPUT_COUNT", "10"))
FF_OUTPUT_COUNT = int(os.getenv("OPENROUTER_FF_OUTPUT_COUNT", "20"))
MIN_CREATIVE_ITEMS_PER_TASK = int(os.getenv("OPENROUTER_MIN_CREATIVE_ITEMS_PER_TASK", "16"))
MIN_PROP_CONJ_ITEMS_PER_TASK = int(os.getenv("OPENROUTER_MIN_PROP_CONJ_ITEMS_PER_TASK", "9"))
MIN_CJST_ITEMS_PER_TASK = int(os.getenv("OPENROUTER_MIN_CJST_ITEMS_PER_TASK", "9"))
MIN_CJST_ITEMS_PER_TIER = int(os.getenv("OPENROUTER_MIN_CJST_ITEMS_PER_TIER", "2"))
MIN_MACGYVER_PLANS_PER_TASK = int(os.getenv("OPENROUTER_MIN_MACGYVER_PLANS_PER_TASK", "3"))
MIN_HYPOUSESPACE_ITEMS_PER_TASK = int(os.getenv("OPENROUTER_MIN_HYPOUSESPACE_ITEMS_PER_TASK", "4"))
MIN_GCW_BEATS_PER_TASK = int(os.getenv("OPENROUTER_MIN_GCW_BEATS_PER_TASK", "4"))
MIN_NEOCODER_ITEMS_PER_TASK = int(os.getenv("OPENROUTER_MIN_NEOCODER_ITEMS_PER_TASK", "1"))
MIN_CLOSED_WORLD_FACT_ITEMS_PER_TASK = int(os.getenv("OPENROUTER_MIN_CLOSED_WORLD_FACT_ITEMS_PER_TASK", "1"))
MIN_ANALOGY_TRANSFER_ITEMS_PER_TASK = int(os.getenv("OPENROUTER_MIN_ANALOGY_TRANSFER_ITEMS_PER_TASK", "1"))
MIN_FF_WORDS_PER_TASK = int(os.getenv("OPENROUTER_MIN_FF_WORDS_PER_TASK", "16"))



MODEL_SAMPLE_REPEATS = int(os.getenv("OPENROUTER_MODEL_SAMPLE_REPEATS", "1"))
REPEAT_ELIGIBLE_FRACTION = float(os.getenv("OPENROUTER_REPEAT_ELIGIBLE_FRACTION", "0.67"))
BOOTSTRAP_SAMPLES = int(os.getenv("OPENROUTER_BOOTSTRAP_SAMPLES", "400"))
SAMPLING_SEED_BASE = int(os.getenv("OPENROUTER_SAMPLING_SEED_BASE", "1729"))
TASK_TEMPERATURES = {
    "creative": float(os.getenv("OPENROUTER_CREATIVE_TEMPERATURE", "0.85")),
    "CJST": float(os.getenv("OPENROUTER_CJST_TEMPERATURE", "0.85")),
    "MacGyver": float(os.getenv("OPENROUTER_MACGYVER_TEMPERATURE", "0.70")),
    "HypoUseSpace": float(os.getenv("OPENROUTER_HYPOUSESPACE_TEMPERATURE", "0.85")),
    "GCW": float(os.getenv("OPENROUTER_GCW_TEMPERATURE", "0.85")),
    "NeoCoder": float(os.getenv("OPENROUTER_NEOCODER_TEMPERATURE", "0.55")),
    "ClosedWorldFact": float(os.getenv("OPENROUTER_CLOSED_WORLD_FACT_TEMPERATURE", "0.20")),
    "AnalogyTransfer": float(os.getenv("OPENROUTER_ANALOGY_TRANSFER_TEMPERATURE", "0.55")),
    "DAT": float(os.getenv("OPENROUTER_DAT_TEMPERATURE", "0.70")),
    "CDAT": float(os.getenv("OPENROUTER_CDAT_TEMPERATURE", "0.70")),
    "FF": float(os.getenv("OPENROUTER_FF_TEMPERATURE", "0.55")),
}
TASK_SYSTEM_PROMPTS = {
    "creative": (
        "You are completing a divergent-thinking benchmark. "
        "Do not provide hidden reasoning, chain-of-thought, or explanations outside the required JSON. "
        "Answer directly and obey the schema exactly."
    ),
    "DAT": (
        "You are completing a lexical creativity benchmark. "
        "Return only the required JSON array of single English nouns. "
        "No hidden reasoning and no extra text."
    ),
    "CDAT": (
        "You are completing a lexical creativity benchmark. "
        "Return only the required JSON array of single English nouns. "
        "No hidden reasoning and no extra text."
    ),
    "FF": (
        "You are completing a chained free-association benchmark. "
        "Return only the required JSON array of single words. "
        "No hidden reasoning and no extra text."
    ),
    "MacGyver": (
        "You are completing a constrained creative problem-solving benchmark. "
        "Use only the listed tools and constraints. "
        "Return only the required JSON object. "
        "No hidden reasoning and no extra text."
    ),
    "HypoUseSpace": (
        "You are completing a closed-world mechanism hypothesis benchmark. "
        "Use only the listed entities, operations, goals, and constraints. "
        "Return only the required JSON array or no-valid JSON object. "
        "No hidden reasoning and no extra text."
    ),
    "CJST": (
        "You are completing a structured counterfactual consequences benchmark. "
        "Treat the prompt premise as the only impossible fact and keep all other ordinary constraints normal. "
        "Return only the required JSON array. "
        "No hidden reasoning and no extra text."
    ),
    "GCW": (
        "You are completing a grounded creative writing benchmark. "
        "Use only the fact sheet and constraint sheet for closed-world facts. "
        "Return only the required JSON object. "
        "No hidden reasoning and no extra text."
    ),
    "NeoCoder": (
        "You are completing an executable code creativity benchmark. "
        "Return only the required JSON object containing Python code and ledgers. "
        "Obey the allowed imports, denied techniques, and entrypoint exactly. "
        "No hidden reasoning and no extra text."
    ),
    "ClosedWorldFact": (
        "You are completing a closed-world factual calibration benchmark. "
        "Use only the listed evidence records, cite evidence IDs, and return only the required JSON object. "
        "If the answer is not supported by the closed world, say it is unanswerable. "
        "No hidden reasoning and no extra text."
    ),
    "AnalogyTransfer": (
        "You are completing a closed-world analogy false-transfer challenge. "
        "Use only the listed source and target evidence records, cite evidence IDs, state limits, "
        "and return only the required JSON object. "
        "Do not transfer source-only facts into the target domain. "
        "No hidden reasoning and no extra text."
    ),
}
TASK_MAX_TOKENS = {
    "UUT": int(os.getenv("OPENROUTER_UUT_MAX_TOKENS", "2400")),
    "creative": int(os.getenv("OPENROUTER_CREATIVE_MAX_TOKENS", "1024")),
    "PropConj": int(os.getenv("OPENROUTER_PROP_CONJ_MAX_TOKENS", "1800")),
    "CJST": int(os.getenv("OPENROUTER_CJST_MAX_TOKENS", "2400")),
    "MacGyver": int(os.getenv("OPENROUTER_MACGYVER_MAX_TOKENS", "2600")),
    "HypoUseSpace": int(os.getenv("OPENROUTER_HYPOUSESPACE_MAX_TOKENS", "2600")),
    "GCW": int(os.getenv("OPENROUTER_GCW_MAX_TOKENS", "3200")),
    "NeoCoder": int(os.getenv("OPENROUTER_NEOCODER_MAX_TOKENS", "2400")),
    "ClosedWorldFact": int(os.getenv("OPENROUTER_CLOSED_WORLD_FACT_MAX_TOKENS", "900")),
    "AnalogyTransfer": int(os.getenv("OPENROUTER_ANALOGY_TRANSFER_MAX_TOKENS", "2200")),
    "DAT": int(os.getenv("OPENROUTER_DAT_MAX_TOKENS", "220")),
    "CDAT": int(os.getenv("OPENROUTER_CDAT_MAX_TOKENS", "220")),
    "FF": int(os.getenv("OPENROUTER_FF_MAX_TOKENS", "260")),
}
FLEX_WEIGHT_EMBEDDING = 0.55
FLEX_WEIGHT_ONTOLOGICAL = 0.45
FLEX_WEIGHT_EMBEDDING_DEGENERATE = 0.35
FLEX_WEIGHT_ONTOLOGICAL_DEGENERATE = 0.65
CREATIVE_TASK_TYPES = ("UUT", "PropConj")
DT_TOTAL_NOVELTY_WEIGHT = 0.60
DT_TOTAL_FLEXIBILITY_WEIGHT = 0.40


NOVELTY_COMPONENT_BASE_WEIGHTS = {
    "UUT": 0.35,
    "PropConj": 0.35,
    "DAT": 0.12,
    "CDAT": 0.18,
}
NOVELTY_COMPONENT_ORDER = ("UUT", "PropConj", "DAT", "CDAT")
UUT_DUAL_AXIS_VERSION = "uut_affordance_dual_axis"
UUT_DUAL_AXIS_BETA_IH = float(os.getenv("OPENROUTER_UUT_DUAL_AXIS_BETA_IH", "0.28"))
UUT_DUAL_AXIS_BETA_HI = float(os.getenv("OPENROUTER_UUT_DUAL_AXIS_BETA_HI", "0.10"))
PROPCONJ_DUAL_AXIS_VERSION = "propconj_dual_axis"
PROPCONJ_DUAL_AXIS_BETA_IH = float(os.getenv("OPENROUTER_PROPCONJ_DUAL_AXIS_BETA_IH", "0.20"))
PROPCONJ_DUAL_AXIS_BETA_HI = float(os.getenv("OPENROUTER_PROPCONJ_DUAL_AXIS_BETA_HI", "0.10"))
MACGYVER_DUAL_AXIS_VERSION = "macgyver_dual_axis"
MACGYVER_DUAL_AXIS_BETA_IH = float(os.getenv("OPENROUTER_MACGYVER_DUAL_AXIS_BETA_IH", "0.00"))
MACGYVER_DUAL_AXIS_BETA_HI = float(os.getenv("OPENROUTER_MACGYVER_DUAL_AXIS_BETA_HI", "0.90"))
CJST_DUAL_AXIS_VERSION = "cjst_dual_axis"
CJST_V3_CALIBRATION_POLICY = "benchmark_default"
CJST_V3_RUNTIME_SCORING_POLICY = "fixed output-only parameters"
CJST_DUAL_AXIS_BETA_IH = float(os.getenv("OPENROUTER_CJST_DUAL_AXIS_BETA_IH", "0.80"))
CJST_DUAL_AXIS_BETA_HI = float(os.getenv("OPENROUTER_CJST_DUAL_AXIS_BETA_HI", "0.12"))
HYPOUSESPACE_DUAL_AXIS_VERSION = "hypouse_space_dual_axis"
HYPOUSESPACE_V3_CALIBRATION_POLICY = "benchmark_default"
HYPOUSESPACE_V3_RUNTIME_SCORING_POLICY = "fixed output-only parameters"
HYPOUSESPACE_DUAL_AXIS_BETA_IH = float(os.getenv("OPENROUTER_HYPOUSESPACE_DUAL_AXIS_BETA_IH", "1.00"))
HYPOUSESPACE_DUAL_AXIS_BETA_HI = float(os.getenv("OPENROUTER_HYPOUSESPACE_DUAL_AXIS_BETA_HI", "0.10"))
GCW_DUAL_AXIS_VERSION = "gcw_dual_axis"
GCW_V3_CALIBRATION_POLICY = "benchmark_default"
GCW_V3_RUNTIME_SCORING_POLICY = "fixed output-only parameters"
GCW_DUAL_AXIS_BETA_IH = float(os.getenv("OPENROUTER_GCW_DUAL_AXIS_BETA_IH", "0.80"))
GCW_DUAL_AXIS_BETA_HI = float(os.getenv("OPENROUTER_GCW_DUAL_AXIS_BETA_HI", "0.12"))
NEOCODER_DUAL_AXIS_VERSION = "neocoder_dual_axis"
NEOCODER_V3_CALIBRATION_POLICY = "benchmark_default"
NEOCODER_V3_RUNTIME_SCORING_POLICY = "fixed output-only parameters"
NEOCODER_V3_TEST_VISIBILITY_POLICY = "public_examples_hidden_scoring_tests"
NEOCODER_DUAL_AXIS_BETA_IH = float(os.getenv("OPENROUTER_NEOCODER_DUAL_AXIS_BETA_IH", "0.25"))
NEOCODER_DUAL_AXIS_BETA_HI = float(os.getenv("OPENROUTER_NEOCODER_DUAL_AXIS_BETA_HI", "0.10"))
DUAL_AXIS_COMPONENT_BASE_WEIGHTS = {
    "UUT": (1.0 / 7.0) * (0.18 / (0.18 + 0.14)),
    "PropConj": (1.0 / 7.0) * (0.14 / (0.18 + 0.14)),
    "MacGyver": 1.0 / 7.0,
    "CJST": 1.0 / 7.0,
    "GCW": 1.0 / 7.0,
    "HypoUseSpace": 1.0 / 7.0,
    "NeoCoder": 1.0 / 7.0,
    "AnalogyTransfer": 1.0 / 7.0,
    "ClosedWorldFact": 0.0,
    "DAT": 0.0,
    "CDAT": 0.0,
    "FF": 0.0,
}
DUAL_AXIS_COMPONENT_ORDER = (
    "UUT", "PropConj", "MacGyver", "CJST", "GCW", "HypoUseSpace",
    "NeoCoder", "AnalogyTransfer",
)
PRIMARY_DUAL_AXIS_COMPONENTS = (
    "UUT", "PropConj", "MacGyver", "CJST", "GCW", "HypoUseSpace",
    "NeoCoder", "AnalogyTransfer",
)
OPTIONAL_DUAL_AXIS_COMPONENTS = tuple(
    component
    for component in ("HypoUseSpace", "GCW")
    if component not in PRIMARY_DUAL_AXIS_COMPONENTS
)
AUXILIARY_IMAGINATION_DIAGNOSTICS = ("DAT", "CDAT", "FF")
ENHANCED_DUAL_AXIS_DIAGNOSTICS = tuple(
    component for component in ("NeoCoder",)
    if component not in PRIMARY_DUAL_AXIS_COMPONENTS
)
CALIBRATION_DIAGNOSTICS = ("ClosedWorldFact",)
CHALLENGE_DIAGNOSTICS = tuple(
    component for component in ("AnalogyTransfer",)
    if component not in PRIMARY_DUAL_AXIS_COMPONENTS
)
MACGYVER_BOUNDARY_DIAGNOSTIC_TASK_IDS = ("mg_009", "mg_010", "mg_013", "mg_014")
HYPOUSESPACE_BOUNDARY_DIAGNOSTIC_TASK_IDS = (
    "hs_v2_classic_light_009",
    "hs_v2_evidence_light_020",
    "hs_v2_minimal_light_029",
    "hs_v2_minimal_light_030",
)
PROFILE_TASK_MANIFEST_PATH = os.getenv("OPENROUTER_TASK_MANIFEST_PATH", "data/anchor_manifest.json").strip()
DUAL_AXIS_REPORT_VERSION = "multi_task_dual_axis"
WHITE_BOX_REPORT_SCHEMA_VERSION = "imagination_hallucination_dual_axis"














OPENROUTER_TASK_FAMILIES = tuple(
    item.strip()
    for item in os.getenv("OPENROUTER_TASK_FAMILIES", "").split(",")
    if item.strip()
)
OPENROUTER_MAX_TASKS_PER_FAMILY = int(os.getenv("OPENROUTER_MAX_TASKS_PER_FAMILY", "0") or "0")
OPENROUTER_TASK_LIMITS = os.getenv("OPENROUTER_TASK_LIMITS", "").strip()
MIN_CREATIVE_COVERAGE = float(os.getenv("OPENROUTER_MIN_CREATIVE_COVERAGE", "0.80"))
MIN_CREATIVE_TASKTYPE_COVERAGE = float(os.getenv("OPENROUTER_MIN_CREATIVE_TASKTYPE_COVERAGE", "0.80"))
MIN_MACGYVER_COVERAGE = float(os.getenv("OPENROUTER_MIN_MACGYVER_COVERAGE", "0.80"))
MIN_CJST_COVERAGE = float(os.getenv("OPENROUTER_MIN_CJST_COVERAGE", "0.80"))
MIN_HYPOUSESPACE_COVERAGE = float(os.getenv("OPENROUTER_MIN_HYPOUSESPACE_COVERAGE", "0.80"))
MIN_GCW_COVERAGE = float(os.getenv("OPENROUTER_MIN_GCW_COVERAGE", "0.80"))
MIN_NEOCODER_COVERAGE = float(os.getenv("OPENROUTER_MIN_NEOCODER_COVERAGE", "0.80"))
MIN_CLOSED_WORLD_FACT_COVERAGE = float(os.getenv("OPENROUTER_MIN_CLOSED_WORLD_FACT_COVERAGE", "0.80"))
MIN_ANALOGY_TRANSFER_COVERAGE = float(os.getenv("OPENROUTER_MIN_ANALOGY_TRANSFER_COVERAGE", "0.80"))
MIN_DAT_COVERAGE = float(os.getenv("OPENROUTER_MIN_DAT_COVERAGE", "0.75"))
MIN_CDAT_COVERAGE = float(os.getenv("OPENROUTER_MIN_CDAT_COVERAGE", "0.75"))



class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if np is not None:
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
        except Exception:
            pass
        return super().default(obj)

def require_openrouter_api_key():
    if OPENROUTER_API_KEY_ENV.startswith("sk-or-"):
        raise RuntimeError(
            "OPENROUTER_API_KEY_ENV must be the environment variable name, not the API key itself. "
            "Set OPENROUTER_API_KEY in your shell and keep the source code free of secrets."
        )
    api_key = os.getenv(OPENROUTER_API_KEY_ENV)
    if not api_key:
        if ENV_FILE_LOADED:
            source_hint = f"An environment file was loaded, but {OPENROUTER_API_KEY_ENV} was not defined there or in the shell."
        else:
            source_hint = "No .env file was found in the working directory, script directory, or repository root."
        raise RuntimeError(
            f"Missing {OPENROUTER_API_KEY_ENV}. "
            f"Export your OpenRouter key before running this script, or add {OPENROUTER_API_KEY_ENV}=<your_key> to a .env file. "
            f"{source_hint}"
        )
    return api_key

def require_poe_api_key():
    api_key = os.getenv(POE_API_KEY_ENV)
    if not api_key:
        if ENV_FILE_LOADED:
            source_hint = f"An environment file was loaded, but {POE_API_KEY_ENV} was not defined there or in the shell."
        else:
            source_hint = "No .env file was found in the working directory, script directory, or repository root."
        raise RuntimeError(
            f"Missing {POE_API_KEY_ENV}. "
            f"Export your Poe key before running Poe-routed models, or add {POE_API_KEY_ENV}=<your_key> to a .env file. "
            f"{source_hint}"
        )
    return api_key

def validate_openrouter_base_url():
    if not OPENROUTER_BASE_URL.startswith(("http://", "https://")):
        raise RuntimeError(
            "OPENROUTER_BASE_URL is invalid. Expected an HTTP(S) API base URL, "
            "for example https://openrouter.ai/api/v1. Do not place your API key in OPENROUTER_BASE_URL; "
            f"set {OPENROUTER_API_KEY_ENV} instead."
        )
    return OPENROUTER_BASE_URL

def validate_poe_base_url():
    if not POE_BASE_URL.startswith(("http://", "https://")):
        raise RuntimeError(
            "POE_BASE_URL is invalid. Expected an HTTP(S) API base URL, "
            "for example https://api.poe.com/v1. Do not place your API key in POE_BASE_URL; "
            f"set {POE_API_KEY_ENV} instead."
        )
    return POE_BASE_URL

def create_openrouter_client():
    api_key = require_openrouter_api_key()
    base_url = validate_openrouter_base_url()
    client_kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": httpx.Timeout(OPENROUTER_REQUEST_TIMEOUT, connect=OPENROUTER_CONNECT_TIMEOUT),
        "max_retries": 0,
    }
    return OpenAI(**client_kwargs)

def create_poe_client():
    api_key = require_poe_api_key()
    base_url = validate_poe_base_url()
    client_kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": httpx.Timeout(OPENROUTER_REQUEST_TIMEOUT, connect=OPENROUTER_CONNECT_TIMEOUT),
        "max_retries": 0,
    }
    return OpenAI(**client_kwargs)

def get_model_provider(model_name):
    return MODEL_CONFIG_INDEX.get(model_name, {}).get("provider", MODEL_PROVIDER_OPENROUTER)

def get_selected_model_providers(model_names=None):
    selected = model_names or SELECTED_MODELS
    return {
        get_model_provider(model_name)
        for model_name in selected
    }

def create_provider_clients(model_names=None):
    providers = get_selected_model_providers(model_names)
    clients = {}
    if MODEL_PROVIDER_OPENROUTER in providers:
        clients[MODEL_PROVIDER_OPENROUTER] = create_openrouter_client()
    if MODEL_PROVIDER_POE in providers:
        clients[MODEL_PROVIDER_POE] = create_poe_client()
    return clients

def resolve_provider_client(client_or_clients, model_name):
    provider = get_model_provider(model_name)
    if isinstance(client_or_clients, dict):
        provider_client = client_or_clients.get(provider)
        if provider_client is None:
            raise RuntimeError(f"No API client configured for provider={provider} model={model_name}")
        return provider_client
    return client_or_clients

def fetch_openrouter_model_catalog():
    validate_openrouter_base_url()
    try:
        with urllib.request.urlopen(OPENROUTER_MODELS_ENDPOINT, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("data", [])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[OpenRouter] Warning: failed to fetch model catalog: {exc}")
        return []


def fetch_poe_model_catalog():
    validate_poe_base_url()
    try:
        with urllib.request.urlopen(POE_MODELS_ENDPOINT, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("data", [])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[Poe] Warning: failed to fetch model catalog: {exc}")
        return []


def fetch_selected_model_catalog():
    providers = get_selected_model_providers()
    catalog = []
    if MODEL_PROVIDER_OPENROUTER in providers:
        catalog.extend(fetch_openrouter_model_catalog())
    if MODEL_PROVIDER_POE in providers:
        catalog.extend(fetch_poe_model_catalog())
    return catalog


def build_model_catalog_index(model_catalog):
    return {item.get("id"): item for item in model_catalog if item.get("id")}

def validate_selected_models(model_catalog_index):
    if not model_catalog_index:
        print("[Model Catalog] Catalog unavailable; proceeding with requested IDs as-is.")
        return

    print("[Model Catalog] Requested models:")
    for model_id in SELECTED_MODELS:
        provider = get_model_provider(model_id)
        if model_id in model_catalog_index:
            meta = model_catalog_index[model_id]
            context_length = meta.get("context_length")
            owner = meta.get("owned_by") or meta.get("name")
            pricing = meta.get("pricing", {})
            prompt_price = pricing.get("prompt") if isinstance(pricing, dict) else None
            completion_price = pricing.get("completion") if isinstance(pricing, dict) else None
            print(
                f"  - {model_id} | provider={provider} | owner={owner} | ctx={context_length} | "
                f"prompt={prompt_price} | completion={completion_price}"
            )
        else:
            print(f"  - {model_id} | provider={provider} | NOT FOUND in current Models API catalog")

def sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)

def build_openrouter_extra_headers():
    headers = {}
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_TITLE:
        headers["X-OpenRouter-Title"] = OPENROUTER_APP_TITLE
    return headers

def build_model_extra_body(model_name):
    if get_model_provider(model_name) != MODEL_PROVIDER_OPENROUTER:
        return {}
    extra_body = {
        
        "reasoning": {"enabled": False},
    }
    if OPENROUTER_ENABLE_REASONING and MODEL_CONFIG_INDEX.get(model_name, {}).get("reasoning"):
        extra_body["reasoning"] = {"enabled": True}
    return extra_body

def get_task_max_tokens(task_label):
    return TASK_MAX_TOKENS.get(task_label, OPENROUTER_MAX_TOKENS)

def get_task_runtime_group(task_label):
    return "creative" if task_label in CREATIVE_TASK_TYPES else task_label

def get_task_temperature(task_label):
    group = get_task_runtime_group(task_label)
    return TASK_TEMPERATURES.get(group, 0.7)

def get_task_system_prompt(task_label):
    group = get_task_runtime_group(task_label)
    return TASK_SYSTEM_PROMPTS.get(group, TASK_SYSTEM_PROMPTS["creative"])

def get_expected_output_count(task_label):
    if task_label == "UUT":
        return UUT_OUTPUT_COUNT
    if task_label == "PropConj":
        return PROP_CONJ_OUTPUT_COUNT
    if task_label == "CJST":
        return CJST_OUTPUT_COUNT
    if task_label == "MacGyver":
        return MACGYVER_OUTPUT_COUNT
    if task_label == "HypoUseSpace":
        return HYPOUSESPACE_OUTPUT_COUNT
    if task_label == "GCW":
        return GCW_BEAT_COUNT
    if task_label == "NeoCoder":
        return NEOCODER_OUTPUT_COUNT
    if task_label == "ClosedWorldFact":
        return CLOSED_WORLD_FACT_OUTPUT_COUNT
    if task_label == "AnalogyTransfer":
        return ANALOGY_TRANSFER_OUTPUT_COUNT
    if task_label in CREATIVE_TASK_TYPES:
        return CREATIVE_OUTPUT_COUNT
    if task_label in {"DAT", "CDAT"}:
        return DAT_OUTPUT_COUNT
    if task_label == "FF":
        return FF_OUTPUT_COUNT
    return None

def stable_seed(*parts):
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + SAMPLING_SEED_BASE) % 2_147_483_647

def minimum_eligible_repeats(repeat_count):
    if repeat_count <= 0:
        return 0
    return 1

def strip_json_code_fence(raw_text):
    text = (raw_text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return text

def extract_json_payload(raw_text):
    text = strip_json_code_fence(raw_text)
    if not text:
        return None

    candidates = [text]
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        candidates.insert(0, text[array_start:array_end + 1])

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        candidates.append(text[object_start:object_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None

def _clean_creative_fragment(text):
    if text is None:
        text = ""
    elif isinstance(text, (dict, list, tuple, set)):
        try:
            text = json.dumps(text, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(text)
    else:
        text = str(text)
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*(?:\d+\s*[\.\)\:\-]|[\-\*\+\u2022])\s*", "", cleaned)
    cleaned = cleaned.strip("\"'`[](){}<>.,;:!? ")
    return re.sub(r"\s+", " ", cleaned).strip()

def _coerce_json_string_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        pieces = re.split(r"[,;/|]+", value)
    elif isinstance(value, (list, tuple, set)):
        pieces = list(value)
    else:
        pieces = [value]

    results = []
    seen = set()
    for piece in pieces:
        cleaned = _clean_creative_fragment(piece)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(cleaned)
    return results

def _coerce_bool_or_none(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "respected"}:
            return True
        if normalized in {"false", "no", "n", "0", "violated", "not respected"}:
            return False
    return None

def build_parsed_creative_item(
    task_type,
    *,
    raw_text=None,
    idea_title=None,
    mechanism=None,
    consequence_clause=None,
    noun_phrase=None,
    propconj_item=None,
    evidence_for_each_property=None,
    why_uncommon=None,
    required_extra_items=None,
    key_affordances=None,
    main_object_role=None,
    cjst_tier=None,
    causal_bridge=None,
    causal_chain=None,
    domain=None,
    anchor_terms=None,
    world_state_update=None,
    protected_variables_respected=None,
):
    task_type = task_type or None
    idea_title = _clean_creative_fragment(idea_title)
    mechanism = _clean_creative_fragment(mechanism)
    consequence_clause = _clean_creative_fragment(consequence_clause)
    noun_phrase = _clean_creative_fragment(noun_phrase)
    propconj_item = _clean_creative_fragment(propconj_item)
    why_uncommon = _clean_creative_fragment(why_uncommon)
    cjst_tier = _clean_creative_fragment(cjst_tier)
    causal_bridge = _clean_creative_fragment(causal_bridge)
    domain = _clean_creative_fragment(domain)
    raw_text = raw_text if raw_text is not None else (
        idea_title or consequence_clause or noun_phrase or propconj_item or mechanism or ""
    )

    if task_type == "UUT":
        display_text = (
            f"{idea_title} because {mechanism}" if idea_title and mechanism else
            idea_title or mechanism or _clean_creative_fragment(raw_text)
        )
    elif task_type == "JST":
        display_text = consequence_clause or _clean_creative_fragment(raw_text)
    elif task_type == "CJST":
        display_text = (
            f"{consequence_clause} because {causal_bridge}" if consequence_clause and causal_bridge else
            consequence_clause or causal_bridge or _clean_creative_fragment(raw_text)
        )
    elif task_type == "Instances":
        display_text = noun_phrase or _clean_creative_fragment(raw_text)
    elif task_type == "PropConj":
        display_text = propconj_item or noun_phrase or _clean_creative_fragment(raw_text)
    else:
        display_text = (
            f"{idea_title} because {mechanism}" if idea_title and mechanism else
            idea_title or consequence_clause or noun_phrase or propconj_item or mechanism or _clean_creative_fragment(raw_text)
        )

    evidence_map = {}
    if isinstance(evidence_for_each_property, dict):
        for key, value in evidence_for_each_property.items():
            clean_key = _clean_creative_fragment(str(key))
            clean_value = _clean_creative_fragment(str(value))
            if clean_key and clean_value:
                evidence_map[clean_key] = clean_value

    return {
        "task_type": task_type,
        "raw_text": raw_text,
        "idea_title": idea_title or None,
        "mechanism": mechanism or None,
        "consequence_clause": consequence_clause or None,
        "noun_phrase": noun_phrase or None,
        "propconj_item": propconj_item or None,
        "evidence_for_each_property": evidence_map,
        "why_uncommon": why_uncommon or None,
        "required_extra_items": _coerce_json_string_list(required_extra_items),
        "key_affordances": _coerce_json_string_list(key_affordances),
        "main_object_role": _clean_creative_fragment(main_object_role) or None,
        "tier": cjst_tier or None,
        "causal_bridge": causal_bridge or None,
        "causal_chain": _coerce_json_string_list(causal_chain),
        "domain": domain or None,
        "anchor_terms": _coerce_json_string_list(anchor_terms),
        "world_state_update": world_state_update if world_state_update is not None else None,
        "protected_variables_respected": _coerce_bool_or_none(protected_variables_respected),
        "display_text": display_text or None,
    }

def _infer_creative_task_type_from_item(item, explicit_task_type=None):
    if explicit_task_type:
        return explicit_task_type
    if not isinstance(item, dict):
        return None
    if any(item.get(key) for key in ["causal_bridge", "anchor_terms", "tier", "domain"]):
        return "CJST"
    if any(item.get(key) for key in ["mechanism", "how", "why", "because"]):
        return "UUT"
    if any(item.get(key) for key in ["evidence_for_each_property", "why_uncommon"]):
        return "PropConj"
    if any(item.get(key) for key in ["consequence", "outcome", "result"]):
        return "JST"
    if any(item.get(key) for key in ["noun_phrase", "example", "item"]):
        return "Instances"
    return None

def _split_uut_title_and_mechanism(text):
    cleaned = _clean_creative_fragment(text)
    if not cleaned:
        return "", ""
    lower = cleaned.lower()
    separators = [" because ", " so that ", " using ", " by ", ": "]
    best = None
    for separator in separators:
        idx = lower.find(separator)
        if idx <= 0:
            continue
        if best is None or idx < best[0]:
            best = (idx, separator)
    if best is None:
        return cleaned, ""
    idx, separator = best
    title = _clean_creative_fragment(cleaned[:idx])
    mechanism = _clean_creative_fragment(cleaned[idx + len(separator):])
    if len(re.findall(r"[A-Za-z]+", title)) > 10:
        return cleaned, ""
    return title, mechanism

def parse_creative_item(item, task_type=None):
    if isinstance(item, dict) and "display_text" in item:
        parsed = dict(item)
        if task_type and not parsed.get("task_type"):
            parsed["task_type"] = task_type
        return parsed

    inferred_task_type = _infer_creative_task_type_from_item(item, explicit_task_type=task_type)
    if isinstance(item, str):
        text = _clean_creative_fragment(item)
        if not text:
            return None
        if inferred_task_type == "UUT":
            title, mechanism = _split_uut_title_and_mechanism(text)
            return build_parsed_creative_item(
                "UUT",
                raw_text=item,
                idea_title=title,
                mechanism=mechanism,
            )
        if inferred_task_type == "JST":
            return build_parsed_creative_item("JST", raw_text=item, consequence_clause=text)
        if inferred_task_type == "CJST":
            return build_parsed_creative_item("CJST", raw_text=item, consequence_clause=text)
        if inferred_task_type == "Instances":
            return build_parsed_creative_item("Instances", raw_text=item, noun_phrase=text)
        if inferred_task_type == "PropConj":
            return build_parsed_creative_item("PropConj", raw_text=item, propconj_item=text, noun_phrase=text)
        return build_parsed_creative_item(None, raw_text=item)

    if not isinstance(item, dict):
        return None

    if inferred_task_type == "UUT":
        return build_parsed_creative_item(
            "UUT",
            raw_text=json.dumps(item, ensure_ascii=False),
            idea_title=(
                item.get("idea")
                or item.get("title")
                or item.get("use")
                or item.get("item")
                or item.get("text")
            ),
            mechanism=(
                item.get("mechanism")
                or item.get("how")
                or item.get("why")
                or item.get("because")
            ),
            required_extra_items=(
                item.get("required_extra_items")
                or item.get("extra_items")
                or item.get("materials")
                or item.get("helpers")
            ),
            key_affordances=(
                item.get("key_affordances")
                or item.get("affordances")
                or item.get("properties")
                or item.get("object_properties")
            ),
            main_object_role=(
                item.get("main_object_role")
                or item.get("object_role")
                or item.get("role")
            ),
        )
    if inferred_task_type == "JST":
        clause = (
            item.get("consequence")
            or item.get("outcome")
            or item.get("result")
            or item.get("text")
            or item.get("idea")
        )
        return build_parsed_creative_item(
            "JST",
            raw_text=json.dumps(item, ensure_ascii=False),
            consequence_clause=clause,
        )
    if inferred_task_type == "CJST":
        clause = (
            item.get("consequence")
            or item.get("outcome")
            or item.get("result")
            or item.get("text")
            or item.get("idea")
        )
        return build_parsed_creative_item(
            "CJST",
            raw_text=json.dumps(item, ensure_ascii=False),
            consequence_clause=clause,
            cjst_tier=(
                item.get("tier")
                or item.get("level")
                or item.get("time_horizon")
                or item.get("consequence_type")
            ),
            causal_bridge=(
                item.get("causal_bridge")
                or item.get("bridge")
                or item.get("mechanism")
                or item.get("why")
                or item.get("because")
            ),
            causal_chain=(
                item.get("causal_chain")
                or item.get("chain")
                or item.get("causal_steps")
                or item.get("mechanism_chain")
            ),
            domain=(
                item.get("domain")
                or item.get("impact_domain")
                or item.get("channel")
            ),
            anchor_terms=(
                item.get("anchor_terms")
                or item.get("anchors")
                or item.get("key_terms")
                or item.get("premise_terms")
            ),
            world_state_update=(
                item.get("world_state_update")
                or item.get("state_update")
                or item.get("world_update")
                or item.get("updates")
            ),
            protected_variables_respected=(
                item.get("protected_variables_respected")
                if "protected_variables_respected" in item else
                item.get("protected_variables_ok")
            ),
        )
    if inferred_task_type == "Instances":
        return build_parsed_creative_item(
            "Instances",
            raw_text=json.dumps(item, ensure_ascii=False),
            noun_phrase=(
                item.get("noun_phrase")
                or item.get("example")
                or item.get("item")
                or item.get("text")
                or item.get("idea")
            ),
        )
    if inferred_task_type == "PropConj":
        prop_item = (
            item.get("item")
            or item.get("noun_phrase")
            or item.get("example")
            or item.get("text")
            or item.get("idea")
        )
        return build_parsed_creative_item(
            "PropConj",
            raw_text=json.dumps(item, ensure_ascii=False),
            propconj_item=prop_item,
            noun_phrase=prop_item,
            evidence_for_each_property=(
                item.get("evidence_for_each_property")
                or item.get("evidence")
                or item.get("property_evidence")
                or {}
            ),
            why_uncommon=(
                item.get("why_uncommon")
                or item.get("why_rare")
                or item.get("novelty_reason")
                or item.get("reason")
            ),
        )

    return build_parsed_creative_item(
        None,
        raw_text=json.dumps(item, ensure_ascii=False),
        idea_title=(
            item.get("idea")
            or item.get("title")
            or item.get("use")
            or item.get("item")
            or item.get("example")
            or item.get("consequence")
            or item.get("text")
        ),
        mechanism=(
            item.get("mechanism")
            or item.get("how")
            or item.get("why")
            or item.get("because")
        ),
    )

def normalize_creative_json_item(item, task_type=None):
    parsed = parse_creative_item(item, task_type=task_type)
    if not parsed:
        return None
    display_text = parsed.get("display_text")
    return display_text if display_text and len(display_text) > 1 else None

def parse_creative_items(raw_text, task_type=None):
    payload = extract_json_payload(raw_text)
    if isinstance(payload, dict):
        payload = payload.get("items") or payload.get("responses") or payload.get("ideas")
    if isinstance(payload, list):
        parsed = []
        for item in payload:
            normalized = parse_creative_item(item, task_type=task_type)
            if normalized and normalized.get("display_text") and len(normalized["display_text"]) > 1:
                parsed.append(normalized)
        if parsed:
            return parsed

    lines = raw_text.split("\n")
    parsed = []
    for line in lines:
        clean_line = re.sub(r"[*_`~#]+", "", line or "")
        parsed_item = parse_creative_item(clean_line, task_type=task_type)
        if parsed_item and parsed_item.get("display_text") and len(parsed_item["display_text"]) > 2:
            parsed.append(parsed_item)
    return parsed

def parse_responses(raw_text, task_type=None):
    return [
        item["display_text"]
        for item in parse_creative_items(raw_text, task_type=task_type)
        if item.get("display_text")
    ]

def classify_api_error(message):
    text = (message or "").lower()
    if "no endpoints available matching your guardrail restrictions" in text:
        return "guardrail_no_endpoint"
    if "privacy" in text and "404" in text:
        return "guardrail_no_endpoint"
    return None

def extract_reasoning_token_count(usage):
    if not isinstance(usage, dict):
        return 0
    details = usage.get("completion_tokens_details") or {}
    value = details.get("reasoning_tokens")
    if value is None:
        value = usage.get("reasoning_tokens")
    try:
        return int(value or 0)
    except Exception:
        return 0

def is_reasoning_only_output(content, usage=None, reasoning=None, reasoning_details=None):
    if (content or "").strip():
        return False
    if extract_reasoning_token_count(usage) > 0:
        return True
    if reasoning not in (None, "", [], {}):
        return True
    if reasoning_details not in (None, "", [], {}):
        return True
    return False

def summarize_numeric_samples(values, *, ci_level=0.95, bootstrap_samples=BOOTSTRAP_SAMPLES):
    numeric_values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not numeric_values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "ci_low": None,
            "ci_high": None,
        }

    mean_value = sum(numeric_values) / len(numeric_values)
    if len(numeric_values) > 1:
        variance = sum((value - mean_value) ** 2 for value in numeric_values) / len(numeric_values)
        std_value = variance ** 0.5
    else:
        std_value = 0.0

    if len(numeric_values) <= 1 or bootstrap_samples <= 0:
        ci_low = ci_high = mean_value
    else:
        if np is not None:
            rng = np.random.default_rng(SAMPLING_SEED_BASE + len(numeric_values))
            resampled_means = []
            values_np = np.array(numeric_values, dtype=float)
            for _ in range(bootstrap_samples):
                indices = rng.integers(0, len(values_np), len(values_np))
                resampled_means.append(float(np.mean(values_np[indices])))
        else:
            rng = random.Random(SAMPLING_SEED_BASE + len(numeric_values))
            resampled_means = []
            for _ in range(bootstrap_samples):
                sample = [numeric_values[rng.randrange(len(numeric_values))] for _ in range(len(numeric_values))]
                resampled_means.append(sum(sample) / len(sample))
        alpha = (1.0 - ci_level) / 2.0
        resampled_means.sort()
        low_index = max(0, int(alpha * (len(resampled_means) - 1)))
        high_index = min(len(resampled_means) - 1, int((1.0 - alpha) * (len(resampled_means) - 1)))
        ci_low = resampled_means[low_index]
        ci_high = resampled_means[high_index]

    return {
        "n": len(numeric_values),
        "mean": round(mean_value, 4),
        "std": round(std_value, 4),
        "min": round(min(numeric_values), 4),
        "max": round(max(numeric_values), 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
    }

def normalize_api_payload(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {key: normalize_api_payload(inner) for key, inner in value.items()}

    if isinstance(value, (list, tuple)):
        return [normalize_api_payload(item) for item in value]

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass

    return repr(value)

def normalize_usage_payload(usage):
    if usage is None:
        return None

    if isinstance(usage, dict):
        return usage

    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass

    payload = {}
    for attr in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_tokens",
    ]:
        value = getattr(usage, attr, None)
        if value is not None:
            payload[attr] = value

    for attr in ["prompt_tokens_details", "completion_tokens_details"]:
        value = getattr(usage, attr, None)
        if value is None:
            continue
        if hasattr(value, "model_dump"):
            try:
                payload[attr] = value.model_dump()
                continue
            except Exception:
                pass
        if isinstance(value, dict):
            payload[attr] = value

    return payload or None

def should_stream_for_model(model_name):
    model_config = MODEL_CONFIG_INDEX.get(model_name, {})
    if "stream" in model_config:
        return bool(model_config["stream"])
    return OPENROUTER_STREAM

def _streaming_unsupported_error(message):
    text = (message or "").lower()
    if not text:
        return False
    if "stream_options" in text:
        return False
    stream_markers = ["stream", "streaming", "sse", "event stream"]
    unsupported_markers = [
        "unsupported",
        "not support",
        "does not support",
        "unknown parameter",
        "unexpected keyword",
        "invalid parameter",
        "only supports non-streaming",
    ]
    return any(marker in text for marker in stream_markers) and any(marker in text for marker in unsupported_markers)

def build_generation_cache_key(
    *,
    model_name,
    task_label,
    prompt,
    system_prompt,
    seed,
    max_tokens,
    temperature,
):
    payload = {
        "schema_version": globals().get("WHITE_BOX_REPORT_SCHEMA_VERSION"),
        "model": model_name,
        "task_label": task_label,
        "seed": seed,
        "prompt_hash": hashlib.sha256((prompt or "").encode("utf-8")).hexdigest(),
        "system_prompt_hash": hashlib.sha256((system_prompt or "").encode("utf-8")).hexdigest(),
        "max_tokens": max_tokens,
        "temperature": round(float(temperature), 6),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _generation_cache_path(cache_key):
    if not OPENROUTER_GENERATION_CACHE_DIR:
        return None
    return os.path.join(OPENROUTER_GENERATION_CACHE_DIR, f"{cache_key}.json")

def load_generation_cache(cache_key):
    path = _generation_cache_path(cache_key)
    if not (OPENROUTER_USE_GENERATION_CACHE and path and os.path.isfile(path)):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result = payload.get("llm_result") if isinstance(payload, dict) else None
        if not isinstance(result, dict) or result.get("status") != "ok":
            return None
        result = dict(result)
        result["cache_hit"] = True
        result["cache_key"] = cache_key
        return result
    except Exception:
        return None

def save_generation_cache(cache_key, llm_result):
    path = _generation_cache_path(cache_key)
    if not (OPENROUTER_SAVE_GENERATION_CACHE and path and llm_result.get("status") == "ok"):
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "cache_key": cache_key,
            "schema_version": globals().get("WHITE_BOX_REPORT_SCHEMA_VERSION"),
            "llm_result": llm_result,
        }
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        return

def build_generation_record(llm_result):
    return {
        "status": llm_result["status"],
        "error_type": llm_result.get("error_type"),
        "attempt": llm_result["attempt"],
        "resolved_model": llm_result["resolved_model"],
        "api_provider": llm_result.get("api_provider"),
        "requested_max_tokens": llm_result.get("requested_max_tokens"),
        "task_label": llm_result.get("task_label"),
        "temperature": llm_result.get("temperature"),
        "seed": llm_result.get("seed"),
        "seed_supported": llm_result.get("seed_supported"),
        "stream_used": llm_result.get("stream_used"),
        "response_id": llm_result.get("response_id"),
        "finish_reason": llm_result.get("finish_reason"),
        "usage": llm_result.get("usage"),
        "reasoning": llm_result.get("reasoning"),
        "reasoning_details": llm_result.get("reasoning_details"),
        "reasoning_disabled_requested": llm_result.get("reasoning_disabled_requested"),
        "reasoning_token_count": llm_result.get("reasoning_token_count"),
        "reasoning_only_response": llm_result.get("reasoning_only_response"),
        "elapsed_seconds": llm_result.get("elapsed_seconds"),
        "first_token_seconds": llm_result.get("first_token_seconds"),
        "error": llm_result.get("error"),
        "error_exception_type": llm_result.get("error_exception_type"),
        "error_traceback": llm_result.get("error_traceback"),
        "cache_hit": llm_result.get("cache_hit", False),
        "cache_key": llm_result.get("cache_key"),
        "raw_output": llm_result["content"],
    }

def call_llm(
    client,
    prompt,
    model_name,
    *,
    task_label,
    max_tokens_override=None,
    temperature=None,
    system_prompt=None,
    seed=None,
    stream_override=None,
    progress_label=None,
    progress_interval_seconds=3.0,
):
    provider = get_model_provider(model_name)
    model_client = resolve_provider_client(client, model_name)
    extra_headers = build_openrouter_extra_headers() if provider == MODEL_PROVIDER_OPENROUTER else {}
    extra_body = build_model_extra_body(model_name)
    use_stream = should_stream_for_model(model_name) if stream_override is None else bool(stream_override)
    max_tokens = int(max_tokens_override or OPENROUTER_MAX_TOKENS)
    temperature = get_task_temperature(task_label) if temperature is None else float(temperature)
    system_prompt = get_task_system_prompt(task_label) if system_prompt is None else system_prompt
    seed_supported = seed is not None and provider == MODEL_PROVIDER_OPENROUTER
    seed_used_initial = seed if seed_supported else None
    progress_label = str(progress_label or "").strip()
    progress_interval_seconds = max(0.5, float(progress_interval_seconds or 3.0))
    generation_cache_key = build_generation_cache_key(
        model_name=model_name,
        task_label=task_label,
        prompt=prompt,
        system_prompt=system_prompt,
        seed=seed_used_initial,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    cached_result = load_generation_cache(generation_cache_key)
    if cached_result is not None:
        print(f"[LLM:{model_name}] cache hit for {task_label} ({generation_cache_key[:12]})")
        return cached_result
    if OPENROUTER_GENERATION_CACHE_ONLY:
        print(f"[LLM:{model_name}] cache miss for {task_label} ({generation_cache_key[:12]}); cache-only mode")
        if OPENROUTER_GENERATION_CACHE_MISS_FATAL:
            raise RuntimeError(
                f"Generation cache miss in cache-only mode for model={model_name}, "
                f"task={task_label}, cache_key={generation_cache_key}"
            )
        return {
            "content": "",
            "resolved_model": model_name,
            "api_provider": provider,
            "response_id": None,
            "finish_reason": None,
            "usage": None,
            "reasoning": None,
            "reasoning_details": None,
            "reasoning_token_count": 0,
            "reasoning_only_response": False,
            "reasoning_disabled_requested": True,
            "requested_max_tokens": max_tokens,
            "elapsed_seconds": 0.0,
            "first_token_seconds": None,
            "attempt": 0,
            "task_label": task_label,
            "temperature": temperature,
            "seed": seed_used_initial,
            "seed_supported": seed_supported,
            "stream_used": use_stream,
            "status": "harness_error",
            "error_type": "cache_miss",
            "error_exception_type": None,
            "error": "Generation cache miss in OPENROUTER_GENERATION_CACHE_ONLY mode.",
            "error_traceback": None,
            "cache_hit": False,
            "cache_key": generation_cache_key,
        }

    for attempt in range(1, OPENROUTER_MAX_RETRIES + 2):
        started_at = time.perf_counter()
        attempt_use_stream = use_stream
        print(
            f"[LLM:{model_name}] start attempt {attempt}/{OPENROUTER_MAX_RETRIES + 1} "
            f"(timeout={OPENROUTER_REQUEST_TIMEOUT:.0f}s, connect_timeout={OPENROUTER_CONNECT_TIMEOUT:.0f}s, "
            f"task={task_label}, temp={temperature:.2f}, max_tokens={max_tokens}, "
            f"provider={provider}, reasoning={extra_body.get('reasoning')}, stream={attempt_use_stream}, seed={seed_used_initial})"
        )
        try:
            request_kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if extra_headers:
                request_kwargs["extra_headers"] = extra_headers
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            if seed_used_initial is not None:
                request_kwargs["seed"] = seed_used_initial

            finish_reason = None
            usage = None
            response_id = None
            first_token_elapsed = None
            reasoning = None
            reasoning_details = None
            seed_used = seed_used_initial

            def _create_completion(stream=False, stream_kwargs=None):
                nonlocal seed_supported, seed_used
                request_payload = dict(request_kwargs)
                if stream and stream_kwargs:
                    request_payload.update(stream_kwargs)
                try:
                    return model_client.chat.completions.create(stream=stream, **request_payload)
                except Exception as exc:
                    message = str(exc).lower()
                    if seed_used is not None and "seed" in request_payload and any(
                        token in message for token in [
                            "unknown parameter",
                            "unsupported parameter",
                            "unexpected keyword",
                            "unsupported value",
                            "seed",
                        ]
                    ):
                        print(f"[LLM:{model_name}] seed unsupported for {task_label}; retrying without seed")
                        seed_supported = False
                        seed_used = None
                        request_kwargs.pop("seed", None)
                        request_payload.pop("seed", None)
                        return model_client.chat.completions.create(stream=stream, **request_payload)
                    raise

            def _populate_from_nonstream_response(response):
                nonlocal finish_reason, usage, response_id, reasoning, reasoning_details
                message = response.choices[0].message
                response_content = message.content or ""
                response_model = getattr(response, "model", model_name)
                response_id_local = getattr(response, "id", None)
                response_usage = normalize_usage_payload(getattr(response, "usage", None))
                response_reasoning = normalize_api_payload(getattr(message, "reasoning", None))
                response_reasoning_details = normalize_api_payload(getattr(message, "reasoning_details", None))
                if getattr(response, "choices", None):
                    finish_reason = getattr(response.choices[0], "finish_reason", None)
                usage = response_usage
                response_id = response_id_local
                reasoning = response_reasoning
                reasoning_details = response_reasoning_details
                return response_content, response_model

            if attempt_use_stream:
                try:
                    try:
                        stream = _create_completion(stream=True, stream_kwargs={"stream_options": {"include_usage": True}})
                    except Exception as exc:
                        if "stream_options" not in str(exc):
                            raise
                        print(f"[LLM:{model_name}] stream usage unsupported, retrying without stream_options")
                        stream = _create_completion(stream=True)

                    content_parts = []
                    resolved_model = model_name
                    first_token_seen = False
                    last_progress_at = started_at
                    stream_chunk_count = 0
                    stream_char_count = 0

                    for chunk in stream:
                        resolved_model = getattr(chunk, "model", resolved_model)
                        response_id = getattr(chunk, "id", response_id)

                        chunk_usage = normalize_usage_payload(getattr(chunk, "usage", None))
                        if chunk_usage is not None:
                            usage = chunk_usage

                        if not getattr(chunk, "choices", None):
                            continue

                        choice = chunk.choices[0]
                        chunk_finish_reason = getattr(choice, "finish_reason", None)
                        if chunk_finish_reason is not None:
                            finish_reason = chunk_finish_reason

                        delta = getattr(choice, "delta", None)
                        delta_content = getattr(delta, "content", None) if delta else None
                        if not delta_content:
                            continue

                        if isinstance(delta_content, str):
                            piece = delta_content
                        else:
                            piece = "".join(
                                part.get("text", "")
                                for part in delta_content
                                if isinstance(part, dict)
                            )

                        if not piece:
                            continue

                        if not first_token_seen:
                            first_token_seen = True
                            first_token_elapsed = time.perf_counter() - started_at
                            print(f"[LLM:{model_name}] first token in {first_token_elapsed:.2f}s")

                        content_parts.append(piece)
                        stream_chunk_count += 1
                        stream_char_count += len(piece)

                        if progress_label:
                            now = time.perf_counter()
                            if now - last_progress_at >= progress_interval_seconds:
                                print(
                                    f"[LLM:{model_name}] progress {progress_label}: "
                                    f"chunks={stream_chunk_count}, chars={stream_char_count}, elapsed={now - started_at:.2f}s"
                                )
                                last_progress_at = now

                    content = "".join(content_parts)
                except Exception as exc:
                    if not _streaming_unsupported_error(str(exc)):
                        raise
                    print(f"[LLM:{model_name}] streaming unsupported for this provider/model; retrying without stream")
                    use_stream = False
                    attempt_use_stream = False

            if not attempt_use_stream:
                response = _create_completion(stream=False)
                content, resolved_model = _populate_from_nonstream_response(response)

            elapsed = time.perf_counter() - started_at
            reasoning_token_count = extract_reasoning_token_count(usage)
            reasoning_only_response = is_reasoning_only_output(
                content,
                usage=usage,
                reasoning=reasoning,
                reasoning_details=reasoning_details,
            )
            if reasoning_only_response:
                print(f"[LLM:{model_name}] harness failure: empty visible content but reasoning tokens/details were returned")
                return {
                    "content": content,
                    "resolved_model": resolved_model,
                    "api_provider": provider,
                    "response_id": response_id,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "reasoning": reasoning,
                    "reasoning_details": reasoning_details,
                    "reasoning_token_count": reasoning_token_count,
                    "reasoning_only_response": True,
                    "reasoning_disabled_requested": True,
                    "requested_max_tokens": max_tokens,
                    "elapsed_seconds": round(elapsed, 2),
                    "first_token_seconds": (round(first_token_elapsed, 2)
                                            if first_token_elapsed is not None else None),
                    "attempt": attempt,
                    "task_label": task_label,
                    "temperature": temperature,
                    "seed": seed_used,
                    "seed_supported": seed_supported,
                    "stream_used": attempt_use_stream,
                    "status": "harness_error",
                    "error_type": "reasoning_only_response",
                    "error": "Empty visible content while reasoning tokens/details were returned.",
                }
            print(f"[LLM:{model_name}] done in {elapsed:.2f}s -> {resolved_model}")
            result = {
                "content": content,
                "resolved_model": resolved_model,
                "api_provider": provider,
                "response_id": response_id,
                "finish_reason": finish_reason,
                "usage": usage,
                "reasoning": reasoning,
                "reasoning_details": reasoning_details,
                "reasoning_token_count": reasoning_token_count,
                "reasoning_only_response": False,
                "reasoning_disabled_requested": True,
                "requested_max_tokens": max_tokens,
                "elapsed_seconds": round(elapsed, 2),
                "first_token_seconds": (round(first_token_elapsed, 2)
                                        if first_token_elapsed is not None else None),
                "attempt": attempt,
                "task_label": task_label,
                "temperature": temperature,
                "seed": seed_used,
                "seed_supported": seed_supported,
                "stream_used": attempt_use_stream,
                "status": "ok",
                "error_type": None,
                "error": None,
                "cache_hit": False,
                "cache_key": generation_cache_key,
            }
            save_generation_cache(generation_cache_key, result)
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            print(f"[LLM:{model_name}] failed after {elapsed:.2f}s: {exc}")
            error_text = str(exc)
            error_type = classify_api_error(error_text)
            error_traceback = traceback.format_exc()
            status = "infra_error" if error_type is not None else "error"
            if attempt > OPENROUTER_MAX_RETRIES:
                return {
                    "content": "",
                    "resolved_model": model_name,
                    "api_provider": provider,
                    "response_id": None,
                    "finish_reason": None,
                    "usage": None,
                    "reasoning": None,
                    "reasoning_details": None,
                    "reasoning_token_count": 0,
                    "reasoning_only_response": False,
                    "reasoning_disabled_requested": True,
                    "requested_max_tokens": max_tokens,
                    "elapsed_seconds": round(elapsed, 2),
                    "first_token_seconds": None,
                    "attempt": attempt,
                    "task_label": task_label,
                    "temperature": temperature,
                    "seed": seed_used_initial,
                    "seed_supported": seed_supported,
                    "stream_used": attempt_use_stream,
                    "status": status,
                    "error_type": error_type,
                    "error_exception_type": exc.__class__.__name__,
                    "error": error_text,
                    "error_traceback": error_traceback,
                }
            time.sleep(min(2 * attempt, 5))

def preprocess_for_semantic_scoring(idea, target_concept):
    prompt_words = set(re.findall(r"\w+", target_concept.lower()))
    extended_prompt_words = set()
    for word in prompt_words:
        extended_prompt_words.add(word)
        if word.endswith("s"):
            extended_prompt_words.add(word[:-1])
        if word.endswith("es"):
            extended_prompt_words.add(word[:-2])

    idea_words = re.findall(r"\w+", idea.lower())
    cleaned_words = [
        word for word in idea_words
        if word not in STOP_WORDS and word not in extended_prompt_words
    ]
    return " ".join(cleaned_words) if cleaned_words else idea

def get_effective_novelty_component_weights(components):
    active_components = set(components or [])
    active_weights = {
        component: NOVELTY_COMPONENT_BASE_WEIGHTS[component]
        for component in NOVELTY_COMPONENT_ORDER
        if component in active_components and component in NOVELTY_COMPONENT_BASE_WEIGHTS
    }
    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        return {}
    return {
        component: weight / total_weight
        for component, weight in active_weights.items()
    }

def format_novelty_component_formula(component_weights):
    if not component_weights:
        return None
    terms = [
        f"{component_weights[component]:.3f}*{component}"
        for component in NOVELTY_COMPONENT_ORDER
        if component in component_weights
    ]
    return " + ".join(terms) if terms else None

def _finite_float(value):
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric

def normalize_dual_axis_component_weights(weight_map, components):
    if not isinstance(weight_map, dict):
        return {}
    active_components = set(components or [])
    active_weights = {}
    for component in DUAL_AXIS_COMPONENT_ORDER:
        if component not in active_components:
            continue
        weight = _finite_float(weight_map.get(component))
        if weight is not None and weight > 0:
            active_weights[component] = weight
    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        return {}
    return {
        component: weight / total_weight
        for component, weight in active_weights.items()
    }

def get_dual_axis_component_base_weights_for_axis(axis="imagination"):
    return DUAL_AXIS_COMPONENT_BASE_WEIGHTS

def get_effective_dual_axis_component_weights_for_axis(components, axis="imagination"):
    return normalize_dual_axis_component_weights(
        get_dual_axis_component_base_weights_for_axis(axis),
        components,
    )

def get_effective_dual_axis_component_weights(components):
    return get_effective_dual_axis_component_weights_for_axis(components, axis="imagination")

def aggregate_dual_axis_component_scores(component_scores, components, axis="imagination"):
    axis_key = "hallucination" if axis == "hallucination" else "imagination"
    axis_label = "H" if axis_key == "hallucination" else "I"
    active_components = [
        component for component in DUAL_AXIS_COMPONENT_ORDER
        if component in set(components or [])
    ]
    weights = get_effective_dual_axis_component_weights_for_axis(active_components, axis_key)
    raw_values = {}
    for component in active_components:
        value = _finite_float((component_scores or {}).get(component))
        if value is None or component not in weights:
            continue
        raw_values[component] = value

    scored_components = [
        component for component in active_components
        if component in raw_values and component in weights
    ]
    kept_components = list(scored_components)
    kept_weight_total = sum(weights.get(component, 0.0) for component in kept_components)
    aggregation_weights = {}
    if kept_weight_total > 0:
        aggregation_weights = {
            component: weights[component] / kept_weight_total
            for component in kept_components
        }
    score = None
    if aggregation_weights:
        score = sum(
            raw_values[component] * aggregation_weights[component]
            for component in kept_components
        )

    weight_formula = format_dual_axis_component_formula(weights, axis_label)
    rounded = lambda mapping: {
        component: round(value, 6)
        for component, value in mapping.items()
    }
    policy = {
        "version": "weighted_mean",
        "axis": axis_key,
        "method": "weighted_mean",
        "component_weights": rounded(aggregation_weights),
    }
    return {
        "score": score,
        "formula": weight_formula,
        "weights": weights,
        "aggregation_weights": aggregation_weights,
        "raw_task_type_scores": raw_values,
        "aggregation_policy": policy,
    }

def format_dual_axis_component_formula(component_weights, axis_label):
    if not component_weights:
        return None
    terms = [
        f"{component_weights[component]:.3f}*{component}_{axis_label}"
        for component in DUAL_AXIS_COMPONENT_ORDER
        if component in component_weights
    ]
    return " + ".join(terms) if terms else None

def compute_stats_delta(before, after):
    delta = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if isinstance(after_value, (int, float)) and isinstance(before_value, (int, float)):
            delta[key] = after_value - before_value
    return delta

def extract_component_score(category_originality_scores, key):
    value = category_originality_scores.get(key)
    return value if value is not None else float("nan")

def extract_groundedness_metric(report, *keys):
    groundedness = report.get("overall_summary", {}).get("axes", {}).get("groundedness", {})
    for key in keys:
        value = groundedness.get(key)
        if value is not None:
            return value
    return None

def extract_groundedness_task_type_score(report, task_type):
    task_type_scores = extract_groundedness_metric(report, "task_type_scores")
    if not isinstance(task_type_scores, dict):
        return None
    return task_type_scores.get(task_type)

def mean_or_none(values):
    filtered = [value for value in values if value is not None]
    return (sum(filtered) / len(filtered)) if filtered else None

def save_json(path, payload):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=4, cls=NumpyJSONEncoder)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def _parse_task_limit_overrides(raw_value):
    limits = {}
    for raw_item in (raw_value or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            key, value = item.split(":", 1)
        elif "=" in item:
            key, value = item.split("=", 1)
        else:
            continue
        key = key.strip()
        try:
            limit = int(value.strip())
        except ValueError:
            continue
        if key and limit >= 0:
            limits[key] = limit
    return limits

_PROFILE_TASK_MANIFEST_CACHE = None

def load_profile_task_manifest():
    global _PROFILE_TASK_MANIFEST_CACHE
    if _PROFILE_TASK_MANIFEST_CACHE is not None:
        return _PROFILE_TASK_MANIFEST_CACHE
    if not PROFILE_TASK_MANIFEST_PATH:
        _PROFILE_TASK_MANIFEST_CACHE = {}
        return _PROFILE_TASK_MANIFEST_CACHE
    try:
        with open(PROFILE_TASK_MANIFEST_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Configured task manifest not found: {PROFILE_TASK_MANIFEST_PATH}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Configured task manifest must be a JSON object: {PROFILE_TASK_MANIFEST_PATH}")
    task_ids = payload.get("task_ids_by_component") or {}
    if not isinstance(task_ids, dict):
        raise RuntimeError(
            f"Configured task manifest task_ids_by_component must be a JSON object: {PROFILE_TASK_MANIFEST_PATH}"
        )
    _PROFILE_TASK_MANIFEST_CACHE = payload
    return _PROFILE_TASK_MANIFEST_CACHE

def get_profile_manifest_task_ids():
    manifest = load_profile_task_manifest()
    task_ids = manifest.get("task_ids_by_component") or {}
    return {
        str(component): [str(task_id) for task_id in ids]
        for component, ids in task_ids.items()
        if isinstance(ids, list)
    }

def _select_manifest_tasks(task_type, task_list, manifest_ids):
    if task_type not in manifest_ids:
        return task_list, None
    requested_ids = list(manifest_ids[task_type])
    by_id = {str(task.get("id")): task for task in task_list}
    missing_ids = [task_id for task_id in requested_ids if task_id not in by_id]
    if missing_ids:
        raise RuntimeError(
            f"Task manifest {PROFILE_TASK_MANIFEST_PATH} references missing {task_type} ids: "
            + ", ".join(missing_ids)
        )
    return [by_id[task_id] for task_id in requested_ids], {
        "original": len(task_list),
        "kept": len(requested_ids),
        "ids": requested_ids,
    }

def get_runtime_task_policy():
    included = set(OPENROUTER_TASK_FAMILIES or PRIMARY_DUAL_AXIS_COMPONENTS)
    return {
        "experiment_profile": OPENROUTER_EXPERIMENT_PROFILE,
        "profile_task_manifest_path": PROFILE_TASK_MANIFEST_PATH or None,
        "explicit_task_families": list(OPENROUTER_TASK_FAMILIES),
        "included_task_families": [
            task_type for task_type in PRIMARY_DUAL_AXIS_COMPONENTS
            if task_type in included
        ],
        "max_tasks_per_family": OPENROUTER_MAX_TASKS_PER_FAMILY,
        "task_limits": _parse_task_limit_overrides(OPENROUTER_TASK_LIMITS),
    }

def filter_dataset_for_runtime(dataset):
    policy = get_runtime_task_policy()
    included = set(policy["included_task_families"])
    task_limits = policy["task_limits"]
    manifest_ids = get_profile_manifest_task_ids()
    filtered = {}
    skipped = {}
    limited = {}
    manifest_selected = {}
    for task_type, tasks in (dataset or {}).items():
        if task_type not in included:
            skipped[task_type] = len(tasks or [])
            continue
        task_list = list(tasks or [])
        task_list, manifest_record = _select_manifest_tasks(task_type, task_list, manifest_ids)
        if manifest_record is not None:
            manifest_selected[task_type] = manifest_record
        limit = task_limits.get(task_type, policy["max_tasks_per_family"])
        if limit and limit > 0:
            original_count = len(task_list)
            task_list = task_list[:limit]
            if len(task_list) < original_count:
                limited[task_type] = {
                    "original": original_count,
                    "kept": len(task_list),
                }
        filtered[task_type] = task_list
    policy = dict(policy)
    policy["skipped_task_families"] = skipped
    policy["limited_task_families"] = limited
    policy["manifest_selected_task_families"] = manifest_selected
    policy["prompt_counts"] = {task_type: len(tasks) for task_type, tasks in filtered.items()}
    policy["total_prompts_per_repeat"] = sum(policy["prompt_counts"].values())
    return filtered, policy

def build_prompt_manifest(dataset):
    manifest = {}
    for task_type, tasks in dataset.items():
        manifest[task_type] = []
        for task in tasks:
            metadata = {k: v for k, v in task.items() if k not in {"id", "prompt"}}
            manifest[task_type].append({
                "id": task.get("id"),
                "prompt": task.get("prompt"),
                "runtime_role": "main_benchmark",
                "metadata": metadata,
            })
    return manifest

__all__ = [
    'env_flag',
    'ENV_FILE_LOADED',
    'OPENROUTER_API_KEY_ENV',
    'OPENROUTER_BASE_URL',
    'OPENROUTER_MODELS_ENDPOINT',
    'OPENROUTER_HTTP_REFERER',
    'OPENROUTER_APP_TITLE',
    'POE_API_KEY_ENV',
    'POE_BASE_URL',
    'POE_MODELS_ENDPOINT',
    'MODEL_PROVIDER_OPENROUTER',
    'MODEL_PROVIDER_POE',
    'OPENROUTER_REQUEST_TIMEOUT',
    'OPENROUTER_CONNECT_TIMEOUT',
    'OPENROUTER_MAX_RETRIES',
    'OPENROUTER_MAX_TOKENS',
    'OPENROUTER_GENERATION_CACHE_DIR',
    'OPENROUTER_USE_GENERATION_CACHE',
    'OPENROUTER_SAVE_GENERATION_CACHE',
    'OPENROUTER_GENERATION_CACHE_ONLY',
    'OPENROUTER_GENERATION_CACHE_MISS_FATAL',
    'OPENROUTER_STREAM',
    'OPENROUTER_ENABLE_REASONING',
    'poe_model_config',
    'openrouter_model_config',
    'DEFAULT_MODEL_CONFIGS',
    'ALL_MODEL_CONFIGS',
    'ALL_MODEL_CONFIG_INDEX',
    'MODEL_IDS_OVERRIDE',
    'dedupe_model_configs',
    'ACTIVE_MODEL_SET',
    'MODEL_CONFIGS',
    'SELECTED_MODELS',
    'MODEL_CONFIG_INDEX',
    'BENCHMARK_PROFILE',
    'OPENROUTER_EXPERIMENT_PROFILE',
    'REPORTS_DIR',
    'make_file_tag',
    'MODEL_SET_FILE_TAG',
    'MULTI_MODEL_REPORT_JSON',
    'MULTI_MODEL_REPORT_MD',
    'MULTI_MODEL_CHART',
    'STOP_WORDS',
    'SWOW_DATA_PATH',
    'DEDUP_SIMILARITY_THRESHOLD',
    'UUT_OUTPUT_COUNT',
    'CREATIVE_OUTPUT_COUNT',
    'PROP_CONJ_OUTPUT_COUNT',
    'CJST_OUTPUT_COUNT',
    'MACGYVER_OUTPUT_COUNT',
    'HYPOUSESPACE_OUTPUT_COUNT',
    'GCW_BEAT_COUNT',
    'NEOCODER_OUTPUT_COUNT',
    'CLOSED_WORLD_FACT_OUTPUT_COUNT',
    'ANALOGY_TRANSFER_OUTPUT_COUNT',
    'DAT_OUTPUT_COUNT',
    'FF_OUTPUT_COUNT',
    'MIN_CREATIVE_ITEMS_PER_TASK',
    'MIN_PROP_CONJ_ITEMS_PER_TASK',
    'MIN_CJST_ITEMS_PER_TASK',
    'MIN_CJST_ITEMS_PER_TIER',
    'MIN_MACGYVER_PLANS_PER_TASK',
    'MIN_HYPOUSESPACE_ITEMS_PER_TASK',
    'MIN_GCW_BEATS_PER_TASK',
    'MIN_NEOCODER_ITEMS_PER_TASK',
    'MIN_CLOSED_WORLD_FACT_ITEMS_PER_TASK',
    'MIN_ANALOGY_TRANSFER_ITEMS_PER_TASK',
    'MIN_FF_WORDS_PER_TASK',
    'MODEL_SAMPLE_REPEATS',
    'REPEAT_ELIGIBLE_FRACTION',
    'BOOTSTRAP_SAMPLES',
    'SAMPLING_SEED_BASE',
    'TASK_TEMPERATURES',
    'TASK_SYSTEM_PROMPTS',
    'TASK_MAX_TOKENS',
    'FLEX_WEIGHT_EMBEDDING',
    'FLEX_WEIGHT_ONTOLOGICAL',
    'FLEX_WEIGHT_EMBEDDING_DEGENERATE',
    'FLEX_WEIGHT_ONTOLOGICAL_DEGENERATE',
    'CREATIVE_TASK_TYPES',
    'DT_TOTAL_NOVELTY_WEIGHT',
    'DT_TOTAL_FLEXIBILITY_WEIGHT',
    'NOVELTY_COMPONENT_BASE_WEIGHTS',
    'NOVELTY_COMPONENT_ORDER',
    'UUT_DUAL_AXIS_VERSION',
    'UUT_DUAL_AXIS_BETA_IH',
    'UUT_DUAL_AXIS_BETA_HI',
    'PROPCONJ_DUAL_AXIS_VERSION',
    'PROPCONJ_DUAL_AXIS_BETA_IH',
    'PROPCONJ_DUAL_AXIS_BETA_HI',
    'MACGYVER_DUAL_AXIS_VERSION',
    'MACGYVER_DUAL_AXIS_BETA_IH',
    'MACGYVER_DUAL_AXIS_BETA_HI',
    'CJST_DUAL_AXIS_VERSION',
    'CJST_V3_CALIBRATION_POLICY',
    'CJST_V3_RUNTIME_SCORING_POLICY',
    'CJST_DUAL_AXIS_BETA_IH',
    'CJST_DUAL_AXIS_BETA_HI',
    'HYPOUSESPACE_DUAL_AXIS_VERSION',
    'HYPOUSESPACE_V3_CALIBRATION_POLICY',
    'HYPOUSESPACE_V3_RUNTIME_SCORING_POLICY',
    'HYPOUSESPACE_DUAL_AXIS_BETA_IH',
    'HYPOUSESPACE_DUAL_AXIS_BETA_HI',
    'GCW_DUAL_AXIS_VERSION',
    'GCW_V3_CALIBRATION_POLICY',
    'GCW_V3_RUNTIME_SCORING_POLICY',
    'GCW_DUAL_AXIS_BETA_IH',
    'GCW_DUAL_AXIS_BETA_HI',
    'NEOCODER_DUAL_AXIS_VERSION',
    'NEOCODER_V3_CALIBRATION_POLICY',
    'NEOCODER_V3_RUNTIME_SCORING_POLICY',
    'NEOCODER_V3_TEST_VISIBILITY_POLICY',
    'NEOCODER_DUAL_AXIS_BETA_IH',
    'NEOCODER_DUAL_AXIS_BETA_HI',
    'DUAL_AXIS_COMPONENT_BASE_WEIGHTS',
    'DUAL_AXIS_COMPONENT_ORDER',
    'PRIMARY_DUAL_AXIS_COMPONENTS',
    'OPTIONAL_DUAL_AXIS_COMPONENTS',
    'AUXILIARY_IMAGINATION_DIAGNOSTICS',
    'ENHANCED_DUAL_AXIS_DIAGNOSTICS',
    'CALIBRATION_DIAGNOSTICS',
    'CHALLENGE_DIAGNOSTICS',
    'MACGYVER_BOUNDARY_DIAGNOSTIC_TASK_IDS',
    'HYPOUSESPACE_BOUNDARY_DIAGNOSTIC_TASK_IDS',
    'PROFILE_TASK_MANIFEST_PATH',
    'DUAL_AXIS_REPORT_VERSION',
    'WHITE_BOX_REPORT_SCHEMA_VERSION',
    'OPENROUTER_TASK_FAMILIES',
    'OPENROUTER_MAX_TASKS_PER_FAMILY',
    'OPENROUTER_TASK_LIMITS',
    'MIN_CREATIVE_COVERAGE',
    'MIN_CREATIVE_TASKTYPE_COVERAGE',
    'MIN_MACGYVER_COVERAGE',
    'MIN_CJST_COVERAGE',
    'MIN_HYPOUSESPACE_COVERAGE',
    'MIN_GCW_COVERAGE',
    'MIN_NEOCODER_COVERAGE',
    'MIN_CLOSED_WORLD_FACT_COVERAGE',
    'MIN_ANALOGY_TRANSFER_COVERAGE',
    'MIN_DAT_COVERAGE',
    'MIN_CDAT_COVERAGE',
    'NumpyJSONEncoder',
    'require_openrouter_api_key',
    'require_poe_api_key',
    'validate_openrouter_base_url',
    'validate_poe_base_url',
    'create_openrouter_client',
    'create_poe_client',
    'get_model_provider',
    'get_selected_model_providers',
    'create_provider_clients',
    'resolve_provider_client',
    'fetch_openrouter_model_catalog',
    'fetch_poe_model_catalog',
    'fetch_selected_model_catalog',
    'build_model_catalog_index',
    'validate_selected_models',
    'sanitize_filename',
    'build_openrouter_extra_headers',
    'build_model_extra_body',
    'get_task_max_tokens',
    'get_task_runtime_group',
    'get_task_temperature',
    'get_task_system_prompt',
    'get_expected_output_count',
    'stable_seed',
    'minimum_eligible_repeats',
    'strip_json_code_fence',
    'extract_json_payload',
    'build_parsed_creative_item',
    'parse_creative_item',
    'normalize_creative_json_item',
    'parse_creative_items',
    'parse_responses',
    'classify_api_error',
    'extract_reasoning_token_count',
    'is_reasoning_only_output',
    'summarize_numeric_samples',
    'normalize_api_payload',
    'normalize_usage_payload',
    'should_stream_for_model',
    'build_generation_cache_key',
    'load_generation_cache',
    'save_generation_cache',
    'build_generation_record',
    'call_llm',
    'preprocess_for_semantic_scoring',
    'get_effective_novelty_component_weights',
    'format_novelty_component_formula',
    'normalize_dual_axis_component_weights',
    'get_dual_axis_component_base_weights_for_axis',
    'get_effective_dual_axis_component_weights_for_axis',
    'get_effective_dual_axis_component_weights',
    'aggregate_dual_axis_component_scores',
    'format_dual_axis_component_formula',
    'compute_stats_delta',
    'extract_component_score',
    'extract_groundedness_metric',
    'extract_groundedness_task_type_score',
    'mean_or_none',
    'save_json',
    'load_profile_task_manifest',
    'get_profile_manifest_task_ids',
    'get_runtime_task_policy',
    'filter_dataset_for_runtime',
    'build_prompt_manifest',
]
