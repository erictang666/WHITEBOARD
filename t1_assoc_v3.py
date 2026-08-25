from __future__ import annotations

"""T1 discriminative scoring helpers used by the released benchmark."""

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from scorer_hyperparameters import get_scorer_hyperparameter


T1_ASSOC_VERSION = "t1_assoc"
T1_ASSOC_SCORING_CONFIG_PATH = Path(__file__).resolve().parent / "data" / "t1_assoc_scoring_config.json"

_COMMON_ANSWER_DISTANCE_TRANSFORM_FALLBACK = {
    "lo": 0.05,
    "hi": 0.95,
    "shape": "sigmoid_tail_preserving",
    "slope": 6.0,
}
DEFAULT_T1_ASSOC_PARAMS = {
    "common_answer_distance_transform": get_scorer_hyperparameter(
        "t1_assoc_v3",
        "common_answer_distance_transform",
        default=_COMMON_ANSWER_DISTANCE_TRANSFORM_FALLBACK,
    ),
    "UUT": {
        "rarity_gamma": 1.8,
        "support_gamma": 1.2,
        "task_weights": {
            "quality_mass_top8": 0.40,
            "elite_tail_top3": 0.30,
            "diversity_eff": 0.05,
            "valid_ratio": 0.15,
            "mechanism_elaboration": 0.10,
        },
    },
    "PropConj": {
        "rarity_gamma": 1.7,
        "grounding_gamma": 1.4,
        "task_weights": {
            "quality_mass_top6": 0.40,
            "elite_tail_top3": 0.30,
            "diversity_eff": 0.08,
            "hard_valid_ratio": 0.10,
            "soft_valid_ratio": 0.12,
        },
        "conjunction_difficulty_bonus": 0.15,
    },
}

MECHANISM_ELABORATION_MARKERS = {
    "because", "so", "therefore", "by", "using", "through", "prevents",
    "allows", "turns", "converts", "holds", "supports", "anchors",
    "stabilizes", "channels", "redirects", "folds", "clamps", "wedges",
}


def clip01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def mean_or_zero(values: Iterable[float]) -> float:
    usable = [float(value) for value in values]
    return sum(usable) / len(usable) if usable else 0.0


def _deep_merge(base: Dict[str, object], override: Mapping[str, object]) -> Dict[str, object]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def load_t1_assoc_params() -> Dict[str, object]:
    if not T1_ASSOC_SCORING_CONFIG_PATH.exists():
        return dict(DEFAULT_T1_ASSOC_PARAMS)
    try:
        with T1_ASSOC_SCORING_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return dict(DEFAULT_T1_ASSOC_PARAMS)
    final_params = payload.get("final_params") if isinstance(payload, dict) else None
    if not isinstance(final_params, dict):
        return dict(DEFAULT_T1_ASSOC_PARAMS)
    return _deep_merge(dict(DEFAULT_T1_ASSOC_PARAMS), final_params)


def get_component_params(component: str) -> Dict[str, object]:
    params = load_t1_assoc_params()
    component_params = params.get(component) if isinstance(params, dict) else None
    return dict(component_params) if isinstance(component_params, dict) else {}


def transform_common_answer_rarity(value, *, params: Mapping[str, object] | None = None) -> float:
    """Map bank/novelty distance to a rarity score with a fixed low-common floor."""

    transform = (params or load_t1_assoc_params()).get("common_answer_distance_transform", {})
    lo = float(transform.get("lo", 0.05) if isinstance(transform, Mapping) else 0.05)
    hi = float(transform.get("hi", 0.95) if isinstance(transform, Mapping) else 0.95)
    if hi <= lo:
        lo, hi = 0.05, 0.95
    linear = clip01((clip01(value) - lo) / (hi - lo))
    if linear <= 0.0 or linear >= 1.0:
        return linear
    if isinstance(transform, Mapping) and transform.get("shape") == "sigmoid_tail_preserving":
        slope = float(transform.get("slope", 6.0) or 6.0)
        return clip01(1.0 / (1.0 + math.exp(-slope * (linear - 0.5))))
    return linear


def mechanism_elaboration_score(
    text: str,
    *,
    support: float = 0.0,
    mechanism: float = 0.0,
    rarity: float = 0.0,
) -> float:
    normalized = " ".join(str(text or "").lower().replace("_", " ").replace("-", " ").split())
    tokens = [token.strip(".,;:!?()[]{}\"'") for token in normalized.split() if token.strip(".,;:!?()[]{}\"'")]
    marker_hits = len({token for token in tokens if token in MECHANISM_ELABORATION_MARKERS})
    marker_score = clip01(marker_hits / 3.0)
    length_score = clip01(len([token for token in tokens if len(token) >= 4]) / 28.0)
    non_template_score = clip01(rarity)
    return clip01(
        0.30 * clip01(support) +
        0.25 * clip01(mechanism) +
        0.25 * marker_score +
        0.10 * length_score +
        0.10 * non_template_score
    )


def uut_item_quality(
    *,
    rarity,
    support,
    gate,
    mechanism,
    params: Mapping[str, object] | None = None,
) -> float:
    component = params or get_component_params("UUT")
    rarity_gamma = float(component.get("rarity_gamma", 1.4))
    support_gamma = float(component.get("support_gamma", 1.2))
    return clip01(
        (clip01(rarity) ** rarity_gamma) *
        (clip01(support) ** support_gamma) *
        clip01(gate) *
        (0.5 + 0.5 * clip01(mechanism))
    )


def propconj_item_quality(
    *,
    rarity,
    grounding,
    gate,
    intent_coverage,
    params: Mapping[str, object] | None = None,
) -> float:
    component = params or get_component_params("PropConj")
    rarity_gamma = float(component.get("rarity_gamma", 1.3))
    grounding_gamma = float(component.get("grounding_gamma", 1.4))
    return clip01(
        (clip01(rarity) ** rarity_gamma) *
        (clip01(grounding) ** grounding_gamma) *
        clip01(gate) *
        clip01(intent_coverage)
    )


def top_mean(values: Sequence[float], top_n: int, *, denominator: int | None = None) -> float:
    ordered = sorted((clip01(value) for value in values), reverse=True)
    if not ordered or top_n <= 0:
        return 0.0
    selected = ordered[:min(top_n, len(ordered))]
    denom = denominator if denominator is not None else len(selected)
    return sum(selected) / max(1, int(denom))


def effective_diversity(diversity_score, valid_ratio=1.0) -> float:
    return clip01(diversity_score) * (0.35 + 0.65 * clip01(valid_ratio))
