from __future__ import annotations

"""Prompt-specific common-answer novelty scoring."""

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  
    np = None

from ttct_zero_originality import (
    ZERO_ORIGINALITY_STATIC_BANK,
    extract_task_specific_core,
    get_zero_originality_runtime_context,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
COMMON_ANSWER_REFERENCE_BANK_PATH = DATA_DIR / "common_answer_reference_bank.json"
T1_V3_COMMON_ANSWER_BANK_OVERLAY_PATH = DATA_DIR / "t1_v3_common_answer_bank_overlay.json"
COMMON_ANSWER_BANK_SCHEMA_VERSION = "common_answer_bank"
COMMON_ANSWER_BANK_BLEND_WEIGHTS = {
    "UUT": {"legacy": 0.5, "bank": 0.5},
    "Instances": {"legacy": 0.5, "bank": 0.5},
    "PropConj": {"legacy": 0.5, "bank": 0.5},
    "JST": {"legacy": 0.3, "bank": 0.7},
}
COMMON_ANSWER_BANK_HYBRID_FORMULAS = {
    "UUT": "0.50*legacy_distance(target_label, cleaned_display_text) + 0.50*bank_distance(core_title, prompt_common_bank)",
    "JST": "0.30*legacy_distance(short_label, cleaned_display_text) + 0.70*bank_distance(core_clause, scenario_common_bank)",
    "Instances": "0.50*legacy_distance(trait_label, cleaned_display_text) + 0.50*bank_distance(canonical_concept, trait_common_bank)",
    "PropConj": "0.50*legacy_distance(property_conjunction_label, item) + 0.50*bank_distance(canonical_item, property_conjunction_common_bank)",
}
COMMON_ANSWER_BASIC_VALIDITY_BOUNDS = {
    "UUT": (1, 8),
    "JST": (2, 18),
    "Instances": (1, 6),
    "PropConj": (1, 6),
}
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for", "with",
    "in", "on", "at", "by", "is", "are", "was", "were", "be", "being",
    "been", "it", "this", "that", "these", "those", "into", "from", "as",
    "would", "could", "should", "can", "may", "might", "will", "just", "very",
    "really", "some", "someone", "something", "make", "makes", "made", "use",
    "used", "using", "people", "thing", "things", "item", "items",
}
GENERIC_CONCEPTS = {"object", "thing", "item", "stuff", "something"}


def _infer_task_type(item_id: Optional[str], task_type: Optional[str] = None) -> Optional[str]:
    if task_type:
        return task_type
    if not item_id:
        return None
    prefix = item_id.split("_", 1)[0].lower()
    if prefix == "uut":
        return "UUT"
    if prefix == "jst":
        return "JST"
    if prefix == "ins":
        return "Instances"
    if prefix in {"pc", "propconj"}:
        return "PropConj"
    return None


def _normalize_phrase(text: str) -> str:
    text = (text or "").lower().strip().replace("_", " ")
    tokens = re.findall(r"[a-zA-Z-]+", text)
    normalized = []
    for token in tokens:
        token = token.lower().strip("-")
        if not token:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith(("ches", "shes", "xes", "zes")) and len(token) > 4:
            token = token[:-2]
        elif token.endswith("es") and len(token) > 4 and not token.endswith(("ses", "xes", "zes")):
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is", "as")):
            token = token[:-1]
        if token and token not in {"a", "an", "the", "of", "for", "to"}:
            normalized.append(token)
    return " ".join(normalized).strip()


def _tokenize_content(text: str) -> List[str]:
    return [
        token for token in _normalize_phrase(text).split()
        if token and token not in STOP_WORDS
    ]


@lru_cache(maxsize=8)
def _load_json_resource(filename: str, default_type: str = "dict"):
    path = DATA_DIR / filename
    if not path.exists():
        return {} if default_type == "dict" else []
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=2)
def _load_common_answer_reference_bank(path: str = str(COMMON_ANSWER_REFERENCE_BANK_PATH)) -> Dict[str, Dict[str, object]]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    if isinstance(payload, dict):
        if isinstance(payload.get("runtime_bank"), dict):
            return dict(payload.get("runtime_bank") or {})
        if isinstance(payload.get("tasks"), dict):
            return dict(payload.get("tasks") or {})
        if all(isinstance(value, dict) for value in payload.values()):
            return dict(payload)
    return {}


@lru_cache(maxsize=2)
def _load_t1_v3_common_answer_bank_overlay(path: str = str(T1_V3_COMMON_ANSWER_BANK_OVERLAY_PATH)) -> Dict[str, Dict[str, object]]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), dict):
        return dict(payload.get("tasks") or {})
    if isinstance(payload, dict) and all(isinstance(value, dict) for value in payload.values()):
        return dict(payload)
    return {}


def has_common_answer_reference_bank(path: str = str(COMMON_ANSWER_REFERENCE_BANK_PATH)) -> bool:
    return bool(_load_common_answer_reference_bank(path))


def _iter_family_aliases(entry: Dict[str, object], group_name: str) -> Iterable[Tuple[str, str]]:
    for family in entry.get(group_name, []) or []:
        family_name = str(family.get("family") or "").strip()
        for alias in family.get("aliases") or []:
            alias_text = str(alias).strip()
            if alias_text:
                yield family_name, alias_text


def _dedupe_phrase_list(entries: Sequence[str], *, max_words: int = 8, limit: Optional[int] = None) -> List[str]:
    results: List[str] = []
    seen = set()
    for entry in entries:
        normalized = _normalize_phrase(entry)
        if not normalized:
            continue
        if len(normalized.split()) > max_words:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)
        if limit is not None and len(results) >= limit:
            break
    return results


def collect_static_common_answer_bank(
    item_id: str,
    *,
    include_broad_common: bool = True,
    include_reference_bank: bool = True,
    include_t1_v3_overlay: bool = True,
) -> Dict[str, object]:
    entry = ZERO_ORIGINALITY_STATIC_BANK.get(item_id, {})
    reference_entry = _load_common_answer_reference_bank().get(item_id, {}) if include_reference_bank else {}
    overlay_entry = _load_t1_v3_common_answer_bank_overlay().get(item_id, {}) if include_t1_v3_overlay else {}

    aliases: List[str] = []
    trace = {
        "built_in_hard_zero": [],
        "built_in_broad_common": [],
        "mined_hard_zero": [],
        "mined_broad_common": [],
        "t1_v3_overlay_hard_zero": [],
        "t1_v3_overlay_broad_common": [],
    }

    for family_name, alias in _iter_family_aliases(entry, "hard_zero_families"):
        aliases.append(alias)
        trace["built_in_hard_zero"].append({"family": family_name, "alias": _normalize_phrase(alias)})
    if include_broad_common:
        for family_name, alias in _iter_family_aliases(entry, "broad_common_families"):
            aliases.append(alias)
            trace["built_in_broad_common"].append({"family": family_name, "alias": _normalize_phrase(alias)})

    if reference_entry:
        for family_name, alias in _iter_family_aliases(reference_entry, "hard_zero_families"):
            aliases.append(alias)
            trace["mined_hard_zero"].append({"family": family_name, "alias": _normalize_phrase(alias)})
        if include_broad_common:
            for family_name, alias in _iter_family_aliases(reference_entry, "broad_common_families"):
                aliases.append(alias)
                trace["mined_broad_common"].append({"family": family_name, "alias": _normalize_phrase(alias)})

    if overlay_entry:
        for family_name, alias in _iter_family_aliases(overlay_entry, "hard_zero_families"):
            aliases.append(alias)
            trace["t1_v3_overlay_hard_zero"].append({"family": family_name, "alias": _normalize_phrase(alias)})
        if include_broad_common:
            for family_name, alias in _iter_family_aliases(overlay_entry, "broad_common_families"):
                aliases.append(alias)
                trace["t1_v3_overlay_broad_common"].append({"family": family_name, "alias": _normalize_phrase(alias)})

    deduped = _dedupe_phrase_list(aliases, max_words=8)
    return {
        "entries": deduped,
        "size": len(deduped),
        "preview": deduped[:16],
        "trace": trace,
        "reference_bank_available": bool(reference_entry),
        "t1_v3_overlay_available": bool(overlay_entry),
    }


def _collect_swow_expansions(seed_terms: Sequence[str], swow_graph=None, *, k: int = 8, min_strength: float = 0.02) -> List[str]:
    if swow_graph is None or not getattr(swow_graph, "available", False):
        return []
    results = []
    seed_norms = {_normalize_phrase(term) for term in seed_terms if term}
    for seed in seed_terms:
        for response, strength in swow_graph.top_associates(seed, k=k):
            if float(strength) < min_strength:
                continue
            normalized = _normalize_phrase(response)
            if not normalized or normalized in seed_norms or normalized in STOP_WORDS:
                continue
            results.append(normalized)
    return _dedupe_phrase_list(results, max_words=5, limit=32)


def _collect_cognitive_baseline_entries(dynamic_context: str, cognitive_baseline=None, *, limit: int = 24) -> List[str]:
    if cognitive_baseline is None or not dynamic_context:
        return []
    try:
        baseline = cognitive_baseline.get_dynamic_baseline(dynamic_context)
    except Exception:
        return []
    return _dedupe_phrase_list(baseline, max_words=5, limit=limit)


def _collect_uut_dynamic_bank(item_id: str, target_concept: Optional[str], cognitive_baseline=None, swow_graph=None, word_norms2=None) -> Dict[str, object]:
    profiles = _load_json_resource("uut_affordance_profiles.json")
    profile = profiles.get(item_id, {}) if isinstance(profiles, dict) else {}
    shared = profiles.get("__shared__", {}) if isinstance(profiles, dict) else {}
    required_affordances = (shared.get("required_affordances") or {}) if isinstance(shared, dict) else {}

    seed_terms = [
        str(target_concept or ""),
        *(profile.get("target_aliases") or []),
        *(profile.get("prompt_cues") or []),
    ]
    phrase_entries: List[str] = []
    affordance_trace = []

    for affordance, strength in sorted((profile.get("base_properties") or {}).items(), key=lambda item: float(item[1]), reverse=True):
        if float(strength) < 0.45:
            continue
        affordance_trace.append({"affordance": affordance, "strength": round(float(strength), 4), "source": "profile"})
        phrase_entries.append(affordance.replace("_", " "))
        phrase_entries.extend((required_affordances.get(affordance) or [])[:6])

    if word_norms2 is not None and getattr(word_norms2, "available", False) and target_concept:
        try:
            derived = word_norms2.derive_affordance_profile(str(target_concept))
        except Exception:
            derived = {}
        for affordance, strength in sorted((derived.get("base_properties") or {}).items(), key=lambda item: float(item[1]), reverse=True):
            if float(strength) < 0.35:
                continue
            affordance_trace.append({"affordance": affordance, "strength": round(float(strength), 4), "source": "word_norms2"})
            phrase_entries.append(affordance.replace("_", " "))
            phrase_entries.extend((required_affordances.get(affordance) or [])[:4])

    context_info = get_zero_originality_runtime_context(
        item_id=item_id,
        task_type="UUT",
        target_concept=target_concept,
        task_metadata={},
    )
    dynamic_context = context_info.get("dynamic_context") or str(target_concept or "")
    baseline_entries = _collect_cognitive_baseline_entries(dynamic_context, cognitive_baseline)
    swow_entries = _collect_swow_expansions(seed_terms, swow_graph)

    dynamic_entries = _dedupe_phrase_list(
        phrase_entries + baseline_entries + swow_entries,
        max_words=6,
        limit=48,
    )
    return {
        "entries": dynamic_entries,
        "trace": {
            "seed_terms": _dedupe_phrase_list(seed_terms, max_words=4, limit=18),
            "affordances": affordance_trace,
            "dynamic_context": dynamic_context,
            "cognitive_baseline_preview": baseline_entries[:16],
            "swow_preview": swow_entries[:16],
        },
    }


def _collect_jst_dynamic_bank(item_id: str, target_concept: Optional[str], task_metadata: Optional[Dict[str, object]], cognitive_baseline=None, swow_graph=None) -> Dict[str, object]:
    task_metadata = task_metadata or {}
    templates = _load_json_resource("jst_scenario_templates.json")
    template = templates.get(item_id, {}) if isinstance(templates, dict) else {}
    scenario_text = str(task_metadata.get("scenario_text") or target_concept or "")
    anchors = list(template.get("anchors") or [])
    impact_channels = template.get("impact_channels") or {}

    channel_labels = [str(name).replace("_", " ") for name in impact_channels.keys()]
    channel_keywords: List[str] = []
    for keywords in impact_channels.values():
        channel_keywords.extend(list(keywords or []))

    context_info = get_zero_originality_runtime_context(
        item_id=item_id,
        task_type="JST",
        target_concept=target_concept,
        task_metadata=task_metadata,
    )
    dynamic_context = context_info.get("dynamic_context") or scenario_text
    baseline_entries = _collect_cognitive_baseline_entries(dynamic_context, cognitive_baseline)
    swow_entries = _collect_swow_expansions(anchors + channel_keywords[:20], swow_graph, k=6)

    phrase_entries = channel_labels + channel_keywords
    dynamic_entries = _dedupe_phrase_list(
        phrase_entries + baseline_entries + swow_entries,
        max_words=8,
        limit=56,
    )
    return {
        "entries": dynamic_entries,
        "trace": {
            "scenario_text": scenario_text,
            "anchors": _dedupe_phrase_list(anchors, max_words=3, limit=16),
            "impact_channel_labels": channel_labels,
            "impact_channel_keywords": _dedupe_phrase_list(channel_keywords, max_words=5, limit=24),
            "dynamic_context": dynamic_context,
            "cognitive_baseline_preview": baseline_entries[:16],
            "swow_preview": swow_entries[:16],
        },
    }


def _score_word_norms2_trait_candidate(entry: Dict[str, object], positive_predicates: Sequence[str], negative_predicates: Sequence[str]) -> float:
    features = entry.get("features") or {}
    if not isinstance(features, dict):
        return 0.0
    positive = sum(float(features.get(predicate, 0.0)) for predicate in positive_predicates)
    negative = sum(float(features.get(predicate, 0.0)) for predicate in negative_predicates)
    return positive - 0.6 * negative


def _collect_instances_word_norms2_candidates(task_lexicon: Dict[str, object], word_norms2=None, *, limit: int = 24) -> List[str]:
    if word_norms2 is None or not getattr(word_norms2, "available", False):
        return []

    positive_predicates: List[str] = []
    negative_predicates: List[str] = []
    for group in task_lexicon.get("trait_groups") or []:
        weight = float(group.get("weight", 1.0) or 1.0)
        for predicate in group.get("positive_predicates") or []:
            positive_predicates.extend([predicate] * max(1, int(round(weight))))
        for predicate in group.get("negative_predicates") or []:
            negative_predicates.append(predicate)

    if not positive_predicates:
        return []

    scored: List[Tuple[float, str]] = []
    for concept, entry in (getattr(word_norms2, "concept_db", {}) or {}).items():
        concept_norm = _normalize_phrase(concept)
        if not concept_norm or concept_norm in GENERIC_CONCEPTS:
            continue
        if len(concept_norm.split()) > 3:
            continue
        score = _score_word_norms2_trait_candidate(entry, positive_predicates, negative_predicates)
        if score < 0.55:
            continue
        scored.append((float(score), concept_norm))

    scored.sort(key=lambda item: (-item[0], len(item[1].split()), item[1]))
    return _dedupe_phrase_list([concept for _, concept in scored], max_words=3, limit=limit)


def _collect_instances_dynamic_bank(item_id: str, target_concept: Optional[str], cognitive_baseline=None, swow_graph=None, word_norms2=None) -> Dict[str, object]:
    lexicon = _load_json_resource("instances_trait_lexicon.json")
    task_lexicon = lexicon.get(item_id, {}) if isinstance(lexicon, dict) else {}
    prompt_cues = list(task_lexicon.get("prompt_cues") or [])

    fallback_keywords: List[str] = []
    for rule in task_lexicon.get("fallback_concept_rules") or []:
        fallback_keywords.extend(list(rule.get("keywords") or []))

    context_info = get_zero_originality_runtime_context(
        item_id=item_id,
        task_type="Instances",
        target_concept=target_concept,
        task_metadata={},
    )
    dynamic_context = context_info.get("dynamic_context") or str(target_concept or "")
    baseline_entries = _collect_cognitive_baseline_entries(dynamic_context, cognitive_baseline)
    swow_entries = _collect_swow_expansions(prompt_cues, swow_graph, k=8)
    norms2_entries = _collect_instances_word_norms2_candidates(task_lexicon, word_norms2)

    dynamic_entries = _dedupe_phrase_list(
        fallback_keywords + norms2_entries + baseline_entries + swow_entries,
        max_words=4,
        limit=56,
    )
    return {
        "entries": dynamic_entries,
        "trace": {
            "prompt_cues": _dedupe_phrase_list(prompt_cues, max_words=3, limit=16),
            "fallback_keywords": _dedupe_phrase_list(fallback_keywords, max_words=4, limit=20),
            "word_norms2_preview": norms2_entries[:16],
            "dynamic_context": dynamic_context,
            "cognitive_baseline_preview": baseline_entries[:16],
            "swow_preview": swow_entries[:16],
        },
    }


def _collect_propconj_dynamic_bank(item_id: str, target_concept: Optional[str], task_metadata: Optional[Dict[str, object]] = None,
                                   cognitive_baseline=None, swow_graph=None, word_norms2=None) -> Dict[str, object]:
    task_metadata = task_metadata or {}
    properties = list(task_metadata.get("properties") or [])
    property_labels = [str(prop.get("label") or prop.get("id") or "") for prop in properties]

    fallback_keywords: List[str] = []
    trait_groups = []
    for prop in properties:
        fallback_keywords.extend(list(prop.get("positive_keywords") or [])[:10])
        fallback_keywords.extend(list(prop.get("evidence_keywords") or [])[:8])
        trait_groups.append({
            "weight": float(prop.get("weight", 1.0) or 1.0),
            "positive_predicates": list(prop.get("positive_predicates") or []),
            "negative_predicates": list(prop.get("negative_predicates") or []),
        })

    dynamic_context = " ".join(
        part for part in [str(target_concept or ""), " ".join(property_labels)] if part
    ).strip()
    baseline_entries = _collect_cognitive_baseline_entries(dynamic_context, cognitive_baseline)
    swow_entries = _collect_swow_expansions(property_labels + fallback_keywords[:24], swow_graph, k=5)
    norms2_entries = _collect_instances_word_norms2_candidates({"trait_groups": trait_groups}, word_norms2)

    dynamic_entries = _dedupe_phrase_list(
        fallback_keywords + norms2_entries + baseline_entries + swow_entries,
        max_words=4,
        limit=64,
    )
    return {
        "entries": dynamic_entries,
        "trace": {
            "property_labels": _dedupe_phrase_list(property_labels, max_words=4, limit=16),
            "fallback_keywords": _dedupe_phrase_list(fallback_keywords, max_words=4, limit=24),
            "word_norms2_preview": norms2_entries[:16],
            "dynamic_context": dynamic_context,
            "cognitive_baseline_preview": baseline_entries[:16],
            "swow_preview": swow_entries[:16],
        },
    }


def build_common_answer_bank_context(
    item_id: str,
    *,
    task_type: Optional[str] = None,
    target_concept: Optional[str] = None,
    task_metadata: Optional[Dict[str, object]] = None,
    cognitive_baseline=None,
    swow_graph=None,
    word_norms2=None,
    include_broad_common: bool = True,
    include_reference_bank: bool = True,
) -> Dict[str, object]:
    task_type = _infer_task_type(item_id, task_type)
    static_bank = collect_static_common_answer_bank(
        item_id,
        include_broad_common=include_broad_common,
        include_reference_bank=include_reference_bank,
    )

    if task_type == "UUT":
        dynamic_bank = _collect_uut_dynamic_bank(
            item_id,
            target_concept,
            cognitive_baseline=cognitive_baseline,
            swow_graph=swow_graph,
            word_norms2=word_norms2,
        )
    elif task_type == "JST":
        dynamic_bank = _collect_jst_dynamic_bank(
            item_id,
            target_concept,
            task_metadata,
            cognitive_baseline=cognitive_baseline,
            swow_graph=swow_graph,
        )
    elif task_type == "PropConj":
        dynamic_bank = _collect_propconj_dynamic_bank(
            item_id,
            target_concept,
            task_metadata,
            cognitive_baseline=cognitive_baseline,
            swow_graph=swow_graph,
            word_norms2=word_norms2,
        )
    else:
        dynamic_bank = _collect_instances_dynamic_bank(
            item_id,
            target_concept,
            cognitive_baseline=cognitive_baseline,
            swow_graph=swow_graph,
            word_norms2=word_norms2,
        )

    combined = _dedupe_phrase_list(
        list(static_bank.get("entries") or []) + list(dynamic_bank.get("entries") or []),
        max_words=8,
        limit=96,
    )
    return {
        "schema_version": COMMON_ANSWER_BANK_SCHEMA_VERSION,
        "item_id": item_id,
        "task_type": task_type,
        "static_bank": static_bank,
        "dynamic_bank": {
            "entries": list(dynamic_bank.get("entries") or []),
            "size": len(dynamic_bank.get("entries") or []),
            "preview": list(dynamic_bank.get("entries") or [])[:16],
            "trace": dynamic_bank.get("trace") or {},
        },
        "combined_bank": combined,
        "combined_bank_size": len(combined),
        "combined_bank_preview": combined[:20],
        "blend_weights": get_common_answer_bank_blend_weights(task_type),
    }


def get_common_answer_bank_blend_weights(task_type: Optional[str]) -> Dict[str, float]:
    weights = COMMON_ANSWER_BANK_BLEND_WEIGHTS.get(task_type or "", {"legacy": 1.0, "bank": 0.0})
    legacy = float(weights.get("legacy", 0.0))
    bank = float(weights.get("bank", 0.0))
    total = legacy + bank
    if total <= 0:
        return {"legacy": 1.0, "bank": 0.0}
    return {"legacy": legacy / total, "bank": bank / total}


def get_common_answer_bank_hybrid_formula(task_type: Optional[str]) -> str:
    return COMMON_ANSWER_BANK_HYBRID_FORMULAS.get(
        task_type or "",
        "legacy_distance + bank_distance",
    )


def is_task_basic_valid(task_type: Optional[str], core_text: Optional[str]) -> bool:
    words = _tokenize_content(core_text or "")
    lower, upper = COMMON_ANSWER_BASIC_VALIDITY_BOUNDS.get(task_type or "", (1, 24))
    return lower <= len(words) <= upper


def _cosine_distances_to_bank(query_text: str, bank_entries: Sequence[str], scorer) -> Optional[List[float]]:
    model = getattr(scorer, "model", None)
    if model is None:
        return None
    if np is None:
        return None
    try:
        embeddings = model.encode([query_text] + list(bank_entries))
    except Exception:
        return None
    vectors = np.asarray(embeddings, dtype=float)
    if vectors.ndim != 2 or len(vectors) != len(bank_entries) + 1:
        return None
    query_vec = vectors[0]
    bank_vecs = vectors[1:]
    query_norm = np.linalg.norm(query_vec)
    bank_norms = np.linalg.norm(bank_vecs, axis=1)
    denom = bank_norms * query_norm
    denom = np.where(denom == 0.0, 1e-12, denom)
    sims = np.clip(bank_vecs @ query_vec / denom, -1.0, 1.0)
    distances = 1.0 - sims
    distances = np.clip(distances, 0.0, 2.0)
    return [float(value) for value in distances.tolist()]


def _nearest_bank_match(query_text: str, bank_entries: Sequence[str], scorer) -> Tuple[Optional[str], Optional[float]]:
    if not query_text or not bank_entries or scorer is None:
        return None, None
    distances = _cosine_distances_to_bank(query_text, bank_entries, scorer)
    if distances is None:
        distances = [float(scorer.calculate_originality(query_text, entry)) for entry in bank_entries]
    if not distances:
        return None, None
    best_index = min(range(len(distances)), key=lambda index: distances[index])
    return bank_entries[best_index], round(float(distances[best_index]), 4)


def blend_creative_novelty(task_type: Optional[str], legacy_score: Optional[float], bank_score: Optional[float]) -> Dict[str, object]:
    legacy = None if legacy_score is None else float(legacy_score)
    bank = None if bank_score is None else float(bank_score)
    base_weights = get_common_answer_bank_blend_weights(task_type)

    if legacy is None and bank is None:
        return {
            "blended_score": None,
            "effective_weights": {"legacy": 0.0, "bank": 0.0},
            "base_weights": base_weights,
            "formula": get_common_answer_bank_hybrid_formula(task_type),
        }
    if legacy is None:
        return {
            "blended_score": round(bank, 4),
            "effective_weights": {"legacy": 0.0, "bank": 1.0},
            "base_weights": base_weights,
            "formula": f"bank_only :: {get_common_answer_bank_hybrid_formula(task_type)}",
        }
    if bank is None:
        return {
            "blended_score": round(legacy, 4),
            "effective_weights": {"legacy": 1.0, "bank": 0.0},
            "base_weights": base_weights,
            "formula": f"legacy_only :: {get_common_answer_bank_hybrid_formula(task_type)}",
        }

    blended = base_weights["legacy"] * legacy + base_weights["bank"] * bank
    formula = get_common_answer_bank_hybrid_formula(task_type)
    return {
        "blended_score": round(float(blended), 4),
        "effective_weights": dict(base_weights),
        "base_weights": base_weights,
        "formula": formula,
    }


def score_common_answer_bank_novelty(
    item_id: str,
    *,
    task_type: Optional[str] = None,
    response_text=None,
    parsed_item=None,
    scorer=None,
    bank_context: Optional[Dict[str, object]] = None,
    zero_orig_trace: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    task_type = _infer_task_type(item_id, task_type)
    if zero_orig_trace is not None:
        core_text = zero_orig_trace.get("zero_orig_core_form") or ""
        core_norm = zero_orig_trace.get("zero_orig_core_norm") or _normalize_phrase(core_text)
        match_field = zero_orig_trace.get("zero_orig_match_field")
    else:
        core_info = extract_task_specific_core(
            item_id,
            response_text=response_text,
            parsed_item=parsed_item,
            task_type=task_type,
        )
        core_text = core_info.get("core_text") or ""
        core_norm = core_info.get("core_norm") or _normalize_phrase(core_text)
        match_field = core_info.get("match_field")

    bank_context = bank_context or {
        "static_bank": {"entries": [], "size": 0, "preview": []},
        "dynamic_bank": {"entries": [], "size": 0, "preview": []},
        "combined_bank": [],
        "combined_bank_size": 0,
        "combined_bank_preview": [],
        "blend_weights": get_common_answer_bank_blend_weights(task_type),
    }

    static_entries = list((bank_context.get("static_bank") or {}).get("entries") or [])
    dynamic_entries = list((bank_context.get("dynamic_bank") or {}).get("entries") or [])
    combined_entries = list(bank_context.get("combined_bank") or [])
    basic_validity = is_task_basic_valid(task_type, core_text)
    forced_zero = bool((zero_orig_trace or {}).get("zero_orig_final"))

    nearest_static = nearest_dynamic = nearest_overall = None
    dist_static = dist_dynamic = dist_overall = None
    bank_score = None

    if forced_zero:
        nearest_static = (zero_orig_trace or {}).get("zero_orig_static_alias") or core_norm or core_text
        nearest_dynamic = None
        nearest_overall = nearest_static or (zero_orig_trace or {}).get("zero_orig_core_form")
        dist_static = 0.0 if nearest_static else None
        dist_dynamic = 0.0 if (zero_orig_trace or {}).get("zero_orig_dynamic") else None
        dist_overall = 0.0
        bank_score = 0.0
    elif basic_validity and scorer is not None and combined_entries:
        nearest_static, dist_static = _nearest_bank_match(core_text or core_norm, static_entries, scorer)
        nearest_dynamic, dist_dynamic = _nearest_bank_match(core_text or core_norm, dynamic_entries, scorer)
        nearest_overall, dist_overall = _nearest_bank_match(core_text or core_norm, combined_entries, scorer)
        bank_score = dist_overall

    return {
        "task_type": task_type,
        "core_text": core_text,
        "core_norm": core_norm,
        "match_field": match_field,
        "task_basic_validity": bool(basic_validity),
        "forced_zero_due_to_zero_originality": forced_zero,
        "bank_score": bank_score,
        "nearest_static_entry": nearest_static,
        "nearest_static_distance": dist_static,
        "nearest_dynamic_entry": nearest_dynamic,
        "nearest_dynamic_distance": dist_dynamic,
        "nearest_overall_entry": nearest_overall,
        "nearest_overall_distance": dist_overall,
        "static_bank_size": len(static_entries),
        "dynamic_bank_size": len(dynamic_entries),
        "combined_bank_size": len(combined_entries),
        "static_bank_preview": static_entries[:12],
        "dynamic_bank_preview": dynamic_entries[:12],
        "combined_bank_preview": combined_entries[:16],
        "blend_weights": bank_context.get("blend_weights") or get_common_answer_bank_blend_weights(task_type),
        "formula": "distance(core_idea, nearest common-answer bank entry)",
    }


def bank_distance_for_anti_cliche(
    *,
    parsed_item=None,
    bank_trace: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Expose the nearest common-answer distance as an anti-cliche signal.

    Groundedness v5 treats distance from prompt-common answers as positive
    evidence only after common-answer novelty has already computed the bank
    trace. This helper intentionally does not rebuild banks or load models.
    """
    trace = bank_trace or {}
    distance = trace.get("nearest_overall_distance")
    if distance is None:
        distance = trace.get("bank_score")
    try:
        distance_value = None if distance is None else float(distance)
    except Exception:
        distance_value = None

    return {
        "nearest_overall_distance": distance_value,
        "nearest_overall_entry": trace.get("nearest_overall_entry"),
        "core_text": trace.get("core_text") or (parsed_item or {}).get("display_text"),
        "core_norm": trace.get("core_norm"),
        "source": "common_answer_bank_trace" if bank_trace else "missing",
    }


__all__ = [
    "COMMON_ANSWER_BANK_SCHEMA_VERSION",
    "COMMON_ANSWER_BANK_BLEND_WEIGHTS",
    "COMMON_ANSWER_REFERENCE_BANK_PATH",
    "T1_V3_COMMON_ANSWER_BANK_OVERLAY_PATH",
    "has_common_answer_reference_bank",
    "COMMON_ANSWER_BANK_HYBRID_FORMULAS",
    "collect_static_common_answer_bank",
    "build_common_answer_bank_context",
    "get_common_answer_bank_blend_weights",
    "get_common_answer_bank_hybrid_formula",
    "is_task_basic_valid",
    "blend_creative_novelty",
    "score_common_answer_bank_novelty",
    "bank_distance_for_anti_cliche",
]
