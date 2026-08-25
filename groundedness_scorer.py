


from __future__ import annotations

import json
import math
import re
import copy
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  
    np = None

from word_norms2_norms import UUT_AFFORDANCE_MAP, WordNorms2Norms
from swow_graph import SWOWGraph
from common_answer_bank import bank_distance_for_anti_cliche, score_common_answer_bank_novelty
from typed_axis_aggregation import build_uut_idea_subtype_contributions


DATA_DIR = Path(__file__).resolve().parent / "data"
WHITE_BOX_GROUNDEDNESS_VERSION = "white_box_groundedness"
GROUNDEDNESS_PENALTY_SCALE = 0.85

DEFAULT_V5_TASK_PARAMS = {
    "Instances": {"tau_abs_floor": 0.20, "lambda": 0.40, "legacy_tau": 0.60},
    "UUT": {"tau_abs_floor": 0.20, "lambda": 0.32, "legacy_tau": 0.50},
    "JST": {"tau_abs_floor": 0.20, "lambda": 0.18, "legacy_tau": 0.35},
}

DEFAULT_V5_PENALTY_CONFIG = {
    "scale": 0.85,
    "deficit_shape_exp": 1.4,
    "novelty_scale_lo": 0.5,
    "novelty_scale_hi": 2.0,
    "low_confidence_cutoff": 0.30,
    "low_confidence_multiplier": 0.5,
}

DEFAULT_ANTI_CLICHE_EFFECTIVE_CONFIG = {
    "formula": "clip(raw_anti + feature_rescue*g_feat_v2 - drift_suppression*drift)",
    "feature_rescue": 0.08,
    "drift_suppression": 1.14,
}

DEFAULT_REFERENCE_COHORT = {
    "schema": "v5p0-internal-prior",
    "source": "internal_reference",
    "per_task": {
        "Instances": {"g_median": 0.42, "g_mad": 0.06, "novelty_median": 0.55, "novelty_mad": 0.08},
        "UUT": {"g_median": 0.44, "g_mad": 0.06, "novelty_median": 0.55, "novelty_mad": 0.08},
        "JST": {"g_median": 0.40, "g_mad": 0.06, "novelty_median": 0.55, "novelty_mad": 0.08},
    },
}

V5_TASK_WEIGHTS = {
    "Instances": {"path": 0.20, "feat": 0.45, "mech": 0.10, "anti": 0.25, "drift": 0.0},
    "UUT": {"path": 0.20, "feat": 0.30, "mech": 0.25, "anti": 0.20, "drift": 0.05},
    "JST": {"path": 0.30, "feat": 0.20, "mech": 0.30, "anti": 0.15, "drift": 0.05},
}

MECHANISM_CONNECTIVE_RE = re.compile(
    r"\b(because|so that|so|via|by|using|through|enables|enable|allows|allow|lets|"
    r"in order to|when|then|therefore|as a result|acts as|serves as|turns into|to)\b",
    re.IGNORECASE,
)

MECHANISM_VERB_HINTS = {
    "absorb", "amplify", "anchor", "attach", "balance", "bend", "block", "bounce",
    "carry", "catch", "channel", "conduct", "connect", "contain", "convert", "cover",
    "cushion", "direct", "filter", "float", "fold", "guide", "hang", "heat", "hold",
    "insulate", "lift", "mark", "muffle", "protect", "reflect", "release", "repel",
    "resonate", "signal", "slice", "store", "support", "suspend", "trap", "turn",
    "weigh", "wrap", "write",
}

IMPOSSIBILITY_PATTERNS = [
    (re.compile(r"\bphotosynthesi[sz](e|es|ing)?\b|\bphotosynthesis\b", re.IGNORECASE), 1.0, "nonbiological_photosynthesis"),
    (re.compile(r"\bcures?\s+(any|all|every)?\s*diseases?\b|\bcure\s+any\s+disease\b", re.IGNORECASE), 1.0, "miracle_cure"),
    (re.compile(r"\b(endless|infinite|unlimited)\s+electricity\b|\bperpetual\s+motion\b", re.IGNORECASE), 0.95, "free_energy"),
    (re.compile(r"\bwithout\s+(batteries?|wires?|fuel|motion)\b", re.IGNORECASE), 0.65, "missing_energy_source"),
    (re.compile(r"\bgrows?\s+(coins?|money|fruit)\b", re.IGNORECASE), 0.90, "impossible_growth_product"),
    (re.compile(r"\bliving\s+seed\b.*\bgrows?\b.*\btree\b", re.IGNORECASE), 0.85, "inorganic_living_seed"),
    (re.compile(r"\bprime\s+numbers?\b.*\binvisible\b", re.IGNORECASE), 0.90, "abstract_entity_physical_state"),
    (re.compile(r"\boceans?\b.*\bcheese\b", re.IGNORECASE), 0.95, "large_scale_material_magic"),
    (re.compile(r"\bmoon\b.*\blearn(s|ed|ing)?\b", re.IGNORECASE), 0.85, "astronomical_language_agent"),
    (re.compile(r"\bfish\b.*\b(tax|taxes|paying)\b", re.IGNORECASE), 0.85, "animal_institutional_action"),
    (re.compile(r"\bgravity\b.*\bstop(s|ped)?\s+working\b", re.IGNORECASE), 0.95, "local_physics_suspension"),
    (re.compile(r"\bclocks?\b.*\bsquare\b", re.IGNORECASE), 0.65, "arbitrary_clock_geometry"),
]

INSTANCE_TRAIT_CONFLICT_TOKENS = {
    "ins_01": {"soft", "pillow", "cloud", "foam"},
    "ins_02": {"feather", "silent", "quiet", "whisper", "cotton"},
    "ins_03": {"blue", "screwdriver", "metal", "plastic"},
    "ins_04": {"banana", "peel", "cloth", "paper", "soft"},
    "ins_05": {"glass", "bowling", "ball", "stone", "metal"},
}

GENERIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for", "with", "in",
    "on", "at", "by", "is", "are", "it", "this", "that", "as", "be", "was",
    "were", "been", "being", "would", "could", "should", "might", "will", "can",
    "may", "just", "only", "very", "more", "most", "into", "from", "we", "our",
    "they", "their", "them", "he", "she", "his", "her", "you", "your", "all",
    "some", "any", "each", "every", "what", "when", "where", "how", "why",
}

UUT_AFFORDANCE_ALIASES = {
    "waterproof": "water_resistant",
    "water resistant": "water_resistant",
    "rainproof": "water_resistant",
    "conductive": "metal_conductive",
    "metal": "metal_conductive",
    "flat": "flat_surface",
    "flat surface": "flat_surface",
    "writable": "writable_surface",
    "writeable": "writable_surface",
    "malleable": "shapeable_surface",
    "bendable": "shapeable_surface",
    "foldable": "shapeable_surface",
    "hollow": "acoustic_cavity",
    "cavity": "acoustic_cavity",
    "light": "lightweight",
    "rigid": "rigid_shell",
    "hard": "rigid_shell",
    "support": "load_bearing",
    "load bearing": "load_bearing",
    "reflective surface": "reflective",
    "sharp": "sharp_edge",
    "heatproof": "heat_resistant",
    "heat resistant": "heat_resistant",
}

UUT_MAJOR_EXTRA_TOOL_TERMS = {
    "battery", "motor", "engine", "generator", "computer", "phone", "robot",
    "drone", "laser", "gps", "sensor", "pump", "propeller", "fan", "camera",
    "speaker", "microphone", "magnet", "welding", "welder", "electricity",
    "electronics", "software", "hydraulic", "rocket", "chemical", "acid",
}

UUT_MINOR_HELPER_TERMS = {
    "tape", "glue", "string", "rope", "cord", "paint", "marker", "pen",
    "pencil", "paper", "clip", "scissors", "knife", "rubber band", "strap",
    "hook", "nail", "screw", "zip tie", "cloth", "foil", "wax", "sealant",
    "reflective tape", "label", "sticker", "water", "sand", "soil",
}

UUT_SUPPORT_FIT_THRESHOLD = 0.35


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class GroundednessScorer:
    def __init__(
        self,
        swow_graph: Optional[SWOWGraph] = None,
        word_norms2_norms: Optional[WordNorms2Norms] = None,
        wn_analyzer=None,
        data_dir: Path = DATA_DIR,
    ):
        self.data_dir = Path(data_dir)
        self.swow = swow_graph or SWOWGraph(str(self.data_dir))
        self.word_norms2 = word_norms2_norms or WordNorms2Norms(data_dir=str(self.data_dir))
        self.wn_analyzer = wn_analyzer

        self.instances_trait_lexicon = self._load_json("instances_trait_lexicon.json", default={})
        self.uut_profiles = self._load_uut_profiles()
        self.uut_minor_helper_terms = self._init_uut_minor_helper_terms()
        self.jst_templates = self._load_json("jst_scenario_templates.json", default={})
        self.instances_trait_anchors = self._load_json("instances_trait_anchors.json", default={})
        self.uut_affordance_anchors = self._load_json("uut_affordance_anchors.json", default={})
        self.jst_consequence_anchors = self._load_json("jst_consequence_anchors.json", default={})
        self.v5_config = self._load_v5_config()
        self.reference_cohort = self._load_reference_cohort()
        self._embedding_anchor_cache: Dict[Tuple[int, str, str], object] = {}

    
    
    

    def score_idea(
        self,
        task_type: str,
        task_id: str,
        target_concept: str,
        idea_text: str,
        raw_originality: float = 0.0,
        parsed_item: Optional[Dict[str, object]] = None,
        semantic_scorer=None,
        common_answer_bank_trace: Optional[Dict[str, object]] = None,
        common_answer_bank_context: Optional[Dict[str, object]] = None,
        cohort_stats: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        task_type = task_type.strip()
        if task_type == "Instances":
            result = self._score_instances(
                task_id,
                target_concept,
                idea_text,
                parsed_item=parsed_item,
                semantic_scorer=semantic_scorer,
                common_answer_bank_trace=common_answer_bank_trace,
                common_answer_bank_context=common_answer_bank_context,
            )
        elif task_type == "UUT":
            result = self._score_uut(
                task_id,
                target_concept,
                idea_text,
                parsed_item=parsed_item,
                semantic_scorer=semantic_scorer,
                common_answer_bank_trace=common_answer_bank_trace,
                common_answer_bank_context=common_answer_bank_context,
            )
        elif task_type == "JST":
            result = self._score_jst(
                task_id,
                target_concept,
                idea_text,
                parsed_item=parsed_item,
                semantic_scorer=semantic_scorer,
                common_answer_bank_trace=common_answer_bank_trace,
                common_answer_bank_context=common_answer_bank_context,
            )
        else:
            result = {
                "groundedness_score": 0.0,
                "groundedness_confidence": 0.0,
                "formula": "unsupported_task",
                "subscores": {},
                "evidence": {"reason": f"Unsupported task_type={task_type}"},
            }

        groundedness = clip(float(result.get("groundedness_score", 0.0)))
        confidence = clip(float(result.get("groundedness_confidence", 0.0)))
        task_penalty_stats = self._resolve_task_penalty_stats(task_type, cohort_stats=cohort_stats)
        penalty, penalty_trace = self._compute_penalty(
            task_type=task_type,
            groundedness=groundedness,
            confidence=confidence,
            raw_originality=raw_originality,
            task_stats=task_penalty_stats,
            return_trace=True,
        )
        task_threshold = penalty_trace.get("tau_eff")
        if task_type == "UUT" and isinstance(result.get("dual_axis_primitives"), dict):
            primitives = dict(result["dual_axis_primitives"])
            novelty = clip(float(raw_originality or 0.0))
            supported = clip(float(primitives.get("supported_affordance_ratio") or 0.0))
            mechanism = clip(float(primitives.get("mechanism_completeness") or 0.0))
            appropriateness_gate = clip(float(primitives.get("appropriateness_gate") or 0.0))
            primitives.update({
                "novelty": round(novelty, 4),
                "novelty_raw_input": round(float(raw_originality or 0.0), 4),
                "novelty_times_affordance_support": round(novelty * supported, 4),
                "novelty_times_groundedness": round(novelty * supported, 4),
                "appropriateness_gated_novelty_affordance_support": round(
                    novelty * supported * appropriateness_gate, 4
                ),
                "mechanism_times_affordance_support": round(mechanism * supported, 4),
                "mechanism_times_groundedness": round(mechanism * supported, 4),
                "appropriateness_gated_mechanism_affordance_support": round(
                    mechanism * supported * appropriateness_gate, 4
                ),
            })
            primitives["subtype_contributions"] = build_uut_idea_subtype_contributions(primitives)
            result["dual_axis_primitives"] = primitives
        result.update({
            "score_version": WHITE_BOX_GROUNDEDNESS_VERSION,
            "g_new": round(groundedness, 4),
            "g_subscores_v5": result.get("subscores", {}),
            "cohort_z": penalty_trace.get("cohort_z"),
            "cohort_stats": penalty_trace.get("task_stats"),
            "penalty_trace_v5": penalty_trace,
            "anti_cliche_score": result.get("subscores", {}).get("anti_cliche"),
            "mech_score": result.get("subscores", {}).get("mechanism_score"),
            "groundedness_score": round(groundedness, 4),
            "groundedness_confidence": round(confidence, 4),
            "groundedness_penalty": round(penalty, 4),
            "low_groundedness": groundedness < task_threshold,
            "task_threshold": task_threshold,
        })
        return result

    def get_data_source_label(self) -> Dict[str, str]:
        return {
            "swow": "active" if self.swow.available else "missing",
            "word_norms2": "active" if self.word_norms2.available else "missing",
            "wordnet": "active" if getattr(self.wn_analyzer, 'available', False) else "fallback",
            "reference_cohort": self.reference_cohort.get("source", "unknown"),
        }

    
    
    

    def _load_json(self, filename: str, default):
        path = self.data_dir / filename
        if not path.exists():
            return default
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)

    def _load_uut_profiles(self) -> Dict[str, object]:
        base = self._load_json("uut_affordance_profiles.json", default={})
        if not isinstance(base, dict):
            base = {}
        overlay = self._load_json("uut_affordance_profiles_v2.json", default={})
        if not isinstance(overlay, dict) or not overlay:
            return base

        merged = copy.deepcopy(base)
        shared = dict(merged.get("__shared__", {}))
        if isinstance(overlay.get("__shared__"), dict):
            shared.update(overlay.get("__shared__") or {})
        merged["__shared__"] = shared
        merged["__schema__"] = overlay.get("schema", "uut_affordance_profiles_v2")

        profiles = overlay.get("profiles")
        if not isinstance(profiles, dict):
            profiles = {
                key: value for key, value in overlay.items()
                if isinstance(value, dict) and not key.startswith("__") and key not in {"profiles"}
            }
        for task_id, metadata in profiles.items():
            if not isinstance(metadata, dict):
                continue
            profile = dict(merged.get(task_id) or {})
            for key, value in metadata.items():
                if key in {"base_properties", "negative_properties"} and isinstance(value, dict):
                    existing = dict(profile.get(key) or {})
                    existing.update(value)
                    profile[key] = existing
                else:
                    profile[key] = value
            profile["profile_schema"] = overlay.get("schema", "uut_affordance_profiles_v2")
            merged[task_id] = profile
        return merged

    def _init_uut_minor_helper_terms(self) -> set:
        terms = {self._normalize_phrase(term) for term in UUT_MINOR_HELPER_TERMS}
        shared = self.uut_profiles.get("__shared__", {}) if isinstance(self.uut_profiles, dict) else {}
        for term in shared.get("minor_helper_whitelist", []) if isinstance(shared, dict) else []:
            normalized = self._normalize_phrase(str(term or ""))
            if normalized:
                terms.add(normalized)
        return {term for term in terms if term}

    def _load_v5_config(self) -> Dict[str, object]:
        payload = self._load_json("groundedness_v5_config.json", default={})
        task_params = {
            task_type: dict(values)
            for task_type, values in DEFAULT_V5_TASK_PARAMS.items()
        }
        penalty_config = dict(DEFAULT_V5_PENALTY_CONFIG)
        task_weights = {
            task_type: dict(values)
            for task_type, values in V5_TASK_WEIGHTS.items()
        }

        if isinstance(payload, dict):
            for task_type, values in (payload.get("task_params") or {}).items():
                if isinstance(values, dict):
                    task_params.setdefault(task_type, {}).update(values)
            if isinstance(payload.get("penalty"), dict):
                penalty_config.update(payload.get("penalty") or {})
            for task_type, values in (payload.get("task_weights") or {}).items():
                if isinstance(values, dict):
                    task_weights.setdefault(task_type, {}).update(values)

        return {
            "schema": payload.get("schema", "benchmark_groundedness") if isinstance(payload, dict) else "benchmark_groundedness",
            "task_params": task_params,
            "penalty": penalty_config,
            "task_weights": task_weights,
            "source": "groundedness_v5_config.json" if isinstance(payload, dict) and payload else "internal_defaults",
        }

    def _load_reference_cohort(self) -> Dict[str, object]:
        payload = self._load_json("groundedness_reference_cohort.json", default={})
        if not isinstance(payload, dict) or not isinstance(payload.get("per_task"), dict):
            fallback = {
                "schema": DEFAULT_REFERENCE_COHORT["schema"],
                "source": DEFAULT_REFERENCE_COHORT["source"],
                "fallback": True,
                "per_task": {
                    task_type: dict(values)
                    for task_type, values in DEFAULT_REFERENCE_COHORT["per_task"].items()
                },
            }
            return fallback

        merged = {
            "schema": payload.get("schema", "benchmark_groundedness"),
            "source": "groundedness_reference_cohort.json",
            "fallback": False,
            "per_task": {
                task_type: dict(DEFAULT_REFERENCE_COHORT["per_task"].get(task_type, {}))
                for task_type in DEFAULT_V5_TASK_PARAMS
            },
        }
        for task_type, values in payload.get("per_task", {}).items():
            if isinstance(values, dict):
                merged["per_task"].setdefault(task_type, {}).update(values)
        return merged

    def get_reference_cohort(self) -> Dict[str, object]:
        return self.reference_cohort

    @staticmethod
    def _safe_float(value, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _resolve_task_penalty_stats(
        self,
        task_type: str,
        *,
        cohort_stats: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        task_params = dict(DEFAULT_V5_TASK_PARAMS.get(task_type, DEFAULT_V5_TASK_PARAMS["UUT"]))
        configured_task = (self.v5_config.get("task_params") or {}).get(task_type, {})
        if isinstance(configured_task, dict):
            task_params.update(configured_task)

        source = self.reference_cohort.get("source", "internal_prior")
        fallback = bool(self.reference_cohort.get("fallback", False))
        source_payload = cohort_stats if isinstance(cohort_stats, dict) else self.reference_cohort
        if isinstance(source_payload.get("per_task"), dict):
            task_stats = dict(source_payload.get("per_task", {}).get(task_type, {}))
            source = source_payload.get("source", source)
            fallback = bool(source_payload.get("fallback", fallback))
        else:
            task_stats = dict(source_payload)
            if task_stats:
                source = task_stats.get("source", source)
                fallback = bool(task_stats.get("fallback", fallback))

        defaults = DEFAULT_REFERENCE_COHORT["per_task"].get(task_type, DEFAULT_REFERENCE_COHORT["per_task"]["UUT"])
        g_median = clip(self._safe_float(task_stats.get("g_median"), defaults["g_median"]))
        g_mad = max(1e-3, self._safe_float(task_stats.get("g_mad"), defaults["g_mad"]))
        novelty_median = max(1e-3, self._safe_float(task_stats.get("novelty_median"), defaults["novelty_median"]))
        novelty_mad = max(1e-3, self._safe_float(task_stats.get("novelty_mad"), defaults["novelty_mad"]))
        tau_abs_floor = clip(self._safe_float(task_params.get("tau_abs_floor"), 0.20))
        tau_eff = max(tau_abs_floor, g_median)

        return {
            "task_type": task_type,
            "g_median": round(g_median, 6),
            "g_mad": round(g_mad, 6),
            "novelty_median": round(novelty_median, 6),
            "novelty_mad": round(novelty_mad, 6),
            "tau_abs_floor": round(tau_abs_floor, 6),
            "tau_eff": round(tau_eff, 6),
            "lambda": self._safe_float(task_params.get("lambda"), 0.20),
            "source": source,
            "fallback": fallback,
            "schema": source_payload.get("schema") if isinstance(source_payload, dict) else None,
        }

    def _lemmatize(self, token: str) -> str:
        token = token.lower().strip()
        if not token:
            return token
        if self.wn_analyzer and hasattr(self.wn_analyzer, 'lemmatize_token'):
            try:
                return self.wn_analyzer.lemmatize_token(token)
            except Exception:
                pass
        if token.endswith('ies') and len(token) > 4:
            return token[:-3] + 'y'
        if token.endswith('es') and len(token) > 4 and not token.endswith(('ses', 'xes', 'zes')):
            return token[:-2]
        if token.endswith('s') and len(token) > 3 and not token.endswith('ss'):
            return token[:-1]
        return token

    def _normalize_phrase(self, text: str) -> str:
        if self.wn_analyzer and hasattr(self.wn_analyzer, 'normalize_phrase'):
            try:
                normalized = self.wn_analyzer.normalize_phrase(text)
                if normalized:
                    return normalized
            except Exception:
                pass
        tokens = [self._lemmatize(token) for token in re.findall(r"[a-zA-Z]+", text.lower())]
        tokens = [token for token in tokens if token and token not in {'a', 'an', 'the'}]
        return ' '.join(tokens).strip()

    def _extract_content_tokens(self, text: str, excluded: Optional[Iterable[str]] = None) -> List[str]:
        excluded_set = {self._lemmatize(token) for token in (excluded or []) if token}
        tokens = []
        for raw in re.findall(r"[a-zA-Z]+", text.lower()):
            token = self._lemmatize(raw)
            if not token or token in GENERIC_STOPWORDS or token in excluded_set:
                continue
            tokens.append(token)
        return tokens

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def _compute_penalty(
        self,
        *,
        task_type: str,
        groundedness: float,
        confidence: float,
        raw_originality: float,
        task_stats: Optional[Dict[str, object]] = None,
        return_trace: bool = False,
    ):
        stats = task_stats or self._resolve_task_penalty_stats(task_type)
        penalty_config = self.v5_config.get("penalty") or {}
        scale = self._safe_float(penalty_config.get("scale"), DEFAULT_V5_PENALTY_CONFIG["scale"])
        shape_exp = self._safe_float(
            penalty_config.get("deficit_shape_exp"),
            DEFAULT_V5_PENALTY_CONFIG["deficit_shape_exp"],
        )
        nov_lo = self._safe_float(
            penalty_config.get("novelty_scale_lo"),
            DEFAULT_V5_PENALTY_CONFIG["novelty_scale_lo"],
        )
        nov_hi = self._safe_float(
            penalty_config.get("novelty_scale_hi"),
            DEFAULT_V5_PENALTY_CONFIG["novelty_scale_hi"],
        )
        low_conf_cutoff = self._safe_float(
            penalty_config.get("low_confidence_cutoff"),
            DEFAULT_V5_PENALTY_CONFIG["low_confidence_cutoff"],
        )
        low_conf_multiplier = self._safe_float(
            penalty_config.get("low_confidence_multiplier"),
            DEFAULT_V5_PENALTY_CONFIG["low_confidence_multiplier"],
        )

        novelty = max(0.0, float(raw_originality or 0.0))
        groundedness = clip(float(groundedness or 0.0))
        confidence = clip(float(confidence or 0.0))
        g_median = clip(self._safe_float(stats.get("g_median"), DEFAULT_REFERENCE_COHORT["per_task"]["UUT"]["g_median"]))
        g_mad = max(1e-3, self._safe_float(stats.get("g_mad"), DEFAULT_REFERENCE_COHORT["per_task"]["UUT"]["g_mad"]))
        novelty_median = max(
            1e-3,
            self._safe_float(stats.get("novelty_median"), DEFAULT_REFERENCE_COHORT["per_task"]["UUT"]["novelty_median"]),
        )
        lam = max(0.0, self._safe_float(stats.get("lambda"), DEFAULT_V5_TASK_PARAMS.get(task_type, DEFAULT_V5_TASK_PARAMS["UUT"])["lambda"]))
        sigma = max(1e-3, 1.4826 * g_mad)
        z = (g_median - groundedness) / sigma
        raw_sigmoid_deficit = self._sigmoid(z) - 0.5
        deficit = max(0.0, raw_sigmoid_deficit) ** max(0.1, shape_exp)
        nov_scale = clip(novelty / novelty_median, min(nov_lo, nov_hi), max(nov_lo, nov_hi))
        penalty = confidence * lam * scale * deficit * nov_scale
        confidence_deferral_applied = False
        if confidence < low_conf_cutoff and penalty > 0.0:
            penalty *= low_conf_multiplier
            confidence_deferral_applied = True
        penalty = min(novelty, penalty) if novelty > 0.0 else 0.0

        trace = {
            "formula": "cohort_relative_sigmoid_deficit_v5",
            "task_type": task_type,
            "task_stats": {
                "g_median": round(g_median, 6),
                "g_mad": round(g_mad, 6),
                "novelty_median": round(novelty_median, 6),
                "novelty_mad": round(self._safe_float(stats.get("novelty_mad"), 0.0), 6),
                "tau_abs_floor": round(self._safe_float(stats.get("tau_abs_floor"), 0.20), 6),
                "tau_eff": round(self._safe_float(stats.get("tau_eff"), max(0.20, g_median)), 6),
                "lambda": round(lam, 6),
                "source": stats.get("source"),
                "fallback": bool(stats.get("fallback", False)),
                "schema": stats.get("schema"),
            },
            "tau_eff": round(self._safe_float(stats.get("tau_eff"), max(0.20, g_median)), 6),
            "cohort_z": round(z, 6),
            "sigma": round(sigma, 6),
            "raw_sigmoid_deficit": round(raw_sigmoid_deficit, 6),
            "deficit": round(deficit, 6),
            "novelty_scale": round(nov_scale, 6),
            "confidence_deferral_applied": confidence_deferral_applied,
            "parameters": {
                "scale": round(scale, 6),
                "deficit_shape_exp": round(shape_exp, 6),
                "novelty_scale_lo": round(min(nov_lo, nov_hi), 6),
                "novelty_scale_hi": round(max(nov_lo, nov_hi), 6),
                "low_confidence_cutoff": round(low_conf_cutoff, 6),
                "low_confidence_multiplier": round(low_conf_multiplier, 6),
            },
        }

        if return_trace:
            return float(penalty), trace
        return float(penalty)

    def _compute_penalty_legacy(self, groundedness: float, confidence: float, tau: float, lam: float, raw_originality: float) -> float:
        if raw_originality <= 0:
            return 0.0
        deficit = max(0.0, tau - groundedness) / tau if tau > 0 else 0.0
        shaped_deficit = 0.45 * deficit + 0.55 * (deficit ** 2)
        penalty = confidence * lam * GROUNDEDNESS_PENALTY_SCALE * shaped_deficit
        return min(float(raw_originality), penalty)

    def _swow_support(self, cues: Sequence[str], idea_text: str, excluded_tokens: Optional[Iterable[str]] = None) -> Dict[str, object]:
        cues = [self._normalize_phrase(cue) for cue in cues if cue]
        cues = [cue for cue in cues if cue]
        return self.swow.score_answer_support(
            cues=cues,
            answer_text=idea_text,
            excluded_tokens=excluded_tokens,
        )

    def _combine_v5_groundedness(
        self,
        task_type: str,
        *,
        g_swow_path: float,
        g_feat_v2: float,
        g_mech: float,
        anti_cliche: float,
        drift_score: float = 0.0,
    ) -> float:
        weights = self._get_v5_task_weights(task_type)
        anti_effective = self._effective_anti_cliche(
            g_swow_path=g_swow_path,
            g_feat_v2=g_feat_v2,
            g_mech=g_mech,
            anti_cliche=anti_cliche,
            drift_score=drift_score,
        )
        return clip(
            weights["path"] * clip(g_swow_path) +
            weights["feat"] * clip(g_feat_v2) +
            weights["mech"] * clip(g_mech) +
            weights["anti"] * anti_effective -
            weights["drift"] * clip(drift_score)
        )

    def _effective_anti_cliche(
        self,
        *,
        g_swow_path: float,
        g_feat_v2: float,
        g_mech: float,
        anti_cliche: float,
        drift_score: float,
    ) -> float:
        del g_swow_path, g_mech
        cfg = self.v5_config.get("anti_cliche_effective") or {}
        feature_rescue = self._safe_float(
            cfg.get("feature_rescue"),
            DEFAULT_ANTI_CLICHE_EFFECTIVE_CONFIG["feature_rescue"],
        )
        drift_suppression = self._safe_float(
            cfg.get("drift_suppression"),
            DEFAULT_ANTI_CLICHE_EFFECTIVE_CONFIG["drift_suppression"],
        )
        return clip(
            clip(anti_cliche) +
            feature_rescue * clip(g_feat_v2) -
            drift_suppression * clip(drift_score)
        )

    def _get_v5_task_weights(self, task_type: str) -> Dict[str, float]:
        configured = (self.v5_config.get("task_weights") or {}).get(task_type)
        defaults = V5_TASK_WEIGHTS.get(task_type, V5_TASK_WEIGHTS["UUT"])
        weights = dict(defaults)
        if isinstance(configured, dict):
            for key in ["path", "feat", "mech", "anti", "drift"]:
                if key in configured:
                    weights[key] = clip(self._safe_float(configured.get(key), weights[key]))
        total = sum(max(0.0, float(weights.get(key, 0.0))) for key in ["path", "feat", "mech", "anti", "drift"])
        if total <= 0.0:
            return dict(defaults)
        return {
            key: round(max(0.0, float(weights.get(key, 0.0))) / total, 6)
            for key in ["path", "feat", "mech", "anti", "drift"]
        }

    def _swow_activation_k(
        self,
        cue_tokens: Sequence[str],
        idea_tokens: Sequence[str],
        *,
        k: int = 3,
        alpha: float = 0.15,
        top_n: int = 200,
    ) -> Dict[str, object]:
        cue_terms: List[str] = []
        seen_cues = set()
        for cue in cue_tokens or []:
            normalized_phrase = self._normalize_phrase(str(cue))
            for candidate in [normalized_phrase, *self._extract_content_tokens(str(cue))]:
                normalized = self.swow.normalize_token(candidate) if self.swow.available else self._normalize_phrase(candidate)
                if normalized and normalized not in seen_cues:
                    seen_cues.add(normalized)
                    cue_terms.append(normalized)

        idea_terms = []
        seen_terms = set()
        for token in idea_tokens or []:
            normalized = self.swow.normalize_token(str(token)) if self.swow.available else self._lemmatize(str(token))
            if normalized and normalized not in GENERIC_STOPWORDS and normalized not in seen_terms:
                seen_terms.add(normalized)
                idea_terms.append(normalized)

        if not cue_terms or not idea_terms or not self.swow.available:
            return {
                "score": 0.0,
                "m_by_hop": {},
                "token_scores": {},
                "hit_tokens_by_hop": {},
                "cue_terms": cue_terms,
                "idea_terms": idea_terms,
                "available": self.swow.available,
            }

        frontier = {cue: 1.0 / len(cue_terms) for cue in cue_terms}
        hop_weights = {1: 0.55, 2: 0.30, 3: 0.15}
        m_by_hop: Dict[int, float] = {}
        hit_tokens_by_hop: Dict[int, List[str]] = {}
        token_scores: Dict[str, float] = {}

        for hop in range(1, k + 1):
            next_frontier: Dict[str, float] = defaultdict(float)
            for node, mass in frontier.items():
                neighbors = [(resp, float(strength)) for resp, strength in self.swow.top_associates(node, k=top_n) if float(strength) > 0.0]
                if not neighbors:
                    continue
                total_strength = sum(strength for _, strength in neighbors) or 1.0
                for response, strength in neighbors:
                    next_frontier[response] += mass * (1.0 - alpha) * (strength / total_strength)

            if not next_frontier:
                m_by_hop[hop] = 0.0
                hit_tokens_by_hop[hop] = []
                frontier = {}
                continue

            ranked_frontier = sorted(next_frontier.items(), key=lambda item: item[1], reverse=True)[:top_n]
            frontier = dict(ranked_frontier)
            max_mass = max(frontier.values()) if frontier else 0.0
            hop_hits = []
            normalized_scores = []
            for token in idea_terms:
                mass = float(frontier.get(token, 0.0))
                if mass <= 0.0:
                    continue
                normalized = clip(mass / max(max_mass, 1e-12))
                token_scores[token] = max(token_scores.get(token, 0.0), normalized * hop_weights.get(hop, 0.0))
                normalized_scores.append(normalized)
                hop_hits.append(token)

            coverage = len(hop_hits) / len(idea_terms) if idea_terms else 0.0
            if normalized_scores:
                top_k = min(3, len(normalized_scores))
                top_mean = sum(sorted(normalized_scores, reverse=True)[:top_k]) / top_k
                m_h = clip(0.65 * top_mean + 0.35 * coverage)
            else:
                m_h = 0.0
            m_by_hop[hop] = m_h
            hit_tokens_by_hop[hop] = sorted(hop_hits)

        score = clip(sum(hop_weights.get(hop, 0.0) * m_by_hop.get(hop, 0.0) for hop in range(1, k + 1)))
        return {
            "score": round(score, 4),
            "m_by_hop": {str(hop): round(value, 4) for hop, value in m_by_hop.items()},
            "token_scores": {token: round(value, 4) for token, value in sorted(token_scores.items(), key=lambda item: item[1], reverse=True)[:12]},
            "hit_tokens_by_hop": {str(hop): tokens for hop, tokens in hit_tokens_by_hop.items()},
            "cue_terms": cue_terms[:24],
            "idea_terms": idea_terms[:24],
            "available": self.swow.available,
            "alpha": alpha,
            "top_n": top_n,
        }

    def _semantic_model(self, semantic_scorer):
        if semantic_scorer is None:
            return None
        return getattr(semantic_scorer, "model", None)

    def _cosine_scores(self, query_vector, candidate_vectors) -> List[float]:
        if np is not None:
            query = np.asarray(query_vector, dtype=float)
            candidates = np.asarray(candidate_vectors, dtype=float)
            if query.ndim != 1 or candidates.ndim != 2 or candidates.shape[0] == 0:
                return []
            query_norm = np.linalg.norm(query)
            candidate_norms = np.linalg.norm(candidates, axis=1)
            denom = np.where(candidate_norms * query_norm == 0.0, 1e-12, candidate_norms * query_norm)
            sims = np.clip(candidates @ query / denom, -1.0, 1.0)
            return [float(value) for value in sims.tolist()]

        q = [float(value) for value in query_vector]
        q_norm = math.sqrt(sum(value * value for value in q)) or 1e-12
        scores = []
        for vector in candidate_vectors:
            candidate = [float(value) for value in vector]
            c_norm = math.sqrt(sum(value * value for value in candidate)) or 1e-12
            dot = sum(a * b for a, b in zip(q, candidate))
            scores.append(max(-1.0, min(1.0, dot / (q_norm * c_norm))))
        return scores

    def _anchor_vectors(self, semantic_scorer, cache_key: str, anchors: Sequence[str]):
        model = self._semantic_model(semantic_scorer)
        if model is None or not anchors:
            return None
        key = (id(model), cache_key, "|".join(anchors))
        if key not in self._embedding_anchor_cache:
            self._embedding_anchor_cache[key] = model.encode(list(anchors))
        return self._embedding_anchor_cache[key]

    def _embedding_score_against_anchors(
        self,
        *,
        semantic_scorer,
        cache_key: str,
        query: str,
        anchors: Sequence[str],
    ) -> Dict[str, object]:
        model = self._semantic_model(semantic_scorer)
        anchors = [str(anchor).strip() for anchor in anchors or [] if str(anchor).strip()]
        if model is None or not query or not anchors:
            return {
                "score": 0.0,
                "available": False,
                "query": query,
                "anchor_count": len(anchors),
            }
        try:
            query_vector = model.encode([query])[0]
            anchor_vectors = self._anchor_vectors(semantic_scorer, cache_key, anchors)
            similarities = self._cosine_scores(query_vector, anchor_vectors)
        except Exception as exc:
            return {
                "score": 0.0,
                "available": False,
                "query": query,
                "anchor_count": len(anchors),
                "error": str(exc)[:160],
            }
        if not similarities:
            return {
                "score": 0.0,
                "available": False,
                "query": query,
                "anchor_count": len(anchors),
            }
        best_index = max(range(len(similarities)), key=lambda index: similarities[index])
        best_similarity = float(similarities[best_index])
        score = clip((best_similarity - 0.35) / 0.35)
        return {
            "score": round(score, 4),
            "available": True,
            "query": query,
            "anchor_count": len(anchors),
            "best_anchor": anchors[best_index],
            "best_similarity": round(best_similarity, 4),
        }

    def _embedding_property_fit(
        self,
        task_type: str,
        task_id: str,
        cue: str,
        idea_text: str,
        *,
        parsed_item: Optional[Dict[str, object]] = None,
        semantic_scorer=None,
    ) -> Dict[str, object]:
        parsed_item = parsed_item or {}
        if task_type == "Instances":
            entry = self.instances_trait_lexicon.get(task_id) or {}
            anchor_entry = self.instances_trait_anchors.get(task_id) or {}
            trait = anchor_entry.get("trait") or entry.get("trait_text") or cue
            core = parsed_item.get("noun_phrase") or parsed_item.get("display_text") or idea_text
            query = f"{core} is {trait}"
            anchors = list(anchor_entry.get("anchors") or [])
        elif task_type == "UUT":
            anchor_entry = self.uut_affordance_anchors.get(task_id) or {}
            shared = self.uut_affordance_anchors.get("__affordance_anchors__", {})
            profile = self.uut_profiles.get(task_id) or {}
            title = parsed_item.get("idea_title") or parsed_item.get("display_text") or idea_text
            mechanism = parsed_item.get("mechanism") or ""
            idea_action = f"{title} {mechanism}".strip()
            query = f"{cue} can be used to {idea_action}"
            anchors = list(anchor_entry.get("anchors") or [])
            affordances = anchor_entry.get("affordances") or [
                name for name, strength in (profile.get("base_properties") or {}).items()
                if float(strength) >= 0.45
            ]
            for affordance in affordances:
                anchors.extend(shared.get(affordance, [])[:8])
        elif task_type == "JST":
            anchor_entry = self.jst_consequence_anchors.get(task_id) or {}
            scenario = anchor_entry.get("scenario_text") or cue
            consequence = parsed_item.get("consequence_clause") or parsed_item.get("display_text") or idea_text
            query = f"If {scenario}, then {consequence}"
            anchors = list(anchor_entry.get("anchors") or [])
        else:
            query = idea_text
            anchors = []

        return self._embedding_score_against_anchors(
            semantic_scorer=semantic_scorer,
            cache_key=f"{task_type}:{task_id}",
            query=query,
            anchors=anchors,
        )

    def _embedding_text_similarity(self, text_a: str, text_b: str, semantic_scorer=None) -> Optional[float]:
        model = self._semantic_model(semantic_scorer)
        if model is None or not text_a or not text_b:
            return None
        try:
            vectors = model.encode([text_a, text_b])
            scores = self._cosine_scores(vectors[0], [vectors[1]])
            return scores[0] if scores else None
        except Exception:
            return None

    def _extract_mechanism_triples(self, idea_text: str, cue: str) -> List[Dict[str, str]]:
        raw_tokens = re.findall(r"[A-Za-z]+", idea_text or "")
        content_tokens = self._extract_content_tokens(idea_text)
        cue_tokens = self._extract_content_tokens(cue)
        subject_default = " ".join(cue_tokens[:2]) or self._normalize_phrase(cue) or "object"
        triples: List[Dict[str, str]] = []

        def add(subject: str, verb: str, obj: str, source: str):
            subject_norm = self._normalize_phrase(subject)
            verb_norm = self._lemmatize(verb)
            object_norm = self._normalize_phrase(obj)
            if not verb_norm or not object_norm:
                return
            record = {
                "subject": subject_norm or subject_default,
                "verb": verb_norm,
                "object": object_norm,
                "source": source,
            }
            key = (record["subject"], record["verb"], record["object"])
            if key not in {(item["subject"], item["verb"], item["object"]) for item in triples}:
                triples.append(record)

        lower = f" {idea_text.lower()} "
        for marker in [" because ", " so that ", " by ", " using ", " via ", " through ", " to ", " when ", " then "]:
            if marker not in lower:
                continue
            segment = lower.split(marker, 1)[1]
            segment = re.split(r"[.;,]", segment, maxsplit=1)[0]
            seg_tokens = self._extract_content_tokens(segment)
            if len(seg_tokens) >= 2:
                verb_index = next(
                    (
                        idx for idx, token in enumerate(seg_tokens)
                        if token in MECHANISM_VERB_HINTS or token.endswith(("ing", "ed", "s"))
                    ),
                    0,
                )
                if verb_index + 1 < len(seg_tokens):
                    add(subject_default, seg_tokens[verb_index], " ".join(seg_tokens[verb_index + 1:verb_index + 4]), f"marker:{marker.strip()}")

        try:
            import nltk
            tagged = nltk.pos_tag(raw_tokens)
        except Exception:
            tagged = []

        if tagged:
            normalized_raw = [self._lemmatize(token) for token, _ in tagged]
            for index, (token, tag) in enumerate(tagged):
                verb = self._lemmatize(token)
                if not (tag.startswith("VB") or verb in MECHANISM_VERB_HINTS):
                    continue
                left = [
                    normalized_raw[pos] for pos in range(max(0, index - 4), index)
                    if normalized_raw[pos] not in GENERIC_STOPWORDS
                ]
                right = [
                    normalized_raw[pos] for pos in range(index + 1, min(len(normalized_raw), index + 5))
                    if normalized_raw[pos] not in GENERIC_STOPWORDS
                ]
                if right:
                    add(" ".join(left[-2:]) or subject_default, verb, " ".join(right[:3]), "pos")

        if not triples and len(content_tokens) >= 3:
            verb_index = next(
                (idx for idx, token in enumerate(content_tokens) if token in MECHANISM_VERB_HINTS or token.endswith(("ing", "ed"))),
                None,
            )
            if verb_index is not None and verb_index + 1 < len(content_tokens):
                add(subject_default, content_tokens[verb_index], " ".join(content_tokens[verb_index + 1:verb_index + 4]), "fallback")

        return triples[:6]

    def _license_mechanism_triple(
        self,
        triple: Dict[str, str],
        cue: str,
        idea_text: str,
        semantic_scorer=None,
    ) -> Dict[str, object]:
        subject = triple.get("subject") or cue
        verb = triple.get("verb") or ""
        obj = triple.get("object") or ""
        cue_subject_tokens = set(self._extract_content_tokens(f"{cue} {subject}"))
        object_tokens = self._extract_content_tokens(obj)
        verb_tokens = self._extract_content_tokens(verb)
        target_tokens = {
            token for token in object_tokens + verb_tokens
            if token not in cue_subject_tokens and token not in GENERIC_STOPWORDS
        }

        feature_hit = False
        feature_predicates = []
        if self.word_norms2.available:
            for phrase in [subject, cue]:
                features = self.word_norms2.get_concept_features(phrase)
                for predicate, strength in features.items():
                    predicate_tokens = set(self._extract_content_tokens(predicate.replace(":", " ")))
                    if float(strength) >= 0.04 and predicate_tokens.intersection(target_tokens):
                        feature_hit = True
                        feature_predicates.append(predicate)

        swow_hit = False
        swow_evidence = []
        if self.swow.available and target_tokens:
            cue_terms = [subject, cue]
            for token in sorted(target_tokens)[:6]:
                support = self.swow.compute_token_support(cue_terms, token, alpha=0.7, bridge_limit=35)
                direct = float(support.get("direct", 0.0))
                two_hop = float(support.get("two_hop", 0.0))
                support_score = float(support.get("score", 0.0))
                if direct >= 0.004 or (two_hop >= 0.0015 and support_score >= 0.0015):
                    swow_hit = True
                    swow_evidence.append({
                        "token": token,
                        "support_type": support.get("support_type"),
                        "score": round(support_score, 6),
                        "bridge": support.get("bridge"),
                    })

        embedding_similarity = self._embedding_text_similarity(
            f"{cue} can {verb} {obj}",
            idea_text,
            semantic_scorer=semantic_scorer,
        )
        embedding_hit = embedding_similarity is not None and embedding_similarity >= 0.55
        licensed = feature_hit or swow_hit or embedding_hit
        return {
            "triple": triple,
            "licensed": bool(licensed),
            "feature_hit": bool(feature_hit),
            "feature_predicates": sorted(set(feature_predicates))[:8],
            "swow_hit": bool(swow_hit),
            "swow_evidence": swow_evidence[:4],
            "embedding_similarity": round(float(embedding_similarity), 4) if embedding_similarity is not None else None,
            "embedding_hit": bool(embedding_hit),
        }

    def _mechanism_score(
        self,
        idea_text: str,
        cue: str,
        task_type: str,
        *,
        parsed_item: Optional[Dict[str, object]] = None,
        semantic_scorer=None,
    ) -> Dict[str, object]:
        parsed_item = parsed_item or {}
        has_connective = bool(MECHANISM_CONNECTIVE_RE.search(idea_text or ""))
        if task_type == "UUT" and parsed_item.get("mechanism"):
            has_connective = True
        triples = self._extract_mechanism_triples(idea_text, cue)
        license_records = [
            self._license_mechanism_triple(triple, cue, idea_text, semantic_scorer=semantic_scorer)
            for triple in triples
        ]
        licensed_ratio = (
            sum(1 for item in license_records if item.get("licensed")) / len(license_records)
            if license_records else 0.0
        )
        triple_count_norm = min(1.0, len(triples) / 2.0)
        score = clip(0.4 * float(has_connective) + 0.4 * licensed_ratio + 0.2 * triple_count_norm)

        if score == 0.0 and task_type == "UUT" and len(self._extract_content_tokens(idea_text)) >= 5:
            score = 0.15
        if task_type == "Instances" and not has_connective:
            score = min(score, 0.20)

        return {
            "score": round(score, 4),
            "has_connective": bool(has_connective),
            "triple_count_norm": round(triple_count_norm, 4),
            "licensed_ratio": round(licensed_ratio, 4),
            "triples": triples,
            "license_records": license_records,
        }

    def _impossibility_drift_score(
        self,
        task_id: str,
        task_type: str,
        idea_text: str,
        *,
        g_feat_v2: float = 0.0,
        base_drift: float = 0.0,
    ) -> Dict[str, object]:
        hits = []
        for pattern, score, reason in IMPOSSIBILITY_PATTERNS:
            if pattern.search(idea_text or ""):
                hits.append({"reason": reason, "score": score})

        trait_conflict = 0.0
        conflict_tokens = []
        if task_type == "Instances":
            idea_tokens = set(self._extract_content_tokens(idea_text))
            conflict_tokens = sorted(idea_tokens.intersection(INSTANCE_TRAIT_CONFLICT_TOKENS.get(task_id, set())))
            if conflict_tokens:
                trait_conflict = min(1.0, 0.45 + 0.15 * len(conflict_tokens))
            if g_feat_v2 < 0.25 and not hits:
                trait_conflict = max(trait_conflict, 0.35)

        impossibility = max([float(item["score"]) for item in hits], default=0.0)
        score = clip(max(float(base_drift or 0.0), impossibility, trait_conflict))
        return {
            "score": round(score, 4),
            "base_drift": round(float(base_drift or 0.0), 4),
            "impossibility_score": round(impossibility, 4),
            "trait_conflict_score": round(trait_conflict, 4),
            "trait_conflict_tokens": conflict_tokens,
            "hits": hits,
        }

    def _anti_cliche_signal(
        self,
        *,
        task_id: str,
        task_type: str,
        idea_text: str,
        parsed_item: Optional[Dict[str, object]] = None,
        semantic_scorer=None,
        common_answer_bank_trace: Optional[Dict[str, object]] = None,
        common_answer_bank_context: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        trace = common_answer_bank_trace
        if trace is None and semantic_scorer is not None and common_answer_bank_context is not None:
            try:
                trace = score_common_answer_bank_novelty(
                    task_id,
                    task_type=task_type,
                    response_text=idea_text,
                    parsed_item=parsed_item,
                    scorer=semantic_scorer,
                    bank_context=common_answer_bank_context,
                )
            except Exception:
                trace = None

        bank_result = bank_distance_for_anti_cliche(
            parsed_item=parsed_item,
            bank_trace=trace,
        )
        distance = bank_result.get("nearest_overall_distance")
        if distance is None:
            return {
                "score": 0.0,
                "available": False,
                "nearest_overall_distance": None,
                "nearest_overall_entry": None,
            }

        
        
        
        anti_score = clip((float(distance) - 0.18) / 0.62)
        return {
            "score": round(anti_score, 4),
            "available": True,
            "nearest_overall_distance": round(float(distance), 4),
            "nearest_overall_entry": bank_result.get("nearest_overall_entry"),
            "core_text": bank_result.get("core_text"),
            "source": bank_result.get("source"),
        }

    def _confidence_from_evidence(
        self,
        *,
        evidence_strength: float = 0.0,
        coverage: float = 0.0,
        agreement: float = 0.0,
        specificity: float = 0.0,
        ambiguity: float = 0.0,
        cap: float = 1.0,
    ) -> Tuple[float, Dict[str, float]]:
        strength = clip(evidence_strength)
        coverage = clip(coverage)
        agreement = clip(agreement)
        specificity = clip(specificity)
        ambiguity = clip(ambiguity)

        confidence = clip(
            0.08 +
            0.34 * strength +
            0.24 * coverage +
            0.16 * agreement +
            0.18 * specificity -
            0.20 * ambiguity,
            0.0,
            cap,
        )
        components = {
            "evidence_strength": round(strength, 4),
            "coverage": round(coverage, 4),
            "agreement": round(agreement, 4),
            "specificity": round(specificity, 4),
            "ambiguity": round(ambiguity, 4),
        }
        return confidence, components

    def _match_keywords_in_text(self, keywords: Sequence[str], idea_text: str) -> List[str]:
        idea_lower = idea_text.lower()
        normalized_idea = self._normalize_phrase(idea_text)
        idea_tokens = set(self._extract_content_tokens(idea_text))
        hits: List[str] = []
        for keyword in keywords or []:
            keyword_text = str(keyword).strip().lower()
            if not keyword_text:
                continue
            normalized_keyword = self._normalize_phrase(keyword_text)
            keyword_tokens = set(self._extract_content_tokens(keyword_text))
            if keyword_text in idea_lower:
                hits.append(keyword_text)
                continue
            if normalized_keyword and normalized_keyword in normalized_idea:
                hits.append(keyword_text)
                continue
            if keyword_tokens and keyword_tokens.issubset(idea_tokens):
                hits.append(keyword_text)
        return hits

    def _max_predicate_strength(self, source_map: Dict[str, float], predicates: Sequence[str]) -> Tuple[float, Optional[str]]:
        best_strength = 0.0
        best_predicate = None
        for predicate in predicates:
            strength = float(source_map.get(predicate, 0.0))
            if strength > best_strength:
                best_strength = strength
                best_predicate = predicate
        return best_strength, best_predicate

    def _match_fallback_rule(self, rules: Sequence[Dict[str, object]], idea_text: str) -> Tuple[Dict[str, float], List[str]]:
        idea_lower = idea_text.lower()
        strengths: Dict[str, float] = {}
        matched_keywords: List[str] = []
        for rule in rules or []:
            keywords = [str(item).lower() for item in rule.get("keywords", [])]
            if any(keyword in idea_lower for keyword in keywords):
                matched_keywords.extend([keyword for keyword in keywords if keyword in idea_lower])
                for predicate, value in rule.get("predicate_strengths", {}).items():
                    strengths[predicate] = max(float(value), strengths.get(predicate, 0.0))
        return strengths, matched_keywords

    def _resolve_word_norms2_match(self, phrase: str, aliases: Optional[Sequence[str]] = None):
        candidates = [phrase]
        candidates.extend(list(aliases or []))
        for candidate in candidates:
            match = self.word_norms2.match_concept(candidate) if self.word_norms2.available else None
            if match and match.concept:
                return match
        return self.word_norms2.match_concept(phrase) if self.word_norms2.available else None

    def _extract_candidate_phrases(self, text: str, *, max_ngram: int = 4, max_candidates: int = 16) -> List[str]:
        candidates: List[str] = []
        seen = set()

        def add(value: Optional[str]):
            normalized = self._normalize_phrase(value or "")
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        raw = (text or "").strip()
        add(raw)

        tokens = self._extract_content_tokens(raw)
        if self.wn_analyzer and hasattr(self.wn_analyzer, "analyze_idea"):
            try:
                analysis = self.wn_analyzer.analyze_idea(raw)
            except Exception:
                analysis = None
            if isinstance(analysis, dict):
                add(analysis.get("representative_concept"))
                for noun in analysis.get("nouns", [])[:8]:
                    add(noun)

        lowered = raw.lower()
        for marker in [" because ", " with ", " using ", " by ", " for ", " of ", " made of "]:
            marker_index = lowered.find(marker)
            if marker_index >= 0:
                left = raw[:marker_index]
                right = raw[marker_index + len(marker):]
                add(left)
                add(right)

        for n in range(min(max_ngram, len(tokens)), 0, -1):
            for start in range(0, max(0, len(tokens) - n + 1)):
                add(" ".join(tokens[start:start + n]))

        return candidates[:max_candidates]

    def _expand_swow_cues(self, cues: Sequence[str], *, top_k: int = 4, min_strength: float = 0.02) -> List[str]:
        expanded: List[str] = []
        seen = set()

        for cue in cues or []:
            cue_norm = self._normalize_phrase(cue)
            if not cue_norm or cue_norm in seen:
                continue
            seen.add(cue_norm)
            expanded.append(cue_norm)

            if not self.swow.available:
                continue
            for response, strength in self.swow.top_associates(cue_norm, k=top_k):
                response_norm = self._normalize_phrase(response)
                if strength < min_strength or not response_norm or response_norm in seen:
                    continue
                seen.add(response_norm)
                expanded.append(response_norm)

        return expanded

    def _estimate_ambiguity(self, tokens: Sequence[str], supported_tokens: Iterable[str]) -> float:
        token_list = [token for token in tokens if token]
        if not token_list:
            return 0.0
        supported = {self._lemmatize(token) for token in supported_tokens if token}
        unsupported_count = sum(1 for token in token_list if token not in supported)
        unsupported_ratio = unsupported_count / len(token_list)
        length_penalty = max(0.0, len(token_list) - 6) / max(1.0, len(token_list))
        return clip(0.65 * unsupported_ratio + 0.35 * length_penalty)

    def _infer_answer_affordances(self, idea_text: str) -> Dict[str, object]:
        affordance_scores: Dict[str, float] = {}
        concept_evidence: List[Dict[str, object]] = []

        for phrase in self._extract_candidate_phrases(idea_text, max_ngram=4, max_candidates=16):
            if not phrase:
                continue
            profile = self.word_norms2.derive_affordance_profile(phrase) if self.word_norms2.available else {}
            base_props = profile.get("base_properties", {}) if isinstance(profile, dict) else {}
            if not base_props:
                continue

            matched = {
                affordance: round(float(strength), 4)
                for affordance, strength in base_props.items()
                if float(strength) >= 0.08
            }
            if not matched:
                continue

            concept_evidence.append({
                "phrase": phrase,
                "affordances": matched,
            })
            for affordance, strength in matched.items():
                affordance_scores[affordance] = max(float(strength), affordance_scores.get(affordance, 0.0))

        return {
            "affordance_scores": affordance_scores,
            "concept_evidence": concept_evidence,
        }

    def _soft_channel_score(self, keywords: Sequence[str], idea_text: str, excluded_tokens: Optional[Iterable[str]] = None) -> Dict[str, object]:
        hits = self._match_keywords_in_text(keywords, idea_text)
        keyword_score = min(1.0, len(set(hits)) / 2.0) if hits else 0.0
        expanded_keywords = self._expand_swow_cues(keywords, top_k=3, min_strength=0.018)
        swow_support = self._swow_support(expanded_keywords, idea_text, excluded_tokens=excluded_tokens)
        semantic_score = clip(float(swow_support.get("score", 0.0)) / 0.22)
        score = max(keyword_score, semantic_score)
        return {
            "score": round(score, 4),
            "keyword_score": round(keyword_score, 4),
            "semantic_score": round(semantic_score, 4),
            "hits": hits,
            "expanded_keywords": expanded_keywords[:12],
            "swow_support": swow_support,
        }

    
    
    

    def _score_instances(
        self,
        task_id: str,
        target_concept: str,
        idea_text: str,
        *,
        parsed_item: Optional[Dict[str, object]] = None,
        semantic_scorer=None,
        common_answer_bank_trace: Optional[Dict[str, object]] = None,
        common_answer_bank_context: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        entry = self.instances_trait_lexicon.get(task_id) or {}
        cues = entry.get("prompt_cues") or self._extract_content_tokens(target_concept)
        excluded = set(self._extract_content_tokens(target_concept))
        swow_result = self._swow_support(cues, idea_text, excluded_tokens=excluded)
        candidate_phrases = self._extract_candidate_phrases(idea_text, max_ngram=4, max_candidates=18)
        idea_tokens = self._extract_content_tokens(idea_text, excluded=excluded)
        swow_path = self._swow_activation_k(cues, idea_tokens)

        candidate_records = []
        supported_tokens = set()
        for token_info in swow_result.get("token_evidence", []):
            if float(token_info.get("score", 0.0)) >= 0.004:
                supported_tokens.add(token_info.get("token"))

        for phrase in candidate_phrases:
            phrase_support = self._swow_support(cues, phrase, excluded_tokens=excluded)
            word_norms2_match = self._resolve_word_norms2_match(phrase)
            word_norms2_features = self.word_norms2.get_concept_features(phrase) if self.word_norms2.available else {}
            fallback_strengths, fallback_keywords = self._match_fallback_rule(entry.get("fallback_concept_rules", []), phrase)

            trait_scores = []
            trait_evidence = []
            source_hits = set()
            for group in entry.get("trait_groups", []):
                name = group.get("name")
                weight = float(group.get("weight", 1.0))
                pos_predicates = group.get("positive_predicates", [])
                neg_predicates = group.get("negative_predicates", [])

                pos_strength, pos_predicate = self._max_predicate_strength(word_norms2_features, pos_predicates)
                neg_strength, neg_predicate = self._max_predicate_strength(word_norms2_features, neg_predicates)
                source = "word_norms2" if pos_strength > 0 or neg_strength > 0 else None
                if source is None:
                    pos_strength, pos_predicate = self._max_predicate_strength(fallback_strengths, pos_predicates)
                    neg_strength, neg_predicate = self._max_predicate_strength(fallback_strengths, neg_predicates)
                    if pos_strength > 0 or neg_strength > 0:
                        source = "fallback_rule"
                if source:
                    source_hits.add(source)

                score = clip(pos_strength - 0.85 * neg_strength)
                trait_scores.append((weight, score))
                trait_evidence.append({
                    "name": name,
                    "weight": weight,
                    "score": round(score, 4),
                    "positive_predicate": pos_predicate,
                    "positive_strength": round(pos_strength, 4),
                    "negative_predicate": neg_predicate,
                    "negative_strength": round(neg_strength, 4),
                    "source": source,
                })

            if trait_scores:
                weighted_mean = sum(weight * score for weight, score in trait_scores) / sum(weight for weight, _ in trait_scores)
                min_score = min(score for _, score in trait_scores)
                feat_score = 0.65 * weighted_mean + 0.35 * min_score
            else:
                feat_score = 0.0

            candidate_strength = clip(0.78 * feat_score + 0.22 * float(phrase_support.get("score", 0.0)))
            candidate_records.append({
                "phrase": phrase,
                "candidate_score": round(candidate_strength, 4),
                "feature_fit": round(feat_score, 4),
                "swow_support": phrase_support.get("score", 0.0),
                "trait_scores": trait_evidence,
                "word_norms2_match": {
                    "concept": word_norms2_match.concept,
                    "matched_alias": word_norms2_match.matched_alias,
                    "confidence": word_norms2_match.confidence,
                } if word_norms2_match else None,
                "word_norms2_features": word_norms2_features if word_norms2_features else None,
                "fallback_keywords": sorted(set(fallback_keywords)),
                "sources": sorted(source_hits),
            })

        best_candidate = max(candidate_records, key=lambda item: item["candidate_score"], default=None)
        feat_score = float(best_candidate["feature_fit"]) if best_candidate else 0.0
        source_count = len(best_candidate.get("sources", [])) if best_candidate else 0
        if best_candidate:
            supported_tokens.update(self._extract_content_tokens(best_candidate["phrase"]))

        candidate_coverage = (
            sum(1 for item in candidate_records if item["candidate_score"] >= 0.35) / len(candidate_records)
            if candidate_records else 0.0
        )
        embedding_fit = self._embedding_property_fit(
            "Instances",
            task_id,
            target_concept,
            idea_text,
            parsed_item=parsed_item,
            semantic_scorer=semantic_scorer,
        )
        g_feat_v2 = max(feat_score, float(embedding_fit.get("score", 0.0)))
        mech_result = self._mechanism_score(
            idea_text,
            target_concept,
            "Instances",
            parsed_item=parsed_item,
            semantic_scorer=semantic_scorer,
        )
        anti_cliche = self._anti_cliche_signal(
            task_id=task_id,
            task_type="Instances",
            idea_text=idea_text,
            parsed_item=parsed_item,
            semantic_scorer=semantic_scorer,
            common_answer_bank_trace=common_answer_bank_trace,
            common_answer_bank_context=common_answer_bank_context,
        )
        drift_result = self._impossibility_drift_score(
            task_id,
            "Instances",
            idea_text,
            g_feat_v2=g_feat_v2,
            base_drift=0.0,
        )
        groundedness = self._combine_v5_groundedness(
            "Instances",
            g_swow_path=float(swow_path.get("score", 0.0)),
            g_feat_v2=g_feat_v2,
            g_mech=float(mech_result.get("score", 0.0)),
            anti_cliche=float(anti_cliche.get("score", 0.0)),
            drift_score=float(drift_result.get("score", 0.0)),
        )

        specificity = 0.0
        if best_candidate and best_candidate.get("word_norms2_match"):
            specificity = float(best_candidate["word_norms2_match"].get("confidence", 0.0))
        elif best_candidate and best_candidate.get("phrase"):
            specificity = min(1.0, len(best_candidate["phrase"].split()) / 3.0)
        legacy_agreement = 1.0 if source_count >= 2 else 0.55 if source_count == 1 else 0.0
        embedding_agreement = (
            1.0 if float(embedding_fit.get("score", 0.0)) >= 0.55 and feat_score >= 0.25 else
            0.65 if float(embedding_fit.get("score", 0.0)) >= 0.55 or float(swow_path.get("score", 0.0)) >= 0.30 else
            0.0
        )
        agreement = max(legacy_agreement, embedding_agreement)
        ambiguity = self._estimate_ambiguity(self._extract_content_tokens(idea_text, excluded=excluded), supported_tokens)
        confidence, confidence_components = self._confidence_from_evidence(
            evidence_strength=max(g_feat_v2, float(swow_path.get("score", 0.0)), float(mech_result.get("score", 0.0))),
            coverage=max(candidate_coverage, float(swow_path.get("score", 0.0))),
            agreement=agreement,
            specificity=max(specificity, float(mech_result.get("score", 0.0))),
            ambiguity=ambiguity,
        )
        return {
            "groundedness_score": clip(groundedness),
            "groundedness_confidence": confidence,
            "formula": "instances_groundedness_v5p0_signal",
            "subscores": {
                "g_swow_path": swow_path.get("score", 0.0),
                "feature_fit_legacy": round(feat_score, 4),
                "embedding_fit": embedding_fit.get("score", 0.0),
                "g_feat_v2": round(g_feat_v2, 4),
                "mechanism_score": mech_result.get("score", 0.0),
                "anti_cliche": anti_cliche.get("score", 0.0),
                "drift_score": drift_result.get("score", 0.0),
                "legacy_swow_support": swow_result.get("score", 0.0),
                "candidate_coverage": round(candidate_coverage, 4),
            },
            "evidence": {
                "target": target_concept,
                "v5_weights": self._get_v5_task_weights("Instances"),
                "swow_path": swow_path,
                "embedding_property_fit": embedding_fit,
                "mechanism": mech_result,
                "anti_cliche": anti_cliche,
                "impossibility_drift": drift_result,
                "candidate_phrases": candidate_phrases,
                "best_candidate": best_candidate,
                "candidate_records": candidate_records[:8],
                "confidence_components": confidence_components,
                "swow": swow_result,
            },
        }

    
    
    

    def _normalize_uut_affordance(self, affordance: object) -> Optional[str]:
        text = self._normalize_phrase(str(affordance or ""))
        if not text:
            return None
        canonical = text.replace("-", "_").replace(" ", "_")
        if canonical in UUT_AFFORDANCE_MAP:
            return canonical
        alias = UUT_AFFORDANCE_ALIASES.get(text) or UUT_AFFORDANCE_ALIASES.get(canonical.replace("_", " "))
        if alias in UUT_AFFORDANCE_MAP:
            return alias
        return canonical

    def _coerce_string_list(self, value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = re.split(r"[,;/|]+", value)
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raw_items = [value]

        results = []
        seen = set()
        for item in raw_items:
            text = self._normalize_phrase(str(item or ""))
            if not text or text in seen:
                continue
            seen.add(text)
            results.append(text)
        return results

    def _extract_uut_schema_fields(self, parsed_item: Optional[Dict[str, object]]) -> Dict[str, object]:
        parsed_item = parsed_item or {}
        raw_affordances = self._coerce_string_list(parsed_item.get("key_affordances"))
        declared_affordances = []
        unknown_affordances = []
        for raw in raw_affordances:
            normalized = self._normalize_uut_affordance(raw)
            if not normalized:
                continue
            if normalized in UUT_AFFORDANCE_MAP:
                declared_affordances.append(normalized)
            else:
                unknown_affordances.append(normalized)

        return {
            "required_extra_items": self._coerce_string_list(parsed_item.get("required_extra_items")),
            "declared_affordances": sorted(set(declared_affordances)),
            "unknown_affordances": sorted(set(unknown_affordances)),
            "main_object_role": self._normalize_phrase(str(parsed_item.get("main_object_role") or "")),
        }

    def _score_uut_extra_tool_violation(
        self,
        idea_text: str,
        schema_fields: Dict[str, object],
    ) -> Dict[str, object]:
        extra_items = list(schema_fields.get("required_extra_items") or [])
        main_role = str(schema_fields.get("main_object_role") or "")
        text = self._normalize_phrase(" ".join([idea_text, *extra_items]))
        minor_helper_terms = getattr(self, "uut_minor_helper_terms", UUT_MINOR_HELPER_TERMS)

        major_hits = []
        minor_hits = []
        for term in sorted(UUT_MAJOR_EXTRA_TOOL_TERMS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(term)}s?\b", text):
                major_hits.append(term)
        for term in sorted(minor_helper_terms, key=len, reverse=True):
            if re.search(rf"\b{re.escape(term)}s?\b", text):
                minor_hits.append(term)

        role_penalty = 0.0
        if main_role and main_role not in {"primary", "main", "core"}:
            role_penalty = 0.5 if main_role in {"supporting", "secondary", "helper"} else 0.8

        unknown_extra_count = 0
        for item in extra_items:
            if any(term in item for term in UUT_MAJOR_EXTRA_TOOL_TERMS):
                continue
            if any(term in item for term in minor_helper_terms):
                continue
            unknown_extra_count += 1

        score = clip(
            0.55 * min(1.0, len(major_hits) / 2.0) +
            0.25 * min(1.0, unknown_extra_count / 4.0) +
            0.20 * role_penalty
        )
        return {
            "score": round(score, 4),
            "required_extra_items": extra_items,
            "major_tool_hits": major_hits,
            "minor_helper_hits": minor_hits,
            "unknown_extra_count": unknown_extra_count,
            "main_object_role": main_role or None,
            "role_penalty": round(role_penalty, 4),
        }

    def _build_uut_dual_axis_primitives(
        self,
        *,
        aff_score: float,
        contra_score: float,
        support_items: Sequence[Dict[str, object]],
        conflict_items: Sequence[Dict[str, object]],
        unknown_affordances: Sequence[str],
        g_mech: float,
        drift_score: float,
        extra_tool_result: Dict[str, object],
        swow_path_score: float = 0.0,
        swow_support_score: float = 0.0,
    ) -> Dict[str, object]:
        weighted_total = 0.0
        unsupported_total = 0.0
        for item in support_items:
            weight = max(0.2, float(item.get("weight", 1.0)))
            fit = clip(float(item.get("fit", 0.0)))
            weighted_total += weight
            unsupported_total += weight * max(0.0, 1.0 - fit)

        unknown_weight = 0.7 * len(unknown_affordances)
        weighted_total += unknown_weight
        unsupported_total += unknown_weight
        unsupported_ratio = (
            unsupported_total / weighted_total
            if weighted_total > 0 else 0.45
        )
        conflict_ratio = max(
            clip(contra_score),
            (len(conflict_items) / len(support_items)) if support_items else 0.0,
        )
        extra_tool_violation = clip(float(extra_tool_result.get("score", 0.0)))
        mechanism_completeness = clip(g_mech)
        physical_drift = clip(drift_score)
        semantic_anchor = clip(max(
            float(swow_path_score or 0.0) / 0.12,
            float(swow_support_score or 0.0) / 0.08,
        ))
        cue_support_failure = clip(1.0 - semantic_anchor)
        if clip(aff_score) >= 0.65 and physical_drift <= 0.25 and support_items:
            cue_support_failure = min(cue_support_failure, 0.25)
        cue_drift = clip(0.65 * cue_support_failure + 0.35 * physical_drift)
        appropriateness_gate = clip(
            0.45 * clip(aff_score) +
            0.20 * mechanism_completeness +
            0.20 * (1.0 - extra_tool_violation) +
            0.15 * (1.0 - physical_drift)
        )
        appropriateness_gate = clip(
            appropriateness_gate *
            (1.0 - 0.25 * clip(unsupported_ratio) - 0.20 * clip(conflict_ratio))
        )
        if (
            clip(aff_score) < 0.25 or
            extra_tool_violation >= 0.80 or
            physical_drift >= 0.80 or
            clip(conflict_ratio) >= 0.80
        ):
            appropriateness_gate = min(appropriateness_gate, 0.25)
        idea_hallucination = clip(
            0.50 * clip(unsupported_ratio) +
            0.20 * clip(conflict_ratio) +
            0.20 * extra_tool_violation +
            0.10 * physical_drift
        )

        return {
            "version": "uut_affordance_dual_axis",
            "novelty": None,
            "diversity": None,
            "appropriateness_gate": round(appropriateness_gate, 4),
            "cue_drift": round(cue_drift, 4),
            "semantic_anchor": round(semantic_anchor, 4),
            "cue_support_failure": round(cue_support_failure, 4),
            "supported_affordance_ratio": round(clip(aff_score), 4),
            "unsupported_claim_ratio": round(clip(unsupported_ratio), 4),
            "contradiction_ratio": round(clip(conflict_ratio), 4),
            "extra_tool_violation": round(extra_tool_violation, 4),
            "mechanism_completeness": round(mechanism_completeness, 4),
            "physical_drift": round(physical_drift, 4),
            "idea_hallucination_raw": round(idea_hallucination, 4),
            "support_item_count": len(support_items),
            "conflict_item_count": len(conflict_items),
            "unknown_affordances": list(unknown_affordances),
            "extra_tool_evidence": extra_tool_result,
            "formula": {
                "appropriateness_gate": "clip((0.45*G + 0.20*M + 0.20*(1-E) + 0.15*(1-P)) * unsupported/contradiction discount)",
                "cue_drift": "0.65*(1-semantic_anchor) + 0.35*physical_drift",
                "idea_hallucination_raw": " 0.50*U + 0.20*C + 0.20*E + 0.10*P",
                "task_imagination_raw": "0.55*mean(N*G) + 0.25*D + 0.20*mean(M*G)",
            },
        }

    def _build_uut_profile(self, task_id: str, target_concept: str) -> Tuple[Dict[str, object], Optional[object], Dict[str, object]]:
        manual_profile = self.uut_profiles.get(task_id) or {}
        aliases = manual_profile.get("target_aliases", [])
        word_norms2_match = self._resolve_word_norms2_match(target_concept, aliases=aliases)
        word_norms2_profile = self.word_norms2.derive_affordance_profile(word_norms2_match.concept) if (self.word_norms2.available and word_norms2_match and word_norms2_match.concept) else {"base_properties": {}, "negative_properties": {}}

        merged_base = {k: float(v) for k, v in word_norms2_profile.get("base_properties", {}).items()}
        merged_neg = {k: float(v) for k, v in word_norms2_profile.get("negative_properties", {}).items()}

        for key, value in manual_profile.get("base_properties", {}).items():
            merged_base[key] = max(float(value), merged_base.get(key, 0.0))
        for key, value in manual_profile.get("negative_properties", {}).items():
            merged_neg[key] = max(float(value), merged_neg.get(key, 0.0))

        merged_profile = dict(manual_profile)
        merged_profile["base_properties"] = merged_base
        merged_profile["negative_properties"] = merged_neg
        return merged_profile, word_norms2_match, word_norms2_profile

    def _score_uut(
        self,
        task_id: str,
        target_concept: str,
        idea_text: str,
        *,
        parsed_item: Optional[Dict[str, object]] = None,
        semantic_scorer=None,
        common_answer_bank_trace: Optional[Dict[str, object]] = None,
        common_answer_bank_context: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        profile, word_norms2_match, word_norms2_profile = self._build_uut_profile(task_id, target_concept)
        shared = self.uut_profiles.get("__shared__", {})
        schema_fields = self._extract_uut_schema_fields(parsed_item)
        cues = profile.get("prompt_cues") or self._extract_content_tokens(target_concept)
        excluded = set(self._extract_content_tokens(target_concept))
        expanded_cues = self._expand_swow_cues(cues, top_k=4, min_strength=0.02)
        swow_result = self._swow_support(expanded_cues, idea_text, excluded_tokens=excluded)

        answer_norm = self._normalize_phrase(idea_text)
        answer_tokens = self._extract_content_tokens(idea_text, excluded=excluded)
        swow_path = self._swow_activation_k(expanded_cues, answer_tokens)
        affordance_demands: Dict[str, Dict[str, object]] = {}
        required_map = shared.get("required_affordances", {})
        for affordance, keywords in required_map.items():
            hits = self._match_keywords_in_text(keywords, idea_text)
            if hits:
                affordance_demands[affordance] = {
                    "affordance": affordance,
                    "keywords": hits,
                    "weight": 0.45 + 0.18 * min(3, len(set(hits))),
                    "source": "keyword",
                }

        inferred_affordances = self._infer_answer_affordances(idea_text)
        for affordance, strength in inferred_affordances.get("affordance_scores", {}).items():
            if affordance not in UUT_AFFORDANCE_MAP:
                continue
            if affordance not in affordance_demands:
                affordance_demands[affordance] = {
                    "affordance": affordance,
                    "keywords": [],
                    "weight": float(strength),
                    "source": "concept",
                }
            else:
                affordance_demands[affordance]["weight"] = max(
                    float(affordance_demands[affordance]["weight"]),
                    float(strength),
                )
                if affordance_demands[affordance]["source"] != "concept":
                    affordance_demands[affordance]["source"] = "keyword+concept"

        for affordance in schema_fields.get("declared_affordances", []):
            if affordance not in affordance_demands:
                affordance_demands[affordance] = {
                    "affordance": affordance,
                    "keywords": [affordance.replace("_", " ")],
                    "weight": 0.85,
                    "source": "declared_schema",
                }
            else:
                affordance_demands[affordance]["weight"] = max(
                    float(affordance_demands[affordance]["weight"]),
                    0.85,
                )
                if affordance_demands[affordance]["source"] not in {"keyword+concept", "declared_schema"}:
                    affordance_demands[affordance]["source"] = f"{affordance_demands[affordance]['source']}+declared"

        modifier_hits = []
        enabled_affordances = set()
        for modifier_name, cfg in shared.get("modifier_enablers", {}).items():
            keywords = cfg.get("keywords", [])
            hits = self._match_keywords_in_text(keywords, idea_text)
            if hits:
                modifier_hits.append({
                    "modifier": modifier_name,
                    "keywords": hits,
                })
                enabled_affordances.update(cfg.get("enables", []))

        mechanism_hits = []
        for keyword in shared.get("mechanism_keywords", []):
            if self._match_keywords_in_text([keyword], idea_text):
                mechanism_hits.append(keyword.lower())
        if not mechanism_hits and len(answer_tokens) >= 5:
            mechanism_hits.append("implicit_mechanism")

        base_props = {k: float(v) for k, v in profile.get("base_properties", {}).items()}
        negative_props = {k: float(v) for k, v in profile.get("negative_properties", {}).items()}

        support_items = []
        conflict_items = []
        total_weight = 0.0
        support_weighted_sum = 0.0
        contra_weighted_sum = 0.0

        for item in affordance_demands.values():
            aff = item["affordance"]
            weight = max(0.2, float(item.get("weight", 1.0)))
            base = base_props.get(aff, 0.0)
            enabled = 1.0 if aff in enabled_affordances else 0.0
            fit = max(base, 0.7 * enabled)
            negative = negative_props.get(aff, 0.0)
            conflict = clip(max(0.0, negative - 0.35 * enabled))
            support_items.append({
                "affordance": aff,
                "fit": round(fit, 4),
                "base": round(base, 4),
                "enabled": round(enabled, 4),
                "weight": weight,
                "keywords": item["keywords"],
                "source": item.get("source"),
            })
            total_weight += weight
            support_weighted_sum += weight * fit
            contra_weighted_sum += weight * conflict
            if conflict >= 0.12:
                conflict_items.append({
                    "affordance": aff,
                    "negative_strength": round(negative, 4),
                })

        if total_weight > 0:
            aff_score = support_weighted_sum / total_weight
            contra_score = contra_weighted_sum / total_weight
        else:
            aff_score = 0.0
            contra_score = 0.0

        coverage_score = (
            sum(1 for item in support_items if item["fit"] >= 0.35) / len(support_items)
            if support_items else 0.0
        )
        mech_score = min(1.0, 0.55 * len(set(mechanism_hits)) + 0.20 * min(1.0, len(answer_tokens) / 7.0))
        modifier_score = min(1.0, len(modifier_hits) / 2.0) if modifier_hits else 0.0
        legacy_feature_fit = clip(
            0.72 * aff_score +
            0.16 * coverage_score +
            0.12 * modifier_score -
            0.20 * contra_score
        )
        embedding_fit = self._embedding_property_fit(
            "UUT",
            task_id,
            target_concept,
            idea_text,
            parsed_item=parsed_item,
            semantic_scorer=semantic_scorer,
        )
        g_feat_v2 = max(legacy_feature_fit, float(embedding_fit.get("score", 0.0)))
        mech_result = self._mechanism_score(
            idea_text,
            target_concept,
            "UUT",
            parsed_item=parsed_item,
            semantic_scorer=semantic_scorer,
        )
        g_mech = max(float(mech_result.get("score", 0.0)), 0.35 * mech_score)
        drift_result = self._impossibility_drift_score(
            task_id,
            "UUT",
            idea_text,
            g_feat_v2=g_feat_v2,
            base_drift=clip(contra_score),
        )
        drift_score = float(drift_result.get("score", 0.0))
        extra_tool_result = self._score_uut_extra_tool_violation(idea_text, schema_fields)
        dual_axis_primitives = self._build_uut_dual_axis_primitives(
            aff_score=aff_score,
            contra_score=contra_score,
            support_items=support_items,
            conflict_items=conflict_items,
            unknown_affordances=schema_fields.get("unknown_affordances", []),
            g_mech=g_mech,
            drift_score=drift_score,
            extra_tool_result=extra_tool_result,
            swow_path_score=float(swow_path.get("score", 0.0)),
            swow_support_score=float(swow_result.get("score", 0.0)),
        )
        shared_support_boundary = shared.get("support_boundary") if isinstance(shared, dict) else None
        for metadata_key in ("object_category", "difficulty", "support_boundary"):
            metadata_value = profile.get(metadata_key) or (shared_support_boundary if metadata_key == "support_boundary" else None)
            if metadata_value is not None:
                dual_axis_primitives[metadata_key] = metadata_value
        anti_cliche = self._anti_cliche_signal(
            task_id=task_id,
            task_type="UUT",
            idea_text=idea_text,
            parsed_item=parsed_item,
            semantic_scorer=semantic_scorer,
            common_answer_bank_trace=common_answer_bank_trace,
            common_answer_bank_context=common_answer_bank_context,
        )
        groundedness = self._combine_v5_groundedness(
            "UUT",
            g_swow_path=float(swow_path.get("score", 0.0)),
            g_feat_v2=g_feat_v2,
            g_mech=g_mech,
            anti_cliche=float(anti_cliche.get("score", 0.0)),
            drift_score=drift_score,
        )

        supported_tokens = {
            token_info.get("token")
            for token_info in swow_result.get("token_evidence", [])
            if float(token_info.get("score", 0.0)) >= 0.004
        }
        for item in support_items:
            supported_tokens.update(self._extract_content_tokens(" ".join(item.get("keywords", []))))
        specificity = float(word_norms2_match.confidence) if word_norms2_match and word_norms2_match.concept else (
            min(1.0, len(inferred_affordances.get("concept_evidence", [])) / 3.0)
        )
        agreement = 1.0 if (coverage_score >= 0.5 and float(swow_result.get("score", 0.0)) >= 0.08) else (
            0.6 if support_items else 0.0
        )
        ambiguity = self._estimate_ambiguity(answer_tokens, supported_tokens)
        confidence, confidence_components = self._confidence_from_evidence(
            evidence_strength=max(g_feat_v2, float(swow_path.get("score", 0.0)), g_mech),
            coverage=max(coverage_score, modifier_score),
            agreement=agreement,
            specificity=max(specificity, g_mech),
            ambiguity=ambiguity,
        )
        return {
            "groundedness_score": groundedness,
            "groundedness_confidence": confidence,
            "formula": "uut_groundedness_v5p0_signal",
            "dual_axis_primitives": dual_axis_primitives,
            "subscores": {
                "g_swow_path": swow_path.get("score", 0.0),
                "affordance_fit_legacy": round(aff_score, 4),
                "legacy_feature_fit": round(legacy_feature_fit, 4),
                "embedding_fit": embedding_fit.get("score", 0.0),
                "g_feat_v2": round(g_feat_v2, 4),
                "coverage_fit": round(coverage_score, 4),
                "legacy_swow_support": swow_result.get("score", 0.0),
                "mechanism_fit_legacy": round(mech_score, 4),
                "mechanism_score": round(g_mech, 4),
                "modifier_fit": round(modifier_score, 4),
                "anti_cliche": anti_cliche.get("score", 0.0),
                "drift_score": round(drift_score, 4),
                "contra_score": round(contra_score, 4),
            },
            "evidence": {
                "target": target_concept,
                "v5_weights": self._get_v5_task_weights("UUT"),
                "swow_path": swow_path,
                "embedding_property_fit": embedding_fit,
                "mechanism": mech_result,
                "anti_cliche": anti_cliche,
                "impossibility_drift": drift_result,
                "word_norms2_target_match": {
                    "concept": word_norms2_match.concept,
                    "matched_alias": word_norms2_match.matched_alias,
                    "confidence": word_norms2_match.confidence,
                } if word_norms2_match else None,
                "word_norms2_profile": word_norms2_profile,
                "profile_loaded": bool(profile),
                "schema_fields": schema_fields,
                "required_affordances": support_items,
                "answer_affordance_inference": inferred_affordances,
                "modifier_hits": modifier_hits,
                "extra_tool_violation": extra_tool_result,
                "mechanism_hits": mechanism_hits,
                "conflicts": conflict_items,
                "confidence_components": confidence_components,
                "swow": swow_result,
                "answer_normalized": answer_norm,
            },
        }

    
    
    

    def _score_jst(
        self,
        task_id: str,
        target_concept: str,
        idea_text: str,
        *,
        parsed_item: Optional[Dict[str, object]] = None,
        semantic_scorer=None,
        common_answer_bank_trace: Optional[Dict[str, object]] = None,
        common_answer_bank_context: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        template = self.jst_templates.get(task_id) or {}
        anchors = template.get("anchors") or self._extract_content_tokens(target_concept)
        excluded = set(self._extract_content_tokens(target_concept))
        expanded_anchors = self._expand_swow_cues(anchors, top_k=4, min_strength=0.018)
        swow_result = self._swow_support(expanded_anchors, idea_text, excluded_tokens=excluded)

        tokens = self._extract_content_tokens(idea_text, excluded=excluded)
        normalized_idea = self._normalize_phrase(idea_text)
        idea_token_set = set(tokens)
        swow_path = self._swow_activation_k(expanded_anchors, tokens)
        channel_scores = []
        matched_channel_tokens = set()
        channel_evidence = []

        for channel_name, keywords in template.get("impact_channels", {}).items():
            channel_result = self._soft_channel_score(keywords, idea_text, excluded_tokens=excluded)
            for hit in channel_result["hits"]:
                matched_channel_tokens.update(self._extract_content_tokens(hit))
            for item in channel_result["swow_support"].get("token_evidence", []):
                if float(item.get("score", 0.0)) >= 0.004:
                    matched_channel_tokens.add(item.get("token"))
            ch_score = float(channel_result["score"])
            channel_scores.append(ch_score)
            channel_evidence.append({
                "channel": channel_name,
                "hits": channel_result["hits"],
                "score": round(ch_score, 4),
                "keyword_score": channel_result["keyword_score"],
                "semantic_score": channel_result["semantic_score"],
                "expanded_keywords": channel_result["expanded_keywords"],
            })

        lexical_anchor_hits = []
        for anchor in anchors:
            anchor_norm = self._normalize_phrase(anchor)
            anchor_tokens = set(self._extract_content_tokens(anchor))
            if anchor_norm and anchor_norm in normalized_idea:
                lexical_anchor_hits.append(anchor)
                continue
            if anchor_tokens and anchor_tokens.issubset(idea_token_set):
                lexical_anchor_hits.append(anchor)

        lexical_anchor_score = min(1.0, len(set(lexical_anchor_hits)) / max(2.0, len(set(anchors)) * 0.5)) if lexical_anchor_hits else 0.0
        scenario_score = max(float(swow_result.get("score", 0.0)), 0.32 * lexical_anchor_score)
        if channel_scores:
            ranked = sorted(channel_scores, reverse=True)
            if len(ranked) >= 2:
                channel_score = 0.65 * ranked[0] + 0.35 * ranked[1]
            else:
                channel_score = ranked[0]
        else:
            channel_score = 0.0

        supported_tokens = set()
        epsilon = 0.003
        for item in swow_result.get("token_evidence", []):
            if float(item.get("score", 0.0)) >= epsilon:
                supported_tokens.add(item.get("token"))
        supported_tokens.update(matched_channel_tokens)
        for anchor in lexical_anchor_hits:
            supported_tokens.update(self._extract_content_tokens(anchor))

        if tokens:
            focus = len([token for token in tokens if token in supported_tokens]) / len(tokens)
            drift = len([token for token in tokens if token not in supported_tokens]) / len(tokens)
        else:
            focus = 0.0
            drift = 0.0

        clause_score = 0.0
        if len(tokens) >= 4:
            clause_score = 1.0
        elif len(tokens) >= 2:
            clause_score = 0.55
        elif len(tokens) == 1:
            clause_score = 0.25

        legacy_feature_fit = clip(0.45 * channel_score + 0.35 * scenario_score + 0.20 * focus)
        embedding_fit = self._embedding_property_fit(
            "JST",
            task_id,
            target_concept,
            idea_text,
            parsed_item=parsed_item,
            semantic_scorer=semantic_scorer,
        )
        g_feat_v2 = max(legacy_feature_fit, float(embedding_fit.get("score", 0.0)))
        mech_result = self._mechanism_score(
            idea_text,
            target_concept,
            "JST",
            parsed_item=parsed_item,
            semantic_scorer=semantic_scorer,
        )
        g_mech = max(float(mech_result.get("score", 0.0)), 0.35 * clause_score)
        anti_cliche = self._anti_cliche_signal(
            task_id=task_id,
            task_type="JST",
            idea_text=idea_text,
            parsed_item=parsed_item,
            semantic_scorer=semantic_scorer,
            common_answer_bank_trace=common_answer_bank_trace,
            common_answer_bank_context=common_answer_bank_context,
        )
        drift_result = self._impossibility_drift_score(
            task_id,
            "JST",
            idea_text,
            g_feat_v2=g_feat_v2,
            base_drift=drift,
        )
        drift = float(drift_result.get("score", 0.0))
        groundedness = self._combine_v5_groundedness(
            "JST",
            g_swow_path=float(swow_path.get("score", 0.0)),
            g_feat_v2=g_feat_v2,
            g_mech=g_mech,
            anti_cliche=float(anti_cliche.get("score", 0.0)),
            drift_score=drift,
        )

        anchor_matches = []
        if self.word_norms2.available:
            for anchor in anchors:
                match = self.word_norms2.match_concept(anchor)
                if match and match.concept:
                    anchor_matches.append({
                        "anchor": anchor,
                        "concept": match.concept,
                        "confidence": match.confidence,
                    })

        agreement = 1.0 if (scenario_score >= 0.12 and channel_score >= 0.18) else (
            0.65 if (scenario_score >= 0.08 or channel_score >= 0.15) else 0.0
        )
        specificity = min(1.0, len(lexical_anchor_hits) / 2.0) if lexical_anchor_hits else min(1.0, scenario_score / 0.25)
        ambiguity = self._estimate_ambiguity(tokens, supported_tokens)
        confidence, confidence_components = self._confidence_from_evidence(
            evidence_strength=max(float(swow_path.get("score", 0.0)), g_feat_v2, g_mech),
            coverage=max(focus, min(1.0, len([score for score in channel_scores if score >= 0.18]) / 2.0)),
            agreement=agreement,
            specificity=max(specificity, g_mech),
            ambiguity=ambiguity,
            cap=0.75,
        )
        return {
            "groundedness_score": groundedness,
            "groundedness_confidence": confidence,
            "formula": "jst_groundedness_v5p0_signal",
            "subscores": {
                "g_swow_path": swow_path.get("score", 0.0),
                "scenario_support_legacy": round(scenario_score, 4),
                "lexical_anchor": round(lexical_anchor_score, 4),
                "channel_fit_legacy": round(channel_score, 4),
                "legacy_feature_fit": round(legacy_feature_fit, 4),
                "embedding_fit": embedding_fit.get("score", 0.0),
                "g_feat_v2": round(g_feat_v2, 4),
                "mechanism_score": round(g_mech, 4),
                "anti_cliche": anti_cliche.get("score", 0.0),
                "focus": round(focus, 4),
                "clause_fit": round(clause_score, 4),
                "drift_score": round(drift, 4),
            },
            "evidence": {
                "target": target_concept,
                "v5_weights": self._get_v5_task_weights("JST"),
                "swow_path": swow_path,
                "embedding_property_fit": embedding_fit,
                "mechanism": mech_result,
                "anti_cliche": anti_cliche,
                "impossibility_drift": drift_result,
                "anchors": anchors,
                "expanded_anchors": expanded_anchors[:16],
                "lexical_anchor_hits": lexical_anchor_hits,
                "word_norms2_anchor_matches": anchor_matches,
                "channels": channel_evidence,
                "supported_tokens": sorted(supported_tokens),
                "tokens": tokens,
                "confidence_components": confidence_components,
                "swow": swow_result,
            },
        }
