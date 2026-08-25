
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Set


DATA_DIR = Path(__file__).resolve().parent / "data"
TAXONOMY_V2_PATH = DATA_DIR / "taxonomy_v2.json"


def load_taxonomy_v2(path: Path | None = None) -> Dict[str, object]:
    taxonomy_path = Path(path) if path else TAXONOMY_V2_PATH
    with taxonomy_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_taxonomy_v2(payload)
    return payload


def taxonomy_subtype_ids(payload: Dict[str, object], axis: str) -> Set[str]:
    key = f"{axis}_subtypes"
    values = payload.get(key)
    if not isinstance(values, dict):
        raise ValueError(f"taxonomy_v2 missing object field {key}")
    return {str(item) for item in values.keys()}


def validate_taxonomy_v2(payload: Dict[str, object]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("taxonomy_v2 must be a JSON object")
    for key in ("version", "imagination_subtypes", "hallucination_subtypes"):
        if key not in payload:
            raise ValueError(f"taxonomy_v2 missing {key}")
    for axis in ("imagination", "hallucination"):
        subtypes = taxonomy_subtype_ids(payload, axis)
        if not subtypes:
            raise ValueError(f"taxonomy_v2 has no {axis} subtypes")
        _validate_subtype_definitions(payload[f"{axis}_subtypes"], axis)


def _validate_subtype_definitions(subtypes: object, axis: str) -> None:
    if not isinstance(subtypes, Mapping):
        raise ValueError(f"taxonomy_v2 {axis}_subtypes must be an object")
    for subtype_id, spec in subtypes.items():
        context = f"taxonomy_v2 {axis} subtype {subtype_id}"
        if not isinstance(spec, Mapping):
            raise ValueError(f"{context} must be an object")
        _require_nonempty_string(spec, "label", context)
        _require_nonempty_string(spec, "definition", context)
        facets = spec.get("facets")
        if not isinstance(facets, Mapping) or not facets:
            raise ValueError(f"{context} requires non-empty facets")
        for facet_id, facet in facets.items():
            facet_context = f"{context} facet {facet_id}"
            if not isinstance(facet, Mapping):
                raise ValueError(f"{facet_context} must be an object")
            _require_nonempty_string(facet, "label", facet_context)
            _require_nonempty_string(facet, "definition", facet_context)
            signals = facet.get("white_box_signals")
            if not isinstance(signals, list) or not signals:
                raise ValueError(f"{facet_context} requires non-empty white_box_signals")
            if any(not isinstance(signal, str) or not signal.strip() for signal in signals):
                raise ValueError(f"{facet_context} has invalid white_box_signals")


def _require_nonempty_string(spec: Mapping[str, object], key: str, context: str) -> None:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} missing non-empty {key}")


def ensure_subtype_ids(payload: Dict[str, object], axis: str, subtype_ids: Iterable[str]) -> None:
    known = taxonomy_subtype_ids(payload, axis)
    missing = sorted(str(subtype_id) for subtype_id in subtype_ids if str(subtype_id) not in known)
    if missing:
        raise ValueError(f"Unknown {axis} subtype ids: {missing}")
