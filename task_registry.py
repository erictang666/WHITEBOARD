
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from taxonomy_registry import ensure_subtype_ids, load_taxonomy_v2


DATA_DIR = Path(__file__).resolve().parent / "data"
TASK_REGISTRY_V2_PATH = DATA_DIR / "task_registry_v2.json"


def load_task_registry_v2(path: Path | None = None, taxonomy: Dict[str, object] | None = None) -> Dict[str, object]:
    registry_path = Path(path) if path else TASK_REGISTRY_V2_PATH
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_task_registry_v2(payload, taxonomy=taxonomy)
    return payload


def validate_task_registry_v2(payload: Dict[str, object], taxonomy: Dict[str, object] | None = None) -> None:
    if not isinstance(payload, dict):
        raise ValueError("task_registry_v2 must be a JSON object")
    if "version" not in payload:
        raise ValueError("task_registry_v2 missing version")
    families = payload.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("task_registry_v2 requires non-empty families list")

    taxonomy_payload = taxonomy or load_taxonomy_v2()
    family_ids = set()
    for family in families:
        if not isinstance(family, dict):
            raise ValueError("task_registry_v2 family entries must be objects")
        family_id = str(family.get("family") or "")
        if not family_id:
            raise ValueError("task_registry_v2 family missing family id")
        if family_id in family_ids:
            raise ValueError(f"duplicate family id {family_id}")
        family_ids.add(family_id)
        components = family.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError(f"family {family_id} requires components")
        ensure_subtype_ids(taxonomy_payload, "imagination", family.get("imagination_subtypes") or [])
        ensure_subtype_ids(taxonomy_payload, "hallucination", family.get("hallucination_subtypes") or [])
