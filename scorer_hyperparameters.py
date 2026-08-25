
from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


SCORER_HYPERPARAMETERS_PATH = Path(__file__).resolve().parent / "data" / "scorer_hyperparameters.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=1)
def _load_default_hyperparameters() -> dict[str, Any]:
    return _read_json(SCORER_HYPERPARAMETERS_PATH)


def load_scorer_hyperparameters(path: str | Path | None = None) -> dict[str, Any]:
    """Load scorer hyperparameters, returning a defensive deep copy."""

    if path is None:
        return copy.deepcopy(_load_default_hyperparameters())
    return copy.deepcopy(_read_json(Path(path)))


def get_scorer_hyperparameter(
    section: str,
    key: str | Iterable[str] | None = None,
    *,
    default: Any = None,
    path: str | Path | None = None,
) -> Any:
    """Fetch one hyperparameter section/key with a copied fallback.

    ``key`` may be a string for one-level lookup or an iterable for nested
    lookup.  Missing sections or wrong container types return ``default``.
    """

    payload = load_scorer_hyperparameters(path)
    node: Any = payload.get(section)
    if key is None:
        return copy.deepcopy(node if node is not None else default)
    keys = [key] if isinstance(key, str) else list(key)
    for part in keys:
        if not isinstance(node, dict) or part not in node:
            return copy.deepcopy(default)
        node = node[part]
    return copy.deepcopy(node)
