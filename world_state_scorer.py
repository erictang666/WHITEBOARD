
from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Sequence


WORLD_STATE_SCORER_VERSION = "world_state_scorer"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for", "with",
    "in", "on", "at", "by", "is", "are", "was", "were", "be", "being",
    "been", "would", "could", "should", "might", "may", "can", "will",
    "all", "every", "some", "any", "this", "that", "these", "those",
    "from", "into", "about", "after", "before", "when", "while",
}

PROTECTED_VARIABLE_TERMS = {
    "ordinary_physics": [
        "teleport", "portal", "fly to the moon", "stop gravity", "black hole",
        "physical law changes", "physics changes",
    ],
    "time_direction": [
        "predict the future", "predict tomorrow", "tomorrow's", "next week",
        "prophecy", "precognition", "time travel", "travel back", "travel forward",
    ],
    "human_minds": [
        "read minds", "read thoughts", "control minds", "control emotions",
        "reveal thoughts", "steal memories",
    ],
    "entity_identity": [
        "become conscious", "becomes conscious", "become sentient",
        "becomes sentient", "ghost", "zombie", "dragon", "alien empire",
    ],
    "finite_energy": [
        "infinite energy", "unlimited energy", "unlimited electricity",
        "perpetual motion", "free energy", "create infinite", "generate unlimited",
    ],
    "ordinary_institutions": [
        "grant wishes", "heal diseases", "cure poison", "create money",
        "print money", "accuse criminals", "arrest thieves",
    ],
}

EXTRA_MIRACLE_TERMS = sorted({
    "magic", "magical", "spell", "wizard", "curse", "miracle", "supernatural",
    "psychic", "teleport", "portal", "time travel", "infinite energy",
    "unlimited energy", "grant wishes", "read minds", "predict the future",
    "prophecy", "become conscious", "becomes conscious", "sentient",
})

FORBIDDEN_GENERIC_TOKENS = {
    "create", "creates", "created", "make", "makes", "made", "become",
    "becomes", "became", "turn", "turns", "turned", "cause", "causes",
    "all", "every", "thing", "things", "object", "objects",
}


def clip01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _normalize_text(value) -> str:
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value) -> Sequence[str]:
    return [
        token for token in _normalize_text(value).split()
        if len(token) > 2 and token not in STOPWORDS
    ]


def _flatten_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(f"{key} { _flatten_text(item) }" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def _phrase_or_token_hit(text_norm: str, phrase: str, *, min_overlap: float = 0.55) -> bool:
    phrase_norm = _normalize_text(phrase)
    if not phrase_norm:
        return False
    if f" {phrase_norm} " in f" {text_norm} ":
        return True
    phrase_tokens = [token for token in _tokens(phrase_norm) if token]
    if not phrase_tokens:
        return False
    text_tokens = set(_tokens(text_norm))
    overlap = sum(1 for token in phrase_tokens if token in text_tokens)
    return (overlap / max(1, len(phrase_tokens))) >= min_overlap and overlap >= min(2, len(phrase_tokens))


def _keyword_support(text_norm: str, keywords: Iterable[str], *, denominator: float = 4.0) -> float:
    usable = []
    seen = set()
    for keyword in keywords or []:
        for token in _tokens(keyword):
            if token and token not in seen:
                seen.add(token)
                usable.append(token)
    if not usable:
        return 0.0
    hits = [token for token in usable if token in set(_tokens(text_norm))]
    return clip01(len(hits) / min(max(denominator, 1.0), max(1, len(usable))))


def _forbidden_update_hit(text_norm: str, phrase: str) -> bool:
    phrase_norm = _normalize_text(phrase)
    if not phrase_norm:
        return False
    if f" {phrase_norm} " in f" {text_norm} ":
        return True
    phrase_tokens = [token for token in _tokens(phrase_norm) if token]
    if not phrase_tokens:
        return False
    text_tokens = set(_tokens(text_norm))
    distinctive = [
        token for token in phrase_tokens
        if token not in FORBIDDEN_GENERIC_TOKENS
    ] or phrase_tokens
    distinctive_hits = sum(1 for token in distinctive if token in text_tokens)
    if len(distinctive) <= 3:
        return distinctive_hits == len(distinctive)
    return distinctive_hits >= 3 and (distinctive_hits / len(distinctive)) >= 0.80


def _explicit_false(value) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return _normalize_text(value) in {"false", "no", "not respected", "violated", "violate"}
    return False


class WorldStateScorer:
    """Checks a model's world-state update against one closed counterfactual card."""

    def score_item(
        self,
        card: Mapping[str, object] | None,
        parsed_item: Mapping[str, object] | None,
        *,
        display_text: str = "",
        causal_bridge: str = "",
    ) -> Dict[str, object]:
        card = card or {}
        parsed_item = parsed_item or {}
        update = parsed_item.get("world_state_update")
        update_text = _flatten_text(update)
        combined = " ".join([
            display_text or "",
            causal_bridge or "",
            " ".join(str(item) for item in parsed_item.get("anchor_terms") or []),
            update_text,
        ])
        combined_norm = _normalize_text(combined)

        intervention = card.get("intervention") or {}
        intervention_keywords = []
        if isinstance(intervention, Mapping):
            intervention_keywords.extend(intervention.get("keywords") or [])
            intervention_keywords.append(intervention.get("new_value") or "")
            intervention_keywords.append(intervention.get("variable") or "")
        intervention_keywords.append(card.get("allowed_impossibility") or "")
        intervention_used = _keyword_support(combined_norm, intervention_keywords, denominator=3.0)

        edge_scores = []
        for edge in card.get("causal_edges") or []:
            if isinstance(edge, Mapping):
                edge_keywords = list(edge.get("keywords") or [])
                edge_keywords.extend([edge.get("source") or "", edge.get("target") or ""])
            else:
                edge_keywords = [str(edge)]
            edge_scores.append(_keyword_support(combined_norm, edge_keywords, denominator=3.0))
        causal_edge_support = max(edge_scores) if edge_scores else intervention_used

        protected_variables = card.get("protected_variables") or []
        protected_hits = []
        protected_score = 0.0
        if _explicit_false(parsed_item.get("protected_variables_respected")):
            protected_hits.append({"variable": "declared", "term": "protected_variables_respected=false"})
            protected_score = 1.0
        for variable in protected_variables:
            variable_id = str(variable.get("variable") if isinstance(variable, Mapping) else variable)
            terms = []
            if isinstance(variable, Mapping):
                terms.extend(variable.get("forbidden_terms") or [])
            terms.extend(PROTECTED_VARIABLE_TERMS.get(variable_id, []))
            for term in terms:
                if _phrase_or_token_hit(combined_norm, term, min_overlap=0.50):
                    protected_hits.append({"variable": variable_id, "term": term})
        if protected_hits:
            protected_score = max(protected_score, clip01(len(protected_hits) / 2.0))

        forbidden_hits = []
        for phrase in card.get("forbidden_updates") or []:
            if _forbidden_update_hit(combined_norm, str(phrase)):
                forbidden_hits.append(str(phrase))
        forbidden_update_rate = clip01(len(forbidden_hits) / 2.0)

        extra_miracle_hits = [
            term for term in EXTRA_MIRACLE_TERMS
            if _phrase_or_token_hit(combined_norm, term, min_overlap=0.50)
        ]
        extra_miracle_rate = clip01(len(extra_miracle_hits) / 2.0)

        formal_hallucination_raw = clip01(
            0.15 * (1.0 - intervention_used) +
            0.15 * (1.0 - causal_edge_support) +
            0.30 * protected_score +
            0.25 * forbidden_update_rate +
            0.15 * extra_miracle_rate
        )
        if protected_score >= 0.75 or forbidden_update_rate >= 0.75 or extra_miracle_rate >= 0.75:
            formal_hallucination_raw = max(formal_hallucination_raw, 0.70)

        world_consistency = clip01(
            0.34 * intervention_used +
            0.34 * causal_edge_support +
            0.16 * (1.0 - protected_score) +
            0.10 * (1.0 - forbidden_update_rate) +
            0.06 * (1.0 - extra_miracle_rate)
        )
        if protected_score >= 0.75 or forbidden_update_rate >= 0.75 or extra_miracle_rate >= 0.75:
            world_consistency = min(world_consistency, 0.35)

        return {
            "version": WORLD_STATE_SCORER_VERSION,
            "world_state_update_present": bool(update_text.strip()),
            "intervention_used": round(intervention_used, 4),
            "causal_edge_support": round(causal_edge_support, 4),
            "protected_variable_violation": round(protected_score, 4),
            "forbidden_update_rate": round(forbidden_update_rate, 4),
            "extra_miracle_rate": round(extra_miracle_rate, 4),
            "world_consistency": round(world_consistency, 4),
            "formal_hallucination_raw": round(formal_hallucination_raw, 4),
            "evidence": {
                "protected_hits": protected_hits,
                "forbidden_update_hits": forbidden_hits,
                "extra_miracle_hits": extra_miracle_hits,
                "intervention_keywords": [str(item) for item in intervention_keywords if item],
            },
        }


__all__ = ["WORLD_STATE_SCORER_VERSION", "WorldStateScorer", "clip01"]
