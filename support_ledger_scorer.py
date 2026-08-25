
from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SUPPORT_LEDGER_VERSION = "support_ledger"
PERMITTED_SCENE_CLAIM_TYPES = {
    "permitted_scene_invention",
    "permitted scene invention",
    "scene_invention",
    "scene invention",
    "sensory_detail",
    "sensory detail",
    "minor_action",
    "minor action",
    "dialogue",
    "small_action",
    "small action",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for", "with",
    "in", "on", "at", "by", "is", "are", "was", "were", "be", "being",
    "been", "would", "could", "should", "might", "may", "can", "will",
    "all", "every", "some", "any", "this", "that", "these", "those",
    "one", "two", "three", "before", "after", "when", "then", "there",
    "their", "them", "they", "she", "he", "her", "his", "into", "from",
    "as", "if", "it", "its", "only", "not", "no",
}


def clip01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _clean_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_text(text: str) -> str:
    text = (text or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> List[str]:
    return [
        token for token in _normalize_text(text).split()
        if token and token not in STOPWORDS and len(token) > 1
    ]


def _as_list(value) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _dedupe_strings(values: Iterable[object]) -> List[str]:
    seen = set()
    results = []
    for value in values:
        cleaned = _clean_string(value)
        if not cleaned:
            continue
        key = _normalize_text(cleaned)
        if key in seen:
            continue
        seen.add(key)
        results.append(cleaned)
    return results


def _phrase_hit(text_norm: str, phrase: str) -> bool:
    phrase_norm = _normalize_text(str(phrase))
    if not phrase_norm:
        return False
    if f" {phrase_norm} " in f" {text_norm} ":
        return True
    phrase_tokens = phrase_norm.split()
    token_set = set(text_norm.split())
    return bool(phrase_tokens) and set(phrase_tokens).issubset(token_set)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    left = set(_tokens(text_a))
    right = set(_tokens(text_b))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _keyword_support(claim_text: str, reference: Mapping[str, object]) -> float:
    ref_text = _clean_string(reference.get("text"))
    claim_norm = _normalize_text(claim_text)
    if not claim_norm:
        return 0.0
    if ref_text and _phrase_hit(claim_norm, ref_text):
        return 1.0
    keyword_hits = 0
    keywords = list(reference.get("keywords") or []) + list(reference.get("forbidden_terms") or [])
    for keyword in keywords:
        if _phrase_hit(claim_norm, str(keyword)):
            keyword_hits += 1
    keyword_score = clip01(keyword_hits / max(1, min(2, len(keywords))))
    overlap_score = _jaccard_similarity(claim_text, ref_text)
    return max(keyword_score, overlap_score)


def _looks_like_negated_constraint_claim(text: str) -> bool:
    text_norm = _normalize_text(text)
    return any(
        marker in f" {text_norm} "
        for marker in (
            " no magic ",
            " no prophecy ",
            " no teleport ",
            " cannot ",
            " does not ",
            " do not ",
            " must not ",
            " remains ",
        )
    )


def _alias_variants(text: str) -> List[str]:
    base = _normalize_text(text)
    if not base:
        return []
    variants = {base}
    if base.endswith("s") and len(base) > 3:
        variants.add(base[:-1])
    else:
        variants.add(base + "s")
    if base.endswith("ies") and len(base) > 4:
        variants.add(base[:-3] + "y")
    if base.endswith("y") and len(base) > 2:
        variants.add(base[:-1] + "ies")
    return [item for item in variants if item]


class SupportLedgerScorer:
    """Scores a parsed response's claim ledger against a closed evidence card."""

    def __init__(self, *, patterns: Optional[Mapping[str, object]] = None):
        self.patterns = dict(patterns or {})

    def score_response(
        self,
        task: Mapping[str, object],
        parsed_response: Mapping[str, object],
        *,
        constraint_profile: Optional[Mapping[str, object]] = None,
        full_text: Optional[str] = None,
    ) -> Dict[str, object]:
        profile = dict(constraint_profile or {})
        claims = self._normalize_claims(parsed_response.get("claims") or parsed_response.get("claim_ledger") or [])
        legacy_fallback = False
        if not claims:
            claims = self._claims_from_beats(parsed_response.get("beats") or [])
            legacy_fallback = True

        references = self._build_reference_index(task)
        required_fact_ids = {
            str(item) for item in (
                profile.get("required_fact_ids")
                or task.get("required_facts")
                or []
            )
            if item
        }
        require_evidence_ids = bool(profile.get("require_evidence_ids") or profile.get("require_claim_ledger"))
        if legacy_fallback:
            require_evidence_ids = False

        full_text = full_text if full_text is not None else self._response_text(parsed_response)
        full_norm = _normalize_text(full_text)
        contradiction_terms = self._contradiction_terms(task, profile)
        forbidden_motifs = [str(item) for item in profile.get("forbidden_motifs") or []]

        supported_claims = 0
        unsupported_spans = 0
        citation_issues = 0
        unknown_evidence_issues = 0
        contradicted_claims = 0
        claims_without_evidence = 0
        covered_required_fact_ids = set()
        records = []

        for claim in claims:
            record = self._score_claim(
                claim,
                references,
                contradiction_terms,
                require_evidence_ids=require_evidence_ids,
            )
            records.append(record)
            supported_claims += int(record["supported"])
            unsupported_spans += int(record["unsupported"])
            citation_issues += int(record["citation_mismatch"])
            unknown_evidence_issues += len(record["unknown_evidence_ids"])
            contradicted_claims += int(record["contradicted"])
            claims_without_evidence += int(record["missing_evidence"])
            for ref_id in record["support_ids"]:
                if ref_id in required_fact_ids and record["supported"]:
                    covered_required_fact_ids.add(ref_id)

        checked_claims = max(1, len(claims))
        entity_drift = self._entity_drift(task, parsed_response)
        entity_persistence_failure = self._entity_persistence_failure(task, parsed_response, full_norm, entity_drift)
        forbidden_motif_hits = [
            motif for motif in forbidden_motifs
            if _phrase_hit(full_norm, motif)
        ]
        forbidden_motif_rate = clip01(len(forbidden_motif_hits) / max(1, len(forbidden_motifs)))
        hard_no_drift_violation = 1.0 if (
            profile.get("hard_no_drift") and (
                entity_drift["entity_drift_rate"] > 0.0
                or entity_persistence_failure > 0.0
                or forbidden_motif_rate > 0.0
            )
        ) else 0.0

        precision = supported_claims / checked_claims if claims else (0.65 if legacy_fallback else 0.0)
        recall = (
            len(covered_required_fact_ids) / len(required_fact_ids)
            if required_fact_ids else (1.0 if supported_claims else 0.0)
        )
        unsupported_rate = unsupported_spans / checked_claims
        citation_mismatch_rate = citation_issues / checked_claims
        contradicted_rate = contradicted_claims / checked_claims
        missing_evidence_rate = claims_without_evidence / checked_claims
        unknown_evidence_rate = unknown_evidence_issues / max(1, sum(len(item["support_ids"]) for item in records))

        return {
            "version": SUPPORT_LEDGER_VERSION,
            "legacy_fallback": bool(legacy_fallback),
            "checked_claims": len(claims),
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_spans,
            "contradicted_claims": contradicted_claims,
            "claim_support_precision": round(clip01(precision), 4),
            "claim_support_recall": round(clip01(recall), 4),
            "unsupported_span_rate": round(clip01(unsupported_rate), 4),
            "citation_mismatch_rate": round(clip01(citation_mismatch_rate), 4),
            "contradicted_claim_rate": round(clip01(contradicted_rate), 4),
            "claim_without_evidence_rate": round(clip01(missing_evidence_rate), 4),
            "unknown_evidence_rate": round(clip01(unknown_evidence_rate), 4),
            "entity_drift_rate": round(entity_drift["entity_drift_rate"], 4),
            "unsupported_entity_count": entity_drift["unsupported_entity_count"],
            "declared_entity_count": entity_drift["declared_entity_count"],
            "entity_persistence_failure": round(clip01(entity_persistence_failure), 4),
            "forbidden_motif_rate": round(forbidden_motif_rate, 4),
            "forbidden_motif_hits": sorted(set(forbidden_motif_hits)),
            "hard_no_drift_violation": round(hard_no_drift_violation, 4),
            "covered_required_fact_ids": sorted(covered_required_fact_ids),
            "missing_required_fact_ids": sorted(required_fact_ids - covered_required_fact_ids),
            "claim_records": records,
        }

    def _score_claim(
        self,
        claim: Mapping[str, object],
        references: Mapping[str, Mapping[str, object]],
        contradiction_terms: Sequence[str],
        *,
        require_evidence_ids: bool,
    ) -> Dict[str, object]:
        text = _clean_string(claim.get("text") or claim.get("claim"))
        support_ids = _dedupe_strings(
            list(_as_list(claim.get("support_ids"))) +
            list(_as_list(claim.get("evidence_ids"))) +
            list(_as_list(claim.get("citation_ids")))
        )
        support_ids = [str(item) for item in support_ids]
        unknown_evidence_ids = [ref_id for ref_id in support_ids if ref_id not in references]
        known_refs = [references[ref_id] for ref_id in support_ids if ref_id in references]

        support_scores = [_keyword_support(text, ref) for ref in known_refs]
        best_cited_support = max(support_scores) if support_scores else 0.0
        best_inferred_support = max(
            (_keyword_support(text, ref) for ref in references.values()),
            default=0.0,
        )
        claim_type_norm = _normalize_text(claim.get("claim_type") or claim.get("type") or "claim")
        permitted_scene_invention = claim_type_norm in PERMITTED_SCENE_CLAIM_TYPES
        has_known_support = best_cited_support >= 0.18 or (not support_ids and best_inferred_support >= 0.24)
        contradicted = self._claim_contradicted(text, contradiction_terms)
        missing_evidence = bool(require_evidence_ids and not support_ids)
        citation_mismatch = bool(
            unknown_evidence_ids
            or missing_evidence
            or (support_ids and known_refs and best_cited_support < 0.18 and not _looks_like_negated_constraint_claim(text))
        )
        if permitted_scene_invention and not contradicted and not unknown_evidence_ids:
            
            
            
            has_known_support = True
            missing_evidence = False
            citation_mismatch = False
        unsupported = bool((not has_known_support and not _looks_like_negated_constraint_claim(text)) or citation_mismatch or contradicted)

        return {
            "claim_id": _clean_string(claim.get("claim_id") or claim.get("id")),
            "beat_id": claim.get("beat_id"),
            "claim_type": _clean_string(claim.get("claim_type") or claim.get("type") or "claim"),
            "permitted_scene_invention": bool(permitted_scene_invention),
            "text": text,
            "support_ids": support_ids,
            "unknown_evidence_ids": unknown_evidence_ids,
            "best_cited_support": round(best_cited_support, 4),
            "best_inferred_support": round(best_inferred_support, 4),
            "supported": bool(has_known_support and not contradicted and not citation_mismatch),
            "unsupported": bool(unsupported),
            "citation_mismatch": bool(citation_mismatch),
            "missing_evidence": bool(missing_evidence),
            "contradicted": bool(contradicted),
        }

    def _normalize_claims(self, claims: Sequence[object]) -> List[Dict[str, object]]:
        normalized = []
        for index, raw_claim in enumerate(_as_list(claims), start=1):
            if isinstance(raw_claim, str):
                text = _clean_string(raw_claim)
                if text:
                    normalized.append({
                        "claim_id": f"CL{index}",
                        "text": text,
                        "claim_type": "claim",
                        "support_ids": [],
                    })
                continue
            if not isinstance(raw_claim, Mapping):
                continue
            text = _clean_string(raw_claim.get("text") or raw_claim.get("claim") or raw_claim.get("statement"))
            if not text:
                continue
            normalized.append({
                "claim_id": _clean_string(raw_claim.get("claim_id") or raw_claim.get("id") or f"CL{index}"),
                "beat_id": raw_claim.get("beat_id"),
                "text": text,
                "claim_type": _clean_string(raw_claim.get("claim_type") or raw_claim.get("type") or "claim"),
                "support_ids": _dedupe_strings(raw_claim.get("support_ids") or raw_claim.get("evidence_ids") or raw_claim.get("citation_ids") or []),
                "evidence_ids": _dedupe_strings(raw_claim.get("evidence_ids") or []),
                "raw_claim": raw_claim,
            })
        return normalized

    def _claims_from_beats(self, beats: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
        claims = []
        for beat in beats or []:
            beat_id = beat.get("beat_id")
            paragraph = _clean_string(beat.get("paragraph"))
            for fact_id in _dedupe_strings(beat.get("used_fact_ids") or []):
                claims.append({
                    "claim_id": f"B{beat_id}_{fact_id}",
                    "beat_id": beat_id,
                    "claim_type": "legacy_used_fact",
                    "text": paragraph,
                    "support_ids": [fact_id],
                })
            for index, new_fact in enumerate(_dedupe_strings(beat.get("claimed_new_facts") or []), start=1):
                claims.append({
                    "claim_id": f"B{beat_id}_NEW{index}",
                    "beat_id": beat_id,
                    "claim_type": "permitted_scene_invention",
                    "text": new_fact,
                    "support_ids": [],
                })
        return claims

    def _build_reference_index(self, task: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
        references = {}
        for fact in task.get("fact_sheet") or []:
            if not isinstance(fact, Mapping) or not fact.get("id"):
                continue
            references[str(fact["id"])] = {
                "kind": "fact",
                "text": _clean_string(fact.get("text")),
                "keywords": list(fact.get("keywords") or []),
            }
        for constraint in task.get("constraint_sheet") or []:
            if not isinstance(constraint, Mapping) or not constraint.get("id"):
                continue
            references[str(constraint["id"])] = {
                "kind": "constraint",
                "text": _clean_string(constraint.get("text")),
                "forbidden_terms": list(constraint.get("forbidden_terms") or []),
            }
        evidence_pack = task.get("evidence_pack") or {}
        if isinstance(evidence_pack, Mapping):
            doc_id = _clean_string(evidence_pack.get("doc_id"))
            if doc_id:
                references[doc_id] = {
                    "kind": "evidence_doc",
                    "text": _clean_string(evidence_pack.get("topic") or doc_id),
                    "keywords": [doc_id, _clean_string(evidence_pack.get("topic"))],
                }
            for claim in evidence_pack.get("claims") or []:
                claim_id = _clean_string(claim.get("claim_id") or claim.get("id"))
                if not claim_id:
                    continue
                references[claim_id] = {
                    "kind": "evidence_claim",
                    "text": _clean_string(claim.get("text")),
                    "keywords": list(claim.get("keywords") or []),
                }
            for claim in evidence_pack.get("forbidden_claims") or []:
                claim_id = _clean_string(claim.get("claim_id") or claim.get("id"))
                if not claim_id:
                    continue
                references[claim_id] = {
                    "kind": "forbidden_evidence_claim",
                    "text": _clean_string(claim.get("text")),
                    "keywords": list(claim.get("keywords") or []),
                    "forbidden_terms": list(claim.get("keywords") or []),
                }
        return references

    def _contradiction_terms(
        self,
        task: Mapping[str, object],
        profile: Mapping[str, object],
    ) -> List[str]:
        terms = []
        terms.extend(self.patterns.get("global_forbidden_terms") or [])
        terms.extend(self.patterns.get("hard_contradiction_patterns") or [])
        terms.extend(self.patterns.get("unsupported_major_entity_markers") or [])
        terms.extend(profile.get("forbidden_motifs") or [])
        evidence_pack = task.get("evidence_pack") or {}
        if isinstance(evidence_pack, Mapping):
            for claim in evidence_pack.get("forbidden_claims") or []:
                terms.append(_clean_string(claim.get("text")))
                terms.extend(claim.get("keywords") or [])
        for constraint in task.get("constraint_sheet") or []:
            terms.extend((constraint or {}).get("forbidden_terms") or [])
        for constraint in task.get("constraints") or []:
            terms.extend((constraint or {}).get("forbidden_keywords") or [])
            terms.extend((constraint or {}).get("forbidden_entity_keywords") or [])
        return _dedupe_strings(terms)

    def _claim_contradicted(self, text: str, contradiction_terms: Sequence[str]) -> bool:
        text_norm = _normalize_text(text)
        return any(_phrase_hit(text_norm, term) for term in contradiction_terms)

    def _response_text(self, parsed_response: Mapping[str, object]) -> str:
        chunks = [_clean_string(parsed_response.get("title"))]
        for beat in parsed_response.get("beats") or []:
            if isinstance(beat, Mapping):
                chunks.append(_clean_string(beat.get("paragraph")))
                chunks.extend(_dedupe_strings(beat.get("claimed_new_facts") or []))
        chunks.append(_clean_string(parsed_response.get("ending_callback")))
        chunks.extend(_dedupe_strings(parsed_response.get("style_devices") or []))
        return " ".join(chunk for chunk in chunks if chunk)

    def _entity_drift(
        self,
        task: Mapping[str, object],
        parsed_response: Mapping[str, object],
    ) -> Dict[str, object]:
        allowed = {_normalize_text(item) for item in task.get("allowed_entities") or [] if item}
        alias_payload = task.get("entity_aliases_v3") or task.get("entity_aliases") or []
        alias_values = []
        if isinstance(alias_payload, Mapping):
            for value in alias_payload.values():
                alias_values.extend(_as_list(value))
        else:
            alias_values.extend(_as_list(alias_payload))
        for alias in alias_values:
            if isinstance(alias, (list, tuple)):
                for nested in alias:
                    allowed.update(_alias_variants(str(nested)))
            else:
                allowed.update(_alias_variants(str(alias)))
        for fact in task.get("fact_sheet") or []:
            if not isinstance(fact, Mapping):
                continue
            for keyword in fact.get("keywords") or []:
                allowed.update(_alias_variants(str(keyword)))
        for entity in task.get("available_entities") or []:
            if not isinstance(entity, Mapping):
                continue
            for alias in [entity.get("id"), entity.get("name"), *list(entity.get("aliases") or [])]:
                alias_norm = _normalize_text(str(alias or ""))
                if alias_norm:
                    allowed.update(_alias_variants(alias_norm))
        declared = []
        for beat in parsed_response.get("beats") or []:
            if not isinstance(beat, Mapping):
                continue
            for field in ("characters", "places", "objects"):
                declared.extend(_dedupe_strings(beat.get(field) or []))
        for hypothesis in parsed_response.get("hypotheses") or []:
            if not isinstance(hypothesis, Mapping):
                continue
            declared.extend(_dedupe_strings(hypothesis.get("entities") or []))
        unsupported = []
        for entity in declared:
            entity_norm = _normalize_text(entity)
            if not entity_norm:
                continue
            if entity_norm in allowed or any(entity_norm in item or item in entity_norm for item in allowed):
                continue
            unsupported.append(entity)
        return {
            "declared_entity_count": len(declared),
            "unsupported_entity_count": len(unsupported),
            "unsupported_entities": _dedupe_strings(unsupported),
            "entity_drift_rate": clip01(len(unsupported) / max(1, len(declared))),
        }

    def _entity_persistence_failure(
        self,
        task: Mapping[str, object],
        parsed_response: Mapping[str, object],
        full_norm: str,
        entity_drift: Mapping[str, object],
    ) -> float:
        character_names = []
        for fact in task.get("fact_sheet") or []:
            if not isinstance(fact, Mapping) or fact.get("type") != "character":
                continue
            keywords = fact.get("keywords") or []
            if keywords:
                character_names.append(str(keywords[0]))
        if not character_names:
            return clip01(entity_drift.get("entity_drift_rate"))
        missing = [
            name for name in character_names
            if name and not _phrase_hit(full_norm, name)
        ]
        missing_rate = clip01(len(missing) / max(1, len(character_names)))
        return max(missing_rate, clip01(entity_drift.get("entity_drift_rate")))


__all__ = ["SUPPORT_LEDGER_VERSION", "SupportLedgerScorer"]
