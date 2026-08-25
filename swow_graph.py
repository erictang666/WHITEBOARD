


from __future__ import annotations

import csv
import glob
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SWOW_CANDIDATE_FILENAMES = [
    "strength.SWOW-EN.R123.20180827.csv",
    "strength.SWOW-EN.R123.csv",
    "strength.SWOW-EN.R1.20180827.csv",
    "strength.SWOW-EN.R1.csv",
    "swow_strength.csv",
]

SWOW_GLOB_PATTERNS = [
    "strength.SWOW-EN.R123*.csv",
    "strength.SWOW-EN.R1*.csv",
    "*swow*strength*.csv",
    "*SWOW*strength*.csv",
]

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for",
    "with", "in", "on", "at", "by", "is", "are", "it", "this",
    "that", "as", "be", "been", "being", "was", "were", "am",
    "do", "does", "did", "done", "have", "has", "had", "from",
    "into", "through", "during", "before", "after", "above", "below",
    "under", "over", "again", "further", "then", "once", "here",
    "there", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "will", "just",
}


def find_swow_file(base_path: str) -> Optional[str]:
    if os.path.isfile(base_path):
        return base_path

    dirs_to_check: List[str] = []
    if os.path.isdir(base_path):
        dirs_to_check.append(base_path)
    else:
        parent_dir = os.path.dirname(base_path)
        if parent_dir and os.path.isdir(parent_dir):
            dirs_to_check.append(parent_dir)

    for candidate_dir in ["data", ".", "data/SWOW-EN18", "SWOW-EN18"]:
        if os.path.isdir(candidate_dir) and candidate_dir not in dirs_to_check:
            dirs_to_check.append(candidate_dir)

    for directory in dirs_to_check:
        for candidate in SWOW_CANDIDATE_FILENAMES:
            candidate_path = os.path.join(directory, candidate)
            if os.path.isfile(candidate_path):
                return candidate_path

    for directory in dirs_to_check:
        for pattern in SWOW_GLOB_PATTERNS:
            matches = glob.glob(os.path.join(directory, pattern))
            if matches:
                r123_matches = [m for m in matches if 'R123' in os.path.basename(m)]
                return r123_matches[0] if r123_matches else matches[0]

    return None


class SWOWGraph:
    """Reusable loader/query interface for SWOW association strengths."""

    def __init__(self, swow_path: str = "data"):
        self.swow_path = swow_path
        self.available = False
        self.filepath: Optional[str] = None
        self.associations: Dict[str, List[Tuple[str, float]]] = {}
        self._strength_index: Dict[str, Dict[str, float]] = {}
        self._load()

    
    
    

    def _load(self) -> None:
        filepath = find_swow_file(self.swow_path)
        if not filepath or not os.path.exists(filepath):
            self.available = False
            return

        self.filepath = filepath
        cue_to_rows: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        with open(filepath, 'r', encoding='utf-8-sig') as handle:
            first_line = handle.readline()
            handle.seek(0)
            delimiter = '\t' if '\t' in first_line else ','
            reader = csv.DictReader(
                handle,
                delimiter=delimiter,
                quotechar=None,
                quoting=csv.QUOTE_NONE,
            )
            if not reader.fieldnames:
                return

            fieldnames_lower = {name.lower().strip(): name for name in reader.fieldnames}
            cue_col = self._pick_column(fieldnames_lower, ['cue', 'word', 'stimulus', 'item'], default=reader.fieldnames[0])
            resp_col = self._pick_column(fieldnames_lower, ['response', 'associate', 'target', 'answer'], default=reader.fieldnames[1])
            strength_col = self._pick_column(
                fieldnames_lower,
                ['r123.strength', 'r1.strength', 'r123_strength', 'r1_strength', 'strength', 'prob', 'proportion', 'weight', 'r123', 'r1', 'frequency'],
                default=reader.fieldnames[-1],
            )

            for row in reader:
                try:
                    cue = self.normalize_token(row[cue_col])
                    response = self.normalize_token(row[resp_col])
                    strength = float(str(row[strength_col]).strip())
                except Exception:
                    continue
                if not cue or not response:
                    continue
                cue_to_rows[cue].append((response, strength))

        self.associations = {}
        self._strength_index = {}
        for cue, rows in cue_to_rows.items():
            rows.sort(key=lambda item: item[1], reverse=True)
            self.associations[cue] = rows
            self._strength_index[cue] = {resp: strength for resp, strength in rows}

        self.available = bool(self.associations)

    @staticmethod
    def _pick_column(fieldnames_lower: Dict[str, str], candidates: Sequence[str], default: str) -> str:
        for candidate in candidates:
            if candidate in fieldnames_lower:
                return fieldnames_lower[candidate]
        return default

    
    
    

    @staticmethod
    def normalize_token(token: str) -> str:
        token = token.lower().strip()
        token = token.replace('_', ' ')
        token = re.sub(r'[^a-z\s-]+', ' ', token)
        token = re.sub(r'\s+', ' ', token).strip()
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + 'y'
        elif token.endswith("es") and len(token) > 4 and token[:-2] not in {"th", "sh"}:
            token = token[:-2]
        elif token.endswith('s') and len(token) > 3 and not token.endswith('ss'):
            token = token[:-1]
        return token

    def tokenize_text(self, text: str, excluded_tokens: Optional[Iterable[str]] = None) -> List[str]:
        excluded = {self.normalize_token(token) for token in (excluded_tokens or []) if token}
        tokens = []
        for raw in re.findall(r"[a-zA-Z]+", text.lower()):
            token = self.normalize_token(raw)
            if not token or token in STOP_WORDS or token in excluded:
                continue
            tokens.append(token)
        return tokens

    
    
    

    def get_strength(self, cue: str, response: str) -> float:
        if not self.available:
            return 0.0
        cue_norm = self.normalize_token(cue)
        resp_norm = self.normalize_token(response)
        return float(self._strength_index.get(cue_norm, {}).get(resp_norm, 0.0))

    def top_associates(self, cue: str, k: int = 20) -> List[Tuple[str, float]]:
        cue_norm = self.normalize_token(cue)
        return list(self.associations.get(cue_norm, [])[:k])

    def compute_token_support(
        self,
        cues: Sequence[str],
        token: str,
        alpha: float = 0.7,
        bridge_limit: int = 20,
    ) -> Dict[str, object]:
        token_norm = self.normalize_token(token)
        best = {
            "token": token_norm,
            "score": 0.0,
            "best_cue": None,
            "direct": 0.0,
            "two_hop": 0.0,
            "three_hop": 0.0,
            "bridge": None,
            "bridge2": None,
            "support_type": None,
        }
        if not token_norm or not self.available:
            return best

        for cue in cues:
            cue_norm = self.normalize_token(cue)
            if not cue_norm:
                continue

            direct = self.get_strength(cue_norm, token_norm)
            two_hop = 0.0
            three_hop = 0.0
            best_bridge = None
            best_bridge2 = None

            cue_neighbors = self.top_associates(cue_norm, bridge_limit)
            for bridge, bridge_strength in cue_neighbors:
                if bridge == token_norm:
                    continue
                second_leg = self.get_strength(bridge, token_norm)
                two_score = bridge_strength * second_leg
                if two_score > two_hop:
                    two_hop = two_score
                    best_bridge = bridge
                    best_bridge2 = None

                bridge_neighbors = self.top_associates(bridge, bridge_limit)
                for bridge2, bridge2_strength in bridge_neighbors:
                    if bridge2 in {token_norm, cue_norm}:
                        continue
                    third_leg = self.get_strength(bridge2, token_norm)
                    three_score = bridge_strength * bridge2_strength * third_leg
                    if three_score > three_hop:
                        three_hop = three_score
                        best_bridge = bridge
                        best_bridge2 = bridge2

            direct_term = alpha * direct
            two_term = (alpha ** 2) * two_hop
            three_term = (alpha ** 3) * three_hop
            combined = direct_term + two_term + three_term
            support_type = None
            if direct_term >= two_term and direct_term >= three_term and direct_term > 0:
                support_type = "direct"
            elif two_term >= three_term and two_term > 0:
                support_type = "two_hop"
            elif three_term > 0:
                support_type = "three_hop"

            if combined > best["score"]:
                best.update({
                    "score": float(combined),
                    "best_cue": cue_norm,
                    "direct": float(direct_term),
                    "two_hop": float(two_term),
                    "three_hop": float(three_term),
                    "bridge": best_bridge,
                    "bridge2": best_bridge2,
                    "support_type": support_type,
                })
        return best

    def score_answer_support(
        self,
        cues: Sequence[str],
        answer_text: str,
        *,
        top_k: int = 3,
        epsilon: float = 0.005,
        alpha: float = 0.7,
        bridge_limit: int = 20,
        excluded_tokens: Optional[Iterable[str]] = None,
    ) -> Dict[str, object]:
        tokens = self.tokenize_text(answer_text, excluded_tokens=excluded_tokens)
        if not tokens:
            return {
                "score": 0.0,
                "direct_score": 0.0,
                "coverage_score": 0.0,
                "coverage": 0.0,
                "cue_diversity": 0.0,
                "tokens": [],
                "token_evidence": [],
                "available": self.available,
            }

        evidence = [
            self.compute_token_support(cues, token, alpha=alpha, bridge_limit=bridge_limit)
            for token in tokens
        ]
        sorted_scores = sorted((item["score"] for item in evidence), reverse=True)
        k = min(top_k, len(sorted_scores))
        top_mean = sum(sorted_scores[:k]) / k if k > 0 else 0.0
        coverage_hits = sum(1 for item in evidence if item["score"] >= epsilon)
        coverage = coverage_hits / len(evidence) if evidence else 0.0
        unique_cues = {self.normalize_token(c) for c in cues if c}
        cue_diversity = (
            len({item["best_cue"] for item in evidence if item.get("best_cue") and item["score"] >= epsilon}) /
            max(1, len(unique_cues))
        )
        score = 0.5 * top_mean + 0.3 * coverage + 0.2 * cue_diversity
        direct_score = max((item["direct"] for item in evidence), default=0.0)
        return {
            "score": round(float(score), 4),
            "direct_score": round(float(direct_score), 4),
            "coverage_score": round(float(coverage), 4),
            "coverage": round(float(coverage), 4),
            "cue_diversity": round(float(cue_diversity), 4),
            "tokens": tokens,
            "token_evidence": evidence,
            "available": self.available,
        }

    
    
    

    def get_dynamic_baseline(self, target_concept: str, swow_top_k: int = 12, min_strength: float = 0.02) -> List[str]:
        if not self.available:
            return []
        keywords = self.tokenize_text(target_concept)
        baseline = []
        seen = set()
        for keyword in keywords:
            for response, strength in self.top_associates(keyword, swow_top_k):
                if strength < min_strength:
                    continue
                if response not in seen:
                    baseline.append(response)
                    seen.add(response)
        return baseline
