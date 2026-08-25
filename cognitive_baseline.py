


from __future__ import annotations

import csv
import glob
import os
import re






SWOW_CANDIDATE_FILENAMES = [
    "swow_strength.csv",
    "strength.SWOW-EN.R123.csv",
    "strength.SWOW-EN.R1.csv",
    "strength.SWOW-EN.R123.20180827.csv",
    "strength.SWOW-EN.R1.20180827.csv",
]

SWOW_GLOB_PATTERNS = [
    "strength.SWOW-EN.R123*.csv",
    "strength.SWOW-EN.R1*.csv",
    "*swow*strength*.csv",
    "*SWOW*strength*.csv",
]


def _find_swow_file(base_path):
    """Locate the SWOW-EN strength file from a file path or base directory."""
    if os.path.isfile(base_path):
        return base_path

    dirs_to_check = []
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
                r123_matches = [match for match in matches if "R123" in match]
                return r123_matches[0] if r123_matches else matches[0]

    return base_path


class CognitiveBaseline:
    """SWOW-based reference baseline for common-response detection."""

    _FILLER_WORDS = frozenset({
        'a', 'an', 'the', 'and', 'or', 'but', 'to', 'of', 'for',
        'with', 'in', 'on', 'at', 'by', 'from', 'as', 'into',
        'about', 'between', 'through', 'during', 'before', 'after',
        'above', 'below', 'under', 'over', 'upon', 'within', 'without',
        'it', 'its', 'this', 'that', 'they', 'them', 'their',
        'he', 'she', 'his', 'her', 'we', 'our', 'you', 'your',
        'who', 'whom', 'which', 'what', 'whose',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'has', 'have', 'had', 'do', 'does', 'did',
        'would', 'could', 'should', 'might', 'may', 'can', 'will',
        'shall', 'must', 'not', 'very', 'also', 'just', 'only', 'more', 'most',
        'so', 'than', 'then', 'if', 'when', 'where', 'how',
        'all', 'each', 'every', 'some', 'any', 'both', 'few',
        'no', 'up', 'out', 'much', 'many', 'such', 'own', 'same',
        'other', 'like', 'even', 'still', 'yet', 'too',
        'used', 'using', 'use', 'made', 'make', 'making',
        'get', 'got', 'getting', 'become', 'becoming',
        'one', 'two', 'new', 'way',
    })

    def __init__(self, swow_path="data/swow_strength.csv", use_conceptnet=False):
        self.swow_data = {}
        self._baseline_cache = {}
        self.swow_available = False

        
        
        self.use_conceptnet = False
        self.conceptnet_available = False
        self._cn_local_available = False

        if use_conceptnet:
            print(
                "[CognitiveBaseline] `use_conceptnet=True` was requested, but "
                "ConceptNet has been removed from the active benchmark runtime. "
                "Proceeding with SWOW-only dynamic baselines."
            )

        resolved_path = _find_swow_file(swow_path)
        self._load_swow(resolved_path)

        print(
            f"[CognitiveBaseline] SWOW data: "
            f"{'LOADED' if self.swow_available else 'NOT AVAILABLE'}"
        )
        print("[CognitiveBaseline] Dynamic baseline mode: SWOW-only")

    
    
    

    def _load_swow(self, filepath):
        """
        Load and parse SWOW-EN association strength data.

        The strength files are usually TAB-delimited and contain unescaped quote
        characters, so we intentionally disable csv quote parsing.
        """
        if not os.path.exists(filepath):
            print(f"[CognitiveBaseline] SWOW data not found at: {filepath}")
            print(
                f"[CognitiveBaseline] Searched candidate filenames: "
                f"{SWOW_CANDIDATE_FILENAMES[:3]}..."
            )
            print(
                "[CognitiveBaseline] Please copy "
                "'strength.SWOW-EN.R123.20180827.csv' (~51MB) to the data/ directory."
            )
            self.swow_available = False
            return

        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"[CognitiveBaseline] Loading SWOW data from: {filepath} ({file_size_mb:.1f} MB)")

        try:
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
                    print(f"[CognitiveBaseline] ERROR: No column headers found in {filepath}")
                    self.swow_available = False
                    return

                fieldnames_lower = {field.lower().strip(): field for field in reader.fieldnames}
                delimiter_label = "TAB" if delimiter == '\t' else repr(delimiter)
                print(f"[CognitiveBaseline] Detected columns: {reader.fieldnames}")
                print(f"[CognitiveBaseline] Delimiter: {delimiter_label}")

                cue_col = None
                for candidate in ['cue', 'word', 'stimulus', 'item']:
                    if candidate in fieldnames_lower:
                        cue_col = fieldnames_lower[candidate]
                        break
                if cue_col is None:
                    cue_col = reader.fieldnames[0]

                resp_col = None
                for candidate in ['response', 'associate', 'target', 'answer']:
                    if candidate in fieldnames_lower:
                        resp_col = fieldnames_lower[candidate]
                        break
                if resp_col is None:
                    resp_col = reader.fieldnames[1]

                strength_col = None
                for candidate in [
                    'r123.strength', 'r1.strength', 'r123_strength',
                    'r1_strength', 'strength', 'prob', 'proportion',
                    'weight', 'r123', 'r1', 'frequency',
                ]:
                    if candidate in fieldnames_lower:
                        strength_col = fieldnames_lower[candidate]
                        break
                if strength_col is None:
                    strength_col = reader.fieldnames[-1] if len(reader.fieldnames) >= 3 else None
                if strength_col is None:
                    print(f"[CognitiveBaseline] ERROR: Cannot identify strength column in {filepath}")
                    self.swow_available = False
                    return

                strength_col_lower = strength_col.lower().strip()
                is_probability_col = ('strength' in strength_col_lower or 'prob' in strength_col_lower)
                if not is_probability_col:
                    print(
                        f"[CognitiveBaseline] WARNING: Using column '{strength_col}' which may be a raw count."
                    )

                print(
                    f"[CognitiveBaseline] Using columns: cue='{cue_col}', "
                    f"response='{resp_col}', strength='{strength_col}'"
                )

                row_count = 0
                parse_errors = 0
                swow_data = {}
                for row in reader:
                    try:
                        cue = row[cue_col].strip().lower()
                        response = row[resp_col].strip().lower()
                        strength = float(row[strength_col].strip())
                    except (ValueError, TypeError, KeyError, AttributeError):
                        parse_errors += 1
                        if parse_errors <= 3:
                            try:
                                preview = dict(list(row.items())[:3])
                            except Exception:
                                preview = '<unavailable>'
                            print(f"[CognitiveBaseline]   Parse error in row: {preview}...")
                        elif parse_errors == 4:
                            print("[CognitiveBaseline]   (suppressing further parse error warnings)")
                        continue

                    if not cue or not response:
                        continue
                    swow_data.setdefault(cue, []).append((response, strength))
                    row_count += 1

                for cue in swow_data:
                    swow_data[cue].sort(key=lambda item: item[1], reverse=True)

                self.swow_data = swow_data
                self.swow_available = True
                print(
                    f"[CognitiveBaseline] Successfully loaded {row_count:,} associations "
                    f"for {len(self.swow_data):,} cue words."
                )
                if parse_errors > 0:
                    print(f"[CognitiveBaseline] Skipped {parse_errors:,} malformed rows.")

                if self.swow_data:
                    sample_cue = next(iter(self.swow_data))
                    sample_max = self.swow_data[sample_cue][0][1] if self.swow_data[sample_cue] else 0.0
                    if sample_max > 1.0:
                        print(
                            f"[CognitiveBaseline] WARNING: Max strength for '{sample_cue}' = {sample_max}. "
                            "Values > 1.0 suggest raw counts, not probabilities."
                        )
                    else:
                        print(
                            f"[CognitiveBaseline] Data sanity check OK: '{sample_cue}' top response = "
                            f"'{self.swow_data[sample_cue][0][0]}' (strength={sample_max:.4f})"
                        )
        except Exception as exc:
            print(f"[CognitiveBaseline] ERROR loading SWOW data: {exc}")
            import traceback
            traceback.print_exc()
            self.swow_available = False

    
    
    

    def _extract_keywords(self, concept_phrase):
        """Extract lookup keywords from a multi-word concept phrase."""
        fillers = {'a', 'an', 'the', 'and', 'or', 'for', 'of', 'that',
                   'which', 'to', 'in', 'on', 'is', 'are', 'it', 'be'}

        words = re.findall(r'[a-zA-Z]+', concept_phrase.lower())
        keywords = [word for word in words if word not in fillers and len(word) > 1]

        expanded = set(keywords)
        for word in keywords:
            if word.endswith('ies') and len(word) > 4:
                expanded.add(word[:-3] + 'y')
            elif word.endswith('es') and len(word) > 3:
                expanded.add(word[:-2])
            elif word.endswith('s') and len(word) > 2 and not word.endswith('ss'):
                expanded.add(word[:-1])
            if not word.endswith('s'):
                expanded.add(word + 's')

        return list(expanded)

    def get_dynamic_baseline(self, target_concept, swow_top_k=12, conceptnet_top_k=8):
        """
        Generate a SWOW-only dynamic zero-originality baseline for a concept.

        `conceptnet_top_k` is accepted for backward compatibility and ignored.
        """
        del conceptnet_top_k

        cache_key = (target_concept or '').lower().strip()
        if cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]

        keywords = self._extract_keywords(target_concept or '')
        if not keywords or not self.swow_available:
            self._baseline_cache[cache_key] = []
            return []

        baseline_set = set()
        for keyword in keywords:
            if keyword not in self.swow_data:
                continue
            for response, strength in self.swow_data[keyword][:swow_top_k]:
                if strength >= 0.02:
                    baseline_set.add(response)

        result = list(baseline_set)
        self._baseline_cache[cache_key] = result
        return result

    
    
    

    def is_in_baseline(self, target_concept, response_text, scorer=None, semantic_threshold=0.25):
        """
        Check whether a response falls into the dynamic zero-originality basin.

        Logic:
            1. Generate a SWOW baseline for the prompt concept.
            2. Check short / low-content responses for overlap with that baseline.
            3. Optionally apply semantic matching using the provided scorer.
        """
        baseline = self.get_dynamic_baseline(target_concept)
        if not baseline:
            return False

        response_lower = (response_text or '').lower().strip()
        response_words = set(re.findall(r'[a-z]+', response_lower))

        prompt_words = set(re.findall(r'[a-z]+', (target_concept or '').lower()))
        expanded_prompt = set()
        for word in prompt_words:
            expanded_prompt.add(word)
            if word.endswith('s') and len(word) > 2 and not word.endswith('ss'):
                expanded_prompt.add(word[:-1])
            if word.endswith('es') and len(word) > 3:
                expanded_prompt.add(word[:-2])
            if word.endswith('ies') and len(word) > 4:
                expanded_prompt.add(word[:-3] + 'y')
            if not word.endswith('s'):
                expanded_prompt.add(word + 's')

        baseline_vocab = set()
        for baseline_answer in baseline:
            for word in re.findall(r'[a-z]+', baseline_answer.lower()):
                if len(word) > 2:
                    baseline_vocab.add(word)

        creative_words = {
            word for word in response_words
            if len(word) > 2
            and word not in self._FILLER_WORDS
            and word not in expanded_prompt
            and word not in baseline_vocab
        }
        num_creative = len(creative_words)

        
        for common_answer in baseline:
            common_words = set(re.findall(r'[a-z]+', common_answer.lower()))
            for common_word in common_words:
                if len(common_word) <= 2:
                    continue
                if common_word in response_words:
                    if len(common_word) >= 4:
                        if num_creative <= 2 and len(response_lower) <= 80:
                            return True
                        if len(response_lower) <= 25:
                            return True
                    elif len(common_word) == 3 and len(response_lower) <= 30:
                        return True

        
        for common_answer in baseline:
            if ' ' in common_answer:
                if common_answer in response_lower:
                    return True
            elif common_answer == response_lower:
                return True

        
        if scorer is not None:
            for common_answer in baseline:
                distance = scorer.calculate_originality(common_answer, response_lower)
                effective_threshold = semantic_threshold
                if len(response_lower) > 60:
                    effective_threshold *= 0.7
                if len(response_lower) > 100:
                    effective_threshold *= 0.5
                if distance < effective_threshold:
                    return True

        return False

    
    
    

    def get_baseline_report(self, target_concept):
        """Return a human-readable report for the SWOW-only dynamic baseline."""
        keywords = self._extract_keywords(target_concept)
        baseline = self.get_dynamic_baseline(target_concept)

        report = {
            "target_concept": target_concept,
            "runtime_mode": "swow_only",
            "extracted_keywords": keywords,
            "swow_available": self.swow_available,
            "baseline_size": len(baseline),
            "baseline_words": sorted(baseline),
            "swow_details": {},
            "conceptnet_available": False,
            "conceptnet_local": False,
            "conceptnet_details": {},
        }

        if self.swow_available:
            for keyword in keywords:
                if keyword in self.swow_data:
                    report["swow_details"][keyword] = [
                        {"response": response, "strength": round(strength, 4)}
                        for response, strength in self.swow_data[keyword][:10]
                    ]

        return report
