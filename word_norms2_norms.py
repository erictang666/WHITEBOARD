


from __future__ import annotations

import csv
import glob
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DATA_DIR = "data"
CACHE_FILENAMES = [
    "word_norms2_cache.json",
    "word_norms2_norms_cache.json",
]
RAW_GLOB_PATTERNS = [
    "**/*word_norms2*.csv",
    "**/*word_norms2*.tsv",
    "**/*word*norm*.csv",
    "**/*word*norm*.tsv",
    "**/*semantic*feature*.csv",
    "**/*semantic*feature*.tsv",
    "**/*cue*feature*.csv",
    "**/*cue*feature*.tsv",
    "**/*translated*.csv",
    "**/*translated*.tsv",
    "**/*.zip",
]

GENERIC_ARTIFACT_WORDS = {
    "object", "thing", "item", "stuff", "something",
}

INSTANCE_PREDICATE_HINTS = {
    "shape:round",
    "shape:flat",
    "texture:hard",
    "texture:soft",
    "sound:loud",
    "sound:noise",
    "edible:true",
    "color:red",
    "material:metal",
    "material:paper",
    "container:container",
    "shape:hollow",
    "form:foldable",
    "weight:light",
    "surface:writable",
    "form:stackable",
    "resistance:water",
    "resistance:pressure",
    "strength:load_bearing",
}

UUT_AFFORDANCE_MAP = {
    "container": {"container:container"},
    "flat_surface": {"shape:flat", "surface:flat"},
    "writable_surface": {"surface:writable"},
    "shapeable_surface": {"form:foldable", "texture:soft", "material:paper"},
    "acoustic_cavity": {"shape:hollow", "container:container", "sound:noise"},
    "lightweight": {"weight:light"},
    "hangable": {"affordance:hangable", "part:handle"},
    "rigid_shell": {"texture:hard", "shape:hollow", "material:metal", "strength:load_bearing"},
    "water_resistant": {"resistance:water", "material:metal"},
    "pressure_resistant": {"resistance:pressure", "strength:load_bearing", "material:metal"},
    "load_bearing": {"strength:load_bearing", "texture:hard"},
    "metal_conductive": {"material:metal", "conductivity:conductive"},
    "reflective": {"surface:reflective", "material:metal"},
    "sharp_edge": {"edge:sharp"},
    "heat_resistant": {"resistance:heat", "material:metal"},
}


@dataclass
class ConceptMatch:
    concept: Optional[str]
    matched_alias: Optional[str]
    confidence: float


class WordNorms2Norms:
    """Loader/query interface for word-norms2 semantic feature norms."""

    def __init__(self, data_dir: str = DATA_DIR, cache_path: Optional[str] = None):
        self.data_dir = os.path.abspath(data_dir)
        self.cache_path = os.path.abspath(cache_path) if cache_path else None
        self.available = False
        self.source_path: Optional[str] = None
        self.concept_db: Dict[str, Dict[str, object]] = {}
        self.alias_index: Dict[str, str] = {}
        self._load()

    
    
    

    def _load(self) -> None:
        candidate_cache = self.cache_path or self._find_existing_cache()
        if candidate_cache and os.path.exists(candidate_cache):
            self._load_cache(candidate_cache)
            return

        raw_path = self.find_raw_file(self.data_dir)
        if raw_path:
            try:
                output_cache = candidate_cache or os.path.join(self.data_dir, "word_norms2_cache.json")
                self.build_cache_from_raw(raw_path, output_cache)
                self._load_cache(output_cache)
            except Exception:
                self.available = False

    def _find_existing_cache(self) -> Optional[str]:
        for filename in CACHE_FILENAMES:
            path = os.path.join(self.data_dir, filename)
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def find_raw_file(data_dir: str = DATA_DIR) -> Optional[str]:
        base = os.path.abspath(data_dir)
        if os.path.isfile(base):
            return base
        for pattern in RAW_GLOB_PATTERNS:
            matches = glob.glob(os.path.join(base, pattern), recursive=True)
            preferred = []
            fallback = []
            for match in matches:
                lower = os.path.basename(match).lower()
                if lower.endswith('.zip'):
                    preferred.append(match)
                elif 'cue' in lower or 'feature' in lower or 'translated' in lower or 'word_norms2' in lower:
                    preferred.append(match)
                else:
                    fallback.append(match)
            if preferred:
                return sorted(preferred)[0]
            if fallback:
                return sorted(fallback)[0]
        return None

    def _load_cache(self, path: str) -> None:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        concepts = payload.get("concepts", {}) if isinstance(payload, dict) else {}
        self.concept_db = concepts
        self.alias_index = {}
        for concept, data in concepts.items():
            aliases = set(data.get("aliases", [])) | {concept}
            for alias in aliases:
                norm = self.normalize_phrase(alias)
                if norm:
                    self.alias_index[norm] = concept
        self.available = bool(self.concept_db)
        self.source_path = payload.get("source_path") if isinstance(payload, dict) else path
        self.cache_path = path

    
    
    

    @staticmethod
    def normalize_token(token: str) -> str:
        token = token.lower().strip().replace('_', ' ')
        token = re.sub(r'[^a-z\s-]+', ' ', token)
        token = re.sub(r'\s+', ' ', token).strip()
        if token.endswith('ies') and len(token) > 4:
            return token[:-3] + 'y'
        if token.endswith(('ches', 'shes', 'xes', 'zes')) and len(token) > 4:
            return token[:-2]
        if token.endswith('es') and len(token) > 4 and not token.endswith(('ses', 'xes', 'zes')):
            return token[:-2]
        if token.endswith('s') and len(token) > 3 and not token.endswith('ss'):
            return token[:-1]
        return token

    @classmethod
    def normalize_phrase(cls, phrase: str) -> str:
        tokens = [cls.normalize_token(token) for token in re.findall(r"[a-zA-Z]+", phrase.lower())]
        tokens = [token for token in tokens if token and token not in {'a', 'an', 'the', 'of', 'for', 'to'}]
        return ' '.join(tokens).strip()

    
    
    

    @classmethod
    def canonicalize_feature(cls, feature_text: str, raw_feature_text: Optional[str] = None) -> Optional[str]:
        text = cls.normalize_phrase(feature_text)
        raw = cls.normalize_phrase(raw_feature_text or "")
        combined = ' '.join(part for part in [text, raw] if part).strip()
        if not combined:
            return None

        relation = None
        value = text or raw
        for prefix, rel in [
            ('made of ', 'material'),
            ('used for ', 'function'),
            ('use for ', 'function'),
            ('does ', 'action'),
            ('can ', 'capability'),
            ('has ', 'part'),
            ('is ', 'attr'),
        ]:
            if combined.startswith(prefix):
                relation = rel
                value = combined[len(prefix):].strip()
                break

        if relation is None:
            relation = 'attr'
            value = combined

        predicate_map = [
            (r'\bround\b|\bspherical\b|\bcircular\b|\borb\b|\bball\b', 'shape:round'),
            (r'\bflat\b|\bplanar\b', 'shape:flat'),
            (r'\bsquare\b', 'attr:square'),
            (r'\brectangular\b|\brectangle\b', 'attr:rectangular'),
            (r'\btriangular\b|\btriangle\b', 'attr:triangular'),
            (r'\bhard\b|\brigid\b|\bsolid\b|\bstiff\b|\bfirm\b', 'texture:hard'),
            (r'\bsoft\b|\bflexible\b|\bbendable\b|\bmalleable\b', 'texture:soft'),
            (r'\bfragile\b|\bbreakable\b', 'attr:fragile'),
            (r'\bloud\b|\bbooming\b|\broaring\b|\bblaring\b', 'sound:loud'),
            (r'\bnoise\b|\bsound\b|\bnoisy\b', 'sound:noise'),
            (r'\bedible\b|\beat\b|\beaten\b|\bfood\b', 'edible:true'),
            (r'\bred\b', 'color:red'),
            (r'\bmetal\b|\btin\b|\baluminum\b|\baluminium\b|\bsteel\b|\biron\b', 'material:metal'),
            (r'\bpaper\b|\bcardboard\b|\bpaperboard\b', 'material:paper'),
            (r'\bcontainer\b|\bhold things\b|\bstore things\b|\bstorage\b|\bcarry things\b|\bpackag', 'container:container'),
            (r'\bhollow\b|\bempty inside\b|\bcavity\b|\binside empty\b', 'shape:hollow'),
            (r'\bfold\b|\bfoldable\b|\bcollaps\b', 'form:foldable'),
            (r'\blight\b|\blightweight\b', 'weight:light'),
            (r'\bwrite\b|\bwritable\b|\blabel\b|\bdraw on\b', 'surface:writable'),
            (r'\bstack\b|\bstackable\b', 'form:stackable'),
            (r'\bwaterproof\b|\bwater resistant\b|\bwater resistant\b', 'resistance:water'),
            (r'\bpressure resistant\b|\bpressure proof\b', 'resistance:pressure'),
            (r'\bheat resistant\b|\bfireproof\b', 'resistance:heat'),
            (r'\bload bearing\b|\bsturdy\b|\bstrong\b|\bsupport weight\b', 'strength:load_bearing'),
            (r'\bconductive\b|\bconduct\b|\belectric\b|\belectrical\b', 'conductivity:conductive'),
            (r'\breflective\b|\breflect\b|\bmirror\b|\bshiny\b', 'surface:reflective'),
            (r'\bsharp\b|\bblade\b|\bcut\b|\bslice\b|\bknife edge\b', 'edge:sharp'),
            (r'\bhang\b|\bhanging\b|\bsuspend\b', 'affordance:hangable'),
            (r'\bhandle\b|\bgrip\b', 'part:handle'),
        ]
        for pattern, predicate in predicate_map:
            if re.search(pattern, value):
                return predicate

        if relation == 'material':
            return f'material:{value}'
        if relation == 'function':
            return f'function:{value}'
        if relation == 'part':
            return f'part:{value}'
        if relation == 'capability':
            return f'capability:{value}'
        if relation == 'action':
            return f'action:{value}'
        return f'attr:{value}'

    @staticmethod
    def predicate_to_phrase(predicate: str) -> str:
        phrase_map = {
            'shape:round': 'round',
            'shape:flat': 'flat',
            'texture:hard': 'hard',
            'texture:soft': 'soft / flexible',
            'sound:loud': 'loud',
            'sound:noise': 'makes a noise',
            'edible:true': 'edible',
            'color:red': 'red',
            'material:metal': 'made of metal',
            'material:paper': 'made of paper / cardboard',
            'container:container': 'can hold things',
            'shape:hollow': 'hollow',
            'form:foldable': 'foldable',
            'weight:light': 'lightweight',
            'surface:writable': 'writable',
            'form:stackable': 'stackable',
            'resistance:water': 'water resistant',
            'resistance:pressure': 'pressure resistant',
            'resistance:heat': 'heat resistant',
            'strength:load_bearing': 'strong / load-bearing',
            'conductivity:conductive': 'conductive',
            'surface:reflective': 'reflective',
            'edge:sharp': 'sharp',
            'affordance:hangable': 'hangable',
            'part:handle': 'has a handle',
        }
        if predicate in phrase_map:
            return phrase_map[predicate]
        if ':' in predicate:
            _, tail = predicate.split(':', 1)
            return tail.replace('_', ' ')
        return predicate.replace('_', ' ')

    @staticmethod
    def weight_from_proportion(proportion: float, is_top_feature: bool = False) -> float:
        proportion = max(0.0, min(1.0, float(proportion or 0.0)))
        if proportion >= 0.16:
            gamma = 1.0
        elif proportion >= 0.10:
            gamma = 0.7
        elif is_top_feature:
            gamma = 0.4
        else:
            gamma = 0.0
        return proportion * gamma

    
    
    

    def build_cache_from_raw(self, raw_path: str, output_path: str) -> str:
        rows = self._read_rows(raw_path)
        concept_db: Dict[str, Dict[str, object]] = {}
        for row in rows:
            concept = self.normalize_phrase(row.get('concept', ''))
            translated = str(row.get('translated_feature') or '').strip()
            raw_feature = str(row.get('raw_feature') or '').strip()
            if not concept or not (translated or raw_feature):
                continue

            proportion = self._extract_proportion(row)
            if proportion <= 0.0:
                continue

            predicate = self.canonicalize_feature(translated or raw_feature, raw_feature_text=raw_feature)
            if not predicate:
                continue

            entry = concept_db.setdefault(concept, {
                'aliases': [],
                'features': {},
                'feature_meta': {},
                'raw_features': {},
                'translated_features': {},
                'participant_n': None,
                'sources': ['word_norms2_2019'],
            })
            participant_n = row.get('n')
            try:
                participant_n = int(float(participant_n)) if participant_n is not None and participant_n != '' else None
            except Exception:
                participant_n = None
            if participant_n is not None:
                current_n = entry.get('participant_n')
                entry['participant_n'] = max(int(current_n or 0), participant_n)

            alias = self.normalize_phrase(row.get('alias', '') or row.get('concept', ''))
            if alias and alias != concept and alias not in entry['aliases']:
                entry['aliases'].append(alias)

            meta = entry['feature_meta'].setdefault(predicate, {
                'max_proportion': 0.0,
                'raw_variants': [],
                'translated_variants': [],
                'top_candidate': False,
            })
            meta['max_proportion'] = max(float(meta.get('max_proportion', 0.0)), proportion)
            if raw_feature and raw_feature not in meta['raw_variants']:
                meta['raw_variants'].append(raw_feature)
            if translated and translated not in meta['translated_variants']:
                meta['translated_variants'].append(translated)

        for concept, entry in concept_db.items():
            meta_items = sorted(
                entry['feature_meta'].items(),
                key=lambda item: float(item[1].get('max_proportion', 0.0)),
                reverse=True,
            )
            top_predicates = {predicate for predicate, _ in meta_items[:5]}
            features = {}
            raw_features = {}
            translated_features = {}
            feature_meta = entry['feature_meta']
            for predicate, meta in meta_items:
                proportion = float(meta.get('max_proportion', 0.0))
                is_top_feature = predicate in top_predicates
                weight = self.weight_from_proportion(proportion, is_top_feature=is_top_feature)
                if weight <= 0.0:
                    continue
                meta['top_candidate'] = is_top_feature
                features[predicate] = round(weight, 6)
                raw_features[predicate] = meta.get('raw_variants', [])
                translated_features[predicate] = meta.get('translated_variants', [])
            entry['top_predicates'] = sorted(
                list(top_predicates),
                key=lambda predicate: features.get(predicate, 0.0),
                reverse=True,
            )
            entry['features'] = features
            entry['raw_features'] = raw_features
            entry['translated_features'] = translated_features
            entry['feature_meta'] = feature_meta

        payload = {
            'source_path': raw_path,
            'concept_count': len(concept_db),
            'concepts': concept_db,
            'resource': 'word_norms2_2019',
        }
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return output_path

    def _extract_proportion(self, row: Dict[str, object]) -> float:
        n_value = self._parse_float(row.get('n'))
        normalized_translated = self._parse_float(row.get('normalized_translated'))
        frequency_translated = self._parse_float(row.get('frequency_translated'))
        normalized_feature = self._parse_float(row.get('normalized_feature'))
        frequency_feature = self._parse_float(row.get('frequency_feature'))

        if normalized_translated is not None:
            return self._coerce_to_proportion(normalized_translated, n_value)
        if frequency_translated is not None:
            return self._coerce_to_proportion(frequency_translated, n_value)
        if normalized_feature is not None:
            return self._coerce_to_proportion(normalized_feature, n_value)
        if frequency_feature is not None:
            return self._coerce_to_proportion(frequency_feature, n_value)
        return 0.0

    @staticmethod
    def _parse_float(value) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        text = text.replace('%', '')
        try:
            return float(text)
        except Exception:
            return None

    @staticmethod
    def _coerce_to_proportion(value: Optional[float], n_value: Optional[float]) -> float:
        if value is None:
            return 0.0
        v = float(value)
        if v <= 0.0:
            return 0.0
        if 0.0 < v <= 1.0:
            return min(1.0, v)
        if n_value and n_value > 1.0 and v <= n_value:
            return min(1.0, v / n_value)
        if 1.0 < v <= 100.0:
            return min(1.0, v / 100.0)
        return 0.0

    def _read_rows(self, raw_path: str) -> List[Dict[str, object]]:
        if raw_path.lower().endswith('.zip'):
            return self._read_rows_from_zip(raw_path)
        return self._read_rows_from_text_file(raw_path)

    def _read_rows_from_zip(self, zip_path: str) -> List[Dict[str, object]]:
        with zipfile.ZipFile(zip_path) as archive:
            candidate_names = [
                name for name in archive.namelist()
                if name.lower().endswith(('.csv', '.tsv'))
            ]
            candidate_names.sort(key=lambda name: (
                'cue' not in name.lower() and 'feature' not in name.lower(),
                'translated' not in name.lower(),
                len(name),
            ))
            for name in candidate_names:
                with archive.open(name, 'r') as handle:
                    text = io.TextIOWrapper(handle, encoding='utf-8-sig', errors='ignore')
                    rows = self._parse_reader_rows(text)
                    if rows:
                        return rows
        return []

    def _read_rows_from_text_file(self, raw_path: str) -> List[Dict[str, object]]:
        with open(raw_path, 'r', encoding='utf-8-sig', errors='ignore') as handle:
            return self._parse_reader_rows(handle)

    def _parse_reader_rows(self, handle) -> List[Dict[str, object]]:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = '\t' if sample.count('\t') > sample.count(',') else ','
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            return []

        field_map = {name.lower().strip(): name for name in reader.fieldnames}
        cue_col = self._pick_col(field_map, ['cue', 'concept', 'item', 'word'])
        raw_feature_col = self._pick_col(field_map, ['feature', 'raw feature', 'raw_feature'])
        translated_col = self._pick_col(field_map, ['translated', 'translated feature', 'translated_feature'])
        freq_feature_col = self._pick_col(field_map, ['frequency feature', 'frequency_feature', 'feature frequency'])
        freq_translated_col = self._pick_col(field_map, ['frequency translated', 'frequency_translated', 'translated frequency'])
        norm_feature_col = self._pick_col(field_map, ['normalized feature', 'normalized_feature', 'feature percent', 'feature percentage'])
        norm_translated_col = self._pick_col(field_map, ['normalized translated', 'normalized_translated', 'translated percent', 'translated percentage'])
        n_col = self._pick_col(field_map, ['n', 'participants', 'num participants'])
        where_col = self._pick_col(field_map, ['where'])

        rows = []
        for row in reader:
            concept = row.get(cue_col, '') if cue_col else ''
            raw_feature = row.get(raw_feature_col, '') if raw_feature_col else ''
            translated = row.get(translated_col, '') if translated_col else ''
            if not concept or not (raw_feature or translated):
                continue
            rows.append({
                'concept': concept,
                'alias': concept,
                'raw_feature': raw_feature,
                'translated_feature': translated or raw_feature,
                'frequency_feature': row.get(freq_feature_col) if freq_feature_col else None,
                'frequency_translated': row.get(freq_translated_col) if freq_translated_col else None,
                'normalized_feature': row.get(norm_feature_col) if norm_feature_col else None,
                'normalized_translated': row.get(norm_translated_col) if norm_translated_col else None,
                'n': row.get(n_col) if n_col else None,
                'where': row.get(where_col) if where_col else None,
            })
        return rows

    @staticmethod
    def _pick_col(field_map: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in field_map:
                return field_map[candidate]
        return None

    
    
    

    def match_concept(self, phrase: str) -> ConceptMatch:
        if not phrase:
            return ConceptMatch(None, None, 0.0)
        normalized = self.normalize_phrase(phrase)
        if not normalized:
            return ConceptMatch(None, None, 0.0)
        if normalized in self.alias_index:
            return ConceptMatch(self.alias_index[normalized], normalized, 1.0)

        tokens = normalized.split()
        candidates = []
        if len(tokens) >= 2:
            candidates.append(' '.join(tokens[-2:]))
        if tokens:
            candidates.extend([tokens[-1], tokens[0]])
        generic_modifiers = {'cardboard', 'tin', 'metal', 'wooden', 'plastic', 'paper'}
        filtered = [tok for tok in tokens if tok not in generic_modifiers]
        if filtered:
            candidates.append(filtered[-1])
            if len(filtered) >= 2:
                candidates.append(' '.join(filtered[-2:]))

        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            if candidate in self.alias_index:
                return ConceptMatch(self.alias_index[candidate], candidate, 0.8)
        return ConceptMatch(None, None, 0.0)

    def get_concept_entry(self, concept_or_phrase: str) -> Dict[str, object]:
        match = self.match_concept(concept_or_phrase)
        if not match.concept:
            return {}
        return dict(self.concept_db.get(match.concept, {}))

    def get_concept_features(self, concept_or_phrase: str) -> Dict[str, float]:
        entry = self.get_concept_entry(concept_or_phrase)
        return dict(entry.get('features', {}))

    def get_feature_strength(self, concept_or_phrase: str, predicate: str) -> float:
        return float(self.get_concept_features(concept_or_phrase).get(predicate, 0.0))

    def get_affordance_strength(self, concept_or_phrase: str, affordance: str) -> float:
        features = self.get_concept_features(concept_or_phrase)
        predicates = UUT_AFFORDANCE_MAP.get(affordance, set())
        return max((float(features.get(predicate, 0.0)) for predicate in predicates), default=0.0)

    def derive_affordance_profile(self, concept_or_phrase: str) -> Dict[str, Dict[str, float]]:
        features = self.get_concept_features(concept_or_phrase)
        if not features:
            return {'base_properties': {}, 'negative_properties': {}}

        base_properties = {}
        for affordance, predicates in UUT_AFFORDANCE_MAP.items():
            strength = max((float(features.get(predicate, 0.0)) for predicate in predicates), default=0.0)
            if strength > 0.0:
                base_properties[affordance] = round(strength, 4)

        negative_properties: Dict[str, float] = {}
        paper_strength = max(float(features.get('material:paper', 0.0)), float(features.get('texture:soft', 0.0)))
        metal_strength = float(features.get('material:metal', 0.0))
        hollow_strength = float(features.get('shape:hollow', 0.0))
        light_strength = float(features.get('weight:light', 0.0))

        if paper_strength >= 0.05:
            negative_properties['water_resistant'] = round(max(negative_properties.get('water_resistant', 0.0), min(1.0, 0.55 + paper_strength)), 4)
            negative_properties['pressure_resistant'] = round(max(negative_properties.get('pressure_resistant', 0.0), min(1.0, 0.50 + paper_strength)), 4)
            negative_properties['heat_resistant'] = round(max(negative_properties.get('heat_resistant', 0.0), min(1.0, 0.55 + paper_strength)), 4)
        if hollow_strength >= 0.05 and light_strength >= 0.05:
            negative_properties['load_bearing'] = round(max(negative_properties.get('load_bearing', 0.0), min(1.0, 0.35 + hollow_strength + 0.5 * light_strength)), 4)
        if metal_strength >= 0.05:
            negative_properties['shapeable_surface'] = round(max(negative_properties.get('shapeable_surface', 0.0), min(1.0, 0.35 + metal_strength)), 4)
            negative_properties['writable_surface'] = round(max(negative_properties.get('writable_surface', 0.0), min(1.0, 0.30 + 0.8 * metal_strength)), 4)

        return {
            'base_properties': base_properties,
            'negative_properties': negative_properties,
        }

    def iter_concepts(self) -> Iterable[Tuple[str, Dict[str, object]]]:
        return self.concept_db.items()

    def list_concepts(self) -> List[str]:
        return sorted(self.concept_db.keys())

    def explain_predicate_hits(
        self,
        concept_or_phrase: str,
        positive_predicates: Sequence[str],
        negative_predicates: Sequence[str],
    ) -> Dict[str, object]:
        match = self.match_concept(concept_or_phrase)
        if not match.concept:
            return {
                'concept': None,
                'match_confidence': 0.0,
                'positive_hits': {},
                'negative_hits': {},
            }
        entry = self.concept_db.get(match.concept, {})
        features = entry.get('features', {})
        meta = entry.get('feature_meta', {})
        positive_hits = {}
        negative_hits = {}
        for predicate in positive_predicates:
            if predicate in features:
                positive_hits[predicate] = {
                    'weight': features[predicate],
                    'proportion': meta.get(predicate, {}).get('max_proportion'),
                    'translated_variants': entry.get('translated_features', {}).get(predicate),
                }
        for predicate in negative_predicates:
            if predicate in features:
                negative_hits[predicate] = {
                    'weight': features[predicate],
                    'proportion': meta.get(predicate, {}).get('max_proportion'),
                    'translated_variants': entry.get('translated_features', {}).get(predicate),
                }
        return {
            'concept': match.concept,
            'matched_alias': match.matched_alias,
            'match_confidence': match.confidence,
            'positive_hits': positive_hits,
            'negative_hits': negative_hits,
        }

