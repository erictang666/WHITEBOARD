


from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from swow_graph import SWOWGraph
from word_norms2_norms import WordNorms2Norms


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class WordNetAnalyzer:
    """WordNet-first, word-norms2/SWOW-fallback semantic category analyzer."""

    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "to", "of", "for", "with",
        "in", "on", "at", "by", "is", "are", "it", "this", "that", "as",
        "be", "been", "being", "was", "were", "from", "into", "up", "down",
        "over", "under", "through", "their", "there", "would", "could", "should",
        "can", "will", "just", "very", "all", "some", "any", "each", "every",
        "thing", "something", "someone", "somebody", "anything", "everything",
    }

    LEXNAME_LABELS = {
        'noun.Tops': 'Top-level Entity',
        'noun.act': 'Act/Action',
        'noun.animal': 'Animal',
        'noun.artifact': 'Artifact/Man-made',
        'noun.attribute': 'Attribute/Quality',
        'noun.body': 'Body Part',
        'noun.cognition': 'Cognition/Knowledge',
        'noun.communication': 'Communication',
        'noun.event': 'Event',
        'noun.feeling': 'Feeling/Emotion',
        'noun.food': 'Food',
        'noun.group': 'Group/Collection',
        'noun.location': 'Location/Place',
        'noun.motive': 'Motive',
        'noun.object': 'Natural Object',
        'noun.person': 'Person',
        'noun.phenomenon': 'Phenomenon',
        'noun.plant': 'Plant',
        'noun.possession': 'Possession',
        'noun.process': 'Process',
        'noun.quantity': 'Quantity/Amount',
        'noun.relation': 'Relation',
        'noun.shape': 'Shape',
        'noun.state': 'State/Condition',
        'noun.substance': 'Substance/Material',
        'noun.time': 'Time',
    }

    FALLBACK_CATEGORY_RULES = [
        {
            "label": "Artifact/Man-made",
            "predicates": {
                "container:container", "form:foldable", "surface:writable",
                "strength:load_bearing", "conductivity:conductive",
                "surface:reflective", "edge:sharp", "part:handle",
                "affordance:hangable",
            },
            "keywords": {
                "box", "carton", "can", "tin", "suitcase", "wallet", "ladder",
                "tool", "hammer", "wrench", "chair", "table", "book", "ticket",
                "card", "map", "poster", "notebook", "label", "window", "helmet",
                "armor", "bridge", "platform", "machine", "device", "phone", "bag",
                "bottle", "sign", "mirror", "antenna", "speaker", "drum", "frame",
            },
        },
        {
            "label": "Animal",
            "predicates": set(),
            "keywords": {
                "animal", "dog", "cat", "bird", "fish", "horse", "cow", "bear",
                "mouse", "elephant", "lion", "tiger", "wolf", "insect", "pet",
            },
        },
        {
            "label": "Plant",
            "predicates": set(),
            "keywords": {
                "plant", "tree", "flower", "grass", "leaf", "bush", "forest",
                "seed", "fruit tree", "vine", "rose", "garden",
            },
        },
        {
            "label": "Food",
            "predicates": {"edible:true"},
            "keywords": {
                "food", "fruit", "vegetable", "apple", "cherry", "strawberry",
                "tomato", "pepper", "bread", "cake", "meat", "meal", "snack",
            },
        },
        {
            "label": "Person/Social",
            "predicates": set(),
            "keywords": {
                "person", "people", "owner", "friend", "teacher", "worker", "child",
                "kid", "tourist", "police", "firefighter", "artist", "customer",
                "family", "crowd", "identity", "reputation", "trust", "honest",
            },
        },
        {
            "label": "Body Part",
            "predicates": set(),
            "keywords": {
                "body", "hand", "arm", "leg", "foot", "feet", "toe", "eye",
                "face", "head", "mouth", "skin", "finger", "shoe", "sock",
            },
        },
        {
            "label": "Location/Place",
            "predicates": set(),
            "keywords": {
                "place", "city", "street", "road", "room", "building", "airport",
                "hotel", "shop", "roof", "tower", "factory", "sidewalk", "earth",
                "sky", "beach", "park", "garden",
            },
        },
        {
            "label": "Substance/Material",
            "predicates": {
                "material:metal", "material:paper", "resistance:water",
                "resistance:heat", "resistance:pressure",
            },
            "keywords": {
                "metal", "tin", "steel", "iron", "paper", "cardboard", "glass",
                "wood", "plastic", "water", "air", "sand", "stone", "rock",
            },
        },
        {
            "label": "Event/Process",
            "predicates": set(),
            "keywords": {
                "event", "process", "journey", "travel", "storm", "rain", "crash",
                "accident", "construction", "maintenance", "repair", "security",
                "tracking", "tourism", "industry", "payment", "sale", "return",
            },
        },
        {
            "label": "Communication/Symbol",
            "predicates": {"surface:writable"},
            "keywords": {
                "message", "music", "song", "sign", "poster", "map", "label",
                "advertising", "language", "symbol", "identity", "braille", "festival",
            },
        },
        {
            "label": "Natural Phenomenon",
            "predicates": set(),
            "keywords": {
                "cloud", "sky", "storm", "rain", "thunder", "fire", "wind",
                "weather", "earth", "sun", "moon", "lightning", "drought",
            },
        },
        {
            "label": "Attribute/Shape",
            "predicates": {
                "shape:round", "shape:flat", "texture:hard", "texture:soft",
                "sound:loud", "sound:noise", "color:red",
            },
            "keywords": {
                "round", "flat", "hard", "soft", "loud", "noise", "red", "shape",
                "texture", "sound", "color", "quiet", "sharp",
            },
        },
        {
            "label": "Abstract/State",
            "predicates": set(),
            "keywords": {
                "idea", "thought", "memory", "dream", "rule", "law", "state",
                "condition", "emotion", "feeling", "love", "fear", "unity", "freedom",
                "value", "future", "past", "time",
            },
        },
    ]

    FALLBACK_ANTONYMS = {
        "round": ["flat", "square"],
        "hard": ["soft", "flexible"],
        "loud": ["quiet", "silent"],
        "noise": ["silence"],
        "red": ["colorless"],
        "metal": ["paper"],
        "paper": ["metal"],
        "writable": ["unwritable"],
        "waterproof": ["porous"],
    }

    V_MAX = 82192

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.available = False
        self.analysis_source = "disabled"
        self.wn = None
        self.nltk = None
        self.lemmatizer = None
        self.word_norms2 = WordNorms2Norms(data_dir=data_dir)
        self.swow = SWOWGraph(data_dir)
        self.num_categories = len(self.LEXNAME_LABELS)
        self._descendant_count_cache: Dict[str, int] = {}
        self._ic_cache: Dict[str, float] = {}

        try:
            import nltk
            from nltk.corpus import wordnet as wn
            from nltk.stem import WordNetLemmatizer

            
            _ = wn.synsets("test", pos=wn.NOUN)
            self.nltk = nltk
            self.wn = wn
            self.lemmatizer = WordNetLemmatizer()
            self.available = True
            self.analysis_source = "wordnet"
            self.num_categories = len(self.LEXNAME_LABELS)
            print("[WordNetAnalyzer] Initialized successfully (WordNet)")
            print(f"[WordNetAnalyzer] {self.num_categories} ontological noun categories available")
            return
        except Exception as exc:
            if self.word_norms2.available or self.swow.available:
                self.available = True
                self.analysis_source = "word_norms2_swow_fallback"
                self.num_categories = len(self.FALLBACK_CATEGORY_RULES)
                print(
                    "[WordNetAnalyzer] WordNet unavailable; using word-norms2/SWOW fallback "
                    f"({self.num_categories} semantic categories)."
                )
            else:
                print(f"[WordNetAnalyzer] No WordNet and no fallback resources available: {exc}")

    
    
    

    def lemmatize_token(self, token: str, pos: str = 'n') -> str:
        token = (token or "").strip().lower()
        if not token:
            return token

        if self.lemmatizer is not None:
            try:
                lemma = self.lemmatizer.lemmatize(token, pos=pos)
                if lemma:
                    return lemma
            except Exception:
                pass

        if token.endswith('ies') and len(token) > 4:
            return token[:-3] + 'y'
        if token.endswith(('ches', 'shes', 'xes', 'zes')) and len(token) > 4:
            return token[:-2]
        if token.endswith('es') and len(token) > 4 and not token.endswith(('ses', 'xes', 'zes')):
            return token[:-2]
        if token.endswith('s') and len(token) > 3 and not token.endswith('ss'):
            return token[:-1]
        return token

    def normalize_phrase(self, text: str) -> str:
        tokens = [self.lemmatize_token(tok) for tok in re.findall(r'[A-Za-z]+', (text or '').lower())]
        tokens = [tok for tok in tokens if tok and tok not in {'a', 'an', 'the', 'of', 'for', 'to'}]
        return ' '.join(tokens).strip()

    def _extract_content_tokens(self, text: str) -> List[str]:
        tokens = [self.lemmatize_token(tok) for tok in re.findall(r'[A-Za-z]+', (text or '').lower())]
        return [tok for tok in tokens if tok and tok not in self.STOP_WORDS]

    def get_antonyms(self, word: str) -> List[str]:
        lemma = self.lemmatize_token(word)
        if self.analysis_source == "wordnet" and self.wn is not None:
            antonyms = set()
            for synset in self.wn.synsets(lemma):
                for item in synset.lemmas():
                    for antonym in item.antonyms():
                        antonyms.add(antonym.name().replace('_', ' '))
            if antonyms:
                return sorted(antonyms)
        return sorted(self.FALLBACK_ANTONYMS.get(lemma, []))

    def get_source_label(self) -> str:
        return self.analysis_source

    
    
    

    def extract_nouns(self, text: str) -> List[str]:
        if self.analysis_source != "wordnet" or self.wn is None:
            return self._fallback_extract_concepts(text)

        nouns = []
        seen = set()
        for token in self._extract_content_tokens(text):
            lemma = self.lemmatize_token(token)
            if lemma in seen:
                continue
            if self.wn.synsets(lemma, pos=self.wn.NOUN):
                seen.add(lemma)
                nouns.append(lemma)
        return nouns

    def _get_best_noun_synset(self, word: str):
        if self.wn is None:
            return None
        synsets = self.wn.synsets(word, pos=self.wn.NOUN)
        return synsets[0] if synsets else None

    def _get_top_category(self, synset) -> str:
        if synset is None:
            return 'Unknown'
        return self.LEXNAME_LABELS.get(synset.lexname(), synset.lexname())

    def _count_descendants(self, synset) -> int:
        if synset is None:
            return 0
        name = synset.name()
        if name in self._descendant_count_cache:
            return self._descendant_count_cache[name]

        visited = {name}
        queue = list(synset.hyponyms())
        while queue:
            current = queue.pop(0)
            cname = current.name()
            if cname not in visited:
                visited.add(cname)
                queue.extend(current.hyponyms())
        count = len(visited)
        self._descendant_count_cache[name] = count
        return count

    def compute_information_content(self, synset) -> float:
        if synset is None:
            return 0.0
        name = synset.name()
        if name in self._ic_cache:
            return self._ic_cache[name]
        descendant_count = self._count_descendants(synset)
        if descendant_count <= 1:
            ic = 1.0
        else:
            ic = 1.0 - math.log(descendant_count) / math.log(self.V_MAX)
        ic = _clip(ic)
        self._ic_cache[name] = ic
        return ic

    def compute_similarity_wup(self, synset1, synset2) -> Optional[float]:
        if synset1 is None or synset2 is None:
            return None
        if synset1 == synset2:
            return 1.0
        return synset1.wup_similarity(synset2)

    
    
    

    def _fallback_extract_concepts(self, text: str) -> List[str]:
        normalized = self.normalize_phrase(text)
        tokens = self._extract_content_tokens(text)
        candidates: List[str] = []
        seen = set()

        def add_candidate(value: Optional[str]):
            value = self.normalize_phrase(value or "")
            if not value or value in seen:
                return
            seen.add(value)
            candidates.append(value)

        if normalized:
            add_candidate(normalized)

        for n in (3, 2, 1):
            for start in range(0, max(0, len(tokens) - n + 1)):
                phrase = ' '.join(tokens[start:start + n])
                if self.word_norms2.available:
                    match = self.word_norms2.match_concept(phrase)
                    if match and match.concept:
                        add_candidate(match.concept)
                        continue
                if n == 1:
                    token = tokens[start]
                    if self.swow.available and token in self.swow.associations:
                        add_candidate(token)
                    else:
                        add_candidate(token)

        return candidates[:6]

    def _get_fallback_entry(self, concept: str) -> Dict[str, object]:
        if not self.word_norms2.available:
            return {}
        return self.word_norms2.get_concept_entry(concept)

    def _categorize_fallback_concept(self, concept: str) -> Tuple[str, Dict[str, float]]:
        entry = self._get_fallback_entry(concept)
        features = dict(entry.get('features', {}))
        translated = entry.get('translated_features', {}) or {}
        raw = entry.get('raw_features', {}) or {}

        lexical_tokens = set(re.findall(r"[a-zA-Z]+", concept.lower()))
        for mapping in (translated, raw):
            for values in mapping.values():
                for value in values or []:
                    lexical_tokens.update(re.findall(r"[a-zA-Z]+", str(value).lower()))

        scores: Dict[str, float] = {}
        for rule in self.FALLBACK_CATEGORY_RULES:
            label = rule["label"]
            score = 0.0
            for predicate in rule["predicates"]:
                score += float(features.get(predicate, 0.0))
            keyword_hits = sum(1 for keyword in rule["keywords"] if keyword in lexical_tokens or keyword in concept)
            score += min(0.9, 0.2 * keyword_hits)
            if score > 0.0:
                scores[label] = score

        if not scores:
            return "Unknown", features
        return max(scores.items(), key=lambda item: item[1])[0], features

    def _fallback_specificity(self, concept: str, features: Dict[str, float]) -> float:
        if features:
            weights = sorted((float(value) for value in features.values()), reverse=True)
            top_mean = sum(weights[:3]) / min(3, len(weights))
            richness = min(1.0, len(features) / 8.0)
            multiword_bonus = 1.0 if len(concept.split()) > 1 else 0.75
            return _clip(0.45 * top_mean + 0.35 * richness + 0.20 * multiword_bonus)
        if concept:
            return 0.25 if len(concept.split()) == 1 else 0.35
        return 0.0

    @staticmethod
    def _sparse_cosine_distance(features_a: Dict[str, float], features_b: Dict[str, float]) -> Optional[float]:
        if not features_a or not features_b:
            return None
        keys = sorted(set(features_a) | set(features_b))
        dot = sum(float(features_a.get(key, 0.0)) * float(features_b.get(key, 0.0)) for key in keys)
        norm_a = math.sqrt(sum(float(features_a.get(key, 0.0)) ** 2 for key in keys))
        norm_b = math.sqrt(sum(float(features_b.get(key, 0.0)) ** 2 for key in keys))
        if norm_a <= 0.0 or norm_b <= 0.0:
            return None
        similarity = dot / (norm_a * norm_b)
        return _clip(1.0 - similarity)

    def _fallback_semantic_distance(self, analysis_a: Dict[str, object], analysis_b: Dict[str, object]) -> float:
        features_a = analysis_a.get('representative_features') or {}
        features_b = analysis_b.get('representative_features') or {}
        primary_a = analysis_a.get('primary_category') or 'Unknown'
        primary_b = analysis_b.get('primary_category') or 'Unknown'

        feature_distance = self._sparse_cosine_distance(features_a, features_b)
        if feature_distance is not None:
            distance = feature_distance
        else:
            rep_a = analysis_a.get('representative_concept') or ''
            rep_b = analysis_b.get('representative_concept') or ''
            assoc = 0.0
            if self.swow.available and rep_a and rep_b:
                assoc = max(self.swow.get_strength(rep_a, rep_b), self.swow.get_strength(rep_b, rep_a))
            if assoc > 0.0:
                distance = max(0.0, 1.0 - min(1.0, assoc / 0.10))
            else:
                distance = 0.35

        if primary_a != 'Unknown' and primary_b != 'Unknown':
            if primary_a != primary_b:
                distance = max(distance, 0.60)
            else:
                distance = min(distance, 0.45)
        return round(_clip(distance), 4)

    
    
    

    def analyze_idea(self, idea_text: str) -> Dict[str, object]:
        if self.analysis_source == "wordnet" and self.wn is not None:
            nouns = self.extract_nouns(idea_text)
            synset_pairs = []
            categories = set()
            ic_values = []
            best_synset = None
            best_ic = -1.0

            for noun in nouns:
                synset = self._get_best_noun_synset(noun)
                if synset is None:
                    continue
                synset_pairs.append((noun, synset))
                category = self._get_top_category(synset)
                categories.add(category)
                ic = self.compute_information_content(synset)
                ic_values.append(ic)
                if ic > best_ic:
                    best_ic = ic
                    best_synset = synset

            avg_ic = sum(ic_values) / len(ic_values) if ic_values else 0.0
            primary_category = self._get_top_category(best_synset) if best_synset is not None else 'Unknown'
            return {
                'nouns': nouns,
                'synsets': synset_pairs,
                'categories': categories,
                'avg_ic': round(avg_ic, 4),
                'representative_synset': best_synset,
                'representative_concept': nouns[0] if nouns else None,
                'representative_features': {},
                'primary_category': primary_category,
            }

        concepts = self._fallback_extract_concepts(idea_text)
        categories = set()
        specificities = []
        best_concept = None
        best_features: Dict[str, float] = {}
        best_category = 'Unknown'
        best_specificity = -1.0

        for concept in concepts:
            category, features = self._categorize_fallback_concept(concept)
            specificity = self._fallback_specificity(concept, features)
            if category != 'Unknown':
                categories.add(category)
            specificities.append(specificity)
            if specificity > best_specificity:
                best_specificity = specificity
                best_concept = concept
                best_features = features
                best_category = category

        avg_ic = sum(specificities) / len(specificities) if specificities else 0.0
        return {
            'nouns': concepts,
            'synsets': [],
            'categories': categories,
            'avg_ic': round(avg_ic, 4),
            'representative_synset': None,
            'representative_concept': best_concept,
            'representative_features': best_features,
            'primary_category': best_category,
        }

    
    
    

    def calculate_ontological_flexibility(self, ideas: Sequence[str], target_concept: Optional[str] = None):
        if not self.available or not ideas:
            return self._empty_result()

        analyses = [self.analyze_idea(idea) for idea in ideas]
        all_categories = set()
        primary_categories = []
        info_values = []
        covered = 0

        for analysis in analyses:
            valid_categories = analysis['categories'] - {'Unknown'}
            all_categories.update(valid_categories)
            primary_categories.append(analysis['primary_category'])
            info_values.append(float(analysis['avg_ic']))
            if analysis.get('representative_synset') is not None or analysis.get('representative_concept'):
                covered += 1

        switches = 0
        for idx in range(1, len(primary_categories)):
            prev_cat = primary_categories[idx - 1]
            curr_cat = primary_categories[idx]
            if prev_cat != curr_cat and prev_cat != 'Unknown' and curr_cat != 'Unknown':
                switches += 1

        distances = []
        for idx in range(1, len(analyses)):
            prev_analysis = analyses[idx - 1]
            curr_analysis = analyses[idx]
            if self.analysis_source == "wordnet":
                prev_synset = prev_analysis.get('representative_synset')
                curr_synset = curr_analysis.get('representative_synset')
                if prev_synset is not None and curr_synset is not None and prev_synset != curr_synset:
                    similarity = self.compute_similarity_wup(prev_synset, curr_synset)
                    if similarity is not None:
                        distances.append(1.0 - similarity)
            else:
                distances.append(self._fallback_semantic_distance(prev_analysis, curr_analysis))

        avg_distance = sum(distances) / len(distances) if distances else 0.0
        valid_info = [value for value in info_values if value > 0.0]
        avg_information_content = sum(valid_info) / len(valid_info) if valid_info else 0.0

        category_counter = Counter(category for category in primary_categories if category != 'Unknown')
        total_categorized = sum(category_counter.values())
        entropy = 0.0
        if total_categorized > 0:
            for count in category_counter.values():
                probability = count / total_categorized
                if probability > 0:
                    entropy -= probability * math.log2(probability)
        max_entropy = math.log2(self.num_categories) if self.num_categories > 1 else 0.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        coverage = covered / len(ideas) if ideas else 0.0

        return {
            'unique_categories': len(all_categories),
            'category_switches': switches,
            'avg_pairwise_wn_distance': round(avg_distance, 4),
            'avg_information_content': round(avg_information_content, 4),
            'category_diversity_index': round(normalized_entropy, 4),
            'category_distribution': dict(category_counter.most_common()),
            'idea_categories': primary_categories,
            'wordnet_coverage': round(coverage, 4),
            'total_ideas_analyzed': len(ideas),
            'ideas_with_wordnet_nouns': covered,
            'analysis_source': self.analysis_source,
            'num_categories': self.num_categories,
        }

    def _empty_result(self):
        return {
            'unique_categories': 0,
            'category_switches': 0,
            'avg_pairwise_wn_distance': 0.0,
            'avg_information_content': 0.0,
            'category_diversity_index': 0.0,
            'category_distribution': {},
            'idea_categories': [],
            'wordnet_coverage': 0.0,
            'total_ideas_analyzed': 0,
            'ideas_with_wordnet_nouns': 0,
            'analysis_source': self.analysis_source,
            'num_categories': self.num_categories,
        }

    def get_idea_detail(self, idea_text: str):
        if not self.available:
            return {
                'wordnet_category': 'N/A',
                'wordnet_nouns': [],
                'wordnet_ic': 0.0,
            }

        analysis = self.analyze_idea(idea_text)
        return {
            'wordnet_category': analysis['primary_category'],
            'wordnet_nouns': analysis['nouns'][:5],
            'wordnet_ic': analysis['avg_ic'],
        }
