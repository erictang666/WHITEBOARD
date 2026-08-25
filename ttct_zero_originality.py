from __future__ import annotations

"""
TTCT-style zero-originality detection for the white-box benchmark.

Key design changes:
- zero-originality no longer consumes the flattened full response by default
- task-specific core extraction is used instead:
    * UUT -> idea title only
    * JST -> short consequence clause
    * Instances / PropConj -> head noun / canonical concept
- static baselines are organized as family banks, split into:
    * hard_zero: direct zero-originality families
    * broad_common: common-but-not-necessarily-zero families kept for audit/future damping
- dynamic matching consumes task-specific context (e.g. JST scenario_text), not the
  display label and not the whole instruction prompt.
"""

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from scorer_hyperparameters import get_scorer_hyperparameter

try:
    from word_norms2_norms import WordNorms2Norms
except ImportError:  
    WordNorms2Norms = None


DATA_DIR = Path(__file__).resolve().parent / "data"
ZERO_ORIG_STATIC_MATCH_PARAMS = get_scorer_hyperparameter(
    "ttct_zero_originality",
    "static_family_match_sigmoid",
    default={
        "midpoint": 0.50,
        "temperature": 0.20,
        "static_threshold": 0.50,
        "exact": 1.0,
        "token_exact": 0.92,
        "alias_token_subset": 0.84,
        "core_token_subset": 0.78,
        "substring": 0.68,
    },
)

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for", "with",
    "in", "on", "at", "by", "is", "are", "was", "were", "be", "being",
    "been", "it", "this", "that", "these", "those", "into", "from", "as",
    "would", "could", "should", "can", "may", "might", "will", "just",
    "very", "really", "kind", "sort", "type", "thing", "stuff", "some",
    "someone", "something", "using", "use", "used", "make", "makes", "made",
    "become", "becomes", "becoming", "because", "through", "their", "there",
    "every", "all", "own", "like", "quietly", "slowly", "people",
}

CORE_WORD_LIMITS = {
    "UUT": 6,
    "JST": 14,
    "Instances": 4,
    "PropConj": 5,
}

DYNAMIC_EXTRA_TOKEN_BUDGET = {
    "UUT": 2,
    "JST": 4,
    "Instances": 1,
    "PropConj": 2,
}

INSTANCE_TRAIT_CUES = {
    "ins_01": ["round", "hard", "ball", "stone", "marble", "sphere"],
    "ins_02": ["loud", "noise", "sound", "alarm", "bang", "siren", "horn"],
    "ins_03": ["red", "edible", "fruit", "food", "berry", "pepper", "tomato"],
    "ins_04": ["metal", "hard", "iron", "steel", "tool", "coin", "pan"],
    "ins_05": ["paper", "cardboard", "writable", "write", "label", "sign", "notebook"],
    "ins_06": ["transparent", "flexible", "clear", "bend", "film", "plastic"],
    "ins_07": ["soft", "waterproof", "water", "rubber", "neoprene", "cushion"],
    "ins_08": ["tiny", "small", "valuable", "precious", "diamond", "coin", "chip"],
    "ins_09": ["hollow", "lightweight", "empty", "shell", "tube", "balloon"],
    "ins_10": ["sticky", "useful", "adhesive", "tape", "glue", "label"],
    "ins_11": ["cold", "edible", "frozen", "food", "ice", "cream"],
    "ins_12": ["sharp", "portable", "carry", "edge", "knife", "blade", "needle"],
    "ins_13": ["bright", "fragile", "light", "glass", "bulb", "break"],
    "ins_14": ["heavy", "movable", "move", "roll", "piano", "safe", "machine"],
    "ins_15": ["natural", "patterned", "stripe", "spot", "shell", "leaf", "skin"],
    "ins_16": ["flat", "reflective", "mirror", "shine", "foil", "screen"],
    "ins_17": ["elastic", "colorful", "stretch", "rubber", "balloon", "band"],
    "ins_18": ["scented", "disposable", "smell", "single-use", "tissue", "wipe"],
    "ins_19": ["mechanical", "small", "gear", "device", "watch", "motor", "toy"],
    "ins_20": ["silent", "protective", "quiet", "shield", "helmet", "case", "pad"],
}

INSTANCE_TRAIT_MODIFIERS = {
    "ins_01": {"steel", "iron", "metal", "crystal"},
    "ins_02": set(),
    "ins_03": {"red", "blood", "orange", "scarlet", "crimson"},
    "ins_04": {"metal", "iron", "steel", "cast", "cast-iron", "tin"},
    "ins_05": {"paper", "cardboard"},
    "ins_06": {"transparent", "clear", "flexible", "plastic", "cellophane"},
    "ins_07": {"soft", "waterproof", "rubber", "silicone", "neoprene"},
    "ins_08": {"tiny", "small", "valuable", "precious", "rare", "gold"},
    "ins_09": {"hollow", "lightweight", "empty", "paper", "plastic"},
    "ins_10": {"sticky", "adhesive", "useful", "duct", "double-sided"},
    "ins_11": {"cold", "frozen", "ice", "iced"},
    "ins_12": {"sharp", "portable", "pocket", "safety", "sewing"},
    "ins_13": {"bright", "fragile", "glass", "neon", "thin"},
    "ins_14": {"heavy", "movable", "rolling"},
    "ins_15": {"natural", "patterned", "striped", "spotted"},
    "ins_16": {"flat", "reflective", "polished", "mirror", "foil"},
    "ins_17": {"elastic", "colorful", "stretch", "rubber"},
    "ins_18": {"scented", "disposable", "perfume", "single-use"},
    "ins_19": {"mechanical", "small", "wind-up", "gear"},
    "ins_20": {"silent", "protective", "quiet", "safety"},
}

COMPOUND_HEAD_NOUNS = {
    "teller", "pan", "ball", "orange", "pepper", "engine", "card",
    "box", "house", "bed", "horn", "wrap", "film", "tape", "sheet",
    "tube", "bottle", "band", "note", "stick", "blade", "knife",
    "needle", "bulb", "light", "coin", "ring", "case", "sleeve",
    "screen", "protector", "toy", "pencil", "guard", "pad", "fan",
    "motor", "machine", "shell", "wing", "skin", "tray",
}

ZERO_ORIGINALITY_STATIC_BANK: Dict[str, Dict[str, object]] = {
    "uut_01": {
        "task_type": "UUT",
        "match_field": "idea_title",
        "hard_zero_families": [
            {"family": "storage_container", "aliases": [
                "storage box", "store things", "hold items", "keep items", "organizer",
                "hold clothes", "clothes storage", "put things inside", "box for storage",
            ]},
            {"family": "moving_box", "aliases": [
                "moving box", "packing box", "carry belongings", "packing container",
                "moving container", "box for moving",
            ]},
            {"family": "trash_recycle", "aliases": [
                "recycle bin", "trash box", "garbage container", "trash container",
                "recycle box", "throw it away", "put it in the trash",
            ]},
            {"family": "pet_shelter", "aliases": [
                "cat house", "dog house", "pet bed", "small animal home", "pet shelter",
                "animal house", "make a house for a pet",
            ]},
            {"family": "burn_kindling", "aliases": [
                "fire starter", "kindling", "burn for heat", "burn it", "fuel for fire",
            ]},
        ],
        "broad_common_families": [
            {"family": "planter_seed_starter", "aliases": ["planter", "seed starter", "plant pot"]},
            {"family": "cardboard_sign", "aliases": ["cardboard sign", "protest sign", "sign board"]},
            {"family": "toy_box", "aliases": ["toy box", "play box"]},
        ],
    },
    "uut_02": {
        "task_type": "UUT",
        "match_field": "idea_title",
        "hard_zero_families": [
            {"family": "food_liquid_container", "aliases": [
                "hold food", "store liquids", "food container", "liquid container", "drink holder",
            ]},
            {"family": "pencil_pen_holder", "aliases": [
                "pencil holder", "pen holder", "put pens in it", "desk organizer", "utensil holder",
            ]},
            {"family": "recycle_scrap_metal", "aliases": [
                "recycle them", "recycle can", "scrap metal", "melt them down", "metal recycling",
            ]},
            {"family": "string_telephone", "aliases": [
                "string telephone", "telephone game", "can phone", "telephone with string",
            ]},
            {"family": "target_practice", "aliases": [
                "kick the can", "target practice", "shooting target", "throwing target",
            ]},
            {"family": "coin_bank", "aliases": [
                "coin bank", "piggy bank", "save coins", "change holder",
            ]},
        ],
        "broad_common_families": [
            {"family": "planter", "aliases": ["planter", "plant pot", "flower pot"]},
            {"family": "screw_nail_holder", "aliases": ["screw holder", "nail holder", "parts can"]},
            {"family": "candle_lantern", "aliases": ["candle holder", "lantern", "tea light holder"]},
        ],
    },
    "uut_03": {
        "task_type": "UUT",
        "match_field": "idea_title",
        "hard_zero_families": [
            {"family": "ordinary_luggage", "aliases": [
                "carry clothes", "carry belongings", "travel case", "ordinary luggage", "luggage",
                "travel bag", "suitcase for travel",
            ]},
            {"family": "storage_chest", "aliases": [
                "storage chest", "storage case", "store clothes", "keep belongings", "closet storage",
            ]},
            {"family": "lockbox_hidden_storage", "aliases": [
                "lockbox", "hidden storage", "secure storage", "keep valuables", "secret storage",
            ]},
            {"family": "pet_bed", "aliases": [
                "pet bed", "dog bed", "cat bed", "pet basket", "animal bed",
            ]},
        ],
        "broad_common_families": [
            {"family": "stool_seat", "aliases": ["stool", "seat", "chair", "bench"]},
            {"family": "side_table_display_case", "aliases": ["side table", "display case", "end table"]},
            {"family": "toy_chest", "aliases": ["toy chest", "toy storage"]},
            {"family": "drum_instrument_case", "aliases": ["drum", "instrument case", "sound box"]},
        ],
    },
    "uut_04": {
        "task_type": "UUT",
        "match_field": "idea_title",
        "hard_zero_families": [
            {"family": "hold_cash_cards_id", "aliases": [
                "hold cash", "hold cards", "hold id", "cash holder", "card holder", "wallet",
                "keep money", "money holder",
            ]},
            {"family": "receipt_business_card_holder", "aliases": [
                "receipt holder", "business card holder", "card case", "receipt organizer",
            ]},
            {"family": "coin_pouch", "aliases": [
                "coin pouch", "change purse", "coin holder", "small money pouch",
            ]},
            {"family": "photo_note_holder", "aliases": [
                "photo holder", "note holder", "memory holder", "picture holder",
            ]},
            {"family": "gift_card_holder", "aliases": [
                "gift card holder", "gift card case", "voucher holder",
            ]},
        ],
        "broad_common_families": [
            {"family": "sim_sd_key_holder", "aliases": ["sim holder", "sd card holder", "key holder"]},
            {"family": "keepsake_pouch", "aliases": ["keepsake pouch", "memento holder"]},
            {"family": "decoy_wallet", "aliases": ["decoy wallet", "fake wallet"]},
        ],
    },
    "uut_05": {
        "task_type": "UUT",
        "match_field": "idea_title",
        "hard_zero_families": [
            {"family": "ordinary_climbing_reach_high_places", "aliases": [
                "reach high places", "climb high", "ordinary climbing", "get to high shelf", "ladder use",
            ]},
            {"family": "shelf_bookshelf", "aliases": [
                "shelf", "bookshelf", "book rack", "shelving unit",
            ]},
            {"family": "clothes_towel_rack", "aliases": [
                "clothes rack", "towel rack", "hanger rack", "dry clothes rack",
            ]},
            {"family": "plant_stand", "aliases": [
                "plant stand", "flower stand", "garden stand",
            ]},
            {"family": "drying_rack", "aliases": [
                "drying rack", "air dry rack", "laundry rack",
            ]},
        ],
        "broad_common_families": [
            {"family": "decor_display_hanger", "aliases": ["decor hanger", "display hanger", "ornament stand"]},
            {"family": "loft_access", "aliases": ["loft access", "attic access"]},
            {"family": "light_or_sign_support", "aliases": ["light support", "sign support", "banner stand"]},
        ],
    },
    "jst_01": {
        "task_type": "JST",
        "match_field": "consequence_clause",
        "hard_zero_families": [
            {"family": "people_pull_clouds", "aliases": [
                "people pull clouds", "pull clouds", "drag clouds", "tug clouds", "move clouds by string",
            ]},
            {"family": "kids_climb_or_swing", "aliases": [
                "kids climb clouds", "climb the strings", "swing on cloud strings", "children climb them",
            ]},
            {"family": "aircraft_or_birds_get_tangled", "aliases": [
                "aircraft get tangled", "birds get tangled", "planes crash into strings", "helicopters get stuck",
            ]},
            {"family": "control_rain_or_shade", "aliases": [
                "make rain by pulling clouds", "control rain", "move shade", "drag clouds to dry places",
            ]},
        ],
        "broad_common_families": [
            {"family": "tourism_festival_economy", "aliases": ["tourism boom", "festival economy", "cloud tourism"]},
            {"family": "sky_harvest_decoration", "aliases": ["sky harvest", "cloud decoration", "hang decorations in sky"]},
        ],
    },
    "jst_02": {
        "task_type": "JST",
        "match_field": "consequence_clause",
        "hard_zero_families": [
            {"family": "identify_people_by_shoes_or_socks", "aliases": [
                "recognize people by shoes", "identify people by socks", "shoes become identity", "socks show identity",
            ]},
            {"family": "more_bumping_tripping_accidents", "aliases": [
                "more tripping accidents", "bump into each other", "car accidents increase", "navigation accidents",
            ]},
            {"family": "stepping_on_toes", "aliases": [
                "step on toes", "toes get stepped on", "toe injuries",
            ]},
            {"family": "shoe_sock_pedicure_industry_boom", "aliases": [
                "shoe industry booms", "sock fashion booms", "pedicure industry booms", "footwear business grows",
            ]},
        ],
        "broad_common_families": [
            {"family": "eye_contact_social_norm_changes", "aliases": ["eye contact changes", "social norms change", "people greet differently"]},
            {"family": "footwear_as_identity_signal", "aliases": ["footwear signals identity", "shoes become status signal"]},
        ],
    },
    "jst_03": {
        "task_type": "JST",
        "match_field": "consequence_clause",
        "hard_zero_families": [
            {"family": "easier_travel_no_need_to_carry_luggage", "aliases": [
                "travel easier", "no need to carry luggage", "bags follow owners", "people stop carrying bags",
                "luggage walks behind owner",
            ]},
            {"family": "theft_harder_or_lost_luggage_reduced", "aliases": [
                "theft harder", "lost luggage reduced", "bags easier to track", "harder to steal suitcases",
            ]},
            {"family": "sidewalks_airports_crowded_or_collisions", "aliases": [
                "airports get crowded", "sidewalk collisions", "luggage traffic jams", "crowded with following bags",
            ]},
            {"family": "suitcases_behave_like_pets", "aliases": [
                "suitcases act like pets", "luggage behaves like pets", "owners walk suitcases like dogs",
            ]},
        ],
        "broad_common_families": [
            {"family": "tracking_privacy_changes", "aliases": ["tracking changes", "privacy changes", "surveillance concerns"]},
            {"family": "hotel_airport_service_changes", "aliases": ["hotel service changes", "airport service changes", "new travel services"]},
        ],
    },
    "jst_04": {
        "task_type": "JST",
        "match_field": "consequence_clause",
        "hard_zero_families": [
            {"family": "ladders_become_taller", "aliases": [
                "ladders get taller", "ladders become taller", "more rungs appear", "ladder keeps growing",
            ]},
            {"family": "easier_to_reach_higher_places", "aliases": [
                "easier to reach higher places", "reach higher", "access taller places", "climb higher buildings",
            ]},
            {"family": "unstable_dangerous_falls", "aliases": [
                "dangerous falls", "unstable ladders", "ladder accidents increase", "ladders become unsafe",
            ]},
            {"family": "storage_space_ceiling_problems", "aliases": [
                "storage problems", "ceiling problems", "ladders no longer fit indoors", "space problems for ladders",
            ]},
            {"family": "people_use_ladders_repeatedly_to_make_them_grow", "aliases": [
                "people climb ladders to grow them", "grow ladders on purpose", "repeated climbing makes taller ladders",
            ]},
        ],
        "broad_common_families": [
            {"family": "construction_maintenance_changes", "aliases": ["construction changes", "maintenance work changes"]},
            {"family": "sports_games_challenges", "aliases": ["ladder games", "climbing challenges", "sports with ladders"]},
        ],
    },
    "jst_05": {
        "task_type": "JST",
        "match_field": "consequence_clause",
        "hard_zero_families": [
            {"family": "lost_wallets_return_automatically", "aliases": [
                "lost wallets return automatically", "wallets crawl back", "wallets return to owners", "wallet finds owner",
            ]},
            {"family": "theft_pickpocketing_becomes_harder", "aliases": [
                "theft harder", "pickpocketing harder", "harder to steal wallets", "crime reduced for wallets",
            ]},
            {"family": "people_worry_less_about_losing_wallets", "aliases": [
                "worry less about losing wallets", "less fear of losing wallet", "people relax about wallets",
            ]},
            {"family": "crawling_wallets_cause_public_scenes", "aliases": [
                "wallets cause public scenes", "crawling wallets on streets", "wallets move through crowds",
            ]},
        ],
        "broad_common_families": [
            {"family": "police_bank_shop_procedure_changes", "aliases": ["police procedures change", "bank procedure changes", "shop procedure changes"]},
            {"family": "honesty_trust_norm_changes", "aliases": ["trust norms change", "honesty norms change", "people seem more honest"]},
        ],
    },
    "ins_01": {
        "task_type": "Instances",
        "match_field": "canonical_concept",
        "hard_zero_families": [
            {"family": "ordinary_round_hard_objects", "aliases": [
                "bowling ball", "billiard ball", "marble", "baseball", "golf ball",
                "cannonball", "rock", "stone", "steel ball",
            ]},
        ],
        "broad_common_families": [
            {"family": "near_common_round_hard", "aliases": ["ball bearing", "crystal ball", "pebble"]},
        ],
    },
    "ins_02": {
        "task_type": "Instances",
        "match_field": "canonical_concept",
        "hard_zero_families": [
            {"family": "ordinary_loud_noise_sources", "aliases": [
                "siren", "alarm", "explosion", "bomb", "gunshot", "thunder",
                "fireworks", "jet engine", "speaker", "drum", "car horn", "scream", "shout",
            ]},
        ],
        "broad_common_families": [
            {"family": "near_common_loud_noise", "aliases": ["air horn", "train whistle", "megaphone"]},
        ],
    },
    "ins_03": {
        "task_type": "Instances",
        "match_field": "canonical_concept",
        "hard_zero_families": [
            {"family": "ordinary_red_edible_foods", "aliases": [
                "apple", "cherry", "strawberry", "tomato", "raspberry", "pomegranate", "red pepper",
            ]},
        ],
        "broad_common_families": [
            {"family": "near_common_red_edible", "aliases": ["watermelon", "radish", "beet", "cranberry", "blood orange"]},
        ],
    },
    "ins_04": {
        "task_type": "Instances",
        "match_field": "canonical_concept",
        "hard_zero_families": [
            {"family": "ordinary_metal_hard_objects", "aliases": [
                "hammer", "wrench", "screwdriver", "crowbar", "anvil", "coin", "chain",
                "frying pan", "nail", "key",
            ]},
        ],
        "broad_common_families": [
            {"family": "near_common_metal_hard", "aliases": ["spoon", "pipe", "lock", "metal ladder"]},
        ],
    },
    "ins_05": {
        "task_type": "Instances",
        "match_field": "canonical_concept",
        "hard_zero_families": [
            {"family": "ordinary_paper_cardboard_writable_objects", "aliases": [
                "notebook", "index card", "postcard", "paper sign", "cardboard sign",
                "paper map", "ticket", "label", "envelope", "notepad",
            ]},
        ],
        "broad_common_families": [
            {"family": "near_common_paper_cardboard_writable", "aliases": ["flashcard", "paper tag", "paper bag with writing", "cardboard box label"]},
        ],
    },
}


@lru_cache(maxsize=1)
def _get_word_norms2():
    if WordNorms2Norms is None:
        return None
    try:
        return WordNorms2Norms(data_dir=str(DATA_DIR))
    except Exception:
        return None


@lru_cache(maxsize=4)
def _load_json_resource(filename: str) -> Dict[str, object]:
    path = DATA_DIR / filename
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _get_wordnet_lemmatizer():
    try:
        from nltk.corpus import wordnet as wn
        from nltk.stem import WordNetLemmatizer

        _ = wn.synsets("test", pos=wn.NOUN)
        return WordNetLemmatizer()
    except Exception:
        return None


def _normalize_token(token: str) -> str:
    token = (token or "").lower().strip().replace("_", " ")
    token = re.sub(r"[^a-z\s-]+", " ", token)
    token = re.sub(r"\s+", " ", token).strip()
    lemmatizer = _get_wordnet_lemmatizer()
    if lemmatizer is not None and token:
        try:
            token = lemmatizer.lemmatize(token, pos="n")
        except Exception:
            pass
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith(("ches", "shes", "xes", "zes")) and len(token) > 4:
        token = token[:-2]
    elif token.endswith("es") and len(token) > 4 and not token.endswith("e"):
        token = token[:-2]
    elif token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is", "as")):
        token = token[:-1]
    return token


def _normalize_phrase(text: str) -> str:
    tokens = [_normalize_token(tok) for tok in re.findall(r"[a-zA-Z-]+", (text or "").lower())]
    tokens = [tok for tok in tokens if tok and tok not in {"a", "an", "the", "of", "for", "to"}]
    return " ".join(tokens).strip()


def _tokenize_content(text: str) -> List[str]:
    return [
        _normalize_token(tok)
        for tok in re.findall(r"[a-zA-Z-]+", (text or "").lower())
        if _normalize_token(tok) and _normalize_token(tok) not in STOP_WORDS
    ]


def _singularize(token: str) -> str:
    return _normalize_token(token)


def _clean_text_fragment(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^[\s\"'`\[\]{}()<>.,;:!?-]+", "", text)
    text = re.sub(r"[\s\"'`\[\]{}()<>.,;:!?-]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _infer_task_type(item_id: Optional[str], task_type: Optional[str] = None) -> Optional[str]:
    if task_type:
        return task_type
    if not item_id:
        return None
    prefix = item_id.split("_", 1)[0].lower()
    if prefix == "uut":
        return "UUT"
    if prefix == "jst":
        return "JST"
    if prefix == "ins":
        return "Instances"
    if prefix in {"pc", "propconj"}:
        return "PropConj"
    return None


def _split_uut_title_and_mechanism(text: str) -> Tuple[str, str]:
    cleaned = _clean_text_fragment(text)
    if not cleaned:
        return "", ""
    separators = [" because ", " so that ", " using ", " by ", ": "]
    lower = cleaned.lower()
    best_idx = None
    best_sep = None
    for sep in separators:
        idx = lower.find(sep)
        if idx <= 0:
            continue
        if best_idx is None or idx < best_idx:
            best_idx = idx
            best_sep = sep
    if best_idx is None or best_sep is None:
        return cleaned, ""
    title = _clean_text_fragment(cleaned[:best_idx])
    mechanism = _clean_text_fragment(cleaned[best_idx + len(best_sep):])
    if len(_tokenize_content(title)) > 10:
        return cleaned, ""
    return title, mechanism


def _build_parsed_item_from_text(task_type: Optional[str], response_text: str) -> Dict[str, Optional[str]]:
    text = _clean_text_fragment(response_text or "")
    parsed = {
        "task_type": task_type,
        "raw_text": response_text,
        "idea_title": None,
        "mechanism": None,
        "consequence_clause": None,
        "noun_phrase": None,
        "propconj_item": None,
        "display_text": text,
    }
    if task_type == "UUT":
        title, mechanism = _split_uut_title_and_mechanism(text)
        parsed["idea_title"] = title or None
        parsed["mechanism"] = mechanism or None
    elif task_type == "JST":
        parsed["consequence_clause"] = text or None
    elif task_type == "Instances":
        parsed["noun_phrase"] = text or None
    elif task_type == "PropConj":
        parsed["noun_phrase"] = text or None
        parsed["propconj_item"] = text or None
    return parsed


def _ensure_parsed_item(response_text=None, parsed_item=None, task_type=None):
    if isinstance(parsed_item, dict):
        result = dict(parsed_item)
        if task_type and not result.get("task_type"):
            result["task_type"] = task_type
        return result
    if isinstance(response_text, dict):
        result = dict(response_text)
        if task_type and not result.get("task_type"):
            result["task_type"] = task_type
        if not result.get("display_text"):
            display = (
                result.get("raw_text")
                or result.get("idea_title")
                or result.get("consequence_clause")
                or result.get("propconj_item")
                or result.get("noun_phrase")
            )
            result["display_text"] = _clean_text_fragment(str(display or ""))
        return result
    return _build_parsed_item_from_text(task_type, str(response_text or ""))


def _candidate_concept_spans(text: str) -> List[str]:
    normalized = _normalize_phrase(text)
    tokens = normalized.split()
    candidates = []
    seen = set()
    for span_len in range(len(tokens), 0, -1):
        for start in range(0, len(tokens) - span_len + 1):
            span = " ".join(tokens[start:start + span_len]).strip()
            if span and span not in seen:
                candidates.append(span)
                seen.add(span)
    return candidates


def _known_aliases_for_item(item_id: str) -> List[str]:
    entry = ZERO_ORIGINALITY_STATIC_BANK.get(item_id, {})
    aliases = []
    for group_name in ["hard_zero_families", "broad_common_families"]:
        for family in entry.get(group_name, []) or []:
            for alias in family.get("aliases") or []:
                alias_norm = _normalize_phrase(str(alias))
                if alias_norm:
                    aliases.append(alias_norm)
    return sorted(set(aliases), key=lambda value: (len(value.split()), len(value)), reverse=True)


def _match_known_alias_span(item_id: str, normalized_phrase: str) -> Optional[str]:
    for alias in _known_aliases_for_item(item_id):
        if alias == normalized_phrase:
            return alias
        if f" {alias} " in f" {normalized_phrase} ":
            return alias
    return None


def canonicalize_instances_concept(item_id: str, noun_phrase: str) -> str:
    phrase = _clean_text_fragment(noun_phrase or "")
    normalized = _normalize_phrase(phrase)
    if not normalized:
        return ""

    alias_span = _match_known_alias_span(item_id, normalized)
    if alias_span:
        return alias_span

    word_norms2 = _get_word_norms2()
    best_match = None
    best_signature = None
    original_modifiers = [tok for tok in normalized.split() if tok in INSTANCE_TRAIT_MODIFIERS.get(item_id, set())]
    if word_norms2 is not None and getattr(word_norms2, "available", False):
        for candidate in _candidate_concept_spans(normalized):
            match = word_norms2.match_concept(candidate)
            if not match or not match.concept:
                continue
            concept_norm = _normalize_phrase(match.concept)
            candidate_tokens = candidate.split()
            concept_tokens = concept_norm.split()
            candidate_modifiers = [tok for tok in candidate_tokens if tok in INSTANCE_TRAIT_MODIFIERS.get(item_id, set())]
            concept_modifiers = [tok for tok in concept_tokens if tok in INSTANCE_TRAIT_MODIFIERS.get(item_id, set())]
            if (
                len(concept_tokens) == 1 and
                concept_tokens[0] in INSTANCE_TRAIT_MODIFIERS.get(item_id, set())
            ):
                continue
            if (
                len(candidate_tokens) > 1 and
                len(concept_tokens) == 1 and
                concept_tokens[0] != candidate_tokens[-1]
            ):
                continue
            if len(candidate_tokens) == 1 and concept_norm != candidate:
                continue
            if len(candidate_tokens) == 1 and original_modifiers and candidate != normalized:
                continue
            if original_modifiers and not candidate_modifiers and len(candidate_tokens) < len(normalized.split()):
                continue
            if candidate_modifiers and not concept_modifiers and len(candidate_tokens) > len(concept_tokens):
                continue
            signature = (float(match.confidence), len(candidate.split()), len(concept_norm.split()))
            if best_signature is None or signature > best_signature:
                best_signature = signature
                best_match = concept_norm
        if best_match:
            return _normalize_phrase(best_match)

    tokens = _tokenize_content(normalized)
    if not tokens:
        return normalized

    head = _singularize(tokens[-1])
    modifiers = [tok for tok in tokens[:-1] if tok in INSTANCE_TRAIT_MODIFIERS.get(item_id, set())]
    compound_tail = None
    if len(tokens) >= 2:
        tail_head = _singularize(tokens[-1])
        tail_prev = _singularize(tokens[-2])
        if tail_head in COMPOUND_HEAD_NOUNS and tail_prev not in INSTANCE_TRAIT_MODIFIERS.get(item_id, set()):
            compound_tail = f"{tail_prev} {tail_head}".strip()
    if modifiers:
        kept = modifiers[-2:]
        if compound_tail is not None:
            return " ".join(kept + compound_tail.split()).strip()
        return " ".join(kept + [head]).strip()
    if compound_tail is not None:
        return compound_tail
    return head


def _extract_zero_orig_core(item_id: str, task_type: str, item: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    display_text = _clean_text_fragment(str(item.get("display_text") or item.get("raw_text") or ""))
    if task_type == "UUT":
        title = _clean_text_fragment(str(item.get("idea_title") or ""))
        if not title:
            title, mechanism = _split_uut_title_and_mechanism(display_text)
            if title and not item.get("mechanism"):
                item["mechanism"] = mechanism or None
        return {
            "match_field": "idea_title",
            "core_text": title or display_text,
            "core_norm": _normalize_phrase(title or display_text),
            "display_text": display_text,
        }
    if task_type == "JST":
        clause = _clean_text_fragment(str(item.get("consequence_clause") or display_text))
        return {
            "match_field": "consequence_clause",
            "core_text": clause,
            "core_norm": _normalize_phrase(clause),
            "display_text": display_text,
        }
    noun_phrase = _clean_text_fragment(str(item.get("propconj_item") or item.get("noun_phrase") or display_text))
    canonical = canonicalize_instances_concept(item_id, noun_phrase)
    return {
        "match_field": "canonical_concept" if task_type != "PropConj" else "propconj_canonical_item",
        "core_text": canonical,
        "core_norm": _normalize_phrase(canonical),
        "display_text": display_text,
    }


def extract_task_specific_core(item_id, response_text=None, parsed_item=None, task_type=None):
    """Public helper shared by zero-origin + common-answer-bank scoring.

    Returns the task-specific core form used for:
        - UUT: idea_title
        - JST: consequence_clause
        - Instances / PropConj: canonical concept
    """
    task_type = _infer_task_type(item_id, task_type)
    item = _ensure_parsed_item(response_text=response_text, parsed_item=parsed_item, task_type=task_type)
    core_info = _extract_zero_orig_core(item_id, task_type or "", item)
    core_info["parsed_item"] = item
    return core_info


def _clip01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _sigmoid_match_score(raw_score: float) -> float:
    raw_score = _clip01(raw_score)
    if raw_score <= 0.0:
        return 0.0
    midpoint = float(ZERO_ORIG_STATIC_MATCH_PARAMS.get("midpoint", 0.50) or 0.50)
    temperature = max(1e-6, float(ZERO_ORIG_STATIC_MATCH_PARAMS.get("temperature", 0.20) or 0.20))
    return _clip01(1.0 / (1.0 + math.exp(-(raw_score - midpoint) / temperature)))


def _family_alias_match_raw(core_norm: str, alias_norm: str, task_type: str) -> Tuple[bool, Optional[str]]:
    if not core_norm or not alias_norm:
        return False, None
    if core_norm == alias_norm:
        return True, "exact"

    core_tokens = set(_tokenize_content(core_norm))
    alias_tokens = set(_tokenize_content(alias_norm))
    if not core_tokens or not alias_tokens:
        return False, None

    if task_type in {"Instances", "PropConj"}:
        if core_norm == alias_norm:
            return True, "exact"
        if len(core_tokens.symmetric_difference(alias_tokens)) == 0:
            return True, "token_exact"
        return False, None

    extra_budget = 2 if task_type == "UUT" else 4
    if alias_tokens.issubset(core_tokens) and len(core_tokens - alias_tokens) <= extra_budget:
        return True, "alias_token_subset"
    if core_tokens.issubset(alias_tokens) and len(alias_tokens - core_tokens) <= 1:
        return True, "core_token_subset"
    if alias_norm in core_norm or core_norm in alias_norm:
        return True, "substring"
    return False, None


def _family_alias_match(core_norm: str, alias_norm: str, task_type: str) -> Tuple[bool, Optional[str]]:
    return _family_alias_match_raw(core_norm, alias_norm, task_type)


def _family_alias_match_score(core_norm: str, alias_norm: str, task_type: str) -> Tuple[float, Optional[str], float]:
    matched, mode = _family_alias_match_raw(core_norm, alias_norm, task_type)
    if not matched or not mode:
        return 0.0, None, 0.0
    raw = float(ZERO_ORIG_STATIC_MATCH_PARAMS.get(mode, 0.0) or 0.0)
    return _sigmoid_match_score(raw), mode, _clip01(raw)


def _match_family_bank(core_norm: str, task_type: str, families: Sequence[Dict[str, object]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    family_name, alias_norm, mode, _, _ = _score_family_bank(core_norm, task_type, families)
    return family_name, alias_norm, mode


def _score_family_bank(
    core_norm: str,
    task_type: str,
    families: Sequence[Dict[str, object]],
) -> Tuple[Optional[str], Optional[str], Optional[str], float, float]:
    for family in families or []:
        family_name = str(family.get("family") or "")
        for alias in family.get("aliases") or []:
            alias_norm = _normalize_phrase(str(alias))
            score, mode, raw = _family_alias_match_score(core_norm, alias_norm, task_type)
            if mode is not None:
                return family_name or None, alias_norm, mode, round(score, 4), round(raw, 4)
    return None, None, None, 0.0, 0.0


def get_zero_originality_runtime_context(item_id: str, task_type: Optional[str] = None,
                                         target_concept: Optional[str] = None,
                                         task_metadata: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    task_type = _infer_task_type(item_id, task_type)
    task_metadata = task_metadata or {}
    context_parts: List[str] = []
    context_source = []

    if task_type == "UUT":
        profiles = _load_json_resource("uut_affordance_profiles.json")
        profile = profiles.get(item_id, {}) if isinstance(profiles, dict) else {}
        aliases = list(profile.get("target_aliases") or [])
        cues = list(profile.get("prompt_cues") or [])
        if target_concept:
            context_parts.append(str(target_concept))
            context_source.append("target_concept")
        if aliases:
            context_parts.extend(aliases)
            context_source.append("target_aliases")
        if cues:
            context_parts.extend(cues)
            context_source.append("prompt_cues")
    elif task_type == "JST":
        templates = _load_json_resource("jst_scenario_templates.json")
        template = templates.get(item_id, {}) if isinstance(templates, dict) else {}
        scenario_text = str(task_metadata.get("scenario_text") or target_concept or "")
        anchors = list(template.get("anchors") or [])
        channel_keywords = []
        for keywords in (template.get("impact_channels") or {}).values():
            channel_keywords.extend(list(keywords or []))
        if scenario_text:
            context_parts.append(scenario_text)
            context_source.append("scenario_text")
        if anchors:
            context_parts.extend(anchors)
            context_source.append("anchors")
        if channel_keywords:
            context_parts.extend(channel_keywords)
            context_source.append("impact_channels")
    elif task_type == "Instances":
        if target_concept:
            context_parts.append(str(target_concept))
            context_source.append("trait")
        trait_cues = INSTANCE_TRAIT_CUES.get(item_id, [])
        if trait_cues:
            context_parts.extend(trait_cues)
            context_source.append("trait_cues")
    elif task_type == "PropConj":
        if target_concept:
            context_parts.append(str(target_concept))
            context_source.append("property_conjunction")
        for prop in task_metadata.get("properties") or []:
            label = str(prop.get("label") or prop.get("id") or "")
            if label:
                context_parts.append(label)
            context_parts.extend(list(prop.get("positive_keywords") or [])[:8])
            context_parts.extend(list(prop.get("evidence_keywords") or [])[:6])
        if task_metadata.get("properties"):
            context_source.append("propconj_properties")

    context_text = " ".join(part for part in context_parts if part).strip()
    return {
        "task_type": task_type,
        "dynamic_context": context_text,
        "context_sources": context_source,
        "context_tokens": sorted(set(_tokenize_content(context_text))),
    }


def analyze_zero_originality(item_id, response_text=None, scorer=None, threshold=0.2,
                             cognitive_baseline=None, target_concept=None,
                             task_type=None, task_metadata=None, parsed_item=None):
    task_type = _infer_task_type(item_id, task_type)
    entry = ZERO_ORIGINALITY_STATIC_BANK.get(item_id, {})
    task_type = task_type or str(entry.get("task_type") or "") or None
    task_metadata = task_metadata or {}

    core_info = extract_task_specific_core(
        item_id,
        response_text=response_text,
        parsed_item=parsed_item,
        task_type=task_type,
    )
    item = core_info.get("parsed_item") or {}
    core_text = core_info.get("core_text") or ""
    core_norm = core_info.get("core_norm") or ""

    hard_family, hard_alias, hard_mode, hard_score, hard_raw_score = _score_family_bank(
        core_norm,
        task_type or "",
        entry.get("hard_zero_families") or [],
    )
    broad_family, broad_alias, broad_mode, broad_score, broad_raw_score = _score_family_bank(
        core_norm,
        task_type or "",
        entry.get("broad_common_families") or [],
    )
    static_threshold = float(ZERO_ORIG_STATIC_MATCH_PARAMS.get("static_threshold", 0.50) or 0.50)
    zero_orig_static = bool(hard_family is not None and hard_score >= static_threshold)

    context_info = get_zero_originality_runtime_context(
        item_id=item_id,
        task_type=task_type,
        target_concept=target_concept,
        task_metadata=task_metadata,
    )
    dynamic_context = context_info.get("dynamic_context") or ""
    zero_orig_dynamic = False
    dynamic_evidence = {
        "dynamic_context": dynamic_context,
        "context_sources": context_info.get("context_sources") or [],
        "baseline_size": 0,
        "baseline_preview": [],
        "eligible_for_dynamic": False,
        "core_token_count": len(_tokenize_content(core_text)),
        "creative_extra_tokens": [],
        "semantic_threshold": None,
        "match": False,
    }

    if cognitive_baseline is not None and dynamic_context and core_norm:
        core_tokens = _tokenize_content(core_text)
        max_words = CORE_WORD_LIMITS.get(task_type or "", 8)
        baseline = cognitive_baseline.get_dynamic_baseline(dynamic_context)
        baseline_vocab = set()
        for answer in baseline:
            baseline_vocab.update(_tokenize_content(answer))
        context_vocab = set(context_info.get("context_tokens") or [])
        creative_extra = sorted(
            token for token in core_tokens
            if token not in context_vocab and token not in baseline_vocab and token not in STOP_WORDS
        )
        semantic_threshold = threshold + (0.03 if task_type == "JST" else 0.05)
        eligible = len(core_tokens) <= max_words and len(creative_extra) <= DYNAMIC_EXTRA_TOKEN_BUDGET.get(task_type or "", 2)
        dynamic_evidence.update({
            "baseline_size": len(baseline),
            "baseline_preview": baseline[:12],
            "eligible_for_dynamic": eligible,
            "creative_extra_tokens": creative_extra,
            "semantic_threshold": round(semantic_threshold, 4),
        })
        if eligible and baseline:
            zero_orig_dynamic = bool(
                cognitive_baseline.is_in_baseline(
                    dynamic_context,
                    core_text,
                    scorer=scorer,
                    semantic_threshold=semantic_threshold,
                )
            )
            dynamic_evidence["match"] = zero_orig_dynamic

    return {
        "zero_orig_static": zero_orig_static,
        "zero_orig_dynamic": zero_orig_dynamic,
        "zero_orig_final": bool(zero_orig_static or zero_orig_dynamic),
        "zero_orig_core_form": core_text,
        "zero_orig_core_norm": core_norm,
        "zero_orig_match_field": core_info.get("match_field"),
        "zero_orig_static_family": hard_family,
        "zero_orig_static_alias": hard_alias,
        "zero_orig_static_match_mode": hard_mode,
        "zero_orig_static_score": hard_score,
        "zero_orig_static_raw_score": hard_raw_score,
        "zero_orig_static_threshold": round(static_threshold, 4),
        "zero_orig_static_transform": "sigmoid_temperature",
        "zero_orig_broad_common_family": broad_family,
        "zero_orig_broad_common_alias": broad_alias,
        "zero_orig_broad_common_match_mode": broad_mode,
        "zero_orig_broad_common_score": broad_score,
        "zero_orig_broad_common_raw_score": broad_raw_score,
        "zero_orig_dynamic_evidence": dynamic_evidence,
        "parsed_item": item,
    }



def is_zero_originality(item_id, response_text, scorer=None, threshold=0.2,
                        cognitive_baseline=None, target_concept=None,
                        task_type=None, task_metadata=None, parsed_item=None):
    trace = analyze_zero_originality(
        item_id=item_id,
        response_text=response_text,
        scorer=scorer,
        threshold=threshold,
        cognitive_baseline=cognitive_baseline,
        target_concept=target_concept,
        task_type=task_type,
        task_metadata=task_metadata,
        parsed_item=parsed_item,
    )
    return bool(trace.get("zero_orig_final"))


__all__ = [
    "ZERO_ORIGINALITY_STATIC_BANK",
    "canonicalize_instances_concept",
    "extract_task_specific_core",
    "get_zero_originality_runtime_context",
    "analyze_zero_originality",
    "is_zero_originality",
]
