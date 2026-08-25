


import copy
import json
import os
from pathlib import Path


UUT_OUTPUT_COUNT = int(os.getenv("OPENROUTER_UUT_OUTPUT_COUNT", os.getenv("OPENROUTER_CREATIVE_OUTPUT_COUNT", "20")))
CREATIVE_OUTPUT_COUNT = UUT_OUTPUT_COUNT
PROP_CONJ_OUTPUT_COUNT = int(os.getenv("OPENROUTER_PROP_CONJ_OUTPUT_COUNT", "12"))
CJST_OUTPUT_COUNT = 12
MACGYVER_PLAN_COUNT = int(os.getenv("OPENROUTER_MACGYVER_OUTPUT_COUNT", "5"))
HYPOUSESPACE_OUTPUT_COUNT = int(os.getenv("OPENROUTER_HYPOUSESPACE_OUTPUT_COUNT", "6"))
GCW_BEAT_COUNT = int(os.getenv("OPENROUTER_GCW_BEAT_COUNT", "6"))
NEOCODER_OUTPUT_COUNT = 1
CLOSED_WORLD_FACT_OUTPUT_COUNT = 1
ANALOGY_TRANSFER_OUTPUT_COUNT = 1
DAT_OUTPUT_COUNT = 10
FF_OUTPUT_COUNT = 20
SECTION12_PROMPT_EXPANSION_FACTOR = int(os.getenv("OPENROUTER_PROMPT_EXPANSION_FACTOR", "5"))


def _auto_article(noun_phrase):
    text = (noun_phrase or "").strip()
    if not text:
        return ""

    lower = text.lower()
    if lower.split()[0] in {"a", "an", "the", "some"}:
        return ""

    tokens = lower.split()
    head = tokens[-1]
    if head.endswith("s") and not head.endswith(("ss", "us")):
        return ""
    return "an" if head[:1] in {"a", "e", "i", "o", "u"} else "a"


def generate_uut_prompt(item, output_count=CREATIVE_OUTPUT_COUNT):
    article = _auto_article(item)
    item_phrase = f"{article} {item}" if article else item
    return (
        f"Think of exactly {output_count} unusual, creative, and physically implementable uses for {item_phrase}. "
        f"Reward surprise and divergence, but every idea must be based on real properties, structure, or affordances of the object. "
        f"Do not invent a new main tool, violate obvious physics, or make the object perform a role it could not plausibly support. "
        f"Return exactly one JSON array with {output_count} objects. "
        f"Each object must have five keys: "
        f"\"idea\" = a short title of 2-8 words, and "
        f"\"mechanism\" = one concise clause of at most 18 words explaining how the object makes the use work, "
        f"\"required_extra_items\" = a JSON array of minor helper items only, or [] if none, "
        f"\"key_affordances\" = a JSON array of 1-4 short affordance/property phrases used by the idea, and "
        f"\"main_object_role\" = \"primary\" if the {item} is the main functional object, otherwise \"supporting\". "
        f"No markdown, no numbering, and no text before or after the JSON array."
    )


def generate_jst_prompt(scenario, output_count=CREATIVE_OUTPUT_COUNT):
    return (
        f"Just suppose {scenario}. "
        f"Generate exactly {output_count} creative and unusual consequences or outcomes. "
        f"Return exactly one JSON array with {output_count} strings. "
        f"Each string must be a short consequence clause of about 4-18 words. "
        f"No markdown, no numbering, and no text before or after the JSON array."
    )


def generate_cjst_v2_prompt(scenario, output_count=CJST_OUTPUT_COUNT):
    return (
        f"Just suppose {scenario}. "
        "Assume this is the only impossible fact. All other ordinary facts about physics, "
        "biology, materials, human behavior, institutions, and language remain normal. "
        f"Generate exactly {output_count} creative consequences: "
        "4 immediate consequences, 4 adaptive consequences, and 4 second-order consequences. "
        f"Return exactly one JSON array with {output_count} objects. "
        "Each object must have exactly eight keys: "
        "\"tier\" = one of \"immediate\", \"adaptive\", or \"second_order\", "
        "\"consequence\" = a short consequence clause of about 5-18 words, "
        "\"causal_bridge\" = one concise clause explaining how the premise leads to the consequence, "
        "\"causal_chain\" = a JSON array of 2-3 short steps that starts with the impossible premise, "
        "passes through an intermediate changed state, and ends at the consequence; second_order items must "
        "make clear which adaptation or repeated behavior creates the later consequence, "
        "\"domain\" = one short impact domain such as safety, infrastructure, economy, privacy, culture, ecology, or daily_life, and "
        "\"anchor_terms\" = a JSON array of 1-5 short words or phrases from the premise that support the consequence, "
        "\"world_state_update\" = a JSON object with exactly four short string keys: "
        "\"variable\", \"old_value\", \"new_value\", and \"licensed_by\", describing only the changed state licensed by the premise, and "
        "\"protected_variables_respected\" = true if ordinary physics, time, minds, entity identity, finite energy, and institutions remain unchanged. "
        "Do not add extra magic beyond the premise. Do not make unrelated inventions, supernatural powers, "
        "time travel, sentient planets, or impossible energy sources unless the premise explicitly licenses them. "
        "No markdown, no numbering, and no text before or after the JSON array."
    )


def generate_cjst_prompt(scenario, output_count=CJST_OUTPUT_COUNT):
    return generate_cjst_v2_prompt(scenario, output_count=output_count)


def generate_instances_prompt(trait, output_count=CREATIVE_OUTPUT_COUNT):
    normalized_trait = (trait or "").strip()
    connector = "that"
    lower = normalized_trait.lower()
    if not lower.startswith(("make ", "makes ", "have ", "has ", "can ", "do ", "does ")):
        connector = "that are"
    return (
        f"Name exactly {output_count} things {connector} {normalized_trait}. "
        f"Prefer unusual and creative examples. "
        f"Return exactly one JSON array with {output_count} short noun phrases. "
        f"Each item must be a noun phrase of about 1-6 words, with no explanation sentence. "
        f"No markdown, no numbering, and no text before or after the JSON array."
    )


def generate_propconj_prompt(task, output_count=PROP_CONJ_OUTPUT_COUNT):
    properties = task.get("properties") or []
    property_labels = [prop["label"] for prop in properties]
    property_ids = [prop["id"] for prop in properties]
    constraints_text = " + ".join(property_labels)
    evidence_keys = ", ".join(f'"{prop_id}"' for prop_id in property_ids)
    return (
        f"Generate exactly {output_count} real-world things that satisfy all of these properties at once: "
        f"{constraints_text}. "
        f"Prefer uncommon, specific, and diverse instances, but do not invent fictional objects, impossible objects, "
        f"or examples that only satisfy some properties. "
        f"Return exactly one JSON array with {output_count} objects. "
        f"Each object must have exactly three keys: "
        f"\"item\" = a short noun phrase of about 1-6 words, "
        f"\"evidence_for_each_property\" = an object with exactly these keys: {evidence_keys}, where each value is a concise factual phrase, and "
        f"\"why_uncommon\" = one concise phrase explaining why the item is not an obvious/common answer. "
        f"No markdown, no numbering, and no text before or after the JSON array."
    )


def _format_macgyver_tools(task):
    chunks = []
    for tool in task.get("tools") or []:
        tool_name = tool.get("name") or tool.get("id")
        affordances = tool.get("affordances") or {}
        if isinstance(affordances, dict):
            affordance_labels = ", ".join(sorted(str(key) for key in affordances.keys()))
        else:
            affordance_labels = ", ".join(str(item) for item in affordances)
        chunks.append(f"{tool_name} ({affordance_labels})")
    return "; ".join(chunks)


def _format_macgyver_constraints(task):
    return "; ".join(
        str(constraint.get("description") or constraint.get("id"))
        for constraint in task.get("constraints") or []
    )


def generate_macgyver_prompt(task, output_count=MACGYVER_PLAN_COUNT):
    clarification_clause = (
        "If the task is underspecified and the missing information changes whether a safe solution is possible, "
        "set \"solvability\" to \"needs_clarification\", set \"plans\" to an empty array, and provide "
        "\"clarification_questions\" as a JSON array of concise questions. "
    )
    return (
        "Solve the following constrained MacGyver-style problem. "
        f"Scene: {task.get('scene')} "
        f"Goal: {task.get('goal')} "
        f"Available tools only: {_format_macgyver_tools(task)}. "
        f"Constraints: {_format_macgyver_constraints(task)}. "
        f"Return exactly one JSON object, with no markdown and no extra text. "
        f"The first character of your answer must be {{ and the last must be }}. "
        f"If the problem is solvable, set \"solvability\" to \"solvable\" and provide exactly {output_count} different plans. "
        f"If the problem is unsolvable using only the listed tools, set \"solvability\" to \"unsolvable\", "
        f"set \"plans\" to an empty array, and give a concise \"impossibility_reason\". "
        f"{clarification_clause}"
        f"For solvable answers, each plan must have exactly these keys: "
        f"\"plan_name\", \"core_trick\", \"used_tools\", \"tool_chain\", \"steps\", "
        f"\"final_state\", \"failure_mode\", \"why_distinct\", and \"risk_note\". "
        f"Each step must be an object with exactly these keys: "
        f"\"action\", \"tools\", \"mechanism\", and \"target_effect\". "
        f"Every tool named anywhere must come from the available tool list. "
        f"Every mechanism must explain what physical property makes the step work. "
        f"The {output_count} plans must use meaningfully different core physical tricks, "
        f"not just wording variants of the same mechanism."
    )


def _format_hypospace_entities(task):
    formatted = []
    for entity in task.get("available_entities") or []:
        affordances = entity.get("affordances") or {}
        affordance_tags = ", ".join(sorted(str(tag) for tag in affordances.keys())) if isinstance(affordances, dict) else ", ".join(str(tag) for tag in affordances)
        aliases = ", ".join(str(alias) for alias in entity.get("aliases") or [])
        formatted.append(
            f"{entity.get('id')} ({entity.get('name')}; family={entity.get('family')}; "
            f"aliases=[{aliases}]; affordances=[{affordance_tags}])"
        )
    return " ".join(formatted)


def _format_hypospace_operations(task):
    formatted = []
    for operation in task.get("allowed_operations") or []:
        aliases = ", ".join(str(alias) for alias in operation.get("aliases") or [])
        formatted.append(
            f"{operation.get('id')} ({operation.get('name')}; family={operation.get('family')}; aliases=[{aliases}])"
        )
    return " ".join(formatted)


def _format_hypospace_goals(task):
    return " ".join(
        f"{goal.get('id')} ({', '.join(str(keyword) for keyword in goal.get('keywords') or [])})"
        for goal in task.get("goal_predicates") or []
    )


def _format_hypospace_constraints(task):
    return " ".join(
        f"{constraint.get('id')}: {constraint.get('description')}"
        for constraint in task.get("constraints") or []
    )


def _format_hypospace_evidence_pack(task):
    pack = task.get("evidence_pack") or {}
    if not isinstance(pack, dict):
        return ""
    claims = []
    for claim in pack.get("claims") or []:
        claims.append(f"{claim.get('claim_id')} [{pack.get('doc_id')}]: {claim.get('text')}")
    forbidden = []
    for claim in pack.get("forbidden_claims") or []:
        forbidden.append(f"{claim.get('claim_id')} [forbidden]: {claim.get('text')}")
    parts = []
    if pack.get("topic"):
        parts.append(f"Evidence topic: {pack.get('topic')}.")
    if claims:
        parts.append("Supported evidence claims: " + " ".join(claims))
    if forbidden:
        parts.append("Forbidden evidence claims: " + " ".join(forbidden))
    return " ".join(parts)


def _hypospace_evidence_ids(task):
    pack = task.get("evidence_pack") or {}
    ids = []
    if isinstance(pack, dict):
        if pack.get("doc_id"):
            ids.append(str(pack.get("doc_id")))
        ids.extend(str(claim.get("claim_id")) for claim in pack.get("claims") or [] if claim.get("claim_id"))
        ids.extend(str(claim.get("claim_id")) for claim in pack.get("forbidden_claims") or [] if claim.get("claim_id"))
    return ids


def generate_hypospace_v2_prompt(task, output_count=HYPOUSESPACE_OUTPUT_COUNT):
    subtype = task.get("task_subtype") or "EvidenceHypoSpace"
    minimal_clause = ""
    if subtype == "MinimalContextIdeation":
        minimal_clause = (
            f"Minimal cue: {task.get('minimal_cue') or task.get('goal')}. "
            "Use the cue only as the ideation target; all factual support must come from the evidence pack and closed entity space. "
        )
    evidence_ids = ", ".join(_hypospace_evidence_ids(task))
    return (
        "Generate closed-world evidence-constrained mechanism hypotheses. "
        "Use only the listed entities, operation tags, mechanism tags, goal predicates, and evidence IDs. "
        "Do not introduce unlisted tools, hidden materials, unsupported feasibility claims, impossible physics, or fabricated citations. "
        f"Task subtype: {subtype}. {minimal_clause}"
        f"Task: {task.get('title')}. Scene: {task.get('scene')} Goal: {task.get('goal')} "
        f"Available entities only: {_format_hypospace_entities(task)} "
        f"Allowed operation tags only: {_format_hypospace_operations(task)} "
        f"Goal predicates: {_format_hypospace_goals(task)} "
        f"Constraints: {_format_hypospace_constraints(task)} "
        f"Closed evidence pack: {_format_hypospace_evidence_pack(task)} "
        f"Allowed evidence IDs only: {evidence_ids}. "
        "If no valid hypothesis is possible using only this closed world, return exactly one JSON object "
        "with keys \"no_valid_hypothesis\" = true, \"reason\", \"hypotheses\" = [], and \"claim_ledger\" = []. "
        f"Otherwise return exactly one JSON object with no markdown and no extra text. "
        f"The first character of your answer must be {{ and the last must be }}. "
        "The object must have exactly these keys: \"hypotheses\" and \"claim_ledger\". "
        f"\"hypotheses\" must be an array with exactly {output_count} different hypothesis objects. "
        "Each hypothesis object must have exactly these keys: "
        "\"hypothesis\" = one concise natural-language mechanism, "
        "\"entities\" = JSON array using only available entity IDs, "
        "\"operation_tags\" = JSON array using only allowed operation tag IDs, "
        "\"mechanism_tags\" = JSON array of affordance/mechanism tags grounded in the listed entities, "
        "\"expected_effects\" = JSON array using goal predicate IDs, "
        "\"evidence\" = JSON array of short physical support claims, "
        "\"evidence_ids\" = JSON array using only allowed evidence IDs, and "
        "\"claim_ids\" = JSON array linking this hypothesis to claim_ledger IDs, "
        "\"core_mechanism\" = one short phrase naming the central mechanism, "
        "\"evidence_chain\" = JSON array of 2-3 short steps from cited evidence to expected effect, "
        "\"why_distinct\" = one short phrase explaining how this differs from the other hypotheses, "
        "\"boundary_note\" = one short phrase naming the closed-world boundary respected, and "
        "\"testable_prediction\" = one short checkable consequence if the hypothesis works. "
        "\"claim_ledger\" must be an array of atomic claim objects with exactly these keys: "
        "\"claim_id\", \"hypothesis_id\", \"text\", \"claim_type\", \"support_ids\", and \"evidence_ids\". "
        "Every feasibility, mechanism, boundary, or causal claim must cite evidence IDs that actually support it. "
        "Make hypotheses mutually distinct in entities, operations, or mechanisms."
    )


def generate_hypospace_prompt(task, output_count=HYPOUSESPACE_OUTPUT_COUNT):
    if task.get("task_schema") == "hypouse_space_tasks_v2" or task.get("evidence_pack"):
        return generate_hypospace_v2_prompt(task, output_count=output_count)
    return (
        "Generate a set of closed-world mechanism hypotheses for the following small-world task. "
        "Use only the listed entities, operation tags, and goal predicates. "
        "Do not introduce unlisted tools, hidden materials, magical effects, impossible physics, or contradictory claims. "
        f"Task: {task.get('title')}. Scene: {task.get('scene')} Goal: {task.get('goal')} "
        f"Available entities only: {_format_hypospace_entities(task)} "
        f"Allowed operation tags only: {_format_hypospace_operations(task)} "
        f"Goal predicates: {_format_hypospace_goals(task)} "
        f"Constraints: {_format_hypospace_constraints(task)} "
        "If no valid hypothesis is possible using only this closed world, return exactly one JSON object "
        "with keys \"no_valid_hypothesis\" = true, \"reason\", and \"hypotheses\" = []. "
        f"Otherwise return exactly one JSON array with {output_count} different hypothesis objects, with no markdown and no extra text. "
        "Each object must have exactly these keys: "
        "\"hypothesis\" = one concise natural-language mechanism, "
        "\"entities\" = JSON array using only available entity IDs, "
        "\"operation_tags\" = JSON array using only allowed operation tag IDs, "
        "\"mechanism_tags\" = JSON array of affordance/mechanism tags grounded in the listed entities, "
        "\"expected_effects\" = JSON array using goal predicate IDs, and "
        "\"evidence\" = JSON array of short physical support claims. "
        "Make the hypotheses mutually distinct in entities, operations, or mechanisms."
    )


def _format_gcw_sheet(rows):
    formatted = []
    for row in rows or []:
        row_id = row.get("id")
        row_type = row.get("type") or row.get("severity") or "fact"
        text = row.get("text") or ""
        formatted.append(f"{row_id} [{row_type}]: {text}")
    return " ".join(formatted)


def _load_gcw_constraint_ladders():
    path = Path(__file__).resolve().parent / "data" / "gcw_constraint_ladders.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_gcw_constraint_profile(task, ladders=None):
    ladders = ladders or {}
    level_defaults = copy.deepcopy(ladders.get("level_defaults") or {})
    card_profiles = ladders.get("cards") or {}
    task_id = task.get("id")
    card_profile = card_profiles.get(task_id) or {}
    selected_level = (
        task.get("constraint_level")
        or card_profile.get("selected_level")
        or ladders.get("default_selected_level")
        or "gcw_l3"
    )
    profile = dict(level_defaults.get(selected_level) or {})
    profile.update((card_profile.get("levels") or {}).get(selected_level) or {})
    profile["level_id"] = selected_level
    profile["selected_level"] = selected_level
    profile["required_fact_ids"] = list(task.get("required_facts") or [])
    profile["evidence_ids"] = [
        str(row.get("id"))
        for row in (task.get("fact_sheet") or [])
        if row.get("id")
    ]
    profile["constraint_ids"] = [
        str(row.get("id"))
        for row in (task.get("constraint_sheet") or [])
        if row.get("id")
    ]
    if "required_causal_callback_terms" not in profile:
        profile["required_causal_callback_terms"] = list(task.get("motifs") or [])
    return profile


def _format_gcw_constraint_profile(profile):
    level_id = profile.get("level_id") or "gcw_l3"
    parts = [
        f"Constraint level: {level_id}",
        f"required_fact_coverage={profile.get('required_fact_coverage', 1.0)}",
    ]
    if profile.get("entity_persistence"):
        parts.append("keep core characters/entities persistent; do not rename or replace them")
    if profile.get("require_evidence_ids"):
        parts.append("every closed-world claim must cite evidence IDs")
    if profile.get("require_claim_ledger"):
        parts.append("include a claims ledger with atomic factual/causal claims")
    if profile.get("required_causal_callback"):
        terms = ", ".join(profile.get("required_causal_callback_terms") or [])
        parts.append(f"the ending must causally callback at least one motif term: {terms}")
    if profile.get("hard_no_drift"):
        parts.append("hard no-drift: no unlisted major entity, device, power, location, or rescue")
    return " ".join(parts)


def generate_gcw_v2_prompt(task, beat_count=GCW_BEAT_COUNT):
    allowed_entities = task.get("allowed_entities") or []
    required_facts = task.get("required_facts") or []
    fact_sheet = _format_gcw_sheet(task.get("fact_sheet") or [])
    constraint_sheet = _format_gcw_sheet(task.get("constraint_sheet") or [])
    profile = task.get("constraint_profile") or _resolve_gcw_constraint_profile(task)
    evidence_ids = ", ".join(list(profile.get("evidence_ids") or []) + list(profile.get("constraint_ids") or []))
    level_requirements = _format_gcw_constraint_profile(profile)
    return (
        "Write one grounded realist microfiction from the given fact sheet and constraint sheet. "
        "You may invent sensory detail, dialogue, and small actions, but you must not rewrite closed-world facts, "
        "add new named characters, violate hard constraints, or make unsupported physical/mechanical claims. "
        f"Story card: {task.get('title')}. Genre mode: {task.get('genre_mode', 'realist_microfiction')}. "
        f"Fact sheet: {fact_sheet} "
        f"Constraint sheet: {constraint_sheet} "
        f"CS4-style constraint ladder requirements: {level_requirements} "
        f"Evidence IDs available for citations: {evidence_ids}. "
        f"Required fact IDs that must be used: {', '.join(required_facts)}. "
        f"Allowed major entities only: {', '.join(allowed_entities)}. "
        "Return exactly one JSON object, with no markdown and no text before or after it. "
        "The first character of your answer must be { and the last must be }. "
        "The object must have exactly these keys: "
        "\"title\", \"grounded_turn\", \"constraint_strategy\", \"payoff_ledger\", "
        "\"beats\", \"claims\", \"ending_callback\", and \"style_devices\". "
        "\"grounded_turn\" must name the fact/constraint IDs used, explain how the story avoids the forbidden obvious solution, "
        "and state the non-obvious supported turn. "
        "\"constraint_strategy\" must briefly describe how hard constraints stay intact. "
        "\"payoff_ledger\" must be an array of 2-4 objects with \"payoff\", \"evidence_ids\", and \"beat_ids\". "
        f"\"beats\" must be an array of exactly {beat_count} objects. "
        "Each beat object must have exactly these keys: "
        "\"beat_id\" = an integer starting at 1, "
        "\"beat_role\" = one of setup, pressure, turn, consequence, payoff, callback, "
        "\"causal_function\" = one sentence explaining this beat's causal work, "
        "\"paragraph\" = one paragraph of 35-80 words, "
        "\"used_fact_ids\" = a JSON array of fact IDs from the fact sheet, "
        "\"characters\" = a JSON array using only allowed character/entity names, "
        "\"places\" = a JSON array using only allowed places/entities, "
        "\"objects\" = a JSON array using only allowed objects/entities, and "
        "\"claimed_new_facts\" = a JSON array of concise new story facts or [] if none. "
        "\"claims\" must be a JSON array of atomic closed-world or causal claims used by the story. "
        "Each claim object must have exactly these keys: "
        "\"claim_id\" = a short ID such as CL1, "
        "\"beat_id\" = the beat number it supports, "
        "\"text\" = one atomic factual or causal claim, "
        "\"claim_type\" = one of fact, entity, causal_callback, permitted_scene_invention, constraint_respect, "
        "\"support_ids\" = a JSON array of fact or constraint IDs that support the claim, and "
        "\"evidence_ids\" = the same evidence IDs repeated for citation checking. "
        "Do not cite evidence IDs that are not listed above. "
        "Use the required facts across the beats, make at least one surprising but grounded turn, "
        "and keep the ending earned rather than arbitrary."
    )


def generate_gcw_prompt(task, beat_count=GCW_BEAT_COUNT):
    return generate_gcw_v2_prompt(task, beat_count=beat_count)






def generate_dat_prompt(output_count=DAT_OUTPUT_COUNT):
    return (
        f"Please enter exactly {output_count} words that are as different from each other as possible, "
        "in all meanings and uses of the words. "
        "Rules: "
        "Only single words in English. "
        "Only nouns (e.g., things, objects, concepts). "
        "No proper nouns (e.g., no specific people or places). "
        "No specialized vocabulary (e.g., no technical terms). "
        "Think of the words on your own (e.g., do not just look at objects "
        "in your surroundings). "
        f"Return exactly one JSON array with {output_count} strings, one noun per string. "
        "No markdown and no text before or after the JSON array."
    )






def generate_cdat_prompt(cue, output_count=DAT_OUTPUT_COUNT):
    return (
        f"Please enter exactly {output_count} words that are as different from each other as possible, "
        f"in all meanings and uses of the words, yet semantically associated "
        f"with the following cue word: {cue}. "
        f"Rules: "
        f"Only single words in English. "
        f"Only nouns (e.g., things, objects, concepts). "
        f"No proper nouns (e.g., no specific people or places). "
        f"No specialized vocabulary (e.g., no technical terms). "
        f"Think of the words on your own (e.g., do not just look at objects "
        f"in your surroundings). "
        f"Return exactly one JSON array with {output_count} strings, one noun per string. "
        f"No markdown and no text before or after the JSON array."
    )






def generate_ff_prompt(seed_word, output_count=FF_OUTPUT_COUNT):
    
    return (
        f"Starting with the word '{seed_word}', write down the next word "
        f"that comes to mind from the previous word. Then write the next "
        f"word that comes to mind from THAT word, and so on. "
        f"Continue this chain for {output_count - 1} more words ({output_count} total including "
        f"'{seed_word}'). "
        f"Rules: "
        f"Only single words. "
        f"No proper nouns (no names, brands, or places). "
        f"Each word should be inspired by the word immediately before it, "
        f"not by the starting word. "
        f"Return exactly one JSON array with {output_count} strings. "
        f"The first string must be '{seed_word}'. "
        f"No markdown and no text before or after the JSON array."
    )


def generate_neocoder_prompt(task, output_count=NEOCODER_OUTPUT_COUNT):
    allowed_imports = task.get("allowed_imports") or []
    denied = task.get("denied_techniques") or []
    public_examples = task.get("public_examples") or task.get("unit_tests") or []
    complexity_claims = task.get("accepted_complexity_claims") or []
    allowed_import_text = ", ".join(allowed_imports) if allowed_imports else "none"
    denied_text = ", ".join(denied) if denied else "none"
    examples_text = json.dumps(public_examples, ensure_ascii=False, separators=(",", ":"))
    return (
        "Solve this Python-only executable code creativity task. "
        "Return exactly one JSON object with no markdown and no text before or after it. "
        "The object must have exactly these keys: "
        "\"code\", \"technique_ledger\", \"constraint_ledger\", \"complexity_claim\", "
        "\"strategy_summary\", \"denial_adaptation\", \"invariant_notes\", and \"edge_case_notes\". "
        f"Task id: {task.get('id')}. Denial state: {task.get('denial_state', 0)}. "
        f"Title: {task.get('title')}. "
        f"Required entrypoint signature: {task.get('signature')}. "
        f"Problem: {task.get('problem_statement')} "
        f"Input contract: {task.get('input_contract')} "
        f"Output contract: {task.get('output_contract')} "
        f"Allowed imports only: {allowed_import_text}. "
        f"Denied techniques: {denied_text}. "
        f"Public examples the function must satisfy: {examples_text}. "
        f"Accepted complexity claims: {', '.join(complexity_claims)}. "
        "\"code\" must be a JSON string containing only Python source that defines the required function. "
        "Do not include stdin/stdout code, file access, networking, subprocesses, eval/exec/open/input, or global side effects. "
        "\"technique_ledger\" must be an array of objects with keys \"id\", \"text\", and \"used\" describing the main algorithmic techniques. "
        "\"constraint_ledger\" must be an array of objects with keys \"id\", \"text\", and \"satisfied\" covering allowed imports, denied techniques, entrypoint, and tests. "
        "\"complexity_claim\" must be one concise time/space complexity claim from the accepted set when possible. "
        "\"strategy_summary\" must summarize the algorithm, \"denial_adaptation\" must explain how denied techniques are avoided, "
        "\"invariant_notes\" must state the main correctness invariant, and \"edge_case_notes\" must identify relevant boundary cases. "
        "Prefer a correct strategy that avoids the denied techniques rather than reusing the most common solution."
    )


def _format_closed_world_fact_evidence(task):
    facts = task.get("support_facts") or []
    return " ".join(
        f"{fact.get('id')}: {fact.get('text')}"
        for fact in facts
        if fact.get("id") and fact.get("text")
    )


def generate_closed_world_fact_prompt(task, output_count=CLOSED_WORLD_FACT_OUTPUT_COUNT):
    evidence_ids = ", ".join(str(item) for item in task.get("required_evidence_ids") or [])
    known_entities = ", ".join(str(item) for item in task.get("known_entities") or [])
    evidence_text = _format_closed_world_fact_evidence(task)
    return (
        "Answer this closed-world factual calibration question using only the listed evidence. "
        "Do not use outside knowledge. If the answer cannot be determined from this closed world, "
        "set \"answerable\" to false and explain the missing or conflicting support. "
        f"Question type: {task.get('question_type')}. "
        f"Question: {task.get('question')} "
        f"Known entities for this question: {known_entities or 'none beyond the evidence'}. "
        f"Closed evidence records: {evidence_text} "
        f"Expected evidence IDs when answerable: {evidence_ids or 'none'}. "
        "Return exactly one JSON object with no markdown and no extra text. "
        "The object must have exactly these keys: "
        "\"answerable\" = true or false, "
        "\"answer\" = a short string, number, boolean, list, or null, "
        "\"evidence_ids\" = a JSON array using only evidence IDs listed above, "
        "\"rationale_steps\" = a JSON array of short evidence-grounded steps in order, "
        "\"unknown_or_conflict_reason\" = a short string or empty string, and "
        "\"confidence\" = a number from 0 to 1. "
        "For list answers, include every item and no extras. For comparison answers, name the winning entity. "
        "For unanswerable questions, keep answer null or an explicit unknown phrase and cite only evidence that shows the boundary."
    )


def _format_analogy_facts(facts):
    return " ".join(
        f"{fact.get('id')}: {fact.get('text')} [relation={fact.get('relation')}]"
        for fact in facts or []
        if fact.get("id") and fact.get("text")
    )


def _format_analogy_gold_mappings(mappings):
    return " ".join(
        (
            f"{item.get('mapping_id')}: dimension={item.get('dimension')}; "
            f"source_relation={item.get('source_relation')}; "
            f"target_relation={item.get('target_relation')}; "
            f"abstraction={item.get('abstraction')}; "
            f"source_evidence_ids={item.get('source_evidence_ids')}; "
            f"target_evidence_ids={item.get('target_evidence_ids')}."
        )
        for item in mappings or []
        if item.get("mapping_id")
    )


def _format_analogy_limits(limits):
    return " ".join(
        f"{item.get('limit_id')}: {item.get('text')} keywords={item.get('keywords')}"
        for item in limits or []
        if item.get("limit_id") and item.get("text")
    )


def _format_analogy_forbidden(forbidden):
    return " ".join(
        f"{item.get('id')}: {item.get('text')} keywords={item.get('keywords')}"
        for item in forbidden or []
        if item.get("id") and item.get("text")
    )


def generate_analogy_transfer_prompt(task, output_count=ANALOGY_TRANSFER_OUTPUT_COUNT):
    cluster = task.get("cluster") or task
    source_domain = cluster.get("source_domain")
    target_domain = cluster.get("target_domain")
    dimensions = cluster.get("dimensions") or {}
    dimension_text = " ".join(
        f"{name}: source={value.get('source')}; target={value.get('target')}."
        for name, value in dimensions.items()
        if isinstance(value, dict)
    )
    source_facts = _format_analogy_facts(cluster.get("source_facts"))
    target_facts = _format_analogy_facts(cluster.get("target_facts"))
    limits = _format_analogy_limits(cluster.get("required_limits"))
    forbidden = _format_analogy_forbidden(cluster.get("forbidden_transfers"))
    support_boundary = cluster.get("support_boundary") or {}
    support_ids = ", ".join(str(item) for item in support_boundary.get("evidence_ids") or [])
    known_entities = ", ".join(str(item) for item in cluster.get("known_entities") or [])
    return (
        "Complete this closed-world analogy false-transfer challenge using only the listed source and target facts. "
        "Do not use outside knowledge and do not transfer source-only details into the target domain. "
        f"Variant: {task.get('variant')}. Instruction: {task.get('instruction')} "
        f"Source domain: {source_domain}. Target domain: {target_domain}. "
        f"Historical-analogy dimensions: {dimension_text} "
        f"Source evidence records: {source_facts} "
        f"Target evidence records: {target_facts} "
        f"Known entities: {known_entities or 'only entities in the evidence records'}. "
        f"Support boundary evidence IDs: {support_ids or 'all listed IDs only'}. "
        f"Support boundary policy: {support_boundary.get('policy') or 'Use only listed facts.'} "
        f"Required limits of analogy: {limits} "
        f"Forbidden transfers to avoid and warn about: {forbidden} "
        "Return exactly one JSON object with no markdown and no extra text. "
        "The object must have exactly these keys: "
        "\"analogy_summary\" = a concise source-to-target analogy summary, "
        "\"candidate_abstractions\" = an array of candidate structural abstractions, "
        "\"mapping_ledger\" = an array of mapping objects, "
        "\"mapping_chain\" = an array of 2-3 short steps explaining the strongest cross-domain mapping, "
        "\"transfer_tests\" = an array of cautious target-side tests or predictions licensed by target evidence, "
        "\"negative_transfer_tests\" = an array of tempting source-only transfers that should fail in the target, "
        "\"transferred_inferences\" = an array of cautious target inferences licensed by mapped evidence, "
        "\"limits_of_analogy\" = an array of explicit limits, "
        "\"unsupported_transfer_warnings\" = an array naming tempting but unsupported transfers, and "
        "\"boundary_rationale\" = a concise explanation of why the valid transfer stops at the listed limits, and "
        "\"confidence\" = a number from 0 to 1. "
        "Each mapping_ledger item must include \"source_evidence_ids\", \"target_evidence_ids\", "
        "\"dimension\", \"abstraction\", \"mapped_relation\", and \"role_alignment\". "
        "Each transferred inference should include \"text\" and \"evidence_ids\". "
        "Each limit should include \"text\" and optionally \"limit_id\". "
        "Use only listed evidence IDs; unknown IDs, invented facts, surface-only matches, or literal forbidden transfers will be penalized."
    )






UUT_ITEMS = [
    ("uut_01", "Carton"),
    ("uut_02", "Tin can"),
    ("uut_03", "Suitcase"),
    ("uut_04", "Wallet"),
    ("uut_05", "Ladder"),
    ("uut_06", "Umbrella"),
    ("uut_07", "Newspaper"),
    ("uut_08", "Spoon"),
    ("uut_09", "Rope"),
    ("uut_10", "Brick"),
    ("uut_11", "Sock"),
    ("uut_12", "Plastic bottle"),
    ("uut_13", "Towel"),
    ("uut_14", "Key"),
    ("uut_15", "Plate"),
    ("uut_16", "Book"),
    ("uut_17", "Chair"),
    ("uut_18", "Toothbrush"),
    ("uut_19", "Shoe"),
    ("uut_20", "Paper clip"),
]

_UUT_AUGMENTED_ITEMS = [
    "Paper bag", "Cardboard box", "Glass jar", "Rubber band", "Pencil",
    "Envelope", "Button", "Bottle cap", "Coffee mug", "Scarf",
    "Tennis ball", "Binder clip", "Plastic fork", "Mason lid", "Cereal box",
    "Clothespin", "Sponge", "Cork", "Shoelace", "Egg carton",
    "Ice cube tray", "Popsicle stick", "Aluminum foil", "Kitchen sponge", "Tea towel",
    "Garden hose", "Bucket", "Coat hanger", "Notebook", "Index card",
    "Straw", "Napkin", "Jar lid", "Rubber glove", "Hair tie",
    "Clipboard", "Magazine", "Pizza box", "Milk jug", "Candle",
    "Flashlight", "Backpack", "Belt", "Cup", "Bowl",
    "Fork", "Comb", "Mirror", "Pillowcase", "Blanket",
    "Mop handle", "Broom", "Dustpan", "Plastic wrap", "Foam cup",
    "Toy block", "Balloon", "Ribbon", "Safety pin", "Matchbox",
    "Paintbrush", "Hanger", "Screwdriver", "Ruler", "Tape roll",
    "Cloth rag", "Mesh bag", "Plant pot", "Tile", "Wooden spoon",
    "Remote control", "Phone case", "Mouse pad", "Calendar", "Greeting card",
    "Laundry basket", "Plastic tray", "Shipping tube", "Tin foil tray", "Wire",
]

if SECTION12_PROMPT_EXPANSION_FACTOR > 1:
    _uut_target_count = len(UUT_ITEMS) * SECTION12_PROMPT_EXPANSION_FACTOR
    for index, item in enumerate(_UUT_AUGMENTED_ITEMS[:max(0, _uut_target_count - len(UUT_ITEMS))], start=len(UUT_ITEMS) + 1):
        UUT_ITEMS.append((f"uut_{index:03d}", item))

JST_SCENARIOS = [
    (
        "jst_01",
        "Cloud strings",
        "the clouds had strings attached to them which hang down to earth",
    ),
    (
        "jst_02",
        "See feet",
        "for some reason, all we could see of people would be their feet",
    ),
    (
        "jst_03",
        "Following suitcases",
        "every suitcase would quietly follow its owner everywhere like a loyal pet",
    ),
    (
        "jst_04",
        "Growing ladders",
        "every ladder grew one new rung each time someone climbed it",
    ),
    (
        "jst_05",
        "Returning wallets",
        "every lost wallet slowly crawled back to its owner on its own",
    ),
    (
        "jst_06",
        "Whispering doors",
        "every door whispered what happened the last time it opened",
    ),
    (
        "jst_07",
        "Floating books",
        "every book floated a few centimeters above any table",
    ),
    (
        "jst_08",
        "Color-changing shadows",
        "people's shadows changed color with their mood",
    ),
    (
        "jst_09",
        "Reversible rain",
        "rain fell upward from puddles into clouds every afternoon",
    ),
    (
        "jst_10",
        "Slow mirrors",
        "mirrors showed reflections from ten minutes ago",
    ),
    (
        "jst_11",
        "Singing traffic lights",
        "traffic lights sang instead of changing silently",
    ),
    (
        "jst_12",
        "Stretching roads",
        "every road stretched longer during rush hour",
    ),
    (
        "jst_13",
        "Talking receipts",
        "receipts read purchases aloud when touched",
    ),
    (
        "jst_14",
        "Transparent walls",
        "all building walls became transparent for one hour each day",
    ),
    (
        "jst_15",
        "Heavy balloons",
        "balloons became heavier the more people laughed nearby",
    ),
    (
        "jst_16",
        "Remembering cups",
        "every cup remembered the last drink poured into it",
    ),
    (
        "jst_17",
        "Borrowed voices",
        "people briefly swapped voices whenever they shook hands",
    ),
    (
        "jst_18",
        "Indoor snow",
        "snow fell only inside rooms with closed windows",
    ),
    (
        "jst_19",
        "Magnetic sidewalks",
        "sidewalks gently pulled metal objects toward the curb",
    ),
    (
        "jst_20",
        "Clockless mornings",
        "all clocks stopped working until noon every day",
    ),
]

INSTANCE_TRAITS = [
    ("ins_01", "Round and Hard", "round and hard"),
    ("ins_02", "Makes a loud noise", "make a loud noise"),
    ("ins_03", "Red and Edible", "red and edible"),
    ("ins_04", "Made of Metal and Hard", "made of metal and hard"),
    (
        "ins_05",
        "Made of Paper or Cardboard and Writable",
        "made of paper or cardboard and writable",
    ),
    ("ins_06", "Transparent and Flexible", "transparent and flexible"),
    ("ins_07", "Soft and Waterproof", "soft and waterproof"),
    ("ins_08", "Tiny and Valuable", "tiny and valuable"),
    ("ins_09", "Hollow and Lightweight", "hollow and lightweight"),
    ("ins_10", "Sticky and Useful", "sticky and useful"),
    ("ins_11", "Cold and Edible", "cold and edible"),
    ("ins_12", "Sharp and Portable", "sharp and portable"),
    ("ins_13", "Bright and Fragile", "bright and fragile"),
    ("ins_14", "Heavy and Movable", "heavy and movable"),
    ("ins_15", "Natural and Patterned", "natural and patterned"),
    ("ins_16", "Flat and Reflective", "flat and reflective"),
    ("ins_17", "Elastic and Colorful", "elastic and colorful"),
    ("ins_18", "Scented and Disposable", "scented and disposable"),
    ("ins_19", "Mechanical and Small", "mechanical and small"),
    ("ins_20", "Silent and Protective", "silent and protective"),
]


def _prop(
    prop_id,
    label,
    *,
    positive_predicates=None,
    negative_predicates=None,
    positive_keywords=None,
    negative_keywords=None,
    evidence_keywords=None,
    weight=1.0,
):
    return {
        "id": prop_id,
        "label": label,
        "weight": weight,
        "positive_predicates": list(positive_predicates or []),
        "negative_predicates": list(negative_predicates or []),
        "positive_keywords": list(positive_keywords or []),
        "negative_keywords": list(negative_keywords or []),
        "evidence_keywords": list(evidence_keywords or positive_keywords or []),
    }


PROPCONJ_PROPERTIES = {
    "round": _prop(
        "round", "round",
        positive_predicates=["shape:round"],
        negative_predicates=["shape:flat", "attr:square", "attr:rectangular", "attr:triangular"],
        positive_keywords=["ball", "marble", "bead", "wheel", "coin", "ring", "pearl", "orange", "button", "sphere", "orb"],
        negative_keywords=["square", "cube", "rectangular", "flat sheet"],
    ),
    "hard": _prop(
        "hard", "hard",
        positive_predicates=["texture:hard", "material:metal", "strength:load_bearing"],
        negative_predicates=["texture:soft"],
        positive_keywords=["stone", "metal", "glass", "ceramic", "shell", "coin", "key", "bolt", "pebble", "steel"],
        negative_keywords=["soft", "cloth", "foam", "rubber band"],
    ),
    "loud_noise": _prop(
        "loud_noise", "makes a loud noise",
        positive_predicates=["sound:loud", "sound:noise"],
        positive_keywords=["alarm", "bell", "siren", "horn", "whistle", "drum", "rattle", "clapper", "firecracker", "gong"],
        negative_keywords=["silent", "quiet", "mute"],
    ),
    "portable": _prop(
        "portable", "portable",
        positive_keywords=["pocket", "handheld", "keychain", "travel", "mini", "folding", "compact", "portable", "pen", "card"],
        negative_keywords=["building", "piano", "boulder", "refrigerator", "sofa"],
    ),
    "red": _prop(
        "red", "red",
        positive_predicates=["color:red"],
        positive_keywords=["red", "ruby", "strawberry", "tomato", "cherry", "cranberry", "raspberry", "crimson", "scarlet"],
        negative_keywords=["blue", "green", "white", "black"],
    ),
    "edible": _prop(
        "edible", "edible",
        positive_predicates=["edible:true", "function:food"],
        positive_keywords=["fruit", "berry", "tomato", "pepper", "cake", "candy", "vegetable", "food", "chocolate", "ice cube"],
        negative_keywords=["stone", "metal", "plastic", "battery", "glass", "ruby"],
    ),
    "metal": _prop(
        "metal", "made of metal",
        positive_predicates=["material:metal", "conductivity:conductive"],
        positive_keywords=["metal", "steel", "iron", "aluminum", "aluminium", "copper", "brass", "tin", "wire", "coin", "key", "bolt"],
        negative_keywords=["paper", "wood", "rubber", "cloth", "plastic"],
    ),
    "paper_cardboard": _prop(
        "paper_cardboard", "made of paper or cardboard",
        positive_predicates=["material:paper"],
        positive_keywords=["paper", "cardboard", "card", "notebook", "index card", "label", "box", "ticket", "paperboard"],
        negative_keywords=["metal", "glass", "stone", "ceramic"],
    ),
    "writable": _prop(
        "writable", "writable",
        positive_predicates=["surface:writable"],
        positive_keywords=["writable", "label", "notebook", "card", "paper", "chalkboard", "whiteboard", "tag", "write"],
        negative_keywords=["slippery", "nonstick"],
    ),
    "transparent": _prop(
        "transparent", "transparent",
        positive_keywords=["transparent", "clear", "glass", "cellophane", "acrylic", "window", "bottle", "lens", "film", "see-through"],
        negative_keywords=["opaque", "painted", "cloudy"],
    ),
    "flexible": _prop(
        "flexible", "flexible",
        positive_predicates=["texture:soft", "form:foldable"],
        positive_keywords=["flexible", "bendable", "rubber", "silicone", "cloth", "film", "wire", "hose", "spring", "foldable"],
        negative_keywords=["rigid", "brittle", "stiff", "ceramic"],
    ),
    "soft": _prop(
        "soft", "soft",
        positive_predicates=["texture:soft"],
        negative_predicates=["texture:hard"],
        positive_keywords=["soft", "foam", "cloth", "cotton", "felt", "sponge", "pillow", "rubber", "silicone"],
        negative_keywords=["hard", "stone", "steel", "glass"],
    ),
    "waterproof": _prop(
        "waterproof", "waterproof",
        positive_predicates=["resistance:water"],
        positive_keywords=["waterproof", "water-resistant", "rubber", "silicone", "plastic", "raincoat", "zip bag", "sealed", "waxed"],
        negative_keywords=["paper", "cardboard", "sponge", "absorbent"],
    ),
    "tiny": _prop(
        "tiny", "tiny",
        positive_keywords=["tiny", "small", "mini", "micro", "seed", "bead", "pin", "grain", "chip", "pearl", "gem"],
        negative_keywords=["large", "giant", "huge", "building"],
    ),
    "valuable": _prop(
        "valuable", "valuable",
        positive_keywords=["gold", "diamond", "pearl", "gem", "ruby", "sapphire", "silver", "rare", "antique", "coin", "stamp"],
        negative_keywords=["trash", "disposable", "cheap", "scrap"],
    ),
    "hollow": _prop(
        "hollow", "hollow",
        positive_predicates=["shape:hollow", "container:container"],
        positive_keywords=["hollow", "tube", "shell", "bottle", "straw", "balloon", "pipe", "egg", "capsule", "thermos"],
        negative_keywords=["solid", "filled"],
    ),
    "lightweight": _prop(
        "lightweight", "lightweight",
        positive_predicates=["weight:light"],
        negative_predicates=["material:metal", "strength:load_bearing"],
        positive_keywords=["lightweight", "light", "foam", "paper", "balloon", "feather", "cork", "plastic", "balsa"],
        negative_keywords=["heavy", "lead", "stone", "iron"],
    ),
    "sticky": _prop(
        "sticky", "sticky",
        positive_keywords=["sticky", "adhesive", "tape", "glue", "sticker", "label", "post-it", "bandage", "tack"],
        negative_keywords=["slippery", "nonstick"],
    ),
    "useful": _prop(
        "useful", "useful",
        positive_keywords=["tool", "tape", "clip", "bandage", "label", "hook", "pin", "key", "utensil", "repair", "organizer"],
        negative_keywords=["decorative only", "useless"],
    ),
    "cold": _prop(
        "cold", "cold",
        positive_keywords=["ice", "frozen", "gel pack", "popsicle", "sorbet", "snow", "cold", "freezer"],
        negative_keywords=["hot", "warm", "heated", "flame"],
    ),
    "sharp": _prop(
        "sharp", "sharp",
        positive_predicates=["edge:sharp"],
        positive_keywords=["sharp", "blade", "knife", "razor", "needle", "pin", "thorn", "scalpel", "edge"],
        negative_keywords=["blunt", "rounded"],
    ),
    "bright": _prop(
        "bright", "bright",
        positive_keywords=["bright", "lamp", "led", "neon", "reflective", "fluorescent", "glow", "laser", "mirror", "light"],
        negative_keywords=["dim", "dark", "matte"],
    ),
    "fragile": _prop(
        "fragile", "fragile",
        positive_predicates=["attr:fragile"],
        positive_keywords=["fragile", "glass", "thin", "shell", "ceramic", "bulb", "crystal", "eggshell"],
        negative_keywords=["rugged", "unbreakable", "steel"],
    ),
    "heavy": _prop(
        "heavy", "heavy",
        positive_keywords=["heavy", "lead", "stone", "iron", "cast iron", "dumbbell", "piano", "safe", "anchor"],
        negative_keywords=["lightweight", "feather", "paper"],
    ),
    "movable": _prop(
        "movable", "movable",
        positive_keywords=["movable", "wheeled", "rolling", "portable", "cart", "suitcase", "trolley", "caster", "folding"],
        negative_keywords=["fixed", "bolted", "building"],
    ),
    "natural": _prop(
        "natural", "natural",
        positive_keywords=["natural", "leaf", "shell", "stone", "seed", "fruit", "wood", "flower", "feather", "bone"],
        negative_keywords=["synthetic", "plastic", "machine-made"],
    ),
    "patterned": _prop(
        "patterned", "patterned",
        positive_keywords=["patterned", "striped", "spotted", "speckled", "marbled", "veined", "banded", "mottled"],
        negative_keywords=["plain", "solid color"],
    ),
    "flat": _prop(
        "flat", "flat",
        positive_predicates=["shape:flat", "surface:flat"],
        negative_predicates=["shape:round", "shape:hollow"],
        positive_keywords=["flat", "sheet", "card", "foil", "mirror", "plate", "screen", "tile", "blanket"],
        negative_keywords=["sphere", "ball", "tube"],
    ),
    "reflective": _prop(
        "reflective", "reflective",
        positive_predicates=["surface:reflective", "material:metal"],
        positive_keywords=["reflective", "mirror", "foil", "polished", "shiny", "chrome", "silver", "glossy"],
        negative_keywords=["matte", "dull", "absorbs light"],
    ),
    "elastic": _prop(
        "elastic", "elastic",
        positive_keywords=["elastic", "rubber band", "stretchy", "spring", "bungee", "balloon", "spandex", "latex"],
        negative_keywords=["rigid", "brittle"],
    ),
    "colorful": _prop(
        "colorful", "colorful",
        positive_keywords=["colorful", "rainbow", "multi-colored", "bright colors", "painted", "marker", "crayon", "confetti"],
        negative_keywords=["monochrome", "plain"],
    ),
    "scented": _prop(
        "scented", "scented",
        positive_keywords=["scented", "fragrant", "perfumed", "aromatic", "lavender", "mint", "rose", "citrus", "soap"],
        negative_keywords=["odorless", "unscented"],
    ),
    "disposable": _prop(
        "disposable", "disposable",
        positive_keywords=["disposable", "single-use", "paper cup", "tissue", "wipe", "straw", "wrapper", "razor", "glove"],
        negative_keywords=["reusable", "permanent", "heirloom"],
    ),
    "mechanical": _prop(
        "mechanical", "mechanical",
        positive_keywords=["mechanical", "gear", "spring", "wind-up", "watch", "motor", "hinge", "clockwork", "ratchet"],
        negative_keywords=["digital only", "biological"],
    ),
    "small": _prop(
        "small", "small",
        positive_keywords=["small", "tiny", "mini", "pocket", "compact", "bead", "pin", "watch", "keychain", "coin"],
        negative_keywords=["large", "huge", "building"],
    ),
    "silent": _prop(
        "silent", "silent",
        positive_keywords=["silent", "quiet", "mute", "soft", "pad", "cushion", "earplug", "foam"],
        negative_keywords=["loud", "alarm", "siren", "bell"],
    ),
    "protective": _prop(
        "protective", "protective",
        positive_keywords=["protective", "guard", "case", "helmet", "shield", "sleeve", "cover", "pad", "glove"],
        negative_keywords=["exposed", "fragile shell only"],
    ),
    "glows": _prop(
        "glows", "glows",
        positive_keywords=["glow", "glows", "firefly", "glow stick", "phosphorescent", "bioluminescent", "luminous", "glow-in-the-dark"],
        negative_keywords=["matte", "unlit", "black stone"],
        evidence_keywords=["bioluminescence", "chemiluminescence", "phosphorescent", "luminous", "glow", "emits light"],
    ),
    "fits_in_pocket": _prop(
        "fits_in_pocket", "fits in a pocket",
        positive_keywords=["pocket", "small", "tiny", "mini", "compact", "keychain", "pen-sized", "bead", "coin", "insect"],
        negative_keywords=["large", "tabletop", "building", "piano", "bicycle"],
    ),
    "no_battery": _prop(
        "no_battery", "needs no battery",
        positive_keywords=["no battery", "battery-free", "natural", "chemical", "wind-up", "hand-crank", "solar", "firefly", "glow stick", "candle"],
        negative_keywords=["battery", "phone", "flashlight", "led", "electric", "rechargeable"],
        evidence_keywords=["natural", "chemical reaction", "chemiluminescence", "bioluminescence", "wind-up", "hand crank", "no battery"],
    ),
    "magnetic": _prop(
        "magnetic", "magnetic",
        positive_keywords=["magnet", "magnetic", "lodestone", "fridge magnet", "neodymium", "compass"],
        negative_keywords=["wood", "paper", "plastic only"],
    ),
    "biodegradable": _prop(
        "biodegradable", "biodegradable",
        positive_keywords=["biodegradable", "compostable", "paper", "leaf", "wood", "cotton", "bamboo", "food", "starch"],
        negative_keywords=["plastic", "metal", "glass"],
    ),
    "reusable": _prop(
        "reusable", "reusable",
        positive_keywords=["reusable", "washable", "refillable", "durable", "metal bottle", "jar", "cloth", "case"],
        negative_keywords=["disposable", "single-use", "throwaway"],
    ),
    "heat_resistant": _prop(
        "heat_resistant", "heat resistant",
        positive_predicates=["resistance:heat", "material:metal"],
        positive_keywords=["heat-resistant", "fireproof", "silicone", "ceramic", "borosilicate", "metal", "oven mitt", "glass"],
        negative_keywords=["paper", "wax", "ice", "plastic bag"],
    ),
    "absorbs_water": _prop(
        "absorbs_water", "absorbs water",
        positive_keywords=["absorbent", "sponge", "cotton", "paper towel", "cloth", "diaper", "moss", "towel"],
        negative_keywords=["waterproof", "waxed", "plastic", "rubber"],
    ),
    "floats": _prop(
        "floats", "floats on water",
        positive_keywords=["float", "floats", "cork", "foam", "balloon", "leaf", "wood", "hollow plastic", "life jacket"],
        negative_keywords=["sinks", "lead", "stone", "iron"],
    ),
    "conductive": _prop(
        "conductive", "electrically conductive",
        positive_predicates=["conductivity:conductive", "material:metal"],
        positive_keywords=["conductive", "copper", "metal", "wire", "aluminum foil", "graphite", "silver", "steel"],
        negative_keywords=["rubber", "plastic", "wood", "ceramic"],
    ),
    "foldable": _prop(
        "foldable", "foldable",
        positive_predicates=["form:foldable", "material:paper"],
        positive_keywords=["foldable", "folding", "paper", "cloth", "foil", "map", "umbrella", "fan", "blanket"],
        negative_keywords=["rigid", "solid block"],
    ),
    "rolls": _prop(
        "rolls", "rolls",
        positive_predicates=["shape:round"],
        positive_keywords=["rolls", "wheel", "ball", "cylinder", "roller", "caster", "marble", "tube", "rolling"],
        negative_keywords=["flat sheet", "cube", "fixed"],
    ),
    "breathable": _prop(
        "breathable", "breathable",
        positive_keywords=["breathable", "mesh", "cotton", "fabric", "vented", "perforated", "gore-tex"],
        negative_keywords=["airtight", "sealed", "plastic wrap"],
    ),
}


PROPCONJ_TASK_COMBOS = [
    ("pc_001", ("round", "hard")),
    ("pc_002", ("loud_noise", "portable")),
    ("pc_003", ("red", "edible")),
    ("pc_004", ("metal", "hard")),
    ("pc_005", ("paper_cardboard", "writable")),
    ("pc_006", ("transparent", "flexible")),
    ("pc_007", ("soft", "waterproof")),
    ("pc_008", ("tiny", "valuable")),
    ("pc_009", ("hollow", "lightweight")),
    ("pc_010", ("sticky", "useful")),
    ("pc_011", ("cold", "edible")),
    ("pc_012", ("sharp", "portable")),
    ("pc_013", ("bright", "fragile")),
    ("pc_014", ("heavy", "movable")),
    ("pc_015", ("natural", "patterned")),
    ("pc_016", ("flat", "reflective")),
    ("pc_017", ("elastic", "colorful")),
    ("pc_018", ("scented", "disposable")),
    ("pc_019", ("mechanical", "small")),
    ("pc_020", ("silent", "protective")),
    ("pc_021", ("glows", "fits_in_pocket", "no_battery")),
    ("pc_022", ("red", "transparent", "flexible")),
    ("pc_023", ("edible", "cold", "portable")),
    ("pc_024", ("magnetic", "metal", "small")),
    ("pc_025", ("biodegradable", "disposable", "scented")),
    ("pc_026", ("reusable", "waterproof", "soft")),
    ("pc_027", ("heat_resistant", "metal", "portable")),
    ("pc_028", ("absorbs_water", "soft", "disposable")),
    ("pc_029", ("floats", "hollow", "lightweight")),
    ("pc_030", ("conductive", "metal", "flexible")),
    ("pc_031", ("foldable", "waterproof", "lightweight")),
    ("pc_032", ("rolls", "round", "movable")),
    ("pc_033", ("breathable", "soft", "protective")),
    ("pc_034", ("transparent", "waterproof", "flexible")),
    ("pc_035", ("glows", "natural", "patterned")),
    ("pc_036", ("edible", "red", "tiny")),
    ("pc_037", ("reflective", "flat", "portable")),
    ("pc_038", ("sticky", "transparent", "useful")),
    ("pc_039", ("sharp", "metal", "small")),
    ("pc_040", ("elastic", "waterproof", "colorful")),
    ("pc_041", ("silent", "mechanical", "small")),
    ("pc_042", ("natural", "scented", "disposable")),
    ("pc_043", ("bright", "flat", "reflective")),
    ("pc_044", ("hollow", "metal", "hard")),
    ("pc_045", ("tiny", "mechanical", "valuable")),
    ("pc_046", ("round", "edible", "red")),
    ("pc_047", ("transparent", "hard", "fragile")),
    ("pc_048", ("protective", "lightweight", "portable")),
    ("pc_049", ("no_battery", "mechanical", "small")),
    ("pc_050", ("glows", "waterproof", "portable")),
    ("pc_051", ("magnetic", "hard", "tiny")),
    ("pc_052", ("biodegradable", "edible", "natural")),
    ("pc_053", ("reusable", "metal", "writable")),
    ("pc_054", ("heat_resistant", "transparent", "hard")),
    ("pc_055", ("absorbs_water", "natural", "biodegradable")),
    ("pc_056", ("floats", "waterproof", "soft")),
    ("pc_057", ("conductive", "flexible", "small")),
    ("pc_058", ("foldable", "flat", "reflective")),
    ("pc_059", ("rolls", "heavy", "movable")),
    ("pc_060", ("breathable", "waterproof", "protective")),
    ("pc_061", ("scented", "natural", "patterned")),
    ("pc_062", ("colorful", "disposable", "paper_cardboard")),
    ("pc_063", ("sticky", "flexible", "transparent")),
    ("pc_064", ("sharp", "disposable", "portable")),
    ("pc_065", ("bright", "no_battery", "portable")),
    ("pc_066", ("mechanical", "silent", "protective")),
    ("pc_067", ("metal", "reflective", "flat")),
    ("pc_068", ("edible", "patterned", "natural")),
    ("pc_069", ("waterproof", "transparent", "reusable")),
    ("pc_070", ("hard", "natural", "round")),
    ("pc_071", ("soft", "silent", "protective")),
    ("pc_072", ("tiny", "transparent", "valuable")),
    ("pc_073", ("hollow", "waterproof", "portable")),
    ("pc_074", ("red", "hard", "valuable")),
    ("pc_075", ("cold", "transparent", "edible")),
    ("pc_076", ("paper_cardboard", "flat", "lightweight")),
    ("pc_077", ("flexible", "conductive", "metal")),
    ("pc_078", ("glows", "fragile", "transparent")),
    ("pc_079", ("magnetic", "reusable", "hard")),
    ("pc_080", ("disposable", "waterproof", "transparent")),
]


def _load_propconj_property_bank_v2():
    path = Path(__file__).resolve().parent / "data" / "propconj_property_bank_v2.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _coerce_propconj_task_combos_from_bank(bank):
    raw_combos = bank.get("task_combos") if isinstance(bank, dict) else None
    if not isinstance(raw_combos, list):
        return []
    combos = []
    for entry in raw_combos:
        if isinstance(entry, dict):
            task_id = entry.get("id") or entry.get("task_id")
            property_ids = entry.get("property_ids")
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            task_id, property_ids = entry
        else:
            continue
        if not task_id or not isinstance(property_ids, (list, tuple)):
            continue
        if all(prop_id in PROPCONJ_PROPERTIES for prop_id in property_ids):
            combos.append((str(task_id), tuple(str(prop_id) for prop_id in property_ids)))
    return combos


def _default_propconj_difficulty(property_ids):
    property_count = len(property_ids or [])
    adversarial_ids = {
        "glows",
        "no_battery",
        "conductive",
        "magnetic",
        "silent",
        "mechanical",
        "fragile",
        "transparent",
        "valuable",
    }
    if property_count <= 2:
        return "2_property_easy"
    if any(prop_id in adversarial_ids for prop_id in property_ids or []):
        return "3_property_adversarial"
    return "3_property_medium"


_PROPCONJ_PROPERTY_BANK_V2 = _load_propconj_property_bank_v2()
_PROPCONJ_TASK_DIFFICULTIES = (
    _PROPCONJ_PROPERTY_BANK_V2.get("task_difficulties", {})
    if isinstance(_PROPCONJ_PROPERTY_BANK_V2.get("task_difficulties"), dict)
    else {}
)
_PROPCONJ_TASK_COMBOS_EFFECTIVE = (
    _coerce_propconj_task_combos_from_bank(_PROPCONJ_PROPERTY_BANK_V2) or PROPCONJ_TASK_COMBOS
)


def _expand_propconj_task_combos(task_combos, factor=SECTION12_PROMPT_EXPANSION_FACTOR):
    combos = [(str(task_id), tuple(property_ids)) for task_id, property_ids in task_combos]
    target_count = len(combos) * max(1, factor)
    if len(combos) >= target_count:
        return combos

    seen_property_sets = {tuple(property_ids) for _task_id, property_ids in combos}
    property_ids = list(PROPCONJ_PROPERTIES)
    next_index = len(combos) + 1
    for size in (2, 3, 4):
        if len(combos) >= target_count:
            break
        for start_index, first_prop in enumerate(property_ids):
            if len(combos) >= target_count:
                break
            for step in range(1, len(property_ids)):
                if len(combos) >= target_count:
                    break
                candidate = [first_prop]
                cursor = (start_index + step) % len(property_ids)
                while len(candidate) < size:
                    prop_id = property_ids[cursor % len(property_ids)]
                    if prop_id not in candidate:
                        candidate.append(prop_id)
                    cursor += step + 1
                candidate_tuple = tuple(candidate)
                if candidate_tuple in seen_property_sets:
                    continue
                seen_property_sets.add(candidate_tuple)
                combos.append((f"pc_{next_index:03d}", candidate_tuple))
                next_index += 1
    return combos


_PROPCONJ_TASK_COMBOS_EFFECTIVE = _expand_propconj_task_combos(_PROPCONJ_TASK_COMBOS_EFFECTIVE)


def _build_propconj_task(task_id, property_ids):
    properties = [dict(PROPCONJ_PROPERTIES[prop_id]) for prop_id in property_ids]
    label = " + ".join(prop["label"] for prop in properties)
    task = {
        "id": task_id,
        "trait": label,
        "property_ids": list(property_ids),
        "properties": properties,
        "difficulty": _PROPCONJ_TASK_DIFFICULTIES.get(task_id) or _default_propconj_difficulty(property_ids),
        "property_bank_schema": _PROPCONJ_PROPERTY_BANK_V2.get("schema") or "inline_fallback",
    }
    task["prompt"] = generate_propconj_prompt(task)
    return task


PROP_CONJ_TASKS = [
    _build_propconj_task(task_id, property_ids)
    for task_id, property_ids in _PROPCONJ_TASK_COMBOS_EFFECTIVE
]


def _load_macgyver_tasks():
    data_dir = Path(__file__).resolve().parent / "data"
    path = data_dir / "macgyver_tasks.json"
    with path.open("r", encoding="utf-8") as handle:
        base_payload = json.load(handle)
    payload = base_payload
    v2_path = data_dir / "macgyver_tasks_v2.json"
    if v2_path.exists():
        with v2_path.open("r", encoding="utf-8") as handle:
            v2_payload = json.load(handle)
        if isinstance(v2_payload, dict):
            merged_tasks = []
            overlays = v2_payload.get("task_overlays") if isinstance(v2_payload.get("task_overlays"), dict) else {}
            for task in base_payload.get("tasks") or []:
                task_copy = copy.deepcopy(task)
                overlay = overlays.get(task_copy.get("id")) or {}
                if isinstance(overlay, dict):
                    task_copy.update(overlay)
                task_copy.setdefault(
                    "expected_response_mode",
                    "unsolvable" if task_copy.get("unsolvable") else v2_payload.get("default_expected_response_mode", "solvable"),
                )
                task_copy.setdefault(
                    "task_subtype",
                    "MacGyverUnsolvable" if task_copy.get("expected_response_mode") == "unsolvable" else "MacGyverSolvable",
                )
                task_copy.setdefault("clarification_fields", [])
                task_copy.setdefault("boundary_expectation", {})
                task_copy["task_schema"] = v2_payload.get("schema", "macgyver_dual_axis_tasks_v2")
                merged_tasks.append(task_copy)
            for task in v2_payload.get("additional_tasks") or []:
                if isinstance(task, dict):
                    task_copy = copy.deepcopy(task)
                    task_copy.setdefault("task_schema", v2_payload.get("schema", "macgyver_dual_axis_tasks_v2"))
                    merged_tasks.append(task_copy)
            payload = {
                "schema": v2_payload.get("schema", base_payload.get("schema")),
                "plan_count": int(os.getenv(
                    "OPENROUTER_MACGYVER_OUTPUT_COUNT",
                    str(v2_payload.get("plan_count", base_payload.get("plan_count", MACGYVER_PLAN_COUNT))),
                )),
                "tasks": merged_tasks,
            }
    tasks = []
    for task in payload.get("tasks") or []:
        task_copy = copy.deepcopy(task)
        task_copy["prompt"] = generate_macgyver_prompt(task_copy, output_count=payload.get("plan_count", MACGYVER_PLAN_COUNT))
        tasks.append(task_copy)
    return tasks


MACGYVER_TASKS = _load_macgyver_tasks()


def _load_hypospace_tasks():
    path = Path(__file__).resolve().parent / "data" / "hypospace_tasks.json"
    with path.open("r", encoding="utf-8") as handle:
        base_payload = json.load(handle)
    payload = base_payload
    v2_path = Path(__file__).resolve().parent / "data" / "hypospace_tasks_v2.json"
    evidence_path = Path(__file__).resolve().parent / "data" / "evidence_packs_v2.json"
    if v2_path.exists():
        with v2_path.open("r", encoding="utf-8") as handle:
            v2_payload = json.load(handle)
        evidence_payload = {}
        if evidence_path.exists():
            with evidence_path.open("r", encoding="utf-8") as handle:
                evidence_payload = json.load(handle)
        base_by_id = {
            task.get("id"): task
            for task in base_payload.get("tasks") or []
            if task.get("id")
        }
        packs = evidence_payload.get("packs") or {}
        expanded_tasks = []
        for spec in v2_payload.get("tasks") or []:
            base_task = base_by_id.get(spec.get("base_task_id"))
            if not isinstance(base_task, dict):
                continue
            task_copy = copy.deepcopy(base_task)
            task_copy["id"] = spec.get("id") or task_copy.get("id")
            task_copy["base_task_id"] = spec.get("base_task_id")
            task_copy["task_schema"] = v2_payload.get("schema_version", "hypouse_space_tasks_v2")
            task_copy["task_subtype"] = spec.get("task_subtype", "EvidenceHypoSpace")
            task_copy["title"] = spec.get("title") or task_copy.get("title")
            if spec.get("scene_suffix"):
                task_copy["scene"] = f"{task_copy.get('scene', '')} {spec.get('scene_suffix')}".strip()
            if spec.get("goal_suffix"):
                task_copy["goal"] = f"{task_copy.get('goal', '')} {spec.get('goal_suffix')}".strip()
            if spec.get("minimal_cue"):
                task_copy["minimal_cue"] = spec.get("minimal_cue")
            task_copy["evidence_pack_id"] = spec.get("evidence_pack_id")
            evidence_pack = copy.deepcopy(packs.get(spec.get("evidence_pack_id")) or {})
            if evidence_pack:
                task_copy["evidence_pack"] = evidence_pack
            task_copy["requires_claim_ledger"] = bool(v2_payload.get("default_requires_claim_ledger", True))
            task_copy["support_boundary"] = {
                "evidence_pack_id": task_copy.get("evidence_pack_id"),
                "allowed_evidence_ids": _hypospace_evidence_ids(task_copy),
                "runtime_family": "HypoUseSpace",
            }
            expanded_tasks.append(task_copy)
        if expanded_tasks:
            payload = {
                "schema_version": v2_payload.get("schema_version", base_payload.get("schema_version")),
                "output_count": v2_payload.get("output_count", base_payload.get("output_count", HYPOUSESPACE_OUTPUT_COUNT)),
                "tasks": expanded_tasks,
            }
    output_count = int(payload.get("output_count", HYPOUSESPACE_OUTPUT_COUNT))
    tasks = []
    for task in payload.get("tasks") or []:
        task_copy = copy.deepcopy(task)
        valid_count = len(task_copy.get("valid_hypotheses") or [])
        task_output_count = min(output_count, valid_count) if valid_count else output_count
        task_copy["output_count"] = task_output_count
        task_copy["prompt"] = generate_hypospace_prompt(task_copy, output_count=task_output_count)
        tasks.append(task_copy)
    return tasks


HYPOUSESPACE_TASKS = _load_hypospace_tasks()


def _load_gcw_tasks():
    path = Path(__file__).resolve().parent / "data" / "gcw_cards.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    ladders = _load_gcw_constraint_ladders()
    beat_count = int(payload.get("beat_count", GCW_BEAT_COUNT))
    tasks = []
    for task_id, card in sorted((payload.get("cards") or {}).items()):
        task = dict(card)
        task["id"] = task_id
        task["beat_count"] = beat_count
        task["constraint_profile"] = _resolve_gcw_constraint_profile(task, ladders=ladders)
        task["constraint_level"] = task["constraint_profile"].get("level_id")
        task["constraint_ladder"] = (ladders.get("cards") or {}).get(task_id) or {}
        task["prompt"] = generate_gcw_v2_prompt(task, beat_count=beat_count)
        tasks.append(task)
    return tasks


GCW_TASKS = _load_gcw_tasks()


def _load_cjst_tasks():
    path = Path(__file__).resolve().parent / "data" / "cjst_scenario_cards.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cards = payload.get("cards") or {}
    tasks = []
    for task_id, card in sorted(cards.items()):
        scenario_text = card.get("premise") or card.get("scenario_text") or ""
        tasks.append({
            "id": task_id,
            "scenario": card.get("label") or task_id,
            "scenario_text": scenario_text,
            "legacy_jst_id": card.get("legacy_jst_id"),
            "scenario_card": card,
            "prompt": generate_cjst_prompt(scenario_text, output_count=payload.get("output_count", CJST_OUTPUT_COUNT)),
        })
    return tasks


CJST_TASKS = _load_cjst_tasks()


def _load_neocoder_v3_overlay():
    path = Path(__file__).resolve().parent / "data" / "neocoder_v3_task_overlay.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("tasks") or {}


def _load_neocoder_tasks():
    path = Path(__file__).resolve().parent / "data" / "neocoder_tasks_v2.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    overlay_by_id = _load_neocoder_v3_overlay()
    output_count = int(payload.get("output_count", NEOCODER_OUTPUT_COUNT))
    tasks = []
    for task in payload.get("tasks") or []:
        task_copy = copy.deepcopy(task)
        overlay = overlay_by_id.get(task_copy.get("id"))
        if isinstance(overlay, dict):
            task_copy.update(copy.deepcopy(overlay))
            task_copy["task_overlay_version"] = "neocoder_task_overlay"
            task_copy["test_visibility_policy"] = "public_examples_hidden_scoring_tests"
        task_copy.setdefault("task_schema", payload.get("schema_version", "neocoder_tasks_v2"))
        task_copy["output_count"] = output_count
        task_copy["prompt"] = generate_neocoder_prompt(task_copy, output_count=output_count)
        tasks.append(task_copy)
    return tasks


NEOCODER_TASKS = _load_neocoder_tasks()


def _load_closed_world_fact_tasks():
    path = Path(__file__).resolve().parent / "data" / "closed_world_fact_cards_v2.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    facts_by_id = {
        str(fact.get("id")): fact
        for fact in ((payload.get("database") or {}).get("facts") or [])
        if fact.get("id")
    }
    output_count = int(payload.get("output_count", CLOSED_WORLD_FACT_OUTPUT_COUNT))
    tasks = []
    for task in payload.get("tasks") or []:
        task_copy = copy.deepcopy(task)
        task_copy.setdefault("task_schema", payload.get("schema_version", "closed_world_fact_cards_v2"))
        task_copy["database"] = copy.deepcopy(payload.get("database") or {})
        support_ids = list((task_copy.get("support_boundary") or {}).get("evidence_ids") or task_copy.get("required_evidence_ids") or [])
        task_copy["support_facts"] = [
            copy.deepcopy(facts_by_id[str(fact_id)])
            for fact_id in support_ids
            if str(fact_id) in facts_by_id
        ]
        task_copy["output_count"] = output_count
        task_copy["prompt"] = generate_closed_world_fact_prompt(task_copy, output_count=output_count)
        tasks.append(task_copy)
    return tasks


CLOSED_WORLD_FACT_TASKS = _load_closed_world_fact_tasks()


def _load_analogy_transfer_tasks():
    path = Path(__file__).resolve().parent / "data" / "analogy_tasks_v2.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    clusters = {
        str(cluster.get("cluster_id")): cluster
        for cluster in payload.get("clusters") or []
        if cluster.get("cluster_id")
    }
    output_count = int(payload.get("output_count", ANALOGY_TRANSFER_OUTPUT_COUNT))
    tasks = []
    for task in payload.get("tasks") or []:
        task_copy = copy.deepcopy(task)
        cluster = copy.deepcopy(clusters.get(str(task_copy.get("cluster_id"))) or {})
        if not cluster:
            continue
        source_ids = [fact.get("id") for fact in cluster.get("source_facts") or [] if fact.get("id")]
        target_ids = [fact.get("id") for fact in cluster.get("target_facts") or [] if fact.get("id")]
        support_boundary = cluster.setdefault("support_boundary", {})
        support_boundary.setdefault("evidence_ids", source_ids + target_ids)
        support_boundary.setdefault("policy", "Use only the listed source and target facts.")
        task_copy.setdefault("task_schema", payload.get("schema_version", "analogy_tasks_v2"))
        task_copy["cluster"] = cluster
        task_copy["source_domain"] = cluster.get("source_domain")
        task_copy["target_domain"] = cluster.get("target_domain")
        task_copy["support_boundary"] = copy.deepcopy(support_boundary)
        task_copy["known_entities"] = list(cluster.get("known_entities") or [])
        task_copy["required_mapping_ids"] = [
            mapping.get("mapping_id")
            for mapping in cluster.get("gold_mappings") or []
            if mapping.get("mapping_id")
        ]
        task_copy["required_limit_ids"] = [
            limit.get("limit_id")
            for limit in cluster.get("required_limits") or []
            if limit.get("limit_id")
        ]
        task_copy["forbidden_transfer_ids"] = [
            transfer.get("id")
            for transfer in cluster.get("forbidden_transfers") or []
            if transfer.get("id")
        ]
        task_copy["output_count"] = output_count
        task_copy["prompt"] = generate_analogy_transfer_prompt(task_copy, output_count=output_count)
        tasks.append(task_copy)
    return tasks


ANALOGY_TRANSFER_TASKS = _load_analogy_transfer_tasks()


_AUGMENTATION_CONTEXTS = [
    {
        "label": "home setting",
        "scene": "The same constraints hold in a quiet home setting; no extra tools, entities, or facts are available.",
        "motif": "threshold",
    },
    {
        "label": "school setting",
        "scene": "The same constraints hold during a classroom demonstration; no extra tools, entities, or facts are available.",
        "motif": "chalk",
    },
    {
        "label": "workshop setting",
        "scene": "The same constraints hold on a small workshop bench; no extra tools, entities, or facts are available.",
        "motif": "bench",
    },
    {
        "label": "outdoor setting",
        "scene": "The same constraints hold in a sheltered outdoor area; no extra tools, entities, or facts are available.",
        "motif": "wind",
    },
]


def _expanded_target_count(tasks, factor=SECTION12_PROMPT_EXPANSION_FACTOR):
    return len(tasks) * max(1, factor)


def _with_augmented_common_fields(task, *, source_task_id, augmentation_round, context):
    task["source_task_id"] = source_task_id
    task["augmentation_round"] = augmentation_round
    task["augmentation_context"] = context["label"]
    return task


def _expand_macgyver_tasks(tasks):
    result = copy.deepcopy(tasks)
    target_count = _expanded_target_count(tasks)
    next_index = len(result) + 1
    for round_index in range(1, max(1, SECTION12_PROMPT_EXPANSION_FACTOR)):
        for task in tasks:
            if len(result) >= target_count:
                return result
            context = _AUGMENTATION_CONTEXTS[(round_index - 1) % len(_AUGMENTATION_CONTEXTS)]
            clone = copy.deepcopy(task)
            source_id = str(task.get("id"))
            clone["id"] = f"mg_{next_index:03d}"
            clone["title"] = f"{task.get('title', source_id)} ({context['label']})"
            clone["scene"] = f"{task.get('scene', '')} {context['scene']}".strip()
            _with_augmented_common_fields(
                clone,
                source_task_id=source_id,
                augmentation_round=round_index,
                context=context,
            )
            clone["prompt"] = generate_macgyver_prompt(clone, output_count=MACGYVER_PLAN_COUNT)
            result.append(clone)
            next_index += 1
    return result


def _expand_hypospace_tasks(tasks):
    result = copy.deepcopy(tasks)
    target_count = _expanded_target_count(tasks)
    next_index = len(result) + 1
    for round_index in range(1, max(1, SECTION12_PROMPT_EXPANSION_FACTOR)):
        for task in tasks:
            if len(result) >= target_count:
                return result
            context = _AUGMENTATION_CONTEXTS[(round_index - 1) % len(_AUGMENTATION_CONTEXTS)]
            clone = copy.deepcopy(task)
            source_id = str(task.get("id"))
            clone["id"] = f"hs_aug_{next_index:03d}"
            clone["title"] = f"{task.get('title', source_id)} ({context['label']})"
            clone["scene"] = f"{task.get('scene', '')} {context['scene']}".strip()
            _with_augmented_common_fields(
                clone,
                source_task_id=source_id,
                augmentation_round=round_index,
                context=context,
            )
            clone["prompt"] = generate_hypospace_prompt(
                clone,
                output_count=clone.get("output_count", HYPOUSESPACE_OUTPUT_COUNT),
            )
            result.append(clone)
            next_index += 1
    return result


def _expand_gcw_tasks(tasks):
    result = copy.deepcopy(tasks)
    target_count = _expanded_target_count(tasks)
    next_index = len(result) + 1
    for round_index in range(1, max(1, SECTION12_PROMPT_EXPANSION_FACTOR)):
        for task in tasks:
            if len(result) >= target_count:
                return result
            context = _AUGMENTATION_CONTEXTS[(round_index - 1) % len(_AUGMENTATION_CONTEXTS)]
            clone = copy.deepcopy(task)
            source_id = str(task.get("id"))
            clone["id"] = f"gcw_{next_index:03d}"
            clone["title"] = f"{task.get('title', source_id)} ({context['label']})"
            fact_sheet = copy.deepcopy(task.get("fact_sheet") or [])
            if isinstance(fact_sheet, list):
                fact_sheet.append({
                    "id": f"AUG{round_index}",
                    "type": "setting",
                    "text": context["scene"],
                    "keywords": [context["label"].split()[0], "same constraints"],
                })
                clone["fact_sheet"] = fact_sheet
            motifs = list(clone.get("motifs") or [])
            if context["motif"] not in motifs:
                motifs.append(context["motif"])
            clone["motifs"] = motifs
            _with_augmented_common_fields(
                clone,
                source_task_id=source_id,
                augmentation_round=round_index,
                context=context,
            )
            clone["prompt"] = generate_gcw_v2_prompt(
                clone,
                beat_count=clone.get("beat_count", GCW_BEAT_COUNT),
            )
            result.append(clone)
            next_index += 1
    return result


def _expand_cjst_tasks(tasks):
    result = copy.deepcopy(tasks)
    target_count = _expanded_target_count(tasks)
    next_index = len(result) + 1
    for round_index in range(1, max(1, SECTION12_PROMPT_EXPANSION_FACTOR)):
        for task in tasks:
            if len(result) >= target_count:
                return result
            context = _AUGMENTATION_CONTEXTS[(round_index - 1) % len(_AUGMENTATION_CONTEXTS)]
            clone = copy.deepcopy(task)
            source_id = str(task.get("id"))
            clone["id"] = f"cjst_{next_index:03d}"
            clone["scenario"] = f"{task.get('scenario', source_id)} ({context['label']})"
            scenario_text = str(task.get("scenario_text") or "")
            clone["scenario_text"] = f"{scenario_text}, while the situation occurs in the {context['label']}"
            _with_augmented_common_fields(
                clone,
                source_task_id=source_id,
                augmentation_round=round_index,
                context=context,
            )
            clone["prompt"] = generate_cjst_prompt(clone["scenario_text"], output_count=CJST_OUTPUT_COUNT)
            result.append(clone)
            next_index += 1
    return result


def _expand_neocoder_tasks(tasks):
    result = copy.deepcopy(tasks)
    target_count = _expanded_target_count(tasks)
    next_index = len(result) + 1
    for round_index in range(1, max(1, SECTION12_PROMPT_EXPANSION_FACTOR)):
        for task in tasks:
            if len(result) >= target_count:
                return result
            context = _AUGMENTATION_CONTEXTS[(round_index - 1) % len(_AUGMENTATION_CONTEXTS)]
            clone = copy.deepcopy(task)
            source_id = str(task.get("id"))
            clone["id"] = f"neo_aug_{next_index:03d}"
            clone["title"] = f"{task.get('title', source_id)} ({context['label']})"
            clone["problem_statement"] = (
                f"{task.get('problem_statement', '')}\n\n"
                f"Benchmark variant: solve this as an independent {context['label']} version. "
                "The function signature, input contract, output contract, denied techniques, and tests are unchanged."
            ).strip()
            _with_augmented_common_fields(
                clone,
                source_task_id=source_id,
                augmentation_round=round_index,
                context=context,
            )
            clone["prompt"] = generate_neocoder_prompt(
                clone,
                output_count=clone.get("output_count", NEOCODER_OUTPUT_COUNT),
            )
            result.append(clone)
            next_index += 1
    return result


def _expand_closed_world_fact_tasks(tasks):
    result = copy.deepcopy(tasks)
    target_count = _expanded_target_count(tasks)
    next_index = len(result) + 1
    for round_index in range(1, max(1, SECTION12_PROMPT_EXPANSION_FACTOR)):
        for task in tasks:
            if len(result) >= target_count:
                return result
            context = _AUGMENTATION_CONTEXTS[(round_index - 1) % len(_AUGMENTATION_CONTEXTS)]
            clone = copy.deepcopy(task)
            source_id = str(task.get("id"))
            clone["id"] = f"cwf_aug_{next_index:03d}"
            clone["question"] = (
                f"{task.get('question', '')} "
                f"Treat this as an independent {context['label']} variant and use only the closed evidence records."
            ).strip()
            _with_augmented_common_fields(
                clone,
                source_task_id=source_id,
                augmentation_round=round_index,
                context=context,
            )
            clone["prompt"] = generate_closed_world_fact_prompt(
                clone,
                output_count=clone.get("output_count", CLOSED_WORLD_FACT_OUTPUT_COUNT),
            )
            result.append(clone)
            next_index += 1
    return result


def _expand_analogy_transfer_tasks(tasks):
    result = copy.deepcopy(tasks)
    target_count = _expanded_target_count(tasks)
    next_index = len(result) + 1
    for round_index in range(1, max(1, SECTION12_PROMPT_EXPANSION_FACTOR)):
        for task in tasks:
            if len(result) >= target_count:
                return result
            context = _AUGMENTATION_CONTEXTS[(round_index - 1) % len(_AUGMENTATION_CONTEXTS)]
            clone = copy.deepcopy(task)
            source_id = str(task.get("id"))
            clone["id"] = f"analogy_aug_{next_index:03d}"
            clone["instruction"] = (
                f"{task.get('instruction', '')} "
                f"Evaluate this as an independent {context['label']} variant; evidence boundaries are unchanged."
            ).strip()
            _with_augmented_common_fields(
                clone,
                source_task_id=source_id,
                augmentation_round=round_index,
                context=context,
            )
            clone["prompt"] = generate_analogy_transfer_prompt(
                clone,
                output_count=clone.get("output_count", ANALOGY_TRANSFER_OUTPUT_COUNT),
            )
            result.append(clone)
            next_index += 1
    return result


MACGYVER_TASKS = _expand_macgyver_tasks(MACGYVER_TASKS)
HYPOUSESPACE_TASKS = _expand_hypospace_tasks(HYPOUSESPACE_TASKS)
GCW_TASKS = _expand_gcw_tasks(GCW_TASKS)
CJST_TASKS = _expand_cjst_tasks(CJST_TASKS)
NEOCODER_TASKS = _expand_neocoder_tasks(NEOCODER_TASKS)
CLOSED_WORLD_FACT_TASKS = _expand_closed_world_fact_tasks(CLOSED_WORLD_FACT_TASKS)
ANALOGY_TRANSFER_TASKS = _expand_analogy_transfer_tasks(ANALOGY_TRANSFER_TASKS)

_CDAT_CUES = [
    "unity", "river", "music", "journey", "stone",
    "laughter", "bridge", "shadow", "memory", "signal",
    "garden", "machine", "ocean", "pattern", "flame",
    "language", "shelter", "horizon", "thread", "magnet",
    "door", "forest", "window", "clock", "island",
    "market", "engine", "mirror", "harbor", "seed",
    "circle", "needle", "lantern", "weather", "map",
    "root", "fabric", "bell", "valley", "book",
    "cloud", "wheel", "knife", "basket", "ladder",
    "feather", "coin", "storm", "vessel", "mask",
    "path", "cave", "harvest", "spark", "anchor",
    "shell", "tower", "meadow", "cradle", "drum",
    "wire", "planet", "nest", "fountain", "gate",
    "desert", "hammer", "camera", "mirror", "balance",
    "rope", "lantern", "garden", "river", "archive",
    "compass", "filter", "threshold", "orbit", "needle",
    "echo", "harvest", "quilt", "station", "canal",
    "branch", "reservoir", "battery", "library", "seedbank",
    "firewall", "transit", "trial", "rehearsal", "simulation",
    "signal", "shelter", "machine", "pattern", "horizon",
]

_FF_SEEDS = [
    "table", "bear", "candle", "window", "garden",
    "river", "ladder", "mirror", "apple", "engine",
    "forest", "island", "jacket", "kettle", "needle",
    "pillow", "temple", "umbrella", "valley", "zebra",
    "basket", "button", "cloud", "door", "feather",
    "guitar", "hammer", "ink", "jewel", "kite",
    "leaf", "marble", "notebook", "orange", "pencil",
    "quilt", "radio", "shell", "ticket", "vase",
    "wallet", "yarn", "anchor", "bridge", "coin",
    "drum", "envelope", "fountain", "glove", "harbor",
    "lamp", "map", "nest", "ocean", "paint",
    "rope", "spoon", "towel", "village", "wheel",
    "bottle", "camera", "dust", "fabric", "grain",
    "helmet", "lantern", "magnet", "paper", "saddle",
    "tool", "wire", "beacon", "cabin", "desert",
    "field", "gate", "hinge", "meadow", "orbit",
    "path", "screen", "tile", "branch", "cork",
    "drawer", "filter", "mug", "ribbon", "stone",
    "tray", "blanket", "bucket", "comb", "scarf",
    "sock", "brick", "chair", "book", "key",
]

_CDAT_CUES = list(dict.fromkeys(_CDAT_CUES + [
    "workshop", "gallery", "tunnel", "orchard", "courtyard",
    "beacon", "ledger", "hinge", "terrace", "laboratory",
    "platform", "corridor", "companion", "festival", "weather",
    "artifact", "granary", "village", "cathedral", "workbench",
]))


CREATIVITY_PROMPTS = {
    "UUT": [
        {"id": task_id, "item": item, "prompt": generate_uut_prompt(item)}
        for task_id, item in UUT_ITEMS
    ],
    "JST": [
        {
            "id": task_id,
            "scenario": scenario,
            "scenario_text": scenario_text,
            "prompt": generate_jst_prompt(scenario_text),
        }
        for task_id, scenario, scenario_text in JST_SCENARIOS
    ],
    "CJST": CJST_TASKS,
    "Instances": [
        {"id": task_id, "trait": label, "prompt": generate_instances_prompt(prompt_trait)}
        for task_id, label, prompt_trait in INSTANCE_TRAITS
    ],
    "PropConj": PROP_CONJ_TASKS,
    "MacGyver": MACGYVER_TASKS,
    "HypoUseSpace": HYPOUSESPACE_TASKS,
    "GCW": GCW_TASKS,
    "NeoCoder": NEOCODER_TASKS,
    "ClosedWorldFact": CLOSED_WORLD_FACT_TASKS,
    "AnalogyTransfer": ANALOGY_TRANSFER_TASKS,

    "DAT": [
        {"id": f"dat_{index:02d}", "prompt": generate_dat_prompt()}
        for index in range(1, 20 * SECTION12_PROMPT_EXPANSION_FACTOR + 1)
    ],

    "CDAT": [
        {"id": f"cdat_{index:02d}", "cue": cue, "prompt": generate_cdat_prompt(cue)}
        for index, cue in enumerate(_CDAT_CUES[:20 * SECTION12_PROMPT_EXPANSION_FACTOR], start=1)
    ],

    
    
    
    
    
    
    "FF": [
        {"id": f"ff_{index:02d}", "seed": seed, "prompt": generate_ff_prompt(seed)}
        for index, seed in enumerate(_FF_SEEDS[:20 * SECTION12_PROMPT_EXPANSION_FACTOR], start=1)
    ],
}


def get_all_prompts():
    return CREATIVITY_PROMPTS
