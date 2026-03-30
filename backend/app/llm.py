"""OpenAI / LLM calls: extraction, question generation, error messages, query answers.

All prompts combine:
1. Engine-level rules (universal, in code)
2. Form-level context (from form metadata: system_prompt + field descriptions)
"""

import json

from langfuse.openai import OpenAI

from app.hierarchy import get_field, has_options, get_valid_dropdown_values
from app.validation import validate_field

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# === Shared helpers ===

def _build_fields_context(form, collected_data):
    """Build a dynamic description of all fields with current state.
    Reused across all LLM calls.
    """
    lines = []
    for f in form["fields"]:
        desc = f"- {f['field_id']} ({f['label']}): type={f['type']}"
        if f.get("required"):
            desc += ", required"
        if f.get("validation_rules"):
            rules = dict(f["validation_rules"])
            rules.pop("conditional_rules", None)
            if rules:
                desc += f", rules={json.dumps(rules)}"
        if f.get("parent_field_id"):
            desc += f", child of {f['parent_field_id']}"
        if has_options(f):
            valid_opts = get_valid_dropdown_values(form, f["field_id"], collected_data)
            if valid_opts:
                desc += f", valid_options={valid_opts}"
        if f.get("conditional_rules"):
            desc += f", conditional_rules={json.dumps(f['conditional_rules'])}"
        lines.append(desc)
    return "\n".join(lines)


def _get_form_prompt(form):
    """Get the form-specific system prompt from metadata.
    Returns empty string if not defined.
    """
    return form.get("system_prompt", "")


def _filter_options_by_collected(form, field_id, options, collected_data):
    """Filter dropdown options by validating all collected fields against each option.

    For each option (e.g., country=India), simulate setting it and re-validate
    all collected fields that have conditional rules depending on this field.
    If any collected field fails, exclude the option.
    """
    if not options or not collected_data:
        return options

    valid_options = []
    for opt in options:
        test_data = {**collected_data, field_id: opt}
        option_valid = True
        for field in form["fields"]:
            fid = field["field_id"]
            if fid == field_id or fid not in collected_data:
                continue
            cond_rules = field.get("validation_rules", {}).get("conditional_rules", [])
            if not any(r.get("if", {}).get("field") == field_id for r in cond_rules):
                continue
            is_valid, _ = validate_field(form, fid, collected_data[fid], test_data)
            if not is_valid:
                option_valid = False
                break
        if option_valid:
            valid_options.append(opt)
    return valid_options


# === Extraction ===

EXTRACTION_RULES = """RULES:

1. HIGH-CONFIDENCE PATTERN OVERRIDE (HIGHEST PRIORITY):
   These patterns ALWAYS map to their field, REGARDLESS of which field is currently being asked:
   - Email pattern (contains @ and domain) → email field
   - Phone pattern (7-15 digits, optional +) → phone field
   - Dropdown close match: if input closely matches a valid_option of ANY dropdown field (e.g., "saving" ≈ "Savings", "curren" ≈ "Current"), map to THAT dropdown field
   - Known structured patterns defined in FORM CONTEXT (e.g., ward patterns) → respective field
   If input matches a high-confidence pattern, map it immediately. Do NOT force it to the currently asking field.

2. BEST-FIT MATCHING (NOT currently-asking-first):
   - Compare input against ALL uncollected fields
   - Choose the field with the STRONGEST semantic + type match
   - Currently asking field gets priority ONLY when match strength is equal
   - Example: asking for "name", user says "adhi@gmail.com" → email (pattern match wins)
   - Example: asking for "name", user says "saving" → account_type="Savings" (dropdown fuzzy match wins over text field)
   - Example: asking for "name", user says "adhi" → full_name (semantic match, fits asked field)

3. TYPE MATCH (STRICT):
   - number → must be numeric (e.g., 25)
   - text → must be meaningful text
   - dropdown → must be one of valid_options (allow fuzzy: "kerla" → "Kerala", "saving" → "Savings")

4. SEMANTIC MATCH:
   - Assign value only to the field it logically belongs to
   - Example: Asking age, User says "India" → map to country, NOT age
   - Example: Asking age, User says "adhi" → NOT a number, DO NOT guess age=18

5. FUZZY MATCHING:
   - Normalize casing and spacing to match known options
   - Allow minor spelling variations for known values (defined in FORM CONTEXT)
   - "kerla" → "Kerala", "banglore" → "Bangalore", "ward 200" → "Ward200"
   - ONLY fuzzy-match against values that EXIST in the form. NEVER invent new values

6. MULTI-VALUE EXTRACTION:
   - User may provide multiple values in one message
   - Extract ALL recognizable values, each to its correct field
   - Example: "adhi 20 india" → name + age + country
   - Example: "adhi@g.com 9876543210" → email + phone

7. NO GUESSING / NO HALLUCINATION:
   - NEVER infer or generate values the user did not say
   - NEVER convert unrelated text to numbers
   - "adhi" CANNOT become age=18

8. UNCERTAINTY (STRICT):
   - Return "_uncertain": true ONLY when ZERO fields match
   - If at least ONE field was extracted → return that field WITHOUT _uncertain
   - NEVER return _uncertain alongside extracted fields

9. UPDATE INTENT:
   - User changes existing value (e.g., "change age to 30") → "_intent": "update"

10. DELETE INTENT:
    - User removes a field (e.g., "delete age") → "_delete": ["field_id", ...]

11. QUERY INTENT:
    - User asks a question → "_query": "<question as-is>"
    - Can combine with data operations

12. STRICT OUTPUT:
    - Return ONLY confident mappings
    - For numbers, return actual numbers not strings
    - Precision > Recall (better to miss than to be wrong)

13. SENSITIVE FIELDS:
    - Password fields → always return ""

14. CONVERSATIONAL INTENTS:
    - If user says ONLY "yes", "yeah", "yep", "sure", "ok", "okay", "correct", "right", "proceed", "go ahead", "continue"
      → Look at the conversation history to understand WHAT the user is confirming
      → If confirming a previously suggested value, extract that value to the correct field
      → If no value to confirm, return "_confirm": true
    - If user says ONLY "no", "nope", "nah", "wrong", "not that", "cancel"
      → return "_deny": true (user rejects the last suggestion or wants to re-answer)
    - If user says ONLY "skip", "later", "next", "move on"
      → return "_skip": true (user wants to skip the current field)
    - If user says ONLY "wait", "hold on", "pause", "stop"
      → return "_wait": true
    - These intents should ONLY be set when the message is PURELY conversational with no field data

Return ONLY a JSON object. No explanation."""


def call_openai_extract(user_message, form, collected_data, currently_asking=None, currently_asking_field=None, messages_history=None):
    """Extract field values from user message."""
    print(f"    [llm] call_openai_extract (currently_asking={currently_asking})")
    fields_context = _build_fields_context(form, collected_data)
    form_prompt = _get_form_prompt(form)


    asking_context = ""
    if currently_asking and currently_asking_field:
        label = currently_asking_field.get("label", currently_asking)
        ftype = currently_asking_field.get("type", "text")
        asking_context = f'\nCURRENTLY ASKING: "{label}" (field_id: {currently_asking}, type: {ftype})\n'
    else:
        asking_context = "\nCURRENTLY ASKING: None\n"

    system_prompt = f"""You are a STRICT data extraction assistant.

Form: {form['title']}
{f"FORM CONTEXT: {form_prompt}" if form_prompt else ""}

Fields:
{fields_context}

Already collected: {json.dumps(collected_data)}
{asking_context}
{EXTRACTION_RULES}"""

    # Build messages with conversation history for context
    llm_messages = [{"role": "system", "content": system_prompt}]
    if messages_history:
        # Include last 10 messages for context (skip the current user message, it's added separately)
        recent = messages_history[-11:-1] if len(messages_history) > 1 else []
        for msg in recent:
            llm_messages.append({"role": msg["role"], "content": msg["content"]})
    llm_messages.append({"role": "user", "content": user_message})

    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=llm_messages,
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {}


# === Next Question ===

QUESTION_RULES = """YOUR RESPONSE MUST FOLLOW THIS STRUCTURE:

1. ACKNOWLEDGE (if LAST ACTION exists):
   - Briefly confirm what was just saved/updated/inferred/deleted
   - Warm tone: "Great!", "Perfect!", "Got it!", "Noted!"
   - ONE short sentence max

2. REJECTED VALUES (MANDATORY — do NOT skip this):
   - If LAST ACTION contains any REJECTED items, you MUST mention them
   - Explain briefly why each was rejected
   - Show valid alternatives if provided
   - Example: "However, Ward2 isn't available for Alappuzha — you can choose from Ward1 or Ward6."
   - NEVER silently skip rejected values

3. ASK THE NEXT QUESTION:
   - Ask for ONE field only
   - Be natural and conversational
   - For dropdowns: list options in a friendly way
   - For password: mention the requirements clearly

FORMATTING:
- Line break between acknowledgment and question
- Max 4 sentences total
- No bullet points, no raw field names
- Sound human, not robotic

LANGUAGE (CRITICAL):
- NEVER use internal terms: "dropdown", "options list", "field_id", "valid_options", "hierarchy", "metadata"
- Say "you can choose from" or "would you prefer" — NOT "select from dropdown options"
- Use natural everyday language as if talking to a real person"""


def call_openai_next_question(form, collected_data, missing_fields, last_action=None, messages_history=None):
    """Generate the next question to ask the user."""
    print(f"    [llm] call_openai_next_question")
    print(f"    [llm]   missing_fields: {missing_fields}")
    print(f"    [llm]   last_action: {last_action}")
    fields_info = []
    for fid in missing_fields:
        field = get_field(form, fid)
        if field:
            info = f"- {field['field_id']}: {field['label']} (type: {field['type']})"
            if has_options(field):
                valid_opts = get_valid_dropdown_values(form, fid, collected_data)
                if valid_opts:
                    valid_opts = _filter_options_by_collected(form, fid, valid_opts, collected_data)
                if valid_opts:
                    info += f" [options: {', '.join(valid_opts)}]"
                else:
                    continue
            if field["type"] == "password":
                pw_rules = field.get("validation_rules", {})
                reqs_text = pw_rules.get("regex_description", "a secure password")
                info += f" [PASSWORD FIELD - ask user to set a password with {reqs_text}. Do NOT suggest any password.]"
            fields_info.append(info)

    if not fields_info:
        return "All fields have been collected!"

    # Build last action context
    action_context = ""
    if last_action:
        parts = []
        stored = last_action.get("stored", {})
        if stored:
            items = ", ".join(f"{k}='{v}'" for k, v in stored.items())
            parts.append(f"User just provided: {items}. Acknowledge briefly.")

        auto_filled = last_action.get("auto_filled", {})
        if auto_filled:
            items = ", ".join(f"{k}='{v}'" for k, v in auto_filled.items())
            parts.append(f"Auto-filled (only one option): {items}. Mention to user.")

        inferred = last_action.get("inferred", {})
        if inferred:
            items = ", ".join(f"{k}='{v}'" for k, v in inferred.items())
            parts.append(f"System inferred: {items}. Mention naturally.")

        updated = last_action.get("updated", {})
        if updated:
            items = ", ".join(f"{k}='{v}'" for k, v in updated.items())
            parts.append(f"User updated: {items}. Confirm briefly.")

        deleted = last_action.get("deleted", [])
        if deleted:
            items = ", ".join(deleted)
            parts.append(f"User deleted: {items}. Confirm deletion briefly.")

        rejected = last_action.get("rejected", [])
        if rejected:
            for r in rejected:
                parts.append(
                    f"REJECTED: User tried to set {r['field']}='{r['value']}' but it was rejected: {r['reason']}. "
                    f"Briefly acknowledge this and explain why."
                )

        unanswered = last_action.get("unanswered_field")
        if unanswered:
            field = get_field(form, unanswered)
            label = field["label"] if field else unanswered
            parts.append(f"User did NOT answer '{label}'. Remind them and ask for it now.")

        if parts:
            action_context = "\n\nLAST ACTION:\n" + "\n".join(f"- {p}" for p in parts)

    system_prompt = f"""You are a warm , short , professional form assistant.

Form: {form['title']}

Already collected: {json.dumps(collected_data)}

Missing fields (ask the FIRST one that makes sense):
{chr(10).join(fields_info)}
{action_context}

{QUESTION_RULES}

IMPORTANT:
- The options listed above for each field are already filtered and validated. Present them EXACTLY as shown — do NOT add or remove any options. They are the only valid choices.
- Do NOT fabricate or invent rejection messages. If LAST ACTION does not contain any REJECTED items, do NOT say anything was rejected or invalid. All stored values are already validated and correct — do not question them.
- If a field was successfully stored (in LAST ACTION "stored" or in "Already collected"), it passed all validation. Do NOT warn the user about it.

Return ONLY the message text."""

    # Build messages with conversation history for continuity
    llm_messages = [{"role": "system", "content": system_prompt}]
    if messages_history:
        recent = messages_history[-6:]
        for msg in recent:
            llm_messages.append({"role": msg["role"], "content": msg["content"]})

    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=llm_messages,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


# === Error Message ===

def call_openai_error_message(form, field_errors, user_message, collected_data, missing_fields=None, last_action=None, messages_history=None):
    """Generate a unified response: explain rejections + acknowledge stored fields + ask next question."""
    print(f"    [llm] call_openai_error_message")
    print(f"    [llm]   field_errors: {field_errors}")
    print(f"    [llm]   last_action: {last_action}")

    # Enrich errors with field labels and valid options
    enriched_errors = []
    for err in field_errors:
        enriched = dict(err)
        fid = err.get("field_id", "")
        field = get_field(form, fid) if fid and fid not in ("hierarchy", "dependency_note", "batch_note") else None
        if field:
            enriched["field_label"] = field.get("label", fid)
            if has_options(field):
                valid_opts = get_valid_dropdown_values(form, fid, collected_data)
                if valid_opts:
                    enriched["valid_options"] = valid_opts
            if field.get("parent_field_id"):
                parent = get_field(form, field["parent_field_id"])
                parent_val = collected_data.get(field["parent_field_id"])
                if parent and parent_val:
                    enriched["parent_field"] = parent.get("label", field["parent_field_id"])
                    enriched["parent_value"] = parent_val
        enriched_errors.append(enriched)

    # Build next question context
    next_field_info = ""
    if missing_fields:
        for fid in missing_fields:
            field = get_field(form, fid)
            if field:
                info = f"{field['label']} (type: {field['type']})"
                if has_options(field):
                    valid_opts = get_valid_dropdown_values(form, fid, collected_data)
                    if valid_opts:
                        valid_opts = _filter_options_by_collected(form, fid, valid_opts, collected_data)
                    if valid_opts:
                        info += f" — choices: {', '.join(valid_opts)}"
                    else:
                        continue
                next_field_info = info
                break
        if next_field_info:
            next_field_info = f"\n\nNEXT FIELD TO ASK: {next_field_info}"

    # Build stored/acknowledged context
    stored_context = ""
    if last_action:
        parts = []
        if last_action.get("stored"):
            items = ", ".join(f"{k}='{v}'" for k, v in last_action["stored"].items())
            parts.append(f"Successfully stored: {items}")
        if last_action.get("inferred"):
            items = ", ".join(f"{k}='{v}'" for k, v in last_action["inferred"].items())
            parts.append(f"Auto-inferred: {items}")
        if last_action.get("deleted"):
            parts.append(f"Deleted: {', '.join(last_action['deleted'])}")
        if last_action.get("rejected"):
            for r in last_action["rejected"]:
                parts.append(f"Rejected {r['field']}='{r['value']}': {r['reason']}")
        if parts:
            stored_context = "\n\nSUCCESSFULLY PROCESSED:\n" + "\n".join(f"- {p}" for p in parts)

    system_prompt = f"""You are a friendly, concise form assistant. The user provided data, some of which was rejected.

Form: {form['title']}

Already collected: {json.dumps(collected_data)}
User said: "{user_message}"

Rejected fields:
{json.dumps(enriched_errors, indent=2)}
{stored_context}
{next_field_info}

GENERATE A SINGLE UNIFIED RESPONSE with this structure:

1. ACKNOWLEDGE stored fields (if any were successfully saved) — one short sentence

2. EXPLAIN REJECTIONS:
   - If a rejected field has "ambiguous_source", ALWAYS mention it first — this is the field that determined the possible values.
     Example: if ambiguous_source is Ward=Ward200 and triggered_by is Country=India,China:
     → Say "Ward200 belongs to India or China" FIRST, then explain the validation failure for each
   - If multiple rejections are CAUSALLY LINKED, explain the ROOT CAUSE first, then dependent failures
   - If rejections are independent, explain each briefly
   - Suggest HOW to fix it (e.g., "you could delete the ward to open up other countries, or update your age")

3. ASK NEXT QUESTION (ONLY if NEXT FIELD TO ASK is provided):
   - After a line break, ask for the next field naturally
   - The choices shown are already validated — present them EXACTLY as listed
   - If NO next field is provided, do NOT ask any question — just end with the fix suggestion

TONE:
- Be casual and direct — like a helpful person, NOT a bureaucrat
- NEVER use phrases like: "I must inform you", "unfortunately", "I'm sorry to say", "please be advised", "I regret to inform"
- Instead use: "heads up", "just so you know", "the thing is", "here's what happened"
- Keep it short — max 3-4 sentences total
- NEVER use: "dropdown", "field_id", "valid_options", "hierarchy", "metadata", "Phase 1", "Phase 3"
- Use field labels only"""

    llm_messages = [{"role": "system", "content": system_prompt}]
    if messages_history:
        recent = messages_history[-6:]
        for msg in recent:
            llm_messages.append({"role": msg["role"], "content": msg["content"]})

    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=llm_messages,
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()


# === Nudge (uncertain input) ===

def call_openai_nudge_message(user_message, form, collected_data, currently_asking=None, currently_asking_field=None, dropped_fields=None, messages_history=None):
    """Generate a helpful message when the system couldn't process the user's input.
    Includes context about WHY specific values were rejected.
    """
    print(f"    [llm] call_openai_nudge_message (currently_asking={currently_asking})")
    print(f"    [llm]   dropped_fields: {dropped_fields}")
    form_prompt = _get_form_prompt(form)

    asking_info = ""
    if currently_asking and currently_asking_field:
        label = currently_asking_field.get("label", currently_asking)
        ftype = currently_asking_field.get("type", "text")
        asking_info = f'We were asking for: "{label}" (type: {ftype})'

        if has_options(currently_asking_field):
            valid_opts = get_valid_dropdown_values(form, currently_asking, collected_data)
            if valid_opts:
                asking_info += f'\nAvailable choices: {", ".join(valid_opts)}'

        rules = currently_asking_field.get("validation_rules", {})
        if rules:
            clean_rules = {k: v for k, v in rules.items() if k not in ("conditional_rules", "regex")}
            if "regex_description" in rules:
                clean_rules["format"] = rules["regex_description"]
            if clean_rules:
                asking_info += f"\nRequirements: {json.dumps(clean_rules)}"

    # Build dropped context — explains WHY the user's input was rejected
    dropped_context = ""
    if dropped_fields:
        dropped_lines = []
        for d in dropped_fields:
            dropped_lines.append(f'- Tried to set {d["field"]}: "{d["value"]}" → Rejected because: {d["reason"]}')
        dropped_context = f"\n\nREJECTED VALUES (explain these to the user):\n" + "\n".join(dropped_lines)

    system_prompt = f"""You are a warm, helpful form assistant. The user provided input that couldn't be processed.

Form: {form['title']}
{f"FORM CONTEXT: {form_prompt}" if form_prompt else ""}

Already collected: {json.dumps(collected_data)}

{asking_info}

User said: "{user_message}"
{dropped_context}

Generate a friendly response that:
1. If values were REJECTED: explain WHY each was rejected using the reasons above.
   - For hierarchy rejections: explain which parent selection limits the choices
   - Show the available alternatives naturally
2. Acknowledge the user's intent (don't dismiss what they tried to do)
3. Then re-ask for the field we need, with helpful hints
4. Keep it concise — max 3-4 sentences total

LANGUAGE:
- NEVER use internal terms like "dropdown", "field_id", "valid_options", "extraction", "sanitizer"
- Say "since you selected [parent] as [value]" — NOT "the parent_field_id limits options"
- Sound like a real person helping, not a system error
- Be encouraging, not robotic

Return ONLY the message text."""

    llm_messages = [{"role": "system", "content": system_prompt}]
    if messages_history:
        recent = messages_history[-6:]
        for msg in recent:
            llm_messages.append({"role": msg["role"], "content": msg["content"]})

    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=llm_messages,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


# === Query Answer ===

def call_openai_answer_query(query, form, collected_data):
    """Answer a user's question about the form using the full form definition."""
    print(f"    [llm] call_openai_answer_query")
    print(f"    [llm]   query: {query!r}")
    print(f"    [llm]   collected_data: {collected_data}")
    query_prompt = form.get("query_prompt", "")

    system_prompt = f"""You are a helpful form assistant answering a user's question.

FULL FORM DEFINITION:
{json.dumps(form, indent=2)}

ALREADY COLLECTED DATA:
{json.dumps(collected_data, indent=2)}

{f"FORM-SPECIFIC QUERY INSTRUCTIONS:{chr(10)}{query_prompt}" if query_prompt else ""}

HOW TO ANSWER — follow these steps IN ORDER:

STEP 1: IDENTIFY ALL COLLECTED FIELDS THAT CONSTRAIN THE ANSWER
   - Before doing anything else, list EVERY collected field that is related to the question
   - Related means: it is a parent, child, grandchild, or any ancestor/descendant in the hierarchy
   - Also include fields connected through validation rules (e.g., age constrains country via conditional_rules)

STEP 2: EXHAUSTIVE TRAVERSAL
   - Walk through EVERY branch of the form definition to collect ALL possible values for the asked field
   - Dropdown fields with "dropdown_options" contain nested "children" that define parent-child relationships
   - Walk through "dropdown_options" → "children" → "options" recursively through ALL branches
   - Do NOT stop at the first match — check every branch

STEP 3: FILTER USING EVERY COLLECTED FIELD (MOST IMPORTANT STEP)
   - For EACH value found in Step 2, verify it is compatible with ALL constraining fields from Step 1
   - A value is compatible ONLY if it appears in a branch that contains ALL the collected values
   - Filter in BOTH directions:
     PARENT → CHILD: if country=India is collected, only keep states that are under India
     CHILD → PARENT: if district=Mangalore is collected, only keep states where Mangalore appears as a child
   - Apply ALL constraints together. If country=India AND district=Mangalore are both collected:
     → First: only states under India (Kerala, Tamil Nadu, Karnataka)
     → Then: of those, only states that contain Mangalore (Karnataka)
     → Final answer: Karnataka only
   - A value that fails ANY constraint is excluded

STEP 4: APPLY VALIDATION RULES
   - Check "conditional_rules" in "validation_rules" — these change constraints based on other fields
   - Example: if age=16 is collected and user asks "which countries", exclude countries where minimum age > 16

STEP 5: FORM YOUR ANSWER
   - Present only the values that survived ALL filters
   - If only one value remains, state it clearly
   - If question is about collected data, format as a nice summary
   - When asked "how many X", count accurately and list them

LANGUAGE (CRITICAL):
- NEVER use technical terms: "dropdown", "field_id", "dropdown_options", "children", "parent_field_id", "conditional_rules", "validation_rules", "hierarchy", "metadata", "JSON"
- Speak naturally: "you can choose from", "there are X available", "since your district is Mangalore, the only state available is Karnataka"
- Sound like a real person, not a system
- Keep answers concise but complete

Return ONLY the answer text."""

    response = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()
