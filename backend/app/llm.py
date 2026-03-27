"""OpenAI / LLM calls: extraction, question generation, error messages, query answers.

All prompts combine:
1. Engine-level rules (universal, in code)
2. Form-level context (from form metadata: system_prompt + field descriptions)
"""

import json

from langfuse.openai import OpenAI

from app.hierarchy import get_field, has_options, get_valid_dropdown_values

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


# === Extraction ===

EXTRACTION_RULES = """RULES:

1. HIGH-CONFIDENCE PATTERN OVERRIDE (HIGHEST PRIORITY):
   These patterns ALWAYS map to their field, REGARDLESS of which field is currently being asked:
   - Email pattern (contains @ and domain) → email field
   - Phone pattern (7-15 digits, optional +) → phone field
   - Known structured patterns defined in FORM CONTEXT (e.g., ward patterns) → respective field
   If input matches a high-confidence pattern, map it immediately. Do NOT force it to the currently asking field.

2. BEST-FIT MATCHING (NOT currently-asking-first):
   - Compare input against ALL uncollected fields
   - Choose the field with the STRONGEST semantic + type match
   - Currently asking field gets priority ONLY when match strength is equal
   - Example: asking for "name", user says "adhi@gmail.com" → email (pattern match wins)
   - Example: asking for "name", user says "adhi" → full_name (semantic match, fits asked field)

3. TYPE MATCH (STRICT):
   - number → must be numeric (e.g., 25)
   - text → must be meaningful text
   - dropdown → must be one of valid_options (allow fuzzy: "kerla" → "Kerala")

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

Return ONLY a JSON object. No explanation."""


def call_openai_extract(user_message, form, collected_data, currently_asking=None, currently_asking_field=None):
    """Extract field values from user message."""
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

    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
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
   - If user skipped the asked question, gently remind them

2. ASK THE NEXT QUESTION:
   - Ask for ONE field only
   - Be natural and conversational
   - For dropdowns: list options in a friendly way
   - For password: mention the requirements clearly

FORMATTING:
- Line break between acknowledgment and question
- Max 3 sentences total
- No bullet points, no raw field names
- Sound human, not robotic

LANGUAGE (CRITICAL):
- NEVER use internal terms: "dropdown", "options list", "field_id", "valid_options", "hierarchy", "metadata"
- Say "you can choose from" or "would you prefer" — NOT "select from dropdown options"
- Use natural everyday language as if talking to a real person"""


def call_openai_next_question(form, collected_data, missing_fields, last_action=None):
    """Generate the next question to ask the user."""
    form_prompt = _get_form_prompt(form)

    fields_info = []
    for fid in missing_fields:
        field = get_field(form, fid)
        if field:
            info = f"- {field['field_id']}: {field['label']} (type: {field['type']})"
            if has_options(field):
                valid_opts = get_valid_dropdown_values(form, fid, collected_data)
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

        unanswered = last_action.get("unanswered_field")
        if unanswered:
            field = get_field(form, unanswered)
            label = field["label"] if field else unanswered
            parts.append(f"User did NOT answer '{label}'. Remind them and ask for it now.")

        if parts:
            action_context = "\n\nLAST ACTION:\n" + "\n".join(f"- {p}" for p in parts)

    system_prompt = f"""You are a warm, professional form assistant.

Form: {form['title']}
{f"FORM CONTEXT: {form_prompt}" if form_prompt else ""}

Already collected: {json.dumps(collected_data)}

Missing fields (ask the FIRST one that makes sense):
{chr(10).join(fields_info)}
{action_context}

{QUESTION_RULES}

Return ONLY the message text."""

    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


# === Error Message ===

def call_openai_error_message(form, field_errors, user_message, collected_data):
    """Generate a user-friendly error message."""
    form_prompt = _get_form_prompt(form)

    system_prompt = f"""You are a helpful form assistant. The user provided data that failed validation.

Form: {form['title']}
{f"FORM CONTEXT: {form_prompt}" if form_prompt else ""}

Already collected: {json.dumps(collected_data)}
User said: "{user_message}"

Validation errors:
{json.dumps(field_errors, indent=2)}

Generate a clear, friendly message explaining what went wrong and how to fix it. Be concise."""

    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()


# === Nudge (uncertain input) ===

def call_openai_nudge_message(user_message, form, collected_data, currently_asking=None, currently_asking_field=None):
    """Generate a helpful message when the system couldn't understand the user's input."""
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
            clean_rules = {k: v for k, v in rules.items() if k != "conditional_rules"}
            if clean_rules:
                asking_info += f"\nRequirements: {json.dumps(clean_rules)}"

    system_prompt = f"""You are a warm, helpful form assistant. The user provided input that we couldn't match to any field.

Form: {form['title']}
{f"FORM CONTEXT: {form_prompt}" if form_prompt else ""}

Already collected: {json.dumps(collected_data)}

{asking_info}

User said: "{user_message}"

Generate a friendly response that:
1. Acknowledges what the user said (don't ignore their input)
2. Explains briefly why it didn't match (wrong type, not a valid option, etc.)
3. Clearly re-asks for the field we need, with helpful hints
4. If the field has specific options, mention them naturally
5. Keep it concise — max 2-3 sentences

LANGUAGE:
- NEVER use internal terms like "dropdown", "field_id", "valid_options", "extraction"
- Sound like a real person helping, not a system error
- Be encouraging, not robotic

Return ONLY the message text."""

    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


# === Query Answer ===

def call_openai_answer_query(query, form, collected_data):
    """Answer a user's question about the form, fields, options, or their data."""
    fields_context = _build_fields_context(form, collected_data)
    form_prompt = _get_form_prompt(form)

    # Include full dropdown hierarchy for accurate answers
    hierarchy_info = []
    for f in form["fields"]:
        if f.get("dropdown_options"):
            hierarchy_info.append(f"- {f['field_id']}: options={json.dumps(f['dropdown_options'])}")

    system_prompt = f"""You are a helpful form assistant answering a user's question.

Form: {form['title']}
{f"FORM CONTEXT: {form_prompt}" if form_prompt else ""}

Fields:
{fields_context}

Full dropdown hierarchy:
{chr(10).join(hierarchy_info) if hierarchy_info else "None"}

Currently collected data: {json.dumps(collected_data)}

RULES:
1. Answer ONLY based on form metadata and collected data above.
2. If question is about options/choices:
   - Check if parent field is already selected → show only relevant children
   - If no parent selected → show all available
3. If question is about collected data → format as a nice summary.
4. Keep answer concise, friendly, accurate.
5. Do NOT make up data — always depend on given data, not from outside.
6. LANGUAGE (CRITICAL):
   - NEVER use technical/internal terms like "dropdown", "options", "valid_options", "field_id", "hierarchy", "metadata"
   - Speak naturally as if talking to a real person
   - Say "available choices" or "you can choose from" — NOT "dropdown options"
   - Say "under Tokyo" — NOT "part of the Tokyo district in the dropdown"
   - Example BAD: "In the dropdown options, there are 2 wards: Ward1 and Ward2"
   - Example GOOD: "Under Tokyo, you can choose between Ward1 and Ward2"

Return ONLY the answer text."""

    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()
