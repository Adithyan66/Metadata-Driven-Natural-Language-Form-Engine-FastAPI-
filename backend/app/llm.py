"""OpenAI / LLM calls: extraction, question generation, error messages."""

import json

from langfuse.openai import OpenAI

from app.hierarchy import get_field, has_options, get_valid_dropdown_values

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def call_openai_extract(user_message, form, collected_data, currently_asking=None, currently_asking_field=None):
    """Ask OpenAI to extract field values from user message.

    Args:
        currently_asking: field_id of the field we just asked about
        currently_asking_field: full field definition for currently_asking
    """
    fields_desc = []
    for f in form["fields"]:
        desc = f"- {f['field_id']} ({f['label']}): type={f['type']}"
        if f.get("required"):
            desc += ", required"
        if f.get("validation_rules"):
            desc += f", rules={json.dumps(f['validation_rules'])}"
        if has_options(f):
            valid_opts = get_valid_dropdown_values(form, f["field_id"], collected_data)
            if valid_opts:
                desc += f", valid_options={valid_opts}"
        fields_desc.append(desc)

    # Build currently_asking context for the prompt
    asking_context = ""
    if currently_asking and currently_asking_field:
        label = currently_asking_field.get("label", currently_asking)
        ftype = currently_asking_field.get("type", "text")
        asking_context = f"""
CURRENTLY ASKING: "{label}" (field_id: {currently_asking}, type: {ftype})
"""
    else:
        asking_context = """
CURRENTLY ASKING: None
"""

    system_prompt = f"""You are a STRICT data extraction assistant for a form.

Form: {form['title']}
Fields:
{chr(10).join(fields_desc)}

Already collected: {json.dumps(collected_data)}
{asking_context}
RULES:

1. Extract ALL fields explicitly or implicitly mentioned by the user.

2. CURRENTLY ASKING PRIORITY:
   - If the user gives a short/direct answer, assume it is for the CURRENTLY ASKING field
   - BUT ONLY if it passes BOTH:
     a) TYPE MATCH
     b) SEMANTIC MATCH (the value logically belongs to that field)

3. TYPE MATCH (STRICT):
   - number → must be numeric (e.g., 25)
   - text → must be meaningful text (not random unrelated values)
   - dropdown → must be one of valid_options

4. SEMANTIC MATCH (VERY IMPORTANT):
   - Even if a value is valid somewhere in the form, DO NOT assign it unless it logically belongs to the field
   - Example:
     - Asking: age (number)
     - User: "India"
       → "India" is valid for "country" but NOT for "age"
       → DO NOT map it to age → map to country instead
   - Example:
     - Asking: age
     - User: "adhi"
       → Not a number, not meaningful for age
       → DO NOT guess age like 18

5. CROSS-FIELD MATCHING:
   - If the value does NOT fit CURRENTLY ASKING field:
     → Try matching it with OTHER uncollected fields
   - Example:
     - Asking: age
     - User: "India"
       → map to "country" (not age)

6. NO GUESSING / NO DEFAULTS:
   - NEVER infer or generate values the user did not say
   - "adhi" → CANNOT become age=18
   - NEVER auto-correct or transform unrelated inputs

7. INVALID OR IRRELEVANT INPUT:
   - If value does not match ANY field at all → return:
     {{"_uncertain": true}}

8. PARTIAL EXTRACTION:
   - Extract valid fields even if some parts are invalid
   - Example:
     "adhi 25 india"
     → extract full_name="adhi", age=25, country="India" (if those fields exist and are uncollected)
   - Example:
     - Asking: full_name
     - User: "age is 19"
     → full_name is NOT answered, but age=19 IS valid → return {{"age": 19}}
     → Do NOT return _uncertain if you found a valid match for another field

9. UPDATE INTENT:
   - If user explicitly changes existing value (e.g., "change age to 30") → include:
     "_intent": "update"

10. STRICT OUTPUT:
    - Return ONLY valid mappings
    - Do NOT include fields with weak or doubtful matches
    - For number fields, return actual numbers not strings

11. CONFIDENCE RULE:
    - Return {{"_uncertain": true}} ONLY when NO field matches at all
    - If at least one field was extracted confidently, return that field WITHOUT _uncertain
    - Precision > Recall (better to miss than to be wrong)

12. SENSITIVE FIELDS:
    - Password fields → always return ""

Return ONLY a JSON object. No explanation."""

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


def call_openai_next_question(form, collected_data, missing_fields, last_action=None):
    """Ask OpenAI to generate the next question.

    Args:
        last_action: dict describing what just happened, e.g.:
            {"stored": {"age": 25, "country": "India"}, "auto_filled": {"state": "Kerala"},
             "inferred": {"country": "India"}, "updated": {"age": 30},
             "unanswered_field": "full_name"}
    """
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

    # Build last action context for the prompt
    action_context = ""
    if last_action:
        parts = []
        stored = last_action.get("stored", {})
        if stored:
            items = ", ".join(f"{k}='{v}'" for k, v in stored.items())
            parts.append(f"User just provided: {items}. Acknowledge this briefly.")

        auto_filled = last_action.get("auto_filled", {})
        if auto_filled:
            items = ", ".join(f"{k}='{v}'" for k, v in auto_filled.items())
            parts.append(f"Auto-filled (only one valid option): {items}. Mention this to the user.")

        inferred = last_action.get("inferred", {})
        if inferred:
            items = ", ".join(f"{k}='{v}'" for k, v in inferred.items())
            parts.append(f"System inferred from hierarchy: {items}. Mention this naturally (e.g., 'Since you selected X, I've set Y').")

        updated = last_action.get("updated", {})
        if updated:
            items = ", ".join(f"{k}='{v}'" for k, v in updated.items())
            parts.append(f"User updated: {items}. Confirm the update briefly.")

        unanswered = last_action.get("unanswered_field")
        if unanswered:
            field = get_field(form, unanswered)
            label = field["label"] if field else unanswered
            parts.append(f"The user did NOT answer the previously asked question about '{label}'. After acknowledging what was stored, remind them and ask for '{label}' now.")

        if parts:
            action_context = "\n\nLAST ACTION:\n" + "\n".join(f"- {p}" for p in parts)

    system_prompt = f"""You are a friendly form assistant. Generate the next message for the user.

Form: {form['title']}
Already collected: {json.dumps(collected_data)}

Missing fields (ask the FIRST one that makes sense):
{chr(10).join(fields_info)}
{action_context}
RULES:
1. First, briefly acknowledge what just happened (if LAST ACTION is provided).
2. Then ask for ONE missing field.
3. Be conversational and friendly. Keep it short.
4. If it's a dropdown, mention the available options naturally.
5. Respect field ordering — ask parent fields before child fields.
6. Do NOT repeat already collected data unnecessarily.

Return ONLY the message text."""

    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


def call_openai_error_message(form, field_errors, user_message, collected_data):
    """Ask OpenAI to generate a user-friendly error message."""
    system_prompt = f"""You are a helpful form assistant. The user provided some data that failed validation.

Form: {form['title']}
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
