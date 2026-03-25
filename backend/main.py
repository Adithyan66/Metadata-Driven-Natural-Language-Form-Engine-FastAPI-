import json
import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langfuse.openai import OpenAI

app = FastAPI(title="Dynamic Form Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# --- JSON helpers ---

def read_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)


def write_json(filename, data):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --- Dependency helpers ---

def _find_hierarchy_root(form, field_id):
    """Find the top-level field that owns the hierarchy containing field_id."""
    field = get_field(form, field_id)
    if not field:
        return None
    if not field.get("parent_field_id"):
        # Check if this is itself a root or if another field's children reference it
        for f in form["fields"]:
            if f["type"] == "dropdown" and not f.get("parent_field_id"):
                if _tree_contains_field(f.get("dropdown_options", []), field_id):
                    return f
        return field
    return _find_hierarchy_root(form, field["parent_field_id"])


def _tree_contains_field(options, field_id):
    """Check if any branch of the options tree references field_id."""
    for opt in options:
        children = opt.get("children", {})
        if children.get("field_id") == field_id:
            return True
        if _tree_contains_field(children.get("options", []), field_id):
            return True
    return False


def find_value_in_hierarchy(form, value):
    """Find all places a value appears in any dropdown hierarchy.
    Returns list of dicts: {field_id, value, parents: {field_id: value, ...}}
    """
    if value is None:
        return []
    value = str(value)
    matches = []
    for field in form["fields"]:
        if field["type"] == "dropdown" and not field.get("parent_field_id"):
            _search_tree(field["field_id"], field.get("dropdown_options", []), value, {}, matches)
    return matches


def _search_tree(current_field_id, options, target_value, parent_context, matches):
    for opt in options:
        if opt["value"].lower() == target_value.lower():
            match = {
                "field_id": current_field_id,
                "value": opt["value"],
                "parents": dict(parent_context),
            }
            # Also add children context
            children = opt.get("children", {})
            if children:
                match["children_field_id"] = children.get("field_id")
            matches.append(match)

        # Recurse into children
        children = opt.get("children", {})
        if children:
            child_field_id = children.get("field_id")
            new_context = dict(parent_context)
            new_context[current_field_id] = opt["value"]
            _search_tree(child_field_id, children.get("options", []), target_value, new_context, matches)


def get_all_descendant_field_ids(form, field_id):
    """Get all descendant field IDs (children, grandchildren, etc.) of a field."""
    descendants = []
    for f in form["fields"]:
        if f.get("parent_field_id") == field_id:
            descendants.append(f["field_id"])
            descendants.extend(get_all_descendant_field_ids(form, f["field_id"]))
    return descendants


def get_all_ancestor_field_ids(form, field_id):
    """Get all ancestor field IDs (parent, grandparent, etc.) of a field."""
    ancestors = []
    field = get_field(form, field_id)
    while field and field.get("parent_field_id"):
        ancestors.append(field["parent_field_id"])
        field = get_field(form, field["parent_field_id"])
    return ancestors


def check_hierarchy_conflict(form, field_id, value, collected_data):
    """Check if setting field_id=value conflicts with already collected hierarchical data.
    Checks ALL ancestors AND all descendants (not just direct parent/child).
    Returns (is_valid, conflict_message).
    """
    matches = find_value_in_hierarchy(form, value)

    # Find matches for the correct field_id
    field_matches = [m for m in matches if m["field_id"] == field_id]

    if not field_matches:
        return False, f"'{value}' is not a valid option for {field_id}."

    # Get ALL descendants (children, grandchildren, etc.)
    all_descendants = get_all_descendant_field_ids(form, field_id)

    # For each possible match, check if it's consistent with ALL collected data
    for match in field_matches:
        # --- Check ancestor consistency ---
        ancestor_conflict = False
        for parent_fid, parent_val in match["parents"].items():
            if parent_fid in collected_data and collected_data[parent_fid].lower() != parent_val.lower():
                ancestor_conflict = True
                break
        if ancestor_conflict:
            continue

        # --- Check ALL descendant consistency (children + grandchildren + ...) ---
        descendant_conflict = False
        descendant_conflict_detail = None
        for desc_fid in all_descendants:
            if desc_fid not in collected_data:
                continue
            desc_val = collected_data[desc_fid]
            # Check if the descendant's value exists under this match's branch
            desc_matches = find_value_in_hierarchy(form, desc_val)
            valid_under_branch = any(
                m["field_id"] == desc_fid and
                m.get("parents", {}).get(field_id, "").lower() == value.lower()
                for m in desc_matches
            )
            if not valid_under_branch:
                descendant_conflict = True
                descendant_conflict_detail = (desc_fid, desc_val)
                break

        if not descendant_conflict:
            return True, None

    # All matches have conflicts — build a clear error message
    for match in field_matches:
        for parent_fid, parent_val in match["parents"].items():
            if parent_fid in collected_data and collected_data[parent_fid].lower() != parent_val.lower():
                return False, (
                    f"'{value}' belongs to {parent_fid}='{parent_val}', "
                    f"but you already selected {parent_fid}='{collected_data[parent_fid]}'."
                )

    # Descendant conflict — find the specific one
    for desc_fid in all_descendants:
        if desc_fid in collected_data:
            desc_val = collected_data[desc_fid]
            desc_matches = find_value_in_hierarchy(form, desc_val)
            valid = any(
                m["field_id"] == desc_fid and
                m.get("parents", {}).get(field_id, "").lower() == value.lower()
                for m in desc_matches
            )
            if not valid:
                return False, (
                    f"'{desc_val}' ({desc_fid}) does not belong to "
                    f"{field_id}='{value}'. Please correct {desc_fid} or "
                    f"choose a different {field_id}."
                )

    return False, f"'{value}' is not valid for {field_id} given the current selections."


def validate_hierarchy_consistency(form, collected_data):
    """Check that all collected hierarchical fields form a valid path.
    Returns list of conflict descriptions. Empty list = consistent.
    """
    conflicts = []
    for field in form["fields"]:
        if field["type"] != "dropdown" or not field.get("parent_field_id"):
            continue
        child_fid = field["field_id"]
        parent_fid = field["parent_field_id"]
        if child_fid not in collected_data or parent_fid not in collected_data:
            continue
        child_val = collected_data[child_fid]
        parent_val = collected_data[parent_fid]
        # Check if child_val exists under parent_val in the hierarchy
        child_matches = find_value_in_hierarchy(form, child_val)
        valid = any(
            m["field_id"] == child_fid and
            m.get("parents", {}).get(parent_fid, "").lower() == parent_val.lower()
            for m in child_matches
        )
        if not valid:
            conflicts.append(
                f"'{child_val}' ({child_fid}) does not belong to "
                f"'{parent_val}' ({parent_fid})."
            )
    return conflicts


def get_field(form, field_id):
    for f in form["fields"]:
        if f["field_id"] == field_id:
            return f
    return None


# --- Validation ---

def validate_field(form, field_id, value, collected_data):
    """Validate a single field value against metadata rules.
    Returns (is_valid, error_message).
    """
    field = get_field(form, field_id)
    if not field:
        return False, f"Unknown field: {field_id}"

    field_type = field.get("type", "text")
    rules = field.get("validation_rules", {})

    # Type check
    if field_type == "number":
        try:
            value = int(value) if isinstance(value, str) and value.isdigit() else value
            if not isinstance(value, (int, float)):
                value = float(value)
        except (ValueError, TypeError):
            return False, f"{field['label']} must be a number."

        if "min" in rules and value < rules["min"]:
            return False, f"{field['label']} must be at least {rules['min']}."
        if "max" in rules and value > rules["max"]:
            return False, f"{field['label']} must be at most {rules['max']}."

    if field_type == "text":
        value = str(value)
        if "min_length" in rules and len(value) < rules["min_length"]:
            return False, f"{field['label']} must be at least {rules['min_length']} characters."
        if "max_length" in rules and len(value) > rules["max_length"]:
            return False, f"{field['label']} must be at most {rules['max_length']} characters."
        if "regex" in rules:
            if not re.match(rules["regex"], value):
                desc = rules.get("regex_description", f"matching pattern {rules['regex']}")
                return False, f"{field['label']} must be {desc}."

    # Dropdown validation
    if field_type == "dropdown":
        value = str(value) if value is not None else ""
        if not value:
            return False, f"{field['label']} cannot be empty."
        # Check hierarchy conflicts (deferred — only when related fields exist)
        is_valid, conflict_msg = check_hierarchy_conflict(form, field_id, value, collected_data)
        if not is_valid:
            return False, conflict_msg

        # Check if value is in valid options
        parent_fid = field.get("parent_field_id")
        valid_options = get_valid_dropdown_values(form, field_id, collected_data)

        if valid_options:
            # Parent is set — strict validation against filtered options
            matched = [o for o in valid_options if o.lower() == str(value).lower()]
            if not matched:
                return False, f"'{value}' is not valid for {field['label']}. Valid options: {valid_options}"
        elif parent_fid and parent_fid not in collected_data:
            # Parent NOT yet collected — accept if value exists in hierarchy
            # BUT still filter against any collected ANCESTOR (grandparent, etc.)
            all_matches = find_value_in_hierarchy(form, str(value))
            matched_for_field = [m for m in all_matches if m["field_id"] == field_id]
            if not matched_for_field:
                return False, f"'{value}' is not a valid option for {field['label']}."
            # Filter by any already-collected ancestors
            all_ancestors = get_all_ancestor_field_ids(form, field_id)
            collected_ancestors = {a: collected_data[a] for a in all_ancestors if a in collected_data}
            if collected_ancestors:
                compatible = [
                    m for m in matched_for_field
                    if all(
                        m.get("parents", {}).get(anc_fid, "").lower() == anc_val.lower()
                        for anc_fid, anc_val in collected_ancestors.items()
                    )
                ]
                if not compatible:
                    ancestor_desc = ", ".join(f"{k}='{v}'" for k, v in collected_ancestors.items())
                    return False, (
                        f"'{value}' does not belong to {ancestor_desc}. "
                        f"Please choose a valid {field['label']} or change your earlier selections."
                    )
        elif valid_options is not None:
            # Has options list but value not in it
            matched = [o for o in (valid_options or []) if o.lower() == str(value).lower()]
            if not matched:
                return False, f"'{value}' is not valid for {field['label']}."

    return True, None


def get_valid_dropdown_values(form, field_id, collected_data):
    """Get valid dropdown values filtered BIDIRECTIONALLY.
    Filters by collected ancestors (top-down) AND collected descendants (bottom-up).
    """
    field = get_field(form, field_id)
    if not field or field["type"] != "dropdown":
        return None

    # Simple dropdown (no hierarchy involvement)
    if not field.get("parent_field_id"):
        root_field = _find_hierarchy_root(form, field_id)
        if root_field and root_field["field_id"] == field_id:
            # This IS the root of a hierarchy — check if any descendant narrows it
            all_options = [o["value"] for o in field.get("dropdown_options", [])]
            if not all_options:
                return None
            return _filter_by_descendants(form, field_id, all_options, collected_data)
        # Not part of any hierarchy
        own_options = [o["value"] for o in field.get("dropdown_options", [])]
        return own_options if own_options else None

    # Part of a hierarchy — collect ALL possible values from the tree
    root_field = _find_hierarchy_root(form, field_id)
    if not root_field:
        return []

    # Gather all occurrences of this field in the tree with their full context
    all_occurrences = _gather_field_occurrences(
        root_field["field_id"], root_field.get("dropdown_options", []), field_id
    )

    # Filter occurrences by ALL collected hierarchy data (ancestors AND descendants)
    valid_values = set()
    for occ in all_occurrences:
        context = occ["context"]  # {field_id: value} for all ancestors in this branch
        value = occ["value"]

        # Check ancestor consistency
        ancestor_ok = all(
            context.get(fid, "").lower() == collected_data[fid].lower()
            for fid in context
            if fid in collected_data
        )
        if not ancestor_ok:
            continue

        # Check descendant consistency
        descendant_ok = True
        occ_descendants = occ.get("descendants", {})
        # Check 1: collected descendants that ARE in this subtree must match
        for desc_fid, desc_vals in occ_descendants.items():
            if desc_fid in collected_data:
                if collected_data[desc_fid].lower() not in [v.lower() for v in desc_vals]:
                    descendant_ok = False
                    break
        # Check 2: collected descendants NOT in this subtree — if they're part
        # of the hierarchy under this field, the subtree can't satisfy them
        if descendant_ok:
            all_possible_descs = get_all_descendant_field_ids(form, field_id)
            for desc_fid in all_possible_descs:
                if desc_fid in collected_data and desc_fid not in occ_descendants:
                    # This branch doesn't contain the field at all — incompatible
                    descendant_ok = False
                    break
        if not descendant_ok:
            continue

        valid_values.add(value)

    return sorted(valid_values) if valid_values else []


def _gather_field_occurrences(current_field_id, options, target_field_id, context=None):
    """Walk the tree and collect every occurrence of target_field_id with full context.
    Each occurrence has: value, context (ancestors), descendants (children values).
    """
    if context is None:
        context = {}
    results = []

    for opt in options:
        children = opt.get("children", {})
        child_field_id = children.get("field_id")
        child_options = children.get("options", [])

        if current_field_id == target_field_id:
            # This level IS the target — collect descendants
            descendants = {}
            _collect_descendants(opt, descendants)
            results.append({
                "value": opt["value"],
                "context": dict(context),
                "descendants": descendants,
            })
        elif child_field_id:
            # Go deeper with updated context
            new_context = dict(context)
            new_context[current_field_id] = opt["value"]
            results.extend(_gather_field_occurrences(
                child_field_id, child_options, target_field_id, new_context
            ))

    return results


def _collect_descendants(node, descendants):
    """Collect all descendant field values under a node."""
    children = node.get("children", {})
    child_field_id = children.get("field_id")
    child_options = children.get("options", [])
    if not child_field_id:
        return
    vals = [co["value"] for co in child_options]
    if child_field_id not in descendants:
        descendants[child_field_id] = []
    descendants[child_field_id].extend(vals)
    for co in child_options:
        _collect_descendants(co, descendants)


def _filter_by_descendants(form, field_id, options, collected_data):
    """Filter root-level dropdown options by any collected descendant values."""
    all_descendants = get_all_descendant_field_ids(form, field_id)
    collected_descs = {fid: collected_data[fid] for fid in all_descendants if fid in collected_data}
    if not collected_descs:
        return options  # No descendants collected — all options valid

    # Find root field for the tree
    field = get_field(form, field_id)
    if not field:
        return options

    filtered = []
    for opt_val in options:
        # Check if this option's subtree contains all collected descendant values
        opt_node = next(
            (o for o in field.get("dropdown_options", []) if o["value"] == opt_val), None
        )
        if not opt_node:
            continue
        descendants = {}
        _collect_descendants(opt_node, descendants)
        # Also check the direct children level
        children = opt_node.get("children", {})
        if children.get("field_id"):
            if children["field_id"] not in descendants:
                descendants[children["field_id"]] = []
            descendants[children["field_id"]].extend(
                [co["value"] for co in children.get("options", [])]
            )

        all_match = True
        for desc_fid, desc_val in collected_descs.items():
            if desc_fid in descendants:
                if desc_val.lower() not in [v.lower() for v in descendants[desc_fid]]:
                    all_match = False
                    break
            # If descendant field not in this subtree at all, it doesn't match
            elif desc_fid in [f["field_id"] for f in form["fields"] if f.get("parent_field_id")]:
                all_match = False
                break
        if all_match:
            filtered.append(opt_val)

    return filtered


def get_missing_fields(form, collected_data):
    missing = []
    for field in form["fields"]:
        if field.get("required", False) and field["field_id"] not in collected_data:
            missing.append(field["field_id"])
    return missing


# --- OpenAI calls ---

def call_openai_extract(user_message, form, collected_data, last_question):
    """Ask OpenAI to extract field values from user message."""
    fields_desc = []
    for f in form["fields"]:
        desc = f"- {f['field_id']} ({f['label']}): type={f['type']}"
        if f.get("required"):
            desc += ", required"
        if f.get("validation_rules"):
            desc += f", rules={json.dumps(f['validation_rules'])}"
        if f["type"] == "dropdown":
            valid_opts = get_valid_dropdown_values(form, f["field_id"], collected_data)
            if valid_opts:
                desc += f", valid_options={valid_opts}"
        fields_desc.append(desc)

    system_prompt = f"""You are a data extraction assistant. Extract ALL possible field values from the user's message.

Form: {form['title']}
Fields:
{chr(10).join(fields_desc)}

Already collected data: {json.dumps(collected_data)}

The last question asked to the user was: "{last_question}"

RULES:
1. Extract ALL fields that the user mentions or implies — not just the one being asked.
   The user may answer the current question AND provide other field values in the same message.
   Example: Question was "What is your name?" but user says "my name is adhi and my district is alappuzha"
   → Extract BOTH name and district.
2. If the user gives a short/direct answer (like just a name or number) with no other context,
   map it to the field from the last question asked.
3. Return STRICT JSON object with field_id as keys and extracted values.
4. For number fields, return actual numbers not strings.
5. Do NOT make up or guess values. Only extract what the user actually said.
6. Do NOT re-extract already collected fields unless the user is explicitly correcting them.
7. If you cannot extract any field, return empty object {{}}.

IMPORTANT: Every piece of data the user provides is valuable. Never ignore a field value
just because it wasn't the current question. Extract everything recognizable.

Return ONLY the JSON object, no explanation."""

    response = client.chat.completions.create(
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


def call_openai_next_question(form, collected_data, missing_fields, auto_filled=None):
    """Ask OpenAI to generate the next question."""
    fields_info = []
    for fid in missing_fields:
        field = get_field(form, fid)
        if field:
            info = f"- {field['field_id']}: {field['label']} (type: {field['type']})"
            if field["type"] == "dropdown":
                valid_opts = get_valid_dropdown_values(form, fid, collected_data)
                if valid_opts:
                    info += f" [options: {', '.join(valid_opts)}]"
                else:
                    # No valid options available yet — skip this field for now
                    continue
            fields_info.append(info)

    if not fields_info:
        return "All fields have been collected!"

    auto_fill_note = ""
    if auto_filled:
        filled_desc = ", ".join(f"{k}='{v}'" for k, v in auto_filled.items())
        auto_fill_note = f"\n\nNote: The following fields were auto-filled because only one valid option existed: {filled_desc}. Mention this to the user briefly."

    system_prompt = f"""You are a friendly form assistant. Generate the next question to ask the user.

Form: {form['title']}
Already collected: {json.dumps(collected_data)}

Missing fields (ask the FIRST one that makes sense):
{chr(10).join(fields_info)}
{auto_fill_note}
RULES:
1. Ask for ONE field at a time.
2. Be conversational and friendly.
3. If it's a dropdown, mention the available options naturally.
4. Respect field ordering — ask parent fields before child fields.
5. Keep it short and natural.

Return ONLY the question text."""

    response = client.chat.completions.create(
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()


# --- Request models ---

class SelectFormRequest(BaseModel):
    form_id: str

class ChatRequest(BaseModel):
    message: str


# --- API endpoints ---

@app.get("/forms")
def get_forms():
    forms = read_json("forms.json")
    return [{"form_id": f["form_id"], "title": f["title"]} for f in forms]


@app.post("/select-form")
def select_form(req: SelectFormRequest):
    forms = read_json("forms.json")
    form = next((f for f in forms if f["form_id"] == req.form_id), None)
    if not form:
        raise HTTPException(status_code=404, detail="Form not found")

    write_json("active_form.json", form)
    write_json("collected_data.json", {})
    write_json("messages.json", [])

    # Generate first question
    missing = get_missing_fields(form, {})
    question = call_openai_next_question(form, {}, missing)

    messages = [{"role": "assistant", "content": question}]
    write_json("messages.json", messages)

    return {
        "status": "pending",
        "message": question,
        "collected_data": {},
        "missing_fields": missing,
        "invalid_fields": [],
    }


@app.post("/reset")
def reset():
    write_json("active_form.json", None)
    write_json("collected_data.json", {})
    write_json("messages.json", [])
    return {"status": "reset", "message": "All data cleared."}


@app.post("/chat")
def chat(req: ChatRequest):
    # Step 1: Load
    form = read_json("active_form.json")
    if not form:
        raise HTTPException(status_code=400, detail="No active form. Select a form first.")

    collected_data = read_json("collected_data.json")
    messages = read_json("messages.json")

    # Find last assistant question
    last_question = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            last_question = msg["content"]
            break

    # Save user message
    messages.append({"role": "user", "content": req.message})

    # Step 2: Extract via OpenAI
    extracted = call_openai_extract(req.message, form, collected_data, last_question)

    print('openai extracted',extracted)
    # Step 3: Validate — process in form field order, with running snapshot
    valid_data = {}
    invalid_fields = []

    # Sort extracted fields by their position in the form definition
    field_order = {f["field_id"]: i for i, f in enumerate(form["fields"])}
    sorted_fields = sorted(
        extracted.items(),
        key=lambda item: field_order.get(item[0], 999),
    )

    for field_id, value in sorted_fields:
        # Skip None/empty values from LLM extraction
        if value is None or value == "":
            continue

        # Skip already collected fields (first valid wins)
        if field_id in collected_data:
            continue

        # Use running snapshot: original collected + already validated in this batch
        running_data = {**collected_data, **valid_data}

        is_valid, error = validate_field(form, field_id, value, running_data)
        if is_valid:
            # Re-resolve correct casing for dropdowns
            field = get_field(form, field_id)
            if field and field["type"] == "dropdown":
                valid_opts = get_valid_dropdown_values(form, field_id, running_data)
                if valid_opts:
                    matched = [o for o in valid_opts if o.lower() == str(value).lower()]
                    if matched:
                        value = matched[0]
            if field and field["type"] == "number":
                try:
                    value = int(value) if isinstance(value, str) and value.isdigit() else value
                    if isinstance(value, str):
                        value = float(value)
                except (ValueError, TypeError):
                    pass
            valid_data[field_id] = value
        else:
            invalid_fields.append({"field_id": field_id, "value": value, "error": error})

    # Step 4: Store valid data
    collected_data.update(valid_data)

    # Step 4b: Auto-fill dropdown fields that have exactly one valid option
    auto_filled = {}
    changed = True
    while changed:
        changed = False
        for field in form["fields"]:
            fid = field["field_id"]
            if fid in collected_data or field["type"] != "dropdown":
                continue
            valid_opts = get_valid_dropdown_values(form, fid, collected_data)
            if valid_opts and len(valid_opts) == 1:
                collected_data[fid] = valid_opts[0]
                auto_filled[fid] = valid_opts[0]
                changed = True  # Re-check — filling one field may narrow others

    write_json("collected_data.json", collected_data)

    # Step 5: Determine response
    missing = get_missing_fields(form, collected_data)

    if not missing:
        # All fields present — verify hierarchy consistency before completing
        hierarchy_conflicts = validate_hierarchy_consistency(form, collected_data)
        if hierarchy_conflicts:
            # Hierarchy is inconsistent — ask user to fix
            # Find which fields need correction
            conflict_field_ids = []
            for field in form["fields"]:
                if field["type"] == "dropdown" and field.get("parent_field_id"):
                    child_fid = field["field_id"]
                    parent_fid = field["parent_field_id"]
                    if child_fid in collected_data and parent_fid in collected_data:
                        child_val = collected_data[child_fid]
                        parent_val = collected_data[parent_fid]
                        child_matches = find_value_in_hierarchy(form, child_val)
                        valid = any(
                            m["field_id"] == child_fid and
                            m.get("parents", {}).get(parent_fid, "").lower() == parent_val.lower()
                            for m in child_matches
                        )
                        if not valid:
                            conflict_field_ids.append(child_fid)
            error_msg = call_openai_error_message(
                form,
                [{"field_id": fid, "value": collected_data[fid],
                  "error": c} for fid, c in zip(conflict_field_ids, hierarchy_conflicts)],
                req.message,
                collected_data,
            )
            # Remove conflicting child fields so user can re-enter
            for fid in conflict_field_ids:
                collected_data.pop(fid, None)
            write_json("collected_data.json", collected_data)
            missing = get_missing_fields(form, collected_data)
            next_q = call_openai_next_question(form, collected_data, missing)
            response_msg = error_msg + "\n\n" + next_q
            status = "pending"
        else:
            response_msg = "All information has been collected. Thank you!"
            status = "complete"
    elif invalid_fields:
        # Some fields were invalid — generate helpful error
        error_msg = call_openai_error_message(form, invalid_fields, req.message, collected_data)
        # Also ask next question if there are still missing fields
        next_q = call_openai_next_question(form, collected_data, missing, auto_filled=auto_filled)
        response_msg = error_msg + "\n\n" + next_q
        status = "pending"
    else:
        # Ask next question
        response_msg = call_openai_next_question(form, collected_data, missing, auto_filled=auto_filled)
        status = "pending"

    messages.append({"role": "assistant", "content": response_msg})
    write_json("messages.json", messages)

    return {
        "status": status,
        "message": response_msg,
        "collected_data": collected_data,
        "missing_fields": missing,
        "invalid_fields": invalid_fields,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
