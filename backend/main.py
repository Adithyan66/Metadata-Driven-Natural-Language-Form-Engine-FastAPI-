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


def has_options(field):
    """Check if a field type has selectable options (metadata-driven).
    Instead of hardcoding type names, checks if the field has dropdown_options defined.
    """
    if not field:
        return False
    return bool(field.get("dropdown_options")) or bool(field.get("parent_field_id"))


# --- Dependency graph & inference engine ---

def build_dependency_graph(form):
    """Build a dependency graph from metadata. Returns:
    - hierarchy_deps: {child_field_id: parent_field_id} from parent_field_id
    - conditional_deps: {field_id: [condition_field_id, ...]} from conditional_rules
    Both are metadata-driven, no hardcoded field names.
    """
    hierarchy_deps = {}
    conditional_deps = {}

    for field in form["fields"]:
        fid = field["field_id"]

        if field.get("parent_field_id"):
            hierarchy_deps[fid] = field["parent_field_id"]

        # Gather from validation_rules.conditional_rules AND field-level conditional_rules
        all_cond_rules = field.get("validation_rules", {}).get("conditional_rules", [])
        all_cond_rules += field.get("conditional_rules", [])

        for rule in all_cond_rules:
            cond_field = rule.get("if", {}).get("field")
            if cond_field:
                if fid not in conditional_deps:
                    conditional_deps[fid] = []
                if cond_field not in conditional_deps[fid]:
                    conditional_deps[fid].append(cond_field)

    return hierarchy_deps, conditional_deps


def infer_parents_from_hierarchy(form, data):
    """Forward inference: if a child value is set but parent is not,
    infer the parent value from the hierarchy tree.
    e.g., state=Kerala → infer country=India (if unambiguous).
    Returns dict of inferred {field_id: value} — does NOT modify data.
    """
    inferred = {}

    for field in form["fields"]:
        fid = field["field_id"]
        if field.get("type") != "dropdown" or not field.get("parent_field_id"):
            continue
        if fid not in data:
            continue

        # Walk up the ancestor chain, inferring missing parents
        child_fid = fid
        child_val = data[child_fid]

        while True:
            child_field = get_field(form, child_fid)
            if not child_field or not child_field.get("parent_field_id"):
                break
            parent_fid = child_field["parent_field_id"]

            # If parent already known (in data or already inferred), stop
            if parent_fid in data or parent_fid in inferred:
                break

            # Find what parent values are compatible with child_val
            matches = find_value_in_hierarchy(form, child_val)
            parent_values = set()
            for m in matches:
                if m["field_id"] == child_fid:
                    parent_val = m.get("parents", {}).get(parent_fid)
                    if parent_val:
                        parent_values.add(parent_val)

            if len(parent_values) == 1:
                # Unambiguous → infer
                inferred[parent_fid] = parent_values.pop()
                # Continue up the chain from the inferred parent
                child_fid = parent_fid
                child_val = inferred[parent_fid]
            else:
                # Ambiguous or no match → can't infer, stop
                break

    return inferred


def _get_ambiguous_parents(form, data):
    """Find fields with ambiguous parent inference.
    Returns {parent_field_id: [possible_value_1, possible_value_2, ...]}
    """
    ambiguous = {}
    for field in form["fields"]:
        fid = field["field_id"]
        if field.get("type") != "dropdown" or not field.get("parent_field_id"):
            continue
        if fid not in data:
            continue
        parent_fid = field["parent_field_id"]
        if parent_fid in data:
            continue  # parent already known

        matches = find_value_in_hierarchy(form, data[fid])
        parent_values = set()
        for m in matches:
            if m["field_id"] == fid:
                pv = m.get("parents", {}).get(parent_fid)
                if pv:
                    parent_values.add(pv)

        if len(parent_values) > 1:
            ambiguous[parent_fid] = list(parent_values)

    return ambiguous


def resolve_and_validate(form, candidate_data):
    """Full fixpoint engine:
    1. Resolve all field states (active, required, validation_rules)
    2. Clean data for inactive fields
    3. Infer missing parents from hierarchy
    4. Re-resolve states (inferred data may change states)
    5. Validate ALL fields with full context
    6. Repeat until stable

    Returns (resolved_data, inferred, conflicts, removed_fields):
    - resolved_data: candidate + inferred, cleaned of inactive data
    - inferred: {field_id: value} of inferred-only fields
    - conflicts: list of conflict dicts (empty = valid)
    - removed_fields: list of field_ids removed due to becoming inactive
    """
    resolved = dict(candidate_data)
    all_inferred = {}
    all_removed = []

    # Fixpoint loop — infer + resolve states + clean, repeat until stable
    for _ in range(10):
        # Step 1: Resolve field states
        field_states = resolve_all_field_states(form, resolved)

        # Step 2: Clean inactive field data
        resolved, removed = cleanup_inactive_data(form, resolved, field_states)
        all_removed.extend(removed)

        # Step 3: Infer missing parents
        new_inferred = infer_parents_from_hierarchy(form, resolved)
        fresh = {k: v for k, v in new_inferred.items() if k not in resolved}

        if not fresh and not removed:
            break  # stable — no new inferences, no new removals
        resolved.update(fresh)
        all_inferred.update(fresh)

    # Final state resolution after fixpoint
    field_states = resolve_all_field_states(form, resolved)

    conflicts = []

    # 1. Cross-field conditional conflicts (re-validate with full context)
    for field in form["fields"]:
        fid = field["field_id"]
        if fid not in resolved:
            continue
        if not field_states.get(fid, {}).get("active", True):
            continue
        if field.get("type") == "dropdown":
            continue
        if not field.get("validation_rules", {}).get("conditional_rules"):
            continue

        value = resolved[fid]
        is_valid, error = validate_field(form, fid, value, resolved)
        if not is_valid:
            triggered_by = _find_trigger_from_conditions(field, resolved)
            conflicts.append({
                "field": fid,
                "value": value,
                "reason": error + (f" (due to {triggered_by['field']}={triggered_by['value']})" if triggered_by else ""),
                "triggered_by": triggered_by,
            })

    # 2. Ambiguous parent validation
    ambiguous = _get_ambiguous_parents(form, resolved)
    for parent_fid, possible_values in ambiguous.items():
        for field in form["fields"]:
            fid = field["field_id"]
            if fid not in resolved:
                continue
            cond_rules = field.get("validation_rules", {}).get("conditional_rules", [])
            depends_on_parent = any(
                r.get("if", {}).get("field") == parent_fid for r in cond_rules
            )
            if not depends_on_parent:
                continue
            if parent_fid in resolved:
                continue

            value = resolved[fid]
            valid_under_any = False
            for pv in possible_values:
                test_data = {**resolved, parent_fid: pv}
                is_valid, _ = validate_field(form, fid, value, test_data)
                if is_valid:
                    valid_under_any = True
                    break

            if not valid_under_any:
                parent_field = get_field(form, parent_fid)
                parent_label = parent_field["label"] if parent_field else parent_fid
                conflicts.append({
                    "field": fid,
                    "value": value,
                    "reason": (
                        f"{field['label']}={value} is invalid for all possible "
                        f"{parent_label} values ({', '.join(possible_values)})"
                    ),
                    "triggered_by": {"field": parent_fid, "value": ", ".join(possible_values)},
                })

    # 3. Hierarchy consistency
    hierarchy_issues = validate_hierarchy_consistency(form, resolved)
    for hc in hierarchy_issues:
        conflicts.append({"field": "hierarchy", "reason": hc})

    return resolved, all_inferred, conflicts, all_removed


# === RULE ENGINE ===

# --- 1. Condition Evaluator (generic, operator-based) ---

def evaluate_condition(condition, collected_data):
    """Evaluate a single condition against collected_data.
    Supports operators: equals, not_equals, greater_than, less_than, in, not_in.
    Falls back to 'equals' if 'operator' is missing (backward compat).
    Returns True if condition is met.
    """
    cond_field = condition.get("field")
    if not cond_field or cond_field not in collected_data:
        return False

    actual = collected_data[cond_field]
    operator = condition.get("operator", "equals")
    expected = condition.get("value", condition.get("equals"))  # backward compat

    # Normalize for comparison
    actual_str = str(actual).lower()
    expected_str = str(expected).lower() if expected is not None else ""

    if operator == "equals":
        return actual_str == expected_str
    elif operator == "not_equals":
        return actual_str != expected_str
    elif operator == "greater_than":
        try:
            return float(actual) > float(expected)
        except (ValueError, TypeError):
            return False
    elif operator == "less_than":
        try:
            return float(actual) < float(expected)
        except (ValueError, TypeError):
            return False
    elif operator == "in":
        if isinstance(expected, list):
            return actual_str in [str(v).lower() for v in expected]
        return False
    elif operator == "not_in":
        if isinstance(expected, list):
            return actual_str not in [str(v).lower() for v in expected]
        return True

    return False


# --- 2. Field State Resolver ---

def resolve_field_state(field, collected_data):
    """Compute the dynamic state of a field based on its conditional_rules.
    Returns {active, required, validation_rules} — fully resolved.
    """
    # Base state from metadata
    state = {
        "active": field.get("active", True),
        "required": field.get("required", False),
        "validation_rules": {},
    }

    # Start with base validation rules (strip out conditional_rules)
    base_rules = dict(field.get("validation_rules", {}))
    validation_cond_rules = base_rules.pop("conditional_rules", [])
    state["validation_rules"] = base_rules

    # Merge field-level conditional_rules + validation conditional_rules
    all_conditional_rules = field.get("conditional_rules", []) + validation_cond_rules

    # Apply conditional_rules — each matching rule can override active, required, validation
    for rule in all_conditional_rules:
        condition = rule.get("if", {})
        if evaluate_condition(condition, collected_data):
            then = rule.get("then", {})
            # Override active/required if specified
            if "active" in then:
                state["active"] = then["active"]
            if "required" in then:
                state["required"] = then["required"]
            # Merge validation rule overrides
            rule_overrides = {k: v for k, v in then.items() if k not in ("active", "required")}
            state["validation_rules"].update(rule_overrides)

    return state


def resolve_all_field_states(form, collected_data):
    """Resolve states for ALL fields. Returns {field_id: {active, required, validation_rules}}."""
    states = {}
    for field in form["fields"]:
        states[field["field_id"]] = resolve_field_state(field, collected_data)
    return states


def cleanup_inactive_data(form, collected_data, field_states):
    """Remove collected_data for fields that are no longer active.
    Returns (cleaned_data, removed_fields).
    """
    cleaned = dict(collected_data)
    removed = []
    for fid, state in field_states.items():
        if not state["active"] and fid in cleaned:
            cleaned.pop(fid)
            removed.append(fid)
    return cleaned, removed


def resolve_rules(field, collected_data):
    """Resolve effective validation rules for a field (convenience wrapper).
    Uses the full field state resolver.
    """
    state = resolve_field_state(field, collected_data)
    return state["validation_rules"]


def _find_trigger_from_conditions(field, collected_data):
    """Find which conditional rule's condition is currently active for this field."""
    all_rules = field.get("validation_rules", {}).get("conditional_rules", [])
    all_rules += field.get("conditional_rules", [])
    for rule in all_rules:
        condition = rule.get("if", {})
        if evaluate_condition(condition, collected_data):
            cond_field = condition.get("field")
            if cond_field and cond_field in collected_data:
                return {"field": cond_field, "value": collected_data[cond_field]}
    return None


def build_conflict_suggestions(form, conflicts, collected_data):
    """Build smart user guidance for resolving conflicts.
    Fully metadata-driven — reads conflict details and suggests fixes.
    """
    suggestions = []
    for conflict in conflicts:
        fid = conflict.get("field")
        field = get_field(form, fid) if fid else None

        if field:
            suggestions.append(f"Change {field['label']} to a valid value")

        triggered_by = conflict.get("triggered_by")
        if triggered_by:
            trigger_field = get_field(form, triggered_by["field"])
            if trigger_field:
                alt_values = _get_alternative_values(form, triggered_by["field"], triggered_by["value"], collected_data)
                if alt_values:
                    suggestions.append(
                        f"Change {trigger_field['label']} to {' or '.join(alt_values)}"
                    )

    return suggestions


def _get_alternative_values(form, field_id, current_value, collected_data):
    """Get alternative values for a field (excluding the current one)."""
    field = get_field(form, field_id)
    if not field:
        return []

    if field["type"] == "dropdown":
        valid_opts = get_valid_dropdown_values(form, field_id, collected_data)
        if valid_opts:
            return [v for v in valid_opts if v.lower() != str(current_value).lower()]

    alt_values = set()
    for f in form["fields"]:
        for rule in f.get("validation_rules", {}).get("conditional_rules", []):
            cond = rule.get("if", {})
            if cond.get("field") == field_id and str(cond.get("equals", "")).lower() != str(current_value).lower():
                alt_values.add(cond["equals"])
    return list(alt_values)


# --- Validation ---

def validate_field(form, field_id, value, collected_data):
    """Validate a single field value against metadata rules.
    Returns (is_valid, error_message).
    """
    field = get_field(form, field_id)
    if not field:
        return False, f"Unknown field: {field_id}"

    field_type = field.get("type", "text")
    rules = resolve_rules(field, collected_data)

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

    # Password validation — fully metadata-driven via regex
    if field_type == "password":
        value = str(value)
        if "regex" in rules:
            match_result = re.match(rules["regex"], value)
            print(f"[PASSWORD VALIDATION] input='{value}', regex='{rules['regex']}', result={'PASS' if match_result else 'FAIL'}")
            if not match_result:
                desc = rules.get("regex_description", f"matching pattern {rules['regex']}")
                return False, f"{field['label']} must have {desc}."
        return True, None

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
    Works for any field type that has dropdown_options (dropdown, multi_select, etc.).
    Filters by collected ancestors (top-down) AND collected descendants (bottom-up).
    """
    field = get_field(form, field_id)
    if not field or not has_options(field):
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
    """Get missing fields using dynamic field states.
    Only active + required fields that are not yet collected count as missing.
    """
    field_states = resolve_all_field_states(form, collected_data)
    missing = []
    for field in form["fields"]:
        fid = field["field_id"]
        state = field_states.get(fid, {})
        if state.get("active", True) and state.get("required", False) and fid not in collected_data:
            missing.append(fid)
    return missing


def get_suggestions(form, collected_data, missing_fields, invalid_fields=None):
    """Generate suggestions for fields that have selectable options.
    Works for any field type with dropdown_options (dropdown, multi_select, etc.).
    Only suggests for the currently relevant field, not all future fields.
    """
    suggestions = []
    target_field_ids = []

    # Suggest for invalid fields that have options
    if invalid_fields:
        for inv in invalid_fields:
            field = get_field(form, inv["field_id"])
            if field and has_options(field):
                target_field_ids.append(inv["field_id"])

    # Suggest for the NEXT missing field — only if it has options
    if missing_fields:
        next_fid = missing_fields[0]
        next_field = get_field(form, next_fid)
        if next_field and has_options(next_field):
            if next_fid not in target_field_ids:
                target_field_ids.append(next_fid)

    for fid in target_field_ids:
        field = get_field(form, fid)
        if not field:
            continue

        valid_opts = get_valid_dropdown_values(form, fid, collected_data)
        if valid_opts:
            suggestions.append({
                "field_id": fid,
                "label": field["label"],
                "options": valid_opts,
            })

    return suggestions


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
6. If the user wants to UPDATE/CHANGE an already collected field (e.g., "change age to 30", "set country to Japan",
   "update my name to X"), extract that field with the new value — even if it was already collected.
   Also set "_intent": "update" in the JSON.
7. If you cannot extract any field, return empty object {{}}.
8. SENSITIVE FIELDS (type=password): Do NOT extract password fields. Return them as empty string "".
   Password values will be handled separately for security. NEVER guess or generate a password.

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
            if field["type"] == "password":
                pw_rules = field.get("validation_rules", {})
                reqs_text = pw_rules.get("regex_description", "a secure password")
                info += f" [PASSWORD FIELD - ask user to set a password with {reqs_text}. Do NOT suggest any password.]"
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
        "suggestions": get_suggestions(form, {}, missing),
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

    # Detect intent (update vs normal)
    intent = extracted.pop("_intent", "normal")
    is_update = intent == "update"

    # Step 2b: Handle sensitive fields (password) — use exact user input, not LLM output
    # Detect if the last question was about a password field by checking
    # what the next question WOULD be for the current collected_data state
    password_field_ids = [f["field_id"] for f in form["fields"] if f.get("type") == "password"]
    is_password_question = False

    # Determine which field was being asked: find the first missing field that
    # would have been asked (same logic as call_openai_next_question)
    missing_now = get_missing_fields(form, collected_data)
    currently_asking = None
    for mfid in missing_now:
        mfield = get_field(form, mfid)
        if mfield and mfield["type"] == "dropdown":
            valid_opts = get_valid_dropdown_values(form, mfid, collected_data)
            if not valid_opts:
                continue  # skip fields with no valid options yet
        currently_asking = mfid
        break

    for fid in password_field_ids:
        # Always remove LLM-extracted passwords (prevents hallucination)
        extracted.pop(fid, None)

        if fid not in collected_data and currently_asking == fid:
            is_password_question = True
            # Use raw user message as the password value
            extracted = {fid: req.message.strip()}

    # ===== ATOMIC TRANSACTION: validate everything before storing anything =====
    #
    # Phase 1: Filter & normalize extracted fields (no storage)
    # Phase 2: Build candidate state with ALL new fields applied
    # Phase 3: Validate candidate as a whole (cross-field + conditional + hierarchy)
    # Phase 4: COMMIT only if everything passes, else ROLLBACK entire batch

    # --- Phase 1: Filter, normalize, check field priority ---
    pending_data = {}  # fields that passed basic checks, NOT yet stored
    invalid_fields = []

    field_order = {f["field_id"]: i for i, f in enumerate(form["fields"])}
    sorted_fields = sorted(
        extracted.items(),
        key=lambda item: field_order.get(item[0], 999),
    )

    for field_id, value in sorted_fields:
        if value is None or value == "":
            continue

        # Field priority: first valid value is preserved unless explicit update
        if field_id in collected_data and not is_update:
            existing = collected_data[field_id]
            if str(existing).lower() == str(value).lower():
                continue
            field = get_field(form, field_id)
            label = field["label"] if field else field_id
            invalid_fields.append({
                "field_id": field_id,
                "value": value,
                "error": (
                    f"You already provided {label} as '{existing}'. "
                    f"If you want to change it, say 'change {label.lower()} to {value}'."
                ),
            })
            continue

        # Basic per-field validation (type, format, regex, dropdown options)
        running_data = {**collected_data, **pending_data}
        if is_update and field_id in running_data:
            running_data.pop(field_id)

        is_valid, error = validate_field(form, field_id, value, running_data)
        if is_valid:
            # Normalize value (correct casing for dropdowns, type for numbers)
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
            pending_data[field_id] = value
        else:
            invalid_fields.append({"field_id": field_id, "value": value, "error": error})

    # --- Phase 2: Build candidate state (collected + ALL pending) ---
    candidate_data = dict(collected_data)

    if is_update and pending_data:
        for updated_fid in pending_data:
            field_def = get_field(form, updated_fid)
            if field_def and field_def["type"] == "dropdown":
                descendants = get_all_descendant_field_ids(form, updated_fid)
                for desc_fid in descendants:
                    candidate_data.pop(desc_fid, None)

    candidate_data.update(pending_data)

    # Auto-fill dropdown fields that have exactly one valid option
    auto_filled = {}
    changed = True
    while changed:
        changed = False
        for field in form["fields"]:
            fid = field["field_id"]
            if fid in candidate_data or field["type"] != "dropdown":
                continue
            valid_opts = get_valid_dropdown_values(form, fid, candidate_data)
            if valid_opts and len(valid_opts) == 1:
                candidate_data[fid] = valid_opts[0]
                auto_filled[fid] = valid_opts[0]
                changed = True

    # --- Phase 3: Fixpoint resolve + validate atomically ---
    # 1. Infer missing parents (state=Kerala → country=India)
    # 2. Re-validate ALL fields with full inferred context
    # 3. Repeat until stable (fixpoint)
    resolved_data, inferred, all_conflicts, removed_fields = resolve_and_validate(form, candidate_data)

    # Track which fields were inferred (not explicitly provided by user)
    auto_filled.update(inferred)

    if all_conflicts or invalid_fields:
        # REJECT new input — existing state is immutable
        # Never remove or modify existing valid data due to new conflicting input
        write_json("collected_data.json", collected_data)

        # Build error details
        all_errors = []
        for c in all_conflicts:
            all_errors.append({"field_id": c["field"], "value": c.get("value"), "error": c["reason"]})
        all_errors.extend(invalid_fields)

        # Build suggestions for conflicts
        conflict_suggestions = build_conflict_suggestions(form, all_conflicts, resolved_data) if all_conflicts else []

        error_msg = call_openai_error_message(form, all_errors, req.message, collected_data)
        missing = get_missing_fields(form, collected_data)

        if missing:
            next_q = call_openai_next_question(form, collected_data, missing)
            response_msg = error_msg + "\n\n" + next_q
        else:
            response_msg = error_msg

        status = "conflict" if all_conflicts else "pending"

        messages.append({"role": "assistant", "content": response_msg})
        write_json("messages.json", messages)

        safe_collected = dict(collected_data)
        for field in form["fields"]:
            if field.get("type") == "password" and field["field_id"] in safe_collected:
                safe_collected[field["field_id"]] = "********"

        result = {
            "status": status,
            "message": response_msg,
            "collected_data": safe_collected,
            "missing_fields": missing,
            "invalid_fields": invalid_fields,
            "suggestions": conflict_suggestions + get_suggestions(form, collected_data, missing, invalid_fields),
        }
        if all_conflicts:
            result["conflicts"] = [{"field": c["field"], "reason": c["reason"]} for c in all_conflicts]
        return result

    # --- Phase 4: COMMIT — all validations passed, no conflicts ---
    collected_data = resolved_data
    write_json("collected_data.json", collected_data)

    # Step 5: Determine response (we only reach here if Phase 3 passed — state is clean)
    missing = get_missing_fields(form, collected_data)

    if not missing:
        response_msg = "All information has been collected. Thank you!"
        status = "complete"
    else:
        response_msg = call_openai_next_question(form, collected_data, missing, auto_filled=auto_filled)
        status = "pending"

    messages.append({"role": "assistant", "content": response_msg})
    write_json("messages.json", messages)

    # Mask sensitive fields in the response
    safe_collected = dict(collected_data)
    for field in form["fields"]:
        if field.get("type") == "password" and field["field_id"] in safe_collected:
            safe_collected[field["field_id"]] = "********"

    return {
        "status": status,
        "message": response_msg,
        "collected_data": safe_collected,
        "missing_fields": missing,
        "invalid_fields": invalid_fields,
        "suggestions": get_suggestions(form, collected_data, missing, invalid_fields),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
