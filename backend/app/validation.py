"""Rule engine + field validation.

Condition evaluator, field state resolver, field validation, suggestions, conflicts.
"""

import re

from app.hierarchy import (
    get_field,
    has_options,
    check_hierarchy_conflict,
    get_valid_dropdown_values,
    get_all_ancestor_field_ids,
    find_value_in_hierarchy,
)


# === Condition Evaluator ===

def evaluate_condition(condition, collected_data):
    """Evaluate a single condition against collected_data.
    Supports: equals, not_equals, greater_than, less_than, in, not_in.
    """
    cond_field = condition.get("field")
    if not cond_field or cond_field not in collected_data:
        return False

    actual = collected_data[cond_field]
    operator = condition.get("operator", "equals")
    expected = condition.get("value", condition.get("equals"))  # backward compat

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


# === Field State Resolver ===

def resolve_field_state(field, collected_data):
    """Compute the dynamic state of a field: {active, required, validation_rules}."""
    state = {
        "active": field.get("active", True),
        "required": field.get("required", False),
        "validation_rules": {},
    }

    base_rules = dict(field.get("validation_rules", {}))
    validation_cond_rules = base_rules.pop("conditional_rules", [])
    state["validation_rules"] = base_rules

    all_conditional_rules = field.get("conditional_rules", []) + validation_cond_rules

    for rule in all_conditional_rules:
        condition = rule.get("if", {})
        if evaluate_condition(condition, collected_data):
            then = rule.get("then", {})
            if "active" in then:
                state["active"] = then["active"]
            if "required" in then:
                state["required"] = then["required"]
            rule_overrides = {k: v for k, v in then.items() if k not in ("active", "required")}
            state["validation_rules"].update(rule_overrides)

    return state


def resolve_all_field_states(form, collected_data):
    """Resolve states for ALL fields."""
    states = {}
    for field in form["fields"]:
        states[field["field_id"]] = resolve_field_state(field, collected_data)
    return states


def cleanup_inactive_data(form, collected_data, field_states):
    """Remove collected_data for fields that are no longer active."""
    cleaned = dict(collected_data)
    removed = []
    for fid, state in field_states.items():
        if not state["active"] and fid in cleaned:
            cleaned.pop(fid)
            removed.append(fid)
    return cleaned, removed


def resolve_rules(field, collected_data):
    """Resolve effective validation rules for a field (convenience wrapper)."""
    state = resolve_field_state(field, collected_data)
    return state["validation_rules"]


# === Field Validation ===

def validate_field(form, field_id, value, collected_data):
    """Validate a single field value against metadata rules.
    Returns (is_valid, error_message).
    """
    field = get_field(form, field_id)
    if not field:
        return False, f"Unknown field: {field_id}"

    field_type = field.get("type", "text")
    rules = resolve_rules(field, collected_data)

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

    if field_type == "password":
        value = str(value)
        if "regex" in rules:
            if not re.match(rules["regex"], value):
                desc = rules.get("regex_description", f"matching pattern {rules['regex']}")
                return False, f"{field['label']} must have {desc}."
        return True, None

    if field_type == "dropdown":
        value = str(value) if value is not None else ""
        if not value:
            return False, f"{field['label']} cannot be empty."
        is_valid, conflict_msg = check_hierarchy_conflict(form, field_id, value, collected_data)
        if not is_valid:
            return False, conflict_msg

        parent_fid = field.get("parent_field_id")
        valid_options = get_valid_dropdown_values(form, field_id, collected_data)

        if valid_options:
            matched = [o for o in valid_options if o.lower() == str(value).lower()]
            if not matched:
                # Build a clear reason explaining WHY options are limited
                # Check if a descendant field is constraining this field
                from app.hierarchy import get_all_descendant_field_ids
                constraining_fields = []
                descendants = get_all_descendant_field_ids(form, field_id)
                for desc_fid in descendants:
                    if desc_fid in collected_data:
                        desc_field = get_field(form, desc_fid)
                        desc_label = desc_field["label"] if desc_field else desc_fid
                        constraining_fields.append(f"{desc_label}='{collected_data[desc_fid]}'")

                if constraining_fields:
                    reason = (
                        f"'{value}' is not valid for {field['label']} because you already selected "
                        f"{', '.join(constraining_fields)} which only belongs under "
                        f"{', '.join(valid_options)}. To change {field['label']}, "
                        f"first update or remove {', '.join(constraining_fields)}."
                    )
                else:
                    reason = f"'{value}' is not valid for {field['label']}. Valid options: {', '.join(valid_options)}"
                return False, reason
        elif parent_fid and parent_fid not in collected_data:
            all_matches = find_value_in_hierarchy(form, str(value))
            matched_for_field = [m for m in all_matches if m["field_id"] == field_id]
            if not matched_for_field:
                return False, f"'{value}' is not a valid option for {field['label']}."
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
            matched = [o for o in (valid_options or []) if o.lower() == str(value).lower()]
            if not matched:
                return False, f"'{value}' is not valid for {field['label']}."

    return True, None


# === Missing fields & suggestions ===

def get_missing_fields(form, collected_data):
    """Only active + required fields that are not yet collected."""
    field_states = resolve_all_field_states(form, collected_data)
    missing = []
    for field in form["fields"]:
        fid = field["field_id"]
        state = field_states.get(fid, {})
        if state.get("active", True) and state.get("required", False) and fid not in collected_data:
            missing.append(fid)
    return missing


def get_currently_asking(form, collected_data):
    """Determine which field would be asked next.
    Skips option-based fields that have no valid options yet.
    Returns (field_id, field_def) or (None, None).
    """
    missing = get_missing_fields(form, collected_data)
    for fid in missing:
        field = get_field(form, fid)
        if not field:
            continue
        if has_options(field):
            valid_opts = get_valid_dropdown_values(form, fid, collected_data)
            if not valid_opts:
                continue
        return fid, field
    return None, None


def get_suggestions(form, collected_data, missing_fields, invalid_fields=None, currently_asking=None):
    """Suggestions only for the field being asked, and only if it has options."""
    suggestions = []
    target_field_ids = []

    # Only suggest for currently_asking if it has options
    if currently_asking:
        field = get_field(form, currently_asking)
        if field and has_options(field):
            target_field_ids.append(currently_asking)

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


# === Conflict helpers ===

def _find_trigger_from_conditions(field, collected_data):
    """Find which conditional rule's condition is currently active."""
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
    """Build user guidance for resolving conflicts."""
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

    if has_options(field):
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
