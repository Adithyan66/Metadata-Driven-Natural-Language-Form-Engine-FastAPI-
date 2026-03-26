"""Core engine: dependency graph, inference, fixpoint resolve-and-validate."""

from app.hierarchy import (
    get_field,
    find_value_in_hierarchy,
    validate_hierarchy_consistency,
)
from app.validation import (
    validate_field,
    resolve_all_field_states,
    cleanup_inactive_data,
    _find_trigger_from_conditions,
)


def build_dependency_graph(form):
    """Build a dependency graph from metadata."""
    hierarchy_deps = {}
    conditional_deps = {}

    for field in form["fields"]:
        fid = field["field_id"]

        if field.get("parent_field_id"):
            hierarchy_deps[fid] = field["parent_field_id"]

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
    """Forward inference: child value → unambiguous parent inference."""
    inferred = {}

    for field in form["fields"]:
        fid = field["field_id"]
        if field.get("type") != "dropdown" or not field.get("parent_field_id"):
            continue
        if fid not in data:
            continue

        child_fid = fid
        child_val = data[child_fid]

        while True:
            child_field = get_field(form, child_fid)
            if not child_field or not child_field.get("parent_field_id"):
                break
            parent_fid = child_field["parent_field_id"]

            if parent_fid in data or parent_fid in inferred:
                break

            matches = find_value_in_hierarchy(form, child_val)
            parent_values = set()
            for m in matches:
                if m["field_id"] == child_fid:
                    parent_val = m.get("parents", {}).get(parent_fid)
                    if parent_val:
                        parent_values.add(parent_val)

            if len(parent_values) == 1:
                inferred[parent_fid] = parent_values.pop()
                child_fid = parent_fid
                child_val = inferred[parent_fid]
            else:
                break

    return inferred


def _get_ambiguous_parents(form, data):
    """Find fields with ambiguous parent inference."""
    ambiguous = {}
    for field in form["fields"]:
        fid = field["field_id"]
        if field.get("type") != "dropdown" or not field.get("parent_field_id"):
            continue
        if fid not in data:
            continue
        parent_fid = field["parent_field_id"]
        if parent_fid in data:
            continue

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
    1. Resolve field states (active, required, validation_rules)
    2. Clean inactive data
    3. Infer missing parents
    4. Repeat until stable
    5. Validate ALL fields

    Returns (resolved_data, inferred, conflicts, removed_fields).
    """
    resolved = dict(candidate_data)
    all_inferred = {}
    all_removed = []

    for _ in range(10):
        field_states = resolve_all_field_states(form, resolved)
        resolved, removed = cleanup_inactive_data(form, resolved, field_states)
        all_removed.extend(removed)

        new_inferred = infer_parents_from_hierarchy(form, resolved)
        fresh = {k: v for k, v in new_inferred.items() if k not in resolved}

        if not fresh and not removed:
            break
        resolved.update(fresh)
        all_inferred.update(fresh)

    field_states = resolve_all_field_states(form, resolved)
    conflicts = []

    # 1. Cross-field conditional conflicts
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
