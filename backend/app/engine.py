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
    print("    [engine] infer_parents_from_hierarchy")
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
            # Filter matches by already-known ancestors (in data or inferred)
            known = {**data, **inferred}
            compatible_matches = []
            for m in matches:
                if m["field_id"] != child_fid:
                    continue
                # Check all known ancestors match this path
                is_compatible = True
                for ancestor_fid, ancestor_val in m.get("parents", {}).items():
                    if ancestor_fid in known and known[ancestor_fid].lower() != ancestor_val.lower():
                        is_compatible = False
                        break
                if is_compatible:
                    compatible_matches.append(m)

            parent_values = set()
            for m in compatible_matches:
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
    """Find fields with ambiguous parent inference.
    Walks up the ENTIRE ancestor chain — if district is ambiguous,
    state and country are also ambiguous.
    """
    ambiguous = {}
    for field in form["fields"]:
        fid = field["field_id"]
        if field.get("type") != "dropdown" or not field.get("parent_field_id"):
            continue
        if fid not in data:
            continue

        # Find all possible parent paths for this value
        matches = find_value_in_hierarchy(form, data[fid])
        field_matches = [m for m in matches if m["field_id"] == fid]

        if len(field_matches) <= 1:
            continue

        # For EACH ancestor level, collect possible values
        # e.g., Ward200 → district: [Alappuzha, Shenzhen], state: [Kerala, Guangdong], country: [India, China]
        all_parent_contexts = [m.get("parents", {}) for m in field_matches]

        for ancestor_fid in set().union(*[ctx.keys() for ctx in all_parent_contexts]):
            if ancestor_fid in data:
                continue  # already known, not ambiguous
            ancestor_values = set()
            for ctx in all_parent_contexts:
                if ancestor_fid in ctx:
                    ancestor_values.add(ctx[ancestor_fid])
            if len(ancestor_values) > 1:
                if ancestor_fid not in ambiguous:
                    ambiguous[ancestor_fid] = list(ancestor_values)

    # Also handle the direct parent case
    for field in form["fields"]:
        fid = field["field_id"]
        if field.get("type") != "dropdown" or not field.get("parent_field_id"):
            continue
        if fid not in data:
            continue
        parent_fid = field["parent_field_id"]
        if parent_fid in data or parent_fid in ambiguous:
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
    print("    [engine] resolve_and_validate START")
    print(f"    [engine] candidate_data: {candidate_data}")
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

    # 2. Ambiguous parent validation + narrowing
    # For each ambiguous parent, test which values are valid.
    # If only ONE survives → infer it and re-validate.
    ambiguous = _get_ambiguous_parents(form, resolved)
    narrowed = {}  # parent_fid → narrowed possible values

    for parent_fid, possible_values in ambiguous.items():
        surviving_values = set(possible_values)

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
            # Test each possible parent value — keep only those that pass
            valid_parents = set()
            for pv in possible_values:
                test_data = {**resolved, parent_fid: pv}
                is_valid, _ = validate_field(form, fid, value, test_data)
                if is_valid:
                    valid_parents.add(pv)

            # Narrow: surviving = intersection with this field's valid parents
            surviving_values &= valid_parents

        narrowed[parent_fid] = surviving_values

        if len(surviving_values) == 0:
            # Invalid under ALL possibilities → conflict
            parent_field = get_field(form, parent_fid)
            parent_label = parent_field["label"] if parent_field else parent_fid

            # Find the child field that made the parent ambiguous (e.g., Ward200)
            ambiguous_source = None
            for f in form["fields"]:
                f_fid = f["field_id"]
                if f_fid in resolved and f.get("parent_field_id"):
                    matches = find_value_in_hierarchy(form, resolved[f_fid])
                    field_matches = [m for m in matches if m["field_id"] == f_fid]
                    if len(field_matches) > 1:
                        parent_vals = set()
                        for m in field_matches:
                            pv = m.get("parents", {}).get(parent_fid)
                            if pv:
                                parent_vals.add(pv)
                        if parent_vals == set(possible_values):
                            ambiguous_source = {
                                "field": f.get("label", f_fid),
                                "value": resolved[f_fid],
                            }
                            break

            # Find which field caused the conflict and build per-value reasons
            for field in form["fields"]:
                fid = field["field_id"]
                if fid not in resolved:
                    continue
                cond_rules = field.get("validation_rules", {}).get("conditional_rules", [])
                if not any(r.get("if", {}).get("field") == parent_fid for r in cond_rules):
                    continue
                if parent_fid in resolved:
                    continue

                # Build per-parent-value detail (e.g., "India: Age must be at least 18")
                per_value_details = []
                for pv in possible_values:
                    test_data = {**resolved, parent_fid: pv}
                    _, err = validate_field(form, fid, resolved[fid], test_data)
                    if err:
                        per_value_details.append(f"{pv}: {err}")

                reason = (
                    f"{field['label']}={resolved[fid]} is invalid for all possible "
                    f"{parent_label} values ({', '.join(possible_values)})"
                )
                if per_value_details:
                    reason += ". " + "; ".join(per_value_details)

                conflict = {
                    "field": fid,
                    "value": resolved[fid],
                    "reason": reason,
                    "triggered_by": {"field": parent_fid, "value": ", ".join(possible_values)},
                }
                if ambiguous_source:
                    conflict["ambiguous_source"] = ambiguous_source
                conflicts.append(conflict)

        elif len(surviving_values) == 1:
            # Exactly ONE valid parent → infer it!
            inferred_val = surviving_values.pop()
            resolved[parent_fid] = inferred_val
            all_inferred[parent_fid] = inferred_val

    # If we inferred new parents from narrowing, re-run fixpoint to cascade
    newly_inferred = {k for k, v in narrowed.items() if k in all_inferred}
    if newly_inferred:
        for _ in range(10):
            new_inferred = infer_parents_from_hierarchy(form, resolved)
            fresh = {k: v for k, v in new_inferred.items() if k not in resolved}
            if not fresh:
                break
            resolved.update(fresh)
            all_inferred.update(fresh)

    # 3. Hierarchy consistency
    hierarchy_issues = validate_hierarchy_consistency(form, resolved)
    for hc in hierarchy_issues:
        conflicts.append({
            "field": "hierarchy",
            "reason": hc["reason"],
            "involved_fields": hc["involved_fields"],
        })

    print(f"    [engine] resolve_and_validate END")
    print(f"    [engine] resolved: {resolved}")
    print(f"    [engine] inferred: {all_inferred}")
    print(f"    [engine] conflicts: {conflicts}")
    return resolved, all_inferred, conflicts, all_removed
