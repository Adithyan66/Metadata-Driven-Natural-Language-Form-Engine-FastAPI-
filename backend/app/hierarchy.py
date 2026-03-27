"""Dropdown hierarchy traversal helpers.

All functions that walk the tree structure in dropdown_options:
parent-child lookups, descendant/ancestor discovery, filtering.
"""


def get_field(form, field_id):
    for f in form["fields"]:
        if f["field_id"] == field_id:
            return f
    return None


def has_options(field):
    """Check if a field has selectable options (metadata-driven).
    Checks for dropdown_options or parent_field_id instead of hardcoding type names.
    """
    if not field:
        return False
    return bool(field.get("dropdown_options")) or bool(field.get("parent_field_id"))


def find_hierarchy_root(form, field_id):
    """Find the top-level field that owns the hierarchy containing field_id."""
    field = get_field(form, field_id)
    if not field:
        return None
    if not field.get("parent_field_id"):
        for f in form["fields"]:
            if f["type"] == "dropdown" and not f.get("parent_field_id"):
                if _tree_contains_field(f.get("dropdown_options", []), field_id):
                    return f
        return field
    return find_hierarchy_root(form, field["parent_field_id"])


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
            children = opt.get("children", {})
            if children:
                match["children_field_id"] = children.get("field_id")
            matches.append(match)

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


def get_valid_dropdown_values(form, field_id, collected_data):
    """Get valid dropdown values filtered BIDIRECTIONALLY.
    Works for any field type that has dropdown_options (dropdown, multi_select, etc.).
    """
    field = get_field(form, field_id)
    if not field or not has_options(field):
        return None

    # Simple dropdown (no hierarchy)
    if not field.get("parent_field_id"):
        root_field = find_hierarchy_root(form, field_id)
        if root_field and root_field["field_id"] == field_id:
            all_options = [o["value"] for o in field.get("dropdown_options", [])]
            if not all_options:
                return None
            return _filter_by_descendants(form, field_id, all_options, collected_data)
        own_options = [o["value"] for o in field.get("dropdown_options", [])]
        return own_options if own_options else None

    # Part of a hierarchy
    root_field = find_hierarchy_root(form, field_id)
    if not root_field:
        return []

    all_occurrences = _gather_field_occurrences(
        root_field["field_id"], root_field.get("dropdown_options", []), field_id
    )

    valid_values = set()
    for occ in all_occurrences:
        context = occ["context"]
        value = occ["value"]

        ancestor_ok = all(
            context.get(fid, "").lower() == collected_data[fid].lower()
            for fid in context
            if fid in collected_data
        )
        if not ancestor_ok:
            continue

        descendant_ok = True
        occ_descendants = occ.get("descendants", {})
        for desc_fid, desc_vals in occ_descendants.items():
            if desc_fid in collected_data:
                if collected_data[desc_fid].lower() not in [v.lower() for v in desc_vals]:
                    descendant_ok = False
                    break
        if descendant_ok:
            all_possible_descs = get_all_descendant_field_ids(form, field_id)
            for desc_fid in all_possible_descs:
                if desc_fid in collected_data and desc_fid not in occ_descendants:
                    descendant_ok = False
                    break
        if not descendant_ok:
            continue

        valid_values.add(value)

    return sorted(valid_values) if valid_values else []


def _gather_field_occurrences(current_field_id, options, target_field_id, context=None):
    """Walk the tree and collect every occurrence of target_field_id with full context."""
    if context is None:
        context = {}
    results = []

    for opt in options:
        children = opt.get("children", {})
        child_field_id = children.get("field_id")
        child_options = children.get("options", [])

        if current_field_id == target_field_id:
            descendants = {}
            _collect_descendants(opt, descendants)
            results.append({
                "value": opt["value"],
                "context": dict(context),
                "descendants": descendants,
            })
        elif child_field_id:
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
        return options

    field = get_field(form, field_id)
    if not field:
        return options

    filtered = []
    for opt_val in options:
        opt_node = next(
            (o for o in field.get("dropdown_options", []) if o["value"] == opt_val), None
        )
        if not opt_node:
            continue
        descendants = {}
        _collect_descendants(opt_node, descendants)
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
            elif desc_fid in [f["field_id"] for f in form["fields"] if f.get("parent_field_id")]:
                all_match = False
                break
        if all_match:
            filtered.append(opt_val)

    return filtered


def check_hierarchy_conflict(form, field_id, value, collected_data):
    """Check if setting field_id=value conflicts with already collected hierarchical data."""
    matches = find_value_in_hierarchy(form, value)
    field_matches = [m for m in matches if m["field_id"] == field_id]

    if not field_matches:
        return False, f"'{value}' is not a valid option for {field_id}."

    all_descendants = get_all_descendant_field_ids(form, field_id)

    for match in field_matches:
        ancestor_conflict = False
        for parent_fid, parent_val in match["parents"].items():
            if parent_fid in collected_data and collected_data[parent_fid].lower() != parent_val.lower():
                ancestor_conflict = True
                break
        if ancestor_conflict:
            continue

        descendant_conflict = False
        for desc_fid in all_descendants:
            if desc_fid not in collected_data:
                continue
            desc_val = collected_data[desc_fid]
            desc_matches = find_value_in_hierarchy(form, desc_val)
            valid_under_branch = any(
                m["field_id"] == desc_fid and
                m.get("parents", {}).get(field_id, "").lower() == value.lower()
                for m in desc_matches
            )
            if not valid_under_branch:
                descendant_conflict = True
                break

        if not descendant_conflict:
            return True, None

    # Build error with field labels (not raw field_ids)
    field_def = get_field(form, field_id)
    field_label = field_def["label"] if field_def else field_id

    # Ancestor conflict — value belongs to a different parent than what's collected
    for match in field_matches:
        for parent_fid, parent_val in match["parents"].items():
            if parent_fid in collected_data and collected_data[parent_fid].lower() != parent_val.lower():
                parent_field = get_field(form, parent_fid)
                parent_label = parent_field["label"] if parent_field else parent_fid
                return False, (
                    f"'{value}' belongs to {parent_label}='{parent_val}', "
                    f"but you already selected {parent_label}='{collected_data[parent_fid]}'."
                )

    # Descendant conflict — a child/grandchild value doesn't exist under the new value
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
                desc_field = get_field(form, desc_fid)
                desc_label = desc_field["label"] if desc_field else desc_fid
                return False, (
                    f"Cannot set {field_label} to '{value}' because you already selected "
                    f"{desc_label}='{desc_val}', which does not belong under '{value}'. "
                    f"To change {field_label}, first remove or update {desc_label} "
                    f"(say 'delete {desc_label.lower()}' or 'change {desc_label.lower()} to ...')."
                )

    return False, f"'{value}' is not valid for {field_label} given the current selections."


def validate_hierarchy_consistency(form, collected_data):
    """Check that all collected hierarchical fields form a valid path.
    Returns list of dicts: {reason, involved_fields: [field_id, ...]}.
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
        child_matches = find_value_in_hierarchy(form, child_val)
        valid = any(
            m["field_id"] == child_fid and
            m.get("parents", {}).get(parent_fid, "").lower() == parent_val.lower()
            for m in child_matches
        )
        if not valid:
            conflicts.append({
                "reason": (
                    f"'{child_val}' ({child_fid}) does not belong to "
                    f"'{parent_val}' ({parent_fid})."
                ),
                "involved_fields": [child_fid, parent_fid],
            })
    return conflicts
