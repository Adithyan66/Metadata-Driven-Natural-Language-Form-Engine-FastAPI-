"""All graph node functions — each handles one step of the chat flow."""

from app.hierarchy import (
    get_field,
    has_options,
    get_all_descendant_field_ids,
    get_valid_dropdown_values,
)
from app.validation import (
    validate_field,
    get_missing_fields,
    get_currently_asking,
    build_conflict_suggestions,
)
from app.engine import resolve_and_validate
from app.llm import (
    call_openai_extract,
    call_openai_next_question,
    call_openai_error_message,
    call_openai_answer_query,
    call_openai_nudge_message,
)
from app.graph.state import FormState


# --- Helpers ---

def _with_query(query_answer, msg):
    """Prepend query answer to a message."""
    return f"{query_answer}\n\n{msg}" if query_answer else msg


# --- Nodes ---

def load_state(state: FormState) -> dict:
    """Determine currently_asking field."""
    asking_fid, asking_field = get_currently_asking(state["form"], state["collected_data"])
    return {
        "currently_asking": asking_fid,
        "currently_asking_field": asking_field,
    }


def extract(state: FormState) -> dict:
    """Run LLM extraction or handle sensitive fields."""
    form = state["form"]
    collected_data = state["collected_data"]
    currently_asking = state["currently_asking"]
    user_message = state["user_message"]

    password_field_ids = [f["field_id"] for f in form["fields"] if f.get("type") == "password"]

    if currently_asking in password_field_ids:
        extracted = {currently_asking: user_message.strip()}
    else:
        extracted = call_openai_extract(
            user_message, form, collected_data,
            currently_asking=currently_asking,
            currently_asking_field=state["currently_asking_field"],
        )
        for fid in password_field_ids:
            extracted.pop(fid, None)

    return {"extracted": extracted}


def parse_intent(state: FormState) -> dict:
    """Separate metadata keys (_intent, _delete, _query, _uncertain) from extracted data."""
    extracted = dict(state["extracted"])

    is_uncertain = extracted.pop("_uncertain", False)
    intent = extracted.pop("_intent", "normal")
    is_update = intent == "update"

    delete_fields = extracted.pop("_delete", [])
    if not isinstance(delete_fields, list):
        delete_fields = [delete_fields] if delete_fields else []

    query = extracted.pop("_query", None)

    return {
        "extracted": extracted,
        "is_uncertain": is_uncertain,
        "is_update": is_update,
        "intent": intent,
        "delete_fields": delete_fields,
        "query": query,
    }


def process_query(state: FormState) -> dict:
    """Answer user's question if _query is present."""
    query = state["query"]
    if not query:
        return {"query_answer": None}
    return {"query_answer": call_openai_answer_query(query, state["form"], state["collected_data"])}


def process_deletes(state: FormState) -> dict:
    """Cascade delete fields + re-validate."""
    delete_fields = state["delete_fields"]
    if not delete_fields:
        return {"deleted_labels": [], "collected_data": state["collected_data"]}

    form = state["form"]
    collected_data = dict(state["collected_data"])
    deleted_labels = []

    for del_fid in delete_fields:
        if del_fid not in collected_data:
            continue
        field_def = get_field(form, del_fid)
        label = field_def["label"] if field_def else del_fid
        collected_data.pop(del_fid, None)
        deleted_labels.append(label)

        for desc_fid in get_all_descendant_field_ids(form, del_fid):
            if desc_fid in collected_data:
                desc_field = get_field(form, desc_fid)
                deleted_labels.append(f"{desc_field['label'] if desc_field else desc_fid} (dependent)")
                collected_data.pop(desc_fid)

    resolved, _, _, removed = resolve_and_validate(form, collected_data)
    collected_data = resolved
    for rfid in removed:
        f = get_field(form, rfid)
        deleted_labels.append(f"{f['label']} (no longer active)" if f else rfid)

    return {"deleted_labels": deleted_labels, "collected_data": collected_data}


def sanitize(state: FormState) -> dict:
    """Reject hallucinated values via source verification. Tracks dropped values with reasons."""
    extracted = state["extracted"]
    form = state["form"]
    collected_data = state["collected_data"]
    user_msg_lower = state["user_message"].strip().lower()

    sanitized = {}
    dropped = []

    for fid, val in extracted.items():
        if val is None or val == "":
            continue
        field_def = get_field(form, fid)
        if not field_def:
            continue
        ftype = field_def.get("type", "text")
        label = field_def.get("label", fid)
        val_str = str(val).lower()

        if ftype == "number":
            try:
                float(val)
            except (ValueError, TypeError):
                dropped.append({"field": label, "value": val, "reason": f"'{val}' is not a valid number"})
                continue
            if not any(ch.isdigit() for ch in user_msg_lower):
                dropped.append({"field": label, "value": val, "reason": f"No number found in your message for {label}"})
                continue
            sanitized[fid] = val

        elif has_options(field_def):
            valid_opts = get_valid_dropdown_values(form, fid, collected_data)
            if valid_opts:
                if any(o.lower() == val_str for o in valid_opts):
                    sanitized[fid] = val
                else:
                    # Build reason: check descendants first (child constraining parent),
                    # then parent context, then generic
                    reason = None

                    # Check if a DESCENDANT field is constraining this field
                    descendants = get_all_descendant_field_ids(form, fid)
                    constraining = []
                    for desc_fid in descendants:
                        if desc_fid in collected_data:
                            desc_field = get_field(form, desc_fid)
                            desc_label = desc_field["label"] if desc_field else desc_fid
                            constraining.append(f"{desc_label}='{collected_data[desc_fid]}'")
                    if constraining:
                        reason = (
                            f"Cannot set {label} to '{val}' because you already selected "
                            f"{', '.join(constraining)} which only belongs under "
                            f"{', '.join(valid_opts)}. To change {label}, first remove or "
                            f"update {', '.join(constraining)}."
                        )

                    # Check parent context
                    if not reason:
                        parent_fid = field_def.get("parent_field_id")
                        if parent_fid:
                            parent_field = get_field(form, parent_fid)
                            parent_label = parent_field["label"] if parent_field else parent_fid
                            parent_val = collected_data.get(parent_fid)
                            if parent_val:
                                reason = f"'{val}' is not available under {parent_label} '{parent_val}'. Available: {', '.join(valid_opts)}"

                    if not reason:
                        reason = f"'{val}' is not a valid {label}. Available: {', '.join(valid_opts)}"

                    dropped.append({"field": label, "value": val, "reason": reason})
                    continue
            else:
                # valid_opts is empty or None
                if valid_opts is not None and len(valid_opts) == 0:
                    # Empty list = no options exist for current selection
                    parent_fid = field_def.get("parent_field_id")
                    if parent_fid and parent_fid in collected_data:
                        parent_field = get_field(form, parent_fid)
                        parent_label = parent_field["label"] if parent_field else parent_fid
                        parent_val = collected_data[parent_fid]
                        reason = (
                            f"{label} is not applicable for your current {parent_label} '{parent_val}'. "
                            f"There are no {label.lower()} options available under '{parent_val}'."
                        )
                    else:
                        reason = f"No {label.lower()} options are available for your current selections."
                    dropped.append({"field": label, "value": val, "reason": reason})
                    continue
                else:
                    # None = can't determine options yet (parent not set), accept for now
                    sanitized[fid] = val

        else:
            if val_str in user_msg_lower:
                sanitized[fid] = val
            else:
                val_words = [w for w in val_str.split() if len(w) > 2]
                if val_words and any(w in user_msg_lower for w in val_words):
                    sanitized[fid] = val
                else:
                    dropped.append({"field": label, "value": val, "reason": f"'{val}' doesn't appear to match what you typed"})
                    continue

    return {"extracted": sanitized, "dropped_fields": dropped}


def respond_empty(state: FormState) -> dict:
    """Handle empty extraction: query-only, delete-only, or nudge."""
    form = state["form"]
    collected_data = state["collected_data"]
    query_answer = state["query_answer"]
    deleted_labels = state.get("deleted_labels", [])

    # Query only
    if query_answer and not deleted_labels:
        missing = get_missing_fields(form, collected_data)
        if missing:
            msg = _with_query(query_answer, call_openai_next_question(form, collected_data, missing))
        else:
            msg = _with_query(query_answer, "All information has been collected. Thank you!")
        return {"response_msg": msg, "status": "pending" if missing else "complete"}

    # Delete only
    if deleted_labels:
        missing = get_missing_fields(form, collected_data)
        last_action = {"deleted": deleted_labels}
        if missing:
            msg = _with_query(query_answer, call_openai_next_question(form, collected_data, missing, last_action=last_action))
        else:
            msg = _with_query(query_answer, "All information has been collected. Thank you!")
        return {"response_msg": msg, "status": "pending" if missing else "complete"}

    # Nudge — pass dropped_fields so LLM can explain WHY values were rejected
    currently_asking, currently_asking_field = get_currently_asking(form, collected_data)
    dropped_fields = state.get("dropped_fields", [])
    nudge_msg = call_openai_nudge_message(
        state["user_message"], form, collected_data,
        currently_asking=currently_asking,
        currently_asking_field=currently_asking_field,
        dropped_fields=dropped_fields,
    )
    return {
        "response_msg": _with_query(query_answer, nudge_msg),
        "status": "pending",
    }


def validate_fields(state: FormState) -> dict:
    """Per-field validation, build pending_data."""
    form = state["form"]
    collected_data = state["collected_data"]
    extracted = state["extracted"]
    is_update = state["is_update"]

    pending_data = {}
    invalid_fields = []

    field_order = {f["field_id"]: i for i, f in enumerate(form["fields"])}
    sorted_fields = sorted(extracted.items(), key=lambda item: field_order.get(item[0], 999))

    for field_id, value in sorted_fields:
        if value is None or value == "":
            continue

        # Field priority check
        if field_id in collected_data and not is_update:
            existing = collected_data[field_id]
            if str(existing).lower() == str(value).lower():
                continue
            field = get_field(form, field_id)
            label = field["label"] if field else field_id
            invalid_fields.append({
                "field_id": field_id,
                "value": value,
                "error": f"You already provided {label} as '{existing}'. If you want to change it, say 'change {label.lower()} to {value}'.",
            })
            continue

        # Validate
        running_data = {**collected_data, **pending_data}
        if is_update and field_id in running_data:
            running_data.pop(field_id)

        is_valid, error = validate_field(form, field_id, value, running_data)
        if is_valid:
            field = get_field(form, field_id)
            # Normalize dropdown casing
            if field and has_options(field):
                valid_opts = get_valid_dropdown_values(form, field_id, running_data)
                if valid_opts:
                    matched = [o for o in valid_opts if o.lower() == str(value).lower()]
                    if matched:
                        value = matched[0]
            # Normalize number type
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

    return {"pending_data": pending_data, "invalid_fields": invalid_fields}


def build_candidate(state: FormState) -> dict:
    """Build candidate state + auto-fill single-option fields."""
    form = state["form"]
    collected_data = state["collected_data"]
    pending_data = state["pending_data"]
    is_update = state["is_update"]

    candidate_data = dict(collected_data)

    # Clear hierarchy descendants on update
    if is_update and pending_data:
        for updated_fid in pending_data:
            field_def = get_field(form, updated_fid)
            if field_def and has_options(field_def) and field_def.get("parent_field_id") is None:
                for desc_fid in get_all_descendant_field_ids(form, updated_fid):
                    candidate_data.pop(desc_fid, None)

    candidate_data.update(pending_data)

    # Auto-fill
    auto_filled = {}
    changed = True
    while changed:
        changed = False
        for field in form["fields"]:
            fid = field["field_id"]
            if fid in candidate_data or not has_options(field):
                continue
            valid_opts = get_valid_dropdown_values(form, fid, candidate_data)
            if valid_opts and len(valid_opts) == 1:
                candidate_data[fid] = valid_opts[0]
                auto_filled[fid] = valid_opts[0]
                changed = True

    return {"candidate_data": candidate_data, "auto_filled": auto_filled}


def resolve_validate(state: FormState) -> dict:
    """Fixpoint resolve + validate atomically."""
    auto_filled = dict(state["auto_filled"])
    resolved_data, inferred, all_conflicts, _ = resolve_and_validate(state["form"], state["candidate_data"])
    auto_filled.update(inferred)

    return {
        "resolved_data": resolved_data,
        "inferred": inferred,
        "all_conflicts": all_conflicts,
        "auto_filled": auto_filled,
    }


def handle_conflicts(state: FormState) -> dict:
    """Partial commit clean fields, reject conflicting ones, generate error response."""
    form = state["form"]
    collected_data = dict(state["collected_data"])
    pending_data = state["pending_data"]
    all_conflicts = state["all_conflicts"]
    invalid_fields = state["invalid_fields"]
    resolved_data = state["resolved_data"]
    auto_filled = dict(state["auto_filled"])
    deleted_labels = state.get("deleted_labels", [])
    query_answer = state.get("query_answer")

    # Identify conflicting fields
    conflicting_ids = set()
    for c in all_conflicts:
        if c.get("field") and c["field"] != "hierarchy":
            conflicting_ids.add(c["field"])
        if c.get("triggered_by", {}).get("field"):
            conflicting_ids.add(c["triggered_by"]["field"])
        for fid in c.get("involved_fields", []):
            conflicting_ids.add(fid)
    for inv in invalid_fields:
        conflicting_ids.add(inv["field_id"])

    # Partial commit clean fields
    clean_fields = {fid: val for fid, val in pending_data.items() if fid not in conflicting_ids}
    if clean_fields:
        partial = dict(collected_data)
        partial.update(clean_fields)
        _, partial_inferred, partial_conflicts, _ = resolve_and_validate(form, partial)
        if not partial_conflicts:
            collected_data = partial
            collected_data.update(partial_inferred)
            auto_filled.update(partial_inferred)

    # Build errors
    all_errors = [{"field_id": c["field"], "value": c.get("value"), "error": c["reason"]} for c in all_conflicts]
    all_errors.extend(invalid_fields)

    conflict_suggestions = build_conflict_suggestions(form, all_conflicts, resolved_data) if all_conflicts else []
    missing = get_missing_fields(form, collected_data)

    # Build last_action
    last_action = {}
    stored_clean = {fid: collected_data[fid] for fid in clean_fields if fid in collected_data}
    if stored_clean:
        last_action["stored"] = stored_clean
    inferred_clean = {k: v for k, v in auto_filled.items() if k not in pending_data and k in collected_data}
    if inferred_clean:
        last_action["inferred"] = inferred_clean
    if deleted_labels:
        last_action["deleted"] = deleted_labels
    dropped_fields = state.get("dropped_fields", [])
    if dropped_fields:
        last_action["rejected"] = dropped_fields

    error_msg = call_openai_error_message(form, all_errors, state["user_message"], collected_data)
    if missing:
        next_q = call_openai_next_question(form, collected_data, missing, last_action=last_action if last_action else None)
        response_msg = _with_query(query_answer, error_msg + "\n\n" + next_q)
    else:
        response_msg = _with_query(query_answer, error_msg)

    return {
        "collected_data": collected_data,
        "auto_filled": auto_filled,
        "clean_fields": clean_fields,
        "response_msg": response_msg,
        "status": "conflict" if all_conflicts else "pending",
    }


def commit(state: FormState) -> dict:
    """Commit resolved data + generate next question."""
    form = state["form"]
    resolved_data = state["resolved_data"]
    pending_data = state["pending_data"]
    auto_filled = state["auto_filled"]
    inferred = state["inferred"]
    is_update = state["is_update"]
    currently_asking = state["currently_asking"]
    deleted_labels = state.get("deleted_labels", [])
    extracted = state["extracted"]
    query_answer = state.get("query_answer")

    collected_data = resolved_data
    missing = get_missing_fields(form, collected_data)

    if not missing:
        return {
            "collected_data": collected_data,
            "response_msg": _with_query(query_answer, "All information has been collected. Thank you!"),
            "status": "complete",
        }

    # Build last_action
    last_action = {}
    stored = {fid: collected_data[fid] for fid in pending_data if fid in collected_data}
    if stored:
        last_action["updated" if is_update else "stored"] = stored

    auto_only = {k: v for k, v in auto_filled.items() if k not in pending_data}
    inferred_only = {k: v for k, v in inferred.items() if k not in pending_data}
    if auto_only:
        last_action["auto_filled"] = auto_only
    if inferred_only:
        last_action["inferred"] = inferred_only
    if deleted_labels:
        last_action["deleted"] = deleted_labels
    if currently_asking and currently_asking in missing and extracted:
        last_action["unanswered_field"] = currently_asking

    # Include dropped fields so LLM can acknowledge rejected values
    dropped_fields = state.get("dropped_fields", [])
    if dropped_fields:
        last_action["rejected"] = dropped_fields

    response_msg = _with_query(query_answer, call_openai_next_question(form, collected_data, missing, last_action=last_action))

    return {
        "collected_data": collected_data,
        "response_msg": response_msg,
        "status": "pending",
    }
