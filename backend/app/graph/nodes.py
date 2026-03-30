"""All graph node functions — each handles one step of the chat flow."""

from app.hierarchy import (
    get_field,
    has_options,
    get_all_descendant_field_ids,
    get_valid_dropdown_values,
)
from app.validation import (
    validate_field,
    resolve_field_state,
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
    print("\n>>> [1/12] load_state")
    asking_fid, asking_field = get_currently_asking(state["form"], state["collected_data"])
    return {
        "currently_asking": asking_fid,
        "currently_asking_field": asking_field,
    }


def extract(state: FormState) -> dict:
    """Run LLM extraction or handle sensitive fields."""
    print("\n>>> [2/12] extract")
    print(f"    user_message: {state['user_message']!r}")
    form = state["form"]
    collected_data = state["collected_data"]
    currently_asking = state["currently_asking"]
    user_message = state["user_message"]
    messages = state.get("messages", [])

    password_field_ids = [f["field_id"] for f in form["fields"] if f.get("type") == "password"]

    if currently_asking in password_field_ids:
        extracted = {currently_asking: user_message.strip()}
    else:
        extracted = call_openai_extract(
            user_message, form, collected_data,
            currently_asking=currently_asking,
            currently_asking_field=state["currently_asking_field"],
            messages_history=messages,
        )
        for fid in password_field_ids:
            extracted.pop(fid, None)

    return {"extracted": extracted}


def parse_intent(state: FormState) -> dict:
    """Separate metadata keys (_intent, _delete, _query, _uncertain) from extracted data."""
    print("\n>>> [3/12] parse_intent")
    print(f"    extracted: {state['extracted']}")
    extracted = dict(state["extracted"])
    collected_data = state["collected_data"]
    user_msg = state["user_message"].lower()

    is_uncertain = extracted.pop("_uncertain", False)
    intent = extracted.pop("_intent", "normal")
    is_update = intent == "update"

    delete_fields = extracted.pop("_delete", [])
    if not isinstance(delete_fields, list):
        delete_fields = [delete_fields] if delete_fields else []

    query = extracted.pop("_query", None)

    # Handle conversational intents
    is_confirm = extracted.pop("_confirm", False)
    is_deny = extracted.pop("_deny", False)
    is_skip = extracted.pop("_skip", False)
    is_wait = extracted.pop("_wait", False)

    # Rule-based fallback: detect conversational intents the LLM may have missed
    msg_stripped = user_msg.strip().rstrip(".!?")
    CONFIRM_WORDS = {"yes", "yeah", "yep", "sure", "ok", "okay", "correct", "right", "proceed", "go ahead", "continue", "yea", "y"}
    DENY_WORDS = {"no", "nope", "nah", "wrong", "not that", "cancel", "n"}
    SKIP_WORDS = {"skip", "later", "next", "move on"}
    WAIT_WORDS = {"wait", "hold on", "pause", "stop"}

    if not extracted and not is_confirm and not is_deny and not is_skip and not is_wait:
        if msg_stripped in CONFIRM_WORDS:
            is_confirm = True
        elif msg_stripped in DENY_WORDS:
            is_deny = True
        elif msg_stripped in SKIP_WORDS:
            is_skip = True
        elif msg_stripped in WAIT_WORDS:
            is_wait = True

    # Auto-detect update intent if LLM missed it
    # If user message has update keywords AND extracted fields overlap with collected_data
    if not is_update and extracted:
        update_keywords = ["change", "chnage", "update", "set", "modify", "switch", "replace", "make"]
        has_update_keyword = any(kw in user_msg for kw in update_keywords)
        # Also detect if extracted fields have DIFFERENT values than collected
        data_fields = {fid for fid in extracted if fid not in ("_uncertain", "_intent", "_delete", "_query")}
        has_changed_values = any(
            fid in collected_data and str(collected_data[fid]).lower() != str(extracted[fid]).lower()
            for fid in data_fields
        )

        if has_update_keyword and has_changed_values:
            is_update = True
            intent = "update"

    return {
        "extracted": extracted,
        "is_uncertain": is_uncertain,
        "is_update": is_update,
        "intent": intent,
        "delete_fields": delete_fields,
        "query": query,
        "is_confirm": is_confirm,
        "is_deny": is_deny,
        "is_skip": is_skip,
        "is_wait": is_wait,
    }


def process_query(state: FormState) -> dict:
    """Answer user's question if _query is present."""
    print("\n>>> [4/12] process_query")
    print(f"    query: {state['query']!r}")
    query = state["query"]
    if not query:
        return {"query_answer": None}
    return {"query_answer": call_openai_answer_query(query, state["form"], state["collected_data"])}


def process_deletes(state: FormState) -> dict:
    """Cascade delete fields + re-validate."""
    print("\n>>> [5/12] process_deletes")
    print(f"    delete_fields: {state['delete_fields']}")
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
    print("\n>>> [6/12] sanitize")
    print(f"    extracted: {state['extracted']}")
    extracted = state["extracted"]
    form = state["form"]
    collected_data = state["collected_data"]
    user_msg_lower = state["user_message"].strip().lower()

    # Build batch-aware context: collected + all extracted values
    # So when checking ward options, it uses country=India (from same message)
    # instead of the old country value
    batch_context = dict(collected_data)
    for efid, eval in extracted.items():
        if eval is not None and eval != "":
            batch_context[efid] = eval

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
            # Use batch_context (includes other extracted values from same message)
            # Remove the field being checked to avoid self-reference
            check_context = {k: v for k, v in batch_context.items() if k != fid}
            valid_opts = get_valid_dropdown_values(form, fid, check_context)
            if valid_opts:
                if any(o.lower() == val_str for o in valid_opts):
                    sanitized[fid] = val
                else:
                    reason = None

                    # Check if a DESCENDANT field is constraining this field
                    descendants = get_all_descendant_field_ids(form, fid)
                    constraining = []
                    for desc_fid in descendants:
                        if desc_fid in check_context:
                            desc_field = get_field(form, desc_fid)
                            desc_label = desc_field["label"] if desc_field else desc_fid
                            constraining.append(f"{desc_label}='{check_context[desc_fid]}'")
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
                            parent_val = check_context.get(parent_fid)
                            if parent_val:
                                reason = f"'{val}' is not available under {parent_label} '{parent_val}'. Available: {', '.join(valid_opts)}"

                    if not reason:
                        reason = f"'{val}' is not a valid {label}. Available: {', '.join(valid_opts)}"

                    dropped.append({"field": label, "value": val, "reason": reason})
                    continue
            else:
                # valid_opts is empty or None
                if valid_opts is not None and len(valid_opts) == 0:
                    parent_fid = field_def.get("parent_field_id")
                    if parent_fid and parent_fid in check_context:
                        parent_field = get_field(form, parent_fid)
                        parent_label = parent_field["label"] if parent_field else parent_fid
                        parent_val = check_context[parent_fid]
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
    """Handle empty extraction: query-only, delete-only, conversational intents, or nudge."""
    print("\n>>> [7/12] respond_empty (no extracted values)")
    form = state["form"]
    collected_data = state["collected_data"]
    query_answer = state["query_answer"]
    deleted_labels = state.get("deleted_labels", [])
    messages = state.get("messages", [])

    # Query only
    if query_answer and not deleted_labels:
        missing = get_missing_fields(form, collected_data)
        if missing:
            msg = _with_query(query_answer, call_openai_next_question(form, collected_data, missing, messages_history=messages))
        else:
            msg = _with_query(query_answer, "All information has been collected. Thank you!")
        return {"response_msg": msg, "status": "pending" if missing else "complete"}

    # Delete only
    if deleted_labels:
        missing = get_missing_fields(form, collected_data)
        last_action = {"deleted": deleted_labels}
        if missing:
            msg = _with_query(query_answer, call_openai_next_question(form, collected_data, missing, last_action=last_action, messages_history=messages))
        else:
            msg = _with_query(query_answer, "All information has been collected. Thank you!")
        return {"response_msg": msg, "status": "pending" if missing else "complete"}

    # Handle conversational intents
    is_confirm = state.get("is_confirm", False)
    is_deny = state.get("is_deny", False)
    is_skip = state.get("is_skip", False)
    is_wait = state.get("is_wait", False)

    currently_asking, currently_asking_field = get_currently_asking(form, collected_data)
    missing = get_missing_fields(form, collected_data)

    if is_wait:
        return {
            "response_msg": "No problem, take your time! Just continue whenever you're ready.",
            "status": "pending",
        }

    if is_skip:
        if currently_asking and currently_asking_field:
            label = currently_asking_field.get("label", currently_asking)
            is_required = currently_asking_field.get("required", False)
            if is_required:
                return {
                    "response_msg": f"I'd love to let you skip, but {label} is required for this form. Could you please provide it?",
                    "status": "pending",
                }
            else:
                # Skip optional field — move to next question
                if missing:
                    remaining = [f for f in missing if f != currently_asking]
                    if remaining:
                        msg = call_openai_next_question(form, collected_data, remaining, messages_history=messages)
                        return {"response_msg": f"No problem, skipping {label}.\n\n{msg}", "status": "pending"}
                return {"response_msg": f"Skipped {label}. All other fields have been collected. Thank you!", "status": "complete"}

    if is_deny:
        if currently_asking and currently_asking_field:
            label = currently_asking_field.get("label", currently_asking)
            msg = call_openai_next_question(form, collected_data, missing, messages_history=messages)
            return {"response_msg": msg, "status": "pending"}

    if is_confirm:
        if missing:
            msg = call_openai_next_question(form, collected_data, missing, messages_history=messages)
            return {"response_msg": msg, "status": "pending"}
        else:
            return {"response_msg": "All information has been collected. Thank you!", "status": "complete"}

    # Nudge — pass dropped_fields so LLM can explain WHY values were rejected
    dropped_fields = state.get("dropped_fields", [])
    nudge_msg = call_openai_nudge_message(
        state["user_message"], form, collected_data,
        currently_asking=currently_asking,
        currently_asking_field=currently_asking_field,
        dropped_fields=dropped_fields,
        messages_history=messages,
    )
    return {
        "response_msg": _with_query(query_answer, nudge_msg),
        "status": "pending",
    }


def validate_fields(state: FormState) -> dict:
    """Per-field validation, build pending_data."""
    print("\n>>> [8/12] validate_fields")
    print(f"    extracted: {state['extracted']}")
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

        # Inactive field check — reject early if field isn't active for current data
        field_def = get_field(form, field_id)
        if field_def:
            running_context = {**collected_data, **{k: v for k, v in extracted.items() if k != field_id}}
            field_state = resolve_field_state(field_def, running_context)
            if not field_state["active"]:
                label = field_def.get("label", field_id)
                invalid_fields.append({
                    "field_id": field_id,
                    "value": value,
                    "error": f"{label} is not applicable for your current selections.",
                })
                continue

        # Validate — build running_data with the full intended state:
        # 1. Start with collected_data + already-validated pending_data
        # 2. Override with ALL other extracted fields from this batch
        #    (so age sees country=Japan if both are in the same message)
        # 3. Remove the field being validated (so it doesn't conflict with itself)
        running_data = {**collected_data, **pending_data}
        for other_fid, other_val in extracted.items():
            if other_fid != field_id:
                running_data[other_fid] = other_val
        running_data.pop(field_id, None)

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
    print("\n>>> [9/12] build_candidate")
    print(f"    pending_data: {state['pending_data']}")
    print(f"    invalid_fields: {state['invalid_fields']}")
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
    print("\n>>> [10/12] resolve_validate")
    print(f"    candidate_data: {state['candidate_data']}")
    auto_filled = dict(state["auto_filled"])
    resolved_data, inferred, all_conflicts, removed_fields = resolve_and_validate(state["form"], state["candidate_data"])
    auto_filled.update(inferred)

    # Build labels for removed (inactive) fields that user tried to provide
    form = state["form"]
    pending_data = state["pending_data"]
    removed_labels = []
    for rfid in removed_fields:
        if rfid in pending_data:
            field = get_field(form, rfid)
            label = field["label"] if field else rfid
            removed_labels.append({"field": label, "value": pending_data[rfid], "reason": f"{label} is not applicable for your current selections"})
    if removed_labels:
        print(f"    removed (inactive) fields user provided: {removed_labels}")

    return {
        "resolved_data": resolved_data,
        "inferred": inferred,
        "all_conflicts": all_conflicts,
        "auto_filled": auto_filled,
        "removed_fields": removed_labels,
    }


def handle_conflicts(state: FormState) -> dict:
    """Partial commit clean fields, reject conflicting ones, generate error response."""
    print("\n>>> [11/12] handle_conflicts")
    print(f"    all_conflicts: {state['all_conflicts']}")
    print(f"    invalid_fields: {state['invalid_fields']}")
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

    # Build errors with causal chain context (include ambiguous_source if present)
    all_errors = []
    for c in all_conflicts:
        err = {"field_id": c["field"], "value": c.get("value"), "error": c["reason"]}
        if c.get("ambiguous_source"):
            err["ambiguous_source"] = c["ambiguous_source"]
        if c.get("triggered_by"):
            err["triggered_by"] = c["triggered_by"]
        all_errors.append(err)
    all_errors.extend(invalid_fields)

    # Detect causal chains: if field A was rejected in Phase 1 (invalid_fields)
    # and field B was rejected in Phase 3 (all_conflicts) because it depended on A,
    # add a note explaining the dependency
    rejected_in_phase1 = {inv["field_id"] for inv in invalid_fields}
    rejected_in_phase3 = {c["field"] for c in all_conflicts if c.get("field")}
    pending_field_ids = set(pending_data.keys())

    # Fields that were in pending but got rejected by resolve_and_validate
    # AND whose rejection is caused by another field that failed in Phase 1
    for c in all_conflicts:
        triggered = c.get("triggered_by", {}).get("field")
        if triggered and triggered in pending_field_ids and triggered in rejected_in_phase1:
            c_field = get_field(form, c["field"])
            t_field = get_field(form, triggered)
            if c_field and t_field:
                all_errors.append({
                    "field_id": "dependency_note",
                    "error": (
                        f"{c_field['label']} was also rejected because it depended on "
                        f"{t_field['label']} being changed, which itself couldn't be applied. "
                        f"Fix {t_field['label']} first, then {c_field['label']} can be updated."
                    ),
                })

    # If both fields were in the same batch and one blocks the other,
    # add a note about the batch relationship
    if len(rejected_in_phase1) > 0 and len(all_conflicts) > 0:
        batch_note = "Note: Some values were rejected because they depend on other changes in the same request that couldn't be applied. Resolve the blocking issue first."
        all_errors.append({"field_id": "batch_note", "error": batch_note})

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
    rejected = []
    dropped_fields = state.get("dropped_fields", [])
    if dropped_fields:
        rejected.extend(dropped_fields)
    removed_fields = state.get("removed_fields", [])
    if removed_fields:
        rejected.extend(removed_fields)
    if rejected:
        last_action["rejected"] = rejected

    # Don't ask next question when there are conflicts — user needs to fix the conflict first
    error_msg = call_openai_error_message(
        form, all_errors, state["user_message"], collected_data,
        missing_fields=None,
        last_action=last_action if last_action else None,
        messages_history=state.get("messages", []),
    )
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
    print("\n>>> [12/12] commit")
    print(f"    resolved_data: {state['resolved_data']}")
    print(f"    inferred: {state['inferred']}")
    print(f"    auto_filled: {state['auto_filled']}")
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

    # Include dropped fields + removed inactive fields so LLM can acknowledge rejected values
    rejected = []
    dropped_fields = state.get("dropped_fields", [])
    if dropped_fields:
        rejected.extend(dropped_fields)
    removed_fields = state.get("removed_fields", [])
    if removed_fields:
        rejected.extend(removed_fields)
    if rejected:
        last_action["rejected"] = rejected

    response_msg = _with_query(query_answer, call_openai_next_question(form, collected_data, missing, last_action=last_action, messages_history=state.get("messages", [])))

    return {
        "collected_data": collected_data,
        "response_msg": response_msg,
        "status": "pending",
    }
