"""API endpoints for the dynamic form engine."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.storage import read_json, write_json
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
    get_suggestions,
    build_conflict_suggestions,
)
from app.engine import resolve_and_validate
from app.llm import (
    call_openai_extract,
    call_openai_next_question,
    call_openai_error_message,
)

router = APIRouter()


# --- Request models ---

class SelectFormRequest(BaseModel):
    form_id: str


class ChatRequest(BaseModel):
    message: str


# --- Helpers ---

def _mask_sensitive(form, collected_data):
    """Mask sensitive field values in response data."""
    safe = dict(collected_data)
    for field in form["fields"]:
        if field.get("type") == "password" and field["field_id"] in safe:
            safe[field["field_id"]] = "********"
    return safe


def _save_currently_asking(form, collected_data):
    """Persist which field will be asked next."""
    fid, _ = get_currently_asking(form, collected_data)
    write_json("currently_asking.json", {"field_id": fid})


# --- Endpoints ---

@router.get("/forms")
def get_forms():
    forms = read_json("forms.json")
    return [{"form_id": f["form_id"], "title": f["title"]} for f in forms]


@router.post("/select-form")
def select_form(req: SelectFormRequest):
    forms = read_json("forms.json")
    form = next((f for f in forms if f["form_id"] == req.form_id), None)
    if not form:
        raise HTTPException(status_code=404, detail="Form not found")

    write_json("active_form.json", form)
    write_json("collected_data.json", {})
    write_json("messages.json", [])

    missing = get_missing_fields(form, {})
    question = call_openai_next_question(form, {}, missing)

    first_asking, _ = get_currently_asking(form, {})
    _save_currently_asking(form, {})

    messages = [{"role": "assistant", "content": question}]
    write_json("messages.json", messages)

    return {
        "status": "pending",
        "message": question,
        "collected_data": {},
        "missing_fields": missing,
        "invalid_fields": [],
        "suggestions": get_suggestions(form, {}, missing, currently_asking=first_asking),
    }


@router.post("/reset")
def reset():
    write_json("active_form.json", None)
    write_json("collected_data.json", {})
    write_json("messages.json", [])
    write_json("currently_asking.json", {"field_id": None})
    return {"status": "reset", "message": "All data cleared."}


@router.post("/chat")
def chat(req: ChatRequest):
    # Step 1: Load state
    form = read_json("active_form.json")
    if not form:
        raise HTTPException(status_code=400, detail="No active form. Select a form first.")

    collected_data = read_json("collected_data.json")
    messages = read_json("messages.json")

    messages.append({"role": "user", "content": req.message})

    # Step 2: Determine which field we're currently asking
    currently_asking, currently_asking_field = get_currently_asking(form, collected_data)

    # Step 2b: Handle sensitive fields — use exact user input, not LLM output
    password_field_ids = [f["field_id"] for f in form["fields"] if f.get("type") == "password"]

    if currently_asking in password_field_ids:
        extracted = {currently_asking: req.message.strip()}
    else:
        # Normal extraction via LLM
        extracted = call_openai_extract(
            req.message, form, collected_data,
            currently_asking=currently_asking,
            currently_asking_field=currently_asking_field,
        )
        print(extracted)
        # Remove any LLM-extracted passwords
        for fid in password_field_ids:
            extracted.pop(fid, None)

    # Clean metadata keys from extracted
    is_uncertain = extracted.pop("_uncertain", False)
    intent = extracted.pop("_intent", "normal")
    is_update = intent == "update"
    delete_fields = extracted.pop("_delete", [])
    if not isinstance(delete_fields, list):
        delete_fields = [delete_fields] if delete_fields else []

    # Process deletes — remove fields + cascade hierarchy children + re-validate
    deleted_labels = []
    if delete_fields:
        for del_fid in delete_fields:
            if del_fid not in collected_data:
                continue
            field_def = get_field(form, del_fid)
            label = field_def["label"] if field_def else del_fid

            # Remove the field itself
            collected_data.pop(del_fid, None)
            deleted_labels.append(label)

            # Cascade: remove hierarchy children (metadata-driven)
            descendants = get_all_descendant_field_ids(form, del_fid)
            for desc_fid in descendants:
                if desc_fid in collected_data:
                    desc_field = get_field(form, desc_fid)
                    desc_label = desc_field["label"] if desc_field else desc_fid
                    collected_data.pop(desc_fid)
                    deleted_labels.append(f"{desc_label} (dependent)")

        # Re-run fixpoint to clean up inactive fields after deletion
        resolved_after_delete, _, _, removed_by_engine = resolve_and_validate(form, collected_data)
        collected_data = resolved_after_delete
        for rfid in removed_by_engine:
            f = get_field(form, rfid)
            deleted_labels.append(f"{f['label']} (no longer active)" if f else rfid)

        write_json("collected_data.json", collected_data)

    # Post-extraction sanitizer — reject hallucinated values
    user_msg_lower = req.message.strip().lower()
    sanitized = {}
    for fid, val in extracted.items():
        if val is None or val == "":
            continue
        field_def = get_field(form, fid)
        if not field_def:
            continue
        ftype = field_def.get("type", "text")
        val_str = str(val).lower()

        if ftype == "number":
            try:
                float(val)
            except (ValueError, TypeError):
                continue
            if not any(ch.isdigit() for ch in user_msg_lower):
                continue
            sanitized[fid] = val

        elif has_options(field_def):
            valid_opts = get_valid_dropdown_values(form, fid, collected_data)
            if valid_opts:
                matched = [o for o in valid_opts if o.lower() == val_str]
                if matched:
                    sanitized[fid] = val
                else:
                    continue
            else:
                sanitized[fid] = val

        else:
            # text — check if value (or significant words) appear in message
            if val_str in user_msg_lower:
                sanitized[fid] = val
            else:
                val_words = [w for w in val_str.split() if len(w) > 2]
                if val_words and any(w in user_msg_lower for w in val_words):
                    sanitized[fid] = val
                else:
                    continue

    extracted = sanitized

    # If nothing survived extraction + sanitization
    if not extracted:
        # If deletes happened, acknowledge them and ask next question
        if deleted_labels:
            missing = get_missing_fields(form, collected_data)
            last_action = {"deleted": deleted_labels}
            if missing:
                response_msg = call_openai_next_question(form, collected_data, missing, last_action=last_action)
            else:
                response_msg = "All information has been collected. Thank you!"

            messages.append({"role": "assistant", "content": response_msg})
            write_json("messages.json", messages)
            _save_currently_asking(form, collected_data)

            new_asking, _ = get_currently_asking(form, collected_data)
            return {
                "status": "pending" if missing else "complete",
                "message": response_msg,
                "collected_data": _mask_sensitive(form, collected_data),
                "missing_fields": missing,
                "invalid_fields": [],
                "suggestions": get_suggestions(form, collected_data, missing, currently_asking=new_asking),
            }

        # No deletes, no extractions — re-ask
        currently_asking, currently_asking_field = get_currently_asking(form, collected_data)
        if currently_asking and currently_asking_field:
            label = currently_asking_field["label"]
            nudge_msg = f"I didn't quite catch that. Could you please provide your {label}?"
            if has_options(currently_asking_field):
                valid_opts = get_valid_dropdown_values(form, currently_asking, collected_data)
                if valid_opts:
                    nudge_msg += f" You can choose from: {', '.join(valid_opts)}."
        else:
            nudge_msg = "I didn't quite understand that. Could you please rephrase?"

        messages.append({"role": "assistant", "content": nudge_msg})
        write_json("messages.json", messages)
        _save_currently_asking(form, collected_data)

        missing = get_missing_fields(form, collected_data)
        return {
            "status": "pending",
            "message": nudge_msg,
            "collected_data": _mask_sensitive(form, collected_data),
            "missing_fields": missing,
            "invalid_fields": [],
            "suggestions": get_suggestions(form, collected_data, missing, currently_asking=currently_asking),
        }

    # ===== ATOMIC TRANSACTION =====
    # Phase 1: Filter, normalize, check field priority
    pending_data = {}
    invalid_fields = []

    field_order = {f["field_id"]: i for i, f in enumerate(form["fields"])}
    sorted_fields = sorted(
        extracted.items(),
        key=lambda item: field_order.get(item[0], 999),
    )

    for field_id, value in sorted_fields:
        if value is None or value == "":
            continue

        # Field priority: first valid value preserved unless explicit update
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

        # Per-field validation
        running_data = {**collected_data, **pending_data}
        if is_update and field_id in running_data:
            running_data.pop(field_id)

        is_valid, error = validate_field(form, field_id, value, running_data)
        if is_valid:
            field = get_field(form, field_id)
            if field and has_options(field):
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

    # Phase 2: Build candidate state
    candidate_data = dict(collected_data)

    if is_update and pending_data:
        for updated_fid in pending_data:
            field_def = get_field(form, updated_fid)
            if field_def and has_options(field_def) and field_def.get("parent_field_id") is None:
                descendants = get_all_descendant_field_ids(form, updated_fid)
                for desc_fid in descendants:
                    candidate_data.pop(desc_fid, None)

    candidate_data.update(pending_data)

    # Auto-fill fields with exactly one valid option
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

    # Phase 3: Fixpoint resolve + validate
    resolved_data, inferred, all_conflicts, removed_fields = resolve_and_validate(form, candidate_data)
    auto_filled.update(inferred)

    if all_conflicts or invalid_fields:
        # Identify which NEW fields are involved in conflicts
        conflicting_field_ids = set()
        for c in all_conflicts:
            # Fields directly named in the conflict
            if c.get("field") and c["field"] != "hierarchy":
                conflicting_field_ids.add(c["field"])
            # triggered_by field
            if c.get("triggered_by", {}).get("field"):
                conflicting_field_ids.add(c["triggered_by"]["field"])
            # involved_fields from hierarchy conflicts
            for fid in c.get("involved_fields", []):
                conflicting_field_ids.add(fid)
        # Also mark per-field invalid fields
        for inv in invalid_fields:
            conflicting_field_ids.add(inv["field_id"])

        # Split pending_data into clean (not in any conflict) vs conflicting
        clean_fields = {}
        for fid, val in pending_data.items():
            if fid not in conflicting_field_ids:
                clean_fields[fid] = val

        # Partial commit: store clean fields that pass validation on their own
        if clean_fields:
            partial_candidate = dict(collected_data)
            partial_candidate.update(clean_fields)
            # Re-validate just the clean subset (no conflicting fields)
            _, partial_inferred, partial_conflicts, _ = resolve_and_validate(form, partial_candidate)
            if not partial_conflicts:
                # Clean fields are valid — commit them
                collected_data = partial_candidate
                collected_data.update(partial_inferred)
                auto_filled.update(partial_inferred)

        write_json("collected_data.json", collected_data)

        # Build error details for rejected fields only
        all_errors = []
        for c in all_conflicts:
            all_errors.append({"field_id": c["field"], "value": c.get("value"), "error": c["reason"]})
        all_errors.extend(invalid_fields)

        conflict_suggestions = build_conflict_suggestions(form, all_conflicts, resolved_data) if all_conflicts else []

        missing = get_missing_fields(form, collected_data)

        # Build last_action for what WAS stored
        last_action = {}
        stored_clean = {fid: collected_data[fid] for fid in clean_fields if fid in collected_data}
        if stored_clean:
            last_action["stored"] = stored_clean
        inferred_clean = {k: v for k, v in auto_filled.items() if k not in pending_data and k in collected_data}
        if inferred_clean:
            last_action["inferred"] = inferred_clean
        if deleted_labels:
            last_action["deleted"] = deleted_labels

        error_msg = call_openai_error_message(form, all_errors, req.message, collected_data)
        if missing:
            next_q = call_openai_next_question(form, collected_data, missing, last_action=last_action if last_action else None)
            response_msg = error_msg + "\n\n" + next_q
        else:
            response_msg = error_msg

        status = "conflict" if all_conflicts else "pending"

        messages.append({"role": "assistant", "content": response_msg})
        write_json("messages.json", messages)
        _save_currently_asking(form, collected_data)

        new_asking, _ = get_currently_asking(form, collected_data)
        result = {
            "status": status,
            "message": response_msg,
            "collected_data": _mask_sensitive(form, collected_data),
            "missing_fields": missing,
            "invalid_fields": invalid_fields,
            "suggestions": conflict_suggestions + get_suggestions(form, collected_data, missing, invalid_fields, currently_asking=new_asking),
        }
        if all_conflicts:
            result["conflicts"] = [{"field": c["field"], "reason": c["reason"]} for c in all_conflicts]
        return result

    # Phase 4: COMMIT
    collected_data = resolved_data
    write_json("collected_data.json", collected_data)

    missing = get_missing_fields(form, collected_data)

    if not missing:
        response_msg = "All information has been collected. Thank you!"
        status = "complete"
    else:
        # Build last_action context for the LLM
        last_action = {}

        # What user explicitly provided
        stored = {fid: collected_data[fid] for fid in pending_data if fid in collected_data}
        if stored:
            if is_update:
                last_action["updated"] = stored
            else:
                last_action["stored"] = stored

        # What was auto-filled or inferred
        auto_only = {k: v for k, v in auto_filled.items() if k not in pending_data}
        inferred_only = {k: v for k, v in inferred.items() if k not in pending_data}
        if auto_only:
            last_action["auto_filled"] = auto_only
        if inferred_only:
            last_action["inferred"] = inferred_only

        # What was deleted
        if deleted_labels:
            last_action["deleted"] = deleted_labels

        # Did user skip the asked question?
        if currently_asking and currently_asking in missing and extracted:
            last_action["unanswered_field"] = currently_asking

        response_msg = call_openai_next_question(form, collected_data, missing, last_action=last_action)
        status = "pending"

    messages.append({"role": "assistant", "content": response_msg})
    write_json("messages.json", messages)
    _save_currently_asking(form, collected_data)

    # Get the NEW currently_asking after commit for suggestions
    new_asking, _ = get_currently_asking(form, collected_data)

    return {
        "status": status,
        "message": response_msg,
        "collected_data": _mask_sensitive(form, collected_data),
        "missing_fields": missing,
        "invalid_fields": invalid_fields,
        "suggestions": get_suggestions(form, collected_data, missing, currently_asking=new_asking),
    }
