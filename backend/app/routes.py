"""API endpoints for the dynamic form engine."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.storage import read_json, write_json
from app.hierarchy import get_field, has_options, get_valid_dropdown_values
from app.validation import (
    get_missing_fields,
    get_currently_asking,
    get_suggestions,
)
from app.llm import call_openai_next_question
from app.graph import form_graph

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
    # Load persisted state
    form = read_json("active_form.json")
    if not form:
        raise HTTPException(status_code=400, detail="No active form. Select a form first.")

    collected_data = read_json("collected_data.json")
    messages = read_json("messages.json")
    messages.append({"role": "user", "content": req.message})

    # Run the LangGraph flow
    initial_state = {
        "user_message": req.message,
        "form": form,
        "collected_data": collected_data,
        "messages": messages,
        # Initialize intermediate fields
        "currently_asking": None,
        "currently_asking_field": None,
        "extracted": {},
        "is_uncertain": False,
        "is_update": False,
        "intent": "normal",
        "delete_fields": [],
        "query": None,
        "query_answer": None,
        "deleted_labels": [],
        "pending_data": {},
        "invalid_fields": [],
        "candidate_data": {},
        "auto_filled": {},
        "resolved_data": {},
        "inferred": {},
        "all_conflicts": [],
        "clean_fields": {},
        "dropped_fields": [],
        "response_msg": "",
        "status": "pending",
        "result": None,
    }

    final_state = form_graph.invoke(initial_state)

    # Persist state
    collected_data = final_state["collected_data"]
    response_msg = final_state["response_msg"]
    status = final_state["status"]
    invalid_fields = final_state.get("invalid_fields", [])

    write_json("collected_data.json", collected_data)
    messages.append({"role": "assistant", "content": response_msg})
    write_json("messages.json", messages)
    _save_currently_asking(form, collected_data)

    # Build response
    missing = get_missing_fields(form, collected_data)
    new_asking, _ = get_currently_asking(form, collected_data)

    result = {
        "status": status,
        "message": response_msg,
        "collected_data": _mask_sensitive(form, collected_data),
        "missing_fields": missing,
        "invalid_fields": invalid_fields,
        "suggestions": get_suggestions(form, collected_data, missing, currently_asking=new_asking),
    }

    all_conflicts = final_state.get("all_conflicts", [])
    if all_conflicts:
        result["conflicts"] = [{"field": c["field"], "reason": c["reason"]} for c in all_conflicts]

    return result
