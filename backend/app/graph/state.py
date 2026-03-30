"""FormState definition — shared across all graph nodes."""

from typing import TypedDict


class FormState(TypedDict):
    # Inputs
    user_message: str
    form: dict
    collected_data: dict
    messages: list

    # Intermediate
    currently_asking: str | None
    currently_asking_field: dict | None
    extracted: dict
    is_uncertain: bool
    is_update: bool
    is_confirm: bool
    is_deny: bool
    is_skip: bool
    is_wait: bool
    intent: str
    delete_fields: list
    query: str | None
    query_answer: str | None
    deleted_labels: list
    pending_data: dict
    invalid_fields: list
    candidate_data: dict
    auto_filled: dict
    resolved_data: dict
    inferred: dict
    all_conflicts: list
    clean_fields: dict
    dropped_fields: list  # values rejected by sanitizer with reasons
    removed_fields: list  # fields removed because they became inactive

    # Output
    response_msg: str
    status: str
    result: dict | None
