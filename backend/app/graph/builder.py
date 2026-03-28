"""Graph builder — wires nodes and edges into the LangGraph state machine."""

from langgraph.graph import StateGraph, END

from app.graph.state import FormState
from app.graph.nodes import (
    load_state,
    extract,
    parse_intent,
    process_query,
    process_deletes,
    sanitize,
    respond_empty,
    validate_fields,
    build_candidate,
    resolve_validate,
    handle_conflicts,
    commit,
)


# === Routing functions ===

def route_after_sanitize(state: FormState) -> str:
    """Empty extraction → respond_empty, else → validate_fields."""
    extracted = state["extracted"]
    if any(v is not None and v != "" for v in extracted.values()):
        print("\n    [route] after sanitize → validate_fields")
        return "validate_fields"
    print("\n    [route] after sanitize → respond_empty")
    return "respond_empty"


def route_after_resolve(state: FormState) -> str:
    """Conflicts → handle_conflicts, else → commit."""
    if state["all_conflicts"] or state["invalid_fields"]:
        print("\n    [route] after resolve → handle_conflicts")
        return "handle_conflicts"
    print("\n    [route] after resolve → commit")
    return "commit"


# === Build & compile ===

def build_form_graph() -> StateGraph:
    """Build the LangGraph state machine for the chat flow."""
    graph = StateGraph(FormState)

    # Nodes
    graph.add_node("load_state", load_state)
    graph.add_node("extract", extract)
    graph.add_node("parse_intent", parse_intent)
    graph.add_node("process_query", process_query)
    graph.add_node("process_deletes", process_deletes)
    graph.add_node("sanitize", sanitize)
    graph.add_node("respond_empty", respond_empty)
    graph.add_node("validate_fields", validate_fields)
    graph.add_node("build_candidate", build_candidate)
    graph.add_node("resolve_validate", resolve_validate)
    graph.add_node("handle_conflicts", handle_conflicts)
    graph.add_node("commit", commit)

    # Linear edges
    graph.set_entry_point("load_state")
    graph.add_edge("load_state", "extract")
    graph.add_edge("extract", "parse_intent")
    graph.add_edge("parse_intent", "process_query")
    graph.add_edge("process_query", "process_deletes")
    graph.add_edge("process_deletes", "sanitize")

    # Branch 1: after sanitize
    graph.add_conditional_edges("sanitize", route_after_sanitize, {
        "respond_empty": "respond_empty",
        "validate_fields": "validate_fields",
    })
    graph.add_edge("respond_empty", END)

    # Linear: validate → build → resolve
    graph.add_edge("validate_fields", "build_candidate")
    graph.add_edge("build_candidate", "resolve_validate")

    # Branch 2: after resolve
    graph.add_conditional_edges("resolve_validate", route_after_resolve, {
        "handle_conflicts": "handle_conflicts",
        "commit": "commit",
    })
    graph.add_edge("handle_conflicts", END)
    graph.add_edge("commit", END)

    return graph


# Compile once at module level
form_graph = build_form_graph().compile()
