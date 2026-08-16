"""LangGraph orchestration for the ClaimsVoice mock claims workflow."""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, StateGraph

from app.rules import calculate_missing_fields, claim_details_from_state
from app.sarvam import ClaimExtraction, extract_claim_information
from app.state import ClaimState
from app.tools import (
    MockBackendError,
    check_coverage as tool_check_coverage,
    create_claim as tool_create_claim,
    escalate_claim as tool_escalate_claim,
    get_customer as tool_get_customer,
    get_document_requirements as tool_get_document_requirements,
    get_policy as tool_get_policy,
)


GRAPH_NODE_STRUCTURE = [
    "START -> understand_intent",
    "understand_intent -> identify_customer",
    "identify_customer -> retrieve_policy | generate_response",
    "retrieve_policy -> collect_incident_information | generate_response",
    "collect_incident_information -> check_coverage | generate_response",
    "check_coverage -> evaluate_claim",
    "evaluate_claim -> create_or_escalate_claim | generate_response",
    "create_or_escalate_claim -> get_document_requirements | generate_response",
    "get_document_requirements -> generate_response",
    "generate_response -> END",
]


def _history(state: ClaimState, node_name: str) -> list[str]:
    return [*state.get("route_history", []), node_name]


def _messages(state: ClaimState, message: str) -> list[str]:
    return [*state.get("conversation_messages", []), message]


def _error_update(state: ClaimState, node_name: str, error: MockBackendError) -> ClaimState:
    return {
        "workflow_status": error.code,
        "next_action": "STOP_WORKFLOW",
        "response_message": error.message,
        "conversation_messages": _messages(state, error.message),
        "route_history": _history(state, node_name),
    }


def understand_intent(state: ClaimState) -> ClaimState:
    message = "Intent understood as motor insurance claim intake."
    return {
        "intent": "motor_claim",
        "conversation_messages": _messages(state, message),
        "route_history": _history(state, "understand_intent"),
    }


def identify_customer(state: ClaimState) -> ClaimState:
    try:
        result = tool_get_customer(state.get("mobile_number", ""))
    except MockBackendError as error:
        return _error_update(state, "identify_customer", error)

    customer = result["customer"]
    return {
        "customer_id": customer["customer_id"],
        "customer_name": customer["name"],
        "policy_id": customer["policy_id"],
        "conversation_messages": _messages(state, f"Customer identified: {customer['name']}."),
        "workflow_status": "CUSTOMER_IDENTIFIED",
        "identity_confirmation_requested": state.get("identity_confirmed") is not True,
        "route_history": _history(state, "identify_customer"),
    }


def route_after_customer(state: ClaimState) -> str:
    if not state.get("customer_id"):
        return "generate_response"
    if state.get("identity_confirmed") is not True:
        return "generate_response"
    return "retrieve_policy"


def retrieve_policy(state: ClaimState) -> ClaimState:
    try:
        result = tool_get_policy(state.get("customer_id", ""))
    except MockBackendError as error:
        return _error_update(state, "retrieve_policy", error)

    policy = result["policy"]
    update: ClaimState = {
        "policy_id": policy["policy_id"],
        "policy_status": policy["status"],
        "policy_type": policy["policy_type"],
        "vehicle_name": policy.get("vehicle_name", ""),
        "vehicle_registration": policy["registration"],
        "conversation_messages": _messages(
            state,
            f"Policy retrieved: {policy['policy_id']} is {policy['status']}.",
        ),
        "route_history": _history(state, "retrieve_policy"),
    }

    if policy["status"] != "ACTIVE":
        update.update(
            {
                "workflow_status": "POLICY_NOT_ACTIVE",
                "next_action": "POLICY_REVIEW",
                "response_message": "This policy is not active, so automatic claim creation is blocked.",
            }
        )
    else:
        update["workflow_status"] = "POLICY_ACTIVE"

    return update


def route_after_policy(state: ClaimState) -> str:
    if not state.get("policy_id") or state.get("policy_status") != "ACTIVE":
        return "generate_response"
    return "collect_incident_information"


def collect_incident_information(state: ClaimState) -> ClaimState:
    missing_fields = calculate_missing_fields(state)
    update: ClaimState = {
        "missing_fields": missing_fields,
        "route_history": _history(state, "collect_incident_information"),
    }

    if missing_fields:
        update.update(
            {
                "workflow_status": "MISSING_INFORMATION",
                "next_action": "ASK_CUSTOMER_FOR_INFORMATION",
                "conversation_messages": _messages(
                    state,
                    "Required incident information is missing.",
                ),
            }
        )
    else:
        update.update(
            {
                "workflow_status": "INCIDENT_INFORMATION_COMPLETE",
                "conversation_messages": _messages(
                    state,
                    "Incident information is complete.",
                ),
            }
        )

    return update


def route_after_incident_information(state: ClaimState) -> str:
    if state.get("missing_fields") or calculate_missing_fields(state):
        return "generate_response"
    return "check_coverage"


def check_coverage(state: ClaimState) -> ClaimState:
    missing_fields = calculate_missing_fields(state)
    if missing_fields:
        return {
            "missing_fields": missing_fields,
            "workflow_status": "MISSING_INFORMATION",
            "next_action": "ASK_CUSTOMER_FOR_INFORMATION",
            "conversation_messages": _messages(
                state,
                "Required incident information is missing.",
            ),
            "route_history": _history(state, "check_coverage"),
        }

    try:
        result = tool_check_coverage(
            state.get("policy_id", ""),
            state.get("incident_type") or "",
        )
    except MockBackendError as error:
        return _error_update(state, "check_coverage", error)

    coverage_status = "COVERED" if result["covered"] else "NOT_COVERED"
    return {
        "coverage_status": coverage_status,
        "coverage_type": result["coverage_type"],
        "coverage_reason": result["reason"],
        "conversation_messages": _messages(
            state,
            f"Coverage check completed: {coverage_status}.",
        ),
        "workflow_status": "COVERAGE_CHECKED",
        "route_history": _history(state, "check_coverage"),
    }


def evaluate_claim(state: ClaimState) -> ClaimState:
    missing_fields = calculate_missing_fields(state)
    update: ClaimState = {
        "claim_eligible": False,
        "missing_fields": missing_fields,
        "route_history": _history(state, "evaluate_claim"),
    }

    if state.get("policy_status") != "ACTIVE":
        update.update(
            {
                "workflow_status": "POLICY_NOT_ACTIVE",
                "next_action": "POLICY_REVIEW",
            }
        )
    elif missing_fields:
        update.update(
            {
                "workflow_status": "MISSING_INFORMATION",
                "next_action": "ASK_CUSTOMER_FOR_INFORMATION",
            }
        )
    elif state.get("coverage_status") != "COVERED":
        update.update(
            {
                "workflow_status": "COVERAGE_REVIEW_REQUIRED",
                "next_action": "COVERAGE_REVIEW",
            }
        )
    elif state.get("injury_reported") is True:
        update.update(
            {
                "workflow_status": "ESCALATION_REQUIRED",
                "next_action": "HUMAN_REVIEW",
                "escalation_required": True,
                "escalation_reason": "Third-party injury reported",
            }
        )
    else:
        update.update(
            {
                "workflow_status": "CLAIM_ELIGIBLE",
                "next_action": "CREATE_CLAIM",
                "claim_eligible": True,
            }
        )

    return update


def route_after_evaluation(state: ClaimState) -> str:
    if state.get("claim_eligible") or state.get("escalation_required"):
        return "create_or_escalate_claim"
    return "generate_response"


def create_or_escalate_claim(state: ClaimState) -> ClaimState:
    claim_details = claim_details_from_state(state)

    try:
        claim_result = tool_create_claim(claim_details)
    except MockBackendError as error:
        return _error_update(state, "create_or_escalate_claim", error)

    claim = claim_result["claim"]
    update: ClaimState = {
        "claim_id": claim["claim_id"],
        "claim_status": claim["status"],
        "next_action": claim["next_action"],
        "conversation_messages": _messages(
            state,
            f"Claim intake record created: {claim['claim_id']}.",
        ),
        "route_history": _history(state, "create_or_escalate_claim"),
    }

    if state.get("escalation_required"):
        escalation = tool_escalate_claim(
            claim["claim_id"],
            state.get("escalation_reason", "Human review required"),
        )
        update.update(
            {
                "claim_status": escalation["status"],
                "workflow_status": "HUMAN_REVIEW",
                "next_action": "Claims specialist review",
                "claim_eligible": False,
                "conversation_messages": _messages(
                    state,
                    f"Claim {claim['claim_id']} routed to human review.",
                ),
            }
        )
    else:
        update["workflow_status"] = "CLAIM_CREATED"

    return update


def route_after_claim_action(state: ClaimState) -> str:
    if state.get("claim_id") and state.get("claim_status") != "HUMAN_REVIEW":
        return "get_document_requirements"
    return "generate_response"


def get_document_requirements(state: ClaimState) -> ClaimState:
    try:
        result = tool_get_document_requirements(state.get("claim_id", ""))
    except MockBackendError as error:
        return _error_update(state, "get_document_requirements", error)

    return {
        "required_documents": result["documents"],
        "next_action": result["next_action"],
        "conversation_messages": _messages(
            state,
            "Document requirements retrieved.",
        ),
        "route_history": _history(state, "get_document_requirements"),
    }


def generate_response(state: ClaimState) -> ClaimState:
    status = state.get("workflow_status", "")

    if status == "CUSTOMER_NOT_FOUND":
        response = "I could not find a customer or policy for this mobile number."
    elif status == "POLICY_NOT_FOUND":
        response = "I could not find a policy for this customer."
    elif status == "CUSTOMER_IDENTIFIED" and state.get("identity_confirmed") is not True:
        response = "Please confirm your identity before I continue."
    elif status == "POLICY_NOT_ACTIVE":
        response = "Your policy is not active, so I cannot create an automatic claim. A review is required."
    elif status == "MISSING_INFORMATION":
        fields = ", ".join(state.get("missing_fields", []))
        response = f"I need a little more information before creating the claim: {fields}."
    elif status == "COVERAGE_REVIEW_REQUIRED":
        response = "This incident is not covered for automatic claim creation and needs review."
    elif status == "HUMAN_REVIEW":
        response = "Because an injury was reported, I am routing this to a claims specialist for human review."
    elif status == "CLAIM_CREATED":
        response = (
            f"Your claim has been created successfully. Claim ID: {state.get('claim_id')}. "
            f"Next step: {state.get('next_action')}."
        )
    else:
        response = state.get("response_message") or "The claim workflow has stopped for review."

    return {
        "response_message": response,
        "conversation_messages": _messages(state, response),
        "route_history": _history(state, "generate_response"),
    }


def build_claim_graph():
    workflow = StateGraph(ClaimState)

    workflow.add_node("understand_intent", understand_intent)
    workflow.add_node("identify_customer", identify_customer)
    workflow.add_node("retrieve_policy", retrieve_policy)
    workflow.add_node("collect_incident_information", collect_incident_information)
    workflow.add_node("check_coverage", check_coverage)
    workflow.add_node("evaluate_claim", evaluate_claim)
    workflow.add_node("create_or_escalate_claim", create_or_escalate_claim)
    workflow.add_node("get_document_requirements", get_document_requirements)
    workflow.add_node("generate_response", generate_response)

    workflow.set_entry_point("understand_intent")
    workflow.add_edge("understand_intent", "identify_customer")
    workflow.add_conditional_edges(
        "identify_customer",
        route_after_customer,
        {
            "retrieve_policy": "retrieve_policy",
            "generate_response": "generate_response",
        },
    )
    workflow.add_conditional_edges(
        "retrieve_policy",
        route_after_policy,
        {
            "collect_incident_information": "collect_incident_information",
            "generate_response": "generate_response",
        },
    )
    workflow.add_conditional_edges(
        "collect_incident_information",
        route_after_incident_information,
        {
            "check_coverage": "check_coverage",
            "generate_response": "generate_response",
        },
    )
    workflow.add_edge("check_coverage", "evaluate_claim")
    workflow.add_conditional_edges(
        "evaluate_claim",
        route_after_evaluation,
        {
            "create_or_escalate_claim": "create_or_escalate_claim",
            "generate_response": "generate_response",
        },
    )
    workflow.add_conditional_edges(
        "create_or_escalate_claim",
        route_after_claim_action,
        {
            "get_document_requirements": "get_document_requirements",
            "generate_response": "generate_response",
        },
    )
    workflow.add_edge("get_document_requirements", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()


claims_graph = build_claim_graph()


def run_claim_workflow(initial_state: ClaimState) -> ClaimState:
    return claims_graph.invoke(initial_state)


def state_updates_from_extraction(extraction: ClaimExtraction) -> ClaimState:
    return {
        "intent": extraction.intent or "",
        "incident_date": extraction.incident_date,
        "incident_time": extraction.incident_time,
        "incident_location": extraction.incident_location,
        "incident_type": extraction.incident_type,
        "vehicle_damage": extraction.vehicle_damage,
        "third_party_involved": extraction.third_party_involved,
        "injury_reported": extraction.injury_reported,
        "vehicle_drivable": extraction.vehicle_drivable,
    }


def build_state_from_customer_text(
    *,
    mobile_number: str,
    customer_text: str,
    extraction: ClaimExtraction,
) -> ClaimState:
    from app.state import build_initial_state

    initial_state = build_initial_state(
        mobile_number=mobile_number,
        raw_customer_input=customer_text,
    )
    initial_state["identity_confirmed"] = True
    initial_state.update(state_updates_from_extraction(extraction))
    return initial_state


def run_text_claim_workflow(
    *,
    mobile_number: str,
    customer_text: str,
    extractor=extract_claim_information,
) -> ClaimState:
    extraction = extractor(customer_text)
    initial_state = build_state_from_customer_text(
        mobile_number=mobile_number,
        customer_text=customer_text,
        extraction=extraction,
    )
    return run_claim_workflow(initial_state)


def get_graph_structure() -> list[str]:
    return GRAPH_NODE_STRUCTURE.copy()
