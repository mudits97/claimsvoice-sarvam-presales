"""State shape and helpers for the ClaimsVoice LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class ClaimState(TypedDict, total=False):
    customer_id: str
    customer_name: str
    mobile_number: str
    policy_id: str
    policy_status: str
    policy_type: str
    vehicle_name: str
    vehicle_registration: str
    identity_confirmed: bool | None
    identity_confirmation_requested: bool
    identity_mismatch: bool
    identity_just_confirmed: bool
    speaker_claimed_name: str
    incident_date: str | None
    incident_time: str | None
    incident_location: str | None
    incident_type: str | None
    vehicle_damage: str | None
    vehicle_damage_code: str
    vehicle_damage_raw_evidence: str | None
    third_party_involved: bool | None
    injury_reported: bool | None
    vehicle_drivable: bool | None
    additional_details: str | None
    coverage_status: str
    coverage_type: str | None
    coverage_reason: str
    claim_eligible: bool
    claim_id: str
    claim_status: str
    required_documents: list[dict[str, Any]]
    next_action: str
    escalation_required: bool
    escalation_reason: str
    conversation_language: str
    stt_language: str
    response_language: str
    tts_language: str
    ui_language: str
    last_captured_fields: list[str]
    conversation_messages: list[str]
    raw_customer_input: str
    intent: str
    missing_fields: list[str]
    next_missing_field: str
    last_requested_field: str
    last_question: str
    response_message: str
    workflow_status: str
    route_history: list[str]


def build_initial_state(
    *,
    mobile_number: str = "",
    raw_customer_input: str = "",
    incident_date: str | None = None,
    incident_time: str | None = None,
    incident_location: str | None = None,
    incident_type: str | None = None,
    vehicle_damage: str | None = None,
    third_party_involved: bool | None = None,
    injury_reported: bool | None = None,
    vehicle_drivable: bool | None = None,
) -> ClaimState:
    """Create a complete initial state for deterministic workflow tests."""
    return {
        "customer_id": "",
        "customer_name": "",
        "mobile_number": mobile_number,
        "policy_id": "",
        "policy_status": "",
        "policy_type": "",
        "vehicle_name": "",
        "vehicle_registration": "",
        "identity_confirmed": None,
        "identity_confirmation_requested": False,
        "identity_mismatch": False,
        "identity_just_confirmed": False,
        "speaker_claimed_name": "",
        "incident_date": incident_date,
        "incident_time": incident_time,
        "incident_location": incident_location,
        "incident_type": incident_type,
        "vehicle_damage": vehicle_damage,
        "vehicle_damage_code": "",
        "vehicle_damage_raw_evidence": None,
        "third_party_involved": third_party_involved,
        "injury_reported": injury_reported,
        "vehicle_drivable": vehicle_drivable,
        "additional_details": None,
        "coverage_status": "UNKNOWN",
        "coverage_type": None,
        "coverage_reason": "",
        "claim_eligible": False,
        "claim_id": "",
        "claim_status": "",
        "required_documents": [],
        "next_action": "",
        "escalation_required": False,
        "escalation_reason": "",
        "conversation_language": "",
        "stt_language": "",
        "response_language": "",
        "tts_language": "",
        "ui_language": "",
        "last_captured_fields": [],
        "conversation_messages": [],
        "raw_customer_input": raw_customer_input,
        "intent": "",
        "missing_fields": [],
        "next_missing_field": "",
        "last_requested_field": "",
        "last_question": "",
        "response_message": "",
        "workflow_status": "STARTED",
        "route_history": [],
    }
