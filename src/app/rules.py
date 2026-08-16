"""Deterministic insurance eligibility rules for the mock backend."""

from __future__ import annotations

from typing import Any


INCIDENT_COVERAGE_MAP = {
    "accident": "accidental_damage",
    "collision": "accidental_damage",
    "car_accident": "accidental_damage",
    "front_collision": "accidental_damage",
    "front_impact": "accidental_damage",
    "motor_accident": "accidental_damage",
    "road_accident": "accidental_damage",
    "road_collision": "accidental_damage",
    "vehicle_accident": "accidental_damage",
    "vehicle_collision": "accidental_damage",
    "own_damage": "accidental_damage",
    "own_vehicle_accident": "accidental_damage",
    "own_vehicle_damage": "accidental_damage",
    "vehicle_damage": "accidental_damage",
    "third_party": "third_party_damage",
    "third_party_accident": "third_party_damage",
    "third_party_damage": "third_party_damage",
    "theft": "theft",
    "stolen_vehicle": "theft",
    "flood": "natural_calamity",
    "natural_calamity": "natural_calamity",
    "fire": "natural_calamity",
}

REQUIRED_INCIDENT_FIELDS = (
    "date",
    "time",
    "location",
    "incident_type",
    "vehicle_damage",
    "third_party_involved",
    "injury_reported",
    "vehicle_drivable",
)

REQUIRED_STATE_INCIDENT_FIELDS = (
    "incident_date",
    "incident_time",
    "incident_location",
    "incident_type",
    "vehicle_damage",
    "third_party_involved",
    "injury_reported",
    "vehicle_drivable",
)

STATE_TO_CLAIM_INCIDENT_FIELD_MAP = {
    "incident_date": "date",
    "incident_time": "time",
    "incident_location": "location",
    "incident_type": "incident_type",
    "vehicle_damage": "vehicle_damage",
    "third_party_involved": "third_party_involved",
    "injury_reported": "injury_reported",
    "vehicle_drivable": "vehicle_drivable",
}


def normalize_incident_type(incident_type: str) -> str:
    return incident_type.strip().lower().replace("-", "_").replace(" ", "_")


def coverage_key_for_incident_type(incident_type: str) -> str | None:
    return INCIDENT_COVERAGE_MAP.get(normalize_incident_type(incident_type))


def missing_required_claim_fields(claim_details: dict[str, Any]) -> list[str]:
    missing: list[str] = []

    for field_name in ("customer_id", "policy_id"):
        if not claim_details.get(field_name):
            missing.append(field_name)

    incident = claim_details.get("incident") or {}
    for field_name in REQUIRED_INCIDENT_FIELDS:
        if field_name not in incident or incident[field_name] in ("", None):
            missing.append(f"incident.{field_name}")

    return missing


def missing_required_state_fields(state: dict[str, Any]) -> list[str]:
    missing: list[str] = []

    for field_name in REQUIRED_STATE_INCIDENT_FIELDS:
        if field_name not in state or state[field_name] in ("", None):
            missing.append(field_name)

    return missing


def calculate_missing_fields(state: dict[str, Any]) -> list[str]:
    """Return required FNOL fields that are genuinely unknown in ClaimState."""
    return missing_required_state_fields(state)


def claim_details_from_state(state: dict[str, Any]) -> dict[str, Any]:
    incident = {
        claim_field: state.get(state_field)
        for state_field, claim_field in STATE_TO_CLAIM_INCIDENT_FIELD_MAP.items()
    }

    return {
        "customer_id": state.get("customer_id", ""),
        "policy_id": state.get("policy_id", ""),
        "incident": incident,
    }


def document_requirements_for_claim(claim: dict[str, Any]) -> list[dict[str, Any]]:
    incident = claim.get("incident") or {}
    documents = [
        {"name": "Driving Licence", "mandatory": True},
        {"name": "Vehicle Registration Certificate", "mandatory": True},
        {"name": "Vehicle photographs", "mandatory": True},
        {"name": "Repair estimate from garage", "mandatory": True},
    ]

    if incident.get("third_party_involved"):
        documents.append({"name": "Third-party vehicle details", "mandatory": True})
        documents.append({"name": "Police intimation or FIR copy", "mandatory": True})

    if incident.get("injury_reported"):
        documents.append({"name": "Medical report for injured person", "mandatory": True})

    if incident.get("vehicle_drivable") is False:
        documents.append({"name": "Towing or roadside assistance receipt", "mandatory": False})

    return documents


def next_action_for_claim(claim: dict[str, Any]) -> str:
    incident = claim.get("incident") or {}

    if claim.get("status") == "HUMAN_REVIEW":
        return "Claims specialist review"
    if incident.get("vehicle_drivable") is False:
        return "Priority roadside assistance and vehicle inspection"
    return "Vehicle inspection"


def priority_for_escalation(reason: str) -> str:
    reason_lower = reason.lower()
    high_risk_terms = ("injury", "injured", "hospital", "death", "fatal", "third-party injury")

    if any(term in reason_lower for term in high_risk_terms):
        return "HIGH"
    return "MEDIUM"
