"""JSON-backed mock insurance API functions for ClaimsVoice."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from app.rules import (
    coverage_key_for_incident_type,
    document_requirements_for_claim,
    missing_required_claim_fields,
    next_action_for_claim,
    priority_for_escalation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR_ENV = "CLAIMSVOICE_DATA_DIR"


class MockBackendError(Exception):
    """Structured error used by both direct functions and FastAPI endpoints."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}

    def to_detail(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            **({"details": self.details} if self.details else {}),
        }


def get_data_dir() -> Path:
    return Path(os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR))


def _load_json(filename: str) -> dict[str, Any]:
    with (get_data_dir() / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(filename: str, payload: dict[str, Any]) -> None:
    with (get_data_dir() / filename).open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _customers() -> list[dict[str, Any]]:
    return _load_json("customers.json")["customers"]


def _policies() -> list[dict[str, Any]]:
    return _load_json("policies.json")["policies"]


def _claims() -> list[dict[str, Any]]:
    return _load_json("claims.json")["claims"]


def _save_claims(claims: list[dict[str, Any]]) -> None:
    _write_json("claims.json", {"claims": claims})


def _find_policy_by_id(policy_id: str) -> dict[str, Any] | None:
    return next((policy for policy in _policies() if policy["policy_id"] == policy_id), None)


def _find_claim_by_id(claim_id: str) -> dict[str, Any] | None:
    return next((claim for claim in _claims() if claim["claim_id"] == claim_id), None)


def _vehicle_summary(vehicle: dict[str, Any]) -> str:
    return f"{vehicle['make']} {vehicle['model']}"


def _next_claim_id(claims: list[dict[str, Any]]) -> str:
    year = datetime.now().year
    prefix = f"CLM{year}"
    matching_numbers = [
        int(claim["claim_id"].replace(prefix, ""))
        for claim in claims
        if claim.get("claim_id", "").startswith(prefix)
        and claim["claim_id"].replace(prefix, "").isdigit()
    ]
    next_number = max(matching_numbers, default=0) + 1
    return f"{prefix}{next_number:06d}"


def get_customer(mobile_number: str) -> dict[str, Any]:
    customer = next(
        (record for record in _customers() if record["mobile"] == mobile_number),
        None,
    )

    if customer is None:
        raise MockBackendError(
            status_code=404,
            code="CUSTOMER_NOT_FOUND",
            message="No customer found for this mobile number.",
        )

    return {"found": True, "customer": deepcopy(customer)}


def get_policy(customer_id: str) -> dict[str, Any]:
    policy = next(
        (record for record in _policies() if record["customer_id"] == customer_id),
        None,
    )

    if policy is None:
        raise MockBackendError(
            status_code=404,
            code="POLICY_NOT_FOUND",
            message="No policy found for this customer.",
        )

    policy_copy = deepcopy(policy)
    vehicle = policy_copy["vehicle"]
    return {
        "found": True,
        "policy": {
            **policy_copy,
            "vehicle_name": _vehicle_summary(vehicle),
            "registration": vehicle["registration"],
        },
    }


def check_coverage(policy_id: str, incident_type: str) -> dict[str, Any]:
    policy = _find_policy_by_id(policy_id)
    if policy is None:
        raise MockBackendError(
            status_code=404,
            code="POLICY_NOT_FOUND",
            message="No policy found for this policy ID.",
        )

    coverage_type = coverage_key_for_incident_type(incident_type)
    if coverage_type is None:
        return {
            "policy_id": policy_id,
            "incident_type": incident_type,
            "coverage_type": None,
            "covered": False,
            "reason": "This incident type is not recognized by the mock coverage rules.",
        }

    covered = bool(policy["coverage"].get(coverage_type, False))
    reason = (
        f"{incident_type} is covered under {coverage_type}."
        if covered
        else f"{incident_type} is not covered under this policy."
    )

    return {
        "policy_id": policy_id,
        "incident_type": incident_type,
        "coverage_type": coverage_type,
        "covered": covered,
        "reason": reason,
    }


def create_claim(claim_details: dict[str, Any]) -> dict[str, Any]:
    missing_fields = missing_required_claim_fields(claim_details)
    if missing_fields:
        raise MockBackendError(
            status_code=422,
            code="MISSING_INFORMATION",
            message="More incident information is required before a claim can be created.",
            details={
                "missing_fields": missing_fields,
                "next_action": "ASK_CUSTOMER_FOR_INFORMATION",
            },
        )

    policy = _find_policy_by_id(claim_details["policy_id"])
    if policy is None:
        raise MockBackendError(
            status_code=404,
            code="POLICY_NOT_FOUND",
            message="No policy found for this policy ID.",
        )

    if policy["customer_id"] != claim_details["customer_id"]:
        raise MockBackendError(
            status_code=409,
            code="POLICY_CUSTOMER_MISMATCH",
            message="The policy does not belong to this customer.",
        )

    if policy["status"] != "ACTIVE":
        raise MockBackendError(
            status_code=409,
            code="POLICY_NOT_ACTIVE",
            message="Automatic claim creation is blocked because the policy is not active.",
            details={"policy_status": policy["status"]},
        )

    coverage_result = check_coverage(
        claim_details["policy_id"],
        claim_details["incident"]["incident_type"],
    )
    if not coverage_result["covered"]:
        raise MockBackendError(
            status_code=409,
            code="INCIDENT_NOT_COVERED",
            message="Automatic claim creation is blocked because this incident is not covered.",
            details=coverage_result,
        )

    claims = _claims()
    claim = {
        "claim_id": _next_claim_id(claims),
        "customer_id": claim_details["customer_id"],
        "policy_id": claim_details["policy_id"],
        "status": "INITIATED",
        "incident": deepcopy(claim_details["incident"]),
        "coverage_type": coverage_result["coverage_type"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "next_action": "Vehicle inspection",
    }
    claim["next_action"] = next_action_for_claim(claim)

    claims.append(claim)
    _save_claims(claims)

    return {"success": True, "claim": deepcopy(claim)}


def get_document_requirements(claim_id: str) -> dict[str, Any]:
    claim = _find_claim_by_id(claim_id)
    if claim is None:
        raise MockBackendError(
            status_code=404,
            code="CLAIM_NOT_FOUND",
            message="No claim found for this claim ID.",
        )

    return {
        "claim_id": claim_id,
        "documents": document_requirements_for_claim(claim),
        "next_action": next_action_for_claim(claim),
    }


def escalate_claim(claim_id: str, reason: str) -> dict[str, Any]:
    claims = _claims()
    claim = next((record for record in claims if record["claim_id"] == claim_id), None)
    if claim is None:
        raise MockBackendError(
            status_code=404,
            code="CLAIM_NOT_FOUND",
            message="No claim found for this claim ID.",
        )

    priority = priority_for_escalation(reason)
    claim["status"] = "HUMAN_REVIEW"
    claim["escalation"] = {
        "reason": reason,
        "priority": priority,
        "queue": "Claims Specialist",
    }
    claim["next_action"] = "Claims specialist review"

    _save_claims(claims)

    return {
        "success": True,
        "claim_id": claim_id,
        "status": claim["status"],
        "queue": claim["escalation"]["queue"],
        "priority": priority,
        "reason": reason,
    }
