from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from app.graph import (
    check_coverage,
    collect_incident_information,
    evaluate_claim,
    identify_customer,
    retrieve_policy,
    run_claim_workflow,
    understand_intent,
)
from app.state import ClaimState, build_initial_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = PROJECT_ROOT / "data"
RUNTIME_DATA_DIR = PROJECT_ROOT / "tests" / "runtime_graph_data"


@pytest.fixture()
def graph_data_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
    if RUNTIME_DATA_DIR.exists():
        shutil.rmtree(RUNTIME_DATA_DIR)
    RUNTIME_DATA_DIR.mkdir(parents=True)

    for filename in ("customers.json", "policies.json"):
        shutil.copyfile(SOURCE_DATA_DIR / filename, RUNTIME_DATA_DIR / filename)
    (RUNTIME_DATA_DIR / "claims.json").write_text('{"claims": []}\n', encoding="utf-8")

    monkeypatch.setenv("CLAIMSVOICE_DATA_DIR", str(RUNTIME_DATA_DIR))
    yield RUNTIME_DATA_DIR

    if RUNTIME_DATA_DIR.exists():
        shutil.rmtree(RUNTIME_DATA_DIR)


def read_claims(data_dir: Path) -> list[dict[str, Any]]:
    return json.loads((data_dir / "claims.json").read_text(encoding="utf-8"))["claims"]


def complete_incident_state(mobile_number: str, **overrides: Any) -> ClaimState:
    values = {
        "mobile_number": mobile_number,
        "raw_customer_input": "Meri car ka accident ho gaya hai.",
        "incident_date": "2026-08-10",
        "incident_time": "19:00",
        "incident_location": "Andheri West, Mumbai",
        "incident_type": "collision",
        "vehicle_damage": "Front bumper damaged",
        "third_party_involved": True,
        "injury_reported": False,
        "vehicle_drivable": True,
    }
    values.update(overrides)
    state = build_initial_state(**values)
    state["identity_confirmed"] = True
    return state


def test_rajesh_happy_path_creates_claim(graph_data_dir: Path) -> None:
    result = run_claim_workflow(complete_incident_state("9876543210"))

    assert result["customer_id"] == "CUS10001"
    assert result["customer_name"] == "Rajesh Kumar"
    assert result["policy_status"] == "ACTIVE"
    assert result["coverage_status"] == "COVERED"
    assert result["claim_eligible"] is True
    assert result["claim_status"] == "INITIATED"
    assert result["claim_id"].startswith("CLM2026")
    assert len(result["required_documents"]) >= 4
    assert result["next_action"] == "Vehicle inspection"
    assert result["route_history"] == [
        "understand_intent",
        "identify_customer",
        "retrieve_policy",
        "collect_incident_information",
        "check_coverage",
        "evaluate_claim",
        "create_or_escalate_claim",
        "get_document_requirements",
        "generate_response",
    ]

    stored_claims = read_claims(graph_data_dir)
    assert len(stored_claims) == 1
    assert stored_claims[0]["claim_id"] == result["claim_id"]


def test_kavya_injury_routes_to_human_review(graph_data_dir: Path) -> None:
    result = run_claim_workflow(
        complete_incident_state(
            "9876543215",
            injury_reported=True,
            vehicle_damage="Side door damaged and third party injury reported",
        )
    )

    assert result["customer_id"] == "CUS10006"
    assert result["coverage_status"] == "COVERED"
    assert result["claim_eligible"] is False
    assert result["escalation_required"] is True
    assert result["escalation_reason"] == "Third-party injury reported"
    assert result["claim_status"] == "HUMAN_REVIEW"
    assert result["workflow_status"] == "HUMAN_REVIEW"
    assert "get_document_requirements" not in result["route_history"]

    stored_claims = read_claims(graph_data_dir)
    assert len(stored_claims) == 1
    assert stored_claims[0]["status"] == "HUMAN_REVIEW"


def test_rohit_expired_policy_stops_before_claim_creation(graph_data_dir: Path) -> None:
    result = run_claim_workflow(complete_incident_state("9876543216"))

    assert result["customer_id"] == "CUS10007"
    assert result["policy_status"] == "EXPIRED"
    assert result["claim_id"] == ""
    assert result["claim_eligible"] is False
    assert result["workflow_status"] == "POLICY_NOT_ACTIVE"
    assert "create_or_escalate_claim" not in result["route_history"]
    assert read_claims(graph_data_dir) == []


def test_ananya_missing_incident_time_asks_for_information(graph_data_dir: Path) -> None:
    result = run_claim_workflow(
        complete_incident_state(
            "9876543217",
            incident_time=None,
            incident_location="T Nagar, Chennai",
        )
    )

    assert result["customer_id"] == "CUS10008"
    assert result["workflow_status"] == "MISSING_INFORMATION"
    assert result["next_action"] == "ASK_CUSTOMER_FOR_INFORMATION"
    assert result["missing_fields"] == ["incident_time"]
    assert result["claim_id"] == ""
    assert "check_coverage" not in result["route_history"]
    assert "incident_time" in result["response_message"]
    assert read_claims(graph_data_dir) == []


def test_coverage_is_blocked_when_required_information_is_missing(
    graph_data_dir: Path,
) -> None:
    state = complete_incident_state(
        "9876543210",
        incident_time=None,
        injury_reported=None,
        vehicle_drivable=None,
    )
    state.update(understand_intent(state))
    state.update(identify_customer(state))
    state.update(retrieve_policy(state))

    result = check_coverage(state)

    assert result["workflow_status"] == "MISSING_INFORMATION"
    assert result["next_action"] == "ASK_CUSTOMER_FOR_INFORMATION"
    assert result["missing_fields"] == [
        "incident_time",
        "injury_reported",
        "vehicle_drivable",
    ]
    assert "coverage_status" not in result


def test_evaluation_recomputes_completeness_before_review_or_claim_creation() -> None:
    state = complete_incident_state(
        "9876543210",
        injury_reported=None,
        vehicle_drivable=None,
    )
    state.update(
        {
            "policy_status": "ACTIVE",
            "coverage_status": "COVERED",
            "missing_fields": [],
        }
    )

    result = evaluate_claim(state)

    assert result["workflow_status"] == "MISSING_INFORMATION"
    assert result["claim_eligible"] is False
    assert result["next_action"] == "ASK_CUSTOMER_FOR_INFORMATION"
    assert result["missing_fields"] == ["injury_reported", "vehicle_drivable"]


def test_amit_third_party_only_policy_blocks_own_damage_claim(
    graph_data_dir: Path,
) -> None:
    result = run_claim_workflow(
        complete_incident_state(
            "9876543214",
            incident_type="own_vehicle_accident",
            third_party_involved=False,
            vehicle_damage="Left fender and headlamp damaged",
        )
    )

    assert result["customer_id"] == "CUS10005"
    assert result["policy_status"] == "ACTIVE"
    assert result["coverage_status"] == "NOT_COVERED"
    assert result["coverage_type"] == "accidental_damage"
    assert result["claim_id"] == ""
    assert result["claim_eligible"] is False
    assert result["workflow_status"] == "COVERAGE_REVIEW_REQUIRED"
    assert "create_or_escalate_claim" not in result["route_history"]
    assert read_claims(graph_data_dir) == []


def test_state_updates_between_individual_nodes(graph_data_dir: Path) -> None:
    state = complete_incident_state("9876543210")

    state.update(understand_intent(state))
    assert state["intent"] == "motor_claim"
    assert state["route_history"] == ["understand_intent"]

    state.update(identify_customer(state))
    assert state["customer_id"] == "CUS10001"
    assert state["policy_id"] == "POL10001"
    assert state["route_history"][-1] == "identify_customer"

    state.update(retrieve_policy(state))
    assert state["policy_status"] == "ACTIVE"
    assert state["vehicle_registration"] == "MH01AB1234"

    state.update(collect_incident_information(state))
    assert state["missing_fields"] == []
    assert state["workflow_status"] == "INCIDENT_INFORMATION_COMPLETE"
