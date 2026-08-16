from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = PROJECT_ROOT / "data"
RUNTIME_DATA_DIR = PROJECT_ROOT / "tests" / "runtime_data"


@pytest.fixture()
def test_data_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
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


@pytest.fixture()
def client(test_data_dir: Path) -> TestClient:
    return TestClient(app)


def happy_path_claim() -> dict[str, Any]:
    return {
        "customer_id": "CUS10001",
        "policy_id": "POL10001",
        "incident": {
            "date": "2026-08-10",
            "time": "19:00",
            "location": "Andheri West, Mumbai",
            "incident_type": "collision",
            "vehicle_damage": "Front bumper damaged",
            "third_party_involved": True,
            "injury_reported": False,
            "vehicle_drivable": True,
        },
    }


def read_claims(data_dir: Path) -> list[dict[str, Any]]:
    return json.loads((data_dir / "claims.json").read_text(encoding="utf-8"))["claims"]


def test_active_comprehensive_policy_collision_is_covered(client: TestClient) -> None:
    response = client.post(
        "/api/coverage/check",
        json={"policy_id": "POL10001", "incident_type": "collision"},
    )

    assert response.status_code == 200
    assert response.json()["covered"] is True
    assert response.json()["coverage_type"] == "accidental_damage"


def test_third_party_only_policy_own_vehicle_accident_is_not_covered(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/coverage/check",
        json={"policy_id": "POL10005", "incident_type": "own_vehicle_accident"},
    )

    assert response.status_code == 200
    assert response.json()["covered"] is False
    assert response.json()["coverage_type"] == "accidental_damage"


def test_expired_policy_cannot_create_claim(
    client: TestClient,
    test_data_dir: Path,
) -> None:
    claim_request = happy_path_claim()
    claim_request["customer_id"] = "CUS10007"
    claim_request["policy_id"] = "POL10007"

    response = client.post("/api/claims", json=claim_request)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "POLICY_NOT_ACTIVE"
    assert read_claims(test_data_dir) == []


def test_successful_claim_creation_persists_claim(
    client: TestClient,
    test_data_dir: Path,
) -> None:
    response = client.post("/api/claims", json=happy_path_claim())

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["claim"]["claim_id"].startswith("CLM2026")
    assert body["claim"]["status"] == "INITIATED"

    stored_claims = read_claims(test_data_dir)
    assert len(stored_claims) == 1
    assert stored_claims[0]["claim_id"] == body["claim"]["claim_id"]


def test_escalation_sets_claim_to_human_review(
    client: TestClient,
) -> None:
    create_response = client.post("/api/claims", json=happy_path_claim())
    claim_id = create_response.json()["claim"]["claim_id"]

    response = client.post(
        f"/api/claims/{claim_id}/escalate",
        json={"reason": "Third-party injury reported"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "HUMAN_REVIEW"
    assert response.json()["priority"] == "HIGH"


def test_unknown_customer_returns_error(client: TestClient) -> None:
    response = client.get("/api/customer/9000000000")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "CUSTOMER_NOT_FOUND"


def test_missing_information_asks_customer_for_details(client: TestClient) -> None:
    claim_request = happy_path_claim()
    claim_request["incident"]["time"] = None

    response = client.post("/api/claims", json=claim_request)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "MISSING_INFORMATION"
    assert response.json()["detail"]["details"]["next_action"] == "ASK_CUSTOMER_FOR_INFORMATION"
    assert "incident.time" in response.json()["detail"]["details"]["missing_fields"]
