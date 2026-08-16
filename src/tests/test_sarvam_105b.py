from __future__ import annotations

import json
import os
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.graph import run_text_claim_workflow
from app.sarvam import (
    ClaimExtraction,
    SarvamConfigurationError,
    _load_project_env,
    build_claim_extraction_messages,
    build_customer_response_messages,
    claim_extraction_response_format,
    create_sarvam_client,
    extract_claim_information,
    generate_customer_response,
    normalize_claim_extraction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = PROJECT_ROOT / "data"
RUNTIME_DATA_DIR = PROJECT_ROOT / "tests" / "runtime_sarvam_data"


class FakeCompletions:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.last_kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )
            ]
        )


class FakeSarvamClient:
    def __init__(self, payload: dict[str, Any] | str) -> None:
        self.completions = FakeCompletions(payload)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture()
def sarvam_data_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
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


def happy_path_extraction_payload() -> dict[str, Any]:
    return {
        "intent": "motor_claim",
        "incident_date": "2026-08-11",
        "incident_time": "evening",
        "incident_location": "Andheri",
        "incident_type": "collision",
        "vehicle_damage": "bumper damage",
        "third_party_involved": True,
        "injury_reported": False,
        "vehicle_drivable": True,
    }


def missing_time_extraction_payload() -> dict[str, Any]:
    return {
        "intent": "motor_claim",
        "incident_date": None,
        "incident_time": None,
        "incident_location": None,
        "incident_type": "accident",
        "vehicle_damage": None,
        "third_party_involved": None,
        "injury_reported": None,
        "vehicle_drivable": None,
    }


def claim_extraction_with(**overrides: Any) -> ClaimExtraction:
    payload: dict[str, Any] = {
        "intent": "motor_claim",
        "incident_date": None,
        "incident_time": None,
        "incident_location": None,
        "incident_type": None,
        "vehicle_damage": None,
        "third_party_involved": None,
        "injury_reported": None,
        "vehicle_drivable": None,
    }
    payload.update(overrides)
    return ClaimExtraction.model_validate(payload)


def test_prompt_contains_safety_guardrails() -> None:
    messages = build_claim_extraction_messages(
        "Accident hua tha but exact time yaad nahi hai.",
        reference_date=date(2026, 8, 12),
    )

    prompt = "\n".join(message["content"] for message in messages)
    assert "Do not invent missing information" in prompt
    assert "Use null for information not provided" in prompt
    assert "Extract only information explicitly stated" in prompt
    assert "Collision or accident alone does not mean another vehicle or person was involved" in prompt
    assert "Do not decide insurance coverage" in prompt
    assert "Do not approve, reject, create, or escalate claims" in prompt
    assert "2026-08-12" in prompt


def test_response_format_uses_strict_json_schema() -> None:
    response_format = claim_extraction_response_format()

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert "incident_time" in schema["required"]
    assert schema["properties"]["injury_reported"]["type"] == ["boolean", "null"]


def test_customer_response_prompt_uses_controlled_workflow_context() -> None:
    messages = build_customer_response_messages(
        customer_text="Not really, but I got a broken leg.",
        state={
            "customer_name": "Rajesh Kumar",
            "workflow_status": "HUMAN_REVIEW",
            "claim_status": "HUMAN_REVIEW",
            "injury_reported": True,
            "missing_fields": [],
            "next_missing_field": "",
            "route_history": ["check_coverage", "evaluate_claim"],
        },
        raw_extraction=claim_extraction_with(injury_reported=False),
        normalized_extraction=claim_extraction_with(injury_reported=True),
        fallback_response="Since an injury was reported, your case needs review.",
        response_language="en",
    )

    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    assert "Use the provided deterministic workflow state as the only source of allowed business action" in system_prompt
    assert "Do not say a claim is registered unless claim_status is INITIATED" in system_prompt
    assert "Not really, but I got a broken leg." in user_prompt
    assert "HUMAN_REVIEW" in user_prompt
    assert "injury_reported" in user_prompt
    assert "route_history" not in user_prompt
    assert "check_coverage" not in user_prompt


def test_customer_response_generation_calls_sarvam_105b() -> None:
    client = FakeSarvamClient("I'm sorry about the injury. A specialist will review this case.")

    response = generate_customer_response(
        customer_text="Not really, but I got a broken leg.",
        state={
            "workflow_status": "HUMAN_REVIEW",
            "claim_status": "HUMAN_REVIEW",
            "injury_reported": True,
            "missing_fields": [],
        },
        raw_extraction=claim_extraction_with(injury_reported=False),
        normalized_extraction=claim_extraction_with(injury_reported=True),
        fallback_response="Since an injury was reported, your case needs review.",
        response_language="en",
        client=client,
    )

    assert response == "I'm sorry about the injury. A specialist will review this case."
    kwargs = client.completions.last_kwargs
    assert kwargs is not None
    assert kwargs["model"] == "sarvam-105b"
    assert kwargs["temperature"] == 0.35
    assert "response_format" not in kwargs.get("request_options", {})


def test_extract_claim_information_calls_sarvam_105b_with_structured_output() -> None:
    client = FakeSarvamClient(happy_path_extraction_payload())

    extraction = extract_claim_information(
        "Meri car ka accident kal shaam ko Andheri mein hua tha.",
        client=client,
        reference_date=date(2026, 8, 12),
    )

    assert extraction.incident_location == "Andheri"
    assert extraction.third_party_involved is None
    assert extraction.injury_reported is None
    assert extraction.vehicle_damage is None

    kwargs = client.completions.last_kwargs
    assert kwargs is not None
    assert kwargs["model"] == "sarvam-105b"
    assert kwargs["temperature"] == 0
    assert kwargs["reasoning_effort"] is None
    assert kwargs["request_options"]["additional_body_parameters"]["response_format"]["type"] == "json_schema"


def test_missing_time_remains_null_from_sarvam_extraction() -> None:
    client = FakeSarvamClient(missing_time_extraction_payload())

    extraction = extract_claim_information(
        "Accident hua tha but exact time yaad nahi hai.",
        client=client,
    )

    assert extraction.intent == "motor_claim"
    assert extraction.incident_time is None
    assert extraction.vehicle_damage is None


def test_tyre_damage_hinglish_text_populates_vehicle_damage() -> None:
    extraction = normalize_claim_extraction(
        "नुकसान नहीं हुआ but मेरे 1 टायर टूट गई",
        ClaimExtraction.model_validate(
            {
                "intent": "motor_claim",
                "incident_date": None,
                "incident_time": None,
                "incident_location": None,
                "incident_type": None,
                "vehicle_damage": None,
                "third_party_involved": None,
                "injury_reported": None,
                "vehicle_drivable": None,
            }
        ),
        reference_date=date(2026, 8, 12),
    )

    assert extraction.vehicle_damage is not None
    assert "टायर" in extraction.vehicle_damage


def test_minor_tyre_damage_sentence_populates_vehicle_damage() -> None:
    extraction = normalize_claim_extraction(
        "गाड़ी में कुछ ज्यादा नुकसान नहीं हुआ, बस एक टायर खराब हो गया है.",
        ClaimExtraction.model_validate(
            {
                "intent": "motor_claim",
                "incident_date": None,
                "incident_time": None,
                "incident_location": None,
                "incident_type": None,
                "vehicle_damage": None,
                "third_party_involved": None,
                "injury_reported": None,
                "vehicle_drivable": None,
            }
        ),
        reference_date=date(2026, 8, 12),
    )

    assert extraction.vehicle_damage is not None
    assert "टायर" in extraction.vehicle_damage


def test_observed_accident_text_extracts_core_facts_without_inventing_location() -> None:
    extraction = normalize_claim_extraction(
        "मेरा accident हो गया कल रात में 12 बजे मैं office से निकल के आ रहा था तो किसी ने सामने से आके गाड़ी ठोक दी",
        ClaimExtraction.model_validate(
            {
                "intent": "motor_claim",
                "incident_date": None,
                "incident_time": None,
                "incident_location": None,
                "incident_type": None,
                "vehicle_damage": None,
                "third_party_involved": None,
                "injury_reported": None,
                "vehicle_drivable": None,
            }
        ),
        reference_date=date(2026, 8, 12),
    )

    assert extraction.incident_date == "2026-08-11"
    assert extraction.incident_time == "00:00"
    assert extraction.incident_location is None
    assert extraction.incident_type == "collision"
    assert extraction.third_party_involved is True
    assert extraction.injury_reported is None
    assert extraction.vehicle_drivable is None


def test_collision_alone_does_not_infer_third_party() -> None:
    extraction = normalize_claim_extraction(
        "My car was involved in a collision.",
        ClaimExtraction.model_validate(
            {
                "intent": "motor_claim",
                "incident_date": None,
                "incident_time": None,
                "incident_location": None,
                "incident_type": "collision",
                "vehicle_damage": None,
                "third_party_involved": True,
                "injury_reported": None,
                "vehicle_drivable": None,
            }
        ),
        reference_date=date(2026, 8, 16),
    )

    assert extraction.incident_type == "collision"
    assert extraction.third_party_involved is None


def test_another_car_hit_my_car_sets_third_party_true() -> None:
    extraction = normalize_claim_extraction(
        (
            "It happened around 8:00 p.m. in the evening near Andheri when I "
            "was coming out of my office and there was another car which just "
            "came in and hit my car."
        ),
        ClaimExtraction.model_validate(
            {
                "intent": "motor_claim",
                "incident_date": None,
                "incident_time": None,
                "incident_location": None,
                "incident_type": None,
                "vehicle_damage": None,
                "third_party_involved": None,
                "injury_reported": None,
                "vehicle_drivable": None,
            }
        ),
        reference_date=date(2026, 8, 16),
    )

    assert extraction.incident_time == "20:00"
    assert extraction.incident_location == "Andheri"
    assert extraction.incident_type == "collision"
    assert extraction.third_party_involved is True


def test_white_car_does_not_trigger_hit_or_third_party() -> None:
    extraction = normalize_claim_extraction(
        "My white car was parked near Andheri.",
        claim_extraction_with(
            incident_location="Andheri",
            incident_type="collision",
            third_party_involved=True,
        ),
        reference_date=date(2026, 8, 16),
    )

    assert extraction.incident_location == "Andheri"
    assert extraction.incident_type is None
    assert extraction.third_party_involved is None


def test_personal_injury_does_not_trigger_third_party_person() -> None:
    extraction = normalize_claim_extraction(
        "No personal injury was reported.",
        claim_extraction_with(
            third_party_involved=True,
            injury_reported=True,
        ),
        reference_date=date(2026, 8, 16),
    )

    assert extraction.third_party_involved is None
    assert extraction.injury_reported is False


def test_nobody_hurt_but_another_car_hit_me_keeps_both_facts() -> None:
    extraction = normalize_claim_extraction(
        "Nobody was hurt, but another car hit me.",
        claim_extraction_with(),
        reference_date=date(2026, 8, 16),
    )

    assert extraction.incident_type == "collision"
    assert extraction.third_party_involved is True
    assert extraction.injury_reported is False


def test_not_really_but_broken_leg_sets_injury_true() -> None:
    extraction = normalize_claim_extraction(
        "Not really, but I got a broken leg.",
        claim_extraction_with(injury_reported=False),
        reference_date=date(2026, 8, 16),
        last_requested_field="injury_reported",
    )

    assert extraction.injury_reported is True


def test_no_serious_injury_but_small_cut_sets_injury_true() -> None:
    extraction = normalize_claim_extraction(
        "No serious injury, only a small cut.",
        claim_extraction_with(injury_reported=False),
        reference_date=date(2026, 8, 16),
        last_requested_field="injury_reported",
    )

    assert extraction.injury_reported is True


def test_no_one_was_injured_sets_injury_false() -> None:
    extraction = normalize_claim_extraction(
        "No one was injured.",
        claim_extraction_with(injury_reported=True),
        reference_date=date(2026, 8, 16),
        last_requested_field="injury_reported",
    )

    assert extraction.injury_reported is False


def test_unknown_injury_answer_remains_null() -> None:
    extraction = normalize_claim_extraction(
        "I don't know if anyone was injured.",
        claim_extraction_with(injury_reported=True),
        reference_date=date(2026, 8, 16),
        last_requested_field="injury_reported",
    )

    assert extraction.injury_reported is None


def test_multi_field_answer_captures_injury_and_driveability() -> None:
    extraction = normalize_claim_extraction(
        "No one else was injured, but I hurt my leg. The car is still driveable.",
        claim_extraction_with(),
        reference_date=date(2026, 8, 16),
        last_requested_field="injury_reported",
    )

    assert extraction.injury_reported is True
    assert extraction.vehicle_drivable is True


def test_undrivable_is_not_treated_as_drivable() -> None:
    extraction = normalize_claim_extraction(
        "The car is undrivable.",
        claim_extraction_with(vehicle_drivable=True),
        reference_date=date(2026, 8, 16),
    )

    assert extraction.vehicle_drivable is False


def test_can_still_be_driven_sets_vehicle_drivable_true() -> None:
    extraction = normalize_claim_extraction(
        "Yes, car can still be driven.",
        claim_extraction_with(vehicle_drivable=None),
        reference_date=date(2026, 8, 16),
        last_requested_field="vehicle_drivable",
    )

    assert extraction.vehicle_drivable is True


def test_vehicle_wont_start_sets_vehicle_drivable_false() -> None:
    extraction = normalize_claim_extraction(
        "No, the vehicle won't start.",
        claim_extraction_with(vehicle_drivable=True),
        reference_date=date(2026, 8, 16),
        last_requested_field="vehicle_drivable",
    )

    assert extraction.vehicle_drivable is False


def test_not_sure_safe_to_drive_remains_null() -> None:
    extraction = normalize_claim_extraction(
        "Not sure if it is safe to drive.",
        claim_extraction_with(vehicle_drivable=True),
        reference_date=date(2026, 8, 16),
        last_requested_field="vehicle_drivable",
    )

    assert extraction.vehicle_drivable is None


def test_entire_car_damage_does_not_become_tyre_damage() -> None:
    extraction = normalize_claim_extraction(
        "The entire car was damaged near Andheri.",
        claim_extraction_with(vehicle_damage="tyre damaged"),
        reference_date=date(2026, 8, 16),
    )

    assert extraction.vehicle_damage is None


def test_may_i_sentence_does_not_support_model_date() -> None:
    extraction = normalize_claim_extraction(
        "May I call you later?",
        claim_extraction_with(incident_date="2026-08-15"),
        reference_date=date(2026, 8, 16),
    )

    assert extraction.incident_date is None


def test_sometimes_does_not_support_model_time() -> None:
    extraction = normalize_claim_extraction(
        "Sometimes I need help with claims.",
        claim_extraction_with(incident_time="20:00"),
        reference_date=date(2026, 8, 16),
    )

    assert extraction.incident_time is None


def test_no_one_injured_is_known_false_not_missing() -> None:
    extraction = normalize_claim_extraction(
        "No one was injured.",
        ClaimExtraction.model_validate(
            {
                "intent": "motor_claim",
                "incident_date": None,
                "incident_time": None,
                "incident_location": None,
                "incident_type": None,
                "vehicle_damage": None,
                "third_party_involved": None,
                "injury_reported": True,
                "vehicle_drivable": None,
            }
        ),
        reference_date=date(2026, 8, 16),
    )

    assert extraction.injury_reported is False


def test_unsupported_model_fields_are_cleared_to_null() -> None:
    extraction = normalize_claim_extraction(
        "My car was in an accident.",
        ClaimExtraction.model_validate(
            {
                "intent": "motor_claim",
                "incident_date": "2026-08-15",
                "incident_time": "20:00",
                "incident_location": "Andheri",
                "incident_type": "collision",
                "vehicle_damage": "bumper damage",
                "third_party_involved": True,
                "injury_reported": False,
                "vehicle_drivable": True,
            }
        ),
        reference_date=date(2026, 8, 16),
    )

    assert extraction.incident_date is None
    assert extraction.incident_time is None
    assert extraction.incident_location is None
    assert extraction.incident_type == "accident"
    assert extraction.vehicle_damage is None
    assert extraction.third_party_involved is None
    assert extraction.injury_reported is None
    assert extraction.vehicle_drivable is None


def test_missing_api_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    with pytest.raises(SarvamConfigurationError):
        create_sarvam_client()


def test_project_env_loader_reads_api_key(
    sarvam_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    env_file = sarvam_data_dir / ".env"
    env_file.write_text("SARVAM_API_KEY=test-secret\n", encoding="utf-8")

    _load_project_env(env_file)

    assert os.environ["SARVAM_API_KEY"] == "test-secret"


def test_text_claim_workflow_uses_extraction_then_creates_claim(
    sarvam_data_dir: Path,
) -> None:
    def fake_extractor(customer_text: str) -> ClaimExtraction:
        assert "Meri car" in customer_text
        return ClaimExtraction.model_validate(happy_path_extraction_payload())

    result = run_text_claim_workflow(
        mobile_number="9876543210",
        customer_text=(
            "Meri car ka accident kal shaam ko Andheri mein hua tha. "
            "Bike wale ko hit hua aur bumper damage ho gaya. Kisi ko injury nahi hui."
        ),
        extractor=fake_extractor,
    )

    assert result["customer_id"] == "CUS10001"
    assert result["coverage_status"] == "COVERED"
    assert result["claim_status"] == "INITIATED"
    assert result["claim_id"].startswith("CLM2026")
    assert "get_document_requirements" in result["route_history"]


def test_text_claim_workflow_asks_for_information_when_extraction_is_incomplete(
    sarvam_data_dir: Path,
) -> None:
    def fake_extractor(customer_text: str) -> ClaimExtraction:
        return ClaimExtraction.model_validate(missing_time_extraction_payload())

    result = run_text_claim_workflow(
        mobile_number="9876543217",
        customer_text="Accident hua tha but exact time yaad nahi hai.",
        extractor=fake_extractor,
    )

    assert result["customer_id"] == "CUS10008"
    assert result["workflow_status"] == "MISSING_INFORMATION"
    assert "incident_time" in result["missing_fields"]
    assert result["claim_id"] == ""
