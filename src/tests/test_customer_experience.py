from __future__ import annotations

import json
import re
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import main
from app.customer_experience import (
    SESSIONS,
    detect_conversation_language,
    process_chat_message,
    process_voice_message,
    reset_session,
    specific_missing_field_question,
)
from app.rules import missing_required_state_fields
from app.sarvam import ClaimExtraction, SarvamSpeechError, SpeechSynthesisResult, TranscriptionResult
from app.state import build_initial_state
from app.tools import get_customer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = PROJECT_ROOT / "data"
RUNTIME_DATA_DIR = PROJECT_ROOT / "tests" / "runtime_experience_data"


@pytest.fixture()
def experience_data_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
    SESSIONS.clear()
    monkeypatch.setenv("CLAIMSVOICE_DISABLE_LLM_RESPONSES", "1")
    if RUNTIME_DATA_DIR.exists():
        shutil.rmtree(RUNTIME_DATA_DIR)
    RUNTIME_DATA_DIR.mkdir(parents=True)

    for filename in ("customers.json", "policies.json"):
        shutil.copyfile(SOURCE_DATA_DIR / filename, RUNTIME_DATA_DIR / filename)
    (RUNTIME_DATA_DIR / "claims.json").write_text('{"claims": []}\n', encoding="utf-8")

    monkeypatch.setenv("CLAIMSVOICE_DATA_DIR", str(RUNTIME_DATA_DIR))
    yield RUNTIME_DATA_DIR

    SESSIONS.clear()
    if RUNTIME_DATA_DIR.exists():
        shutil.rmtree(RUNTIME_DATA_DIR)


@pytest.fixture()
def client(experience_data_dir: Path) -> TestClient:
    return TestClient(main.app)


def extraction(**overrides: Any) -> ClaimExtraction:
    payload = {
        "intent": "motor_claim",
        "incident_date": "2026-08-11",
        "incident_time": "19:00",
        "incident_location": "Andheri",
        "incident_type": "collision",
        "vehicle_damage": "bumper damage",
        "third_party_involved": True,
        "injury_reported": False,
        "vehicle_drivable": True,
    }
    payload.update(overrides)
    return ClaimExtraction.model_validate(payload)


def empty_extraction(**overrides: Any) -> ClaimExtraction:
    payload = {
        "intent": None,
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


def fake_audio(text: str, **kwargs: Any) -> SpeechSynthesisResult:
    return SpeechSynthesisResult(
        audio_path=str(PROJECT_ROOT / "static" / "generated" / "audio" / "fake.mp3"),
        audio_url="/media/audio/fake.mp3",
    )


def read_claims(data_dir: Path) -> list[dict[str, Any]]:
    return json.loads((data_dir / "claims.json").read_text(encoding="utf-8"))["claims"]


def ready_incident_state(**overrides: Any) -> dict[str, Any]:
    state = build_initial_state(
        mobile_number="9876543210",
        incident_date="2026-08-12",
        incident_time="19:00",
        incident_location="Andheri",
        incident_type="collision",
        vehicle_damage="bumper damage",
        third_party_involved=False,
        injury_reported=False,
        vehicle_drivable=True,
    )
    state.update(
        {
            "customer_id": "CUS10001",
            "customer_name": "Rajesh Kumar",
            "policy_id": "POL10001",
            "policy_status": "ACTIVE",
            "policy_type": "Comprehensive",
            "vehicle_name": "Hyundai Creta",
            "vehicle_registration": "MH01AB1234",
            "identity_confirmed": True,
            "conversation_language": "en",
            "response_language": "en",
            "workflow_status": "MISSING_INFORMATION",
        }
    )
    state.update(overrides)
    state["missing_fields"] = missing_required_state_fields(state)
    if not state.get("next_missing_field") and state["missing_fields"]:
        state["next_missing_field"] = state["missing_fields"][0]
    if not state.get("last_requested_field"):
        state["last_requested_field"] = state.get("next_missing_field", "")
    return state


def flattened_summary_items(summary: dict[str, Any]) -> dict[str, str]:
    return {
        item["label"]: item["value"]
        for section in summary.get("sections", [])
        for item in section.get("items", [])
    }


def captured_summary_items(payload: dict[str, Any]) -> dict[str, str]:
    summary = payload["captured_summary"]
    return {item["label"]: item["value"] for item in summary.get("items", [])}


def start_then_confirm(
    *,
    message: str,
    mobile_number: str,
    session_id: str,
    first_extraction: ClaimExtraction,
    confirm_message: str = "haan",
    confirm_extraction: ClaimExtraction | None = None,
    first_audio=fake_audio,
    confirm_audio=fake_audio,
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = process_chat_message(
        message=message,
        mobile_number=mobile_number,
        session_id=session_id,
        extractor=lambda _text: first_extraction,
        speech_synthesizer=first_audio,
    )
    second = process_chat_message(
        message=confirm_message,
        mobile_number=mobile_number,
        session_id=session_id,
        extractor=lambda _text: confirm_extraction or empty_extraction(),
        speech_synthesizer=confirm_audio,
    )
    return first, second


BANNED_HINDI_ENGLISH_TERMS = (
    "claim",
    "accident",
    "damage",
    "vehicle",
    "review",
    "specialist",
    "involved",
    "driveable",
    "coverage",
    "policy",
    "case",
    "claims",
    "information",
    "next steps",
)


def assert_hindi_response(text: str) -> None:
    assert re.search(r"[\u0900-\u097F]", text)
    lower_text = text.lower()
    for term in BANNED_HINDI_ENGLISH_TERMS:
        assert term not in lower_text


def test_chat_happy_path_returns_claim_confirmation(experience_data_dir: Path) -> None:
    first, result = start_then_confirm(
        message=(
            "Meri car ka accident kal shaam ko Andheri mein hua tha. "
            "Bumper damage hai, koi dusri gaadi involved nahi thi, "
            "kisi ko injury nahi hui, car drive ho sakti hai."
        ),
        mobile_number="9876543210",
        session_id="happy",
        first_extraction=extraction(),
    )

    assert first["claim_status"] == "CUSTOMER_IDENTIFIED"
    assert "Rajesh Kumar" in first["response_text"]
    assert result["claim_id"].startswith("CLM2026")
    assert result["claim_status"] == "INITIATED"
    assert result["audio_url"] == "/media/audio/fake.mp3"
    assert result["audio_available"] is True
    assert result["claim_summary"]["type"] == "claim_success"
    assert result["claim_summary"]["title"] == "आपके दावे का सारांश"
    assert result["claim_summary"]["reference_label"] == "दावा क्रमांक"
    assert result["claim_summary"]["reference_value"] == result["claim_id"]
    assert result["claim_summary"]["next_steps_heading"] == "अगले चरण"
    summary_items = flattened_summary_items(result["claim_summary"])
    assert summary_items["ग्राहक"] == "Rajesh Kumar"
    assert summary_items["पॉलिसी क्रमांक"] == "POL10001"
    assert summary_items["दुर्घटना की तारीख"] == (date.today() - timedelta(days=1)).isoformat()
    assert summary_items["दुर्घटना का समय"] == "19:00"
    assert summary_items["स्थान"] == "Andheri"
    assert summary_items["घटना का प्रकार"] == "वाहन दुर्घटना"
    assert summary_items["वाहन का नुकसान"] == "बम्पर पर डेंट"
    assert summary_items["किसी को चोट"] == "नहीं"
    assert summary_items["दावे की स्थिति"] == "दर्ज"
    assert summary_items["ड्राइविंग लाइसेंस"] == "आवश्यक"
    assert result["progress"][-1]["status"] == "complete"
    assert "दावा सफलतापूर्वक दर्ज" in result["response_text"]
    assert_hindi_response(result["response_text"])
    stored_claims = read_claims(experience_data_dir)
    assert len(stored_claims) == 1
    stored_claim = stored_claims[0]
    assert stored_claim["claim_id"] == result["claim_id"]
    assert summary_items["दुर्घटना की तारीख"] == stored_claim["incident"]["date"]
    assert summary_items["दुर्घटना का समय"] == stored_claim["incident"]["time"]
    assert summary_items["स्थान"] == stored_claim["incident"]["location"]
    assert summary_items["वाहन का नुकसान"] == stored_claim["incident"]["vehicle_damage"]


def test_chat_human_review_is_customer_safe(experience_data_dir: Path) -> None:
    _first, result = start_then_confirm(
        message=(
            "Accident kal shaam 7 baje Koramangala mein hua. "
            "Another person was involved, side door damage hua, "
            "injury bhi hui thi, car drive ho sakti hai."
        ),
        mobile_number="9876543215",
        session_id="injury",
        first_extraction=extraction(
            incident_location="Koramangala",
            injury_reported=True,
            vehicle_damage="side door damage and injury reported",
        ),
    )

    assert result["claim_status"] == "HUMAN_REVIEW"
    assert result["claim_summary"]["type"] == "human_review"
    assert result["claim_summary"]["title"] == "दर्ज की गई जानकारी"
    assert result["claim_summary"]["reference_label"] == "संदर्भ क्रमांक"
    assert result["claim_summary"]["reference_value"] == result["claim_id"]
    summary_items = flattened_summary_items(result["claim_summary"])
    assert summary_items["ग्राहक"] == "Kavya Nair"
    assert summary_items["स्थान"] == "Koramangala"
    assert summary_items["किसी को चोट"] == "हाँ"
    assert summary_items["दावे की स्थिति"] == "विशेषज्ञ समीक्षा आवश्यक"
    assert "दावा क्रमांक" not in str(result["claim_summary"])
    assert "Claim Registered" not in str(result["claim_summary"])
    progress_labels = [step["label"] for step in result["progress"]]
    assert "Specialist Review Required" in progress_labels
    assert "Claim Registered" not in progress_labels
    assert "विशेषज्ञ" in result["response_text"]
    assert_hindi_response(result["response_text"])
    assert "HIGH" not in result["response_text"]
    assert "queue" not in result["response_text"].lower()
    assert read_claims(experience_data_dir)[0]["status"] == "HUMAN_REVIEW"


def test_chat_expired_policy_does_not_create_claim(experience_data_dir: Path) -> None:
    _first, result = start_then_confirm(
        message="Meri car ka accident hua hai.",
        mobile_number="9876543216",
        session_id="expired",
        first_extraction=extraction(incident_location="Jaipur"),
    )

    assert result["claim_id"] == ""
    assert result["claim_status"] == "POLICY_NOT_ACTIVE"
    assert "सक्रिय नहीं" in result["response_text"]
    assert_hindi_response(result["response_text"])
    assert read_claims(experience_data_dir) == []


def test_chat_coverage_failure_does_not_create_claim(experience_data_dir: Path) -> None:
    _first, result = start_then_confirm(
        message=(
            "Meri hi car ka accident kal shaam 7 baje Lucknow mein hua. "
            "Bumper damage hua hai, koi dusri gaadi involved nahi thi, "
            "kisi ko injury nahi hui, car drive ho sakti hai."
        ),
        mobile_number="9876543214",
        session_id="coverage",
        first_extraction=extraction(
            incident_type="own_vehicle_accident",
            third_party_involved=False,
        ),
    )

    assert result["claim_id"] == ""
    assert result["claim_status"] == "COVERAGE_REVIEW_REQUIRED"
    assert "समीक्षा" in result["response_text"]
    assert result["claim_summary"]["type"] == "review_required"
    assert result["claim_summary"]["title"] == "दर्ज की गई जानकारी"
    assert result["claim_summary"]["reference_value"] == ""
    summary_items = flattened_summary_items(result["claim_summary"])
    assert summary_items["दावे की स्थिति"] == "समीक्षा आवश्यक"
    progress_labels = [step["label"] for step in result["progress"]]
    assert "Review Required" in progress_labels
    assert "Claim Registered" not in progress_labels
    assert_hindi_response(result["response_text"])
    assert read_claims(experience_data_dir) == []


def test_language_detection_examples() -> None:
    assert detect_conversation_language("मेरी कार का एक्सीडेंट हुआ है") == "hi"
    assert detect_conversation_language("Meri car ka accident hua hai") == "hi"
    assert detect_conversation_language("My car was in an accident") == "en"


def test_mobile_number_maps_to_correct_customer(experience_data_dir: Path) -> None:
    result = get_customer("9876543210")

    assert result["customer"]["customer_id"] == "CUS10001"
    assert result["customer"]["name"] == "Rajesh Kumar"


def test_customer_name_comes_only_from_lookup(experience_data_dir: Path) -> None:
    result = process_chat_message(
        message="I am Neha and my car was in an accident.",
        mobile_number="9876543210",
        session_id="lookup-name-only",
        extractor=lambda _text: extraction(),
        speech_synthesizer=fake_audio,
    )

    assert "Rajesh Kumar" in result["response_text"]
    assert "Neha" not in result["response_text"]
    assert SESSIONS["lookup-name-only"]["customer_name"] == "Rajesh Kumar"


def test_greeting_uses_returned_customer_name(experience_data_dir: Path) -> None:
    result = process_chat_message(
        message="Meri car ka accident hua hai.",
        mobile_number="9876543215",
        session_id="greeting-name",
        extractor=lambda _text: extraction(),
        speech_synthesizer=fake_audio,
    )

    assert "Kavya Nair" in result["response_text"]
    assert SESSIONS["greeting-name"]["customer_name"] == "Kavya Nair"


def test_identity_confirmation_required_before_policy_details(
    experience_data_dir: Path,
) -> None:
    result = process_chat_message(
        message="Meri car ka accident hua hai.",
        mobile_number="9876543210",
        session_id="policy-hidden",
        extractor=lambda _text: extraction(),
        speech_synthesizer=fake_audio,
    )

    assert result["claim_status"] == "CUSTOMER_IDENTIFIED"
    assert SESSIONS["policy-hidden"]["identity_confirmed"] is None
    assert SESSIONS["policy-hidden"]["policy_status"] == ""
    assert "बीमा पॉलिसी" not in result["response_text"]
    assert "POL" not in result["response_text"]


def test_confirmed_identity_allows_policy_lookup(experience_data_dir: Path) -> None:
    first, confirmed = start_then_confirm(
        message="Meri car ka accident hua hai.",
        mobile_number="9876543210",
        session_id="policy-after-confirm",
        first_extraction=extraction(
            incident_date=None,
            incident_time=None,
            incident_location=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
    )

    assert first["claim_status"] == "CUSTOMER_IDENTIFIED"
    assert SESSIONS["policy-after-confirm"]["identity_confirmed"] is True
    assert SESSIONS["policy-after-confirm"]["policy_status"] == "ACTIVE"
    assert "बीमा पॉलिसी" in confirmed["response_text"]


def test_live_captured_summary_shows_only_known_fields_before_identity(
    experience_data_dir: Path,
) -> None:
    result = process_chat_message(
        message="मेरा accident हो गया कल रात में 12 बजे मैं office से निकल के आ रहा था तो किसी ने सामने से आके गाड़ी ठोक दी",
        mobile_number="9876543211",
        session_id="live-before-identity",
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    assert result["captured_summary"]["title"] == "अब तक दर्ज जानकारी"
    items = captured_summary_items(result)
    assert items["ग्राहक"] == "Priya Sharma"
    assert items["दुर्घटना का समय"] == "00:00"
    assert items["घटना का प्रकार"] == "वाहन टक्कर"
    assert items["दूसरी गाड़ी/व्यक्ति शामिल"] == "हाँ"
    assert "पॉलिसी क्रमांक" not in items
    assert "किसी को चोट" not in items
    assert "गाड़ी चलने की स्थिति में" not in items
    assert "नहीं" not in items.values()


def test_live_captured_summary_updates_after_each_turn_and_retains_values(
    experience_data_dir: Path,
) -> None:
    session_id = "live-summary-updates"
    process_chat_message(
        message="मेरा accident हो गया कल रात में 12 बजे मैं office से निकल के आ रहा था तो किसी ने सामने से आके गाड़ी ठोक दी",
        mobile_number="9876543211",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )
    confirmed = process_chat_message(
        message="हाँ",
        mobile_number="9876543211",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )
    confirmed_items = captured_summary_items(confirmed)
    assert confirmed_items["ग्राहक"] == "Priya Sharma"
    assert confirmed_items["पॉलिसी क्रमांक"] == "POL10002"
    assert confirmed_items["दुर्घटना का समय"] == "00:00"
    assert "स्थान" not in confirmed_items

    location = process_chat_message(
        message="मेरे office के बाहर",
        mobile_number="9876543211",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )
    location_items = captured_summary_items(location)
    assert location_items["दुर्घटना का समय"] == "00:00"
    assert location_items["स्थान"] == "office के बाहर"

    damage = process_chat_message(
        message="गाड़ी में कुछ ज्यादा नुकसान नहीं हुआ but मेरा 1 टायर टूट गया",
        mobile_number="9876543211",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )
    damage_items = captured_summary_items(damage)
    assert damage_items["स्थान"] == "office के बाहर"
    assert damage_items["वाहन का नुकसान"] == "एक टायर क्षतिग्रस्त"
    assert "किसी को चोट" not in damage_items

    no_injury = process_chat_message(
        message="नहीं, किसी को चोट नहीं लगी",
        mobile_number="9876543211",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )
    no_injury_items = captured_summary_items(no_injury)
    assert no_injury_items["वाहन का नुकसान"] == "एक टायर क्षतिग्रस्त"
    assert no_injury_items["किसी को चोट"] == "नहीं"
    assert "गाड़ी चलने की स्थिति में" not in no_injury_items


def test_live_captured_summary_uses_english_labels(experience_data_dir: Path) -> None:
    result = process_chat_message(
        message="My car was in a collision yesterday near the office.",
        mobile_number="9876543210",
        session_id="live-english",
        language="en-IN",
        extractor=lambda _text: extraction(
            incident_date="2026-08-11",
            incident_time=None,
            incident_location="near the office",
            incident_type="collision",
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=fake_audio,
    )

    assert result["captured_summary"]["title"] == "Information captured so far"
    items = captured_summary_items(result)
    assert items["Customer name"] == "Rajesh Kumar"
    assert items["Incident date"] == (date.today() - timedelta(days=1)).isoformat()
    assert items["Incident location"] == "near the office"
    assert items["Incident type"] == "Collision"
    assert "Injury reported" not in items


def test_policy_number_is_not_unnecessarily_requested(
    experience_data_dir: Path,
) -> None:
    _first, confirmed = start_then_confirm(
        message="Meri car ka accident hua hai.",
        mobile_number="9876543210",
        session_id="no-policy-number-request",
        first_extraction=extraction(
            incident_date=None,
            incident_time=None,
            incident_location=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
    )

    lower_response = confirmed["response_text"].lower()
    assert "policy number" not in lower_response
    assert "पॉलिसी नंबर" not in confirmed["response_text"]
    assert "POL10001" not in confirmed["response_text"]


def test_identity_mismatch_prevents_policy_disclosure(
    experience_data_dir: Path,
) -> None:
    process_chat_message(
        message="Meri car ka accident hua hai.",
        mobile_number="9876543210",
        session_id="identity-mismatch",
        extractor=lambda _text: extraction(),
        speech_synthesizer=fake_audio,
    )

    result = process_chat_message(
        message="nahi, main Rajesh Kumar nahi hoon",
        mobile_number="9876543210",
        session_id="identity-mismatch",
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    assert result["claim_status"] == "IDENTITY_MISMATCH"
    assert SESSIONS["identity-mismatch"]["identity_mismatch"] is True
    assert SESSIONS["identity-mismatch"]["customer_name"] == ""
    assert SESSIONS["identity-mismatch"]["policy_status"] == ""
    assert "Rajesh" not in result["response_text"]
    assert "बीमा पॉलिसी की जानकारी साझा नहीं" in result["response_text"]


def test_identity_mismatch_from_different_name_does_not_repeat_prompt(
    experience_data_dir: Path,
) -> None:
    process_chat_message(
        message="My car was in an accident.",
        mobile_number="9876543210",
        session_id="identity-different-name",
        language="en-IN",
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    result = process_chat_message(
        message="Suresh Kumar only.",
        mobile_number="9876543210",
        session_id="identity-different-name",
        language="en-IN",
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    state = SESSIONS["identity-different-name"]
    assert state["identity_confirmed"] is False
    assert state["identity_mismatch"] is True
    assert state["policy_status"] == ""
    assert "Rajesh" not in result["response_text"]
    assert "May I confirm" not in result["response_text"]


def test_identity_success_that_is_me_progresses_to_missing_field(
    experience_data_dir: Path,
) -> None:
    process_chat_message(
        message="My car was in an accident.",
        mobile_number="9876543210",
        session_id="identity-thats-me",
        language="en-IN",
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    result = process_chat_message(
        message="Yes, that's me.",
        mobile_number="9876543210",
        session_id="identity-thats-me",
        language="en-IN",
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    state = SESSIONS["identity-thats-me"]
    assert state["identity_confirmed"] is True
    assert state["policy_status"] == "ACTIVE"
    assert result["claim_status"] == "MISSING_INFORMATION"
    assert "May I confirm" not in result["response_text"]
    assert state["last_requested_field"] == "incident_date"


def test_claim_success_uses_customer_name_and_generated_claim_id(
    experience_data_dir: Path,
) -> None:
    _first, result = start_then_confirm(
        message=(
            "Meri car ka accident kal shaam ko Andheri mein hua tha. "
            "Bumper damage hai, koi dusri gaadi involved nahi thi, "
            "kisi ko injury nahi hui, car drive ho sakti hai."
        ),
        mobile_number="9876543210",
        session_id="success-personalized",
        first_extraction=extraction(),
    )

    assert result["claim_id"].startswith("CLM2026")
    assert result["claim_id"] in result["response_text"]
    assert "Rajesh जी" in result["response_text"]
    assert "दावा क्रमांक" in result["response_text"]
    assert_hindi_response(result["response_text"])


def test_exact_current_case_keeps_unanswered_facts_unknown(
    experience_data_dir: Path,
) -> None:
    result = process_chat_message(
        message=(
            "I met an accident yesterday night and accident happened around "
            "Andheri around 8 PM."
        ),
        mobile_number="9876543210",
        session_id="exact-current-case",
        language="en-IN",
        extractor=lambda _text: extraction(
            incident_type="collision",
            vehicle_damage="bumper damage",
            third_party_involved=True,
            injury_reported=False,
            vehicle_drivable=True,
        ),
        speech_synthesizer=fake_audio,
    )

    state = SESSIONS["exact-current-case"]
    assert state["incident_date"] == (date.today() - timedelta(days=1)).isoformat()
    assert state["incident_time"] == "20:00"
    assert state["incident_location"] == "Andheri"
    assert state["incident_type"] == "accident"
    assert state["third_party_involved"] is None
    assert state["injury_reported"] is None
    assert state["vehicle_damage"] is None
    assert state["vehicle_drivable"] is None
    items = captured_summary_items(result)
    assert "Third-party involvement" not in items
    assert "Injury reported" not in items
    assert "Vehicle damage" not in items
    assert "Vehicle driveable" not in items


def test_customer_switch_same_session_starts_clean_state(
    experience_data_dir: Path,
) -> None:
    process_chat_message(
        message=(
            "Meri car ka accident kal shaam ko Andheri mein hua tha. "
            "Bumper damage hai, koi dusri gaadi involved nahi thi, "
            "kisi ko injury nahi hui, car drive ho sakti hai."
        ),
        mobile_number="9876543210",
        session_id="customer-switch",
        extractor=lambda _text: extraction(),
        speech_synthesizer=fake_audio,
    )
    process_chat_message(
        message="haan",
        mobile_number="9876543210",
        session_id="customer-switch",
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    result = process_chat_message(
        message="My car was in an accident.",
        mobile_number="9876543215",
        session_id="customer-switch",
        language="en-IN",
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    state = SESSIONS["customer-switch"]
    assert state["customer_name"] == "Kavya Nair"
    assert state["mobile_number"] == "9876543215"
    assert state["claim_id"] == ""
    assert state["incident_date"] is None
    assert state["third_party_involved"] is None
    assert "Rajesh" not in str(result["captured_summary"])


def test_reset_session_returns_fresh_empty_session_id(
    experience_data_dir: Path,
) -> None:
    SESSIONS["old-session"] = build_initial_state(
        mobile_number="9876543210",
        incident_date="2026-08-15",
        incident_time="20:00",
        incident_location="Andheri",
        incident_type="accident",
        vehicle_damage="bumper damage",
        third_party_involved=False,
        injury_reported=False,
        vehicle_drivable=True,
    )

    new_id = reset_session("old-session")

    assert new_id != "old-session"
    assert "old-session" not in SESSIONS
    assert new_id not in SESSIONS


def test_customer_name_is_not_used_every_turn(experience_data_dir: Path) -> None:
    _first, confirmed = start_then_confirm(
        message="Meri car ka accident hua hai.",
        mobile_number="9876543210",
        session_id="name-usage",
        first_extraction=extraction(
            incident_date=None,
            incident_time=None,
            incident_location=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
    )
    follow_up = process_chat_message(
        message="Aaj school ke bahar hua.",
        mobile_number="9876543210",
        session_id="name-usage",
        extractor=lambda _text: extraction(
            incident_date=date.today().isoformat(),
            incident_time=None,
            incident_location="school ke bahar",
            incident_type=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=fake_audio,
    )

    assert "Rajesh" in confirmed["response_text"]
    assert "Rajesh" not in follow_up["response_text"]
    assert "स्थान नोट" in follow_up["response_text"]


def test_hindi_user_receives_hindi_response(experience_data_dir: Path) -> None:
    result = process_chat_message(
        message="मेरी कार का एक्सीडेंट हुआ है।",
        mobile_number="9876543210",
        session_id="hindi-language",
        extractor=lambda _text: extraction(
            incident_date=None,
            incident_time=None,
            incident_location=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=fake_audio,
    )

    assert result["response_language"] == "hi"
    assert "पुष्टि" in result["response_text"]
    assert "Rajesh Kumar" in result["response_text"]
    assert SESSIONS["hindi-language"]["customer_name"] == "Rajesh Kumar"
    assert SESSIONS["hindi-language"]["identity_confirmed"] is None
    assert SESSIONS["hindi-language"]["policy_status"] == ""


def test_english_user_receives_english_response(experience_data_dir: Path) -> None:
    result = process_chat_message(
        message="My car was in an accident",
        mobile_number="9876543210",
        session_id="english-language",
        extractor=lambda _text: extraction(
            incident_date=None,
            incident_time=None,
            incident_location=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=fake_audio,
    )

    assert result["response_language"] == "en"
    assert result["response_text"] == "Hello Rajesh. May I confirm that I'm speaking with Rajesh Kumar?"


def test_bulbul_receives_exact_displayed_text_and_language_code(
    experience_data_dir: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def recording_audio(text: str, **kwargs: Any) -> SpeechSynthesisResult:
        calls.append({"text": text, **kwargs})
        return fake_audio(text, **kwargs)

    hindi = process_chat_message(
        message="Meri car ka accident hua hai",
        mobile_number="9876543210",
        session_id="tts-hindi",
        extractor=lambda _text: extraction(
            incident_date=None,
            incident_time=None,
            incident_location=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=recording_audio,
    )
    english = process_chat_message(
        message="My car was in an accident",
        mobile_number="9876543210",
        session_id="tts-english",
        language="en-IN",
        extractor=lambda _text: extraction(
            incident_date=None,
            incident_time=None,
            incident_location=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=recording_audio,
    )

    assert calls[0]["text"] == hindi["response_text"]
    assert calls[0]["language_code"] == "hi-IN"
    assert hindi["audio_available"] is True
    assert calls[1]["text"] == english["response_text"]
    assert calls[1]["language_code"] == "en-IN"
    assert english["audio_available"] is True


def test_hinglish_user_receives_hindi_response(experience_data_dir: Path) -> None:
    result = process_chat_message(
        message="Meri car ka accident hua hai",
        mobile_number="9876543210",
        session_id="hinglish-language",
        extractor=lambda _text: extraction(
            incident_date=None,
            incident_time=None,
            incident_location=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=fake_audio,
    )

    assert result["response_language"] == "hi"
    assert "पुष्टि" in result["response_text"]
    assert_hindi_response(result["response_text"])


def test_language_persists_across_multiple_turns(experience_data_dir: Path) -> None:
    process_chat_message(
        message="मेरी कार का एक्सीडेंट हुआ है।",
        mobile_number="9876543210",
        session_id="language-persist",
        extractor=lambda _text: extraction(
            incident_date=None,
            incident_time=None,
            incident_location=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=fake_audio,
    )

    result = process_chat_message(
        message="हाँ",
        mobile_number="9876543210",
        session_id="language-persist",
        extractor=lambda _text: extraction(
            incident_date=None,
            incident_time=None,
            incident_location=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=fake_audio,
    )

    assert result["response_language"] == "hi"
    assert "दुर्घटना" in result["response_text"]


def test_previously_captured_incident_fields_are_retained(
    experience_data_dir: Path,
) -> None:
    process_chat_message(
        message="Accident aaj school ke bahar hua.",
        mobile_number="9876543210",
        session_id="state-retention",
        extractor=lambda _text: extraction(
            incident_date="2026-08-12",
            incident_time=None,
            incident_location="school ke bahar",
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=fake_audio,
    )

    process_chat_message(
        message="haan, shaam 6 baje hua tha.",
        mobile_number="9876543210",
        session_id="state-retention",
        extractor=lambda _text: extraction(
            incident_date=None,
            incident_time="18:00",
            incident_location=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=fake_audio,
    )

    state = SESSIONS["state-retention"]
    assert state["incident_date"] == date.today().isoformat()
    assert state["incident_location"] == "school ke bahar"
    assert state["incident_time"] == "18:00"


def test_null_extraction_does_not_erase_prior_state(experience_data_dir: Path) -> None:
    process_chat_message(
        message="Accident Andheri mein hua.",
        mobile_number="9876543210",
        session_id="null-safe",
        extractor=lambda _text: extraction(
            incident_location="Andheri",
            incident_time=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=fake_audio,
    )

    process_chat_message(
        message="haan",
        mobile_number="9876543210",
        session_id="null-safe",
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    process_chat_message(
        message="details chahiye aap bata do, aap mere ko main bolta hoon",
        mobile_number="9876543210",
        session_id="null-safe",
        extractor=lambda _text: extraction(
            incident_date=None,
            incident_time=None,
            incident_location=None,
            incident_type=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=fake_audio,
    )

    assert SESSIONS["null-safe"]["incident_location"] == "Andheri"


def test_missing_field_calculation_returns_exact_missing_fields() -> None:
    state = build_initial_state(
        mobile_number="9876543210",
        incident_date="2026-08-12",
        incident_location="school ke bahar",
    )

    assert missing_required_state_fields(state) == [
        "incident_time",
        "incident_type",
        "vehicle_damage",
        "third_party_involved",
        "injury_reported",
        "vehicle_drivable",
    ]


def test_agent_asks_specific_missing_field_question() -> None:
    state = build_initial_state(
        mobile_number="9876543210",
        incident_date="2026-08-12",
        incident_time="18:00",
        incident_location="school ke bahar",
        incident_type="collision",
        third_party_involved=False,
        injury_reported=False,
        vehicle_drivable=True,
    )
    state["response_language"] = "hi"
    state["missing_fields"] = missing_required_state_fields(state)

    question = specific_missing_field_question(state)
    assert "गाड़ी में क्या नुकसान" in question
    assert_hindi_response(question)


def test_agent_never_asks_for_already_known_field() -> None:
    state = build_initial_state(
        mobile_number="9876543210",
        incident_date="2026-08-12",
        incident_time="18:00",
        incident_location="school ke bahar",
        incident_type="collision",
        vehicle_damage="bumper damage",
        third_party_involved=False,
        vehicle_drivable=True,
    )
    state["response_language"] = "hi"
    state["missing_fields"] = missing_required_state_fields(state)

    question = specific_missing_field_question(state)
    assert "किसी को चोट" in question
    assert "नुकसान" not in question
    assert "कहां" not in question
    assert_hindi_response(question)


def test_minor_damage_gets_contextual_acknowledgment() -> None:
    state = build_initial_state(
        mobile_number="9876543210",
        incident_date="2026-08-12",
        incident_time="18:00",
        incident_location="school ke bahar",
        incident_type="collision",
        vehicle_damage="हल्का bumper dent",
        third_party_involved=False,
        vehicle_drivable=True,
    )
    state["response_language"] = "hi"
    state["last_captured_fields"] = ["vehicle_damage"]
    state["missing_fields"] = missing_required_state_fields(state)

    question = specific_missing_field_question(state)
    assert "नुकसान सीमित है" in question
    assert "किसी को चोट" in question
    assert_hindi_response(question)


def test_no_injury_gets_contextual_acknowledgment() -> None:
    state = build_initial_state(
        mobile_number="9876543210",
        incident_date="2026-08-12",
        incident_time="18:00",
        incident_location="school ke bahar",
        incident_type="collision",
        vehicle_damage="हल्का नुकसान",
        injury_reported=False,
        vehicle_drivable=True,
    )
    state["response_language"] = "hi"
    state["last_captured_fields"] = ["injury_reported"]
    state["missing_fields"] = missing_required_state_fields(state)

    question = specific_missing_field_question(state)
    assert "अच्छा है कि किसी को चोट नहीं लगी" in question
    assert "दूसरी गाड़ी" in question
    assert_hindi_response(question)


def test_injury_gets_contextual_acknowledgment() -> None:
    state = build_initial_state(
        mobile_number="9876543210",
        incident_date="2026-08-12",
        incident_time="18:00",
        incident_location="school ke bahar",
        incident_type="collision",
        vehicle_damage="side damage",
        third_party_involved=True,
        injury_reported=True,
    )
    state["response_language"] = "hi"
    state["last_captured_fields"] = ["injury_reported"]
    state["missing_fields"] = missing_required_state_fields(state)

    question = specific_missing_field_question(state)
    assert "यह सुनकर चिंता हुई" in question
    assert "चोट लगी है" in question
    assert "विशेषज्ञ" in question
    assert "गाड़ी अभी चलने" in question
    assert_hindi_response(question)


def test_multi_turn_claim_eventually_progresses_to_claim_creation(
    experience_data_dir: Path,
) -> None:
    today = date.today().isoformat()
    extractions = iter(
        [
            extraction(
                incident_date=None,
                incident_time=None,
                incident_location=None,
                incident_type="collision",
                vehicle_damage=None,
                third_party_involved=None,
                injury_reported=None,
                vehicle_drivable=None,
            ),
            extraction(
                incident_date=today,
                incident_time=None,
                incident_location="school ke bahar",
                incident_type=None,
                vehicle_damage=None,
                third_party_involved=None,
                injury_reported=None,
                vehicle_drivable=None,
            ),
            extraction(
                incident_date=None,
                incident_time="18:00",
                incident_location=None,
                incident_type=None,
                vehicle_damage=None,
                third_party_involved=None,
                injury_reported=None,
                vehicle_drivable=None,
            ),
            extraction(
                incident_date=None,
                incident_time=None,
                incident_location=None,
                incident_type=None,
                vehicle_damage="हल्का bumper dent",
                third_party_involved=None,
                injury_reported=None,
                vehicle_drivable=None,
            ),
            extraction(
                incident_date=None,
                incident_time=None,
                incident_location=None,
                incident_type=None,
                vehicle_damage=None,
                third_party_involved=None,
                injury_reported=False,
                vehicle_drivable=None,
            ),
            extraction(
                incident_date=None,
                incident_time=None,
                incident_location=None,
                incident_type=None,
                vehicle_damage=None,
                third_party_involved=True,
                injury_reported=None,
                vehicle_drivable=None,
            ),
            extraction(
                incident_date=None,
                incident_time=None,
                incident_location=None,
                incident_type=None,
                vehicle_damage=None,
                third_party_involved=None,
                injury_reported=None,
                vehicle_drivable=True,
            ),
        ]
    )

    def sequential_extractor(_text: str) -> ClaimExtraction:
        return next(extractions)

    first = process_chat_message(
        message="Meri car ka accident hua hai.",
        mobile_number="9876543210",
        session_id="multi-turn-hindi",
        extractor=sequential_extractor,
        speech_synthesizer=fake_audio,
    )
    assert first["claim_status"] == "CUSTOMER_IDENTIFIED"
    assert first["response_language"] == "hi"
    assert "पुष्टि" in first["response_text"]
    assert_hindi_response(first["response_text"])

    confirmed = process_chat_message(
        message="हाँ",
        mobile_number="9876543210",
        session_id="multi-turn-hindi",
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )
    assert confirmed["claim_status"] == "MISSING_INFORMATION"
    assert "Rajesh जी" in confirmed["response_text"]
    assert "बीमा पॉलिसी" in confirmed["response_text"]
    assert "तारीख" in confirmed["response_text"]
    assert_hindi_response(confirmed["response_text"])

    second = process_chat_message(
        message="आज school के बाहर हुआ।",
        mobile_number="9876543210",
        session_id="multi-turn-hindi",
        extractor=sequential_extractor,
        speech_synthesizer=fake_audio,
    )
    assert SESSIONS["multi-turn-hindi"]["incident_date"] == today
    assert SESSIONS["multi-turn-hindi"]["incident_location"] == "school ke bahar"
    assert "स्थान नोट" in second["response_text"]
    assert "बजे" in second["response_text"]
    assert_hindi_response(second["response_text"])

    third = process_chat_message(
        message="शाम करीब छह बजे।",
        mobile_number="9876543210",
        session_id="multi-turn-hindi",
        extractor=sequential_extractor,
        speech_synthesizer=fake_audio,
    )
    assert SESSIONS["multi-turn-hindi"]["incident_time"] == "18:00"
    assert "गाड़ी में क्या नुकसान" in third["response_text"]
    assert_hindi_response(third["response_text"])

    fourth = process_chat_message(
        message="हल्का सा bumper पर dent आया है।",
        mobile_number="9876543210",
        session_id="multi-turn-hindi",
        extractor=sequential_extractor,
        speech_synthesizer=fake_audio,
    )
    assert SESSIONS["multi-turn-hindi"]["vehicle_damage"] == "बम्पर पर डेंट"
    assert "नुकसान सीमित है" in fourth["response_text"]
    assert "किसी को चोट" in fourth["response_text"]
    assert_hindi_response(fourth["response_text"])

    fifth = process_chat_message(
        message="नहीं, किसी को चोट नहीं लगी।",
        mobile_number="9876543210",
        session_id="multi-turn-hindi",
        extractor=sequential_extractor,
        speech_synthesizer=fake_audio,
    )
    assert SESSIONS["multi-turn-hindi"]["injury_reported"] is False
    assert "अच्छा है कि किसी को चोट नहीं लगी" in fifth["response_text"]
    assert "दूसरी गाड़ी" in fifth["response_text"]
    assert_hindi_response(fifth["response_text"])

    sixth = process_chat_message(
        message="हाँ, एक bike थी।",
        mobile_number="9876543210",
        session_id="multi-turn-hindi",
        extractor=sequential_extractor,
        speech_synthesizer=fake_audio,
    )
    assert SESSIONS["multi-turn-hindi"]["third_party_involved"] is True
    assert "नोट कर लिया" in sixth["response_text"]
    assert "गाड़ी अभी चलने" in sixth["response_text"]
    assert_hindi_response(sixth["response_text"])

    seventh = process_chat_message(
        message="हाँ, गाड़ी चल रही है।",
        mobile_number="9876543210",
        session_id="multi-turn-hindi",
        extractor=sequential_extractor,
        speech_synthesizer=fake_audio,
    )

    assert seventh["claim_id"].startswith("CLM2026")
    assert seventh["claim_status"] == "INITIATED"
    assert "दावा सफलतापूर्वक दर्ज" in seventh["response_text"]
    assert_hindi_response(seventh["response_text"])
    assert len(read_claims(experience_data_dir)) == 1


def test_observed_priya_conversation_keeps_collecting_until_complete(
    experience_data_dir: Path,
) -> None:
    session_id = "observed-priya-flow"

    first = process_chat_message(
        message="मेरा accident हो गया कल रात में 12 बजे मैं office से निकल के आ रहा था तो किसी ने सामने से आके गाड़ी ठोक दी",
        mobile_number="9876543211",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    state = SESSIONS[session_id]
    assert first["claim_status"] == "CUSTOMER_IDENTIFIED"
    assert "Priya Sharma" in first["response_text"]
    assert state["incident_date"] is not None
    assert state["incident_time"] == "00:00"
    assert state["incident_location"] is None
    assert state["incident_type"] == "collision"
    assert state["third_party_involved"] is True
    assert state["injury_reported"] is None
    assert state["vehicle_drivable"] is None
    assert_hindi_response(first["response_text"])

    confirmed = process_chat_message(
        message="हाँ",
        mobile_number="9876543211",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    assert confirmed["claim_status"] == "MISSING_INFORMATION"
    assert "Priya जी" in confirmed["response_text"]
    assert "कहां हुई" in confirmed["response_text"]
    assert "check_coverage" not in SESSIONS[session_id]["route_history"]
    progress_by_key = {step["key"]: step["status"] for step in confirmed["progress"]}
    assert progress_by_key["information"] == "current"
    assert progress_by_key["claim"] == "pending"
    assert_hindi_response(confirmed["response_text"])

    location = process_chat_message(
        message="मेरे office के बाहर",
        mobile_number="9876543211",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    assert SESSIONS[session_id]["incident_location"] == "office के बाहर"
    assert "गाड़ी में क्या नुकसान" in location["response_text"]
    assert "check_coverage" not in SESSIONS[session_id]["route_history"]
    assert_hindi_response(location["response_text"])

    damage = process_chat_message(
        message="गाड़ी में कुछ ज्यादा नुकसान नहीं हुआ but मेरा 1 टायर टूट गया",
        mobile_number="9876543211",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    assert SESSIONS[session_id]["vehicle_damage"] is not None
    assert "टायर" in SESSIONS[session_id]["vehicle_damage"]
    assert "टायर के नुकसान" in damage["response_text"]
    assert "किसी को चोट" in damage["response_text"]
    assert "गाड़ी में क्या नुकसान" not in damage["response_text"]
    assert "check_coverage" not in SESSIONS[session_id]["route_history"]
    assert_hindi_response(damage["response_text"])

    no_injury = process_chat_message(
        message="नहीं, किसी को चोट नहीं लगी",
        mobile_number="9876543211",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    assert SESSIONS[session_id]["injury_reported"] is False
    assert SESSIONS[session_id]["vehicle_drivable"] is None
    assert "अच्छा है कि किसी को चोट नहीं लगी" in no_injury["response_text"]
    assert "गाड़ी अभी चलने" in no_injury["response_text"]
    assert "check_coverage" not in SESSIONS[session_id]["route_history"]
    assert_hindi_response(no_injury["response_text"])

    complete = process_chat_message(
        message="हाँ, गाड़ी चल रही है",
        mobile_number="9876543211",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    assert complete["claim_status"] == "INITIATED"
    assert complete["claim_id"].startswith("CLM2026")
    assert SESSIONS[session_id]["coverage_status"] == "COVERED"
    assert SESSIONS[session_id]["missing_fields"] == []
    assert "check_coverage" in SESSIONS[session_id]["route_history"]
    assert "दावा सफलतापूर्वक दर्ज" in complete["response_text"]
    assert complete["claim_summary"]["type"] == "claim_success"
    assert_hindi_response(complete["response_text"])


def test_injury_answer_with_broken_leg_resolves_pending_field(
    experience_data_dir: Path,
) -> None:
    session_id = "injury-loop-fix"
    SESSIONS[session_id] = ready_incident_state(
        injury_reported=None,
        vehicle_drivable=True,
        next_missing_field="injury_reported",
        last_requested_field="injury_reported",
    )

    result = process_chat_message(
        message="Not really, but I got a broken leg.",
        mobile_number="9876543210",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(injury_reported=False),
        speech_synthesizer=fake_audio,
    )

    state = SESSIONS[session_id]
    assert state["injury_reported"] is True
    assert state["missing_fields"] == []
    assert state["last_requested_field"] == ""
    assert result["claim_status"] == "HUMAN_REVIEW"
    assert "Was anyone injured in the accident?" not in result["response_text"]
    assert "specialist" in result["response_text"].lower()


def test_driveability_answer_resolves_pending_field_and_creates_claim(
    experience_data_dir: Path,
) -> None:
    session_id = "driveability-loop-fix"
    SESSIONS[session_id] = ready_incident_state(
        vehicle_drivable=None,
        next_missing_field="vehicle_drivable",
        last_requested_field="vehicle_drivable",
    )

    result = process_chat_message(
        message="Yes, car can still be driven.",
        mobile_number="9876543210",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    state = SESSIONS[session_id]
    assert state["vehicle_drivable"] is True
    assert state["missing_fields"] == []
    assert state["last_requested_field"] == ""
    assert result["claim_status"] == "INITIATED"
    assert "Can the car still be driven?" not in result["response_text"]
    assert result["claim_id"].startswith("CLM2026")


def test_ambiguous_injury_answer_gets_clarification_not_repeat(
    experience_data_dir: Path,
) -> None:
    session_id = "injury-clarification"
    SESSIONS[session_id] = ready_incident_state(
        injury_reported=None,
        vehicle_drivable=True,
        next_missing_field="injury_reported",
        last_requested_field="injury_reported",
    )

    result = process_chat_message(
        message="Not really.",
        mobile_number="9876543210",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )

    state = SESSIONS[session_id]
    assert state["injury_reported"] is None
    assert state["last_requested_field"] == "injury_reported"
    assert result["claim_status"] == "MISSING_INFORMATION"
    assert "Just to confirm" in result["response_text"]
    assert result["response_text"] != "Was anyone injured in the accident?"


def test_last_requested_field_is_passed_to_extractor(
    experience_data_dir: Path,
) -> None:
    session_id = "field-context"
    seen: dict[str, Any] = {}
    SESSIONS[session_id] = ready_incident_state(
        vehicle_drivable=None,
        next_missing_field="vehicle_drivable",
        last_requested_field="vehicle_drivable",
    )

    def context_extractor(_text: str, **kwargs: Any) -> ClaimExtraction:
        seen.update(kwargs)
        return empty_extraction(vehicle_drivable=True)

    result = process_chat_message(
        message="It is roadworthy.",
        mobile_number="9876543210",
        session_id=session_id,
        extractor=context_extractor,
        speech_synthesizer=fake_audio,
    )

    assert seen["last_requested_field"] == "vehicle_drivable"
    assert SESSIONS[session_id]["vehicle_drivable"] is True
    assert result["claim_status"] == "INITIATED"


def test_dynamic_response_generator_receives_controlled_workflow_context(
    experience_data_dir: Path,
) -> None:
    session_id = "dynamic-response-context"
    seen: dict[str, Any] = {}
    SESSIONS[session_id] = ready_incident_state(
        injury_reported=None,
        vehicle_drivable=True,
        next_missing_field="injury_reported",
        last_requested_field="injury_reported",
    )

    def dynamic_response(**kwargs: Any) -> str:
        seen.update(kwargs)
        return "I'm sorry about the broken leg. I'll mark this for specialist review."

    result = process_chat_message(
        message="Not really, but I got a broken leg.",
        mobile_number="9876543210",
        session_id=session_id,
        extractor=lambda _text: empty_extraction(injury_reported=False),
        response_generator=dynamic_response,
        speech_synthesizer=fake_audio,
    )

    assert seen["customer_text"] == "Not really, but I got a broken leg."
    assert seen["state"]["workflow_status"] == "HUMAN_REVIEW"
    assert seen["normalized_extraction"].injury_reported is True
    assert "fallback_response" in seen
    assert result["response_text"] == "I'm sorry about the broken leg. I'll mark this for specialist review."
    assert SESSIONS[session_id]["last_requested_field"] == ""


def test_missing_information_can_continue_same_session(experience_data_dir: Path) -> None:
    first = process_chat_message(
        message="Accident kal hua tha.",
        mobile_number="9876543217",
        session_id="missing",
        extractor=lambda _text: extraction(
            incident_time=None,
            incident_location=None,
            vehicle_damage=None,
            third_party_involved=None,
            injury_reported=None,
            vehicle_drivable=None,
        ),
        speech_synthesizer=fake_audio,
    )

    assert first["claim_id"] == ""
    assert first["claim_status"] == "CUSTOMER_IDENTIFIED"
    assert "पुष्टि" in first["response_text"]
    assert_hindi_response(first["response_text"])

    confirmed = process_chat_message(
        message="haan",
        mobile_number="9876543217",
        session_id="missing",
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=fake_audio,
    )
    assert confirmed["claim_status"] == "MISSING_INFORMATION"
    assert "कितने बजे" in confirmed["response_text"]
    assert_hindi_response(confirmed["response_text"])

    second = process_chat_message(
        message=(
            "Shaam 7 baje T Nagar mein hua, bumper damage hai, "
            "dusri gaadi nahi thi, injury nahi hui, car drive ho sakti hai."
        ),
        mobile_number="9876543217",
        session_id="missing",
        extractor=lambda _text: extraction(
            incident_time="19:00",
            incident_location="T Nagar",
            vehicle_damage="bumper damage",
            third_party_involved=False,
            injury_reported=False,
            vehicle_drivable=True,
        ),
        speech_synthesizer=fake_audio,
    )

    assert second["claim_id"].startswith("CLM2026")
    assert second["claim_status"] == "INITIATED"
    assert len(read_claims(experience_data_dir)) == 1


def test_voice_pipeline_transcribes_then_returns_audio(experience_data_dir: Path) -> None:
    def fake_transcriber(audio_bytes: bytes, **kwargs: Any) -> TranscriptionResult:
        assert audio_bytes == b"fake-webm"
        return TranscriptionResult(
            transcript="Meri car ka accident kal shaam ko Andheri mein hua tha.",
            language_code="hi-IN",
        )

    result = process_voice_message(
        audio_bytes=b"fake-webm",
        filename="recording.webm",
        content_type="audio/webm",
        mobile_number="9876543210",
        session_id="voice",
        transcriber=fake_transcriber,
        extractor=lambda _text: extraction(),
        speech_synthesizer=fake_audio,
    )

    assert result["transcript"].startswith("Meri car")
    assert result["detected_language"] == "hi-IN"
    assert result["claim_status"] == "CUSTOMER_IDENTIFIED"
    assert result["claim_id"] == ""
    assert "पुष्टि" in result["response_text"]
    assert result["audio_url"] == "/media/audio/fake.mp3"
    assert result["audio_available"] is True
    assert result["success"] is True


def test_voice_pipeline_returns_structured_stt_failure(
    experience_data_dir: Path,
) -> None:
    def failing_transcriber(audio_bytes: bytes, **kwargs: Any) -> TranscriptionResult:
        assert audio_bytes == b"fake-webm"
        raise SarvamSpeechError("Saaras transcription failed: ConnectError")

    result = process_voice_message(
        audio_bytes=b"fake-webm",
        filename="browser_recording.webm",
        content_type="audio/webm;codecs=opus",
        mobile_number="9876543210",
        session_id="voice-failure",
        transcriber=failing_transcriber,
        extractor=lambda _text: extraction(),
        speech_synthesizer=fake_audio,
    )

    assert result["success"] is False
    assert result["error_type"] == "NETWORK_FAILURE"
    assert result["transcript"] == ""
    assert result["claim_id"] == ""
    assert result["response_text"]


def test_bulbul_failure_keeps_text_response(experience_data_dir: Path) -> None:
    def failing_audio(text: str, **kwargs: Any) -> SpeechSynthesisResult:
        raise SarvamSpeechError("TTS failed")

    _first = process_chat_message(
        message=(
            "Meri car ka accident kal shaam ko Andheri mein hua tha. "
            "Bumper damage hai, koi dusri gaadi involved nahi thi, "
            "kisi ko injury nahi hui, car drive ho sakti hai."
        ),
        mobile_number="9876543210",
        session_id="tts-fail",
        extractor=lambda _text: extraction(),
        speech_synthesizer=fake_audio,
    )

    result = process_chat_message(
        message="haan",
        mobile_number="9876543210",
        session_id="tts-fail",
        extractor=lambda _text: empty_extraction(),
        speech_synthesizer=failing_audio,
    )

    assert result["claim_id"].startswith("CLM2026")
    assert result["audio_url"] is None
    assert result["audio_available"] is False
    assert "Audio response is unavailable" in result["audio_error"]
    assert "दावा सफलतापूर्वक दर्ज" in result["response_text"]


def test_chat_endpoint_returns_customer_facing_payload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_process_chat_message(**kwargs: Any) -> dict[str, Any]:
        return {
            "session_id": kwargs["session_id"],
            "transcript": kwargs["message"],
            "response_text": "Your claim has been created.",
            "audio_url": "/media/audio/fake.mp3",
            "claim_status": "INITIATED",
            "claim_id": "CLM2026000001",
            "progress": [],
            "next_action": "Vehicle inspection",
            "required_documents": [],
            "claim_summary": {"title": "Claim Registered Successfully!"},
        }

    monkeypatch.setattr(main, "process_chat_message", fake_process_chat_message)

    response = client.post(
        "/api/chat",
        json={
            "message": "Meri car ka accident hua hai.",
            "mobile_number": "9876543210",
            "session_id": "endpoint",
            "language": "hi-IN",
        },
    )

    assert response.status_code == 200
    assert response.json()["claim_id"] == "CLM2026000001"
    assert "route_history" not in response.json()


def test_voice_endpoint_accepts_audio_upload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    experience_data_dir: Path,
) -> None:
    def fake_process_voice_message(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["audio_bytes"] == b"voice-bytes"
        return {
            "session_id": kwargs["session_id"],
            "transcript": "Meri car ka accident hua hai.",
            "response_text": "Your claim has been created.",
            "audio_url": "/media/audio/fake.mp3",
            "claim_status": "INITIATED",
            "claim_id": "CLM2026000001",
            "progress": [],
            "next_action": "Vehicle inspection",
            "required_documents": [],
            "claim_summary": {"title": "Claim Registered Successfully!"},
        }

    monkeypatch.setattr(main, "process_voice_message", fake_process_voice_message)
    debug_dir = experience_data_dir / "debug_audio"
    monkeypatch.setattr(main, "DEBUG_AUDIO_DIR", debug_dir)

    response = client.post(
        "/api/voice/process",
        data={"mobile_number": "9876543210", "session_id": "voice-endpoint"},
        files={"audio": ("browser_recording.webm", b"voice-bytes", "audio/webm;codecs=opus")},
    )

    assert response.status_code == 200
    assert response.json()["transcript"].startswith("Meri car")
    assert response.json()["audio_url"] == "/media/audio/fake.mp3"
    assert (debug_dir / "browser_test.webm").read_bytes() == b"voice-bytes"


def test_session_reset_endpoint(client: TestClient) -> None:
    response = client.post("/api/session/reset", json={"session_id": "abc"})

    assert response.status_code == 200
    assert response.json()["session_id"] != "abc"
    assert response.json()["session_id"]
    assert response.json()["status"] == "reset"
