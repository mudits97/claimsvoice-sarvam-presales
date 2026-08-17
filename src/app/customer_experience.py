"""Customer-facing orchestration for the ClaimsVoice browser experience."""

from __future__ import annotations

import os
import re
import logging
import threading
import time
from inspect import Parameter, signature
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.graph import (
    run_claim_workflow,
    state_updates_from_extraction,
)
from app.sarvam import (
    ClaimExtraction,
    SarvamConfigurationError,
    SarvamExtractionError,
    SarvamResponseGenerationError,
    SarvamSpeechError,
    deterministic_claim_extraction,
    extract_claim_information,
    generate_customer_response,
    normalize_claim_extraction,
    synthesize_speech_to_file,
    transcribe_audio_bytes,
)
from app.rules import REQUIRED_STATE_INCIDENT_FIELDS, calculate_missing_fields
from app.state import ClaimState, build_initial_state
from app.tools import MockBackendError, get_document_requirements


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_AUDIO_DIR = PROJECT_ROOT / "static" / "generated" / "audio"
DISABLE_LLM_RESPONSES_ENV = "CLAIMSVOICE_DISABLE_LLM_RESPONSES"

SESSIONS: dict[str, ClaimState] = {}
logger = logging.getLogger(__name__)


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


CUSTOMER_PROGRESS_LABELS = [
    ("customer", "Customer Identified"),
    ("policy", "Policy Verified"),
    ("information", "Information Captured"),
    ("claim", "Claim Registered"),
    ("documents", "Documents & Next Steps"),
]
HUMAN_REVIEW_PROGRESS_LABELS = [
    ("customer", "Customer Identified"),
    ("policy", "Policy Verified"),
    ("information", "Information Captured"),
    ("review", "Specialist Review Required"),
]
REVIEW_PROGRESS_LABELS = [
    ("customer", "Customer Identified"),
    ("policy", "Policy Verified"),
    ("information", "Information Captured"),
    ("review", "Review Required"),
]

LANGUAGE_STYLES = {"hi", "en"}
QUESTION_FIELD_PRIORITY = (
    "incident_date",
    "incident_time",
    "incident_location",
    "incident_type",
    "vehicle_damage",
    "injury_reported",
    "third_party_involved",
    "vehicle_drivable",
)
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
WORD_PATTERN = re.compile(r"[a-zA-Z]+")
ROMAN_HINDI_MARKERS = {
    "aaj",
    "abhi",
    "aap",
    "bata",
    "baje",
    "chahiye",
    "chot",
    "gaadi",
    "gadi",
    "hai",
    "hain",
    "hua",
    "hui",
    "ka",
    "kal",
    "ke",
    "kisi",
    "ko",
    "kya",
    "mein",
    "meri",
    "mere",
    "nahi",
    "nuksan",
    "paas",
    "shaam",
}
ENGLISH_SWITCH_MARKERS = {
    "my",
    "was",
    "were",
    "the",
    "in",
    "near",
    "outside",
    "vehicle",
    "policy",
    "claim",
    "happened",
    "damaged",
    "injured",
}

MISSING_FIELD_QUESTIONS = {
    "hi": {
        "incident_date": "दुर्घटना किस तारीख को हुई थी?",
        "incident_time": "दुर्घटना लगभग कितने बजे हुई थी?",
        "incident_location": "दुर्घटना कहां हुई थी?",
        "incident_type": "क्या यह टक्कर, चोरी, आग, बाढ़ या किसी दूसरी गाड़ी से जुड़ी घटना थी?",
        "vehicle_damage": "गाड़ी में क्या नुकसान हुआ है?",
        "third_party_involved": "क्या इस दुर्घटना में कोई दूसरी गाड़ी या व्यक्ति शामिल था?",
        "injury_reported": "क्या इस दुर्घटना में किसी को चोट लगी थी?",
        "vehicle_drivable": "क्या गाड़ी अभी चलने की स्थिति में है?",
    },
    "en": {
        "incident_date": "What date did the accident happen?",
        "incident_time": "Approximately what time did the accident happen?",
        "incident_location": "Where did the accident happen?",
        "incident_type": "Was this a collision, theft, fire, flood, or third-party accident?",
        "vehicle_damage": "What damage happened to the vehicle?",
        "third_party_involved": "Was another vehicle or person involved in the accident?",
        "injury_reported": "Was anyone injured in the accident?",
        "vehicle_drivable": "Can the car still be driven?",
    },
}

CLARIFICATION_FIELD_QUESTIONS = {
    "hi": {
        "incident_date": "कृपया दुर्घटना की तारीख स्पष्ट कर दें।",
        "incident_time": "कृपया दुर्घटना का समय थोड़ा स्पष्ट कर दें।",
        "incident_location": "कृपया दुर्घटना का स्थान स्पष्ट कर दें।",
        "incident_type": "कृपया बताएं यह किस तरह की घटना थी।",
        "vehicle_damage": "कृपया गाड़ी के नुकसान के बारे में थोड़ा स्पष्ट बताएं।",
        "third_party_involved": "कृपया स्पष्ट करें कि कोई दूसरी गाड़ी या व्यक्ति शामिल था या नहीं।",
        "injury_reported": "कृपया पुष्टि करें कि आपको या किसी और को कोई शारीरिक चोट लगी थी या नहीं, छोटी चोट भी शामिल है।",
        "vehicle_drivable": "कृपया पुष्टि करें कि गाड़ी अभी सुरक्षित रूप से चलाई जा सकती है या नहीं।",
    },
    "en": {
        "incident_date": "Just to confirm, what date did the accident happen?",
        "incident_time": "Just to confirm, approximately what time did the accident happen?",
        "incident_location": "Just to confirm, where did the accident happen?",
        "incident_type": "Just to confirm, what kind of incident was this?",
        "vehicle_damage": "Just to confirm, what damage happened to the vehicle?",
        "third_party_involved": "Just to confirm, was another vehicle or person involved?",
        "injury_reported": "Just to confirm, did you or anyone else suffer any physical injury, even a minor one?",
        "vehicle_drivable": "Just to confirm, can the car still be driven safely?",
    },
}

RESPONSE_TEMPLATES = {
    "identity_confirmation": {
        "hi": "नमस्ते {salutation}। क्या मैं पुष्टि कर सकता हूँ कि मैं {full_name} जी से बात कर रहा हूँ?",
        "en": "Hello {first_name}. May I confirm that I'm speaking with {full_name}?",
    },
    "identity_mismatch": {
        "hi": "ठीक है। सुरक्षा के लिए मैं इस नंबर से जुड़ी बीमा पॉलिसी की जानकारी साझा नहीं कर सकता। कृपया सही पंजीकृत मोबाइल नंबर से संपर्क करें या सहायता टीम से बात करें।",
        "en": "Thank you for confirming. For your security, I cannot share policy information for this mobile number. Please use the correct registered mobile number or contact support.",
    },
    "post_confirmation_help": {
        "hi": "धन्यवाद {salutation}। बताइए, मैं आपकी किस तरह मदद कर सकता हूँ?",
        "en": "Thank you, {first_name}. How can I help you today?",
    },
    "customer_not_found": {
        "hi": "इस मोबाइल नंबर से जुड़ा ग्राहक मुझे नहीं मिला। कृपया नंबर जांच लें या सहायता केंद्र से संपर्क करें।",
        "en": "I could not find a customer linked to this mobile number. Please check the number or contact support.",
    },
    "policy_not_found": {
        "hi": "आपका ग्राहक विवरण मिला, लेकिन उससे जुड़ी बीमा पॉलिसी नहीं मिली।",
        "en": "I found your customer profile, but could not find an insurance policy linked to it.",
    },
    "policy_not_active": {
        "hi": "आपकी बीमा पॉलिसी अभी सक्रिय नहीं है, इसलिए मैं दावा अपने-आप दर्ज नहीं कर सकता। हमारा प्रतिनिधि आगे की प्रक्रिया में आपकी मदद करेगा।",
        "en": "I found your policy, but it is currently not active. I will not register the claim automatically. A representative can help you with the next steps.",
    },
    "coverage_review": {
        "hi": "उपलब्ध पॉलिसी और दुर्घटना की जानकारी के आधार पर, दावा दर्ज करने से पहले इस मामले की अतिरिक्त समीक्षा आवश्यक है।",
        "en": "Based on the policy information available, I need to have this case reviewed before registering the claim.",
    },
    "human_review": {
        "hi": "यह सुनकर चिंता हुई कि इस दुर्घटना में किसी व्यक्ति को चोट लगी है। ऐसी स्थिति में मामले की समीक्षा बीमा दावा विशेषज्ञ द्वारा करना आवश्यक है। आपकी दी गई जानकारी सुरक्षित रूप से दर्ज कर ली गई है और विशेषज्ञ आगे की प्रक्रिया में आपकी सहायता करेगा।",
        "en": "Since an injury was reported, your case needs to be reviewed by a claims specialist. Your information has been recorded and a specialist will assist you with the next steps.",
    },
    "claim_created": {
        "hi": "धन्यवाद {salutation}। आपका दावा सफलतापूर्वक दर्ज हो गया है। आपका दावा क्रमांक {claim_id} है। अब मैं आपको अगले चरण बता देता हूँ।",
        "en": "{first_name}, your claim has been successfully registered. Your claim number is {claim_id}. I'll now take you through the next steps.",
    },
    "generic_missing": {
        "hi": "दावा दर्ज करने से पहले मुझे थोड़ी और जानकारी चाहिए।",
        "en": "I need a little more information before registering the claim.",
    },
    "generic_error": {
        "hi": "मुझे आपका अनुरोध समझने में परेशानी हो रही है। कृपया फिर से कोशिश करें।",
        "en": "I am having trouble processing your request. Please try again.",
    },
    "voice_error": {
        "hi": "मुझे आपकी आवाज़ समझने में परेशानी हो रही है। आप अपना जवाब लिखकर भी भेज सकते हैं।",
        "en": "I am having trouble processing your voice. You can also type your response.",
    },
}

DOCUMENT_LABELS_HI = {
    "Driving Licence": "ड्राइविंग लाइसेंस",
    "Vehicle Registration Certificate": "वाहन पंजीकरण प्रमाणपत्र",
    "Vehicle photographs": "वाहन की तस्वीरें",
    "Repair estimate from garage": "गैरेज से मरम्मत का अनुमान",
    "Third-party vehicle details": "दूसरी गाड़ी की जानकारी",
    "Police intimation or FIR copy": "पुलिस सूचना या एफआईआर प्रति",
    "Medical report for injured person": "घायल व्यक्ति की मेडिकल रिपोर्ट",
    "Towing or roadside assistance receipt": "टोइंग या सड़क सहायता की रसीद",
}

NEXT_ACTION_LABELS_HI = {
    "Vehicle inspection": "वाहन निरीक्षण",
    "Claims specialist review": "बीमा दावा विशेषज्ञ समीक्षा",
    "Priority roadside assistance and vehicle inspection": "प्राथमिक सड़क सहायता और वाहन निरीक्षण",
}

INCIDENT_TYPE_LABELS_HI = {
    "accident": "वाहन दुर्घटना",
    "collision": "वाहन टक्कर",
    "own_damage": "अपनी गाड़ी का नुकसान",
    "own_vehicle_accident": "अपनी गाड़ी की दुर्घटना",
    "own_vehicle_damage": "अपनी गाड़ी का नुकसान",
    "vehicle_damage": "वाहन नुकसान",
    "third_party": "दूसरे पक्ष से जुड़ी दुर्घटना",
    "third_party_accident": "दूसरे पक्ष से जुड़ी दुर्घटना",
    "third_party_damage": "दूसरे पक्ष का नुकसान",
    "theft": "चोरी",
    "stolen_vehicle": "वाहन चोरी",
    "flood": "बाढ़",
    "natural_calamity": "प्राकृतिक आपदा",
    "fire": "आग",
}

LOCATION_LABELS_HI = {
    "andheri": "अंधेरी",
    "dwarka": "द्वारका",
    "janakpuri": "जनकपुरी",
    "connaught place": "कनॉट प्लेस",
    "koramangala": "कोरमंगला",
    "t nagar": "टी नगर",
    "nh48": "एनएच 48",
    "mathura": "मथुरा",
    "delhi": "दिल्ली",
    "mumbai": "मुंबई",
    "pune": "पुणे",
    "hyderabad": "हैदराबाद",
    "lucknow": "लखनऊ",
    "bengaluru": "बेंगलुरु",
    "chennai": "चेन्नई",
    "jaipur": "जयपुर",
    "ahmedabad": "अहमदाबाद",
    "chandigarh": "चंडीगढ़",
}

VEHICLE_DAMAGE_DISPLAY = {
    "BUMPER_DENT": {
        "en": "Bumper dent",
        "hi": "बम्पर पर डेंट",
    },
    "REAR_BUMPER_DENT": {
        "en": "Rear bumper dent",
        "hi": "पिछले बम्पर पर डेंट",
    },
    "HEADLIGHT_DAMAGE": {
        "en": "Headlight damage",
        "hi": "हेडलाइट क्षतिग्रस्त",
    },
    "DOOR_DAMAGE": {
        "en": "Door damage",
        "hi": "दरवाजे का नुकसान",
    },
    "TYRE_DAMAGE": {
        "en": "Tyre damage",
        "hi": "एक टायर क्षतिग्रस्त",
    },
    "SIDE_MIRROR_DAMAGE": {
        "en": "Side mirror damage",
        "hi": "साइड मिरर क्षतिग्रस्त",
    },
    "WINDSHIELD_DAMAGE": {
        "en": "Windshield damage",
        "hi": "विंडशील्ड क्षतिग्रस्त",
    },
    "NO_MAJOR_DAMAGE_REPORTED": {
        "en": "No major damage reported",
        "hi": "कोई बड़ा नुकसान नहीं बताया गया",
    },
}


def new_session_id() -> str:
    return uuid4().hex


def _normalize_language_style(language_style: str | None) -> str:
    if language_style in LANGUAGE_STYLES:
        return language_style
    return ""


def _language_style_from_code(language_code: str | None) -> str:
    value = (language_code or "").lower()
    if value.startswith("hi"):
        return "hi"
    if value.startswith("en"):
        return "en"
    return ""


def detect_conversation_language(
    customer_text: str,
    *,
    previous_language: str | None = None,
    requested_language: str | None = None,
) -> str:
    previous = _normalize_language_style(previous_language)
    requested = _language_style_from_code(requested_language)
    text = customer_text.strip()

    if not text:
        return previous or requested or "en"
    if DEVANAGARI_PATTERN.search(text):
        return "hi"

    tokens = WORD_PATTERN.findall(text.lower())
    if not tokens:
        return previous or requested or "en"
    if any(token in ROMAN_HINDI_MARKERS for token in tokens):
        return "hi"
    if previous == "hi" and not any(token in ENGLISH_SWITCH_MARKERS for token in tokens):
        return "hi"
    if any(token in ENGLISH_SWITCH_MARKERS for token in tokens) or len(tokens) >= 4:
        return "en"
    return previous or requested or "en"


def _response_language_from_state(state: ClaimState) -> str:
    return (
        _normalize_language_style(state.get("response_language"))
        or _normalize_language_style(state.get("conversation_language"))
        or "en"
    )


def _summary_language_from_state(state: ClaimState) -> str:
    return _normalize_language_style(state.get("ui_language")) or _response_language_from_state(state)


def _localized_template(state: ClaimState, key: str, **kwargs: Any) -> str:
    language = _response_language_from_state(state)
    template = RESPONSE_TEMPLATES[key][language]
    return template.format(**kwargs)


def _first_name(customer_name: str | None) -> str:
    return (customer_name or "").strip().split()[0] if (customer_name or "").strip() else ""


def _name_context(state: ClaimState) -> dict[str, str]:
    full_name = state.get("customer_name", "").strip()
    first_name = _first_name(full_name) or "there"
    return {
        "full_name": full_name or first_name,
        "first_name": first_name,
        "salutation": f"{first_name} जी" if first_name != "there" else "जी",
    }


def _localized_named_template(state: ClaimState, key: str, **kwargs: Any) -> str:
    return _localized_template(state, key, **_name_context(state), **kwargs)


def _has_incident_context(state: ClaimState) -> bool:
    return any(
        state.get(field_name) not in ("", None)
        for field_name in REQUIRED_STATE_INCIDENT_FIELDS
    )


def _identity_words(value: str | None) -> list[str]:
    return [word for word in WORD_PATTERN.findall((value or "").lower()) if len(word) > 1]


def _clean_claimed_name(value: str) -> str:
    words = [
        word
        for word in WORD_PATTERN.findall(value.lower())
        if word
        not in {
            "yes",
            "yeah",
            "yep",
            "no",
            "nope",
            "mr",
            "mrs",
            "ms",
            "miss",
            "ji",
            "only",
            "here",
            "speaking",
            "this",
            "is",
            "am",
            "i",
            "im",
        }
    ]
    return " ".join(word.capitalize() for word in words[:3])


def speaker_claimed_name_from_text(customer_text: str) -> str:
    text = customer_text.strip().lower()
    if not text:
        return ""

    patterns = (
        r"\b(?:you are|you're)\s+(?:speaking|talking)\s+with\s+([a-z]+(?:\s+[a-z]+){0,2})\b",
        r"\b(?:i am|i'm|im|this is|my name is)\s+([a-z]+(?:\s+[a-z]+){0,2})\b",
        r"\b(?:main|mein)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:hoon|hu|hun)\b",
        r"\b([a-z]+(?:\s+[a-z]+){0,2})\s+speaking\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _clean_claimed_name(match.group(1))

    if re.search(r"\b(?:no|nope|not|nahi|nahin)\b", text) or "नहीं" in customer_text:
        name_match = re.search(r"\b([a-z]+)\s+kumar\b", text)
        if name_match:
            return _clean_claimed_name(name_match.group(0))
    return ""


def _claimed_name_matches_expected(claimed_name: str, expected_customer_name: str | None) -> bool:
    claimed_words = _identity_words(claimed_name)
    expected_words = _identity_words(expected_customer_name)
    if not claimed_words or not expected_words:
        return False
    expected_first_name = expected_words[0]
    if claimed_words[0] != expected_first_name:
        return False
    return all(word in expected_words for word in claimed_words)


def _mentions_expected_as_other_person(text: str, expected_first_name: str) -> bool:
    if not expected_first_name:
        return False
    other_person_patterns = (
        rf"\b(?:father|mother|brother|sister|friend|owner)\s+{re.escape(expected_first_name)}\b",
        rf"\b{re.escape(expected_first_name)}\s+(?:owns|owner|is\s+my\s+father|is\s+my\s+mother)\b",
    )
    return any(re.search(pattern, text) for pattern in other_person_patterns)


def identity_confirmation_from_text(
    customer_text: str,
    expected_customer_name: str | None = None,
) -> bool | None:
    text = customer_text.strip().lower()
    if not text:
        return None

    expected_words = _identity_words(expected_customer_name)
    expected_first_name = expected_words[0] if expected_words else ""
    claimed_name = speaker_claimed_name_from_text(customer_text)
    if claimed_name:
        return _claimed_name_matches_expected(claimed_name, expected_customer_name)

    negative_terms = (
        "no",
        "nope",
        "not me",
        f"not {expected_first_name}" if expected_first_name else "not registered customer",
        "wrong",
        "incorrect",
        "nahi",
        "nahin",
        "main nahi",
        "not the",
        "नहीं",
        "गलत",
        "मैं नहीं",
    )
    if any(term in text for term in negative_terms):
        return False

    positive_terms = (
        "yes",
        "yeah",
        "yep",
        "correct",
        "right",
        "confirm",
        "that's me",
        "that is me",
        "it is me",
        "it's me",
        "its me",
        "haan",
        "han",
        "ha",
        "ji",
        "sahi",
        "हाँ",
        "हां",
        "जी",
        "सही",
        "ठीक",
    )
    tokens = set(WORD_PATTERN.findall(text))
    if _mentions_expected_as_other_person(text, expected_first_name):
        return False
    if expected_words and any(
        re.search(rf"\b{re.escape(word)}\b", text)
        for word in expected_words
    ) and any(term in text for term in ("i am", "i'm", "this is", "main", "mein", "speaking")):
        return True
    if expected_first_name and "kumar" in tokens and expected_first_name not in tokens:
        return False
    if expected_first_name and re.search(r"\b[a-z]+\s+kumar\b", text):
        matched_first = re.search(r"\b([a-z]+)\s+kumar\b", text)
        if matched_first and matched_first.group(1) != expected_first_name:
            return False
    if any(term in text for term in positive_terms if len(term) > 2):
        return True
    if tokens & set(positive_terms):
        return True
    return None


def _next_missing_field(state: ClaimState) -> str:
    current_missing = state.get("missing_fields") or calculate_missing_fields(state)
    for field_name in QUESTION_FIELD_PRIORITY:
        if field_name in current_missing:
            return field_name
    return ""


def specific_missing_field_question(state: ClaimState) -> str:
    language = _response_language_from_state(state)
    missing_field = _next_missing_field(state)
    if not missing_field:
        return _localized_template(state, "generic_missing")
    question_set = MISSING_FIELD_QUESTIONS
    if state.get("last_requested_field") == missing_field and not state.get("last_captured_fields"):
        question_set = CLARIFICATION_FIELD_QUESTIONS
    question = question_set[language][missing_field]
    acknowledgment = _contextual_acknowledgment(state)
    return f"{acknowledgment} {question}" if acknowledgment else question


def _minor_damage_reported(state: ClaimState) -> bool:
    damage = (state.get("vehicle_damage") or "").lower()
    minor_terms = ("minor", "small", "dent", "डेंट", "हल्का", "हल्की", "थोड़ा", "थोड़ी", "सीमित")
    return any(term in damage for term in minor_terms)


def _tyre_damage_reported(state: ClaimState) -> bool:
    damage = (state.get("vehicle_damage") or "").lower()
    return "tyre" in damage or "tire" in damage or "टायर" in damage


def _captured_field_summary(captured_fields: list[str], language: str) -> str:
    if language == "en":
        names = {
            "incident_date": "date",
            "incident_time": "time",
            "incident_location": "location",
            "incident_type": "incident type",
            "vehicle_damage": "vehicle damage",
            "third_party_involved": "third-party involvement",
            "injury_reported": "injury status",
            "vehicle_drivable": "vehicle drivability",
        }
        captured = [names[field] for field in captured_fields if field in names]
        if not captured:
            return "I understand."
        return f"I have noted the {' and '.join(captured[:2])}."

    names = {
        "incident_date": "तारीख",
        "incident_time": "समय",
        "incident_location": "स्थान",
        "incident_type": "घटना का प्रकार",
        "vehicle_damage": "नुकसान",
        "third_party_involved": "दूसरे पक्ष की जानकारी",
        "injury_reported": "चोट की जानकारी",
        "vehicle_drivable": "गाड़ी की स्थिति",
    }
    captured = [names[field] for field in captured_fields if field in names]
    if not captured:
        return "समझ गया।"
    return f"ठीक है, मैंने {' और '.join(captured[:2])} नोट कर लिया है।"


def _contextual_acknowledgment(state: ClaimState) -> str:
    language = _response_language_from_state(state)
    captured_fields = state.get("last_captured_fields", [])

    if state.get("identity_just_confirmed"):
        names = _name_context(state)
        if language == "en":
            if state.get("policy_status") == "ACTIVE":
                return (
                    f"I understand, {names['first_name']}. I've found your insurance policy "
                    "and confirmed that it is active. I'll help you register the claim."
                )
            return f"I understand, {names['first_name']}. I'll help you register the claim."
        if state.get("policy_status") == "ACTIVE":
            return (
                f"समझ गया {names['salutation']}। मुझे आपकी बीमा पॉलिसी मिल गई है "
                "और वह सक्रिय है। मैं आपका दावा दर्ज करने में मदद करता हूँ।"
            )
        return f"समझ गया {names['salutation']}। मैं आपका दावा दर्ज करने में मदद करता हूँ।"

    if "injury_reported" in captured_fields:
        if state.get("injury_reported") is True:
            if language == "en":
                return "I understand. Because someone was injured, this matter may need specialist review."
            return "यह सुनकर चिंता हुई कि किसी व्यक्ति को चोट लगी है। इस मामले की विशेषज्ञ द्वारा समीक्षा आवश्यक हो सकती है।"
        if state.get("injury_reported") is False:
            if language == "en":
                return "Good to know that no one was injured."
            return "अच्छा है कि किसी को चोट नहीं लगी।"

    if "vehicle_drivable" in captured_fields and state.get("vehicle_drivable") is False:
        if language == "en":
            return "I understand. Since the car cannot be driven, you may need further assistance."
        return "ठीक है। चूँकि गाड़ी चलने की स्थिति में नहीं है, आगे सहायता की आवश्यकता हो सकती है।"

    if "vehicle_drivable" in captured_fields and state.get("vehicle_drivable") is True:
        if language == "en":
            return "Understood. I've noted that the vehicle can still be driven."
        return "ठीक है, मैंने नोट कर लिया है कि गाड़ी अभी चलने की स्थिति में है।"

    if "vehicle_damage" in captured_fields:
        if language == "en":
            if _tyre_damage_reported(state):
                return "I have noted the tyre damage."
            if _minor_damage_reported(state):
                return "I understand. It is good that the damage appears limited."
            return "I have noted the vehicle damage."
        if _tyre_damage_reported(state):
            return "ठीक है, मैंने टायर के नुकसान की जानकारी दर्ज कर ली है।"
        if _minor_damage_reported(state):
            return "ठीक है, समझ गया। अच्छा है कि नुकसान सीमित है।"
        return "ठीक है, मैंने गाड़ी के नुकसान की जानकारी नोट कर ली है।"

    if "third_party_involved" in captured_fields:
        if language == "en":
            return "I have noted that."
        return "ठीक है, मैंने नोट कर लिया।"

    if captured_fields:
        return _captured_field_summary(captured_fields, language)

    if language == "en":
        return "I understand. I can help you register the claim."
    return "समझ गया। मैं आपका दावा दर्ज करने में मदद करता हूँ।"


def _remember_language(
    state: ClaimState,
    *,
    customer_text: str,
    requested_language: str | None,
) -> ClaimState:
    requested = _language_style_from_code(requested_language)
    if requested:
        state["ui_language"] = requested
    language = detect_conversation_language(
        customer_text,
        previous_language=state.get("response_language") or state.get("conversation_language"),
        requested_language=requested_language,
    )
    state["conversation_language"] = language
    state["response_language"] = requested or language
    return state


def _tts_language_code(response_language: str, requested_language: str) -> str:
    if response_language == "en":
        return "en-IN"
    return "hi-IN"


def _queue_tts_generation(
    *,
    response_text: str,
    language_code: str,
    speech_synthesizer,
    latency_trace: dict[str, Any],
) -> str:
    filename = f"claimsvoice-{uuid4().hex}.mp3"
    audio_url = f"/media/audio/{filename}"

    def worker() -> None:
        started_at = time.perf_counter()
        logger.warning("VOICE TURN LATENCY T10 Bulbul TTS started audio_url=%s", audio_url)
        try:
            speech_synthesizer(
                response_text,
                output_dir=GENERATED_AUDIO_DIR,
                language_code=language_code,
                filename=filename,
            )
            duration_ms = _elapsed_ms(started_at)
            logger.warning(
                "VOICE TURN LATENCY T11 first playable audio available audio_url=%s bulbul_tts_ms=%s",
                audio_url,
                duration_ms,
            )
        except Exception as exc:
            logger.warning(
                "VOICE TURN LATENCY Bulbul TTS failed audio_url=%s error=%s",
                audio_url,
                type(exc).__name__,
            )

    queued_at = time.perf_counter()
    threading.Thread(target=worker, daemon=True).start()
    latency_trace["bulbul_tts_status"] = "queued"
    latency_trace["bulbul_tts_blocking"] = False
    latency_trace["tts_queue_ms"] = _elapsed_ms(queued_at)
    return audio_url


def reset_session(session_id: str | None = None) -> str:
    if session_id:
        SESSIONS.pop(session_id, None)
    return new_session_id()


def render_session_view(session_id: str, language: str = "hi-IN") -> dict[str, Any]:
    state = deepcopy(SESSIONS.get(session_id) or build_initial_state())
    requested = _language_style_from_code(language)
    if requested:
        state["ui_language"] = requested
        state["response_language"] = requested
    SESSIONS[session_id] = state
    return _response_payload(
        session_id=session_id,
        transcript="",
        state=state,
        response_text=state.get("response_message", ""),
    )


def _get_session_state(session_id: str, mobile_number: str, customer_text: str) -> ClaimState:
    existing = deepcopy(SESSIONS.get(session_id))
    if existing:
        previous_mobile_number = existing.get("mobile_number", "")
        if mobile_number and previous_mobile_number and mobile_number != previous_mobile_number:
            return build_initial_state(mobile_number=mobile_number, raw_customer_input=customer_text)
        existing["mobile_number"] = mobile_number or existing.get("mobile_number", "")
        existing["raw_customer_input"] = customer_text
        return existing
    return build_initial_state(mobile_number=mobile_number, raw_customer_input=customer_text)


def _merge_extraction_into_state(
    state: ClaimState,
    extraction: ClaimExtraction,
    *,
    raw_extraction: ClaimExtraction | None = None,
) -> ClaimState:
    merged = deepcopy(state)
    updates = state_updates_from_extraction(extraction)
    captured_fields: list[str] = []
    for key, value in updates.items():
        if key == "intent" and value:
            merged[key] = value
        elif key != "intent" and value not in ("", None):
            if merged.get(key) != value:
                captured_fields.append(key)
            merged[key] = value
            if key == "vehicle_damage":
                normalized_damage = str(value)
                raw_damage = raw_extraction.vehicle_damage if raw_extraction else None
                merged["vehicle_damage_code"] = _vehicle_damage_code_from_text(normalized_damage)
                merged["vehicle_damage_raw_evidence"] = _vehicle_damage_raw_evidence(
                    customer_text=state.get("raw_customer_input", ""),
                    normalized_value=normalized_damage,
                    raw_value=raw_damage,
                )
    merged["last_captured_fields"] = captured_fields
    return merged


def _extract_with_field_context(
    extractor,
    customer_text: str,
    last_requested_field: str,
) -> ClaimExtraction:
    try:
        extractor_signature = signature(extractor)
    except (TypeError, ValueError):
        return extractor(customer_text)

    parameters = extractor_signature.parameters
    accepts_context = (
        "last_requested_field" in parameters
        or any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())
    )
    if accepts_context:
        return extractor(customer_text, last_requested_field=last_requested_field or None)
    return extractor(customer_text)


def _llm_response_disabled(response_generator) -> bool:
    return (
        response_generator is generate_customer_response
        and os.environ.get(DISABLE_LLM_RESPONSES_ENV, "").strip().lower() in {"1", "true", "yes"}
    )


def _requires_deterministic_response(state: ClaimState) -> bool:
    status = state.get("workflow_status", "")
    return (
        (status == "CUSTOMER_IDENTIFIED" and state.get("identity_confirmed") is not True)
        or status in {"IDENTITY_MISMATCH", "CUSTOMER_NOT_FOUND", "POLICY_NOT_FOUND"}
    )


def _customer_response_with_dynamic_fallback(
    *,
    customer_text: str,
    state: ClaimState,
    raw_extraction: ClaimExtraction | None,
    normalized_extraction: ClaimExtraction | None,
    response_generator,
) -> str:
    fallback_response = _customer_response_from_state(state)
    if _requires_deterministic_response(state):
        return fallback_response
    if response_generator is None or _llm_response_disabled(response_generator):
        return fallback_response

    try:
        return response_generator(
            customer_text=customer_text,
            state=state,
            raw_extraction=raw_extraction,
            normalized_extraction=normalized_extraction,
            fallback_response=fallback_response,
            response_language=_response_language_from_state(state),
        )
    except (SarvamConfigurationError, SarvamResponseGenerationError, Exception) as exc:
        logger.warning(
            "Dynamic customer response unavailable; using deterministic fallback. error=%s",
            type(exc).__name__,
        )
        return fallback_response


def _progress_from_state(state: ClaimState) -> list[dict[str, str]]:
    status = state.get("workflow_status", "")
    claim_status = state.get("claim_status", "")
    incident_complete = not calculate_missing_fields(state)
    if status == "HUMAN_REVIEW":
        labels = HUMAN_REVIEW_PROGRESS_LABELS
    elif status in {"COVERAGE_REVIEW_REQUIRED", "POLICY_NOT_ACTIVE"}:
        labels = REVIEW_PROGRESS_LABELS
    else:
        labels = CUSTOMER_PROGRESS_LABELS

    completed = set()
    current = "customer"

    if state.get("customer_id") and state.get("identity_confirmed") is True:
        completed.add("customer")
        current = "policy"
    elif state.get("customer_id") and state.get("identity_mismatch") is not True:
        current = "customer"
    if state.get("identity_confirmed") is True and state.get("policy_id") and state.get("policy_status"):
        completed.add("policy")
        current = "information"
    if incident_complete:
        completed.add("information")
        current = "claim"
    if state.get("claim_id") and claim_status == "INITIATED":
        completed.add("claim")
        current = "documents"
    if status == "HUMAN_REVIEW":
        if incident_complete:
            completed.add("information")
        current = "review"
    if state.get("required_documents"):
        completed.add("documents")
        current = "documents"

    if status in {"POLICY_NOT_ACTIVE", "COVERAGE_REVIEW_REQUIRED"}:
        current = "review"

    progress = []
    for key, label in labels:
        if key in completed:
            item_status = "complete"
        elif key == current:
            item_status = "current"
        else:
            item_status = "pending"
        progress.append({"key": key, "label": label, "status": item_status})
    return progress


def _customer_response_from_state(state: ClaimState) -> str:
    status = state.get("workflow_status", "")

    if status == "IDENTITY_MISMATCH":
        return _localized_template(state, "identity_mismatch")
    if status == "CUSTOMER_NOT_FOUND":
        return _localized_template(state, "customer_not_found")
    if status == "CUSTOMER_IDENTIFIED" and state.get("identity_confirmed") is not True:
        return _localized_named_template(state, "identity_confirmation")
    if status == "POLICY_NOT_FOUND":
        return _localized_template(state, "policy_not_found")
    if status == "POLICY_NOT_ACTIVE":
        return _localized_template(state, "policy_not_active")
    if status == "MISSING_INFORMATION":
        return specific_missing_field_question(state)
    if status == "COVERAGE_REVIEW_REQUIRED":
        return _localized_template(state, "coverage_review")
    if status == "HUMAN_REVIEW":
        return _localized_template(state, "human_review")
    if state.get("claim_status") == "INITIATED":
        return _localized_named_template(
            state,
            "claim_created",
            claim_id=state.get("claim_id"),
        )
    return state.get("response_message") or _localized_template(state, "generic_error")


def _finalize_response_state(state: ClaimState, response_text: str) -> ClaimState:
    finalized = deepcopy(state)
    if finalized.get("workflow_status") == "MISSING_INFORMATION":
        finalized["missing_fields"] = calculate_missing_fields(finalized)
        next_field = _next_missing_field(finalized)
        finalized["next_missing_field"] = next_field
        finalized["last_requested_field"] = next_field
        finalized["last_question"] = response_text
    else:
        finalized["next_missing_field"] = ""
        if finalized.get("workflow_status") in {
            "CLAIM_CREATED",
            "HUMAN_REVIEW",
            "COVERAGE_REVIEW_REQUIRED",
            "POLICY_NOT_ACTIVE",
            "IDENTITY_MISMATCH",
        }:
            finalized["last_requested_field"] = ""
            finalized["last_question"] = ""
    finalized["response_message"] = response_text
    return finalized


def _debug_state_trace(
    *,
    session_id: str,
    customer_text: str,
    raw_extraction: ClaimExtraction | None,
    normalized_extraction: ClaimExtraction | None,
    state_before: ClaimState,
    state_after: ClaimState,
) -> None:
    logger.info(
        "ClaimsVoice state trace session_id=%s turn=%s user_message=%r raw_extraction=%s "
        "normalized_extraction=%s state_before=%s state_after=%s missing_fields=%s "
        "next_requested_field=%s workflow_route=%s",
        session_id,
        len(state_after.get("conversation_messages", [])),
        customer_text,
        raw_extraction.model_dump() if raw_extraction else None,
        normalized_extraction.model_dump() if normalized_extraction else None,
        state_before,
        state_after,
        calculate_missing_fields(state_after),
        state_after.get("next_missing_field", ""),
        state_after.get("route_history", []),
    )


def _yes_no(value: bool | None, language: str) -> str:
    if value is True:
        return "हाँ" if language == "hi" else "Yes"
    if value is False:
        return "नहीं" if language == "hi" else "No"
    return "दर्ज नहीं" if language == "hi" else "Not captured"


def _display_value(value: Any, language: str) -> str:
    if value in ("", None, []):
        return "दर्ज नहीं" if language == "hi" else "Not captured"
    return str(value)


def _sentence_case(value: str) -> str:
    cleaned = value.replace("_", " ").strip()
    if not cleaned:
        return ""
    return cleaned[:1].upper() + cleaned[1:]


def _vehicle_damage_code_from_text(value: str | None) -> str:
    if not value:
        return ""
    raw_value = value.strip()
    text = raw_value.lower()
    if not text:
        return ""
    if any(term in text for term in ("tyre", "tire", "puncture")) or any(
        term in raw_value for term in ("टायर", "पंक्चर")
    ):
        return "TYRE_DAMAGE"
    if "bumper" in text or "बम्पर" in raw_value or "बंपर" in raw_value:
        if "rear" in text or "पीछे" in raw_value or "पिछ" in raw_value:
            return "REAR_BUMPER_DENT"
        return "BUMPER_DENT"
    if "headlight" in text or "headlamp" in text or "हेडलाइट" in raw_value:
        return "HEADLIGHT_DAMAGE"
    if "door" in text or "दरवाज" in raw_value:
        return "DOOR_DAMAGE"
    if "mirror" in text or "शीशा" in raw_value:
        return "SIDE_MIRROR_DAMAGE"
    if any(term in text for term in ("windshield", "windscreen")) or any(
        term in raw_value for term in ("कांच", "विंडशील्ड")
    ):
        return "WINDSHIELD_DAMAGE"
    if "no major damage" in text or "ज्यादा नुकसान नहीं" in raw_value:
        return "NO_MAJOR_DAMAGE_REPORTED"
    return ""


def _vehicle_damage_raw_evidence(
    *,
    customer_text: str,
    normalized_value: str,
    raw_value: str | None,
) -> str:
    if raw_value and raw_value.strip():
        return raw_value.strip()
    code = _vehicle_damage_code_from_text(normalized_value)
    if DEVANAGARI_PATTERN.search(customer_text) and code in VEHICLE_DAMAGE_DISPLAY:
        return VEHICLE_DAMAGE_DISPLAY[code]["hi"]
    return normalized_value


def _vehicle_damage_display(state: ClaimState, language: str) -> str:
    normalized_value = state.get("vehicle_damage") or ""
    raw_evidence = state.get("vehicle_damage_raw_evidence") or ""
    code = (
        state.get("vehicle_damage_code")
        or _vehicle_damage_code_from_text(normalized_value)
        or _vehicle_damage_code_from_text(raw_evidence)
    )

    if language == "hi":
        if raw_evidence and DEVANAGARI_PATTERN.search(raw_evidence):
            return raw_evidence
        if code in VEHICLE_DAMAGE_DISPLAY:
            return VEHICLE_DAMAGE_DISPLAY[code]["hi"]
        return _display_value(raw_evidence or normalized_value, language)

    if code in VEHICLE_DAMAGE_DISPLAY:
        return VEHICLE_DAMAGE_DISPLAY[code]["en"]
    return _sentence_case(str(normalized_value or raw_evidence))


def _incident_location_display(value: str | None, language: str) -> str:
    if not value:
        return _display_value(value, language)
    location = value.strip()
    lower_location = location.lower()
    if language == "hi":
        if "office" in lower_location and ("बाहर" in location or "bahar" in lower_location):
            return "ऑफिस के बाहर"
        if "school" in lower_location and ("बाहर" in location or "bahar" in lower_location):
            return "स्कूल के बाहर"
        if lower_location in LOCATION_LABELS_HI:
            return LOCATION_LABELS_HI[lower_location]
        return location

    if "office" in lower_location and ("बाहर" in location or "bahar" in lower_location):
        return "Outside the office"
    if "school" in lower_location and ("बाहर" in location or "bahar" in lower_location):
        return "Outside school"
    return location


def _incident_type_display(value: str | None, language: str) -> str:
    if not value:
        return _display_value(value, language)
    if language == "hi":
        return INCIDENT_TYPE_LABELS_HI.get(value, value.replace("_", " "))
    return value.replace("_", " ").title()


def _claim_status_display(state: ClaimState, language: str) -> str:
    claim_status = state.get("claim_status", "")
    workflow_status = state.get("workflow_status", "")
    if claim_status == "INITIATED":
        return "दर्ज" if language == "hi" else "Registered"
    if claim_status == "HUMAN_REVIEW" or workflow_status == "HUMAN_REVIEW":
        return "विशेषज्ञ समीक्षा आवश्यक" if language == "hi" else "Specialist review required"
    if workflow_status == "COVERAGE_REVIEW_REQUIRED":
        return "समीक्षा आवश्यक" if language == "hi" else "Review required"
    if workflow_status == "POLICY_NOT_ACTIVE":
        return "पॉलिसी समीक्षा आवश्यक" if language == "hi" else "Policy review required"
    return _display_value(claim_status or workflow_status, language)


def _document_label(document: dict[str, Any], language: str) -> str:
    name = document.get("name", "")
    if language == "hi":
        return DOCUMENT_LABELS_HI.get(name, name)
    return name


def _next_action_display(next_action: str, language: str) -> str:
    if not next_action:
        return _display_value(next_action, language)
    if language == "hi":
        return NEXT_ACTION_LABELS_HI.get(next_action, next_action)
    return next_action


def _vehicle_display(state: ClaimState, language: str) -> str:
    vehicle_name = state.get("vehicle_name", "")
    registration = state.get("vehicle_registration", "")
    if vehicle_name and registration:
        return f"{vehicle_name} / {registration}"
    return _display_value(vehicle_name or registration, language)


def _summary_labels(language: str) -> dict[str, str]:
    if language == "hi":
        return {
            "claim_id": "दावा क्रमांक",
            "reference_id": "संदर्भ क्रमांक",
            "customer": "ग्राहक",
            "policy": "पॉलिसी क्रमांक",
            "vehicle": "वाहन / गाड़ी",
            "date": "दुर्घटना की तारीख",
            "time": "दुर्घटना का समय",
            "location": "स्थान",
            "incident_type": "घटना का प्रकार",
            "damage": "वाहन का नुकसान",
            "third_party": "दूसरी गाड़ी/व्यक्ति शामिल",
            "injury": "किसी को चोट",
            "drivable": "गाड़ी चलने की स्थिति में",
            "status": "दावे की स्थिति",
            "documents": "आवश्यक दस्तावेज",
            "next_action": "अगला चरण",
            "live_title": "अब तक दर्ज जानकारी",
        }
    return {
        "claim_id": "Claim ID",
        "reference_id": "Reference number",
        "customer": "Customer name",
        "policy": "Policy number",
        "vehicle": "Vehicle",
        "date": "Incident date",
        "time": "Incident time",
        "location": "Incident location",
        "incident_type": "Incident type",
        "damage": "Vehicle damage",
        "third_party": "Third-party involvement",
        "injury": "Injury reported",
        "drivable": "Vehicle driveable",
        "status": "Claim status",
        "documents": "Required documents",
        "next_action": "Next action",
        "live_title": "Information captured so far",
    }


def _summary_items(state: ClaimState, language: str) -> list[dict[str, str]]:
    labels = _summary_labels(language)
    items = _captured_summary_items(state, language)
    status_value = _claim_status_display(state, language)
    if status_value not in ("दर्ज नहीं", "Not captured"):
        items.append({"label": labels["status"], "value": status_value})
    return items


def _known_value(value: Any) -> bool:
    return value not in ("", None, [])


def _captured_summary_items(state: ClaimState, language: str) -> list[dict[str, str]]:
    if state.get("identity_mismatch"):
        return []

    labels = _summary_labels(language)
    items: list[dict[str, str]] = []

    if _known_value(state.get("customer_name")):
        items.append({"label": labels["customer"], "value": str(state["customer_name"])})
    policy_visible = state.get("identity_confirmed") is True or _known_value(state.get("policy_status"))
    if policy_visible and _known_value(state.get("policy_id")):
        items.append({"label": labels["policy"], "value": str(state["policy_id"])})
    if policy_visible and (
        _known_value(state.get("vehicle_name")) or _known_value(state.get("vehicle_registration"))
    ):
        items.append({"label": labels["vehicle"], "value": _vehicle_display(state, language)})
    if _known_value(state.get("incident_date")):
        items.append({"label": labels["date"], "value": str(state["incident_date"])})
    if _known_value(state.get("incident_time")):
        items.append({"label": labels["time"], "value": str(state["incident_time"])})
    if _known_value(state.get("incident_location")):
        items.append(
            {
                "label": labels["location"],
                "value": _incident_location_display(state.get("incident_location"), language),
            }
        )
    if _known_value(state.get("incident_type")):
        items.append(
            {
                "label": labels["incident_type"],
                "value": _incident_type_display(state.get("incident_type"), language),
            }
        )
    if _known_value(state.get("vehicle_damage")):
        items.append({"label": labels["damage"], "value": _vehicle_damage_display(state, language)})
    if state.get("third_party_involved") is not None:
        items.append(
            {
                "label": labels["third_party"],
                "value": _yes_no(state.get("third_party_involved"), language),
            }
        )
    if state.get("injury_reported") is not None:
        items.append({"label": labels["injury"], "value": _yes_no(state.get("injury_reported"), language)})
    if state.get("vehicle_drivable") is not None:
        items.append({"label": labels["drivable"], "value": _yes_no(state.get("vehicle_drivable"), language)})
    return items


def _identity_mismatch_summary(state: ClaimState, language: str) -> dict[str, Any]:
    if language == "hi":
        return {
            "type": "identity_verification",
            "title": "पहचान सत्यापन आवश्यक",
            "items": [
                {
                    "label": "स्थिति",
                    "value": "ग्राहक पहचान सत्यापित नहीं हुई",
                }
            ],
        }
    return {
        "type": "identity_verification",
        "title": "Identity verification required",
        "items": [
            {
                "label": "Status",
                "value": "Customer identity could not be verified",
            }
        ],
    }


def _captured_summary(state: ClaimState) -> dict[str, Any] | None:
    language = _summary_language_from_state(state)
    if state.get("identity_mismatch"):
        return _identity_mismatch_summary(state, language)

    labels = _summary_labels(language)
    items = _captured_summary_items(state, language)
    if not items:
        return None
    return {
        "type": "captured_so_far",
        "title": labels["live_title"],
        "items": items,
    }


def _document_items(state: ClaimState, language: str) -> list[dict[str, str]]:
    return [
        {
            "label": _document_label(document, language),
            "value": "आवश्यक" if language == "hi" else "Required",
        }
        for document in state.get("required_documents", [])
    ]


def _summary_next_steps(state: ClaimState, language: str) -> list[dict[str, str]]:
    if state.get("workflow_status") == "HUMAN_REVIEW":
        if language == "hi":
            return [
                {
                    "title": "बीमा दावा विशेषज्ञ समीक्षा",
                    "description": "बीमा दावा विशेषज्ञ इस मामले की समीक्षा करेगा और आगे की प्रक्रिया में आपकी सहायता करेगा।",
                }
            ]
        return [
            {
                "title": "Claims specialist review",
                "description": "A claims specialist will review this case and help you with the next steps.",
            }
        ]
    if state.get("workflow_status") == "COVERAGE_REVIEW_REQUIRED":
        if language == "hi":
            return [
                {
                    "title": "अतिरिक्त समीक्षा",
                    "description": "बीमा टीम उपलब्ध पॉलिसी और दुर्घटना की जानकारी देखकर आगे की प्रक्रिया बताएगी।",
                }
            ]
        return [
            {
                "title": "Additional review",
                "description": "The insurance team will review the available policy and incident details before the next step.",
            }
        ]

    if language == "hi":
        return [
            {
                "title": "वाहन की तस्वीरें अपलोड करें",
                "description": "गाड़ी के नुकसान की साफ तस्वीरें साझा करें।",
            },
            {
                "title": "आवश्यक दस्तावेज़ जमा करें",
                "description": "ड्राइविंग लाइसेंस, वाहन पंजीकरण प्रमाणपत्र और बाकी जरूरी दस्तावेज़ जमा करें।",
            },
            {
                "title": "वाहन निरीक्षण",
                "description": "हम आपके साथ वाहन निरीक्षण निर्धारित करेंगे।",
            },
        ]
    return [
        {
            "title": "Upload vehicle photographs",
            "description": "Share clear photos of the vehicle damage.",
        },
        {
            "title": "Submit required documents",
            "description": "Driving licence, RC and other required documents.",
        },
        {
            "title": "Vehicle inspection",
            "description": "We will schedule inspection with you.",
        },
    ]


def _claim_summary(state: ClaimState) -> dict[str, Any] | None:
    language = _summary_language_from_state(state)
    labels = _summary_labels(language)
    is_success = state.get("claim_status") == "INITIATED"
    workflow_status = state.get("workflow_status")
    is_review = workflow_status in {"HUMAN_REVIEW", "COVERAGE_REVIEW_REQUIRED"}
    if not is_success and not is_review:
        return None

    if language == "hi":
        if is_success:
            title = "आपके दावे का सारांश"
        else:
            title = "दर्ज की गई जानकारी"
        subtitle = (
            "यह वह जानकारी है जो दावा दर्ज करने के लिए भेजी गई है।"
            if is_success
            else "हमने यह जानकारी दर्ज की है।"
        )
        section_heading = "घटना की जानकारी"
        next_heading = "अगले चरण"
    else:
        if is_success:
            title = "Claim Summary"
        else:
            title = "Recorded Information"
        subtitle = (
            "This is the information submitted for your claim."
            if is_success
            else "We have captured the following information for review."
        )
        section_heading = "Incident details"
        next_heading = "Next steps"

    summary: dict[str, Any] = {
        "type": "claim_success" if is_success else ("human_review" if workflow_status == "HUMAN_REVIEW" else "review_required"),
        "title": title,
        "subtitle": subtitle,
        "reference_label": labels["claim_id"] if is_success else labels["reference_id"],
        "reference_value": state.get("claim_id", ""),
        "sections": [
            {
                "heading": section_heading,
                "items": _summary_items(state, language),
            }
        ],
        "next_steps_heading": next_heading,
        "next_steps": _summary_next_steps(state, language),
    }
    document_items = _document_items(state, language)
    if document_items:
        summary["sections"].append(
            {
                "heading": labels["documents"],
                "items": document_items,
            }
        )
    if state.get("next_action"):
        summary["sections"].append(
            {
                "heading": labels["next_action"],
                "items": [
                    {
                        "label": labels["next_action"],
                        "value": _next_action_display(state.get("next_action", ""), language),
                    }
                ],
            }
        )
    return summary


def _response_payload(
    *,
    session_id: str,
    transcript: str,
    state: ClaimState,
    response_text: str,
    audio_url: str | None = None,
    audio_error: str | None = None,
    audio_pending: bool = False,
    latency_trace: dict[str, Any] | None = None,
    success: bool = True,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "success": success,
        "error_type": error_type or "",
        "error_message": error_message or "",
        "session_id": session_id,
        "transcript": transcript,
        "response_text": response_text,
        "audio_url": audio_url,
        "audio_available": bool(audio_url) and not audio_pending,
        "audio_pending": audio_pending,
        "audio_error": audio_error,
        "ui_language": state.get("ui_language", ""),
        "conversation_language": state.get("conversation_language", ""),
        "response_language": state.get("response_language", ""),
        "stt_language": state.get("stt_language", ""),
        "tts_language": state.get("tts_language", ""),
        "latency_trace": latency_trace or {},
        "claim_status": state.get("claim_status") or state.get("workflow_status", ""),
        "claim_id": state.get("claim_id", ""),
        "progress": _progress_from_state(state),
        "next_action": state.get("next_action", ""),
        "required_documents": state.get("required_documents", []),
        "captured_summary": _captured_summary(state),
        "claim_summary": _claim_summary(state),
    }


def _voice_error_type(error: Exception) -> str:
    message = str(error).lower()
    if "no audio" in message or "did not return a transcript" in message:
        return "NO_SPEECH"
    if "connect" in message or "network" in message or "timeout" in message:
        return "NETWORK_FAILURE"
    if "configuration" in message or "required" in message:
        return "SARVAM_API_FAILURE"
    return "STT_FAILURE"


def _run_text_pipeline(
    *,
    session_id: str,
    mobile_number: str,
    customer_text: str,
    requested_language: str = "hi-IN",
    extractor=extract_claim_information,
    response_generator=generate_customer_response,
    latency_trace: dict[str, Any] | None = None,
) -> tuple[ClaimState, str]:
    trace = latency_trace if latency_trace is not None else {}
    trace["transcript_processing_started"] = True
    state = _get_session_state(session_id, mobile_number, customer_text)
    state = _remember_language(
        state,
        customer_text=customer_text,
        requested_language=requested_language,
    )

    if state.get("identity_mismatch"):
        response_text = _customer_response_from_state(state)
        state["response_message"] = response_text
        SESSIONS[session_id] = state
        trace["workflow_ms"] = 0
        trace["response_generation_ms"] = 0
        trace["response_text_finalized_ms"] = 0
        return state, response_text

    if state.get("customer_id") and state.get("identity_confirmed") is not True:
        claimed_name = speaker_claimed_name_from_text(customer_text)
        confirmation = identity_confirmation_from_text(
            customer_text,
            state.get("customer_name"),
        )
        if confirmation is False:
            state_before = deepcopy(state)
            state.update(
                {
                    "policy_status": "",
                    "policy_type": "",
                    "vehicle_name": "",
                    "vehicle_registration": "",
                    "identity_confirmed": False,
                    "identity_mismatch": True,
                    "speaker_claimed_name": claimed_name,
                    "workflow_status": "IDENTITY_MISMATCH",
                    "next_action": "MANUAL_IDENTITY_VERIFICATION",
                }
            )
            response_text = _customer_response_from_state(state)
            state = _finalize_response_state(state, response_text)
            trace["workflow_ms"] = 0
            trace["response_generation_ms"] = 0
            trace["response_text_finalized_ms"] = 0
            _debug_state_trace(
                session_id=session_id,
                customer_text=customer_text,
                raw_extraction=None,
                normalized_extraction=None,
                state_before=state_before,
                state_after=state,
            )
            SESSIONS[session_id] = state
            return state, response_text
        if confirmation is True:
            state["identity_confirmed"] = True
            state["identity_just_confirmed"] = True
            state["speaker_claimed_name"] = claimed_name or state.get("customer_name", "")
        else:
            state_before = deepcopy(state)
            state["speaker_claimed_name"] = claimed_name
            state["workflow_status"] = "CUSTOMER_IDENTIFIED"
            state["identity_confirmation_requested"] = True
            response_text = _customer_response_from_state(state)
            state = _finalize_response_state(state, response_text)
            trace["workflow_ms"] = 0
            trace["response_generation_ms"] = 0
            trace["response_text_finalized_ms"] = 0
            _debug_state_trace(
                session_id=session_id,
                customer_text=customer_text,
                raw_extraction=None,
                normalized_extraction=None,
                state_before=state_before,
                state_after=state,
            )
            SESSIONS[session_id] = state
            return state, response_text

    state_before = deepcopy(state)
    last_requested_field = state.get("last_requested_field") or state.get("next_missing_field", "")
    extraction_started_at = time.perf_counter()
    try:
        raw_extraction = _extract_with_field_context(
            extractor,
            customer_text,
            last_requested_field,
        )
    except SarvamExtractionError as exc:
        logger.warning(
            "Sarvam extraction unavailable; using deterministic extraction fallback. error=%s",
            type(exc).__name__,
        )
        trace["sarvam_extraction_fallback"] = True
        raw_extraction = deterministic_claim_extraction(
            customer_text,
            last_requested_field=last_requested_field,
        )
    trace["sarvam_reasoning_ms"] = _elapsed_ms(extraction_started_at)
    normalization_started_at = time.perf_counter()
    extraction = normalize_claim_extraction(
        customer_text,
        raw_extraction,
        last_requested_field=last_requested_field,
    )
    trace["normalization_ms"] = _elapsed_ms(normalization_started_at)
    merge_started_at = time.perf_counter()
    state = _merge_extraction_into_state(state, extraction, raw_extraction=raw_extraction)
    trace["state_merge_ms"] = _elapsed_ms(merge_started_at)
    workflow_started_at = time.perf_counter()
    result = run_claim_workflow(state)
    trace["workflow_ms"] = _elapsed_ms(workflow_started_at)
    response_started_at = time.perf_counter()
    response_text = _customer_response_with_dynamic_fallback(
        customer_text=customer_text,
        state=result,
        raw_extraction=raw_extraction,
        normalized_extraction=extraction,
        response_generator=response_generator,
    )
    trace["response_generation_ms"] = _elapsed_ms(response_started_at)
    finalize_started_at = time.perf_counter()
    result = _finalize_response_state(result, response_text)
    trace["response_text_finalized_ms"] = _elapsed_ms(finalize_started_at)
    _debug_state_trace(
        session_id=session_id,
        customer_text=customer_text,
        raw_extraction=raw_extraction,
        normalized_extraction=extraction,
        state_before=state_before,
        state_after=result,
    )
    stored_result = deepcopy(result)
    stored_result["identity_just_confirmed"] = False
    SESSIONS[session_id] = stored_result
    return result, response_text


def process_chat_message(
    *,
    message: str,
    mobile_number: str,
    session_id: str | None = None,
    language: str = "hi-IN",
    extractor=extract_claim_information,
    response_generator=generate_customer_response,
    speech_synthesizer=synthesize_speech_to_file,
    latency_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_session_id = session_id or new_session_id()
    latency_trace = latency_trace if latency_trace is not None else {}
    latency_trace.setdefault("turn_type", "text")
    latency_trace["tts_non_blocking"] = speech_synthesizer is synthesize_speech_to_file
    turn_started_at = time.perf_counter()
    try:
        state, response_text = _run_text_pipeline(
            session_id=active_session_id,
            mobile_number=mobile_number,
            customer_text=message,
            requested_language=language,
            extractor=extractor,
            response_generator=response_generator,
            latency_trace=latency_trace,
        )
    except (SarvamConfigurationError, SarvamExtractionError):
        fallback_state = _get_session_state(active_session_id, mobile_number, message)
        fallback_state = _remember_language(
            fallback_state,
            customer_text=message,
            requested_language=language,
        )
        return _response_payload(
            session_id=active_session_id,
            transcript=message,
            state=fallback_state,
            response_text=_localized_template(fallback_state, "generic_error"),
            latency_trace=latency_trace,
        )

    audio_url = None
    audio_error = None
    audio_pending = False
    tts_language = _tts_language_code(
        _response_language_from_state(state),
        language,
    )
    state["tts_language"] = tts_language
    if active_session_id in SESSIONS:
        SESSIONS[active_session_id]["tts_language"] = tts_language
    try:
        if speech_synthesizer is synthesize_speech_to_file:
            audio_url = _queue_tts_generation(
                response_text=response_text,
                language_code=tts_language,
                speech_synthesizer=speech_synthesizer,
                latency_trace=latency_trace,
            )
            audio_pending = True
        else:
            tts_started_at = time.perf_counter()
            audio = speech_synthesizer(
                response_text,
                output_dir=GENERATED_AUDIO_DIR,
                language_code=tts_language,
            )
            latency_trace["bulbul_tts_ms"] = _elapsed_ms(tts_started_at)
            latency_trace["bulbul_tts_blocking"] = True
            audio_url = audio.audio_url
    except (SarvamConfigurationError, SarvamSpeechError, Exception):
        audio_error = "Audio response is unavailable, but you can continue with the text response."
        latency_trace["bulbul_tts_status"] = "failed"

    latency_trace["backend_text_response_total_ms"] = _elapsed_ms(turn_started_at)
    logger.warning("VOICE TURN LATENCY text/chat trace=%s", latency_trace)
    return _response_payload(
        session_id=active_session_id,
        transcript=message,
        state=state,
        response_text=response_text,
        audio_url=audio_url,
        audio_error=audio_error,
        audio_pending=audio_pending,
        latency_trace=latency_trace,
    )


def process_voice_message(
    *,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    mobile_number: str,
    session_id: str | None = None,
    language: str = "hi-IN",
    transcriber=transcribe_audio_bytes,
    extractor=extract_claim_information,
    response_generator=generate_customer_response,
    speech_synthesizer=synthesize_speech_to_file,
) -> dict[str, Any]:
    active_session_id = session_id or new_session_id()
    latency_trace: dict[str, Any] = {
        "turn_type": "voice",
        "backend_audio_received": True,
    }
    voice_started_at = time.perf_counter()
    try:
        stt_started_at = time.perf_counter()
        logger.warning("VOICE TURN LATENCY T2 Saaras STT request started")
        transcription = transcriber(
            audio_bytes,
            filename=filename,
            content_type=content_type,
        )
        latency_trace["saaras_stt_ms"] = _elapsed_ms(stt_started_at)
        logger.warning(
            "VOICE TURN LATENCY T3 Saaras STT completed saaras_stt_ms=%s",
            latency_trace["saaras_stt_ms"],
        )
    except (SarvamConfigurationError, SarvamSpeechError) as error:
        state = _get_session_state(active_session_id, mobile_number, "")
        state = _remember_language(
            state,
            customer_text="",
            requested_language=language,
        )
        error_type = _voice_error_type(error)
        return _response_payload(
            session_id=active_session_id,
            transcript="",
            state=state,
            response_text=_localized_template(state, "voice_error"),
            success=False,
            error_type=error_type,
            error_message=error_type,
            latency_trace=latency_trace,
        )

    latency_trace["transcript_length"] = len(transcription.transcript)
    latency_trace["stt_language"] = transcription.language_code or ""
    logger.warning("VOICE TURN LATENCY T4 transcript processing started")
    payload = process_chat_message(
        message=transcription.transcript,
        mobile_number=mobile_number,
        session_id=active_session_id,
        language=language,
        extractor=extractor,
        response_generator=response_generator,
        speech_synthesizer=speech_synthesizer,
        latency_trace=latency_trace,
    )
    if active_session_id in SESSIONS:
        SESSIONS[active_session_id]["stt_language"] = transcription.language_code or ""
    payload["stt_language"] = transcription.language_code or ""
    payload["transcript"] = transcription.transcript
    payload["detected_language"] = transcription.language_code
    payload["latency_trace"]["voice_backend_total_ms"] = _elapsed_ms(voice_started_at)
    logger.warning("VOICE TURN LATENCY total trace=%s", payload["latency_trace"])
    return payload


def get_customer_claim(claim_id: str) -> dict[str, Any]:
    try:
        docs = get_document_requirements(claim_id)
    except MockBackendError as error:
        raise error
    return {
        "claim_id": claim_id,
        "required_documents": docs["documents"],
        "next_action": docs["next_action"],
    }
