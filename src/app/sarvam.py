"""Sarvam AI service wrappers for ClaimsVoice."""

from __future__ import annotations

import json
import os
import base64
import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sarvamai import SarvamAI


SARVAM_API_KEY_ENV = "SARVAM_API_KEY"
SARVAM_CHAT_MODEL = "sarvam-105b"
SAARAS_MODEL = "saaras:v3"
BULBUL_MODEL = "bulbul:v3"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_TTL_SECONDS = 60 * 60
SARVAM_HTTP_TIMEOUT_SECONDS = 30.0
logger = logging.getLogger(__name__)
_SARVAM_HTTP_CLIENT: httpx.Client | None = None


def _load_project_env(env_path: Path | None = None) -> None:
    path = env_path or PROJECT_ROOT / ".env"
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_project_env()


def _cleanup_old_audio_files(output_dir: Path) -> None:
    cutoff = time.time() - AUDIO_TTL_SECONDS
    for path in output_dir.glob("claimsvoice-*.mp3"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            logger.debug("Skipping generated audio cleanup for %s", path)


def _get_sarvam_http_client() -> httpx.Client:
    global _SARVAM_HTTP_CLIENT
    if _SARVAM_HTTP_CLIENT is None or _SARVAM_HTTP_CLIENT.is_closed:
        _SARVAM_HTTP_CLIENT = httpx.Client(
            timeout=SARVAM_HTTP_TIMEOUT_SECONDS,
            follow_redirects=True,
            trust_env=False,
        )
    return _SARVAM_HTTP_CLIENT


def _sarvam_error_summary(error: Exception) -> str:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code:
        return f"{type(error).__name__} status={status_code}"
    return type(error).__name__


class SarvamConfigurationError(RuntimeError):
    """Raised when Sarvam is called without the required configuration."""


class SarvamExtractionError(RuntimeError):
    """Raised when a Sarvam response cannot be parsed or validated."""


class SarvamResponseGenerationError(RuntimeError):
    """Raised when Sarvam cannot generate a customer-facing response."""


class SarvamSpeechError(RuntimeError):
    """Raised when Sarvam speech-to-text or text-to-speech fails."""


class ClaimExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str | None = None
    incident_date: str | None = None
    incident_time: str | None = None
    incident_location: str | None = None
    incident_type: str | None = None
    vehicle_damage: str | None = None
    third_party_involved: bool | None = None
    injury_reported: bool | None = None
    vehicle_drivable: bool | None = None


class TranscriptionResult(BaseModel):
    transcript: str
    language_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpeechSynthesisResult(BaseModel):
    audio_path: str
    audio_url: str
    content_type: str = "audio/mpeg"


CLAIM_EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": ["string", "null"]},
        "incident_date": {"type": ["string", "null"]},
        "incident_time": {"type": ["string", "null"]},
        "incident_location": {"type": ["string", "null"]},
        "incident_type": {"type": ["string", "null"]},
        "vehicle_damage": {"type": ["string", "null"]},
        "third_party_involved": {"type": ["boolean", "null"]},
        "injury_reported": {"type": ["boolean", "null"]},
        "vehicle_drivable": {"type": ["boolean", "null"]},
    },
    "required": [
        "intent",
        "incident_date",
        "incident_time",
        "incident_location",
        "incident_type",
        "vehicle_damage",
        "third_party_involved",
        "injury_reported",
        "vehicle_drivable",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You extract motor insurance FNOL claim facts from Indian customer text.

Rules:
- Return only the requested JSON structure.
- Do not invent missing information.
- Use null for information not provided.
- Extract only information explicitly stated or unambiguously supported by the customer's words.
- Do not infer missing business facts simply because they are common in accidents.
- Collision or accident alone does not mean another vehicle or person was involved.
- If a claim fact is not known, return null.
- Preserve uncertainty using the original wording when needed, for example "evening".
- If a relative date such as "kal" or "yesterday" is clear, resolve it using the reference date.
- Do not decide insurance coverage.
- Do not determine claim eligibility.
- Do not approve, reject, create, or escalate claims.
- Do not call tools.
- Normalize obvious vehicle damage phrases into vehicle_damage, even when the customer says there is no major body damage. Examples: tyre damaged, tyre punctured, bumper dent, mirror broken, windshield cracked.
- For collisions described as another person or vehicle hitting the car, use incident_type "collision" and third_party_involved true.
- When a statement contains apparently contradictory clauses, prioritize explicit factual evidence over vague conversational qualifiers. For example, "not really, but I got a broken leg" means injury_reported true.
- For injury_reported, explicit injury evidence such as broken bone, bleeding, cut, pain, hospital, ambulance, unconscious, injury, or hurt means true. Clear denial such as "no one was injured" means false. Ambiguous statements such as "not sure" or "I don't know" mean null.
- For vehicle_drivable, "can still be driven", "running fine", or "I can still drive it" means true. "won't start", "cannot be moved", or "cannot drive" means false. Ambiguous statements such as "not sure if safe to drive" mean null.
"""

FIELD_FOCUS_GUIDANCE = {
    "injury_reported": (
        "The workflow just asked whether anyone was injured. Treat explicit physical injury evidence "
        "such as broken bone, bleeding, cut, pain, hospital, ambulance, unconscious, injury, or hurt as true. "
        "Treat clear denial such as no one was injured as false. Treat uncertainty as null."
    ),
    "vehicle_drivable": (
        "The workflow just asked whether the vehicle can still be driven. Treat can still be driven, "
        "running fine, or I can still drive it as true. Treat won't start, cannot be moved, or cannot drive "
        "as false. Treat uncertainty as null."
    ),
}

CUSTOMER_RESPONSE_SYSTEM_PROMPT = """You write the customer-facing reply for ClaimsVoice, a motor insurance FNOL assistant.

Use the LLM for natural acknowledgement, empathy, concise phrasing, and clarification wording.
Use the provided deterministic workflow state as the only source of allowed business action.

Rules:
- Output only the message the customer should hear or read.
- Keep it concise for voice: 1 to 3 short sentences.
- Use the requested response language.
- Ask at most one next question.
- If missing_fields is not empty, ask only for next_missing_field.
- Do not mention internal workflow codes, graph nodes, tools, rule names, queues, JSON, confidence, or evidence labels.
- Do not say a claim is registered unless claim_status is INITIATED and claim_id is present.
- Do not say a case is approved, rejected, or eligible beyond what the deterministic workflow allows.
- If workflow_status is HUMAN_REVIEW, explain that specialist review is needed and do not say "Claim Registered".
- If the fallback response includes required policy or claim wording, preserve that business meaning.
- When the customer's latest message contains meaningful new information, acknowledge that fact naturally before the next step.
"""


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _contains_roman_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _contains_any_roman_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains_roman_phrase(text, phrase) for phrase in phrases)


def _text_has_roman_word(text: str, word: str) -> bool:
    return _contains_roman_phrase(text, word)


def _has_direct_yes(text: str) -> bool:
    return bool(re.search(r"^\s*(?:yes|yeah|yep|haan|han|ha|ji)\b|^\s*(?:हाँ|हां|जी)", text))


def _has_direct_no(text: str) -> bool:
    return bool(re.search(r"^\s*(?:no|nope|nahi|nahin)\b|^\s*नहीं", text))


def _nearby_negation_before(text: str, start: int, *, window: int = 55) -> bool:
    prefix = text[max(0, start - window):start]
    clause = re.split(
        r"\b(?:but|however|though|except|only|just|bas)\b|लेकिन|पर|बस",
        prefix,
    )[-1]
    return bool(
        re.search(
            r"\b(?:no|not|nahi|nahin|nobody|none|without|never|dont|don't|do not)\b|नहीं",
            clause,
        )
    )


def _has_unnegated_evidence(text: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            if not _nearby_negation_before(text, match.start()):
                return True
    return False


def _is_ambiguous_answer(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:not sure|dont know|don't know|do not know|not certain|unclear|maybe|pata nahi)\b",
            text,
        )
    )


def _infer_incident_date(customer_text: str, reference_date: date) -> str | None:
    text = customer_text.lower()
    if "कल" in customer_text or _text_has_roman_word(text, "kal") or _text_has_roman_word(text, "yesterday"):
        return (reference_date - timedelta(days=1)).isoformat()
    if "आज" in customer_text or _text_has_roman_word(text, "aaj") or _text_has_roman_word(text, "today"):
        return reference_date.isoformat()
    return None


def _incident_date_supported_by_text(customer_text: str) -> bool:
    text = customer_text.lower()
    relative_terms = ("kal", "yesterday", "aaj", "today", "tomorrow", "date")
    month_terms = (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    )
    month_pattern = "|".join(month_terms)
    has_written_date = bool(
        re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        or re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text)
        or re.search(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{month_pattern})\b", text)
        or re.search(rf"\b(?:{month_pattern})\s+\d{{1,2}}(?:st|nd|rd|th)?\b", text)
    )
    return (
        _contains_any_roman_phrase(text, relative_terms)
        or has_written_date
        or _contains_any(customer_text, ("कल", "आज", "तारीख"))
    )


def _infer_incident_time(customer_text: str) -> str | None:
    text = customer_text.lower()
    match = re.search(
        r"\b(?:around|about|at|लगभग)?\s*(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(pm|p\.m\.|am|a\.m\.)(?![a-z])",
        text,
    )
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or "0")
        meridiem = match.group(3).replace(".", "")
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"
    baje_match = re.search(r"\b(1[0-2]|0?[1-9])\s*(?:baje|बजे)\b", text)
    if baje_match:
        hour = int(baje_match.group(1))
        if _contains_any_roman_phrase(text, ("shaam", "evening")) or "शाम" in customer_text:
            if hour != 12:
                hour += 12
        elif (_contains_any_roman_phrase(text, ("raat", "night")) or "रात" in customer_text) and hour <= 5:
            hour += 12
        return f"{hour:02d}:00"
    if ("रात" in customer_text or "raat" in text or "midnight" in text) and re.search(r"(12|१२)\s*(बजे|baje|am|a\.m\.)?", text):
        return "00:00"
    if ("शाम" in customer_text or "shaam" in text or "evening" in text) and re.search(r"\b6\b|छह|६", text):
        return "18:00"
    return None


def _incident_time_supported_by_text(customer_text: str) -> bool:
    text = customer_text.lower()
    return bool(
        re.search(r"\b(?:1[0-2]|0?[1-9])(?::[0-5]\d)?\s*(?:pm|p\.m\.|am|a\.m\.)(?![a-z])", text)
        or re.search(r"\b(?:1[0-9]|2[0-3]|0?[1-9])\s*(?:baje|बजे)\b", text)
    ) or _contains_any_roman_phrase(
        text,
        ("shaam", "evening", "raat", "night", "morning", "afternoon", "time"),
    ) or _contains_any(customer_text, ("शाम", "रात", "सुबह", "दोपहर", "समय"))


def _infer_incident_location(customer_text: str) -> str | None:
    text = customer_text.lower()
    if (_contains_roman_phrase(text, "office") or "ऑफिस" in customer_text) and (
        "बाहर" in customer_text or _contains_roman_phrase(text, "bahar") or _contains_roman_phrase(text, "outside")
    ):
        return "office के बाहर"
    if _contains_roman_phrase(text, "office") and _contains_any_roman_phrase(text, ("near the office", "near office")):
        return "near the office"
    known_places = {
        "andheri": "Andheri",
        "koramangala": "Koramangala",
        "t nagar": "T Nagar",
        "mathura": "Mathura",
        "delhi": "Delhi",
        "lucknow": "Lucknow",
    }
    for marker, display in known_places.items():
        if _contains_roman_phrase(text, marker):
            return display
    matches = re.finditer(
        r"\b(?:in|at|near|around|outside)\s+([A-Za-z][A-Za-z\s]{1,32}?)(?=\s+(?:around|at|on|yesterday|today|tomorrow|near|in)\b|[,.]|$)",
        customer_text,
        flags=re.IGNORECASE,
    )
    for match in matches:
        candidate = match.group(1).strip()
        if candidate.lower() not in {"a collision", "collision", "an accident", "accident"}:
            return candidate
    return None


def _incident_location_supported_by_text(customer_text: str, existing_value: str | None) -> bool:
    text = customer_text.lower()
    existing = (existing_value or "").strip().lower()
    if existing and existing in text:
        return True
    return _contains_any_roman_phrase(
        text,
        ("near", "around", "outside", "inside", "paas", "bahar", "location"),
    ) or _contains_any(customer_text, ("पास", "बाहर", "स्थान", "कहाँ"))


def _canonical_incident_type(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    collision_aliases = {
        "front_collision",
        "front_impact",
        "vehicle_collision",
        "car_collision",
        "road_accident",
        "road_collision",
        "minor_collision",
        "major_collision",
    }
    if normalized in collision_aliases:
        return "collision"
    if "collision" in normalized:
        return "collision"
    if normalized in {"car_accident", "vehicle_accident", "motor_accident"}:
        return "accident"
    return normalized


def _infer_incident_type(customer_text: str, existing_value: str | None) -> str | None:
    canonical = _canonical_incident_type(existing_value)
    text = customer_text.lower()
    has_collision_words = _contains_any(customer_text, ("ठोक", "टक्कर", "सामने से")) or _contains_any_roman_phrase(
        text,
        ("collision", "hit", "front impact"),
    )
    if has_collision_words:
        return "collision"
    if _contains_any_roman_phrase(text, ("accident", "crash")) or _contains_any(customer_text, ("दुर्घटना", "हादसा")):
        return "accident"
    if canonical in {"collision", "accident", "own_vehicle_accident", "third_party_accident"} and (
        _contains_any_roman_phrase(text, ("accident", "crash", "collision", "hit"))
        or _contains_any(customer_text, ("दुर्घटना", "हादसा", "टक्कर", "ठोक"))
    ):
        return canonical
    return None


def _vehicle_damage_supported_by_text(customer_text: str) -> bool:
    text = customer_text.lower()
    hindi_supported = _contains_any(
        customer_text,
        (
            "नुकसान",
            "डेंट",
            "टायर",
            "पंक्चर",
            "टूट",
            "खराब",
            "फट",
            "शीशा",
            "कांच",
            "विंडशील्ड",
        ),
    )
    english_terms = (
        "damage",
        "damaged",
        "dent",
        "dented",
        "bumper",
        "tyre",
        "tire",
        "puncture",
        "punctured",
        "broken",
        "scratch",
        "scratched",
        "cracked",
        "mirror",
        "windshield",
        "windscreen",
    )
    english_supported = any(
        re.search(rf"\b{re.escape(term)}\b", text)
        for term in english_terms
    )
    return hindi_supported or english_supported


def _infer_vehicle_damage(customer_text: str, existing_value: str | None) -> str | None:
    text = customer_text.lower()
    has_existing_value = bool(existing_value and existing_value.strip())
    existing_raw = existing_value or ""
    existing_text = existing_raw.lower()
    existing_mentions_no_damage = has_existing_value and _contains_any(
        existing_text,
        ("no damage", "none", "नुकसान नहीं"),
    )

    combined_text = customer_text + " " + text
    has_tyre_reference = _contains_any_roman_phrase(text, ("tyre", "tire", "puncture")) or _contains_any(
        customer_text,
        ("टायर", "पंक्चर"),
    )
    has_tyre_damage = _contains_any_roman_phrase(
        text,
        ("puncture", "punctured", "damaged", "broken", "burst"),
    ) or _contains_any(customer_text, ("टूट", "टुट", "खराब", "फट"))
    if has_tyre_reference and has_tyre_damage:
        return "एक टायर क्षतिग्रस्त"

    has_bumper_reference = _contains_roman_phrase(text, "bumper") or "बम्पर" in customer_text or "बंपर" in customer_text
    if has_bumper_reference:
        if _contains_any(combined_text, ("dent", "डेंट", "टूट", "damaged", "damage")):
            return "बम्पर पर डेंट"

    has_mirror_reference = _contains_any_roman_phrase(text, ("mirror", "side mirror")) or "शीशा" in customer_text
    if has_mirror_reference:
        if _contains_any(combined_text, ("broken", "टूट", "cracked", "damage", "damaged")):
            return "साइड मिरर क्षतिग्रस्त"

    has_windshield_reference = (
        _contains_any_roman_phrase(text, ("windshield", "windscreen"))
        or "कांच" in customer_text
        or "विंडशील्ड" in customer_text
    )
    if has_windshield_reference:
        if _contains_any(combined_text, ("cracked", "broken", "टूट", "दरार", "damage", "damaged")):
            return "विंडशील्ड क्षतिग्रस्त"

    if _contains_any(combined_text, ("ज्यादा नुकसान नहीं", "no major body damage", "no major damage")):
        return "कोई बड़ा नुकसान नहीं बताया गया"

    if has_existing_value and not existing_mentions_no_damage and _vehicle_damage_supported_by_text(customer_text):
        if _contains_any_roman_phrase(existing_text, ("tyre", "tire", "puncture")) or _contains_any(
            existing_raw,
            ("टायर", "पंक्चर"),
        ):
            return existing_value if has_tyre_reference else None
        if _contains_roman_phrase(existing_text, "bumper") or _contains_any(existing_raw, ("बम्पर", "बंपर")):
            return existing_value if has_bumper_reference else None
        if _contains_any_roman_phrase(existing_text, ("mirror", "side mirror")) or "शीशा" in existing_raw:
            return existing_value if has_mirror_reference else None
        if _contains_any_roman_phrase(existing_text, ("windshield", "windscreen")) or _contains_any(
            existing_raw,
            ("कांच", "विंडशील्ड"),
        ):
            return existing_value if has_windshield_reference else None
        return existing_value
    return None


def _explicit_third_party_from_text(customer_text: str) -> bool | None:
    text = customer_text.lower()
    roman_party_terms = r"\b(?:other|another|third party|person|vehicle|car|bike|dusri|doosri|gaadi|gadi)\b"
    party_terms = rf"(?:{roman_party_terms}|गाड़ी|व्यक्ति|शामिल)"
    negative_patterns = (
        rf"(?:\b(?:no|not|nahi|nahin)\b|नहीं).{{0,40}}{party_terms}",
        rf"{party_terms}.{{0,40}}(?:\b(?:no|not|nahi|nahin)\b|नहीं)",
        r"(only|sirf|सिर्फ).{0,20}(my|meri|मेरी).{0,20}(car|gaadi|gadi|गाड़ी)",
    )
    if any(re.search(pattern, text) for pattern in negative_patterns):
        return False
    if _contains_any_roman_phrase(
        text,
        (
            "divider",
            "median",
            "barrier",
            "pole",
            "tree",
            "wall",
            "footpath",
            "single vehicle",
        ),
    ) or _contains_any(customer_text, ("डिवाइडर", "खंभा", "पेड़", "दीवार", "फुटपाथ")):
        return False

    if _contains_any(customer_text, ("किसी ने", "सामने से", "दूसरी गाड़ी", "बाइक", "व्यक्ति", "आदमी", "बस", "ट्रक", "ऑटो")):
        return True
    if _contains_any_roman_phrase(
        text,
        (
            "bike",
            "scooter",
            "another vehicle",
            "another car",
            "another person",
            "other vehicle",
            "other car",
            "other person",
            "someone hit",
            "hit me",
            "third party",
            "pedestrian",
            "person involved",
            "person hit",
            "bus hit",
            "truck hit",
            "auto hit",
        ),
    ):
        return True
    return None


def _infer_third_party_involved(customer_text: str, existing_value: bool | None) -> bool | None:
    explicit_value = _explicit_third_party_from_text(customer_text)
    if explicit_value is not None:
        return explicit_value
    return None


def _explicit_injury_from_text(customer_text: str, last_requested_field: str | None = None) -> bool | None:
    text = customer_text.lower()
    strong_injury_patterns = (
        r"\b(?:broken|fractured)\s+(?:leg|arm|bone|hand|foot|ankle|shoulder|rib|finger|wrist|knee)\b",
        r"\b(?:leg|arm|hand|foot|ankle|shoulder|rib|finger|wrist|knee|head|neck|back)\s+(?:is\s+)?(?:broken|fractured|hurt|injured|bleeding|paining)\b",
        r"\b(?:bleeding|blood|cut|wound|pain|hospital|ambulance|unconscious|fracture|fractured|sprain|sprained|bruise|bruised)\b",
        r"\b(?:hurt|injured)\s+(?:my|me|his|her|their|passenger|driver|someone|anyone|person)\b",
        r"\b(?:my|his|her|their|passenger|driver)\s+(?:.*\s)?(?:hurt|injured)\b",
    )
    generic_injury_patterns = (
        r"\b(?:injury|injuries|injured|hurt)\b",
    )
    denial_patterns = (
        r"\b(?:no one|nobody|none)\s+(?:was\s+|got\s+)?(?:injured|hurt)\b",
        r"\bno\s+(?:personal\s+)?(?:injury|injuries)\b",
        r"\b(?:nahi|nahin)\s+(?:.*\s)?(?:injury|injured|hurt)\b",
    )
    if _contains_roman_phrase(text, "uninjured"):
        return False
    if _has_unnegated_evidence(text, strong_injury_patterns) or _contains_any(customer_text, ("चोट लगी", "घायल")):
        return True
    if re.search(r"(?:किसी|कोई).{0,20}चोट.{0,20}नहीं|नहीं.{0,20}चोट|चोट.{0,20}नहीं", customer_text):
        return False
    if any(re.search(pattern, text) for pattern in denial_patterns) or re.search(
        r"(?:injury|injured|hurt).{0,30}\b(?:no|not|nahi|nahin)\b",
        text,
    ):
        return False
    if _is_ambiguous_answer(text):
        return None
    if _has_unnegated_evidence(text, generic_injury_patterns):
        return True
    if last_requested_field == "injury_reported":
        if _has_direct_yes(text):
            return True
        if _has_direct_no(text) and not _contains_roman_phrase(text, "not really"):
            return False
    return None


def _infer_injury_reported(
    customer_text: str,
    existing_value: bool | None,
    last_requested_field: str | None = None,
) -> bool | None:
    explicit_value = _explicit_injury_from_text(customer_text, last_requested_field)
    if explicit_value is not None:
        return explicit_value
    if last_requested_field == "injury_reported" and _is_ambiguous_answer(customer_text.lower()):
        return None
    if last_requested_field == "injury_reported" and existing_value is not None:
        return existing_value
    return None


def _explicit_vehicle_drivable_from_text(customer_text: str, last_requested_field: str | None = None) -> bool | None:
    text = customer_text.lower()
    false_patterns = (
        r"\b(?:won't|wont|will not|doesn't|does not|didn't|did not)\s+start\b",
        r"\b(?:cannot|can't|cant|can not)\s+(?:be\s+)?(?:moved|move|driven|drive)\b",
        r"\b(?:not|nahi|nahin)\s+(?:safe\s+to\s+)?(?:drive|drivable|driveable|running|start)\b",
        r"\b(?:not|nahi|nahin)\s+(?:in\s+)?(?:drivable|driveable)\s+condition\b",
    )
    true_patterns = (
        r"\bcan\s+(?:still\s+)?drive\b",
        r"\bcan\s+(?:still\s+)?be\s+driven\b",
        r"\bstill\s+(?:drive|drives|driven|drivable|driveable)\b",
        r"\b(?:car|vehicle|gaadi|gadi)\s+(?:is\s+)?running\s+fine\b",
        r"\b(?:car|vehicle|gaadi|gadi)\s+(?:can\s+)?(?:still\s+)?(?:run|runs|move|moves)\b",
        r"\b(?:drivable|driveable)\b",
        r"\bdrive\s+ho\s+sakti\b",
        r"\bdrive\s+kar\s+sak\b",
    )
    if _contains_roman_phrase(text, "undrivable"):
        return False
    if _is_ambiguous_answer(text):
        return None
    if any(re.search(pattern, text) for pattern in false_patterns):
        return False
    if any(re.search(pattern, text) for pattern in true_patterns) or _contains_any(
        customer_text,
        ("गाड़ी चल रही", "चलने की स्थिति में है"),
    ):
        return True
    if last_requested_field == "vehicle_drivable":
        if _has_direct_yes(text):
            return True
        if _has_direct_no(text):
            return False
    return None


def _infer_vehicle_drivable(
    customer_text: str,
    existing_value: bool | None,
    last_requested_field: str | None = None,
) -> bool | None:
    explicit_value = _explicit_vehicle_drivable_from_text(customer_text, last_requested_field)
    if explicit_value is not None:
        return explicit_value
    if last_requested_field == "vehicle_drivable" and _is_ambiguous_answer(customer_text.lower()):
        return None
    if last_requested_field == "vehicle_drivable" and existing_value is not None:
        return existing_value
    return None


def normalize_claim_extraction(
    customer_text: str,
    extraction: ClaimExtraction,
    *,
    reference_date: date | None = None,
    last_requested_field: str | None = None,
) -> ClaimExtraction:
    """Deterministically clean obvious structured FNOL facts after model extraction."""
    reference = reference_date or date.today()
    payload = extraction.model_dump()

    inferred_date = _infer_incident_date(customer_text, reference)
    payload["incident_date"] = (
        inferred_date
        or (payload["incident_date"] if _incident_date_supported_by_text(customer_text) else None)
    )
    inferred_time = _infer_incident_time(customer_text)
    payload["incident_time"] = (
        inferred_time
        or (payload["incident_time"] if _incident_time_supported_by_text(customer_text) else None)
    )
    inferred_location = _infer_incident_location(customer_text)
    payload["incident_location"] = (
        inferred_location
        or (
            payload["incident_location"]
            if _incident_location_supported_by_text(customer_text, payload["incident_location"])
            else None
        )
    )
    payload["incident_type"] = _infer_incident_type(customer_text, payload["incident_type"])
    payload["vehicle_damage"] = _infer_vehicle_damage(customer_text, payload["vehicle_damage"])
    payload["third_party_involved"] = _infer_third_party_involved(
        customer_text,
        payload["third_party_involved"],
    )
    payload["injury_reported"] = _infer_injury_reported(
        customer_text,
        payload["injury_reported"],
        last_requested_field,
    )
    payload["vehicle_drivable"] = _infer_vehicle_drivable(
        customer_text,
        payload["vehicle_drivable"],
        last_requested_field,
    )

    return ClaimExtraction.model_validate(payload)


def create_sarvam_client(api_key: str | None = None) -> SarvamAI:
    key = api_key or os.environ.get(SARVAM_API_KEY_ENV)
    if not key:
        raise SarvamConfigurationError(
            f"{SARVAM_API_KEY_ENV} is required to call Sarvam AI."
        )
    return SarvamAI(
        api_subscription_key=key,
        httpx_client=_get_sarvam_http_client(),
    )


def build_claim_extraction_messages(
    customer_text: str,
    reference_date: date | None = None,
    last_requested_field: str | None = None,
) -> list[dict[str, str]]:
    demo_date = reference_date or date.today()
    focus_guidance = FIELD_FOCUS_GUIDANCE.get(last_requested_field or "")
    focus_text = f"\n\nField context: {focus_guidance}" if focus_guidance else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Reference date: {demo_date.isoformat()}\n"
                "Extract the structured claim facts from this customer message:\n"
                f"{customer_text}"
                f"{focus_text}"
            ),
        },
    ]


def claim_extraction_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "claimsvoice_claim_extraction",
            "description": "Structured FNOL claim facts extracted from customer text.",
            "strict": True,
            "schema": CLAIM_EXTRACTION_JSON_SCHEMA,
        },
    }


def _message_content_from_response(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise SarvamExtractionError("Sarvam response did not contain message content.") from exc

    if not content:
        raise SarvamExtractionError("Sarvam response content was empty.")
    return content


def extract_claim_information(
    customer_text: str,
    *,
    client: Any | None = None,
    reference_date: date | None = None,
    last_requested_field: str | None = None,
    model: str = SARVAM_CHAT_MODEL,
) -> ClaimExtraction:
    """Call Sarvam-105B and return validated claim fact extraction."""
    sarvam_client = client or create_sarvam_client()
    try:
        response = sarvam_client.chat.completions(
            model=model,
            messages=build_claim_extraction_messages(customer_text, reference_date, last_requested_field),
            temperature=0,
            max_tokens=800,
            reasoning_effort=None,
            request_options={
                "additional_body_parameters": {
                    "response_format": claim_extraction_response_format()
                }
            },
        )
    except Exception as exc:
        logger.warning(
            "Sarvam-105B request failed model=%s error=%s",
            model,
            _sarvam_error_summary(exc),
        )
        raise SarvamExtractionError(
            f"Sarvam-105B request failed: {_sarvam_error_summary(exc)}"
        ) from exc

    raw_content = _message_content_from_response(response)
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise SarvamExtractionError("Sarvam response was not valid JSON.") from exc

    try:
        extraction = ClaimExtraction.model_validate(payload)
    except ValueError as exc:
        raise SarvamExtractionError("Sarvam response did not match the extraction schema.") from exc

    return normalize_claim_extraction(
        customer_text,
        extraction,
        reference_date=reference_date,
        last_requested_field=last_requested_field,
    )


def _supported_extraction_facts(extraction: ClaimExtraction | None) -> dict[str, Any]:
    if extraction is None:
        return {}
    return {
        key: value
        for key, value in extraction.model_dump().items()
        if value not in ("", None, [])
    }


def _customer_response_state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = (
        "customer_name",
        "policy_id",
        "policy_status",
        "policy_type",
        "vehicle_name",
        "vehicle_registration",
        "incident_date",
        "incident_time",
        "incident_location",
        "incident_type",
        "vehicle_damage",
        "third_party_involved",
        "injury_reported",
        "vehicle_drivable",
        "workflow_status",
        "claim_status",
        "claim_id",
        "missing_fields",
        "next_missing_field",
        "last_requested_field",
        "next_action",
    )
    snapshot = {
        key: state.get(key)
        for key in allowed_keys
        if state.get(key) not in ("", None, [])
    }
    documents = state.get("required_documents") or []
    if documents:
        snapshot["required_documents"] = [
            document.get("name", "")
            for document in documents
            if document.get("name")
        ]
    return snapshot


def build_customer_response_messages(
    *,
    customer_text: str,
    state: dict[str, Any],
    raw_extraction: ClaimExtraction | None,
    normalized_extraction: ClaimExtraction | None,
    fallback_response: str,
    response_language: str,
) -> list[dict[str, str]]:
    """Build a constrained prompt for natural customer-facing response generation."""
    payload = {
        "latest_customer_message": customer_text,
        "response_language": "Hindi" if response_language == "hi" else "English",
        "workflow_state": _customer_response_state_snapshot(state),
        "facts_extracted_this_turn": _supported_extraction_facts(normalized_extraction),
        "raw_model_extraction": _supported_extraction_facts(raw_extraction),
        "deterministic_fallback_response": fallback_response,
    }
    return [
        {"role": "system", "content": CUSTOMER_RESPONSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def generate_customer_response(
    *,
    customer_text: str,
    state: dict[str, Any],
    raw_extraction: ClaimExtraction | None = None,
    normalized_extraction: ClaimExtraction | None = None,
    fallback_response: str,
    response_language: str = "en",
    client: Any | None = None,
    model: str = SARVAM_CHAT_MODEL,
) -> str:
    """Use Sarvam-105B to phrase the final customer-facing response."""
    sarvam_client = client or create_sarvam_client()
    try:
        response = sarvam_client.chat.completions(
            model=model,
            messages=build_customer_response_messages(
                customer_text=customer_text,
                state=state,
                raw_extraction=raw_extraction,
                normalized_extraction=normalized_extraction,
                fallback_response=fallback_response,
                response_language=response_language,
            ),
            temperature=0.35,
            max_tokens=220,
            reasoning_effort=None,
        )
    except Exception as exc:
        logger.warning(
            "Sarvam response generation failed model=%s error=%s",
            model,
            _sarvam_error_summary(exc),
        )
        raise SarvamResponseGenerationError(
            f"Sarvam response generation failed: {_sarvam_error_summary(exc)}"
        ) from exc

    text = _message_content_from_response(response).strip().strip('"')
    if not text:
        raise SarvamResponseGenerationError("Sarvam response generation returned empty text.")
    return text


def _content_type_to_audio_codec(content_type: str | None) -> str | None:
    if not content_type:
        return None
    if "webm" in content_type:
        return "webm"
    if "wav" in content_type:
        return "wav"
    if "mpeg" in content_type or "mp3" in content_type:
        return "mp3"
    if "ogg" in content_type or "opus" in content_type:
        return "opus"
    if "mp4" in content_type or "m4a" in content_type:
        return "mp4"
    return None


def _content_type_for_sarvam_upload(content_type: str | None, codec: str | None) -> str:
    if codec == "webm":
        return "audio/webm"
    if codec == "wav":
        return "audio/wav"
    if codec == "mp3":
        return "audio/mpeg"
    if codec == "opus":
        return "audio/ogg"
    if codec == "mp4":
        return "audio/mp4"
    return (content_type or "application/octet-stream").split(";", 1)[0].strip()


def transcribe_audio_bytes(
    audio_bytes: bytes,
    *,
    filename: str = "recording.webm",
    content_type: str = "audio/webm",
    client: Any | None = None,
) -> TranscriptionResult:
    """Transcribe a short browser recording with Saaras v3."""
    if not audio_bytes:
        raise SarvamSpeechError("No audio was provided for transcription.")

    sarvam_client = client or create_sarvam_client()
    codec = _content_type_to_audio_codec(content_type)
    upload_content_type = _content_type_for_sarvam_upload(content_type, codec)
    file_payload = (filename, audio_bytes, upload_content_type)
    started_at = time.perf_counter()

    kwargs: dict[str, Any] = {
        "file": file_payload,
        "model": SAARAS_MODEL,
        "mode": "codemix",
        "language_code": "unknown",
    }
    if codec:
        kwargs["input_audio_codec"] = codec

    logger.warning(
        "STT request started model=%s filename=%s content_type=%s upload_content_type=%s bytes=%s codec=%s",
        SAARAS_MODEL,
        filename,
        content_type,
        upload_content_type,
        len(audio_bytes),
        codec or "",
    )
    try:
        response = sarvam_client.speech_to_text.transcribe(**kwargs)
    except Exception as exc:
        logger.warning(
            "STT request failed model=%s error=%s",
            SAARAS_MODEL,
            _sarvam_error_summary(exc),
        )
        raise SarvamSpeechError(
            f"Saaras transcription failed: {_sarvam_error_summary(exc)}"
        ) from exc

    transcript = getattr(response, "transcript", None)
    if not transcript:
        raise SarvamSpeechError("Saaras did not return a transcript.")

    language_code = (
        getattr(response, "language_code", None)
        or getattr(response, "detected_language", None)
    )
    metadata = {}
    if hasattr(response, "model_dump"):
        metadata = response.model_dump(exclude={"transcript"}, exclude_none=True)

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    logger.warning(
        "STT request completed model=%s language=%s transcript_length=%s duration_ms=%s",
        SAARAS_MODEL,
        language_code or "",
        len(transcript),
        duration_ms,
    )

    return TranscriptionResult(
        transcript=transcript,
        language_code=language_code,
        metadata=metadata,
    )


def synthesize_speech_to_file(
    text: str,
    *,
    output_dir: Path,
    public_url_prefix: str = "/media/audio",
    client: Any | None = None,
    language_code: str = "hi-IN",
    speaker: str = "shubh",
) -> SpeechSynthesisResult:
    """Generate a Bulbul v3 audio file for the final customer response."""
    if not text.strip():
        raise SarvamSpeechError("No text was provided for speech synthesis.")

    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_audio_files(output_dir)
    sarvam_client = client or create_sarvam_client()
    started_at = time.perf_counter()
    logger.warning(
        "TTS request started model=%s language=%s output_codec=mp3",
        BULBUL_MODEL,
        language_code,
    )
    try:
        response = sarvam_client.text_to_speech.convert(
            text=text[:2500],
            language_code=language_code,
            speaker=speaker,
            model=BULBUL_MODEL,
            output_audio_codec="mp3",
            speech_sample_rate=24000,
            pace=1.0,
        )
    except Exception:
        logger.warning(
            "TTS request failed model=%s language=%s",
            BULBUL_MODEL,
            language_code,
            exc_info=False,
        )
        raise

    audios = getattr(response, "audios", None)
    if not audios and isinstance(response, dict):
        audios = response.get("audios")
    if not audios:
        raise SarvamSpeechError("Bulbul did not return audio data.")

    audio_payload = audios[0]
    if isinstance(audio_payload, str):
        audio_bytes = base64.b64decode(audio_payload)
    else:
        audio_bytes = bytes(audio_payload)

    filename = f"claimsvoice-{uuid4().hex}.mp3"
    output_path = output_dir / filename
    output_path.write_bytes(audio_bytes)
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    logger.warning(
        "TTS request completed audio_path=%s duration_ms=%s",
        output_path,
        duration_ms,
    )

    return SpeechSynthesisResult(
        audio_path=str(output_path),
        audio_url=f"{public_url_prefix}/{filename}",
    )
