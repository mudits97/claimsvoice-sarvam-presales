"""FastAPI entry point for the ClaimsVoice mock insurance backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.customer_experience import (
    _elapsed_ms,
    get_customer_claim,
    process_chat_message,
    process_voice_message,
    render_session_view,
    reset_session,
)
from app.sarvam import SarvamConfigurationError, SarvamSpeechError, transcribe_audio_bytes
from app.tools import (
    MockBackendError,
    check_coverage,
    create_claim,
    escalate_claim,
    get_customer,
    get_document_requirements,
    get_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR = PROJECT_ROOT / "static"
GENERATED_AUDIO_DIR = STATIC_DIR / "generated" / "audio"
DEBUG_AUDIO_DIR = PROJECT_ROOT / "debug_audio"
GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ClaimsVoice Mock Insurance API",
    description="JSON-backed mock APIs for the ClaimsVoice FNOL proof of concept.",
    version="0.2.0",
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/media", StaticFiles(directory=STATIC_DIR / "generated"), name="media")


class CoverageCheckRequest(BaseModel):
    policy_id: str = Field(..., examples=["POL10001"])
    incident_type: str = Field(..., examples=["collision"])


class IncidentDetails(BaseModel):
    date: str | None = Field(None, examples=["2026-08-10"])
    time: str | None = Field(None, examples=["19:00"])
    location: str | None = Field(None, examples=["Andheri West, Mumbai"])
    incident_type: str | None = Field(None, examples=["collision"])
    vehicle_damage: str | None = Field(None, examples=["Front bumper damaged"])
    third_party_involved: bool | None = Field(None, examples=[True])
    injury_reported: bool | None = Field(None, examples=[False])
    vehicle_drivable: bool | None = Field(None, examples=[True])


class ClaimCreateRequest(BaseModel):
    customer_id: str = Field(..., examples=["CUS10001"])
    policy_id: str = Field(..., examples=["POL10001"])
    incident: IncidentDetails


class EscalationRequest(BaseModel):
    reason: str = Field(..., examples=["Third-party injury reported"])


class ChatRequest(BaseModel):
    message: str = Field(..., examples=["Meri car ka accident kal shaam ko Andheri mein hua tha."])
    mobile_number: str = Field(..., examples=["9876543210"])
    session_id: str | None = Field(None, examples=["demo-session"])
    language: str = Field("hi-IN", examples=["hi-IN"])


class SessionResetRequest(BaseModel):
    session_id: str | None = None


class SessionViewRequest(BaseModel):
    session_id: str
    language: str = Field("hi-IN", examples=["hi-IN"])


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _api_response(action):
    try:
        return action()
    except MockBackendError as error:
        raise HTTPException(status_code=error.status_code, detail=error.to_detail()) from error


def _audio_extension(content_type: str | None, filename: str | None) -> str:
    lowered = (content_type or "").lower()
    if "webm" in lowered:
        return ".webm"
    if "wav" in lowered:
        return ".wav"
    if "mpeg" in lowered or "mp3" in lowered:
        return ".mp3"
    if "ogg" in lowered or "opus" in lowered:
        return ".ogg"
    if "mp4" in lowered or "m4a" in lowered:
        return ".m4a"

    suffix = Path(filename or "").suffix.lower()
    if suffix in {".webm", ".wav", ".mp3", ".ogg", ".m4a", ".mp4"}:
        return suffix
    return ".bin"


def _save_single_debug_audio(
    audio_bytes: bytes,
    *,
    filename: str | None,
    content_type: str | None,
) -> Path | None:
    if not audio_bytes:
        return None

    DEBUG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(DEBUG_AUDIO_DIR.glob("browser_test.*"))
    if existing:
        return existing[0]

    path = DEBUG_AUDIO_DIR / f"browser_test{_audio_extension(content_type, filename)}"
    path.write_bytes(audio_bytes)
    return path


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def frontend_home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        return process_chat_message(
            message=request.message,
            mobile_number=request.mobile_number,
            session_id=request.session_id,
            language=request.language,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CHAT_PROCESSING_FAILED",
                "message": "I am having trouble processing your request. Please try again.",
            },
        ) from error


@app.post("/api/voice/process")
async def voice_process(
    audio: UploadFile = File(...),
    mobile_number: str = Form(...),
    session_id: str | None = Form(None),
    language: str = Form("hi-IN"),
) -> dict[str, Any]:
    try:
        audio_bytes = await audio.read()
        debug_path = _save_single_debug_audio(
            audio_bytes,
            filename=audio.filename,
            content_type=audio.content_type,
        )
        logger.warning(
            "Voice endpoint hit content_type=%s filename=%s bytes=%s non_empty=%s debug_audio=%s",
            audio.content_type or "",
            audio.filename or "",
            len(audio_bytes),
            bool(audio_bytes),
            str(debug_path) if debug_path else "",
        )
        return process_voice_message(
            audio_bytes=audio_bytes,
            filename=audio.filename or "recording.webm",
            content_type=audio.content_type or "audio/webm",
            mobile_number=mobile_number,
            session_id=session_id,
            language=language,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "VOICE_PROCESSING_FAILED",
                "message": "I am having trouble processing your voice. You can also type your response.",
            },
        ) from error


@app.post("/api/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
    mobile_number: str = Form(...),
    session_id: str | None = Form(None),
    language: str = Form("hi-IN"),
) -> dict[str, Any]:
    import time

    request_started_at = time.perf_counter()
    try:
        audio_bytes = await audio.read()
        debug_path = _save_single_debug_audio(
            audio_bytes,
            filename=audio.filename,
            content_type=audio.content_type,
        )
        logger.warning(
            "VOICE TURN LATENCY T1 audio received by backend endpoint=/api/voice/transcribe "
            "content_type=%s filename=%s bytes=%s debug_audio=%s",
            audio.content_type or "",
            audio.filename or "",
            len(audio_bytes),
            str(debug_path) if debug_path else "",
        )
        stt_started_at = time.perf_counter()
        logger.warning("VOICE TURN LATENCY T2 Saaras STT request started")
        transcription = transcribe_audio_bytes(
            audio_bytes,
            filename=audio.filename or "recording.webm",
            content_type=audio.content_type or "audio/webm",
        )
        stt_ms = _elapsed_ms(stt_started_at)
        total_ms = _elapsed_ms(request_started_at)
        logger.warning(
            "VOICE TURN LATENCY T3 Saaras STT completed stt_ms=%s total_transcribe_ms=%s",
            stt_ms,
            total_ms,
        )
        return {
            "success": True,
            "session_id": session_id,
            "transcript": transcription.transcript,
            "detected_language": transcription.language_code,
            "selected_language": language,
            "latency_trace": {
                "turn_type": "voice_transcribe",
                "audio_bytes": len(audio_bytes),
                "saaras_stt_ms": stt_ms,
                "transcribe_backend_total_ms": total_ms,
            },
        }
    except (SarvamConfigurationError, SarvamSpeechError) as error:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "VOICE_TRANSCRIPTION_FAILED",
                "message": "I am having trouble processing your voice. You can also type your response.",
                "error_type": type(error).__name__,
            },
        ) from error


@app.post("/api/session/reset")
def session_reset(request: SessionResetRequest) -> dict[str, str]:
    session_id = reset_session(request.session_id)
    return {"session_id": session_id, "status": "reset"}


@app.post("/api/session/view")
def session_view(request: SessionViewRequest) -> dict[str, Any]:
    return render_session_view(request.session_id, request.language)


@app.get("/api/claim/{claim_id}")
def claim_lookup(claim_id: str) -> dict[str, Any]:
    return _api_response(lambda: get_customer_claim(claim_id))


@app.get("/api/customer/{mobile_number}")
def customer_lookup(mobile_number: str) -> dict[str, Any]:
    return _api_response(lambda: get_customer(mobile_number))


@app.get("/api/policy/{customer_id}")
def policy_lookup(customer_id: str) -> dict[str, Any]:
    return _api_response(lambda: get_policy(customer_id))


@app.post("/api/coverage/check")
def coverage_check(request: CoverageCheckRequest) -> dict[str, Any]:
    return _api_response(lambda: check_coverage(request.policy_id, request.incident_type))


@app.post("/api/claims")
def claim_create(request: ClaimCreateRequest) -> dict[str, Any]:
    return _api_response(lambda: create_claim(_model_to_dict(request)))


@app.get("/api/claims/{claim_id}/documents")
def claim_documents(claim_id: str) -> dict[str, Any]:
    return _api_response(lambda: get_document_requirements(claim_id))


@app.post("/api/claims/{claim_id}/escalate")
def claim_escalate(claim_id: str, request: EscalationRequest) -> dict[str, Any]:
    return _api_response(lambda: escalate_claim(claim_id, request.reason))


def load_json(filename: str) -> dict:
    """Load one of the mock data files from the local data folder."""
    with (DATA_DIR / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    customers = load_json("customers.json")["customers"]
    policies = load_json("policies.json")["policies"]
    claims = load_json("claims.json")["claims"]

    print("ClaimsVoice mock backend data check")
    print(f"Customers: {len(customers)}")
    print(f"Policies: {len(policies)}")
    print(f"Claims: {len(claims)}")


if __name__ == "__main__":
    main()
