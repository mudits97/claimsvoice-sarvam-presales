from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = PROJECT_ROOT / "frontend" / "index.html"


def test_frontend_contains_required_customer_experience_elements() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "ClaimsVoice" in html
    assert "AI Insurance Assistant" in html
    assert "Live Conversation" in html
    assert "Your Claim Progress" in html
    assert "Customer Identified" in html
    assert "Policy Verified" in html
    assert "Information Captured" in html
    assert "Claim Registered" in html
    assert "Documents & Next Steps" in html
    assert "Type instead" in html
    assert "Start New Claim" in html


def test_frontend_uses_mediarecorder_and_customer_endpoints() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "MediaRecorder" in html
    assert "/api/chat" in html
    assert "/api/voice/process" in html
    assert "/api/session/reset" in html
    assert "/api/claim/" in html


def test_frontend_preserves_browser_recording_format_and_blocks_duplicates() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "selectRecordingMimeType()" in html
    assert "audio/webm;codecs=opus" in html
    assert "state.mediaRecorder.mimeType" in html
    assert "browser_recording.${extension}" in html
    assert "state.isSendingRecording" in html
    assert "duplicate send blocked" in html
    assert "[ClaimsVoice voice]" in html


def test_frontend_plays_and_replays_bulbul_audio() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "payload.audio_available" in html
    assert "playResponseAudio(audioUrl, payload)" in html
    assert "new Audio(audioUrl)" in html
    assert "Play response" in html
    assert "replay-button" in html


def test_frontend_does_not_expose_internal_agent_terms() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    visible_body = html.split("<body>", maxsplit=1)[1]

    assert "LangGraph" not in visible_body
    assert "tool names" not in visible_body
    assert "route_history" not in visible_body
    assert "internal queue" not in visible_body
    assert "priority HIGH" not in visible_body


def test_frontend_renders_backend_summary_states() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert 'summary.type === "claim_success"' in html
    assert 'summary.type === "human_review"' in html
    assert 'summary.type === "review_required"' in html
    assert "summary.reference_label" in html
    assert "summary.reference_value" in html
    assert "renderSummarySections(summary)" in html
    assert "renderSummaryNextSteps(summary)" in html


def test_frontend_renders_live_captured_summary() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert 'id="capturePanel"' in html
    assert "renderCapturedSummary(payload.captured_summary" in html
    assert "capture-row" in html
    assert "captured_so_far" in html or "captured_summary" in html


def test_frontend_hindi_greeting_and_fallbacks_are_localized() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "नमस्ते। मैं आपका बीमा दावा दर्ज करने में मदद कर सकता हूँ।" in html
    assert "Namaste! Main aapki insurance claim register" not in html
    assert "आपके मामले की आगे समीक्षा आवश्यक है" in html
    assert "थोड़ी और जानकारी चाहिए" in html
    assert "मुझे आपका अनुरोध समझने में परेशानी हो रही है" in html
