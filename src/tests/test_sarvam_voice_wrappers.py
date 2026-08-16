from __future__ import annotations

import base64
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app import sarvam as sarvam_module
from app.sarvam import (
    SarvamSpeechError,
    create_sarvam_client,
    synthesize_speech_to_file,
    transcribe_audio_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_AUDIO_DIR = PROJECT_ROOT / "tests" / "runtime_audio"


class FakeSpeechToText:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def transcribe(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(
            transcript="Meri car ka accident hua hai.",
            language_code="hi-IN",
            model_dump=lambda **_kwargs: {"language_code": "hi-IN"},
        )


class FakeTextToSpeech:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def convert(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(audios=[base64.b64encode(b"fake-mp3").decode("ascii")])


class FakeSarvamClient:
    def __init__(self) -> None:
        self.speech_to_text = FakeSpeechToText()
        self.text_to_speech = FakeTextToSpeech()


class FailingSpeechToText:
    def transcribe(self, **_kwargs: Any) -> Any:
        raise RuntimeError("connection refused through local proxy")


class FailingSpeechClient:
    def __init__(self) -> None:
        self.speech_to_text = FailingSpeechToText()


@pytest.fixture()
def audio_dir() -> Path:
    if RUNTIME_AUDIO_DIR.exists():
        shutil.rmtree(RUNTIME_AUDIO_DIR)
    RUNTIME_AUDIO_DIR.mkdir(parents=True)
    yield RUNTIME_AUDIO_DIR
    if RUNTIME_AUDIO_DIR.exists():
        shutil.rmtree(RUNTIME_AUDIO_DIR)


def test_saaras_wrapper_transcribes_browser_audio() -> None:
    client = FakeSarvamClient()

    result = transcribe_audio_bytes(
        b"webm-bytes",
        filename="recording.webm",
        content_type="audio/webm;codecs=opus",
        client=client,
    )

    assert result.transcript == "Meri car ka accident hua hai."
    assert result.language_code == "hi-IN"
    assert client.speech_to_text.kwargs["model"] == "saaras:v3"
    assert client.speech_to_text.kwargs["mode"] == "codemix"
    assert client.speech_to_text.kwargs["input_audio_codec"] == "webm"
    assert client.speech_to_text.kwargs["file"][2] == "audio/webm"


def test_sarvam_client_ignores_proxy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeSarvamAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    monkeypatch.setattr(sarvam_module, "SarvamAI", FakeSarvamAI)
    monkeypatch.setattr(sarvam_module, "_SARVAM_HTTP_CLIENT", None)

    create_sarvam_client()

    httpx_client = captured["httpx_client"]
    try:
        assert getattr(httpx_client, "_trust_env") is False
    finally:
        httpx_client.close()


def test_saaras_wrapper_returns_sanitized_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="app.sarvam")

    with pytest.raises(SarvamSpeechError, match="Saaras transcription failed"):
        transcribe_audio_bytes(
            b"webm-bytes",
            filename="browser_recording.webm",
            content_type="audio/webm;codecs=opus",
            client=FailingSpeechClient(),
        )

    assert "STT request failed" in caplog.text
    assert "SARVAM_API_KEY" not in caplog.text
    assert "test-key" not in caplog.text


def test_bulbul_wrapper_writes_audio_file(audio_dir: Path) -> None:
    client = FakeSarvamClient()

    result = synthesize_speech_to_file(
        "Aapka claim register ho gaya hai.",
        output_dir=audio_dir,
        client=client,
        language_code="hi-IN",
    )

    output_path = Path(result.audio_path)
    assert output_path.exists()
    assert output_path.read_bytes() == b"fake-mp3"
    assert result.audio_url.startswith("/media/audio/")
    assert client.text_to_speech.kwargs["model"] == "bulbul:v3"
    assert client.text_to_speech.kwargs["language_code"] == "hi-IN"
    assert client.text_to_speech.kwargs["speaker"] == "shubh"


def test_bulbul_wrapper_logs_safe_lifecycle(
    audio_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeSarvamClient()
    caplog.set_level(logging.INFO, logger="app.sarvam")

    result = synthesize_speech_to_file(
        "Your claim has been registered.",
        output_dir=audio_dir,
        client=client,
        language_code="en-IN",
    )

    assert Path(result.audio_path).exists()
    assert "TTS request started" in caplog.text
    assert "TTS request completed" in caplog.text
    assert result.audio_path in caplog.text
    assert "SARVAM_API_KEY" not in caplog.text
