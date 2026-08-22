"""
Speech-to-text client.

The task requires picking ONE of Sarvam or ElevenLabs. We default to
ElevenLabs (Scribe v2) but keep a Sarvam implementation behind the same
interface so switching providers is a one-line config change
(`STT_PROVIDER=sarvam`) rather than a rewrite.

Note: the S.W.A.N frontend (index.html/script.js) only *captures* the mic
stream in-browser via getUserMedia — it does not transcribe. The recorded
blob is expected to be POSTed to `/api/v1/voice-query` as multipart/form-data,
which is where this module gets invoked server-side.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.logger import logger

settings = get_settings()


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    provider: str
    raw: dict | None = None


class BaseTranscriber(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> TranscriptionResult:
        ...


class ElevenLabsTranscriber(BaseTranscriber):
    """Wraps ElevenLabs Scribe (batch Speech-to-Text)."""

    def __init__(self, api_key: str | None = None, model_id: str | None = None):
        self.api_key = api_key or settings.ELEVENLABS_API_KEY
        self.model_id = model_id or settings.ELEVENLABS_STT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            from elevenlabs.client import ElevenLabs

            self._client = ElevenLabs(api_key=self.api_key)
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.2, min=0.2, max=2))
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> TranscriptionResult:
        client = self._get_client()
        audio_stream = io.BytesIO(audio_bytes)
        audio_stream.name = filename  # some SDK versions use this for content-type sniffing

        transcription = client.speech_to_text.convert(
            file=audio_stream,
            model_id=self.model_id,
            tag_audio_events=False,
            language_code=settings.STT_LANGUAGE_CODE,
            diarize=False,
        )

        text = getattr(transcription, "text", None) or ""
        language = getattr(transcription, "language_code", None)
        return TranscriptionResult(
            text=text.strip(),
            language=language,
            provider="elevenlabs",
            raw=getattr(transcription, "__dict__", None),
        )


class SarvamTranscriber(BaseTranscriber):
    """Wraps Sarvam AI's Speech-to-Text REST API (kept as a drop-in alternative)."""

    ENDPOINT = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.model = model or settings.SARVAM_STT_MODEL

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.2, min=0.2, max=2))
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> TranscriptionResult:
        headers = {"api-subscription-key": self.api_key}
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"model": self.model}
        if settings.STT_LANGUAGE_CODE:
            data["language_code"] = settings.STT_LANGUAGE_CODE

        resp = requests.post(self.ENDPOINT, headers=headers, files=files, data=data, timeout=10)
        resp.raise_for_status()
        payload = resp.json()

        return TranscriptionResult(
            text=(payload.get("transcript") or "").strip(),
            language=payload.get("language_code"),
            provider="sarvam",
            raw=payload,
        )


def get_transcriber() -> BaseTranscriber:
    if settings.STT_PROVIDER == "sarvam":
        return SarvamTranscriber()
    return ElevenLabsTranscriber()


class TranscriptionError(Exception):
    pass


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> TranscriptionResult:
    """Entry point used by the API layer. Wraps provider errors uniformly."""
    transcriber = get_transcriber()
    try:
        result = transcriber.transcribe(audio_bytes, filename=filename)
    except Exception as exc:  # noqa: BLE001 - we deliberately normalize all provider errors
        logger.error(f'"STT failed via {settings.STT_PROVIDER}: {exc}"')
        raise TranscriptionError(str(exc)) from exc

    if not result.text:
        raise TranscriptionError("Empty transcript returned by STT provider.")
    return result
