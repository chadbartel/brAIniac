"""
brainiac/servers/voice-server/src/voice_server/server.py

FastMCP voice server exposing:
  - transcribe_audio : faster-whisper STT (file path or base64 bytes)
  - synthesise_speech : TTS via Kokoro or Piper (returns base64 WAV)

NOTE: This is a scaffold. Install your preferred TTS backend by
      un-commenting the relevant section in pyproject.toml and below.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path

from fastmcp import FastMCP
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voice-server")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# ---------------------------------------------------------------------------
# Lazy model loaders
# ---------------------------------------------------------------------------
_whisper_model: WhisperModel | None = None


def _get_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        logger.info(
            "Loading Whisper model '%s' on %s (%s)",
            WHISPER_MODEL_SIZE,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------
mcp: FastMCP = FastMCP(
    name="brainiac-voice-server",
    version="0.1.0",
    description="STT (faster-whisper) and TTS (Kokoro/Piper) MCP server.",
)


class TranscriptionResult(BaseModel):
    text: str
    language: str
    segments: list[dict[str, object]] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded WAV audio bytes.")
    sample_rate: int


@mcp.tool()
def transcribe_audio(
    audio_b64: str = Field(
        ...,
        description="Base64-encoded audio file bytes (WAV, MP3, FLAC, etc.).",
    ),
    language: str | None = Field(
        None,
        description="BCP-47 language code hint (e.g. 'en'). Auto-detect if omitted.",
    ),
) -> TranscriptionResult:
    """Transcribe audio to text using faster-whisper.

    Args:
        audio_b64: Base64-encoded audio file bytes.
        language: Optional language hint for faster detection.

    Returns:
        TranscriptionResult with full text, detected language, and segments.
    """
    logger.info("[transcribe_audio] bytes=%d", len(audio_b64))
    raw_bytes = base64.b64decode(audio_b64)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        model = _get_whisper()
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
        )
        segment_list = [
            {"start": s.start, "end": s.end, "text": s.text} for s in segments
        ]
        full_text = " ".join(s["text"] for s in segment_list).strip()
        logger.info(
            "[transcribe_audio] language=%s text_len=%d",
            info.language,
            len(full_text),
        )
        return TranscriptionResult(
            text=full_text,
            language=info.language,
            segments=segment_list,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@mcp.tool()
def synthesise_speech(
    text: str = Field(..., description="Text to synthesise."),
    voice: str = Field(
        "default",
        description="Voice identifier (model/speaker dependent).",
    ),
) -> SynthesisResult:
    """Convert text to speech and return base64-encoded WAV audio.

    Args:
        text: The text to synthesise.
        voice: Voice/speaker selection string.

    Returns:
        SynthesisResult with base64 WAV bytes and sample rate.

    Raises:
        NotImplementedError: Until a TTS backend is wired up.
    """
    logger.info("[synthesise_speech] voice=%s chars=%d", voice, len(text))
    # TODO: wire in Kokoro or Piper TTS here.
    # Example Kokoro integration:
    #   from kokoro import KPipeline
    #   pipeline = KPipeline(lang_code="a")
    #   generator = pipeline(text, voice=voice)
    #   for _, _, audio in generator:
    #       audio_bytes = (audio.numpy() * 32767).astype("int16").tobytes()
    #       ...
    raise NotImplementedError(
        "TTS backend not configured. "
        "Install kokoro or piper-tts and uncomment the implementation."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Launch the voice server."""
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8200"))
    logger.info("Starting brainiac-voice-server on %s:%d", host, port)
    mcp.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    run()
