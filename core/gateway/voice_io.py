"""
MYCONEX Voice I/O
------------------
Local speech-to-text (Whisper) and text-to-speech (Piper/pyttsx3) support.

Works in two modes:
  1. Discord voice message attachments — `.ogg`/`.mp3`/`.wav` files attached to
     a message are transcribed and injected as text into the chat pipeline.
  2. TTS reply — optionally synthesises responses as audio files and attaches
     them to the Discord reply (when VOICE_TTS_ENABLED=true).

Env vars:
  VOICE_STT_MODEL      — Whisper model size: tiny/base/small/medium/large (default: base)
  VOICE_STT_LANGUAGE   — ISO-639-1 language hint, e.g. "en" (default: auto-detect)
  VOICE_TTS_ENABLED    — "true" to enable TTS replies (default: false)
  VOICE_TTS_ENGINE     — "piper" | "pyttsx3" (default: pyttsx3)
  VOICE_PIPER_MODEL    — Path to Piper .onnx model file (required when engine=piper)
  VOICE_PIPER_CONFIG   — Path to Piper model .json config (required when engine=piper)
  VOICE_MAX_AUDIO_MB   — Max audio size to process in MB (default: 25)
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".ogg", ".mp3", ".wav", ".m4a", ".flac", ".opus", ".webm"}
_STT_MODEL   = os.getenv("VOICE_STT_MODEL", "base")
_STT_LANG    = os.getenv("VOICE_STT_LANGUAGE") or None
_TTS_ENABLED = os.getenv("VOICE_TTS_ENABLED", "false").lower() == "true"
_TTS_ENGINE  = os.getenv("VOICE_TTS_ENGINE", "pyttsx3")
_PIPER_MODEL = os.getenv("VOICE_PIPER_MODEL", "")
_PIPER_CFG   = os.getenv("VOICE_PIPER_CONFIG", "")
_MAX_MB      = float(os.getenv("VOICE_MAX_AUDIO_MB", "25"))

# ── Lazy Whisper loader ────────────────────────────────────────────────────────

_whisper_model = None
_whisper_lock  = asyncio.Lock()


async def _get_whisper():
    global _whisper_model
    async with _whisper_lock:
        if _whisper_model is None:
            try:
                import whisper  # openai-whisper
                logger.info("[voice-stt] loading Whisper model '%s'…", _STT_MODEL)
                _whisper_model = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: whisper.load_model(_STT_MODEL)
                )
                logger.info("[voice-stt] Whisper ready")
            except ImportError:
                logger.warning("[voice-stt] openai-whisper not installed — STT unavailable")
                return None
    return _whisper_model


# ── STT ───────────────────────────────────────────────────────────────────────

def is_audio_url(url: str) -> bool:
    """Return True if the URL looks like an audio attachment."""
    path = url.split("?")[0].lower()
    return any(path.endswith(ext) for ext in _AUDIO_EXTS)


async def transcribe_url(url: str, http_session=None) -> Optional[str]:
    """
    Download audio from *url* and transcribe it with Whisper.
    Returns the transcribed text, or None on failure.
    Requires openai-whisper + ffmpeg installed.
    """
    import urllib.request

    model = await _get_whisper()
    if model is None:
        return None

    # Size check via HEAD
    try:
        with urllib.request.urlopen(url) as resp:
            cl = int(resp.headers.get("Content-Length", 0))
        if cl and cl > _MAX_MB * 1024 * 1024:
            logger.warning("[voice-stt] audio too large (%dMB), skipping", cl // (1024 * 1024))
            return None
    except Exception:
        pass  # proceed anyway

    try:
        # Download
        def _download() -> bytes:
            with urllib.request.urlopen(url) as r:
                return r.read()

        data = await asyncio.get_event_loop().run_in_executor(None, _download)

        # Write to temp file (Whisper needs a path)
        suffix = Path(url.split("?")[0]).suffix or ".ogg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(data)
            tmp_path = tf.name

        # Transcribe
        def _transcribe() -> str:
            result = model.transcribe(tmp_path, language=_STT_LANG, fp16=False)
            return result.get("text", "").strip()

        text = await asyncio.get_event_loop().run_in_executor(None, _transcribe)
        Path(tmp_path).unlink(missing_ok=True)

        if text:
            logger.info("[voice-stt] transcribed %d chars from %s", len(text), url[-40:])
        return text or None

    except Exception as exc:
        logger.warning("[voice-stt] transcription failed: %s", exc)
        return None


async def transcribe_attachment_urls(urls: list[str]) -> list[tuple[str, str]]:
    """
    Filter audio URLs from *urls*, transcribe each one.
    Returns list of (url, transcript) tuples for successful transcriptions.
    """
    audio_urls = [u for u in urls if is_audio_url(u)]
    if not audio_urls:
        return []

    results = []
    for url in audio_urls[:3]:  # cap at 3 per message
        text = await transcribe_url(url)
        if text:
            results.append((url, text))
    return results


def format_transcription_context(transcriptions: list[tuple[str, str]]) -> str:
    """Format transcriptions for injection into the user message."""
    if not transcriptions:
        return ""
    lines = ["[Voice message transcription]"]
    for i, (url, text) in enumerate(transcriptions, 1):
        prefix = f"[Audio {i}]: " if len(transcriptions) > 1 else ""
        lines.append(f"{prefix}{text}")
    return "\n".join(lines)


# ── TTS ───────────────────────────────────────────────────────────────────────

async def synthesise(text: str) -> Optional[bytes]:
    """
    Convert *text* to speech.  Returns WAV bytes or None if TTS is disabled
    or unavailable.
    """
    if not _TTS_ENABLED:
        return None

    if _TTS_ENGINE == "piper":
        return await _tts_piper(text)
    else:
        return await _tts_pyttsx3(text)


async def _tts_piper(text: str) -> Optional[bytes]:
    """Synthesise using Piper (high-quality, local neural TTS)."""
    if not _PIPER_MODEL or not _PIPER_CFG:
        logger.warning("[voice-tts] VOICE_PIPER_MODEL / VOICE_PIPER_CONFIG not set")
        return None
    try:
        import subprocess, shutil
        if not shutil.which("piper"):
            logger.warning("[voice-tts] 'piper' binary not found in PATH")
            return None

        proc = await asyncio.create_subprocess_exec(
            "piper",
            "--model", _PIPER_MODEL,
            "--config", _PIPER_CFG,
            "--output_raw",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        raw, _ = await proc.communicate(input=text.encode())

        # raw is 16-bit PCM @ 22050Hz; wrap in a minimal WAV header
        wav = _pcm_to_wav(raw, sample_rate=22050, channels=1, sampwidth=2)
        logger.info("[voice-tts] Piper synthesised %d bytes for %d chars", len(wav), len(text))
        return wav

    except Exception as exc:
        logger.warning("[voice-tts] Piper TTS failed: %s", exc)
        return None


async def _tts_pyttsx3(text: str) -> Optional[bytes]:
    """Synthesise using pyttsx3 (offline, cross-platform, lower quality)."""
    try:
        import pyttsx3

        def _run() -> bytes:
            engine = pyttsx3.init()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                path = tf.name
            engine.save_to_file(text, path)
            engine.runAndWait()
            data = Path(path).read_bytes()
            Path(path).unlink(missing_ok=True)
            return data

        wav = await asyncio.get_event_loop().run_in_executor(None, _run)
        logger.info("[voice-tts] pyttsx3 synthesised %d bytes", len(wav))
        return wav

    except ImportError:
        logger.warning("[voice-tts] pyttsx3 not installed")
        return None
    except Exception as exc:
        logger.warning("[voice-tts] pyttsx3 TTS failed: %s", exc)
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int, sampwidth: int) -> bytes:
    """Wrap raw PCM in a minimal RIFF/WAV header."""
    import struct
    data_len  = len(pcm)
    byte_rate = sample_rate * channels * sampwidth
    block_align = channels * sampwidth
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_len))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, sampwidth * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    buf.write(pcm)
    return buf.getvalue()
