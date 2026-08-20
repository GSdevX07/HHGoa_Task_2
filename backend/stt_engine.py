"""
Speech-to-Text (STT) Engine for RAG Pipeline
Supports Sarvam AI (https://api.sarvam.ai/speech-to-text) and ElevenLabs (Scribe STT).
Includes latency tracking and automated fallback handling.
"""

import os
import time
import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger("stt_engine")

class STTEngine:
    def __init__(self):
        self.sarvam_api_key = os.getenv("SARVAM_API_KEY", "").strip()
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        self.preferred_provider = os.getenv("STT_PROVIDER", "sarvam").lower() # "sarvam" or "elevenlabs"

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "input.wav",
        language_code: str = "hi-IN",
        provider_override: Optional[str] = None,
        transcript_fallback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio bytes using Sarvam AI or ElevenLabs API.
        Returns dict with transcribed text, latency_ms, provider used, and status.
        """
        provider = (provider_override or self.preferred_provider).lower()
        start_time = time.perf_counter()

        # Call Sarvam AI API if API key is configured
        if provider == "sarvam" and self.sarvam_api_key and audio_bytes:
            res = await self._transcribe_sarvam(audio_bytes, filename, language_code)
            if res.get("success"):
                res["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
                return res

        # Call ElevenLabs API if API key is configured
        if provider == "elevenlabs" and self.elevenlabs_api_key and audio_bytes:
            res = await self._transcribe_elevenlabs(audio_bytes, filename)
            if res.get("success"):
                res["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
                return res

        # Try alternate provider if main provider key is missing
        if self.sarvam_api_key and provider != "sarvam" and audio_bytes:
            res = await self._transcribe_sarvam(audio_bytes, filename, language_code)
            if res.get("success"):
                res["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
                return res
        elif self.elevenlabs_api_key and provider != "elevenlabs" and audio_bytes:
            res = await self._transcribe_elevenlabs(audio_bytes, filename)
            if res.get("success"):
                res["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
                return res

        # No API keys configured — return an honest failure.
        # The caller (voice endpoint in main.py) will handle this by:
        #   a) returning a 422 with a clear message, OR
        #   b) accepting a transcript_fallback form field for demo/testing mode.
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        logger.warning(
            f"STT provider '{provider}' has no API key configured. "
            "Set SARVAM_API_KEY or ELEVENLABS_API_KEY in backend/.env"
        )
        return {
            "success": False,
            "transcript": "",
            "provider": "none",
            "latency_ms": round(elapsed, 2),
            "language": language_code,
            "error": (
                f"No API key configured for STT provider '{provider}'. "
                "Set SARVAM_API_KEY or ELEVENLABS_API_KEY in backend/.env to enable voice transcription. "
                "Text queries via /api/query/text work without any STT keys."
            ),
        }

    async def _transcribe_sarvam(self, audio_bytes: bytes, filename: str, language_code: str) -> Dict[str, Any]:
        """
        Sarvam AI STT API call
        Endpoint: https://api.sarvam.ai/speech-to-text
        """
        url = "https://api.sarvam.ai/speech-to-text"
        headers = {
            "api-subscription-key": self.sarvam_api_key
        }
        files = {
            "file": (filename, audio_bytes, "audio/wav")
        }
        data = {
            "language_code": language_code,
            "model": "saarika:v2.5"
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=headers, files=files, data=data)
                if response.status_code == 200:
                    result = response.json()
                    transcript = result.get("transcript", "")
                    return {
                        "success": True,
                        "transcript": transcript,
                        "provider": "Sarvam AI (saarika:v2.5)",
                        "raw_response": result
                    }
                else:
                    logger.error(f"Sarvam API error {response.status_code}: {response.text}")
                    return {"success": False, "error": f"Sarvam API error: {response.status_code}"}
        except Exception as e:
            logger.error(f"Sarvam STT exception: {e}")
            return {"success": False, "error": str(e)}

    async def _transcribe_elevenlabs(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        ElevenLabs Scribe STT API call
        Endpoint: https://api.elevenlabs.io/v1/speech-to-text
        """
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {
            "xi-api-key": self.elevenlabs_api_key
        }
        files = {
            "file": (filename, audio_bytes, "audio/wav")
        }
        data = {
            "model_id": "scribe_v1"
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=headers, files=files, data=data)
                if response.status_code == 200:
                    result = response.json()
                    transcript = result.get("text", "")
                    return {
                        "success": True,
                        "transcript": transcript,
                        "provider": "ElevenLabs Scribe",
                        "raw_response": result
                    }
                else:
                    logger.error(f"ElevenLabs API error {response.status_code}: {response.text}")
                    return {"success": False, "error": f"ElevenLabs API error: {response.status_code}"}
        except Exception as e:
            logger.error(f"ElevenLabs STT exception: {e}")
            return {"success": False, "error": str(e)}
