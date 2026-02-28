"""
backend/agents/visual_assessor.py

Agent 2 — Visual Road Distress Assessor

Uses Qwen3-VL served via local Ollama — no HuggingFace download required
for the VLM. Ollama manages VRAM allocation and model loading automatically.

Model selection for RTX 4050 (6 GB VRAM):
    Primary:  qwen3-vl:4b   (3.3 GB) — leaves ~2.7 GB for depth pipeline
    Fallback: qwen3-vl:2b   (1.9 GB) — ultra-safe, use if OOM

Pull models before first run:
    ollama pull qwen3-vl:4b
    ollama pull qwen3-vl:2b   # optional fallback

Ollama is also used for the LLM narrative agents (economic cascade,
government pipeline). Running everything via Ollama means a single
process manages all GPU memory — no cross-process VRAM fragmentation.
"""

import base64
import json
import logging
import time
from io import BytesIO
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Ollama model priority for RTX 4050 6 GB VRAM
# 4b = 3.3 GB: safe headroom when Depth Anything V2 (~500 MB) is co-loaded
# 2b = 1.9 GB: fallback when VRAM is under pressure
VLM_OLLAMA_MODELS = [
    "qwen3-vl:4b",
    "qwen3-vl:2b",
]

SYSTEM_PROMPT = """You are a pavement engineer. Assess road surface condition from images. Output only valid JSON."""

ASSESSMENT_PROMPT = """Analyse this road surface image. Return ONLY this JSON structure:
{
    "surface_type": "BC|WBM|Granular|Rigid|Unknown",
    "overall_condition": "Good|Fair|Poor|Very Poor",
    "pci_estimate": <0-100>,
    "distresses": [
        {
            "type": "pothole|alligator_crack|longitudinal_crack|transverse_crack|raveling|bleeding|rutting|edge_drop|corrugation|drainage",
            "severity": "Low|Medium|High",
            "extent_percent": <0-100>,
            "notes": "<specific observation>"
        }
    ],
    "drainage_adequacy": "Adequate|Inadequate|Blocked",
    "recommended_intervention": "Routine|Preventive|Rehabilitation|Reconstruction",
    "confidence": "High|Medium|Low",
    "limiting_factor": "<what reduced confidence, or empty string if High>"
}"""


def _frame_to_base64(frame_pil) -> str:
    """Convert a PIL Image to base64 string for Ollama API."""
    buf = BytesIO()
    frame_pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class VisualRoadAssessor:
    """
    Qwen3-VL visual road assessor via Ollama API.

    Ollama manages VRAM — no manual model loading or quantization needed.
    Auto-falls back from 4b to 2b if the primary model is not pulled.
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = VLM_OLLAMA_MODELS[0],
    ):
        self.host  = ollama_host
        self.model = model
        self._confirmed_model: Optional[str] = None
        self._probe_models()

    def _probe_models(self):
        """
        Check which Qwen3-VL models are pulled locally.
        Sets self._confirmed_model to the best available.
        """
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            if resp.status_code != 200:
                logger.warning("Ollama not reachable — visual assessment will be disabled.")
                return

            pulled = {m["name"] for m in resp.json().get("models", [])}

            for candidate in VLM_OLLAMA_MODELS:
                if candidate in pulled:
                    self._confirmed_model = candidate
                    logger.info(f"VisualRoadAssessor using Ollama model: {candidate}")
                    return

            # None pulled yet — log helpful instructions
            logger.warning(
                "No Qwen3-VL model found in Ollama. Pull one before driving:\n"
                "  ollama pull qwen3-vl:4b   ← recommended for RTX 4050 6GB\n"
                "  ollama pull qwen3-vl:2b   ← fallback\n"
                "Visual assessment will be degraded (returns Unknown condition)."
            )
        except Exception as exc:
            logger.warning(f"Could not probe Ollama models: {exc}")

    def assess_segment(
        self,
        frames: list,           # List of PIL Images from the segment
        segment_id: str,
        max_frames: int = 3,
    ) -> dict:
        """
        Assess a road segment from a list of PIL frames.

        Args:
            frames:     PIL Images (typically 5–10 from a 100m segment).
            segment_id: GPS-based identifier.
            max_frames: Max frames to send to VLM (keep low for speed).

        Returns:
            Structured assessment dict.
        """
        if not frames:
            return self._error_response(segment_id, "no_frames")

        model = self._confirmed_model
        if model is None:
            # Retry probe in case Ollama started after init
            self._probe_models()
            model = self._confirmed_model
        if model is None:
            return self._error_response(segment_id, "no_vlm_model_pulled")

        try:
            # Select evenly spaced frames
            import numpy as np
            indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames)), dtype=int)
            selected = [frames[i] for i in indices]

            # Encode frames as base64 JPEG
            images_b64 = [_frame_to_base64(f) for f in selected]

            # Build Ollama /api/chat multimodal request
            # Qwen3-VL requires the chat endpoint with messages array
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": ASSESSMENT_PROMPT,
                    "images": images_b64,
                },
            ]

            payload = {
                "model":  model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    # Qwen3-VL is a "thinking" model — it spends ~500 tokens on
                    # internal <think> reasoning before producing the JSON answer.
                    # num_predict must be large enough for thinking + response.
                    "num_predict": 4096,
                },
            }

            t0 = time.monotonic()
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=120,   # VLM inference can take 5–30s on 4b
            )
            elapsed = time.monotonic() - t0

            if resp.status_code != 200:
                logger.error(f"Ollama VLM request failed: HTTP {resp.status_code} — {resp.text[:200]}")
                return self._error_response(segment_id, f"ollama_http_{resp.status_code}")

            resp_json = resp.json()

            # Extract content — /api/chat returns {message: {content: ...}}
            raw_text = resp_json.get("message", {}).get("content", "")
            logger.info(f"VLM inference: {elapsed:.1f}s, model={model}, segment={segment_id}, response_len={len(raw_text)}")

            assessment = self._parse_response(raw_text)
            assessment["segment_id"]       = segment_id
            assessment["frames_analysed"]  = len(selected)
            assessment["model_used"]        = model
            assessment["inference_time_s"] = round(elapsed, 1)
            assessment["raw_response"]     = raw_text
            return assessment

        except requests.Timeout:
            logger.error(f"VLM inference timed out for segment {segment_id}")
            return self._error_response(segment_id, "timeout")
        except Exception as exc:
            logger.error(f"Visual assessment failed for {segment_id}: {exc}")
            return self._error_response(segment_id, str(exc))

    def _parse_response(self, response: str) -> dict:
        """Parse JSON from model response. Strips accidental markdown fences."""
        clean = (
            response.strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        # Strip Qwen3 thinking tags if present  (<think>...</think>)
        if "<think>" in clean:
            end_think = clean.find("</think>")
            if end_think != -1:
                clean = clean[end_think + len("</think>"):].strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract JSON substring
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(clean[start:end])
                except json.JSONDecodeError:
                    pass

        logger.warning(f"Could not parse VLM response as JSON: {clean[:200]}")
        return {
            "error":             "parse_failed",
            "raw_response":      clean[:500],
            "confidence":        "Low",
            "overall_condition": "Unknown",
        }

    def _error_response(self, segment_id: str, error: str) -> dict:
        return {
            "segment_id":        segment_id,
            "error":             error,
            "overall_condition": "Unknown",
            "confidence":        "Low",
            "distresses":        [],
            "pci_estimate":      None,
        }

    @property
    def is_ready(self) -> bool:
        return self._confirmed_model is not None
