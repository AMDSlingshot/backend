"""
backend/debug_logger.py

Full-visibility debug logger for the PULSE pipeline.

Saves to output/debug/<session_id>/<segment_id>/:
    raw_sensor_stats.json   — IMU count, frame count, GPS points, speed
    vlm_input_frames/       — actual JPEG images sent to Qwen3-VL
    vlm_raw_response.txt    — exact text that came back from the model
    vlm_parsed.json         — parsed JSON assessment (or parse error)
    depth_result.json       — depth pipeline output
    iri_result.json         — IRI computation output
    fusion_result.json      — sensor fusion output
    pipeline_result.json    — full final pipeline output
    acoustic_result.json    — acoustic classifier output
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEBUG_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "debug")


class DebugLogger:
    """
    Writes intermediate pipeline data to disk for human inspection.
    """

    def __init__(self, session_id: str, enabled: bool = True):
        self.enabled = enabled
        self.session_id = session_id
        self.session_dir = os.path.join(DEBUG_ROOT, session_id)
        if self.enabled:
            os.makedirs(self.session_dir, exist_ok=True)
            logger.info(f"[DEBUG] Debug output directory: {self.session_dir}")

    def _seg_dir(self, segment_id: str) -> str:
        d = os.path.join(self.session_dir, segment_id)
        os.makedirs(d, exist_ok=True)
        return d

    def _write_json(self, path: str, data: dict):
        """Write a dict to a JSON file, handling non-serializable types."""
        def default_serializer(obj):
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=default_serializer, ensure_ascii=False)

    # ── Raw Sensor Data ───────────────────────────────────────────────────

    def log_raw_segment(self, segment: dict):
        """Log statistics about the raw data received from the smartphone."""
        if not self.enabled:
            return

        seg_id = segment.get("segment_id", "unknown")
        d = self._seg_dir(seg_id)

        stats = {
            "segment_id": seg_id,
            "timestamp": segment.get("timestamp"),
            "gps_midpoint": segment.get("gps"),
            "avg_speed_kmh": segment.get("avg_speed_kmh"),
            "avg_speed_ms": segment.get("avg_speed_ms"),
            "length_km": segment.get("length_km"),
            "imu_readings_count": len(segment.get("imu_buffer", [])),
            "frames_count": len(segment.get("frames", [])),
            "gps_points_count": len(segment.get("gps_buffer", [])),
            "audio_packets_count": len(segment.get("audio_buffer", [])),
        }

        # Save first 5 IMU readings as sample
        imu_buf = segment.get("imu_buffer", [])
        if imu_buf:
            stats["imu_sample_first_5"] = imu_buf[:5]
            stats["imu_sample_last_5"] = imu_buf[-5:]

        # Save first 5 GPS readings
        gps_buf = segment.get("gps_buffer", [])
        if gps_buf:
            stats["gps_sample_first_3"] = gps_buf[:3]
            stats["gps_sample_last_3"] = gps_buf[-3:]

        self._write_json(os.path.join(d, "raw_sensor_stats.json"), stats)
        logger.info(f"[DEBUG] Saved raw sensor stats for {seg_id}")

    # ── Camera Frames ────────────────────────────────────────────────────

    def log_frames(self, segment_id: str, frames: list):
        """Save actual camera frames that were captured from the phone."""
        if not self.enabled or not frames:
            return

        d = self._seg_dir(segment_id)
        frames_dir = os.path.join(d, "captured_frames")
        os.makedirs(frames_dir, exist_ok=True)

        import cv2
        for i, frame in enumerate(frames):
            if frame is not None:
                path = os.path.join(frames_dir, f"frame_{i:03d}.jpg")
                cv2.imwrite(path, frame)

        logger.info(f"[DEBUG] Saved {len(frames)} captured frames for {segment_id}")

    # ── VLM Input/Output ─────────────────────────────────────────────────

    def log_vlm_input(self, segment_id: str, pil_frames: list, prompt: str, system_prompt: str):
        """Save the exact images and prompt sent to Qwen3-VL."""
        if not self.enabled:
            return

        d = self._seg_dir(segment_id)
        vlm_dir = os.path.join(d, "vlm_input_frames")
        os.makedirs(vlm_dir, exist_ok=True)

        for i, pil_img in enumerate(pil_frames):
            path = os.path.join(vlm_dir, f"vlm_frame_{i:03d}.jpg")
            pil_img.save(path, "JPEG", quality=85)

        # Save the prompt
        self._write_json(os.path.join(d, "vlm_prompt.json"), {
            "system_prompt": system_prompt,
            "assessment_prompt": prompt,
            "num_images_sent": len(pil_frames),
            "image_sizes": [f"{img.width}x{img.height}" for img in pil_frames],
        })
        logger.info(f"[DEBUG] Saved VLM input ({len(pil_frames)} frames) for {segment_id}")

    def log_vlm_output(self, segment_id: str, raw_response: str, parsed_result: dict, inference_time_s: float):
        """Save the exact raw text from Ollama and the parsed JSON."""
        if not self.enabled:
            return

        d = self._seg_dir(segment_id)

        # Raw text — exactly what the model returned
        with open(os.path.join(d, "vlm_raw_response.txt"), "w", encoding="utf-8") as f:
            f.write(raw_response)

        # Parsed JSON result
        self._write_json(os.path.join(d, "vlm_parsed.json"), {
            "inference_time_s": inference_time_s,
            "parse_success": "error" not in parsed_result or parsed_result.get("error") != "parse_failed",
            "parsed_result": parsed_result,
        })
        logger.info(f"[DEBUG] Saved VLM output for {segment_id} ({inference_time_s:.1f}s)")

    # ── Other Pipeline Stages ────────────────────────────────────────────

    def log_stage(self, segment_id: str, stage_name: str, result: dict):
        """Generic logger for any pipeline stage (IRI, depth, fusion, etc.)."""
        if not self.enabled:
            return

        d = self._seg_dir(segment_id)
        self._write_json(os.path.join(d, f"{stage_name}.json"), result)

    def log_final_result(self, segment_id: str, result: dict):
        """Save the complete final pipeline output."""
        if not self.enabled:
            return

        d = self._seg_dir(segment_id)
        # Remove heavy data (frames) before saving
        result_clean = {k: v for k, v in result.items() if k != "frames"}
        self._write_json(os.path.join(d, "pipeline_result.json"), result_clean)
        logger.info(f"[DEBUG] Full pipeline result saved for {segment_id}")
