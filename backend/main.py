"""
backend/main.py

FastAPI entrypoint for PULSE.
Provides:
- WebSocket endpoint for real-time smartphone PWA data ingestion
- REST API for the Next.js dashboard
- Debug endpoints for inspecting pipeline data
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Load .env from project root BEFORE anything reads os.getenv()
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        _env_example = Path(__file__).parent.parent / ".env.example"
        if _env_example.exists():
            load_dotenv(_env_example)
except ImportError:
    pass

import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .pipeline import PULSEPipeline
from .segment_manager import SegmentManager

# Suppress Windows ProactorEventLoop connection reset noise
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DEBUG_DIR = PROJECT_ROOT / "output" / "debug"

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="PULSE Backend API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve debug images/files statically
if DEBUG_DIR.exists():
    app.mount("/debug-files", StaticFiles(directory=str(DEBUG_DIR)), name="debug-files")

# Active session states: session_id -> {"manager", "pipeline", "results"}
active_sessions: Dict[str, dict] = {}


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}


# ── WebSocket — PWA Data Ingestion ───────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Main ingestion endpoint for smartphone PWA data."""
    await websocket.accept()
    logger.info(f"Client connected for session: {session_id}")

    manager = SegmentManager(segment_length_m=100.0)
    pipeline = PULSEPipeline(session_id=session_id)

    active_sessions[session_id] = {
        "manager": manager,
        "pipeline": pipeline,
        "results": [],  # Store segment results for dashboard API
    }

    try:
        while True:
            packet = await websocket.receive_json()

            # Handle PWA heartbeat
            if packet.get("type") == "PING":
                await websocket.send_json({"type": "PONG", "timestamp": packet.get("timestamp")})
                continue

            manager.ingest_packet(packet)

            for segment in manager.get_ready_segments():
                logger.info(f"[{session_id}] Processing segment: {segment['segment_id']}")
                asyncio.create_task(process_and_notify(websocket, pipeline, segment, session_id))

    except WebSocketDisconnect:
        logger.info(f"Client disconnected for session: {session_id}")
        manager.flush()
        for segment in manager.get_ready_segments():
            asyncio.create_task(process_and_notify(websocket, pipeline, segment, session_id))

    finally:
        pipeline.finalise()
        active_sessions.pop(session_id, None)


async def process_and_notify(
    websocket: WebSocket,
    pipeline: PULSEPipeline,
    segment: dict,
    session_id: str,
):
    """Run the ML pipeline and stream results back to the PWA."""
    try:
        result = await pipeline.process_segment(segment)

        # Store result for dashboard REST API
        if session_id in active_sessions:
            active_sessions[session_id]["results"].append(result)

        # Send to PWA with SEGMENT_COMPLETE type (matches PWA expectation)
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_json({"type": "SEGMENT_COMPLETE", "segment": result})
    except Exception as e:
        logger.error(f"Pipeline error on segment {segment.get('segment_id')}: {e}")
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_json({"type": "ERROR", "message": str(e)})


# ── HTTP Video Upload — PWA sends recorded video clips here ─────────────────


@app.post("/upload/video/{session_id}")
async def upload_video(session_id: str, video: UploadFile = File(...)):
    """
    Receive a short video clip from the PWA, extract frames using OpenCV,
    and inject them into the active session's SegmentManager buffer.

    This avoids takePictureAsync (shutter sound + preview freeze) on the phone.
    The PWA records silent video via recordAsync() and uploads here.
    """
    if session_id not in active_sessions:
        return {"error": "Session not found or already closed", "frames_extracted": 0}

    manager = active_sessions[session_id]["manager"]

    # Write the uploaded bytes to a temp file so OpenCV can read it
    tmp = None
    frames_extracted = 0
    try:
        contents = await video.read()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(contents)
        tmp.flush()
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            logger.warning(f"[Upload] Could not open video file for session {session_id}")
            return {"error": "Could not decode video", "frames_extracted": 0}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample ~2 frames per second (evenly spaced) to match the pipeline's needs
        target_fps = 2.0
        frame_interval = max(1, int(fps / target_fps))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                manager._buffer["frames"].append(frame)
                frames_extracted += 1
            frame_idx += 1

        cap.release()
        logger.info(
            f"[Upload] Session {session_id}: extracted {frames_extracted} frames "
            f"from {frame_idx} total ({fps:.0f} fps video)"
        )
    except Exception as e:
        logger.error(f"[Upload] Frame extraction failed for {session_id}: {e}")
        return {"error": str(e), "frames_extracted": frames_extracted}
    finally:
        if tmp:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return {"ok": True, "frames_extracted": frames_extracted}


# ── REST API — Dashboard ────────────────────────────────────────────────────

@app.get("/api/sessions")
def api_list_sessions():
    """List all sessions (active + completed debug sessions)."""
    sessions = []

    # Active sessions
    for sid, state in active_sessions.items():
        results = state.get("results", [])
        sessions.append({
            "session_id": sid,
            "status": "active",
            "segment_count": len(results),
            "avg_iri": _avg([r.get("iri", {}).get("iri_value") for r in results]),
            "avg_pci": _avg([r.get("visual", {}).get("pci_estimate") for r in results]),
            "total_distance_km": sum(r.get("length_km", 0) for r in results),
        })

    # Completed sessions from debug output
    if DEBUG_DIR.exists():
        for d in sorted(DEBUG_DIR.iterdir(), reverse=True):
            if d.is_dir() and d.name not in {s["session_id"] for s in sessions}:
                segments = [s.name for s in sorted(d.iterdir()) if s.is_dir()]
                seg_data = _load_session_segments(d.name)
                sessions.append({
                    "session_id": d.name,
                    "status": "completed",
                    "segment_count": len(segments),
                    "avg_iri": _avg([s.get("iri_value") for s in seg_data]),
                    "avg_pci": _avg([s.get("pci_estimate") for s in seg_data]),
                    "total_distance_km": sum(s.get("length_km", 0) for s in seg_data),
                })

    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str):
    """Get full session detail with all segment results."""
    # Check active sessions first
    if session_id in active_sessions:
        results = active_sessions[session_id]["results"]
        return {
            "session_id": session_id,
            "status": "active",
            "segments": results,
            "summary": _compute_summary(results),
        }

    # Check debug output
    session_dir = DEBUG_DIR / session_id
    if session_dir.exists():
        segments = _load_full_segments(session_id)
        return {
            "session_id": session_id,
            "status": "completed",
            "segments": segments,
            "summary": _compute_summary(segments),
        }

    return {"error": "Session not found"}


@app.get("/api/sessions/{session_id}/segments")
def api_get_segments(session_id: str):
    """Get all segments for a session."""
    if session_id in active_sessions:
        return {"segments": active_sessions[session_id]["results"]}

    session_dir = DEBUG_DIR / session_id
    if session_dir.exists():
        return {"segments": _load_full_segments(session_id)}

    return {"error": "Session not found"}


@app.get("/api/live")
def api_live_status():
    """Get current active session status for live dashboard."""
    if not active_sessions:
        return {"active": False, "sessions": []}

    live = []
    for sid, state in active_sessions.items():
        results = state.get("results", [])
        latest = results[-1] if results else {}
        live.append({
            "session_id": sid,
            "segment_count": len(results),
            "latest_segment": latest,
            "current_gps": latest.get("gps"),
            "current_iri": latest.get("iri", {}).get("iri_value"),
            "current_pci": latest.get("visual", {}).get("pci_estimate"),
            "current_speed_kmh": latest.get("avg_speed_kmh"),
        })

    return {"active": True, "sessions": live}


@app.get("/api/stats")
def api_global_stats():
    """Global statistics across all sessions."""
    all_segments = []

    # Collect from active sessions
    for state in active_sessions.values():
        all_segments.extend(state.get("results", []))

    # Collect from debug output
    if DEBUG_DIR.exists():
        for session_dir in DEBUG_DIR.iterdir():
            if session_dir.is_dir():
                all_segments.extend(_load_session_segments(session_dir.name))

    total_km = sum(s.get("length_km", 0) for s in all_segments)
    iri_values = [s.get("iri_value") or s.get("iri", {}).get("iri_value") for s in all_segments]
    pci_values = [s.get("pci_estimate") or s.get("visual", {}).get("pci_estimate") for s in all_segments]

    return {
        "total_sessions": len(set(s.get("session_id", "") for s in all_segments)) if all_segments else 0,
        "total_segments": len(all_segments),
        "total_distance_km": round(total_km, 2),
        "avg_iri": _avg(iri_values),
        "avg_pci": _avg(pci_values),
        "distress_count": sum(
            len(s.get("visual", {}).get("distresses", []))
            for s in all_segments
        ),
    }


# ── Debug Endpoints ──────────────────────────────────────────────────────────


@app.get("/api/frames/{session_id}/{segment_id}")
def api_get_frames(session_id: str, segment_id: str):
    """List all VLM input frames for a segment (for dashboard collage view)."""
    frames_dir = DEBUG_DIR / session_id / segment_id / "vlm_input_frames"
    if not frames_dir.exists():
        return {"frames": [], "base_url": ""}

    images = sorted(
        f.name for f in frames_dir.iterdir()
        if f.suffix in (".jpg", ".png", ".jpeg")
    )
    return {
        "frames": images,
        "base_url": f"/debug-files/{session_id}/{segment_id}/vlm_input_frames",
        "count": len(images),
    }


@app.get("/api/frames/{session_id}")
def api_get_session_frames(session_id: str):
    """List all segments with their VLM frames for a session."""
    session_dir = DEBUG_DIR / session_id
    if not session_dir.exists():
        return {"segments": []}

    result = []
    for seg_dir in sorted(session_dir.iterdir()):
        if not seg_dir.is_dir():
            continue
        frames_dir = seg_dir / "vlm_input_frames"
        if frames_dir.exists():
            images = sorted(
                f.name for f in frames_dir.iterdir()
                if f.suffix in (".jpg", ".png", ".jpeg")
            )
            result.append({
                "segment_id": seg_dir.name,
                "frames": images,
                "base_url": f"/debug-files/{session_id}/{seg_dir.name}/vlm_input_frames",
                "count": len(images),
            })
    return {"segments": result}


@app.get("/debug/sessions")
def list_debug_sessions():
    """List all debug sessions that have been recorded."""
    if not DEBUG_DIR.exists():
        return {"sessions": []}
    sessions = []
    for d in sorted(DEBUG_DIR.iterdir()):
        if d.is_dir():
            segments = [s.name for s in sorted(d.iterdir()) if s.is_dir()]
            sessions.append({"session_id": d.name, "segments": segments})
    return {"sessions": sessions}


@app.get("/debug/{session_id}/{segment_id}")
def get_debug_data(session_id: str, segment_id: str):
    """Return all debug JSON files for a specific segment."""
    seg_dir = DEBUG_DIR / session_id / segment_id
    if not seg_dir.exists():
        return {"error": "Segment not found"}

    result = {"session_id": session_id, "segment_id": segment_id, "files": {}}

    for f in sorted(seg_dir.iterdir()):
        if f.suffix == ".json":
            try:
                result["files"][f.name] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                result["files"][f.name] = {"error": "Could not parse"}
        elif f.suffix == ".txt":
            result["files"][f.name] = f.read_text(encoding="utf-8")
        elif f.is_dir():
            images = [img.name for img in sorted(f.iterdir()) if img.suffix in (".jpg", ".png")]
            result["files"][f.name] = {
                "type": "image_directory",
                "count": len(images),
                "files": images,
                "base_url": f"/debug-files/{session_id}/{segment_id}/{f.name}"
            }

    return result


# ── Helpers ──────────────────────────────────────────────────────────────────

def _avg(values: list) -> Optional[float]:
    """Average of non-None values, or None if empty."""
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 2) if clean else None


def _load_session_segments(session_id: str) -> List[dict]:
    """Load summary data from debug pipeline_result.json files."""
    session_dir = DEBUG_DIR / session_id
    if not session_dir.exists():
        return []

    segments = []
    for seg_dir in sorted(session_dir.iterdir()):
        if not seg_dir.is_dir():
            continue
        result_file = seg_dir / "pipeline_result.json"
        if result_file.exists():
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
                data["segment_id"] = seg_dir.name
                segments.append(data)
            except Exception:
                pass
    return segments


def _load_full_segments(session_id: str) -> List[dict]:
    """Load full debug data for all segments in a session."""
    session_dir = DEBUG_DIR / session_id
    if not session_dir.exists():
        return []

    segments = []
    for seg_dir in sorted(session_dir.iterdir()):
        if not seg_dir.is_dir():
            continue
        seg = {"segment_id": seg_dir.name}
        for f in seg_dir.iterdir():
            if f.suffix == ".json":
                try:
                    seg[f.stem] = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    pass
            elif f.suffix == ".txt":
                seg[f.stem] = f.read_text(encoding="utf-8")
        segments.append(seg)
    return segments


def _compute_summary(results: list) -> dict:
    """Compute summary statistics from a list of segment results."""
    iri_values = [r.get("iri", {}).get("iri_value") for r in results]
    pci_values = [r.get("visual", {}).get("pci_estimate") for r in results]
    speeds = [r.get("avg_speed_kmh") for r in results]

    return {
        "segment_count": len(results),
        "total_distance_km": round(sum(r.get("length_km", 0) for r in results), 2),
        "avg_iri": _avg(iri_values),
        "avg_pci": _avg(pci_values),
        "avg_speed_kmh": _avg(speeds),
        "distress_count": sum(
            len(r.get("visual", {}).get("distresses", []))
            for r in results
        ),
    }
