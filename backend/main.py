"""
backend/main.py

FastAPI entrypoint for PULSE.
Provides:
- WebSocket endpoint for real-time smartphone data ingestion
- REST endpoints for session summaries and report generation
- REST API endpoints for Next.js dashboard (live, sessions, stats, segments)
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
    pass  # Will use system environment variables

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DEBUG_DIR = PROJECT_ROOT / "output" / "debug"

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="PULSE Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# Ensure debug dir exists before mounting so the route doesn't fail silently on startup
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/debug-files", StaticFiles(directory=str(DEBUG_DIR)), name="debug-files")

# Active session states
active_sessions: Dict[str, dict] = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_all_session_results() -> List[dict]:
    """
    Read pipeline_result.json files from every debug session/segment directory.
    Returns a flat list of all segment result dicts.
    """
    results = []
    if not DEBUG_DIR.exists():
        return results
    for session_dir in sorted(DEBUG_DIR.iterdir()):
        if not session_dir.is_dir():
            continue
        for seg_dir in sorted(session_dir.iterdir()):
            if not seg_dir.is_dir():
                continue
            result_file = seg_dir / "pipeline_result.json"
            if result_file.exists():
                try:
                    results.append(json.loads(result_file.read_text(encoding="utf-8")))
                except Exception:
                    pass
    return results


def _session_summary_from_segments(session_id: str, segs: List[dict]) -> dict:
    """Build a SessionSummary dict from a list of segment result dicts."""
    iri_values = [s["iri"]["iri_value"] for s in segs
                  if s.get("iri", {}).get("iri_value") is not None]
    pci_values = [s.get("pci_estimate") for s in segs if s.get("pci_estimate") is not None]
    total_km = sum(s.get("length_km", 0) for s in segs)
    return {
        "session_id": session_id,
        "status": "completed",
        "segment_count": len(segs),
        "avg_iri": round(sum(iri_values) / len(iri_values), 2) if iri_values else None,
        "avg_pci": round(sum(pci_values) / len(pci_values), 1) if pci_values else None,
        "total_distance_km": round(total_km, 4),
    }


# ── Core Endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Main ingestion endpoint for smartphone data."""
    await websocket.accept()
    logger.info(f"Client connected for session: {session_id}")

    manager = SegmentManager(segment_length_m=100.0)
    pipeline = PULSEPipeline(session_id=session_id)

    active_sessions[session_id] = {
        "manager": manager,
        "pipeline": pipeline,
        "current_gps": None,
        "current_speed_kmh": 0.0,
    }

    try:
        while True:
            packet = await websocket.receive_json()
            manager.ingest_packet(packet)

            # Update live telemetry for /api/live endpoint
            ptype = packet.get("type", "")
            pdata = packet.get("data", {})
            if ptype == "gps":
                active_sessions[session_id]["current_gps"] = {
                    "lat": pdata.get("lat", 0),
                    "lng": pdata.get("lng", 0),
                }
                speed_ms = pdata.get("speed", 0)
                active_sessions[session_id]["current_speed_kmh"] = round(speed_ms * 3.6, 1)

            for segment in manager.get_ready_segments():
                logger.info(f"[{session_id}] Processing segment: {segment['segment_id']}")
                asyncio.create_task(process_and_notify(websocket, pipeline, segment))

    except WebSocketDisconnect:
        logger.info(f"Client disconnected for session: {session_id}")
        manager.flush()
        for segment in manager.get_ready_segments():
            asyncio.create_task(process_and_notify(websocket, pipeline, segment))

    finally:
        pipeline.finalise()
        active_sessions.pop(session_id, None)


async def process_and_notify(websocket: WebSocket, pipeline: PULSEPipeline, segment: dict):
    """Run the ML pipeline and stream results back to the frontend."""
    try:
        result = await pipeline.process_segment(segment)
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_json({"type": "segment_result", "data": result})
    except Exception as e:
        logger.error(f"Pipeline error on segment {segment.get('segment_id')}: {e}")
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_json({"type": "error", "message": str(e)})


# ── Dashboard REST API ────────────────────────────────────────────────────────

@app.get("/api/live")
def get_live_status():
    """
    Return currently active WebSocket sessions with live telemetry.
    Consumed by Next.js dashboard Zustand store every 5 seconds.
    """
    live_sessions = []
    for sid, state in active_sessions.items():
        summary = state["pipeline"].get_session_summary()
        live_sessions.append({
            "session_id": sid,
            "status": "active",
            "current_gps": state.get("current_gps"),
            "current_speed_kmh": state.get("current_speed_kmh", 0.0),
            "current_iri": summary.get("avg_iri"),
            "current_pci": None,  # Not computed live — available post-segment
            "segment_count": summary.get("segments_processed", 0),
        })
    return {
        "active": len(live_sessions) > 0,
        "sessions": live_sessions,
    }


@app.get("/api/sessions")
def list_sessions():
    """
    Return list of all historical sessions (from debug output files) plus any active sessions.
    """
    all_segments = _load_all_session_results()

    # Group by session_id
    session_map: Dict[str, List[dict]] = {}
    for seg in all_segments:
        sid = seg.get("session_id", "unknown")
        session_map.setdefault(sid, []).append(seg)

    sessions = [
        _session_summary_from_segments(sid, segs)
        for sid, segs in sorted(session_map.items())
    ]

    # Also include currently active sessions not yet in files
    active_ids = set(active_sessions.keys())
    file_ids = {s["session_id"] for s in sessions}
    for sid in active_ids - file_ids:
        summary = active_sessions[sid]["pipeline"].get_session_summary()
        sessions.append({
            "session_id": sid,
            "status": "active",
            "segment_count": summary.get("segments_processed", 0),
            "avg_iri": summary.get("avg_iri"),
            "avg_pci": None,
            "total_distance_km": summary.get("total_length_km", 0),
        })

    return {"sessions": sessions}


@app.get("/api/stats")
def get_global_stats():
    """
    Return aggregate statistics across all historical sessions.
    """
    all_segments = _load_all_session_results()

    # Count unique sessions from files
    session_ids = {s.get("session_id") for s in all_segments}

    iri_values = [s["iri"]["iri_value"] for s in all_segments
                  if s.get("iri", {}).get("iri_value") is not None]
    pci_values = [s.get("pci_estimate") for s in all_segments
                  if s.get("pci_estimate") is not None]
    total_km = sum(s.get("length_km", 0) for s in all_segments)
    distress_count = sum(len(s.get("distresses", [])) for s in all_segments)

    return {
        "total_sessions": len(session_ids) + len(active_sessions),
        "total_segments": len(all_segments),
        "total_distance_km": round(total_km, 4),
        "avg_iri": round(sum(iri_values) / len(iri_values), 2) if iri_values else None,
        "avg_pci": round(sum(pci_values) / len(pci_values), 1) if pci_values else None,
        "distress_count": distress_count,
    }


@app.get("/api/sessions/{session_id}/segments")
def get_session_segments(session_id: str):
    """
    Return all segment results for a specific session.
    First checks in-memory (active), then on-disk debug files.
    """
    # Check active sessions first (in-memory)
    if session_id in active_sessions:
        summary = active_sessions[session_id]["pipeline"].get_session_summary()
        segs = summary.get("segments", [])
        return {"session_id": session_id, "status": "active", "segments": segs}

    # Load from debug output files
    session_dir = DEBUG_DIR / session_id
    if not session_dir.exists():
        return {"error": f"Session {session_id} not found", "segments": []}

    segs = []
    for seg_dir in sorted(session_dir.iterdir()):
        if not seg_dir.is_dir():
            continue
        result_file = seg_dir / "pipeline_result.json"
        if result_file.exists():
            try:
                segs.append(json.loads(result_file.read_text(encoding="utf-8")))
            except Exception:
                pass

    return {"session_id": session_id, "status": "completed", "segments": segs}


@app.get("/api/frames/{session_id}")
def get_session_frames(session_id: str):
    """
    Return frame lists for each segment in a session.
    Used by the Visual Feed (context) page to display VLM input images.
    Images are served via the /debug-files/* static mount.
    """
    session_dir = DEBUG_DIR / session_id
    if not session_dir.exists():
        return {"session_id": session_id, "segments": []}

    segment_frames = []
    for seg_dir in sorted(session_dir.iterdir()):
        if not seg_dir.is_dir():
            continue
        seg_id = seg_dir.name

        # vlm_input_frames first (frames actually sent to VLM), fallback to captured_frames
        for frame_dir_name in ("vlm_input_frames", "captured_frames"):
            frame_dir = seg_dir / frame_dir_name
            if frame_dir.exists():
                frames = sorted(
                    f.name for f in frame_dir.iterdir()
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                )
                if frames:
                    base_url = f"/debug-files/{session_id}/{seg_id}/{frame_dir_name}"
                    segment_frames.append({
                        "segment_id": seg_id,
                        "frame_source": frame_dir_name,
                        "frames": frames,
                        "count": len(frames),
                        "base_url": base_url,
                    })
                    break

    return {"session_id": session_id, "segments": segment_frames}


@app.get("/session/{session_id}/summary")
def get_session_summary(session_id: str):
    """Fetch aggregate data for an entire drive (legacy endpoint)."""
    if session_id in active_sessions:
        return active_sessions[session_id]["pipeline"].get_session_summary()
    return {"error": "Session not found or already closed."}


# ── Debug Endpoints ──────────────────────────────────────────────────────────

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


@app.get("/debug/viewer", response_class=HTMLResponse)
def debug_viewer():
    """Serve the debug viewer from frontend/debug.html."""
    html_path = FRONTEND_DIR / "debug.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Debug viewer not found</h1><p>Expected at frontend/debug.html</p>"
