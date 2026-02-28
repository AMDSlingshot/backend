"""
backend/main.py

FastAPI entrypoint for PULSE.
Provides:
- WebSocket endpoint for real-time smartphone data ingestion
- REST endpoints for session summaries and report generation
- Debug endpoints for inspecting pipeline data
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

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

if DEBUG_DIR.exists():
    app.mount("/debug-files", StaticFiles(directory=str(DEBUG_DIR)), name="debug-files")

# Active session states
active_sessions: Dict[str, dict] = {}


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
    }

    try:
        while True:
            packet = await websocket.receive_json()
            manager.ingest_packet(packet)

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


@app.get("/session/{session_id}/summary")
def get_session_summary(session_id: str):
    """Fetch aggregate data for an entire drive."""
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
