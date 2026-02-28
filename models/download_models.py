"""
models/download_models.py

PULSE model setup script.
Downloads all required models and provides Ollama pull instructions.

Run:
    python models/download_models.py

Models:
    1. Depth Anything V2 Small — HuggingFace (~100 MB) — depth estimation
    2. Qwen3-VL 4b            — Ollama (3.3 GB)        — visual road assessment
    3. ORB vocab → not needed  (using DPVO / stella_vslam instead)

GPU memory budget (RTX 4050, 6 GB VRAM):
    Depth Anything V2 Small : ~500 MB  (always loaded in-process)
    Qwen3-VL 4b             : ~3.3 GB  (served by Ollama, loaded on demand)
    ─────────────────────────────────────────────────────────────────────────
    Max simultaneous        : ~3.8 GB  
    ✅ Safe for 6 GB VRAM
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── 1. Depth Anything V2 Small (HuggingFace) ──────────────────────────────

def download_depth_anything():
    logger.info("Downloading Depth Anything V2 Small (HuggingFace)...")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id="depth-anything/Depth-Anything-V2-Small-hf",
            ignore_patterns=["*.gguf"],
        )
        logger.info(f"  ✓ Depth Anything V2 Small cached at: {path}")
        return True
    except Exception as exc:
        logger.error(f"  ✗ Depth Anything V2 failed: {exc}")
        return False


# ── 2. Ollama model pulls ──────────────────────────────────────────────────

def check_ollama_running() -> bool:
    """Check if Ollama daemon is reachable."""
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def ollama_pull(model: str) -> bool:
    """Pull a model via Ollama CLI."""
    logger.info(f"  Pulling {model} via Ollama (may take several minutes)...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=False,  # Show progress live
            timeout=1800,          # 30 min max (large models)
        )
        return result.returncode == 0
    except FileNotFoundError:
        logger.error("  'ollama' command not found. Install from https://ollama.ai")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"  Pull timed out for {model}")
        return False


def get_pulled_models() -> set:
    """Return set of currently pulled Ollama model names."""
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        pass
    return set()


def setup_ollama_models():
    """Pull required Ollama models if not already present."""
    if not check_ollama_running():
        logger.error(
            "Ollama is not running. Start it first:\n"
            "  Windows: ollama serve   (or start from system tray)\n"
            "  Then re-run this script."
        )
        return False

    pulled = get_pulled_models()
    logger.info(f"  Currently pulled models: {pulled or '(none)'}")

    required = [
        # (model_name, description, required_or_optional)
        ("qwen3-vl:4b",   "Visual road assessment (3.3 GB)",     "required"),
        ("qwen3-vl:2b",   "VLM fallback (1.9 GB)",               "optional"),
    ]

    all_ok = True
    for model, desc, importance in required:
        if model in pulled:
            logger.info(f"  ✓ {model} already pulled ({desc})")
            continue

        if importance == "optional":
            logger.info(f"  - Skipping optional {model} ({desc}). Pull manually if needed.")
            continue

        logger.info(f"\nPulling {model} — {desc}")
        ok = ollama_pull(model)
        if ok:
            logger.info(f"  ✓ {model} pulled successfully")
        else:
            logger.error(f"  ✗ {model} pull failed")
            all_ok = False

    return all_ok


# ── 3. Optional visual odometry ───────────────────────────────────────────

def check_dpvo():
    """Check if DPVO is installed."""
    try:
        import dpvo  # noqa: F401
        logger.info("  ✓ DPVO installed — full 3-anchor scale fusion active")
        return True
    except ImportError:
        logger.warning(
            "  - DPVO not installed. System will use 2/3 scale anchors.\n"
            "    To install: pip install dpvo  (requires CUDA)"
        )
        return False


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("PULSE — Model Setup Script")
    logger.info("GPU: RTX 4050 6 GB VRAM")
    logger.info("=" * 60)

    results = {}

    logger.info("\n[1/3] Depth Anything V2 Small (HuggingFace)")
    results["depth"] = download_depth_anything()

    logger.info("\n[2/3] Ollama visual models")
    results["ollama"] = setup_ollama_models()

    logger.info("\n[3/3] DPVO visual odometry (optional)")
    results["dpvo"] = check_dpvo()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Setup Summary:")
    for name, ok in results.items():
        status = "✓" if ok else "✗ (see warnings above)"
        logger.info(f"  {status}  {name}")

    logger.info("""
Next steps:
  1. Copy environment template:
       copy .env.example .env
     Then edit .env to add your GEMINI_API_KEY.

  2. Run camera calibration (once):
       python calibration/camera_calibration.py

  3. (Optional) Train acoustic classifier:
       python backend/sensors/train_acoustic.py --data_dir data/audio

  4. Start backend over HTTPS:
       python run_https.py

  5. Connect phone to same Wi-Fi → open https://<YOUR_LAPTOP_IP>:8000/app/index.html
""")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
