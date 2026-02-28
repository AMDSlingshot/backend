"""
backend/sensors/depth_pipeline.py

Channel 2: Camera + IMU → Metric 3D Point Cloud

Pipeline:
    RGB Frame → Depth Anything V2 Small → Relative depth map
    IMU data  → SLAMWrapper               → Scale Anchor 1 (S_imu)
    Ground plane → RANSAC fit             → Scale Anchor 2 (S_ground)
    GPS + Optical flow                    → Scale Anchor 3 (S_motion)

    Metric depth = relative_depth × fused_scale
    3D Point cloud → rut depth (mm), cross-section profile

GPU: NVIDIA CUDA (RTX 4050). Falls back to CPU gracefully.
Model: depth-anything/Depth-Anything-V2-Small-hf (25M params, ~30ms/frame on GPU)
"""

import numpy as np
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Optional heavy imports — lazy-loaded to allow testing without GPU ──────────
_TORCH_AVAILABLE = False
_OPEN3D_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available — depth pipeline will not function.")

try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("Open3D not available — point cloud extraction disabled.")


# Scale fusion weights (must sum to 1.0)
SCALE_WEIGHTS = {
    "imu":    0.50,
    "ground": 0.30,
    "motion": 0.20,
}

# Valid scale range (metres per depth unit)
SCALE_VALID_MIN = 0.1
SCALE_VALID_MAX = 100.0

# Sanity clamp: real road ruts never exceed ~80mm even on worst rural roads
# Values above this indicate uncalibrated depth or non-road surfaces
MAX_RUT_DEPTH_MM = 100.0


class MetricDepthPipeline:
    """
    Converts smartphone RGB video + IMU → metric 3D point cloud.

    Three independent scale anchors fuse to recover absolute depth:
      Anchor 1 (S_imu):    ORB-SLAM3 or IMU double-integration (weight 0.50)
      Anchor 2 (S_ground): Known camera height + RANSAC road plane (weight 0.30)
      Anchor 3 (S_motion): GPS speed + optical flow pixel displacement (weight 0.20)

    Instantiate once; call process_frame() per video frame.
    """

    def __init__(
        self,
        camera_height_m: float = 1.2,
        device: Optional[str] = None,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    ):
        """
        Args:
            camera_height_m: Physical height of phone above road (metres).
                             Measure with a ruler before driving. Critical for Anchor 2.
            device:          "cuda" | "cpu" | None (auto-detect).
            model_id:        HuggingFace model identifier for Depth Anything V2.
        """
        self.camera_height = camera_height_m
        self.model_id = model_id

        # Device selection
        if device is None:
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"DepthPipeline using device: {self.device}")

        # Lazy-load depth model
        self._depth_pipeline = None
        self._model_loaded = False

        # Previous frame for optical flow (Anchor 3)
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_depth: Optional[np.ndarray] = None

        # Camera intrinsics (updated via set_intrinsics)
        self._intrinsics: Optional[dict] = None

    def load_model(self):
        """Load Depth Anything V2 model. Called lazily on first use."""
        if self._model_loaded:
            return
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. Cannot load depth model.")

        try:
            from transformers import pipeline as hf_pipeline

            gpu_device = 0 if self.device == "cuda" else -1
            self._depth_pipeline = hf_pipeline(
                task="depth-estimation",
                model=self.model_id,
                device=gpu_device,
            )
            self._model_loaded = True
            logger.info(f"Depth Anything V2 loaded: {self.model_id}")
        except Exception as exc:
            logger.error(f"Failed to load depth model: {exc}")
            raise

    def set_intrinsics(self, intrinsics: dict):
        """
        Set camera intrinsics from calibration file.

        Args:
            intrinsics: dict with keys fx, fy, cx, cy (pixels).
        """
        self._intrinsics = intrinsics

    def get_relative_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run Depth Anything V2 on a single BGR frame.

        Args:
            frame: BGR numpy array (H, W, 3).

        Returns:
            Depth map (H, W) normalised to [0, 1].
            Larger values = farther away in relative terms.
            Returns None if model not loaded.
        """
        if not self._model_loaded:
            try:
                self.load_model()
            except Exception:
                return None

        try:
            import cv2
            from PIL import Image

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self._depth_pipeline(img_pil)
            depth_raw = np.array(result["depth"], dtype=np.float32)

            # Normalise to [0, 1]
            dmin, dmax = depth_raw.min(), depth_raw.max()
            if dmax - dmin < 1e-8:
                return np.zeros_like(depth_raw)
            depth_norm = (depth_raw - dmin) / (dmax - dmin)
            return depth_norm

        except Exception as exc:
            logger.error(f"Depth estimation failed: {exc}")
            return None

    # ── Scale Anchor 2: Ground Plane ────────────────────────────────────────

    def _recover_scale_ground_plane(
        self, depth_relative: np.ndarray, h: int, w: int
    ) -> Optional[float]:
        """
        Anchor 2: Use known camera height to derive metric scale.

        Road surface pixels (lower 60% of frame) form a plane at roughly
        `camera_height` metres from the camera. Median depth in that region
        maps to camera_height → scale = camera_height / median(road_depth).
        """
        # Road region = lower 60% of frame (above horizon = sky/obstacles)
        road_region = depth_relative[int(h * 0.4):, :]
        road_depth_median = float(np.median(road_region))

        if road_depth_median < 0.01:
            return None  # Degenerate frame (e.g. camera looking at sky)

        scale = self.camera_height / road_depth_median
        return float(scale) if SCALE_VALID_MIN < scale < SCALE_VALID_MAX else None

    # ── Scale Anchor 3: GPS + Optical Flow ──────────────────────────────────

    def _recover_scale_optical_flow(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray,
        gps_speed_ms: float,
        fps: float = 30.0,
    ) -> Optional[float]:
        """
        Anchor 3: GPS speed + dense optical flow → pixel-to-metre ratio.

        distance_per_frame = gps_speed / fps (metres)
        pixel_displacement  = median optical flow magnitude (pixels)
        scale = distance_per_frame / pixel_displacement_converted
        """
        if gps_speed_ms < 0.5:  # Nearly stationary — flow is noise
            return None

        import cv2
        prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        # Forward motion shows as y-component flow in forward-facing camera
        h = frame_prev.shape[0]
        road_flow_y = flow[int(h * 0.4):, :, 1]  # y-component, road region
        median_pixel_disp = float(np.median(np.abs(road_flow_y)))

        if median_pixel_disp < 1.0:
            return None  # Too little motion to estimate

        distance_per_frame_m = gps_speed_ms / fps
        # Scale approximation: convert pixel displacement to metric
        scale = distance_per_frame_m / (median_pixel_disp * 0.001)
        return float(scale) if SCALE_VALID_MIN < scale < SCALE_VALID_MAX else None

    # ── Scale Fusion ─────────────────────────────────────────────────────────

    def fuse_scales(
        self,
        s_imu: Optional[float],
        s_ground: Optional[float],
        s_motion: Optional[float],
    ) -> float:
        """
        Weighted fusion of three independent scale anchors.

        Any anchor that is None or out of valid range is dropped.
        Remaining weights are renormalised to sum to 1.0.

        Raises ValueError if ALL anchors are invalid.
        """
        candidates = [
            ("imu",    s_imu),
            ("ground", s_ground),
            ("motion", s_motion),
        ]

        valid = [
            (name, s)
            for name, s in candidates
            if s is not None and SCALE_VALID_MIN < s < SCALE_VALID_MAX
        ]

        if not valid:
            raise ValueError(
                "All scale anchors failed. Check sensor data. "
                "Need at least one of: IMU/SLAM, camera height, GPS speed."
            )

        total_w = sum(SCALE_WEIGHTS[name] for name, _ in valid)
        fused = sum(SCALE_WEIGHTS[name] / total_w * s for name, s in valid)

        logger.debug(
            f"Scale fusion: imu={s_imu}, ground={s_ground}, "
            f"motion={s_motion} → fused={fused:.3f} "
            f"(using {[n for n, _ in valid]})"
        )
        return float(fused)

    # ── Point Cloud Generation ────────────────────────────────────────────────

    def depth_to_pointcloud(
        self,
        frame: np.ndarray,
        metric_depth: np.ndarray,
        intrinsics: Optional[dict] = None,
    ):
        """
        Back-project depth map to 3D point cloud using pinhole camera model.

        Args:
            frame:        BGR frame (H, W, 3).
            metric_depth: Metric depth map (H, W) in metres.
            intrinsics:   Camera intrinsics dict (fx, fy, cx, cy).
                          Falls back to self._intrinsics or sensible defaults.

        Returns:
            open3d.geometry.PointCloud, or None if Open3D unavailable.
        """
        if not _OPEN3D_AVAILABLE:
            logger.warning("Open3D not available — skipping point cloud generation.")
            return None

        cam = intrinsics or self._intrinsics or {}
        h, w = metric_depth.shape

        fx = cam.get("fx", w * 0.8)
        fy = cam.get("fy", h * 0.8)
        cx = cam.get("cx", w / 2.0)
        cy = cam.get("cy", h / 2.0)

        # Back-projection
        x_idx, y_idx = np.meshgrid(np.arange(w), np.arange(h))
        z = metric_depth
        x = (x_idx - cx) * z / fx
        y = (y_idx - cy) * z / fy

        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = (frame.reshape(-1, 3) / 255.0)[:, ::-1]  # BGR → RGB

        # Filter: keep only road-plausible depths
        valid_mask = (z.flatten() > 0.3) & (z.flatten() < 8.0)
        points = points[valid_mask]
        colors = colors[valid_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    # ── Rut Depth Extraction ─────────────────────────────────────────────────

    def extract_rut_depth(self, pcd) -> dict:
        """
        Extract road distress metrics from 3D point cloud.

        Algorithm:
          - Select road-corridor points (in front, within ±2m lateral)
          - Slice into 20 longitudinal strips
          - In each strip find transverse cross-section profile
          - Rut depth = max depression below fitted baseline (90th percentile)

        Returns:
            dict with rut_depth_mm, severity, confidence.
        """
        if not _OPEN3D_AVAILABLE or pcd is None:
            return {"rut_depth_mm": None, "severity": "Unknown", "confidence": "low"}

        points = np.asarray(pcd.points)

        # Road corridor: 0.5m–5m ahead, ±2m lateral
        road_mask = (
            (points[:, 2] > 0.5)
            & (points[:, 2] < 5.0)
            & (np.abs(points[:, 0]) < 2.0)
        )
        road_pts = points[road_mask]

        if len(road_pts) < 100:
            return {"rut_depth_mm": None, "severity": "Unknown", "confidence": "low"}

        # Longitudinal slices
        z_edges = np.linspace(road_pts[:, 2].min(), road_pts[:, 2].max(), 21)
        rut_depths = []

        for z_lo, z_hi in zip(z_edges[:-1], z_edges[1:]):
            slice_pts = road_pts[(road_pts[:, 2] >= z_lo) & (road_pts[:, 2] < z_hi)]
            if len(slice_pts) < 10:
                continue

            # Transverse profile: y-values (up/down in camera frame)
            y_vals = slice_pts[:, 1]
            baseline = float(np.percentile(y_vals, 90))  # Road surface level
            depressions = baseline - y_vals
            pos_depressions = depressions[depressions > 0]

            if len(pos_depressions) > 0:
                rut_depths.append(float(np.max(pos_depressions)))

        if not rut_depths:
            return {"rut_depth_mm": 0.0, "severity": "None/Slight", "confidence": "low"}

        rut_depth_m = float(np.median(rut_depths))
        rut_depth_mm = rut_depth_m * 1000.0

        # Sanity clamp: values above MAX_RUT_DEPTH_MM indicate bad scale or non-road
        if rut_depth_mm > MAX_RUT_DEPTH_MM:
            logger.warning(
                f"Rut depth {rut_depth_mm:.1f}mm exceeds {MAX_RUT_DEPTH_MM}mm — "
                f"clamping (likely uncalibrated depth or non-road surface)"
            )
            rut_depth_mm = MAX_RUT_DEPTH_MM

        # IRC:SP:20 severity thresholds
        if rut_depth_mm < 10:
            severity = "None/Slight"
        elif rut_depth_mm < 20:
            severity = "Moderate"
        else:
            severity = "Severe"

        confidence = "high" if len(rut_depths) >= 10 else "medium"

        return {
            "rut_depth_mm": round(rut_depth_mm, 1),
            "severity": severity,
            "confidence": confidence,
            "samples": len(rut_depths),
        }

    # ── Main Processing Entry Point ──────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        gps_speed_ms: float = 0.0,
        imu_scale: Optional[float] = None,
        fps: float = 30.0,
    ) -> dict:
        """
        Full pipeline: BGR frame → metric depth → point cloud → road metrics.

        Args:
            frame:         BGR numpy array.
            gps_speed_ms:  Current vehicle speed in m/s (from GPS).
            imu_scale:     Scale from SLAMWrapper (Anchor 1), or None.
            fps:           Camera frame rate.

        Returns:
            dict with keys: metric_depth (np.ndarray), point_cloud,
                            rut_depth_mm, scale_used, scale_anchors.
        """
        h, w = frame.shape[:2]

        # Step 1: Relative depth
        depth_rel = self.get_relative_depth(frame)
        if depth_rel is None:
            return {"error": "depth_estimation_failed"}

        # Step 2: Scale anchors
        s_ground = self._recover_scale_ground_plane(depth_rel, h, w)
        s_motion = None
        if self._prev_frame is not None:
            s_motion = self._recover_scale_optical_flow(
                self._prev_frame, frame, gps_speed_ms, fps
            )

        # Step 3: Fuse scales
        try:
            scale = self.fuse_scales(imu_scale, s_ground, s_motion)
        except ValueError as exc:
            logger.warning(f"Scale fusion failed: {exc}")
            # Last resort: use camera height directly as scale
            scale = self.camera_height / 0.5  # Assume median road depth is 0.5 relative

        # Step 4: Apply scale
        metric_depth = depth_rel * scale

        # Step 5: Point cloud
        pcd = self.depth_to_pointcloud(frame, metric_depth)

        # Step 6: Road metrics
        rut_info = self.extract_rut_depth(pcd)

        # Update state for next frame
        self._prev_frame = frame.copy()
        self._prev_depth = depth_rel.copy()

        return {
            "metric_depth": metric_depth,
            "point_cloud": pcd,
            "rut_depth_mm": rut_info.get("rut_depth_mm"),
            "rut_severity": rut_info.get("severity"),
            "rut_confidence": rut_info.get("confidence"),
            "scale_used": round(scale, 4),
            "scale_anchors": {
                "imu": imu_scale,
                "ground": s_ground,
                "motion": s_motion,
            },
        }
