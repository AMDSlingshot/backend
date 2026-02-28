"""
backend/segment_manager.py

Orchestrates incoming data streams from a smartphone WebSocket, grouping them
into ~100-metre physical segments based on GPS coordinates.

Once a 100m segment is complete, it yields the aggregated buffer for the
agents to process.
"""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Earth radius in kilometres
R_EARTH = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in kilometres."""
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) * math.sin(dLon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_EARTH * c


class SegmentManager:
    """
    Manages live data accumulation for a single ongoing recording session.
    Buffers incoming packets until the physical vehicle has moved SEGMENT_LENGTH_M.
    """

    def __init__(self, segment_length_m: float = 100.0):
        self.segment_length_km = segment_length_m / 1000.0
        
        # Current accumulating segment
        self._current_segment_id = 1
        self._reset_buffer()
        
        # State tracking
        self._last_gps: Optional[dict] = None
        self._distance_accumulated_km = 0.0

        # Ready queue for the pipeline
        self.ready_segments = []

    def _reset_buffer(self):
        """Clear buffers for the new physical segment."""
        self._buffer = {
            "segment_id": f"seg_{self._current_segment_id:04d}",
            "gps_buffer": [],
            "imu_buffer": [],
            "frames": [],      # Base64 strings or numpy arrays
            "audio_buffer": []
        }
        self._distance_accumulated_km = 0.0

    def ingest_packet(self, packet: dict):
        """
        Takes an incoming dictionary packet from the WebSocket.
        Expected format: {"type": "gps"|"imu"|"camera"|"audio", "data": {...}}
        """
        p_type = packet.get("type")
        data = packet.get("data", {})

        if p_type == "gps":
            self._handle_gps(data)
        elif p_type == "imu":
            self._buffer["imu_buffer"].append(data)
        elif p_type == "camera":
            # For efficiency in a real system, you'd decode Base64 to cv2 here.
            # We assume it's pre-processed or we decode it later.
            import cv2
            import numpy as np
            import base64
            
            b64_img = data.get("image", "")
            if b64_img:
                try:
                    # Strip data URI header if present
                    if "," in b64_img:
                        b64_img = b64_img.split(",")[1]
                    img_data = base64.b64decode(b64_img)
                    np_arr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._buffer["frames"].append(frame)
                except Exception as e:
                    logger.warning(f"Failed to decode image frame: {e}")

        elif p_type == "audio":
            self._buffer["audio_buffer"].append(data)

    def _handle_gps(self, gps_data: dict):
        """Process a GPS point and check if we've crossed the segment threshold."""
        lat = gps_data.get("lat")
        lng = gps_data.get("lng")
        
        if lat is None or lng is None:
            return

        self._buffer["gps_buffer"].append(gps_data)

        if self._last_gps is not None:
            dist = haversine_distance(
                self._last_gps["lat"], self._last_gps["lng"],
                lat, lng
            )
            self._distance_accumulated_km += dist
            
            # Check if we've reached 100 meters
            if self._distance_accumulated_km >= self.segment_length_km:
                self._finalize_segment()

        self._last_gps = {"lat": lat, "lng": lng}

    def _finalize_segment(self):
        """Package the current buffer and put it in the ready queue."""
        segment = self._buffer.copy()
        
        # Assign an average speed and midpoint GPS for the pipeline
        if segment["gps_buffer"]:
            mid_idx = len(segment["gps_buffer"]) // 2
            segment["gps"] = segment["gps_buffer"][mid_idx]
            
            # Calculate average speed if available in GPS data (m/s)
            speeds = [p.get("speed", 0) for p in segment["gps_buffer"] if p.get("speed") is not None]
            if speeds:
                segment["avg_speed_ms"] = sum(speeds) / len(speeds)
                segment["avg_speed_kmh"] = segment["avg_speed_ms"] * 3.6
            else:
                segment["avg_speed_ms"] = 0.0
                segment["avg_speed_kmh"] = 0.0
                
            # Use timestamp from mid point
            segment["timestamp"] = segment["gps"].get("timestamp", 0)
        else:
            segment["gps"] = {"lat": 0.0, "lng": 0.0}
            segment["avg_speed_ms"] = 0.0
            segment["avg_speed_kmh"] = 0.0
            segment["timestamp"] = 0

        segment["length_km"] = self._distance_accumulated_km
        
        self.ready_segments.append(segment)
        logger.info(f"Segment {segment['segment_id']} assembled with {len(segment['imu_buffer'])} IMU readings and {len(segment['frames'])} frames.")
        
        # Setup for the next physical segment
        self._current_segment_id += 1
        self._reset_buffer()

    def get_ready_segments(self) -> list:
        """Returns all fully assembled 100m segments, clearing the queue."""
        segments = self.ready_segments.copy()
        self.ready_segments.clear()
        return segments

    def flush(self):
        """Force finalize whatever is currently in the buffer. Used at the end of a session."""
        if len(self._buffer["imu_buffer"]) > 0 or len(self._buffer["gps_buffer"]) > 0:
            self._finalize_segment()
