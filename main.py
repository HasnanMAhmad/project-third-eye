#!/usr/bin/env python3
"""
========================================================================================
Project Third Eye - Real-Time Edge Spatial Hazard Detection & Navigation Assistant
========================================================================================
A real-time edge computer vision and spatial audio navigation assistant designed for
visually impaired individuals using laptop webcams.

Core Components:
1. High-throughput OpenCV webcam acquisition pipeline (1280x720 @ 30 FPS).
2. Ultralytics YOLOv8n object detection engine with class filtering and conf=0.50.
3. 3-Way vertical spatial corridor partitioning (Left, Center collision vector, Right).
4. Monocular geometry and bounding box occupancy ratio-based distance estimation.
5. Thread-safe, non-blocking asynchronous TTS audio engine (pyttsx3) with 1.8s cooldown.
6. Real-time AR/HUD overlay with corridor lines, color-coded bounding boxes, and top status banner.
========================================================================================
"""

import sys
import time
import math
import queue
import threading
import argparse
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

# Ultralytics YOLOv8 import
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Text-To-Speech engine import
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


# ======================================================================================
# CONSTANTS & CONFIGURATION
# ======================================================================================
TARGET_CLASSES = {
    "person", "chair", "couch", "bottle", "dining table",
    "door", "pole", "laptop", "bicycle", "motorcycle", "car", "bus"
}

# Real-World Heights for COCO Classes (in meters) for Pinhole Metric Distance Estimation
CLASS_REAL_HEIGHTS: Dict[str, float] = {
    "person": 1.70,
    "chair": 0.85,
    "couch": 0.85,
    "bottle": 0.22,
    "dining table": 0.75,
    "door": 2.00,
    "pole": 2.50,
    "laptop": 0.25,
    "bicycle": 1.00,
    "motorcycle": 1.10,
    "car": 1.50,
    "bus": 3.20,
}
DEFAULT_REAL_HEIGHT = 1.00  # Fallback real-world height for unlisted classes

# Camera Geometry (assuming standard 60-degree vertical FOV)
CAMERA_FOV_V_DEG = 60.0

# Metric Distance Thresholds (in meters)
CRITICAL_DISTANCE_METERS = 1.20   # Z < 1.2m -> CRITICAL HAZARD
WARNING_DISTANCE_METERS = 2.50    # 1.2m <= Z <= 2.5m -> WARNING HAZARD
                                  # Z > 2.5m -> SAFE / FAR

# Legacy Thresholds based on Bounding Box Height / Frame Height (retained for backward compatibility)
CRITICAL_RATIO_THRESHOLD = 0.50
WARNING_RATIO_THRESHOLD = 0.25

# Spatial Corridor Boundary Ratios
LEFT_CORRIDOR_RATIO = 0.33
RIGHT_CORRIDOR_RATIO = 0.66

# Audio Throttle
AUDIO_COOLDOWN_SECONDS = 1.8

# Color Palette (BGR format for OpenCV)
COLOR_CRITICAL = (0, 0, 240)       # Vivid Red
COLOR_WARNING = (0, 215, 255)      # Bright Yellow
COLOR_SAFE = (50, 205, 50)         # Forest Green
COLOR_CORRIDOR_LINE = (180, 180, 180)  # Light Gray
COLOR_HUD_TEXT = (255, 255, 255)   # White
COLOR_DARK_BG = (20, 20, 20)       # Near Black


# ======================================================================================
# PINHOLE METRIC DISTANCE CALCULATION
# ======================================================================================
def compute_focal_length_y(frame_height: int, fov_v_deg: float = CAMERA_FOV_V_DEG) -> float:
    """
    Computes vertical focal length in pixels assuming pinhole camera model:
    f_y = (frame_height / 2) / tan(fov_v / 2)
    """
    fov_v_rad = math.radians(fov_v_deg)
    return (frame_height / 2.0) / math.tan(fov_v_rad / 2.0)


def estimate_metric_distance(
    bbox_height_px: float,
    class_name: str,
    frame_height: int,
    fov_v_deg: float = CAMERA_FOV_V_DEG,
) -> float:
    """
    Pinhole metric distance function:
    Z = (f_y * H_real) / bbox_height_px
    """
    h_real = CLASS_REAL_HEIGHTS.get(class_name.lower(), DEFAULT_REAL_HEIGHT)
    f_y = compute_focal_length_y(frame_height, fov_v_deg)
    h_px = max(1.0, float(bbox_height_px))
    return float((f_y * h_real) / h_px)



# ======================================================================================
# THREAD-SAFE NON-BLOCKING TTS AUDIO ENGINE
# ======================================================================================
class AudioNavigationEngine:
    """
    Asynchronous, non-blocking Text-To-Speech engine using a dedicated background daemon
    thread and queue to ensure video capture and inference FPS never drop or stutter.
    Supports priority alert preemption for CRITICAL hazards and STOP directives.
    """
    def __init__(self, cooldown: float = AUDIO_COOLDOWN_SECONDS):
        self.cooldown = cooldown
        self.message_queue: queue.Queue = queue.Queue(maxsize=4)
        self.last_spoken_time: float = 0.0
        self.last_enqueued_phrase: str = ""
        self.lock = threading.Lock()
        self.is_running = True
        self.sapi_voice = None
        self.pyttsx3_engine = None

        # Attempt to bind direct native Windows SAPI.SpVoice for zero-latency async speech
        try:
            import win32com.client
            self.sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
            self.sapi_voice.Rate = 1
            self.sapi_voice.Volume = 100
        except Exception:
            self.sapi_voice = None

        # Background worker fallback thread (for pyttsx3 on non-SAPI environments)
        if self.sapi_voice is None:
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="TTS-Worker-Thread")
            self.worker_thread.start()
        else:
            self.worker_thread = None

    def _worker_loop(self):
        """Worker thread entry point for speech synthesis fallback."""
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        if pyttsx3 is not None:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self.pyttsx3_engine.setProperty('rate', 165)
                self.pyttsx3_engine.setProperty('volume', 1.0)
            except Exception as e:
                print(f"[TTS AudioEngine] Warning: pyttsx3 init failed: {e}", file=sys.stderr)
                self.pyttsx3_engine = None

        while self.is_running:
            try:
                phrase = self.message_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.pyttsx3_engine is not None:
                try:
                    self.pyttsx3_engine.say(phrase)
                    self.pyttsx3_engine.runAndWait()
                except Exception as err:
                    print(f"[TTS AudioEngine] Playback error: {err}", file=sys.stderr)
            else:
                print(f"\a[TTS Voice Output]: \"{phrase}\"", flush=True)

            with self.lock:
                self.last_spoken_time = time.time()

            self.message_queue.task_done()

    def announce(self, phrase: str, force: bool = False, hazard_level: str = "SAFE") -> bool:
        """
        Submits a speech command if cooldown has expired (or if forced/critical).
        If hazard_level == 'CRITICAL' or 'STOP' is in phrase (case-insensitive):
        - Bypasses cooldown check
        - Purges message queue
        - Interrupts any ongoing SAPI or pyttsx3 speech immediately
        """
        if not phrase:
            return False

        is_critical = (
            (hazard_level is not None and str(hazard_level).strip().upper() == "CRITICAL")
            or ("STOP" in phrase.upper())
            or force
        )

        current_time = time.time()
        with self.lock:
            if not is_critical and (current_time - self.last_spoken_time < self.cooldown):
                return False
            self.last_spoken_time = current_time

        self.last_enqueued_phrase = phrase

        # Direct Windows SAPI Asynchronous Execution (Native OS background audio)
        if self.sapi_voice is not None:
            try:
                # Flag 1 = SVSFlagsAsync
                # Flag 3 = SVSFPurgeBeforeSpeak | SVSFlagsAsync (interrupts ongoing speech instantly)
                flags = 3 if is_critical else 1
                self.sapi_voice.Speak(phrase, flags)
                with self.lock:
                    self.last_spoken_time = time.time()
                return True
            except Exception as err:
                print(f"[TTS AudioEngine] SAPI error: {err}", file=sys.stderr)

        # Fallback to queue worker
        # 1. Purge queued messages immediately if critical
        if is_critical:
            with self.lock:
                while not self.message_queue.empty():
                    try:
                        self.message_queue.get_nowait()
                        self.message_queue.task_done()
                    except (queue.Empty, ValueError):
                        break
            if self.pyttsx3_engine is not None:
                try:
                    self.pyttsx3_engine.stop()
                except Exception:
                    pass
        else:
            # Drain older unplayed non-critical messages so queue doesn't back up
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.task_done()
                except (queue.Empty, ValueError):
                    break

        try:
            self.message_queue.put_nowait(phrase)
            return True
        except queue.Full:
            return False

    def shutdown(self):
        """Cleanly terminates the background audio daemon."""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.5)


# ======================================================================================
# SPATIAL HAZARD & GEOMETRY ANALYZER
# ======================================================================================
class SpatialHazardAnalyzer:
    """
    Computes spatial partitioning, corridor occupancy, monocular distance estimation,
    threat classification, and directional navigation directives.
    """
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.left_bound = int(frame_width * LEFT_CORRIDOR_RATIO)
        self.right_bound = int(frame_width * RIGHT_CORRIDOR_RATIO)

    def update_frame_dimensions(self, width: int, height: int):
        self.frame_width = width
        self.frame_height = height
        self.left_bound = int(width * LEFT_CORRIDOR_RATIO)
        self.right_bound = int(width * RIGHT_CORRIDOR_RATIO)

    def classify_corridor(self, cx: float) -> str:
        """Determines whether a coordinate belongs to LEFT, CENTER, or RIGHT corridor."""
        if cx < self.left_bound:
            return "LEFT"
        elif cx <= self.right_bound:
            return "CENTER"
        else:
            return "RIGHT"

    def analyze_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates detected objects across corridors using pinhole metric distances and executes
        the Third Eye Decision Matrix.

        Decision Matrix Rules:
        - Primary obstacle in Center & Critical (distance < 1.2m or ratio > 0.50):
            -> Voice: "STOP! Obstacle directly ahead!"
            -> Status: "STATUS: CRITICAL HAZARD DETECTED"
        - Primary obstacle in Center & Warning (1.2m <= distance <= 2.5m or 0.25 <= ratio <= 0.50):
            -> If Left Corridor is clearer than Right:
                -> Voice: "Obstacle ahead. Veer Left"
                -> Status: "STATUS: WARNING - OBSTACLE AHEAD [VEER LEFT]"
            -> Else:
                -> Voice: "Obstacle ahead. Veer Right"
                -> Status: "STATUS: WARNING - OBSTACLE AHEAD [VEER RIGHT]"
        - Clear Path:
            -> Voice: Silent
            -> Status: "STATUS: CLEAR PATH AHEAD"
        """
        left_obstacles = []
        center_obstacles = []
        right_obstacles = []

        for det in detections:
            corridor = det["corridor"]
            if corridor == "LEFT":
                left_obstacles.append(det)
            elif corridor == "CENTER":
                center_obstacles.append(det)
            else:
                right_obstacles.append(det)

        # Primary obstacle in center corridor: closest object (min distance_m or max height ratio)
        primary_center = None
        if center_obstacles:
            if any("distance_m" in d for d in center_obstacles):
                primary_center = min(center_obstacles, key=lambda d: d.get("distance_m", float("inf")))
            else:
                primary_center = max(center_obstacles, key=lambda d: d.get("ratio", 0.0))

        # Determine corridor occupancy / clearance scores (inverse distance or ratio)
        def _get_obstacle_threat_score(d: Dict[str, Any]) -> float:
            if "distance_m" in d:
                return 1.0 / max(0.1, float(d["distance_m"]))
            return float(d.get("ratio", 0.0))

        left_threat_score = sum(_get_obstacle_threat_score(d) for d in left_obstacles)
        right_threat_score = sum(_get_obstacle_threat_score(d) for d in right_obstacles)

        audio_cue: Optional[str] = None
        status_text: str = "STATUS: CLEAR PATH AHEAD"
        status_color: Tuple[int, int, int] = COLOR_SAFE
        threat_level: str = "SAFE"

        if primary_center is not None:
            # 1. Metric distance-based classification
            if "distance_m" in primary_center:
                dist_z = primary_center["distance_m"]
                if dist_z < CRITICAL_DISTANCE_METERS:
                    threat_level = "CRITICAL"
                    status_text = "STATUS: CRITICAL HAZARD DETECTED"
                    status_color = COLOR_CRITICAL
                    audio_cue = "STOP! Obstacle directly ahead!"
                elif dist_z <= WARNING_DISTANCE_METERS:
                    threat_level = "WARNING"
                    status_color = COLOR_WARNING
                    if left_threat_score < right_threat_score:
                        veer_direction = "VEER LEFT"
                        audio_cue = "Obstacle ahead. Veer Left"
                    else:
                        veer_direction = "VEER RIGHT"
                        audio_cue = "Obstacle ahead. Veer Right"
                    status_text = f"STATUS: WARNING - OBSTACLE AHEAD [{veer_direction}]"
                else:
                    threat_level = "SAFE"
                    status_text = "STATUS: CLEAR PATH AHEAD"
                    status_color = COLOR_SAFE
                    audio_cue = None

            # 2. Backward-compatible ratio/threat classification (for tests passing synthetic ratio/threat)
            else:
                c_ratio = primary_center.get("ratio", 0.0)
                is_crit = (c_ratio > CRITICAL_RATIO_THRESHOLD) or (primary_center.get("threat") == "CRITICAL")
                is_warn = (c_ratio >= WARNING_RATIO_THRESHOLD) or (primary_center.get("threat") == "WARNING")

                if is_crit:
                    threat_level = "CRITICAL"
                    status_text = "STATUS: CRITICAL HAZARD DETECTED"
                    status_color = COLOR_CRITICAL
                    audio_cue = "STOP! Obstacle directly ahead!"
                elif is_warn:
                    threat_level = "WARNING"
                    status_color = COLOR_WARNING
                    if left_threat_score < right_threat_score:
                        veer_direction = "VEER LEFT"
                        audio_cue = "Obstacle ahead. Veer Left"
                    else:
                        veer_direction = "VEER RIGHT"
                        audio_cue = "Obstacle ahead. Veer Right"
                    status_text = f"STATUS: WARNING - OBSTACLE AHEAD [{veer_direction}]"
                else:
                    threat_level = "SAFE"
                    status_text = "STATUS: CLEAR PATH AHEAD"
                    status_color = COLOR_SAFE
                    audio_cue = None

        return {
            "threat_level": threat_level,
            "hazard_level": threat_level,
            "status_text": status_text,
            "status_color": status_color,
            "audio_cue": audio_cue,
            "primary_center": primary_center,
            "left_count": len(left_obstacles),
            "center_count": len(center_obstacles),
            "right_count": len(right_obstacles),
        }


# ======================================================================================
# HUD RENDERING & VISUALIZATION PIPELINE
# ======================================================================================
def draw_dashed_vertical_line(
    image: np.ndarray,
    x: int,
    y_start: int,
    y_end: int,
    color: Tuple[int, int, int] = COLOR_CORRIDOR_LINE,
    thickness: int = 2,
    dash_length: int = 15,
    gap_length: int = 10,
):
    """Draws a vertical dashed line in the OpenCV canvas."""
    for y in range(y_start, y_end, dash_length + gap_length):
        y2 = min(y + dash_length, y_end)
        cv2.line(image, (x, y), (x, y2), color, thickness, cv2.LINE_AA)


def render_hud(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    fps: float,
    latency_ms: float,
) -> np.ndarray:
    """
    Renders high-contrast AR HUD:
    1. Top HUD Status Banner (Red/Yellow/Green based on threat).
    2. Dashed corridor partition lines and corridor headers.
    3. Color-coded bounding boxes and distance estimation tags.
    4. Diagnostics & performance telemetry footer.
    """
    h, w, _ = frame.shape
    banner_height = 64
    hud = frame.copy()

    # 1. Top HUD Status Banner with semi-transparent background
    banner_color = analysis["status_color"]
    overlay = hud.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_color, -1)
    cv2.addWeighted(overlay, 0.90, hud, 0.10, 0, hud)
    cv2.line(hud, (0, banner_height), (w, banner_height), (255, 255, 255), 2)

    # Banner Text Centering
    status_msg = analysis["status_text"]
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.85
    thickness = 2
    (tw, th), _ = cv2.getTextSize(status_msg, font, font_scale, thickness)
    tx = max(20, (w - tw) // 2)
    ty = (banner_height + th) // 2
    cv2.putText(hud, status_msg, (tx, ty), font, font_scale, COLOR_HUD_TEXT, thickness, cv2.LINE_AA)

    # 2. Corridor Partition Lines
    left_x = int(w * LEFT_CORRIDOR_RATIO)
    right_x = int(w * RIGHT_CORRIDOR_RATIO)

    draw_dashed_vertical_line(hud, left_x, banner_height + 5, h - 35, COLOR_CORRIDOR_LINE, thickness=2)
    draw_dashed_vertical_line(hud, right_x, banner_height + 5, h - 35, COLOR_CORRIDOR_LINE, thickness=2)

    # Corridor Title Labels at Top (below banner)
    header_y = banner_height + 28
    corridor_labels = [
        ("LEFT CORRIDOR", int(left_x * 0.5)),
        ("CENTER CORRIDOR (COLLISION VECTOR)", int((left_x + right_x) * 0.5)),
        ("RIGHT CORRIDOR", int((right_x + w) * 0.5)),
    ]

    for label, center_x in corridor_labels:
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        lx = max(5, center_x - lw // 2)
        # Background badge for text legibility
        cv2.rectangle(hud, (lx - 6, header_y - lh - 4), (lx + lw + 6, header_y + 4), (20, 20, 20), -1)
        cv2.putText(hud, label, (lx, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # 3. Detections & Color-Coded Bounding Boxes
    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = box
        ratio = det["ratio"]
        cls_name = det["class_name"]
        conf = det["confidence"]
        dist_desc = det["dist_desc"]
        color = det["color"]

        # Bounding Box
        cv2.rectangle(hud, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Distance & Threat Badge Label
        badge_text = f"{cls_name.upper()} {conf:.2f} | {dist_desc}"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        badge_y1 = max(y1 - bh - 8, banner_height + 35)
        badge_y2 = badge_y1 + bh + 8
        badge_x2 = min(x1 + bw + 10, w - 2)

        cv2.rectangle(hud, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            hud,
            badge_text,
            (x1 + 5, badge_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0) if color == COLOR_WARNING else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 4. Telemetry Footer (FPS, Latency, Audio status)
    footer_y = h - 12
    audio_status = analysis.get("audio_cue") or "Silent (Path Clear)"
    footer_text = f"FPS: {fps:4.1f} | Latency: {latency_ms:4.1f}ms | Audio: {audio_status} | Press 'q' to Quit"
    cv2.rectangle(hud, (0, h - 30), (w, h), COLOR_DARK_BG, -1)
    cv2.putText(hud, footer_text, (15, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    return hud


# ======================================================================================
# THREADED CAMERA INGESTION PIPELINE
# ======================================================================================
class ThreadedCameraStream:
    """
    Non-blocking, threaded camera ingestion worker running cv2.VideoCapture in a background thread.
    Uses a queue.Queue(maxsize=1) that always drops old frames so the main processing loop
    only reads the freshest available camera frame, eliminating buffer latency.
    """
    def __init__(self, source: Any = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.is_running = True
        self.grabbed = False
        self.worker_thread = None

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.grabbed = True
                self.frame_queue.put(frame)
                self.worker_thread = threading.Thread(
                    target=self._capture_worker,
                    daemon=True,
                    name="Camera-Worker-Thread"
                )
                self.worker_thread.start()

    def _capture_worker(self):
        """Continuously reads frames from cv2.VideoCapture and drops stale frames in the queue."""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.grabbed = False
                break
            self.grabbed = True

            # Drop older frame if present in queue to guarantee freshest frame delivery
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def isOpened(self) -> bool:
        return self.cap.isOpened() and self.grabbed

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the freshest frame from the queue."""
        if not self.is_running or not self.grabbed:
            return False, None
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None

    def get(self, prop_id: int) -> float:
        return self.cap.get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        return self.cap.set(prop_id, value)

    def release(self):
        """Terminates background worker thread and releases VideoCapture."""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.5)
        if self.cap and self.cap.isOpened():
            self.cap.release()


# ======================================================================================
# SYNTHETIC TEST FRAME GENERATOR (HEADLESS / TEST FALLBACK)
# ======================================================================================
class SyntheticCameraStream:
    """
    Generates a deterministic synthetic video stream when no physical camera is attached,
    ensuring continuous pipeline testability and zero-crash execution.
    """
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_idx = 0
        self.start_time = time.time()

    def isOpened(self) -> bool:
        return True

    def read(self) -> Tuple[bool, np.ndarray]:
        self.frame_idx += 1
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Gradient room background
        for i in range(self.height):
            c = int(40 + (i / self.height) * 45)
            canvas[i, :] = (c, c, c + 15)

        # Oscillating synthetic obstacle in Center Corridor to trigger warning/critical test cases
        t = self.frame_idx * 0.05
        cycle = (np.sin(t) + 1.0) / 2.0  # 0.0 to 1.0
        # Obstacle height varies between 0.15 and 0.65 of frame height
        box_h = int(self.height * (0.15 + cycle * 0.50))
        box_w = int(box_h * 0.6)
        cx = int(self.width * 0.50 + np.cos(t * 0.5) * 50)
        cy = int(self.height * 0.60)

        x1 = max(0, cx - box_w // 2)
        y1 = max(0, cy - box_h // 2)
        x2 = min(self.width, x1 + box_w)
        y2 = min(self.height, y1 + box_h)

        # Draw simulated human obstacle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (90, 110, 140), -1)
        cv2.circle(canvas, ((x1 + x2) // 2, y1 + 35), 25, (160, 180, 200), -1)
        cv2.putText(canvas, "[SIMULATED OBSTACLE]", (x1 + 10, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        time.sleep(1.0 / self.fps)
        return True, canvas

    def release(self):
        pass


# ======================================================================================
# MAIN EXECUTION PIPELINE
# ======================================================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project Third Eye - Edge Spatial Hazard Detection & Navigation Assistant"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="System camera index (default: 0).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional path to a recorded video file or image instead of webcam.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Detection confidence threshold (default: 0.50).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model checkpoint name or path (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--simulated-fallback",
        action="store_true",
        help="Automatically fallback to simulated stream if camera 0 is unavailable.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying cv2.imshow window (for automated testing or servers).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting cleanly (useful for test validation).",
    )
    return parser.parse_args()


def initialize_camera(args: argparse.Namespace) -> Tuple[Any, bool]:
    """
    Initializes camera feed with ThreadedCameraStream and fallback to SyntheticCameraStream.
    """
    source = args.source if args.source is not None else args.camera
    print(f"[Project Third Eye] Initializing video capture source: {source}...")

    stream = ThreadedCameraStream(source=source, width=1280, height=720, fps=30)

    if not stream.isOpened():
        print(f"[Project Third Eye] ERROR: Failed to open video source: {source}", file=sys.stderr)
        stream.release()
        if args.simulated_fallback or args.source is None:
            print("[Project Third Eye] Notice: Falling back to SyntheticCameraStream for testing/verification.")
            return SyntheticCameraStream(width=1280, height=720, fps=30), True
        return None, False

    actual_w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = stream.get(cv2.CAP_PROP_FPS)
    if actual_w <= 0:
        actual_w, actual_h = 1280, 720
    print(f"[Project Third Eye] Camera successfully online: {actual_w}x{actual_h} @ ~{actual_fps:.1f} FPS (Threaded).")
    return stream, False


def run_pipeline():
    args = parse_arguments()
    print("=" * 76)
    print(" PROJECT THIRD EYE - REAL-TIME EDGE SPATIAL HAZARD DETECTION SYSTEM")
    print("=" * 76)

    # 1. Initialize Audio Navigation Daemon
    audio_engine = AudioNavigationEngine(cooldown=AUDIO_COOLDOWN_SECONDS)
    backend_name = "Native Windows SAPI.SpVoice (Zero-Latency Async)" if audio_engine.sapi_voice else "Daemon Worker Thread"
    print(f"[Audio Engine] Initialized {backend_name} with {AUDIO_COOLDOWN_SECONDS}s cooldown.", flush=True)

    # 2. Load YOLOv8 Model
    print(f"[YOLOv8 Engine] Loading model '{args.model}'...")
    if YOLO is None:
        print("[YOLOv8 Engine] ERROR: 'ultralytics' library is not installed.", file=sys.stderr)
        sys.exit(1)

    try:
        model = YOLO(args.model)
        print(f"[YOLOv8 Engine] Successfully loaded '{args.model}'.")
    except Exception as e:
        print(f"[YOLOv8 Engine] Failed to load model '{args.model}': {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Initialize Video Feed
    # Default simulated-fallback to True so headless/camera-less environments execute gracefully
    if args.source is None:
        args.simulated_fallback = True

    cap, is_simulated = initialize_camera(args)
    if cap is None:
        print("[Project Third Eye] Execution aborted due to camera acquisition failure.", file=sys.stderr)
        audio_engine.shutdown()
        sys.exit(1)

    # Setup Spatial Analyzer
    analyzer = SpatialHazardAnalyzer(frame_width=1280, frame_height=720)

    # FPS Calculation trackers
    prev_time = time.time()
    fps = 30.0
    processed_frames = 0

    window_name = "Project Third Eye - Spatial Hazard Assistant"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    print("\n[Project Third Eye] System operational. Tracking spatial hazards. Press 'q' to quit.\n", flush=True)
    # Auditory confirmation that speech synthesis and spatial monitoring are active
    audio_engine.announce("Third Eye active. Monitoring path.", force=True)

    try:
        while True:
            start_loop = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Project Third Eye] Video stream completed or interrupted.")
                break

            h, w, _ = frame.shape
            analyzer.update_frame_dimensions(w, h)

            # Perform Object Detection
            t_infer_start = time.time()
            results = model.predict(
                source=frame,
                conf=args.conf,
                verbose=False,
                device="cpu",  # Edge default; will leverage GPU if configured
            )
            latency_ms = (time.time() - t_infer_start) * 1000.0

            detections: List[Dict[str, Any]] = []

            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        cls_name = model.names.get(cls_id, str(cls_id)).lower()

                        # Filter for spatial navigation hazards
                        if cls_name not in TARGET_CLASSES:
                            continue

                        conf = float(boxes.conf[i].item())
                        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy

                        # Pinhole Metric Distance & Geometry
                        box_height = max(1, y2 - y1)
                        distance_m = estimate_metric_distance(box_height, cls_name, h, CAMERA_FOV_V_DEG)
                        ratio = float(box_height) / float(h)
                        cx = (x1 + x2) / 2.0
                        corridor = analyzer.classify_corridor(cx)

                        # Threat Level & Color Classification based on Metric Distance
                        if distance_m < CRITICAL_DISTANCE_METERS:
                            threat = "CRITICAL"
                            color = COLOR_CRITICAL
                            dist_desc = f"{distance_m:.1f}m (CRITICAL)"
                        elif distance_m <= WARNING_DISTANCE_METERS:
                            threat = "WARNING"
                            color = COLOR_WARNING
                            dist_desc = f"{distance_m:.1f}m (WARNING)"
                        else:
                            threat = "SAFE"
                            color = COLOR_SAFE
                            dist_desc = f"{distance_m:.1f}m (SAFE)"

                        detections.append({
                            "class_name": cls_name,
                            "confidence": conf,
                            "box": (x1, y1, x2, y2),
                            "distance_m": distance_m,
                            "ratio": ratio,
                            "cx": cx,
                            "corridor": corridor,
                            "threat": threat,
                            "color": color,
                            "dist_desc": dist_desc,
                        })

            # Spatial Hazard Analysis & Decision Matrix
            analysis = analyzer.analyze_detections(detections)

            # Trigger Audio Navigation Directives asynchronously
            if analysis["audio_cue"] is not None:
                announced = audio_engine.announce(
                    analysis["audio_cue"],
                    hazard_level=analysis["hazard_level"]
                )
                if announced:
                    print(f"[AUDIO DIRECTIVE SPOKEN]: \"{analysis['audio_cue']}\"", flush=True)

            # Compute smoothed FPS
            loop_duration = time.time() - start_loop
            instant_fps = 1.0 / max(loop_duration, 1e-5)
            fps = 0.9 * fps + 0.1 * instant_fps

            # Render HUD AR Overlay
            annotated_frame = render_hud(frame, detections, analysis, fps, latency_ms)

            # Display Window
            if not args.headless:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[Project Third Eye] 'q' pressed. Gracefully shutting down...")
                    break

            processed_frames += 1
            if args.max_frames and processed_frames >= args.max_frames:
                print(f"[Project Third Eye] Reached limit of {args.max_frames} frames. Exiting.")
                break

    except KeyboardInterrupt:
        print("\n[Project Third Eye] Interrupted by user (Ctrl+C).")
    finally:
        print("[Project Third Eye] Cleaning up resources...")
        cap.release()
        audio_engine.shutdown()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[Project Third Eye] Pipeline closed safely.")


if __name__ == "__main__":
    run_pipeline()
