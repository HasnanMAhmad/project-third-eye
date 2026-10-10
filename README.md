# Project Third Eye: Real-Time Edge Spatial Hazard Detection & Navigation Assistant

Project Third Eye is a real-time, low-latency edge computer vision and spatial audio navigation assistant designed to empower visually impaired individuals using standard laptop webcams.

The system partitions the environment into 3 spatial corridors, detects navigation obstacles with YOLOv8, computes monocular distance/threat metrics, delivers real-time thread-safe directional audio cues via `pyttsx3`, and displays a high-contrast Augmented Reality (AR) HUD.

---

## 1. System Architecture

```
[Laptop Webcam (1280x720 @ 30 FPS)]
                │
                ▼
[YOLOv8n Inference Engine (conf >= 0.50)]
  - Filters: person, chair, couch, bottle, dining table, door, pole, laptop, bicycle, motorcycle, car, bus
                │
                ▼
[Spatial Corridor Partitioning]
  - LEFT CORRIDOR:   0.00 * W to 0.33 * W
  - CENTER CORRIDOR: 0.33 * W to 0.66 * W (Active collision vector)
  - RIGHT CORRIDOR:  0.66 * W to 1.00 * W
                │
                ▼
[Monocular Geometry & Distance Estimation]
  - ratio = bounding_box_height / frame_height
  - CRITICAL HAZARD: ratio > 0.50 (< 1.2m)
  - WARNING HAZARD:  0.25 <= ratio <= 0.50 (1.2m - 2.5m)
  - SAFE / FAR:      ratio < 0.25 (> 2.5m)
                │
        ┌───────┴────────────────────────┐
        ▼                                ▼
[Decision Matrix & Audio Engine]   [AR HUD Visualizer]
  - Non-blocking daemon thread      - Corridor boundary dashed lines
  - 1.8s audio cooldown throttle    - Color-coded boxes (Red/Yellow/Green)
  - Natural directional cues        - Top Status Banner (Critical/Warning/Safe)
```

---

## 2. Decision Matrix & Audio Directives

| Center Hazard Status | Spatial Conditions | Spoken Audio Cue | HUD Banner |
|---|---|---|---|
| **Critical Hazard** (`ratio > 0.50`) | Primary obstacle directly in Center corridor (< 1.2m) | *"STOP! Obstacle directly ahead!"* | `STATUS: CRITICAL HAZARD DETECTED` (Red) |
| **Warning Hazard** (`0.25 <= ratio <= 0.50`) | Primary obstacle in Center; Left corridor is clearer than Right | *"Obstacle ahead. Veer Left"* | `STATUS: WARNING - OBSTACLE AHEAD [VEER LEFT]` (Yellow) |
| **Warning Hazard** (`0.25 <= ratio <= 0.50`) | Primary obstacle in Center; Right corridor is clearer or equal | *"Obstacle ahead. Veer Right"* | `STATUS: WARNING - OBSTACLE AHEAD [VEER RIGHT]` (Yellow) |
| **Safe / Clear** (`ratio < 0.25` or no obstacle) | No collision vector threats | *Silent* | `STATUS: CLEAR PATH AHEAD` (Green) |

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+
- Webcam or integrated camera

### Setup Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Running the System

### Standard Webcam Mode
```powershell
.\venv\Scripts\python main.py
```

### Command-Line Options
```text
Options:
  --camera INT           Webcam index (default: 0)
  --source PATH          Optional path to video file or image
  --conf FLOAT           Detection confidence threshold (default: 0.50)
  --model PATH           YOLO model weights (default: yolov8n.pt)
  --headless             Run without GUI window
  --max-frames INT       Limit frame processing for automated testing
```

Press **`q`** in the window to cleanly shut down.
