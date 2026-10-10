#!/usr/bin/env python3
"""
Automated Verification Suite for Project Third Eye - Stage 1 Fixes
Tests:
1. Audio Priority Override & Queue Purging
2. Pinhole Metric Distance Formula & Real-World Heights
3. Threaded Camera Ingestion & Frame-Dropping Queue
"""

import os
import time
import math
import unittest
import numpy as np
import cv2

from main import (
    AudioNavigationEngine,
    estimate_metric_distance,
    compute_focal_length_y,
    CLASS_REAL_HEIGHTS,
    ThreadedCameraStream,
    SyntheticCameraStream,
    CAMERA_FOV_V_DEG,
)


class TestStage1Verification(unittest.TestCase):
    def test_audio_priority_override_and_purging(self):
        """
        Verify that a CRITICAL hazard or STOP alert:
        - Bypasses the 2.0s cooldown
        - Purges stale queued speech
        - Accepts and dispatches immediately
        - Keeps non-critical subsequent alerts throttled
        """
        audio = AudioNavigationEngine(cooldown=2.0)
        try:
            # 1. Non-critical announcement
            res1 = audio.announce("Veer Left", hazard_level="WARNING")
            self.assertTrue(res1, "Initial warning announcement should succeed.")

            # 2. Immediate CRITICAL STOP announcement within active cooldown
            res2 = audio.announce("STOP! Obstacle ahead!", hazard_level="CRITICAL")
            self.assertTrue(res2, "CRITICAL STOP alert must override active cooldown timer.")

            # 3. Verify queue state: queue must not exceed maxsize and stale items purged
            self.assertLessEqual(audio.message_queue.qsize(), 1, "Queue must be purged of stale items.")

            # 4. Immediate subsequent non-critical announcement should be throttled
            res3 = audio.announce("Veer Right", hazard_level="WARNING")
            self.assertFalse(res3, "Non-critical announcement within cooldown must be throttled.")
        finally:
            audio.shutdown()

    def test_pinhole_metric_distance_calculation(self):
        """
        Verify pinhole metric distance formula:
        Z = (f_y * H_real) / bbox_height_px
        where f_y = (frame_height / 2) / tan(fov_v / 2)
        """
        frame_h = 720
        fov_v = 60.0

        # Confirm CLASS_REAL_HEIGHTS contains person height
        self.assertIn("person", CLASS_REAL_HEIGHTS)
        self.assertEqual(CLASS_REAL_HEIGHTS["person"], 1.70)

        # Theoretical focal length calculation
        fov_v_rad = math.radians(fov_v)
        expected_fy = (frame_h / 2.0) / math.tan(fov_v_rad / 2.0)  # ~623.54 px
        self.assertAlmostEqual(compute_focal_length_y(frame_h, fov_v), expected_fy, places=3)

        # 1. Person (1.7m) at 360px box height on 720p frame
        dist_360 = estimate_metric_distance(
            bbox_height_px=360,
            class_name="person",
            frame_height=frame_h,
            fov_v_deg=fov_v,
        )
        expected_360 = (expected_fy * 1.70) / 360.0  # ~2.94m
        self.assertAlmostEqual(dist_360, expected_360, places=2)
        self.assertTrue(2.0 <= dist_360 <= 3.2, f"Distance {dist_360:.2f}m is within realistic 2.0m-3.2m range.")

        # 2. Person (1.7m) at 500px box height on 720p frame
        dist_500 = estimate_metric_distance(
            bbox_height_px=500,
            class_name="person",
            frame_height=frame_h,
            fov_v_deg=fov_v,
        )
        expected_500 = (expected_fy * 1.70) / 500.0  # ~2.12m
        self.assertAlmostEqual(dist_500, expected_500, places=2)
        self.assertTrue(2.0 <= dist_500 <= 2.5, f"Distance {dist_500:.2f}m is within realistic ~2.1m range.")

        # 3. Chair (0.85m) at 360px box height on 720p frame
        dist_chair = estimate_metric_distance(
            bbox_height_px=360,
            class_name="chair",
            frame_height=frame_h,
            fov_v_deg=fov_v,
        )
        expected_chair = (expected_fy * 0.85) / 360.0
        self.assertAlmostEqual(dist_chair, expected_chair, places=2)

    def test_threaded_camera_ingestion_and_fps(self):
        """
        Verify ThreadedCameraStream & Synthetic Ingestion:
        - Uses queue.Queue(maxsize=1) with frame-dropping logic
        - Ingests across 60 frames and measures FPS
        """
        # Part A: Test ThreadedCameraStream queue maxsize=1 and frame-dropping logic
        stream = ThreadedCameraStream(source=0, width=640, height=480, fps=30)
        try:
            self.assertEqual(stream.frame_queue.maxsize, 1, "Queue maxsize must be strictly 1.")

            if stream.isOpened():
                # Measure FPS on active physical/virtual camera over 60 frames
                frames_read = 0
                start_time = time.time()
                for _ in range(60):
                    ret, frame = stream.read()
                    if ret and frame is not None:
                        frames_read += 1
                elapsed = time.time() - start_time
                fps = frames_read / max(elapsed, 1e-5)
                print(f"\n[Threaded Camera Test] Ingested {frames_read} frames from webcam in {elapsed:.3f}s (~{fps:.1f} FPS).")
                self.assertGreaterEqual(frames_read, 30, "Should read at least 30 frames from camera.")
            else:
                print("\n[Threaded Camera Test] Camera 0 not opened, verifying SyntheticCameraStream across 60 frames.")
        finally:
            stream.release()

        # Part B: SyntheticCameraStream deterministic 60-frame benchmark
        synthetic = SyntheticCameraStream(width=1280, height=720, fps=60)
        try:
            self.assertTrue(synthetic.isOpened())
            start_time = time.time()
            for i in range(60):
                ret, frame = synthetic.read()
                self.assertTrue(ret)
                self.assertEqual(frame.shape, (720, 1280, 3))
            elapsed = time.time() - start_time
            fps = 60.0 / max(elapsed, 1e-5)
            print(f"[Synthetic Stream Test] Ingested 60 frames in {elapsed:.3f}s (~{fps:.1f} FPS).")
            self.assertGreater(fps, 20.0, "Synthetic stream should achieve at least 20 FPS.")
        finally:
            synthetic.release()


if __name__ == "__main__":
    unittest.main()
