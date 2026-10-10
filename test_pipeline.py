#!/usr/bin/env python3
"""
Unit and Integration Tests for Project Third Eye
"""

import time
import unittest
import numpy as np

from main import (
    SpatialHazardAnalyzer,
    AudioNavigationEngine,
    CRITICAL_RATIO_THRESHOLD,
    WARNING_RATIO_THRESHOLD,
    COLOR_CRITICAL,
    COLOR_WARNING,
    COLOR_SAFE,
)


class TestProjectThirdEye(unittest.TestCase):
    def setUp(self):
        self.width = 1280
        self.height = 720
        self.analyzer = SpatialHazardAnalyzer(self.width, self.height)

    def test_corridor_partitioning(self):
        """Test vertical corridor boundary partitioning."""
        left_x = int(0.20 * self.width)
        center_x = int(0.50 * self.width)
        right_x = int(0.80 * self.width)

        self.assertEqual(self.analyzer.classify_corridor(left_x), "LEFT")
        self.assertEqual(self.analyzer.classify_corridor(center_x), "CENTER")
        self.assertEqual(self.analyzer.classify_corridor(right_x), "RIGHT")

    def test_decision_matrix_clear_path(self):
        """Test decision matrix when no obstacles exist."""
        analysis = self.analyzer.analyze_detections([])
        self.assertEqual(analysis["threat_level"], "SAFE")
        self.assertEqual(analysis["status_text"], "STATUS: CLEAR PATH AHEAD")
        self.assertIsNone(analysis["audio_cue"])
        self.assertEqual(analysis["status_color"], COLOR_SAFE)

    def test_decision_matrix_critical_obstacle_ahead(self):
        """Test decision matrix when center obstacle is in critical zone (ratio > 0.50)."""
        det = {
            "class_name": "person",
            "confidence": 0.88,
            "box": (500, 100, 700, 600),  # height = 500 / 720 = 0.694
            "ratio": 500.0 / 720.0,
            "cx": 600.0,
            "corridor": "CENTER",
            "threat": "CRITICAL",
            "color": COLOR_CRITICAL,
            "dist_desc": "< 1.2m (CRITICAL)",
        }
        analysis = self.analyzer.analyze_detections([det])
        self.assertEqual(analysis["threat_level"], "CRITICAL")
        self.assertEqual(analysis["status_text"], "STATUS: CRITICAL HAZARD DETECTED")
        self.assertEqual(analysis["audio_cue"], "STOP! Obstacle directly ahead!")
        self.assertEqual(analysis["status_color"], COLOR_CRITICAL)

    def test_decision_matrix_warning_veer_left(self):
        """
        Test decision matrix: obstacle in center warning zone,
        right corridor has another obstacle, so left corridor is clearer -> Veer Left.
        """
        center_det = {
            "class_name": "chair",
            "confidence": 0.75,
            "box": (500, 300, 700, 588),  # height = 288 / 720 = 0.40
            "ratio": 0.40,
            "cx": 600.0,
            "corridor": "CENTER",
            "threat": "WARNING",
            "color": COLOR_WARNING,
            "dist_desc": "1.2m-2.5m (WARNING)",
        }
        right_det = {
            "class_name": "couch",
            "confidence": 0.80,
            "box": (900, 300, 1200, 550),
            "ratio": 0.35,
            "cx": 1050.0,
            "corridor": "RIGHT",
            "threat": "WARNING",
            "color": COLOR_WARNING,
            "dist_desc": "1.2m-2.5m (WARNING)",
        }
        # Left corridor has 0 obstacles, Right corridor has 1 obstacle (0.35 threat)
        analysis = self.analyzer.analyze_detections([center_det, right_det])
        self.assertEqual(analysis["threat_level"], "WARNING")
        self.assertEqual(analysis["status_text"], "STATUS: WARNING - OBSTACLE AHEAD [VEER LEFT]")
        self.assertEqual(analysis["audio_cue"], "Obstacle ahead. Veer Left")
        self.assertEqual(analysis["status_color"], COLOR_WARNING)

    def test_decision_matrix_warning_veer_right(self):
        """
        Test decision matrix: obstacle in center warning zone,
        left corridor has another obstacle, so right corridor is clearer -> Veer Right.
        """
        center_det = {
            "class_name": "chair",
            "confidence": 0.75,
            "box": (500, 300, 700, 588),
            "ratio": 0.40,
            "cx": 600.0,
            "corridor": "CENTER",
            "threat": "WARNING",
            "color": COLOR_WARNING,
            "dist_desc": "1.2m-2.5m (WARNING)",
        }
        left_det = {
            "class_name": "dining table",
            "confidence": 0.82,
            "box": (50, 200, 350, 500),
            "ratio": 0.42,
            "cx": 200.0,
            "corridor": "LEFT",
            "threat": "WARNING",
            "color": COLOR_WARNING,
            "dist_desc": "1.2m-2.5m (WARNING)",
        }
        analysis = self.analyzer.analyze_detections([center_det, left_det])
        self.assertEqual(analysis["threat_level"], "WARNING")
        self.assertEqual(analysis["status_text"], "STATUS: WARNING - OBSTACLE AHEAD [VEER RIGHT]")
        self.assertEqual(analysis["audio_cue"], "Obstacle ahead. Veer Right")
        self.assertEqual(analysis["status_color"], COLOR_WARNING)

    def test_audio_engine_non_blocking_cooldown(self):
        """Test audio engine daemon non-blocking enqueue and throttle cooldown."""
        audio = AudioNavigationEngine(cooldown=0.3)
        try:
            # First announcement should succeed
            res1 = audio.announce("Testing speech one")
            self.assertTrue(res1)

            # Immediate second announcement within cooldown should be dropped by throttle
            res2 = audio.announce("Testing speech two")
            self.assertFalse(res2)

            # Wait for cooldown to elapse
            time.sleep(0.35)
            res3 = audio.announce("Testing speech three")
            self.assertTrue(res3)
        finally:
            audio.shutdown()


if __name__ == "__main__":
    unittest.main()
