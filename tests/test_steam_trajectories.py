"""验证通用轨迹的公开合同、冲突及几何安全边界。"""

import copy
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main"))
from spatial_schema import SpatialRenderManifest  # noqa: E402


def with_trajectory(trajectory):
    """构造具有局部时间偏移的最小轨迹对象。"""
    return {
        "timeline_duration_ms": 30000,
        "sources": [
            {
                "id": "car",
                "kind": "sfx",
                "asset_id": "car",
                "asset_filename": "car.wav",
                "start_ms": 1000,
                "duration_ms": 20000,
                "spatial": {"trajectory": trajectory},
            }
        ],
    }


CASES = [
    {"type": "static", "position_m": [1, 0, -2]},
    {"type": "linear", "start_m": [-4, 0, -2], "end_m": [4, 0, -2], "duration_seconds": 8},
    {
        "type": "approach",
        "azimuth_degrees": 90,
        "elevation_degrees": 0,
        "start_distance_m": 8,
        "end_distance_m": 1,
        "duration_seconds": 8,
    },
    {
        "type": "recede",
        "azimuth_degrees": -90,
        "elevation_degrees": 30,
        "start_distance_m": 1,
        "end_distance_m": 8,
        "duration_seconds": 8,
    },
    {
        "type": "circular",
        "radius_m": 2,
        "height_m": 1,
        "start_angle_degrees": 170,
        "end_angle_degrees": 530,
        "duration_seconds": 8,
        "repeat": "loop",
    },
    {
        "type": "waypoints",
        "points": [
            {"time_seconds": 0, "position_m": [-2, 0, -2]},
            {"time_seconds": 2, "position_m": [0, 1, -2]},
            {"time_seconds": 8, "position_m": [2, 0, -2]},
        ],
        "repeat": "ping_pong",
        "interpolation": "smoothstep",
    },
]


class TrajectoryContractTests(unittest.TestCase):
    def test_all_six_types_survive_manifest_serialization(self):
        for trajectory in CASES:
            with self.subTest(kind=trajectory["type"]):
                result = SpatialRenderManifest.model_validate(with_trajectory(trajectory))
                saved = result.model_dump(mode="json")["sources"][0]
                self.assertEqual(saved["spatial"]["trajectory"]["type"], trajectory["type"])
                self.assertEqual(saved["start_ms"], 1000)
                for name, value in trajectory.items():
                    self.assertEqual(saved["spatial"]["trajectory"][name], value)

    def test_rejects_invalid_geometry_times_repeats_and_unknown_parameters(self):
        invalid = [
            {"type": "static", "position_m": [0, 0, 0]},
            {"type": "static", "position_m": [1000, 1000, 0]},
            {"type": "static", "position_m": [True, 0, -1]},
            {"type": "static", "position_m": ["1", 0, -1]},
            {"type": "static", "position_m": [float("inf"), 0, -1]},
            {**CASES[0], "radius_m": 1},
            {**CASES[1], "duration_seconds": 0},
            {**CASES[1], "duration_seconds": True},
            {**CASES[1], "duration_seconds": 7201},
            {**CASES[1], "start_m": [-2, 0, 0], "end_m": [2, 0, 0]},
            {**CASES[1], "repeat": "loop"},
            {**CASES[2], "start_distance_m": 1, "end_distance_m": 8},
            {**CASES[3], "elevation_degrees": 100},
            {**CASES[4], "radius_m": 0},
            {**CASES[4], "end_angle_degrees": 360},
            {**CASES[5], "points": [{"time_seconds": 0, "position_m": [1, 0, 0]}]},
            {
                **CASES[5],
                "points": [
                    {"time_seconds": 1, "position_m": [1, 0, 0]},
                    {"time_seconds": 2, "position_m": [1, 0, 0]},
                ],
            },
            {
                **CASES[5],
                "points": [
                    {"time_seconds": 0, "position_m": [-1, 0, 0]},
                    {"time_seconds": 1, "position_m": [1, 0, 0]},
                ],
            },
            {
                **CASES[5],
                "points": [
                    {"time_seconds": 0, "position_m": [1, 0, 0]},
                    {"time_seconds": 0, "position_m": [2, 0, 0]},
                ],
            },
        ]
        for trajectory in invalid:
            with self.subTest(trajectory=trajectory), self.assertRaises(ValidationError):
                SpatialRenderManifest.model_validate(with_trajectory(trajectory))

    def test_explicit_trajectory_cannot_be_silently_ignored_or_conflict(self):
        for field, value in (
            ("mode", "dry_center"),
            ("mode", "preserve_stereo"),
            ("movement", "approaching"),
        ):
            payload = with_trajectory(copy.deepcopy(CASES[0]))
            payload["sources"][0]["spatial"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                SpatialRenderManifest.model_validate(payload)

    def test_old_manifest_does_not_require_a_trajectory(self):
        payload = with_trajectory(None)
        del payload["sources"][0]["spatial"]["trajectory"]
        self.assertIsNone(
            SpatialRenderManifest.model_validate(payload).sources[0].spatial.trajectory
        )


if __name__ == "__main__":
    unittest.main()
