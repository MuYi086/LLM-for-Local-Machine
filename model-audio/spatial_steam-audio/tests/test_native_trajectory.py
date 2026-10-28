"""显式运行的 CPU SDK 集成测试：不加载任何生成模型。"""

import json
import os
import random
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

BIN = os.environ.get("STEAM_AUDIO_TEST_BIN", "")


def float_wav(path):
    """读取 renderer 的 float WAV，独立检查帧数与数值。"""
    data = path.read_bytes()
    offset = 12
    while offset + 8 <= len(data):
        tag, size = struct.unpack_from("<4sI", data, offset)
        if tag == b"data":
            values = struct.unpack_from(f"<{size // 4}f", data, offset + 8)
            return list(zip(values[::2], values[1::2], strict=True))
        offset += 8 + size + size % 2
    raise AssertionError("missing WAV data")


@unittest.skipUnless(BIN, "显式设置 STEAM_AUDIO_TEST_BIN 才运行真实 CPU SDK")
class NativeTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = random.Random(17)
        self.noise = [int(5000 * rng.uniform(-1, 1)) for _ in range(48000)]

    def render(self, name, trajectory, samples=None):
        source = self.root / f"{name}.wav"
        with wave.open(str(source), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(48000)
            samples = samples if samples is not None else self.noise
            out.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        config = self.root / f"{name}.json"
        config.write_text(json.dumps(trajectory))
        output = self.root / f"{name}-out.wav"
        trace = self.root / f"{name}.tsv"
        run = subprocess.run(
            [
                BIN,
                "--input",
                str(source),
                "--trajectory",
                str(config),
                "--output",
                str(output),
                "--trajectory-report",
                str(trace),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(run.stdout)
        audio = float_wav(output)
        self.assertEqual(report["output_frames"], len(audio))
        self.assertEqual(report["output_frames"], len(samples) + report["tail_samples"])
        self.assertEqual(report["mix_gain"], 1)
        self.assertFalse(report["peak_normalized"])
        return audio, report, trace

    def test_info_and_left_right_mirror(self):
        info = json.loads(subprocess.check_output([BIN, "--info"], text=True))
        self.assertEqual(info["trajectory_version"], 1)
        left, _, _ = self.render("left", {"type": "static", "position_m": [-1, 0, 0]})
        right, _, _ = self.render("right", {"type": "static", "position_m": [1, 0, 0]})

        def energy(audio, ear):
            return sum(row[ear] ** 2 for row in audio)

        self.assertGreater(energy(left, 0), energy(left, 1) * 1.5)
        self.assertGreater(energy(right, 1), energy(right, 0) * 1.5)

    def test_approach_changes_level_and_trace_uses_local_seconds(self):
        audio, _, trace = self.render(
            "approach",
            {"type": "approach", "start_distance_m": 8, "end_distance_m": 1, "duration_seconds": 1},
        )

        def energy(rows):
            return sum(a * a + b * b for a, b in rows)

        self.assertGreater(energy(audio[36000:47000]), energy(audio[1000:12000]) * 3)
        rows = [line.split() for line in trace.read_text().splitlines()[1:]]
        self.assertEqual(rows[0][0], "single_source")
        self.assertGreater(float(rows[0][1]), 0)
        self.assertLess(float(rows[0][1]), 0.03)
        self.assertLess(float(rows[0][4]), float(rows[-1][4]))

    def test_manifest_offset_does_not_advance_trajectory_clock(self):
        source = self.root / "normalized.wav"
        with wave.open(str(source), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(48000)
            output.writeframes(struct.pack(f"<{len(self.noise)}h", *self.noise))
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "sample_rate": 48000,
                    "timeline_duration_ms": 1500,
                    "scene": {
                        "acoustic_quality": "immersive",
                        "room": "dry_studio",
                        "listener_pose": "center_forward",
                    },
                    "sources": [
                        {
                            "id": "delayed",
                            "kind": "sfx",
                            "asset_id": "a",
                            "asset_filename": source.name,
                            "start_ms": 250,
                            "gain_db": 0,
                            "duration_ms": 1000,
                            "spatial": {
                                "mode": "point",
                                "location": "front_center",
                                "distance": "near",
                                "occlusion": "none",
                                "spatial_blend": "full",
                                "movement": "static",
                                "trajectory": {
                                    "type": "linear",
                                    "start_m": [-2, 0, -2],
                                    "end_m": [2, 0, -2],
                                    "duration_seconds": 1,
                                },
                            },
                        }
                    ],
                }
            )
        )
        trace = self.root / "delayed.tsv"
        output = self.root / "delayed.wav"
        result = subprocess.run(
            [
                BIN,
                "--manifest",
                str(manifest),
                "--assets-dir",
                str(self.root),
                "--output",
                str(output),
                "--trajectory-report",
                str(trace),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        first = trace.read_text().splitlines()[1].split()
        local_seconds = float(first[1])
        self.assertLess(local_seconds, 0.03)
        self.assertAlmostEqual(float(first[2]), -2 + 4 * local_seconds, places=5)
        self.assertTrue(all(a == 0 and b == 0 for a, b in float_wav(output)[:12000]))

    def test_relative_output_basename(self):
        self.render("fixture", {"type": "static", "position_m": [0, 0, -1]})
        result = subprocess.run(
            [
                BIN,
                "--input",
                "fixture.wav",
                "--trajectory",
                "fixture.json",
                "--output",
                "relative.wav",
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "relative.wav").is_file())

    def test_full_block_eof_preserves_direct_filter_tail(self):
        trajectory = {"type": "static", "position_m": [0, 0, -8]}
        short, _, _ = self.render("aligned", trajectory, [0] * 1023 + [16000])
        padded, _, _ = self.render("padded", trajectory, [0] * 1023 + [16000] + [0] * 3072)
        error = max(
            abs(a - b)
            for actual, expected in zip(short, padded, strict=False)
            for a, b in zip(actual, expected, strict=True)
        )
        self.assertLess(error, 1e-6)
        self.assertLess(sum(a * a + b * b for a, b in padded[len(short) :]), 1e-10)

    def test_last_sample_tail_and_no_hidden_peak_normalization(self):
        audio, report, _ = self.render(
            "tail", {"type": "static", "position_m": [0, 0, -1]}, [0] * 48002 + [16000]
        )
        self.assertGreater(report["tail_samples"], 0)
        self.assertGreater(sum(abs(a) + abs(b) for a, b in audio[48003:]), 0.01)
        base, _, _ = self.render(
            "base",
            {"type": "static", "position_m": [0, 0, -0.25]},
            [int(v / 10) for v in self.noise],
        )
        high, _, _ = self.render(
            "high",
            {"type": "static", "position_m": [0, 0, -0.25]},
            [int(v / 10) * 50 for v in self.noise],
        )
        error = max(
            abs(high[i][ear] - base[i][ear] * 50) for i in range(len(base)) for ear in (0, 1)
        )
        self.assertLess(error, 1e-4)


if __name__ == "__main__":
    unittest.main()
