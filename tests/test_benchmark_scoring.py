from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_scoring import boundary_transform, export_benchmark_scores, load_scoring_config


class BenchmarkScoringTest(unittest.TestCase):
    def test_boundary_transform(self):
        self.assertEqual(boundary_transform(0.1, floor=0.1, gamma=0.6), 0.0)
        self.assertEqual(boundary_transform(1.0, floor=0.1, gamma=0.6), 1.0)
        expected = ((0.55 - 0.1) / 0.9) ** 0.6
        self.assertTrue(math.isclose(boundary_transform(0.55, floor=0.1, gamma=0.6), expected))

    def test_frozen_beta_vector(self):
        config = load_scoring_config()
        self.assertEqual(config["betas"]["MacGyver"], {"beta_ih": 0.0, "beta_hi": 0.9})
        self.assertEqual(config["betas"]["AnalogyTransfer"], {"beta_ih": 0.9, "beta_hi": 0.3})

    def test_synthetic_report_scores(self):
        contribution = {
            "raw": {"imagination": {"assoc": 0.6}, "hallucination": {
                "context": 0.2, "drift": 0.3, "intent": 0.4, "logic": 0.5,
            }},
            "gated": {"imagination": {"assoc": 0.6}, "hallucination": {
                "context": 0.2, "drift": 0.3, "intent": 0.4, "logic": 0.5,
            }},
            "residual": {"imagination": {"assoc": 0.5}, "hallucination": {
                "context": 0.1, "drift": 0.2, "intent": 0.3, "logic": 0.4,
            }},
        }
        report = {
            "model_id": "synthetic-model",
            "task_results": [{
                "task_type": "UUT",
                "task_id": "uut_synthetic",
                "repeat_index": 0,
                "valid_run": True,
                "dual_axis": {"subtype_contributions": contribution},
            }],
        }
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "synthetic-model_report.json").write_text(json.dumps(report))
            result = export_benchmark_scores(directory)
            self.assertEqual(result["models"], 1)
            self.assertEqual(result["task_outputs"], 1)
            self.assertTrue((directory / "benchmark_model_scores.csv").is_file())
            self.assertTrue((directory / "benchmark_output_scores.csv").is_file())


if __name__ == "__main__":
    unittest.main()
