from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path

from prompts_dataset import get_all_prompts

import run_benchmark


class BenchmarkRunModeTest(unittest.TestCase):
    def setUp(self):
        self.original_environment = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_environment)

    @staticmethod
    def args(output_dir: str):
        return argparse.Namespace(
            provider="openrouter",
            model_id="test/model",
            output_dir=output_dir,
            embedding_device="cpu",
            smoke=False,
            full=False,
            overwrite=False,
        )

    def test_default_mode_uses_fixed_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            run_benchmark._set_runtime_environment(self.args(temp))
            self.assertEqual(os.environ["BENCHMARK_RUN_MODE"], "anchor")
            self.assertEqual(
                os.environ["OPENROUTER_TASK_MANIFEST_PATH"],
                "data/anchor_manifest.json",
            )
            self.assertEqual(
                tuple(os.environ["OPENROUTER_TASK_FAMILIES"].split(",")),
                run_benchmark.PRIMARY_FAMILIES,
            )

    def test_full_mode_uses_all_main_paper_prompts(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(temp)
            args.full = True
            run_benchmark._set_runtime_environment(args)
            self.assertEqual(os.environ["BENCHMARK_RUN_MODE"], "full")
            self.assertEqual(os.environ["OPENROUTER_TASK_MANIFEST_PATH"], "")
            dataset = get_all_prompts()
            observed_count = sum(len(dataset[family]) for family in run_benchmark.PRIMARY_FAMILIES)
            self.assertEqual(observed_count, run_benchmark.FULL_MAIN_PROMPT_COUNT)

if __name__ == "__main__":
    unittest.main()
