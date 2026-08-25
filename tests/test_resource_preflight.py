from __future__ import annotations

import unittest

import run_benchmark


class ResourcePreflightTests(unittest.TestCase):
    def test_missing_swow_blocks_exact_run(self):
        with self.assertRaisesRegex(SystemExit, "SWOW-EN data is required"):
            run_benchmark._validate_required_resources()


if __name__ == "__main__":
    unittest.main()
