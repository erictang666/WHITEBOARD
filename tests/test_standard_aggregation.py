from __future__ import annotations

import unittest

import benchmark_core


class StandardAggregationTests(unittest.TestCase):
    def test_fixed_weighted_mean_snapshot(self):
        scores = {
            name: (index + 1) / 10
            for index, name in enumerate(benchmark_core.DUAL_AXIS_COMPONENT_ORDER)
        }
        result = benchmark_core.aggregate_dual_axis_component_scores(
            scores,
            benchmark_core.PRIMARY_DUAL_AXIS_COMPONENTS,
            axis="imagination",
        )
        self.assertAlmostEqual(result["score"], 0.4919642857142857, places=15)
        self.assertEqual(
            result["formula"],
            "0.080*UUT_I + 0.062*PropConj_I + 0.143*MacGyver_I + "
            "0.143*CJST_I + 0.143*GCW_I + 0.143*HypoUseSpace_I + "
            "0.143*NeoCoder_I + 0.143*AnalogyTransfer_I",
        )


if __name__ == "__main__":
    unittest.main()
