"""Plot statistics only; no GA or MILP optimization."""
import unittest
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use("Agg")
from plots import generation_quality, draw_surrogate_screening_quality


def record(generation, error, coverage=None, rescue=None):
    return dict(generation=generation, error_j=error,
                interval_covered=coverage, outside_topk_rescue=rescue)


class GenerationQualityTests(unittest.TestCase):
    def test_uses_all_samples_not_last_twenty(self):
        rows = generation_quality([record(2, 4.)]*10 + [record(2, -1.)]*20, [2])
        self.assertEqual(rows[0]["checks"], 30)
        self.assertAlmostEqual(rows[0]["mae"], 2.)

    def test_no_cross_generation_leak_or_forward_fill(self):
        rows = generation_quality([record(2, 8.), record(0, -2.)], [0, 1, 2, 3])
        self.assertEqual(rows[0]["mae"], 2.)
        self.assertEqual(rows[2]["mae"], 8.)
        for i in (1, 3):
            for key in ("mae", "coverage", "rescue"):
                self.assertTrue(np.isnan(rows[i][key]))
            self.assertEqual(rows[i]["checks"], 0)

    def test_separate_denominators_and_real_zero(self):
        checks = [record(1, .1, True, False), record(1, .2, False), record(1, .3)]
        r = generation_quality(checks, [1])[0]
        self.assertEqual((r["checks"], r["interval_checks"], r["audit_checks"]), (3, 2, 1))
        self.assertEqual(r["coverage"], .5)
        self.assertEqual(r["rescue"], 0.)

    def test_empty_input(self):
        self.assertEqual(generation_quality([], []), [])
        self.assertTrue(np.isnan(generation_quality([], [0])[0]["mae"]))

    def test_sparse_audits_are_visible_markers(self):
        captured = []
        def capture(fig, *args, **kwargs):
            captured.append(fig)
        with patch("matplotlib.figure.Figure.savefig", capture):
            draw_surrogate_screening_quality(
                checks=[record(1, .1, True, True), record(3, .2, False, False)],
                generations=[0, 1, 2, 3], output="unused_quality_test.png")
        fig = captured[0]
        self.assertIn("per generation", fig.axes[0].get_title())
        audit = fig.axes[1].lines[1]
        self.assertEqual(audit.get_marker(), "x")
        self.assertEqual(audit.get_linestyle(), "None")
        np.testing.assert_allclose(audit.get_ydata(), [np.nan, 1., np.nan, 0.], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
