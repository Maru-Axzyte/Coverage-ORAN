import unittest
import numpy as np
from line_analysis import parse_progress, observed_curve, carry_forward


class LineTests(unittest.TestCase):
    def test_infeasible_service_never_becomes_feasible_coverage(self):
        text = ('Joint WE 1/300: served 70/70, V=2, PRB 135/135, P 10 W; MILP 84, pool 800\n'
                'Joint WE 10/300: served 60/70, V=0, PRB 135/135, P 10 W; MILP 120, pool 800')
        events = parse_progress(text,'WE')
        x,y = observed_curve(events,1280)
        np.testing.assert_allclose(carry_forward(x,y,np.array([0,84,119,120])),[0,0,0,60/70])

    def test_logs_are_not_linearly_interpolated(self):
        events = parse_progress('MILP 40/1280: served=30/50, V=0\nMILP 80/1280: served=50/50, V=0','PSO-MILP')
        x,y = observed_curve(events,1280)
        self.assertEqual(carry_forward(x,y,np.array([60]))[0],.6)


if __name__ == '__main__':
    unittest.main()
