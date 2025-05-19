import unittest
import numpy as np
import openmdao.api as om
from practice.kinematics.TrajFinalExample import WidthCI


class TestCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'dist', WidthCI(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        prob.setup()
        prob.set_val('x_out:ci_lower', 132.58208526)
        prob.set_val('x_out:ci_upper', 167.91431536)

        prob.run_model()
        self.partials = prob.check_partials(out_stream=None)
        self.prob = prob

    def test_partials(self):
        partial_lb = self.partials['dist'][('width', 'x_out:ci_lower')]['rel error'][0]
        partial_ub = self.partials['dist'][('width', 'x_out:ci_upper')]['rel error'][0]

        self.assertTrue(
            np.isclose(partial_lb, 0), msg="Partial of width wrt CI_lower error."
        )
        self.assertTrue(
            np.isclose(partial_ub, 0), msg="Partial of width wrt CI_upper error."
        )

    def test_compute(self):
        ub = self.prob.get_val('x_out:ci_upper')
        lb = self.prob.get_val('x_out:ci_lower')
        calc = ub - lb
        diff_truth = 167.91431536 - 132.58208526

        self.assertTrue(
            np.isclose(calc, diff_truth), msg="CI width error."
        )


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.main()