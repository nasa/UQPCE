import unittest

import numpy as np
import openmdao.api as om

from uqpce.mdao.coeffcomp import CoefficientsComp

tanh_omega = 1e-6
aleat_cnt = 75_000

class TestCoefficientsComp(unittest.TestCase):
    def setUp(self):

        prob = om.Problem()
        var_basis = np.array([
                [1.00000000e+00, -1.69821781e+00, 9.67008130e-01],
                [ 1.00000000e+00, 2.64654707e-01, -5.54940304e-01],
                [ 1.00000000e+00, 9.04884946e-02, -9.27743660e-01],
                [ 1.00000000e+00, -1.27260262e+00, -3.13090952e-02],
                [ 1.00000000e+00, -7.19942644e-01, 4.53624248e-01,]
            ])
    
        prob.model.add_subsystem(
            'comp', CoefficientsComp(var_basis=var_basis),
            promotes_inputs=['*'], promotes_outputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val(
            'responses', np.array([
                12.16415154, 9.01613385, 7.20322875, 8.36029185, 8.540021
            ])
        )

        prob.run_model()
        self.partials = prob.check_partials(out_stream=None, method='cs')
        self.prob = prob

    def test_partials(self):
        coeff_err_coeffs = (
            self.partials['comp']
            [('matrix_coeffs', 'responses')]['rel error'][0]
        )
        coeff_err_mean = (
            self.partials['comp'][('mean', 'responses')]
            ['rel error'][0]
        )
        self.assertTrue(
            np.isclose(coeff_err_coeffs, 0), 
            msg='CoefficientsComp derivative (\'matrix_coeffs\', \'responses\')'
            ' is not correct'
        )
        self.assertTrue(
            np.isclose(coeff_err_mean, 0), 
            msg='CoefficientsComp derivative (\'mean\', \'responses\') is not '
            'correct'
        )

    def test_coeffs(self):
        act_coeffs = np.array([9.24875496, 0.22670994, 2.18217806])
        calc_coeffs = self.prob.get_val('comp.matrix_coeffs')
        self.assertTrue(
            np.isclose(calc_coeffs, act_coeffs), 
            msg='CoefficientsComp is not calculating coefficients correctly'
        )


if __name__ == '__main__':

    np.random.seed(33)

    suite = unittest.TestSuite()
    unittest.main()
