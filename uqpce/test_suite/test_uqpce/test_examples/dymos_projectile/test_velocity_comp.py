import unittest
import numpy as np
import openmdao.api as om
import openmdao.utils.assert_utils as om_assert
from uqpce.examples.dymos_projectile.dymos_projectile.v_comp import V_Comp


class TestCost(unittest.TestCase):
    def setUp(self):
        # Number of sample points
        resp_cnt = 12
        prob = om.Problem()
        prob.model.add_subsystem(
            'vcomp', V_Comp(num_samples=resp_cnt), promotes_inputs=['v_in', 'theta'], promotes_outputs=['vx', 'vy']
        )

        prob.setup(force_alloc_complex=True)
        v_in = np.repeat(80, resp_cnt)
        # Theta values originate from normal distribution sample points
        theta = [3.017970710922082489e+01,
                 3.318668458265722165e+01,
                 2.805582049505525077e+01,
                 3.208790110869553303e+01,
                 3.189874578803349436e+01,
                 2.891264889449846720e+01,
                 2.677466746927050778e+01,
                 2.919262369162053616e+01,
                 3.101082016363410077e+01,
                 2.820730556964451097e+01,
                 3.042781095372752276e+01,
                 2.971544398669106002e+01]

        prob.set_val('v_in', v_in)
        prob.set_val('theta', theta, units='deg')

        prob.run_model()
        self.partials = prob.check_partials(out_stream=None, method='cs')
        self.prob = prob

    def test_partials(self):
        partials = self.partials
        om_assert.assert_check_partials(partials, atol=1e-6, rtol=1e-6)

    def test_compute(self):
        vx = self.prob.get_val('vx').tolist()
        vy = self.prob.get_val('vy').tolist()

        # Truth values computed from vx = v * cos(theta[i]) and vy = v * sin(theta[i])
        vx_sol = [69.16, 66.95, 70.60, 67.78, 67.92, 70.03, 71.43, 69.84, 68.57, 70.50, 68.98, 69.48]
        vy_sol = [40.21, 43.79, 37.63, 42.50, 42.28, 38.67, 36.03, 39.02, 41.22, 37.82, 40.52, 39.66]

        self.assertTrue(
            np.isclose(vx, vx_sol, rtol=1e-1).all(), msg="vx computation error."
        )
        self.assertTrue(
            np.isclose(vy, vy_sol, rtol=1e-1).all(), msg="vy computation error."
        )


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.main()