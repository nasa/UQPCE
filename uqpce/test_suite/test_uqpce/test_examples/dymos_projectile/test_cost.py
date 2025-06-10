import unittest
import numpy as np
import openmdao.api as om
from uqpce.examples.dymos_projectile.dymos_projectile.cost import Cost


class TestCost(unittest.TestCase):
    def setUp(self):
        # Number of sample points
        resp_cnt = 12
        prob = om.Problem()
        prob.model.add_subsystem(
            'cost', Cost(num_samples=resp_cnt), promotes_inputs=['*'], promotes_outputs=['*']
        )

        prob.setup()
        prob.set_val('v', 80)
        prob.set_val('m', 12)

        prob.run_model()
        self.partials = prob.check_partials(out_stream=None)
        self.prob = prob

    def test_partials(self):
        partial_v = self.partials['cost'][('cost', 'v')]['rel error'][0]
        partial_m = self.partials['cost'][('cost', 'm')]['rel error'][0]

        self.assertTrue(
            np.isclose(partial_v, 0), msg="Partial of cost wrt v error."
        )
        self.assertTrue(
            np.isclose(partial_m, 0), msg="Partial of cost wrt m error."
        )

    def test_compute(self):
        ke = self.prob.get_val('cost')

        # Truth value computed with KE = m*v**2 == (12)*(80)**2
        ke_truth = 38400

        self.assertTrue(
            np.isclose(ke, ke_truth), msg="Area computation error."
        )


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.main()