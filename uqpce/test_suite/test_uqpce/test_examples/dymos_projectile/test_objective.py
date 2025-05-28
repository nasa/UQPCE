import unittest
import numpy as np
import openmdao.api as om
from uqpce.examples.dymos_projectile.dymos_projectile.obj import Obj


class TestCost(unittest.TestCase):
    def setUp(self): 
        # Number of sample points
        resp_cnt = 12
        prob = om.Problem()
        prob.model.add_subsystem(
            'obj_comp', Obj(num_samples=resp_cnt), promotes_inputs=['*'], promotes_outputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('obj_comp.x_out:mean', 140.0)
        prob.set_val('obj_comp.x_out:variance', 62.35886459)

        prob.run_model()
        self.partials = prob.check_partials(out_stream=None, method='cs')
        self.prob = prob

    def test_partials(self):
        partial_mu = self.partials['obj_comp'][('obj', 'x_out:mean')]['rel error'][0]

        self.assertTrue(
            np.isclose(partial_mu, 0), msg="Partial of obj wrt mean error."
        )

    def test_compute(self):
        calc = (150 - self.prob.get_val('x_out:mean')) ** 2

        self.assertTrue(
            np.isclose(calc, 100), msg="Objective computation error."
        )


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.main()