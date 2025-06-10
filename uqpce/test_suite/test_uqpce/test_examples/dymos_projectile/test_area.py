import unittest
import numpy as np
import openmdao.api as om
from uqpce.examples.dymos_projectile.dymos_projectile.area import Area


class TestCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'area', Area(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('m', 1)

        prob.run_model()
        self.partials = prob.check_partials(out_stream=None, method='cs')
        self.prob = prob

    def test_partials(self):
        partial_m = self.partials['area'][('A', 'm')]['rel error'][0]

        self.assertTrue(
            np.isclose(partial_m, 0), msg="Partial of area wrt m error."
        )
    
    def test_compute(self):
        area = self.prob.get_val('A')

        # Truth value computed from derived equation A = pi * (3*m / (4*pi*9340))**(2/3) when m = 1
        area_truth = 0.002726

        self.assertTrue(
            np.isclose(area, area_truth), msg="Area computation error."
        )


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.main()