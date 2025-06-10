import numpy as np
import openmdao.api as om
class WidthCI(om.ExplicitComponent):
    """
    OpenMDAO Explicit Component which computes the difference between
    upper and lower confidence interval bounds (with respect to distance traveled).
    Component behaves as a constraint.
    """
    def setup(self):
        self.add_input('x_out:ci_lower')
        self.add_input('x_out:ci_upper')
        self.add_output('width')

    def setup_partials(self):
        self.declare_partials(of='width', wrt='x_out:ci_lower')
        self.declare_partials(of='width', wrt='x_out:ci_upper')

    def compute(self, inputs, outputs):
        outputs['width'] = inputs['x_out:ci_upper'] - inputs['x_out:ci_lower']

    def compute_partials(self, inputs, partials):
        partials['width', 'x_out:ci_lower'] = -1.0
        partials['width', 'x_out:ci_upper'] = 1.0
