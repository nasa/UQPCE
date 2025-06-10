import numpy as np
import openmdao.api as om
class Cost(om.ExplicitComponent):
    """
    OpenMDAO Explicit Component which computes cost using Kinetic Energy,
    a function of mass and velocity. Component behaves as a constraint.
    Underlying functionality transforms velocity from scalar to vector and
    feeds mass directly through input to output.
    """
    def initialize(self):
        self.options.declare('num_samples', types=int)

    def setup(self):
        n = self.options['num_samples']
        self.add_input('v', units='m/s')
        self.add_input('m', units='kg')

        self.add_output('cost', units='m')
        self.add_output('v_out', shape=(n,), units='m/s')
        self.add_output('m_out', units='kg')

    def setup_partials(self):
        self.declare_partials(of='m_out', wrt='m', val=1.0)
        self.declare_partials(of='v_out', wrt='v', val=1.0)
        self.declare_partials(of='cost', wrt='m')
        self.declare_partials(of='cost', wrt='v')
    
    def compute(self, inputs, outputs):
        n = self.options['num_samples']
        outputs['cost'] = 0.5 * inputs['m'] * inputs['v'] ** 2
        outputs['v_out'] = np.repeat(inputs['v'], n)
        outputs['m_out'] = inputs['m']

    def compute_partials(self, inputs, partials):
        partials['cost', 'v'] = inputs['m'] * inputs['v']
        partials['cost', 'm'] = 0.5 * inputs['v'] ** 2
