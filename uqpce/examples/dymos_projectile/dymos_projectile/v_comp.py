import numpy as np
import jax.numpy as jnp
import openmdao.api as om
class V_Comp(om.ExplicitComponent):
    """
    OpenMDAO Explicit Component which computes x and y velocity components
    provided the angle theta and initial velocity.
    """
    def initialize(self):
        self.options.declare('num_samples', types=int)

    def setup(self):
        n = self.options['num_samples']

        self.add_input('v_in', shape=(n,), units='m/s')
        self.add_input('theta', shape=(n,), units='rad')

        self.add_output('vx', shape=(n,), units='m/s')
        self.add_output('vy', shape=(n,), units='m/s')

    def setup_partials(self):
        n = self.options['num_samples']
        ar = np.arange(n, dtype=int)
        self.declare_partials('*', '*', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        th_rad = inputs['theta']
        vx = inputs['v_in']*jnp.cos(th_rad)
        vy = inputs['v_in']*jnp.sin(th_rad)

        outputs['vx'] = vx
        outputs['vy'] = vy

    def compute_partials(self, inputs, partials):
        n = self.options['num_samples']
        th_rad = inputs['theta']
        partials['vx', 'theta'] = inputs['v_in'] * (-1)*np.sin(th_rad)
        partials['vx', 'v_in'] = np.cos(th_rad)
        partials['vy', 'theta'] = inputs['v_in'] * np.cos(th_rad)
        partials['vy', 'v_in'] = np.sin(th_rad)
