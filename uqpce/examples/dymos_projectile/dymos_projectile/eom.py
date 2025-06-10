import jax.numpy as jnp
import openmdao.api as om
class EOM(om.JaxExplicitComponent):
    """
    Jax Explicit Component which computes the equations of motion used by Dymos
    for trajectory interpolation.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('y', shape=(nn,), units='m')
        self.add_input('vx', shape=(nn,), units='m/s')
        self.add_input('vy', shape=(nn,), units='m/s')
        self.add_input('c_d', shape=(1), units='unitless')
        self.add_input('A', shape=(1), units='m**2')
        self.add_input('m', shape=(1), units='kg')
        self.add_input('g', shape=(1), val=9.80665, units='m/s**2')
        self.add_input('rho_0', shape=(1), val=1.22, units='kg/m**3')

        self.add_output('x_dot', shape=(nn,), units='m/s', tags=['dymos.state_rate_source:x', 'dymos.state_units:m'])
        self.add_output('y_dot', shape=(nn,), units='m/s', tags=['dymos.state_rate_source:y', 'dymos.state_units:m'])
        self.add_output('vx_dot', shape=(nn,), units='m/s**2', tags=['dymos.state_rate_source:vx', 'dymos.state_units:m/s'])
        self.add_output('vy_dot', shape=(nn,), units='m/s**2', tags=['dymos.state_rate_source:vy', 'dymos.state_units:m/s'])
        self.add_output('v', shape=(nn,), units='m/s')
    
    def compute_primal(self, y, vx, vy, c_d, A, m, g, rho_0):
        theta = jnp.arctan2(vy, vx)
        rho = rho_0*jnp.exp(-y/8500)
        v = jnp.sqrt(vx**2 + vy**2)
        D = 0.5*rho*(v**2)*c_d*A

        vx_dot = (-D/m) * jnp.cos(theta)
        vy_dot = -g - (D/m) * jnp.sin(theta)
        x_dot = vx
        y_dot = vy

        return x_dot, y_dot, vx_dot, vy_dot, v
