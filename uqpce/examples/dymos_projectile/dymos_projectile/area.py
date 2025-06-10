import numpy as np
import openmdao.api as om
class Area(om.ExplicitComponent):
    """
    OpenMDAO Explicit Component which computes surface area provided
    the mass of the projectile.
    """
    def setup(self):
        self.add_input('m', units='kg')
        self.add_output('A', units='m**2')

    def setup_partials(self):
        self.declare_partials(of='A', wrt='m')

    def compute(self, inputs, outputs):
        # Density = 9340 kg/m^3 for metal ball
        const = 3 / (4*np.pi*9340)
        m = inputs['m']
        
        outputs['A'] = np.pi * (const * m)**(2/3)

    def compute_partials(self, inputs, partials):
        const = 3 / (4*np.pi*9340)
        m = inputs['m']

        partials['A', 'm'] = ((2*np.pi)/3) * ((1/(const*m))**(1/3)) * const
