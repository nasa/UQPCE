import openmdao.api as om

class Paraboloid(om.ExplicitComponent):
    """
    Evaluates the equation f(a,b,x,y) = (a*x-3)**2 + a*b*x*y + (b*y+4)**2 - 3
    """
    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']

        # Add two uncertain variables of length n
        self.add_input('uncerta', shape=(n,))
        self.add_input('uncertb', shape=(n,))

        # Add two design variables
        self.add_input('desx') 
        self.add_input('desy')

        # Add output of length n
        self.add_output('f_abxy', shape=(n,))

    def setup_partials(self):
        n = self.options['vec_size']
        incr = np.linspace(0, n-1, n)

        self.declare_partials(of='f_abxy', wrt=['desx', 'desy'])
        self.declare_partials(
            of='f_abxy', wrt=['uncerta', 'uncertb'], rows=incr, cols=incr
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        a = inputs['uncerta']
        b = inputs['uncertb']

        x = inputs['desx']
        y = inputs['desy']

        partials['f_abxy', 'desx'] = 2*a**2*x + a*(-6 + b*y)
        partials['f_abxy', 'desy'] =  8*b + a*b*x + 2*b**2*y
        partials['f_abxy', 'uncerta'] =  -6*x + 2*a*x**2 + b*x*y
        partials['f_abxy', 'uncertb'] =  8*y + a*x*y + 2*b*y**2

    def compute(self, inputs, outputs):
        a = inputs['uncerta']
        b = inputs['uncertb']

        x = inputs['desx']
        y = inputs['desy']

        outputs['f_abxy'] = (a*x-3)**2 + a*b*x*y + (b*y+4)**2 - 3

if __name__ == '__main__':
    import numpy as np

    prob = om.Problem()
    prob.model.add_subsystem(
        'comp', Paraboloid(vec_size=4),
        promotes_inputs=['*'], promotes_outputs=['*']
    )
    prob.setup(force_alloc_complex=True)
    prob.set_val('uncerta', np.array([1.25, 1.5, 1.75, 2.0]))
    prob.set_val('uncertb', np.array([5.0, 4.0, 3.0, 2.0]))
    prob.set_val('desx', 8)
    prob.set_val('desy', 12)
    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
