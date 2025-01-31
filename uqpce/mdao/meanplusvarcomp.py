import openmdao.api as om
import numpy as np

class MeanPlusVarComp(om.ExplicitComponent):

    def setup(self):

        self.add_input('mean', shape=(1,))
        self.add_input('variance', shape=(1,))
        self.add_output('mean_plus_var', shape=(1,))

        self.declare_partials(of='mean_plus_var', wrt='mean', val=1)
        self.declare_partials(of='mean_plus_var', wrt='variance', val=1)

    def compute(self, inputs, outputs):

        outputs['mean_plus_var'] = inputs['mean'] + inputs['variance']

if __name__ == '__main__':
    import numpy as np

    prob = om.Problem()
    prob.model.add_subsystem(
        'comp', MeanPlusVarComp(), promotes_inputs=['*'], promotes_outputs=['*']
    )

    prob.setup(force_alloc_complex=True)
    prob.set_val('mean', 4.7)
    prob.set_val('variance', 15)

    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
