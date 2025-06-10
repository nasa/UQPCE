import openmdao.api as om
import numpy as np
from numpy.linalg import solve

class CoefficientsComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('var_basis', types=np.ndarray)

        self._no_check_partials = True

    def setup(self):
        var_basis = self.options['var_basis']
        resp_cnt, term_cnt = var_basis.shape

        self._var_basis_T = var_basis.T
        self._basis_transform = np.dot(self._var_basis_T, var_basis)

        self.add_input('responses', shape=(resp_cnt,))
        self.add_output('matrix_coeffs', shape=(term_cnt,))
        self.add_output('mean', shape=(1,))

        self.declare_partials(
            of='matrix_coeffs', wrt='responses', 
            val=np.dot(np.linalg.inv(self._basis_transform), self._var_basis_T)
        )
        self.declare_partials(
            of='mean', wrt='responses', 
            val=np.dot(np.linalg.inv(self._basis_transform), self._var_basis_T)[0,:]
        )

    def compute(self, inputs, outputs):
        responses = inputs['responses']

        outputs['matrix_coeffs'] = solve(
            self._basis_transform, np.dot(self._var_basis_T, responses)
        )
        outputs['mean'] = outputs['matrix_coeffs'][0]


if __name__ == '__main__':
    import numpy as np

    prob = om.Problem()
    var_basis = np.array([
            [1.00000000e+00, -1.69821781e+00, 9.67008130e-01],
            [ 1.00000000e+00, 2.64654707e-01, -5.54940304e-01],
            [ 1.00000000e+00, 9.04884946e-02, -9.27743660e-01],
            [ 1.00000000e+00, -1.27260262e+00, -3.13090952e-02],
            [ 1.00000000e+00, -7.19942644e-01, 4.53624248e-01,]
        ])
    prob.model.add_subsystem(
        'comp', CoefficientsComp(var_basis=var_basis),
        promotes_inputs=['*'], promotes_outputs=['*']
    )

    prob.setup(force_alloc_complex=True)
    prob.set_val(
        'responses', np.array([
            12.16415154, 9.01613385, 7.20322875, 8.36029185, 8.540021
        ])
    )

    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
