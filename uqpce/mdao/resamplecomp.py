import openmdao.api as om
import numpy as np

class ResampleComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('resampled_var_basis', types=np.ndarray)


    def setup(self):
        resamp_var_basis = self.options['resampled_var_basis']
        resamp_resp_cnt, term_cnt = resamp_var_basis.shape

        self.add_input('matrix_coeffs', shape=(term_cnt,))
        self.add_output('resampled_responses', shape=(resamp_resp_cnt,))

        self.declare_partials(
            of='resampled_responses', wrt='matrix_coeffs', 
            val=resamp_var_basis
        )

    def compute(self, inputs, outputs):
        matrix_coeffs = inputs['matrix_coeffs']
        resamp_var_basis = self.options['resampled_var_basis']

        outputs['resampled_responses'] = np.matmul(
            resamp_var_basis, matrix_coeffs
        )

if __name__ == '__main__':
    import numpy as np

    prob = om.Problem()
    resampled_var_basis = np.array([
            [1.00000000e+00, -1.69821781e+00, 9.67008130e-01],
            [ 1.00000000e+00, 2.64654707e-01, -5.54940304e-01],
            [ 1.00000000e+00, 9.04884946e-02, -9.27743660e-01],
            [ 1.00000000e+00, -1.27260262e+00, -3.13090952e-02],
            [ 1.00000000e+00, -7.19942644e-01, 4.53624248e-01,]
        ])
    prob.model.add_subsystem(
        'comp', ResampleComp(resampled_var_basis=resampled_var_basis), 
        promotes_inputs=['*'], promotes_outputs=['*']
    )

    prob.setup(force_alloc_complex=True)
    prob.set_val('matrix_coeffs', np.array([9.24875496, 0.22670994, 2.18217806]))

    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
