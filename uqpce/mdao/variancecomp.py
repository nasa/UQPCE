import openmdao.api as om
import numpy as np

class VarianceComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('norm_sq', types=np.ndarray)

        self._no_check_partials = True

    def setup(self):
        norm_sq = self.options['norm_sq']
        term_cnt = len(norm_sq)

        self.add_input('matrix_coeffs', shape=(term_cnt,))
        self.add_output('variance', shape=(1,))

    def compute(self, inputs, outputs):
        matrix_coeffs = inputs['matrix_coeffs']
        norm_sq = self.options['norm_sq']

        matrix_coeffs_sq = (
            np.reshape(
                matrix_coeffs, (len(matrix_coeffs), 1)
            )[1:] ** 2
        )
        norm_mult_coeff = norm_sq[1:] * matrix_coeffs_sq

        outputs['variance'] = np.sum(norm_mult_coeff)

    def setup_partials(self):
        self.declare_partials(of='variance', wrt='matrix_coeffs')

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        matrix_coeffs = inputs['matrix_coeffs']
        norm_sq = self.options['norm_sq']
        
        partials['variance', 'matrix_coeffs'][0,0] = 0
        partials['variance', 'matrix_coeffs'][0,1:] = 2 * matrix_coeffs[1:] * norm_sq[1:].T


if __name__ == '__main__':

    import numpy as np

    prob = om.Problem()
    norm_sq = np.array([[1], [1], [1 / 3]])
    prob.model.add_subsystem('comp', VarianceComp(norm_sq=norm_sq))

    prob.setup(force_alloc_complex=True)
    prob.set_val('comp.matrix_coeffs', np.array([9.24875496, 0.22670994, 2.18217806]))

    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
