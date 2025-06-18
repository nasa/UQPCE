from typing import List

import openmdao.api as om
import numpy as np

from uqpce.mdao.coeffcomp import CoefficientsComp
from uqpce.mdao.resamplecomp import ResampleComp
from uqpce.mdao.variancecomp import VarianceComp
from uqpce.mdao.cdf.cdfgroup import CDFGroup
from uqpce.mdao.meanplusvarcomp import MeanPlusVarComp

class UQPCEGroup(om.Group):
    """
    Class definition for the UQPCEGroup.

    A UQPCEGroup object builds a Polynomial Chaos Expansion (PCE) model for an 
    arbitrary response. This object outputs statistics for the mean, variance, 
    and confidence interval on a given response.
    """

    def initialize(self):
        """
        Declare any options for a UQPCEGroup.
        """
        self.options.declare(
            'uncert_list', allow_none=False,
            desc='The string names of the uncertain outputs for the user\'s problem.'
        )
        self.options.declare(
            'var_basis', allow_none=False,
            desc='The evaluated variable basis of the PCE model.'
        )
        self.options.declare(
            'resampled_var_basis', allow_none=False,
            desc='The evaluated resampled variable basis of the PCE model.'
        )
        self.options.declare(
            'norm_sq', allow_none=False, 
            desc='The norm squared for the user\'s PCE model.'
        )
        self.options.declare(
            'tail', values=['lower', 'upper', 'both'], allow_none=False,
            desc='The tail from the two-sided uncertainty bound to be calculated.'
        )
        self.options.declare(
            'significance', types=float, default=0.05, 
            desc='The significance level of the uncertain problem (i.e. '
            'significance=0.05 corresponds to a 95% confidence interval).'
        )
        self.options.declare(
            'aleatory_cnt', types=int, allow_none=False, 
            desc='The number of aleatory samples used to resample the surrogate'
        )
        self.options.declare(
            'epistemic_cnt', types=int, allow_none=False, 
            desc='The number of epistemic samples used to resample the surrogate'
        )
        self.options.declare('tanh_omega', types=(list, float, int), default=1e-6)
        self.options.declare(
            'sample_ref0', types=(list, None, int, float), default=None, 
            desc='Reference scale for 0 of the sample data'
        )
        self.options.declare(
            'sample_ref', types=(list, None, int, float), default=None, 
            desc='Reference scale for 1 of the sample data'
        )

    def setup(self):
        """
        Setup the UQPCEGroup.
        """
        uncert_list = self.options['uncert_list']
        tail = self.options['tail']
        aleatory_cnt = self.options['aleatory_cnt']
        epistemic_cnt = self.options['epistemic_cnt']

        cnt = 0

        alpha = self.options['significance']
        vec_size = self.options['resampled_var_basis'].shape[0]
        if vec_size!=aleatory_cnt and vec_size!=(epistemic_cnt*aleatory_cnt):
            exit(
                'The length of your `resampled_var_basis` should equal either '
                'the aleatory count of the aleatory count times the epistemic '
                'count.'
            )

        out_ci = 'f_ci' if vec_size==aleatory_cnt else 'ci'
        use_ref0 = (not self.options['sample_ref0'] is None)
        use_ref = (not self.options['sample_ref'] is None)
        ref0 = 0.0
        ref = 1.0
        oms = self.options['tanh_omega']

        for resp in uncert_list:
            om = oms if ((type(oms) == list and len(oms)==1) or type(oms)==float or type(oms)==int) else oms[cnt]

            if use_ref0:
                ref = list(self.options['sample_ref0'])[cnt]
            if use_ref:
                ref = list(self.options['sample_ref'])[cnt]

            resp_subsys_name = resp.replace(':', '_')

            # Add the system which outputs the matrix coefficients
            self.add_subsystem(
                f'{resp_subsys_name}_coeff_comp', 
                CoefficientsComp(var_basis=self.options['var_basis']), 
                promotes_inputs=[('responses', resp)], 
                promotes_outputs=[
                    ('matrix_coeffs', f'{resp}:matrix_coeffs'),
                    ('mean', f'{resp}:mean')
                ]
            )

            # Add the system which outputs resampled responses
            self.add_subsystem(
                f'{resp_subsys_name}_resamp_comp', 
                ResampleComp(resampled_var_basis=self.options['resampled_var_basis']), 
                promotes_inputs=[('matrix_coeffs', f'{resp}:matrix_coeffs')], 
                promotes_outputs=[('resampled_responses', f'{resp}:resampled_responses')]
            )

            # Add the system which outputs variance
            self.add_subsystem(
                f'{resp_subsys_name}_var_comp', VarianceComp(norm_sq=self.options['norm_sq']), 
                promotes_inputs=[('matrix_coeffs', f'{resp}:matrix_coeffs')], 
                promotes_outputs=[('variance', f'{resp}:variance')]
            )

            self.add_subsystem(
                f'{resp_subsys_name}_mean_plus_var_comp', MeanPlusVarComp(), 
                promotes_inputs=[('mean', f'{resp}:mean'),('variance', f'{resp}:variance')], 
                promotes_outputs=[('mean_plus_var', f'{resp}:mean_plus_var')] 
            )

            if tail == 'lower' or tail == 'both':
                self.add_subsystem(
                    f'{resp_subsys_name}_lower_cdf_group',
                    CDFGroup(
                        alpha=alpha, tail='lower', aleatory_cnt=aleatory_cnt, 
                        epistemic_cnt=epistemic_cnt, vec_size=vec_size,
                        sample_ref0=ref0, sample_ref=ref, tanh_omega=om
                    ),
                    promotes_inputs=[('f_sampled', f'{resp}:resampled_responses')], 
                    promotes_outputs=[(out_ci, f'{resp}:ci_lower')]
                )

            if tail == 'upper' or tail == 'both':
                self.add_subsystem(
                    f'{resp_subsys_name}_upper_cdf_group',
                    CDFGroup(
                        alpha=alpha, tail='upper', aleatory_cnt=aleatory_cnt, 
                        epistemic_cnt=epistemic_cnt, vec_size=vec_size,
                        sample_ref0=ref0, sample_ref=ref, tanh_omega=om
                    ),
                    promotes_inputs=[('f_sampled', f'{resp}:resampled_responses')], 
                    promotes_outputs=[(out_ci, f'{resp}:ci_upper')]
                )

            cnt += 1


if __name__ == '__main__':
    from uqpce.examples.paraboloid.paraboloid import paraboloid

    aleat_cnt = 10_000
    epist_cnt = 250
    total_cnt = aleat_cnt*epist_cnt
    sig = 0.05

    norm_sq = np.array([[1], [1], [1 / 3]])
    var_basis = np.array([
        [1, -1.690e+00,  9.67e-01],
        [1,  2.646e-01, -5.55e-01],
        [1,  9.048e-02, -9.27e-01],
        [1, -1.270e+00, -3.01e-02],
        [1, -7.199e-01,  4.52e-01],
        [1, -1.272e+00, -2.02e-02]
    ])
    resampled_var_basis = np.zeros([total_cnt, 3])
    resampled_var_basis[:,0] = 1
    resampled_var_basis[:,1] = np.random.uniform(low=-2, high=2, size=total_cnt)
    resampled_var_basis[:,2] = np.random.uniform(low=-1, high=1, size=total_cnt)
    
    outputs = ['f_abxy'] 

    prob = om.Problem(reports=False)

    prob.model.add_subsystem(
        'parab', paraboloid.Paraboloid(vec_size=6),
        promotes_inputs=['*'], promotes_outputs=['*']
    )

    prob.model.add_subsystem(
        'comp',
        UQPCEGroup(
            uncert_list=outputs,
            var_basis=var_basis, norm_sq=norm_sq, significance=sig, 
            resampled_var_basis=resampled_var_basis, tail='both',
            aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt, sample_ref0=[100], 
            sample_ref=[125]
        ),
        promotes_inputs=['*'], promotes_outputs=['*']
    )
    # prob.model.add_design_var('desx', lower=0, upper=5) #unitless
    # prob.model.add_design_var('desy', lower=0, upper=5) #unitless
    # prob.model.add_objective('f_abxy:ci_upper')

    prob.setup(force_alloc_complex=True)
    prob.set_val('uncerta', np.array([1, 2, 3, 4, 5, 6]))
    prob.set_val('uncertb', np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    prob.set_val('desx', 2)
    prob.set_val('desy', 3.1)
    prob.run_model()
    # prob.run_driver()
    prob.check_partials(compact_print=True, method='cs')
    # prob.check_totals(method='fd', form='central')

    print(prob.get_val('f_abxy:variance'))
    print(prob.get_val('f_abxy:mean'))
    print('UQPCE OM CI:     ', prob.get_val(f'f_abxy:ci_upper'))
    print('Interpolated CI: ', np.max(np.quantile(np.reshape(prob.get_val(f'f_abxy:resampled_responses'), (-1, aleat_cnt)), 1-sig/2, axis=1)))
    print('UQPCE All Epistemic CIs:', prob.get_val(f'comp.f_abxy_upper_cdf_group.f_ci'))
    print('Interpolated All Epistemic CIs:', np.quantile(np.reshape(prob.get_val(f'f_abxy:resampled_responses'), (-1, aleat_cnt)), 1-sig/2, axis=1))

    print('UQPCE OM CI:     ', prob.get_val(f'f_abxy:ci_lower'))
    print('UQPCE OM CI:     ', prob.get_val(f'comp.f_abxy_lower_cdf_group.ks.g'))
    print('Interpolated CI: ', np.max(np.quantile(np.reshape(prob.get_val(f'f_abxy:resampled_responses'), (-1, aleat_cnt)), sig/2, axis=1)))
    print('UQPCE All Epistemic CIs:', prob.get_val(f'comp.f_abxy_lower_cdf_group.f_ci'))
    print('Interpolated All Epistemic CIs:', np.quantile(np.reshape(prob.get_val(f'f_abxy:resampled_responses'), (-1, aleat_cnt)), sig/2, axis=1))

    # om.n2(prob)