import numpy as np
import openmdao.api as om

from uqpce.mdao.cdf.cdfcomp import CDFComp
from uqpce.mdao.cdf.cdfgroup import CDFGroup
from uqpce.mdao.cvar.cvarcomp import CVaRComp
from uqpce.mdao.cvar.cvargroup import CVaRGroup
from uqpce.mdao.coeffcomp import CoefficientsComp
from uqpce.mdao.meanplusvarcomp import MeanPlusVarComp
from uqpce.mdao.resamplecomp import ResampleComp
from uqpce.mdao.sobolcomp import SobolComp
from uqpce.mdao.variancecomp import VarianceComp


class UQPCEGroup(om.Group):
    """
    Class definition for the UQPCEGroup.

    A UQPCEGroup object builds a Polynomial Chaos Expansion (PCE) model for
    `one` arbitrary response. This object outputs statistics for the mean,
    variance, and confidence interval on a given response.
    """

    def initialize(self):
        """
        Declare any options for a UQPCEGroup.
        """
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
        # Sobol parameters
        self.options.declare(
            'compute_sobols', types=bool, default=False,
            desc='Whether to compute Sobol sensitivity indices. '
                 'If True, model_matrix must be provided.'
        )
        self.options.declare(
            'model_matrix', types=(np.ndarray, type(None)), default=None,
            desc='Interaction matrix for computing Sobol indices. '
                 'Shape: (n_terms, n_vars). Entry [i,j] indicates if variable j '
                 'appears in PCE term i. Required if compute_sobols=True.'
        )
        # Activation function parameters
        self.options.declare(
            'use_tanh_ci', types=bool, default=False,
            desc='A flag for if the former complex-safe tanh method for '
            'calculating the confidence interval should be used.'
        )
        self.options.declare('tanh_omega', types=(float, int), default=1e-6)
        self.options.declare(
            'sample_ref0', types=(int, float), default=0,
            desc='Reference scale for 0 of the sample data'
        )
        self.options.declare(
            'sample_ref', types=(int, float), default=1,
            desc='Reference scale for 1 of the sample data'
        )
        self.options.declare(
            'compute_cvar', types=bool, default=False,
            desc='Whether to also compute the conditional value at risk '
            '(the mean of the tail beyond the confidence interval) for '
            'each requested tail. Uses the tanh-smoothed form when '
            'use_tanh_ci is True and the exact quantile form otherwise.'
        )
        self._coeff_comp = CoefficientsComp

    def _coeff_kwargs(self):
        return {'var_basis': self.options['var_basis']}

    def _coeff_inputs(self):
        return ['responses']

    def _ci_kwargs(self):
        alpha = self.options['significance']
        vec_size = self.options['resampled_var_basis'].shape[0]
        oms = self.options['tanh_omega']
        ref0 = self.options['sample_ref0']
        ref = self.options['sample_ref']
        use_tanh_ci = self.options['use_tanh_ci']
        aleatory_cnt = self.options['aleatory_cnt']
        epistemic_cnt = self.options['epistemic_cnt']

        ci_kwargs = {
            'alpha': alpha, 'aleatory_cnt': aleatory_cnt,
            'epistemic_cnt': epistemic_cnt, 'vec_size': vec_size,
        }
        if use_tanh_ci:
            ci_kwargs.update({
                'sample_ref0': ref0, 'sample_ref': ref, 'tanh_omega': oms
            })

        return ci_kwargs

    def _ci_comp(self):
        use_tanh_ci = self.options['use_tanh_ci']

        if not use_tanh_ci:
            _ci_calc = CDFComp
        else:
            _ci_calc = CDFGroup

        return _ci_calc

    def _out_ci(self):
        aleatory_cnt = self.options['aleatory_cnt']
        vec_size = self.options['resampled_var_basis'].shape[0]
        use_tanh_ci = self.options['use_tanh_ci']

        if not use_tanh_ci:
            out_ci = 'f_ci'
        else:
            out_ci = 'f_ci' if vec_size == aleatory_cnt else 'ci'

        return out_ci

    def _cvar_comp(self):
        use_tanh_ci = self.options['use_tanh_ci']

        if not use_tanh_ci:
            _cvar_calc = CVaRComp
        else:
            _cvar_calc = CVaRGroup

        return _cvar_calc

    def _out_cvar(self):
        aleatory_cnt = self.options['aleatory_cnt']
        vec_size = self.options['resampled_var_basis'].shape[0]
        use_tanh_ci = self.options['use_tanh_ci']

        if not use_tanh_ci:
            out_cvar = 'f_cvar'
        else:
            out_cvar = 'f_cvar' if vec_size == aleatory_cnt else 'cvar'

        return out_cvar

    def setup(self):
        """
        Setup the UQPCEGroup.
        """
        kwargs = self._coeff_kwargs()
        coeff_inputs = self._coeff_inputs()

        ci_kwargs = self._ci_kwargs()
        ci_calc = self._ci_comp()
        out_ci = self._out_ci()

        tail = self.options['tail']
        tails = [tail] if tail != 'both' else ['lower', 'upper']
        aleatory_cnt = self.options['aleatory_cnt']
        epistemic_cnt = self.options['epistemic_cnt']
        vec_size = self.options['resampled_var_basis'].shape[0]
        norm_sq = self.options['norm_sq']

        if vec_size != aleatory_cnt and vec_size != (epistemic_cnt*aleatory_cnt):
            exit(
                'The length of your `resampled_var_basis` should equal either '
                'the aleatory count of the aleatory count times the epistemic '
                'count.'
            )

        # Add the system which outputs the matrix coefficients
        self.add_subsystem(
            'coeff_comp',
            self._coeff_comp(**kwargs),
            promotes_inputs=coeff_inputs,
            promotes_outputs=['matrix_coeffs', 'mean']
        )

        # Add the system which outputs resampled responses
        self.add_subsystem(
            'resamp_comp',
            ResampleComp(resampled_var_basis=self.options['resampled_var_basis']),
            promotes_inputs=['matrix_coeffs'],
            promotes_outputs=['resampled_responses']
        )

        # Add the system which outputs variance
        self.add_subsystem(
            'var_comp', VarianceComp(norm_sq=norm_sq),
            promotes_inputs=['matrix_coeffs'], promotes_outputs=['variance']
        )

        self.add_subsystem(
            'mean_plus_var_comp', MeanPlusVarComp(),
            promotes_inputs=['mean', 'variance'],
            promotes_outputs=['mean_plus_var']
        )

        # Add Sobol sensitivity component if requested
        if self.options['compute_sobols']:
            model_matrix = self.options['model_matrix']
            if model_matrix is None:
                raise ValueError(
                    'model_matrix must be provided when compute_sobols=True'
                )

            self.add_subsystem(
                'sobol_comp',
                SobolComp(norm_sq=norm_sq, model_matrix=model_matrix),
                promotes_inputs=['matrix_coeffs'],
                promotes_outputs=['sobols', 'total_sobols']
            )

        for curr_tail in tails:
            ci_kwargs.update({'tail': curr_tail})
            self.add_subsystem(
                f'{curr_tail}_cdf_group',
                ci_calc(**ci_kwargs),
                promotes_inputs=[('f_sampled', 'resampled_responses')],
                promotes_outputs=[(out_ci, f'ci_{curr_tail}')]
            )

        if self.options['compute_cvar']:
            cvar_calc = self._cvar_comp()
            out_cvar = self._out_cvar()

            for curr_tail in tails:
                cvar_kwargs = dict(ci_kwargs)
                cvar_kwargs.update({'tail': curr_tail})
                self.add_subsystem(
                    f'{curr_tail}_cvar_group',
                    cvar_calc(**cvar_kwargs),
                    promotes_inputs=[('f_sampled', 'resampled_responses')],
                    promotes_outputs=[(out_cvar, f'cvar_{curr_tail}')]
                )


class MultiUQPCEGroup(UQPCEGroup):
    """
    Class definition for the MultiUQPCEGroup.

    A MultiUQPCEGroup object builds a Polynomial Chaos Expansion (PCE) model for an
    arbitrary response. This object outputs statistics for the mean, variance,
    and confidence interval on a given response.
    """

    def initialize(self):
        """
        Declare any options for a MultiUQPCEGroup.
        """
        super().initialize()

        self.options.declare(
            'uncert_list', allow_none=False,
            desc='The string names of the uncertain outputs for the user\'s problem.'
        )
        self.options.declare('tanh_omega', types=(list, float, int), default=1e-6)
        self.options.declare(
            'sample_ref0', types=(list, int, float), default=0,
            desc='Reference scale for 0 of the sample data'
        )
        self.options.declare(
            'sample_ref', types=(list, int, float), default=1,
            desc='Reference scale for 1 of the sample data'
        )
        self.options.declare(
            'use_tanh_ci', types=bool, default=False,
            desc='A flag for if the former complex-safe tanh method for '
            'calculating the confidence interval should be used.'
        )

    def _update_tanh_option(self, option, iter_cnt):
        """
        Ensures the tanh-specific options is a numpy array.
        """
        opt_type = type(option)

        if opt_type is list and len(option) == 1:  # List of size 1
            option = np.ones(iter_cnt, dtype=float) * option[0]
        elif opt_type in (float, int, np.float64, np.int64):
            option = np.ones(iter_cnt, dtype=float) * option
        elif opt_type is list and len(option) == iter_cnt:
            option = np.array(option, dtype=float)
        else:
            raise ValueError('The tanh input {option} should be reformatted.')

        return option

    def setup(self):
        """
        Setup the MultiUQPCEGroup.
        """
        cnt = 0
        uncert_list = self.options['uncert_list']
        resampled_var_basis = self.options['resampled_var_basis']
        var_basis = self.options['var_basis']
        norm_sq = self.options['norm_sq']
        sig = self.options['significance']
        epist_cnt = self.options['epistemic_cnt']
        aleat_cnt = self.options['aleatory_cnt']
        tail = self.options['tail']
        compute_sobols = self.options['compute_sobols']
        use_tanh_ci = self.options['use_tanh_ci']
        compute_cvar = self.options['compute_cvar']

        iter_cnt = len(uncert_list)
        tanh_omega = self._update_tanh_option(
            self.options['tanh_omega'], iter_cnt)
        sample_ref0 = self._update_tanh_option(
            self.options['sample_ref0'], iter_cnt)
        sample_ref = self._update_tanh_option(
            self.options['sample_ref'], iter_cnt)

        pce_outputs = ['variance', 'mean', 'resampled_responses',
                       'matrix_coeffs', 'mean_plus_var']

        if tail == 'lower' or tail == 'both':
            pce_outputs.append('ci_lower')
            if compute_cvar:
                pce_outputs.append('cvar_lower')
        if tail == 'upper' or tail == 'both':
            pce_outputs.append('ci_upper')
            if compute_cvar:
                pce_outputs.append('cvar_upper')
        if compute_sobols:
            pce_outputs.append('sobols')
            pce_outputs.append('total_sobols')

        for resp in uncert_list:

            resp_name = resp.replace(':', '_')

            outputs = []
            for op in pce_outputs:
                outputs.append((op, f'{resp_name}:{op}'))

            self.add_subsystem(
                f'{resp_name}_pce',
                UQPCEGroup(
                    var_basis=var_basis, norm_sq=norm_sq, significance=sig,
                    resampled_var_basis=resampled_var_basis, tail=tail,
                    aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt,
                    compute_sobols=compute_sobols, sample_ref0=sample_ref0[cnt],
                    sample_ref=sample_ref[cnt], tanh_omega=tanh_omega[cnt],
                    use_tanh_ci=use_tanh_ci, compute_cvar=compute_cvar
                ),
                promotes_inputs=[('responses', resp)], promotes_outputs=outputs
            )

            cnt += 1


if __name__ == '__main__':
    from uqpce.examples.paraboloid.paraboloid import paraboloid

    aleat_cnt = 100_000
    epist_cnt = 5
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
        'group',
        MultiUQPCEGroup(
            uncert_list=outputs, var_basis=var_basis, significance=sig,
            resampled_var_basis=resampled_var_basis, tail='both', norm_sq=norm_sq,
            aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt, use_tanh_ci=False
        ),
        promotes_inputs=['*'], promotes_outputs=['*']
    )

    prob.setup(force_alloc_complex=True)
    prob.set_val('uncerta', np.array([1, 2, 3, 4, 5, 6]))
    prob.set_val('uncertb', np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    prob.set_val('desx', 2)
    prob.set_val('desy', 3.1)
    prob.run_model()
    # prob.check_partials(compact_print=True, method='cs')#, form='central')

    print(prob.get_val('f_abxy:variance'))
    print(prob.get_val('f_abxy:mean'))
    print('UQPCE OM CI:     ', prob.get_val('f_abxy:ci_upper'))
    print('Interpolated CI: ', np.max(np.quantile(np.reshape(prob.get_val('f_abxy:resampled_responses'), (-1, aleat_cnt)), 1-sig/2, axis=1)))
    print('Interpolated All Epistemic CIs:', np.quantile(np.reshape(prob.get_val('f_abxy:resampled_responses'), (-1, aleat_cnt)), 1-sig/2, axis=1))

    print('\nUQPCE OM CI:     ', prob.get_val('f_abxy:ci_lower'))
    print('Interpolated CI: ', np.min(np.quantile(np.reshape(prob.get_val('f_abxy:resampled_responses'), (-1, aleat_cnt)), sig/2, axis=1)))
    print('Interpolated All Epistemic CIs:', np.quantile(np.reshape(prob.get_val('f_abxy:resampled_responses'), (-1, aleat_cnt)), sig/2, axis=1))

    # om.n2(prob)