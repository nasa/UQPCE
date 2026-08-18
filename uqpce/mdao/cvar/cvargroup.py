import numpy as np
import openmdao.api as om
import jax.numpy as jnp

from uqpce.mdao.cdf.cdfresidcomp import CDFResidComp
from uqpce.mdao.cvar.cvartailcomp import CVaRTailComp


class CVaRGroup(om.Group):
    """
    Group that calculates the smoothed conditional value at risk (CVaR) of
    the resampled responses.

    The confidence interval is solved with the same tanh-CDF residual and
    balance as CDFGroup, and CVaRTailComp then averages the samples beyond
    it with matching tanh weights. Under mixed uncertainty the epistemic
    CVaR curves are aggregated with a KS function, as in CDFGroup.
    """

    def initialize(self):
        self.options.declare('vec_size', types=int)
        self.options.declare(
            'alpha', types=float, default=0.05,
            desc='Single-sided upper confidence interval of (1-alpha)'
        )
        self.options.declare('tanh_omega', types=float, default=1e-6)
        self.options.declare(
            'tail', values=['lower', 'upper'], allow_none=False
        )
        self.options.declare('aleatory_cnt', types=int, allow_none=False)
        self.options.declare('epistemic_cnt', types=int, allow_none=False)
        self.options.declare(
            'sample_ref0', types=(float, int), default=0.0,
            desc='Scaling parameter. The value in the user-defined units of '
            'this output variable when the scaled value is 0. Default is 0.'
        )
        self.options.declare(
            'sample_ref', types=(float, int), default=1.0,
            desc='Scaling parameter. The value in the user-defined units of '
            'this output variable when the scaled value is 1. Default is 1.'
        )

    def setup(self):

        vec_size = self.options['vec_size']
        alpha = self.options['alpha']
        tanh_omega = self.options['tanh_omega']
        tail = self.options['tail']
        aleat_cnt = self.options['aleatory_cnt']
        epist_cnt = self.options['epistemic_cnt']
        sample_ref0 = self.options['sample_ref0']
        sample_ref = self.options['sample_ref']

        self.add_subsystem(
            'cdf', CDFResidComp(
                vec_size=vec_size, alpha=alpha, tanh_omega=tanh_omega,
                tail=tail, aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt,
                sample_ref0=sample_ref0, sample_ref=sample_ref
            ),
            promotes_inputs=[('samples', 'f_sampled'), 'f_ci'],
            promotes_outputs=['ci_resid']
        )

        bal = self.add_subsystem(
            'bal', om.BalanceComp(val=jnp.ones([epist_cnt])),
            promotes_inputs=['ci_resid'], promotes_outputs=['f_ci']
        )
        bal.add_balance(
            name='f_ci', lhs_name='ci_resid', val=jnp.ones([epist_cnt])
        )

        self.add_subsystem(
            'cvar_tail', CVaRTailComp(
                vec_size=vec_size, alpha=alpha, tanh_omega=tanh_omega,
                tail=tail, aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt,
                sample_ref0=sample_ref0, sample_ref=sample_ref
            ),
            promotes_inputs=[('samples', 'f_sampled'), 'f_ci'],
            promotes_outputs=['f_cvar']
        )

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()

        minimum = (self.options['tail'] == 'lower')

        if vec_size == aleat_cnt:  # purely aleatoric
            pass
        else:
            self.add_subsystem(
                'ks', om.KSComp(width=epist_cnt, minimum=minimum, rho=1000.),
                promotes_outputs=[('KS', 'cvar')]
            )
            self.connect('f_cvar', 'ks.g')

    def guess_nonlinear(self, inputs, outputs, residuals):
        aleatory_cnt = self.options['aleatory_cnt']
        samples = inputs['cdf.samples']
        x = jnp.reshape(samples, (-1, aleatory_cnt))
        outputs['f_ci'] = jnp.percentile(  # find CI of curves
            x, self._get_subsystem('cdf')._sig*100, axis=1
        )


if __name__ == '__main__':

    from scipy.stats import norm

    alpha = 0.05
    aleat_cnt = 100_000
    epist_cnt = 1
    vec_size = aleat_cnt * epist_cnt

    np.random.seed(1)
    samps = norm.rvs(0, 1, size=vec_size)

    prob = om.Problem(reports=False)
    for tail in ('lower', 'upper'):
        prob.model.add_subsystem(
            tail,
            CVaRGroup(
                alpha=alpha, tanh_omega=0.01, tail=tail, vec_size=vec_size,
                epistemic_cnt=epist_cnt, aleatory_cnt=aleat_cnt,
                sample_ref0=0.0, sample_ref=1.0
            ),
            promotes_inputs=['*']
        )

    prob.setup()
    prob.set_val('f_sampled', samps)
    prob.run_model()

    p = alpha/2
    exact = norm.pdf(norm.ppf(p))/p
    print('lower CVaR:', prob.get_val('lower.f_cvar'), 'analytic:', -exact)
    print('upper CVaR:', prob.get_val('upper.f_cvar'), 'analytic:', exact)
