import numpy as np
import openmdao.api as om
import jax
import jax.numpy as jnp


class CVaRTailComp(om.JaxExplicitComponent):
    """
    Component class to calculate the smoothed conditional value at risk
    (CVaR) from the samples and the solved confidence interval, using the
    Rockafellar-Uryasev form with a softplus in place of the hinge:

        CVaR_upper = f_ci + mean(softplus_omega(x - f_ci)) / (alpha/2)
        CVaR_lower = f_ci - mean(softplus_omega(f_ci - x)) / (alpha/2)

    Since softplus_omega(u) = omega*log(1 + exp(u/omega)) >= max(u, 0), the
    smoothing bias is always outward (conservative): the upper CVaR is
    overestimated and the lower CVaR is underestimated, mirroring the
    conservative bias of the tanh-smoothed confidence interval. The
    smoothing width and sample scaling match CDFResidComp: omega is applied
    in the (sample_ref0, sample_ref)-scaled space.
    """

    def initialize(self):
        self.options.declare('vec_size', types=int)

        # The probability of the response is greater than the 1-alpha value
        # i.e. alpha=0.05 corresponds to the cumulative probability of 95%
        self.options.declare(
            'alpha', types=float, default=0.05,
            desc='Single-sided upper confidence interval of (1-alpha)'
        )
        self.options.declare('tanh_omega', types=float, default=1e-6)
        self.options.declare('aleatory_cnt', types=int, allow_none=False)
        self.options.declare('epistemic_cnt', types=int, allow_none=False)
        self.options.declare(
            'tail', values=['lower', 'upper'], allow_none=False
        )
        self.options.declare(
            'sample_ref0', types=(float, int), default=0.0,
            desc='Reference scale for 0 of the sample data'
        )
        self.options.declare(
            'sample_ref', types=(float, int), default=1.0,
            desc='Reference scale for 1 of the sample data'
        )

        self._no_check_partials = True

    def setup(self):
        aleat_cnt = self.options['aleatory_cnt']
        epist_cnt = self.options['epistemic_cnt']

        self.add_input('samples', shape=(epist_cnt*aleat_cnt,))
        self.add_input('f_ci', shape=(epist_cnt,))

        self.add_output('f_cvar', shape=(epist_cnt,))

        self._tail_frac = self.options['alpha']/2

    def get_self_statics(self):
        return (
            self.options['alpha'], self.options['tanh_omega'],
            self.options['aleatory_cnt'], self.options['sample_ref0'],
            self.options['sample_ref'], self.options['tail']
        )

    def compute_primal(self, samples, f_ci):
        sample_ref0 = self.options['sample_ref0']
        sample_ref = self.options['sample_ref']
        aleat_cnt = self.options['aleatory_cnt']
        tanh_omega = self.options['tanh_omega']
        tail = self.options['tail']

        f_sampled = (samples - sample_ref0) / sample_ref
        f_ci_scaled = (f_ci - sample_ref0) / sample_ref

        x = jnp.transpose(jnp.reshape(f_sampled, (-1, aleat_cnt)))

        if tail == 'upper':
            excess = jax.nn.softplus((x - f_ci_scaled)/tanh_omega)*tanh_omega
            cvar_scaled = f_ci_scaled + (
                jnp.sum(jnp.transpose(excess), axis=1)/aleat_cnt
            )/self._tail_frac
        else:
            shortfall = jax.nn.softplus((f_ci_scaled - x)/tanh_omega)*tanh_omega
            cvar_scaled = f_ci_scaled - (
                jnp.sum(jnp.transpose(shortfall), axis=1)/aleat_cnt
            )/self._tail_frac

        return cvar_scaled*sample_ref + sample_ref0


if __name__ == '__main__':
    from scipy.stats import norm

    alpha = 0.05
    epist_cnt = 1
    aleat_cnt = 100_000
    vec_size = aleat_cnt*epist_cnt

    np.random.seed(1)
    samps = norm.rvs(0, 1, size=vec_size)
    ci = np.quantile(samps, alpha/2)

    prob = om.Problem(reports=False)
    prob.model.add_subsystem(
        'cvar_tail', CVaRTailComp(
            vec_size=vec_size, alpha=alpha, tanh_omega=0.01, tail='lower',
            aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt
        ),
        promotes_inputs=['samples', 'f_ci'], promotes_outputs=['f_cvar']
    )

    prob.setup()
    prob.set_val('samples', samps)
    prob.set_val('f_ci', ci)
    prob.run_model()

    p = alpha/2
    print('smoothed CVaR:', prob.get_val('f_cvar'))
    print('exact tail mean:', samps[samps <= ci].mean())
    print('analytic:', -norm.pdf(norm.ppf(p))/p)
