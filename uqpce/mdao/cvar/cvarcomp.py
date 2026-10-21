import jax.numpy as jnp
import openmdao.api as om


class CVaRComp(om.JaxExplicitComponent):
    """
    Component class to calculate the differentiable conditional value at risk
    (CVaR, expected shortfall) of the resampled responses.

    The CVaR is the mean of the samples in the tail beyond the confidence
    interval that CDFComp reports. The tail convention matches CDFComp:
    alpha=0.05 gives a lower CVaR that is the mean of the worst 2.5% low
    samples (E[X | X <= q_{alpha/2}]) and an upper CVaR that is the mean of
    the worst 2.5% high samples (E[X | X >= q_{1-alpha/2}]).

    The calculation uses the Rockafellar-Uryasev form

        CVaR_upper = VaR + mean((x - VaR)_+) / (alpha/2)
        CVaR_lower = VaR - mean((VaR - x)_+) / (alpha/2)

    with VaR the exact sample quantile, which is well defined for repeated
    (discrete) sample values and reduces to the conditional tail mean for
    continuous samples.
    """

    def initialize(self):
        self.options.declare('vec_size', types=int)

        # The probability of the response is greater than the 1-alpha value
        # i.e. alpha=0.05 corresponds to the cumulative probability of 95%
        self.options.declare(
            'alpha', types=float, default=0.05,
            desc='Single-sided upper confidence interval of (1-alpha)'
        )
        self.options.declare('aleatory_cnt', types=int, allow_none=False)
        self.options.declare('epistemic_cnt', types=int, allow_none=False)
        self.options.declare(
            'tail', values=['lower', 'upper'], allow_none=False
        )

        self._no_check_partials = True

    def setup(self):
        alpha = self.options['alpha']
        aleat_cnt = self.options['aleatory_cnt']
        epist_cnt = self.options['epistemic_cnt']

        self.add_input('f_sampled', shape=(epist_cnt*aleat_cnt,))
        self.add_output('f_cvar', shape=(1,))

        self._sig = (1-alpha/2) if self.options['tail'] == 'upper' else alpha/2
        self._tail_frac = alpha/2

    def get_self_statics(self):
        return (
            self.options['alpha'], self.options['epistemic_cnt'],
            self.options['aleatory_cnt'], self.options['tail']
        )

    def compute_primal(self, f_sampled):
        aleat_cnt = self.options['aleatory_cnt']
        epist_cnt = self.options['epistemic_cnt']
        tail = self.options['tail']

        samps = jnp.reshape(f_sampled, (-1, aleat_cnt))

        if aleat_cnt != 1:
            vars_ = jnp.quantile(samps, self._sig, axis=1)

            if tail == 'upper':
                excess = jnp.maximum(samps - vars_[:, None], 0.0)
                cvars = vars_ + jnp.mean(excess, axis=1)/self._tail_frac
            else:
                shortfall = jnp.maximum(vars_[:, None] - samps, 0.0)
                cvars = vars_ - jnp.mean(shortfall, axis=1)/self._tail_frac

            # Mixed uncertainty
            if epist_cnt != 1:
                if tail == 'upper':
                    f_cvar = jnp.atleast_1d(jnp.max(cvars))
                else:
                    f_cvar = jnp.atleast_1d(jnp.min(cvars))
            else:  # Pure aleatory
                f_cvar = jnp.atleast_1d(cvars)
        else:  # Pure epistemic
            if tail == 'upper':
                f_cvar = jnp.atleast_1d(jnp.max(samps))
            else:
                f_cvar = jnp.atleast_1d(jnp.min(samps))

        return f_cvar


if __name__ == '__main__':
    import numpy as np
    from scipy.stats import norm

    alpha = 0.05
    aleat_cnt = 100_000
    epist_cnt = 1
    vec_size = aleat_cnt*epist_cnt

    np.random.seed(1)
    samps = norm.rvs(0, 1, size=vec_size)

    prob = om.Problem(reports=False)
    for tail in ('lower', 'upper'):
        prob.model.add_subsystem(
            tail,
            CVaRComp(
                alpha=alpha, tail=tail, vec_size=vec_size,
                epistemic_cnt=epist_cnt, aleatory_cnt=aleat_cnt
            ),
            promotes_inputs=['*']
        )

    prob.setup()
    prob.set_val('f_sampled', samps)
    prob.run_model()

    # analytic CVaR of a standard normal at tail mass p: -phi(ppf(p))/p
    p = alpha/2
    exact = norm.pdf(norm.ppf(p))/p
    print('lower CVaR:', prob.get_val('lower.f_cvar'), 'analytic:', -exact)
    print('upper CVaR:', prob.get_val('upper.f_cvar'), 'analytic:', exact)
