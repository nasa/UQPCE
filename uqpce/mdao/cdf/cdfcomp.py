import jax.numpy as jnp
import openmdao.api as om


class CDFComp(om.JaxExplicitComponent):
    """
    Component class to calculate the differentiable confidence interval.
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
        self.add_output('f_ci', shape=(1,))

        self._sig = (1-alpha/2) if self.options['tail'] == 'upper' else alpha/2

    def get_self_statics(self):
        return (
            self.options['alpha'], self.options['epistemic_cnt'],
            self.options['aleatory_cnt']
        )

    def compute_primal(self, f_sampled):
        aleat_cnt = self.options['aleatory_cnt']
        epist_cnt = self.options['epistemic_cnt']
        tail = self.options['tail']

        samps = jnp.reshape(f_sampled, (-1, aleat_cnt))

        if aleat_cnt != 1:
            cis = jnp.quantile(samps, self._sig, axis=1)

            # Mixed uncertainty
            if epist_cnt != 1:
                if tail == 'upper':
                    f_ci = jnp.atleast_1d(jnp.max(cis))
                else:
                    f_ci = jnp.atleast_1d(jnp.min(cis))
            else:  # Pure aleatory
                f_ci = jnp.atleast_1d(cis)
        else:  # Pure epistemic
            if tail == 'upper':
                f_ci = jnp.atleast_1d(jnp.max(samps))
            else:
                f_ci = jnp.atleast_1d(jnp.min(samps))

        return f_ci


if __name__ == '__main__':
    import numpy as np
    from scipy.stats import binom, norm

    lower = -2
    upper = 2

    alpha = 0.05
    aleat_cnt = 1
    epist_cnt = 20
    vec_size = aleat_cnt * epist_cnt

    np.random.seed(1)
    samps = (
        binom.rvs(n=5, p=0.3, size=vec_size)
        + norm.rvs(0, 0.05, size=vec_size)
    )

    prob = om.Problem()
    prob.model.add_subsystem(
        'comp',
        CDFComp(
            alpha=alpha, tail='upper', vec_size=vec_size,
            epistemic_cnt=epist_cnt, aleatory_cnt=aleat_cnt
        ),
        promotes_inputs=['*'], promotes_outputs=['*']
    )

    prob.setup(force_alloc_complex=True)
    prob.set_val('f_sampled', samps)
    prob.run_model()

    with np.printoptions(threshold=np.inf):
        partials = prob.check_partials(compact_print=False, method='fd', form='central')

    f_sampled = np.reshape(samps, (-1, aleat_cnt))
    print(np.quantile(f_sampled, [0.025, 1-0.025], axis=1))
    print(np.min(f_sampled), np.max(f_sampled))
    print(prob.get_val('f_ci'))