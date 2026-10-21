import unittest

import numpy as np
import openmdao.api as om
from scipy.stats import beta, expon, norm

from uqpce.mdao.cvar.cvarcomp import CVaRComp

aleat_cnt = 500_000
alpha = 0.05


def np_cvar(samps, tail, alpha=alpha):
    """Rockafellar-Uryasev sample CVaR, matching CVaRComp's definition."""
    sig = alpha/2 if tail == 'lower' else 1 - alpha/2
    q = np.quantile(samps, sig)
    frac = alpha/2

    if tail == 'upper':
        return q + np.mean(np.maximum(samps - q, 0.0))/frac
    return q - np.mean(np.maximum(q - samps, 0.0))/frac


def build_problem(samps):
    prob = om.Problem(reports=False)
    for tail in ('lower', 'upper'):
        prob.model.add_subsystem(
            tail,
            CVaRComp(
                alpha=alpha, tail=tail, vec_size=samps.size,
                epistemic_cnt=1, aleatory_cnt=samps.size
            ),
            promotes_inputs=['*']
        )
    prob.setup()
    prob.set_val('f_sampled', samps)
    prob.run_model()
    return prob


class TestCVaRComp(unittest.TestCase):
    def setUp(self):
        thresh = 1e-8
        pcnts = np.linspace(thresh, 1-thresh, num=aleat_cnt)

        a, b = 2.31, 0.627
        self.beta_samples = beta(a, b).ppf(pcnts)
        self.expon_samples = expon.ppf(pcnts)
        self.norm_samples = norm.ppf(pcnts)

    def test_beta(self):
        prob = build_problem(self.beta_samples)

        for tail in ('lower', 'upper'):
            self.assertTrue(
                np.isclose(
                    prob.get_val(f'{tail}.f_cvar')[0],
                    np_cvar(self.beta_samples, tail), atol=1e-6
                ),
                msg=f'Beta distribution failed with {tail} CVaR.'
            )

    def test_expon(self):
        prob = build_problem(self.expon_samples)

        for tail in ('lower', 'upper'):
            self.assertTrue(
                np.isclose(
                    prob.get_val(f'{tail}.f_cvar')[0],
                    np_cvar(self.expon_samples, tail), atol=1e-6
                ),
                msg=f'Exponential distribution failed with {tail} CVaR.'
            )

    def test_norm_analytic(self):
        # standard normal CVaR at tail mass p is -pdf(ppf(p))/p (lower tail)
        prob = build_problem(self.norm_samples)
        p = alpha/2
        exact = norm.pdf(norm.ppf(p))/p

        self.assertTrue(
            np.isclose(prob.get_val('lower.f_cvar')[0], -exact, atol=1e-2),
            msg='Normal distribution failed the analytic lower CVaR.'
        )
        self.assertTrue(
            np.isclose(prob.get_val('upper.f_cvar')[0], exact, atol=1e-2),
            msg='Normal distribution failed the analytic upper CVaR.'
        )

    def test_cvar_beyond_ci(self):
        # the tail mean must lie beyond the quantile it conditions on
        prob = build_problem(self.beta_samples)
        q_lo = np.quantile(self.beta_samples, alpha/2)
        q_hi = np.quantile(self.beta_samples, 1 - alpha/2)

        self.assertLess(prob.get_val('lower.f_cvar')[0], q_lo)
        self.assertGreater(prob.get_val('upper.f_cvar')[0], q_hi)


if __name__ == '__main__':
    unittest.main()
