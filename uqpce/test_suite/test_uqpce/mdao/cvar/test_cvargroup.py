import unittest

import numpy as np
import openmdao.api as om
from scipy.stats import norm

from uqpce.mdao.cvar.cvargroup import CVaRGroup

alpha = 0.05


def np_cvar(samps, tail, alpha=alpha):
    """Rockafellar-Uryasev sample CVaR (exact, unsmoothed)."""
    sig = alpha/2 if tail == 'lower' else 1 - alpha/2
    q = np.quantile(samps, sig)
    frac = alpha/2

    if tail == 'upper':
        return q + np.mean(np.maximum(samps - q, 0.0))/frac
    return q - np.mean(np.maximum(q - samps, 0.0))/frac


def build_problem(samps, tanh_omega):
    prob = om.Problem(reports=False)
    for tail in ('lower', 'upper'):
        prob.model.add_subsystem(
            tail,
            CVaRGroup(
                alpha=alpha, tanh_omega=tanh_omega, tail=tail,
                vec_size=samps.size, epistemic_cnt=1, aleatory_cnt=samps.size,
                sample_ref0=float(samps.mean()), sample_ref=float(samps.std())
            ),
            promotes_inputs=['*']
        )
    prob.setup()
    prob.set_val('f_sampled', samps)
    prob.run_model()
    return prob


class TestCVaRGroup(unittest.TestCase):
    def setUp(self):
        thresh = 1e-8
        pcnts = np.linspace(thresh, 1-thresh, num=500_000)
        self.norm_samples = norm.ppf(pcnts)

    def test_norm(self):
        prob = build_problem(self.norm_samples, tanh_omega=1e-3)
        p = alpha/2
        exact = norm.pdf(norm.ppf(p))/p

        self.assertTrue(
            np.isclose(prob.get_val('lower.f_cvar')[0], -exact, atol=1e-2),
            msg='Normal distribution failed the smoothed lower CVaR.'
        )
        self.assertTrue(
            np.isclose(prob.get_val('upper.f_cvar')[0], exact, atol=1e-2),
            msg='Normal distribution failed the smoothed upper CVaR.'
        )

    def test_conservative_bias(self):
        # softplus smoothing must bias outward: lower CVaR under the exact
        # value, upper CVaR over it
        prob = build_problem(self.norm_samples, tanh_omega=5e-3)

        self.assertLessEqual(
            prob.get_val('lower.f_cvar')[0],
            np_cvar(self.norm_samples, 'lower') + 1e-9
        )
        self.assertGreaterEqual(
            prob.get_val('upper.f_cvar')[0],
            np_cvar(self.norm_samples, 'upper') - 1e-9
        )

    def test_totals(self):
        # analytic totals through the balance solve vs finite difference.
        # tanh_omega must exceed the local sample spacing (~0.06 at the
        # 2.5% tail of 300 normal samples) or the CDF kernel saturates
        # between samples and the balance Jacobian goes singular.
        n = 300
        thresh = 1e-4
        samps = norm.ppf(np.linspace(thresh, 1-thresh, num=n))

        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'lower',
            CVaRGroup(
                alpha=alpha, tanh_omega=0.1, tail='lower', vec_size=n,
                epistemic_cnt=1, aleatory_cnt=n,
                sample_ref0=0.0, sample_ref=1.0
            ),
            promotes_inputs=['*']
        )
        prob.setup()
        prob.set_val('f_sampled', samps)
        prob.run_model()

        data = prob.check_totals(
            of=['lower.f_cvar'], wrt=['f_sampled'],
            method='fd', form='central', step=1e-7, out_stream=None
        )
        for key, d in data.items():
            J = d.get('J_fwd', d.get('J_rev'))
            rel = (np.max(np.abs(J - d['J_fd']))
                   / max(np.max(np.abs(d['J_fd'])), 1e-12))
            self.assertLess(rel, 1e-5, msg=f'Total derivative check failed: {key}')


if __name__ == '__main__':
    unittest.main()
