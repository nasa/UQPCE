import unittest

import numpy as np
import openmdao.api as om

from uqpce.mdao.uqpcegroup import MultiUQPCEGroup

alpha = 0.05
n_fit = 9
aleat_cnt = 20_000


def np_cvar(samps, tail, alpha=alpha):
    """Rockafellar-Uryasev sample CVaR (exact, unsmoothed)."""
    sig = alpha/2 if tail == 'lower' else 1 - alpha/2
    q = np.quantile(samps, sig)
    frac = alpha/2

    if tail == 'upper':
        return q + np.mean(np.maximum(samps - q, 0.0))/frac
    return q - np.mean(np.maximum(q - samps, 0.0))/frac


class TestCVaRIntegration(unittest.TestCase):
    """
    compute_cvar=True in MultiUQPCEGroup: cvar_lower/cvar_upper outputs
    exist, match a numpy reference through the PCE fit + resample chain,
    and carry accurate analytic total derivatives.
    """

    def setUp(self):
        # order-2 PCE in one standard normal variable: psi = [1, x, x^2 - 1]
        rng = np.random.default_rng(3)
        x_fit = rng.standard_normal(n_fit)
        self.var_basis = np.column_stack(
            [np.ones(n_fit), x_fit, x_fit**2 - 1])
        self.norm_sq = np.array([[1.0], [1.0], [2.0]])

        x_re = rng.standard_normal(aleat_cnt)
        self.rvb = np.column_stack([np.ones(aleat_cnt), x_re, x_re**2 - 1])

        self.responses = (
            1.0 + 0.5*x_fit + 0.15*(x_fit**2 - 1)
            + 0.05*rng.standard_normal(n_fit)
        )

        coeffs = np.linalg.lstsq(self.var_basis, self.responses, rcond=None)[0]
        resampled = self.rvb @ coeffs
        self.ref = {t: np_cvar(resampled, t) for t in ('lower', 'upper')}

    def build(self, use_tanh_ci):
        kwargs = dict(
            uncert_list=['f'], var_basis=self.var_basis, significance=alpha,
            resampled_var_basis=self.rvb, tail='both', norm_sq=self.norm_sq,
            aleatory_cnt=aleat_cnt, epistemic_cnt=1,
            use_tanh_ci=use_tanh_ci, compute_cvar=True,
        )
        if use_tanh_ci:
            kwargs.update(
                tanh_omega=[0.005], sample_ref0=[0.0], sample_ref=[1.0])

        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'uq', MultiUQPCEGroup(**kwargs),
            promotes_inputs=['*'], promotes_outputs=['*']
        )
        prob.setup(force_alloc_complex=True)
        prob.set_val('f', self.responses)
        prob.run_model()
        return prob

    def check_totals(self, prob):
        data = prob.check_totals(
            of=['f:cvar_lower', 'f:cvar_upper'], wrt=['f'],
            method='fd', form='central', step=1e-6, out_stream=None
        )
        for key, d in data.items():
            J = d.get('J_fwd', d.get('J_rev'))
            rel = (np.max(np.abs(J - d['J_fd']))
                   / max(np.max(np.abs(d['J_fd'])), 1e-12))
            self.assertLess(
                rel, 1e-5, msg=f'Total derivative check failed: {key}')

    def test_exact(self):
        prob = self.build(use_tanh_ci=False)

        self.assertAlmostEqual(
            prob.get_val('f:cvar_lower')[0], self.ref['lower'], places=5)
        self.assertAlmostEqual(
            prob.get_val('f:cvar_upper')[0], self.ref['upper'], places=5)
        self.check_totals(prob)

    def test_tanh(self):
        prob = self.build(use_tanh_ci=True)
        lo = prob.get_val('f:cvar_lower')[0]
        hi = prob.get_val('f:cvar_upper')[0]

        self.assertTrue(np.isclose(lo, self.ref['lower'], atol=2e-2))
        self.assertTrue(np.isclose(hi, self.ref['upper'], atol=2e-2))
        # smoothing bias must be outward (conservative)
        self.assertLessEqual(lo, self.ref['lower'] + 1e-9)
        self.assertGreaterEqual(hi, self.ref['upper'] - 1e-9)
        self.check_totals(prob)

    def test_default_off(self):
        # compute_cvar defaults False: no cvar outputs are created
        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'uq', MultiUQPCEGroup(
                uncert_list=['f'], var_basis=self.var_basis,
                significance=alpha, resampled_var_basis=self.rvb,
                tail='both', norm_sq=self.norm_sq,
                aleatory_cnt=aleat_cnt, epistemic_cnt=1,
            ),
            promotes_inputs=['*'], promotes_outputs=['*']
        )
        prob.setup()
        prob.set_val('f', self.responses)
        prob.run_model()

        with self.assertRaises(KeyError):
            prob.get_val('f:cvar_lower')


if __name__ == '__main__':
    unittest.main()
