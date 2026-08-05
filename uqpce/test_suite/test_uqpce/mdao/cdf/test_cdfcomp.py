import unittest

import numpy as np
import openmdao.api as om
from scipy.stats import beta, expon, nbinom

from uqpce.mdao.cdf.cdfcomp import CDFComp

aleat_cnt = 500_000

class TestCDFComp(unittest.TestCase):
    def setUp(self):

        self.sig = 0.05
        self.cil = self.sig/2
        self.cih = 1-self.cil
        a, b = 2.31, 0.627
        thresh = 1e-8
        pcnts = np.linspace(thresh, 1-thresh, num=aleat_cnt)
        self.beta_samples = beta(a, b).ppf(pcnts)

        self.expon_samples = expon.ppf(pcnts)

        n, p = 5, 0.5
        self.nbinom_samples = nbinom(n, p).ppf(pcnts)

    def test_beta(self):
        alpha = 0.05

        prob = om.Problem(reports=False)

        prob.model.add_subsystem(
            'lower',
            CDFComp(
                alpha=alpha, tail='lower', vec_size=aleat_cnt,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
            ), promotes_inputs=['*']
        )
        prob.model.add_subsystem(
            'upper',
            CDFComp(
                alpha=alpha, tail='upper', vec_size=aleat_cnt,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
            ), promotes_inputs=['*']
        )

        prob.setup()
        prob.set_val('f_sampled', self.beta_samples)
        prob.set_val('f_sampled', self.beta_samples)
        prob.run_model()

        ci_lower = prob.get_val('lower.f_ci')[0]
        ci_upper = prob.get_val('upper.f_ci')[0]

        self.assertTrue(
            np.isclose(ci_lower, np.quantile(self.beta_samples, self.cil), atol=1e-6),
            msg='Beta distribution failed with lower confidence interval.'
        )
        self.assertTrue(
            np.isclose(ci_upper, np.quantile(self.beta_samples, self.cih), atol=1e-6),
            msg='Beta distribution failed with upper confidence interval.'
        )


    def test_expon(self):
        alpha = 0.05

        prob = om.Problem(reports=False)

        prob.model.add_subsystem(
            'lower',
            CDFComp(
                alpha=alpha, tail='lower', vec_size=aleat_cnt,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
            ), promotes_inputs=['*']
        )
        prob.model.add_subsystem(
            'upper',
            CDFComp(
                alpha=alpha, tail='upper', vec_size=aleat_cnt,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
            ), promotes_inputs=['*']
        )

        prob.setup()
        prob.set_val('f_sampled', self.expon_samples)
        prob.set_val('f_sampled', self.expon_samples)
        prob.run_model()

        ci_lower = prob.get_val('lower.f_ci')[0]
        ci_upper = prob.get_val('upper.f_ci')[0]

        self.assertTrue(
            np.isclose(ci_lower, np.quantile(self.expon_samples, self.cil), atol=1e-6),
            msg='Exponential distribution failed with lower confidence interval.'
        )
        self.assertTrue(
            np.isclose(ci_upper, np.quantile(self.expon_samples, self.cih), atol=1e-6),
            msg='Exponential distribution failed with upper confidence interval.'
        )


    def test_nbinom(self):
        alpha = 0.05

        prob = om.Problem(reports=False)

        prob.model.add_subsystem(
            'lower',
            CDFComp(
                alpha=alpha, tail='lower', vec_size=aleat_cnt,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
            ), promotes_inputs=['*']
        )
        prob.model.add_subsystem(
            'upper',
            CDFComp(
                alpha=alpha, tail='upper', vec_size=aleat_cnt,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
            ), promotes_inputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('f_sampled', self.nbinom_samples)
        prob.set_val('f_sampled', self.nbinom_samples)
        prob.run_model()

        ci_lower = prob.get_val('lower.f_ci')[0]
        ci_upper = prob.get_val('upper.f_ci')[0]

        self.assertTrue(
            np.isclose(ci_lower, np.quantile(self.nbinom_samples, self.cil), atol=1e-6),
            msg='Negative Binomial distribution failed with lower confidence interval.'
        )
        self.assertTrue(
            np.isclose(ci_upper, np.quantile(self.nbinom_samples, self.cih), atol=1e-6),
            msg='Negative Binomial distribution failed with upper confidence interval.'
        )

    def test_combined(self):
        alpha = 0.05

        samps = self.beta_samples - self.expon_samples + self.nbinom_samples

        prob = om.Problem(reports=False)

        prob.model.add_subsystem(
            'lower',
            CDFComp(
                alpha=alpha, tail='lower', vec_size=aleat_cnt,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
            ), promotes_inputs=['*']
        )
        prob.model.add_subsystem(
            'upper',
            CDFComp(
                alpha=alpha, tail='upper', vec_size=aleat_cnt,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
            ), promotes_inputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('f_sampled', samps)
        prob.run_model()

        ci_lower = prob.get_val('lower.f_ci')[0]
        ci_upper = prob.get_val('upper.f_ci')[0]

        self.assertTrue(
            np.isclose(ci_lower, np.quantile(samps, self.cil), atol=1e-6),
            msg='Combined distribution failed with lower confidence interval.'
        )
        self.assertTrue(
            np.isclose(ci_upper, np.quantile(samps, self.cih), atol=1e-6),
            msg='Combined distribution failed with upper confidence interval.'
        )

    def test_combined_high_order(self):
        alpha = 0.05

        samps = self.beta_samples**6 + self.expon_samples**5 - self.nbinom_samples**3

        prob = om.Problem(reports=False)

        prob.model.add_subsystem(
            'lower',
            CDFComp(
                alpha=alpha, tail='lower', vec_size=aleat_cnt,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
            ), promotes_inputs=['*']
        )
        prob.model.add_subsystem(
            'upper',
            CDFComp(
                alpha=alpha, tail='upper', vec_size=aleat_cnt,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
            ), promotes_inputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('f_sampled', samps)
        prob.set_val('f_sampled', samps)
        prob.run_model()

        ci_lower = prob.get_val('lower.f_ci')[0]
        ci_upper = prob.get_val('upper.f_ci')[0]

        self.assertTrue(
            np.isclose(ci_lower, np.quantile(samps, self.cil), atol=1e-6),
            msg='Combined high order failed with lower confidence interval.'
        )
        self.assertTrue(
            np.isclose(ci_upper, np.quantile(samps, self.cih), atol=1e-6),
            msg='Combined high order failed with upper confidence interval.'
        )


if __name__ == '__main__':

    np.random.seed(33)

    suite = unittest.TestSuite()
    unittest.main()