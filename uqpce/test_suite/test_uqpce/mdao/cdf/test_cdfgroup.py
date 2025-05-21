import unittest

import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt

from scipy.stats import beta, nbinom, expon

from uqpce.mdao.cdf.cdfgroup import CDFGroup
from uqpce.mdao.cdf.cdfresidcomp import tanh_activation

tanh_omega = 1e-6
aleat_cnt = 75_000

class TestCDFGroup(unittest.TestCase):
    def setUp(self):

        self.sig = 0.05
        self.cil = self.sig/2
        self.cih = 1-self.cil
        a, b = 2.31, 0.627
        self.beta_samples = beta.rvs(a, b, size=aleat_cnt)

        self.expon_samples = expon.rvs(size=aleat_cnt)

        n, p = 5, 0.5
        self.nbinom_samples = nbinom.rvs(n, p, size=aleat_cnt)

    def test_beta(self):
        alpha = 0.05

        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'lower',
            CDFGroup(
                alpha=alpha, tail='lower', vec_size=aleat_cnt, tanh_omega=tanh_omega,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
                sample_ref0=self.beta_samples.min(), sample_ref=self.beta_samples.max()
            ), promotes_inputs=['*']
        )
        prob.model.add_subsystem(
            'upper',
            CDFGroup(
                alpha=alpha, tail='upper', vec_size=aleat_cnt, tanh_omega=tanh_omega,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
                sample_ref0=self.beta_samples.min(), sample_ref=self.beta_samples.max()
            ), promotes_inputs=['*']
        )

        prob.setup()
        prob.set_val('lower.cdf.samples', self.beta_samples)
        prob.set_val('upper.cdf.samples', self.beta_samples)
        prob.run_model()

        ci_lower = prob.get_val('lower.f_ci')[0]
        ci_upper = prob.get_val('upper.f_ci')[0]

        self.assertTrue(
            np.isclose(ci_lower, np.quantile(self.beta_samples, self.cil), atol=1e-2),
            msg='Beta distribution failed with lower confidence interval.'
        )
        self.assertTrue(
            np.isclose(ci_upper, np.quantile(self.beta_samples, self.cih), atol=1e-2),
            msg='Beta distribution failed with upper confidence interval.'
        )

        plt.figure()
        x = np.linspace(0,1,aleat_cnt)
        plt.plot(np.sort(self.beta_samples), x, '-o')
        plt.plot([ci_lower, ci_lower], [0,1], 'r--')
        plt.plot([ci_upper, ci_upper], [0,1], 'r--')
        plt.title('Beta Distribution')
        # plt.savefig('beta_cdf')

        x = np.sort(self.beta_samples)

        plt.figure()
        act = tanh_activation(x, omega=1e-12, z=ci_lower, a=1, b=0)
        plt.plot(x, act, '-o')
        plt.plot([ci_lower, ci_lower], [act.min(), act.max()], 'k--')
        plt.title('Beta Distribution')
        # plt.savefig('beta_lower_ci')
        
        plt.figure()
        act = tanh_activation(x, omega=1e-12, z=ci_upper, a=1, b=0)
        plt.plot(x, act, '-o')
        plt.plot([ci_upper, ci_upper], [act.min(), act.max()], 'k--')
        plt.title('Beta Distribution')
        # plt.savefig('beta_upper_ci')


    def test_expon(self):
        alpha = 0.05
        tanh_omega = 1e-8

        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'lower',
            CDFGroup(
                alpha=alpha, tail='lower', vec_size=aleat_cnt, tanh_omega=tanh_omega,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
                sample_ref0=self.expon_samples.min(), sample_ref=self.expon_samples.max()
            ), promotes_inputs=['*']
        )
        prob.model.add_subsystem(
            'upper',
            CDFGroup(
                alpha=alpha, tail='upper', vec_size=aleat_cnt, tanh_omega=tanh_omega,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
                sample_ref0=self.expon_samples.min(), sample_ref=self.expon_samples.max()
            ), promotes_inputs=['*']
        )

        prob.setup()
        prob.set_val('lower.cdf.samples', self.expon_samples)
        prob.set_val('upper.cdf.samples', self.expon_samples)
        prob.run_model()

        ci_lower = prob.get_val('lower.f_ci')[0]
        ci_upper = prob.get_val('upper.f_ci')[0]

        self.assertTrue(
            np.isclose(ci_lower, np.quantile(self.expon_samples, self.cil), atol=1e-2),
            msg='Exponential distribution failed with lower confidence interval.'
        )
        self.assertTrue(
            np.isclose(ci_upper, np.quantile(self.expon_samples, self.cih), atol=1e-2),
            msg='Exponential distribution failed with upper confidence interval.'
        )

        plt.figure()
        x = np.linspace(0,1,aleat_cnt)
        plt.plot(np.sort(self.expon_samples), x, '-o')
        plt.plot([ci_lower, ci_lower], [0,1], 'r--')
        plt.plot([ci_upper, ci_upper], [0,1], 'r--')
        plt.title('Exponential Distribution')
        # plt.savefig('expon_cdf')

        x = np.sort(self.expon_samples)

        plt.figure()
        act = tanh_activation(x, omega=tanh_omega, z=ci_lower, a=1, b=0)
        plt.plot(x, act, '-o')
        plt.plot([ci_lower, ci_lower], [act.min(), act.max()], 'k--')
        plt.title('Exponential Distribution')
        # plt.savefig('expon_lower_ci')
        
        plt.figure()
        act = tanh_activation(x, omega=tanh_omega, z=ci_upper, a=1, b=0)
        plt.plot(x, act, '-o')
        plt.plot([ci_upper, ci_upper], [act.min(), act.max()], 'k--')
        plt.title('Exponential Distribution')
        # plt.savefig('expon_upper_ci')

    def test_nbinom(self):
        alpha = 0.05
        tanh_omega = 1e-10

        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'lower',
            CDFGroup(
                alpha=alpha, tail='lower', vec_size=aleat_cnt, tanh_omega=tanh_omega,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
                sample_ref0=float(self.nbinom_samples.min()), sample_ref=float(self.nbinom_samples.max())
            ), promotes_inputs=['*']
        )
        prob.model.add_subsystem(
            'upper',
            CDFGroup(
                alpha=alpha, tail='upper', vec_size=aleat_cnt, tanh_omega=tanh_omega,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt, 
                sample_ref0=float(self.nbinom_samples.min()), sample_ref=float(self.nbinom_samples.max())
            ), promotes_inputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('lower.cdf.samples', self.nbinom_samples)
        prob.set_val('upper.cdf.samples', self.nbinom_samples)
        prob.run_model()

        ci_lower = prob.get_val('lower.f_ci')[0]
        ci_upper = prob.get_val('upper.f_ci')[0]

        self.assertTrue(
            np.isclose(ci_lower, np.quantile(self.nbinom_samples, self.cil), atol=1e-2),
            msg='Negative Binomial distribution failed with lower confidence interval.'
        )
        self.assertTrue(
            np.isclose(ci_upper, np.quantile(self.nbinom_samples, self.cih), atol=1e-2),
            msg='Negative Binomial distribution failed with upper confidence interval.'
        )

        plt.figure()
        x = np.linspace(0,1,aleat_cnt)
        plt.plot(np.sort(self.nbinom_samples), x, '-o')
        plt.plot([ci_lower, ci_lower], [0,1], 'r--')
        plt.plot([ci_upper, ci_upper], [0,1], 'r--')
        plt.title('Binomial Distribution')
        # plt.savefig('nbinom_cdf')

        x = np.sort(self.nbinom_samples)

        plt.figure()
        act = tanh_activation(x, omega=tanh_omega, z=ci_lower, a=1, b=0)
        plt.plot(x, act, '-o')
        plt.plot([ci_lower, ci_lower], [act.min()-0.1, act.max()+0.1], 'k--')
        plt.title('Binomial Distribution')
        # plt.savefig('nbinom_lower_ci')
        
        plt.figure()
        act = tanh_activation(x, omega=tanh_omega, z=ci_upper, a=1, b=0)
        plt.plot(x, act, '-o')
        plt.plot([ci_upper, ci_upper], [act.min(), act.max()], 'k--')
        plt.title('Binomial Distribution')
        # plt.savefig('nbinom_upper_ci')

    def test_combined(self):
        alpha = 0.05
        tanh_omega = 1e-6

        samps = self.beta_samples - self.expon_samples + self.nbinom_samples

        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'lower',
            CDFGroup(
                alpha=alpha, tail='lower', vec_size=aleat_cnt, tanh_omega=tanh_omega,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
                sample_ref0=samps.min(), sample_ref=samps.max()
            ), promotes_inputs=['*']
        )
        prob.model.add_subsystem(
            'upper',
            CDFGroup(
                alpha=alpha, tail='upper', vec_size=aleat_cnt, tanh_omega=tanh_omega,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
                sample_ref0=samps.min(), sample_ref=samps.max()
            ), promotes_inputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('lower.cdf.samples', samps)
        prob.set_val('upper.cdf.samples', samps)
        prob.run_model()

        ci_lower = prob.get_val('lower.f_ci')[0]
        ci_upper = prob.get_val('upper.f_ci')[0]

        self.assertTrue(
            np.isclose(ci_lower, np.quantile(samps, self.cil), atol=1e-2),
            msg='Combined distribution failed with lower confidence interval.'
        )
        self.assertTrue(
            np.isclose(ci_upper, np.quantile(samps, self.cih), atol=1e-2),
            msg='Combined distribution failed with upper confidence interval.'
        )

        plt.figure()
        x = np.linspace(0,1,aleat_cnt)
        plt.plot(np.sort(samps), x, '-o')
        plt.plot([ci_lower, ci_lower], [0,1], 'r--')
        plt.plot([ci_upper, ci_upper], [0,1], 'r--')
        plt.title('Combined Distributions')
        # plt.savefig('combined_cdf')

        x = np.sort(samps)

        plt.figure()
        act = tanh_activation(x, omega=tanh_omega, z=ci_lower, a=1, b=0)
        plt.plot(x, act, '-o')
        plt.plot([ci_lower, ci_lower], [act.min(), act.max()], 'k--')
        plt.title('Combined Distributions')
        # plt.savefig('combined_lower_ci')
        
        plt.figure()
        act = tanh_activation(x, omega=tanh_omega, z=ci_upper, a=1, b=0)
        plt.plot(x, act, '-o')
        plt.plot([ci_upper, ci_upper], [act.min(), act.max()], 'k--')
        plt.title('Combined Distributions')
        # plt.savefig('combined_upper_ci')


    def test_combined_high_order(self):
        alpha = 0.05
        tanh_omega = 1e-8

        samps = self.beta_samples**6 + self.expon_samples**5 - self.nbinom_samples**3

        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'lower',
            CDFGroup(
                alpha=alpha, tail='lower', vec_size=aleat_cnt, tanh_omega=tanh_omega,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
                sample_ref0=samps.min(), sample_ref=samps.max()
            ), promotes_inputs=['*']
        )
        prob.model.add_subsystem(
            'upper',
            CDFGroup(
                alpha=alpha, tail='upper', vec_size=aleat_cnt, tanh_omega=tanh_omega,
                epistemic_cnt=1, aleatory_cnt=aleat_cnt,
                sample_ref0=samps.min(), sample_ref=samps.max()
            ), promotes_inputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('lower.cdf.samples', samps)
        prob.set_val('upper.cdf.samples', samps)
        prob.run_model()

        ci_lower = prob.get_val('lower.f_ci')[0]
        ci_upper = prob.get_val('upper.f_ci')[0]

        self.assertTrue(
            np.isclose(ci_lower, np.quantile(samps, self.cil), atol=1e-2),
            msg='Combined high order failed with lower confidence interval.'
        )
        self.assertTrue(
            np.isclose(ci_upper, np.quantile(samps, self.cih), atol=1e-2),
            msg='Combined high order failed with upper confidence interval.'
        )

        plt.figure()
        x = np.linspace(0,1,aleat_cnt)
        plt.plot(np.sort(samps), x, '-o')
        plt.plot([ci_lower, ci_lower], [0,1], 'r--')
        plt.plot([ci_upper, ci_upper], [0,1], 'r--')
        plt.title('High-Order Combined Distributions')
        # plt.savefig('high_order_cdf')

        x = np.sort(samps)

        plt.figure()
        act = tanh_activation(x, omega=1e-12, z=ci_lower, a=1, b=0)
        plt.plot(x, act, '-o')
        plt.plot([ci_lower, ci_lower], [act.min(), act.max()], 'k--')
        plt.title('High-Order Combined Distributions')
        # plt.savefig('high_order_lower_ci')
        
        plt.figure()
        act = tanh_activation(x, omega=1e-12, z=ci_upper, a=1, b=0)
        plt.plot(x, act, '-o')
        plt.plot([ci_upper, ci_upper], [act.min(), act.max()], 'k--')
        plt.title('High-Order Combined Distributions')
        # plt.savefig('high_order_upper_ci')

if __name__ == '__main__':

    np.random.seed(33)

    suite = unittest.TestSuite()
    unittest.main()
