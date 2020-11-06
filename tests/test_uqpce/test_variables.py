import unittest

import numpy as np
from scipy.stats import kstest
from sympy import symbols, Matrix, Eq, N, sympify, expand, exp, integrate, gamma
from sympy.parsing.sympy_parser import parse_expr

from PCE_Codes.UQPCE import (
    Variable, UniformVariable, NormalVariable, BetaVariable,
    ExponentialVariable, GammaVariable
)


class TestVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        number = 0
        order = 3

        # region: scipy.stats.invgauss, mu=3
        low_int = 0

        var_dict = {
            'distribution':'(1 /(sqrt(2*pi*x**3))) * exp(-(x-3)**2/(2*x*3**2))',
            'type':'aleatory', 'name':'x0', 'interval_low':low_int,
            'interval_high':'oo'
        }

        self.invgauss_var = Variable(number, order)
        self.invgauss_dist = parse_expr(
            var_dict['distribution'], local_dict={'x':symbols(f'x{number}')}
        )
        self.invgauss_var.initialize(var_dict)
        self.invgauss_var.check_num_string()
        # endregion: scipy.stats.invgauss, mu=3

        samp_size = 50

        self.invgauss_var.generate_samples(samp_size)
        self.invgauss_var.standardize('vals', 'std_vals')
        self.inv_warn = self.invgauss_var.check_distribution()
        self.invgauss_samps = self.invgauss_var.get_resamp_vals(samp_size)

    def test_standardize(self):
        """
        Testing the general Variable standardize and insuring that the values
        remain unchanged.
        """
        self.assertTrue(
            np.isclose(
                self.invgauss_var.vals, self.invgauss_var.std_vals, rtol=0,
                atol=1e-6
            ).all(),
            msg='Variable standardize is not correct'
        )

    def test_check_distribution(self):
        """
        Testing the general Variable check_distribution and insuring that no
        warning is raised.
        """
        no_warn = None

        self.assertEqual(
            self.inv_warn, no_warn,
            msg='Variable check_distribution is not correct'
        )

    def test_generate_samples(self):
        """
        Testing the general Variable generate_samples for several
        distributions and comparing them against the numpy distribution in a
        Kolmogorov-Smirnov test.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        # region: scipy.stats.invgauss, mu=3
        mu = 3

        invgauss_ks_stat, invgauss_p_val = (
            kstest(self.invgauss_var.vals, 'invgauss', args=(mu,))
        )

        self.assertTrue(
            invgauss_p_val > p_val_min,
            msg='Variable generate_samples is not correct'
        )

        self.assertTrue(
            invgauss_ks_stat < ks_stat_max,
            msg='Variable generate_samples is not correct'
        )
        # endregion: scipy.stats.invgauss, mu=3

    def test_get_resamp_vals(self):
        """
        Testing the general Variable get_resamp_vals for several distributions
        and comparing them against the numpy distribution in a
        Kolmogorov-Smirnov test.
        """
        # acceptance-rejection method can lead to poor stats, especially for a
        # large sample number; these thresholds are adjusted accordingly
        p_val_min = 0.05
        ks_stat_max = 0.2

        # region: scipy.stats.invgauss, mu=3
        mu = 3

        invgauss_ks_stat, invgauss_p_val = (
            kstest(self.invgauss_samps, 'invgauss', args=(mu,))
        )

        self.assertTrue(
            invgauss_p_val > p_val_min,
            msg='Variable get_resamp_vals is not correct'
        )

        self.assertTrue(
            invgauss_ks_stat < ks_stat_max,
            msg='Variable get_resamp_vals is not correct'
        )
        # endregion: scipy.stats.invgauss, mu=3

    def test_get_probability_density_func(self):
        """
        Testing the general Variable get_probability_density_func for several
        distributions and ensuring that the standardized distribution is
        approximately equal to 1.
        """
        true_integral = 1

        invgauss_integral = float(integrate(
            self.invgauss_var.distribution,
            (
                self.invgauss_var.x, self.invgauss_var.low_approx,
                self.invgauss_var.high_approx
            )
        ))

        self.invgauss_var.get_probability_density_func()

        self.assertTrue(
            np.isclose(invgauss_integral, true_integral, rtol=0, atol=1e-6),
            msg='Variable get_probability_density_func is not correct'
        )

    def test_check_num_string(self):
        """
        Testing the general Variable check_num_string and ensuring it works
        properly.
        """
        neg_pi = -np.pi
        pos_pi = np.pi

        self.invgauss_var.interval_low = '-pi'
        self.invgauss_var.interval_high = 'pi'
        self.invgauss_var.check_num_string()

        self.assertEqual(
            self.invgauss_var.interval_low, neg_pi,
            msg='Variable check_num_string is not correct'
        )

        self.assertEqual(
            self.invgauss_var.interval_high, pos_pi,
            msg='Variable check_num_string is not correct'
        )

    def test_create_norm_sq(self):
        """
        Testing the general Variable create_norm_sq for several distributions
        and ensuring that the norm squared values are correct.
        
        Verified by hand.
        """
        norm_sq_count = 4

        # region: scipy.stats.invgauss, mu=3
        true_norm_sq = np.array([
            1.0001184, 27.0031965, 14581.7261254, 24673592.9594914,
#             86076162282.29724
        ])
        norm_sq = np.zeros(norm_sq_count)

        for i in range(norm_sq_count):
            norm_sq[i] = self.invgauss_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(true_norm_sq, norm_sq, rtol=0, atol=1e-4).all(),
            msg='Variable create_norm_sq is not correct'
        )
        # endregion: scipy.stats.invgauss, mu=3

    def test_recursive_var_basis(self):
        """
        Testing the general Variable recursive_var_basis for several
        distributions and ensuring that the orthogonal polynomials are correct.
        
        Verified by hand.
        """
        tol = 1e-6
        x0 = symbols('x0')

        # region: scipy.stats.invgauss, mu=3
        true_var_orthopoly_vect = Matrix([
            [1], [x0 - 3], [x0 ** 2 - 33 * x0 + 63],
            [x0 ** 3 - 98.1 * x0 ** 2 + 1671.3 * x0 - 2481.3],
#             [x0 ** 4 - 198.8157 * x0 ** 3 + 9859.4203 * x0 ** 2
#              -114968.4815 * x0 + 143304.1996]
        ])

        basis_size = len(true_var_orthopoly_vect)

        equal = [
            str(
                Eq(
                    N(sympify(expand(true_var_orthopoly_vect[i])), tol),
                    N(
                        sympify(expand(self.invgauss_var.var_orthopoly_vect[i])),
                        tol
                    )
                )
            ) for i in range(basis_size)
        ]

        eval_loc = locals().copy()
        eval_glob = globals().copy()

        evaled = np.array([
            eval(equal[i], eval_loc, eval_glob) for i in range(len(equal))
        ])

        self.assertTrue(
            evaled.all(),
            msg='Variable recursive_var_basis is not correct'
        )
        # endregion: scipy.stats.invgauss, mu=3


class TestUniformVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        self.interval_low = -2
        self.interval_high = 0

        number = 0
        order = 5

        var_dict = {
            'distribution':'uniform', 'interval_low':self.interval_low,
            'interval_high':self.interval_high, 'type':'aleatory', 'name':'x0'
        }

        self.unif_var = UniformVariable(number, order)
        self.unif_var.initialize(var_dict)

        samp_size = 100
        self.unif_var.generate_samples(samp_size)
        self.unif_var.standardize('vals', 'std_vals')
        self.unif_warn = self.unif_var.check_distribution()
        self.unif_samps = self.unif_var.get_resamp_vals(samp_size)

    def test_standardize(self):
        """
        Testing the UniformVariable standardize and insuring that the values
        follow a standardized distribution.
        """
        min_bnd = -1
        max_bnd = 1

        thresh = 1e-6

        mn = np.min(self.unif_var.std_vals)
        mx = np.max(self.unif_var.std_vals)

        self.assertTrue(
            mn >= min_bnd - thresh,
            msg='UniformVariable standardize is not correct'
        )

        self.assertTrue(
            mx <= max_bnd + thresh,
            msg='UniformVariable standardize is not correct'
        )

    def test_check_distribution(self):
        """
        Testing the UniformVariable check_distribution and insuring that no
        warning is raised.
        """
        no_warn = None

        self.assertEqual(
            self.unif_warn, no_warn,
            msg='UniformVariable check_distribution is not correct'
        )

    def test_generate_samples(self):
        """
        Testing the UniformVariable generate_samples and comparing the output
        to the numpy distribution in a Kolmogorov-Smirnov test.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        scale = self.interval_high - self.interval_low
        ks_stat, p_val = kstest(
            self.unif_var.vals, 'uniform', args=(self.interval_low, scale)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='UniformVariable generate_samples is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='UniformVariable generate_samples is not correct'
        )

    def test_get_resamp_vals(self):
        """
        Testing the UniformVariable get_resamp_vals and comparing the output to
        the numpy distribution in a Kolmogorov-Smirnov test.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        scale = 2
        interval_low = -1

        ks_stat, p_val = kstest(
            self.unif_samps, 'uniform', args=(interval_low, scale)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='UniformVariable get_resamp_vals is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='UniformVariable get_resamp_vals is not correct'
        )

    def test_check_num_string(self):
        """
        Testing the UniformVariable check_num_string and ensuring it works
        properly.
        """
        neg_pi = -np.pi
        pos_pi = np.pi

        self.unif_var.interval_low = '-pi'
        self.unif_var.interval_high = 'pi'
        self.unif_var.check_num_string()

        self.assertEqual(
            self.unif_var.interval_low, neg_pi,
            msg='UniformVariable check_num_string is not correct'
        )

        self.assertEqual(
            self.unif_var.interval_high, pos_pi,
            msg='UniformVariable check_num_string is not correct'
        )

    def test_get_norm_sq_val(self):
        """
        Testing the UniformVariable get_norm_sq_val and ensuring that the norm
        squared values are correct.
        """
        norm_sq_count = 6

        true_norm_sq = np.array([1, 1 / 3, 1 / 5, 1 / 7, 1 / 9, 1 / 11])  # , 1 / 13
        norm_sq = np.zeros(norm_sq_count)

        for i in range(norm_sq_count):
            norm_sq[i] = self.unif_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(true_norm_sq, norm_sq, rtol=0, atol=1e-6).all(),
            msg='UniformVariable get_norm_sq_val is not correct'
        )

    def test_generate_orthopoly(self):
        """
        Testing the UniformVariable generate_orthopoly and ensuring that the
        orthogonal polynomials are correct.
        """
        tol = 1e-6
        x0 = symbols('x0')

        true_var_orthopoly_vect = Matrix([
            [1], [x0], [3 * x0 ** 2 / 2 - 1 / 2], [x0 * (5 * x0 ** 2 - 3) / 2],
            [35 * x0 ** 4 / 8 - 15 * x0 ** 2 / 4 + 3 / 8],
            [x0 * (63 * x0 ** 4 - 70 * x0 ** 2 + 15) / 8],
#             [231 * x0 ** 6 / 16 - 315 * x0 ** 4 / 16 + 105 * x0 ** 2 / 16
#              -5 / 16]
        ])

        basis_size = len(true_var_orthopoly_vect)

        equal = [
            str(
                Eq(
                    N(sympify(expand(true_var_orthopoly_vect[i])), tol),
                    N(sympify(expand(self.unif_var.var_orthopoly_vect[i])), tol)
                )
            ) for i in range(basis_size)
        ]

        eval_loc = locals().copy()
        eval_glob = globals().copy()

        evaled = np.array([
            eval(equal[i], eval_loc, eval_glob) for i in range(len(equal))
        ])

        self.assertTrue(
            evaled.all(),
            msg='UniformVariable generate_orthopoly is not correct'
        )


class TestNormalVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        self.mean = 17
        self.stdev = 0.3

        number = 0
        order = 5

        var_dict = {
            'distribution':'normal', 'mean':self.mean, 'stdev':self.stdev,
            'type':'aleatory', 'name':'x0'
        }

        self.norm_var = NormalVariable(number, order)
        self.norm_var.initialize(var_dict)

        samp_size = 100
        self.norm_var.generate_samples(samp_size)
        self.norm_var.standardize('vals', 'std_vals')
        self.norm_warn = self.norm_var.check_distribution()
        self.norm_samps = self.norm_var.get_resamp_vals(samp_size)

    def test_standardize(self):
        """
        Testing the NormalVariable standardize and insuring that the values
        follow a standardized distribution.
        """
        min_bnd = -4.5
        max_bnd = 4.5

        thresh = 1e-6

        mn = np.min(self.norm_var.std_vals)
        mx = np.max(self.norm_var.std_vals)

        self.assertTrue(
            mn >= min_bnd - thresh,
            msg='NormalVariable standardize is not correct'
        )

        self.assertTrue(
            mx <= max_bnd + thresh,
            msg='NormalVariable standardize is not correct'
        )

    def test_check_distribution(self):
        """
        Testing the NormalVariable check_distribution and insuring that no
        warning is raised.
        """
        no_warn = None

        self.assertEqual(
            self.norm_warn, no_warn,
            msg='NormalVariable check_distribution is not correct'
        )

    def test_generate_samples(self):
        """
        Testing the NormalVariable generate_samples and comparing the output
        to the numpy distribution in a Kolmogorov-Smirnov test.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        ks_stat, p_val = kstest(
            self.norm_var.vals, 'norm', args=(self.mean, self.stdev)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='NormalVariable generate_samples is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='NormalVariable generate_samples is not correct'
        )

    def test_get_resamp_vals(self):
        """
        Testing the NormalVariable get_resamp_vals and comparing the output to
        the numpy distribution in a Kolmogorov-Smirnov test.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        mean = 0
        stdev = 1

        ks_stat, p_val = kstest(
            self.norm_samps, 'norm', args=(mean, stdev,)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='NormalVariable get_resamp_vals is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='NormalVariable get_resamp_vals is not correct'
        )

    def test_check_num_string(self):
        """
        Testing the NormalVariable check_num_string and ensuring it works
        properly.
        """
        neg_pi = -np.pi
        pos_pi = np.pi

        self.norm_var.mean = '-pi'
        self.norm_var.stdev = 'pi'
        self.norm_var.check_num_string()

        self.assertEqual(
            self.norm_var.mean, neg_pi,
            msg='NormalVariable check_num_string is not correct'
        )

        self.assertEqual(
            self.norm_var.stdev, pos_pi,
            msg='NormalVariable check_num_string is not correct'
        )

    def test_create_norm_sq(self):
        """
        Testing the NormalVariable create_norm_sq and ensuring that the norm
        squared values are correct.
        """
        norm_sq_count = 6

        true_norm_sq = np.array([1, 1, 2, 6, 24, 120])  # , 720
        norm_sq = np.zeros(norm_sq_count)

        for i in range(norm_sq_count):
            norm_sq[i] = self.norm_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(true_norm_sq, norm_sq, rtol=0, atol=1e-6).all(),
            msg='NormalVariable create_norm_sq is not correct'
        )

    def test_generate_orthopoly(self):
        """
        Testing the NormalVariable generate_orthopoly and ensuring that the
        orthogonal polynomials are correct.
        """
        tol = 1e-6
        x0 = symbols('x0')

        true_var_orthopoly_vect = Matrix([
            [1], [x0], [x0 ** 2 - 1], [x0 * (x0 ** 2 - 3)],
            [x0 ** 4 - 6 * x0 ** 2 + 3], [x0 * (x0 ** 4 - 10 * x0 ** 2 + 15)],
#             [1 * x0 ** 6 - 15 * x0 ** 4 + 45 * x0 ** 2 - 15]
        ])

        basis_size = len(true_var_orthopoly_vect)

        equal = [
            str(
                Eq(
                    N(sympify(expand(true_var_orthopoly_vect[i])), tol),
                    N(sympify(expand(self.norm_var.var_orthopoly_vect[i])), tol)
                )
            ) for i in range(basis_size)
        ]

        eval_loc = locals().copy()
        eval_glob = globals().copy()

        evaled = np.array([
            eval(equal[i], eval_loc, eval_glob) for i in range(len(equal))
        ])

        self.assertTrue(
            evaled.all(),
            msg='NormalVariable generate_orthopoly is not correct'
        )


class TestBetaVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        self.number = 2
        self.alpha = 7
        self.beta = 9
        self.interval_low = -2
        self.interval_high = 3

        number = 0
        order = 5
        self.scale = self.interval_high - self.interval_low

        var_dict = {
            'distribution':'beta', 'alpha':self.alpha, 'beta':self.beta,
            'type':'aleatory', 'interval_low':self.interval_low,
            'interval_high':self.interval_high, 'name':'x0'
        }

        self.beta_var = BetaVariable(number, order)
        self.beta_var.initialize(var_dict)

        samp_size = 100
        self.beta_var.generate_samples(samp_size)
        self.beta_var.standardize('vals', 'std_vals')
        self.beta_warn = self.beta_var.check_distribution()
        self.beta_samps = self.beta_var.get_resamp_vals(samp_size)

    def test_standardize(self):
        """
        Testing the BetaVariable standardize and insuring that the values
        follow a standardized distribution.
        """
        min_bnd = 0
        max_bnd = 1

        thresh = 1e-6

        mn = np.min(self.beta_var.std_vals)
        mx = np.max(self.beta_var.std_vals)

        self.assertTrue(
            mn >= min_bnd - thresh,
            msg='BetaVariable standardize is not correct'
        )

        self.assertTrue(
            mx <= max_bnd + thresh,
            msg='BetaVariable standardize is not correct'
        )

    def test_check_distribution(self):
        """
        Testing the BetaVariable check_distribution and insuring that no
        warning is raised.
        """
        no_warn = None

        self.assertEqual(
            self.beta_warn, no_warn,
            msg='BetaVariable check_distribution is not correct'
        )

    def test_generate_samples(self):
        """
        Testing the BetaVariable generate_samples and comparing the output
        to the numpy distribution in a Kolmogorov-Smirnov test.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        ks_stat, p_val = kstest(# args=(loc, scale)
            self.beta_var.vals, 'beta',
            args=(self.alpha, self.beta, self.interval_low, self.scale)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='BetaVariable generate_samples is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='BetaVariable generate_samples is not correct'
        )

    def test_get_resamp_vals(self):
        """
        Testing the BetaVariable get_resamp_vals and comparing the output to
        the numpy distribution in a Kolmogorov-Smirnov test.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        interval_low = 0
        scale = 1

        ks_stat, p_val = kstest(self.beta_samps, 'beta',
            args=(self.alpha, self.beta, interval_low, scale))

        self.assertTrue(
            p_val > p_val_min,
            msg='BetaVariable get_resamp_vals is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='BetaVariable get_resamp_vals is not correct'
        )

    def test_check_num_string(self):
        """
        Testing the BetaVariable check_num_string and ensuring it works
        properly.
        """
        neg_pi = -np.pi
        pos_pi = np.pi

        self.beta_var.alpha = '-pi'
        self.beta_var.beta = 'pi'
        self.beta_var.check_num_string()

        self.assertEqual(
            self.beta_var.alpha, neg_pi,
            msg='BetaVariable check_num_string is not correct'
        )

        self.assertEqual(
            self.beta_var.beta, pos_pi,
            msg='BetaVariable check_num_string is not correct'
        )

    def test_create_norm_sq(self):
        """
        Testing the BetaVariable create_norm_sq and ensuring that the norm
        squared values are correct.
        """
        norm_sq_count = 6

        true_norm_sq = np.array([
            1, 1.44761000e-02, 3.54114814e-04, 1.12025420e-05, 4.14035738e-07,
            1.69827159e-08
        ])  # , 7.50321175e-10
        norm_sq = np.zeros(norm_sq_count)

        for i in range(norm_sq_count):
            norm_sq[i] = self.beta_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(true_norm_sq, norm_sq, rtol=0, atol=1e-6).all(),
            msg='BetaVariable create_norm_sq is not correct'
        )

    def test_generate_orthopoly(self):
        """
        Testing the BetaVariable generate_orthopoly and ensuring that the
        orthogonal polynomials are correct.
        """
        tol = 1e-6
        x0 = symbols('x0')

        true_var_orthopoly_vect = Matrix([
            [1], [x0 - 0.4375],
            [x0 ** 2 - 0.888888888888889 * x0 + 0.183006535947712],
            [x0 ** 3 - 1.35 * x0 ** 2 + 0.568421052631579 * x0
             -0.0736842105263152],
            [x0 ** 4 - 1.81818181818182 * x0 ** 3 + 1.16883116883117 * x0 ** 2
             -0.311688311688312 * x0 + 0.0287081339712913],
            [x0 ** 5 - 2.29166666666667 * x0 ** 4 + 1.99275362318841 * x0 ** 3
             -0.815217391304351 * x0 ** 2 + 0.155279503105591 * x0
             -0.0108695652173908],
#             [x0 ** 6 - 2.76923076923077 * x0 ** 5 + 3.04615384615386 * x0 ** 4
#              -1.6923076923077 * x0 ** 3 + 0.496655518394654 * x0 ** 2
#              -0.072240802675587 * x0 + 0.00401337792642093]
        ])

        basis_size = len(true_var_orthopoly_vect)

        equal = [
            str(
                Eq(
                    N(sympify(expand(true_var_orthopoly_vect[i])), tol),
                    N(sympify(expand(self.beta_var.var_orthopoly_vect[i])), tol)
                )
            ) for i in range(basis_size)
        ]

        eval_loc = locals().copy()
        eval_glob = globals().copy()

        evaled = np.array([
            eval(equal[i], eval_loc, eval_glob) for i in range(len(equal))
        ])

        self.assertTrue(
            evaled.all(),
            msg='BetaVariable generate_orthopoly is not correct'
        )


class TestExponentialVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        self.lambd = 7
        self.interval_low = 12

        number = 0
        order = 5

        var_dict = {
            'distribution':'exponential', 'lambda':self.lambd,
            'type':'aleatory', 'interval_low':self.interval_low, 'name':'x0'
        }

        self.exp_var = ExponentialVariable(number, order)
        self.exp_var.initialize(var_dict)

        samp_size = 100
        self.exp_var.generate_samples(samp_size)
        self.exp_var.standardize('vals', 'std_vals')
        self.exp_warn = self.exp_var.check_distribution()
        self.exp_samps = self.exp_var.get_resamp_vals(samp_size)

    def test_standardize(self):
        """
        Testing the ExponentialVariable standardize and insuring that the values
        follow a standardized distribution.
        """
        min_bnd = 0
        max_bnd = 1

        thresh = 1e-6

        mn = np.min(self.exp_var.std_vals)
        mx = np.max(self.exp_var.std_vals)

        self.assertTrue(
            mn >= min_bnd - thresh,
            msg='ExponentialVariable standardize is not correct'
        )

        self.assertTrue(
            mx <= max_bnd + thresh,
            msg='ExponentialVariable standardize is not correct'
        )

    def test_check_distribution(self):
        """
        Testing the ExponentialVariable check_distribution and insuring that no
        warning is raised.
        """
        no_warn = None

        self.assertEqual(
            self.exp_warn, no_warn,
            msg='ExponentialVariable check_distribution is not correct'
        )

    def test_generate_samples(self):
        """
        Testing the ExponentialVariable generate_samples and comparing the
        output to the numpy distribution in a Kolmogorov-Smirnov test.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        ks_stat, p_val = kstest(# args=(loc, scale)
            self.exp_var.vals, 'expon', args=(self.interval_low, 1 / self.lambd)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='ExponentialVariable generate_samples is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='ExponentialVariable generate_samples is not correct'
        )

    def test_get_resamp_vals(self):
        """
        Testing the ExponentialVariable get_resamp_vals and comparing the
        output to the numpy distribution in a Kolmogorov-Smirnov test.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        interval_low = 0

        ks_stat, p_val = kstest(
            self.exp_samps, 'expon', args=(interval_low, 1 / self.lambd)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='ExponentialVariable get_resamp_vals is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='ExponentialVariable get_resamp_vals is not correct'
        )

    def test_check_num_string(self):
        """
        Testing the ExponentialVariable check_num_string and ensuring it works
        properly.
        """
        pos_pi = np.pi

        setattr(self.exp_var, 'lambda', 'pi')
        self.exp_var.check_num_string()

        self.assertEqual(
            getattr(self.exp_var, 'lambda'), pos_pi,
            msg='ExponentialVariable check_num_string is not correct'
        )

    def test_create_norm_sq(self):
        """
        Testing the ExponentialVariable create_norm_sq and ensuring that the
        norm squared values are correct.
        """
        norm_sq_count = 6

        true_norm_sq = np.array([
            1, 2.04082000e-02, 1.66597251e-03, 3.05994951e-04, 9.99167187e-05,
            5.09779177e-05
        ])  # , 3.74531640e-05
        norm_sq = np.zeros(norm_sq_count)

        for i in range(norm_sq_count):
            norm_sq[i] = self.exp_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(true_norm_sq, norm_sq, rtol=0, atol=1e-6).all(),
            msg='ExponentialVariable create_norm_sq is not correct'
        )

    def test_generate_orthopoly(self):
        """
        Testing the ExponentialVariable generate_orthopoly and ensuring that the
        orthogonal polynomials are correct.
        """
        tol = 1e-6
        x0 = symbols('x0')

        true_var_orthopoly_vect = Matrix([
            [1], [x0 - 0.142857142857143],
            [x0 ** 2 - 0.571428571428572 * x0 + 0.0408163265306123],
            [x0 ** 3 - 1.28571428571428 * x0 ** 2 + 0.367346938775509 * x0
             -0.0174927113702623],
            [x0 ** 4 - 2.28571428571422 * x0 ** 3 + 1.46938775510196 * x0 ** 2
             -0.279883381924174 * x0 + 0.00999583506872018],
            [x0 ** 5 - 3.57142857142935 * x0 ** 4 + 4.08163265306289 * x0 ** 3
             -1.74927113702723 * x0 ** 2 + 0.249895876718207 * x0
             -0.00713988219194941],
#             [x0 ** 6 - 5.14285714283134 * x0 ** 5 + 9.18367346929817 * x0 ** 4
#              -6.99708454800531 * x0 ** 3 + 2.24906289042068 * x0 ** 2
#              -0.257035758904169 * x0 + 0.00611989902150362]
        ])

        basis_size = len(true_var_orthopoly_vect)

        equal = [
            str(
                Eq(
                    N(sympify(expand(true_var_orthopoly_vect[i])), tol),
                    N(sympify(expand(self.exp_var.var_orthopoly_vect[i])), tol)
                )
            ) for i in range(basis_size)
        ]

        eval_loc = locals().copy()
        eval_glob = globals().copy()

        evaled = np.array([
            eval(equal[i], eval_loc, eval_glob) for i in range(len(equal))]
        )

        self.assertTrue(
            evaled.all(),
            msg='ExponentialVariable generate_orthopoly is not correct'
        )


class TestGammaVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        self.alpha = 3
        self.theta = 6
        self.interval_low = -5

        number = 0
        order = 5

        var_dict = {
            'distribution':'gamma', 'alpha':self.alpha, 'theta':self.theta,
            'type':'aleatory', 'interval_low':self.interval_low, 'name':'x0'
        }

        self.gamma_var = GammaVariable(number, order)
        self.gamma_var.initialize(var_dict)

        samp_size = 100
        self.gamma_var.generate_samples(samp_size)
        self.gamma_var.standardize('vals', 'std_vals')
        self.gamma_warn = self.gamma_var.check_distribution()
        self.gamma_samps = self.gamma_var.get_resamp_vals(samp_size)

    def test_standardize(self):
        """
        Testing the GammaVariable standardize and insuring that the values
        follow a standardized distribution.
        """
        min_bnd = 0
        max_bnd = 15

        thresh = 1e-6

        mn = np.min(self.gamma_var.std_vals)
        mx = np.max(self.gamma_var.std_vals)

        self.assertTrue(
            mn >= min_bnd - thresh,
            msg='GammaVariable standardize is not correct'
        )

        self.assertTrue(
            mx <= max_bnd + thresh,
            msg='GammaVariable standardize is not correct'
        )

    def test_check_distribution(self):
        """
        Testing the GammaVariable check_distribution and insuring that no
        warning is raised.
        """
        no_warn = None

        self.assertEqual(
            self.gamma_warn, no_warn,
            msg='GammaVariable check_distribution is not correct'
        )

    def test_generate_samples(self):
        """
        Testing the GammaVariable generate_samples and comparing the output
        to the numpy distribution in a Kolmogorov-Smirnov test.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        ks_stat, p_val = kstest(
            self.gamma_var.vals, 'gamma',
            args=(self.alpha, self.interval_low, self.theta)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='GammaVariable generate_samples is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='GammaVariable generate_samples is not correct'
        )

    def test_get_resamp_vals(self):
        """
        Testing the GammaVariable get_resamp_vals and comparing the output to
        the numpy distribution in a Kolmogorov-Smirnov test.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        ks_stat, p_val = kstest(
            self.gamma_samps, 'gamma', args=(self.alpha,)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='GammaVariable get_resamp_vals is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='GammaVariable get_resamp_vals is not correct'
        )

    def test_check_num_string(self):
        """
        Testing the GammaVariable check_num_string and ensuring it works
        properly.
        """
        neg_pi = -np.pi
        pos_pi = np.pi

        self.gamma_var.alpha = '-pi'
        self.gamma_var.theta = 'pi'

        self.gamma_var.check_num_string()

        self.assertEqual(
            self.gamma_var.alpha, neg_pi,
            msg='GammaVariable check_num_string is not correct'
        )

        self.assertEqual(
            self.gamma_var.theta, pos_pi,
            msg='GammaVariable check_num_string is not correct'
        )

    def test_create_norm_sq(self):
        """
        Testing the GammaVariable create_norm_sq and ensuring that the norm
        squared values are correct.
        """
        norm_sq_count = 6

        true_norm_sq = np.array([1, 3, 24, 360, 8640, 302400])  # , 14515200
        norm_sq = np.zeros(norm_sq_count)

        for i in range(norm_sq_count):
            norm_sq[i] = self.gamma_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(true_norm_sq, norm_sq, rtol=0, atol=1e-6).all(),
            msg='GammaVariable create_norm_sq is not correct'
        )

    def test_recursive_var_basis(self):
        """
        Testing the GammaVariable generate_orthopoly and ensuring that the
        orthogonal polynomials are correct.
        """
        tol = 1e-6
        x0 = symbols('x0')

        true_var_orthopoly_vect = Matrix([
            [1], [x0 - 3], [x0 ** 2 - 8 * x0 + 12],
            [x0 ** 3 - 15 * x0 ** 2 + 60 * x0 - 60],
            [x0 ** 4 - 24 * x0 ** 3 + 180 * x0 ** 2 - 480 * x0 + 360],
            [x0 ** 5 - 35 * x0 ** 4 + 420 * x0 ** 3 - 2100 * x0 ** 2
             +4200 * x0 - 2520],
#             [x0 ** 6 - 48 * x0 ** 5 + 840 * x0 ** 4 - 6720 * x0 ** 3
#              +25200 * x0 ** 2 - 40320 * x0 + 20160]
        ])

        basis_size = len(true_var_orthopoly_vect)

        equal = [
            str(
                Eq(
                    N(sympify(expand(true_var_orthopoly_vect[i])), tol),
                    N(sympify(expand(self.gamma_var.var_orthopoly_vect[i])), tol)
                )
            ) for i in range(basis_size)
        ]

        eval_loc = locals().copy()
        eval_glob = globals().copy()

        evaled = np.array([
            eval(equal[i], eval_loc, eval_glob) for i in range(len(equal))
        ])

        self.assertTrue(
            evaled.all(),
            msg='GammaVariable recursive_var_basis is not correct'
        )
