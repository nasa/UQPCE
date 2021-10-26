import unittest
from multiprocessing import Manager

from scipy.integrate import quad
from scipy.stats import kstest, beta, expon, gamma
from sympy import (
    symbols, Eq, N, sympify, expand, Matrix, sqrt, erf, sqrt, erfinv
)
from sympy.utilities.lambdify import lambdify
import numpy as np

from PCE_Codes.variables.continuous import (
    ContinuousVariable, UniformVariable, NormalVariable, BetaVariable,
    ExponentialVariable, GammaVariable
)


class TestGeneralVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        order = 4
        samp_size = 5000

        dist = '(1 /(sqrt(2*pi*x**3))) * exp(-(x-3)**2/(2*x*3**2))'
        interval_low = 0
        interval_high = 'oo'

        self.invgauss_var = ContinuousVariable(
            dist, interval_low, interval_high, order=order
        )

        self.invgauss_var.vals = self.invgauss_var.generate_samples(samp_size)
        self.invgauss_samps = self.invgauss_var.get_resamp_vals(samp_size)
        self.invgauss_var.standardize('vals', 'std_vals')
        self.inv_warn = self.invgauss_var.check_distribution()

        dist = '1/(sqrt(2*pi)) * exp(-1/2 * x**2)'
        interval_low = '-oo'
        interval_high = 'oo'
        self.norm_var = ContinuousVariable(
            dist, interval_low, interval_high, order=order
        )

    def test_standardize(self):
        """
        This method does nothing for the general Variable class since it is
        required that the input equation to be standardized.
        """
        self.assertTrue(
            np.isclose(
                self.invgauss_var.vals, self.invgauss_var.std_vals, rtol=0,
                atol=1e-6
            ).all(),
            msg='Variable standardize is not correct'
        )

    def test_standardize_points(self):
        """
        This method does nothing for the general Variable class since it is
        required that the input equation to be standardized.
        """
        invgauss_stand_samps = self.invgauss_var.standardize_points(
            self.invgauss_samps
        )

        self.assertTrue(
            np.isclose(
                self.invgauss_samps, invgauss_stand_samps, rtol=0, atol=1e-6
            ).all(),
            msg='Variable standardize_points is not correct'
        )

    def test_unstandardize_points(self):
        """
        This method does nothing for the general Variable class since it is
        required that the input equation to be standardized.
        """
        invgauss_stand_samps = self.invgauss_var.unstandardize_points(
            self.invgauss_samps
        )

        self.assertTrue(
            np.isclose(
                self.invgauss_samps, invgauss_stand_samps, rtol=0, atol=1e-6
            ).all(),
            msg='Variable unstandardize_points is not correct'
        )

    def test_check_distribution(self):
        """
        This method does nothing for the general variable class since there is
        limited information about the general Variable.
        """
        no_warn = None

        self.assertEqual(
            self.inv_warn, no_warn,
            msg='Variable check_distribution is not correct'
        )

    def test_generate_samples(self):
        """
        Testing the general Variable sample generation for several
        distributions and comparing them against the numpy distribution in a
        Kolmogorov-Smirnov test.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

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

    def test_get_probability_density_func(self):
        """
        Testing the general Variable get_probability_density_func for several
        distributions and ensuring that the standardized distribution is
        approximately equal to 1.
        """
        true_integral = 1
        tol = 1e-6

        self.invgauss_var.get_probability_density_func()
        func = lambdify(
            self.invgauss_var.x, self.invgauss_var.distribution,
            ('numpy', 'sympy')
        )

        invgauss_integral = quad(
            func, self.invgauss_var.low_approx, self.invgauss_var.high_approx
        )[0]

        self.assertTrue(
            np.isclose(invgauss_integral, true_integral, rtol=0, atol=tol),
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
        """
        norm_sq_count = 4

        true_norm_sq = np.array([
            1.0008192, 27.0221173, 14591.9433628, 24690881.4447931
        ])
        norm_sq = np.zeros(norm_sq_count)

        for i in range(norm_sq_count):
            norm_sq[i] = self.invgauss_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(true_norm_sq, norm_sq, rtol=0, atol=1e-4).all(),
            msg='Variable create_norm_sq is not correct'
        )

    def test_recursive_var_basis(self):
        """
        Testing the general Variable recursive_var_basis for several
        distributions and ensuring that the orthogonal polynomials are correct.

        Verified by hand.
        """
        tol = 1e-6
        x0 = symbols('x0')

        true_var_orthopoly_vect = Matrix(np.array([
            [1], [x0 - 3], [x0 ** 2 - 33 * x0 + 63],
            [x0 ** 3 - 98.1 * x0 ** 2 + 1671.3 * x0 - 2481.3],
            [x0 ** 4 - 198.8157 * x0 ** 3 + 9859.4203 * x0 ** 2
             -114968.4815 * x0 + 143304.1996]
        ]))

        basis_size = len(true_var_orthopoly_vect)
        orthopoly = Matrix(self.invgauss_var.var_orthopoly_vect)

        equal = [
            str(
                Eq(
                    N(sympify(expand(true_var_orthopoly_vect[i])), tol),
                    N(sympify(expand(orthopoly[i])), tol)
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

    def test__norm_sq(self):
        """
        Testing the general Variable _norm_sq to ensure that the three 
        different calculation attempts all give the same correct answer.
        """
        i = 2
        norm_sq_val = 14591.9433628

        low = self.invgauss_var.interval_low
        high = self.invgauss_var.interval_high

        func = self.invgauss_var.distribution

        proc_dict = {}

        region = 0
        self.invgauss_var._norm_sq(low, high, func, i, region, proc_dict)
        norm_0 = proc_dict['out']

        region = 1
        self.invgauss_var._norm_sq(low, high, func, i, region, proc_dict)
        norm_1 = proc_dict['out']

        region = 2
        self.invgauss_var._norm_sq(low, high, func, i, region, proc_dict)
        norm_2 = proc_dict['out']

        self.assertEqual(
            norm_sq_val, norm_0, msg='Variable _norm_sq is not correct'
        )

        self.assertEqual(
            norm_sq_val, norm_1, msg='Variable _norm_sq is not correct'
        )

        self.assertEqual(
            norm_sq_val, norm_2, msg='Variable _norm_sq is not correct'
        )

    def test__invert(self):
        """
        Testing the general Variable _invert to ensure that the correct 
        equation is found when _invert is called.
        """
        manager = Manager()
        proc_dict = manager.dict()
        y = symbols('y')

        inv_func_act = sqrt(2) * erfinv(2.0 * y - 1.0)

        self.norm_var._calc_cdf(proc_dict)
        self.norm_var.cum_dens_func = proc_dict['cum_dens_func']
        self.norm_var._invert(proc_dict)

        self.assertEqual(
            proc_dict['inverse_func'][0], N(inv_func_act),
            msg='Variable _invert is not correct'
        )

    def test__calc_cdf(self):
        """
        Testing the general Variable _calc_cdf to ensure that the correct 
        cumulative density function is found when _calc_cdf is called.
        """
        manager = Manager()
        proc_dict = manager.dict()
        x0 = symbols('x0')

        cdf_act = 0.5 * erf(sqrt(2) * x0 / 2) + 0.5
        self.norm_var._calc_cdf(proc_dict)

        self.assertEqual(
            cdf_act, proc_dict['cum_dens_func'],
            msg='Variable _calc_cdf is not correct'
        )

    def test_get_mean(self):
        """
        Tests that the mean for the distributions from integration are
        consistent with the true mean value.
        """
        invgauss_true_mean = 3
        tol = 1e-5

        mean = self.invgauss_var.get_mean()
        self.assertTrue(
            np.abs(mean - invgauss_true_mean) <= tol,
            msg='Variable get_mean is not correct'
        )


class TestUniformVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        self.interval_low = -2
        self.interval_high = 0
        order = 5

        samp_size = 5000

        self.unif_var = UniformVariable(
            self.interval_low, self.interval_high, order=order
        )

        self.unif_var.vals = np.array([
            -1.13195492, -0.25059365, -0.79204424, -0.31193718, -1.48149601,
            -1.75890592, -0.66485926, -1.85500246, -1.51296043, -0.58363648,
            -1.97514899, -0.05487085, -1.05134119, -1.39803386, -0.40168332,
            -0.86066013, -0.15990408, -1.2046748, -1.67397003, -0.90349732
        ])
        self.unif_var.standardize('vals', 'std_vals')
        self.unif_warn = self.unif_var.check_distribution()
        self.unif_samps = self.unif_var.get_resamp_vals(samp_size)

    def test_standardize(self):
        """
        Testing the UniformVariable standardize and insuring that the values
        follow a standardized distribution.
        """
        stand_points_act = np.array([
            -0.13195492, 0.74940634, 0.20795576, 0.68806282, -0.48149602,
            -0.75890592, 0.33514074, -0.85500246, -0.51296042, 0.41636352,
            -0.975149  , 0.94512914, -0.0513412 , -0.39803386, 0.59831668,
            0.13933988, 0.84009592, -0.2046748 , -0.67397002, 0.09650268
        ])

        self.assertTrue(
            np.isclose(stand_points_act, self.unif_var.std_vals).all(),
            msg='UniformVariable standardize is not correct'
        )

    def test_standardize_points(self):
        """
        Testing the UniformVariable standardize_points and insuring that the 
        values follow a standardized distribution.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        points = np.array([
            -1.31593915, -1.1659186 , -1.79209697, -1.2637438 , -0.18374311,
            -1.86460178, -0.01976728, -0.24042643, -1.58342055, -0.4693803 ,
            -0.72684234, -1.02895983, -0.62267842, -0.38633504, -0.85276121,
            -1.64440856, -1.96685233, -0.99682922, -0.57587552, -1.47047622
        ])
        stand_points_act = np.array([
            -0.31593915, -0.1659186 , -0.79209697, -0.2637438 , 0.81625689,
            -0.86460178, 0.98023272, 0.75957357, -0.58342055, 0.5306197 ,
            0.27315766, -0.02895983, 0.37732158, 0.61366496, 0.14723879,
            -0.64440856, -0.96685233, 0.00317078, 0.42412448, -0.47047622
        ])
        stand_points_calc = self.unif_var.standardize_points(points)

        low = -1
        high = 1

        scale = high - low
        ks_stat, p_val = kstest(
            stand_points_calc, 'uniform', args=(low, scale)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='UniformVariable standardize_points is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='UniformVariable standardize_points is not correct'
        )

        self.assertTrue(
            np.isclose(stand_points_act, stand_points_calc).all(),
            msg='UniformVariable standardize_points is not correct'
        )

    def test_unstandardize_points(self):
        """
        Testing the UniformVariable unstandardize_points and insuring that the 
        values follow an unstandardized distribution.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        points = np.array([
            -0.31593915, -0.1659186, -0.79209697, -0.2637438 , 0.81625689,
            -0.86460178, 0.98023272, 0.75957357, -0.58342055, 0.5306197,
            0.27315766, -0.02895983, 0.37732158, 0.61366496, 0.14723879,
            -0.64440856, -0.96685233, 0.00317078, 0.42412448, -0.47047622
        ])
        unstand_points_act = np.array([
            -1.31593915, -1.1659186, -1.79209697, -1.2637438, -0.18374311,
            -1.86460178, -0.01976728, -0.24042643, -1.58342055, -0.4693803,
            -0.72684234, -1.02895983, -0.62267842, -0.38633504, -0.85276121,
            -1.64440856, -1.96685233, -0.99682922, -0.57587552, -1.47047622
        ])
        unstand_points_calc = self.unif_var.unstandardize_points(points)

        scale = self.interval_high - self.interval_low
        ks_stat, p_val = kstest(
            unstand_points_calc, 'uniform', args=(self.interval_low, scale)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='UniformVariable unstandardize_points is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='UniformVariable unstandardize_points is not correct'
        )

        self.assertTrue(
            np.isclose(unstand_points_act, unstand_points_calc).all(),
            msg='UniformVariable unstandardize_points is not correct'
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

        true_var_orthopoly_vect = Matrix(np.array([
            [1], [x0], [3 * x0 ** 2 / 2 - 1 / 2], [x0 * (5 * x0 ** 2 - 3) / 2],
            [35 * x0 ** 4 / 8 - 15 * x0 ** 2 / 4 + 3 / 8],
            [x0 * (63 * x0 ** 4 - 70 * x0 ** 2 + 15) / 8]
#             [231 * x0 ** 6 / 16 - 315 * x0 ** 4 / 16 + 105 * x0 ** 2 / 16
#              -5 / 16]
        ]))

        basis_size = len(true_var_orthopoly_vect)
        orthopoly = Matrix(self.unif_var.var_orthopoly_vect)

        equal = [
            str(
                Eq(
                    N(sympify(expand(true_var_orthopoly_vect[i])), tol),
                    N(sympify(expand(orthopoly[i])), tol)
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

    def test_get_mean(self):
        """
        Tests that the mean for the distributions is consistent with the true 
        mean value.
        """
        cnt = 100000
        tol = 1e-3
        act_mean = -1

        calc_mean = np.mean(self.unif_var.generate_samples(cnt))

        self.assertEqual(
            act_mean, self.unif_var.get_mean(),
            msg='UniformVariable get_mean is not correct'
        )

        self.assertTrue(
            np.abs(act_mean - calc_mean) < tol,
            msg='UniformVariable get_mean is not correct'
        )


class TestNormalVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        self.mean = 17
        self.stdev = 0.3

        order = 5
        samp_size = 5000

        self.norm_var = NormalVariable(self.mean, self.stdev, order=order)

        self.norm_var.vals = np.array([
            16.73012736, 16.99049448, 16.38575072, 17.23439198, 17.30919136,
            17.00823061, 17.1489481, 17.85874036, 17.46873612, 17.3169995,
            17.19112564, 16.86582408, 16.63673353, 16.52670581, 17.06367273,
            16.90473806, 16.92937104, 16.77880454, 17.09521439, 16.80732891
        ])

        self.norm_var.standardize('vals', 'std_vals')
        self.norm_warn = self.norm_var.check_distribution()
        self.norm_samps = self.norm_var.get_resamp_vals(samp_size)

    def test_standardize(self):
        """
        Testing the NormalVariable standardize and insuring that the values
        follow a standardized distribution.
        """
        stand_points_act = np.array([
            -0.89957547, -0.03168507, -2.0474976 , 0.7813066, 1.03063787,
            0.02743537, 0.49649367, 2.86246787, 1.56245373, 1.056665,
            0.63708547, -0.44725307, -1.21088823, -1.5776473, 0.21224243,
            -0.3175398, -0.23542987, -0.7373182, 0.3173813, -0.64223697
        ])

        self.assertTrue(
            np.isclose(stand_points_act, self.norm_var.std_vals).all(),
            msg='NormalVariable standardize is not correct'
        )

    def test_standardize_points(self):
        """
        Testing the NormalVariable standardize_points and insuring that the 
        values follow a standardized distribution.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        points = np.array([
            16.71289138, 17.2857896, 16.38722038, 17.16280096, 16.82660026,
            17.13116335, 17.39723893, 17.01401991, 16.95621621, 17.07574243,
            16.6086059, 16.88789858, 16.85157194, 16.6253459, 17.32610548,
            16.98834336, 16.76382667, 17.10685334, 17.62909248, 17.23868834
        ])
        stand_points_act = np.array([
            -0.95702873, 0.952632, -2.04259873, 0.54266987, -0.57799913,
            0.43721117, 1.32412977, 0.04673303, -0.14594597, 0.25247477,
            -1.304647, -0.3736714, -0.4947602, -1.248847, 1.08701827,
            -0.03885547, -0.78724443, 0.3561778, 2.09697493, 0.7956278
        ])
        stand_points_calc = self.norm_var.standardize_points(points)

        ks_stat, p_val = kstest(stand_points_calc, 'norm')

        self.assertTrue(
            p_val > p_val_min,
            msg='NormalVariable standardize_points is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='NormalVariable standardize_points is not correct'
        )

        self.assertTrue(
            np.isclose(stand_points_act, stand_points_calc).all(),
            msg='NormalVariable standardize_points is not correct'
        )

    def test_unstandardize_points(self):
        """
        Testing the NormalVariable unstandardize_points and insuring that the 
        values follow an unstandardized distribution.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        points = np.array([
            -0.95702873, 0.952632, -2.04259873, 0.54266987, -0.57799913,
            0.43721117, 1.32412977, 0.04673303, -0.14594597, 0.25247477,
            -1.304647, -0.3736714, -0.4947602, -1.248847, 1.08701827,
            -0.03885547, -0.78724443, 0.3561778, 2.09697493, 0.7956278
        ])
        unstand_points_act = np.array([
            16.71289138, 17.2857896, 16.38722038, 17.16280096, 16.82660026,
            17.13116335, 17.39723893, 17.01401991, 16.95621621, 17.07574243,
            16.6086059, 16.88789858, 16.85157194, 16.6253459, 17.32610548,
            16.98834336, 16.76382667, 17.10685334, 17.62909248, 17.23868834
        ])

        unstand_points_calc = self.norm_var.unstandardize_points(points)

        ks_stat, p_val = kstest(
            unstand_points_calc, 'norm', args=(self.mean, self.stdev)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='NormalVariable unstandardize_points is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='NormalVariable unstandardize_points is not correct'
        )

        self.assertTrue(
            np.isclose(unstand_points_act, unstand_points_calc).all(),
            msg='NormalVariable unstandardize_points is not correct'
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

    def test_get_norm_sq_val(self):
        """
        Testing the NormalVariable get_norm_sq_val and ensuring that the norm
        squared values are correct.
        """
        norm_sq_count = 6

        true_norm_sq = np.array([1, 1, 2, 6, 24, 120])
        norm_sq = np.zeros(norm_sq_count)

        for i in range(norm_sq_count):
            norm_sq[i] = self.norm_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(true_norm_sq, norm_sq, rtol=0, atol=1e-6).all(),
            msg='NormalVariable get_norm_sq_val is not correct'
        )

    def test_generate_orthopoly(self):
        """
        Testing the NormalVariable generate_orthopoly and ensuring that the
        orthogonal polynomials are correct.
        """
        tol = 1e-6
        x0 = symbols('x0')

        true_var_orthopoly_vect = Matrix(np.array([
            [1], [x0], [x0 ** 2 - 1], [x0 * (x0 ** 2 - 3)],
            [x0 ** 4 - 6 * x0 ** 2 + 3], [x0 * (x0 ** 4 - 10 * x0 ** 2 + 15)]
#             [1 * x0 ** 6 - 15 * x0 ** 4 + 45 * x0 ** 2 - 15]
        ]))

        basis_size = len(true_var_orthopoly_vect)
        orthopoly = Matrix(self.norm_var.var_orthopoly_vect)

        equal = [
            str(
                Eq(
                    N(sympify(expand(true_var_orthopoly_vect[i])), tol),
                    N(sympify(expand(orthopoly[i])), tol)
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

    def test_get_mean(self):
        """
        Tests that the mean for the distributions is consistent with the true 
        mean value.
        """
        cnt = 100000
        tol = 1e-3

        act_mean = self.mean
        calc_mean = np.mean(self.norm_var.generate_samples(cnt))

        self.assertEqual(
            self.mean, self.norm_var.get_mean(),
            msg='NormalVariable get_mean is not correct'
        )

        self.assertTrue(
            np.abs(act_mean - calc_mean) < tol,
            msg='NormalVariable get_mean is not correct'
        )


class TestBetaVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        self.number = 2
        self.alpha = 7
        self.beta = 9
        self.interval_low = -2
        self.interval_high = 3
        self.scale = self.interval_high - self.interval_low

        order = 5
        samp_size = 5000

        self.beta_var = BetaVariable(
            self.alpha, self.beta, interval_low=self.interval_low,
            interval_high=self.interval_high, order=order
        )

        self.beta_var.vals = np.array([
            0.07004045, 0.90031624, 0.34088884, 0.81437103, -0.22362193,
            -0.52869249, 0.44862329, -0.68384585, -0.2531082 , 0.52165205,
            -1.06665014, 1.36275001, 0.13370142, -0.14883056, 0.70582787,
            0.28501447, 1.05813268, 0.01203821, -0.42110615, 0.25065737
        ])
        self.beta_var.standardize('vals', 'std_vals')
        self.beta_warn = self.beta_var.check_distribution()
        self.beta_samps = self.beta_var.get_resamp_vals(samp_size)

    def test_standardize(self):
        """
        Testing the BetaVariable standardize and insuring that the values
        follow a standardized distribution.
        """
        stand_points_act = np.array([
            0.41400809, 0.58006325, 0.46817777, 0.56287421, 0.35527561,
            0.2942615, 0.48972466, 0.26323083, 0.34937836, 0.50433041,
            0.18666997, 0.67255, 0.42674028, 0.37023389, 0.54116557,
            0.45700289, 0.61162654, 0.40240764, 0.31577877, 0.45013147
        ])

        self.assertTrue(
            np.isclose(stand_points_act, self.beta_var.std_vals).all(),
            msg='BetaVariable standardize is not correct'
        )

    def test_standardize_points(self):
        """
        Testing the BetaVariable standardize_points and insuring that the values
        follow a standardized distribution.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        points = np.array([
            -0.07891957, 0.04304876, -0.57690413, -0.03582172, 1.01166483,
            -0.70271311, 1.59090727, 0.91584277, -0.32251385, 0.63299745,
            0.39529933, 0.15133571, 0.48603777, 0.72328528, 0.29138855,
            -0.38747927, -1.01584712, 0.17666514, 0.5288584 , -0.21347985
        ])

        stand_points_act = np.array([
            0.38421609, 0.40860975, 0.28461917, 0.39283566, 0.60233297,
            0.25945738, 0.71818145, 0.58316855, 0.33549723, 0.52659949,
            0.47905987, 0.43026714, 0.49720755, 0.54465706, 0.45827771,
            0.32250415, 0.19683058, 0.43533303, 0.50577168, 0.35730403
        ])
        stand_points_calc = self.beta_var.standardize_points(points)

        low = 0
        high = 1

        scale = high - low
        ks_stat, p_val = kstest(
            stand_points_calc, 'beta', args=(self.alpha, self.beta, low, scale)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='BetaVariable standardize_points is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='BetaVariable standardize_points is not correct'
        )

        self.assertTrue(
            np.isclose(stand_points_act, stand_points_calc).all(),
            msg='BetaVariable standardize_points is not correct'
        )

    def test_unstandardize_points(self):
        """
        Testing the NormalVariable unstandardize_points and insuring that the 
        values follow an unstandardized distribution.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        points = np.array([
            0.38421609, 0.40860975, 0.28461917, 0.39283566, 0.60233297,
            0.25945738, 0.71818145, 0.58316855, 0.33549723, 0.52659949,
            0.47905987, 0.43026714, 0.49720755, 0.54465706, 0.45827771,
            0.32250415, 0.19683058, 0.43533303, 0.50577168, 0.35730403
        ])

        unstand_points_act = np.array([
            -0.07891957, 0.04304876, -0.57690413, -0.03582172, 1.01166483,
            -0.70271311, 1.59090727, 0.91584277, -0.32251385, 0.63299745,
            0.39529933, 0.15133571, 0.48603777, 0.72328528, 0.29138855,
            -0.38747927, -1.01584712, 0.17666514, 0.5288584 , -0.21347985
        ])

        unstand_points_calc = self.beta_var.unstandardize_points(points)

        scale = self.interval_high - self.interval_low
        ks_stat, p_val = kstest(
            unstand_points_calc, 'beta',
            args=(self.alpha, self.beta, self.interval_low, scale)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='BetaVariable unstandardize_points is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='BetaVariable unstandardize_points is not correct'
        )

        self.assertTrue(
            np.isclose(unstand_points_act, unstand_points_calc).all(),
            msg='BetaVariable unstandardize_points is not correct'
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

        ks_stat, p_val = kstest(
            self.beta_samps, 'beta',
            args=(self.alpha, self.beta, interval_low, scale)
        )

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

    def test_get_norm_sq_val(self):
        """
        Testing the BetaVariable get_norm_sq_val and ensuring that the norm
        squared values are correct.
        """
        norm_sq_count = 6

        true_norm_sq = np.array([
            1, 1.44761000e-02, 3.54114814e-04, 1.12025420e-05, 4.14035738e-07,
            1.69827159e-08
        ])
        norm_sq = np.zeros(norm_sq_count)

        for i in range(norm_sq_count):
            norm_sq[i] = self.beta_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(true_norm_sq, norm_sq, rtol=0, atol=1e-6).all(),
            msg='BetaVariable get_norm_sq_val is not correct'
        )

    def test_generate_orthopoly(self):
        """
        Testing the BetaVariable generate_orthopoly and ensuring that the
        orthogonal polynomials are correct.
        """
        tol = 1e-6
        x0 = symbols('x0')

        true_var_orthopoly_vect = Matrix(np.array([
            [1], [x0 - 0.4375],
            [x0 ** 2 - 0.888888888888889 * x0 + 0.183006535947712],
            [x0 ** 3 - 1.35 * x0 ** 2 + 0.568421052631579 * x0
             -0.0736842105263152],
            [x0 ** 4 - 1.81818181818182 * x0 ** 3 + 1.16883116883117 * x0 ** 2
             -0.311688311688312 * x0 + 0.0287081339712913],
            [x0 ** 5 - 2.29166666666667 * x0 ** 4 + 1.99275362318841 * x0 ** 3
             -0.815217391304351 * x0 ** 2 + 0.155279503105591 * x0
             -0.0108695652173908]
#             [x0 ** 6 - 2.76923076923077 * x0 ** 5 + 3.04615384615386 * x0 ** 4
#              -1.6923076923077 * x0 ** 3 + 0.496655518394654 * x0 ** 2
#              -0.072240802675587 * x0 + 0.00401337792642093]
        ]))

        basis_size = len(true_var_orthopoly_vect)
        orthopoly = Matrix(self.beta_var.var_orthopoly_vect)

        equal = [
            str(
                Eq(
                    N(sympify(expand(true_var_orthopoly_vect[i])), tol),
                    N(sympify(expand(orthopoly[i])), tol)
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

    def test_get_mean(self):
        """
        Tests that the mean for the distributions is consistent with the true 
        mean value.
        """
        cnt = 100000
        tol = 1e-3

        scale = self.interval_high - self.interval_low
        act_mean = beta.mean(self.alpha, self.beta, self.interval_low, scale)
        calc_mean = np.mean(self.beta_var.generate_samples(cnt))

        self.assertEqual(
            act_mean, self.beta_var.get_mean(),
            msg='BetaVariable get_mean is not correct'
        )

        self.assertTrue(
            np.abs(act_mean - calc_mean) < tol,
            msg='BetaVariable get_mean is not correct'
        )


class TestExponentialVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        self.lambd = 7
        self.interval_low = 12

        order = 5
        samp_size = 5000

        self.exp_var = ExponentialVariable(
            self.lambd, self.interval_low, order=order
        )

        self.exp_var.vals = np.array([
            12.01075159, 12.22931978, 12.15733244, 12.08131443, 12.13232646,
            12.01835074, 12.1759463, 12.0254213, 12.26544295, 12.05115433,
            12.11351847, 12.00178619, 12.03986841, 12.09186864, 12.36090405,
            12.51370288, 12.29672425, 12.04287068, 12.12045754, 12.07241965
        ])
        self.exp_var.standardize('vals', 'std_vals')
        self.exp_warn = self.exp_var.check_distribution()
        self.exp_samps = self.exp_var.get_resamp_vals(samp_size)

    def test_standardize(self):
        """
        Testing the ExponentialVariable standardize and insuring that the values
        follow a standardized distribution.
        """
        stand_points_act = np.array([
            0.01075159, 0.22931978, 0.15733244, 0.08131443, 0.13232646,
            0.01835074, 0.1759463, 0.0254213, 0.26544295, 0.05115433,
            0.11351847, 0.00178619, 0.03986841, 0.09186864, 0.36090405,
            0.51370288, 0.29672425, 0.04287068, 0.12045754, 0.07241965
        ])

        self.assertTrue(
            np.isclose(stand_points_act, self.exp_var.std_vals).all(),
            msg='ExponentialVariable standardize is not correct'
        )

    def test_standardize_points(self):
        """
        Testing the ExponentialVariable standardize_points and insuring that
        the values follow a standardized distribution.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        points = np.array([
            12.21175876, 12.69861934, 12.04882843, 12.12648997, 12.0604722 ,
            12.26653948, 12.00103251, 12.32263972, 12.37010718, 12.0849974 ,
            12.1712394 , 12.17878711, 12.14270891, 12.04101203, 12.02235371,
            12.07086804, 12.08757787, 12.02982745, 12.01094778, 12.10092573
        ])

        stand_points_act = np.array([
            0.21175876, 0.69861934, 0.04882843, 0.12648997, 0.0604722,
            0.26653948, 0.00103251, 0.32263972, 0.37010718, 0.0849974,
            0.1712394, 0.17878711, 0.14270891, 0.04101203, 0.02235371,
            0.07086804, 0.08757787, 0.02982745, 0.01094778, 0.10092573
        ])

        stand_points_calc = self.exp_var.standardize_points(points)

        low = 0
        ks_stat, p_val = kstest(
            stand_points_calc, 'expon', args=(low, 1 / self.lambd)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='ExponentialVariable standardize_points is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='ExponentialVariable standardize_points is not correct'
        )

        self.assertTrue(
            np.isclose(stand_points_act, stand_points_calc).all(),
            msg='ExponentialVariable standardize_points is not correct'
        )

    def test_unstandardize_points(self):
        """
        Testing the ExponentialVariable unstandardize_points and insuring that 
        the values follow an unstandardized distribution.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        points = np.array([
            0.21175876, 0.69861934, 0.04882843, 0.12648997, 0.0604722,
            0.26653948, 0.00103251, 0.32263972, 0.37010718, 0.0849974,
            0.1712394, 0.17878711, 0.14270891, 0.04101203, 0.02235371,
            0.07086804, 0.08757787, 0.02982745, 0.01094778, 0.10092573
        ])

        unstand_points_act = np.array([
            12.21175876, 12.69861934, 12.04882843, 12.12648997, 12.0604722 ,
            12.26653948, 12.00103251, 12.32263972, 12.37010718, 12.0849974 ,
            12.1712394 , 12.17878711, 12.14270891, 12.04101203, 12.02235371,
            12.07086804, 12.08757787, 12.02982745, 12.01094778, 12.10092573
        ])
        unstand_points_calc = self.exp_var.unstandardize_points(points)

        ks_stat, p_val = kstest(
            unstand_points_calc, 'expon',
            args=(self.interval_low, 1 / self.lambd)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='ExponentialVariable unstandardize_points is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='ExponentialVariable unstandardize_points is not correct'
        )

        self.assertTrue(
            np.isclose(unstand_points_act, unstand_points_calc).all(),
            msg='ExponentialVariable unstandardize_points is not correct'
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

        ks_stat, p_val = kstest(
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

        true_var_orthopoly_vect = Matrix(np.array([
            [1], [x0 - 0.142857142857143],
            [x0 ** 2 - 0.571428571428572 * x0 + 0.0408163265306123],
            [x0 ** 3 - 1.28571428571428 * x0 ** 2 + 0.367346938775509 * x0
             -0.0174927113702623],
            [x0 ** 4 - 2.28571428571422 * x0 ** 3 + 1.46938775510196 * x0 ** 2
             -0.279883381924174 * x0 + 0.00999583506872018],
            [x0 ** 5 - 3.57142857142935 * x0 ** 4 + 4.08163265306289 * x0 ** 3
             -1.74927113702723 * x0 ** 2 + 0.249895876718207 * x0
             -0.00713988219194941]
#             [x0 ** 6 - 5.14285714283134 * x0 ** 5 + 9.18367346929817 * x0 ** 4
#              -6.99708454800531 * x0 ** 3 + 2.24906289042068 * x0 ** 2
#              -0.257035758904169 * x0 + 0.00611989902150362]
        ]))

        basis_size = len(true_var_orthopoly_vect)
        orthopoly = Matrix(self.exp_var.var_orthopoly_vect)

        equal = [
            str(
                Eq(
                    N(sympify(expand(true_var_orthopoly_vect[i])), tol),
                    N(sympify(expand(orthopoly[i])), tol)
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

    def test_get_norm_sq_val(self):
        """
        Testing the ExponentialVariable get_norm_sq_val and ensuring that the 
        norm squared values are correct.
        """
        norm_sq_count = 6

        true_norm_sq = np.array([
            1, 2.04082000e-02, 1.66597251e-03, 3.05994951e-04, 9.99167187e-05,
            5.09779177e-05
        ])
        norm_sq = np.zeros(norm_sq_count)

        for i in range(norm_sq_count):
            norm_sq[i] = self.exp_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(true_norm_sq, norm_sq, rtol=0, atol=1e-6).all(),
            msg='ExponentialVariable get_norm_sq_val is not correct'
        )

    def test_get_mean(self):
        """
        Tests that the mean for the distributions is consistent with the true 
        mean value.
        """
        cnt = 100000
        tol = 1e-3
        act_mean = expon.mean(loc=self.interval_low, scale=1 / self.lambd)
        calc_mean = np.mean(self.exp_var.generate_samples(cnt))

        self.assertEqual(
            act_mean, self.exp_var.get_mean(),
            msg='ExponentialVariable get_mean is not correct'
        )

        self.assertTrue(
            np.abs(act_mean - calc_mean) < tol,
            msg='ExponentialVariable get_mean is not correct'
        )


class TestGammaVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        self.alpha = 3
        self.theta = 6
        self.interval_low = -5
        order = 5

        samp_size = 5000

        self.gamma_var = GammaVariable(
            self.alpha, self.theta, interval_low=self.interval_low, order=order
        )

        self.gamma_var.vals = np.array([
            9.48579197, 24.95417016, 13.74310854, 22.98317338, 5.57258319,
            2.20094449, 15.62918513, 0.73975869, 5.21688342, 16.97555267,
            -2.16180763, 37.61290404, 10.42750512, 6.50448665, 20.63443988,
            12.80978757, 28.85558648, 8.65770534, 3.31314678, 12.25054577
        ])

        self.gamma_var.standardize('vals', 'std_vals')
        self.gamma_warn = self.gamma_var.check_distribution()
        self.gamma_samps = self.gamma_var.get_resamp_vals(samp_size)

    def test_standardize(self):
        """
        Testing the GammaVariable standardize and insuring that the values
        follow a standardized distribution.
        """
        stand_points_act = np.array([
            2.41429866, 4.99236169, 3.12385142, 4.66386223, 1.7620972,
            1.20015742, 3.43819752, 0.95662645, 1.7028139 , 3.66259211,
            0.47303206, 7.10215067, 2.57125085, 1.91741444, 4.27240665,
            2.96829793, 5.64259775, 2.27628422, 1.38552446, 2.87509096
        ])

        self.assertTrue(
            np.isclose(stand_points_act, self.gamma_var.std_vals).all(),
            msg='GammaVariable standardize is not correct'
        )

    def test_standardize_points(self):
        """
        Testing the GammaVariable standardize_points and insuring that the 
        values follow a standardized distribution.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        points = np.array([
            7.41484008, 9.09693542, 1.7290065, 7.99537761, 27.66668563,
            0.57340321, 45.52458532, 25.32133284, 4.40527912, 19.14058597,
            14.68111726, 10.69457038, 16.31190679, 21.00209078, 12.91475696,
            3.67771932, -1.83390652, 11.08296807, 17.11149989, 5.69644812
        ])

        stand_points_act = np.array([
            2.06914001, 2.34948924, 1.12150108, 2.16589627, 5.44444761,
            0.92890054, 8.42076422, 5.05355547, 1.56754652, 4.023431,
            3.28018621, 2.61576173, 3.55198446, 4.3336818, 2.98579283,
            1.44628655, 0.52768225, 2.68049468, 3.68524998, 1.78274135
        ])

        stand_points_calc = self.gamma_var.standardize_points(points)

        low = 0
        norm_theta = 1
        ks_stat, p_val = kstest(
            stand_points_calc, 'gamma', args=(self.alpha, low, norm_theta)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='GammaVariable standardize_points is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='GammaVariable standardize_points is not correct'
        )

        self.assertTrue(
            np.isclose(stand_points_act, stand_points_calc).all(),
            msg='GammaVariable standardize_points is not correct'
        )

    def test_unstandardize_points(self):
        """
        Testing the GammaVariable unstandardize_points and insuring that 
        the values follow an unstandardized distribution.
        """
        p_val_min = 0.05
        ks_stat_max = 0.2

        points = np.array([
            2.06914001, 2.34948924, 1.12150108, 2.16589627, 5.44444761,
            0.92890054, 8.42076422, 5.05355547, 1.56754652, 4.023431,
            3.28018621, 2.61576173, 3.55198446, 4.3336818, 2.98579283,
            1.44628655, 0.52768225, 2.68049468, 3.68524998, 1.78274135
        ])

        unstand_points_act = np.array([
            7.41484008, 9.09693542, 1.7290065, 7.99537761, 27.66668563,
            0.57340321, 45.52458532, 25.32133284, 4.40527912, 19.14058597,
            14.68111726, 10.69457038, 16.31190679, 21.00209078, 12.91475696,
            3.67771932, -1.83390652, 11.08296807, 17.11149989, 5.69644812
        ])

        unstand_points_calc = self.gamma_var.unstandardize_points(points)

        ks_stat, p_val = kstest(
            unstand_points_calc, 'gamma',
            args=(self.alpha, self.interval_low, self.theta)
        )

        self.assertTrue(
            p_val > p_val_min,
            msg='GammaVariable unstandardize_points is not correct'
        )

        self.assertTrue(
            ks_stat < ks_stat_max,
            msg='GammaVariable unstandardize_points is not correct'
        )

        self.assertTrue(
            np.isclose(unstand_points_act, unstand_points_calc).all(),
            msg='GammaVariable unstandardize_points is not correct'
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

        true_var_orthopoly_vect = Matrix(np.array([
            [1], [x0 - 3], [x0 ** 2 - 8 * x0 + 12],
            [x0 ** 3 - 15 * x0 ** 2 + 60 * x0 - 60],
            [x0 ** 4 - 24 * x0 ** 3 + 180 * x0 ** 2 - 480 * x0 + 360],
            [x0 ** 5 - 35 * x0 ** 4 + 420 * x0 ** 3 - 2100 * x0 ** 2
             +4200 * x0 - 2520]
#             [x0 ** 6 - 48 * x0 ** 5 + 840 * x0 ** 4 - 6720 * x0 ** 3
#              +25200 * x0 ** 2 - 40320 * x0 + 20160]
        ]))

        basis_size = len(true_var_orthopoly_vect)
        orthopoly = Matrix(self.gamma_var.var_orthopoly_vect)

        equal = [
            str(
                Eq(
                    N(sympify(expand(true_var_orthopoly_vect[i])), tol),
                    N(sympify(expand(orthopoly[i])), tol)
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

    def test_get_norm_sq_val(self):
        """
        Testing the GammaVariable get_norm_sq_val and ensuring that the norm
        squared values are correct.
        """
        norm_sq_count = 6

        true_norm_sq = np.array([1, 3, 24, 360, 8640, 302400])
        norm_sq = np.zeros(norm_sq_count)

        for i in range(norm_sq_count):
            norm_sq[i] = self.gamma_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(true_norm_sq, norm_sq, rtol=0, atol=1e-6).all(),
            msg='GammaVariable get_norm_sq_val is not correct'
        )

    def test_get_mean(self):
        """
        Tests that the mean for the distributions is consistent with the true 
        mean value.
        """
        cnt = 100000
        tol = 1e-3

        act_mean = gamma.mean(self.alpha, self.interval_low, self.theta)
        calc_mean = np.mean(self.gamma_var.generate_samples(cnt))

        self.assertEqual(
            act_mean, self.gamma_var.get_mean(),
            msg='GammaVariable get_mean is not correct'
        )

        self.assertTrue(
            np.abs(act_mean - calc_mean) < tol,
            msg='GammaVariable get_mean is not correct'
        )

