import unittest

from scipy.special import factorial
from scipy.stats import chisquare, boltzmann
import numpy as np
from sympy import Matrix

from PCE_Codes.variables.discrete import (
    DiscreteVariable, PoissonVariable, NegativeBinomialVariable,
    HypergeometricVariable, UniformVariable as DiscreteUniformVariable
)


class TestDiscreteVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        self.order = 5
        samp_size = 5000
        self.N = 15

        # region: binomial distribution
        self.p = 0.35
        self.binom_interval_low = 0
        self.binom_interval_high = self.N
        self.binom_distribution = (
            f'({self.N}/(x! * ({self.N}-x)!)) * {self.p}**x '
            f'* (1-{self.p})**({self.N}-x)'
        )

        self.binom_var = DiscreteVariable(
            self.binom_distribution, self.binom_interval_low,
            self.binom_interval_high, order=self.order
        )

        self.binom_var.vals = self.binom_var.generate_samples(samp_size)
        self.binom_var.standardize('vals', 'std_vals')
        self.binom_warn = self.binom_var.check_distribution()
        # endregion: binomial distribution

        # region: Boltzmann distribution
        self.lambd = 0.4
        self.boltz_interval_low = 0
        self.boltz_interval_high = self.N - 1
        self.boltz_distribution = (
            f'((1 - exp(-{self.lambd})) * exp(-{self.lambd} * x)) '
            f'/ (1 - exp(-{self.lambd} * {self.N}))'
        )

        self.boltz_var = DiscreteVariable(
            self.boltz_distribution, self.boltz_interval_low,
            self.boltz_interval_high, order=self.order
        )

        self.boltz_var.vals = self.boltz_var.generate_samples(samp_size)
        self.boltz_var.standardize('vals', 'std_vals')
        self.boltz_warn = self.boltz_var.check_distribution()
        # endregion: Boltzmann distribution

        # region: exponential distribution
        self.exp_interval_low = 0
        self.exp_interval_high = 'oo'
        self.exp_distribution = f'exp(-x)'

        self.exp_var = DiscreteVariable(
            self.exp_distribution, self.exp_interval_low,
            self.exp_interval_high, order=self.order
        )
        # endregion: exponential distribution

    def test_recursive_var_basis(self):
        """
        Testing test_recursive_var_basis for DiscreteVariable.
        """
        tol = 1e-7

        # region: binomial distribution
        x = self.binom_var.x

        # Calculated using SymPy to gather the exact Sum values at each step.
        binom_var_basis_act = np.array([
            1, x - 5.25, x ** 2 - 30.975 - (10.8 * x - 56.7),
            (
                x ** 3 - 199.47375 - (97.285 * x - 510.74625)
                -(16.65 * x ** 2 - 179.82 * x + 428.32125)
            ),
            (
                x ** 4 - (1379.22225) - (845.466 * x - 4438.6965)
                -(203.56 * x ** 2 - 2198.448 * x + 5236.581)
                -(22.8 * x ** 3 - 379.62 * x ** 2 + 1881.798 * x - 2668.7115)
            ),
            (
                x ** 5 - (10126.0695375) - (7368.23575 * x - 38683.2376875)
                -(2238.585 * x ** 2 - 24176.718 * x + 57587.599125)
                -(354.7 * x ** 3 - 5905.755 * x ** 2 + 29275.1645 * x - 41517.191625)
                -(
                    29.25 * x ** 4 - 666.9 * x ** 3 + 5149.755 * x ** 2
                    -15467.868 * x + 14379.4389375
                )
            )
        ])

        diff_sum = np.zeros(self.order)
        basis_diff = Matrix(binom_var_basis_act - self.binom_var.var_orthopoly_vect)

        # Take a sum of the error of all of the coefficients in order of the
        # variable basis.
        for i in range(self.order):
            try:
                diff_sum[i] = np.sum(
                    np.abs(np.array(basis_diff[i].args))
                ).subs(x, 1)
            except:
                diff_sum[i] = np.sum(np.abs(np.array(basis_diff[i].args)))

        self.assertTrue(
            (diff_sum < tol).all(),
            msg='DiscreteVariable recursive_var_basis is not correct'
        )
        # endregion: binomial distribution

        # region: Boltzmann distribution
        x = self.boltz_var.x

        # Calculated using SymPy to gather the exact Sum values at each step.
        boltz_var_basis_act = np.array([
            1, x - 1.99597110686707,
            x ** 2 - (9.59073533393137) - (8.05796339056724 * x - 16.0834621077648),
            (
                x ** 3 - (64.3224991640849)
                -(72.0936813072422 * x - 143.896904876938)
                -(15.3944811551822 * x ** 2 - 124.048165565235 * x + 99.9521599655256)
            ),
            (
                x ** 4 - (532.603202017583)
                -(711.575240418011 * x - 1420.28362023634)
                -(199.276312255745 * x ** 2 - 1605.76122876403 * x + 1293.84664797366)
                -(
                    22.691729495912 * x ** 3 - 349.327402103311 * x ** 2
                    +1178.93710288201 * x - 462.406487235832
                )
            ),
            (
                x ** 5
                -(5052.74533413142) - (7567.41121915258 * x - 15104.3341472102)
                -(2496.71661221338 * x ** 2 - 20118.4510578364 * x + 16210.4987947925)
                -(
                    378.158114354919 * x ** 3 - 5821.54796511607 * x ** 2
                    +19647.0097992853 * x - 7706.01312297887
                )
                -(
                    29.960525045363 * x ** 4 - 679.856129884851 * x ** 3
                    +4495.60943546166 * x ** 2 - 8531.29289874926 * x
                    +1684.98764281426
                )
            )
        ])

        diff_sum = np.zeros(self.order)

        basis_diff = Matrix(boltz_var_basis_act - self.boltz_var.var_orthopoly_vect)

        # Take a sum of the error of all of the coefficients in order of the
        # variable basis.
        for i in range(self.order):
            try:
                diff_sum[i] = np.sum(
                    np.abs(np.array(basis_diff[i].args))
                ).subs(x, 1)
            except:
                diff_sum[i] = np.sum(np.abs(np.array(basis_diff[i].args)))

        self.assertTrue(
            (diff_sum < tol).all(),
            msg='DiscreteVariable recursive_var_basis is not correct'
        )
        # endregion: Boltzmann distribution

    def test_create_norm_sq(self):
        """
        Testing create_norm_sq for DiscreteVariable.
        """
        # region: binomial distribution
        # Calculated seperately using SymPy Sum for the values
        norm_sq_act = np.array([
            1, 3.4125, 21.737625, 192.8670778125, 2106.10848971249,
            26352.6824775273
        ])

        self.assertTrue(
            np.isclose(self.binom_var.norm_sq_vals, norm_sq_act).all(),
            msg='DiscreteVariable create_norm_sq is not correct'
        )
        # endregion: binomial distribution

        # region: Boltzmann distribution
        # Calculated seperately using SymPy Sum for the values
        norm_sq_act = np.array([
            1, 5.60683467448322, 76.5648826469614, 1090.34108277671,
            14495.6667533856, 181892.849624307
        ])

        self.assertTrue(
            np.isclose(self.boltz_var.norm_sq_vals, norm_sq_act).all(),
            msg='DiscreteVariable create_norm_sq is not correct'
        )
        # endregion: Boltzmann distribution

    def test_get_norm_sq_val(self):
        """
        Testing get_norm_sq_val for DiscreteVariable.
        """
        # region: binomial distribution
        # Calculated seperately using SymPy Sum for the values
        norm_sq_act = np.array([
            1, 3.4125, 21.737625, 192.8670778125, 2106.10848971249
        ])
        norm_sq_calc = np.zeros(self.order)

        for i in range(self.order):
            norm_sq_calc[i] = self.binom_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(norm_sq_calc, norm_sq_act).all(),
            msg='DiscreteVariable get_norm_sq_val is not correct'
        )
        # endregion: binomial distribution

        # region: Boltzmann distribution
        # Calculated seperately using SymPy Sum for the values
        norm_sq_act = np.array([
            1, 5.60683467448322, 76.5648826469614, 1090.34108277671,
            14495.6667533856
        ])
        norm_sq_calc = np.zeros(self.order)

        for i in range(self.order):
            norm_sq_calc[i] = self.boltz_var.get_norm_sq_val(i)

        self.assertTrue(
            np.isclose(norm_sq_calc, norm_sq_act).all(),
            msg='DiscreteVariable get_norm_sq_val is not correct'
        )
        # endregion: Boltzmann distribution

    def test_get_probability_density_func(self):
        """
        Testing get_probability_density_func for DiscreteVariable.
        """
        # region: binomial distribution
        x = self.binom_var.x_values
        prob_act = (
            (self.N / (factorial(x) * factorial(self.N - x)))
            * self.p ** x * (1 - self.p) ** (self.N - x)
        )

        self.assertTrue(
            np.isclose(
                self.binom_var.probabilities.astype(float), prob_act
                / np.sum(prob_act)
            ).all(),
            msg='DiscreteVariable get_probability_density_func is not correct'
        )
        # endregion: binomial distribution

        # region: Boltzmann distribution
        prob_act = (
            (1 - np.exp(-0.4)) / (1 - np.exp(-0.4 * 15))
            * np.exp(-0.4 * self.boltz_var.x_values)
        )

        self.assertTrue(
            np.isclose(self.boltz_var.probabilities, prob_act).all(),
            msg='DiscreteVariable get_probability_density_func is not correct'
        )
        # endregion: Boltzmann distribution

        # region: exponential distribution
        low = 0

        x = self.exp_var.x_values
        prob_act = np.exp(-x) / np.sum(np.exp(-x))  # normalize the distribution

        self.assertTrue(
            np.isclose(prob_act, self.exp_var.probabilities).all(),
            msg='DiscreteVariable get_probability_density_func is not correct'
        )

        self.assertEqual(
            low, self.exp_var.x_values[0],
            msg='DiscreteVariable get_probability_density_func is not correct'
        )

        self.assertTrue(
            1 - 1e-10 <= np.sum(self.exp_var.probabilities),
            msg='DiscreteVariable get_probability_density_func is not correct'
        )
        # endregion: exponential distribution

    def test_standardize(self):
        """
        Testing the DiscreteVariable standardize and insuring that the values
        follow a standardized distribution.
        """
        # region: binomial distribution
        self.assertTrue(
            np.equal(self.binom_var.vals, self.binom_var.std_vals).all(),
            msg='DiscreteVariable standardize is not correct'
        )
        # endregion: binomial distribution

        # region: Boltzmann distribution
        self.assertTrue(
            np.equal(self.boltz_var.vals, self.boltz_var.std_vals).all(),
            msg='DiscreteVariable standardize is not correct'
        )
        # endregion: Boltzmann distribution

    def test_standardize_points(self):
        """
        This method does nothing for the DiscreteVariable class since it is
        required that the input is standardized.
        """
        # region: binomial distribution
        test_points = np.array([
            5, 4, 8, 5, 4, 5, 4, 6, 3, 11, 4, 3, 5, 5, 5, 9, 7, 6, 9, 5, 4, 6,
            7, 6, 8, 6, 2, 3, 3, 6, 4, 10, 4, 5, 4, 4, 5, 4, 6, 6, 5, 3, 3, 6,
            5, 2, 6, 5, 8, 10
        ])

        stand_points = self.binom_var.standardize_points(test_points)

        self.assertTrue(
            (test_points == stand_points).all(),
            msg='DiscreteVariable standardize_points is not correct'
        )
        # endregion: binomial distribution

        # region: Boltzmann distribution
        test_points = np.array([])

        stand_points = self.boltz_var.standardize_points(test_points)

        self.assertTrue(
            (test_points == stand_points).all(),
            msg='DiscreteVariable standardize_points is not correct'
        )
        # endregion: Boltzmann distribution

    def test_unstandardize_points(self):
        """
        This method does nothing for the DiscreteVariable class since it is
        required that the input is standardized.
        """

        # region: binomial distribution
        test_points = np.array([
            7, 7, 5, 5, 3, 6, 5, 5, 6, 5, 5, 7, 6, 6, 7, 6, 6, 6, 6, 5, 4, 6,
            6, 6, 8, 8, 7, 5, 4, 10, 9, 3, 7, 5, 7, 4, 2, 8, 7, 8, 6, 4, 5, 8,
            2, 5, 6, 7, 5, 6,
        ])
        stand_points = self.binom_var.unstandardize_points(test_points)

        self.assertTrue(
            (test_points == stand_points).all(),
            msg='DiscreteVariable standardize_points is not correct'
        )
        # endregion: binomial distribution

        # region: Boltzmann distribution
        test_points = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5,
            5, 6, 6, 7, 9, 14
        ])
        stand_points = self.boltz_var.unstandardize_points(test_points)

        self.assertTrue(
            (test_points == stand_points).all(),
            msg='DiscreteVariable standardize_points is not correct'
        )
        # endregion: Boltzmann distribution

    def test_check_distribution(self):
        """
        This method tests that the values are all values this variable can
        accept.
        """
        no_warn = None

        # region: binomial distribution
        self.assertEqual(
            self.binom_warn, no_warn,
            msg='DiscreteVariable check_distribution is not correct'
        )
        # endregion: binomial distribution

        # region: Boltzmann distribution
        self.assertEqual(
            self.boltz_warn, no_warn,
            msg='DiscreteVariable check_distribution is not correct'
        )
        # endregion: Boltzmann distribution

    def test_generate_samples(self):
        """
        Testing the DiscreteVariable sample generation for generating accepted
        values discrete values according to the probabilities.
        """
        samp_size = 5000
        orig_cnt = 1000000
        p_val_min = 0.05
        chi_sq_max = 0.1

        # region: binomial distribution
        gen_vals = self.binom_var.generate_samples(samp_size)
        obs_x, obs = np.unique(gen_vals, return_counts=True)

        exp_x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        exp = np.array([
            1595, 12706, 47493, 110862, 178900, 211922, 190709, 132120, 71292,
            30020, 9630, 2310, 406, 31, 4
        ])

        idx = np.isin(obs_x, exp_x)
        obs = obs[idx] / samp_size
        obs_x = obs_x[idx]

        idx = np.isin(exp_x, obs_x)
        exp = exp[idx] / orig_cnt
        exp_x = exp_x[idx]

        chi_sq, p_val = chisquare(obs, exp)

        self.assertTrue(
            p_val > p_val_min,
            msg='DiscreteVariable generate_samples is not correct'
        )

        self.assertTrue(
            chi_sq < chi_sq_max,
            msg='DiscreteVariable generate_samples is not correct'
        )
        # endregion: binomial distribution

        # region: Boltzmann distribution
        gen_vals = self.boltz_var.generate_samples(samp_size)
        obs_x, obs = np.unique(gen_vals, return_counts=True)

        exp_x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        exp = np.array([
            330499, 221540, 148503, 99544, 66727, 44728, 29982, 20098, 13472,
            9030, 6054, 4057, 2720, 1823, 1223
        ])

        idx = np.isin(obs_x, exp_x)
        obs = obs[idx] / samp_size
        obs_x = obs_x[idx]

        idx = np.isin(exp_x, obs_x)
        exp = exp[idx] / orig_cnt
        exp_x = exp_x[idx]

        chi_sq, p_val = chisquare(obs, exp)

        self.assertTrue(
            p_val > p_val_min,
            msg='DiscreteVariable generate_samples is not correct'
        )

        self.assertTrue(
            chi_sq < chi_sq_max,
            msg='DiscreteVariable generate_samples is not correct'
        )
        # endregion: Boltzmann distribution

    def test_get_resamp_vals(self):
        """
        Testing that the resampling values are only the allowed values and that
        the distribution is appropriate.
        """
        samp_size = 5000
        orig_cnt = 1000000
        p_val_min = 0.05
        chi_sq_max = 0.1

        # region: binomial distribution
        self.binom_var.get_resamp_vals(samp_size)
        obs_x, obs = np.unique(self.binom_var.resample, return_counts=True)

        exp_x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        exp = np.array([
            1607, 12695, 47527, 111285, 179254, 212406, 190850, 131215, 71477,
            29340, 9580, 2291, 427, 46
        ])

        idx = np.isin(obs_x, exp_x)
        obs = obs[idx] / samp_size
        obs_x = obs_x[idx]

        idx = np.isin(exp_x, obs_x)
        exp = exp[idx] / orig_cnt
        exp_x = exp_x[idx]

        chi_sq, p_val = chisquare(obs, exp)

        self.assertTrue(
            p_val > p_val_min,
            msg='DiscreteVariable get_resamp_vals is not correct'
        )

        self.assertTrue(
            chi_sq < chi_sq_max,
            msg='DiscreteVariable get_resamp_vals is not correct'
        )
        # endregion: binomial distribution

        # region: Boltzmann distribution
        self.boltz_var.get_resamp_vals(samp_size)
        obs_x, obs = np.unique(self.boltz_var.resample, return_counts=True)

        exp_x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        exp = np.array([
            330499, 221540, 148503, 99544, 66727, 44728, 29982, 20098, 13472,
            9030, 6054, 4057, 2720, 1823, 1223
        ])

        idx = np.isin(obs_x, exp_x)
        obs = obs[idx] / samp_size
        obs_x = obs_x[idx]

        idx = np.isin(exp_x, obs_x)
        exp = exp[idx] / orig_cnt
        exp_x = exp_x[idx]

        chi_sq, p_val = chisquare(obs, exp)

        self.assertTrue(
            p_val > p_val_min,
            msg='DiscreteVariable get_resamp_vals is not correct'
        )

        self.assertTrue(
            chi_sq < chi_sq_max,
            msg='DiscreteVariable get_resamp_vals is not correct'
        )
        # endregion: Boltzmann distribution

    def test_check_num_string(self):
        """
        Testing that 'pi' values are replaced in the x_values.
        """
        # region: binomial distribution
        self.binom_var.check_num_string()

        self.assertEqual(
            self.binom_var.interval_low, 0,
            msg='DiscreteVariable check_num_string is not correct'
        )

        self.assertEqual(
            self.binom_var.interval_high, self.N,
            msg='DiscreteVariable check_num_string is not correct'
        )
        # endregion: binomial distribution

        # region: Boltzmann distribution
        self.boltz_var.check_num_string()

        self.assertEqual(
            self.boltz_var.interval_low, 0,
            msg='DiscreteVariable check_num_string is not correct'
        )

        self.assertEqual(
            self.boltz_var.interval_high, self.N - 1,
            msg='DiscreteVariable check_num_string is not correct'
        )
        # endregion: Boltzmann distribution

    def test_get_mean(self):
        """
        Testing the mean calculation for the DiscreteVariable.
        """
        tol = 1e-14

        # region: binomial distribution
        act_mean = self.N * self.p

        calc_mean = self.binom_var.get_mean()

        self.assertTrue(
            np.isclose(act_mean, float(calc_mean), rtol=0, atol=tol),
            msg='DiscreteVariable get_mean is not correct'
        )
        # endregion: binomial distribution

        # region: Boltzmann distribution
        act_mean = boltzmann(self.lambd, self.N).mean()

        calc_mean = self.boltz_var.get_mean()

        self.assertTrue(
            np.isclose(act_mean, float(calc_mean), rtol=0, atol=tol),
            msg='DiscreteVariable get_mean is not correct'
        )
        # endregion: Boltzmann distribution


class TestPoissonVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

        order = 5
        self.lambd = 3
        self.interval_low = 10

        samp_size = 5000

        self.pois_var = PoissonVariable(
            self.lambd, interval_low=self.interval_low, order=order
        )

        self.pois_samps = self.pois_var.get_resamp_vals(samp_size)

        self.pois_var.vals = np.array([
            11, 10, 14, 11, 13, 14, 12, 15, 12, 11, 15, 13, 11, 11, 12, 14, 13,
            12, 12, 15, 13, 16, 12, 15, 13, 14, 12, 14, 12, 14, 13, 13, 18, 13,
            11, 13, 11, 12, 12, 17, 12, 13, 16, 10, 11, 13, 14, 15, 12, 14
       ])

        self.pois_var.standardize('vals', 'std_vals')
        self.pois_warn = self.pois_var.check_distribution()

    def test_find_high_lim(self):
        """
        Testing find_high_lim for PoissonVariable.
        """
        low = 0
        high = 26

        self.assertEqual(
            low, self.pois_var.x_values[0],
            msg='PoissonVariable find_high_lim is not correct'
        )

        self.assertEqual(
            high, self.pois_var.x_values[-1],
            msg='PoissonVariable find_high_lim is not correct'
        )

    def test_get_probability_density_func(self):
        """
        Testing get_probability_density_func for PoissonVariable.
        """
        x = self.pois_var.x_values
        dist = np.exp(-self.lambd) * self.lambd ** (x) / factorial(x)

        self.assertTrue(
            np.isclose(dist, self.pois_var.probabilities).all(),
            msg='PoissonVariable get_probability_density_func is not correct'
        )

    def test_standardize(self):
        """
        Testing the PoissonVariable standardize_points and insuring that the
        values follow a standardized distribution.
        """
        tol = 1e-15

        act_stand = np.array([
            1, 0, 4, 1, 3, 4, 2, 5, 2, 1, 5, 3, 1, 1, 2, 4, 3, 2, 2, 5, 3, 6,
            2, 5, 3, 4, 2, 4, 2, 4, 3, 3, 8, 3, 1, 3, 1, 2, 2, 7, 2, 3, 6, 0,
            1, 3, 4, 5, 2, 4
        ])

        self.assertTrue(
            (
                np.isclose(act_stand, self.pois_var.std_vals, rtol=0, atol=tol)
            ).all(),
            msg='PoissonVariable standardize is not correct'
        )

    def test_standardize_points(self):
        """
        Testing the PoissonVariable standardize_points and ensuring that the
        values follow a standardized distribution.
        """
        tol = 1e-15

        orig_points = np.array([
            10, 15, 16, 11, 14, 11, 13, 11, 12, 12, 13, 14, 12, 13, 15, 14, 12,
            11, 10, 12, 11, 16, 16, 12, 13, 14, 11, 12, 18, 12, 15, 12, 12, 13,
            16, 11, 14, 13, 13, 13, 15, 12, 14, 13, 14, 13, 10, 14, 13, 14
        ])

        act_stand_points = np.array([
            0, 5, 6, 1, 4, 1, 3, 1, 2, 2, 3, 4, 2, 3, 5, 4, 2, 1, 0, 2, 1, 6,
            6, 2, 3, 4, 1, 2, 8, 2, 5, 2, 2, 3, 6, 1, 4, 3, 3, 3, 5, 2, 4, 3,
            4, 3, 0, 4, 3, 4
        ])

        stand_points = self.pois_var.standardize_points(orig_points)

        self.assertTrue(
            np.isclose(stand_points, act_stand_points, rtol=0, atol=tol).all(),
            msg='PoissonVariable standardize_points is not correct'
        )

    def test_unstandardize_points(self):
        """
        Testing the PoissonVariable unstandardize_points and ensuring that the
        values follow an unstandardized distribution.
        """
        tol = 1e-15

        orig_points = np.array([
            3, 5, 6, 1, 1, 2, 1, 2, 0, 1, 2, 5, 3, 5, 6, 2, 3, 2, 5, 2, 4, 3,
            4, 1, 3, 2, 5, 3, 3, 4, 3, 4, 4, 7, 2, 8, 3, 4, 2, 1, 3, 2, 0, 1,
            4, 0, 2, 3, 4, 5
        ])

        act_unstand_points = np.array([
            13, 15, 16, 11, 11, 12, 11, 12, 10, 11, 12, 15, 13, 15, 16, 12, 13,
            12, 15, 12, 14, 13, 14, 11, 13, 12, 15, 13, 13, 14, 13, 14, 14, 17,
            12, 18, 13, 14, 12, 11, 13, 12, 10, 11, 14, 10, 12, 13, 14, 15
        ])

        unstand_points = self.pois_var.unstandardize_points(orig_points)

        self.assertTrue(
            np.isclose(unstand_points, act_unstand_points, rtol=0, atol=tol).all(),
            msg='PoissonVariable unstandardize_points is not correct'
        )

    def test_check_distribution(self):
        """
        Testing the PoissonVariable check_distribution to ensure that no warning
        is raised.
        """
        no_warn = None

        self.assertEqual(
            self.pois_warn, no_warn,
            msg='PoissonVariable check_distribution is not correct'
        )

    def test_generate_samples(self):
        """
        Testing the PoissonVariable generate_samples to ensure that the samples
        generated reasonably fit a Poisson distribution.
        """
        samp_size = 5000
        orig_cnt = 1000000
        p_val_min = 0.05
        chi_sq_max = 0.1

        gen_vals = self.pois_var.generate_samples(samp_size)
        obs_x, obs = np.unique(gen_vals, return_counts=True)

        exp_x = np.array([
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
        ])
        exp = np.array([
            49787, 149361, 224042, 224042, 168031, 100820, 50408, 21604, 8102,
            2700, 810, 222, 55, 13, 2, 1
        ])

        idx = np.isin(obs_x, exp_x)
        obs = obs[idx] / samp_size

        idx = np.isin(exp_x, obs_x)
        exp = exp[idx] / orig_cnt

        chi_sq, p_val = chisquare(obs, exp)

        self.assertTrue(
            p_val > p_val_min,
            msg='PoissonVariable generate_samples is not correct'
        )

        self.assertTrue(
            chi_sq < chi_sq_max,
            msg='PoissonVariable generate_samples is not correct'
        )

    def test_get_resamp_vals(self):
        """
        Testing the PoissonVariable get_resamp_vals to ensure that the samples
        generated reasonably fit a Poisson distribution.
        """
        samp_size = 5000
        orig_cnt = 1000000
        p_val_min = 0.05
        chi_sq_max = 0.1

        gen_vals = self.pois_var.get_resamp_vals(samp_size)
        obs_x, obs = np.unique(gen_vals, return_counts=True)

        exp_x = np.array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ])
        exp = np.array([
            49787, 149361, 224042, 224042, 168031, 100819, 50410, 21604, 8101,
            2700, 810, 222, 55, 13, 2, 1
        ])

        idx = np.isin(obs_x, exp_x)
        obs = obs[idx] / samp_size

        idx = np.isin(exp_x, obs_x)
        exp = exp[idx] / orig_cnt

        chi_sq, p_val = chisquare(obs, exp)

        self.assertTrue(
            p_val > p_val_min,
            msg='PoissonVariable get_resamp_vals is not correct'
        )

        self.assertTrue(
            chi_sq < chi_sq_max,
            msg='PoissonVariable get_resamp_vals is not correct'
        )

    def test_check_num_string(self):
        """
        Testing the PoissonVariable check_num_string to ensure that pi values
        are appropriately converted to floats.
        """
        self.pois_var.interval_low = '-pi'
        setattr(self.pois_var, 'lambda', 'pi')

        self.pois_var.check_num_string()

        self.assertEqual(
            getattr(self.pois_var, 'lambda'), np.pi,
            msg='PoissonVariable check_num_string is not correct'
        )

        self.assertEqual(
            self.pois_var.interval_low, -np.pi,
            msg='PoissonVariable check_num_string is not correct'
        )

    def test_get_mean(self):
        """
        Testing the PoissonVariable get_mean to ensure that the mean value is
        correct.
        """

        act_mean = self.interval_low + self.lambd
        calc_mean = self.pois_var.get_mean()

        self.assertEqual(
            act_mean, calc_mean,
            msg='PoissonVariable get_mean is not correct'
        )


class TestNegativeBinomialVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        order = 5
        self.r = 20
        self.p = 0.95
        self.interval_low = -7

        samp_size = 5000

        self.negbi_var = NegativeBinomialVariable(
            self.r, self.p, order=order, interval_low=self.interval_low
        )

        self.negbi_samps = self.negbi_var.get_resamp_vals(samp_size)

        self.negbi_var.vals = np.array([
            -5, -5, -6, -7, -6, -7, -7, -6, -7, -5, -5, -4, -6, -6, -7, -4, -5,
            -7, -6, -3, -6, -7, -7, -6, -3, -7, -6, -5, -7, -5, -4, -7, -7, -5,
            -6, -5, -7, -7, -6, -6, -6, -5, -5, -4, -7, -7, -4, -7, -6, -6
        ])

        self.negbi_var.standardize('vals', 'std_vals')
        self.negbi_warn = self.negbi_var.check_distribution()

    def test_find_high_lim(self):
        """
        Testing find_high_lim for NegativeBinomialVariable.
        """
        low = 0
        high = 20

        self.assertEqual(
            low, self.negbi_var.x_values[0],
            msg='NegativeBinomialVariable find_high_lim is not correct'
        )
        self.assertEqual(
            high, self.negbi_var.x_values[-1],
            msg='NegativeBinomialVariable find_high_lim is not correct'
        )

    def test_get_probability_density_func(self):
        """
        Testing get_probability_density_func for NegativeBinomialVariable.
        """
        x = self.negbi_var.x_values
        dist = (
            factorial(x + self.r - 1) / (factorial(x) * factorial(self.r - 1))
            * (1 - self.p) ** (x) * self.p ** self.r
        )

        self.assertTrue(
            np.isclose(dist, self.negbi_var.probabilities).all(),
            msg='NegativeBinomialVariable get_probability_density_func is not correct'
        )

    def test_standardize(self):
        """
        Testing the NegativeBinomialVariable standardize_points and insuring that the
        values follow a standardized distribution.
        """
        tol = 1e-15

        act_stand = np.array([
            2, 2, 1, 0, 1, 0, 0, 1, 0, 2, 2, 3, 1, 1, 0, 3, 2, 0, 1, 4, 1, 0,
            0, 1, 4, 0, 1, 2, 0, 2, 3, 0, 0, 2, 1, 2, 0, 0, 1, 1, 1, 2, 2, 3,
            0, 0, 3, 0, 1, 1
        ])

        self.assertTrue(
            (
                np.isclose(act_stand, self.negbi_var.std_vals, rtol=0, atol=tol)
            ).all(),
            msg='NegativeBinomialVariable standardize is not correct'
        )

    def test_standardize_points(self):
        """
        Testing the NegativeBinomialVariable standardize_points and ensuring that the
        values follow a standardized distribution.
        """
        tol = 1e-15

        orig_points = np.array([
            -7, -6, -7, -7, -7, -6, -6, -5, -6, -5, -7, -6, -6, -5, -6, -6, -6,
            -7, -6, -7, -6, -7, -5, -6, -5, -6, -6, -6, -7, -7, -6, -7, -6, -5,
            -4, -6, -6, -6, -7, -5, -7, -7, -6, -5, -6, -7, -4, -6, -7, -6
        ])

        act_stand_points = np.array([
            0, 1, 0, 0, 0, 1, 1, 2, 1, 2, 0, 1, 1, 2, 1, 1, 1, 0, 1, 0, 1, 0,
            2, 1, 2, 1, 1, 1, 0, 0, 1, 0, 1, 2, 3, 1, 1, 1, 0, 2, 0, 0, 1, 2,
            1, 0, 3, 1, 0, 1
        ])

        stand_points = self.negbi_var.standardize_points(orig_points)

        self.assertTrue(
            np.isclose(stand_points, act_stand_points, rtol=0, atol=tol).all(),
            msg='NegativeBinomialVariable standardize_points is not correct'
        )

    def test_unstandardize_points(self):
        """
        Testing the NegativeBinomialVariable unstandardize_points and ensuring that the
        values follow an unstandardized distribution.
        """
        tol = 1e-15

        orig_points = np.array([
            0, 1, 0, 0, 0, 1, 1, 2, 1, 2, 0, 1, 1, 2, 1, 1, 1, 0, 1, 0, 1, 0,
            2, 1, 2, 1, 1, 1, 0, 0, 1, 0, 1, 2, 3, 1, 1, 1, 0, 2, 0, 0, 1, 2,
            1, 0, 3, 1, 0, 1
        ])

        act_unstand_points = np.array([
            -7, -6, -7, -7, -7, -6, -6, -5, -6, -5, -7, -6, -6, -5, -6, -6, -6,
            -7, -6, -7, -6, -7, -5, -6, -5, -6, -6, -6, -7, -7, -6, -7, -6, -5,
            -4, -6, -6, -6, -7, -5, -7, -7, -6, -5, -6, -7, -4, -6, -7, -6
        ])

        unstand_points = self.negbi_var.unstandardize_points(orig_points)

        self.assertTrue(
            np.isclose(unstand_points, act_unstand_points, rtol=0, atol=tol).all(),
            msg='NegativeBinomialVariable unstandardize_points is not correct'
        )

    def test_check_distribution(self):
        """
        Testing the NegativeBinomialVariable check_distribution to ensure that no warning
        is raised.
        """
        no_warn = None

        self.assertEqual(
            self.negbi_warn, no_warn,
            msg='NegativeBinomialVariable check_distribution is not correct'
        )

    def test_generate_samples(self):
        """
        Testing the NegativeBinomialVariable generate_samples to ensure that the samples
        generated reasonably fit a Poisson distribution.
        """
        samp_size = 5000
        orig_cnt = 1000000
        p_val_min = 0.05
        chi_sq_max = 0.1

        gen_vals = self.negbi_var.generate_samples(samp_size)
        obs_x, obs = np.unique(gen_vals, return_counts=True)

        exp_x = np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3])
        exp = np.array([
            358837, 358033, 188198, 69063, 19835, 4837, 975, 184, 32, 4, 2
        ])

        idx = np.isin(obs_x, exp_x)
        obs = obs[idx] / samp_size

        idx = np.isin(exp_x, obs_x)
        exp = exp[idx] / orig_cnt

        chi_sq, p_val = chisquare(obs, exp)

        self.assertTrue(
            p_val > p_val_min,
            msg='NegativeBinomialVariable generate_samples is not correct'
        )

        self.assertTrue(
            chi_sq < chi_sq_max,
            msg='NegativeBinomialVariable generate_samples is not correct'
        )

    def test_get_resamp_vals(self):
        """
        Testing the NegativeBinomialVariable get_resamp_vals to ensure that the samples
        generated reasonably fit a Poisson distribution.
        """
        samp_size = 5000
        orig_cnt = 1000000
        p_val_min = 0.05
        chi_sq_max = 0.1

        gen_vals = self.negbi_var.get_resamp_vals(samp_size)
        obs_x, obs = np.unique(gen_vals, return_counts=True)

        exp_x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        exp = np.array([
            358397, 358069, 189186, 68695, 19675, 4733, 1015, 198, 29, 2, 1
        ])

        idx = np.isin(obs_x, exp_x)
        obs = obs[idx] / samp_size

        idx = np.isin(exp_x, obs_x)
        exp = exp[idx] / orig_cnt

        chi_sq, p_val = chisquare(obs, exp)

        self.assertTrue(
            p_val > p_val_min,
            msg='NegativeBinomialVariable get_resamp_vals is not correct'
        )

        self.assertTrue(
            chi_sq < chi_sq_max,
            msg='NegativeBinomialVariable get_resamp_vals is not correct'
        )

    def test_check_num_string(self):
        """
        Testing the NegativeBinomialVariable check_num_string to ensure that pi values
        are appropriately converted to floats.
        """
        self.negbi_var.interval_low = '-pi'
        self.negbi_var.r = 'pi'

        self.negbi_var.check_num_string()

        self.assertEqual(
            self.negbi_var.interval_low, -np.pi,
            msg='NegativeBinomialVariable check_num_string is not correct'
        )

        self.assertEqual(
            self.negbi_var.r, np.pi,
            msg='NegativeBinomialVariable check_num_string is not correct'
        )

    def test_get_mean(self):
        """
        Testing the NegativeBinomialVariable get_mean to ensure that the mean value is
        correct.
        """
        tol = 1e-14

        act_mean = ((self.r * (1 - self.p)) / self.p) + self.interval_low
        calc_mean = self.negbi_var.get_mean()

        self.assertAlmostEqual(
            act_mean, calc_mean, delta=tol,
            msg='NegativeBinomialVariable get_mean is not correct'
        )


class TestHypergeometricVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        order = 5
        self.M = 15
        self.n = 20
        self.N = 6
        self.interval_shift = 4

        samp_size = 5000

        self.hyp_geo_var = HypergeometricVariable(
            self.M, self.n, self.N, order=order, interval_shift=self.interval_shift
        )
        self.hyp_geo_samps = self.hyp_geo_var.get_resamp_vals(samp_size)

        self.hyp_geo_var.vals = np.array([
            7, 8, 8, 6, 6, 7, 9, 7, 8, 7, 8, 7, 7, 8, 9, 9, 8, 6, 6, 7, 9, 7,
            8, 7, 9, 7, 6, 7, 8, 7, 7, 8, 8, 8, 9, 7, 8, 8, 7, 8, 6, 6, 8, 8,
            9, 6, 8, 7, 5, 7
        ])

        self.hyp_geo_var.standardize('vals', 'std_vals')
        self.hyp_geo_warn = self.hyp_geo_var.check_distribution()

    def test_find_high_lim(self):
        """
        Testing find_high_lim for HypergeometricVariable.
        """
        low = 0
        high = self.N

        self.assertEqual(
            low, self.hyp_geo_var.x_values[0],
            msg='HypergeometricVariable find_high_lim is not correct'
        )

        self.assertEqual(
            high, self.hyp_geo_var.x_values[-1],
            msg='HypergeometricVariable find_high_lim is not correct'
        )

    def test_get_probability_density_func(self):
        """
        Testing get_probability_density_func for HypergeometricVariable.
        """
        x = self.hyp_geo_var.x_values
        dist = (
            (
                factorial(self.M) * factorial(self.n) * factorial(self.N)
                * factorial(self.M + self.n - self.N)
            )
            / (
                factorial(x) * factorial(self.n - x)
                * factorial(self.M + x - self.N) * factorial(self.N - x)
                * factorial(self.M + self.n)
            )
        )

        self.assertTrue(
            np.isclose(dist, self.hyp_geo_var.probabilities).all(),
            msg='HypergeometricVariable get_probability_density_func is not correct'
        )

    def test_standardize(self):
        """
        Testing the HypergeometricVariable standardize_points and insuring that the
        values follow a standardized distribution.
        """
        tol = 1e-15

        act_stand = np.array([
            3, 4, 4, 2, 2, 3, 5, 3, 4, 3, 4, 3, 3, 4, 5, 5, 4, 2, 2, 3, 5, 3,
            4, 3, 5, 3, 2, 3, 4, 3, 3, 4, 4, 4, 5, 3, 4, 4, 3, 4, 2, 2, 4, 4,
            5, 2, 4, 3, 1, 3
        ])

        self.assertTrue(
            (
                np.isclose(act_stand, self.hyp_geo_var.std_vals, rtol=0, atol=tol)
            ).all(),
            msg='HypergeometricVariable standardize is not correct'
        )

    def test_standardize_points(self):
        """
        Testing the HypergeometricVariable standardize_points and ensuring that the
        values follow a standardized distribution.
        """
        tol = 1e-15

        orig_points = np.array([
            6, 8, 7, 8, 6, 8, 7, 8, 8, 8, 7, 6, 7, 9, 6, 8, 8, 10, 6, 7, 6, 9,
            5, 8, 8, 9, 6, 7, 7, 8, 7, 9, 6, 7, 6, 6, 9, 8, 8, 6, 9, 8, 7, 7,
            7, 6, 7, 6, 6, 7
        ])

        act_stand_points = np.array([
            2, 4, 3, 4, 2, 4, 3, 4, 4, 4, 3, 2, 3, 5, 2, 4, 4, 6, 2, 3, 2, 5,
            1, 4, 4, 5, 2, 3, 3, 4, 3, 5, 2, 3, 2, 2, 5, 4, 4, 2, 5, 4, 3, 3,
            3, 2, 3, 2, 2, 3
        ])

        stand_points = self.hyp_geo_var.standardize_points(orig_points)

        self.assertTrue(
            np.isclose(stand_points, act_stand_points, rtol=0, atol=tol).all(),
            msg='HypergeometricVariable standardize_points is not correct'
        )

    def test_unstandardize_points(self):
        """
        Testing the HypergeometricVariable unstandardize_points and ensuring 
        that the values follow an unstandardized distribution.
        """
        tol = 1e-15

        orig_points = np.array([
            4, 3, 4, 4, 3, 5, 5, 4, 4, 4, 3, 3, 3, 3, 5, 2, 3, 2, 3, 3, 3, 5,
            3, 2, 5, 6, 3, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 3, 3, 2, 3, 4, 1,
            5, 4, 3, 4, 3, 3
        ])

        act_unstand_points = np.array([
            8, 7, 8, 8, 7, 9, 9, 8, 8, 8, 7, 7, 7, 7, 9, 6, 7, 6, 7, 7, 7, 9,
            7, 6, 9, 10, 7, 6, 7, 7, 8, 7, 8, 8, 7, 6, 7, 7, 7, 7, 6, 7, 8, 5,
            9, 8, 7, 8, 7, 7
        ])

        unstand_points = self.hyp_geo_var.unstandardize_points(orig_points)

        self.assertTrue(
            np.isclose(unstand_points, act_unstand_points, rtol=0, atol=tol).all(),
            msg='HypergeometricVariable unstandardize_points is not correct'
        )

    def test_check_distribution(self):
        """
        Testing the HypergeometricVariable check_distribution to ensure that 
        no warning is raised.
        """
        no_warn = None

        self.assertEqual(
            self.hyp_geo_warn, no_warn,
            msg='HypergeometricVariable check_distribution is not correct'
        )

    def test_generate_samples(self):
        """
        Testing the HypergeometricVariable generate_samples to ensure that the samples
        generated reasonably fit a Poisson distribution.
        """
        samp_size = 5000
        orig_cnt = 1000000
        p_val_min = 0.05
        chi_sq_max = 0.1

        gen_vals = self.hyp_geo_var.generate_samples(samp_size)
        obs_x, obs = np.unique(gen_vals, return_counts=True)

        exp_x = np.array([
            4, 5, 6, 7, 8, 9, 10
        ])

        exp = np.array([
            3102, 36793, 160128, 319585, 313153, 143231, 24008
        ])

        idx = np.isin(obs_x, exp_x)
        obs = obs[idx] / samp_size

        idx = np.isin(exp_x, obs_x)
        exp = exp[idx] / orig_cnt

        chi_sq, p_val = chisquare(obs, exp)

        self.assertTrue(
            p_val > p_val_min,
            msg='HypergeometricVariable generate_samples is not correct'
        )

        self.assertTrue(
            chi_sq < chi_sq_max,
            msg='HypergeometricVariable generate_samples is not correct'
        )

    def test_get_resamp_vals(self):
        """
        Testing the HypergeometricVariable get_resamp_vals to ensure that the samples
        generated reasonably fit a Poisson distribution.
        """
        samp_size = 5000
        orig_cnt = 1000000
        p_val_min = 0.05
        chi_sq_max = 0.1

        gen_vals = self.hyp_geo_var.get_resamp_vals(samp_size)
        obs_x, obs = np.unique(gen_vals, return_counts=True)

        exp_x = np.array([
            0, 1, 2, 3, 4, 5, 6
        ])

        exp = np.array([
            3187, 37021, 160103, 319357, 313119, 143406, 23807
        ])

        idx = np.isin(obs_x, exp_x)
        obs = obs[idx] / samp_size

        idx = np.isin(exp_x, obs_x)
        exp = exp[idx] / orig_cnt

        chi_sq, p_val = chisquare(obs, exp)

        self.assertTrue(
            p_val > p_val_min,
            msg='HypergeometricVariable get_resamp_vals is not correct'
        )

        self.assertTrue(
            chi_sq < chi_sq_max,
            msg='HypergeometricVariable get_resamp_vals is not correct'
        )

    def test_check_num_string(self):
        """
        Testing the HypergeometricVariable check_num_string to ensure that pi values
        are appropriately converted to floats.
        """
        pass

    def test_get_mean(self):
        """
        Testing the HypergeometricVariable get_mean to ensure that the mean 
        value is correct.
        """
        act_mean = (self.N * self.n) / (self.M + self.n) + self.interval_shift
        calc_mean = self.hyp_geo_var.get_mean()

        self.assertEqual(
            act_mean, calc_mean,
            msg='HypergeometricVariable get_mean is not correct'
        )


class TestDiscreteUniformVariable(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        order = 5
        self.interval_low = 4
        self.interval_high = 9

        samp_size = 5000

        self.disc_unif_var = DiscreteUniformVariable(
            self.interval_low, self.interval_high, order=order
        )

        self.disc_unif_samps = self.disc_unif_var.get_resamp_vals(samp_size)

        self.disc_unif_var.vals = np.array([
            8, 9, 4, 9, 7, 4, 5, 7, 9, 9, 4, 9, 9, 6, 8, 4, 6, 9, 8, 9, 7, 4,
            9, 9, 8, 8, 8, 6, 7, 5, 8, 4, 5, 4, 4, 8, 8, 5, 9, 4, 6, 6, 8, 8,
            5, 6, 6, 7, 8, 8
        ])

        self.disc_unif_var.standardize('vals', 'std_vals')
        self.disc_unif_warn = self.disc_unif_var.check_distribution()

    def test_standardize(self):
        """
        Testing the UniformVariable standardize_points and insuring that the
        values follow a standardized distribution.
        """
        tol = 1e-15

        act_stand = np.array([
            0.6, 1, -1, 1, 0.2, -1, -0.6, 0.2, 1, 1, -1, 1, 1, -0.2, 0.6, -1,
            -0.2, 1, 0.6, 1, 0.2, -1, 1, 1, 0.6, 0.6, 0.6, -0.2, 0.2, -0.6,
            0.6, -1, -0.6, -1, -1, 0.6, 0.6, -0.6, 1, -1, -0.2, -0.2, 0.6, 0.6,
            -0.6, -0.2, -0.2, 0.2, 0.6, 0.6
        ])

        self.assertTrue(
            (
                np.isclose(act_stand, self.disc_unif_var.std_vals, rtol=0, atol=tol)
            ).all(),
            msg='Discrete UniformVariable standardize is not correct'
        )

    def test_standardize_points(self):
        """
        Testing the UniformVariable standardize_points and ensuring that the
        values follow a standardized distribution.
        """
        tol = 1e-15

        orig_points = np.array([
             5, 6, 7, 5, 6, 5, 9, 8, 8, 7, 4, 4, 4, 7, 8, 5, 6, 4, 8, 9, 8, 4,
             4, 4, 7, 6, 4, 9, 9, 8, 8, 6, 4, 8, 4, 7, 8, 6, 9, 6, 6, 6, 8, 6,
             5, 6, 8, 8, 9, 9
        ])

        act_stand_points = np.array([
            -0.6, -0.2, 0.2, -0.6, -0.2, -0.6, 1, 0.6, 0.6, 0.2, -1, -1, -1,
            0.2, 0.6, -0.6, -0.2, -1, 0.6, 1, 0.6, -1, -1, -1, 0.2, -0.2, -1,
            1, 1, 0.6, 0.6, -0.2, -1, 0.6, -1, 0.2, 0.6, -0.2, 1, -0.2, -0.2,
            -0.2, 0.6, -0.2, -0.6, -0.2, 0.6, 0.6, 1, 1
        ])

        stand_points = self.disc_unif_var.standardize_points(orig_points)

        self.assertTrue(
            np.isclose(stand_points, act_stand_points, rtol=0, atol=tol).all(),
            msg='Discrete UniformVariable standardize_points is not correct'
        )

    def test_unstandardize_points(self):
        """
        Testing the UniformVariable unstandardize_points and ensuring that the
        values follow an unstandardized distribution.
        """
        tol = 1e-15

        orig_points = np.array([
            1, 0.6, -0.2, -0.6, -1, 0.6, -1, 0.6, -0.6, 1, -0.6, -0.6, 0.6,
            -0.2, 0.6, 0.2, 1, 0.2, -0.2, -0.6, 0.6, -1, -0.2, -0.6, 0.2, 0.2,
            0.6, 0.2, -0.6, -0.6, -0.2, -1, -1, -1, -0.2, 1, -1, -0.6, 0.2,
            -0.6, 0.6, 0.6, 0.2, 0.2, -0.6, 1, -0.6, -0.6, -0.6, 0.6
        ])

        act_unstand_points = np.array([
            9, 8, 6, 5, 4, 8, 4, 8, 5, 9, 5, 5, 8, 6, 8, 7, 9, 7, 6, 5, 8, 4,
            6, 5, 7, 7, 8, 7, 5, 5, 6, 4, 4, 4, 6, 9, 4, 5, 7, 5, 8, 8, 7, 7,
            5, 9, 5, 5, 5, 8
        ])

        unstand_points = self.disc_unif_var.unstandardize_points(orig_points)

        self.assertTrue(
            np.isclose(unstand_points, act_unstand_points, rtol=0, atol=tol).all(),
            msg='Discrete UniformVariable unstandardize_points is not correct'
        )

    def test_check_distribution(self):
        """
        Testing the UniformVariable check_distribution to ensure that no warning
        is raised.
        """
        no_warn = None

        self.assertEqual(
            self.disc_unif_warn, no_warn,
            msg='Discrete UniformVariable check_distribution is not correct'
        )

    def test_generate_samples(self):
        """
        Testing the UniformVariable generate_samples to ensure that the samples
        generated reasonably fit a Poisson distribution.
        """
        samp_size = 5000
        orig_cnt = 1000000
        p_val_min = 0.05
        chi_sq_max = 0.1

        gen_vals = self.disc_unif_var.generate_samples(samp_size)
        obs_x, obs = np.unique(gen_vals, return_counts=True)

        exp_x = np.array([4, 5, 6, 7, 8, 9])
        exp = np.array([167143, 165837, 166562, 166266, 167511, 166681])

        idx = np.isin(obs_x, exp_x)
        obs = obs[idx] / samp_size

        idx = np.isin(exp_x, obs_x)
        exp = exp[idx] / orig_cnt

        chi_sq, p_val = chisquare(obs, exp)

        self.assertTrue(
            p_val > p_val_min,
            msg='Discrete UniformVariable generate_samples is not correct'
        )

        self.assertTrue(
            chi_sq < chi_sq_max,
            msg='Discrete UniformVariable generate_samples is not correct'
        )

    def test_get_resamp_vals(self):
        """
        Testing the UniformVariable get_resamp_vals to ensure that the samples
        generated reasonably fit a Poisson distribution.
        """
        samp_size = 5000
        orig_cnt = 1000000
        p_val_min = 0.05
        chi_sq_max = 0.1

        gen_vals = self.disc_unif_var.get_resamp_vals(samp_size)
        obs_x, obs = np.unique(gen_vals, return_counts=True)

        exp_x = np.array([-1, -0.6, -0.2, 0.2, 0.6, 1])
        exp = np.array([166678, 166342, 166750, 166893, 166905, 166432])

        idx = np.isin(obs_x, exp_x)
        obs = obs[idx] / samp_size

        idx = np.isin(exp_x, obs_x)
        exp = exp[idx] / orig_cnt

        chi_sq, p_val = chisquare(obs, exp)

        self.assertTrue(
            p_val > p_val_min,
            msg='Discrete UniformVariable get_resamp_vals is not correct'
        )

        self.assertTrue(
            chi_sq < chi_sq_max,
            msg='Discrete UniformVariable get_resamp_vals is not correct'
        )

    def test_check_num_string(self):
        """
        Testing the UniformVariable check_num_string to ensure that pi values
        are appropriately converted to floats.
        """
        self.disc_unif_var.interval_low = '-pi'
        self.disc_unif_var.interval_high = 'pi'

        self.disc_unif_var.check_num_string()

        self.assertEqual(
            self.disc_unif_var.interval_low, -np.pi,
            msg='Discrete UniformVariable check_num_string is not correct'
        )

        self.assertEqual(
            self.disc_unif_var.interval_high, np.pi,
            msg='Discrete UniformVariable check_num_string is not correct'
        )

    def test_get_mean(self):
        """
        Testing the UniformVariable get_mean to ensure that the mean value is
        correct.
        """

        act_mean = (self.interval_low + self.interval_high) / 2
        calc_mean = self.disc_unif_var.get_mean()

        self.assertEqual(
            act_mean, calc_mean,
            msg='Discrete UniformVariable get_mean is not correct'
        )
