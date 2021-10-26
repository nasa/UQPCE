import unittest

import numpy as np
from sympy import symbols, Matrix, lambdify

from PCE_Codes.stats.statistics import (
    calc_error_variance, calc_coeff_conf_int, calc_pred_conf_int,
    calc_R_sq_adj, calc_R_sq, calc_mean_conf_int, calc_term_count,
    calc_min_responses, calc_partial_F, calc_mean_sq_err, calc_sum_sq_regr,
    calc_error_sum_of_sq, calc_total_sum_of_sq, calc_PRESS_res,
    calc_hat_matrix, get_sobol_bounds, calc_var_conf_int
)
from PCE_Codes.uqpce import  SurrogateModel
from PCE_Codes._helpers import solve_coeffs, evaluate_points
from PCE_Codes.variables.continuous import UniformVariable


class TestStatistics(unittest.TestCase):

    def test_calc_R_sq(self):
        """
        Design and Analysis of Experiments 8th ed, pg 454, 464
        Douglas C. Montgomery
        """
        tol = 1e-5  # book rounds, so this is a higher value than default

        responses = np.array([
            2256, 2340, 2426, 2293, 2330, 2368, 2250, 2409, 2364, 2379, 2440,
            2364, 2404, 2317, 2309, 2328
        ])

        interval_low = -10
        interval_high = 10
        order = 2

        var_list = [
            UniformVariable(interval_low, interval_high, order=order, number=0),
            UniformVariable(interval_low, interval_high, order=order, number=1)
        ]

        var_basis = np.array([
            [1, 80, 8], [1, 93, 9], [1, 100, 10], [1, 82, 12],
            [1, 90, 11], [1, 99, 8], [1, 81, 8], [1, 96, 10],
            [1, 94, 12], [1, 93, 11], [1, 97, 13], [1, 95, 11],
            [1, 100, 8], [1, 85, 12], [1, 86, 9], [1, 87, 12]
        ])

        for i in range(1, 3):
            var_list[i - 1].std_vals = var_basis[:, i]
            var_list[i - 1].vals = var_basis[:, i]

        matrix_coeffs = solve_coeffs(var_basis, responses)

        R_sq_calc = calc_R_sq(var_basis, matrix_coeffs, responses)
        R_sq_act = 0.92697

        self.assertTrue(
            np.isclose(R_sq_calc, R_sq_act, rtol=0, atol=tol),
            msg='calc_R_sq is not correct'
        )

    def test_calc_R_sq_adj(self):
        """
        Design and Analysis of Experiments 8th ed, pg 454, 464
        Douglas C. Montgomery
        """
        tol = 1e-5  # book rounds, so this is a higher value than default

        responses = np.array([
            2256, 2340, 2426, 2293, 2330, 2368, 2250, 2409, 2364, 2379, 2440,
            2364, 2404, 2317, 2309, 2328
        ])

        interval_low = -10
        interval_high = 10
        order = 2

        var_list = [
            UniformVariable(interval_low, interval_high, order=order, number=0),
            UniformVariable(interval_low, interval_high, order=order, number=1)
        ]

        var_basis = np.array([
            [1, 80, 8], [1, 93, 9], [1, 100, 10], [1, 82, 12],
            [1, 90, 11], [1, 99, 8], [1, 81, 8], [1, 96, 10],
            [1, 94, 12], [1, 93, 11], [1, 97, 13], [1, 95, 11],
            [1, 100, 8], [1, 85, 12], [1, 86, 9], [1, 87, 12]
        ])

        for i in range(1, 3):
            var_list[i - 1].std_vals = var_basis[:, i]
            var_list[i - 1].vals = var_basis[:, i]

        matrix_coeffs = solve_coeffs(var_basis, responses)

        R_sq_adj_calc = calc_R_sq_adj(var_basis, matrix_coeffs, responses)
        R_sq_adj_act = 0.915735

        self.assertTrue(
            np.isclose(R_sq_adj_calc, R_sq_adj_act, rtol=0, atol=tol),
            msg='statistics function calc_R_sq_adj is not correct'
        )

    def test_calc_PRESS_res(self):
        """
        Design and Analysis of Experiments 8th Ed (2013), (pg. 454, Ex. 10.1)
        Douglas C. Montgomery
        """
        x0 = symbols('x0')
        x1 = symbols('x1')

        var_list_symb = [x0, x1]
        var_basis_vect_symb = Matrix([[1, x0, x1]])

        var_basis_func = lambdify(
            (var_list_symb,), var_basis_vect_symb, modules='numpy'
        )

        # region: new samples
        var_list = []
        order = 1

        interval_low = 80
        interval_high = 100
        var_list.append(
            UniformVariable(interval_low, interval_high, order=order, number=0)
        )

        interval_low = 8
        interval_high = 13
        var_list.append(
            UniformVariable(interval_low, interval_high, order=order, number=1)
        )

        var_list[0].std_vals = np.array([
            80, 93, 100, 82, 90, 99, 81, 96, 94, 93, 97, 95, 100, 85, 86, 87
        ])

        var_list[1].std_vals = np.array([
            8, 9, 10, 12, 11, 8, 8, 10, 12, 11, 13, 11, 8, 12, 9, 12
        ])

        act_model_size = len(var_list[0].std_vals)

        new_resps = np.array([
            2256, 2340, 2426, 2293, 2330, 2368, 2250, 2409, 2364, 2379, 2440,
            2364, 2404, 2317, 2309, 2328
        ])

        new_basis = evaluate_points(
            var_basis_func, 0, act_model_size, var_list, 'std_vals'
        )

        true_press = 5207.7
        press = calc_PRESS_res(new_basis, new_resps)

        self.assertAlmostEqual(
            true_press, press, delta=1e-1,
            msg='statistics function calc_PRESS_res is not correct'
        )

    def test_calc_pred_conf_int(self):
        """
        Design and Analysis of Experiments 8th Ed. (pg 469, 258)
        """
        signif = 0.05
        tol = 0.15  # the book example rounds much more than UQOCE

        # Design and Analysis of Experiments 8th ed, pg 469, 258
        responses = np.array([45, 100, 45, 65, 75, 60, 80, 96])
        var_basis = np.array([
            [1, -1, -1, -1, 1, 1],
            [1, 1, -1, 1, -1, 1],
            [1, -1, -1, 1, 1, -1],
            [1, 1, -1, -1, -1, -1],
            [1, -1, 1, 1, -1, -1],
            [1, 1, 1, -1, 1, -1],
            [1, -1, 1, -1, -1, 1],
            [1, 1, 1, 1, 1, 1]
        ])

        basis_eval_ver = np.array([[1, 1, -1, 1, -1, 1]])

        matrix_coeffs = solve_coeffs(var_basis, responses)

        approx_mean, mean_uncert = (
            calc_pred_conf_int(
                var_basis, matrix_coeffs, responses, signif, basis_eval_ver
            )
        )

        act_mean = 100.25
        act_uncert = 10.25

        self.assertEqual(
            approx_mean, act_mean,
            msg='calc_pred_conf_int is calculating the wrong mean'
        )

        self.assertTrue(
            np.isclose(mean_uncert, act_uncert, rtol=0, atol=tol),
            msg='calc_pred_conf_int is calculating the mean uncertainty'
        )

    def test_calc_mean_conf_int(self):
        """
        Applied Statistics and Probability for Engineers 4th Ed. (2007), 
        (pg. 13, 440, 467)
        Douglas C. Montgomery & George C. Runger
        """
        signif = 0.05
        tol = 1e-2  # book rounds to 2 decimal place

        responses = np.array([
            9.95, 24.45, 31.75, 35.0, 25.02, 16.86, 14.38, 9.6, 24.35, 27.5,
            17.08, 37.0, 41.95, 11.66, 21.65, 17.89, 69.0, 10.3, 34.93, 46.59,
            44.88, 54.12, 56.63, 22.13, 21.15
        ])

        var_basis = np.array([
            [1, 2, 50],
            [1, 8, 110],
            [1, 11, 120],
            [1, 10, 550],
            [1, 8, 295],
            [1, 4, 200],
            [1, 2, 375],
            [1, 2, 52],
            [1, 9, 100],
            [1, 8, 300],
            [1, 4, 412],
            [1, 11, 400],
            [1, 12, 500],
            [1, 2, 360],
            [1, 4, 205],
            [1, 4, 400],
            [1, 20, 600],
            [1, 1, 585],
            [1, 10, 540],
            [1, 15, 250],
            [1, 15, 290],
            [1, 16, 510],
            [1, 17, 590],
            [1, 6, 100],
            [1, 5, 400]
        ])

        matrix_coeffs = solve_coeffs(var_basis, responses)

        var_basis_ver = np.array([[1, 8, 275]])

        mean_calc, uncert_calc = calc_mean_conf_int(
            var_basis, matrix_coeffs, responses, signif, var_basis_ver
        )

        mean_act = 27.66
        uncert_act = 1

        self.assertTrue(
            np.isclose(mean_calc, mean_act, rtol=0, atol=tol),
            msg='calc_mean_conf_int is not correct.'
        )

        self.assertTrue(
            np.isclose(uncert_calc, uncert_act, rtol=0, atol=tol),
            msg='calc_mean_conf_int is not correct.'
        )

    def test_calc_coeff_conf_int(self):
        """
        Design and Analysis of Experiments, 8th Ed. (pg 468)
        Douglas Montgomery
        """
        tol = 1e-3  # book rounds, so this is a higher value than default

        # Design and Analysis of Experiments 8th ed, pg 468
        responses = np.array([
            2256, 2340, 2426, 2293, 2330, 2368, 2250, 2409, 2364, 2379, 2440,
            2364, 2404, 2317, 2309, 2328
        ])

        order = 2
        signif = 0.05

        interval_low = -10
        interval_high = 10

        var_list = [
            UniformVariable(interval_low, interval_high, order=order, number=0),
            UniformVariable(interval_low, interval_high, order=order, number=1)
        ]

        var_basis = np.array([
            [1, 80, 8], [1, 93, 9], [1, 100, 10], [1, 82, 12],
            [1, 90, 11], [1, 99, 8], [1, 81, 8], [1, 96, 10],
            [1, 94, 12], [1, 93, 11], [1, 97, 13], [1, 95, 11],
            [1, 100, 8], [1, 85, 12], [1, 86, 9], [1, 87, 12]
        ])

        for i in range(1, 3):
            var_list[i - 1].std_vals = var_basis[:, i]
            var_list[i - 1].vals = var_basis[:, i]

        matrix_coeffs = solve_coeffs(var_basis, responses)

        coeff_uncert = calc_coeff_conf_int(
            var_basis, matrix_coeffs, responses, signif
        )

        calc_coeff = matrix_coeffs[1]
        act_coeff = 7.62129

        calc_low_bound = matrix_coeffs[1] - coeff_uncert[1]
        act_low_bound = 6.2855

        calc_high_bound = matrix_coeffs[1] + coeff_uncert[1]
        act_high_bound = 8.9570

        self.assertTrue(
            np.isclose(calc_coeff, act_coeff, rtol=0, atol=tol),
            msg='calc_coeff_conf_int is not correct.'
        )

        self.assertTrue(
            np.isclose(calc_low_bound, act_low_bound, rtol=0, atol=tol),
            msg='calc_coeff_conf_int is not correct.'
        )

        self.assertTrue(
            np.isclose(calc_high_bound, act_high_bound, rtol=0, atol=tol),
            msg='calc_coeff_conf_int is not correct.'
        )

    def test_calc_var_conf_int(self):
        """
        Design and Analysis of Experiments, 8th Ed. (pg 468)
        Douglas Montgomery
        """
        tol = 5e-3  # round off error in hand calculation

        # Hand calculated using the matrix coefficients minus and plus
        # the coefficient uncertainty for the lower and upper CI, respectively.
        var_conf_int_act = (
            1 / 3 * (6.285) ** 2 + 1 / 3 * (3.316) ** 2,
            1 / 3 * (8.957) ** 2 + 1 / 3 * (13.853) ** 2
        )

        responses = np.array([
            2256, 2340, 2426, 2293, 2330, 2368, 2250, 2409, 2364, 2379, 2440,
            2364, 2404, 2317, 2309, 2328
        ])

        signif = 0.05

        var_basis = np.array([
            [1, 80, 8], [1, 93, 9], [1, 100, 10], [1, 82, 12],
            [1, 90, 11], [1, 99, 8], [1, 81, 8], [1, 96, 10],
            [1, 94, 12], [1, 93, 11], [1, 97, 13], [1, 95, 11],
            [1, 100, 8], [1, 85, 12], [1, 86, 9], [1, 87, 12]
        ])

        norm_sq = np.array([[1], [1 / 3], [1 / 3]])

        matrix_coeffs = solve_coeffs(var_basis, responses)

        coeff_uncert = calc_coeff_conf_int(
            var_basis, matrix_coeffs, responses, signif
        )

        var_conf_int_calc = calc_var_conf_int(
            matrix_coeffs, coeff_uncert, norm_sq
        )

        self.assertTrue(
            (np.abs(np.array(var_conf_int_act) - np.array(var_conf_int_calc)) < tol).all(),
            msg='calc_var_conf_int is not correct.'
        )

    def test_get_sobol_bounds(self):
        """
        First Tests:
        Design and Analysis of Experiments, 8th Ed. (pg 468)
        Douglas Montgomery
        
        Values compared to hand-calculated values from this data.
        
        Second Tests:
        Calculates the smallest and largest of all possible Sobols- ensures 
        that the lowest corresponds to the low and the highest corresponds to 
        the high.
        """
        # Design and Analysis of Experiments 8th ed, pg 468
        responses = np.array([
            2256, 2340, 2426, 2293, 2330, 2368, 2250, 2409, 2364, 2379, 2440,
            2364, 2404, 2317, 2309, 2328
        ])

        signif = 0.05
        tol = 1e-8

        var_basis = np.array([
            [1, 80, 8], [1, 93, 9], [1, 100, 10], [1, 82, 12],
            [1, 90, 11], [1, 99, 8], [1, 81, 8], [1, 96, 10],
            [1, 94, 12], [1, 93, 11], [1, 97, 13], [1, 95, 11],
            [1, 100, 8], [1, 85, 12], [1, 86, 9], [1, 87, 12]
        ])

        # Hand calculated from coefficients and coefficient uncertainty
        sobol_CIL_act = np.array([0.17070561225122574, 0.12055447113345258])
        sobol_CIH_act = np.array([0.8794455288665474, 0.8292943877487742])

        matrix_coeffs = solve_coeffs(var_basis, responses)
        coeff_uncert = calc_coeff_conf_int(
            var_basis, matrix_coeffs, responses, signif
        )

        norm_sq = np.array([[1], [1 / 3], [1 / 3]])
        mod = SurrogateModel(responses, matrix_coeffs)
        mod.calc_var(norm_sq)
        sobols = mod.get_sobols(norm_sq)

        sobol_CIL_calc, sobol_CIH_calc = get_sobol_bounds(
            matrix_coeffs, sobols, coeff_uncert, norm_sq
        )

        self.assertTrue(
            np.isclose(sobol_CIL_calc, sobol_CIL_act, rtol=0, atol=tol).all(),
            msg='get_sobol_bounds is not correct.'
        )

        self.assertTrue(
            np.isclose(sobol_CIH_calc, sobol_CIH_act, rtol=0, atol=tol).all(),
            msg='get_sobol_bounds is not correct.'
        )

        # Sanity check on the methodology- the low Sobol for first variable and
        # high for the second variable should sum to 1 and vice versa.
        self.assertTrue(
            np.isclose(
                np.sum([sobol_CIL_calc[0], sobol_CIH_calc[1]]), 1, rtol=0,
                atol=tol
            ),
            msg='get_sobol_bounds is not correct.'
        )

        self.assertTrue(
            np.isclose(
                np.sum([sobol_CIL_calc[1], sobol_CIH_calc[0]]), 1, rtol=0,
                atol=tol
            ),
            msg='get_sobol_bounds is not correct.'
        )

        # Special case example- one matrix coefficient of 0 with larg
        matrix_coeffs = np.array([1, 2, 0, 1.5])
        sobols = np.array([0.24615385, 0, 0.08307692])
        coeff_uncert = np.array([0.1, 2, 0.1, 0.1])
        norm_sq = np.array([[1], [1 / 3], [1 / 3], [1 / 3]])

        sobol_CIL_calc, sobol_CIH_calc = get_sobol_bounds(
            matrix_coeffs, sobols, coeff_uncert, norm_sq
        )

        # Sobol 0 should have low bound of 0 since the uncertainty can lead to
        # a coefficient of 0 and therefore Sobol of zero.
        # Sobol 0 should have a high bound of > 0.5 since Sobol 1 can be 0 and
        # coeff 2 at its smallest is smaller than coeff 0 at its largest.
        self.assertTrue(
            sobol_CIL_calc[0] == 0, msg='get_sobol_bounds is not correct.'
        )

        self.assertTrue(
            sobol_CIH_calc[0] > 0.5, msg='get_sobol_bounds is not correct.'
        )

        # Sobol 1 should have low bound of 0 since the calculated Sobol is 0.
        # Sobol 1 should have a high bound that is relatively small since the
        # largest magnitude of coeff 1 is 0.1 and smallest coeff 2 is much
        # larger.
        self.assertTrue(
            sobol_CIL_calc[1] == 0, msg='get_sobol_bounds is not correct.'
        )

        self.assertTrue(
            sobol_CIH_calc[1] < 0.1, msg='get_sobol_bounds is not correct.'
        )

        # Sobol 2 should have a low bound that is small but relatively large
        # since the lowest coeff 2 is 1.4, and highest coeff 0 and coeff 1 are
        # 1.1 and 4, respectively.
        # Sobol 2 should have a high bound of 1 since the other Sobols can
        # both be 0.
        self.assertTrue(
            sobol_CIL_calc[2] > 0 and sobol_CIL_calc[2] < 0.2,
            msg='get_sobol_bounds is not correct.'
        )

        self.assertTrue(
            sobol_CIH_calc[2] == 1, msg='get_sobol_bounds is not correct.'
        )

    def test_calc_error_variance(self):
        """
        Design and Analysis of Experiments 8th ed.
        (pg 469, 258)
        Douglas Montgomery
        """
        responses = np.array([45, 100, 45, 65, 75, 60, 80, 96])
        var_basis = np.array([
            [1, -1, -1, -1, 1, 1],
            [1, 1, -1, 1, -1, 1],
            [1, -1, -1, 1, 1, -1],
            [1, 1, -1, -1, -1, -1],
            [1, -1, 1, 1, -1, -1],
            [1, 1, 1, -1, 1, -1],
            [1, -1, 1, -1, -1, 1],
            [1, 1, 1, 1, 1, 1]
        ])

        matrix_coeffs = solve_coeffs(var_basis, responses)

        calc_err_var = calc_error_variance(var_basis, matrix_coeffs, responses)
        act_err_var = 3.25

        self.assertEqual(
            calc_err_var, act_err_var,
            msg='calc_error_variance is not working correctly.'
        )

    def test_calc_error_sum_of_sq(self):
        """
        Applied Statistics and Probability for Engineers 3rd Ed. (2003), online
        (Example 11-3, section 11-5.2)
        Douglas C. Montgomery & George C. Runger
        """
        SS_E_act = 21.25
        tol = 0.2  # small difference due to rounding differences

        responses = np.array([
            90.1, 89.05, 91.43, 93.74, 96.73, 94.45, 87.59, 91.77, 99.42,
            93.65, 93.54, 92.52, 90.56, 89.54, 89.85, 90.39, 93.25, 93.41,
            94.98, 87.33
        ])

        var_basis = np.array([
            [1, 0.99], [1, 1.02], [1, 1.15], [1, 1.29], [1, 1.46], [1, 1.36],
            [1, 0.87], [1, 1.23], [1, 1.55], [1, 1.40], [1, 1.19], [1, 1.15],
            [1, 0.98], [1, 1.01], [1, 1.11], [1, 1.20], [1, 1.26], [1, 1.32],
            [1, 1.43], [1, 0.95]
        ])

        matrix_coeffs = solve_coeffs(var_basis, responses)

        SS_E_calc = calc_error_sum_of_sq(var_basis, matrix_coeffs, responses)

        self.assertAlmostEqual(
            SS_E_act, SS_E_calc, delta=tol,
            msg='calc_error_sum_of_sq is not working correctly.'
        )

    def test_calc_total_sum_of_sq(self):
        """
        Applied Statistics and Probability for Engineers 3rd Ed. (2003), online
        (Example 11-3, section 11-5.2)
        Douglas C. Montgomery & George C. Runger
        """
        SS_T_act = 173.38
        tol = 0.5  # small difference due to rounding differences

        responses = np.array([
            90.1, 89.05, 91.43, 93.74, 96.73, 94.45, 87.59, 91.77, 99.42,
            93.65, 93.54, 92.52, 90.56, 89.54, 89.85, 90.39, 93.25, 93.41,
            94.98, 87.33
        ])

        var_basis = np.array([
            [1, 0.99], [1, 1.02], [1, 1.15], [1, 1.29], [1, 1.46], [1, 1.36],
            [1, 0.87], [1, 1.23], [1, 1.55], [1, 1.40], [1, 1.19], [1, 1.15],
            [1, 0.98], [1, 1.01], [1, 1.11], [1, 1.20], [1, 1.26], [1, 1.32],
            [1, 1.43], [1, 0.95]
        ])

        SS_T_calc = calc_total_sum_of_sq(var_basis, responses)

        self.assertAlmostEqual(
            SS_T_act, SS_T_calc, delta=tol,
            msg='calc_total_sum_of_sq is not working correctly.'
        )

    def test_calc_sum_sq_regr(self):
        """
        Applied Statistics and Probability for Engineers 4th Ed. (2007), 
        (pg. 462) Douglas C. Montgomery & George C. Runger
        """
        SS_R_act = 5990.7712
        tol = 1e-4

        responses = np.array([
            9.95, 24.45, 31.75, 35.0, 25.02, 16.86, 14.38, 9.6, 24.35, 27.5,
            17.08, 37.0, 41.95, 11.66, 21.65, 17.89, 69.0, 10.3, 34.93, 46.59,
            44.88, 54.12, 56.63, 22.13, 21.15
        ])

        var_basis = np.array([
            [1, 2, 50],
            [1, 8, 110],
            [1, 11, 120],
            [1, 10, 550],
            [1, 8, 295],
            [1, 4, 200],
            [1, 2, 375],
            [1, 2, 52],
            [1, 9, 100],
            [1, 8, 300],
            [1, 4, 412],
            [1, 11, 400],
            [1, 12, 500],
            [1, 2, 360],
            [1, 4, 205],
            [1, 4, 400],
            [1, 20, 600],
            [1, 1, 585],
            [1, 10, 540],
            [1, 15, 250],
            [1, 15, 290],
            [1, 16, 510],
            [1, 17, 590],
            [1, 6, 100],
            [1, 5, 400]
        ])

        matrix_coeffs = solve_coeffs(var_basis, responses)
        SS_R_calc = calc_sum_sq_regr(matrix_coeffs, responses, var_basis)

        self.assertAlmostEqual(
            SS_R_act, SS_R_calc, delta=tol,
            msg='calc_sum_sq_regr is not working correctly.'
        )

    def test_calc_mean_sq_err(self):
        """
        Applied Statistics and Probability for Engineers 4th Ed. (2007), (pg. 462)
        Douglas C. Montgomery & George C. Runger
        """
        tol = 1e-3
        MSE_act = 5.2352

        responses = np.array([
            9.95, 24.45, 31.75, 35.0, 25.02, 16.86, 14.38, 9.6, 24.35, 27.5,
            17.08, 37.0, 41.95, 11.66, 21.65, 17.89, 69.0, 10.3, 34.93, 46.59,
            44.88, 54.12, 56.63, 22.13, 21.15
        ])

        var_basis = np.array([
            [1, 2, 50],
            [1, 8, 110],
            [1, 11, 120],
            [1, 10, 550],
            [1, 8, 295],
            [1, 4, 200],
            [1, 2, 375],
            [1, 2, 52],
            [1, 9, 100],
            [1, 8, 300],
            [1, 4, 412],
            [1, 11, 400],
            [1, 12, 500],
            [1, 2, 360],
            [1, 4, 205],
            [1, 4, 400],
            [1, 20, 600],
            [1, 1, 585],
            [1, 10, 540],
            [1, 15, 250],
            [1, 15, 290],
            [1, 16, 510],
            [1, 17, 590],
            [1, 6, 100],
            [1, 5, 400]
        ])

        matrix_coeffs = solve_coeffs(var_basis, responses)

        MSE_calc = calc_mean_sq_err(responses, matrix_coeffs, var_basis)

        self.assertAlmostEqual(
            MSE_act, MSE_calc, delta=tol,
            msg='calc_mean_sq_err is not working correctly.'
        )

    def test_calc_partial_F(self):
        """
        Applied Statistics and Probability for Engineers 4th Ed. (2007), (pg. 462)
        Douglas C. Montgomery & George C. Runger
        """
        deg_free = 2
        f_stat_act = 4.05

        full_SS = 6024.0
        partial_SS = 5990.8
        mean_sq_err_all = 4.1

        f_stat_calc = calc_partial_F(
            full_SS, partial_SS, mean_sq_err_all, deg_free
        )

        self.assertAlmostEqual(
            f_stat_act, f_stat_calc, delta=1e-2,
            msg='calc_partial_F is not working correctly.'
        )

    def test_calc_hat_matrix(self):
        """
        Applied Statistics and Probability for Engineers 3rd Ed. (2003), online
        (Table 12-11, section 12-5.2)
        Douglas C. Montgomery & George C. Runger
        """
        tol = 1e-4

        h_11_act = 0.1573
        h_1313_act = 0.0820
        h_2525_act = 0.0729

        X = np.array([
            [1, 2, 50],
            [1, 8, 110],
            [1, 11, 120],
            [1, 10, 550],
            [1, 8, 295],
            [1, 4, 200],
            [1, 2, 375],
            [1, 2, 52],
            [1, 9, 100],
            [1, 8, 300],
            [1, 4, 412],
            [1, 11, 400],
            [1, 12, 500],
            [1, 2, 360],
            [1, 4, 205],
            [1, 4, 400],
            [1, 20, 600],
            [1, 1, 585],
            [1, 10, 540],
            [1, 15, 250],
            [1, 15, 290],
            [1, 16, 510],
            [1, 17, 590],
            [1, 6, 100],
            [1, 5, 400]
        ])

        hat_matrix = calc_hat_matrix(X)
        h_11_calc = hat_matrix[0, 0]
        h_1313_calc = hat_matrix[12, 12]
        h_2525_calc = hat_matrix[24, 24]

        self.assertAlmostEqual(
            h_11_act, h_11_calc, delta=tol,
            msg='calc_hat_matrix is not working correctly.'
        )

        self.assertAlmostEqual(
            h_1313_act, h_1313_calc, delta=tol,
            msg='calc_hat_matrix is not working correctly.'
        )

        self.assertAlmostEqual(
            h_2525_act, h_2525_calc, delta=tol,
            msg='calc_hat_matrix is not working correctly.'
        )

    def test_calc_term_count(self):
        """
        Testing calc_term_count against a hand calculated answer.
        """
        order = 2
        var_count = 5
        terms_act = 21

        terms_calc = calc_term_count(order, var_count)

        self.assertEqual(
            terms_act, terms_calc,
            msg='calc_term_count is not working correctly.'
        )

    def test_calc_min_responses(self):
        """
        Testing calc_min_responses against a hand calculated answer.
        """
        order = 2
        var_count = 5
        responses_act = 22

        responses_calc = calc_min_responses(order, var_count)

        self.assertEqual(
            responses_act, responses_calc,
            msg='calc_min_responses is not working correctly.'
        )
