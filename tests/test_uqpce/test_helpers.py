import unittest
from io import StringIO
import sys
import copy

from sympy.utilities.lambdify import lambdify
from sympy import symbols, Matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest

from PCE_Codes._helpers import (
    switch_backend, user_function, get_str_vars, create_total_sobols,
    check_directory, evaluate_points_verbose, evaluate_points,
    calc_difference, calc_mean_err, uniform_hypercube, solve_coeffs,
    generate_sample_set, unstandardize_set, standardize_set, check_error_trends,
    check_error_magnitude, _warn, calc_sobols
)

from PCE_Codes.variables.variable import Variable
from PCE_Codes.variables.continuous import (
    UniformVariable, GammaVariable, ExponentialVariable, BetaVariable
)
from PCE_Codes.variables.discrete import (
    UniformVariable as DiscUniformVariable, NegativeBinomialVariable
)


class TestHelpers(unittest.TestCase):

    def test_switch_backend(self):
        """
        Test that the backend is set to the new backend. This is a wrapper for 
        the MatPlotLib option and does not need testing.
        """
        pass

    def test_user_function(self):
        """
        Testing to ensure that sample generation is correct.
        """
        tol = 1e-8
        func = f'(((x0 ** 2) * (1 + x1**1.5)) + exp(x2**0.1))'

        var_list = [Variable(number=0), Variable(number=1), Variable(number=2)]
        var_list[0].vals = np.array([1, 2, 3, 4, 5])
        var_list[1].vals = np.array([2, 4, 6, 8, 10])
        var_list[2].vals = np.array([3, 6, 9, 12, 15])

        response_act = np.array([
            6.88142242, 39.30762761, 144.74792034, 381.6428324, 819.27936884
        ])
        response_calc = user_function(func, var_list)['generated_responses']

        self.assertTrue(
            np.isclose(response_act, response_calc, rtol=0, atol=tol).all(),
            msg='statistics function calc_R_sq_adj is not correct'
        )

    def test_get_str_vars(self):
        """
        Testing to ensure that the string representations of the terms are 
        correct.
        """
        var_str_act = [
            'x0', 'x1', 'x2', 'x0^2', 'x0*x1', 'x0*x2', 'x1^2', 'x1*x2', 'x2^2'
        ]

        matrix = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 2, 0],
            [0, 1, 1],
            [0, 0, 2]
        ])
        var_str_calc = get_str_vars(matrix)

        self.assertEqual(
            var_str_calc, var_str_act,
            msg='statistics function get_str_vars is not correct'
        )

    def test_create_total_sobols(self):
        """
        Testing total Sobol calculation to ensure that calculation is correct.
        """
        # Sobols tested against results from analytical test case hand calculated

        matrix = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [2, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 2, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 2, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 2]
        ])

        var_count = 5

        sobols = np.array([
            0.33712224, 0.02809352, 0.14983211, 0.10959148, 0.33712224, 0,
            0.00078038, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03745803
        ])
        sobols_T_act = np.array([
            0.33712224 + 0.00078038, 0.02809352 + 0.00078038, 0.14983211,
            0.10959148, 0.33712224 + 0.03745803
        ])
        sobols_T_calc = create_total_sobols(var_count, matrix, sobols)

        self.assertTrue(
            (sobols_T_act == sobols_T_calc).all(),
            msg='statistics function create_total_sobols is not correct'
        )

    def test_check_directory(self):
        """
        Skipping this test for now as to not create and remove directories in 
        the user's directory. This has been tested to work in the verification 
        cases.
        """
        pass

    def test_evaluate_points_verbose(self):
        """
        Testing evaluate_points_verbose to ensure that points are evaluated 
        correctly.
        """
        low = -1
        high = 1
        thresh = 1e-8
        var_list = [
            UniformVariable(low, high, number=0),
            UniformVariable(low, high, number=1),
            UniformVariable(low, high, number=2)
        ]

        var_list[0].stdvals = np.array([
            0.81772598, -0.06381505, 0.14684707, 0.1240739 , 0.9744819 ,
            -0.27647688, -0.9233829 , -0.67351042, -0.09180627, 0.95404397
        ])
        var_list[1].stdvals = np.array([
            -0.2753504 , 0.5171086 , -0.97586483, -0.30631548, 0.50056239,
            -0.02663459, 0.23837264, 0.23206214, 0.99304867, -0.83170756
        ])
        var_list[2].stdvals = np.array([
            0.18747112, -0.31858939, 0.13468345, -0.97464967, -0.73878923,
            -0.68208937, -0.72570261, 0.06601919, 0.34919762, -0.64923021
        ])

        x0 = symbols('x0')
        x1 = symbols('x1')
        x2 = symbols('x2')

        var_list_symb = np.array([x0, x1, x2])
        var_basis_vect_symb = Matrix([[x0 * x1 * x2 + 3], [x0 ** 6 * x1 ** 3 / x2 ** 3]]).T

        func = lambdify(
            (var_list_symb,), var_basis_vect_symb, modules='numpy'
        )

        # Redirect output so the text from this method is not printed during the
        # unittest.
        prev_stdout = sys.stdout
        sys.stdout = StringIO()

        eval_pnts_calc = evaluate_points_verbose(
            func, 0, 10, var_list, 'stdvals'
        )

        # Reset the stdout to original output.
        sys.stdout = prev_stdout

        eval_pnts_act = np.array([
            [2.95778878e+00, -9.47328905e-01],
            [3.01051323e+00, -2.88795246e-07],
            [2.98069947e+00, -3.81432782e-03],
            [3.03704230e+00, 1.13251462e-07],
            [2.63962675e+00, -2.66351803e-01],
            [2.99497720e+00, 2.65929871e-08],
            [3.15973384e+00, -2.19676768e-02],
            [2.98968145e+00, 4.05385508e+00],
            [2.96816432e+00, 1.37699332e-05],
            [3.51515481e+00, 1.58535356e+00]
        ])

        self.assertTrue(
            (np.abs(eval_pnts_calc - eval_pnts_act) < thresh).all(),
            msg='statistics function evaluate_points_verbose is not correct'
        )

    def test_evaluate_points(self):
        """
        Testing evaluate_points to ensure that points are evaluated 
        correctly.
        """
        low = -1
        high = 1
        thresh = 1e-8
        var_list = [
            UniformVariable(low, high, number=0),
            UniformVariable(low, high, number=1),
            UniformVariable(low, high, number=2)
        ]

        var_list[0].stdvals = np.array([
            0.81772598, -0.06381505, 0.14684707, 0.1240739 , 0.9744819 ,
            -0.27647688, -0.9233829 , -0.67351042, -0.09180627, 0.95404397
        ])
        var_list[1].stdvals = np.array([
            -0.2753504 , 0.5171086 , -0.97586483, -0.30631548, 0.50056239,
            -0.02663459, 0.23837264, 0.23206214, 0.99304867, -0.83170756
        ])
        var_list[2].stdvals = np.array([
            0.18747112, -0.31858939, 0.13468345, -0.97464967, -0.73878923,
            -0.68208937, -0.72570261, 0.06601919, 0.34919762, -0.64923021
        ])

        x0 = symbols('x0')
        x1 = symbols('x1')
        x2 = symbols('x2')

        var_list_symb = np.array([x0, x1, x2])
        var_basis_vect_symb = Matrix([[x0 * x1 * x2 + 3], [x0 ** 6 * x1 ** 3 / x2 ** 3]]).T

        func = lambdify(
            (var_list_symb,), var_basis_vect_symb, modules='numpy'
        )

        eval_pnts_calc = evaluate_points(func, 0, 10, var_list, 'stdvals')

        eval_pnts_act = np.array([
            [2.95778878e+00, -9.47328905e-01],
            [3.01051323e+00, -2.88795246e-07],
            [2.98069947e+00, -3.81432782e-03],
            [3.03704230e+00, 1.13251462e-07],
            [2.63962675e+00, -2.66351803e-01],
            [2.99497720e+00, 2.65929871e-08],
            [3.15973384e+00, -2.19676768e-02],
            [2.98968145e+00, 4.05385508e+00],
            [2.96816432e+00, 1.37699332e-05],
            [3.51515481e+00, 1.58535356e+00]
        ])

        self.assertTrue(
            (np.abs(eval_pnts_calc - eval_pnts_act) < thresh).all(),
            msg='statistics function evaluate_points is not correct'
        )

    def test_calc_difference(self):
        """
        Testing calc_difference to ensure that point differences are correct.
        """
        arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        arr2 = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])

        diff_act = np.ones(10) * 3

        diff_calc = calc_difference(arr1, arr2)

        self.assertTrue(
            (diff_act == diff_calc).all(),
            msg='statistics function calc_difference is not correct'
        )

    def test_calc_mean_err(self):
        """
        Testing calc_mean_err to ensure that mean error is correct.
        """
        error = np.array([-1, -1.75, 2, 5.2, -3, 1.3])
        mean_err_act = 2.375

        mean_err_calc = calc_mean_err(error)

        self.assertAlmostEqual(
            mean_err_act, mean_err_calc, delta=1e-4,
            msg='statistics function calc_mean_err is not correct'
        )

    def test_uniform_hypercube(self):
        """
        Testing uniform_hypercube to ensure points are from a standard uniform 
        distribution and the points are spaced as they should be.
        """
        low = 0
        high = 1
        cnt = 10

        within_rng_calc = np.zeros(cnt)
        within_rng_act = np.array([20, 18, 16, 14, 12, 10, 8, 6, 4, 2])

        samps = uniform_hypercube(low, high, samp_size=cnt)
        samps = np.atleast_2d(samps)
        intervals = np.linspace(low, high, cnt + 1)

        biggest_diffs = np.abs(intervals - samps.T)
        for i in range(cnt):
            within_rng_calc[i] = len(biggest_diffs[(biggest_diffs > i * 0.1) * (biggest_diffs < (i + 1) * 0.1)])

        # There should be 20 values within the first range since they're spaced
        # out evenly in 1/10th intervals, then 18 for the next due to edge
        # cases, and so on.
        self.assertTrue(
            (within_rng_calc == within_rng_act).all(),
            msg='statistics function uniform_hypercube is not correct'
        )

    def test_solve_coeffs(self):
        """
        Applied Statistics and Probability for Engineers 3rd Ed. online (2003), 
        (section 12-1.2)
        Douglas C. Montgomery & George C. Runger
        """
        tol = 1e-5

        responses = np.array(
            [9.95, 24.45, 31.75, 35.0, 25.02, 16.86, 14.38, 9.6, 24.35, 27.5,
             17.08, 37.0, 41.95, 11.66, 21.65, 17.89, 69.0, 10.3, 34.93, 46.59,
             44.88, 54.12, 56.63, 22.13, 21.15]
        )

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

        coeffs_act = np.array([2.26379, 2.74427, 0.01253])
        coeffs_calc = solve_coeffs(var_basis, responses)

        self.assertTrue(
            (np.abs(coeffs_calc - coeffs_act) < tol).all(),
            msg='statistics function solve_coeffs is not correct'
        )

    def test_generate_sample_set(self):
        """
        Testing generate_sample_set to ensure points are generated correctly.
        """
        cnt = 50

        p_val_min = 0.05
        ks_stat_max = 0.2

        lows = np.array([-1, 0, -2, 10])
        highs = np.array([1, 12, 2, 11])

        var_list = [
            UniformVariable(lows[0], highs[0], number=0),
            UniformVariable(lows[1], highs[1], number=1),
            UniformVariable(lows[2], highs[2], number=2),
            UniformVariable(lows[3], highs[3], number=3)
        ]

        var_cnt = len(var_list)

        ks_stat = np.zeros(var_cnt)
        p_val = np.zeros(var_cnt)

        set_crt = generate_sample_set(var_list, sample_count=cnt)

        # Verify that standardized points are generated for each variable- since
        # the 'genreate_samples' is tested for every variable, only one variable
        # type needs to be used to ensure this method is working.

        scale = 2
        args = (-1, scale)
        for i in range(var_cnt):

            ks_stat[i], p_val[i] = kstest(
                set_crt[i, :], 'uniform', args=args
            )

        self.assertTrue(
            (ks_stat < ks_stat_max).all(),
            msg='statistics function generate_sample_set is not correct'
        )

        self.assertTrue(
            (p_val > p_val_min).all(),
            msg='statistics function generate_sample_set is not correct'
        )

    def test_unstandardize_set(self):
        """
        Testing unstandardize_set to ensure points are unstandardized correctly.
        """
        var_list = [
            GammaVariable(3, 6, interval_low=-5),
            ExponentialVariable(7, interval_low=12)
        ]

        sample_array = np.array([
            [2.38724789, 7.4998926 , 1.77382463, 1.20950467, 3.25669586],
            [0.06357667, 0.3530339 , 0.08183875, 0.14562894, 0.01910057]
        ])

        unstand_act = np.array([
            [9.32348731, 39.99935561, 5.64294779, 2.25702801, 14.54017514],
            [12.06357667, 12.3530339 , 12.08183875, 12.14562894, 12.01910057]
        ])

        unstand_calc = unstandardize_set(var_list, sample_array)

        self.assertTrue(
            np.isclose(unstand_act, unstand_calc).all(),
            msg='statistics function unstandardize_set is not correct'
        )

    def test_standardize_set(self):
        """
        Testing standardize_set to ensure points are standardized correctly.
        """
        var_list = [
            GammaVariable(3, 6, interval_low=-5),
            ExponentialVariable(7, interval_low=12)
        ]
        sample_array = np.array([
            [9.32348731, 39.99935561, 5.64294779, 2.25702801, 14.54017514],
            [12.06357667, 12.3530339 , 12.08183875, 12.14562894, 12.01910057]
        ])

        stand_act = np.array([
            [2.38724789, 7.4998926 , 1.77382463, 1.20950467, 3.25669586],
            [0.06357667, 0.3530339 , 0.08183875, 0.14562894, 0.01910057]
        ])

        stand_calc = standardize_set(var_list, sample_array)

        self.assertTrue(
            np.isclose(stand_act, stand_calc).all(),
            msg='statistics function standardize_set is not correct'
        )

    def test_check_error_trends(self):
        """
        Testing check_error_trends to ensure most trends in error will be 
        output by this function.
        """
        order = 3

        var_list = [
            DiscUniformVariable(-2, 5, number=0), BetaVariable(2, 3, number=1),
            NegativeBinomialVariable(2, 0.001, number=2)
        ]

        var_list[0].std_vals = np.array([
            0.71428571, 1, -0.42857143, -0.14285714, 1, -0.42857143, 0.14285714,
            0.71428571, -0.14285714, 0.42857143, 0.42857143, -1, 0.42857143,
            0.14285714, -1, 0.71428571, -0.71428571, -1, -0.71428571, -0.14285714
        ])
        var_list[1].std_vals = np.array([
            0.5462244 , 0.20809593, 0.23652854, 0.35380315, 0.51932061,
            0.30080875, 0.75616288, 0.44329766, 0.39408217, 0.46692401,
            0.3174672 , 0.25156626, 0.14046451, 0.68685308, 0.15933271,
            0.49358348, 0.37818204, 0.61942014, 0.08322559, 0.63548133
        ])
        var_list[2].std_vals = np.array([
            1537, 531, 2144, 2295, 4737, 827, 786, 2823, 1826, 1448, 1222,
            3584, 1028, 279, 1374, 3342, 1984, 7028, 2444, 440
        ])

        # Force the error to be a function of the standardized values
        error = (
            1e-3 * var_list[0].std_vals ** 2
            -1e-2 * var_list[1].std_vals ** 5
            -1e-8 * var_list[2].std_vals  # very small error trend
        )

        problem_vars_act = ['x0', 'x1']
        problem_vars_calc = check_error_trends(var_list, error, order)

        self.assertTrue(
            problem_vars_act == problem_vars_calc,
            msg='statistics function check_error_trends is not correct'
        )

    def test_check_error_magnitude(self):
        """
        Testing check_error_magnitude to ensure that error with large outliers 
        will be flagged.
        """
        error = np.array([0.01, 0.2, -0.015, 0.1, -0.0005, -0.002, 0.03, 4])
        err_mag_high = check_error_magnitude(error)

        error = np.array([-2.1, -1.1, 0.1, 1.1, 2.1])
        err_mag_none = check_error_magnitude(error)

        self.assertTrue(
            ('has large outliers' in err_mag_high),
            msg='statistics function check_error_magnitude is not correct'
        )

        self.assertTrue(
            ('no error outliers' in err_mag_none),
            msg='statistics function check_error_magnitude is not correct'
        )

    def test__warn(self):
        """
        Skipping this test for now as to not raise false warnings in unittest. 
        """
        pass

    def test_calc_sobols(self):
        """
        Test case created from three normal variables using equation 
        x0 + x1 + x2 + 0.5*x0**2 + 0.5*x2**2 .
        
        This ensures that Sobols 0 and 2 and Sobols 3 and 8 are the same and 
        that Sobol 1 is smaller than 0 and 2 since some of the higher order will
        be captured by the lower order.
        """

        matrix_coeffs = np.array([
            4.25, 1.0, 0.5, 1.0, 0.125, 0, 0, 0, 0, 0.125
        ])

        norm_sq = np.array([[1], [1], [1], [1], [2], [1], [1], [2], [1], [2]])

        sobols_act = np.array([
            0.43243, 0.10811, 0.43243, 0.013514, 0, 0, 0, 0, 0.013514
        ])

        sobols_calc = calc_sobols(matrix_coeffs, norm_sq)

        self.assertTrue(
            np.isclose(sobols_act, sobols_calc, rtol=0, atol=1e-5).all(),
            msg='statistics function calc_sobols is not correct'
        )
