"""Regression tests for all-zero vector and multi-response coefficient solves."""

import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from uqpce import PCE
from uqpce.pce._helpers import solve_coeffs
from uqpce.pce.model import MatrixSystem


class TestZeroResponses(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(-1.0, 1.0, 5)
        self.basis = np.column_stack((np.ones(5), self.x))

    def test_zero_response_matrix_keeps_every_response_column(self):
        for count in (1, 2, 3, 5):
            for dtype in (np.int64, np.float32, np.float64):
                with self.subTest(count=count, dtype=dtype):
                    responses = np.zeros((5, count), dtype=dtype)
                    original_basis = self.basis.copy()
                    coeffs = solve_coeffs(self.basis, responses)
                    self.assertEqual(coeffs.shape, (2, count))
                    assert_array_equal(coeffs, np.zeros((2, count)))
                    assert_array_equal(self.basis @ coeffs, responses)
                    assert_array_equal(responses, np.zeros((5, count)))
                    assert_array_equal(self.basis, original_basis)

    def test_one_dimensional_zero_response_retains_vector_result(self):
        result = solve_coeffs(self.basis, np.zeros(5))
        self.assertEqual(result.shape, (2,))
        assert_array_equal(result, [0.0, 0.0])

    def test_zero_shortcut_still_handles_a_singular_basis(self):
        basis = np.ones((5, 3))
        for shape in ((5,), (5, 2)):
            with self.subTest(shape=shape):
                result = solve_coeffs(basis, np.zeros(shape))
                self.assertEqual(result.shape, (3,) + shape[1:])
                assert_array_equal(result, np.zeros_like(result))

    def test_nonzero_and_mixed_response_solutions_are_unchanged(self):
        for coefficients in (
            np.array([2.0, 3.0]),
            np.array([[2.0], [3.0]]),
            np.array([[0.0, 2.0, -1.0], [0.0, 3.0, 4.0]]),
        ):
            with self.subTest(shape=coefficients.shape):
                responses = self.basis @ coefficients
                assert_allclose(solve_coeffs(self.basis, responses), coefficients)

    def test_matrix_system_solve_keeps_zero_response_columns(self):
        for count in (1, 2, 4):
            with self.subTest(count=count):
                system = MatrixSystem(np.zeros((5, count)), [])
                system.var_basis_sys_eval = self.basis.copy()
                coefficients = system.solve()
                self.assertIs(system.matrix_coeffs, coefficients)
                self.assertEqual(coefficients.shape, (2, count))
                assert_array_equal(coefficients, np.zeros((2, count)))

    def test_public_fit_and_predict_accept_multiple_zero_responses(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                model = PCE(order=1, outputs=False, verbose=False)
                model.add_variable(
                    distribution="uniform",
                    interval_low=-1,
                    interval_high=1,
                    type="aleatory",
                )
                # Constant responses have undefined signal-to-noise diagnostics.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    coefficients = model.fit(
                        self.x.reshape(-1, 1), np.zeros((5, count))
                    )
                    predictions = model.predict(np.array([[-0.5], [0.5]]))
                self.assertEqual(coefficients.shape, (2, count))
                self.assertEqual(predictions.shape, (2, count))
                assert_array_equal(coefficients, np.zeros((2, count)))
                assert_array_equal(predictions, np.zeros((2, count)))


if __name__ == "__main__":
    unittest.main()
