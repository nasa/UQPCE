#!/usr/bin/env python
import unittest
import warnings

from test_uqpce.test_uqpce import TestMatrixSystem, TestSurrogateModel
from test_uqpce.test_helpers import TestHelpers
from test_uqpce.test_pbox import TestProbabilityBoxes
from test_uqpce.test_variables import (
    TestVariable, TestUniformVariable, TestNormalVariable, TestBetaVariable,
    TestExponentialVariable, TestGammaVariable
)

if __name__ == '__main__':

    test_list = [
        TestMatrixSystem, TestSurrogateModel, TestHelpers, TestProbabilityBoxes,
        TestVariable, TestUniformVariable, TestNormalVariable, TestBetaVariable,
        TestExponentialVariable, TestGammaVariable
    ]

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for test in test_list:
        tests = loader.loadTestsFromTestCase(test)
        suite.addTests(tests)

    unittest.TextTestRunner(verbosity=2).run(suite)
