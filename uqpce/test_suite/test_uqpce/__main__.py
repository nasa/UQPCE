import unittest
import numpy as np


def suite():
    tests = unittest.TestSuite()
    tests.addTest('test_uqpce')

if __name__ == '__main__':

    np.random.seed(33)

    runner = unittest.TextTestRunner()
    runner.run(suite())
    unittest.main()
