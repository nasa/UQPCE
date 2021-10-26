import unittest
import numpy as np

import test_uqpce

if __name__ == '__main__':

    np.random.seed(33)

    suite = unittest.TestSuite(test_uqpce)
    unittest.main()
