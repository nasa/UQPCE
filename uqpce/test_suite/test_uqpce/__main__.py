import unittest
import numpy as np

from test_uqpce import *
from test_continuous_variables import *
from test_discrete_variable import *
from test_helpers import *
from test_pbox import *
from test_pce import *
from test_statistics import *

if __name__ == '__main__':

    np.random.seed(33)

    unittest.main()
