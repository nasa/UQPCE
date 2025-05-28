import unittest

import numpy as np
from sympy import Matrix, symbols, Eq, N, expand, sympify

from uqpce.pce.model import MatrixSystem, SurrogateModel
from uqpce.pce.variables.continuous import (
    UniformVariable, NormalVariable, BetaVariable, ExponentialVariable,
    GammaVariable
)

order = 2

# region: normal variable
mean = 1
stdev = 0.5
norm_var = NormalVariable(mean, stdev, order=order, number=0)
norm_var.vals = np.array([
    0.1508911 , 1.13232735, 1.04524425, 0.36369869, 0.64002868,
    0.74664909, 1.34666796, 0.72736668, 0.42280317, 1.15711335,
    -0.19216545, 1.00792105, 0.78257629, 0.90743312, 1.53291596,
    0.87178893, 0.70325221, 1.7908781 , 1.57110079, 0.58397168,
    0.28099855, 1.05482927, 1.40968597, 1.60103053, 1.76458409,
    1.18931793, 1.67683969, 0.97636684, 0.09914773, 1.12167674,
    0.95950223, 0.57282726, 2.09024103, 1.26232112, 1.87918827,
    1.09499841, 1.4465404 , 1.31132882, 1.20667329, 0.89441882,
    0.5049925 , 0.65833404, 0.31867009, 0.84161706, 1.35815508,
    0.49771936, 0.79380287, 1.47710471, 0.94029816, 1.24945768
])
norm_var.std_vals = norm_var.standardize('vals', 'std_vals')
norm_var.std_verify_vals = norm_var.standardize_points(np.array([1.029939809893880]))
# endregion: normal variable

# region: uniform variable
interval_low = 1.75
interval_high = 2.25
unif_var = UniformVariable(interval_low, interval_high, order=order, number=1)
unif_var.vals = np.array([
    2.24175203, 1.86126492, 1.76806408, 1.99217273, 2.11340606,
    2.21447413, 1.78640677, 1.88201559, 1.85798326, 2.23527088,
    2.16077857, 2.17751665, 2.20046952, 2.05883816, 2.2450643 ,
    2.07235355, 1.95016395, 2.12835819, 1.88891429, 1.94867102,
    1.8546233 , 1.92380071, 2.17457973, 1.81447488, 2.13691741,
    1.84828762, 1.75680435, 1.94961961, 1.8404631 , 1.94830901,
    1.99328695, 1.78768032, 1.90646027, 2.17054326, 1.95000092,
    1.79736887, 2.09034209, 2.09450132, 1.77030467, 1.97063215,
    2.00789363, 1.79097481, 2.24809957, 2.13432934, 1.89638325,
    1.76479892, 2.17388118, 2.10339739, 2.0568413 , 1.92051619
])
unif_var.std_vals = unif_var.standardize('vals', 'std_vals')
unif_var.std_verify_vals = unif_var.standardize_points(np.array([2.103023044009805]))
# endregion: uniform variable

# region: exponential variable
lambd = 3
exp_var = ExponentialVariable(lambd, order=order, number=2)
exp_var.vals = np.array([
    8.69118285e-01, 4.21465457e-01, 7.15743745e-02, 1.51038972e-01,
    3.44761746e-02, 1.76662640e-01, 1.52918571e-01, 2.82762034e-03,
    4.08930330e-02, 4.43610452e-01, 4.21321897e-01, 6.59380913e-02,
    3.89902887e-01, 2.00896646e-01, 2.20632950e-01, 2.07852564e-01,
    5.36888164e-02, 2.59113224e-01, 1.03759066e-03, 1.00716938e-01,
    2.31218307e-01, 9.81283465e-03, 1.04617519e-01, 2.27672062e-02,
    1.61394302e+00, 1.72882144e-01, 5.85475226e-01, 5.72199450e-02,
    5.02091035e-01, 1.13525010e+00, 1.88280783e-01, 3.85514605e-02,
    4.46187126e-02, 1.07683214e-01, 1.58287990e-01, 2.42237977e-01,
    2.17616520e-02, 3.78629099e-01, 4.04839096e-01, 9.03614172e-03,
    6.56655700e-02, 5.87966571e-01, 6.30254268e-01, 3.63501358e-01,
    2.15586609e-01, 2.84195903e-01, 3.14840242e-01, 2.05524179e-02,
    1.56740233e-01, 5.21499683e-01
])
exp_var.std_vals = exp_var.standardize('vals', 'std_vals')
exp_var.std_verify_vals = exp_var.standardize_points(np.array([0.237943609477604]))
# endregion: exponential variable

# region: beta variable
alpha = 0.5
beta = 2
beta_var = BetaVariable(alpha, beta, order=order, number=3)
beta_var.vals = np.array([
    8.29187303e-01, 8.15638554e-02, 1.26223950e-01, 5.97955199e-01,
    4.85083543e-01, 3.38422586e-01, 5.01089814e-02, 2.48329650e-01,
    4.52392180e-02, 4.92172781e-01, 3.95519339e-01, 8.98815195e-02,
    2.34759425e-01, 6.32185551e-03, 1.07133256e-01, 1.28939374e-02,
    7.46551126e-02, 7.25738158e-01, 4.73889026e-03, 1.70018862e-01,
    1.97882157e-01, 9.92888484e-03, 1.41032565e-01, 3.08622785e-01,
    1.76062678e-02, 2.53441064e-03, 1.68339060e-01, 5.63035836e-01,
    7.70790728e-04, 6.76278623e-01, 2.94123939e-01, 1.02961551e-01,
    3.06452443e-02, 4.13467481e-01, 3.44726724e-02, 4.89997326e-02,
    1.33674822e-01, 2.26050969e-01, 4.39001432e-01, 3.01172820e-02,
    3.03288932e-01, 5.09956573e-02, 2.75668137e-01, 1.90131145e-02,
    2.44168677e-01, 4.44050662e-02, 1.61453380e-01, 2.00161749e-01,
    5.09092200e-03, 3.75763378e-04
])
beta_var.std_vals = beta_var.standardize('vals', 'std_vals')
beta_var.std_verify_vals = beta_var.standardize_points(np.array([0.072181544120407]))
# endregion: beta variable

# region: gamma variable
alpha = 1.0
theta = 0.5
gamma_var = GammaVariable(alpha, theta, order=order, number=4)
gamma_var.vals = np.array([
    0.73182462, 0.34443097, 0.05152389, 0.33232168, 0.22579406,
    2.30108506, 0.74698967, 0.32317909, 0.07897738, 1.31743861,
    0.33570981, 0.794461  , 0.10650176, 0.36967217, 2.56562697,
    0.12170762, 0.22378507, 0.6857827 , 1.20483871, 0.19184017,
    2.43441573, 0.59873763, 0.98615485, 0.74329151, 0.18088927,
    1.05630248, 0.07441607, 0.04554521, 0.39627944, 0.43939487,
    0.11383968, 1.93638897, 0.2255472 , 0.20216123, 0.07498395,
    0.22432336, 0.12857658, 0.18502862, 0.87443948, 0.1296126 ,
    0.51159338, 0.12728902, 0.12229771, 1.81307878, 0.93562495,
    0.10540639, 1.42158734, 0.17741185, 0.86773494, 0.4880535
])
gamma_var.std_vals = gamma_var.standardize('vals', 'std_vals')
gamma_var.std_verify_vals = gamma_var.standardize_points(np.array([0.225540215280446]))
# endregion: gamma variable

# region: responses
responses = np.atleast_2d(np.array([
    12.16415154, 9.01613385, 7.20322875, 8.36029185, 8.540021,
    16.30878819, 9.28935313, 7.28973858, 5.31318168, 14.56672245,
    7.00861861, 9.54068883, 9.13214135, 8.02767443, 19.70304007,
    7.63480473, 6.70858894, 14.69559745, 10.99514746, 6.83015151,
    14.35737085, 7.95809045, 11.66095904, 10.73357651, 14.93505371,
    9.78495628, 10.64606339, 9.25057275, 6.02522479, 13.94699517,
    8.72677666, 11.38571172, 10.4209789 , 10.56328017, 10.13697017,
    7.85520941, 9.3960816 , 10.5062668 , 11.4930689 , 6.89224321,
    7.71822458, 7.5307172 , 8.66196112, 13.17345374, 11.16095766,
    6.05241665, 11.90003052, 9.86202873, 9.09930809, 9.78234558
])).T
# endregion: responses
var_list = [norm_var, unif_var, exp_var, beta_var, gamma_var]
matrix = MatrixSystem(responses, var_list)
min_model_size, model_matrix = matrix.create_model_matrix()
norm_sq = matrix.form_norm_sq(order)
var_basis_vect_symb = matrix.build()

X = np.zeros([len(var_list), len(responses)])
Xv = np.zeros([len(var_list), len(responses)])
for i in range(len(var_list)):
    X[i,:] = var_list[i].std_vals
    Xv[i,:] = var_list[i].std_verify_vals

var_basis_sys_eval = matrix.evaluate(X.T)
matrix_coeffs = matrix.solve()

idx = 3
err_ver, mean_err, mean, var = (
    matrix._build_alt_model(
        responses, var_basis_sys_eval, norm_sq, idx
    )
)
press_stats = matrix.get_press_stats()

sig_combo = np.array([0, 1, 2, 3, 4, 5, 7, 20])

var_basis_vect_symb_upd, norm_sq_upd, model_matrix_upd, min_model_size_upd, var_basis_sys_eval_upd = (
    matrix.update(sig_combo)
)


class TestMatrixSystem(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)

        self.tol = 10

        self.matrix = matrix
        self.var_list = var_list
        self.order = order
        self.idx = idx

        self.min_model_size = min_model_size
        self.model_matrix = model_matrix
        self.norm_sq = norm_sq
        self.var_basis_vect_symb = var_basis_vect_symb
        self.var_basis_sys_eval = var_basis_sys_eval
        self.matrix_coeffs = matrix_coeffs

        self.err_ver = err_ver
        self.mean_err = mean_err
        self.mean = mean
        self.var = var
        self.press_stats = press_stats

        self.var_basis_vect_symb_upd = var_basis_vect_symb_upd
        self.norm_sq_upd = norm_sq_upd
        self.model_matrix_upd = model_matrix_upd
        self.min_model_size_upd = min_model_size_upd

        self.sig_combo = sig_combo

    def test_create_model_matrix(self):
        """
        Testing create_model_matrix for a system wtih our 5 common variable types.

        This example is from a well-tested test case.
        """

        # region: model matrix
        true_model_matrix = np.array([
            [0., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.],
            [2., 0., 0., 0., 0.], [1., 1., 0., 0., 0.], [1., 0., 1., 0., 0.],
            [1., 0., 0., 1., 0.], [1., 0., 0., 0., 1.], [0., 2., 0., 0., 0.],
            [0., 1., 1., 0., 0.], [0., 1., 0., 1., 0.], [0., 1., 0., 0., 1.],
            [0., 0., 2., 0., 0.], [0., 0., 1., 1., 0.], [0., 0., 1., 0., 1.],
            [0., 0., 0., 2., 0.], [0., 0., 0., 1., 1.], [0., 0., 0., 0., 2.],
        ])
        # endregion: model matrix

        self.assertTrue(
            (self.model_matrix == true_model_matrix).all(),
            msg='MatrixSystem create_model_matrix is not correct'
        )

    def test_form_norm_sq(self):
        """
        Testing form_norm_sq for a system wtih our 5 common variable types.

        This example is from a well-tested test case.
        """

        # region: norm squared
        true_norm_sq = np.array([
            [1],
            [1],
            [1 / 3],
            [1 / 9],
            [16 / 350],
            [1],
            [2],
            [1 / 3],
            [1 / 9],
            [16 / 350],
            [1],
            [0.2],
            [(1 / 3) * (1 / 9)],
            [(1 / 3) * (16 / 350)],
            [1 / 3],
            [4 / 81],
            [(1 / 9) * (16 / 350)],
            [1 / 9],
            [64 / 24255],
            [16 / 350],
            [4]
        ])

        # endregion: norm squared

        self.assertTrue(
            (np.isclose(true_norm_sq, self.norm_sq, rtol=0, atol=1e-6)).all(),
            msg='MatrixSystem form_norm_sq is not correct'
        )

    def test_build(self):
        """
        Testing form_norm_sq for a system wtih our 5 common variable types.

        This example is from a well-tested test case.
        """
        x0 = symbols('x0')
        x1 = symbols('x1')
        x2 = symbols('x2')
        x3 = symbols('x3')
        x4 = symbols('x4')

        # region: symbolic basis
        true_var_basis_vect_symb = Matrix([[
            1, x0, x1, x2 - (1 / 3), x3 - 0.2, x4 - 1.0, x0 ** 2 - 1.0,
            x0 * x1, x0 * (x2 - (1 / 3)), x0 * (x3 - 0.2), x0 * (x4 - 1),
            1.5 * x1 ** 2 - 0.5, x1 * (x2 - (1 / 3)), x1 * (x3 - 0.2),
            x1 * (x4 - 1), x2 ** 2 - (4 / 3) * x2 + (2 / 9),
            (1 * x2 - (1 / 3)) * (x3 - 0.2), (1 * x2 - (1 / 3)) * (x4 - 1),
            x3 ** 2 - (2 / 3) * x3 + (1 / 21),
            (1 * x3 - 0.2) * (x4 - 1), x4 ** 2 - 4 * x4 + 2
        ]])

        # endregion: symbolic basis

        basis_size = len(true_var_basis_vect_symb)

        equal = [str(Eq(N(sympify(expand(true_var_basis_vect_symb[i])), self.tol), N(sympify(expand(self.var_basis_vect_symb[i])), self.tol))) for i in range(basis_size)]

        eval_loc = locals().copy()
        eval_glob = globals().copy()

        evaled = np.array([eval(equal[i], eval_loc, eval_glob) for i in range(len(equal))])

        self.assertTrue(
            evaled.all(),
            msg='MatrixSystem build is not correct'
        )

    def test_evaluate(self):
        """
        Testing evaluate for a system wtih our 5 common variable types.

        This example is from a well-tested test case.
        """
        # region: variable basis
        true_var_basis_sys_eval = np.array([[
            1.00000000e+00, -1.69821781e+00, 9.67008130e-01,
            5.35784951e-01, 6.29187303e-01, 4.63649236e-01,
            1.88394372e+00, -1.64219043e+00, -9.09879545e-01,
            -1.06849708e+00, -7.87377389e-01, 9.02657086e-01,
            5.18108404e-01, 6.08429238e-01, 4.48352581e-01,
            -1.81235565e-01, 3.37109089e-01, 2.48416284e-01,
            1.82379096e-01, 2.91722213e-01, -1.71232786e+00],
            [ 1.00000000e+00, 2.64654707e-01, -5.54940304e-01,
            8.81321237e-02, -1.18436145e-01, -3.11138051e-01,
            -9.29957886e-01, -1.46867563e-01, 2.33245814e-02,
            -3.13446831e-02, -8.23441497e-02, -3.80618892e-02,
            -4.89080675e-02, 6.57249900e-02, 1.72663045e-01,
            -1.62098589e-01, -1.04380289e-02, -2.74212572e-02,
            -1.04193497e-04, 3.68499912e-02, -2.80917010e-01],
            [ 1.00000000e+00, 9.04884946e-02, -9.27743660e-01,
            -2.61758959e-01, -7.37760503e-02, -8.96952227e-01,
            -9.91811832e-01, -8.39501272e-02, -2.36861741e-02,
            -6.67588373e-03, -8.11638567e-02, 7.91062449e-01,
            2.42845215e-01, 6.84452629e-02, 8.32141742e-01,
            1.31912614e-01, 1.93115421e-02, 2.34785281e-01,
            -2.05977667e-02, 6.61735926e-02, 1.59842775e+00],
            [ 1.00000000e+00, -1.27260262e+00, -3.13090952e-02,
            -1.82294361e-01, 3.97955199e-01, -3.35356642e-01,
            6.19517422e-01, 3.98440364e-02, 2.31988281e-01,
            -5.06438828e-01, 4.26775741e-01, -4.98529611e-01,
            5.70747149e-03, -1.24596172e-02, 1.04997130e-02,
            4.36496968e-02, -7.25449887e-02, 6.11336248e-02,
            6.53266832e-03, -1.33456919e-01, -2.16822638e-01],
            [ 1.00000000e+00, -7.19942644e-01, 4.53624248e-01,
            -2.98857159e-01, 2.85083543e-01, -5.48411877e-01,
            -4.81682589e-01, -3.26583440e-01, 2.15160013e-01,
            -2.05243800e-01, 3.94825096e-01, -1.91337563e-01,
            -1.35568854e-01, 1.29320808e-01, -2.48772925e-01,
            1.77442596e-01, -8.51992576e-02, 1.63896815e-01,
            -4.04639374e-02, -1.56343201e-01, 3.97579340e-01]
        ])
        # endregion: variable basis

        self.assertTrue(
            np.isclose(true_var_basis_sys_eval, self.var_basis_sys_eval[0:5], rtol=0, atol=1e-6).all(),
            msg='MatrixSystem build is not correct'
        )

    def test_solve(self):
        """
        Testing solve for a system wtih our 5 common variable types.

        This example is from a well-tested test case.
        """
        # region: coefficients
        true_matrix_coeffs = np.array([
            9.8, 1.5, 0.75, 3, 4, 1.5, 0, 0.125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0.25
        ])
        # endregion: coefficients

        self.assertTrue(
            np.isclose(true_matrix_coeffs, self.matrix_coeffs.ravel(), rtol=0, atol=1e-5).all(),
            msg='MatrixSystem solve is not correct'
        )

    def test__build_alt_model(self):
        """
        Testing _build_alt_model for a system wtih our 5 common variable types.

        This example is from a well-tested test case.
        """
        true_err_ver = 3e-09
        true_mean_err = 9e-09
        true_mean = 9.8
        true_var = 6.674

        self.assertAlmostEqual(
            true_err_ver, self.err_ver, delta=1e-4,
            msg='MatrixSystem _build_alt_model is not correct'
        )

        self.assertAlmostEqual(
            true_mean_err, self.mean_err[0],
            msg='MatrixSystem _build_alt_model is not correct'
        )

        self.assertAlmostEqual(
            true_mean, self.mean, delta=1e-3,
            msg='MatrixSystem _build_alt_model is not correct'
        )

        self.assertAlmostEqual(
            true_var, self.var, delta=1e-3,
            msg='MatrixSystem _build_alt_model is not correct'
        )

    def test_get_press_stats(self):
        """
        Testing get_press_stats for a system wtih our 5 common variable types.

        This example is from a well-tested test case.
        """
        true_press = 0
        true_mean_of_model_mean_err = 0
        true_variance_of_model_mean_err = 0
        true_mean_of_model_mean = 9.8
        true_variance_of_model_mean = 0
        true_mean_of_model_variance = 6.674137036805925
        true_variance_of_model_variance = 0

        press = self.press_stats['PRESS']
        mean_of_model_mean_err = self.press_stats['mean_of_model_mean_err']
        variance_of_model_mean_err = self.press_stats['variance_of_model_mean_err']
        mean_of_model_mean = self.press_stats['mean_of_model_mean']
        variance_of_model_mean = self.press_stats['variance_of_model_mean']
        mean_of_model_variance = self.press_stats['mean_of_model_variance']
        variance_of_model_variance = self.press_stats['variance_of_model_variance']

        self.assertAlmostEqual(
            true_press, press[0,0],
            msg='MatrixSystem get_press_stats is not correct'
        )

        self.assertAlmostEqual(
            true_mean_of_model_mean_err, mean_of_model_mean_err[0,0],
            msg='MatrixSystem get_press_stats is not correct'
        )

        self.assertAlmostEqual(
            true_variance_of_model_mean_err, variance_of_model_mean_err[0,0],
            msg='MatrixSystem get_press_stats is not correct'
        )

        self.assertAlmostEqual(
            true_mean_of_model_mean, mean_of_model_mean[0,0], delta=1e-8,
            msg='MatrixSystem get_press_stats is not correct'
        )

        self.assertAlmostEqual(
            true_variance_of_model_mean, variance_of_model_mean[0,0],
            msg='MatrixSystem get_press_stats is not correct'
        )

        self.assertAlmostEqual(
            true_mean_of_model_variance, mean_of_model_variance[0,0], delta=1e-9,
            msg='MatrixSystem get_press_stats is not correct'
        )

        self.assertAlmostEqual(
            true_variance_of_model_variance, variance_of_model_variance[0,0],
            msg='MatrixSystem get_press_stats is not correct'
        )

    def test_update(self):
        """
        Testing update for a system wtih our 5 common variable types.

        This example is from a well-tested test case.
        """
        x0 = symbols('x0')
        x1 = symbols('x1')
        x2 = symbols('x2')
        x3 = symbols('x3')
        x4 = symbols('x4')

        true_var_basis_vect_symb = Matrix([[
            1, x0, x1, x2 - (1 / 3), x3 - 0.2, x4 - 1, x0 * x1, x4 ** 2 - 4 * x4 + 2
        ]])
        basis_size = len(true_var_basis_vect_symb)
        equal = [str(Eq(N(sympify(expand(true_var_basis_vect_symb[i])), self.tol), N(sympify(expand(self.var_basis_vect_symb_upd[i])), self.tol))) for i in range(basis_size)]
        eval_loc = locals().copy()
        eval_glob = globals().copy()
        evaled = np.array([eval(equal[i], eval_loc, eval_glob) for i in range(len(equal))])

        true_norm_sq = np.array([
            [1         ],
            [1         ],
            [1 / 3     ],
            [1 / 9     ],
            [16 / 350  ],
            [1         ],
            [1 / 3     ],
            [4         ]
       ])

        true_model_matrix = np.array([
            [0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.],
            [1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 2.]
        ])
        true_min_model_size = len(true_norm_sq)

        self.assertTrue(
            evaled.all(),
            msg='MatrixSystem update is not correct'
        )

        self.assertTrue(
            np.isclose(true_norm_sq, norm_sq_upd, rtol=0, atol=1e-6).all(),
            msg='MatrixSystem update is not correct'
        )

        self.assertTrue(
            np.isclose(true_model_matrix, model_matrix_upd, rtol=0, atol=1e-6).all(),
            msg='MatrixSystem update is not correct'
        )

        self.assertEqual(
            true_min_model_size, min_model_size_upd,
            msg='MatrixSystem update is not correct'
        )


class TestSurrogateModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(33)
        sig = 0.05

        self.var_list = var_list
        self.order = order

        self.min_model_size = min_model_size
        self.model_matrix = model_matrix
        self.norm_sq = norm_sq
        self.var_basis_vect_symb = var_basis_vect_symb
        self.var_basis_sys_eval = var_basis_sys_eval
        self.sig_combo = sig_combo

        # region: coefficients
        self.matrix_coeffs = np.array([
            9.8, 1.5, 0.75, 3, 4, 1.5, 0, 0.125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0.25
        ])
        # endregion: coefficients

        # region: norm squared
        self.norm_sq = np.array([
            [1],
            [1],
            [1 / 3],
            [1 / 9],
            [16 / 350],
            [1],
            [2],
            [1 / 3],
            [1 / 9],
            [16 / 350],
            [1],
            [0.2],
            [(1 / 3) * (1 / 9)],
            [(1 / 3) * (16 / 350)],
            [1 / 3],
            [4 / 81],
            [(1 / 9) * (16 / 350)],
            [1 / 9],
            [64 / 24255],
            [16 / 350],
            [4]
        ])
        # endregion: norm squared

        self.min_model_size = len(self.norm_sq)
        self.responses = responses
        self.model = SurrogateModel(self.responses, self.matrix_coeffs)
        self.resp_mean = self.matrix_coeffs[0]
        self.sigma_sq = self.model.calc_var(self.norm_sq)
        self.sobols = self.model.get_sobols(self.norm_sq)
        self.error, self.pred = self.model.calc_error(self.var_basis_sys_eval)
        self.mean_sq_error, self.hat_matrix, self.shapiro_results = (
            self.model.check_normality(self.var_basis_sys_eval, sig)
        )

        var_list_symb = [symbols('x0'), symbols('x1'), symbols('x2'), symbols('x3'), symbols('x4')]
        self.verify_pred, self.var_basis_sys_eval_verify = self.model.predict(
            self.var_basis_vect_symb, var_list_symb, Xv.T
        )

    def test_get_sobols(self):
        """
        Testing get_sobols for a system wtih our 5 common variable types.

        This example is from a well-tested test case.
        """
        true_sobols = np.array([
            3.37122236e-01, 2.80935197e-02, 1.49832090e-01, 1.09591517e-01,
            3.37122236e-01, 0, 7.80375546e-04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 3.74580262e-02
        ])

        self.assertTrue(
            np.isclose(true_sobols, self.sobols.ravel(), rtol=0, atol=1e-6).all(),
            msg='SurrogateModel get_sobols is not correct'
        )

    def test_calc_var(self):
        """
        Testing calc_var for a system wtih our 5 common variable types.

        This example is from a well-tested test case.
        """
        true_sigma_sq = 6.674137033333334
        true_resp_mean = 9.8

        self.assertAlmostEqual(
            true_sigma_sq, self.sigma_sq, delta=1e-3,
            msg='SurrogateModel calc_var is not correct'
        )

        self.assertAlmostEqual(
            true_resp_mean, self.resp_mean, delta=1e-3,
            msg='SurrogateModel calc_var is not correct'
        )

    def test_calc_error(self):
        """
        Testing calc_error for a system wtih our 5 common variable types.

        This example is from a well-tested test case.
        """
        # region: setting up
        true_error_mean = 1e-08
        true_pred = np.array([
            12.16415155, 9.01613382, 7.20322875, 8.36029185, 8.540021,
            16.30878822, 9.28935314, 7.28973857, 5.31318168, 14.56672247,
            7.00861864, 9.54068885, 9.13214135, 8.02767442, 19.70304005,
            7.63480472, 6.70858894, 14.69559747, 10.99514746, 6.83015149,
            14.35737083, 7.95809041, 11.66095902, 10.73357651, 14.9350537,
            9.78495626, 10.64606339, 9.25057275, 6.02522477, 13.94699517,
            8.72677666, 11.38571172, 10.4209789 , 10.5632802, 10.13697017,
            7.8552094, 9.39608162, 10.50626679, 11.49306889, 6.89224318,
            7.71822459, 7.5307172, 8.66196111, 13.1734537, 11.16095766,
            6.05241666, 11.9000305, 9.86202875, 9.0993081, 9.78234558
        ])
        # endregion: setting up

        error_mean = np.mean(np.abs(self.error))

        self.assertAlmostEqual(
            true_error_mean, error_mean,
            msg='SurrogateModel calc_error is not correct'
        )

        self.assertTrue(
            np.isclose(true_pred, self.pred.ravel(), rtol=0, atol=1e-6).all(),
            msg='SurrogateModel calc_error is not correct'
        )

    def test_check_normality(self):
        tol = 1e-8
        true_mean_sq_error = 3e-08
        true_hat_matrix_diag = np.array([
            0.85301608, 0.14710155, 0.32613936, 0.55408452, 0.44386865,
            0.81320782, 0.1962962 , 0.17939326, 0.35248454, 0.55968826,
            0.62477938, 0.35244729, 0.25953484, 0.18302421, 0.90817508,
            0.31009368, 0.13093016, 0.8703522 , 0.39532409, 0.12536661,
            0.74693703, 0.13340156, 0.34287341, 0.45199607, 0.99011101,
            0.21673877, 0.68270017, 0.47247981, 0.57044074, 0.9031058 ,
            0.19536471, 0.59651909, 0.46501379, 0.3464053 , 0.3337337 ,
            0.1265035 , 0.2272529 , 0.35589019, 0.67320789, 0.22359764,
            0.18542228, 0.44536523, 0.58397094, 0.62470183, 0.24074264,
            0.2890431 , 0.28068778, 0.2432183 , 0.18115646, 0.28611056
        ])
        size = len(true_hat_matrix_diag)

        hat_matrix_diag = np.zeros(size)
        for i in range(size):
            hat_matrix_diag[i] = self.hat_matrix[i, i]

        self.assertAlmostEqual(
            true_mean_sq_error, self.mean_sq_error, delta=tol,
            msg='SurrogateModel check_normality is not correct'
        )

        self.assertTrue(
            np.isclose(true_hat_matrix_diag, hat_matrix_diag, rtol=0, atol=tol).all(),
            msg='SurrogateModel check_normality is not correct'
        )

    def test_predict(self):
        """
        Testing predict for a system wtih our 5 common variable types.

        This example is from a well-tested test case.
        """
        true_verify_pred = 8.6809

        self.assertAlmostEqual(
            true_verify_pred, self.verify_pred[0], delta=1e-4,
            msg='SurrogateModel verify is not correct'
        )

if __name__ == '__main__':

    np.random.seed(33)

    suite = unittest.TestSuite()
    unittest.main()
