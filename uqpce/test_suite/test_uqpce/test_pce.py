import unittest
from io import StringIO
import syslog
import numpy as np

from uqpce import PCE
from uqpce.pce.variables.continuous import (
    UniformVariable, NormalVariable, ContinuousVariable
)

class TestPCE(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.aleat_cnt = 25000
        self.pce = PCE(
            verbose=False, outputs=False, aleat_samp_size=self.aleat_cnt
        )

        self.var_dict = {
            'Variable 0':{'distribution':'uniform', 'interval_low':6, 'interval_high':7, 'type':'aleatory'}, 
            'Variable 1':{'distribution':'normal', 'mean':2, 'stdev':7}, 
            'Variable 2':{'distribution':'continuous', 'pdf':'x', 'interval_low':0, 'interval_high':1}
        }
        for key, value in self.var_dict.items():
            self.pce.add_variable(**value)

        self.X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        self.X_stand = np.array([
            [-11, 0, 3],
            [-5, 0.42857142857142855, 6],
            [1, 0.8571428571428571, 9]
        ])

        self.var_basis = np.array([
            [1, -11, 0, 2+1/3, 181, 0, -25-2/3, -1, 0, 5.7],
            [1, -5, 0.42857143, 5+1/3, 37, -2.14285714, -26-2/3, -0.81632653, 
             2.28571429, 29.1],
            [1, 1, 0.85714286, 8+1/3, 1, 0.85714286, 8+1/3, -0.26530612, 
             7.14285714, 70.5]
        ])

        self.coeffs = np.array([
            [4.46333333e+01],
            [3.15000000e+00],
            [7.35000000e+01],
            [-2.39999991e-01],
            [1.66666672e-02],
            [3.50000000e+00],
            [-1.20884707e-08],
            [7.75582134e-10],
            [-1.91746002e-08],
            [-2.00000024e-01]
        ])

        self.Xfull = np.array([
            [6.9844401, -14.43267517, 0.80832032],
            [6.4546889, 1.88209264, 0.7025658 ],
            [6.55378905, 7.17022173, 0.06456357],
            [6.74649883, 2.84136776, 0.63094464],
            [6.27726664, 5.06353382, 0.72269021],
            [6.85162001, 10.35977754, 0.82224553],
            [6.70757488, 15.53280154, 0.97231201],
            [6.8766664, -7.24820184, 0.51295487],
            [6.06072536, -0.14736632, 0.59577929],
            [6.2261537, 9.43464177, 0.67003164],
            [6.6067751, 0.72806726, 0.83054083],
            [6.48703187, 4.63462365, 0.80485814],
            [6.33901146, -5.56423756, 0.54325641],
            [6.18951312, -1.54274933, 0.86892836],
            [6.11340383, -2.91818499, 0.94443107]
        ])
        self.yfull = np.array([
            -132.83397158,   43.10834087,   99.62875613,   55.24615822,
            74.70703855,  137.53448142,  190.75143817,  -53.53004468,
            20.3018126 ,  118.94526845,   31.76975909,   72.14314046,
            -34.5524004 ,    6.52867479,   -7.61359113
        ])

        self.yv = np.array([
            [55.76950619],
            [24.14615998],
            [-35.77190184],
            [81.51459335],
            [155.45079017]
        ])

    def test_add_variable(self):

        self.assertTrue(
            len(self.pce.variables) == len(self.var_dict),
            msg='PCE add_variable is not correct'
        )
        self.assertTrue(
            type(self.pce.variables[0]) is UniformVariable,
            msg='PCE add_variable is not correct'
        )
        self.assertTrue(
            type(self.pce.variables[1]) is NormalVariable,
            msg='PCE add_variable is not correct'
        )
        self.assertTrue(
            type(self.pce.variables[2]) is ContinuousVariable,
            msg='PCE add_variable is not correct'
        )

    def test_set_samples(self):

        self.pce.set_samples(self.X)
        self.assertTrue(
            (self.pce._X == self.X).all(), 
            msg='PCE set_samples is not correct'
        )
        self.assertTrue(
            (self.pce._X_stand == self.X_stand).all(), 
            msg='PCE set_samples is not correct'
        )

    def test_build_basis(self):
        self.pce.set_samples(self.X)
        self.pce.build_basis(order=2)

        self.assertTrue(
            np.isclose(self.pce.var_basis, self.var_basis).all(), 
            msg='PCE build_basis is not correct'
        )

    def test_fit(self):
        self.pce.fit(self.Xfull, self.yfull)
        self.assertTrue(
            np.isclose(self.pce.matrix_coeffs, self.coeffs).all(), 
            msg='PCE fit is not correct'
        )

    def test_predict(self):
        Xv = np.array([
            [6.12621703, 3.32946119, 0.62066499],
            [6.83541515, -0.09102409, 0.4800609],
            [6.74609318, -5.63265315, 0.4008933],
            [6.577896, 5.43906771, 0.63189715],
            [6.24346511, 12.97452976, 0.63966294]
        ])
        self.pce.fit(self.Xfull, self.yfull)
        resp_pred = self.pce.predict(Xv)

        self.assertTrue(
            np.isclose(resp_pred, self.yv).all(), 
            msg='PCE fit is not correct'
        )

    def test_verification(self):
        self.pce.fit(self.Xfull, self.yfull)
        err = self.pce.verification(self.Xfull, self.yfull)

        self.assertTrue(
            (np.abs(err) < 1e-6).all(), 
            msg='PCE verification is not correct'
        )

    def test_resample_surrogate(self):
        self.pce.fit(self.Xfull, self.yfull)
        eval_resps = self.pce.resample_surrogate()

        self.assertTrue(
            eval_resps.ravel().shape[0] == self.aleat_cnt, 
            msg='PCE resample_surrogate is not correct'
        )

    def test_confidence_interval(self):
        self.pce.fit(self.Xfull, self.yfull)
        cil, cih = self.pce.confidence_interval()

        self.assertTrue(
            cil > -101 and cil < -98,
            msg='PCE resample_surrogate is not correct'
        )
        self.assertTrue(
            cih > 188 and cih < 190,
            msg='PCE resample_surrogate is not correct'
        )

if __name__ == '__main__':

    np.random.seed(0)

    suite = unittest.TestSuite()
    unittest.main()
