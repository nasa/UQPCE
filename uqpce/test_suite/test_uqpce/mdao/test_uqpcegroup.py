import unittest

import numpy as np
import openmdao.api as om

from uqpce import UQPCEGroup

tanh_omega = 1e-6
aleat_cnt = 75_000

class TestUQPCEGroup(unittest.TestCase):
    def setUp(self):

        from uqpce import paraboloid
        from scipy.stats import binom, norm, beta

        aleat_cnt = 10_000
        epist_cnt = 1
        total_cnt = aleat_cnt*epist_cnt
        sig = 0.05

        norm_sq = np.array([[1], [1], [1 / 3]])
        var_basis = np.array([
            [1, -1.690e+00,  9.67e-01],
            [1,  2.646e-01, -5.55e-01],
            [1,  9.048e-02, -9.27e-01],
            [1, -1.270e+00, -3.01e-02],
            [1, -7.199e-01,  4.52e-01],
            [1, -1.272e+00, -2.02e-02]
        ])
        resampled_var_basis = np.zeros([total_cnt, 3])
        resampled_var_basis[:,0] = 1
        resampled_var_basis[:,1] = np.random.uniform(low=-2, high=2, size=total_cnt)
        resampled_var_basis[:,2] = np.random.uniform(low=-1, high=1, size=total_cnt)
        
        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'parab', paraboloid.Paraboloid(vec_size=6),
            promotes_inputs=['*'], promotes_outputs=['*']
        )
        prob.model.add_subsystem(
            'comp',
            UQPCEGroup(
                uncert_list=['f_abxy'],
                var_basis=var_basis, norm_sq=norm_sq, significance=sig, 
                resampled_var_basis=resampled_var_basis, tail='both',
                aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt, sample_ref0=[100], 
                sample_ref=[125]
            ),
            promotes_inputs=['*'], promotes_outputs=['*']
        )
        # prob.model.add_design_var('desx', lower=0, upper=5) #unitless
        # prob.model.add_design_var('desy', lower=0, upper=5) #unitless
        # prob.model.add_objective('f_abxy:ci_upper')

        prob.setup(force_alloc_complex=True)
        prob.set_val('uncerta', np.array([1, 2, 3, 4, 5, 6]))
        prob.set_val('uncertb', np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        prob.set_val('desx', 2)
        prob.set_val('desy', 3.1)
        prob.run_model()

        self.partials = prob.check_partials(out_stream=None, method='cs')


    def test_partials(self):
        # Check partials of CoefficientComp
        coeff_err_coeff = (
            self.partials['comp.f_abxy_coeff_comp']
            [('matrix_coeffs', 'responses')]['rel error'][0]
        )
        coeff_err_mean = (
            self.partials['comp.f_abxy_coeff_comp']
            [('mean', 'responses')]['rel error'][0]
        )
        self.assertTrue(
            np.isclose(coeff_err_coeff, 0), 
            msg='CoefficientComp derivative (\'matrix_coeffs\', \'responses\') '
            'is not correct'
        )
        self.assertTrue(
            np.isclose(coeff_err_mean, 0), 
            msg='CoefficientComp derivative (\'mean\', \'responses\') '
            'is not correct'
        )

        # Check partials of ResampleComp
        resamp_err_resamp = (
            self.partials['comp.f_abxy_resamp_comp']
            [('resampled_responses', 'matrix_coeffs')]['rel error'][0]
        )
        self.assertTrue(
            np.isclose(resamp_err_resamp, 0), 
            msg='ResampleComp derivative (\'resampled_responses\', '
            '\'matrix_coeffs\') is not correct'
        )

        # Check partials of VarianceComp
        var_err_coeff = (
            self.partials['comp.f_abxy_var_comp']
            [('variance', 'matrix_coeffs')]['rel error'][0]
        )
        self.assertTrue(
            np.isclose(var_err_coeff, 0), 
            msg='VarianceComp derivative (\'variance\', \'matrix_coeffs\') '
            'is not correct'
        )

        # Check partials of MeanPlusVarComp
        mpv_err_mean = (
            self.partials['comp.f_abxy_mean_plus_var_comp']
            [('mean_plus_var', 'mean')]['rel error'][0]
        )
        mpv_err_var = (
            self.partials['comp.f_abxy_mean_plus_var_comp']
            [('mean_plus_var', 'variance')]['rel error'][0]
        )
        self.assertTrue(
            np.isclose(mpv_err_mean, 0), 
            msg='MeanPlusVarComp derivative (\'mean_plus_var\', \'mean\') '
            'is not correct'
        )
        self.assertTrue(
            np.isclose(mpv_err_var, 0), 
            msg='MeanPlusVarComp derivative (\'mean_plus_var\', \'variance\') '
            'is not correct'
        )

        # Check partials of CDFGroup
        lower_cdf_samp = (
            self.partials['comp.f_abxy_lower_cdf_group.cdf']
            [('ci_resid', 'samples')]['rel error'][0]
        )
        lower_cdf_fci = (
            self.partials['comp.f_abxy_lower_cdf_group.cdf']
            [('ci_resid', 'f_ci')]['rel error'][0]
        )
        upper_cdf_samp = (
            self.partials['comp.f_abxy_upper_cdf_group.cdf']
            [('ci_resid', 'samples')]['rel error'][0]
        )
        upper_cdf_fci = (
            self.partials['comp.f_abxy_upper_cdf_group.cdf']
            [('ci_resid', 'f_ci')]['rel error'][0]
        )

        self.assertTrue(
            np.isclose(lower_cdf_samp, 0), 
            msg='CDFGroup derivative (\'ci_resid\', \'samples\') is not correct'
        )
        self.assertTrue(
            np.isclose(upper_cdf_samp, 0), 
            msg='CDFGroup derivative (\'ci_resid\', \'samples\') is not correct'
        )
        self.assertTrue(
            np.isclose(lower_cdf_fci, 0), 
            msg='CDFGroup derivative (\'ci_resid\', \'f_ci\') is not correct'
        )
        self.assertTrue(
            np.isclose(upper_cdf_fci, 0), 
            msg='CDFGroup derivative (\'ci_resid\', \'f_ci\') is not correct'
        )



if __name__ == '__main__':

    np.random.seed(33)

    suite = unittest.TestSuite()
    unittest.main()
