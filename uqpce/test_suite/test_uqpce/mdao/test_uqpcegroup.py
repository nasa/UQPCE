import unittest

import numpy as np
import openmdao.api as om
import openmdao.utils.assert_utils as om_assert

from uqpce import MultiUQPCEGroup, UQPCEGroup
from uqpce.examples.paraboloid.paraboloid import paraboloid


class TestUQPCEGroup(unittest.TestCase):
    def setUp(self):
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
        resampled_var_basis[:,1] = np.linspace(-2, 2, num=total_cnt)
        resampled_var_basis[:,2] = np.linspace(-1, 1, num=total_cnt)

        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'parab', paraboloid.Paraboloid(vec_size=6),
            promotes_inputs=['*'], promotes_outputs=['*']
        )
        prob.model.add_subsystem(
            'comp',
            UQPCEGroup(
                var_basis=var_basis, norm_sq=norm_sq, significance=sig,
                resampled_var_basis=resampled_var_basis, tail='both',
                aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt, sample_ref0=100,
                sample_ref=125, use_tanh_ci=True
            ),
            promotes_inputs=[('responses', 'f_abxy')], promotes_outputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('uncerta', np.array([1, 2, 3, 4, 5, 6]))
        prob.set_val('uncertb', np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        prob.set_val('desx', 2)
        prob.set_val('desy', 3.1)
        prob.run_model()

        self.cs_partials = prob.check_partials(out_stream=None, method='cs')

        ########################################################################
        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'parab', paraboloid.Paraboloid(vec_size=6),
            promotes_inputs=['*'], promotes_outputs=['*']
        )
        prob.model.add_subsystem(
            'comp',
            UQPCEGroup(
                var_basis=var_basis, norm_sq=norm_sq, significance=sig,
                resampled_var_basis=resampled_var_basis, tail='both',
                aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt,
                use_tanh_ci=False
            ),
            promotes_inputs=[('responses', 'f_abxy')], promotes_outputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('uncerta', np.array([1, 2, 3, 4, 5, 6]))
        prob.set_val('uncertb', np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        prob.set_val('desx', 2)
        prob.set_val('desy', 3.1)
        prob.run_model()

        self.fd_partials = prob.check_partials(out_stream=None, method='fd')

    def test_partials(self):
        om_assert.assert_check_partials(self.cs_partials, atol=1e-6, rtol=1e-6)
        om_assert.assert_check_partials(self.fd_partials, atol=1e-6, rtol=1e-6)


class TestMultiUQPCEGroup(unittest.TestCase):
    def setUp(self):
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
        resampled_var_basis[:,1] = np.linspace(-2, 2, num=total_cnt)
        resampled_var_basis[:,2] = np.linspace(-1, 1, num=total_cnt)

        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'parab', paraboloid.Paraboloid(vec_size=6),
            promotes_inputs=['*'], promotes_outputs=['*']
        )
        prob.model.add_subsystem(
            'comp',
            MultiUQPCEGroup(
                uncert_list=['f_abxy'],
                var_basis=var_basis, norm_sq=norm_sq, significance=sig,
                resampled_var_basis=resampled_var_basis, tail='both',
                aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt, sample_ref0=[100],
                sample_ref=[125], use_tanh_ci=True
            ),
            promotes_inputs=['*'], promotes_outputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('uncerta', np.array([1, 2, 3, 4, 5, 6]))
        prob.set_val('uncertb', np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        prob.set_val('desx', 2)
        prob.set_val('desy', 3.1)
        prob.run_model()

        self.cs_partials = prob.check_partials(out_stream=None, method='cs')

        ########################################################################
        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            'parab', paraboloid.Paraboloid(vec_size=6),
            promotes_inputs=['*'], promotes_outputs=['*']
        )
        prob.model.add_subsystem(
            'comp',
            MultiUQPCEGroup(
                uncert_list=['f_abxy'],
                var_basis=var_basis, norm_sq=norm_sq, significance=sig,
                resampled_var_basis=resampled_var_basis, tail='both',
                aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt,
                use_tanh_ci=False
            ),
            promotes_inputs=['*'], promotes_outputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('uncerta', np.array([1, 2, 3, 4, 5, 6]))
        prob.set_val('uncertb', np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        prob.set_val('desx', 2)
        prob.set_val('desy', 3.1)
        prob.run_model()

        self.fd_partials = prob.check_partials(out_stream=None, method='fd')

    def test_partials(self):
        om_assert.assert_check_partials(self.cs_partials, atol=1e-6, rtol=1e-6)
        om_assert.assert_check_partials(self.fd_partials, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':

    np.random.seed(33)

    suite = unittest.TestSuite()
    unittest.main()