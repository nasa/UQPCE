import numpy as np
from scipy.optimize import brentq

import openmdao.api as om
from openmdao.recorders.recording_iteration_stack import Recording
from uqpce.mdao.cdf.cdfresidcomp import CDFResidComp

class CDFGroup(om.Group):

    def initialize(self):
        self.options.declare('vec_size', types=int)
        self.options.declare(
            'alpha', types=float, default=0.05,
            desc='Single-sided upper confidence interval of (1-alpha)'
        )
        self.options.declare('tanh_omega', types=float, default=1e-6)
        self.options.declare('tail', values=['lower', 'upper'], allow_none=False)
        self.options.declare('aleatory_cnt', types=int, allow_none=False)
        self.options.declare('epistemic_cnt', types=int, allow_none=False)
        self.options.declare(
            'sample_ref0', types=(float, int), default=0.0,
            desc='Scaling parameter. The value in the user-defined units of '
            'this output variable when the scaled value is 0. Default is 0.'
        )
        self.options.declare(
            'sample_ref', types=(float, int), default=1.0, 
            desc='Scaling parameter. The value in the user-defined units of '
            'this output variable when the scaled value is 1. Default is 1.'
        )


    def setup(self):

        vec_size = self.options['vec_size']
        alpha = self.options['alpha']
        tanh_omega = self.options['tanh_omega']
        tail = self.options['tail']
        aleatory_cnt = self.options['aleatory_cnt']
        epistemic_cnt = self.options['epistemic_cnt']
        sample_ref0 = self.options['sample_ref0']
        sample_ref = self.options['sample_ref']

        self.add_subsystem(
            'cdf', CDFResidComp(
                vec_size=vec_size, alpha=alpha, tanh_omega=tanh_omega, tail=tail,
                aleatory_cnt=aleatory_cnt, epistemic_cnt=epistemic_cnt, 
                sample_ref0=sample_ref0, sample_ref=sample_ref
            ),
            promotes_inputs=[('samples', 'f_sampled'), 'f_ci'],
            promotes_outputs=['ci_resid']
        )

        bal = self.add_subsystem(
            'bal', om.BalanceComp(val=np.ones([epistemic_cnt])),
            promotes_inputs=['ci_resid'], promotes_outputs=['f_ci']
        )
        bal.add_balance(name='f_ci', lhs_name='ci_resid', val=np.ones([epistemic_cnt]))

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()

        minimum = (self.options['tail'] == 'lower')

        if vec_size == aleatory_cnt: # purely aleatoric
            pass
        else:
            self.add_subsystem('ks', om.KSComp(width=epistemic_cnt, minimum=minimum, rho=1000.), promotes_outputs=[('KS','ci')])
            self.connect('f_ci', 'ks.g')


    def guess_nonlinear(self, inputs, outputs, residuals):
        aleatory_cnt = self.options['aleatory_cnt']
        samples = inputs['cdf.samples']
        alpha = self.options['alpha']
        x = np.reshape(samples, (-1, aleatory_cnt))
        outputs['f_ci'] = np.percentile(x, self._get_subsystem('cdf')._sig*100, axis=1) # find CI of curves
        self._truth = np.max(outputs['f_ci'])
        

if __name__ == '__main__':

    import numpy as np
    from scipy.stats import uniform, binom, norm, beta

    lower = -2
    upper = 2

    alpha = 0.05 # 0.0455
    aleat_cnt = 200
    epist_cnt = 5
    vec_size = aleat_cnt * epist_cnt

    np.random.seed(1)
    # samps = np.random.uniform(low=lower, high=upper, size=vec_size) #np.random.normal(loc=mean, scale=stdev, size=vec_size)
    # samps = uniform.rvs(-1,2, size=vec_size)
    samps = binom.rvs(n=5, p=0.3, size=vec_size)+norm.rvs(0,0.05, size=vec_size)
    # samps = beta.rvs(a=2,b=1,size=vec_size)

    prob = om.Problem()
    prob.model.add_subsystem(
        'comp',
        CDFGroup(
            alpha=alpha, tanh_omega=0.005, tail='upper', vec_size=vec_size,
            epistemic_cnt=epist_cnt, aleatory_cnt=aleat_cnt, sample_ref0=-0.13, 
            sample_ref=4.1
        ),
        promotes_inputs=['*'], promotes_outputs=['*']
    )

    prob.setup(force_alloc_complex=True)
    prob.set_val('f_sampled', samps)
    prob.run_model()

    with np.printoptions(threshold=np.inf):
        prob.check_partials(compact_print=True, method='cs')