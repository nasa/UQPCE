import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
from scipy.stats import norm


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    val = array[idx]
    return(idx,val)


def tanh_activation(x, omega=1, z=0, a=-1, b=1):
    """
    https://github.com/OpenMDAO/POEMs/pull/171/files?short_path=59ae36c#diff-59ae36c5f4569d1629b476ea07c65ce0b3081afbe219cf3c8c435bf040d77dc6
    """
    """
    Hyperbolic tangent with adjustable parameters.

    Parameters
    ----------
    x : float
        The nominal argument to hyperbolic tangent.
    omega : float
        The omega parameter impacts the steepness with which the transition in the value of the hyperbolic tangent occurs. The default of 1 provides the nominal behavior of the hyperbolic tangent.
    z : float
        The y-intercept of the activation function. The default of 0 provides the nominal behavior of the hyperbolic tangent.
    a : float
        The initial value of transition function. The default of -1 provides the nominal behavior of the hyperbolic tangent.
    b : float
        The final value of transition function. The default of -1 provides the nominal behavior of the hyperbolic tangent.
    """
    dy = b - a
    tanh_term = np.tanh((x - z) / omega)
    return 0.5 * dy * (1 + tanh_term) + a



def dtanh_act(x, omega=1.0, z=0.0, a=-1.0, b=1.0):
    """
    A function which provides a differentiable activation function based on the hyperbolic tangent.
    Parameters
    ----------
    x : float or np.array
        The input at which the value of the activation function is to be computed.
    omega : float
        A shaping parameter which impacts the "abruptness" of the activation function. As this value approaches zero
        the response approaches that of a step function.
    z : float
        The value of the independent variable about which the activation response is centered.
    a : float
        The initial value that the input asymptotically approaches negative infinity.
    b : float
        The final value that the input asymptotically approaches positive infinity.
    Returns
    -------
    dict
        A dictionary which contains the partial derivatives of the tanh activation function wrt inputs, stored in the
        keys 'x', 'omega', 'z', 'a', 'b'.
    """
    dy = b - a
    xmz = x - z
    tanh_term = np.tanh(xmz / omega)
    partials = {'x': (0.5 * dy) / (omega * np.cosh(xmz / omega)**2),
                'omega': (-0.5 * dy * xmz) / (omega**2 * np.cosh(xmz / omega)**2),
                'z': (-0.5 * dy) / (omega * np.cosh(xmz / omega)**2),
                'a': 0.5 * (1 - tanh_term),
                'b': 0.5 * (1 + tanh_term)}
    
    return partials


class CDFResidComp(om.ExplicitComponent):
    """
    Component class to calculate the residual between the current tanh function 
    evaluation sum and the confidence-interval-based value that it is desired to 
    be.
    """

    def initialize(self):
        self.options.declare('vec_size', types=int)

        # The probability of the response is greater than the 1-alpha value
        # i.e. alpha=0.05 corresponds to the cumulative probability of 95%
        self.options.declare(
            'alpha', types=float, default=0.05,
            desc='Single-sided upper confidence interval of (1-alpha)'
        )
        self.options.declare('tanh_omega', types=float, default=1e-6)
        self.options.declare('aleatory_cnt', types=int, allow_none=False)
        self.options.declare('epistemic_cnt', types=int, allow_none=False)
        self.options.declare('tail', values=['lower', 'upper'], allow_none=False)
        self.options.declare('sample_ref0', types=(float, int), default=0.0, desc='Reference scale for 0 of the sample data')
        self.options.declare('sample_ref', types=(float, int), default=1.0, desc='Reference scale for 1 of the sample data')

    def setup(self):
        alpha = self.options['alpha']
        aleat_cnt = self.options['aleatory_cnt']
        epist_cnt = self.options['epistemic_cnt']

        self.add_input('samples', shape=(epist_cnt*aleat_cnt,))
        self.add_input('f_ci', shape=(epist_cnt,))

        self.add_output('ci_resid', shape=(epist_cnt,))

        self._sig = (1-alpha/2) if self.options['tail'] == 'upper' else alpha/2

    def compute(self, inputs, outputs):
        sample_ref0 = self.options['sample_ref0']
        sample_ref = self.options['sample_ref']
        f_sampled = (inputs['samples'] - sample_ref0) / sample_ref
        f_ci = (inputs['f_ci'] - sample_ref0) / sample_ref

        aleat_cnt = self.options['aleatory_cnt']
        tanh_omega = self.options['tanh_omega']
        x = np.reshape(f_sampled, (-1, aleat_cnt))

        dlt = tanh_activation(x.T, omega=tanh_omega, z=f_ci, a=1, b=0).T

        outputs['ci_resid'] = (np.sum(dlt, axis=1) / aleat_cnt) - self._sig

    def setup_partials(self):
        epist_cnt = self.options['epistemic_cnt']
        aleat_cnt = self.options['aleatory_cnt']

        epist = np.linspace(0, epist_cnt-1, num=epist_cnt)
        aleat = np.linspace(0,aleat_cnt*epist_cnt-1, aleat_cnt*epist_cnt)

        self.declare_partials(of='ci_resid', wrt='f_ci', rows=epist, cols=epist)
        self.declare_partials(of='ci_resid', wrt='samples', rows=np.repeat(epist, aleat_cnt), cols=aleat)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sample_ref0 = self.options['sample_ref0']
        sample_ref = self.options['sample_ref']
        samples = (inputs['samples']-sample_ref0) / sample_ref
        f_ci = (inputs['f_ci']-sample_ref0) / sample_ref
        tanh_omega = self.options['tanh_omega']
        aleat_cnt = self.options['aleatory_cnt']

        x = np.reshape(samples, (-1, aleat_cnt))
        ps = dtanh_act(x.T, omega=tanh_omega, z=f_ci, a=1, b=0)

        # divide by N *see equations*
        partials['ci_resid', 'f_ci'] = (np.sum(ps['z'].T, axis=1))/aleat_cnt/sample_ref # d ci_resid / d f_ci

        partials['ci_resid', 'samples'] = ((ps['x'].T)/aleat_cnt).reshape(1,-1)/sample_ref # d sum(f_ci < f_95) / d f_sampled
 


if __name__ == '__main__':

    import numpy as np

    lower = -2
    upper = 2

    alpha = 0.05 # 0.0455
    epist_cnt = 10
    aleat_cnt = 50
    vec_size = aleat_cnt*epist_cnt

    np.random.seed(1)
    samps = np.random.uniform(low=lower, high=upper, size=vec_size) #np.random.normal(loc=mean, scale=stdev, size=vec_size)
    ci_init = np.quantile(samps.reshape(-1, aleat_cnt), 1-alpha/2)
    prob = om.Problem()
    prob.model.add_subsystem(
        'res_cdf', CDFResidComp(
            vec_size=vec_size, alpha=alpha, tanh_omega=0.001, tail='upper', 
            aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt
        ),
        promotes_inputs=['samples', 'f_ci'], 
        promotes_outputs=['ci_resid']
    )

    bal = prob.model.add_subsystem(
        'bal', om.BalanceComp(val=np.ones([epist_cnt])), 
        promotes_inputs=['ci_resid'], promotes_outputs=['f_ci']
    )
    bal.add_balance(name='f_ci', lhs_name='ci_resid', val=np.ones([epist_cnt]))

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    prob.model.linear_solver = om.DirectSolver()

    prob.setup(force_alloc_complex=True)
    prob.set_val('res_cdf.samples', samps)
    prob.set_val('bal.f_ci', ci_init)
    prob.run_model()

    prob.check_partials(compact_print=True, method='fd')