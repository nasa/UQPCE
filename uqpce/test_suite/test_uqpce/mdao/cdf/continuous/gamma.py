from uqpce.mdao.cdf.cdfgroup import CDFGroup
import matplotlib.pyplot as plt
from uqpce.mdao.cdf.cdfresidcomp import tanh_activation
import openmdao.api as om
from scipy.stats import gamma
import numpy as np

if __name__ == '__main__':
    lfs = 16
    tfs = 24

    size = 100_000
    sig = 0.05
    omega=0.001; a=1; b=0
    p = (sig/2)
    np.random.seed(0)

    dist = gamma(loc=-1, a=2, scale=1/4)
    samples = dist.rvs(size=size)
    x = np.linspace(samples.min(), samples.max(), num=10000)
    guess = np.quantile(samples, p)

    prob = om.Problem()
    prob.model = CDFGroup(
        vec_size=size, aleatory_cnt=size, epistemic_cnt = 1, alpha=sig, 
        tanh_omega=omega, tail='lower'
    )

    prob.setup(force_alloc_complex=True)
    prob.set_val('f_sampled', samples)

    prob.run_model()

    calc = prob.get_val("f_ci")[0]

    print(f'The analytical bound is {dist.ppf(p)}')
    print(f'The interpolated bound is {guess}')
    print(f'The solved bound is {calc}')

    y = tanh_activation(x, omega=omega, z=calc, a=a, b=b)
    tanh_ys = tanh_activation(samples, omega=omega, z=calc, a=a, b=b)

    f, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, gridspec_kw={'width_ratios': [0.5]})

    hist_color = (172/255, 163/255, 197/255)
    ax1.hist(samples, bins=15, density=True, color=hist_color, label='PDF')
    ax1.plot([calc, calc], [ax1.get_ylim()[0], ax1.get_ylim()[1]], 'k--')
    ax1.set_ylabel('PDF', fontsize=tfs)

    ax2.plot(sorted(samples), np.linspace(0, 1, num=len(samples)), '-', color='seagreen', lw=2, label='CDF')
    ax2.plot([calc, calc], [0, 1], 'k--')
    ax2.plot([x.min(), calc], [0.025, 0.025], 'r', ls='dotted')
    ax2.set_ylabel('CDF', fontsize=tfs)

    ax3.plot(x, y, '-', color='dodgerblue', lw=2, label='tanh')
    ax3.plot([calc, calc], [0, 1], 'k--', label='upper CI')
    ax3.set_ylabel('f(x)', fontsize=tfs)
    ax3.set_xlabel('x', fontsize=tfs)

    ax1.tick_params(axis='both', which='major', labelsize=lfs)
    ax2.tick_params(axis='both', which='major', labelsize=lfs)
    ax3.tick_params(axis='both', which='major', labelsize=lfs)
    
    f.align_ylabels([ax1, ax2, ax3])

    f.suptitle('Gamma Distribution', fontsize=tfs)
    f.legend(fontsize=12, bbox_to_anchor=(1, 1.1), fancybox=True, labelspacing=0.1)

    # plt.savefig('tanh_gamma', bbox_inches='tight')