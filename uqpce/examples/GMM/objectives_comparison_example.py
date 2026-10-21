"""Compare the UQPCE optimization objectives on one analytical problem.

The model is

    f = (x - 4)^2 + (y - 4)^2 + 0.1*(x*a1)^2 + 0.22*(y*a2)^2
        + 0.5*a1^2 + 0.1*a3^2

with design variables x, y and the uncertain inputs of this example
(input.yaml): a1 is a three-component Gaussian mixture (multimodal,
unbounded), a2 is uniform (bounded), a3 is normal. Moving toward the
deterministic target (4, 4) amplifies both uncertain terms, so each
statistic picks a different compromise between performance and exposure:

    deterministic   ignores uncertainty, sits at (4, 4)
    mean            best expected value, accepts a wide distribution
    mean_plus_var   narrowest distribution, pays the largest mean penalty
    ci_upper        best 95th percentile (alpha = 0.1)
    cvar_upper      best mean of the worst 5%; relative to ci_upper it
                    shifts exposure away from the unbounded multimodal
                    a1 and onto the bounded a2

f is quadratic in the uncertain inputs, so the order-2 PCE is exact and
the designs differ only because of the statistics, not surrogate error.

Run from this directory (uses the same input.yaml / run_matrix.dat as
GMM_Test.py):

    python objectives_comparison_example.py

Prints a Monte Carlo summary of every optimal design and writes
objectives_comparison.png with the designs and their CDFs/PDFs.
"""
import os

os.environ.setdefault('OPENMDAO_REPORTS', '0')
# single-threaded math keeps the flat CI/CVaR optima reproducible run-to-run
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault(
    'XLA_FLAGS',
    '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
)

import numpy as np
import openmdao.api as om

from uqpce.mdao import interface
from uqpce.mdao.uqpcegroup import UQPCEGroup

X_TARGET = 4.0
ALPHA = 0.1  # ci_upper = 95th percentile, cvar_upper = mean of the worst 5%
OBJECTIVES = ('mean', 'mean_plus_var', 'ci_upper', 'cvar_upper')


class QuadraticExposure(om.ExplicitComponent):
    """
    f = (x-4)**2 + (y-4)**2 + 0.1*(x*a1)**2 + 0.22*(y*a2)**2
        + 0.5*a1**2 + 0.1*a3**2
    """

    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']
        self.add_input('x', val=1.0)
        self.add_input('y', val=1.0)
        self.add_input('a1', shape=(n,))
        self.add_input('a2', shape=(n,))
        self.add_input('a3', shape=(n,))
        self.add_output('f', shape=(n,))

        arange = np.arange(n)
        self.declare_partials('f', ['x', 'y'])
        self.declare_partials('f', ['a1', 'a2', 'a3'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        x, y = inputs['x'], inputs['y']
        a1, a2, a3 = inputs['a1'], inputs['a2'], inputs['a3']

        outputs['f'] = ((x - X_TARGET)**2 + (y - X_TARGET)**2
                        + 0.1*(x*a1)**2 + 0.22*(y*a2)**2
                        + 0.5*a1**2 + 0.1*a3**2)

    def compute_partials(self, inputs, partials):
        x, y = inputs['x'], inputs['y']
        a1, a2, a3 = inputs['a1'], inputs['a2'], inputs['a3']

        partials['f', 'x'] = (2*(x - X_TARGET) + 0.2*x*a1**2).reshape(-1, 1)
        partials['f', 'y'] = (2*(y - X_TARGET) + 0.44*y*a2**2).reshape(-1, 1)
        partials['f', 'a1'] = 0.2*x**2*a1 + a1
        partials['f', 'a2'] = 0.44*y**2*a2
        partials['f', 'a3'] = 0.2*a3


def optimize_deterministic(uq_data):
    """Minimize f with the uncertain inputs held at their means."""
    (var_basis, norm_sq, resampled_var_basis, aleatory_cnt, epistemic_cnt,
     resp_cnt, order, variables, sig, run_matrix) = uq_data

    prob = om.Problem(reports=False)
    prob.model.add_subsystem(
        'func', QuadraticExposure(vec_size=resp_cnt), promotes=['*'])

    prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9)
    prob.model.add_design_var('x', lower=0.0, upper=6.0)
    prob.model.add_design_var('y', lower=0.0, upper=6.0)
    prob.model.add_objective('f', index=0)

    prob.setup()
    interface.set_vals(prob, variables, run_matrix, deterministic=True)
    prob.run_driver()

    return float(prob.get_val('x')[0]), float(prob.get_val('y')[0])


def optimize_statistic(objective, uq_data):
    """Minimize one UQPCE statistic of f over the design variables."""
    (var_basis, norm_sq, resampled_var_basis, aleatory_cnt, epistemic_cnt,
     resp_cnt, order, variables, sig, run_matrix) = uq_data

    prob = om.Problem(reports=False)

    prob.model.add_subsystem(
        'func', QuadraticExposure(vec_size=resp_cnt),
        promotes_inputs=['x', 'y', 'a1', 'a2', 'a3'],
        promotes_outputs=['f']
    )
    prob.model.add_subsystem(
        'uq',
        UQPCEGroup(
            significance=ALPHA, var_basis=var_basis, norm_sq=norm_sq,
            resampled_var_basis=resampled_var_basis, tail='upper',
            aleatory_cnt=aleatory_cnt, epistemic_cnt=epistemic_cnt,
            compute_cvar=True
        ),
        promotes_inputs=[('responses', 'f')],
        promotes_outputs=['mean', 'mean_plus_var', 'ci_upper', 'cvar_upper']
    )

    prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9)
    prob.model.add_design_var('x', lower=0.0, upper=6.0)
    prob.model.add_design_var('y', lower=0.0, upper=6.0)
    prob.model.add_objective(objective)

    prob.setup()
    interface.set_vals(prob, variables, run_matrix)
    prob.set_val('x', 1.0)
    prob.set_val('y', 1.0)
    prob.run_driver()

    return float(prob.get_val('x')[0]), float(prob.get_val('y')[0])


def evaluate_designs(designs, uq_data):
    """UQPCE statistics of f at fixed designs (one problem, re-run per design)."""
    (var_basis, norm_sq, resampled_var_basis, aleatory_cnt, epistemic_cnt,
     resp_cnt, order, variables, sig, run_matrix) = uq_data

    prob = om.Problem(reports=False)
    prob.model.add_subsystem(
        'func', QuadraticExposure(vec_size=resp_cnt),
        promotes_inputs=['x', 'y', 'a1', 'a2', 'a3'],
        promotes_outputs=['f']
    )
    prob.model.add_subsystem(
        'uq',
        UQPCEGroup(
            significance=ALPHA, var_basis=var_basis, norm_sq=norm_sq,
            resampled_var_basis=resampled_var_basis, tail='upper',
            aleatory_cnt=aleatory_cnt, epistemic_cnt=epistemic_cnt,
            compute_cvar=True
        ),
        promotes_inputs=[('responses', 'f')],
        promotes_outputs=['mean', 'variance', 'ci_upper', 'cvar_upper']
    )
    prob.setup()
    interface.set_vals(prob, variables, run_matrix)

    stats = {}
    for name, (x, y) in designs.items():
        prob.set_val('x', x)
        prob.set_val('y', y)
        prob.run_model()
        stats[name] = (
            float(prob.get_val('mean')[0]),
            float(np.sqrt(prob.get_val('variance')[0])),
            float(prob.get_val('ci_upper')[0]),
            float(prob.get_val('cvar_upper')[0]),
        )
    return stats


def sample_uncertain_inputs(n_samples, seed=42):
    """Monte Carlo draws of (a1, a2, a3) matching input.yaml."""
    rng = np.random.default_rng(seed)

    weights = [0.3, 0.5, 0.2]
    means = [-2.0, 0.0, 3.0]
    stdevs = [0.3, 0.5, 0.4]
    comp = rng.choice(3, size=n_samples, p=weights)
    a1 = np.zeros(n_samples)
    for i in range(3):
        mask = comp == i
        a1[mask] = rng.normal(means[i], stdevs[i], mask.sum())

    a2 = rng.uniform(-2.0, 2.0, n_samples)
    a3 = rng.normal(0.0, 1.0, n_samples)
    return a1, a2, a3


def f_true(x, y, a1, a2, a3):
    return ((x - X_TARGET)**2 + (y - X_TARGET)**2
            + 0.1*(x*a1)**2 + 0.22*(y*a2)**2 + 0.5*a1**2 + 0.1*a3**2)


def plot_designs(designs, mc_inputs, sig, output_dir):
    """Design points plus CDF/PDF comparison of the optimal designs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_xy, ax_cdf, ax_pdf) = plt.subplots(1, 3, figsize=(15, 4.4))
    markers = ('s', 'o', '^', 'D', 'v')
    grid = np.linspace(0, 60, 300)

    for (name, (x, y)), marker in zip(designs.items(), markers):
        f = f_true(x, y, *mc_inputs)
        color = ax_cdf.plot(np.sort(f), np.linspace(0, 1, f.size),
                            label=name)[0].get_color()
        ax_pdf.hist(f, bins=grid, density=True, alpha=0.4, color=color,
                    label=name)
        ax_xy.scatter(x, y, s=90, marker=marker, color=color,
                      label=f'{name} ({x:.2f}, {y:.2f})', zorder=5)

    ax_xy.scatter(X_TARGET, X_TARGET, s=40, c='k', marker='+')
    ax_xy.annotate('det. target', (X_TARGET, X_TARGET),
                   textcoords='offset points', xytext=(-10, 8), fontsize=8)
    ax_xy.set_xlabel('x')
    ax_xy.set_ylabel('y')
    ax_xy.set_xlim(0, 5)
    ax_xy.set_ylim(0, 5)
    ax_xy.set_title('Optimal designs')
    ax_xy.legend(fontsize=8)
    ax_xy.grid(alpha=0.3)

    ax_cdf.set_xlabel('f')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.set_xlim(0, 60)
    ax_cdf.set_title('CDFs of optimal designs')
    ax_cdf.legend(fontsize=8)
    ax_cdf.grid(alpha=0.3)

    ax_pdf.set_xlabel('f')
    ax_pdf.set_ylabel('PDF')
    ax_pdf.set_xlim(0, 60)
    ax_pdf.set_title('PDFs of optimal designs')
    ax_pdf.legend(fontsize=8)
    ax_pdf.grid(alpha=0.3)

    fig.tight_layout()
    output_file = os.path.join(output_dir, 'objectives_comparison.png')
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'\nwrote {output_file}')


if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))

    np.random.seed(0)  # fixes the resampling basis
    uq_data = interface.initialize(
        os.path.join(script_dir, 'input.yaml'),
        os.path.join(script_dir, 'run_matrix.dat')
    )
    sig = uq_data[8]

    designs = {'deterministic': optimize_deterministic(uq_data)}
    for objective in OBJECTIVES:
        designs[objective] = optimize_statistic(objective, uq_data)

    # UQPCE statistics of every design: each optimized design minimizes
    # its own column of this table
    stats = evaluate_designs(designs, uq_data)
    print(f'\n{"objective":<15}{"x*":>7}{"y*":>7}{"mean":>8}{"sd":>7}'
          f'{"ci_upper":>10}{"cvar_upper":>12}')
    for name, (x, y) in designs.items():
        mu, sd, ci, cvar = stats[name]
        print(f'{name:<15}{x:>7.3f}{y:>7.3f}{mu:>8.2f}{sd:>7.2f}'
              f'{ci:>10.2f}{cvar:>12.2f}')

    # independent Monte Carlo of the true function for the plots
    mc_inputs = sample_uncertain_inputs(500_000)
    plot_designs(designs, mc_inputs, sig, script_dir)
