"""
GMM-UQPCE: Robust vs Deterministic Optimization Comparison

Compares deterministic optimization (using mean parameter values) with
robust design optimization (minimizing mean + variance).
All uncertain variables have mean = 0 for simplicity.
"""

import os

# Disable OpenMDAO reports before importing
os.environ['OPENMDAO_REPORTS'] = '0'

import time

import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om
from scipy.stats import norm

from uqpce.mdao import interface
from uqpce.mdao.uqpcegroup import UQPCEGroup


class QuadraticFunction(om.ExplicitComponent):
    """
    f = (x - 2)^2 + (y - 2)^2 + 10*(x*a1)^2 + 5*(y*a2)^2 + 0.5*a1^2 + 0.1*a3^2

    With all uncertain parameters having mean=0:
    - Deterministic minimum at (2,2)
    - High variance when x,y are large due to uncertainty amplification
    """

    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)
        self.add_input('a1', shape=(n,))  # GMM
        self.add_input('a2', shape=(n,))  # Uniform
        self.add_input('a3', shape=(n,))  # Normal
        self.add_output('f', shape=(n,))

        arange = np.arange(n)
        self.declare_partials('f', ['x', 'y'])
        self.declare_partials('f', ['a1', 'a2', 'a3'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        x, y = inputs['x'], inputs['y']
        a1, a2, a3 = inputs['a1'], inputs['a2'], inputs['a3']

        outputs['f'] = ((x - 2)**2 + (y - 2)**2 +
                       10*(x*a1)**2 + 5*(y*a2)**2 +
                       0.5*a1**2 + 0.1*a3**2)

    def compute_partials(self, inputs, partials):
        x, y = inputs['x'], inputs['y']
        a1, a2, a3 = inputs['a1'], inputs['a2'], inputs['a3']

        partials['f', 'x'] = (2*(x - 2) + 20*x*a1**2).reshape(-1, 1)
        partials['f', 'y'] = (2*(y - 2) + 10*y*a2**2).reshape(-1, 1)
        partials['f', 'a1'] = 20*x**2*a1 + a1
        partials['f', 'a2'] = 10*y**2*a2
        partials['f', 'a3'] = 0.2*a3


def monte_carlo_uq(x, y, n_samples=100000):
    """Post-optimality UQ via Monte Carlo for deterministic point only"""
    np.random.seed(42)

    # GMM for a1 (zero mean)
    weights = [0.3, 0.5, 0.2]
    means = [-2.0, 0.0, 3.0]  # Weighted mean = 0.3*(-2) + 0.5*0 + 0.2*3 = 0
    stdevs = [0.3, 0.5, 0.4]

    components = np.random.choice(3, size=n_samples, p=weights)
    a1 = np.zeros(n_samples)
    for i in range(3):
        mask = components == i
        a1[mask] = np.random.normal(means[i], stdevs[i], np.sum(mask))

    a2 = np.random.uniform(-2.0, 2.0, n_samples)  # Mean = 0
    a3 = np.random.normal(0.0, 1.0, n_samples)    # Mean = 0

    f = ((x - 2)**2 + (y - 2)**2 +
         10*(x*a1)**2 + 5*(y*a2)**2 +
         0.5*a1**2 + 0.1*a3**2)

    return f


def create_plots(results, output_dir):
    """Create 3x2 visualization grid"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    x_det, y_det, f_det = results['det_point']
    x_rob, y_rob, f_rob = results['rob_point']
    f_det_mc = results['det_mc']
    f_rob_mc = results['rob_mc']

    # 1. GMM PDF with components
    ax = axes[0, 0]
    x_range = np.linspace(-4, 5, 1000)
    weights = [0.3, 0.5, 0.2]
    means = [-2.0, 0.0, 3.0]
    stdevs = [0.3, 0.5, 0.4]

    # Plot individual components
    for w, m, s in zip(weights, means, stdevs):
        pdf_comp = w * norm.pdf(x_range, m, s)
        ax.plot(x_range, pdf_comp, '--', alpha=0.5, label=f'w={w}, μ={m}, σ={s}')

    # Plot total GMM
    pdf_total = sum(w * norm.pdf(x_range, m, s) for w, m, s in zip(weights, means, stdevs))
    ax.plot(x_range, pdf_total, 'k-', linewidth=2, label='Total GMM')
    ax.fill_between(x_range, pdf_total, alpha=0.3)

    # Mark overall mean
    gmm_mean = sum(w*m for w, m in zip(weights, means))
    ax.axvline(gmm_mean, color='red', linestyle='-', linewidth=2, label=f'Mean={gmm_mean:.1f}')

    ax.set_xlabel('a1')
    ax.set_ylabel('PDF')
    ax.set_title('GMM Uncertain Variable Distribution (a1)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Objective function contour (deterministic)
    ax = axes[0, 1]
    X, Y = np.meshgrid(np.linspace(-1, 3, 50), np.linspace(-1, 3, 50))
    # All uncertain params have mean = 0
    Z = (X - 2)**2 + (Y - 2)**2

    contour = ax.contour(X, Y, Z, levels=15, alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.scatter(x_det, y_det, s=150, c='red', marker='s',
               label=f'Deterministic: ({x_det:.2f},{y_det:.2f}), f={f_det:.1f}', zorder=5)
    ax.scatter(x_rob, y_rob, s=150, c='blue', marker='o',
               label=f'Robust: ({x_rob:.2f},{y_rob:.2f}), f={f_rob:.1f}', zorder=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Objective Function $f$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Mean + Variance contour with log scale
    ax = axes[0, 2]
    Z_mv = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            f_samples = monte_carlo_uq(X[i,j], Y[i,j], 10000)
            Z_mv[i,j] = np.mean(f_samples) + np.var(f_samples)

    # Use log scale for better visualization
    Z_mv_log = np.log10(Z_mv + 1)  # Add 1 to avoid log(0)

    # Create contour plot with more levels for better resolution
    levels = np.linspace(Z_mv_log.min(), Z_mv_log.max(), 20)
    contour = ax.contour(X, Y, Z_mv_log, levels=levels, alpha=0.6, cmap='viridis')

    # Add contour labels with actual values (not log)
    fmt = {}
    for level, label in zip(contour.levels, contour.levels):
        fmt[level] = f'{10**label - 1:.0f}'  # Convert back from log scale
    ax.clabel(contour, inline=True, fontsize=7, fmt=fmt)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('log₁₀(μ + σ² + 1)', rotation=270, labelpad=15)

    ax.scatter(x_det, y_det, s=200, c='red', marker='s', label=f'Deterministic',
               edgecolor='white', linewidth=2, zorder=5)
    ax.scatter(x_rob, y_rob, s=200, c='blue', marker='o', label=f'Robust',
               edgecolor='white', linewidth=2, zorder=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'$f_{\mu} + f_{\sigma^2}$ Objective (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Deterministic point PDF
    ax = axes[1, 0]
    ax.hist(f_det_mc, bins=50, density=True, alpha=0.7, color='red', edgecolor='darkred')
    ax.axvline(np.mean(f_det_mc), color='darkred', linestyle='--',
               linewidth=2, label=f'μ={np.mean(f_det_mc):.1f}')
    ax.set_xlabel('f')
    ax.set_ylabel('PDF')
    ax.set_title(f'Distribution of $f$ at Deterministic Optimal Point ({x_det:.2f},{y_det:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Robust point PDF
    ax = axes[1, 1]
    ax.hist(f_rob_mc, bins=50, density=True, alpha=0.7, color='blue', edgecolor='darkblue')
    ax.axvline(np.mean(f_rob_mc), color='darkblue', linestyle='--',
               linewidth=2, label=f'μ={np.mean(f_rob_mc):.1f}')
    ax.set_xlabel('f')
    ax.set_ylabel('PDF')
    ax.set_title(f'Distribution of $f$ at Robust Optimal Point ({x_rob:.2f},{y_rob:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Results table
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')

    table_data = [
        ['Metric', 'Deterministic', 'Robust'],
        ['', '', ''],
        ['$x$', f'{x_det:.3f}', f'{x_rob:.3f}'],
        ['$y$', f'{y_det:.3f}', f'{y_rob:.3f}'],
        ['', '', ''],
        ['$f$', f'{f_det:.1f}', f'{f_rob:.1f}'],
        ['', '', ''],
        [r'Post-optimality  $f_{\mu}$', f'{np.mean(f_det_mc):.1f}', f'{np.mean(f_rob_mc):.1f}'],
        [r'Post-optimality $f_{\sigma^2}$', f'{np.var(f_det_mc):.1f}', f'{np.var(f_rob_mc):.1f}'],
        [r'Post-optimality $f_{\mu} + f_{\sigma^2}$', f'{np.mean(f_det_mc)+np.var(f_det_mc):.1f}',
         f'{np.mean(f_rob_mc)+np.var(f_rob_mc):.1f}'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Format header
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Comparison Results', fontweight='bold')

    plt.suptitle('Gaussian Mixture Model (GMM) Implementation in UQPCE: Robust vs Deterministic Optimization', fontsize=14)
    plt.tight_layout()

    # Save to the same directory as the script
    output_file = os.path.join(output_dir, 'gmm_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.show()


def run_deterministic_with_uqpce(input_file, matrix_file):
    """
    Deterministic optimization using UQPCE framework with deterministic flag.
    """
    print("\n" + "-"*40)
    print("DETERMINISTIC OPTIMIZATION (UQPCE framework)")
    print("-"*40)

    # Initialize UQPCE
    (var_basis, norm_sq, resampled_var_basis,
     aleatory_cnt, epistemic_cnt, resp_cnt, order, variables,
     sig, run_matrix) = interface.initialize(input_file, matrix_file)

    # Verify mean values
    print("Verifying uncertain parameter means:")
    for i, var in enumerate(variables):
        print(f"  {var.name}: {var.get_mean():.3f}")

    # Set up optimization problem
    prob_det = om.Problem(reports=None)

    # Add the quadratic function
    prob_det.model.add_subsystem('func', QuadraticFunction(vec_size=resp_cnt),
                                  promotes=['*'])

    # Use SNOPT and disable reports
    prob_det.driver = om.pyOptSparseDriver(optimizer='SNOPT')
    prob_det.driver.options['print_results'] = False

    prob_det.model.add_design_var('x', lower=-2, upper=3)
    prob_det.model.add_design_var('y', lower=-2, upper=3)

    # Just use f[0] as the objective since all elements are identical
    prob_det.model.add_objective('f', index=0)

    prob_det.setup()

    # Disable OpenMDAO report generation
    prob_det.set_solver_print(level=0)

    # Set initial guess
    prob_det.set_val('x', 0.0)
    prob_det.set_val('y', 0.0)

    # Use interface.set_vals with deterministic=True
    interface.set_vals(prob_det, variables, run_matrix, deterministic=True)

    # Run optimization
    prob_det.run_driver()

    x_det = prob_det.get_val('x')[0]
    y_det = prob_det.get_val('y')[0]
    f_det = prob_det.get_val('f')[0]

    print(f"Optimal point: x = {x_det:.3f}, y = {y_det:.3f}")
    print(f"Objective value: f = {f_det:.3f}")

    return x_det, y_det, f_det


def main():
    start_time = time.time()

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'input.yaml')
    matrix_file = os.path.join(script_dir, 'run_matrix.dat')

    print("\n" + "="*60)
    print("GMM-UQPCE: ROBUST vs DETERMINISTIC OPTIMIZATION")
    print("="*60)
    print(f"Loading files from: {script_dir}")
    print(f"  Input file: {input_file}")
    print(f"  Matrix file: {matrix_file}")
    print("\nAll uncertain parameters have mean = 0")

    # Check if files exist
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Cannot find input file: {input_file}")
    if not os.path.exists(matrix_file):
        raise FileNotFoundError(f"Cannot find matrix file: {matrix_file}")

    # ========== DETERMINISTIC OPTIMIZATION (UQPCE) ==========
    x_det, y_det, f_det = run_deterministic_with_uqpce(input_file, matrix_file)

    # ========== ROBUST OPTIMIZATION ==========
    print("\n" + "-"*40)
    print("ROBUST OPTIMIZATION (minimize μ + σ²)")
    print("-"*40)

    # Initialize UQPCE
    (var_basis, norm_sq, resampled_var_basis,
     aleatory_cnt, epistemic_cnt, resp_cnt, order, variables,
     sig, run_matrix) = interface.initialize(input_file, matrix_file)

    # Set up problem
    prob = om.Problem(reports=None)

    prob.model.add_subsystem('func', QuadraticFunction(vec_size=resp_cnt),
                              promotes_inputs=['x', 'y', 'a1', 'a2', 'a3'],
                              promotes_outputs=['f'])

    prob.model.add_subsystem(
        'UQPCE',
        UQPCEGroup(
            significance=sig, var_basis=var_basis, norm_sq=norm_sq,
            resampled_var_basis=resampled_var_basis, tail='both',
            epistemic_cnt=epistemic_cnt, aleatory_cnt=aleatory_cnt
        ),
        promotes_inputs=[('responses', 'f')],
        promotes_outputs=['mean', 'variance', 'mean_plus_var', 'resampled_responses']
    )

    # Use SNOPT and disable reports
    prob.driver = om.pyOptSparseDriver(optimizer='SNOPT')
    prob.driver.options['print_results'] = False

    prob.model.add_design_var('x', lower=-2, upper=3)
    prob.model.add_design_var('y', lower=-2, upper=3)
    prob.model.add_objective('mean_plus_var')

    prob.setup()

    # Disable OpenMDAO report generation
    prob.set_solver_print(level=0)

    prob.set_val('x', 0.0)
    prob.set_val('y', 0.0)

    # Use interface.set_vals WITHOUT deterministic flag for robust optimization
    interface.set_vals(prob, variables, run_matrix, deterministic=False)

    prob.run_driver()

    x_rob = prob.get_val('x')[0]
    y_rob = prob.get_val('y')[0]

    # Get the resampled responses from UQPCE (these are the Monte Carlo samples)
    f_rob_mc = prob.get_val('resampled_responses').flatten()

    # Get f value at robust point with mean parameters for comparison
    interface.set_vals(prob, variables, run_matrix, deterministic=True)
    prob.run_model()
    f_rob = prob.get_val('f')[0]

    print(f"Optimal point: x = {x_rob:.3f}, y = {y_rob:.3f}")
    print(f"Objective value at means: f = {f_rob:.3f}")
    print(f"UQPCE statistics from resampled responses:")
    print(f"  Mean: {np.mean(f_rob_mc):.3f}")
    print(f"  Variance: {np.var(f_rob_mc):.3f}")

    # ========== POST-OPTIMALITY UQ ==========
    print("\n" + "-"*40)
    print("POST-OPTIMALITY UQ")
    print("-"*40)

    # Only need Monte Carlo for deterministic point
    f_det_mc = monte_carlo_uq(x_det, y_det)

    # ========== RESULTS ==========
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Deterministic: x={x_det:.3f}, y={y_det:.3f}, f={f_det:.1f}")
    print(f"  Post-opt: μ={np.mean(f_det_mc):.1f}, σ²={np.var(f_det_mc):.1f}")

    print(f"\nRobust: x={x_rob:.3f}, y={y_rob:.3f}, f={f_rob:.1f}")
    print(f"  UQPCE: μ={np.mean(f_rob_mc):.1f}, σ²={np.var(f_rob_mc):.1f}")

    print(f"\nTrade-off: Δf = {f_rob - f_det:.1f} (robust accepts higher f for lower variance)")
    print(f"Total objective improvement: {(np.mean(f_det_mc)+np.var(f_det_mc)) - (np.mean(f_rob_mc)+np.var(f_rob_mc)):.1f}")

    # Create visualization
    results = {
        'det_point': (x_det, y_det, f_det),
        'rob_point': (x_rob, y_rob, f_rob),
        'det_mc': f_det_mc,
        'rob_mc': f_rob_mc  # Using UQPCE's resampled responses
    }
    create_plots(results, script_dir)

    print(f"\nTotal runtime: {time.time()-start_time:.1f}s")


if __name__ == '__main__':
    main()