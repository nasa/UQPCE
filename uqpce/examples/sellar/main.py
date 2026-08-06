import os

import numpy as np
import openmdao.api as om

from uqpce.mdao import interface
from uqpce.mdao.uqpcegroup import MultiUQPCEGroup


class SellarDis1(om.ExplicitComponent):
    """
    Modified component for Discipline 1 containing derivatives and uncertain parameters
    """
    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']
        arange = np.arange(n)

        # Global Design Variable
        self.add_input('z', shape=(2,))

        # Local Design Variable
        self.add_input('x')

        # Coupling parameter
        self.add_input('y2', shape=(n,))

        # Uncertain input parameters
        self.add_input('a0', shape=(n,))
        self.add_input('a1', shape=(n,))
        self.add_input('a2', shape=(n,))

        # Coupling output
        self.add_output('y1', shape=(n,))

        # Declare partials
        self.declare_partials('y1', ['z', 'x'])
        self.declare_partials('y1', ['a0', 'a1', 'a2', 'y2'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y1 = z1**2*a0 + z2*a1 + x*a2 - 0.2*y2
        """
        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        a0 = inputs['a0']
        a1 = inputs['a1']
        a2 = inputs['a2']
        x = inputs['x']
        y2 = inputs['y2']

        outputs['y1'] = z1**2 * a0 + z2 * a1 + x*a2 - 0.2*y2

    def compute_partials(self, inputs, partials):

        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        x = inputs['x']
        a0 = inputs['a0']
        a1 = inputs['a1']
        a2 = inputs['a2']

        partials['y1', 'z'] = np.column_stack([2*z1*a0, a1])
        partials['y1', 'x'] = a2
        partials['y1', 'y2'] = -0.2
        partials['y1', 'a0'] = z1**2
        partials['y1', 'a1'] = z2
        partials['y1', 'a2'] = x

class SellarDis2(om.ExplicitComponent):
    """
    Component containing Discipline 2 with analytical derivatives.
    """
    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']
        arange = np.arange(n)

        # Global Design Variable
        self.add_input('z', shape=(2,))

        # Coupling parameter
        self.add_input('y1', shape=(n,))

        # Uncertain input parameters
        self.add_input('a0', shape=(n,))
        self.add_input('a1', shape=(n,))

        # Coupling output
        self.add_output('y2', shape=(n,))

        # Declare partials
        self.declare_partials('y2', ['z'])
        self.declare_partials('y2', ['a0', 'a1', 'y1'], rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1*a0 + z2*a1
        """
        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        y1 = inputs['y1']
        a0 = inputs['a0']
        a1 = inputs['a1']

        outputs['y2'] = y1**.5 + z1 * a0 + z2 * a1

    def compute_partials(self, inputs, partials):
        y1 = inputs['y1']
        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        a0 = inputs['a0']
        a1 = inputs['a1']

        partials['y2', 'z'] = np.column_stack([a0, a1])
        partials['y2', 'y1'] = 0.5 * y1**(-0.5)
        partials['y2', 'a0'] = z1
        partials['y2', 'a1'] = z2

class SellarObj(om.ExplicitComponent):
    """
    Component containing objective function for Sellar problem
    min: x**2 + z2 + y1 + exp(-y2)
    """
    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']
        arange = np.arange(n)
        # Coupling parameter
        self.add_input('x')
        self.add_input('z', shape=(2,))
        self.add_input('y1', shape=(n,))
        self.add_input('y2', shape=(n,))
        self.add_input('a0', shape=(n,))
        self.add_input('a1', shape=(n,))
        self.add_input('a2', shape=(n,))
        self.add_input('a3', shape=(n,))

        # Coupling output
        self.add_output('obj', shape=(n,))

        # Declare partials
        self.declare_partials('obj', ['x', 'z'])
        self.declare_partials('obj', ['y1', 'y2', 'a0', 'a1', 'a2', 'a3'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):

        z2 = inputs['z'][1]
        x = inputs['x']
        y1 = inputs['y1']
        y2 = inputs['y2']
        a0 = inputs['a0']
        a1 = inputs['a1']
        a2 = inputs['a2']
        a3 = inputs['a3']

        outputs['obj'] = x**2*a2 + z2*(a1) + y1*a3 + np.exp(-y2)*(a0)

    def compute_partials(self, inputs, partials):
        n = self.options['vec_size']

        z2 = inputs['z'][1]
        x = inputs['x']
        y1 = inputs['y1']
        y2 = inputs['y2']
        a0 = inputs['a0']
        a1 = inputs['a1']
        a2 = inputs['a2']
        a3 = inputs['a3']

        partials['obj', 'z'] = np.column_stack([np.zeros(n), a1])
        partials['obj', 'x'] = 2*x*a2
        partials['obj', 'y1'] = a3
        partials['obj', 'y2'] = -np.exp(-y2)*a0
        partials['obj', 'a0'] = np.exp(-y2)
        partials['obj', 'a1'] = z2
        partials['obj', 'a2'] = x**2
        partials['obj', 'a3'] = y1

class SellarConst1(om.ExplicitComponent):
    """
    Component containing constraints for component 1
    """
    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']
        arange = np.arange(n)
        # Coupling parameter
        self.add_input('y1', shape=(n,))

        # Coupling output
        self.add_output('const1', shape=(n,))

        # Declare partials
        self.declare_partials('const1', ['y1'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):

        y1 = inputs['y1']

        outputs['const1'] = 8 - y1

    def compute_partials(self, inputs, partials):

        partials['const1', 'y1'] = -1

class SellarConst2(om.ExplicitComponent):
    """
    Component containing constraints for component 1
    """
    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']
        arange = np.arange(n)
        # Coupling parameter
        self.add_input('y2', shape=(n,))

        # Coupling output
        self.add_output('const2', shape=(n,))

        # Declare partials
        self.declare_partials('const2', ['y2'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):

        y2 = inputs['y2']

        outputs['const2'] = y2 - 12.

    def compute_partials(self, inputs, partials):

        partials['const2', 'y2'] = 1

class SellarMDA(om.Group):
    """
    Group containing the Sellar MDA.
    """
    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        vec_size = self.options['vec_size']

        self.add_subsystem('d1', SellarDis1(vec_size=vec_size),
                           promotes_inputs=['x', 'z', 'y2', 'a0', 'a1', 'a2'],
                           promotes_outputs=['y1'])
        self.add_subsystem('d2', SellarDis2(vec_size=vec_size),
                           promotes_inputs=['z', 'y1', 'a0', 'a1'],
                           promotes_outputs=['y2'])


        self.nonlinear_solver = om.NewtonSolver(
            maxiter=30,                           # More iterations allowed
            rtol=1e-12,                            # More reasonable tolerance
            atol=1e-12,                           # Absolute tolerance
            iprint=True,                          # Print iteration info
            debug_print=True,                     # Print debug info when failing
            err_on_non_converge=False,            # Don't error out on non-convergence
            solve_subsystems=True,                # Solve subsystems at each iteration
            stall_limit=4,                        # Stop if not improving
        )

        self.linear_solver = om.DirectSolver()

    def guess_nonlinear(self, inputs, outputs, residuals):

        vec_size = self.options['vec_size']

        outputs['y1'] = 3.*np.ones(vec_size)
        outputs['y2'] = 24.*np.ones(vec_size)
        self.run_apply_nonlinear()

if __name__ == '__main__':

    #---------------------------------------------------------------------------
    #                               Input Files
    #---------------------------------------------------------------------------

    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_yaml = 'input.yaml'
    relative_matrix = 'run_matrix.dat'
    input_file = os.path.join(script_dir, relative_yaml)
    matrix_file  = os.path.join(script_dir, relative_matrix)

    #---------------------------------------------------------------------------
    #                   Setting up for UQPCE and design under uncertainty
    #---------------------------------------------------------------------------

    (
        var_basis, norm_sq, resampled_var_basis,
        aleatory_cnt, epistemic_cnt, resp_cnt, order, variables,
        sig, run_matrix
    ) = interface.initialize(input_file, matrix_file)

    prob = om.Problem()

    #---------------------------------------------------------------------------
    #                   Add Subsystems to Problem
    #---------------------------------------------------------------------------

    prob.model.add_subsystem(
        'MDA',
        SellarMDA(vec_size=resp_cnt),
        promotes_inputs=['z', 'x', 'a0', 'a1', 'a2'],
        promotes_outputs=['y1', 'y2']
    )

    prob.model.add_subsystem(
        'con_cmp1',
        SellarConst1(vec_size=resp_cnt),
        promotes_inputs=['y1'],
        promotes_outputs=['const1']
    )

    prob.model.add_subsystem(
        'con_cmp2',
        SellarConst2(vec_size=resp_cnt),
        promotes_inputs=['y2'],
        promotes_outputs=['const2']
    )

    prob.model.add_subsystem(
        'obj_func',
        SellarObj(vec_size=resp_cnt),
        promotes_inputs=['z', 'x', 'y1', 'y2', 'a0', 'a1', 'a2', 'a3'],
        promotes_outputs=['obj']
    )

    #---------------------------------------------------------------------------
    #                   Add UQPCE Group to Problem
    #---------------------------------------------------------------------------
    prob.model.add_subsystem(
        'UQPCE',
        MultiUQPCEGroup(
            significance=sig, var_basis=var_basis, norm_sq=norm_sq,
            resampled_var_basis=resampled_var_basis, tail='both',
            epistemic_cnt=epistemic_cnt, aleatory_cnt=aleatory_cnt,
            uncert_list=['const1', 'const2', 'obj']
        ),
        promotes_inputs=['const1', 'const2', 'obj'],
        promotes_outputs=['const1:resampled_responses', 'const1:ci_lower', 'const1:ci_upper', 'const1:mean',
                          'const2:resampled_responses', 'const2:ci_lower', 'const2:ci_upper', 'const2:mean',
                          'obj:resampled_responses', 'obj:ci_lower', 'obj:ci_upper', 'obj:mean', 'obj:mean_plus_var']
    )
    #---------------------------------------------------------------------------
    #                   Setting up the OpenMDAO Problem
    #---------------------------------------------------------------------------

    # Set up driver
    prob.driver = om.pyOptSparseDriver(optimizer='SNOPT')
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-8
    prob.driver.opt_settings['Iterations limit'] = 50

    # Initial guesses
    prob.model.set_input_defaults('x', -5)
    prob.model.set_input_defaults('z', np.array([2, 15]))

    # Add design variables and bounds
    prob.model.add_design_var('x', lower=-20, upper=5, ref=1)
    prob.model.add_design_var('z', lower=[-5., -5.0], upper=[10., 25.], ref=1)

    # Assign objective function and constraints in UQPCE formatting
    obj = 'obj:mean'
    C1 = 'const1:ci_lower'
    C2 = 'const2:ci_lower'

    prob.model.add_objective(obj)
    prob.model.add_constraint(C1, lower=0, ref0=1, ref=2) #Use ref0=1 to avoid poor scaling
    prob.model.add_constraint(C2, lower=0, ref0=1, ref=2)
    prob.model.add_constraint('y1', lower=0.01)

    prob.setup(force_alloc_complex=True)
    om.n2(prob, show_browser=False)

    # Use the UQPCE interface to set the uncertain parameters from the run matrix
    interface.set_vals(prob, variables, run_matrix)

    #---------------------------------------------------------------------------
    #                   Run the Problem and Print Results
    #---------------------------------------------------------------------------

    prob.run_driver()

    print('Design Variable x ', prob.get_val('x'))
    print('Design Variable z ', prob.get_val('z'))

    print(f'Constraint {C1}', prob.get_val(C1))
    print(f'Constraint {C2}', prob.get_val(C2))

    print(f'Objective {obj} is', prob.get_val(obj))