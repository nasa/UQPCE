import openmdao.api as om

from uqpce.examples.paraboloid.paraboloid.paraboloid import Paraboloid
from uqpce.mdao import interface
from uqpce.mdao.uqpcegroup import UQPCEGroup

if __name__ == '__main__':

    #---------------------------------------------------------------------------
    #                               Input Files
    #---------------------------------------------------------------------------
    # First the .yaml input file must be created by the user. 
    #   This file contains information describing the distributions of uncertain variables
    #   and the problem setup for UQPCE such as the UQPCE order and number of oversamples. 
    #   See https://nasa.github.io/UQPCE/user-guide/inputs.html for reference
    # 
    input_file = 'input_paraboloid.yaml' # The UQPCE input file

    # After creating the .yaml file, the user must run UQPCE to generate the run matrix file.
    #   this file lists the locations where evaluation of the function will take place.
    #   to generate the run matrix file, execute the following command at the command line:
    #   python -m uqpce -i input_paraboloid.yaml --generate-samples
    #   this will create the run matrix file (run_matrix.data) which we then 
    #   rename to run_matrix_paraboloid.dat for this example. 
    matrix_file = 'run_matrix_paraboloid.dat' # The UQPCE run matrix file

    #---------------------------------------------------------------------------
    #                   Setting up for UQPCE and robust design
    #---------------------------------------------------------------------------
    # Now we can extract a variety of useful information from the two files to setup
    #   the rest of the problem.
    (
        var_basis, norm_sq, resampled_var_basis,
        aleatory_cnt, epistemic_cnt, resp_cnt, order, variables,
        sig, run_matrix
    ) = interface.initialize(input_file, matrix_file)

    prob = om.Problem()

    # ---------------------------------------------------------------------------
    #                               Model Setup 
    #---------------------------------------------------------------------------
    # We add a simple OpenMDAO group to evaluate the paraboloid function. 
    #   uncerta and uncertb both have distributions described in input_paraboloid.yaml.
    prob.model.add_subsystem(
        'parab',
        Paraboloid(vec_size=resp_cnt),
        promotes_inputs=['uncerta', 'uncertb', 'desx', 'desy'],
        promotes_outputs=['f_abxy']
    )

    #---------------------------------------------------------------------------
    #                               UQPCE Setup
    #---------------------------------------------------------------------------
    # Now that we have created the f_abxy output, we need the UQPCE group to evaluate the uncertainty.
    #   The inputs to UQPCE group can be confiugred to produce a number of outputs including the mean
    #   variance, and confidence interval of the output which in thise case is f_abxy.  
    prob.model.add_subsystem(
        'func',
        UQPCEGroup(
            significance=sig, var_basis=var_basis, norm_sq=norm_sq,
            resampled_var_basis=resampled_var_basis, tail='upper',
            epistemic_cnt=epistemic_cnt, aleatory_cnt=aleatory_cnt
        ),
        promotes_inputs=[('responses', 'f_abxy')], promotes_outputs=['ci_upper','mean','variance']
    )

    # OpenMDAO driver setup for optimization
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-9
    prob.driver.options['disp'] = True

    # Add design variables
    prob.model.add_design_var('desx', lower=-10, upper=10)
    prob.model.add_design_var('desy', lower=-10, upper=10)

    # Add objective
    prob.model.add_objective('ci_upper')

    prob.setup()

    # ---------------------------------------------------------------------------
    #                               Set Values
    #---------------------------------------------------------------------------
    # We must now initialize the values of the uncertaint variables contained in run_matrix_paraboloid.dat. 
    #   We do this with the interfance.set_vals() function from UQPCE. This is a robust way to ensure the
    #   values are set . Alternative approaches like using an IndeVarComp() at the top of the problem
    #   will not work in all cases such as when using UQPCE with the Dyreqt library.
    interface.set_vals(prob, variables, run_matrix)

    prob.set_val('desx', 7)
    prob.set_val('desy', -9)

    prob.run_driver()

    prob.model.list_inputs(print_arrays=True)
    prob.model.list_outputs(print_arrays=True)

    print('Design Variable desx is ', prob.get_val('desx'))
    print('Design Variable desy is ', prob.get_val('desy'))
    print('Mean of f_abxy is ', prob.get_val('mean'))
    print('Variance of f_abxy is ', prob.get_val('variance'))
    print('Upper Confidence Interval of f_abxy is ', prob.get_val('ci_upper'))