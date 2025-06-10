import openmdao.api as om
from paraboloid.paraboloid import Paraboloid
from uqpce.mdao.uqpcegroup import UQPCEGroup
from uqpce.mdao import interface

if __name__ == '__main__':

    #---------------------------------------------------------------------------
    #                               Input Files
    #---------------------------------------------------------------------------
    input_file = 'input_paraboloid.yaml' # The UQPCE input file; see UQPCE docs
    matrix_file = 'run_matrix_paraboloid.dat' # The UQPCE run matrix file; see UQPCE docs

    #---------------------------------------------------------------------------
    #                   Setting up for UQPCE and robust design
    #---------------------------------------------------------------------------
    (
        var_basis, norm_sq, resampled_var_basis, 
        aleatory_cnt, epistemic_cnt, resp_cnt, order, variables, 
        sig, run_matrix
    ) = interface.initialize(input_file, matrix_file)

    prob = om.Problem()

    prob.model.add_subsystem(
        'parab', 
        Paraboloid(vec_size=resp_cnt), 
        promotes_inputs=['uncerta', 'uncertb', 'desx', 'desy'], 
        promotes_outputs=['f_abxy']
    )

    prob.model.add_subsystem(
        'func',
        UQPCEGroup(
            significance=sig, var_basis=var_basis, norm_sq=norm_sq, 
            resampled_var_basis=resampled_var_basis, tail='upper',
            epistemic_cnt=epistemic_cnt, aleatory_cnt=aleatory_cnt,
            uncert_list=['f_abxy'], tanh_omega=5e-4
        ),
        promotes_inputs=['f_abxy'], promotes_outputs=['f_abxy:ci_upper']
    )

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-9
    prob.driver.options['disp'] = True

    # Add design variables
    prob.model.add_design_var('desx', lower=-10, upper=10)
    prob.model.add_design_var('desy', lower=-10, upper=10)

    # Add objective
    prob.model.add_objective('f_abxy:ci_upper')

    prob.setup()

    interface.set_vals(prob, variables, run_matrix)

    prob.set_val('desx', 7)
    prob.set_val('desy', -9)

    prob.run_driver()

    prob.model.list_inputs(print_arrays=True)
    prob.model.list_outputs(print_arrays=True)

    print('Design Variable desx is ', prob.get_val('desx'))
    print('Design Variable desy is ', prob.get_val('desy'))
