#!/usr/bin/env python

"""

Notices:
Copyright 2020 United States Government as represented by the Administrator of 
the National Aeronautics and Space Administration. All Rights Reserved.

Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF 
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED 
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY 
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR 
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR 
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE 
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN 
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, 
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS 
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY 
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, 
IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
 
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE 
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY 
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY 
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, 
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S 
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE 
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY 
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR 
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS 
AGREEMENT.

"""

import argparse
from builtins import getattr
import configparser
from datetime import datetime
import math
import os
import warnings
from warnings import warn

import numpy as np
from sympy.utilities.lambdify import lambdify
from sympy import symbols

from PCE_Codes.UQPCE import (
    _warn, read_file, get_str_vars, user_function, switch_backend,
    create_total_sobols, check_directory, calc_PRESS_res, DataSet,
    SurrogateModel, Graphs, ProbabilityBoxes, MatrixSystem)

if __name__ == '__main__':

    # taking in optional commandline args
    parser = argparse.ArgumentParser(
        description='Uncertainty Quantification Polynomial Chaos Expansion',
        prog='UQPCE',
        argument_default=argparse.SUPPRESS
    )

    warnings.formatwarning = _warn

    out_file = 'output.dat'
    sobol_out = 'sobol.dat'
    coef_out = 'coefficients.dat'

    output_directory = 'outputs'

    # defaults
    input_file = 'input.yaml'
    matrix_file = 'run_matrix.dat'
    results_file = 'results.dat'

    verification_results_file = 'verification_results.dat'
    verification_matrix_file = 'verification_run_matrix.dat'

    case = None
    version_num = '0.2.5'
    backend = 'TkAgg'

    version = False
    verbose = False
    verify = False
    plot = False
    plot_stand = False
    generate_samples = False
    user_func = None  # can put a string function here (ex. 'x0*x1+x2')

    track_convergence_off = False  # track and plot confidence interval unless
                                    # turned off (True)
    epist_sub_samp_size = 25  # convergence; CI tracking used unless flag
    aleat_sub_samp_size = 5000

    order = 2  # order of PCE expansion
    over_samp_ratio = 2
    significance = 0.05
    conv_threshold_percent = 0.0005

    # number of times to iterate for each variable type
    epist_samp_size = 125
    aleat_samp_size = 25000

    # optional arguments with a default file option
    parser.add_argument('-i', '--input-file',
        help=f'file containing variables (default: {input_file})')
    parser.add_argument('-m', '--matrix-file',
        help='file containing matrix elements '
        f'(default: {matrix_file})')
    parser.add_argument('-r', '--results-file',
        help=f'file containing results (default: {results_file})')
    parser.add_argument('--verification-results-file',
        help=f'file containing verification results '
        f'(default: {verification_results_file})')
    parser.add_argument('--verification-matrix-file',
        help=f'file containing verification matrix elements '
        f'(default: {verification_matrix_file})')
    parser.add_argument('--output-directory',
        help=f'directory that the outputs will be put in '
        f'(default: {output_directory})')

    parser.add_argument('-c', '--case',
        help=f'case of input data (default: {case})')
    parser.add_argument('-s', '--significance',
        help=f'significance level of the confidence interval (default: '
        f'{significance})', type=float)
    parser.add_argument('-o', '--order',
        help=f'order of PCE expansion (default: {order})', type=int)
    parser.add_argument('-f', '--user-func',
        help='allows the user to specify the analytical function for the '
        f'data (default: {user_func})')
    parser.add_argument('--over-samp-ratio',
        help='over sampling ratio; factor for how many points to be used '
        f'in calculations (default: {over_samp_ratio})', type=float)
    parser.add_argument('-b', '--backend',
        help='the backend that will be used for Matplotlib plotting '
        f'(default: {backend})')
    parser.add_argument('--aleat-sub-samp-size',
        help='the number of samples to check the new high and low intervals at '
        f'for each individual curve (default: {aleat_sub_samp_size})', type=int)
    parser.add_argument('--epist-sub-samp-size',
        help='the number of curves to check the new high and low intervals at '
        f'for a set of curves (default: {epist_sub_samp_size})', type=int)
    parser.add_argument('--conv-threshold-percent',
        help='the percent of the response mean to be used as a threshold for '
        f'tracking convergence (default: {conv_threshold_percent})', type=float)
    parser.add_argument('--epist-samp-size',
        help='the number of times to sample for each varaible with epistemic '
        f'uncertainty (default: {epist_samp_size})', type=int)
    parser.add_argument('--aleat-samp-size',
        help='the number of times to sample for each varaible with aleatory '
        f'uncertainty (default: {aleat_samp_size})', type=int)

    # optional flags while running module
    parser.add_argument('-v', '--version',
        help='displays the version of the software', action='store_true')
    parser.add_argument('--verbose',
        help='increase output verbosity', action='store_true')
    parser.add_argument('--verify',
        help='allows verification of results', action='store_true')
    parser.add_argument('--plot', help='generates factor vs response plots, '
                        'pbox plot, and error plots', action='store_true')
    parser.add_argument('--plot-stand',
        help='plots standardized variables', action='store_true')
    parser.add_argument('--track-convergence-off', help='allows users to '
        'converge on confidence interval until the change between the two '
        'iterations is less than the threshold', action='store_true')
    parser.add_argument('--generate-samples', help='generates the samples used '
                        'for all variables according to the parameters '
                        'provided in the input file', action='store_true')

    args = parser.parse_args()

#******************************************************************************

    init = DataSet()  # setting up the problem via the input files

    input_file = getattr(args, 'input_file', input_file)  # read input first
    var_list, settings = init.read_var_input(input_file, args, order, verbose)
    var_count = len(var_list)

    if 'MPLBACKEND' in os.environ:
        backend = os.environ['MPLBACKEND']
        try:  # if the backend hasn't been set in input file
            if settings['backend'].upper() != backend.upper():
                warn(f'backend changing from OS env {backend} to input file '
                     f'{settings["backend"]}.\n')
                backend = settings['backend']
        except KeyError:
            pass

    try:
        locals().update(settings)  # updates the local variables with input file
    except TypeError:
        pass  # no commands in the input file

    arg_options = [parser._actions[arg_num].dest  # list of all possible cmnd line
                   for arg_num in range(1, len(parser._actions))]  # arg names

    try:
        for key in list(settings):
            try:  # do values from input file agree with values from commandline
                if key not in arg_options:
                    warn(f'Setting {key} in {input_file} is not a valid '
                         'setting.\n')
                if settings[key] != getattr(args, key):
                    warn(f'{key} changing from {settings[key]} to '
                         f'{getattr(args, key)}.\n')

            except AttributeError:
                pass
    except TypeError:
        pass  # no commands in the input file

    # setting the arguments
    matrix_file = getattr(args, 'matrix_file', matrix_file)
    results_file = getattr(args, 'results_file', results_file)

    verification_results_file = getattr(args, 'verification_results_file',
                                        verification_results_file)
    verification_matrix_file = getattr(args, 'verification_matrix_file',
                                       verification_matrix_file)
    output_directory = getattr(args, 'output_directory', output_directory)

    case = getattr(args, 'case', case)
    version = getattr(args, 'version', version)
    verbose = getattr(args, 'verbose', verbose)
    verify = getattr(args, 'verify', verify)
    plot = getattr(args, 'plot', plot)
    plot_stand = getattr(args, 'plot_stand', plot_stand)
    track_convergence_off = getattr(args, 'track_convergence_off', \
                                     track_convergence_off)
    generate_samples = getattr(args, 'generate_samples', generate_samples)

    significance = getattr(args, 'significance', significance)
    order = getattr(args, 'order', order)
    user_func = getattr(args, 'user_func', user_func)
    over_samp_ratio = getattr(args, 'over_samp_ratio', over_samp_ratio)
    backend = getattr(args, 'backend', backend)
    aleat_sub_samp_size = getattr(args, 'aleat_sub_samp_size',
                                  aleat_sub_samp_size)
    epist_sub_samp_size = getattr(args, 'epist_sub_samp_size',
                                  epist_sub_samp_size)
    conv_threshold_percent = getattr(args, 'conv_threshold_percent',
                                           conv_threshold_percent)
    epist_samp_size = getattr(args, 'epist_samp_size', epist_samp_size)
    aleat_samp_size = getattr(args, 'aleat_samp_size', aleat_samp_size)

#******************************************************************************
    if order > 4:
        warn('The results may be less valid for orders as high as 5th order. '
             'Most cases do not require an order above 4.')

    required_sample_count = math.factorial(order + var_count)\
                        / (math.factorial(order) * math.factorial(var_count))
    gen_samp_size = int(np.ceil(required_sample_count * over_samp_ratio))

    if generate_samples:
        for variable in var_list:
            variable.generate_samples(gen_samp_size)
    else:
        init.read_var_vals(matrix_file, 'vals')

    # user_func required when generate_samples, but can use user_func
    if (user_func is None) and generate_samples:  # w/o generate_samples
        print ('Samples will be created according to the input distributions, '
               "but the corresponding \noutputs won't be generated unless the "
               "'--user-func' flag with an equation is also used\n")

    switch_backend(backend)
    if verbose:
        print(f'Using MatPlotLib backend {backend}\n')

    if version:
        print(f'UQPCE: version {version_num}\n')

    time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    time_start = datetime.strptime(time_start, '%Y-%m-%d %H:%M:%S.%f')

    if verify:
        R_verify = read_file(verification_results_file)
        verify_keys = list(R_verify)
        init.read_var_vals(verification_matrix_file, 'verify_vals')

    for variable in var_list:
        variable.standardize('vals', 'std_vals')
        variable.check_distribution()

    if not generate_samples:
        if case is not None:
            output_directory = case + '_' + output_directory
        output_directory = check_directory(output_directory, verbose)

    if generate_samples:
        with open('run_matrix_generated.dat', 'w') as mat_gen:
            for i in range(gen_samp_size):
                for variable in var_list:
                    mat_gen.write(''.join((str(variable.vals[i]), '    ')))
                mat_gen.write('\n')
        if user_func is None:
            if verbose:
                print(f'Generated samples are in run_matrix_generated.dat\n')
            exit()

    if user_func is not None:  # user func given/samples generated
        if verbose:
            print(f'Generating the results from function {user_func}\n')
        results = user_function(user_func, var_list)
    else:
        results = read_file(results_file)

    results_keys = list(results)
    results_column_count = len(results_keys)

    response_count = len(results_keys)
    if verbose and (response_count > 1):
        print(f'{response_count} columns found in the results file\n\n'
              f'{response_count} models will be constructed\n')

    for key_ind in range(results_column_count):  # multiple columns of results

        if response_count > 1:  # results file has multiple columns
            output_col_directory = output_directory + '/response_'\
                                     +results_keys[key_ind]
            output_col_directory = check_directory(output_col_directory,
                                                   verbose)
        else:
            output_col_directory = output_directory

        if user_func is not None:
            with open('results_generated.dat', 'w') as res_gen:
                gen_vals = results[list(results)[0]]
                res_gen.write('\n'.join((str(i) for i in gen_vals)))
                exit()

        graph_directory = output_col_directory + '/graphs'
        graph_directory = check_directory(graph_directory, verbose)

        results_iteration = results[results_keys[key_ind]]
        over_samp_ratio = len(results_iteration) / required_sample_count
        locals().update({'over_samp_ratio':over_samp_ratio})

        if over_samp_ratio < 1.25:
            warn('The oversampling ratio is {:.5}. Consider using at least '
            '{} samples for a more accurate model.'.format(over_samp_ratio,
                                    int(np.ceil(1.25 * required_sample_count))))

        if verify:
            R_verify_iteration = R_verify[verify_keys[key_ind]]

        matrix = MatrixSystem(results_iteration, var_list)
        matrix.verbose = verbose

        if verbose:
            print('Constructing surrogate model\n\n'
                  'Building norm-squared matrix\n')
        min_model_size, inter_matrix, norm_sq = matrix.form_norm_sq(order)

        if verbose:
            print('Assembling Psi matrix\n')
        var_basis_vect_symb = matrix.build()

        if verbose:
            print('Evaluating Psi matrix\n')
        var_basis_sys_eval = matrix.evaluate()

        if verbose:
            print('Solving system\n')
        matrix_coeffs, var_basis = matrix.solve()

        var_list_symb = [''] * var_count
        for j in range(var_count):
            var_list_symb[j] = symbols(f'x{j}')

        var_basis_vect_func = lambdify(
            (var_list_symb,), matrix.var_basis_vect_symb, modules='numpy'
        )

        model = SurrogateModel(results_iteration, matrix_coeffs)
        model.verbose = verbose

        if verbose:
            print('Surrogate model construction complete\n')

        sigma_sq, resp_mean = model.calc_var(norm_sq)
        error, pred = model.calc_error(var_basis)
        err_mean = model.calc_mean_error(error)
        signal_to_noise = sigma_sq / err_mean

        sobols = model.get_sobols(norm_sq, min_model_size)
        total_sobol_str = create_total_sobols(var_count, inter_matrix, sobols)
        mean_sq, hat_matrix, shapiro_results = model.check_normality(
            var_basis_sys_eval, min_model_size, significance, graph_directory)

        press = calc_PRESS_res(var_basis, results_iteration, var_basis_vect_func, var_list)
        press_sr = f'The PRESS value for this model is {press:.5}.\n'

        if verbose:
            for i in range(var_count):
                print('Total Sobol {} = {:.5}'.format(var_list[i].name,
                                                      total_sobol_str[i]))

            print('\nMean of response {:.5}\n'
                  'Variance of response {:.5}\n'
                  'Mean error of surrogate {:.5}\n'
                  'Signal to noise ratio {:.5}\n'
                  .format(resp_mean, sigma_sq, err_mean, signal_to_noise))

            print(press_sr)

        if plot:
            graph = Graphs(plot_stand)
            graph.verbose = verbose
            graph.standardize = plot_stand
            graph.factor_plots(graph_directory, var_list, pred, 'Predicted')
            graph.factor_plots(graph_directory, var_list, error, 'Error')
            graph.error_vs_pred(graph_directory, error, pred,
                                'Error vs Predicted')

        pbox = ProbabilityBoxes(var_list)
        pbox.verbose = verbose
        pbox.plot = plot
        pbox.track_convergence_off = track_convergence_off
        pbox.aleat_sub_samp_size = aleat_sub_samp_size
        pbox.epist_sub_samp_size = epist_sub_samp_size

        pbox.generate_variable_str()

        pbox.count_epistemic()

        if verbose:
            print('Generating resampling values\n')
        pbox.generate_epistemic_samples(epist_samp_size, aleat_samp_size)
        pbox.generate_aleatory_samples()

        if verbose:
            print('Resampling the surrogate model\n')
        results_eval, convergence_message = pbox.evaluate_surrogate(
                                              var_basis_vect_symb, significance,
                                              resp_mean, matrix_coeffs,
                                              conv_threshold_percent,
                                              graph_directory)

        conf_int_low, conf_int_high = pbox.generate(results_eval, significance,
                                                    graph_directory)

        if verbose:
            print('Writing output data\n')

        # writing the labeled output files for sobols and matrix_coeffs coeficients
        str_vars = get_str_vars(inter_matrix)
        with open(output_col_directory + '/' + sobol_out, 'w') as sob:
            for line in range(len(sobols)):
                sob.write('{:40s}    {:.5}\n'.format(str_vars[line],
                                                     sobols[line]))
            sob.write('\n\n')
            for j in range(var_count):
                sob.write('Total Sobol {} = {:.5}\n'.format(var_list[j].name,
                                                            total_sobol_str[j]))

        with open(output_col_directory + '/' + coef_out, 'w') as coef:
            coef.write('{:40s}    {:.5}\n'.format('intercept', matrix_coeffs[0]))
            for line in range(len(matrix_coeffs) - 1):
                coef.write('{:40s}    {:.5}\n'.format(str_vars[line], \
                    matrix_coeffs[line + 1]))

        if verify:
            for variable in var_list:
                variable.standardize('verify_vals', 'std_verify_vals')
            R_verify_pred, var_basis_sys_eval_verify = model.verify(
                                                       var_basis_vect_symb,
                                                       var_list,
                                                       len(R_verify_iteration))
            err_verify = model.calc_difference(R_verify_iteration,
                                               R_verify_pred)
            err_verify_mean = model.calc_mean_error(err_verify)

            if plot:
                graph.factor_plots(graph_directory, var_list, R_verify_pred,
                                   'Verify Predicted', verify=True)
                graph.factor_plots(graph_directory, var_list, err_verify,
                                   'Verify Error', verify=True)
                graph.error_vs_pred(graph_directory, err_verify,
                                    R_verify_pred, 'Verify Error vs Predicted')

            if verbose:
                print('Mean error between model and verification '
                      '{:.5}\n\nThe ratio of verification error to surrogate '
                      'model error is {:.5}\n'.format(err_verify_mean,
                                                      err_verify_mean / err_mean))

        time_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        time_end = datetime.strptime(time_end, '%Y-%m-%d %H:%M:%S.%f')
        time_total = time_end - time_start

        with open(output_col_directory + '/' + out_file, 'w') as fi:
            fi.write(f'###          UQPCE v{version_num} Output\n'
                     f'###  Analysis of case: {case}\n'
                     f'###  Analysis started: {time_start}\n'
                     f'###  Analysis finished: {time_end}\n'
                     f'###  Total compute time: {time_total}\n'
                     '-----------------------------------------'
                     '----------------------------------------------\n'
                    'Mean of response {:.5}\n'
                    'Variance of response {:.5}\n'
                    'Mean error of surrogate {:.5}\n'
                    'Signal to noise ratio {:.5}\n'
                    '{:.1f}% Confidence Interval on Response [{:.5} ,'
                    ' {:.5}]\n'.format(resp_mean, sigma_sq, err_mean,
                    signal_to_noise, 100 - significance * 100, conf_int_low,
                    conf_int_high))

            fi.write('\n'.join((shapiro_results, convergence_message)))
            fi.write('---------------------------------------------------------'
                     '------------------------------\n')

            fi.write(press_sr)
            fi.write('---------------------------------------------------------'
                     '------------------------------\n')

            if verify:
                fi.write('\nMean error between model and verification '
                      '{:.5}\n\nThe ratio of verification error to surrogate model '
                      'error is {:.5}\n'.format(err_verify_mean, err_verify_mean / err_mean))
            fi.write('\nThe settings used to generate these '
                     'results are:\n')
            for i in range(len(arg_options)):
                arg_opt = arg_options[i]
                fi.write(''.join((arg_opt, ': ', str(locals().get(arg_opt)),
                                  '\n')))
            fi.write('---------------------------------------------------------'
                     '------------------------------\n')
            fi.write('\nThe input file used is:\n')
            with open(input_file) as in_file:
                input_file_lines = in_file.readlines()
            fi.writelines(input_file_lines)

    if verbose:
        print('Analysis Complete\n')
