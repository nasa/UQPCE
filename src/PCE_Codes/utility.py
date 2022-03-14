import argparse
from builtins import getattr
from collections import namedtuple
from datetime import datetime
from warnings import warn

try:
    from mpi4py.MPI import COMM_WORLD as MPI_COMM_WORLD
    import numpy as np
    from scipy.stats import shapiro
except:
    warn('Ensure that all required packages are installed.')
    exit()

from PCE_Codes._helpers import (
    _warn, get_str_vars, user_function, switch_backend, create_total_sobols,
    check_directory, calc_difference, calc_mean_err, check_error_trends,
    check_error_magnitude
)
from PCE_Codes.graphs import Graphs
from PCE_Codes.io import (
    read_file, write_sobols, write_coeffs, write_outputs, write_gen_samps,
    write_gen_resps, DataSet
)
from PCE_Codes.pbox import ProbabilityBoxes
from PCE_Codes.report import UQPCEReport
from PCE_Codes.stats.statistics import (
    calc_R_sq, calc_R_sq_adj, calc_pred_conf_int, calc_coeff_conf_int,
    get_sobol_bounds, calc_term_count, calc_var_conf_int
)
from PCE_Codes.uqpce import SurrogateModel, MatrixSystem

ProgSettings = namedtuple(
    'ProgSettings',
    (
        'out_file', 'sobol_out', 'coeff_out', 'input_file',
        'matrix_file', 'results_file', 'verification_results_file',
        'verification_matrix_file', 'case', 'version_num', 'backend', 'verbose',
        'verify', 'version', 'plot', 'plot_stand', 'generate_samples',
        'model_conf_int', 'stats', 'report', 'user_func',
        'track_convergence_off', 'epist_sub_samp_size', 'aleat_sub_samp_size',
        'order', 'over_samp_ratio', 'significance',
        'conv_threshold_percent', 'epist_samp_size', 'aleat_samp_size',
        'bound_limits',
        'arg_options', 'var_thresh', 'output_directory', 'resp_order',
        'seed', 'verify_over_samp_ratio'
    ),
    defaults=[-np.inf] * 37
)

IterSettings = namedtuple(
    'IterSettings',
    (
        'resp_verify', 'verify_keys', 'results', 'results_keys', 'resp_count',
        'var_count', 'var_list', 'max_order', 'graph_dir', 'resp_iter',
        'sobol_str', 'matrix_coeffs', 'norm_sq', 'var_basis_sys_eval',
        'inter_matrix', 'min_model_size', 'var_basis_vect_symb', 'sobol_bounds',
        'mean_sq', 'shapiro_results', 'convergence_message', 'approx_mean',
        'mean_uncert', 'hat_matrix', 'conf_int_low', 'conf_int_high', 'resp_idx',
        'err_mean', 'time_start', 'coeff_uncert', 'output_col_directory',
        'resp_mean', 'sigma_sq', 'signal_to_noise', 'error', 'pred',
        'resp_iter_count', 'err_ratio', 'verify_str', 'sigma_low', 'sigma_high'
    ),
    defaults=[-np.inf] * 41
)


def defaults():
    """
    Sets the default values for the UQPCE settings.
    """
    out_file = 'output.dat'
    sobol_out = 'sobol.dat'
    coeff_out = 'coefficients.dat'
    output_directory = 'outputs'

    input_file = 'input.yaml'
    matrix_file = 'run_matrix.dat'
    results_file = 'results.dat'
    verification_results_file = 'verification_results.dat'
    verification_matrix_file = 'verification_run_matrix.dat'

    case = None
    version_num = '0.3.0'
    backend = 'TkAgg'

    verbose = False
    verify = False
    version = False
    plot = False
    plot_stand = False
    generate_samples = False
    model_conf_int = False
    stats = False
    report = False
    seed = False
    user_func = None  # can put a string function here (ex. 'x0*x1+x2')

    track_convergence_off = False  # track and plot confidence interval unless
                                    # turned off (True)
    epist_sub_samp_size = 25  # convergence; CI tracking used unless flag
    aleat_sub_samp_size = 5000

    order = 2  # order of PCE expansion
    over_samp_ratio = 2
    verify_over_samp_ratio = 0.5
    significance = 0.05
    conv_threshold_percent = 0.0005

    # number of times to iterate for each variable type
    epist_samp_size = 125
    aleat_samp_size = 25000

    bound_limits = None
    var_thresh = None

    prog_set = ProgSettings(
        out_file=out_file, sobol_out=sobol_out, coeff_out=coeff_out,
        input_file=input_file, order=order,
        matrix_file=matrix_file, results_file=results_file, verify=verify,
        verification_results_file=verification_results_file, version=version,
        verification_matrix_file=verification_matrix_file, case=case, plot=plot,
        version_num=version_num, backend=backend, verbose=verbose,
        plot_stand=plot_stand, generate_samples=generate_samples,
        model_conf_int=model_conf_int, stats=stats, report=report,
        user_func=user_func,
        track_convergence_off=track_convergence_off,
        epist_sub_samp_size=epist_sub_samp_size, epist_samp_size=epist_samp_size,
        aleat_sub_samp_size=aleat_sub_samp_size, significance=significance,
        over_samp_ratio=over_samp_ratio,
        conv_threshold_percent=conv_threshold_percent, bound_limits=bound_limits,
        aleat_samp_size=aleat_samp_size,
        output_directory=output_directory,
        var_thresh=var_thresh,
        seed=seed, verify_over_samp_ratio=verify_over_samp_ratio
    )

    return prog_set


def arg_parser(prog_set):
    """
    Inputs: prog_set- the set of UQPCE program settings
    
    Parses the UQPCE arguments and sets them from the defaults, the input yaml 
    file, and the command line options, respectively.
    """
    # region: parsing arguments
    parser = argparse.ArgumentParser(# taking in optional commandline args
        description='Uncertainty Quantification Polynomial Chaos Expansion',
        prog='UQPCE',
        argument_default=argparse.SUPPRESS
    )

    # optional arguments with a default file option
    parser.add_argument(
        '-i', '--input-file', help=f'file containing variables (default: '
        f'{prog_set.input_file})'
    )

    parser.add_argument(
        '-m', '--matrix-file', help='file containing matrix elements (default:'
        f' {prog_set.matrix_file})'
    )

    parser.add_argument(
        '-r', '--results-file', help=f'file containing results (default: '
        f'{prog_set.results_file})'
    )

    parser.add_argument(
        '--verification-results-file', help=f'file containing verification '
        f'results (default: {prog_set.verification_results_file})'
    )

    parser.add_argument(
        '--verification-matrix-file', help=f'file containing verification '
        f'matrix elements (default: {prog_set.verification_matrix_file})'
    )

    parser.add_argument(
        '--output-directory', help=f'directory that the outputs will be put in '
        f'(default: {prog_set.output_directory})'
    )

    parser.add_argument(
        '-c', '--case', help=f'case of input data (default: {prog_set.case})'
    )

    parser.add_argument(
        '-s', '--significance', help=f'significance level of the confidence '
        f'interval (default: {prog_set.significance})', type=float
    )

    parser.add_argument(
        '-o', '--order', help=f'order of polynomial chaos expansion (default: {prog_set.order})',
        type=int, nargs='+'
    )

    parser.add_argument(
        '-f', '--user-func', help='allows the user to specify the analytical '
        f'function for the data (default: {prog_set.user_func})'
    )

    parser.add_argument(
        '--over-samp-ratio', help='over sampling ratio; factor for how many '
        f'points to be used in calculations (default: {prog_set.over_samp_ratio})',
        type=float
    )

    parser.add_argument(
        '--verify-over-samp-ratio', help='over sampling ratio for verification; '
        'factor for how many points to be used in calculations (default: '
        f'{prog_set.over_samp_ratio})',
        type=float
    )

    parser.add_argument(
        '-b', '--backend', help='the backend that will be used for Matplotlib '
        f'plotting (default: {prog_set.backend})'
    )

    parser.add_argument(
        '--aleat-sub-samp-size', help='the number of samples to check the new '
        'high and low intervals at for each individual curve (default: '
        f'{prog_set.aleat_sub_samp_size})', type=int
    )

    parser.add_argument(
        '--epist-sub-samp-size', help='the number of curves to check the new '
        'high and low intervals at for a set of curves (default: '
        f'{prog_set.epist_sub_samp_size})', type=int
    )

    parser.add_argument(
        '--conv-threshold-percent', help='the percent of the response mean to '
        'be used as a threshold for tracking convergence (default: '
        f'{prog_set.conv_threshold_percent})', type=float
    )

    parser.add_argument(
        '--epist-samp-size', help='the number of times to sample for each '
        f'varaible with epistemic uncertainty (default: {prog_set.epist_samp_size})',
        type=int
    )

    parser.add_argument(
        '--aleat-samp-size', help='the number of times to sample for each '
        f'varaible with aleatory uncertainty (default: {prog_set.aleat_samp_size})',
        type=int
    )

    # optional flags while running module
    parser.add_argument(
        '-v', '--version', help='displays the version of the software',
        action='store_true'
    )

    parser.add_argument(
        '--verbose', help='increase output verbosity', action='store_true'
    )

    parser.add_argument(
        '--verify', help='allows verification of results', action='store_true'
    )

    parser.add_argument(
        '--plot', help='generates factor vs response plots, pbox plot, and '
        'error plots', action='store_true'
    )

    parser.add_argument(
        '--plot-stand', help='plots standardized variables', action='store_true'
    )

    parser.add_argument(
        '--track-convergence-off', help='allows users to converge on '
        'confidence interval until the change between the two iters is '
        'less than the threshold', action='store_true'
    )

    parser.add_argument(
        '--generate-samples', help='generates the samples used for all '
        'variables according to the parameters provided in the input file',
        action='store_true'
    )

    parser.add_argument(
        '--model-conf-int', help='includes uncertainties associated with the '
        'model itself', action='store_true'
    )

    parser.add_argument(
        '--stats', help='perform additional statistics for a '
        'more-comprehensive profile of the model', action='store_true'
    )

    parser.add_argument(
        '--report', help='generates a user-friendly report that documents the '
        'statistics and plots that suggest the model is insufficient',
        action='store_true'
    )

    parser.add_argument(
        '--seed', help='if UQPCE should use a seed for random values',
        action='store_true'
    )

    args = parser.parse_args()
    # endregion: parsing arguments

    # region: read settings
    input_file = getattr(args, 'input_file', prog_set.input_file)  # read input first
    arg_options = [parser._actions[arg_num].dest  # list of all possible cmnd line
                   for arg_num in range(1, len(parser._actions))]  # arg names

    init = DataSet(prog_set.verbose)  # setting up the problem via the input files
    var_dict, settings, atmos = init.check_settings(input_file, arg_options, args)
    arg_options.append('resp_order')  # want the iteration order in output.dat

    try:  # updates the local variables with input file
        prog_set = prog_set._replace(**settings)
        prog_set = prog_set._replace(arg_options=arg_options)
    except TypeError:
        pass  # no commands in the input file
    # endregion: read settings

    # region: setting arguments
    matrix_file = getattr(args, 'matrix_file', prog_set.matrix_file)
    results_file = getattr(args, 'results_file', prog_set.results_file)

    verification_results_file = getattr(
        args, 'verification_results_file', prog_set.verification_results_file
    )

    verification_matrix_file = getattr(
        args, 'verification_matrix_file', prog_set.verification_matrix_file
    )

    output_directory = getattr(args, 'output_directory', prog_set.output_directory)
    case = getattr(args, 'case', prog_set.case)
    version = getattr(args, 'version', prog_set.version)
    verbose = getattr(args, 'verbose', prog_set.verbose)
    verify = getattr(args, 'verify', prog_set.verify)
    plot = getattr(args, 'plot', prog_set.plot)
    plot_stand = getattr(args, 'plot_stand', prog_set.plot_stand)
    model_conf_int = getattr(args, 'model_conf_int', prog_set.model_conf_int)
    stats = getattr(args, 'stats', prog_set.stats)
    report = getattr(args, 'report', prog_set.report)
    seed = getattr(args, 'seed', prog_set.seed)

    track_convergence_off = getattr(
        args, 'track_convergence_off', prog_set.track_convergence_off
    )

    generate_samples = getattr(args, 'generate_samples', prog_set.generate_samples)
    significance = getattr(args, 'significance', prog_set.significance)
    order = getattr(args, 'order', prog_set.order)
    user_func = getattr(args, 'user_func', prog_set.user_func)
    over_samp_ratio = getattr(args, 'over_samp_ratio', prog_set.over_samp_ratio)
    verify_over_samp_ratio = getattr(args, 'verify_over_samp_ratio', prog_set.verify_over_samp_ratio)
    backend = getattr(args, 'backend', prog_set.backend)

    aleat_sub_samp_size = getattr(
        args, 'aleat_sub_samp_size', prog_set.aleat_sub_samp_size
    )

    epist_sub_samp_size = getattr(
        args, 'epist_sub_samp_size', prog_set.epist_sub_samp_size
    )

    conv_threshold_percent = getattr(
        args, 'conv_threshold_percent', prog_set.conv_threshold_percent
    )

    epist_samp_size = getattr(args, 'epist_samp_size', prog_set.epist_samp_size)
    aleat_samp_size = getattr(args, 'aleat_samp_size', prog_set.aleat_samp_size)

    bound_limits = getattr(args, 'bound_limits', prog_set.bound_limits)

    out_file = getattr(args, 'out_file', prog_set.out_file)
    sobol_out = getattr(args, 'sobol_out', prog_set.sobol_out)
    coeff_out = getattr(args, 'coeff_out', prog_set.coeff_out)
    version_num = getattr(args, 'version_num', prog_set.version_num)
    # endregion: setting arguments

    prog_set = prog_set._replace(
        out_file=out_file, sobol_out=sobol_out, coeff_out=coeff_out,
        input_file=input_file, order=order,
        matrix_file=matrix_file, results_file=results_file, verify=verify,
        verification_results_file=verification_results_file, version=version,
        verification_matrix_file=verification_matrix_file, case=case, plot=plot,
        version_num=version_num, backend=backend, verbose=verbose,
        plot_stand=plot_stand, generate_samples=generate_samples,
        model_conf_int=model_conf_int, stats=stats, report=report,
        user_func=user_func,
        track_convergence_off=track_convergence_off,
        epist_sub_samp_size=epist_sub_samp_size, epist_samp_size=epist_samp_size,
        aleat_sub_samp_size=aleat_sub_samp_size, significance=significance,
        over_samp_ratio=over_samp_ratio,
        conv_threshold_percent=conv_threshold_percent, bound_limits=bound_limits,
        aleat_samp_size=aleat_samp_size, output_directory=output_directory,
        seed=seed, verify_over_samp_ratio=verify_over_samp_ratio
    )

    return prog_set


def uqpce_setup(prog_set, iter_set=IterSettings()):
    """
    Inputs: prog_set- the set of UQPCE program settings
            iter_set- the set of UQPCE iteration settings
    
    Sets up for UQPCE.
    """

    comm = MPI_COMM_WORLD
    rank = comm.rank
    size = comm.size
    is_manager = (rank == 0)

    # Seed UQPCE if this is desired by the user
    if prog_set.seed:
        np.random.seed(33)

    if prog_set.seed and size > 1:
        raise RuntimeError(
            'UQPCE does not yet support using --seed with more than one '
            'process.'
        )
        exit()

    output_directory = prog_set.output_directory
    verify_keys = False
    resp_verify = False

    # region: setup
    if prog_set.user_func is not None:  # user func given/samples generated
        resp_count = 1  # generating samples means only one set possible
    else:
        results = read_file(prog_set.results_file)
        results_keys = list(results)
        resp_count = len(results_keys)

    # region: ensure order is iterable
    order = prog_set.order
    if isinstance(order, list):
        if len(order) == 1:
            order = order[0]
        else:
            assert len(order), resp_count
            order = np.array(order).astype(int)

    if isinstance(order, int):
        order = np.ones(resp_count, dtype=int) * prog_set.order
    # endregion: ensure order is iterable

    max_order = int(np.max(prog_set.order))

    if max_order > 4:
        warn('The results may be less valid for orders as high as 5th order. '
             'Most cases do not require an order above 4.')

    init = DataSet(prog_set.verbose)  # setting up the problem via the input files
    var_dict = init.check_settings(prog_set.input_file, prog_set.arg_options, prog_set._asdict())[0]

    var_list = init.read_var_input(var_dict, max_order)
    var_count = len(var_list)

    required_sample_count = calc_term_count(max_order, var_count)
    gen_samp_size = int(
        np.ceil(required_sample_count * prog_set.over_samp_ratio)
    )
    verif_gen_samp_size = int(
        np.ceil(required_sample_count * prog_set.verify_over_samp_ratio)
    )

    if prog_set.generate_samples:
        for variable in var_list:
                variable.set_vals(variable.generate_samples(gen_samp_size))
    else:
        init.read_var_vals(prog_set.matrix_file, 'vals')

    if prog_set.generate_samples and prog_set.verify:
        for variable in var_list:
                variable.set_verify_vals(
                    variable.generate_samples(verif_gen_samp_size)
                )

    # region: generate samples
    # user_func required when generate_samples, but can use user_func
    if prog_set.generate_samples:  # w/o generate_samples
        if is_manager and prog_set.verbose:
            print('Generated samples are in run_matrix_generated.dat\n')

        write_gen_samps(gen_samp_size, var_list)

        if prog_set.verify:
            write_gen_samps(
                verif_gen_samp_size, var_list,
                file_name='verification_run_matrix_generated.dat',
                attr='verify_vals'
            )

        if prog_set.user_func is None:
            if is_manager and prog_set.verbose:
                print(
                    'Samples will be created according to the input distributions, '
                    "but the corresponding \noutputs won't be generated unless the "
                    "'--user-func' flag with an equation is also used\n"
                )

            exit()

    if prog_set.user_func is not None:  # user func given/samples generated
        if is_manager and prog_set.verbose:
            print(f'Generating the results from function {prog_set.user_func}\n')

        results = user_function(prog_set.user_func, var_list)
        write_gen_resps(results)

        exit()
    # endregion: generate samples

    switch_backend(prog_set.backend)

    if is_manager and prog_set.verbose:
        print(f'Using MatPlotLib backend {prog_set.backend}\n')

    if is_manager and prog_set.version:
        print(f'UQPCE: version {prog_set.version_num}\n')

    if prog_set.verify:
        resp_verify = read_file(prog_set.verification_results_file)
        verify_keys = list(resp_verify)
        init.read_var_vals(prog_set.verification_matrix_file, 'verify_vals')

    for variable in var_list:
        variable.standardize('vals', 'std_vals')
        variable.check_distribution()

    if prog_set.case is not None:
        output_directory = f'{prog_set.case}_{output_directory}'

    if is_manager:
        output_directory = check_directory(output_directory, prog_set.verbose)
    else:
        output_directory = None

    output_directory = comm.bcast(output_directory, root=0)

    if is_manager and prog_set.verbose and (resp_count > 1):
        print(
            f'{resp_count} columns found in the results file\n\n'
            f'{resp_count} models will be constructed\n'
        )
    # endregion: setup

    iter_set = iter_set._replace(
        resp_verify=resp_verify, verify_keys=verify_keys, results=results,
        results_keys=results_keys, resp_count=resp_count, var_count=var_count,
        var_list=var_list, max_order=max_order
    )

    prog_set = prog_set._replace(output_directory=output_directory, order=order)

    return prog_set, iter_set


def uqpce_iter_setup(prog_set, iter_set):
    """
    Inputs: prog_set- the set of UQPCE program settings
            iter_set- the set of UQPCE iteration settings
    
    Set up for a UQPCE iteration.
    """
    comm = MPI_COMM_WORLD
    size = comm.size
    rank = comm.rank
    is_manager = (rank == 0)

    # region: unpack settings
    verbose = prog_set.verbose
    output_directory = prog_set.output_directory

    results_keys = iter_set.results_keys
    resp_idx = iter_set.resp_idx
    resp_count = iter_set.resp_count
    results = iter_set.results
    # endregion: unpack settings

    sobol_str = ''

    resp_name = results_keys[resp_idx]

    if resp_count > 1:  # results file has multiple columns
        output_col_directory = (
            f'{output_directory}/response_{resp_name}'
        )

        if is_manager:
            output_col_directory = (
                check_directory(output_col_directory, verbose)
            )

            for i in range(1, size):
                comm.send(output_col_directory, dest=i, tag=9)

        else:
            output_col_directory = comm.recv(source=0, tag=9)

    else:
        output_col_directory = output_directory

    # check that graph directory exists
    graph_dir = f'{output_col_directory}/graphs'
    if is_manager:
        graph_dir = check_directory(graph_dir, verbose)

        for i in range(1, size):
            comm.send(graph_dir, dest=i, tag=11)

    else:
        graph_dir = comm.recv(source=0, tag=11)

    resp_iter = results[resp_name]
    # endregion: iteration initialization

    iter_set = iter_set._replace(
        graph_dir=graph_dir, resp_iter=resp_iter, sobol_str=sobol_str,
        output_col_directory=output_col_directory
    )

    return prog_set, iter_set


def uqpce_matrix(prog_set, iter_set):
    """
    Inputs: prog_set- the set of UQPCE program settings
            iter_set- the set of UQPCE iteration settings
            
    Function that handles the UQPCE matrix.
    """
    comm = MPI_COMM_WORLD
    size = comm.size
    rank = comm.rank
    is_manager = (rank == 0)

    # region: unpack settings
    verbose = prog_set.verbose
    resp_order = prog_set.resp_order

    var_list = iter_set.var_list
    resp_iter = iter_set.resp_iter
    # endregion: unpack settings

    # region: setting up matrix
    matrix = MatrixSystem(resp_iter, var_list)
    matrix.verbose = verbose

    if is_manager and verbose:
        print('Constructing surrogate model\n\nBuilding norm-squared matrix\n')

    min_model_size, inter_matrix = matrix.create_model_matrix(resp_order)

    norm_sq = matrix.form_norm_sq(resp_order)

    if is_manager and verbose:
        print('Assembling Psi matrix\n')

    var_basis_vect_symb = matrix.build()

    if is_manager and verbose:
        print('Psi matrix Assembled\n')

    # endregion: setting up matrix

    # region: update over_samp_ratio
    try:
        resp_iter_count = len(resp_iter)
        over_samp_ratio = resp_iter_count / min_model_size

        if over_samp_ratio < 1.25:  # warn if low over_samp_ratio
            warn(
                f'The oversampling ratio is {over_samp_ratio:.5}. Consider '
                f'using at least {int(np.ceil(1.25 * min_model_size))} '
                'samples for a more accurate model.'
            )

    except TypeError:
        over_samp_ratio = 0
        resp_iter_count = 0
        pass
    # endregion: update over_samp_ratio

    # region: solving matrix system
    if is_manager and verbose:
        print('Evaluating Psi matrix\n')

    var_basis_sys_eval = matrix.evaluate()

    if is_manager and verbose:
        print('Psi matrix Evaluated\n')

    if is_manager and verbose:
        print('Solving system\n')

    matrix_coeffs = matrix.solve()
    # endregion: solving matrix system

    iter_set = iter_set._replace(
        var_list=var_list, matrix_coeffs=matrix_coeffs, norm_sq=norm_sq,
        var_basis_sys_eval=var_basis_sys_eval, resp_iter=resp_iter,
        inter_matrix=inter_matrix, min_model_size=min_model_size,
        var_basis_vect_symb=var_basis_vect_symb, resp_iter_count=resp_iter_count
    )

    prog_set = prog_set._replace(over_samp_ratio=over_samp_ratio)

    return prog_set, iter_set, matrix


def uqpce_basis(prog_set, iter_set):
    """
    Inputs: prog_set- the set of UQPCE program settings
            iter_set- the set of UQPCE iteration settings
    
    Function for most of UQPCE, including the SurrogateModel, Graphs, and 
    ProbabilityBoxes.
    """
    comm = MPI_COMM_WORLD
    size = comm.size
    rank = comm.rank
    is_manager = (rank == 0)

    # region: unpack settings
    model_conf_int = prog_set.model_conf_int
    significance = prog_set.significance
    epist_samp_size = prog_set.epist_samp_size
    aleat_samp_size = prog_set.aleat_samp_size
    epist_sub_samp_size = prog_set.epist_sub_samp_size
    aleat_sub_samp_size = prog_set.aleat_sub_samp_size
    track_convergence_off = prog_set.track_convergence_off
    plot = prog_set.plot
    verbose = prog_set.verbose
    conv_threshold_percent = prog_set.conv_threshold_percent
    plot_stand = prog_set.plot_stand

    var_list = iter_set.var_list
    var_count = iter_set.var_count
    graph_dir = iter_set.graph_dir
    matrix_coeffs = iter_set.matrix_coeffs
    resp_iter = iter_set.resp_iter
    norm_sq = iter_set.norm_sq
    var_basis_sys_eval = iter_set.var_basis_sys_eval
    inter_matrix = iter_set.inter_matrix
    sobol_str = iter_set.sobol_str
    var_basis_vect_symb = iter_set.var_basis_vect_symb

    # default if not calculated
    approx_mean = False
    mean_uncert = False
    sobol_bounds = False
    coeff_uncert = False

    sigma_low = False
    sigma_high = False

    # endregion: unpack settings
    # region: setting up model
    model = SurrogateModel(resp_iter, matrix_coeffs, verbose=verbose)

    if is_manager and verbose:
        print('Surrogate model construction complete\n')
    # endregion: setting up model

    # region: basic model stats
    sigma_sq, resp_mean = model.calc_var(norm_sq)
    error, pred = model.calc_error(var_basis_sys_eval)
    err_mean = calc_mean_err(error)
    signal_to_noise = sigma_sq / err_mean

    sobols = model.get_sobols(norm_sq)
    tot_sobol = create_total_sobols(var_count, inter_matrix, sobols)

    mean_sq, hat_matrix, shapiro_results = model.check_normality(
        var_basis_sys_eval, significance, graph_dir
    )

    sobol_str = ''.join((
            sobol_str,
            'Total Sobols\n'
    ))

    for i in range(var_count):
        sobol_str = ''.join((
            sobol_str, f'   Total Sobol {var_list[i].name} = {tot_sobol[i]:.5}\n'
        ))

    sobol_str = ''.join((
            sobol_str,
            '\nRescaled Total Sobols\n'
    ))

    sobol_sum = np.sum(tot_sobol)
    for i in range(var_count):
        sobol_str = ''.join((
            sobol_str, f'   Total Sobol {var_list[i].name} = {tot_sobol[i]/sobol_sum:.5}\n'
        ))

    if is_manager and verbose:
        print(sobol_str)

        print(
            f'Mean of response {resp_mean:.5}\n'
            f'Variance of response {sigma_sq:.5}\n'
            f'Mean error of surrogate {err_mean:.5}\n'
            f'Signal to noise ratio {signal_to_noise:.5}\n'
        )
    # endregion: basic model stats

    # region: confidence intervals
    if model_conf_int:  # calculating the coefficient and sobol uncertainties
        coeff_uncert = (
            calc_coeff_conf_int(
                var_basis_sys_eval, matrix_coeffs, resp_iter, significance
            )
        )

        low_sobol, high_sobol = (
            get_sobol_bounds(matrix_coeffs, sobols, coeff_uncert, norm_sq)
        )

        sobol_bounds = {'low': low_sobol, 'high':high_sobol}

        sigma_low, sigma_high = calc_var_conf_int(
            matrix_coeffs, coeff_uncert, norm_sq
        )
    # endregion: confidence intervals

    # region: plotting
    if plot:
        graph = Graphs(plot_stand)
        graph.verbose = verbose
        graph.factor_plots(graph_dir, var_list, pred, 'Predicted')
        graph.factor_plots(graph_dir, var_list, error, 'Error')
        graph.error_vs_pred(graph_dir, error, pred, 'Error vs Predicted')
    # endregion: plotting

    # region: probability box
    pbox = ProbabilityBoxes(
        var_list, verbose=verbose, plot=plot,
        track_conv_off=track_convergence_off,
        epist_samps=epist_samp_size,
        aleat_samps=aleat_samp_size,
        aleat_sub_size=aleat_sub_samp_size,
        epist_sub_size=epist_sub_samp_size
    )

    pbox.count_epistemic()
    pbox.generate_epistemic_samples()
    pbox.generate_aleatory_samples()

    results_eval, convergence_message = (
        pbox.evaluate_surrogate(
            var_basis_vect_symb, significance, matrix_coeffs,
            conv_threshold_percent, graph_dir=graph_dir
        )
    )

    if model_conf_int:
        approx_mean, mean_uncert = (
            pbox.calc_mean_conf_int(var_basis_sys_eval, resp_iter, significance)
        )

    conf_int_low, conf_int_high = (
        pbox.generate(results_eval, significance, graph_dir)
    )
    # endregion: probability box

    iter_set = iter_set._replace(
        var_list=var_list, mean_sq=mean_sq, sobol_bounds=sobol_bounds,
        shapiro_results=shapiro_results, convergence_message=convergence_message,
        approx_mean=approx_mean, mean_uncert=mean_uncert, hat_matrix=hat_matrix,
        conf_int_low=conf_int_low, conf_int_high=conf_int_high,
        resp_mean=resp_mean, err_mean=err_mean, sigma_sq=sigma_sq,
        signal_to_noise=signal_to_noise, error=error, coeff_uncert=coeff_uncert,
        sobol_str=sobol_str, sigma_low=sigma_low, sigma_high=sigma_high
    )

    return prog_set, iter_set, model


def uqpce_outputs(prog_set, iter_set, matrix, model):
    """
    Inputs: prog_set- the set of UQPCE program settings
            iter_set- the set of UQPCE iteration settings
            matrix- the MatrixSystem object used by UQPCE
            model- the SurrogateModel object used by UQPCE

    Function that handles the output files and verification.
    """

    comm = MPI_COMM_WORLD
    size = comm.size
    rank = comm.rank
    is_manager = (rank == 0)

    resp_iter = iter_set.resp_iter
    resp_idx = iter_set.resp_idx
    verify_keys = iter_set.verify_keys
    resp_verify = iter_set.resp_verify
    var_list = iter_set.var_list
    err_mean = iter_set.err_mean
    sobol_str = iter_set.sobol_str
    sobol_bounds = iter_set.sobol_bounds
    convergence_message = iter_set.convergence_message
    shapiro_results = iter_set.shapiro_results
    time_start = iter_set.time_start
    coeff_uncert = iter_set.coeff_uncert
    output_col_directory = iter_set.output_col_directory
    resp_mean = iter_set.resp_mean
    conf_int_high = iter_set.conf_int_high
    sigma_sq = iter_set.sigma_sq
    conf_int_low = iter_set.conf_int_low
    signal_to_noise = iter_set.signal_to_noise
    graph_dir = iter_set.graph_dir
    sigma_low = iter_set.sigma_low
    sigma_high = iter_set.sigma_high

    signif = prog_set.significance

    press_dict = None
    err_ratio = False
    verify_str = ''
    stats_str = ''
    conf_int_str = ''

    # region: stats
    if prog_set.stats:
        press_dict = matrix.get_press_stats()
        R_sq = calc_R_sq(matrix.var_basis_sys_eval, matrix.matrix_coeffs, resp_iter)
        R_sq_adj = calc_R_sq_adj(matrix.var_basis_sys_eval, matrix.matrix_coeffs, resp_iter)

        # unpacking press stats dict
        press_stat = press_dict['PRESS']
        mean_err_avg = press_dict['mean_of_model_mean_err']
        mean_err_var = press_dict['variance_of_model_mean_err']
        mean_avg = press_dict['mean_of_model_mean']
        mean_var = press_dict['variance_of_model_mean']
        variance_avg = press_dict['mean_of_model_variance']
        variance_var = press_dict['variance_of_model_variance']

        stats_str = (# formatting like this so 5 sig figs and aligned
            f'PRESS statistic:         {f"{press_stat:.5}":10s}\n'
            f'R\u00b2:                      {f"{R_sq:.5}":10s}\n'
            f'R\u00b2 adjusted:             {f"{R_sq_adj:.5}":10s}\n\n'
            'Below are additional statistics from models built during the '
            'calculation of the \nPRESS statistic. These are not needed for '
            'most users.\n'
            f'average of mean error:   {f"{mean_err_avg:.5}":10s}\t\t'
            f'variance of mean error:  {f"{mean_err_var:.5}":10s}\n'
            f'average of mean:         {f"{mean_avg:.5}":10s}\t\t'
            f'variance of mean:        {f"{mean_var:.5}":10s}\n'
            f'average of variance:     {f"{variance_avg:.5}":10s}\t\t'
            f'variance of variance:    {f"{variance_var:.5}":10s}\n'
        )

        if is_manager and prog_set.verbose:
            print(stats_str)
    # endregion: stats

    # region: model_conf_int
    if prog_set.model_conf_int:
        conf_int_str = (
            f'{1-signif:.1%} Confidence Interval on the mean [{resp_mean - coeff_uncert[0]:.5}, '
            f'{resp_mean + coeff_uncert[0]:.5}]\n'
            f'{1-signif:.1%} Confidence Interval on the variance '
            f'[{sigma_low:.5}, {sigma_high:.5}]\n'
        )
        if is_manager and prog_set.verbose:
            print(conf_int_str)
    # endregion: model_conf_int

    # region: verify
    if prog_set.verify:
        resp_iter_ver = resp_verify[verify_keys[resp_idx]]

        for variable in var_list:
            variable.standardize('verify_vals', 'std_verify_vals')

        pred_ver, var_basis_sys_eval_ver = (
            model.verify(
                matrix.var_basis_vect_symb, var_list, len(resp_iter_ver),
                matrix.var_list_symb
            )
        )

        err_ver = calc_difference(resp_iter_ver, pred_ver)
        err_mean_ver = calc_mean_err(err_ver)

        if prog_set.model_conf_int:

            approx_conf_ver, conf_uncert_ver = (
                calc_pred_conf_int(
                    matrix.var_basis_sys_eval, matrix.matrix_coeffs, resp_iter,
                    prog_set.significance, var_basis_sys_eval_ver
                )
            )

            if (err_ver > conf_uncert_ver).any():
                warn(# verify vals outside of pred CI?
                    f'Predicted points are outside of the '
                    f'{1-prog_set.significance:.0%} prediction confidence interval.'
                )

        if prog_set.plot:
            graph = Graphs(prog_set.plot_stand)
            graph.factor_plots(
                graph_dir, var_list, pred_ver, 'Verify Predicted', verify=True
            )

            graph.factor_plots(
                graph_dir, var_list, err_ver, 'Verify Error', verify=True
            )

            graph.error_vs_pred(
                graph_dir, err_ver, pred_ver, 'Verify Error vs Predicted'
            )

            if prog_set.model_conf_int:
                graph.pred_conf(
                    graph_dir, pred_ver, resp_iter_ver, approx_conf_ver, conf_uncert_ver
                )

        err_ratio = err_mean_ver / err_mean

        verify_str = (
            f'Mean error between model and verification {err_mean_ver:.5}'
            '\n\nThe ratio of verification error to surrogate model error '
            f'is {err_ratio:.5}\n'
        )

        if is_manager and prog_set.verbose:
            print(verify_str)
    # endregion: verify

    if is_manager and prog_set.verbose:
        print('Writing output data\n')

    # writing the labeled output files for sobols and matrix_coeffs coeficients
    str_vars = get_str_vars(matrix.model_matrix)

    time_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    time_end = datetime.strptime(time_end, '%Y-%m-%d %H:%M:%S.%f')
    time_total = time_end - time_start

    output_str = (
        f'###  UQPCE v{prog_set.version_num} Output\n###  Analysis of case: {prog_set.case}\n'
        f'###  Analysis started: {time_start}\n###  Analysis finished: '
        f'{time_end}\n###  Total compute time: {time_total}\n--------------'
        '-----------------------------------------------------------'
        f'-------\n\nMean of response {resp_mean:.5}\nVariance of response'
        f' {sigma_sq:.5}\nMean error of surrogate {err_mean:.5}\nSignal to'
        f' noise ratio {signal_to_noise:.5}\n{1 - prog_set.significance:.1%}'
        f' Confidence Interval on Response [{conf_int_low:.5} , '
        f'{conf_int_high:.5}]\n'
    )

    kwargs = {
        'out_str':output_str, 'in_file':prog_set.input_file, 'arg_opts':prog_set.arg_options,
        'shapiro':shapiro_results, 'ver_str':verify_str, 'settings':prog_set,
        'conv_str':convergence_message, 'stats_str':stats_str,
        'conf_int_str':conf_int_str
    }

    sobol_dir = f'{output_col_directory}/{prog_set.sobol_out}'
    coeff_dir = f'{output_col_directory}/{prog_set.coeff_out}'
    out_dir = f'{output_col_directory}/{prog_set.out_file}'

    task_count = 3  # N tasks to execute simultaneously
    reps = int(np.ceil(task_count / size))
    opts = np.repeat(np.arange(0, size), reps)

    write_sobol_file = (rank == opts[0])
    write_coeff_file = (rank == opts[1])
    write_output_file = (rank == opts[2])

    if write_sobol_file:
        write_sobols(sobol_dir, str_vars, model.sobols, sobol_str, sobol_bounds)

    if write_coeff_file:
        write_coeffs(coeff_dir, matrix.matrix_coeffs, str_vars, coeff_uncert)

    if write_output_file:
        write_outputs(out_dir, **kwargs)
    # endregion: outputs

    iter_set = iter_set._replace(err_ratio=err_ratio, verify_str=verify_str)

    return prog_set, iter_set, press_dict


def uqpce_report(prog_set, iter_set, press_dict):
    """
    Inputs: prog_set- the set of UQPCE program settings
        iter_set- the set of UQPCE iteration settings
        press_dict- the dictionary of PRESS results

    Generates the UQPCE report if that flag is used by the user.
    """
    comm = MPI_COMM_WORLD
    rank = comm.rank
    is_manager = (rank == 0)

    # unpacking iter_set
    var_list = iter_set.var_list
    error = iter_set.error
    graph_dir = iter_set.graph_dir
    signal_to_noise = iter_set.signal_to_noise
    resp_mean = iter_set.resp_mean
    err_mean = iter_set.err_mean
    resp_iter_count = iter_set.resp_iter_count
    err_ratio = iter_set.err_ratio
    verify_str = iter_set.verify_str
    shapiro_results = iter_set.shapiro_results
    convergence_message = iter_set.convergence_message
    output_col_directory = iter_set.output_col_directory

    resp_order = prog_set.resp_order

    if prog_set.stats:
        R_sq = calc_R_sq(
            iter_set.var_basis_sys_eval, iter_set.matrix_coeffs,
            iter_set.resp_iter
        )

        R_sq_adj = calc_R_sq_adj(
            iter_set.var_basis_sys_eval, iter_set.matrix_coeffs,
            iter_set.resp_iter
        )

    # unpacking press stats dict
    if press_dict is not None:
        press_stat = press_dict['PRESS']
        mean_err_avg = press_dict['mean_of_model_mean_err']
        mean_err_var = press_dict['variance_of_model_mean_err']
        mean_avg = press_dict['mean_of_model_mean']
        mean_var = press_dict['variance_of_model_mean']
        variance_avg = press_dict['mean_of_model_variance']
        variance_var = press_dict['variance_of_model_variance']

    # region: UQPCE Report
    if prog_set.report:
        empty = []
        uq_report = UQPCEReport()

        if is_manager and prog_set.verbose:
            print('Generating error report\n')

        if prog_set.plot:
            var_names = check_error_trends(var_list, error, resp_order)
            err_paths = [f'{graph_dir}/Error_vs_{name}' for name in var_names]

            uq_report.add_section('Error Plots')

            if err_paths != empty:
                uq_report.add_plot(err_paths)
                uq_report.add_pagebreak()
            else:
                uq_report.add_text(
                    'None of the plots appear to have correlations with error.'
                )

            uq_report.add_text(check_error_magnitude(error))

        uq_report.add_section('Statistics')
        signal_min = 10

        test_stat, p_val_hypoth = shapiro(error)
        if p_val_hypoth < 0.05:
            uq_report.add_stat('Error distribution p-value', p_val_hypoth, prog_set.significance)
            uq_report.add_plot(f'{graph_dir}/error_distribution.png')
            uq_report.add_plot(f'{graph_dir}/normal_prob.png')

        if signal_to_noise < signal_min:
            uq_report.add_stat('Signal to noise ratio', signal_to_noise, signal_min)

        err_mean_max = resp_mean * 0.01  # one percent of mean
        if err_mean > err_mean_max:
            uq_report.add_stat('Mean error', err_mean, err_mean_max)

        if prog_set.stats:
            ind_thresh = 0.01 * resp_mean  # 1% of mean
            press_thresh = resp_iter_count * ind_thresh
            R_sq_thresh = 0.85
            R_sq_adj_thresh = 0.80

            if press_stat > press_thresh:
                uq_report.add_stat('PRESS residual', press_stat, press_thresh)

            if R_sq < R_sq_thresh:
                uq_report.add_stat('R squared', R_sq, R_sq_thresh)

            if R_sq_adj < R_sq_adj_thresh:
                uq_report.add_stat(
                    'R squared adjusted', R_sq_adj, R_sq_adj_thresh
                )

            if ((R_sq_adj > R_sq_adj_thresh) and (R_sq > R_sq_thresh)
                and (press_stat < press_thresh)):
                uq_report.add_text('The statistics do not suggest that the '
                                   'model is insufficient.')

        if prog_set.verify:
            err_ratio_max = 10
            err_ratio_min = 0.1
            err_ratio_ideal = 1

            uq_report.add_section('Verification')

            if err_ratio > err_ratio_max or err_ratio < err_ratio_min:
                uq_report.add_stat(
                    'Verification Error Ratio', err_ratio, err_ratio_ideal
                )

            else:
                uq_report.add_text('The verification data do not suggest the '
                                   'model is insufficient.')

        uq_report.add_section('Messages')
        uq_report.add_text(verify_str)
        uq_report.add_text(shapiro_results)
        uq_report.add_text(convergence_message)
        uq_report.write(output_col_directory)
        uq_report.clear()
    # endregion: UQPCE Report
