import argparse
from builtins import getattr
from collections import namedtuple

try:
    from mpi4py.MPI import COMM_WORLD as MPI_COMM_WORLD
    comm = MPI_COMM_WORLD
    rank = comm.rank
    size = comm.size
    is_manager = (rank == 0)
except:
    comm = None
    rank = 0
    size = 1
    is_manager = True

import numpy as np
from uqpce.pce.io import DataSet

class ProgSet():
    def __init__(self, **kwargs):
        settings_dict = {
            'out_file':-np.inf, 'sobol_out':-np.inf, 'coeff_out':-np.inf, 
            'input_file':-np.inf, 'matrix_file':-np.inf, 'results_file':-np.inf, 
            'verification_results_file':-np.inf, 'verification_matrix_file':-np.inf, 
            'case':-np.inf, 'version_num':-np.inf, 'backend':-np.inf, 
            'verbose':-np.inf, 'verify':-np.inf, 'version':-np.inf, 'plot':-np.inf, 
            'plot_stand':-np.inf, 'generate_samples':-np.inf, 'user_func':-np.inf,
            'model_conf_int':-np.inf, 'stats':-np.inf, 'report':-np.inf, 
            'track_convergence_off':-np.inf, 'epist_sub_samp_size':-np.inf, 
            'aleat_sub_samp_size':-np.inf, 'order':-np.inf, 'significance':-np.inf,
            'over_samp_ratio':-np.inf, 'conv_threshold_percent':-np.inf, 
            'epist_samp_size':-np.inf, 'aleat_samp_size':-np.inf, 'seed':-np.inf, 
            'bound_limits':-np.inf, 'arg_options':-np.inf, 'var_thresh':-np.inf, 
            'output_directory':-np.inf, 'resp_order':-np.inf, 
            'verify_over_samp_ratio':-np.inf
        }
        settings_dict.update(**kwargs)
        self.dict = settings_dict
        
        for key, val in settings_dict.items():
            setattr(self, key, val)
    
    def update(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

        self.dict.update(**kwargs)

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
    version_num = '1.0.0'
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

    prog_set = ProgSet(**{
        'out_file':out_file, 'sobol_out':sobol_out, 'coeff_out':coeff_out,
        'input_file':input_file, 'order':order, 'matrix_file':matrix_file, 
        'results_file':results_file, 'verify':verify, 'user_func':user_func,
        'verification_results_file':verification_results_file, 'plot':plot,
        'verification_matrix_file':verification_matrix_file, 'case':case, 
        'version_num':version_num, 'backend':backend, 'verbose':verbose,
        'plot_stand':plot_stand, 'generate_samples':generate_samples, 
        'model_conf_int':model_conf_int, 'stats':stats, 'report':report,
        'track_convergence_off':track_convergence_off, 'var_thresh':var_thresh, 
        'epist_sub_samp_size':epist_sub_samp_size, 'version':version,
        'aleat_sub_samp_size':aleat_sub_samp_size, 'significance':significance,
        'over_samp_ratio':over_samp_ratio, 'output_directory':output_directory,
        'conv_threshold_percent':conv_threshold_percent, 'seed':seed, 
        'aleat_samp_size':aleat_samp_size, 'bound_limits':bound_limits,
        'verify_over_samp_ratio':verify_over_samp_ratio, 
        'epist_samp_size':epist_samp_size,
    })

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
    var_dict, settings = init.check_settings(input_file, arg_options, args)
    arg_options.append('resp_order')  # want the iteration order in output.dat

    try:  # updates the local variables with input file
        prog_set.update(**settings)
        prog_set.update(arg_options=arg_options)
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

    prog_set.update(**{
        'out_file':out_file, 'sobol_out':sobol_out, 'coeff_out':coeff_out,
        'input_file':input_file, 'order':order,
        'matrix_file':matrix_file, 'results_file':results_file, 'verify':verify,
        'verification_results_file':verification_results_file, 'version':version,
        'verification_matrix_file':verification_matrix_file, 'case':case, 'plot':plot,
        'version_num':version_num, 'backend':backend, 'verbose':verbose,
        'plot_stand':plot_stand, 'generate_samples':generate_samples,
        'model_conf_int':model_conf_int, 'stats':stats, 'report':report,
        'user_func':user_func,
        'track_convergence_off':track_convergence_off,
        'epist_sub_samp_size':epist_sub_samp_size, 'epist_samp_size':epist_samp_size,
        'aleat_sub_samp_size':aleat_sub_samp_size, 'significance':significance,
        'over_samp_ratio':over_samp_ratio,
        'conv_threshold_percent':conv_threshold_percent, 'bound_limits':bound_limits,
        'aleat_samp_size':aleat_samp_size, 'output_directory':output_directory,
        'seed':seed, 'verify_over_samp_ratio':verify_over_samp_ratio
    })

    return prog_set
