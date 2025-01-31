from builtins import setattr, getattr, FileNotFoundError
import os

import numpy as np
from yaml import Loader
import yaml

try:
    from mpi4py.MPI import DOUBLE as MPI_DOUBLE, COMM_WORLD as MPI_COMM_WORLD
    comm = MPI_COMM_WORLD
    rank = comm.rank
    size = comm.size
    is_manager = (rank == 0)
except:
    comm = None
    rank = 0
    size = 1
    is_manager = True

from uqpce.pce.enums import Distribution
from uqpce.pce.variables.continuous import (
    BetaVariable, ExponentialVariable, GammaVariable, NormalVariable,
    UniformVariable, ContinuousVariable
)
from uqpce.pce.variables.discrete import (
    UniformVariable as DiscUniformVariable, DiscreteVariable, NegativeBinomialVariable,
    PoissonVariable, HypergeometricVariable
)
from uqpce.pce._helpers import warn


def write_sobols(file_name, str_vars, sobols, sobol_str, bounds):
    """
    Inputs: file_name- name of the file containing the responses
            str_vars- a list of the variable numbers of type string
            sobols- the sobol values
            sobol_str- the string that contains the total Sobols
            bounds- the Sobol bounds
    
    Writes all of the relevant Sobol information into an output file.
    """
    sob_count = len(sobols)

    low = 'low'
    high = 'high'

    with open(file_name, 'w') as sob:
        for line in range(sob_count):
            if isinstance(bounds, dict):
                sob.write(
                    f'{str_vars[line]:35s}{f"{bounds[low][line]:.5}":10s}\t<\t'
                    f'{f"{sobols[line]:.5}":10s}\t<\t{f"{bounds[high][line]:.5}":10s}\n'
                )

            else:
                sob.write(f'{str_vars[line]:35s}{sobols[line]:.5}\n')

        sob.write('\n\n')
        sob.write(sobol_str)


def write_coeffs(file_name, matrix_coeffs, str_vars, uncert):
    """
    Inputs: name of the file containing the responses
            matrix_coeffs- the matrix coefficient values
            str_vars- a list of the variable numbers of type string
            uncert- the coefficient uncertainty
    
    Writes all of the relevant coefficient information into an output file.
    """
    coeff_count = len(matrix_coeffs)

    with open(file_name, 'w') as coef:

        coef.write(f'{"intercept":35s}\t{str(f"{matrix_coeffs[0]:.5}"):15s}')

        if isinstance(uncert, np.ndarray):
            coef.write(f' +/-\t{str(f"{uncert[0]:.5}"):15s}')

        coef.write('\n')

        for line in range(coeff_count - 1):
            if isinstance(uncert, np.ndarray):
                coef.write(
                    f'{str_vars[line]:35s}\t'
                    f'{str(f"{matrix_coeffs[line + 1]:.5}"):15s} +/-\t'
                    f'{str(f"{uncert[line + 1]:.5}"):15s}\n'
                )

            else:
                coef.write(
                    f'{str_vars[line]:35s}\t{matrix_coeffs[line + 1]:.5}\n'
                )


def write_outputs(
        out_dir, out_str, settings, shapiro, ver_str, conv_str, stats_str,
        conf_int_str, in_file, arg_opts
):
    """
    Inputs: out_dir- the output directory
            out_str- the output string to be written
            settings- the used UQPCE settings
            shapiro- the shapiro result string of the error distribution
            ver_str- the string of the verification information
            conv_str- the convergence message
            stats_str- the statistics information
            conf_int_str- the confidence interval string
            in_file- all of the text from the input file
            arg_opts- the names of the settings
    
    Writes all of the information about the execution to an output file.
    """
    dashed_line = '-' * 80 + '\n\n'

    with open(out_dir, 'w') as fi:

        fi.write(out_str)
        fi.write('\n'.join((shapiro, conv_str)))
        fi.write(dashed_line)
        fi.write(ver_str)
        if ver_str != '':
            fi.write(dashed_line)
        fi.write(conf_int_str)
        if conf_int_str != '':
            fi.write(dashed_line)
        fi.write(stats_str)
        if stats_str != '':
            fi.write(dashed_line)
        fi.write('The settings used to generate these results are:\n')

        for i in range(len(arg_opts)):
            arg_opt = arg_opts[i]

            fi.write(''.join((arg_opt, ': ', str(getattr(settings, arg_opt)), '\n')))

        if in_file:
            fi.write(f'{dashed_line}\nThe input file used is:\n')
            with open(in_file) as in_file:
                input_file_lines = in_file.readlines()
                fi.writelines(input_file_lines)
        else:
            fi.write(f'{dashed_line}\nThere was no input file used')


def read_input_file(input_file):
    """
    Inputs: input_file- the name of the input file to read Variables and 
            Settings from
    
    Reads the input yaml file and separates the variables and settings.
    """
    with open(input_file, 'r') as fi:
        dat = fi.read()

    var_dict = dict(yaml.load(dat, Loader=Loader))

    try:  # check for the Settings dictionary
        settings = var_dict.pop('Settings')

    except KeyError:
        settings = None  # Settings dictionary not present

    return var_dict, settings


class DataSet:
    """
    Inputs: verbose- if the DataSet object should print messages during 
            execution
    
    Reads in, creates, and sets up the variables from the input file. 
    Reads from the matrix file to set the 'vals' (or 'verify_vals') of each 
    variable.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def check_settings(self, input_file, arg_options, args):
        """
        Inputs: input_file- the input (.yaml) file containing the Variable and 
                Settings (optional) information
                arg_options- the names of the available command-line arguments
                args- the command-line arguments
        
        Removes the Settings from the input file, warns users if an argument 
        used isn't valid, and checks the backend in the environment varaibles.
        """
        var_dict, settings = read_input_file(input_file)

        try:
            for key in list(settings):
                try:  # do values from input file agree with values from commandline
                    if key not in arg_options:
                        warn(
                            f'Setting {key} in {input_file} is not a valid '
                             'setting.'
                        )

                    cl_set = settings[key]
                    file_set = getattr(args, key)

                    if cl_set != file_set:
                        try:  # order [2] vs 2 shouldn't alert users
                            np.isclose(cl_set, file_set)

                        except TypeError:
                            warn(
                                f'{key} changing from {settings[key]} to '
                                 f'{getattr(args, key)}.'
                            )

                except AttributeError:  # attribute doesn't exist
                    pass

        except TypeError:
            pass  # no commands in the input file

        if 'MPLBACKEND' in os.environ:
            backend = os.environ['MPLBACKEND']

            try:  # if the backend hasn't been set in input file
                if settings['backend'].upper() != backend.upper():
                    warn(
                        f'backend changing from OS env {backend} to input file '
                        f'{settings["backend"]}.'
                    )

                    backend = settings['backend']
                settings['backend'] = backend

            except KeyError:  # backend isn't in file
                pass

        return var_dict, settings


def write_gen_resps(results, file_name='results_generated.dat'):
    """
    Inputs: results- the array of generated results
            file_name- the name of the output results file
    
    Writes the results array to a file in the UQPCE format.
    """
    with open(file_name, 'w') as res_gen:

        gen_vals = results[list(results)[0]]
        res_gen.write('\n'.join((str(i) for i in gen_vals)))
