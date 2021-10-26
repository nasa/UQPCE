from builtins import setattr, getattr, FileNotFoundError
import os
from warnings import warn

try:
    from mpi4py.MPI import DOUBLE as MPI_DOUBLE, COMM_WORLD as MPI_COMM_WORLD
    import numpy as np
    from yaml import Loader
    import yaml
except:
    warn('Ensure that all required packages are installed.')
    exit()

from PCE_Codes.custom_enums import Distribution
from PCE_Codes._helpers import _warn
from PCE_Codes.variables.continuous import (
    BetaVariable, ExponentialVariable, GammaVariable, NormalVariable,
    UniformVariable, ContinuousVariable
)

from PCE_Codes.variables.discrete import (
    UniformVariable as DiscUniformVariable, DiscreteVariable, NegativeBinomialVariable,
    PoissonVariable, HypergeometricVariable
)

comm = MPI_COMM_WORLD
size = comm.size
rank = comm.rank

is_manager = (rank == 0)


def read_file(file_name):
    """
    Inputs: file_name- name of the file containing the responses
    
    Reads the file and appends the values to a list.
    """
    try:
        with open(file_name, 'r') as fi:
            dat = fi.readlines()

        line_count = len(dat)
        first_line = dat[0].replace(',', '\t').split()

        try:
            float(first_line[0])
            col_names = [f'{i}' for i in range(len(first_line))]

        except ValueError:
            dat = dat[1:]
            col_names = [val for val in first_line]

        results = {name: np.zeros(line_count) for name in col_names}
        keys = results.keys()
        ind = 0

        for line in dat:
            if "#" not in line and line != '\n':
                curr_line = line.replace(',', '\t').split()
                i = 0

                for key in keys:
                    results[key][ind] = curr_line[i]
                    i += 1
                ind += 1

        results = {name : results[name][0:ind] for name in results.keys()}

    except FileNotFoundError:
        results = {'empty' :-1}

    return results


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

        fi.write(f'{dashed_line}\nThe input file used is:\n')

        with open(in_file) as in_file:
            input_file_lines = in_file.readlines()

        fi.writelines(input_file_lines)


def write_gen_samps(
        gen_samp_size, var_list, file_name='run_matrix_generated.dat',
        attr='vals'
    ):
    """
    Inputs: gen_samp_size- the number of generated samples
            var_list- the list of variables
            file_name- the output file name
            attr- the attribute of each Variable object whose values are 
            written to the file
    
    Writes the generated samples to a file.
    """
    with open(file_name, 'w') as mat_gen:
        for i in range(gen_samp_size):
            for var in var_list:
                mat_gen.write(str(getattr(var, attr)[i]) + '\t')
            mat_gen.write('\n')


def write_array(out_file, points, message=''):
    """
    Inputs: out_file- the file to write the outputs to
            points- the 2D array of points to write to a file
            message- the message to print out at the top of the file (optional)
            
    Writes an optional message to a file followed by the 2D array ,'points,' 
    with values separated by spaces.
    """
    x_dim, y_dim = points.shape

    with open(out_file, 'w') as out:
        out.write(message)

        for i in range(x_dim):
            for j in range(y_dim):
                out.write(f'{f"{points[i][j]:.10}":18s}\t')
            out.write('\n')


def read_input_file(input_file):
    """
    Inputs: input_file- the name of the input file to read Variables and 
            Settings from
    
    Reads the input yaml file and separates the variables, settings, and 
    atmosphere.
    """
    with open(input_file, 'r') as fi:
        dat = fi.read()

    var_dict = dict(yaml.load(dat, Loader=Loader))

    try:  # check for the Settings dictionary
        settings = var_dict.pop('Settings')

    except KeyError:
        settings = None  # Settings dictionary not present

    try:  # check for the Settings dictionary
        atmos = var_dict.pop('Atmosphere')

    except KeyError:
        atmos = None  # Settings dictionary not present

    return var_dict, settings, atmos


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
        showwarning = _warn

    def check_settings(self, input_file, arg_options, args):
        """
        Inputs: input_file- the input (.yaml) file containing the Variable and 
                Settings (optional) information
                arg_options- the names of the available command-line arguments
                args- the command-line arguments
        
        Removes the Settings from the input file, warns users if an argument 
        used isn't valid, and checks the backend in the environment varaibles.
        """
        var_dict, settings, atmos = read_input_file(input_file)

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

        return var_dict, settings, atmos

    def read_var_input(self, var_dict, order):
        """
        Inputs: var_dict- the variable dictionaries
                order- the order value at this step in the program
        
        Constructs Variable types in the appropriate subclass based on the 
        information in each of the Variable dictionaries.
        """
        var_names = list(var_dict)
        var_count = len(var_dict)
        self.var_count = var_count

        comm = MPI_COMM_WORLD
        size = comm.size
        rank = comm.rank

        base = var_count // size
        rem = var_count % size
        count = base + (rank < rem)

        var_list = np.zeros(count, dtype=object)

        if is_manager and self.verbose:
            print('Initializing Problem\n\nSetting up Variables\n')

        i = rank
        idx = 0

        while i < var_count:

            curr_var_dict = var_dict[var_names[i]]
            distribution = curr_var_dict.pop('distribution')
            dist = Distribution.from_name(distribution)

            if dist is Distribution.NORMAL:
                var_list[idx] = NormalVariable(
                    curr_var_dict.pop('mean'), curr_var_dict.pop('stdev'),
                    number=i, order=order, **curr_var_dict
                )

            elif dist is Distribution.UNIFORM:
                var_list[idx] = UniformVariable(
                    curr_var_dict.pop('interval_low'),
                    curr_var_dict.pop('interval_high'),
                    number=i, order=order, **curr_var_dict
                )

            elif dist is Distribution.BETA:
                var_list[idx] = BetaVariable(
                    curr_var_dict.pop('alpha'), curr_var_dict.pop('beta'),
                    number=i, order=order, **curr_var_dict
                )

            elif dist is Distribution.EXPONENTIAL:
                var_list[idx] = ExponentialVariable(
                    curr_var_dict.pop('lambda'), number=i, order=order,
                    **curr_var_dict
                )

            elif dist is Distribution.GAMMA:
                var_list[idx] = GammaVariable(
                    curr_var_dict.pop('alpha'), curr_var_dict.pop('theta'),
                    number=i, order=order, **curr_var_dict
                )

            elif dist is Distribution.CONTINUOUS:
                var_list[idx] = ContinuousVariable(
                    curr_var_dict.pop('pdf'), curr_var_dict.pop('interval_low'),
                    curr_var_dict.pop('interval_high'), number=i, order=order,
                    **curr_var_dict
                )

            elif dist is Distribution.DISCRETE_UNIFORM:
                var_list[idx] = DiscUniformVariable(
                    curr_var_dict.pop('interval_low'),
                    curr_var_dict.pop('interval_high'),
                    number=i, order=order, **curr_var_dict
                )

            elif dist is Distribution.NEGATIVE_BINOMIAL:
                var_list[idx] = NegativeBinomialVariable(
                    curr_var_dict.pop('r'), curr_var_dict.pop('p'),
                    number=i, order=order, **curr_var_dict
                )

            elif dist is Distribution.POISSON:
                var_list[idx] = PoissonVariable(
                    curr_var_dict.pop('lambda'), number=i, order=order,
                    **curr_var_dict
                )

            elif dist is Distribution.HYPERGEOMETRIC:
                var_list[idx] = HypergeometricVariable(
                    curr_var_dict.pop('M'), curr_var_dict.pop('n'),
                    curr_var_dict.pop('N'), number=i, order=order,
                    **curr_var_dict
                )

            elif dist is Distribution.DISCRETE:
                var_list[idx] = DiscreteVariable(
                    curr_var_dict.pop('pdf'), curr_var_dict.pop('interval_low'),
                    curr_var_dict.pop('interval_high'), number=i, order=order,
                    **curr_var_dict
                )

            i += size
            idx += 1

        var_list_tot = comm.allgather(var_list)
        self.var_list = np.zeros(var_count, dtype=object)

        for i in range(size):

            idx = i
            var_cnk = var_list_tot[i]
            var_cnk_size = len(var_cnk)

            for j in range(var_cnk_size):
                self.var_list[idx] = var_cnk[j]

                idx += size

        return self.var_list

    def read_var_vals(self, matrix_file, attr):
        """
        Inputs: matrix_file- the file to be read from
                attr- the attribute of the variables to set this data to
        
        Reads the values in the file and sets them to the specified 
        attribute for each variable.
        """
        try:
            with open(matrix_file, 'r') as fi:
                dat = fi.readlines()

            line_count = len(dat)
            var_count = len(self.var_list)
            ind = 0

            for i in range(var_count):
                arr = np.zeros(line_count)
                setattr(self.var_list[i], attr, arr)

            for line in dat:
                if '#' not in line and line != '\n':
                    curr_line = line.replace(',', '\t').split()

                    for i in range(var_count):
                        attribute = getattr(self.var_list[i], attr)
                        attribute[ind] = float(curr_line[i])

                    ind += 1

            for i in range(var_count):
                setattr(self.var_list[i], attr, getattr(self.var_list[i], attr)[0:ind])

        except FileNotFoundError:
            pass


def write_gen_resps(results, file_name='results_generated.dat'):
    """
    Inputs: results- the array of generated results
            file_name- the name of the output results file
    
    Writes the results array to a file in the UQPCE format.
    """
    with open(file_name, 'w') as res_gen:

        gen_vals = results[list(results)[0]]
        res_gen.write('\n'.join((str(i) for i in gen_vals)))
