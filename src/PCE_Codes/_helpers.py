from builtins import getattr
import os
from warnings import warn, showwarning

try:
    from numpy.linalg import inv, pinv, cond, solve
    from scipy.stats import pearsonr
    from sympy import symbols
    from sympy.utilities.lambdify import lambdify
    from sympy.parsing.sympy_parser import parse_expr
    import matplotlib.pyplot as plt
    from mpi4py.MPI import (
        DOUBLE as MPI_DOUBLE, COMM_WORLD as MPI_COMM_WORLD, SUM as MPI_SUM
    )
    import numpy as np
except:
    warn('Ensure that all required packages are installed.')
    exit()

comm = MPI_COMM_WORLD
size = comm.size
rank = comm.rank

is_manager = (rank == 0)


def switch_backend(backend):
    """
    Inputs: backend- the backend to change to
    
    A wrapper for the matplotlib method that switches the backend.
    """
    plt.switch_backend(backend)


def user_function(user_func_str, var_list):
    """ 
    Inputs: user_func_str- a string with which responses will be generated
            var_list- the list of variables (which contain the values used 
            to create the responses)
    
    Generates responses for an function based on the corresponding variable
    values.
    """
    var_count = len(var_list)
    var_list_symb = [''] * var_count

    user_func_str = parse_expr(''.join((user_func_str.split())))

    for j in range(var_count):
        var_list_symb[j] = symbols(f'x{j}')

    func = lambdify((var_list_symb,), user_func_str, ('numpy', 'sympy'))
    gen_res = func([variable.vals for variable in var_list])

    return {'generated_responses': gen_res}


def get_str_vars(matrix):
    """
    Inputs: matrix- the model matrix of the different interraction terms
    
    Generates strings that represent the sobol interactions. (i.e. 'x0**2')
    """
    length, width = matrix.shape
    out_strings = [''] * (length - 1)

    for row in range(1, length):
        curr = ''

        for col in range(width):
            curr_elem = matrix[row][col]

            if curr_elem != 0.:
                if curr_elem != 1.:
                    curr += f'x{col}^{int(matrix[row][col])}*'

                else:
                    curr += f'x{col}*'
        out_strings[row - 1] = curr[0:-1]

    return out_strings


def create_total_sobols(var_count, matrix, sobols):
    """
    Inputs: var_count- number of variables
            matrix- the interaction matrix
            sobols- the previously-calculated sobol indices for each interaction
            
    Using the existing sobol indices, the total sobol index for each variable 
    is created. The string for the output is created.
    """
    comm = MPI_COMM_WORLD
    size = comm.size
    rank = comm.rank

    mat_size = len(matrix)
    iter_count = mat_size - 1

    base = iter_count // size
    rem = iter_count % size
    beg = base * rank + (rank >= rem) * rem + (rank < rem) * rank + 1
    count = base + (rank < rem)
    end = beg + count

    total_sobols = np.zeros(var_count)
    temp_sobols = np.zeros(var_count)

    for i in range(beg, end):
        for j in range(var_count):
            if matrix[i, j] != 0:
                temp_sobols[j] += sobols[i - 1]

    comm.Allreduce(
        [temp_sobols, MPI_DOUBLE], [total_sobols, MPI_DOUBLE], op=MPI_SUM
    )

    return total_sobols


def check_directory(directory, verbose):
    """
    Inputs: directory- the directory for the graphs to be place
            verbose- the verbose flag
    
    Checks to see if the graph directory exists. If it doesn't exit, the 
    folder is created.
    """

    if not os.path.isdir(directory):

        if is_manager:
            os.mkdir(directory)

            if verbose:
                print(f'Making directory {directory}\n')

    else:
        directory_exists = True
        i = 0

        while directory_exists:
            i += 1

            if not os.path.isdir(f'{directory}_{i}'):
                directory = f'{directory}_{i}'

                if is_manager:
                    os.mkdir(directory)

                    if verbose:
                        print(f'Making directory {directory}\n')

                directory_exists = False

    return directory


def evaluate_points_verbose(func, begin, total_samps, var_list, attr):
    """
    Inputs: func- the lambda function used to generate the data from the 
            evaluation vector
            begin- the index to start at in the `attr` array
            total_samps- the total number of samples to generate
            var_list- list of the variables
            attr- the attribute that holds the values to be used in the 
            evaluation vector
            
    From a specified attirbute, a lambda function is used to generate 
    values that populate matrix.
    """
    rank = MPI_COMM_WORLD.rank

    var_count = len(var_list)
    term_count = func(np.zeros(var_count)).shape

    if rank == 0:
        inter_vals = (np.arange(0.1, 1.1, 0.1) * total_samps).astype(int)

    if len(term_count) > 0:
        term_count = term_count[1]  # len(func(np.zeros(var_count)))
    else:
        term_count = 1

    eval_vect = np.zeros([total_samps, var_count])
    matrix = np.zeros([total_samps, term_count])

    end = begin + total_samps

    for j in range(var_count):
        attr_arr = getattr(var_list[j], attr)
        eval_vect[:, j] = attr_arr[begin:end].T

    for i in range(total_samps):
        matrix[i, :] = func(eval_vect[i, :])

        if rank == 0 and (inter_vals is not None) and (i + 1 + begin in inter_vals):
            print(f'{(i+1)/total_samps:.0%} Complete\n')

    return matrix


def evaluate_points(func, begin, total_samps, var_list, attr):
    """
    Inputs: func- the lambda function used to generate the data from the 
            evaluation vector
            begin- the index to start at in the `attr` array
            total_samps- the total number of samples to generate
            var_list- list of the variables
            attr- the attribute that holds the values to be used in the 
            evaluation vector
    
    Identical to evaluate_points_verbose, but doesn't check for a verbose 
    option every iteration. This version also deals with indexing only part of
    eval_vect.
    """
    var_count = len(var_list)
    term_count = func(np.zeros(var_count)).shape

    if len(term_count) > 0:
        term_count = term_count[1]  # len(func(np.zeros(var_count)))
    else:
        term_count = 1

    eval_vect = np.zeros([total_samps, var_count])
    matrix = np.zeros([total_samps, term_count])

    end = begin + total_samps
    for j in range(var_count):
        attr_arr = getattr(var_list[j], attr)
        eval_vect[:, j] = attr_arr[begin:end].T

    for i in range(total_samps):
        matrix[i, :] = func(eval_vect[i, :])

    return matrix


def calc_difference(array_1, array_2):
    """
    Inputs: array_1- the array being subtracted from
            array_2- the array being subtracted
    
    Finds difference between the two input arrays.
    """
    return array_1 - array_2


def calc_mean_err(error):
    """
    Inputs: error- an array of error values
    
    Calculates the mean of the error.
    """
    return np.mean(np.abs(error))


def uniform_hypercube(low, high, samp_size=1):
    """
    Inputs: low- the low bound of the hypercube
            high- the high bound of the hypercube
            samp_size- the number of samples to generate
    
    Generates a uniformly-distributed Latin Hypercube.
    """
    intervals = np.linspace(low, high, samp_size + 1)
    vals = np.zeros(samp_size)

    for i in range(samp_size):
        vals[i] = np.random.uniform(low=intervals[i], high=intervals[i + 1])

    np.random.shuffle(vals)

    return vals


def solve_coeffs(var_basis, responses):
    """
    Inputs: var_basis- the variable basis matrix
            responses- the array of responses
    
    Uses the matrix system to solve for the matrix coefficients.
    """
    cond_num_thresh = 20

    var_basis = np.atleast_2d(var_basis)
    var_basis_T = np.transpose(var_basis)
    basis_transform = np.dot(var_basis_T, var_basis)

    cond_num = cond(basis_transform, -np.inf)

    if cond_num > cond_num_thresh:
        warn(
            'The condition number of the matrix used to solve for the matrix '
            f'coefficients is a large value, {cond_num}.'
        )

    matrix_coeffs = solve(basis_transform, np.dot(var_basis_T, responses))

    return matrix_coeffs


def generate_sample_set(var_list, sample_count=1):
    """
    Inputs: var_list- the list of varibles
            sample_count- the number of samples to generate
    
    Creates and returns a random, standardized value for each variable present.
    """
    # generate random samples for each variable
    var_count = len(var_list)
    test_points = np.zeros([var_count, sample_count])

    i = 0
    for var in var_list:
        test_points[i, :] = var.standardize_points(
            var.generate_samples(sample_count)
        )

        i += 1

    return test_points


def unstandardize_set(var_list, sample_array):
    """
    Inputs: var_list- list of variables
            sample_array- array with one sample corresponding to each variable
    
    Takes an array of standardized values, unstandardizes them, and returns the 
    array of unstandardized values.
    """
    var_count, iters = sample_array.shape
    unstandard_array = np.zeros((var_count, iters))

    for i in range(var_count):
        unstandard_array[i] = var_list[i].unstandardize_points(sample_array[i])

    return unstandard_array


def standardize_set(var_list, sample_array):
    """
    Inputs: var_list- list of variables
            sample_array- array with one sample corresponding to each variable
    
    Takes an array of unstandardized values, standardizes them, and returns the 
    array of standardized values.
    """
    var_count, iters = sample_array.shape
    standard_array = np.zeros((var_count, iters))

    for i in range(var_count):
        standard_array[i] = var_list[i].standardize_points(sample_array[i])

    return standard_array


def check_error_trends(var_list, error, order, thresh=0.5, shift=2):
    """
    Inputs: var_list- list of variables
            error- the array of error
            order- the order to start the error checking at
            thresh- the minimum pearsonr value the correlation must be at to get 
            flagged as an error correlation
            shift- how many orders higher than input `order` to test for 
            correlations
    
    Returns the names of the variables that have a pearsonr correlation higher 
    than 'thresh.'
    """
    i = 0
    var_cnt = len(var_list)
    shift_order = order + shift
    exp_range = range(1, shift_order + 1)

    corr = np.zeros(var_cnt)

    for i in range(var_cnt):
        correlate = np.zeros(shift_order)
        var = var_list[i]

        for exp in exp_range:
            # Check for correlations between the error and x^N
            pear = pearsonr(error, var.std_vals ** exp)[0]
            if np.abs(pear) > thresh:
                correlate[exp - order] = 1

        corr[i] = correlate.any()

        i += 1

    names = [var_list[i].name for i in range(var_cnt) if corr[i]]

    return names


def check_error_magnitude(error):
    """
    Inputs: error- the array of error
    
    Checks for large outliers in the error.
    """
    const = 3
    largest_mag = np.max(np.abs(error))

    mean_error = calc_mean_err(error)

    if largest_mag >= const * mean_error:
        text = (
            'The error has large outliers. The order may not be high enough to '
            'capture the interactions.'
        )

    else:
        text = 'There are no error outliers.'

    return text


def _warn(warn_message, *args, **kwargs):
    """
    Inputs: warn_message- the warning message
    
    Used to override "warnings.formatwarning" to output only the warning 
    message.
    """
    return f'{warn_message}\n\n'


def calc_sobols(matrix_coeffs, norm_sq):
    """
    Inputs: matrix_coeffs- the matrix coefficient of the model
            norm_sq- the norm squared of the model
    
    Returns the sobols from the matrix coefficients and norm squared.
    """
    min_model_size = len(matrix_coeffs)

    norm_sq = norm_sq.reshape(min_model_size, 1)

    matrix_coeffs_sq = (
        np.reshape(
            matrix_coeffs, (len(matrix_coeffs), 1)
        )[1:] ** 2
    )

    prod = (norm_sq[1:] * matrix_coeffs_sq)
    sigma_sq = np.sum(prod)

    sobols = np.zeros(min_model_size - 1)

    for i in range(1, min_model_size):
        sobols[i - 1] = (
            (matrix_coeffs[i] ** 2 * norm_sq[i]) / sigma_sq
        )

    return sobols

