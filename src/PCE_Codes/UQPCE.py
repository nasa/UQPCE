#!/usr/bin/env python
from builtins import setattr, getattr
from enum import auto, Enum
from fractions import Fraction
import math
from multiprocessing import Process
from multiprocessing import Process, Manager
import os
from warnings import showwarning, warn

from numpy.linalg import inv, pinv
from scipy import stats
from scipy.integrate import quad
from scipy.linalg import pascal
from scipy.stats import norm, beta, gamma, expon
from sympy import (
    symbols, Matrix, zeros, ones, integrate, N, sqrt, pi, exp, sinh, cosh, Mul,
    sympify, simplify, factorial
)
from sympy.core.numbers import NaN, Float, One
from sympy.integrals.integrals import Integral
from sympy.parsing.sympy_parser import parse_expr
from sympy.polys.polyerrors import CoercionFailed
from sympy.solvers import solve
from sympy.utilities.lambdify import lambdify
from yaml import Loader
import yaml

import matplotlib.pyplot as plt
import numpy as np


def switch_backend(backend):
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
    for j in range(var_count):
        var_list_symb[j] = symbols(f'x{j}')
    func = lambdify((var_list_symb,), user_func_str, modules='numpy')
    gen_res = func([variable.vals for variable in var_list])
    return({'generated_responses': gen_res})


def get_str_vars(matrix):
    """
    Inputs: matrix- the model matrix of the different interraction terms
    
    Generates strings that represent the sobol interractions. (i.e. 'x0**2')
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
    return(out_strings)


def read_file(file_name):
    """
    Inputs: file_name- name of the file containing the responses
    
    Reads the file and appends the values to a list
    """
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
            if "#" not in line:
                curr_line = line.replace(',', '\t').split()
                i = 0
                for key in keys:
                    results[key][ind] = curr_line[i]
                    i += 1
                ind += 1

        results = {name : results[name][0:ind] for name in results.keys()}

    return(results)


def create_total_sobols(var_count, matrix, sobols):
    """
    Inputs: var_count- number of variables
            matrix- the interaction matrix
            sobols- the previously-calculated sobol indices for each interaction
            
    Using the existing sobol indices, the total sobol index for each variable 
    is created. The string for the output is created.
    """
    total_sobols = np.zeros(var_count)
    for i in range(var_count):
        for j in range(1, len(matrix)):
            if matrix[j, i] != 0:
                total_sobols[i] += sobols[j - 1]

    return(total_sobols)


def check_directory(directory, verbose):
    """
    Inputs: graph_directory- the directory for the graphs to be place
            verbose- the verbose flag
    
    Checks to see if the graph directory exists. If it doesn't exit, the 
    folder is created.
    """
    cwd = os.getcwd() + '/'
    if not os.path.isdir(cwd + directory):
        if verbose:
            print('Making output directory\n')
        os.mkdir(cwd + directory)
    else:
        directory_exists = True
        i = 0
        while directory_exists:
            i += 1
            if not os.path.isdir(cwd + directory + f'_{i}'):
                directory = directory + f'_{i}'
                os.mkdir(cwd + directory)
                directory_exists = False
    return(directory)


def evaluate_points(func, begin, total_samps, matrix, var_list, attr,
                    verbose=None, inter_vals=None):
    """
    Inputs: func- the lambda function used to generate the data from the 
                  evaluation vector
            begin- the first index of matrix to set
            total_samps- the total number of samples to generate
            matrix- the matrix where the transformed data will reside
            var_list- list of the variables
            attr- the attribute that holds the values to be used in the 
            evaluation vector
            
    From a specified attirbute, a lambda function is used to generate 
    values that populate matrix.
    """
    var_count = len(var_list)
    term_count = func(np.zeros(var_count)).shape

    if len(term_count) > 0:
        term_count = term_count[1]  # len(func(np.zeros(var_count)))
    else:
        term_count = 1

    eval_vect = np.zeros([total_samps, var_count])
    end = begin + total_samps

    for j in range(var_count):
        attr_arr = getattr(var_list[j], attr)
        eval_vect[:, j] = attr_arr[begin:end].T

    for i in range(total_samps):
        scaled_ind = i + begin
        matrix[scaled_ind, :] = func(eval_vect[i, :])

        if verbose and (i in inter_vals):
            percent = i / total_samps * 100
            print('{:.1f}% Complete\n'.format(percent))


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

    return(vals)


def calc_difference(array_1, array_2):
    """
    Inputs: array_1- the array being subtracted from
            array_2- the array being subtracted
    
    Finds difference between the two input arrays.
    """
    return(array_1 - array_2)


def solve_coeffs(var_basis, responses):
    """
    Uses the matrix system to solve for the matrix coefficients.
    """
    var_basis_T = np.transpose(var_basis)

    try:
        transformed_var_basis = inv(np.dot(var_basis_T, var_basis))
    except np.linalg.LinAlgError:
        warn('Unable to invert singular matrix, using Moore-Penrose pseudo-inverse')
        transformed_var_basis = pinv(np.dot(var_basis_T, var_basis))

    matrix_coeffs = (
        np.dot(transformed_var_basis, np.dot(var_basis_T, responses))
    )

    return(matrix_coeffs)


def calc_PRESS_res(var_basis, responses, var_basis_func, var_list):
    """
    Inputs: var_basis- evaluated variable basis
            responses- the responses
            var_basis_func- the function that calculates the model variable 
            basis from the input point
    
    An efficient way to calculate the PRESS residual for a given model.
    """
    resp_count = len(responses)
    var_count = len(var_list)

    err_ver_sq = np.zeros(resp_count)
    excl_point = np.zeros(var_count)

    for idx in range(resp_count):
        incr_idx = idx + 1

        for i in range(var_count):
            excl_point[i] = var_list[i].std_vals[idx]

        temp_basis = np.append(var_basis[:idx], var_basis[incr_idx:], axis=0)
        temp_resps = np.append(responses[:idx], responses[incr_idx:])
        matrix_coeffs = solve_coeffs(temp_basis, temp_resps)

        ver_pnt = var_basis_func(excl_point)
        ver_pred = np.matmul(ver_pnt, matrix_coeffs)
        err_ver = calc_difference(responses[idx], ver_pred)
        err_ver_sq[idx] = (err_ver) ** 2

    press_res = np.sum(err_ver_sq)

    return press_res


def _warn(warn_message, *args, **kwargs):
        return(str(warn_message) + '\n\n')

#******************************************************************************


class DataSet:
    """
    Reads in, creates, and sets up the variables from the input file. 
    Reads from the matrix file to set the 'vals' (or 'verify_vals') of each 
    variable.
    """

    def __init__(self):
        self.input = True
        self.verbose = False
        self.var_list = []
        showwarning = _warn

    def read_var_input(self, input_file, args, order, verbose):
        """
        Inputs: input_file- file name for yaml file containing variables
                args- the arguments from the command line parser
                order- the order value at this step in the program
                verbose- the verbose value at this step in the program
        
        Takes in a yaml file, read it, and constructs Variable types in the
        appropriate subclass
        """
        with open(input_file, "r") as fi:
            dat = fi.read()
        var_dict = dict(yaml.load(dat, Loader=Loader))

        try:  # check for the Settings dictionary
            settings = var_dict.pop('Settings')
        except KeyError:
            settings = None  # Settings dictionary not present

        try:  # check if order is listed; must be seperate so 'Settings' isn't
            order = settings['order']  # overwritten if 'order' isn't present
        except KeyError:
            pass  # not present; don't change order
        except TypeError:
            pass

        try:
            verbose = settings['verbose']
        except KeyError:
            pass  # not present; don't change verbose
        except TypeError:
            pass

        order = getattr(args, 'order', order)
        verbose = getattr(args, 'verbose', verbose)

        if verbose:
            print('Initializing Problem\n\nSetting up Variables\n')

        var_names = list(var_dict)
        var_count = len(var_dict)

        for i in range(var_count):
            curr_var_dict = var_dict[var_names[i]]

            try:
                dist = Distribution[curr_var_dict['distribution'].upper()]
                if dist is Distribution.NORMAL:
                    self.var_list.append(NormalVariable(i, order))
                elif dist is Distribution.UNIFORM:
                    self.var_list.append(UniformVariable(i, order))
                elif dist is Distribution.BETA:
                    self.var_list.append(BetaVariable(i, order))
                elif dist is Distribution.EXPONENTIAL:
                    self.var_list.append(ExponentialVariable(i, order))
                elif dist is Distribution.GAMMA:
                    self.var_list.append(GammaVariable(i, order))
            except KeyError:
                dist = curr_var_dict['distribution']
                self.var_list.append(Variable(i, order))
            except AttributeError:
                dist = curr_var_dict['distribution']
                self.var_list.append(Variable(i, order))

            curr_var = self.var_list[i]
            curr_var.name = f'x{i}'
            curr_var.distribution = dist
            curr_var.initialize(curr_var_dict)
            curr_var.check_num_string()

        self.var_count = len(self.var_list)
        return(self.var_list, settings)

    def read_var_vals(self, matrix_file, attr):
        """
        Inputs: matrix_file- the file to be read from
                attr- the attribute of the variables to set this data to
        
        Reads the values in the file and sets them to the specified 
        attribute for each variable.
        """
        with open(matrix_file, "r") as fi:
            dat = fi.readlines()

        line_count = len(dat)
        var_count = len(self.var_list)
        ind = 0
        for i in range(var_count):
            arr = np.zeros(line_count)
            setattr(self.var_list[i], attr, arr)

        for line in dat:
            if "#" not in line:
                curr_line = line.replace(',', '\t').split()
                for i in range(var_count):
                    attribute = getattr(self.var_list[i], attr)
                    attribute[ind] = float(curr_line[i])
                ind += 1


class MatrixSystem:
    """
    Inputs: responses- the array of responses from the results file 
                       (or from the user_function)
            var_list- the lsit of variables
    
    The matrix system built from the responses and input values. The 
    MatrixSystem is built and solved to acquire the matrix coefficients in 
    the systme of equations.
    """

    def __init__(self, responses, var_list):
        self.input = True
        self.verbose = False
        self.inter = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        self.responses = responses
        self.act_model_size = len(responses)
        self.inter_vals = (self.inter * self.act_model_size).astype(int)
        self.var_list = var_list
        self.var_count = len(var_list)
        showwarning = _warn

    def form_norm_sq(self, order):
        """
        Inputs: order- the order of polynomial expansion 
        
        Creates the model matrix and the corresponding norm squared matrix.
        """
        self.min_model_size = int(math.factorial(order + self.var_count) \
                                  / (math.factorial(order) \
                                   * math.factorial(self.var_count)))

        ident_matrix = np.identity(self.var_count)
        prev_matrix = np.zeros([self.var_count, self.var_count])
        final = np.zeros([1, self.var_count])

        if self.min_model_size > self.act_model_size:
            raise ValueError(f'For order={order} and {self.var_count} variables, '
                             'you need to have at least '
                             f'{self.min_model_size+1} responses and matrix '
                             'elements.\n')

        for i in range(order):  # interraction matrix formation
            prev_cols, prev_rows = prev_matrix.shape
            curr_size = prev_cols * self.var_count + 1
            model_matrix = np.zeros([curr_size, self.var_count])
            ind = 0
            for row in range(prev_cols):
                for j in range(self.var_count):
                    model_matrix[ind] = ident_matrix[j, :] + \
                        prev_matrix[row, :]
                    ind += 1
            sorted_indices = np.unique(model_matrix, axis=0, return_index=True)
            prev_matrix = model_matrix[np.sort(sorted_indices[1])]
            final = np.append(final, prev_matrix, axis=0)
        indices = np.unique(final, axis=0, return_index=True)
        self.model_matrix = final[np.sort(indices[1])]

        # Norm Square Matrix formation
        self.norm_sq = np.ones([self.min_model_size, 1])
        for i in range(self.min_model_size):
            for j in range(self.var_count):
                self.norm_sq[i] *= self.var_list[j].get_norm_sq_val(
                    int(self.model_matrix[i, j]))
        return(self.min_model_size, self.model_matrix, self.norm_sq)

    def build(self):
        """
        Builds the symbolic 'psi' matrix that represents the interactions 
        of the variables.
        """
        self.var_basis_mat_symb = Matrix(zeros(self.min_model_size,
                                               self.var_count))
        self.var_basis_vect_symb = Matrix(ones(1, self.min_model_size))

        inter_vals = (
                np.floor(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                 * self.min_model_size)
        )

        matrix_size = len(self.model_matrix)
        for i in range(self.var_count):
            self.var_basis_mat_symb[:, i] = self.var_list[i].get_var_basis(
                matrix_size, self.model_matrix, i)

        percent = 10

        for i in range(self.min_model_size):
            curr_vect = 1
            curr_var_basis_row = self.var_basis_mat_symb[i, :]
            for k in curr_var_basis_row:
                curr_vect *= k
            self.var_basis_vect_symb[i] = curr_vect

            if self.verbose and i in inter_vals:
                print('{:.1f}% Complete\n'.format(percent))
                percent += 10

        return(self.var_basis_vect_symb)

    def evaluate(self):
        """
        Fills the symbolic 'psi' variable basis system with the numbers that 
        correspond to the variables in the matrix.
        """
        self.var_basis_sys_eval = np.zeros([self.act_model_size,
                                               self.min_model_size])
        var_list_symb = [''] * self.var_count
        for j in range(self.var_count):
            var_list_symb[j] = symbols(f'x{j}')

        var_basis_vect_func = lambdify((var_list_symb,),
                                       self.var_basis_vect_symb, modules='numpy')

        evaluate_points(var_basis_vect_func, 0, self.act_model_size,
                        self.var_basis_sys_eval, self.var_list, 'std_vals',
                        verbose=self.verbose, inter_vals=self.inter_vals)

        return(self.var_basis_sys_eval)

    def solve(self):
        """
        Uses the matrix system to solve for the matrix coefficients.
        """
        self.var_basis = self.var_basis_sys_eval
        self.matrix_coeffs = solve_coeffs(self.var_basis, self.responses)

        return(self.matrix_coeffs, self.var_basis)


class SurrogateModel:
    """
    Inputs: responses- the array of responses from the results file 
                       (or from the user_function)
            matrix_coeffs- the matrix coefficients solved for by MatrixSystem
    
    Gets the sobol indices of the varibles. Performs several calculations 
    and checks on the model to check if it is a good representation.
    """

    def __init__(self, responses, matrix_coeffs):
        self.verbose = False
        self.responses = responses
        self.act_model_size = len(responses)
        self.matrix_coeffs = matrix_coeffs
        showwarning = _warn

    def get_sobols(self, norm_sq, min_model_size):
        """
        Inputs: norm_sq- the norm squared matrix
                min_model_size- the number of terms necessary to build the 
                model
                
        Solves for the sobol sensitivities
        """
        self.sobols = np.zeros(min_model_size - 1)
        for i in range(1, min_model_size):
            self.sobols[i - 1] = (self.matrix_coeffs[i] ** 2 * norm_sq[i]) \
                / self.sigma_sq
        return(self.sobols)

    def calc_var(self, norm_sq):
        """
        Inputs: norm_sq- the norm squared matrix
        
        Calculates the mean and variance in the responses
        """
        self.resp_mean = self.matrix_coeffs[0]
        matrix_coeffs_sq = np.reshape(self.matrix_coeffs,
                                      (len(self.matrix_coeffs), 1))[1:] ** 2
        prod = (norm_sq[1:] * matrix_coeffs_sq)
        self.sigma_sq = np.sum(prod)
        return(self.sigma_sq, self.resp_mean)

    def calc_error(self, var_basis):
        """
        Inputs: var_basis- the varibale basis ('psi') matrix that consists 
                of values (not symbols)
        
        Solves for the calculated responses that the matrix coefficients and 
        variable basis ('alpha' and 'psi') result in as well as the difference 
        between these values and the actual values.
        """
        self.pred = np.dot(var_basis, self.matrix_coeffs)
        self.error = self.pred - self.responses
        return(self.error, self.pred)

    def calc_mean_error(self, error):
        """
        Inputs: error- an array of error values
        
        Calculates the mean of the error.
        """
        return(np.mean(np.abs(error)))

    def calc_difference(self, verify, pred):
        """
        Inputs: verify- the verification responses
                pred- the predicted responses
        
        Finds difference between the verification values and the predicted 
        values.
        """
        self.err_verify = pred - verify
        return(self.err_verify)

    def check_normality(self, var_basis_sys_eval, min_model_size, sig,
                        graph_directory=None):
        """
        Inputs: var_basis_sys_eval- the variable basis system (not symbolic) 
                matrix
                min_model_size- the minimum size that the model can to be
                sig- the level of significance for the Shapiro-Wilks test
                graph_directory- the directory that the graphs are put into
        
        Ensures that the err follows a normal distribution.
        """
        var_basis_sys_eval = np.array(var_basis_sys_eval).astype(np.float64)
        std_err_matrix = np.zeros([self.act_model_size])
        [test_stat, p_val_hypoth] = stats.shapiro(self.error)
        if p_val_hypoth < sig:
            shapiro_results = 'Shapiro-Wilks test statistic is {:.5f}, P-value '\
            'is {:.5f}\n\nEvidence exists that errors are not from a normal '\
            'distribution\n'.format(test_stat, p_val_hypoth)
        if p_val_hypoth > sig:
            shapiro_results = 'Shapiro-Wilks test statistic is {:.5f}, P-value '\
            'is {:.5f}\n\nInsufficient evidence to infer errors are not from a '\
            'normal distribution\n'.format(test_stat, p_val_hypoth)
        if self.verbose:
            print(shapiro_results)

        var_basis_sys_eval_T = np.transpose(var_basis_sys_eval)

        hat_matrix = np.matmul(
            np.matmul(
                var_basis_sys_eval,
                inv(np.matmul(var_basis_sys_eval_T, var_basis_sys_eval))
            ),
            var_basis_sys_eval_T
        )

        mean_sq_error = (
            np.matmul(np.transpose(self.responses), self.responses)
            -np.matmul(
                np.matmul(
                    np.transpose(self.matrix_coeffs),
                    var_basis_sys_eval_T
                ),
                self.responses
            )
        ) / (self.act_model_size - min_model_size)

        if (np.min(mean_sq_error) < 0):
            warn('Negative mean squared error\nUsing an alternate equation.')
            mean_sq_error = np.sum(self.error ** 2) / len(self.error)

        elif mean_sq_error == 0:
            warn('Mean squared error was 0\nGiving mean squared error '
                 'value 1e-6.\n')
            mean_sq_error = 1e-6

        sigma = np.sqrt(mean_sq_error)

        for i in range(self.act_model_size):
            std_err_matrix[i] = self.error[i] \
                / (sigma * np.sqrt(1 - hat_matrix[i, i]))

        try:
            if graph_directory is not None:
                plt.hist(std_err_matrix)
                image_path = f'{graph_directory}/error_distribution'
                plt.title('Error_distribution')
                plt.savefig(image_path, dpi=600, bbox_inches='tight')
                plt.clf()
                stats.probplot(self.error, plot=plt)
                image_path = f'{graph_directory}/normal_prob'
                plt.savefig(image_path, dpi=600, bbox_inches='tight')
                plt.clf()
        except ValueError:
            warn('The histogram of the errors was not successfully created.'
                 'Errors were zero.')

        return(mean_sq_error, hat_matrix, shapiro_results)

    def verify(self, var_basis_vect_symb, var_list, verify_size):
        """
        Inputs: var_basis_vect_symb- the symbolic variable basis vector (psi)
                var_list- the list of variables
                verify_size- the size of the verification responses
        
        Verifies the surrogate model by outputting the responses from 
        verification input values put into the model.
        """
        self.var_list = var_list
        self.var_count = len(var_list)
        self.var_basis_sys_eval_verify = np.zeros([verify_size,
                                                      len(self.matrix_coeffs)])

        var_list_symb = [symbols(variable.var_str) for variable in self.var_list]

        var_basis_vect_func = lambdify((var_list_symb,),
                                       var_basis_vect_symb, modules='numpy')

        evaluate_points(var_basis_vect_func, 0, verify_size,
                        self.var_basis_sys_eval_verify, self.var_list,
                        'std_verify_vals')

        self.R_verify_pred = np.matmul(self.var_basis_sys_eval_verify,
                                       self.matrix_coeffs)

        return(self.R_verify_pred, self.var_basis_sys_eval_verify)


class Graphs:
    """
    Inputs: standardize- boolean for if graphs should be standardized or not
    
    Creates the plots of the variable values vs some other value. Plots the 
    model error vs the predicted responses.
    """

    def __init__(self, standardize):
        self.input = True
        self.standardize = standardize
        self.verbose = False
        showwarning = _warn

    def factor_plots(self, graph_directory, var_list, plot_data, plot_name,
                     verify=None):
        """ 
        Inputs: graph_directory- file location where to put plots
                var_list- list of variables
                plot_data- the data to be plotted
                plot_name- 'Predicted' or 'Error'; what data is 
                being plotted
                verify- if these points are the verification points or the 
                input points
        
        Generates plots for each variable against plot_data.
        """
        var_count = len(var_list)
        if self.verbose:
            print(f'Generating {plot_name} vs Factor graphs\n')
        for j in range(var_count):
            curr_var = var_list[j]
            if not verify:
                if self.standardize:
                    plt.scatter(curr_var.std_vals, plot_data)
                    plt.title(f'{plot_name} vs {curr_var.name} (Standardized)')
                else:
                    plt.scatter(curr_var.vals, plot_data)
                    plt.title(f'{plot_name} vs {curr_var.name}')
            else:
                if self.standardize:
                    plt.scatter(curr_var.std_verify_vals, plot_data)
                    plt.title(f'{plot_name} vs {curr_var.name} (Standardized)')
                else:
                    plt.scatter(curr_var.verify_vals, plot_data)
                    plt.title(f'{plot_name} vs {curr_var.name}')

            plt.xlabel(f'{curr_var.name}')
            plt.ylabel(f'{plot_name}')
            image_path = graph_directory + f'/{plot_name}_vs_' + curr_var.name
            plt.savefig(image_path, dpi=600, bbox_inches='tight')
            plt.clf()

    def error_vs_pred(self, graph_directory, err, pred, plot_name):
        """
        Inputs: graph_directory- file location where to put plots
                err- difference between predicted vals and actual vals
                pred- the predicted values
                plot_name- the name of the plot
        
        Generates a plot of the error vs the predicted values.
        """
        if self.verbose:
            print('Generating Error vs Predicted graph\n')
        plt.scatter(pred, err)
        plt.title(f'{plot_name}')
        plt.xlabel('predicted values')
        plt.ylabel('error')
        plt.savefig(graph_directory + f'/{plot_name}',
                    dpi=600, bbox_inches='tight')
        plt.clf()


class ProbabilityBoxes:
    """
    Inputs: var_list- the list of variables
    
    The probability box (pbox) plots that show the confidence interval from 
    the data.
    """

    def __init__(self, var_list):
        self.input = True
        self.verbose = False
        self.var_list = var_list
        self.var_count = len(var_list)
        self.plot = False
        self.track_convergence_off = False
        showwarning = _warn

    def generate_variable_str(self):
        """
        Generates the symbolic variable list.
        """
        self.var_list_symb = [symbols(variable.var_str) for variable in
                              self.var_list]
        return(self.var_list_symb)

    def count_epistemic(self):
        """
        Counts the number of epistemic variables present.
        """
        self.epistemic_list = []
        for i in range(self.var_count):
            curr_var = self.var_list[i]
            if curr_var.type is UncertaintyType.EPISTEMIC:
                self.epistemic_list.append(curr_var)
        self.epist_var_count = len(self.epistemic_list)

    def generate_aleatory_samples(self):
        """
        Inputs: samp_num- epist_samps*aleat_samps; total number of times 
                to resample
        
        Creates resample values for aleatory variables
        """
        samp_num = self.epist_samps * self.aleat_samps

        for i in range(self.var_count):
            curr_var = self.var_list[i]
            if curr_var.type is UncertaintyType.ALEATORY:
                curr_var.resample = curr_var.get_resamp_vals(samp_num)

    def generate_epistemic_samples(self, epist_samps, aleat_samps):
        """
        Inputs: epist_samps- number of times to sample for an epistemic var
                aleat_samps- number of times to sample for an aleatory var
        
        Determines the number of samples needed and generates the resample 
        values for epistemic variables.
        """
        self.epist_samps = epist_samps
        self.aleat_samps = aleat_samps

        if self.verbose:
            print('Resampling surrogate model...\n')

        if self.epist_var_count == 0:
            self.epist_samps = 1
            if self.verbose:
                print('No epistemic variables found, perfoming pure '
                      'aleatory analysis\n')

            if self.verbose and self.track_convergence_off:
                print('Defaulting to {:.1f}k aleatory samples\n'.format(
                        float(self.aleat_samps / 1000)))

        self.total_samps = self.epist_samps * self.aleat_samps
        self.eval_resps = np.zeros([self.total_samps, 1])

        for epistemic_var in self.epistemic_list:
            epistemic_var.resample = np.zeros([self.total_samps, 1])

        ones = np.ones([self.aleat_samps, 1])
        for i in range(self.epist_samps):
            i_begin = i * self.aleat_samps
            i_end = i_begin + self.aleat_samps
            for epistemic_var in self.epistemic_list:
                if i == 0:
                    epistemic_var.resample[i_begin:i_end] = ones * -1  # bound
                elif i == 1:
                    epistemic_var.resample[i_begin:i_end] = ones  # bound
                else:
                    epistemic_var.resample[i_begin:i_end] = ones \
                        * np.random.uniform(-1, 1, 1)

    def evaluate_surrogate(self, var_basis_vect_symb, sig, resp_mean,
                           matrix_coeffs, conv_threshold_percent,
                           graph_directory=None):
        """
        Inputs: var_basis_vect_symb- symbolic variable basis (psi) vector
                sig- the level of significance
                resp_mean- mean of the responses
                matrix_coeffs- the coefficients from solving the matrix (psi) 
                system of equations
                conv_threshold_percent- the percent of the response mean to be 
                used as the threshold for tracking convergence
                graph_directory- name of graph directory
        
        Resamples to generate responses from the model using various variable 
        inputs.
        """
        self.matrix_coeffs = matrix_coeffs
        var_basis_vect_func = lambdify((self.var_list_symb,),
                                       var_basis_vect_symb, modules='numpy')
        self.var_basis_sys_eval_resamp = np.zeros([self.total_samps, len(self.matrix_coeffs)])

        shift_thresh = 1  # 100%

        if not self.track_convergence_off:
            if self.verbose:
                print('Sampling until specified convergence rate of '
                      f'{conv_threshold_percent * 100}% is met.\n')
            lower_bound = sig / 2  # the bounds of the responses that will
            upper_bound = 1 - sig / 2  # be interpolated to get conf ints (CIs)

            aleat_run_count = (self.aleat_samps // self.aleat_sub_samp_size)
            epist_run_count = (self.epist_samps // self.epist_sub_samp_size)

            conf_int_low = [np.inf] * aleat_run_count  # lists of each iteration's
            conf_int_high = [-np.inf] * aleat_run_count  # upper and lower CIs
            outer_conf_int_low = [np.inf] * self.epist_samps
            outer_conf_int_high = [-np.inf] * self.epist_samps
            set_low = [np.inf] * (epist_run_count + 1)  # min will be overall low
            set_high = [-np.inf] * (epist_run_count + 1)  # max will be overall high

            converged = False  # individual curve converged?
            outer_converged = False  # set of curves converged?

            index = 1
            threshold = conv_threshold_percent * abs(resp_mean)
            self.eval_resps = np.zeros(self.total_samps)

            if graph_directory is not None:
                fig1 = plt.figure(1)
                conf_int_low_fig = fig1.subplots()

                fig2 = plt.figure(2)
                conf_int_high_fig = fig2.subplots()
                y_vals = np.arange(self.aleat_sub_samp_size) / (float(self.aleat_sub_samp_size))

                convergence_file = open(f'{graph_directory}'
                                        '/convergence_values.dat', 'w')

            if self.epist_samps == 1:
                out_message = f'The probability curve did not converge. '\
                                f'{self.aleat_samps} samples were used.\n'
            else:
                out_message = f'The probability curves did not converge. '\
                                f'{self.epist_samps} curves were used.\n'

            for ep in range(self.epist_samps):  # number of pbox curves to make
                begining_variable_index = ep * self.aleat_samps
                for k in range(aleat_run_count):  # max iterations per curve
                    iter_end = (k + 1) * self.aleat_sub_samp_size

                    begin_ind = k * self.aleat_sub_samp_size + begining_variable_index
                    end_ind = iter_end + begining_variable_index

                    evaluate_points(var_basis_vect_func, begin_ind,
                                    self.aleat_sub_samp_size,
                                    self.var_basis_sys_eval_resamp,
                                    self.var_list, 'resample')
                    self.var_basis_sys_eval_resamp = np.array(
                            self.var_basis_sys_eval_resamp).astype(np.float64)

                    self.eval_resps = np.matmul(self.var_basis_sys_eval_resamp,
                                                self.matrix_coeffs)

                    # finding the upper and lower conf interval for the curve
                    y_vals = np.arange(iter_end) / (float(iter_end))

                    sorted_data = np.sort(self.eval_resps[
                                    begining_variable_index:end_ind],
                                    axis=0)
                    conf_int_low[k] = np.interp(lower_bound, y_vals,
                                                sorted_data)
                    conf_int_high[k] = np.interp(upper_bound, y_vals,
                                                 sorted_data)

                    if k > 0:
                        inner_low_diff = np.abs(conf_int_low[k] - conf_int_low[k - 1])
                        inner_high_diff = np.abs(conf_int_high[k] - conf_int_high[k - 1])

                        if self.verbose and self.epist_samps == 1:
                            print('low: {:.5f}%    high: {:.5f}%\n'
                                  ''.format(np.abs(inner_low_diff / resp_mean * 100),
                                  np.abs(inner_high_diff / resp_mean * 100)))

                        if converged and self.epist_samps == 1:  # if converges twice, break
                            if (inner_low_diff < threshold) \
                            and (inner_high_diff < threshold):
                                out_message = 'The probability curve has '\
                                                'converged.\n'
                                converged = False
                                break

                            else:  # iforce convergence twice
                                converged = False

                        if (inner_low_diff < threshold) \
                        and (inner_high_diff < threshold):
                            # after the change is below the threshold, we need
                            # one more run
                            converged = True

                if graph_directory is not None:
                    x_values = np.linspace(1, k + 1, k) * self.aleat_sub_samp_size
                    conf_int_low_fig.plot(x_values, conf_int_low[0:k])
                    conf_int_high_fig.plot(x_values, conf_int_high[0:k])

                    outer_conf_int_low[ep] = np.min(conf_int_low[0:k])
                    outer_conf_int_high[ep] = np.max(conf_int_high[0:k])

                # checks to do for each set of curves for set convergence
                if (ep > 1) and ((ep + 1) % self.epist_sub_samp_size == 0):
                    index = (ep // self.epist_sub_samp_size) - 1
                    set_low[index] = np.min(outer_conf_int_low)
                    set_high[index] = np.max(outer_conf_int_high)

                    if graph_directory is not None:
                        convergence_file.write('    set: [{}, {}]\n'
                                            ''.format(set_low[index],
                                                      set_high[index]))

                    if (index > 0):
                        low_diff = np.abs(set_low[index] - set_low[index - 1])
                        high_diff = np.abs(set_high[index] - set_high[index - 1])

                        if self.verbose:
                            print('low: {:.5f}%    high: {:.5f}%\n'
                                  ''.format(np.abs(low_diff / resp_mean * 100),
                                  np.abs(high_diff / resp_mean * 100)))

                        if outer_converged:  # if converges twice, break
                            if (low_diff < threshold) and (high_diff < threshold):
                                out_message = 'The probability curves have '\
                                    'converged.\n'
                                break
                            else:  # force two convergences in a row
                                outer_converged = False

                        if (low_diff < threshold) and (high_diff < threshold):
                                # after the change is below the threshold, we need
                            outer_converged = True  # one more iter after convergence

            # calculating limits to be used for the plots y-axis
            if self.epist_samps > 1:
                outer_conf_int_low = np.array(outer_conf_int_low)
                low = outer_conf_int_low[outer_conf_int_low != np.inf]
                mx = np.max(low)
                mn = np.min(low)
                shift = (mx - mn) * shift_thresh
                low_lim_lower = mn - shift
                low_lim_upper = mx + shift

                outer_conf_int_high = np.array(outer_conf_int_high)
                high = outer_conf_int_high[outer_conf_int_high != -np.inf]
                mx = np.max(high)
                mn = np.min(high)
                shift = (mx - mn) * shift_thresh
                high_lim_lower = mn - shift
                high_lim_upper = mx + shift

            else:  # only one curve; no set
                conf_int_low = np.array(conf_int_low)
                low = conf_int_low[conf_int_low != np.inf]
                mx = np.max(low)
                mn = np.min(low)
                shift = (mx - mn) * shift_thresh
                low_lim_lower = mn - shift
                low_lim_upper = mx + shift

                conf_int_high = np.array(conf_int_high)
                high = conf_int_high[conf_int_high != -np.inf]
                mx = np.max(high)
                mn = np.min(high)
                shift = (mx - mn) * shift_thresh
                high_lim_lower = mn - shift
                high_lim_upper = mx + shift

            if graph_directory is not None and self.epist_samps == 1:
                convergence_file.write(''.join(('\nlow:  ',
                                                str(conf_int_low[0:k]),
                                                '\nhigh: ',
                                                str(conf_int_high[0:k]))))

            if graph_directory is not None:
                convergence_file.close()

                conf_int_low_fig.set_xlabel('number of evaluations')
                conf_int_low_fig.set_ylabel('low confidence interval')
                conf_int_low_fig.set_title('Convergence of Confidence Interval (low)')

                conf_int_low_fig.set_ylim(low_lim_lower, low_lim_upper)
                fig1.savefig(f'{graph_directory}/CIL_convergence',
                             dpi=1200, bbox_inches='tight')
                fig1.clf()

                conf_int_high_fig.set_xlabel('number of evaluations')
                conf_int_high_fig.set_ylabel('high confidence interval')
                conf_int_high_fig.set_title('Convergence of Confidence Interval (high)')

                conf_int_high_fig.set_ylim(high_lim_lower, high_lim_upper)
                fig2.savefig(f'{graph_directory}/CIH_convergence',
                             dpi=1200, bbox_inches='tight')
                fig2.clf()

        else:
            out_message = f'{self.total_samps} were used to resample the '\
                'model.\n'

            intervals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                  1.0]) * self.total_samps
            evaluate_points(var_basis_vect_func, 0, self.total_samps,
                            self.var_basis_sys_eval_resamp, self.var_list,
                            'resample', verbose=True, inter_vals=intervals)

            self.eval_resps = np.matmul(self.var_basis_sys_eval_resamp,
                                        self.matrix_coeffs)

        if self.verbose:
            print(out_message)

        return(self.eval_resps, out_message)

    def generate(self, eval_resps, sig, graph_directory):
        """
        Inputs: eval_resps- the evaluated responses from the pbox curves
                sig- the significance
                graph_directory- the name of the graph directory
        
        Generates the pbox plots from the eval_resps.
        """
        if self.epist_var_count == 0:
            self.epist_samps = 1
        conf_int_low = [np.inf] * self.epist_samps
        conf_int_high = [-np.inf] * self.epist_samps

        for i in range(self.epist_samps):
            try:
                sorted_data = np.sort(np.trim_zeros(eval_resps[i \
                                    * self.aleat_samps:(i + 1) \
                                    * self.aleat_samps]), axis=0)
                y_vals = np.arange(len(sorted_data)) / float(len(sorted_data))
                plt.plot(sorted_data, y_vals)

                conf_int_low[i] = np.interp(sig / 2, y_vals, sorted_data)
                conf_int_high[i] = np.interp(1 - sig / 2, y_vals, sorted_data)
            except ValueError:  # if the values converged and a
                pass  # set of curves are arrays of zeros

        conf_int_low = np.min(conf_int_low)
        conf_int_high = np.max(conf_int_high)

        if self.verbose:
            print('{:.1f}% Confidence Interval on Response'
                  ' [{:.5} , {:.5}]\n'.format(100 - sig * 100,
                                                conf_int_low, conf_int_high))
            print('Generating p-box plot\n')

        if graph_directory != None:
            plt.title('Probability Box')
            plt.xlabel('generated response')
            plt.ylabel('cumulative probability')
            plt.savefig(graph_directory + '/p-box',
                        dpi=1200, bbox_inches='tight')
            plt.clf()

        return(conf_int_low, conf_int_high)


class Variable:
    """
    Inputs: number- the number of the variable from the file
            order- the order of the model to calculate the orthogonal 
            polynomials and norm squared values
    
    Class represents an input variable that is used to create and evaluate 
    the model.
    """

    def __init__(self, number, order):
        self.number = number
        self.order = order
        self.var_str = f'x{self.number}'
        self.x = symbols(self.var_str)
        showwarning = _warn

    __slots__ = ('distribution', 'samples', 'std_vals', 'type', 'name',
                 'resample', 'verify_vals', 'std_verify_vals', 'vals',
                 'number', 'var_str', 'var_orthopoly_vect',
                 'norm_sq_vals', 'low', 'high', 'x', 'order',
                 'interval_low', 'interval_high', 'high_limit', 'low_limit',
                 'cum_dens_func', 'inverse_func', 'low_approx', 'high_approx',
                 'poly_denom')

    def initialize(self, attributes):
        """
        Inputs: attributes- all of the input names and values for the 
        attributes of the variable
        
        Sets the attributes of the variable, calculates the probability density
        function from the input equation, calculates the orthogonal polynomials 
        and norm squared values, and creates the new "interval" for creating 
        resampling values if the limits are '-oo' or 'oo'.
        """
        [setattr(self, attr, attributes[attr]) for attr in list(attributes)]
        self.check_attributes(attributes)

        # split at white space and rejoin to remove all whitespace- make safer
        self.distribution = ''.join(self.distribution.split())
        self.distribution = parse_expr(self.distribution,
                                       local_dict={'x':self.x})

        self.low = self.interval_low
        self.high = self.interval_high
        self.check_num_string()
        self.low_approx = self.low
        self.high_approx = self.high
        self.get_probability_density_func()  # make sure sum over interval = 1

        self.type = UncertaintyType[self.type.upper()]

        if self.type == UncertaintyType.EPISTEMIC:
            warn('The general Variable cannot be epistemic. For an epistemic'
                  ' variable, use the uniform distribution with type epistemic.')
            exit()

        self.recursive_var_basis(self.distribution, self.interval_low, self.interval_high, self.order)
        self.create_norm_sq(self.interval_low, self.interval_high, self.distribution)

        # convert the values into non-fractional values that SymPy can evaluate
        self.convert_to_floating_point(self.var_orthopoly_vect)
        self.convert_to_floating_point(self.norm_sq_vals)

    def check_attributes(self, attributes):
        """
        Inputs: attributes- the attributes of the variable given into the input 
                yaml file
                
        If there is an input in the yaml file that doesn't correspond to one of 
        the allowed inputs for the variable type, a warning is raised.
        """
        options = ['distribution', 'interval_low', 'interval_high', 'type',
                   'name']
        for attr in list(attributes):
            if attr not in options:
                warn(f'{self.name} : {attr} is not an attribute of '
                     'UniformVariable.')

    def get_probability_density_func(self):
        """
        Turns the input function into the corresponding probability density 
        function.
        """
        tol = 1e-5
        const = float(integrate(self.distribution, (self.x, self.low_approx, self.high_approx)))

        if isinstance(const, Integral):
            const = const.as_sum(n=500)

        const_rnd = np.round(const)

        if np.abs(const_rnd - const) < tol:
            const = const_rnd

        self.distribution = self.distribution / const

    def standardize(self, orig, std_vals):
        """
        Inputs: orig- the un-standardized values
                std_vals- the attribue name for the standardized vals
                         
        For each variable, it adds a new attribute for the standardized values 
        from the original input values.
        """
        setattr(self, std_vals, getattr(self, orig))

    def check_distribution(self):
        """
        Checks all values in an array to ensure that they are standardized.
        """
        mx = np.max(self.std_vals)
        mn = np.min(self.std_vals)

        if mx > self.high_approx or mn < self.low_approx:
            warn(
                f'Large standardized value for variable {self.name} '
                'with user distribution found. Check input and run matrix.'
            )
            return -1

    def generate_samples(self, samp_size):
        """ 
        Inputs: samp_size- the number of points needed to be generated
        
        Generates points according to the Latin hypercube; each point is in an 
        interval of equal probability.
        """
        self.vals = self._generate_samples(samp_size)

    def _generate_samples(self, samp_size):
        """ 
        Inputs: samp_size- the number of points needed to be generated
        
        Generates points according to the Latin hypercube; each point is in an 
        interval of equal probability.
        """
        decimals = 50

        try:
            y = symbols('y')

            if hasattr(self, 'failed'):
                raise AttributeError  # skip if has already gon through and failed

            # solve for the cumulative density function with 10s timeout
            if not hasattr(self, 'cum_dens_func'):
                manager = Manager()
                proc_dict = manager.dict()

                cdf_proc = Process(target=self._calc_cdf, args=(proc_dict,))
                cdf_proc.start()

                cdf_proc.join(10.0)
                if cdf_proc.is_alive():
                    cdf_proc.terminate()

                try:
                    self.cum_dens_func = proc_dict['cum_dens_func']

                except KeyError:
                    self.failed = 1
                    raise ValueError

            # solve for the inverse function with 10s timeout
                inv_proc = Process(target=self._invert, args=(proc_dict,))
                inv_proc.start()

                inv_proc.join(10.0)
                if inv_proc.is_alive():
                    inv_proc.terminate()

                try:
                    self.inverse_func = proc_dict['inverse_func']

                except KeyError:
                    self.failed = 1
                    raise ValueError

            # plug in random uniform 0 -> 1 to solve for x vals
            for i in range(len(self.inverse_func)):  # multiple solutions
                inv_func = (
                    np.vectorize(
                        lambdify(y, str(self.inverse_func[i]), ('numpy', 'sympy'))
                    )
                )

                samples = N(inv_func(uniform_hypercube(0, 1, samp_size)), decimals)

                if np.min(samples) >= self.low_approx and np.max(samples) <= self.high_approx:
                    np.random.shuffle(samples)
                    return(samples)

            if not (
                (np.min(samples) >= self.low_approx) and (np.max(samples) <= self.high_approx)
            ):
                raise ValueError

        # if cdf or inverse func can't be found, use rejection-acceptance sampling
        except (ValueError, NameError, AttributeError):
            func = lambdify(self.x, self.distribution, ('numpy', 'sympy'))

            try:
                max_val = (
                    np.max(func(np.random.uniform(self.low_approx, self.high_approx, 5000)))
                )

            except RuntimeError:
                max_val = (
                    np.max(
                        func(np.random.uniform(self.low_approx, self.high_approx, 5000))
                    ).astype('float64')
                )

            samples = np.zeros(samp_size)
            i = 0
            j = 0

            y_vals = np.random.uniform(0, max_val, samp_size)
            x_vals = np.random.uniform(self.low_approx, self.high_approx, samp_size)

            # while loop until all 'samp_size' samples have been generated
            while i < samp_size:

                if j == samp_size:
                    y_vals = np.random.uniform(0, max_val, samp_size)
                    x_vals = np.random.uniform(self.low_approx, self.high_approx, samp_size)
                    j = 0

                if y_vals[j] <= func(x_vals[j]):
                    samples[i] = x_vals[j]
                    i += 1

                j += 1

            np.random.shuffle(samples)

            return samples

    def create_norm_sq(self, low, high, func):
        """
        Inputs: low- the low interval bound for the distribution
                high- the high interval bound for the distribution
                func- the function corresponding to the distribution
        
        Calculates the norm squared values up to the order of polynomial 
        expansion based on the probability density function and its 
        corresponding orthogonal polynomials.
        """
        orthopoly_count = len(self.var_orthopoly_vect)
        self.norm_sq_vals = np.zeros(orthopoly_count)

        tries = 2
        zero = 0

        # is rounded off at 50 decimals, requiring two decimals places
        norm_sq_thresh = 1e-49

        for i in range(orthopoly_count):

            proc_dict = {}

            for j in range(tries):

                self._norm_sq(low, high, func, i, j, proc_dict)

                try:
                    if (proc_dict['out'] is not None) and (not math.isclose(proc_dict['out'], zero)):
                        self.norm_sq_vals[i] = proc_dict['out']
                        break  # only breaks inner loop

                except KeyError:
                    pass

        if (self.norm_sq_vals == zero).any():
            warn(f'Finding the norm squared for variable {self.name} failed.')

        if (self.norm_sq_vals <= norm_sq_thresh).any():
            warn(
                f'At least one norm squared value for variable {self.name} is '
                f'very small. This can introduce error into the model.'
            )

    def _norm_sq(self, low, high, func, i, region, proc_dict):
        """
        Inputs: low- the low interval bound for the distribution
                high- the high interval bound for the distribution
                func- the function corresponding to the distribution
                i- the index of the norm squared to calculate
                region- which sympy calculation to try for the norm squared
        
        An assistant to create_norm_sq; allows the norm squared calculations to 
        have a timeout if an error isn't raised and the solution isn't found 
        reasonably quickly.
        """
        proc_dict['out'] = None

        # round 0.99999999 to 1 to reduce error; if value is small, don't round
        thresh = 1e-2
        decimals = 50

        if high == 'oo':
            ul = np.inf
        elif high == 'pi':
            ul = np.pi
        elif high == '-pi':
            ul = -np.pi
        else:
            ul = high

        if low == '-oo':
            ll = -np.inf
        elif low == 'pi':
            ll = np.pi
        elif low == '-pi':
            ll = -np.pi
        else:
            ll = low

        if region == 0:
            try:
                f = lambdify(self.x, func * self.var_orthopoly_vect[i] ** 2, ('numpy', 'sympy'))
                ans = quad(f, ll, ul)[0]

                if ans > thresh:
                    proc_dict['out'] = round(ans, 7)
                else:
                    proc_dict['out'] = ans
            except:
                pass

        elif region == 1:
            try:
                f = lambdify(self.x, N(func * self.var_orthopoly_vect[i] ** 2, decimals), ('numpy', 'sympy'))
                ans = quad(f, ll, ul)[0]

                if ans > thresh:
                    proc_dict['out'] = round(ans, 7)
                else:
                    proc_dict['out'] = ans
            except:
                pass

        elif region == 2:
            try:
                f = lambdify(self.x, sympify(f'{func} * ({self.var_orthopoly_vect[i]}) ** 2'), ('numpy', 'sympy'))
                ans = quad(f, ll, ul)[0]

                if ans > thresh:
                    proc_dict['out'] = round(ans, 7)
                else:
                    proc_dict['out'] = ans
            except:
                pass

    def get_norm_sq_val(self, matrix_val):
        """
        Inputs: matrix_val- the value in the model matrix to consider
        
        Returns the norm squared value corresponding to the matrix value.
        """
        return(float(self.norm_sq_vals[matrix_val]))

    def get_var_basis(self, matrix_size, model_matrix, index):
        """
        Inputs: matrix_size- the size of the model matrix
                model_matrix- the model matrix
                index- the variable index to consider
        
        Creates the variable basis for the variable based on the values in the 
        model matrix that correspond to the variable index.
        """
        var_basis = Matrix(zeros(matrix_size, 1))
        for i in range(matrix_size):
            var_basis[i] = self.var_orthopoly_vect[int(model_matrix[i, index])]
        return(var_basis)

    def recursive_var_basis(self, func, low, high, order):
        """
        Inputs: func- the probability density function of the input equation
                low- the low bound on the variable 
                high- the high bound on the variable 
                order- the order of polynomial expansion
                
        Recursively calculates the variable basis up to the input 'order'.
        """
        decimals = 50

        if order == 0:
            self.poly_denom = zeros(self.order, 1)
            self.var_orthopoly_vect = zeros(self.order + 1, 1)
            self.var_orthopoly_vect[order] = 1
            return

        else:
            self.recursive_var_basis(func, low, high, order - 1)
            curr = self.x ** order

            integrate_tuple = (self.x, low, high)

            for i in range(order):
                orthopoly = self.var_orthopoly_vect[i]

                if self.poly_denom[i] == 0:
                    self.poly_denom[i] = integrate(
                        orthopoly ** 2 * func, integrate_tuple
                    )

                intergal_eval = N(
                    (
                        integrate(
                            self.x ** order * orthopoly * func,
                            integrate_tuple
                        )
                        / self.poly_denom[i]
                    ) * orthopoly,
                    decimals
                )

                # var_orthopoly_vect[i] needs to be evaluated first or NaN
                if isinstance(intergal_eval, NaN):
                    eval_orthopoly = N(orthopoly, decimals)

                    self.poly_denom[i] = integrate(
                        eval_orthopoly ** 2 * func, integrate_tuple
                    )

                    intergal_eval = (
                        (
                            integrate(
                                (self.x ** order * eval_orthopoly * func),
                                integrate_tuple
                            )
                            / self.poly_denom[i]
                        )
                        * orthopoly
                    )
                curr -= intergal_eval
            self.var_orthopoly_vect[order] = curr

            if order == self.order and (np.array(self.var_orthopoly_vect) == 0).any():
                warn(
                    f'Variable {self.name} has at least one orthogonal polynomial '
                    f'that is zero. The model may not be accurate'
                )

            return

    def convert_to_floating_point(self, array):
        """
        Inputs: array- the array of points to be converted
        
        For functions that result in gamma(1/2) or similar functions, this will 
        convert all of the values into forms that depend on floating point 
        values instead of exact values.
        """
        decimals = 50

        for i in range(len(array)):
            array[i] = N(array[i], decimals)

    def get_resamp_vals(self, samp_size):
        """
        Inputs: samp_size- the number of samples to generate according to the
                distribution
                
        Generates samp_size number of samples according to the pdf of the 
        Variable.
        """
        return self._generate_samples(samp_size)

    def _calc_cdf(self, proc_dict):
        """
        Calculates the cumulative density function of the distribution.
        """
        try:
            proc_dict['cum_dens_func'] = (
                integrate(self.distribution, (self.x, self.interval_low, self.x))
            )

        except RuntimeError:
            pass

    def _invert(self, proc_dict):
        """
        Solves for the inverse function of the cumulative density function.
        """
        y = symbols('y')

        try:
            proc_dict['inverse_func'] = solve(f'{self.cum_dens_func}-y', self.x)

        except (NameError, NotImplementedError, AttributeError, RuntimeError):
            pass

    def check_num_string(self):
        """
        Checks for values in the input file that correspond to pi, -oo, or oo. 
        If these values exist, they are converted into values that Python can 
        use to create resampling points.
        """
        if self.interval_low == '-oo' or self.interval_high == 'oo':
            x = self.x

            mean = integrate(x * self.distribution, (x, self.interval_low,
                                        self.interval_high))
            stdev = math.sqrt(integrate(x ** 2 * self.distribution,
                                        (x, self.interval_low,
                                         self.interval_high))\
                                            -mean ** 2)

        if self.interval_low == 'pi':
            self.interval_low = np.pi
            self.low = np.pi
        elif self.interval_low == '-pi':
            self.interval_low = -np.pi
            self.low = -np.pi
        elif self.interval_low == '-oo':
            self.low = float(mean - 15 * stdev)

        if self.interval_high == 'pi':
            self.interval_high = np.pi
            self.high = np.pi
        elif self.interval_high == '-pi':
            self.interval_high = -np.pi
            self.high = -np.pi
        elif self.interval_high == 'oo':
            self.high = float(mean + 15 * stdev)

    __standardize = standardize
    __check_distribution = check_distribution
    __generate_samples = generate_samples
    __get_norm_sq_val = get_norm_sq_val
    __check_num_string = check_num_string


class UniformVariable(Variable):
    """
    Inputs: number- the number of the variable from the file
            order- the order of the model to calculate the orthogonal 
            polynomials and norm squared values
    
    Represents a uniform variable. The methods in this class correspond to 
    those of a uniform variable.
    """

    def __init__(self, number, order):
        super(UniformVariable, self).__init__(number, order)
        self.order = order

    def initialize(self, attributes):
        """
        Inputs: attributes- all of the input names and values for the 
        attributes of the variable
        
        Sets the attributes of the variable and generates the orthogonal 
        polynomials.
        """
        [setattr(self, attr, attributes[attr]) for attr in list(attributes)]
        self.check_attributes(attributes)
        self.type = UncertaintyType[self.type.upper()]

        self.check_num_string()
        self.generate_orthopoly()

    def check_attributes(self, attributes):
        """
        Overrides the Variable class check_attributes to align with 
        a uniform distribution.
        """
        options = ['distribution', 'interval_low', 'interval_high', 'type',
                   'name']
        for attr in list(attributes):
            if attr not in options:
                warn(f'{self.name} : {attr} is not an attribute of '
                     'UniformVariable.')

    def generate_orthopoly(self):
        """
        Generates the orthogonal polynomials for a uniform variable up to the 
        order of polynomial expansion.
        """
        self.var_orthopoly_vect = zeros(self.order + 1, 1)
        x = self.x
        for n in range(self.order + 1):
            if n == 0:
                self.var_orthopoly_vect[n] = 1
            elif n == 1:
                self.var_orthopoly_vect[n] = x
            else:  # "N" from sympy allows fractions to look like decimals to evaluate gamma(1/3)
                self.var_orthopoly_vect[n] = simplify(((2 * n - 1) * x *
                                              self.var_orthopoly_vect[n - 1]
                                              -(n - 1)
                                              * self.var_orthopoly_vect[n - 2])\
                                                / n)

    def standardize(self, orig, std_vals):
        """
        Overrides the Variable class standardize to align with 
        a uniform distribution.
        """
        original = getattr(self, orig)
        mean = (self.interval_high - self.interval_low) / 2\
                        +self.interval_low
        stdev = (self.interval_high - self.interval_low) / 2
        standard = (original[:] - mean) / stdev
        setattr(self, std_vals, standard)

    def check_distribution(self):
        """
        Overrides the Variable class check_distribution to align with 
        a uniform distribution.
        """
        if (np.max(self.std_vals) > 1.0000001)\
         or (np.min(self.std_vals) < -1.0000001):
            raise ValueError(f'Standardized value for variable {self.name} '
                             'with uniform distribution outside expected '
                             '[-1,1] bounds')
            return -1

    def generate_samples(self, samp_size):
        """ 
        Overrides the Variable class generate_samples to align with 
        an exponential distribution.
        """
        self.vals = (
            uniform_hypercube(self.interval_low, self.interval_high, samp_size)
        )

    def get_norm_sq_val(self, matrix_val):
        """
        Overrides the Variable class get_norm_sq_val to align with 
        an exponential distribution.
        """
        return(1.0 / (2.0 * matrix_val + 1.0))

    def get_resamp_vals(self, samp_num):
        """
        Overrides the Variable class get_resamp_vals to align with 
        an exponential distribution.
        """
        samples = np.random.uniform(-1, 1, samp_num)
        samples[0] = -1  # bound
        samples[1] = 1  # bound
        return(samples)

    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the 
        values that might contain it.
        """
        if self.interval_low == 'pi':
            self.interval_low = np.pi
        elif self.interval_low == '-pi':
            self.interval_low = -np.pi

        if self.interval_high == 'pi':
            self.interval_high = np.pi
        elif self.interval_high == '-pi':
            self.interval_high = -np.pi


class NormalVariable(Variable):
    """
    Inputs: number- the number of the variable from the file
            order- the order of the model to calculate the orthogonal 
            polynomials and norm squared values
    
    Represents a normal variable. The methods in this class correspond to 
    those of a normal variable.
    """
    __slots__ = ('mean', 'stdev')

    def __init__(self, number, order):
        super(NormalVariable, self).__init__(number, order)
        self.order = order

    def initialize(self, attributes):
        """
        Inputs: attributes- all of the input names and values for the 
        attributes of the variable
        
        Sets the attributes of the variable and generates the orthogonal 
        polynomials.
        """
        [setattr(self, attr, attributes[attr]) for attr in list(attributes)]
        self.check_attributes(attributes)

        self.type = UncertaintyType[self.type.upper()]

        if self.type == UncertaintyType.EPISTEMIC:
            warn('The NormalVariable cannot be epistemic. For an epistemic'
                  ' variable, use the uniform distribution with type epistemic.')
            exit()

        self.check_num_string()
        self.generate_orthopoly()

    def check_attributes(self, attributes):
        """
        Overrides the Variable class check_attributes to align with 
        a normal distribution.
        """
        options = ['distribution', 'mean', 'stdev', 'type', 'name']
        for attr in list(attributes):
            if attr not in options:
                warn(f'{self.name} : {attr} is not an attribute of '
                     'NormalVariable.')

    def generate_orthopoly(self):
        """
        Generates the orthogonal polynomials for a normal variable up to the 
        order of polynomial expansion.
        """
        self.var_orthopoly_vect = zeros(self.order + 1, 1)
        x = self.x
        for n in range(self.order + 1):
            if n == 0:
                self.var_orthopoly_vect[n] = 1
            elif n == 1:
                self.var_orthopoly_vect[n] = 2 * x
            else:
                self.var_orthopoly_vect[n] = 2 * x * self.var_orthopoly_vect[n - 1]\
                                             -2 * (n - 1)\
                                             * self.var_orthopoly_vect[n - 2]

        for n in range(self.order + 1):  # transform into probabalists Hermite poly
            self.var_orthopoly_vect[n] = simplify(2 ** (-n / 2)\
                                    * self.var_orthopoly_vect[n].subs(
                                                        {x:x / math.sqrt(2)}))

    def standardize(self, orig, std_vals):
        """
        Overrides the Variable class standardize to align with 
        a normal distribution.
        """
        original = getattr(self, orig)
        standard = (original[:] - self.mean) / (self.stdev)
        setattr(self, std_vals, standard)

    def check_distribution(self):
        """
        Overrides the Variable class check_distribution to align with 
        an exponential distribution.
        """
        if (np.max(self.std_vals) > 4.5) or (np.min(self.std_vals) < -4.5):
            warn(f'Large standardized value for variable {self.name} '
                  'with normal distribution found. Check input and run matrix\n')
            return -1

    def generate_samples(self, samp_size):
        """ 
        Overrides the Variable class generate_samples to align with 
        an exponential distribution.
        """
        low_percent = 1e-10
        high_percent = 1 - low_percent

        dist = norm(loc=self.mean, scale=self.stdev)

        rnd_hypercube = uniform_hypercube(low_percent, high_percent, samp_size)
        self.vals = dist.ppf(rnd_hypercube)

    def get_norm_sq_val(self, matrix_value):
        """
        Overrides the Variable class get_norm_sq_val to align with 
        an exponential distribution.
        """
        return(math.factorial(matrix_value))

    def get_resamp_vals(self, samp_num):
        """
        Overrides the Variable class get_resamp_vals to align with 
        an exponential distribution.
        """
        return(np.random.randn(samp_num))

    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the 
        values that might contain it.
        """
        if self.mean == 'pi':
            self.mean = np.pi
        elif self.mean == '-pi':
            self.mean = -np.pi

        if self.stdev == 'pi':
            self.stdev = np.pi


class BetaVariable(Variable):
    """
    Inputs: number- the number of the variable from the file
            order- the order of the model to calculate the orthogonal 
            polynomials and norm squared values
    
    Represents a beta variable. The methods in this class correspond to 
    those of a beta variable.
    """
    __slots__ = ('alpha', 'beta')
    equation = '((A+B-1)! * (x)**(A-1) * (1-x)**(B-1)) / ((A-1)! * (B-1)!)'
    low = '0'
    high = '1'

    def __init__(self, number, order):
        self.interval_low = 0
        self.interval_high = 1
        super(BetaVariable, self).__init__(number, order)

    def initialize(self, attributes):
        """
        Inputs: attributes- all of the input names and values for the 
        attributes of the variable
        
        Sets the attributes of the variable, and creates the orthogonal 
        polynomials and norm squared values.
        """
        [setattr(self, attr, attributes[attr]) for attr in list(attributes)]
        self.check_attributes(attributes)
        self.check_num_string()

        self.type = UncertaintyType[self.type.upper()]

        if self.type == UncertaintyType.EPISTEMIC:
            warn('The BetaVariable cannot be epistemic. For an epistemic'
                  ' variable, use the uniform distribution with type epistemic.')
            exit()

        dist = parse_expr(
            self.equation,
            local_dict={'A':parse_expr(str(Fraction(self.alpha))),
                        'B':parse_expr(str(Fraction(self.beta))),
                        'x':self.x}
        )

        self.generate_orthopoly()
        # self.recursive_var_basis(dist, self.low, self.high, self.order)
        self.create_norm_sq(int(self.low), int(self.high), dist)

    def generate_orthopoly(self):
        """
        Generates the orthogonal polynomials for a beta variable up to the 
        self.self.order of polynomial expansion.
        """
        self.var_orthopoly_vect = zeros(self.order + 1, 1)
        x = self.x
        a = parse_expr(str(Fraction(self.alpha)))
        b = parse_expr(str(Fraction(self.beta)))

        for n in range(self.order + 1):

            if n == 0:
                self.var_orthopoly_vect[n] = 1

            elif n == 1:
                self.var_orthopoly_vect[n] = x - (a / (a + b))

            else:
                self.var_orthopoly_vect[n] = x ** n
                pasc = pascal(self.order + 1, kind='lower')

                for m in range(n):
                    self.var_orthopoly_vect[n] -= N(
                        parse_expr(
                            f'{pasc[n, m]} * ((a+n-1)!*(a+b+2*m-1)!)/((a+m-1)!*(a+b+n+m-1)!)*({self.var_orthopoly_vect[m]})',
                            local_dict={'a':a, 'b':b, 'n':n, 'm':m, 'x':x}
                        ),
                        50
                    )

        return self.var_orthopoly_vect

    def check_attributes(self, attributes):
        """
        Overrides the Variable class check_attributes to align with 
        a beta distribution.
        """
        options = ['distribution', 'alpha', 'beta', 'type', 'interval_low',
                   'interval_high', 'name']
        for attr in list(attributes):
            if attr not in options:
                warn(f'{self.name} : {attr} is not an attribute of '
                     'BetaVariable.')

    def standardize(self, orig, std_vals):
        """
        Overrides the Variable class standardize to align with 
        a beta distribution.
        """
        original = getattr(self, orig)
        standard = (original[:] - self.interval_low) \
                    / (self.interval_high - self.interval_low)
        setattr(self, std_vals, standard)

    def check_distribution(self):
        """
        Overrides the Variable class check_distribution to align with 
        an beta distribution.
        """
        if (np.max(self.std_vals) > 8) or (np.min(self.std_vals) < -8):
            print(f'Large standardized value for variable {self.name} '
                  'with B distribution found. Check input and run matrix\n')
            return -1

    def generate_samples(self, samp_size):
        """
        Overrides the Variable class generate_samples to align with
        an beta distribution.
        """
        low_percent = 0
        high_percent = 1

        dist = beta(a=self.alpha, b=self.beta)

        rnd_hypercube = uniform_hypercube(low_percent, high_percent, samp_size)

        self.vals = (
            (dist.ppf(rnd_hypercube) * (self.interval_high - self.interval_low))
            +self.interval_low
        )

    def get_resamp_vals(self, samp_num):
        """
        Overrides the Variable class get_resamp_vals to align with 
        an beta distribution.
        """
        samples = np.random.beta(a=self.alpha, b=self.beta, size=samp_num)
        samples[0] = 0  # bounds
        samples[1] = 1  # bounds
        return(samples)

    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the 
        values that might contain it.
        """
        if self.alpha == 'pi':
            self.alpha = np.pi
        elif self.alpha == '-pi':
            self.alpha = -np.pi

        if self.beta == 'pi':
            self.beta = np.pi
        elif self.beta == '-pi':
            self.beta = -np.pi

        if self.interval_low == 'pi':
            self.interval_low = np.pi
        elif self.interval_low == '-pi':
            self.interval_low = -np.pi

        if self.interval_high == 'pi':
            self.interval_high = np.pi
        elif self.interval_high == '-pi':
            self.interval_high = -np.pi


class ExponentialVariable(Variable):
    """
    Inputs: number- the number of the variable from the file
            order- the order of the model to calculate the orthogonal 
            polynomials and norm squared values
    
    Represents an exponential variable. The methods in this class correspond to 
    those of an exponential variable.
    """

    __slots__ = ('lambda', 'interval_low')

    equation = 'lambd * exp(-lambd * x)'
    low = '0'
    high = 'oo'

    def __init__(self, number, order):
        super(ExponentialVariable, self).__init__(number, order)

    def initialize(self, attributes):
        """
        Inputs: attributes- all of the input names and values for the 
        attributes of the variable
        
        Sets the attributes of the variable, and creates the orthogonal 
        polynomials and norm squared values.
        """
        self.interval_low = 0
        [setattr(self, attr, attributes[attr]) for attr in list(attributes)]
        self.check_attributes(attributes)

        self.type = UncertaintyType[self.type.upper()]

        if self.type == UncertaintyType.EPISTEMIC:
            warn('The ExponentialVariable cannot be epistemic. For an epistemic'
                  ' variable, use the uniform distribution with type epistemic.')
            exit()

        self.check_num_string()
        dist = parse_expr(
            self.equation,
            local_dict={'lambd':parse_expr(str(Fraction(getattr(self, 'lambda')))),
            'x':self.x}
        )
        self.recursive_var_basis(dist, int(self.low), self.high, self.order)
        self.create_norm_sq(int(self.low), self.high, dist)

    def check_attributes(self, attributes):
        """
        Overrides the Variable class check_attributes to align with 
        an exponential distribution.
        """
        options = ['distribution', 'lambda', 'interval_low', 'type', 'name']
        for attr in list(attributes):
            if attr not in options:
                warn(f'{self.name} : {attr} is not an attribute of '
                     'ExponentialVariable.')

    def standardize(self, orig, std_vals):
        """
        Overrides the Variable class standardize to align with 
        an exponential distribution.
        """
        original = getattr(self, orig)
        standard = (original[:] - self.interval_low)
        setattr(self, std_vals, standard)

    def check_distribution(self):
        """
        Overrides the Variable class check_distribution to align with 
        an exponential distribution.
        """
        if (np.min(self.std_vals) < 0) or (np.max(self.std_vals) > 15):
            warn(f'Large standardized value for variable {self.name} '
                  'with exponential distribution found. Check input and run '
                  'matrix\n')
            return -1

    def generate_samples(self, samp_size):
        """ 
        Overrides the Variable class generate_samples to align with 
        an exponential distribution.
        """
        percent_shift = 1e-10

        low_percent = 0
        high_percent = 1 - percent_shift

        dist = expon(scale=1 / getattr(self, 'lambda'))

        rnd_hypercube = uniform_hypercube(low_percent, high_percent, samp_size)
        self.vals = dist.ppf(rnd_hypercube) + self.interval_low

    def get_resamp_vals(self, samp_num):
        """
        Overrides the Variable class get_resamp_vals to align with 
        an exponential distribution.
        """
        samples = (
            np.random.exponential(
                scale=(1 / getattr(self, 'lambda')), size=samp_num
            )
        )
        samples[0] = 0  # low bound
        return(samples)

    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the 
        values that might contain it.
        """
        if getattr(self, 'lambda') == 'pi':
            setattr(self, 'lambda', np.pi)
        elif getattr(self, 'lambda') == '-pi':
            setattr(self, 'lambda', -np.pi)


class GammaVariable(Variable):
    """
    Inputs: number- the number of the variable from the file
            order- the order of the model to calculate the orthogonal 
            polynomials and norm squared values
    
    Represents a beta variable. The methods in this class correspond to 
    those of a gamma variable.
    """
    __slots__ = ('alpha', 'theta')
    equation = '(x**(A-1) * exp(-x)) / (A-1)!'
    low = '0'
    high = 'oo'  # SymPy infinity

    def __init__(self, number, order):
        super(GammaVariable, self).__init__(number, order)

    def initialize(self, attributes):
        """
        Inputs: attributes- all of the input names and values for the 
        attributes of the variable
        
        Sets the attributes of the variable, and creates the orthogonal 
        polynomials and norm squared values.
        """
        self.interval_low = 0
        [setattr(self, attr, attributes[attr]) for attr in list(attributes)]
        self.check_attributes(attributes)

        self.type = UncertaintyType[self.type.upper()]

        if self.type == UncertaintyType.EPISTEMIC:
            warn('The GammaVariable cannot be epistemic. For an epistemic'
                  ' variable, use the uniform distribution with type epistemic.')
            exit()

        self.check_num_string()
        x = symbols(self.var_str)
        dist = parse_expr(
            self.equation,
            local_dict={'A':parse_expr(str(Fraction(self.alpha))), 'x':x}
        )
        self.recursive_var_basis(dist, int(self.low), self.high, self.order)
        self.create_norm_sq(int(self.low), self.high, dist)

    def check_attributes(self, attributes):
        """
        Overrides the Variable class check_attributes to align with 
        a gamma distribution.
        """
        options = ['distribution', 'alpha', 'theta', 'type', 'interval_low',
                   'name']
        for attr in list(attributes):
            if attr not in options:
                warn(f'{self.name} : {attr} is not an attribute of '
                     'GammaVariable.')

    def standardize(self, orig, std_vals):
        """
        Overrides the Variable class standardize to align with 
        a gamma distribution.
        """
        standard = (getattr(self, orig) - self.interval_low) / self.theta
        setattr(self, std_vals, standard)

    def check_distribution(self):
        """
        Overrides the Variable class check_distribution to align with 
        a gamma distribution.
        """
        shift = 15

        if (np.max(self.std_vals) > shift) or (np.min(self.std_vals) < 0):
            warn(f'Large standardized value for variable {self.name} '
                  'with gamma distribution found. Check input and run matrix\n')
            return -1

    def generate_samples(self, samp_size):
        """ 
        Overrides the Variable class generate_samples to align with 
        a gamma distribution.
        """
        percent_shift = 1e-10
        low_percent = 0
        high_percent = 1 - percent_shift

        dist = gamma(self.alpha, scale=self.theta)

        rnd_hypercube = uniform_hypercube(low_percent, high_percent, samp_size)
        self.vals = dist.ppf(rnd_hypercube) + self.interval_low

    def get_resamp_vals(self, samp_num):
        """
        Overrides the Variable class get_resamp_vals to align with 
        a gamma distribution.
        """
        samples = np.random.gamma(shape=self.alpha, scale=1, size=samp_num)
        samples[0] = 0  # low bound
        return(samples)

    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the 
        values that might contain it.
        """
        if self.alpha == 'pi':
            self.alpha = np.pi
        elif self.alpha == '-pi':
            self.alpha = -np.pi

        if self.theta == 'pi':
            self.theta = np.pi
        elif self.theta == '-pi':
            self.theta = -np.pi


class Distribution(Enum):
    NORMAL = auto()
    UNIFORM = auto()
    BETA = auto()
    EXPONENTIAL = auto()
    GAMMA = auto()


class UncertaintyType(Enum):
    ALEATORY = auto()
    EPISTEMIC = auto()

