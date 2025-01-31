import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sympy import symbols, Matrix, factorial
from sympy.utilities.lambdify import lambdify

try:
    from mpi4py.MPI import (
        DOUBLE as MPI_DOUBLE, COMM_WORLD as MPI_COMM_WORLD, SUM as MPI_SUM
    )

    comm = MPI_COMM_WORLD
    rank = comm.rank
    size = comm.size
    is_manager = (rank == 0)
except:
    comm = None
    rank = 0
    size = 1
    is_manager = True



from uqpce.pce._helpers import (
    evaluate_points, solve_coeffs, evaluate_points_verbose,
    calc_difference, calc_mean_err, warn
)
from uqpce.pce.stats.statistics import (
    calc_term_count, calc_mean_sq_err, calc_hat_matrix
)


class MatrixSystem:
    """
    Inputs: responses- the array of responses from the results file 
                       (or from the user_function)
            var_list- the lsit of variables
    
    The matrix system built from the responses and input values. The 
    MatrixSystem is built and solved to acquire the matrix coefficients in 
    the systme of equations.
    """

    __slots__ = (
        'verbose', 'responses', 'act_model_size', 'min_model_size', 'var_list',
        'var_count', 'inter_vals', 'var_list_symb', 'model_matrix', 'norm_sq',
        'var_basis_vect_symb', 'matrix_coeffs', 'var_basis_sys_eval_verify',
        'var_basis_sys_eval', '_norm_sq', '_var_basis_vect_symb', '_model_matrix',
        '_var_basis_sys_eval'
    )

    def __init__(self, responses, var_list, verbose=False):
        self.verbose = verbose
        self.responses = responses

        try:
            self.act_model_size = len(responses)
        except TypeError:
            self.act_model_size = 0

        self.var_list = var_list
        self.var_count = len(var_list)

        self.inter_vals = (
            np.linspace(0, 1, 11) * self.act_model_size
        ).astype(int)

        self.var_list_symb = np.array(
            [symbols(f'x{j}') for j in range(self.var_count)]
        )

    def create_model_matrix(self):
        """
        Creates the model matrix to support an Nth order model. Supports 
        creating a model matrix with variables of varying orders.
        """
        orders = np.array([var.order for var in self.var_list])
        max_order = np.max(orders)

        ident_matrix = np.identity(self.var_count)
        prev_matrix = np.zeros([self.var_count, self.var_count])
        final = np.zeros([1, self.var_count])

        y_axis = 0

        for i in range(max_order):  # interraction matrix formation
            prev_cols = prev_matrix.shape[0]
            curr_size = prev_cols * self.var_count + 1
            model_matrix = np.zeros([curr_size, self.var_count])
            idx = 0

            for row in range(prev_cols):

                for j in range(self.var_count):
                    model_matrix[idx] = ident_matrix[j, :] + prev_matrix[row, :]
                    idx += 1

            sorted_indices = np.unique(model_matrix, axis=y_axis, return_index=True)[1]
            prev_matrix = model_matrix[np.sort(sorted_indices)]
            final = np.append(final, prev_matrix, axis=y_axis)

        indices = np.unique(final, axis=y_axis, return_index=True)[1]
        self.model_matrix = final[np.sort(indices)]

        # Below, terms that contain an order higher than their individual order
        # are removed; this is only relevant when any of the variables have
        # different orders.
        for col in range(self.var_count):
            # Remove instances of orders higher than the respective variable
            # order
            disc_idx = self.model_matrix[:, col] > orders[col]
            keep_idx = (disc_idx == False)
            self.model_matrix = self.model_matrix[keep_idx, :]

        keep_rows = [0]  # Force the intercept to be kept

        for row in range(1, self.model_matrix.shape[0]):
            # Remove instances of total interaction order higher than any of
            # the variables in that interaction
            rem_idx = (orders[self.model_matrix[row, :] != 0] < np.sum(self.model_matrix[row, :])).all()

            if not rem_idx:
                keep_rows.append(row)

        self.model_matrix = self.model_matrix[keep_rows, :]

        # The min_model_size is set using the length due to the options to
        # have different orders for different variables.
        self.min_model_size = self.model_matrix.shape[0]
        self._model_matrix = np.copy(self.model_matrix)

        return self.min_model_size, self.model_matrix

    def form_norm_sq(self, order):
        """
        Inputs: order- the order of polynomial expansion 
        
        Creates the model matrix and the corresponding norm squared matrix.
        """
        if (
            not hasattr(self, 'model_matrix')
            or not hasattr(self, 'min_model_size')
        ):
            self.create_model_matrix(order)

        y_size = 1

        base = self.min_model_size // size
        rem = self.min_model_size % size
        beg = base * rank + (rank >= rem) * rem + (rank < rem) * rank
        count = base + (rank < rem)
        end = beg + count

        ranks = np.arange(0, size, dtype=int)
        seq_count = (ranks < rem) + base
        seq_disp = base * ranks + (ranks >= rem) * rem + (ranks < rem) * ranks

        norm_sq = np.zeros([count, y_size])
        self.norm_sq = np.zeros([self.min_model_size, y_size])

        for i in range(beg, end):
            val = 1

            for j in range(self.var_count):
                val *= self.var_list[j].get_norm_sq_val(
                    int(self.model_matrix[i, j])
                )

            norm_sq[i - beg] = val

        if comm:
            comm.Allgatherv(
                [norm_sq, count, MPI_DOUBLE],
                [self.norm_sq, seq_count, seq_disp, MPI_DOUBLE]
            )
        else:
            self.norm_sq = norm_sq

        self._norm_sq = copy.copy(self.norm_sq)

        return self.norm_sq

    def build(self):
        """
        Builds the symbolic 'psi' matrix that represents the interactions 
        of the variables.
        """

        base = self.min_model_size // size
        rem = self.min_model_size % size
        beg = base * rank + (rank >= rem) * rem + (rank < rem) * rank
        count = base + (rank < rem)
        end = beg + count

        var_base = self.var_count // size
        var_rem = self.var_count % size
        var_beg = var_base * rank + (rank >= var_rem) * var_rem + (rank < var_rem) * rank
        var_count = var_base + (rank < var_rem)
        var_end = var_beg + var_count

        var_basis_vect_symb_temp = np.zeros(count, dtype=object)
        var_basis_mat_symb_temp = np.zeros([self.min_model_size, var_count], dtype=object)

        inter_vals = (np.arange(0.1, 1.1, 0.1) * count).astype(int)

        for i in range(var_beg, var_end):  # self.var_count
            var_basis_mat_symb_temp[:, i - var_beg] = self.var_list[i].get_var_basis(
                self.min_model_size, self.model_matrix, i)

        if comm:
            var_basis_mat_symb = comm.allgather(var_basis_mat_symb_temp)
            var_basis_mat_symb = np.concatenate(var_basis_mat_symb, axis=1)
        else:
            var_basis_mat_symb =var_basis_mat_symb_temp        

        for i in range(beg, end):
            curr_vect = 1
            curr_var_basis_row = var_basis_mat_symb[i, :]

            for k in curr_var_basis_row:
                curr_vect *= k

            var_basis_vect_symb_temp[i - beg] = curr_vect

            if rank == 0 and self.verbose and (i + 1) in inter_vals:
                print(f'{(i+1)/count:.0%} Complete\n')

        if comm:
            var_basis_vect_symb_tot = comm.allgather(var_basis_vect_symb_temp)
            self.var_basis_vect_symb = Matrix(np.concatenate(var_basis_vect_symb_tot).reshape(1, self.min_model_size))
        else:
            self.var_basis_vect_symb = Matrix(var_basis_vect_symb_temp)

        self._var_basis_vect_symb = copy.copy(self.var_basis_vect_symb)

        return self.var_basis_vect_symb

    def evaluate(self, X):
        """
        Inputs: attribute- the attribute of variables used to calculate the 
                responses; this is almost always std_vals
        
        Fills the symbolic 'psi' variable basis system with the numbers that 
        correspond to the variables in the matrix.
        """

        var_basis_vect_func = lambdify(
            (self.var_list_symb,), self.var_basis_vect_symb, modules='numpy'
        )

        self.var_basis_sys_eval = evaluate_points(
            var_basis_vect_func, X
        )
        
        self._var_basis_sys_eval = np.copy(self.var_basis_sys_eval)

        return self.var_basis_sys_eval

    def solve(self):
        """
        Uses the matrix system to solve for the matrix coefficients.
        """
        self.matrix_coeffs = solve_coeffs(self.var_basis_sys_eval, self.responses)

        if len(self.matrix_coeffs.shape) != 2:
            self.matrix_coeffs = self.matrix_coeffs.reshape(-1, 1)

        return self.matrix_coeffs

    def _build_alt_model(self, responses, var_basis, norm_sq, idx):
        """
        Inputs: responses- the array of responses
                var_basis- the evaluated variable basis
                norm_sq- the norm squared
                idx- the index of the point that will be omitted
        
        Creates a model for the input combination; the mean, variance, and 
        errors are calculated and returned.
        """
        incr_idx = idx + 1

        # remove the test point to solve for constants and build model
        var_basis_test = np.copy(var_basis[idx])
        var_basis = np.append(var_basis[:idx], var_basis[incr_idx:], axis=0)
        responses = np.append(responses[:idx], responses[incr_idx:])

        matrix_coeffs = solve_coeffs(var_basis, responses)

        # create a model for each of the subsystems; check model error
        temp_model = SurrogateModel(responses, matrix_coeffs)
        err, pred = temp_model.calc_error(var_basis)
        mean_err = calc_mean_err(err)
        var = temp_model.calc_var(norm_sq)

        # evaluate with the test point
        resp_ver = np.matmul(var_basis_test, matrix_coeffs)

        # diff between actual point and calculated value
        err_ver = np.abs(calc_difference(self.responses[idx], resp_ver))
        mean = matrix_coeffs[0]

        return err_ver, mean_err, mean, var

    def get_press_stats(self):
        """
        Calculates the PRESS statistic of the model.
        """

        mean_err = np.zeros(self.responses.shape)
        var = np.zeros(self.responses.shape)
        mean = np.zeros(self.responses.shape)
        ver = np.zeros(self.responses.shape)

        for idx in range(self.act_model_size):

            temp = np.delete(self.var_basis_sys_eval, idx, axis=0)
            responses = np.delete(self.responses, idx)
            matrix_coeffs = solve_coeffs(temp, responses)

            # create a model for each of the subsystems; check model error
            temp_model = SurrogateModel(responses, matrix_coeffs)
            err = temp_model.calc_error(temp)[0]
            mean_err[idx,:] = calc_mean_err(err)
            var[idx,:] = temp_model.calc_var(self.norm_sq)
            mean[idx,:] = matrix_coeffs[0]

            ver[idx,:] = np.matmul(self.var_basis_sys_eval[idx, :], matrix_coeffs)

        press = np.atleast_2d(np.sum((ver - self.responses) ** 2, axis=0))

        mean_err_avg = np.atleast_2d(np.mean(mean_err, axis=0))
        mean_err_var = np.atleast_2d(np.var(mean_err, axis=0))
        mean_avg = np.atleast_2d(np.mean(mean, axis=0))
        mean_var = np.atleast_2d(np.var(mean, axis=0))
        var_avg = np.atleast_2d(np.mean(var, axis=0))
        var_var = np.atleast_2d(np.var(var, axis=0))

        outputs = {
            'PRESS':press,
            'mean_of_model_mean_err':mean_err_avg,
            'variance_of_model_mean_err':mean_err_var,
            'mean_of_model_mean':mean_avg,
            'variance_of_model_mean':mean_var,
            'mean_of_model_variance':var_avg,
            'variance_of_model_variance':var_var
        }

        return outputs

    def update(self, combo):
        """
        Inputs: combo- the combination used to update the attributes
        
        Updates the MatrixSystem attributes to reflect only the model terms 
        that correcpond to `combo`.
        """
        combo = list(combo)
        self.var_basis_vect_symb = self._var_basis_vect_symb[:, combo]
        self.norm_sq = self._norm_sq[combo, :]
        self.model_matrix = self._model_matrix[combo, :]
        self.min_model_size = len(combo)
        self.var_basis_sys_eval = self._var_basis_sys_eval[:, combo]

        return(
            self.var_basis_vect_symb, self.norm_sq, self.model_matrix,
            self.min_model_size, self.var_basis_sys_eval
        )


class SurrogateModel:
    """
    Inputs: responses- the array of responses from the results file 
                       (or from the user_function)
            matrix_coeffs- the matrix coefficients solved for by MatrixSystem
            verbose- if statements should be printed by methods
    
    Gets the sobol indices of the varibles. Performs several calculations 
    and checks on the model to check if it is a good representation.
    """

    __slots__ = (
        'verbose', 'responses', 'matrix_coeffs', 'act_model_size', 'sobols',
        'sigma_sq', 'resp_mean', 'error', 'act_model_size', 'model_cnt'
    )

    def __init__(self, responses=None, matrix_coeffs=None, verbose=False):
        self.verbose = verbose
        if len(responses.shape) == 1:
            responses = responses.reshape(-1, 1)
        self.responses = responses
        self.model_cnt = responses.shape[1]
        self.matrix_coeffs = matrix_coeffs.reshape(-1, self.model_cnt)

        if responses is not None:
            self.act_model_size = len(responses)

    def get_sobols(self, norm_sq):
        """
        Inputs: norm_sq- the norm squared matrix
                
        Solves for the sobol sensitivities.
        """
        tol = 1e-8
        cnt = len(norm_sq) - 1
        self.sobols = np.ones([cnt, self.model_cnt])

        for i in range(1, cnt+1):
            self.sobols[i - 1, :] = (
                (self.matrix_coeffs[i] ** 2 * norm_sq[i]) / self.sigma_sq
            )


        return self.sobols

    def calc_var(self, norm_sq):
        """
        Inputs: norm_sq- the norm squared matrix
        
        Calculates the mean and variance in the responses.
        """

        self.resp_mean = self.matrix_coeffs[0]

        matrix_coeffs_sq = (
            np.reshape(
                self.matrix_coeffs, (len(self.matrix_coeffs), self.model_cnt)
            )[1:] ** 2
        )

        norm_mult_coeff = norm_sq[1:] * matrix_coeffs_sq
        if norm_mult_coeff.shape[0] != 1 and norm_mult_coeff.shape[1] != self.model_cnt:
            warn(
                'Sigma squared does not look correct for calculating the Sobol '
                'indices.'
            )

        self.sigma_sq = np.sum(norm_mult_coeff, axis=0)

        return self.sigma_sq

    def calc_error(self, var_basis):
        """
        Inputs: var_basis- the varibale basis matrix that consists of values 
                (not symbols)
        
        Solves for the calculated responses that the matrix coefficients and 
        variable basis ('alpha' and 'psi') result in as well as the difference 
        between these values and the actual values.
        """
        prod = np.dot(var_basis, self.matrix_coeffs)

        self.error = prod - self.responses

        return self.error, prod

    def check_normality(
            self, var_basis_sys_eval, sig, graph_dir=None, sigfigs=5, plot=False
        ):
        """
        Inputs: var_basis_sys_eval- the variable basis system (not symbolic) 
                matrix
                sig- the level of signif for the Shapiro-Wilks test
                graph_dir- the directory that the graphs are put into
        
        Ensures that the err follows a normal distribution.
        """
        test_stat = np.zeros(self.model_cnt)
        p_val_hypoth = np.zeros(self.model_cnt)
        shapiro_results = np.zeros(self.model_cnt, dtype=object)
        for i in range(self.model_cnt):
            test_stat[i], p_val_hypoth[i] = stats.shapiro(self.error[:,i])

            if p_val_hypoth[i] < sig:
                shapiro_results[i] = (
                    f'Shapiro-Wilks test statistic is {test_stat[i]:.{sigfigs}}, P-value is '
                    f'{p_val_hypoth[i]:.{sigfigs}}\n\nEvidence exists that errors are not '
                    'from a normal distribution\n'
                )

            if p_val_hypoth[i] > sig:
                shapiro_results[i] = (
                    f'Shapiro-Wilks test statistic is {test_stat[i]:.{sigfigs}}, P-value is '
                    f'{p_val_hypoth[i]:.{sigfigs}}\n\nInsufficient evidence to infer errors '
                    'are not from a normal distribution\n'
                )

        hat_matrix = calc_hat_matrix(var_basis_sys_eval)

        mean_sq_error = calc_mean_sq_err(
            self.responses, self.matrix_coeffs, var_basis_sys_eval
        )

        sigma = np.atleast_2d(np.array(np.sqrt(mean_sq_error)))
        hat_adj = np.atleast_2d(np.diag(sigma))*np.atleast_2d(np.sqrt(1 - np.diagonal(hat_matrix))).T
        std_err_matrix = self.error / hat_adj

        if plot and graph_dir is not None:
            for i in range(self.model_cnt):
                try:
                    size_decr = size - 1
                    plot_error_dist = 0
                    plot_normal_prob = np.min([1, size_decr])

                    if rank == plot_error_dist:
                        plt.hist(std_err_matrix[:,i])
                        image_path = f'{graph_dir[i]}/error_distribution'
                        plt.title('Error Distribution')
                        plt.savefig(image_path, dpi=600, bbox_inches='tight')
                        plt.clf()

                    if rank == plot_normal_prob:
                        stats.probplot(self.error[:,i], plot=plt)
                        image_path = f'{graph_dir[i]}/normal_prob'
                        plt.savefig(image_path, dpi=600, bbox_inches='tight')
                        plt.clf()


                except ValueError:
                    warn(
                        'The histogram of the errors was not successfully created. '
                        'Errors were zero.'
                    )

        return mean_sq_error, hat_matrix, shapiro_results

    def predict(self, var_basis_vect_symb, var_list_symb, X):
        """
        Inputs: var_basis_vect_symb- the symbolic variable basis vector (psi)
                var_list- the list of variables
                verify_size- the size of the verification responses
                var_list_symb- the list of the variable string representations
                attr- the Variable attribute to be used for verification
                beg_idx- the index at which to start using points from 
                attribute `attr`
        
        Verifies the surrogate model by outputting the responses from 
        verification input values put into the model.
        """
        var_basis_vect_func = lambdify(
            (var_list_symb,), var_basis_vect_symb, modules='numpy'
        )

        var_basis_sys_eval_pred = evaluate_points(var_basis_vect_func, X)

        pred = np.matmul(
            var_basis_sys_eval_pred, self.matrix_coeffs
        )

        return pred, var_basis_sys_eval_pred