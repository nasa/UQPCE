from warnings import warn

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from sympy import symbols, Matrix, factorial
    from sympy.utilities.lambdify import lambdify

    from mpi4py.MPI import (
        DOUBLE as MPI_DOUBLE, COMM_WORLD as MPI_COMM_WORLD, SUM as MPI_SUM
    )
except:
    warn('Ensure that all required packages are installed.')
    exit()

from PCE_Codes._helpers import (
    _warn, evaluate_points, solve_coeffs, evaluate_points_verbose,
    calc_difference, calc_mean_err
)
from PCE_Codes.stats.statistics import (
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
        'var_basis_sys_eval', '_norm_sq', '_var_basis_vect_symb'
    )

    def __init__(self, responses, var_list):
        self.verbose = False
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

        showwarning = _warn

    def create_model_matrix(self, order):
        """
        Inputs: order- the order of the polynomial expansion
        
        Creates the model matrix to support an Nth order model.
        """
        self.min_model_size = calc_term_count(order, self.var_count)

        ident_matrix = np.identity(self.var_count)
        prev_matrix = np.zeros([self.var_count, self.var_count])
        final = np.zeros([1, self.var_count])

        y_axis = 0

        for i in range(order):  # interraction matrix formation
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

        comm = MPI_COMM_WORLD
        size = comm.size
        rank = comm.rank

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

        comm.Allgatherv(
            [norm_sq, count, MPI_DOUBLE],
            [self.norm_sq, seq_count, seq_disp, MPI_DOUBLE]
        )

        self._norm_sq = np.copy(self.norm_sq)

        return self.norm_sq

    def build(self):
        """
        Builds the symbolic 'psi' matrix that represents the interactions 
        of the variables.
        """
        comm = MPI_COMM_WORLD
        size = comm.size
        rank = comm.rank

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

        var_basis_mat_symb = comm.allgather(var_basis_mat_symb_temp)
        var_basis_mat_symb = np.concatenate(var_basis_mat_symb, axis=1)

        for i in range(beg, end):
            curr_vect = 1
            curr_var_basis_row = var_basis_mat_symb[i, :]

            for k in curr_var_basis_row:
                curr_vect *= k

            var_basis_vect_symb_temp[i - beg] = curr_vect

            if rank == 0 and self.verbose and (i + 1) in inter_vals:
                print(f'{(i+1)/count:.0%} Complete\n')

        var_basis_vect_symb_tot = comm.allgather(var_basis_vect_symb_temp)
        self.var_basis_vect_symb = Matrix(np.concatenate(var_basis_vect_symb_tot).reshape(1, self.min_model_size))
        self._var_basis_vect_symb = np.copy(self.var_basis_vect_symb)

        return self.var_basis_vect_symb

    def evaluate(self, attribute='std_vals'):
        """
        Inputs: attribute- the attribute of variables used to calculate the 
                responses; this is almost always std_vals
        
        Fills the symbolic 'psi' variable basis system with the numbers that 
        correspond to the variables in the matrix.
        """
        comm = MPI_COMM_WORLD
        size = comm.size
        rank = comm.rank

        base = self.act_model_size // size
        rem = self.act_model_size % size
        beg = base * rank + (rank >= rem) * rem + (rank < rem) * rank
        count = base + (rank < rem)

        ranks = np.arange(0, size, dtype=int)
        seq_count = ((ranks < rem) + base) * self.min_model_size

        seq_disp = (
            base * ranks + (ranks >= rem) * rem + (ranks < rem) * ranks
        ) * self.min_model_size

        self.var_basis_sys_eval = np.zeros([self.act_model_size, self.min_model_size])

        var_basis_vect_func = lambdify(
            (self.var_list_symb,), self.var_basis_vect_symb, modules='numpy'
        )

        if self.verbose:

            var_basis_sys_eval = evaluate_points_verbose(
                var_basis_vect_func, beg, count, self.var_list, attribute
            )

        else:
            var_basis_sys_eval = evaluate_points(
                var_basis_vect_func, beg, count, self.var_list, attribute
            )

        comm.Allgatherv(
            [var_basis_sys_eval, count * self.min_model_size, MPI_DOUBLE],
            [self.var_basis_sys_eval, seq_count, seq_disp, MPI_DOUBLE]
        )

        return self.var_basis_sys_eval

    def solve(self):
        """
        Uses the matrix system to solve for the matrix coefficients.
        """
        comm = MPI_COMM_WORLD
        rank = comm.rank

        if rank == 0:
            self.matrix_coeffs = solve_coeffs(self.var_basis_sys_eval, self.responses)
        else:
            self.matrix_coeffs = np.zeros(self.var_basis_sys_eval.shape[1])

        comm.Bcast([self.matrix_coeffs, MPI_DOUBLE], root=0)

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
        eval_count = 1

        # remove the test point to solve for constants and build model
        var_basis = np.append(var_basis[:idx], var_basis[incr_idx:], axis=0)
        responses = np.append(responses[:idx], responses[incr_idx:])

        matrix_coeffs = solve_coeffs(var_basis, responses)

        # create a model for each of the subsystems; check model error
        temp_model = SurrogateModel(responses, matrix_coeffs)
        err, pred = temp_model.calc_error(var_basis)
        mean_err = calc_mean_err(err)
        var, mean = temp_model.calc_var(norm_sq)

        # evaluate with the test point
        resp_ver = (
            temp_model.verify(
                self.var_basis_vect_symb, self.var_list, eval_count,
                self.var_list_symb, 'std_vals', idx
            )
        )[0]

        # diff between actual point and calculated value
        err_ver = np.abs(calc_difference(self.responses[idx], resp_ver)[0])

        return err_ver, mean_err, mean, var

    def get_press_stats(self):
        """
        Calculates the PRESS statistic of the model.
        """
        comm = MPI_COMM_WORLD
        size = comm.size
        rank = comm.rank

        base = self.act_model_size // size
        rem = self.act_model_size % size
        beg = base * rank + (rank >= rem) * rem + (rank < rem) * rank
        count = base + (rank < rem)
        end = beg + count

        ranks = np.arange(0, size, dtype=int)
        seq_count = (ranks < rem) + base
        seq_disp = base * ranks + (ranks >= rem) * rem + (ranks < rem) * ranks

        temp_eval = np.zeros([self.act_model_size, self.min_model_size])
        mean_err = np.zeros(count)
        var = np.zeros(count)
        mean = np.zeros(count)
        ver = np.zeros(count)

        tot_mean_err = np.zeros(self.act_model_size)
        tot_var = np.zeros(self.act_model_size)
        tot_mean = np.zeros(self.act_model_size)
        press = np.zeros(1)

        temp_eval = np.copy(self.var_basis_sys_eval)

        for i in range(beg, end):

            idx = i - beg

            temp = np.delete(temp_eval, i, axis=0)
            responses = np.delete(self.responses, i)
            matrix_coeffs = solve_coeffs(temp, responses)

            # create a model for each of the subsystems; check model error
            temp_model = SurrogateModel(responses, matrix_coeffs)
            err = temp_model.calc_error(temp)[0]
            mean_err[idx] = calc_mean_err(err)
            var[idx], mean[idx] = temp_model.calc_var(self.norm_sq)

            ver[idx] = np.matmul(temp_eval[i, :], matrix_coeffs)

        ver = np.sum((ver - self.responses[beg:end]) ** 2)

        comm.Allreduce(
            [ver, MPI_DOUBLE], [press, MPI_DOUBLE], op=MPI_SUM
        )

        comm.Allgatherv(
            [mean_err, count, MPI_DOUBLE],
            [tot_mean_err, seq_count, seq_disp, MPI_DOUBLE]
        )

        comm.Allgatherv(
            [mean, count, MPI_DOUBLE],
            [tot_mean, seq_count, seq_disp, MPI_DOUBLE]
        )

        comm.Allgatherv(
            [var, count, MPI_DOUBLE],
            [tot_var, seq_count, seq_disp, MPI_DOUBLE]
        )

        mean_err_avg = float(np.mean(tot_mean_err))
        mean_err_var = float(np.var(tot_mean_err))
        mean_avg = float(np.mean(tot_mean))
        mean_var = float(np.var(tot_mean))
        var_avg = float(np.mean(tot_var))
        var_var = float(np.var(tot_var))

        outputs = {
            'PRESS':float(press),
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
        self.model_matrix = self.model_matrix[combo, :]
        self.min_model_size = len(combo)

        return(
            self.var_basis_vect_symb, self.norm_sq, self.model_matrix,
            self.min_model_size
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
        'sigma_sq', 'resp_mean', 'error', 'act_model_size'
    )

    def __init__(self, responses=None, matrix_coeffs=None, verbose=False):
        self.verbose = verbose
        self.responses = responses
        self.matrix_coeffs = matrix_coeffs.reshape(-1,)

        if responses is not None:
            self.act_model_size = len(responses)

        showwarning = _warn

    def get_sobols(self, norm_sq):
        """
        Inputs: norm_sq- the norm squared matrix
                
        Solves for the sobol sensitivities.
        """
        tol = 1e-8

        comm = MPI_COMM_WORLD
        size = comm.size
        rank = comm.rank

        term_decr = len(norm_sq) - 1

        base = term_decr // size
        rem = term_decr % size
        beg = base * rank + (rank >= rem) * rem + (rank < rem) * rank + 1
        count = base + (rank < rem)
        end = beg + count

        ranks = np.arange(0, size, dtype=int)
        seq_count = (ranks < rem) + base
        seq_disp = base * ranks + (ranks >= rem) * rem + (ranks < rem) * ranks

        sobols = np.ones(count)
        self.sobols = np.ones(term_decr)

        for i in range(beg, end):
            sobols[i - beg] = (
                (self.matrix_coeffs[i] ** 2 * norm_sq[i]) / self.sigma_sq
            )

        comm.Allgatherv(
            [sobols, count, MPI_DOUBLE],
            [self.sobols, seq_count, seq_disp, MPI_DOUBLE]
        )

        if np.abs(np.sum(self.sobols) - 1) > tol:
            warn(
                'The Sobols do not sum to 1 within the accepted tolerance of '
                f'{tol}, which suggests that something may be wrong.'
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
                self.matrix_coeffs, (len(self.matrix_coeffs), 1)
            )[1:] ** 2
        )

        norm_mult_coeff = norm_sq[1:] * matrix_coeffs_sq
        if norm_mult_coeff.shape[0] != 1 and norm_mult_coeff.shape[1] != 1:
            warn(
                'Sigma squared does not look correct for calculating the Sobol '
                'indices.'
            )

        self.sigma_sq = np.sum(norm_mult_coeff)

        return self.sigma_sq, self.resp_mean

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

    def check_normality(self, var_basis_sys_eval, sig, graph_dir=None):
        """
        Inputs: var_basis_sys_eval- the variable basis system (not symbolic) 
                matrix
                sig- the level of signif for the Shapiro-Wilks test
                graph_dir- the directory that the graphs are put into
        
        Ensures that the err follows a normal distribution.
        """
        comm = MPI_COMM_WORLD
        size = comm.size
        rank = comm.rank
        is_manager = (rank == 0)

        test_stat, p_val_hypoth = stats.shapiro(self.error)

        if p_val_hypoth < sig:
            shapiro_results = (
                f'Shapiro-Wilks test statistic is {test_stat:.5f}, P-value is '
                f'{p_val_hypoth:.5f}\n\nEvidence exists that errors are not '
                'from a normal distribution\n'
            )

        if p_val_hypoth > sig:
            shapiro_results = (
                f'Shapiro-Wilks test statistic is {test_stat:.5f}, P-value is '
                f'{p_val_hypoth:.5f}\n\nInsufficient evidence to infer errors '
                'are not from a normal distribution\n'
            )

        if is_manager and self.verbose:
            print(shapiro_results)

        hat_matrix = calc_hat_matrix(var_basis_sys_eval)
        mean_sq_error = calc_mean_sq_err(
            self.responses, self.matrix_coeffs, var_basis_sys_eval
        )

        sigma = np.sqrt(mean_sq_error)
        hat_adj = sigma * np.sqrt(1 - np.diagonal(hat_matrix))
        std_err_matrix = self.error / hat_adj

        if graph_dir is not None:
            try:
                size_decr = size - 1
                plot_error_dist = 0
                plot_normal_prob = np.min([1, size_decr])

                if rank == plot_error_dist:
                    plt.hist(std_err_matrix)
                    image_path = f'{graph_dir}/error_distribution'
                    plt.title('Error Distribution')
                    plt.savefig(image_path, dpi=600, bbox_inches='tight')
                    plt.clf()

                if rank == plot_normal_prob:
                    stats.probplot(self.error, plot=plt)
                    image_path = f'{graph_dir}/normal_prob'
                    plt.savefig(image_path, dpi=600, bbox_inches='tight')
                    plt.clf()

            except ValueError:
                warn(
                    'The histogram of the errors was not successfully created. '
                    'Errors were zero.'
                )

        return mean_sq_error, hat_matrix, shapiro_results

    def verify(
            self, var_basis_vect_symb, var_list, verify_size, var_list_symb,
            attr='std_verify_vals', beg_idx=0
        ):
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
        comm = MPI_COMM_WORLD
        size = comm.size
        rank = comm.rank

        base = verify_size // size
        rem = verify_size % size
        beg = base * rank + (rank >= rem) * rem + (rank < rem) * rank
        count = base + (rank < rem)

        ranks = np.arange(0, size, dtype=int)
        seq_count = (ranks < rem) + base
        seq_disp = base * ranks + (ranks >= rem) * rem + (ranks < rem) * ranks

        min_model_size = len(self.matrix_coeffs)

        var_basis_sys_eval_verify = np.zeros([verify_size, min_model_size])
        verify_pred = np.zeros(verify_size)

        var_basis_vect_func = lambdify(
            (var_list_symb,), var_basis_vect_symb, modules='numpy'
        )

        var_basis_sys_eval_verify_temp = evaluate_points(
            var_basis_vect_func, beg_idx + beg, count, var_list, attr
        )

        verify_pred_temp = np.matmul(
            var_basis_sys_eval_verify_temp, self.matrix_coeffs
        )

        comm.Allgatherv(
            [
                var_basis_sys_eval_verify_temp,
                count * min_model_size, MPI_DOUBLE
            ],
            [
                var_basis_sys_eval_verify, seq_count * min_model_size,
                seq_disp * min_model_size, MPI_DOUBLE
            ]
        )

        comm.Allgatherv(
            [verify_pred_temp, count, MPI_DOUBLE],
            [verify_pred, seq_count , seq_disp, MPI_DOUBLE]
        )

        return verify_pred, var_basis_sys_eval_verify
