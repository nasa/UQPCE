import math
from warnings import warn

try:
    from mpi4py.MPI import DOUBLE as MPI_DOUBLE, COMM_WORLD as MPI_COMM_WORLD
    import numpy as np
    from numpy.linalg import inv
    from scipy.stats import t as t_stat
except:
    warn('Ensure that all required packages are installed.')
    exit()

from PCE_Codes._helpers import solve_coeffs


def calc_R_sq(var_basis, matrix_coeffs, responses):
    """
    Inputs: var_basis- evaluated variable basis
            matrix_coeffs- matrix coefficients
            responses- vector of responses
            
    Calculates the R squared statistic for the given system.
    
    Design and Analysis of Experiments (8th) by Douglas Montgomery (pg. 464)
    """
    thresh = 1e-8

    # SS_T and SS_E
    tot_sum_sq = calc_total_sum_of_sq(var_basis, responses)

    err_sum_sq = calc_error_sum_of_sq(
        var_basis, matrix_coeffs, responses
    )

    R_sq = 1 - (err_sum_sq / tot_sum_sq)  # equation for R^2

    if R_sq > 1:
        if R_sq > 1 + thresh:
            warn(
                f'R squared value was {R_sq}. Check the variable basis and '
                'responses.'
            )

        R_sq = 1.0

    return R_sq


def calc_R_sq_adj(var_basis, matrix_coeffs, responses):
    """
    Inputs: var_basis- evaluated variable basis
            matrix_coeffs- matrix coefficients
            responses- vector of responses
    
    Calculates the adjusted R squared statistic for the given system.
    
    Design and Analysis of Experiments (8th) by Douglas Montgomery (pg. 464)
    """
    act_model_size, min_model_size = var_basis.shape
    R_sq = calc_R_sq(var_basis, matrix_coeffs, responses)

    ratio = (# (n-1)/(n-p)    p == model size
        (act_model_size - 1)
        / (act_model_size - min_model_size)
    )

    R_sq_adj = 1 - ratio * (1 - R_sq)

    return R_sq_adj


def calc_PRESS_res(var_basis, responses):
    """
    Inputs: var_basis- evaluated variable basis
            responses- the responses
            var_basis_func- the function that calculates the model variable 
            basis from the input point
    
    An efficient way to calculate the PRESS residual for a given model.
    """
    # TODO: cite
    resp_count = len(responses)
    err_ver_sq = np.zeros(resp_count)

    comm = MPI_COMM_WORLD
    size = comm.size
    rank = comm.rank

    base = resp_count // size
    rem = resp_count % size
    beg = base * rank + (rank >= rem) * rem + (rank < rem) * rank
    count = base + (rank < rem)
    end = beg + count

    ranks = np.arange(0, size, dtype=int)
    seq_count = (ranks < rem) + base
    seq_disp = base * ranks + (ranks >= rem) * rem + (ranks < rem) * ranks

    err_ver_sq_temp = np.zeros(count)

    for idx in range(beg, end):

        temp_basis = np.delete(var_basis, idx, axis=0)
        temp_resps = np.delete(responses, idx,)
        matrix_coeffs = solve_coeffs(temp_basis, temp_resps)

        ver_pred = np.matmul(var_basis[idx], matrix_coeffs)
        err_ver_sq_temp[idx - beg] = (responses[idx] - ver_pred) ** 2

    comm.Allgatherv(
        [err_ver_sq_temp, count, MPI_DOUBLE],
        [err_ver_sq, seq_count, seq_disp, MPI_DOUBLE]
    )

    press_res = np.sum(err_ver_sq)

    return press_res


def calc_pred_conf_int(var_basis, matrix_coeffs, responses, signif, var_basis_ver):
    """
    Inputs: var_basis- evaluated variable basis
            matrix_coeffs- the matrix coefficients
            responses- the responses
            signif- significance of the model (i.e. 95% conf int has signif=0.05)
            var_basis_ver- the variable basis whose mean confidence interval 
            will be evaluated against the model variable basis
    
    Calcualtes the prediction confidence interval for a given point.
    
    Design and Analysis of Experiments (8th) by Douglas Montgomery (pg. 469)
    """
    const = 1
    resp_count, term_count = var_basis.shape
    deg_of_free = resp_count - term_count

    err_var = calc_error_variance(var_basis, matrix_coeffs, responses)

    basis_mat = np.matmul(var_basis.T, var_basis)  # X'X
    inverse_mat = inv(basis_mat)  # (X'X)^-1

    # t_alpha/2, n-p
    t_val = t_stat(deg_of_free).ppf(1 - signif / 2)  # high CI bound

    resamp_size = len(var_basis_ver)
    approx_conf = np.zeros(resamp_size)
    conf_uncert = np.zeros(resamp_size)

    for i in range(resamp_size):
        x_i = var_basis_ver[i]
        mult_matrices = np.matmul(np.matmul(x_i.T, inverse_mat), x_i)

        approx_conf[i] = np.matmul(x_i, matrix_coeffs)  # y_hat(x)
        conf_uncert[i] = t_val * np.sqrt(err_var * (const + mult_matrices))

    return approx_conf, conf_uncert


def calc_mean_conf_int(var_basis, matrix_coeffs, responses, signif, var_basis_ver):
    """
    Inputs: var_basis- evaluated variable basis
            matrix_coeffs- the matrix coefficients
            responses- the responses
            signif- significance of the model (i.e. 95% conf int has signif=0.05)
            var_basis_ver- the variable basis whose mean confidence interval 
            will be evaluated against the model variable basis
    
    Calculates the confidence interval on the mean response.
    
    Design and Analysis of Experiments (8th) by Douglas Montgomery (pg. 468)
    """
    var_basis_ver = np.atleast_2d(var_basis_ver)

    resp_count, term_count = var_basis.shape
    deg_of_free = resp_count - term_count

    err_var = calc_error_variance(var_basis, matrix_coeffs, responses)

    basis_mat = np.matmul(var_basis.T, var_basis)  # X'X
    inverse_mat = inv(basis_mat)  # (X'X)^-1

    # t_alpha/2, n-p
    t_val = t_stat(deg_of_free).ppf(1 - signif / 2)  # high CI bound

    resamp_size = len(var_basis_ver)

    comm = MPI_COMM_WORLD
    size = comm.size
    rank = comm.rank

    approx_mean_tot = np.zeros(resamp_size)
    pred_uncert_tot = np.zeros(resamp_size)

    base = resamp_size // size
    rem = resamp_size % size
    rank_l_rem = (rank < rem)
    beg = base * rank + (rank >= rem) * rem + rank_l_rem * rank
    count = base + rank_l_rem
    end = beg + count

    ranks = np.arange(0, size, dtype=int)
    ranks_l_rem = (ranks < rem)
    seq_count = ranks_l_rem + base
    seq_disp = base * ranks + (ranks >= rem) * rem + ranks_l_rem * ranks

    approx_mean = np.zeros(count)
    pred_uncert = np.zeros(count)

    for i in range(beg, end):
        x_i = var_basis_ver[i]
        mult_matrices = np.matmul(np.matmul(x_i.T, inverse_mat), x_i)

        idx = i - beg
        approx_mean[idx] = np.matmul(x_i, matrix_coeffs)  # y_hat(x)
        pred_uncert[idx] = t_val * np.sqrt(err_var * mult_matrices)

    comm.Allgatherv(
        [approx_mean, count, MPI_DOUBLE],
        [approx_mean_tot, seq_count, seq_disp, MPI_DOUBLE]
    )

    comm.Allgatherv(
        [pred_uncert, count, MPI_DOUBLE],
        [pred_uncert_tot, seq_count, seq_disp, MPI_DOUBLE]
    )

    return approx_mean_tot, pred_uncert_tot


def calc_coeff_conf_int(var_basis, matrix_coeffs, responses, signif):
    """
    Inputs: var_basis- evaluated variable basis
            matrix_coeffs- the matrix coefficients
            responses- the responses
            signif- significance of the model (i.e. 95% conf int has signif=0.05)
    
    Design and Analysis of Experiments (8th) by Douglas Montgomery (pg. 467)
    """
    resp_count, term_count = var_basis.shape

    comm = MPI_COMM_WORLD
    size = comm.size
    rank = comm.rank

    base = term_count // size
    rem = term_count % size
    beg = base * rank + (rank >= rem) * rem + (rank < rem) * rank
    count = base + (rank < rem)
    end = beg + count

    ranks = np.arange(0, size, dtype=int)
    seq_count = (ranks < rem) + base
    seq_disp = base * ranks + (ranks >= rem) * rem + (ranks < rem) * ranks

    deg_of_free = resp_count - term_count

    coeff_uncert = np.zeros(count)
    coeff_uncert_tot = np.zeros(term_count)

    err_var = calc_error_variance(var_basis, matrix_coeffs, responses)

    basis_mat = np.matmul(var_basis.T, var_basis)  # X'X
    inverse_mat = inv(basis_mat)  # (X'X)^-1

    # t_alpha/2, n-p
    t_val = t_stat(deg_of_free).ppf(1 - signif / 2)  # high CI bound

    for i in range(beg, end):
        coeff_uncert[i - beg] = t_val * np.sqrt(err_var * inverse_mat[i, i])

    comm.Allgatherv(
        [coeff_uncert, count, MPI_DOUBLE],
        [coeff_uncert_tot, seq_count, seq_disp, MPI_DOUBLE]
    )

    return coeff_uncert_tot


def calc_var_conf_int(matrix_coeffs, coeff_uncert, norm_sq):
    """
    Inputs: matrix_coeffs- an array of the matrix coefficients
            coeff_uncert- the uncertainty of each coefficient
            norm_sq- the norm squared of the model
    
    Calculates the bounds on the variance from the calculated bounds on the 
    matrix coefficients. This equation was created by replacing the matrix 
    coefficients in the variance equation with the lower and upper bounds on the 
    matrix coefficients.
    
    Equation derived from method of calculating variance and the matrix 
    coefficient uncertainties.
    """
    coeff_mag = np.abs(matrix_coeffs)

    L_coeffs = coeff_mag - coeff_uncert
    # Edge case of when uncertainty magnitude is larger than coefficient
    L_coeffs[coeff_mag <= coeff_uncert] = 0

    # To have the largest magnitude of coefficients squared, add uncertainty to
    # postitive coeffs and subtract it from negative ones
    H_coeffs = coeff_mag + coeff_uncert

    low_matrix_coeffs_sq = (
        np.reshape(L_coeffs, (len(L_coeffs), 1))[1:] ** 2
    )

    low_norm_mult_coeff = norm_sq[1:] * low_matrix_coeffs_sq
    if low_norm_mult_coeff.shape[0] != 1 and low_norm_mult_coeff.shape[1] != 1:
        warn(
            'Sigma squared does not look correct for calculating the Sobol '
            'indices.'
        )

    low_variance = np.sum(low_norm_mult_coeff)

    high_matrix_coeffs_sq = (
        np.reshape(H_coeffs, (len(H_coeffs), 1))[1:] ** 2
    )

    high_norm_mult_coeff = norm_sq[1:] * high_matrix_coeffs_sq
    if high_norm_mult_coeff.shape[0] != 1 and high_norm_mult_coeff.shape[1] != 1:
        warn(
            'Sigma squared does not look correct for calculating the Sobol '
            'indices.'
        )

    high_variance = np.sum(high_norm_mult_coeff)

    return low_variance, high_variance


def get_sobol_bounds(matrix_coeffs, sobols, coeff_uncert, norm_sq):
    """
    Inputs: matrix_coeffs- matrix coefficients
            sobols- sobol sensitivities
            coeff_uncert- uncertainties of the coefficients
            norm_sq- the norm squared
    
    Calculates the bounds on the sobols from the coefficient uncertainty.
    
    Equation derived from method of calculating sobols and the matrix 
    coefficient uncertainties.
    """
    threshold = 1e-4  # sobol significance level

    comm = MPI_COMM_WORLD
    size = comm.size
    rank = comm.rank

    iter_count = len(matrix_coeffs) - 1

    base = iter_count // size
    rem = iter_count % size
    beg = base * rank + (rank >= rem) * rem + (rank < rem) * rank + 1
    count = base + (rank < rem)
    end = beg + count

    ranks = np.arange(0, size, dtype=int)
    seq_count = (ranks < rem) + base
    seq_disp = base * ranks + (ranks >= rem) * rem + (ranks < rem) * ranks

    lower_sobols = np.zeros(count)
    upper_sobols = np.zeros(count)

    lower_sobols_tot = np.zeros(iter_count)
    upper_sobols_tot = np.zeros(iter_count)

    coeff_mag = np.abs(matrix_coeffs)

    for i in range(beg, end):
        i_incr = i + 1
        i_decr = i - 1

        i_sc = i - beg

        # all coeffs squared- can make them all positive for ease of magnitude
        L_coeffs = np.copy(coeff_mag)
        H_coeffs = np.copy(coeff_mag)

        # make one smallest coeff, all other largest coeff
        L_coeffs[0:i] += coeff_uncert[0:i]
        L_coeffs[i_incr:] += coeff_uncert[i_incr:]
        L_coeffs[i] = (
            0 if coeff_mag[i] <= coeff_uncert[i]
            else coeff_mag[i] - coeff_uncert[i]
        )

        # make one largest coeff, all other smallest coeff
        H_coeffs[0:i] -= coeff_uncert[0:i]
        H_coeffs[i_incr:] -= coeff_uncert[i_incr:]
        H_coeffs[coeff_mag < coeff_uncert] = 0
        H_coeffs[i] = coeff_mag[i] + coeff_uncert[i]

        # If the uncertainty is larger than the magnitude, then the smallest
        # coefficient magnitude is 0; don't do this for the L_coeffs because we
        # want denominator to be the largest possible for the smallest Sobol
        H_coeffs[coeff_mag <= coeff_uncert][0:i] = 0
        H_coeffs[coeff_mag <= coeff_uncert][i_incr:] = 0
        L_coeffs[i] = 0 if coeff_mag[i] <= coeff_uncert[i] else  L_coeffs[i]

        # squaring and reshaping coefficients- positive column array
        L_alt_matrix_coeffs_sq = (
            np.reshape(L_coeffs, (len(L_coeffs), 1))[1:] ** 2
        )

        H_alt_matrix_coeffs_sq = (
            np.reshape(H_coeffs, (len(H_coeffs), 1))[1:] ** 2
        )

        # sobols from altered coeffs- sobols are based on coeff magnitude
        lower_sobols[i_sc] = (
            (L_coeffs[i] ** 2 * norm_sq[i])
            / np.sum(norm_sq[1:] * L_alt_matrix_coeffs_sq)  # lower_sigma_sq
        )

        upper_sobols[i_sc] = (
            (H_coeffs[i] ** 2 * norm_sq[i])
            / np.sum(norm_sq[1:] * H_alt_matrix_coeffs_sq)  # upper_sigma_sq
        )

        lower_sobol_wrong = (lower_sobols[i_sc] > sobols[i_decr])
        upper_sobol_wrong = (upper_sobols[i_sc] < sobols[i_decr])

        sobol_is_small = sobols[i_decr] < threshold

        # if a significantly-large sobol has wrong bounds, warn the user
        if (lower_sobol_wrong or upper_sobol_wrong):
            warn(f'Sobol number {i_sc} has limits that are not correct.')

            if lower_sobol_wrong and sobol_is_small:
                lower_sobols[i_sc] = np.copy(sobols[i_decr])

            if upper_sobol_wrong and sobol_is_small:
                upper_sobols[i_sc] = np.copy(sobols[i_decr])

    comm.Allgatherv(
        [lower_sobols, count, MPI_DOUBLE],
        [lower_sobols_tot, seq_count, seq_disp, MPI_DOUBLE]
    )

    comm.Allgatherv(
        [upper_sobols, count, MPI_DOUBLE],
        [upper_sobols_tot, seq_count, seq_disp, MPI_DOUBLE]
    )

    return lower_sobols_tot, upper_sobols_tot


def calc_error_variance(var_basis, matrix_coeffs, responses):
    """
    Inputs: var_basis- evaluated variable basis
            matrix_coeffs- the matrix coefficients
            responses- the responses
    
    Calculates the unbiased estimator of the error variance.
    
    Design and Analysis of Experiments (8th) by Douglas Montgomery (pg. 453)
    """
    # sigma squared eq (Design and Analysis of Experiments, pg 453)
    resp_count, term_count = var_basis.shape
    high_err_thresh = -1e-4

    if (resp_count - term_count) < 1:
        raise ValueError(
            'You must have more responses than terms in your model. For a '
            f'{term_count} term model, add {(term_count - resp_count) + 1} more '
            'responses.'
        )

    err_sum_sq = calc_error_sum_of_sq(var_basis, matrix_coeffs, responses)
    err_var = err_sum_sq / (resp_count - term_count)

    if err_var < 0:
        if err_var >= high_err_thresh:
            warn(
                'Error variance is small and negative. Taking the absolute '
                'value for use in calculations.'
            )
            err_var = np.abs(err_var)
        else:
            raise ValueError(
                f'The error variance for this model is {err_var} and should be '
                'positive. Make sure variable basis and responses are correct.'
            )

    return err_var


def calc_error_sum_of_sq(var_basis, matrix_coeffs, responses):
    """
    Inputs: var_basis- the variable basis
            matrix_coeffs- the matrix coefficients
            responses- the responses
    
    Calculates the error sum of squares.
    
    Design and Analysis of Experiments (8th) by Douglas Montgomery (pg. 453)
    """
    resp_mat = np.matmul(responses.T, responses)
    coeff_dot_basis = np.matmul(matrix_coeffs.T, var_basis.T)

    return resp_mat - np.matmul(coeff_dot_basis, responses)


def calc_total_sum_of_sq(var_basis, responses):
    """
    Inputs: var_basis- the variable basis
            responses- the responses 
            
    Calculates the total sum of squares.
    
    Design and Analysis of Experiments (8th) by Douglas Montgomery (pg. 463)
    """
    act_model_size = var_basis.shape[0]
    resp_mat = np.matmul(responses.T, responses)

    return resp_mat - (np.sum(responses) ** 2 / act_model_size)


def calc_sum_sq_regr(matrix_coeffs, responses, var_basis):
    """
    Inputs: matrix_coeffs- the matrix coefficients
            responses- the responses
            var_basis- the variable basis
    
    Calculates the regression sum of squares.
    
    Design and Analysis of Experiments (8th) by Douglas Montgomery (pg. 463)
    """
    resp_cnt = len(responses)

    regr_sum_sq = np.matmul(
        np.matmul(np.transpose(matrix_coeffs), np.transpose(var_basis)),
        responses
    ) - (np.sum(responses) ** 2 / resp_cnt)

    return regr_sum_sq


def calc_mean_sq_err(responses, matrix_coeffs, var_basis):
    """
    Inputs: responses- the actual responses from the analysis tool
            matrix_coeffs - the coefficients for the terms from the 
            MatrixSystem
            var_basis- the varible basis with values plugged in
            
    Calculates the mean squared error for the given matrices.
    
    Equation from original UQPCE version.
    """
    comm = MPI_COMM_WORLD
    rank = comm.rank
    mean_sq_err_thresh = 1e-15
    resp_count, term_count = var_basis.shape
    mean_sq_error = -1.0

    if rank == 0:
        expect_resp_dot_resp = (
            np.matmul(
                np.matmul(np.transpose(matrix_coeffs), np.transpose(var_basis)),
                responses
            )
        )

        # ((R.T R) - (a.T X.T R)) / (n - p)
        mean_sq_error = (
            (np.matmul(np.transpose(responses), responses) - expect_resp_dot_resp)
            / (resp_count - term_count)
        )

        # if value is negative, return the value that must be positive
        if (mean_sq_error < mean_sq_err_thresh):
            pred = np.dot(var_basis, matrix_coeffs)
            error = pred - responses

            warn('Negative mean squared error\n\nUsing an alternate equation')

            mean_sq_error = np.sum(error ** 2) / len(error)

    mean_sq_error = comm.bcast(mean_sq_error, root=0)

    return mean_sq_error


def calc_partial_F(regr_sum_sq_all, regr_sum_sq, mean_sq_err_all, deg_free):
    """
    Inputs: regr_sum_sq_all- the regression sum of squares value for the model 
            with the larger number of terms
            regr_sum_sq- the regression sum of squares value for the model 
            with the smaller number of terms
            mean_sq_err_all- the mean squared error for the model with the 
            larger number of terms
            deg_free- the number of degrees of freedom
            
    Calculates the partial F statistic used in stepwise regression.
    
    "Applied Statistics and Probablity for Engineers" (4th) by Douglas 
    Montgomery (pg. 486)
    """
    partial_F = ((regr_sum_sq_all - regr_sum_sq) / deg_free) / mean_sq_err_all
    return partial_F


def calc_hat_matrix(var_basis):
    """
    Inputs: var_basis- the evaluated variable basis

    Calculates the hat matrix for a variable basis.
    """
    comm = MPI_COMM_WORLD
    rank = comm.rank

    resp_count = var_basis.shape[0]

    if rank == 0:
        var_basis_T = np.transpose(var_basis)
        transformed_matrix = inv(np.matmul(var_basis_T, var_basis))

        hat_matrix = np.matmul(
            np.matmul(var_basis, transformed_matrix), var_basis_T
        )

#         hat_matrix = np.matmul(
#             var_basis,
#             np.linalg.solve(np.matmul(var_basis_T, var_basis), var_basis_T)
#         )
    else:
        hat_matrix = np.zeros([resp_count, resp_count])

    comm.Bcast([hat_matrix, MPI_DOUBLE], root=0)

    return hat_matrix


def calc_term_count(order, var_count):
    """
    Inputs: order- the order of the model
            var_count- the number of variables in the model
    
    Calculates the number of terms in a model.
    """
    term_count = int(
        math.factorial(order + var_count)
        / (math.factorial(order) * math.factorial(var_count))
    )

    return term_count


def calc_min_responses(order, var_count):
    """
    Inputs: order- the order of the model
            var_count- the number of variables in the model
            
    Calculates the minimum responses required for a model.
    """
    return calc_term_count(order, var_count) + 1
