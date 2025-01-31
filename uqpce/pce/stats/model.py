from itertools import combinations

import numpy as np
from scipy.stats import f as f_dist, pearsonr

from uqpce.pce._helpers import solve_coeffs
from uqpce.pce.stats.statistics import (
    calc_sum_sq_regr, calc_mean_sq_err, calc_partial_F
)


def stepwise_regression(
        orig_var_basis, responses, alpha_in=0.15, alpha_out=0.15,
        partial_model=True, sort=True
    ):
    """
    Inputs: orig_var_basis- the full variable basis
            responses- the array of responses
            alpha_in- the alpha parameter used for the F-distribution threshold 
            for values to be added to the model (default: 0.15)
            alpha_out- the alpha parameter used for the F-distribution threshold 
            for values to be removed to the model (default: 0.15)
            partial_model- boolean if the model will be built using only 
            (model_size + 1) terms from variable basis and responses (True) or 
            if all of will be used (False) (default: True)
            sort- whether the indices will be returned in the sorted order or 
            the order in which they're added
    
    Uses stepwise regression to build a model of the most relevant terms of 
    the model.
    
    Applied Statistics and Probability for Engineers, 4th ed (pg. 486)
    """

    curr_term_count = 2  # starts with intercept + one term

    if partial_model:
        resps = responses[0:curr_term_count]
    else:
        resps = responses

    if partial_model:
        var_basis = orig_var_basis[0:curr_term_count, :]
    else:
        var_basis = orig_var_basis

    idx = get_most_correlated(var_basis, resps)
    combo = np.array([0, int(idx)])  # intercept + best term

    combo = adapt_stepwise_regression(
        combo, orig_var_basis, responses, alpha_in=alpha_in, alpha_out=alpha_out,
        partial_model=partial_model, sort=sort
    )

    return combo


def adapt_stepwise_regression(
        combo, orig_var_basis, responses, alpha_in=0.15, alpha_out=0.15,
        partial_model=True, sort=True
    ):
    """
    Inputs: combo- the starting combination (intercept + strongest term)
            orig_var_basis- the full variable basis
            responses- the array of responses
            alpha_in- the alpha parameter used for the F-distribution threshold 
            for values to be added to the model (default: 0.15)
            alpha_out- the alpha parameter used for the F-distribution threshold 
            for values to be removed to the model (default: 0.15)
            partial_model- boolean if the model will be built using only 
            (model_size + 1) terms from variable basis and responses (True) or 
            if all of will be used (False) (default: True)
            sort- whether the indices will be returned in the sorted order or 
            the order in which they're added
    
    Uses stepwise regression to build a model of the most relevant terms of 
    the model.
    
    Applied Statistics and Probability for Engineers, 4th ed (pg. 486)
    """

    curr_term_count = len(combo)  # starts with intercept + one term
    deg_free = 1  # each iteration only has one more or one less term than the previous

    if partial_model:
        resps = responses[0:curr_term_count]
    else:
        resps = responses

    tot_response_count, tot_term_count = orig_var_basis.shape

    all_combos = np.array(range(1, tot_term_count))

    if partial_model:
        var_basis = orig_var_basis[0:curr_term_count, :]
    else:
        var_basis = orig_var_basis

    combo_list = list(combo)

    shift = 1
    term_change = 1  # always adding or removing only one term

    mean_sq_err_thresh = 0
    combo_size = len(combo)
    unincl_term_count = tot_term_count - combo_size

    curr_term_count = combo_size + shift

    if partial_model:
        var_basis = orig_var_basis[0:curr_term_count, combo_list]
        resps = responses[0:curr_term_count]
    else:
        var_basis = orig_var_basis[:, combo_list]
        resps = responses

    matrix_coeffs = solve_coeffs(var_basis, resps)
    sum_sq_reg_0 = calc_sum_sq_regr(matrix_coeffs, resps, var_basis)

    prev_combos = []

    # dont build model w/o deg of freedom for err
    while combo_size < tot_term_count and (tot_response_count - combo_size) > 1:

        if combo_list in prev_combos:
            combo = np.array(prev_combos[-1])
            combo_list = list(combo)
            break

        else:
            prev_combos.append(combo_list)

        unincl_term_count = tot_term_count - combo_size
        unincl_terms = np.delete(all_combos, combo[shift:].astype(int) - shift)
        curr_term_count = combo_size + shift
        model_size = curr_term_count + shift  # one more than term count

        if partial_model:
            resps = responses[0:model_size]
        else:
            resps = responses

        response_count = len(resps)

        signif_F_in = (
            f_dist(
                term_change, (response_count - curr_term_count)
            ).ppf(1 - alpha_in)
        )

        # ------------------------ add in terms ----------------------------
        partial_F_in = np.zeros(unincl_term_count)

        for i in range(unincl_term_count):
            temp_combo = np.append(combo, unincl_terms[i])
            temp_combo_list = list(temp_combo)

            if partial_model:
                var_basis = orig_var_basis[0:model_size, temp_combo_list]
            else:
                var_basis = orig_var_basis[:, temp_combo_list]

            matrix_coeffs = solve_coeffs(var_basis, resps)

            sum_sq_reg_curr = (
                calc_sum_sq_regr(matrix_coeffs, resps, var_basis)
            )

            mean_sq_err = (
                calc_mean_sq_err(resps, matrix_coeffs, var_basis)
            )

            partial_F_in[i] = (
                calc_partial_F(sum_sq_reg_curr, sum_sq_reg_0, mean_sq_err, deg_free)
            )

        max_idx = np.argmax(partial_F_in)

        # ------------------------ calc new vals ---------------------------
        if partial_F_in[max_idx] > signif_F_in:
            combo = np.append(combo, unincl_terms[max_idx])
            combo_list = list(combo)
            combo_size += shift

        if partial_model:
            var_basis = orig_var_basis[0:model_size, combo_list]
            resps = responses[0:model_size]
        else:
            var_basis = orig_var_basis[:, combo_list]
            resps = responses

        matrix_coeffs = solve_coeffs(var_basis, resps)

        sum_sq_reg_0 = (
            calc_sum_sq_regr(matrix_coeffs, resps, var_basis)
        )

        mean_sq_err = (
            calc_mean_sq_err(resps, matrix_coeffs, var_basis)
        )

        curr_term_count = combo_size - shift
        model_size = curr_term_count + shift  # one more than term count

        # ------------------------ remove terms ----------------------------
        if mean_sq_err > mean_sq_err_thresh:

            if partial_model:
                resps = responses[0:model_size]
            else:
                resps = responses

            response_count = len(resps)
            partial_F_out = np.zeros(curr_term_count)  # - shift

            signif_F_out = (
                f_dist(
                    term_change, (response_count - curr_term_count)
                ).ppf(1 - alpha_out)
            )

            for i in range(shift, curr_term_count + shift):  # dont remove added param

                temp_combo = np.delete(combo, i)
                temp_combo_list = list(temp_combo)

                if partial_model:
                    var_basis = orig_var_basis[0:model_size, temp_combo_list]
                else:
                    var_basis = orig_var_basis[:, temp_combo_list]

                matrix_coeffs = solve_coeffs(var_basis, resps)

                sum_sq_reg_curr = (
                    calc_sum_sq_regr(
                        matrix_coeffs, resps, var_basis
                    )
                )

                # sum_sq_reg_0 &sum_sq_reg_curr swapped bc sum_sq_reg_0 more terms
                partial_F_out[i - shift] = (
                    calc_partial_F(
                        sum_sq_reg_0, sum_sq_reg_curr, mean_sq_err, deg_free
                    )
                )

            min_idx = np.argmin(partial_F_out)

            if partial_F_out[min_idx] < signif_F_out:
                combo = np.delete(combo, min_idx + shift)  # combo[0]=intercept
                combo_list = list(combo)
                combo_size -= shift

        else:
            break

        if sort:
            combo = np.sort(combo)
        combo_list = list(combo)

    if sort:
        combo = np.sort(combo)

    return combo


def backward_elimination(var_basis, responses, alpha_out=0.15, sort=False):
    """
    Inputs: var_basis- the evaluated variable basis
            responses- the array of responses
            alpha_out- the alpha parameter used for the F-distribution threshold
            for values to be removed to the model (default: 0.15)

    Uses backward elimination to build a model of the most relevant terms of
    the model.

    Applied Statistics and Probability for Engineers, 4th ed (pg. 486)
    """

    response_count, tot_term_count = var_basis.shape

    if (response_count - tot_term_count) < 1:
        raise RuntimeError(
            f'There must be {tot_term_count+1} response values to use backward '
            f'elimination with a model of size {tot_term_count}'
        )

    shift = 1
    term_change = 1  # always adding or removing only one term
    min_terms = 1
    mean_sq_err_thresh = 1e-15
    deg_free = 1  # response_count - curr_term_count

    combo = np.arange(0, tot_term_count)  # all terms
    combo_list = list(combo)
    combo_size = len(combo)
    curr_term_count = combo_size + shift

    temp_eval = var_basis[:, combo_list]

    matrix_coeffs = solve_coeffs(temp_eval, responses)
    sum_sq_reg_0 = calc_sum_sq_regr(matrix_coeffs, responses, temp_eval)

    prev_combos = []

    while combo_size > min_terms:

        if combo_list in prev_combos:
            combo = np.array(prev_combos[-1])
            combo_list = list(combo)
            break

        else:
            prev_combos.append(combo_list)

        temp_eval = var_basis[:, combo_list]
        matrix_coeffs = solve_coeffs(temp_eval, responses)

        sum_sq_reg_0 = calc_sum_sq_regr(matrix_coeffs, responses, temp_eval)
        mean_sq_err = calc_mean_sq_err(responses, matrix_coeffs, temp_eval)
        curr_term_count = combo_size - shift

        # ------------------------ remove terms ----------------------------
        if mean_sq_err > mean_sq_err_thresh:

            partial_F_out = np.zeros(curr_term_count)

            signif_F_out = f_dist(
                    term_change, (response_count - curr_term_count)
            ).ppf(1 - alpha_out)

            for i in range(shift, curr_term_count + shift):  # dont remove intercept

                temp_combo = np.delete(combo, i)
                temp_combo_list = list(temp_combo)
                temp_eval = var_basis[:, temp_combo_list]
                matrix_coeffs = solve_coeffs(temp_eval, responses)

                sum_sq_reg_curr = calc_sum_sq_regr(
                    matrix_coeffs, responses, temp_eval
                )

                # sum_sq_reg_0 &sum_sq_reg_curr swapped bc sum_sq_reg_0 more terms
                partial_F_out[i - shift] = calc_partial_F(
                    sum_sq_reg_0, sum_sq_reg_curr, mean_sq_err, deg_free
                )

            min_idx = np.argmin(partial_F_out)

            if partial_F_out[min_idx] < signif_F_out:
                combo = np.delete(combo, min_idx + shift)  # combo[0]=intercept
                combo_list = list(combo)
                combo_size -= shift

        else:
            break

        combo_list = list(combo)

    # update object attributes and variables
    if sort:
        combo = np.sort(combo)

    return combo


def get_most_correlated(var_basis, responses):
    """
    Inputs: var_basis- the variable basis matrix
            responses- the vector of responses
    
    Returns the index of the term that correlates most strongly to the 
    responses.
    """
    intercept_shift = 1

    term_count = var_basis.shape[1]
    pear_coeffs = np.zeros(term_count - intercept_shift)

    for i in range(intercept_shift, term_count):  # ignore intercept
        pear_coeffs[i - intercept_shift] = np.abs(
            pearsonr(var_basis[:, i], responses)[0]
        )

    # Constant arrays will output nan, which is seen as larger than 1. These
    # values are excluded, so the nan is replaced with 0.
    pear_coeffs[np.isnan(pear_coeffs)] = 0

    return np.argmax(pear_coeffs) + intercept_shift


def find_lower_terms(model_matrix, row_idx, combo):
    """
    Inputs: model_matrix- the matrix that contains integers for the variable 
            and order present for each term
            row_idx- an array containing the indices of the rows in the 
            model_matrix whose lower-order terms that make them up will be 
            found
            combo- the input combo whose lower terms will be found
    
    Takes some input of the non-zero terms in the combination array. 
    Generates all model terms that make up the higher terms.
    (i.e. if x0*x1**2 is chosen, terms x0, x1, x0*x1, and x1**2 are added 
    if they aren't already included.)
    """

    for row in row_idx:  # each row index value
        indices = np.argwhere(model_matrix[row] != 0)

        for idx in indices:  # 0, 1 for [1, 2] but 1 for [0, 2] bc a[0] = 0
            arr = np.copy(model_matrix[row])

            while arr[idx] > 0:  # decrement until the value reaches zero
                arr[idx] -= 1
                new_idx = np.where((model_matrix == arr).all(axis=1))

                # the model_matrix row isn't already in combo_best
                if new_idx not in combo:
                    combo = np.append(combo, new_idx)

                # recursive call for each decrement to get all terms
                sub_row_idx = arr[arr > 0].astype('int')
                find_lower_terms(model_matrix, sub_row_idx, combo)

    return np.sort(combo)


def best_subset(var_basis, responses, model_size):
    """
    Inputs: var_basis- the variable basis matrix
            responses- the response matrix
            model_size- the size of the desired model; must be smaller than the 
            full model size for the given order

    Finds the subset of size 'model_size' with the smallest amount of model 
    error.
    """
    combo_temp = range(1, model_size + 1)
    all_combos = combinations(combo_temp, model_size - 1)
    all_combos = np.asarray(list(all_combos))
    x_size, y_size = all_combos.shape
    fin_combo = np.zeros([x_size, y_size + 1])
    fin_combo[:, 1:] = all_combos

    error = np.zeros(x_size)

    for i in range(x_size):
        combo = fin_combo[i, :].astype('int')
        combo_list = list(combo)
        basis = var_basis[:, combo_list]

        matrix_coeffs = solve_coeffs(basis, responses)
        pred = np.dot(basis, matrix_coeffs)
        error[i] = np.sum(np.abs(pred - responses))

    return fin_combo[np.argmin(error), :].astype('int')

