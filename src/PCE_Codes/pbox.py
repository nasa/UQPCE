from warnings import warn
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from sympy import symbols
    from sympy.utilities.lambdify import lambdify
    from mpi4py.MPI import DOUBLE as MPI_DOUBLE, COMM_WORLD as MPI_COMM_WORLD
except:
    warn('Ensure that all required packages are installed.')
    exit()

from PCE_Codes.custom_enums import UncertaintyType
from PCE_Codes._helpers import _warn, evaluate_points, evaluate_points_verbose
from PCE_Codes.stats.statistics import calc_mean_conf_int
from PCE_Codes.variables.continuous import ContinuousVariable


class ProbabilityBoxes:
    """
    Inputs: var_list- the list of variables
            verbose- if in verbose mode or not
            plot- if the plots should be generated
            track_conv_off- if convergence tracking is off
            epist_samps- the number of epistemic samples
            aleat_samps- the number of aleatory samples
            aleat_sub_size- the number of aleatory samples for a sub iteration 
            when convergence tracking is used
            epist_sub_size- the number of epistemic samples for a sub iteration 
            when convergence tracking is used
    
    The probability box (pbox) plots that show the confidence interval from 
    the data.
    """
    __slots__ = (
        'var_list_symb', 'epistemic_list', 'var_list', 'var_count',
        'epist_var_count', 'input', 'verbose', 'plot', 'track_conv_off',
        'epist_samps', 'aleat_samps', 'total_samps', 'eval_resps',
        'matrix_coeffs', 'var_basis_resamp', 'aleat_sub_size', 'epist_sub_size',
        'mean_uncert'
    )

    def __init__(
            self, var_list, verbose=False, plot=False, track_conv_off=False,
            epist_samps=125, aleat_samps=25000, aleat_sub_size=5000,
            epist_sub_size=25
        ):

        self.input = True
        self.verbose = verbose
        self.var_list = var_list
        self.var_count = len(var_list)
        self.plot = plot
        self.track_conv_off = track_conv_off

        self.epist_samps = epist_samps
        self.aleat_samps = aleat_samps

        self.aleat_sub_size = aleat_sub_size
        self.epist_sub_size = epist_sub_size
        showwarning = _warn

        self.var_list_symb = np.array(
            [symbols(f'x{j}') for j in range(self.var_count)]
        )

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
        Creates resample values for aleatory variables
        """
        comm = MPI_COMM_WORLD
        rank = comm.rank

        if self.epist_var_count == 0:
            self.epist_samps = 1

        samp_count = self.aleat_samps * self.epist_samps

        if rank == 0 and self.verbose:
            print('Generating aleatory resampling values\n')

        for i in range(self.var_count):
            curr_var = self.var_list[i]

            if curr_var.type is UncertaintyType.ALEATORY:
                curr_var.get_resamp_vals(samp_count)

    def generate_epistemic_samples(self):
        """
        Determines the number of samples needed and generates the resample 
        values for epistemic variables.
        """
        comm = MPI_COMM_WORLD
        rank = comm.rank

        if rank == 0 and self.verbose:
            print('Generating epistemic resampling values\n')

        y_size = 1

        ones = np.ones([self.aleat_samps, y_size])

        if self.epist_var_count == 0:
            self.epist_samps = 1

        self.total_samps = self.epist_samps * self.aleat_samps

        for epistemic_var in self.epistemic_list:
            resample = np.zeros([self.total_samps, y_size])

            # shift due to using the standard bounds for curves
            eps = epistemic_var.get_resamp_vals(self.epist_samps)

            for i in range(self.epist_samps):
                i_begin = i * self.aleat_samps
                i_end = i_begin + self.aleat_samps

                resample[i_begin:i_end] = ones * eps[i]

            epistemic_var.resample = np.copy(resample)

    def evaluate_surrogate(
            self, var_basis_vect_symb, sig, matrix_coeffs, conv_thresh_percent,
            graph_dir=None
        ):
        """
        Inputs: var_basis_vect_symb- symbolic variable basis (psi) vector
                sig- the level of significance
                matrix_coeffs- the coefficients from solving the matrix (psi)
                system of equations
                conv_thresh_percent- the percent of the response mean to be
                used as the thresh for tracking convergence
                graph_dir- name of graph directory

        Resamples to generate responses from the model using various variable
        inputs.
        """
        comm = MPI_COMM_WORLD
        rank = comm.rank
        size = comm.size

        task_count = 3  # N tasks to execute simultaneously
        reps = int(np.ceil(task_count / size))
        opts = np.repeat(np.arange(0, size), reps)

        handle_conv_file = (rank == opts[0])
        handle_cih_file = (rank == opts[1])
        handle_cil_file = (rank == opts[2])

        resp_mean = np.abs(matrix_coeffs[0])
        shift_thresh = 0.3  # 30%
        self.matrix_coeffs = matrix_coeffs

        epist_is_one = self.epist_samps == 1
        term_count = len(self.matrix_coeffs)

        # initialized for the non-zero rank processes
        self.var_basis_resamp = np.zeros([self.total_samps, term_count])
        self.eval_resps = np.zeros(self.total_samps)

        is_manager = (rank == 0)
        if self.verbose and is_manager:
            print('Resampling the surrogate model\n')

        if epist_is_one:
            if self.verbose and is_manager:
                print(
                    'No epistemic variables found, perfoming pure aleatory '
                    'analysis\n'
                )

        var_basis_vect_func = lambdify(
            (self.var_list_symb,), var_basis_vect_symb, modules='numpy'
        )

        if not self.track_conv_off:
            if self.verbose and is_manager:
                print(
                    'Sampling until specified convergence rate of '
                    f'{conv_thresh_percent:.3%} is met\n'
                )

            low_bnd = sig / 2  # the bounds of the responses that will
            high_bnd = 1 - sig / 2  # be interpolated to get conf ints (CIs)

            aleat_run_count = (self.aleat_samps // self.aleat_sub_size)
            epist_run_count = (self.epist_samps // self.epist_sub_size)

            conf_int_low = np.ones([self.epist_samps, aleat_run_count]) * np.inf  # lists of each iteration's
            conf_int_high = np.ones([self.epist_samps, aleat_run_count]) * -np.inf  # upper and lower CIs

            out_conf_int_low = np.ones(self.epist_samps) * np.inf
            out_conf_int_high = np.ones(self.epist_samps) * -np.inf

            set_low = np.ones(epist_run_count) * np.inf
            set_high = np.ones(epist_run_count) * -np.inf

            converged = False  # individual curve converged? False
            outer_converged = False  # set of curves converged? False

            idx = 1
            thresh = conv_thresh_percent * np.abs(resp_mean)

            if graph_dir is not None and handle_conv_file:
                conv_file_name = f'{graph_dir}/convergence_values.dat'
                conv_file = open(conv_file_name, 'w')

            if graph_dir is not None and handle_cil_file:
                fig_file_low = f'{graph_dir}/CIL_convergence'
                fig_low = plt.figure(1)
                fig_low.suptitle('Convergence of Confidence Interval (low)')
                conf_int_low_fig = fig_low.subplots()

            if graph_dir is not None and handle_cih_file:
                fig_file_high = f'{graph_dir}/CIH_convergence'
                fig_high = plt.figure(2)
                fig_high.suptitle = ('Convergence of Confidence Interval (high)')
                conf_int_high_fig = fig_high.subplots()

            if epist_is_one:
                out_msg = (
                    f'The probability curve did not converge. '
                    f'{self.aleat_samps} samples were used.\n'
                )

            else:
                out_msg = (
                    f'The probability curves did not converge- '
                    f'{self.epist_samps} curves were used\n'
                )

            if rank == 0:

                self.eval_resps = np.ones(self.total_samps) * np.inf
                self.var_basis_resamp = np.ones([self.total_samps, term_count]) * np.inf

    #--------------------------   create pbox curve(s)    --------------------------
                for ep in range(self.epist_samps):  # number of pbox curves to make
                    beg_var_idx = ep * self.aleat_samps

                    for al in range(aleat_run_count):  # max iterations per curve
                        al_inc = al + 1
                        al_dec = al - 1

                        iter_end = al_inc * self.aleat_sub_size
                        beg_idx = al * self.aleat_sub_size + beg_var_idx
                        end_idx = iter_end + beg_var_idx

                        self.var_basis_resamp[beg_idx:end_idx, :] = evaluate_points(
                            var_basis_vect_func, beg_idx, self.aleat_sub_size,
                            self.var_list, 'resample'
                        )

                        self.eval_resps[beg_idx:end_idx] = np.matmul(
                            self.var_basis_resamp[beg_idx:end_idx, :],
                            self.matrix_coeffs
                        )

                        # finding the upper and lower conf interval for the curve
                        y_vals = np.linspace(0, 1, num=iter_end, endpoint=True)
                        sorted_data = np.sort(self.eval_resps[beg_var_idx:end_idx], axis=0)

                        conf_int_low[ep, al] = np.interp(low_bnd, y_vals, sorted_data)
                        conf_int_high[ep, al] = np.interp(high_bnd, y_vals, sorted_data)

                        if al > 0:
                            in_low_diff = np.abs(
                                conf_int_low[ep, al] - conf_int_low[ep, al_dec]
                            )

                            in_high_diff = np.abs(
                                conf_int_high[ep, al] - conf_int_high[ep, al_dec]
                            )

                            if self.verbose and epist_is_one and is_manager:
                                print(
                                    f'low: {in_low_diff / resp_mean:.5%}'
                                    f'    high: {in_high_diff / resp_mean:.5%}\n'
                                )

                            if converged and epist_is_one:  # if converges twice, break
                                if (
                                    (in_low_diff < thresh)
                                    and (in_high_diff < thresh)
                                ):
                                    out_msg = (
                                        'The probability curve has converged.\n'
                                    )

                                    converged = False
                                    break

                                else:  # iforce convergence twice
                                    converged = False

                            if (
                                (in_low_diff < thresh)
                                and (in_high_diff < thresh)
                            ):
                                # after the change is below the thresh, we need
                                # one more run
                                converged = True

                    out_conf_int_low[ep] = np.min(conf_int_low[ep, 0:al])
                    out_conf_int_high[ep] = np.max(conf_int_high[ep, 0:al])

                    # checks to do for each set of curves for set convergence
                    if (ep > 1) and ((ep + 1) % self.epist_sub_size == 0):
                        idx = (ep // self.epist_sub_size) - 1
                        set_low[idx] = np.min(out_conf_int_low)
                        set_high[idx] = np.max(out_conf_int_high)

                        if graph_dir is not None:
                            conv_file.write(
                                f'    set: [{set_low[idx]} , {set_high[idx]}]\n'
                            )

                        if (idx > 0):
                            low_diff = np.abs(set_low[idx] - set_low[idx - 1])
                            high_diff = np.abs(set_high[idx] - set_high[idx - 1])

                            if self.verbose and is_manager:
                                print(
                                    f'low: {low_diff / resp_mean:.5%}    '
                                    f'high: {high_diff / resp_mean:.5%}\n'
                                )

                            low_conv = low_diff < thresh
                            high_conv = high_diff < thresh

                            if outer_converged:  # if converges twice, break
                                if low_conv and high_conv:
                                    out_msg = (
                                        'The probability curves have converged.\n'
                                    )
                                    break

                                else:  # force two convergences in a row
                                    outer_converged = False

                            if low_conv and high_conv:
                                    # after the change is below the thresh, we need
                                outer_converged = True  # one more iter after convergence

            comm.Bcast([conf_int_low, MPI_DOUBLE], root=0)
            comm.Bcast([conf_int_high, MPI_DOUBLE], root=0)
            comm.Bcast([out_conf_int_low, MPI_DOUBLE], root=0)
            comm.Bcast([out_conf_int_high, MPI_DOUBLE], root=0)
            out_msg = comm.bcast(out_msg, root=0)

            if graph_dir is not None and epist_is_one and handle_conv_file:
                conv_file.write(
                    ''.join(('\nlow:  ', str(conf_int_low[0, 0:al]), '\nhigh: ',
                    str(conf_int_high[0, 0:al])))
                )
                conv_file.close()

            if graph_dir is not None:

                if epist_is_one:
                    low_arr = conf_int_low
                    high_arr = conf_int_high

                else:
                    low_arr = out_conf_int_low
                    high_arr = out_conf_int_high

                mx = np.max(low_arr[low_arr != np.inf])
                mn = np.min(low_arr[low_arr != np.inf])
                shift = (mx - mn) * shift_thresh
                low_lim_lower = mn - shift
                low_lim_upper = mx + shift

                mx = np.max(high_arr[high_arr != -np.inf])
                mn = np.min(high_arr[high_arr != -np.inf])
                shift = (mx - mn) * shift_thresh
                high_lim_lower = mn - shift
                high_lim_upper = mx + shift

                if handle_cil_file:
                    for ep in range(self.epist_samps):
                        lows = conf_int_low[ep, :][conf_int_low[ep, :] != np.inf]
                        sz = len(lows)
                        x_values = np.linspace(1, sz, sz, endpoint=True) * self.aleat_sub_size
                        conf_int_low_fig.plot(x_values, lows)

                    conf_int_low_fig.set_xlabel('number of evaluations')
                    conf_int_low_fig.set_ylabel('low confidence interval')
                    conf_int_low_fig.set_ylim(low_lim_lower, low_lim_upper)
                    fig_low.savefig(fig_file_low, dpi=1200, bbox_inches='tight')
                    fig_low.clf()

                if handle_cih_file:
                    for ep in range(self.epist_samps):
                        highs = conf_int_high[ep, :][conf_int_high[ep, :] != np.inf]
                        sz = len(highs)
                        x_values = np.linspace(1, sz, sz, endpoint=True) * self.aleat_sub_size
                        conf_int_high_fig.plot(x_values, highs)

                    conf_int_high_fig.set_xlabel('number of evaluations')
                    conf_int_high_fig.set_ylabel('high confidence interval')
                    conf_int_high_fig.set_ylim(high_lim_lower, high_lim_upper)
                    fig_high.savefig(fig_file_high, dpi=1200, bbox_inches='tight')
                    fig_high.clf()

        else:
            out_msg = (
                f'{self.total_samps} were used to resample the model.\n'
            )

            zero = 0

            if self.verbose:

                if is_manager:
                    print(
                        f'Defaulting to {float(self.aleat_samps / 1000):.1f}k '
                        'aleatory samples\n'
                    )

                self.var_basis_resamp = evaluate_points_verbose(
                    var_basis_vect_func, zero, self.total_samps,
                    self.var_list, 'resample'
                )

            else:
                self.var_basis_resamp = evaluate_points(
                    var_basis_vect_func, zero, self.total_samps,
                    self.var_list, 'resample'
                )

            self.eval_resps = np.matmul(
                self.var_basis_resamp, self.matrix_coeffs
            )

        comm.Bcast([self.eval_resps, MPI_DOUBLE], root=0)
        comm.Bcast([self.var_basis_resamp, MPI_DOUBLE], root=0)

        if self.verbose and is_manager:
            print(out_msg)

        return self.eval_resps, out_msg

    def generate(self, eval_resps, sig, graph_dir):
        """
        Inputs: eval_resps- the evaluated responses from the pbox curves
                sig- the significance
                graph_dir- the name of the graph directory
        
        Generates the pbox plots from the eval_resps.
        """
        comm = MPI_COMM_WORLD
        rank = comm.rank

        conf_int_low = None
        conf_int_high = None

        idx = 0
        if rank == 0:
            if self.epist_var_count == 0:
                self.epist_samps = 1

            conf_int_low = np.ones(self.epist_samps) * np.inf
            conf_int_high = np.ones(self.epist_samps) * -np.inf

            for i in range(self.epist_samps):
                try:

                    eval_part = (
                        eval_resps[i * self.aleat_samps:(i + 1) * self.aleat_samps]
                    )

                    finite_eval = eval_part != np.inf
                    data = eval_part[finite_eval]
                    iter_size = len(data)
                    y_vals = np.linspace(0, 1, iter_size)

                    if not hasattr(self, 'mean_uncert'):
                        data = np.sort(data)

                        if self.plot:
                            plt.plot(data, y_vals)

                        conf_int_low[i] = np.interp(sig / 2, y_vals, data)
                        conf_int_high[i] = np.interp(1 - sig / 2, y_vals, data)

                    else:
                        idx_end = idx + self.aleat_samps
                        low = np.sort(data - self.mean_uncert[idx:idx_end][finite_eval])
                        high = np.sort(data + self.mean_uncert[idx:idx_end][finite_eval])

                        if self.plot:
                            color = np.random.uniform(size=3)
                            plt.plot(low, y_vals, high, y_vals, c=color, lw=0.75)

                        conf_int_low[i] = np.interp(sig / 2, y_vals, low)
                        conf_int_high[i] = np.interp(1 - sig / 2, y_vals, high)

                        idx += self.aleat_samps

                except ValueError:  # if the values converged and a
                    pass  # set of curves are arrays of zeros

            conf_int_low = np.min(conf_int_low)
            conf_int_high = np.max(conf_int_high)

            if self.verbose:
                print(
                    f'{1 - sig:.1%} Confidence Interval on Response '
                    f'[{conf_int_low:.5} , {conf_int_high:.5}]\n\nGenerating p-box '
                    'plot\n'
                )

            if self.plot:
                plt.title('Probability Box')
                plt.xlabel('generated response')
                plt.ylabel('cumulative probability')
                plt.savefig(f'{graph_dir}/p-box', dpi=1200, bbox_inches='tight')
                plt.clf()

        conf_int_low = comm.bcast(conf_int_low, root=0)
        conf_int_high = comm.bcast(conf_int_high, root=0)

        return conf_int_low, conf_int_high

    def calc_mean_conf_int(self, var_basis_sys_eval, responses, signif):
        """
        Inputs: var_basis_sys_eval- the evaluated variable basis
                responses- the matrix of responses
                signif- the level of significance of the model
        
        Calculates the confidence interval for each point in the 
        ProbabilityBoxes.
        """

        approx_mean, self.mean_uncert = calc_mean_conf_int(
            var_basis_sys_eval, self.matrix_coeffs, responses, signif,
            self.var_basis_resamp
        )

        return approx_mean, self.mean_uncert
