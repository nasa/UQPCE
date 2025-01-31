
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols
from sympy.utilities.lambdify import lambdify

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

from uqpce.pce.enums import UncertaintyType
from uqpce.pce._helpers import evaluate_points, evaluate_points_verbose, warn
from uqpce.pce.stats.statistics import calc_mean_conf_int
from uqpce.pce.variables.continuous import ContinuousVariable


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
        'var_list_symb', 'var_list', 'var_count',
        'epist_var_count', 'aleat_var_count', 'input', 'verbose', 'plot', 
        'track_conv_off', 'epist_samps', 'aleat_samps', 'total_samps', 
        'eval_resps', 'matrix_coeffs', 'var_basis_resamp', 'aleat_sub_size', 
        'epist_sub_size', 'mean_uncert', 'rank', 'size', 'aleat_resample', 
        'epist_resample', 'aleat_list_symb', 'epist_list_symb'
    )

    def __init__(
            self, var_list, verbose=False, plot=False, track_conv_off=False,
            epist_samps=125, aleat_samps=25000, aleat_sub_size=5000,
            epist_sub_size=25
        ):

        self.rank = rank
        self.size = size

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

        self.var_list_symb = np.array(
            [symbols(f'x{j}') for j in range(self.var_count)]
        )

        self._count_var_types()

        if self.epist_var_count == 0:
            self.epist_samps = 1

        if self.aleat_var_count == 0:
            self.aleat_samps = 1

        self.total_samps = self.epist_samps * self.aleat_samps

        self._generate_aleatory_samples()
        self._generate_epistemic_samples()

    def _count_var_types(self):
        """
        Counts the number of epistemic variables present.
        """
        self.epist_var_count = 0
        self.aleat_list_symb = []
        self.epist_list_symb = []

        for i in range(self.var_count):
            curr_var = self.var_list[i]

            if curr_var.type is UncertaintyType.EPISTEMIC:
                self.epist_var_count += 1
                self.epist_list_symb.append(self.var_list_symb[i])
            else:
                self.aleat_list_symb.append(self.var_list_symb[i])
    
        self.aleat_var_count = len(self.var_list) - self.epist_var_count

    def _generate_aleatory_samples(self):
        """
        Creates resample values for aleatory variables
        """
        if self.aleat_var_count == 0:
            return
        
        if self.rank == 0 and self.verbose:
            print('Generating aleatory resampling values\n')

        self.aleat_resample = np.zeros([self.aleat_samps, self.aleat_var_count])
        decr = 0

        for i in range(self.var_count):
            curr_var = self.var_list[i]

            if curr_var.type is UncertaintyType.ALEATORY:
                self.aleat_resample[:,i-decr] = curr_var.resample(
                    self.aleat_samps
                )
            else:
                decr += 1

    def _generate_epistemic_samples(self):
        """
        Determines the number of samples needed and generates the resample 
        values for epistemic variables.
        """
        if self.epist_var_count == 0:
            return
        
        if self.rank == 0 and self.verbose:
            print('Generating epistemic resampling values\n')

        self.epist_resample = np.zeros([self.epist_samps, self.epist_var_count])
        decr = 0

        for i in range(self.var_count):
            curr_var = self.var_list[i]
            
            if curr_var.type is UncertaintyType.EPISTEMIC:
                self.epist_resample[:,i-decr] = curr_var.resample(
                    self.epist_samps
                )
            else:
                decr += 1

    def evaluate_surrogate(
            self, var_basis_vect_symb, sig, matrix_coeffs, conv_thresh,
            graph_dir=None, **kwargs
        ):
        """
        Inputs: var_basis_vect_symb- symbolic variable basis (psi) vector
                sig- the level of significance
                matrix_coeffs- the coefficients from solving the matrix (psi)
                system of equations
                conv_thresh- the convergence threshold
                graph_dir- name of graph directory

        Resamples to generate responses from the model using various variable
        inputs.
        """
        mod_cnt = matrix_coeffs.shape[1]
        coeff_cnt = len(matrix_coeffs)

        out_msg = [''] * mod_cnt
        converged = False

        # Threshold is based on mean and input convergence threshold
        thresh = (matrix_coeffs[0] * conv_thresh)

        # The percents to interpolate for the confidence interval
        low_bnd = sig / 2
        high_bnd = 1 - low_bnd

        cils = np.zeros([3, mod_cnt])
        cihs = np.zeros([3, mod_cnt])

        if 'complex' in kwargs.keys() and kwargs['complex']:
            self.var_basis_resamp = (
                np.ones([self.total_samps, coeff_cnt], dtype=complex) * np.inf
            )
            self.eval_resps = (
                np.ones([self.total_samps, mod_cnt], dtype=complex) * np.inf
            )
        else:
            self.var_basis_resamp = (
                np.ones([self.total_samps, coeff_cnt]) * np.inf
            )
            self.eval_resps = (
                np.ones([self.total_samps, mod_cnt]) * np.inf
            )

        #--------------------------   create pbox curve(s)    --------------------------
        for ep in range(self.epist_samps):  # number of pbox curves to make
            beg_var_idx = ep * self.aleat_samps
            end_var_idx = (ep+1) * self.aleat_samps

            new_eq = var_basis_vect_symb.subs(
                {self.epist_list_symb[e]:self.epist_resample[ep,e] for e in 
                range(self.epist_var_count)}
            )
            var_basis_vect_func = lambdify(
                (self.aleat_list_symb,), new_eq, modules='numpy'
            )

            self.var_basis_resamp[beg_var_idx:end_var_idx, :] = evaluate_points(
                var_basis_vect_func, self.aleat_resample
            )

            self.eval_resps[beg_var_idx:end_var_idx, :] = np.matmul(
                self.var_basis_resamp[beg_var_idx:end_var_idx, :],
                matrix_coeffs
            )

        if not self.track_conv_off:
            if self.epist_samps == 1:
                cnt = self.aleat_sub_size
                cils[0,:] = np.quantile(
                    self.eval_resps[:-2*cnt], low_bnd, axis=0
                ).reshape(1,-1)
                cils[1,:] = np.quantile(
                    self.eval_resps[:-cnt], low_bnd, axis=0
                ).reshape(1,-1)
                cils[2,:] = np.quantile(
                    self.eval_resps, low_bnd, axis=0
                ).reshape(1,-1)

                cihs[0,:] = np.quantile(
                    self.eval_resps[:-2*cnt], high_bnd, axis=0
                ).reshape(1,-1)
                cihs[1,:] = np.quantile(
                    self.eval_resps[:-cnt], high_bnd, axis=0
                ).reshape(1,-1)
                cihs[2,:] = np.quantile(
                    self.eval_resps, high_bnd, axis=0
                ).reshape(1,-1)

            else:
                cnt = self.epist_sub_size
                cils[0,:] = np.quantile(
                    self.eval_resps[:-2*cnt,:], low_bnd, axis=0
                ).reshape(1,-1)
                cils[1,:] = np.quantile(
                    self.eval_resps[:-cnt], low_bnd, axis=0
                ).reshape(1,-1)
                cils[2,:] = np.quantile(
                    self.eval_resps, low_bnd, axis=0
                ).reshape(1,-1)
                
                cihs[0,:] = np.quantile(
                    self.eval_resps[:-2*cnt,:], high_bnd, axis=0
                ).reshape(1,-1)
                cihs[1,:] = np.quantile(
                    self.eval_resps[:-cnt], high_bnd, axis=0
                ).reshape(1,-1)
                cihs[2,:] = np.quantile(
                    self.eval_resps, high_bnd, axis=0
                ).reshape(1,-1)

        if not self.track_conv_off:
            converged = (
                (np.abs(np.diff(cils, axis=0)) < thresh).all(axis=0) 
                * (np.abs(np.diff(cihs, axis=0)) < thresh).all(axis=0)
            )
            for i in range(mod_cnt):
                if converged[i]:
                    out_msg[i] = 'The probability curves have converged.\n'
                else:
                    out_msg[i] = 'The probability curves did not converge.\n'

        return self.eval_resps, out_msg


    def generate(self, eval_resps, sig, graph_dir, **kwargs):
        """
        Inputs: eval_resps- the evaluated responses from the pbox curves
                sig- the significance
                graph_dir- the name of the graph directory
        
        Generates the pbox plots from the eval_resps.
        """
        model_cnt = eval_resps.shape[1]

        if 'complex' in kwargs.keys() and kwargs['complex']:
            conf_int_low = np.ones([self.epist_samps, model_cnt], dtype=complex) * np.inf
            conf_int_high = np.ones([self.epist_samps, model_cnt], dtype=complex) * -np.inf
        else:
            conf_int_low = np.ones([self.epist_samps, model_cnt]) * np.inf
            conf_int_high = np.ones([self.epist_samps, model_cnt]) * -np.inf
        
        qs = [sig/2, 1-sig/2]

        for i in range(self.epist_samps):

            try:
                eval_part = (
                    eval_resps[i * self.aleat_samps:(i + 1) * self.aleat_samps]
                )

                # Only keep finite part; unevaluated part is left as infinite
                data = eval_part[(eval_part[:,0] != np.inf)]

                if self.plot:
                    ys = np.linspace(0,1,len(data))
                    for p in range(model_cnt):
                        plt.figure(p)
                        plt.plot(np.sort(data[:,p]), ys, '-')

                conf_int_low[i,:], conf_int_high[i,:] = np.quantile(data, qs, axis=0)

            except:
                pass

        conf_int_low = np.min(conf_int_low, axis=0)
        conf_int_high = np.max(conf_int_high, axis=0)

        if self.verbose and self.plot:
            print('Generating p-box plot\n')

        if self.plot:
            for p in range(model_cnt):
                plt.figure(p)
                plt.title('Probability Box')
                plt.xlabel('resampled response')
                plt.ylabel('cumulative probability')
                plt.savefig(f'{graph_dir[p]}/p-box', dpi=1200, bbox_inches='tight')
                plt.clf()

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
