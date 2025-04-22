import sys
import os
import copy
from datetime import datetime
import pickle

try:
    from mpi4py.MPI import COMM_WORLD as MPI_COMM_WORLD
    comm = MPI_COMM_WORLD
    rank = comm.rank
    size = comm.size
    is_manager = (rank == 0)
except:
    comm = None
    rank = 0
    size = 1
    is_manager = True

import matplotlib.pyplot as plt
import numpy as np

from uqpce.pce.variables.variable import Variable
from uqpce.pce.variables.continuous import (
    NormalVariable, UniformVariable, BetaVariable, ExponentialVariable, 
    GammaVariable, LogNormalVariable, EpistemicVariable
)
from uqpce.pce.variables.discrete import (
    DiscreteVariable, PoissonVariable, NegativeBinomialVariable, 
    HypergeometricVariable, UniformVariable as DiscUniformVariable, 
    EpistemicVariable as DiscEpistemicVariable
)
from uqpce.pce._helpers import check_directory, term_count
from uqpce.pce.stats.statistics import (
    calc_coeff_conf_int, get_sobol_bounds, calc_var_conf_int, calc_R_sq, 
    calc_R_sq_adj
)

class PCE():
    """
    Parameters
    ----------
    kwargs :
        key word arguments used by UQPCE

        input_file :
            file containing variables
        matrix_file :
            file containing matrix elements
        results_file  :
            file containing results 
        verification_results_file : 
            file containing verification results
        verification_matrix_file : 
            file containing verification matrix elements
        output_directory : 
            directory that the outputs will be put in
        case : 
            case name of input data
        significance : 
            significance level of the confidence interval
        order : 
            order of polynomial chaos expansion
        over_samp_ratio : 
            over sampling ratio; factor for how many points to be used in 
            calculations
        verify_over_samp_ratio : 
            over sampling ratio for verification; factor for how many points to be 
            used in calculations
        aleat_sub_samp_size : 
            the number of samples to check the new high and low intervals at for 
            each individual curve
        epist_sub_samp_size : 
            the number of curves to check the new high and low intervals at for a 
            set of curves
        conv_threshold_percent : 
            the percent of the response mean to be used as a threshold for tracking 
            convergence
        epist_samp_size : 
            the number of times to sample for each variable with epistemic 
            uncertainty
        aleat_samp_size : 
            the number of times to sample for each variable with aleatory 
            uncertainty
        version : 
            displays the version of the software
        verbose : 
            increase output verbosity
        verify : 
            allows verification of results
        plot : 
            generates factor vs response plots, pbox plot, and error plots
        plot_stand : 
            plots standardized variables
        track_convergence_off : 
            allows users to converge on confidence interval until the change between 
            the two iters is less than the threshold
        model_conf_int : 
            includes uncertainties associated with the model itself
        stats : 
            perform additional statistics for a more-comprehensive profile 
            of the model
        seed : 
            if UQPCE should use a seed for random values
        
    The Polynomial Chaos Exapansion (PCE) model class. This class is intended to 
    be an interface for programmers.
    """

    def __init__(self, **kwargs):

        self._is_manager = is_manager

        self.verbose = True
        self.symbolic = False
        self.alpha = 0.05

        self.input_file = None

        self.out_file = 'output.dat'
        self.sobol_file = 'sobol.dat'
        self.coeff_file = 'coefficients.dat'

        self.output_directory = 'outputs'
        self._output_base_directory = 'outputs'
        self.graph_directory = 'graphs'
        self._graph_base_directory = 'graphs'

        self.version_num = '1.0.0'
        self.backend = 'TkAgg'

        self.verbose = False
        self.version = False
        self.seed = False
        self.plot = False
        self.plot_stand = False
        self.outputs = True # Boolean for if output files should be saved
        self.model_conf_int = False # Provide CIs on coeffs, Sobols, etc
        self.stats = False

        # track and plot confidence interval unless turned off (True)
        self.track_convergence_off = False
        self.epist_sub_samp_size = 25
        self.aleat_sub_samp_size = 5000

        self.order = 2  # order of PCE expansion
        self.over_samp_ratio = 2
        self.verify_over_samp_ratio = 0.5
        self.adapt_samp_terms = 4
        self.significance = 0.05
        self.conv_threshold_percent = 0.0005
        self.sigfigs = 5
        self.alpha_out = 0.15

        # number of times to iterate for each variable type
        self.epist_samp_size = 125
        self.aleat_samp_size = 25000

        # Attributes needed for class
        self.variables = np.array([], dtype=Variable)
        self._var_count = 0
        self._is_built = False # PCE model has been built or not
        self._bin_const = 3 # Divide by this for the number of hist bins
        self.time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.time_start = datetime.strptime(self.time_start, '%Y-%m-%d %H:%M:%S.%f')
        self.case = None
        self._conf_int_str = ['']
        self.verify_str = ['']
        self._stats_str = ['']
        self._sobol_str = ['']
        self._output_str = ['']
        self._mod_str = ['']
        self._sobol_bounds = False
        self._coeff_uncert = False

        self.update_settings(**kwargs)

    def update_settings(self, **kwargs) -> None:
        """
        Parameters
        ----------
        kwargs : 
            The key-value pairs for the keyword arguments to update for the
            PCE model

        A method to update the object attributes with the input kwargs.
        """
        for key, val in kwargs.items():
            setattr(self, key, val)

            if key == 'output_directory':
                setattr(self, '_output_base_directory', val)

        if self.seed:
            np.random.seed(0)

    def _build_directory(self, output_directory: str, graph_directory: str):
        """
        A method to check if a directory exists, increment it if it does, and 
        create the directory structure for the outputs.
        """
        output_directory = check_directory(
            output_directory, verbose=self.verbose
        )

        # Update the name of the graph dir incase the output dir was incremented
        graph_directory = os.path.join(
            output_directory, os.path.split(graph_directory)[1]
        )

        graph_directory = check_directory(
            graph_directory, verbose=self.verbose
        )

        return output_directory, graph_directory


    def _add_variable(self, variable: Variable) -> None:
        """
        Parameters
        ----------
        variable : 
            A Variable object to add to the model

        Method to add a variable to the model and increment the PCE object's 
        variable count.
        """
        if self.verbose:
            print(f'Adding variable number {self._var_count+1}\n')

        self.variables = np.append(self.variables, variable)
        self._var_count += 1


    def add_variable(self, **kwargs) -> None:
        """
        Parameters
        ----------
        **kwargs : 
            The input arguments required for the variable being added and the 
            name of the distribution being added.
        
        Adds a variable with **kwargs to the PCE object.
        """
        from uqpce.pce.enums import Distribution, UncertaintyType
        from uqpce.pce.variables.continuous import (
            BetaVariable, ExponentialVariable, GammaVariable, NormalVariable,
            UniformVariable, ContinuousVariable, EpistemicVariable
        )
        from uqpce.pce.variables.discrete import (
            UniformVariable as DiscUniformVariable, DiscreteVariable,
            NegativeBinomialVariable, PoissonVariable, HypergeometricVariable, 
            EpistemicVariable as DiscreteEpistemicVariable
        )

        distribution = kwargs.pop('distribution')

        # For backwards compatability
        type_exists = 'type' in kwargs
        if type_exists:
            unc_type = UncertaintyType[kwargs.pop('type').upper()]

        # Get the enum of the variable
        dist = Distribution[distribution.upper()]

        # Setting the order for variables; default is order of class
        curr_order = kwargs.get('order', self.order)
        if 'order' in kwargs:
            kwargs.pop('order')

        if dist is Distribution.NORMAL:
            req_1 = 'mean'
            req_2 = 'stdev'
            try:
                mean = kwargs.pop(req_1)
                stdev = kwargs.pop(req_2)
            except:
                print(
                    f'Key word arguments `{req_1}` and `{req_2}` are required '
                    f'inputs for the {distribution} variable.', file=sys.stderr
                )
            
            var = NormalVariable(
                mean=mean, stdev=stdev, number=self._var_count, order=curr_order, 
                **kwargs
            )

        elif dist is Distribution.UNIFORM:
            if (not type_exists) or (type_exists and unc_type is UncertaintyType.ALEATORY):
                req_1 = 'interval_low'
                req_2 = 'interval_high'
                try:
                    interval_low = kwargs.pop(req_1)
                    interval_high = kwargs.pop(req_2)
                except:
                    print(
                        f'Key word arguments `{req_1}` and `{req_2}` are required '
                        f'inputs for the {distribution} variable.', file=sys.stderr
                    )
                var = UniformVariable(
                    interval_low, interval_high, number=self._var_count, 
                    order=curr_order, **kwargs
                )

            if type_exists and unc_type is UncertaintyType.EPISTEMIC:
                req_1 = 'interval_low'
                req_2 = 'interval_high'
                try:
                    interval_low = kwargs.pop(req_1)
                    interval_high = kwargs.pop(req_2)
                except:
                    print(
                        f'Key word arguments `{req_1}` and `{req_2}` are required '
                        f'inputs for the {distribution} variable.', file=sys.stderr
                    )
                var = EpistemicVariable(
                    interval_low, interval_high, number=self._var_count, 
                    order=curr_order, **kwargs
                )            

        elif dist is Distribution.BETA:
            req_1 = 'alpha'
            req_2 = 'beta'
            try:
                alpha = kwargs.pop(req_1)
                beta = kwargs.pop(req_2)
            except:
                print(
                    f'Key word arguments `{req_1}` and `{req_2}` are required '
                    f'inputs for the {distribution} variable.', file=sys.stderr
                )
            var = BetaVariable(
                alpha, beta, number=self._var_count, order=curr_order, **kwargs
            )

        elif dist is Distribution.EXPONENTIAL:
            req_1 = 'lambda'
            try:
                lambd = kwargs.pop(req_1)
            except:
                print(
                    f'Key word argument `{req_1}` is a required input for the '
                    f'{distribution} variable.', file=sys.stderr
                )
            var = ExponentialVariable(
                lambd, number=self._var_count, order=curr_order, **kwargs
            )

        elif dist is Distribution.GAMMA:
            req_1 = 'alpha'
            req_2 = 'theta'
            try:
                alpha = kwargs.pop(req_1)
                theta = kwargs.pop(req_2)
            except:
                print(
                    f'Key word arguments `{req_1}` and `{req_2}` are required '
                    f'inputs for the {distribution} variable.', file=sys.stderr
                )
            var = GammaVariable(
                alpha, theta, number=self._var_count, order=curr_order, **kwargs
            )

        elif dist is Distribution.LOGNORMAL:
            req_1 = 'mu'
            req_2 = 'stdev'
            try:
                mu = kwargs.pop(req_1)
                stdev = kwargs.pop(req_2)
            except:
                print(
                    f'Key word arguments `{req_1}` and `{req_2}` are required '
                    f'inputs for the {distribution} variable.', file=sys.stderr
                )
            var = LogNormalVariable(
                mu, stdev, number=self._var_count, order=curr_order, **kwargs
            )

        elif dist is Distribution.CONTINUOUS:
            req_1 = 'pdf'
            req_2 = 'interval_low'
            req_3 = 'interval_high'
            try:
                pdf = kwargs.pop(req_1)
                interval_low = kwargs.pop(req_2)
                interval_high= kwargs.pop(req_3)
            except:
                print(
                    f'Key word arguments `{req_1}`, `{req_2}`, and `{req_3}` '
                    f'are required inputs for the user-input continuous variable.', 
                    file=sys.stderr
                )
            var = ContinuousVariable(
                pdf, interval_low, interval_high, number=self._var_count, 
                order=curr_order, **kwargs
            )

        elif dist is Distribution.DISCRETE_UNIFORM:
            if (not type_exists) or (type_exists and unc_type is UncertaintyType.ALEATORY):
                req_1 = 'interval_low'
                req_2 = 'interval_high'
                try:
                    interval_low = kwargs.pop(req_1)
                    interval_high = kwargs.pop(req_2)
                except:
                    print(
                        f'Key word arguments `{req_1}` and `{req_2}` are required '
                        f'inputs for the {distribution} variable.', file=sys.stderr
                    )
                var = DiscUniformVariable(
                    interval_low, interval_high, number=self._var_count, 
                    order=curr_order, **kwargs
                )
            if type_exists and unc_type is UncertaintyType.EPISTEMIC:
                req_1 = 'interval_low'
                req_2 = 'interval_high'
                try:
                    interval_low = kwargs.pop(req_1)
                    interval_high = kwargs.pop(req_2)
                except:
                    print(
                        f'Key word arguments `{req_1}` and `{req_2}` are required '
                        f'inputs for the {distribution} variable.', file=sys.stderr
                    )
                var = DiscreteEpistemicVariable(
                    interval_low, interval_high, number=self._var_count, 
                    order=curr_order, **kwargs
                )

        elif dist is Distribution.NEGATIVE_BINOMIAL:
            req_1 = 'r'
            req_2 = 'p'
            try:
                r = kwargs.pop(req_1)
                p = kwargs.pop(req_2)
            except:
                print(
                    f'Key word arguments `{req_1}` and `{req_2}` are required '
                    f'inputs for the {distribution} variable.', file=sys.stderr
                )
            var = NegativeBinomialVariable(
                r, p, number=self._var_count, order=curr_order, **kwargs
            )

        elif dist is Distribution.POISSON:
            req_1 = 'lambda'
            try:
                lambd = kwargs.pop(req_1)
            except:
                print(
                    f'Key word argument `{req_1}` is a required input for the '
                    f'{distribution} variable.', file=sys.stderr
                )
            var = PoissonVariable(
                lambd, number=self._var_count, order=curr_order, **kwargs
            )

        elif dist is Distribution.HYPERGEOMETRIC:
            req_1 = 'M'
            req_2 = 'n'
            req_3 = 'N'
            try:
                M = kwargs.pop(req_1)
                n = kwargs.pop(req_2)
                N = kwargs.pop(req_3)
            except:
                print(
                    f'Key word arguments `{req_1}`, `{req_2}`, and {req_3} are '
                    f'required inputs for the {distribution} variable.', 
                    file=sys.stderr
                )
            var = HypergeometricVariable(
                M, n, N, number=self._var_count, order=curr_order, **kwargs
            )

        elif dist is Distribution.DISCRETE:
            req_1 = 'pdf'
            req_2 = 'interval_low'
            req_3 = 'interval_high'
            try:
                pdf = kwargs.pop(req_1)
                interval_low = kwargs.pop(req_2)
                interval_high= kwargs.pop(req_3)
            except:
                print(
                    f'Key word arguments `{req_1}`, `{req_2}`, and `{req_3}` '
                    f'are required inputs for the user-input discrete variable.', 
                    file=sys.stderr
                )
            var = DiscreteVariable(
                pdf, interval_low, interval_high, number=self._var_count, 
                order=curr_order, **kwargs
            )

        self._add_variable(var)


    def from_yaml(self, input_file: str) -> dict:
        """
        Parameters
        ----------
        input_file : 
            A string for the file name of the input yaml file

        Update the PCE object from the UQPCE YAML file. Adds the variables to 
        the object and updates the settings.
        """
        from uqpce.pce.io import read_input_file
        var_dict, settings_dict = read_input_file(input_file)
        self.input_file = input_file

        self.update_settings(**settings_dict)

        for key, val in var_dict.items():
            self.add_variable(**val)

        return settings_dict


    def load_matrix_file(self, filename: str) -> np.ndarray:
        """
        Parameters
        ----------
        filename : 
            A string for the file name of the run matrix file

        Loads and returns a matrix file.
        """
        return np.loadtxt(filename, ndmin=2)


    def set_samples(self, X: np.ndarray) -> None:
        """
        Sets the run matrix samples for the PCE model.
        """
        self._X = np.copy(np.atleast_2d(X))

        self._X_stand = np.zeros(X.shape)
        for i in range(self._var_count):
            self._X_stand[:,i] = self.variables[i].standardize_points(
                self._X[:,i]
            )

    def build_basis(self, order: int) -> None:
        """
        Parameters
        ----------
        order : 
            An int for the order of the model to build

        Builds the variable basis and norm squared for the model. This sets up 
        the variable- and order- based information for the model that is 
        independent of the response samples.
        """
        from uqpce.pce.model import MatrixSystem

        if not hasattr(self, '_y'):
            terms = term_count(self._var_count, order) + 1
            self._y = np.zeros(terms)

        if not hasattr(self, '_X'):
            self._X = np.zeros([terms, self._var_count])
            self._X_stand = np.zeros([terms, self._var_count])

        self._matrix = MatrixSystem(self._y, self.variables, verbose=self.verbose)

        if self._is_manager and self.verbose:
            print('Constructing surrogate model\n\nBuilding norm-squared matrix\n')
        self._matrix.create_model_matrix()
        self._matrix.form_norm_sq(order)
        if self._is_manager and self.verbose:
            print('Assembling psi matrix\n')
        self._matrix.build()
        if self._is_manager and self.verbose:
            print('Psi matrix assembled\n')
            
        if self._is_manager and self.verbose:
            print('Evaluating psi matrix\n')
        self._matrix.evaluate(self._X_stand)

        self.norm_sq = np.copy(self._matrix.norm_sq)
        self.var_basis = np.copy(self._matrix.var_basis_sys_eval)

    def _build_matrix(self, order: int=None) -> None:
        self.build_basis(order)

        self._matrix.solve()
        self.matrix_coeffs = np.copy(self._matrix.matrix_coeffs)

    def _build_model(self) -> None:

        from uqpce.pce.model import SurrogateModel
        from uqpce.pce._helpers import calc_mean_err

        self._model = SurrogateModel(
            self._y, self._matrix.matrix_coeffs, verbose=self.verbose
        )

        if self._is_manager and self.verbose:
            print('Surrogate model construction complete\n')

        try:
            resp_iter_count = len(self._y)
            over_samp_ratio = resp_iter_count / self._matrix.min_model_size

            if over_samp_ratio < 1.25:  # warn if low over_samp_ratio
                print(
                    f'The oversampling ratio is {over_samp_ratio:.{self.sigfigs}}. Consider '
                    f'using at least {int(np.ceil(1.25 * self._matrix.min_model_size))} '
                    'samples for a more accurate model.', file=sys.stderr
                )
        except TypeError:
            over_samp_ratio = 0
            resp_iter_count = 0
            pass

        self._mean = self._matrix.matrix_coeffs[0]
        self._sigma_sq = self._model.calc_var(self._matrix.norm_sq)
        self._error, pred = self._model.calc_error(self._matrix.var_basis_sys_eval)
        self._err_mean = calc_mean_err(self._error)
        self._signal_to_noise = self._sigma_sq / self._err_mean
        self._mean_sq_error, hat_matrix, self._shapiro = self._model.check_normality(
            self._matrix.var_basis_sys_eval, self.significance, 
            self.graph_directory, sigfigs=self.sigfigs, plot=self.plot
        )

        if self.plot:
            self._figures(self._error, pred)

        if self.model_conf_int:
            self._coeff_uncert = (
                calc_coeff_conf_int(
                    self._matrix.var_basis_sys_eval, self._matrix.matrix_coeffs, 
                    self._y, self.significance
                )
            )
            self._sigma_sq_low, self._sigma_sq_high = calc_var_conf_int(
                self._matrix.matrix_coeffs, self._coeff_uncert, 
                self._matrix.norm_sq
            )

            for mod in range(self.model_cnt):
                self._conf_int_str[mod] = (
                    f'{1-self.significance:.1%} Confidence Interval on the mean [{self._mean[mod] - self._coeff_uncert[mod, 0]:.5}, '
                    f'{self._mean[mod] + self._coeff_uncert[mod, 0]:.5}]\n'
                    f'{1-self.significance:.1%} Confidence Interval on the variance '
                    f'[{self._sigma_sq_low:.5}, {self._sigma_sq_high:.5}]\n'
                )


    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : 
            An m-by-n matrix with the first dimension having the number of 
            samples in the model (m) and the second having the number of 
            variables in the model (n).

        y : 
            The 2D numpy array of responses from the user's analytic tool. The 

        Fits the PCE model and returns the matrix coefficients of the model.
        """
        self.set_samples(X)

        self._y = np.copy(np.atleast_2d(y)).reshape(self._X.shape[0], -1)
        self.model_cnt = self._y.shape[1]

        self._stats_str = [''] * self.model_cnt
        self._conf_int_str = [''] * self.model_cnt
        self._sobol_str = [''] * self.model_cnt
        self._output_str = [''] * self.model_cnt
        self.verify_str = [''] * self.model_cnt
        self._mod_str = [''] * self.model_cnt

        if self.outputs:
            self.output_directory = np.zeros(self.model_cnt, dtype=object)
            self.graph_directory = np.zeros(self.model_cnt, dtype=object)
            self._output_base_directory = check_directory(
                self._output_base_directory, verbose=False
            )
            if self.model_cnt > 1:
                for i in range(self.model_cnt):
                    self.output_directory[i] = os.path.join(self._output_base_directory, f'response_{i}')
                    self.output_directory[i] = self._build_directory(self.output_directory[i], self._graph_base_directory)[0]
                    self.graph_directory[i] = os.path.join(self.output_directory[i], self._graph_base_directory)
            else:
                self.output_directory[0] = self._output_base_directory
                self.graph_directory[0] = check_directory(
                    os.path.join(self._output_base_directory, self._graph_base_directory), 
                    verbose=self.verbose
            )

        X = np.atleast_2d(X)
        if X.shape[0] == 1: # only one uncertain variable
            X = X.T # transpose to ensure the responses are in correct dimension
        
        # This can be expensive; ensure that this is only done once
        if not self._is_built:
            self._is_build = True
        
        self._build_matrix(self.order)
        self._build_model()

        for i in range(self.model_cnt):
            self._mod_str[i] = (
                f'Mean of response {self._mean[i]:.5}\nVariance of response'
                f' {self._sigma_sq[i]:.5}\nMean error of surrogate '
                f'{self._err_mean[i]:.5}\nSignal to noise ratio '
                f'{self._signal_to_noise[i]:.5}\n'
            )

        return self._model.matrix_coeffs

    def print(self) -> None:
        """
        Print the outputs for the model.
        """

        for i in range(self.model_cnt):

            print(self._shapiro[i])

            print(self._sobol_str[i])

            print(self._mod_str[i])

            print(self._conf_int_str[i])

    def _figures(self, error:np.ndarray, pred:np.ndarray) -> None:
        from uqpce.pce.graphs import Graphs
        if self.plot_stand:
            X = self._X_stand
        else:
            X = self._X

        self._graph = Graphs(self.plot_stand)
        for i in range(self.model_cnt):
            self._graph.factor_plots(self.graph_directory[i], self.variables, pred[:,i], X, 'Predicted')
            self._graph.factor_plots(self.graph_directory[i], self.variables, error[:,i], X, 'Error')
            self._graph.error_vs_pred(self.graph_directory[i], error[:,i], pred[:,i], 'Error vs Predicted')

    def sobols(self) -> str:
        from uqpce.pce._helpers import create_total_sobols

        self._sobols = self._model.get_sobols(self._matrix.norm_sq)
        tot_sobol = create_total_sobols(
            self._var_count, self._matrix.model_matrix, self._sobols
        )

        sobol_sum = np.sum(tot_sobol, axis=0)

        for m in range(self.model_cnt):
            unsc = 'Total Sobols\n'
            resc = 'Rescaled Total Sobols\n'
            for i in range(self._var_count):
                var_name = self.variables[i].name 
                unsc = ''.join((
                    unsc, 
                    f'   Total Sobol {var_name} = {tot_sobol[i, m]:.5}\n'
                ))
                resc = ''.join((
                    resc, 
                    f'   Total Sobol {var_name} = {tot_sobol[i, m]/sobol_sum[m]:.5}\n'
                ))

            self._sobol_str[m] = unsc + resc

        if self.model_conf_int:
            low_sobol, high_sobol = get_sobol_bounds(
                self._matrix.matrix_coeffs, self._sobols, self._coeff_uncert, 
                self._matrix.norm_sq
            )

            self._sobol_bounds = {'low': low_sobol, 'high':high_sobol}

        return self._sobol_str

    def predict(self, X: np.ndarray, return_uncert: bool=False) -> np.ndarray:
        """
        Parameters
        ----------
        X : 
            An m-by-n matrix with the first dimension having the number of 
            samples in the model (m) and the second having the number of 
            variables in the model (n).

        return_uncert : 
            Boolean for if the method should return the uncertainty associated 
            with the predicted responses.

        Returns
        -------
        resp_pred : the predicted value
        uncert_mean : the uncertainty on the predicted value

        Predicts the model responses of an input matrix of values.
        """
        Xnew = np.zeros(X.shape)
        for var in range(self._var_count):
            Xnew[:,var] = self.variables[var].standardize_points(X[:,var])

        resp_pred, var_basis_sys_eval_pred = self._model.predict(
            self._matrix.var_basis_vect_symb, self._matrix.var_list_symb, Xnew
        )

        if return_uncert:
            from uqpce.pce.stats.statistics import calc_mean_conf_int
            approx_mean, uncert_mean = calc_mean_conf_int(
                self._matrix.var_basis_sys_eval, self._matrix.matrix_coeffs, 
                self._y, self.significance, var_basis_sys_eval_pred
            )

            # These values should be the same; this is a little sanity check
            assert np.isclose(approx_mean, resp_pred).all()

            return resp_pred, uncert_mean

        return resp_pred

    def verification(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : 
            An m-by-n matrix with the first dimension having the number of 
            samples in the model (m) and the second having the number of 
            variables in the model (n).

        y : 
            The 2D numpy array of responses from the user's analytic tool.

        Returns
        -------
        ver_error :
            The error between the truth and the predicted reponses.

        Predicts the response for the verification samples and compares the 
        predicted responses to the user-provided verification response file.
        """
        y = y.reshape(-1, self.model_cnt)
        pred_resp = self.predict(X)
        ver_error = pred_resp - y
        self.verify_str = [''] * self.model_cnt

        for i in range(self.model_cnt):
            if self.plot:

                fig = plt.figure('Verify')
                f, ax = plt.subplots(
                    1, 2, gridspec_kw={'width_ratios': [3, 1]}
                )

                ax[0].plot(self._y[:,i], self.error[:,i], 'bo', label='train')
                ax[0].plot(y[:,i], ver_error[:,i], 'ro', label='verify')

                f.suptitle('Verification Error')
                ax[0].set_xlabel('response')
                ax[0].set_ylabel('residual')

                vp1 = ax[1].violinplot(self.error[:,i], positions=[1])
                for vp in vp1['bodies']:
                    vp.set_facecolor('b')
                    vp.set_edgecolor('b')

                vp2 = ax[1].violinplot(ver_error[:,i], positions=[2])
                for vp in vp2['bodies']:
                    vp.set_facecolor('r')
                    vp.set_edgecolor('r')

                ax[1].set_xticks([])
                ax[1].set_yticks([])

                ax[0].legend()

                plt.savefig(
                    os.path.join(self.graph_directory[i], 'verify_err'), dpi=600, 
                    bbox_inches='tight'
                )
                plt.clf()

                self._graph.factor_plots(
                    self.graph_directory[i], self.variables, ver_error[:,i], X, 
                    'Verification Error', verify=True
                )

            mean_ver_err = np.mean(np.abs(ver_error[:,i]))
            self.verify_str[i] = f'Mean error between model and verification {mean_ver_err}\n'

        return ver_error

    def verify_sample(self, count: int=-1, seed=None) -> np.ndarray:
        """
        Sample the number of verification values according to the input count or 
        the set attribute verify_over_samp_ratio.
        """
        from uqpce.pce._helpers import term_count

        if count == -1:
            req_samps = term_count(self._var_count, self.order)
            count = int(np.ceil(self.verify_over_samp_ratio * req_samps))

        return self.sample(count=count, seed=seed)

    def sample(self, count: int=-1, seed=None) -> np.ndarray:
        """
        This method returns a matrix of samples from the input variables.
        """
        from scipy.stats.qmc import LatinHypercube
        from uqpce.pce.variables.continuous import ContinuousVariable
        from uqpce.pce.variables.discrete import DiscreteVariable
        from uqpce.pce._helpers import term_count

        if count == -1:
            req_samps = term_count(self._var_count, self.order)
            count = int(np.ceil(self.over_samp_ratio * req_samps))
            
        sampler = LatinHypercube(d=self._var_count, seed=seed)
        samps = sampler.random(n=count)

        for vidx in range(self._var_count):
            var = self.variables[vidx]
            vtype = type(var)

            if vtype is ContinuousVariable or vtype is DiscreteVariable:
                # The continuous and discrete user-input variables do not 
                # currently support reliable sampling from the cumulative 
                # density functions (CDFs)
                samps[:,vidx] = var.generate_samples(count)
            else:
                # All other variables support sampling from their CDF
                samps[:,vidx] = var.cdf_sample(samps[:,vidx])

        return samps

    def resample_surrogate(self) -> np.ndarray:
        """
        Resamples the surrogate model.
        """
        from uqpce.pce.pbox import ProbabilityBoxes

        if hasattr(self, '_model'):
            coeffs = self._model.matrix_coeffs
        else:
            terms = term_count(self._var_count, self.order)
            coeffs = np.zeros([terms, 1])

        self._pbox = ProbabilityBoxes(
            self.variables, verbose=self.verbose, plot=self.plot, 
            track_conv_off=self.track_convergence_off, 
            epist_samps=self.epist_samp_size, aleat_samps=self.aleat_samp_size, 
            aleat_sub_size=self.aleat_sub_samp_size, 
            epist_sub_size=self.epist_sub_samp_size
        )

        eval_resps, out_msg = self._pbox.evaluate_surrogate(
            self._matrix.var_basis_vect_symb, self.significance, 
            coeffs, self.conv_threshold_percent,
            graph_dir=self.graph_directory
        )

        self.resampled_var_basis = np.copy(self._pbox.var_basis_resamp)
        
        return eval_resps

    def confidence_interval(self) -> tuple:
        """
        Calls the ProbabilityBoxes class, resamples the model, and outputs the 
        confidence interval at the set significance level.
        """
        eval_resps = self.resample_surrogate()

        self._cil, self._cih = self._pbox.generate(
            eval_resps, self.significance, graph_dir=self.graph_directory
        )

        self.epist_samp_size = copy.copy(self._pbox.epist_samps)
        self.aleat_samp_size = copy.copy(self._pbox.aleat_samps)

        for i in range(self.model_cnt):
            self._mod_str[i] += (
                f'{1 - self.significance:.1%} Confidence Interval on Response '
                f'[{np.real(self._cil[i]):.5} , {np.real(self._cih[i]):.5}]\n'
            )

        return self._cil, self._cih

    def backward_elimination(self) -> None:
        """
        Peforms backward elemination on the existing model.
        """
        from uqpce.pce.stats.model import backward_elimination, find_lower_terms

        if self._is_manager and self.verbose:
            print('Performing backward elimination\n')

        combo = backward_elimination(
            self._matrix.var_basis_sys_eval, self._y, alpha_out=self.alpha_out
        )

        combo = sorted(combo)
        combo = find_lower_terms(self._matrix._model_matrix, combo, combo)

        if self._is_manager and self.verbose:
            print('Updating matrix attributes\n')

        # Update the model such that the mean, variance, etc are correct
        self._matrix.update(combo)
        self._build_model()

    def check_variables(self, X: np.ndarray) -> bool:
        """
        Saves figures for all of the variables with their values plotted on 
        them, if the data is available, which serves as a check if the user's 
        run matrix does not match their distributions well.
        """
        from uqpce.pce.variables.continuous import ContinuousVariable
        from uqpce.pce.variables.discrete import DiscreteVariable

        X = np.atleast_2d(X)

        for vidx in range(self._var_count):
            curr_var = self.variables[vidx]

            curr_var.check_distribution(X[:,vidx])

            if self.plot:
                plt.figure(f'Variable {curr_var.name} Distribution')
                plt.hist(X[:,vidx], density=True, label='Samples', bins=50)

                if (
                    (type(curr_var) is not ContinuousVariable) and 
                    (type(curr_var) is not DiscreteVariable)
                ):
                    x = np.linspace(
                        curr_var.bounds[0], 
                        curr_var.bounds[1], num=1000
                    )
                    if type(curr_var) is ContinuousVariable or type(curr_var) is DiscreteVariable:
                        # plt.plot(x, [curr_var.distribution.subs(curr_var.x, i) for i in x], 'k-', label='Analytical PDF')
                        pass
                    else:
                        curr_type = type(curr_var)
                        cont_vars = [
                            ContinuousVariable, NormalVariable, UniformVariable,
                            BetaVariable, ExponentialVariable, GammaVariable,
                            EpistemicVariable
                        ]
                        disc_vars = [
                            DiscreteVariable, PoissonVariable, DiscUniformVariable,
                            NegativeBinomialVariable, HypergeometricVariable, 
                            DiscEpistemicVariable
                        ]
                        if curr_type in cont_vars:
                            plt.plot(x, curr_var.dist.pdf(x), 'k-', label='Analytical PDF')
                        elif curr_type in disc_vars:
                            plt.plot(x, curr_var.dist.pmf(x), 'k-', label='Analytical PDF')

                    plt.title(f'Variable {curr_var.name} Distribution')
                    plt.xlabel('value')
                    plt.ylabel('Probability Density Function (PDF)')

                plt.legend()
                plt.savefig(
                    os.path.join(self._output_base_directory, f'{curr_var.name}').replace(':', '_'), 
                    dpi=600, bbox_inches='tight'
                )
                plt.clf()

        return True

    def check_fit(self) -> str:
        """
        Provides statistics on the fit of the model.
        """
        from uqpce.pce.stats.statistics import calc_R_sq, calc_R_sq_adj

        press_dict = self._matrix.get_press_stats()
        R_sq = calc_R_sq(
            self._matrix.var_basis_sys_eval, self._matrix.matrix_coeffs, self._y
        )
        R_sq_adj = calc_R_sq_adj(
            self._matrix.var_basis_sys_eval, self._matrix.matrix_coeffs, self._y
        )

        # unpacking press stats dict
        press_stat = press_dict['PRESS']

        shift = 5
        spaces = self.sigfigs + shift

        for m in range(self.model_cnt):
            self._stats_str[m] = (# formatting like this so 5 sig figs and aligned
                f'PRESS statistic:         {f"{float(press_stat[m]):.{self.sigfigs}}":{spaces}s}\n'
                f'R\u00b2:                      {f"{float(R_sq[m]):.{self.sigfigs}}":{spaces}s}\n'
                f'R\u00b2 adjusted:             {f"{float(R_sq_adj[m]):.{self.sigfigs}}":{spaces}s}\n\n'
            )

        return self._stats_str

    def generate_responses(self, X, equation: str) -> np.ndarray:
        """
        Parameters
        ----------
        X : 
            An m-by-n matrix with the first dimension having the number of 
            samples in the model (m) and the second having the number of 
            variables in the model (n).
        
        equation : 
            A string represenation of the desired equation, using x0 for the 
            first variable, x1 for the second, and so on.

        For testing purposes; allows users to generate samples according to 
        an input function.
        """
        from uqpce.pce._helpers import user_function
        from uqpce.pce.error import DimensionError

        shapex, shapey = X.shape

        if shapex <= shapey:
            raise DimensionError(
                'The first dimension of the matrix should be larger than the '
                'second dimension'
            )

        if self._is_manager and self.verbose:
            print(f'Generating the results from function {equation}\n')

        return user_function(equation, X)

    def save_model(self) -> bool:
        """
        Pickles the entire PCE object.
        """
        time = time.time()
        try:
            with open(f'uqpce_{time}.pickle', 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            return True
        
        except:
            print('Failed to save PCE object', file=sys.stderr)
            return False


    def write_outputs(self) -> None:
        """
        Writes all of the output files for all of the models associated with the 
        PCE object.
        """
        from uqpce.pce.io import write_sobols, write_coeffs, write_outputs
        from uqpce.pce._helpers import get_str_vars

        time_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        time_end = datetime.strptime(time_end, '%Y-%m-%d %H:%M:%S.%f')
        time_total = time_end - self.time_start

        for i in range(self.model_cnt):
            sobol_bounds = False if (self._sobol_bounds == False) else {'low':self._sobol_bounds['low'][i], 'high':self._sobol_bounds['high'][i]}
            coeff_uncert = False if (type(self._coeff_uncert) == bool and self._coeff_uncert == False) else self._coeff_uncert[:,i]

            self._output_str[i] = (
                f'###  UQPCE v{self.version_num} Output\n###  Analysis of case: {self.case}\n'
                f'###  Analysis started: {self.time_start}\n###  Analysis finished: '
                f'{time_end}\n###  Total compute time: {time_total}\n--------------'
                '-----------------------------------------------------------'
                f'-------\n\nMean of response {self._mean[i]:.5}\nVariance of response'
                f' {self._sigma_sq[i]:.5}\nMean error of surrogate {self._err_mean[i]:.5}\nSignal to'
                f' noise ratio {self._signal_to_noise[i]:.5}\n{1 - self.significance:.1%}'
                f' Confidence Interval on Response [{np.real(self._cil[i]):.5} , '
                f'{np.real(self._cih[i]):.5}]\n'
            )

            kwargs = {
                'out_str':self._output_str[i], 'in_file':self.input_file, 'arg_opts':'',
                'shapiro':self._shapiro[i], 'ver_str':self.verify_str[i], 'settings':'',
                'conv_str':'', 'stats_str':self._stats_str[i],
                'conf_int_str':self._conf_int_str[i]
            }

            str_vars = get_str_vars(self._matrix.model_matrix)

            write_sobols(
                os.path.join(self.output_directory[i], self.sobol_file), str_vars, 
                self._model.sobols[:,i], self._sobol_str[i], sobol_bounds
            )
            write_coeffs(
                os.path.join(self.output_directory[i], self.coeff_file), 
                self._matrix.matrix_coeffs[:,i], str_vars, coeff_uncert
            )
            write_outputs(os.path.join(self.output_directory[i], self.out_file), **kwargs)

    @property
    def variance(self) -> float:
        return self._sigma_sq

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def error(self) -> float:
        return self._error

    @property
    def signal_to_noise(self) -> float:
        return self._signal_to_noise

    @property
    def mean_sq_error(self) -> float:
        return self._mean_sq_error


if __name__ == '__main__':

    eq = '0.1*x0**2 + x0*x1 - 0.2*x1**2 + 3*x0 + 4*x1'
    settings = {
        'order':2, 'verbose':True, 'plot':True, 'model_conf_int':False, 
        'stats':False, 'verify':False
    }
    var_dict = {
        'Variable 0':{'distribution':'uniform', 'interval_low':6, 'interval_high':7, 'type':'aleatory'}, 
        'Variable 1':{'distribution':'normal', 'mean':2, 'stdev':7}, 
        'Variable 2':{'distribution':'continuous', 'pdf':'x', 'interval_low':0, 'interval_high':1}
    }

    pce = PCE(**settings)

    for key, value in var_dict.items():
        pce.add_variable(**value)

    Xt = pce.sample(count=10000)
    yt = pce.generate_responses(Xt, equation=eq)

    Xv = pce.sample(count=100)
    yv = pce.generate_responses(Xv, equation=eq)

    pce.fit(Xt, yt)
    # pce.check_fit()
    pce.check_variables(Xt)
    pce.sobols()

    # yverr = pce.verification(Xv, yv)
    cil, cih = pce.confidence_interval()

    pce.write_outputs()
