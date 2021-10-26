from builtins import getattr
from warnings import warn

try:
    from mpi4py.MPI import DOUBLE as MPI_DOUBLE, COMM_WORLD as MPI_COMM_WORLD
    import matplotlib.pyplot as plt
    import numpy as np
except:
    warn('Ensure that all required packages are installed.')
    exit()

from PCE_Codes._helpers import _warn
from PCE_Codes.variables.discrete import DiscreteVariable

comm = MPI_COMM_WORLD
size = comm.size
rank = comm.rank

is_manager = (rank == 0)


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

    def factor_plots(self, graph_dir, var_list, plot_data, plot_name,
                     verify=False):
        """ 
        Inputs: graph_dir- file location where to put plots
                var_list- list of variables
                plot_data- the data to be plotted
                plot_name- 'Predicted' or 'Error'; what data is 
                being plotted
                verify- if these points are the verification points or the 
                input points
        
        Generates plots for each variable against plot_data.
        """
        var_count = len(var_list)
        attr = ''
        stand = ''

        if is_manager and self.verbose:
            print(f'Generating {plot_name} vs Factor graphs\n')

        if self.standardize:
            attr = ''.join((attr, 'std_'))
            stand = ' (Standardized)'

        if verify:
            attr = ''.join((attr, 'verify_'))

        attr = ''.join((attr, 'vals'))

        j = rank

        while j < var_count:

            curr_var = var_list[j]
            plt.scatter(getattr(curr_var, attr), plot_data)
            plt.title(f'{plot_name} vs {curr_var.name}{stand}')
            plt.xlabel(f'{curr_var.name}')
            plt.ylabel(f'{plot_name}')
            image_path = f'{graph_dir}/{plot_name}_vs_{curr_var.name}'

            if isinstance(curr_var, DiscreteVariable):

                if hasattr(curr_var, 'categories'):

                    if not self.standardize:
                        ticks = curr_var.unstandardize_points(curr_var.x_values)
                    else:
                        ticks = curr_var.x_values

                    plt.xticks(ticks, curr_var.categories)

            plt.savefig(image_path, dpi=600, bbox_inches='tight')
            plt.clf()

            j += size

    def error_vs_pred(self, graph_dir, err, pred, plot_name):
        """
        Inputs: graph_dir- file location where to put plots
                err- difference between predicted vals and actual vals
                pred- the predicted values
                plot_name- the name of the plot
        
        Generates a plot of the error vs the predicted values.
        """
        if is_manager and self.verbose:
            print('Generating Error vs Predicted graph\n')

        plt.scatter(pred, err)
        plt.title(f'{plot_name}')
        plt.xlabel('predicted values')
        plt.ylabel('error')
        plt.savefig(f'{graph_dir}/{plot_name}', dpi=600, bbox_inches='tight')
        plt.clf()

    def pred_conf(self, graph_dir, pred, act, mean, conf):
        """
        Inputs: graph_dir- file location where to put plots
                pred- the predicted values
                act- actual response values
                mean- predicted mean for each point
                conf- predicted mean confidence intervals for each point
        
        Generates a plot of the predicted values, the verification points, and 
        the prediction interval at each point.
        """
        plot_name = 'Predicted and Actual Responses'
        xs = np.arange(1, len(pred) + 1)

        plt.errorbar(
            xs, mean, yerr=conf, capsize=4, fmt='ok', markersize=4, alpha=0.4,
            label='predicted interval'
        )
        plt.errorbar(xs, pred, fmt='xb', markersize=7, label='predicted responses')
        plt.errorbar(xs, act, fmt='xg', markersize=7, label='actual responses')
        plt.title(f'{plot_name}')
        plt.xlabel('count')
        plt.ylabel('response')
        plt.legend(loc="upper left")
        plt.savefig(f'{graph_dir}/{plot_name}', dpi=600, bbox_inches='tight')
        plt.clf()
