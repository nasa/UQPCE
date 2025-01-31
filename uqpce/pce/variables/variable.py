
import numpy as np


class Variable():
    """
    Inputs: number- the number of the variable from the file
    
    Class represents an input variable that is used to create and evaluate 
    the model.
    """
    __slots__ = (
        'distribution', 'samples', 'std_vals', 'type', 'name', 'verify_vals', 'std_verify_vals', 'vals', 'number', 'var_str',
        'var_orthopoly_vect', 'norm_sq_vals', 'low_approx', 'high_approx', 'x',
        'order', 'interval_low', 'interval_high', 'mean', 'cum_dens_func',
        'failed', 'bounds', 'test_samples', 'poly_denom', 'dist', 'std_bounds'
    )

    def __init__(self, number=0):
        pass

    def get_norm_sq_val(self, matrix_val):
        """
        Inputs: matrix_val- the value in the model matrix to consider
        
        Returns the norm squared value corresponding to the matrix value.
        """
        return float(self.norm_sq_vals[matrix_val])

    def get_var_basis(self, matrix_size, model_matrix, index):
        """
        Inputs: matrix_size- the size of the model matrix
                model_matrix- the model matrix
                index- the variable index to consider
        
        Creates the variable basis for the variable based on the values in the 
        model matrix that correspond to the variable index.
        """
        var_basis = np.zeros(matrix_size, dtype=object)

        for i in range(matrix_size):
            var_basis[i] = self.var_orthopoly_vect[int(model_matrix[i, index])]

        return var_basis

    def resample(self, count):
        samps = self.generate_samples(count, standardize=True)
        # samps[np.argmin(samps)] = self.std_bounds[0]
        # samps[np.argmax(samps)] = self.std_bounds[1]
        return samps

    def generate_samples(self, count, standardize=False):
        """
        Inputs: count- the number of points needed to be generated

        Overrides the Variable class generate_samples to align with
        a normal distribution.
        """
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=1)
        samps = sampler.random(n=count)

        vals = self.cdf_sample(samps)

        if standardize:
            return self.standardize_points(vals)

        return vals
    
    
    def standardize(self, vals):
        """
        Force the child classes to implement this function to prevent an 
        inaccurate default.
        """
        raise NotImplementedError
    
    def cdf_sample(self, cdf):
        """
        Sample from the CDF location of the distribution.
        """

        return self.dist.ppf(cdf).reshape(-1,)
    
    def get_mean(self):
        """
        Return the mean of the variable.
        """
        return self.dist.mean()