from warnings import warn

try:
    import numpy as np
except:
    warn('Ensure that all required packages are installed.')
    exit()


class Variable():
    """
    Inputs: number- the number of the variable from the file
    
    Class represents an input variable that is used to create and evaluate 
    the model.
    """
    __slots__ = (
        'distribution', 'samples', 'std_vals', 'type', 'name', 'resample',
        'verify_vals', 'std_verify_vals', 'vals', 'number', 'var_str',
        'var_orthopoly_vect', 'norm_sq_vals', 'low_approx', 'high_approx', 'x',
        'order', 'interval_low', 'interval_high', 'mean', 'cum_dens_func',
        'failed', 'bounds', 'test_samples', 'poly_denom', 'std_bounds'
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

    def set_vals(self, vals):
        """
        Inputs: vals- values to be set to the variable's attribute, 'vals'.
        
        Sets vals to Variable attribute 'vals.'
        """
        self.vals = vals

    def set_verify_vals(self, vals):
        """
        Inputs: vals- values to be set to the variable's attribute, 'vals'.
        
        Sets vals to Variable attribute 'vals.'
        """
        self.verify_vals = vals
