import sys
from builtins import setattr, getattr
import random

import numpy as np
from scipy.stats import poisson, randint, nbinom, hypergeom
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify
from sympy import *
from sympy import symbols, Sum

try:
    from mpi4py.MPI import (
        COMM_WORLD as MPI_COMM_WORLD, DOUBLE as MPI_DOUBLE
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

from uqpce.pce._helpers import uniform_hypercube
from uqpce.pce.enums import Distribution, UncertaintyType
from uqpce.pce.variables.variable import Variable
from uqpce.pce.error import VariableInputError


class DiscreteVariable(Variable):
    """
    Inputs: pdf- the equation that defines the pdf of the variable values
            interval_low- the low interval of the variable
            interval_high- the high interval of the variable
            order- the order of the model to calculate the orthogonal
            polynomials and norm squared values
            type- the type of variable
            name- the name of the variable
            number- the number of the variable from the file
    
    Class represents a discrete variable. When using a variable of only this 
    type, the x_values must be standardized for the desired distribution and 
    the probabilities must add up to 1.
    """

    __slots__ = ('distribution', 'x_values', 'probabilities')

    def __init__(
            self, pdf, interval_low, interval_high, order=2, name='', number=0
        ):

        self.interval_low = interval_low
        self.interval_high = interval_high
        self.order = order
        self.bounds = (self.interval_low, self.interval_high)
        self.std_bounds = self.bounds
        
        self.name = f'x{number}' if name == '' else name
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)

        self.distribution = pdf
        self.distribution = ''.join(self.distribution.split())
        self.distribution = parse_expr(
            self.distribution, local_dict={'x':self.x}
        )
        self.type = UncertaintyType.ALEATORY

        self.get_probability_density_func()
        self.check_num_string()

        self.recursive_var_basis(self.x_values, self.probabilities, self.order)
        self.create_norm_sq(self.x_values, self.probabilities)

        self.low_approx = np.min(self.x_values)
        self.high_approx = np.max(self.x_values)
        self.std_bounds = (self.low_approx, self.high_approx)


    def check_distribution(self, X):
        """
        Checks all values in an array to ensure that they are standardized.
        """
        std_vals = self.standardize_points(X)

        mx = np.max(std_vals)
        mn = np.min(std_vals)

        if rank == 0 and mx > self.high_approx or mn < self.low_approx:
            print(
                f'Large standardized value for variable {self.name} '
                'with user distribution found. Check input and run matrix.', 
                file=sys.stderr
            )
            return -1


    def get_probability_density_func(self):
        """
        Ensures that the probabilities sum to be 1.
        """
        tol = 1e-12
        target = 1 - tol
        integ = 0
        inc = 5
        beg = 20

        dist_sum = N(
            Sum(
                self.distribution,
                (self.x, self.interval_low, self.interval_high)
            ).doit()
        )
        self.distribution = self.distribution / dist_sum
        f = np.vectorize(lambdify(self.x, self.distribution, ('numpy', 'sympy')))

        if self.interval_low != '-oo' and self.interval_high != 'oo':
            low_approx = self.interval_low
            high_approx = self.interval_high
            self.x_values = np.arange(self.interval_low, self.interval_high + 1)

        else:
            low_approx = -beg if self.interval_low == '-oo' else self.interval_low
            high_approx = beg if self.interval_high == 'oo' else self.interval_high

            while integ < target:
                xs = np.arange(low_approx, high_approx + 1)
                func = f(xs)

                integ = np.sum(func)
                low_approx = low_approx - inc if self.interval_low == '-oo' else low_approx
                high_approx = high_approx + inc if self.interval_high == 'oo' else high_approx

            self.x_values = xs

        prob_acc = N(
            Sum(self.distribution, (self.x, low_approx, high_approx)).doit()
        )

        if prob_acc < target:
            exit(f'Variable {self.name} is not correct.')

        self.probabilities = f(self.x_values)


    def generate_samples(self, samp_size, **kwargs):
        """ 
        Inputs: samp_size- the number of points needed to be generated

        Overrides the Variable class generate_samples to align with 
        a discrete uniform distribution.
        """
        vals = np.array(
            random.choices(
                self.x_values, weights=self.probabilities, k=samp_size
            )
        )

        return vals


    def resample(self, samp_size):
        """
        Inputs: samp_size- the number of samples to generate according to the
                distribution

        Overrides the Variable class resample to align with 
        a discrete uniform distribution.
        """
        return self.generate_samples(samp_size, standardize=True)



    def create_norm_sq(self, x_values, probabilities):
        """
        Inputs: x_values- the x-values associated with the variable
                probabilities- the probabilities associated with the x-values
        
        Calculates the norm squared values up to the order of polynomial 
        expansion based on the probability density function and its 
        corresponding orthogonal polynomials.
        """
        orthopoly_count = len(self.var_orthopoly_vect)
        self.norm_sq_vals = np.zeros(orthopoly_count)

        norm_sq_thresh = 1e-16

        for i in range(orthopoly_count):

            expr = self.var_orthopoly_vect[i] ** 2
            sum_func = np.vectorize(lambdify(self.x, expr))
            self.norm_sq_vals[i] = np.sum(sum_func(x_values) * probabilities)

        if (self.norm_sq_vals == 0).any():
            print(f'Finding the norm squared for variable {self.name} failed.', file=sys.stderr)

        if (self.norm_sq_vals <= norm_sq_thresh).any():
            print(
                f'At least one norm squared value for variable {self.name} is '
                f'very small. This can introduce error into the model.', 
                file=sys.stderr
            )


    def recursive_var_basis(self, x_values, probabilities, order):
        """
        Inputs: x_values- the x-values associated with the variable
                probabilities- the probabilities associated with the x-values
                order- the order of polynomial expansion
                
        Recursively calculates the variable basis up to the input 'order'.
        """

        if order == 0:
            self.poly_denom = np.zeros(self.order, dtype=object)
            self.var_orthopoly_vect = np.zeros(self.order + 1, dtype=object)
            self.var_orthopoly_vect[order] = 1
            return

        else:
            self.recursive_var_basis(x_values, probabilities, order - 1)
            curr = self.x ** order

            for i in range(order):
                orthopoly = self.var_orthopoly_vect[i]

                poly_denom_expr = orthopoly ** 2
                poly_denom = np.vectorize(lambdify((self.x,), poly_denom_expr))

                expr = self.x ** order * orthopoly
                numer = np.vectorize(lambdify((self.x,), expr))
                num = np.sum(numer(x_values) * probabilities)
                intergal_eval = (
                    num / np.sum(poly_denom(x_values) * probabilities)
                    * orthopoly
                )

                curr -= intergal_eval

            self.var_orthopoly_vect[order] = curr

            if order == self.order and (np.array(self.var_orthopoly_vect) == 0).any():
                print(
                    f'Variable {self.name} has at least one orthogonal polynomial '
                    f'that is zero. The model may not be accurate', file=sys.stderr
                )

            return


    def standardize(self, orig, std_vals):
        """
        Inputs: orig- the un-standardized values
                std_vals- the attribue name for the standardized vals
        
        Overrides the Variable class standardize to align with 
        a discrete uniform distribution.
        """
        setattr(self, std_vals, getattr(self, orig))

        return getattr(self, std_vals)


    def standardize_points(self, values):
        """
        Inputs: values- unstandardized points corresponding to the variable's
        distribution

        Standardizes and returns the inputs points.
        """
        return values  # general discrete variable must already be standardized


    def unstandardize_points(self, value):
        """
        Inputs: values- unstandardized points corresponding to the variable's
        distribution

        Calculates and returns the unscaled variable value from the 
        standardized value.
        """
        return value  # general discrete variable must already be standardized


    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the 
        values that might contain it.
        """
        pass

    def get_mean(self):
        """
        Returns the mean of a DiscreteVariable.
        """
        decimals = 30
        if not hasattr(self, 'mean'):
            self.mean = np.sum(self.x_values * self.probabilities)

        return N(self.mean, decimals)

class PoissonVariable(DiscreteVariable):
    """
    Inputs: lambd- the lambda parameter of the variable
            interval_low- the low interval of the variable
            order- the order of the model to calculate the orthogonal
            polynomials and norm squared values
            type- the type of variable
            name- the name of the variable
            number- the number of the variable from the file
            
    Represents a discrete poisson variable. The methods in this class correspond to 
    those of a discrete poisson variable.
    """
    __slots__ = ('lambda')

    equation = 'exp(-c) * (c**x / x!)'

    def __init__(self, lambd, interval_low=0, order=2, name='', number=0):

        if not lambd >= 0:
            raise VariableInputError(
                'PoissonVariable lambd must be greater than 0.'
            )

        self.interval_low = interval_low
        self.order = order

        self.name = f'x{number}' if name == '' else name
        setattr(self, 'lambda', lambd)
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)

        self.distribution = Distribution.POISSON
        self.type = UncertaintyType.ALEATORY
        self.dist = poisson(mu=getattr(self, 'lambda'), loc=self.interval_low)

        self.find_high_lim()

        low_percent = 8e-17
        high_percent = 1 - low_percent
        self.low_approx = self.interval_low
        self.high_approx = self.dist.ppf(high_percent)
        self.bounds = (self.interval_low, self.high_approx)
        self.std_bounds = (0, self.standardize_points(self.high_approx))

        self.get_probability_density_func()
        self.check_num_string()

        self.recursive_var_basis(self.x_values, self.probabilities, self.order)
        self.create_norm_sq(self.x_values, self.probabilities)

        self.low_approx = np.min(self.x_values)
        self.high_approx = np.max(self.x_values)
        self.std_bounds = (self.low_approx, self.high_approx)


    def find_high_lim(self):
        """
        Finds the high interval to use in calculations for the variable basis 
        and univariate norm squared values.
        """
        low_percent = 8e-17
        high_percent = 1 - low_percent

        stand_dist = poisson(mu=getattr(self, 'lambda'))
        high = np.ceil(stand_dist.ppf(high_percent))
        low = np.floor(stand_dist.ppf(low_percent))

        self.x_values = np.arange(low, high + 1)


    def get_probability_density_func(self):
        """
        Calculates the probabilities for the PoissonVariable x_values.
        """
        dist = poisson(mu=getattr(self, 'lambda'))
        self.probabilities = dist.pmf(self.x_values)


    def standardize(self, orig, std_vals):
        """
        Inputs: orig- the un-standardized values
                std_vals- the attribue name for the standardized vals

        Overrides the Variable class standardize to align with 
        a discrete poisson distribution.
        """
        standardized = getattr(self, orig) - self.interval_low
        setattr(self, std_vals, standardized)
        return standardized


    def standardize_points(self, values):
        """
        Inputs: values- unstandardized points corresponding to the variable's
        distribution

        Standardizes and returns the inputs points.
        """
        standardized = values - self.interval_low
        return standardized


    def unstandardize_points(self, values):
        """
        Inputs: values- the standardized value to be unstandardized
        
        Calculates and returns the unscaled variable value from the 
        standardized value.
        """
        unstandardized = values + self.interval_low
        return unstandardized


    def check_distribution(self, X):
        """
        Overrides the Variable class check_distribution to align with 
        a discrete poisson distribution.
        """
        std_vals = self.standardize_points(X)

        mx = np.max(std_vals)
        mn = np.min(std_vals)

        if rank == 0 and (mx > 40) or (mn < 0):
            print(
                f'Large standardized value for variable {self.name} '
                'with Poisson distribution found. Check input and run matrix.', 
                file=sys.stderr
            )
            return -1
        

    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the 
        values that might contain it.
        """
        lambd = getattr(self, 'lambda')

        if isinstance(self.interval_low, str) and 'pi' in self.interval_low :
            self.interval_low = float(self.interval_low.replace('pi', str(np.pi)))

        if isinstance(lambd, str) and 'pi' in lambd:
            setattr(self, 'lambda', float(lambd.replace('pi', str(np.pi))))

    def generate_samples(self, count, standardize=False):
        return super(DiscreteVariable, self).generate_samples(count, standardize=standardize)

    def get_mean(self):
        return self.dist.mean()

class NegativeBinomialVariable(DiscreteVariable):
    """
    Inputs: r- the r parameter of the variable
            p- the p parameter of the variable
            interval_low- the low interval of the variable
            order- the order of the model to calculate the orthogonal
            polynomials and norm squared values
            type- the type of variable
            name- the name of the variable
            number- the number of the variable from the file
    
    Represents a discrete NegativeBinomial variable. The methods in this class correspond to 
    those of a discrete NegativeBinomial variable.
    """

    __slots__ = ('r', 'p', 'divisions', 'dist')

    equation = '((x+r-1)!/(x!*(r-1)!))*(1-p)**x*p**r'

    def __init__(self, r, p, interval_low=0, order=2, name='', number=0):

        if (p < 0) or (p > 1):
            raise VariableInputError(
                'NegativeBinomialVariable p must be greater than 0 and less than 1.'
            )

        if r <= 0:
            raise VariableInputError(
                'NegativeBinomialVariable r must be greater than 0.'
            )

        self.interval_low = interval_low
        self.order = order
        self.r = r
        self.p = p

        self.name = f'x{number}' if name == '' else name
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)

        self.distribution = Distribution.NEGATIVE_BINOMIAL
        self.type = UncertaintyType.ALEATORY
        self.dist = nbinom(n=self.r, p=self.p, loc=self.interval_low)

        self.find_high_lim()

        low_percent = 8e-17
        high_percent = 1 - low_percent
        self.low_approx = self.interval_low
        self.high_approx = self.dist.ppf(high_percent)
        self.bounds = (self.interval_low, self.high_approx)
        self.std_bounds = (0, self.standardize_points(self.high_approx))

        self.get_probability_density_func()
        self.check_num_string()

        self.recursive_var_basis(self.x_values, self.probabilities, self.order)
        self.create_norm_sq(self.x_values, self.probabilities)

        self.low_approx = np.min(self.x_values)
        self.high_approx = np.max(self.x_values)
        self.std_bounds = (self.low_approx, self.high_approx)


    def find_high_lim(self):
        """
        Finds the high interval to use in calculations for the variable basis 
        and univariate norm squared values.
        """
        low_percent = 8e-17
        high_percent = 1 - low_percent

        stand_dist = nbinom(n=self.r, p=self.p)
        high = np.ceil(stand_dist.ppf(high_percent))
        low = np.floor(stand_dist.ppf(low_percent))

        self.x_values = np.arange(low, high + 1)


    def get_probability_density_func(self):
        """
        Calculates the probabilities for the NegativeBinomial x_values.
        """
        dist = nbinom(n=self.r, p=self.p)
        self.probabilities = dist.pmf(self.x_values)


    def standardize(self, orig, std_vals):
        """
        Inputs: orig- the un-standardized values
                std_vals- the attribue name for the standardized vals

        Overrides the Variable class standardize to align with 
        a discrete NegativeBinomial distribution.
        """
        original = getattr(self, orig)
        standard = original - self.interval_low
        setattr(self, std_vals, standard)

        return getattr(self, std_vals)


    def standardize_points(self, values):
        """
        Inputs: values- unstandardized points corresponding to the variable's
        distribution

        Standardizes and returns the inputs points.
        """
        standard = values - self.interval_low

        return standard


    def unstandardize_points(self, values):
        """
        Inputs: values- the standardized value to be unstandardized
        
        Calculates and returns the unscaled variable value from the 
        standardized value.
        """
        unstandard = values + self.interval_low

        return unstandard


    def check_distribution(self, X):
        """
        Overrides the Variable class check_distribution to align with 
        a discrete NegativeBinomial distribution.
        """
        std_vals = self.standardize_points(X)

        mx = np.max(std_vals)
        mn = np.min(std_vals)

        std_min = np.min(self.x_values)
        std_max = np.max(self.x_values)

        if rank == 0 and mx > std_max or mn < std_min:
            print(
                f'Large standardized value for variable {self.name} '
                'with negative binomial distribution found. Check input and '
                'run matrix.', file=sys.stderr
            )
            return -1


    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the 
        values that might contain it.
        """

        if isinstance(self.interval_low, str) and 'pi' in self.interval_low:
            self.interval_low = float(self.interval_low.replace('pi', str(np.pi)))

        if isinstance(self.r, str) and 'pi' in self.r:
            self.r = float(self.r.replace('pi', str(np.pi)))
            

    def get_mean(self):
        return self.dist.mean()

    def generate_samples(self, count, standardize=False):
        return super(DiscreteVariable, self).generate_samples(count, standardize=standardize)


class HypergeometricVariable(DiscreteVariable):
    """
    Inputs: M- the M parameter of the variable
            n- the n parameter of the variable
            N- the N parameter of the variable
            interval_low- the low interval of the variable
            order- the order of the model to calculate the orthogonal
            polynomials and norm squared values
            type- the type of variable
            name- the name of the variable
            number- the number of the variable from the file
    
    Represents a discrete hypergeometric variable. The methods in this class correspond to 
    those of a discrete hypergeometric variable.
    """
    __slots__ = ('M', 'n', 'N', 'dist')

    # https://mathworld.wolfram.com/HypergeometricDistribution.html
    equation = '(M! * n! * N! * (M+n-N)!)/(k! * (n-k)! * (M+k-N)! * (N-k)! * (M+n)!)'

    def __init__(self, M, n, N, interval_low=0, order=2, name='', number=0):

        if M < 0:
            raise VariableInputError(
                'HypergeometricVariable M must be greater or equal to 0.'
            )

        if n < 0:
            raise VariableInputError(
                'HypergeometricVariable n must be greater or equal to 0.'
            )

        if (N < 1) or (N > (M + n)):
            raise VariableInputError(
                'HypergeometricVariable M must be greater than 1 and less than M+n.'
            )

        self.interval_low = interval_low
        self.order = order
        self.M = M
        self.n = n
        self.N = N
        
        self.name = f'x{number}' if name == '' else name
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)

        self.distribution = Distribution.HYPERGEOMETRIC
        self.type = UncertaintyType.ALEATORY
        self.dist = hypergeom(M=self.M + self.n, n=self.n, N=self.N, loc=self.interval_low)

        self.find_high_lim()

        low_percent = 8e-17
        high_percent = 1 - low_percent
        self.low_approx = self.interval_low
        self.high_approx = self.dist.ppf(high_percent)
        self.bounds = (self.interval_low, self.high_approx)
        self.std_bounds = (0, self.standardize_points(self.high_approx))

        self.get_probability_density_func()
        self.check_num_string()

        self.recursive_var_basis(self.x_values, self.probabilities, self.order)
        self.create_norm_sq(self.x_values, self.probabilities)

        self.low_approx = np.min(self.x_values)
        self.high_approx = np.max(self.x_values)
        self.std_bounds = (self.low_approx, self.high_approx)


    def find_high_lim(self):
        """
        Finds the high interval to use in calculations for the variable basis 
        and univariate norm squared values.
        """
        low_percent = 8e-17
        high_percent = 1 - low_percent

        stand_dist = hypergeom(M=self.M + self.n, n=self.n, N=self.N)
        high = np.ceil(stand_dist.ppf(high_percent))
        low = np.floor(stand_dist.ppf(low_percent))

        self.x_values = np.arange(low, high + 1)


    def get_probability_density_func(self):
        """
        Calculates the probabilities for the HypergeomericVariable x_values.
        """
        dist = hypergeom(M=self.M + self.n, n=self.n, N=self.N)
        self.probabilities = dist.pmf(self.x_values)


    def standardize(self, orig, std_vals):
        """
        Inputs: orig- the un-standardized values
                std_vals- the attribue name for the standardized vals
        
        Overrides the Variable class standardize to align with 
        a discrete Hypergeomeric distribution.
        """
        original = getattr(self, orig)
        standard = original - self.interval_low
        setattr(self, std_vals, standard)

        return getattr(self, std_vals)


    def standardize_points(self, values):
        """
        Inputs: values- unstandardized points corresponding to the variable's
        distribution

        Standardizes and returns the inputs points.
        """
        standard = values - self.interval_low

        return standard


    def unstandardize_points(self, values):
        """
        Inputs: values- the standardized value to be unstandardized
        
        Calculates and returns the unscaled variable value from the 
        standardized value.
        """
        unstandard = values + self.interval_low

        return unstandard


    def check_distribution(self, X):
        """
        Overrides the Variable class check_distribution to align with 
        a discrete hypergeometric distribution.
        """
        std_vals = self.standardize_points(X)

        mx = np.max(std_vals)
        mn = np.min(std_vals)

        if rank == 0 and (mx > self.n) or (mn < 0):
            print(
                f'Large standardized value for variable {self.name} with '
                'hypergeometric distribution found. Check input and run matrix.', 
                file=sys.stderr
            )
            return -1


    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the 
        values that might contain it.
        """

        if isinstance(self.interval_low, str) and 'pi' in self.interval_low:
            self.interval_low = float(self.interval_low.replace('pi', str(np.pi)))

    def generate_samples(self, count, standardize=False):
        return super(DiscreteVariable, self).generate_samples(count, standardize=standardize)

    def get_mean(self):
        return self.dist.mean()

class UniformVariable(DiscreteVariable):
    """
    Inputs: interval_low- the low interval of the variable
            interval_high- the high interval of the variable
            order- the order of the model to calculate the orthogonal
            polynomials and norm squared values
            type- the type of variable
            name- the name of the variable
            number- the number of the variable from the file
    
    Represents a discrete uniform variable. The methods in this class correspond to 
    those of a categorical variable.
    """

    def __init__(
            self, interval_low, interval_high, order=2, name='', number=0
        ):

        if not interval_low < interval_high:
            raise VariableInputError(
                'UniformVariable interval_high must be greater than '
                'interval_low.'
            )

        self.interval_low = interval_low
        self.interval_high = interval_high
        self.order = order
        
        self.name = f'x{number}' if name == '' else name
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)

        self.distribution = Distribution.DISCRETE_UNIFORM
        self.type = UncertaintyType.ALEATORY

        low = -1
        high = 1

        self.check_num_string()

        # SciPy stats randint doesn't include the high interval, so add 1
        self.dist = randint(self.interval_low, self.interval_high + 1)

        self.bounds = (self.interval_low, self.interval_high)
        self.std_bounds = (0, self.standardize_points(self.interval_high))

        # Add one to the range to include both boundaries.
        rng = self.interval_high - self.interval_low + 1
        self.x_values = np.linspace(low, high, num=rng)
        divisions = len(self.x_values)
        self.probabilities = np.ones(divisions) / divisions

        self.recursive_var_basis(self.x_values, self.probabilities, self.order)
        self.create_norm_sq(self.x_values, self.probabilities)

        self.low_approx = np.min(self.x_values)
        self.high_approx = np.max(self.x_values)
        self.std_bounds = (self.low_approx, self.high_approx)


    def standardize(self, orig, std_vals):
        """
        Inputs: orig- the un-standardized values
                std_vals- the attribue name for the standardized vals
        
        Overrides the Variable class standardize to align with 
        a discrete uniform distribution.
        """
        original = getattr(self, orig)

        rng = self.interval_high - self.interval_low
        mean = rng / 2 + self.interval_low
        stdev = rng / 2

        standard = (original[:] - mean) / stdev
        setattr(self, std_vals, standard)

        return getattr(self, std_vals)


    def standardize_points(self, values):
        """
        Inputs: values- unstandardized points corresponding to the variable's
        distribution

        Standardizes and returns the inputs points.
        """
        rng = self.interval_high - self.interval_low
        mean = rng / 2 + self.interval_low
        stdev = rng / 2

        return (values - mean) / stdev


    def unstandardize_points(self, value):
        """
        Inputs: value- the standardized value to be unstandardized
        
        Calculates and returns the unscaled variable value from the 
        standardized value.
        """
        rng = self.interval_high - self.interval_low
        mean = rng / 2 + self.interval_low
        stdev = rng / 2

        return (value * stdev) + mean


    def check_distribution(self, X):
        """
        Overrides the Variable class check_distribution to align with 
        a discrete uniform distribution.
        """
        std_vals = self.standardize_points(X)

        mx = np.max(std_vals)
        mn = np.min(std_vals)

        std_min = -1
        std_max = 1

        if rank == 0 and mx > std_max or mn < std_min:
            print(
                f'Large standardized value for variable {self.name} '
                'with discrete uniform distribution found. Check input and '
                'run matrix.', file=sys.stderr
            )
            return -1


    def check_num_string(self):
        """
        Searches to replace string 'pi' with its numpy equivalent in any of the 
        values that might contain it.
        """
        if isinstance(self.interval_low, str) and 'pi' in self.interval_low:
            self.interval_low = float(self.interval_low.replace('pi', str(np.pi)))

        if isinstance(self.interval_high, str) and 'pi' in self.interval_high:
            self.interval_high = float(self.interval_high.replace('pi', str(np.pi)))

    def generate_samples(self, count, standardize=False):
        return super(DiscreteVariable, self).generate_samples(count, standardize=standardize)

    def get_mean(self):
        return self.dist.mean()

class EpistemicVariable(UniformVariable):

    def __init__(self, interval_low, interval_high, **kwargs):
        super().__init__(interval_low, interval_high, **kwargs)
        self.type = UncertaintyType.EPISTEMIC

    def generate_samples(self, count, standardize=False):
        return super(DiscreteVariable, self).generate_samples(count, standardize=standardize)
