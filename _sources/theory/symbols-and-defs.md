# Symbols and Definitions

(symbols)=

## Symbols

In the next sections, concepts that UQPCE relies on will be covered. Below are the symbols and their meanings.

$X$ - the variable basis matrix

$x$ - the set of variable values from the ``run_matrix.dat`` file

$y$ - the responses of the system from the ``results.dat`` file

$\hat{y}$ - the vector of predicted responses

$x_v$ - the set of variable values from the ``verification_run_matrix.dat`` file

$y_v$ - the responses of the system from the ``verification_results.dat`` file

$\beta$ - the vector of matrix coefficients

$\langle\bf{\Psi}^{2}\rangle$ - the multivariate norm squared matrix

$\langle \Psi^{2} \rangle$ - the univariate norm squared matrix

$\psi_i$ - the $i^{th}$ order orthogonal polynomial basis component

$p$ - the number of terms, including the intercept, in the model

$k$ - the number of terms, not including the intercept, in the model

$n$ - the number of responses used to create the model

$n_v$ - the number of responses used to verify the model

$m$ - the number of variables

$M$ - the model interaction matrix 

$\alpha$ - the significance level from confidence interval

$t$ - a Student's t-distributed random variable

$f$ - an *f* distributed random variable

$\mathcal{R}$ - residual


## Abbreviations

**PDF**
: Probability Density Function

**CDF**
: Cumulative Distribution Function


## Definitions

There are a few definitions that the user will need to be familiar with in order 
to correctly use UQPCE and analyze its results.

**Aleatory**
: This is a type of uncertainty due to inherent variation of a parameter, physical system or environment. This type of uncertainty is often referred to as irreducible uncertainty.

**Epistemic**
: This is a type of uncertainty due to imperfect knowledge or ignorance of a variable.

**Sobol**
: Sobol indices are global non-linear estimates of sensitivity. In general, the larger a Sobol index, the more sensitive the model is to that term.