
# Release Notes

## Version 0.2.0
- $N^{th}$-order polynomial expansion
- user-defined function input
- Latin hypercube sample generation
- command line inputs and options
- BetaVariable
- ExponentialVariable
- GammaVariable
- user-input Variable

## Version 0.2.1
- bug fixes to sample generation for the general Variable

## Version 0.2.2
- bug fixes to sample generation for BetaVariable, GammaVariable, and ExponentialVariable classes when variables have non-standard bounds

## Version 0.2.3
- bug fixes to sample generation

## Version 0.2.4
- updated sample generation that is accurate no matter how many points are generated

## Version 0.2.5
- recursive variable basis equation for BetaVariable to reduce time
- norm squared values for each variable calculated with scipy integral to reduce time and improve accuracy
- PRESS residual
- speed improvements

## Version 0.3.0
- statistics
    - mean statistics
    - variance statistics
    - mean error statistics
    - R-squared
    - R-squared adjusted
- uncertainty bounds
    - mean confidence interval
    - confidence interval uncertainty
    - Sobol uncertainty
    - coefficient uncertainty
    - mean uncertainty
- discrete variables
    - user-input
    - Poisson
    - discrete uniform
    - negative binomial
    - hypergeometric
- stepwise regression
- A-optimal design
- D-optimal design
- report generation capability
- adaptive sampling
- parallelized using mpi4py
- robust unittest
- ``UQPCE`` wrapper
- Robust Design Capability
- orthogonal polynomial updates
    - replaced SymPy integrals in Gram-Schmidt method with scipy integrals to reduce time
    - **added a symbolic option to calculate symbolic integrals for debugging purposes**
- improved variable initialization to allow user to easily use variables for other purposes
- optional seeding for random values
- less rounding error in variable basis and norm squared values
- errors raised when users input parameters that are not in the correct ranges for the variable type

## Version 1.0.0
- OpenMDAO compatability
- analytical derivatives
- stepwise regression
- backward elimination
- more efficient ProbabilityBox class
- programmer-focused package and PCE object
- restructured model construction to build N models simultaneously
- improved logo