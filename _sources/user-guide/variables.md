
# Variables

The following section will discuss what is required for each variable when using the UQPCE input ``yaml`` file. Refer to the [variables](../theory/variables) section for more information on the different distributions.

## Supported Distributions

| Class | Distribution | Required Parameters | Optional Parameters | 
| --- | --- | --- | --- |
| **ContinuousVariable** | ``continuous`` | pdf, interval_low, interval_high |  |
| **EpistemicVariable** | ``epistemic`` | interval_low, interval_high |  |
| **NormalVariable** | ``normal`` | mean, stdev |  |
| **UniformVariable** | ``uniform`` | interval_low, interval_high |  |
| **BetaVariable** | ``beta`` | alpha, beta | interval_low *(default = 0)*, <br /> interval_high *(default = 1)* |
| **ExponentialVariable** | ``exponential`` | lambda | interval_low *(default = 0)* |
| **GammaVariable** | ``gamma`` | alpha, theta | interval_low *(default = 0)* |
| **LognormalVariable** | ``lognormal`` | mean, stdev | interval_low *(default = 0)* |
| **DiscreteVariable** | ``discrete`` | pdf, interval_low, interval_high |  |
| **DiscreteEpistemicVariable** | ``discrete_uniform`` | interval_low, interval_high |  |
| **NegativeBinomialVariable** | ``negative_binomial`` | r, p | interval_low *(default = 0)* |
| **DiscreteUniformVariable** | ``discrete_uniform`` | interval_low, interval_high |  |
| **PoissonVariable** | ``poisson`` | lambda | interval_low *(default = 0)* |
| **HypergeometricVariable** | ``hypergeometric`` | M, n, N | interval_low *(default = 0)* |

&nbsp;

## Variable Parameters
| Name | Type | Description | 
| --- | --- | --- |
|**name** | ``str`` | The name for the physical meaning of the variable *(default = x{number})*|
|**distribution** | ``str`` | a string value for the chosen distribution; can be any of the types in the above ``Distribution`` column|
|**pdf** | ``str`` | The [PDF](../theory/symbols-and-defs) for a user-input variable; must follow [SymPy notation](https://docs.sympy.org/latest/tutorials/intro-tutorial/simplification.html)|
|**mean** | ``float`` | This option is the mean of a NormalVariable and the mean of the natural logarithm for the LognormalVariable|
|**stdev** | ``float`` | This option is the standard deviation of a NormalVariable and the mean of the natural logarithm for the LognormalVariable|
|**interval_low** | ``float`` | This option is the lower interval on which the data lies|
|**interval_high** | ``float`` | This option is the upper interval on which the data lies|
|**alpha** | ``float`` | The $\alpha$ parameter of the BetaVariable and the GammaVariable|
|**beta** | ``float`` | The $\beta$ parameter of the BetaVariable|
|**theta** | ``float`` | The $\theta$ parameter of the GammaVariable|
|**lambda** | ``float`` | The $\lambda$ parameter of the ExponentialVariable and PoissonVariable|
|**r** | ``float`` | The $r$ parameter of the NegativeBinomialVariable|
|**p** | ``float`` | The $p$ parameter of the NeativeBinomialVariable|
|**M** | ``float`` | The $M$ parameter of the HypergeometricVariable|
|**n** | ``float`` | The $n$ parameter of the HypergeometricVariable|
|**N** | ``float`` | The $N$ parameter of the HypergeometricVariable|


```{note}
The **pdf** parameter must be in the [SymPy notation](https://docs.sympy.org/latest/tutorials/intro-tutorial/simplification.html) with **x** as the variable in the PDF.

sqrt(2 * pi)  =  $\sqrt{2\pi}$

exp(-x)  =  $e^{-x}$
```


```{note}
YAML files are sensitive to spacing and tabs. Ensure that there are no trailing spaces in the lines. See the [PyYAML documentation](https://pyyaml.org/wiki/PyYAMLDocumentation) for more information. 
```