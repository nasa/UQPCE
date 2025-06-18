
# Variables
In this section, the variables used by UQPCE will be documented.

## Continuous
Multiple continuous variables are supported in UQPCE, and these variables will be discussed below.

All bounds $a$ and $b$ are such that $a \ \epsilon \ \mathbb{R}$, $b \ \epsilon \ \mathbb{R}$, and both are finite.

### Normal Variable
The equation UQPCE uses for a normal variable is shown below where $\mu \ \epsilon \ \mathbb{R}$ and $\sigma^{2} > 0$. The support range of this variable is [$-\infty$, $\infty$].

$$
	f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x - \mu)^{2}}{2\sigma^{2}}}
$$

### Uniform Variable
The equation UQPCE uses for a uniform variable is shown below. The support range of this variable is [a, b].

$$
	f(x) = \frac{1}{b - a}
$$

### Beta Variable
The equation UQPCE uses for a beta variable is shown below where $\alpha > 0$, $\beta > 0$, and the gamma function $\Gamma(n) = (n - 1)!$. The support range of this variable is [a, b].

$$
	f(x) = \frac{1}{b - a} \frac{\Gamma(\alpha + \beta)}
	{\Gamma(\alpha) \Gamma(\beta)} 
	\Bigg(\frac{x - a}{b - a}\Bigg)^{\alpha - 1} 
	\Bigg(\frac{b - x}{b - a}\Bigg)^{\beta - 1} 
$$


```{note}
The implementation of the density function used for this program is the more-common equation for the beta distribution, while common orthogonal polynomial conventions use a different equation. The user must treat their data according to the above equation.
```

### Exponential Variable
The equation UQPCE uses for an exponential variable is shown below where $\lambda > 0$. The support range of this variable is [a, $\infty$].

$$
	f(x) = \lambda e^{-\lambda (x\ -\ a)}
$$


### Gamma Variable
The equation UQPCE uses for a gamma variable is shown below where $\alpha > 0$, $\theta > 0$, and the gamma function $\Gamma(n) = (n - 1)!$. The support range of this variable is [a, $\infty$].

$$
f(x) = \frac{1}{\Gamma(\alpha)\theta^{\alpha}} (x\ -\ a)^{\alpha - 1} e^{-\frac{(x\ -\ a)}{\theta}}
$$


### Lognormal Variable
The equation UQPCE uses for a lognormal variable is shown below where $\mu \ \epsilon \ \mathbb{R}$ and $\sigma > 0$. Parameter $\mu$ is the mean of the variable's natural logarithm, and $\sigma$ is the standard deviation of the variable's natural logarithm. The support range of this variable is [a, $\infty$].

$$
f(x) = \frac{1}{(x - a) \sigma \sqrt{2 \pi}} \cdot e^{-\frac{\bigg(ln \big(x-a \big) - \mu \bigg)^{2}}{2 \sigma}}
$$


(variable)=


### User-Input Variable
This is an option for the user to input a variable that has an arbitrary continuous distribution. While this gives the user flexibility, there are some requirements the distribution must adhere to:

* The distribution must have all values included in the PDF. For example, a user-input variable for a normal distribution would need to input the mean and standard deviation values into the equation explicitly.
* The distribution and samples need to be standardized as inputs, as UQPCE currently has no way of standardizing the ContinuousVariable class distribution and samples.
* The distribution must be continuous.
* The distribution must have a finite integral over its support range.

```{note}
Since every continuous distribution is impossible to test and this variable type relies on the user's understanding of the inputs, it is recommended that users first test a UQPCE case using their user-input variables against a Monte Carlo to ensure that the variable works correctly for their purposes.

For many distributions, this user-input variable works well. It is, however, possible that the orthogonal polynomials and/or norm squared values for an equation take significant computational time to converge.
```

## Discrete
Multiple discrete variables are supported in UQPCE, and these variables will be discussed below.

All bounds $a$ and $b$ are such that $a \ \epsilon \ \mathbb{R}$, $b \ \epsilon \ \mathbb{R}$, and both are finite.

### Poisson
The equation UQPCE uses for a Poisson variable is shown below where $\lambda \geq 0$. The support range of this variable is [a, $\infty$].

$$
	f(x) = e^{-\lambda} \frac{\lambda^{(x-a)}}{(x-a)!}
$$


### Uniform
The equation UQPCE uses for a discrete uniform variable is shown below for $x = \{a, a+1, ..., b-1, b\}$. The support range of this variable is [a, b].

$$
	f(x) = \frac{1}{b-a}
$$

### Negative Binomial
The equation UQPCE uses for a negative binomial variable is shown below where $0 \leq p \leq 1$ and $r > 0$. The support range of this variable is [a, $\infty$].

$$
	f(x) = \frac{((x-a)+r-1)!}{(x-a)!(r-1)!} \ (1-p)^{(x-a)} p^r
$$

### Hypergeometric
The equation UQPCE uses for a hypergeometric variable is shown below where $M \geq 0$, $n \geq 0$, and $1 \leq N \leq M+n$. The support range of this variable is [a, $N$ + a].

$$
	f(x) = \frac{M! \ n! \ N! \ (M+n-N)!}{(x-a)! \ (n-(x-a))! \ (M+(x-a)-N)! \ (N-(x-a))! \ (M+n)!}
$$


### User-Input Variable
This is an option for the user to input a variable that has an arbitrary discrete distribution. While this gives the user flexibility, there are some requirements the distribution must adhere to:

* The distribution must have numbers plugged in for any parameters.
* The distribution and samples need to be standardized as inputs, as UQPCE currently has no way of standardizing the DiscreteVariable class distribution and samples.
* The distribution must be discrete.
* The distribution must have a finite integral over its support range.

```{note}
Since every discrete distribution is impossible to test and this variable type relies on the user's understanding of the inputs, it is recommended that users first test a UQPCE case using their user-input variables against a Monte Carlo to ensure that the variable works correctly for their purposes.
```
