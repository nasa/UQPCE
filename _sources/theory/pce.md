# Polynomial Chaos Expansion
The following section will cover the main mathematical concepts used in the PCE codes.

## Orthogonal Polynomials
Each variable has orthogonal polynomials that depend on its distribution. For the ExponentialVariable, GammaVariable, ContinuousVariable, and all discrete variable classes, the Gram-Schmidt process is used to find the orthogonal polynomials. The NormalVariable, UniformVariable, and BetaVariable classes use recursive equations to calculate each distribution's orthogonal polynomials.

The Gram-Schmidt process uses the below equation to solve for the $i^{th}$ orthogonal polynomial for a distribution with the weighting function, $w(x)$, over the variable's support range [a, b].

$$
	\psi_{0}(x)=1
$$

$$
	\psi_{i}(x) = x^i + \sum_{j=0}^{i-1}\ -\frac{\int_{a}^{b} x^i\ \psi_{j}(x)\ w(x)\ dx}{\int_{a}^{b} \psi_{j}(x)^2\ w(x)\ dx}\ \psi_{j}(x)
$$

This leads to the orthogonal polynomials

$$
	\psi_{1}(x)=x -\frac{\int_{a}^{b} x\ \psi_{0}(x)\ w(x)\ dx}{\int_{a}^{b} \psi_{0}(x)^2\ w(x)\ dx}\psi_{0}(x)
$$
	
$$
	\psi_{2}(x)=x^2 -\frac{\int_{a}^{b} x^2\ \psi_{0}(x)\ w(x)\ dx}{\int_{a}^{b} \psi_{0}(x)^2\ w(x)\ dx}
	\ \psi_{0}(x) -\frac{\int_{a}^{b} x^2\ \psi_{1}(x)\ w(x)\ dx}{\int_{a}^{b} \psi_{1}(x)^2\ w(x)\ dx}\psi_{1}(x)
$$

$$
	...
$$


Since $\psi_{0}(x)=1$, the first orthogonal polynomials become
	
$$
	\psi_{1}(x)=x -\frac{\int_{a}^{b} x\ w(x)\ dx}{\int_{a}^{b} w(x)\ dx}
$$

$$
	\psi_{2}(x)=x^2 -\frac{\int_{a}^{b} x^2\ w(x)\ dx}{\int_{a}^{b}\ w(x)\ dx}
	\ -\frac{\int_{a}^{b} x^2\ \psi_{1}(x)\ w(x)\ dx}{\int_{a}^{b} \psi_{1}(x)^2\ w(x)\ dx}\ \psi_{1}(x)
$$	

$$
    ...
$$

## Norm Squared
The BetaVariable, ExponentialVariable, GammaVariable, ContinuousVariable, and all discrete classes use the following method to calculate each variable's univariate norm squared. The UniformVariable and NormalVariable classes use equations to generate the univariate norm squared values.

The univariate norm squared is given by 

$$
	\langle\Psi_{i}^{2}\rangle = \int_{a}^{b} w(x) (\psi_{i}(x))^{2} dx
$$

where $a$ and $b$ are the bounds of the variable's support range, function $w(x)$ is the weighting function, and $\psi_{i}(x)$ is the orthogonal polynomial for order $i$.

For example, given that the second order Hermite polynomial is

$$
	\psi_{2} = H_{2} = x^{2} - 1
$$

a normal variable where $i = 2$ would have a univariate norm squared of

$$
	\langle\Psi_{2}^{2}\rangle = \int_{a}^{b} f(x) (\psi_{2}(x))^{2} dx
$$

$$
	\langle\Psi_{2}^{2}\rangle = {\int_{-\infty}^{\infty} 
	\frac{1}{\sqrt{2\pi}} e^{\frac{-x^{2}}{2}} (x^{2}-1)^{2} dx}
$$

Once these are calculated for each variable, the norm squared of the multivariate orthogonal polynomial can be calculated {cite:ps}`Eldred2009`

$$
	\langle{\bf \Psi}^{2}_{j}\rangle{}=\prod_{i=1}^{m}\langle\Psi_{M_i^j}^{2}\rangle
$$

where $j$ is the index of the model term, $i$ represents the index of the variable, and $M$ is the model interaction matrix.

(system_of_equations)=

## Solving the System of Equations
A system of equations is created in the format of

$$
	y(x) \cong \beta X
$$

This technique is used initially to solve for the $\beta$ coefficients, and it is later used to solve for the responses that the model generates from the $\beta$ coefficients and the $X$ matrix.


## Model Interaction Matrix
The model interaction matrix for two variables with $order = 2$ is given by

$$
	M =		\begin{pmatrix}
			0&0\\
			1&0\\
			0&1\\
			2&0\\
			1&1\\
			0&2\\
			\end{pmatrix}
$$

where the columns represent the variables present and the rows represent the combinations of interactions up to $order = 2$ {cite:ps}`Eldred2009`.

## Variable Basis
The values from the model interaction matrix are used as indices for forming the variable basis, $X$, as shown below

$$
	X_{0}(\textbf{x}) = \psi_{0}(x_{1}) \psi_{0}(x_{2})\\
	
	X_{1}(\textbf{x}) = \psi_{1}(x_{1}) \psi_{0}(x_{2})\\
	
	X_{2}(\textbf{x}) = \psi_{0}(x_{1}) \psi_{1}(x_{2})\\
	
	X_{3}(\textbf{x}) = \psi_{2}(x_{1}) \psi_{0}(x_{2})\\
	
	X_{4}(\textbf{x}) = \psi_{1}(x_{1}) \psi_{1}(x_{2})\\
	
	X_{5}(\textbf{x}) = \psi_{0}(x_{1}) \psi_{2}(x_{2})\\
$$

The integer $i$ in $\psi_{i}$ represents the order of each variable in each row of the interaction matrix, and $\psi_{i}$ is a function for the respective variable's orthogonal polynomial basis at the given order {cite:ps}`Eldred2009`.

## Sobol Sensitivity Indices

The Sobol sensitivity indices that are output by UQPCE allow for users to see which input parameters the model is most affected by for the given input space.

The Sobol indices are calculated using the matrix coefficients, $\beta$, and the norm squared, $\langle\Psi^{2}\rangle$, using the below equation for all $i \geq 1$.

$$
	S_{i-1} = \frac{\beta_i^2 \langle\bf{\Psi}_{i}^{2}\rangle}{\sum_{j=1}^{p} \beta_j^2 \langle\bf{\Psi}_{j}^{2}\rangle}
$$

```{note}
There is no Sobol is for the intercept term, $\beta_0$.
```


The total Sobol for each variable, $x_i$, is given by

$$
	S_{T_{x_i}} = \sum_{j=1}^{p} P_{i,j}
$$

where 

$$
	P_{i,j}= \left\{
	  \begin{array}{lr} 
	      S_{j-1} \ , & M_{j,i} > 0\\
	      0 \ , & M_{j,i} = 0\\
	      \end{array}
	\right
	\}
$$

{cite:ps}`Crestaux2009`