# Techniques

The techniques used by UQPCE are discussed in this section. The additional statistics, confidence intervals on model parameters, and experimental designs will be discussed.

## Statistics

UQPCE provides several useful model statistics when the ``--stats`` flag is used. These statistics are discussed below.

### R-Squared

The $R^2$ statistic is a measure of how well the results are modeled by the surrogate model.

To calculate the $R^2$ statistic, the total sum or squares, $SS_T$, and the error sum of squares, $SS_E$ must first be calculated

$$
	SS_T = y'y - \frac{\big(\sum_{i=1}^{n} y_i\big)^2}{n}
$$
	
$$
	SS_E = y'y - \hat{\beta}'X'y
$$

From these two statistics, we can calculate $R^2$

$$
	R^2 = 1 - \frac{SS_E}{SS_T}
$$

{cite:ps}`Montgomery2013`

### R-Squared Adjusted

The $R^2_{adj}$ statistic, much like $R^2$, is a measure of how well the results are modeled by the surrogate model. However, this statistic reflects the fit for the number of terms included. If terms that only slightly improve the model are included, $R^2_{adj}$ is penalized.

The $R^2_{adj}$ is given by
 
$$
	R^2_{adj} = 1\ -\ \frac{n - 1}{n - p}\bigg(1\ -\ R^2\bigg)
$$

{cite:ps}`Montgomery2013`

<!-- ### Mallow's $C_p$

The Mallow's $C_p$ statistic compares how well a model with fewer terms compares to the full-term model, which is treated as the truth. The $C_p$statistic is given by the equation

$$
	C_p = \frac{SSE_{k}}{MSE_{full}} + 2 (k + 1) - n
$$


### PRESS Residual

The PRESS residual is the sum of error for all models created with ``n - 1`` results, where ``n`` is the total number of results.

$$
	PRESS = \sum_{i=1}^{n} e^2_{(i)} = \sum_{i=1}^{n} [y_{i} - \hat{y}_{(i)}]^2
$$

{cite:ps}`Montgomery2013`
	
As shown above, the error is calculated from the actual response and the predicted response from a model that omits point ``i``. The error in the prediction is squared and the sum of all of those ``n`` values is the PRESS residual.

### Mean

The average and variance of the mean are calculated from all models created with ``n - 1`` results, where ``n`` is the total number of results.

The estimate of the mean of a model is given by

$$
	\mu \cong \sum_{i=0}^{p} \beta_i X_i = \beta_0
$$

{cite:ps}`Eldred2009`

The average of the mean is given by

$$
	\mu_{avg} = \frac{\sum_{i=0}^{n} \mu_{i}}{n}
$$
	
While the variance of the mean is given by

$$
	\mu_{var} = \frac{\sum_{i=0}^{n} \big(\mu_{i} - \mu_{avg}\big)^2}{n}
$$

### Variance

The average and variance of the variance are calculated from all models created with ``n - 1`` results, where ``n`` is the total number of results.
	
The estimate of the variance of a model is given by

$$
	\sigma^{2} = \sum_{i=1}^{p} \beta^{2}_i \langle\bf{\Psi}^{2}_i\rangle
$$
	
{cite:ps}`Eldred2009`
	
The average of the variance is given by

$$
	\sigma^{2}_{avg} = \frac{\sum_{i=1}^{n} \sigma^{2}_{i}}{n}
$$
	
While the variance of the variance is given by

$$
	\sigma^{2}_{var} = \frac{\sum_{i=1}^{n} \big(\sigma^{2}_{i} - \sigma^{2}_{avg}\big)^2}{n}
$$

### Mean Error

The average and variance of the mean error are calculated by creating all models that omit one response value, determining their mean errors, and calculating the average and variance of these values.
	
The mean error is the average differential between model predicted response and observed response.

Mean error for a model is calculated with

$$
	\varepsilon = \frac{\sum_{i=0}^{n} |\hat{y}_i - y_i|}{n}
$$

The average of the mean error is given by

$$
	\varepsilon_{avg} = \frac{\sum_{i=0}^{n} \varepsilon_{i}}{n}
$$
	
While the variance of the mean error is given by

$$
	\varepsilon_{var} = \frac{\sum_{i=0}^{n} \big(\varepsilon_{i} - \varepsilon_{avg}\big)^2}{n}
$$


### Signal-to-Noise Ratio

The signal-to-noise ratio estimates the relative magnitude of the model variance to the model error. This calculation used the model variance, $\sigma^{2}$, and the model mean error, $\varepsilon$.

$$
    SNR = \frac{\sigma^{2}}{\varepsilon}
$$


### Verification Error

The verification error is a similar metric to the mean error but is calculated using the model verification responses.

$$
    \varepsilon_v = \frac{\sum_{i=0}^{n_v} |\hat{y}_{v_{i}} - y_{v_{i}}|}{n_v}
$$


## Confidence Intervals

In addition to the mean and variance of the surrogate model, UQPCE provides confidence intervals for Sobols, coefficients, and the mean when the ``--model-conf-int`` flag is used.

### Confidence Interval on the Coefficients

The bounds on the coefficients are solved by finding the uncertainty of the coefficients and then adding and subtracting that from the values.

$$
	\Delta B_i = t_{\alpha/2, n-p} \sqrt{\hat{\sigma}^2 C_{ii}}
$$

where

$$
	C = \bigg( X'X \bigg)^{-1}
$$
	
and 

$$
	\hat{\sigma}^2 = \frac{SS_E}{n - p}
$$

Once the uncertainty has been calculated, the bounds can be found by adding and subtracting this value from the original $\beta$.

$$
	\beta_{L} = \beta - \Delta \beta
$$

$$
	\beta_{H} = \beta + \Delta \beta
$$

{cite:ps}`Montgomery2013`

### Confidence Interval on the Sobols

The bounds of the Sobols are solved by recalculating the Sobols while changing the coefficients of each Sobol to minimize the magnitude of all coefficients except coefficient $\beta_i$, who has its magnitude maximized. The same is done to maximize all the magnitude of all coefficients except $\beta_i$. Doing this makes coefficient $\beta_i$ its most and least influential. With these new coefficients, the Sobols are recalculated.

To calculate the low Sobol bound, the coefficients are altered such that the coefficient of interest has a smaller magnitude and the others have a larger magnitude.

$$
	\beta_{L_j} = |\beta_j| + \Delta \beta_j \ \ \text{if} \ \ \beta_j \neq \beta_i
$$


$$
	\beta_{L_j} = \left\{
	  \begin{array}{lr} 
	      |\beta_j| - \Delta \beta_j \ , & |\beta_j| > \Delta \beta_j\\
	      0 \ , & |\beta_j| \leq \Delta \beta_j\\
	      \end{array}
	\right\} \ \ \text{if} \ \ \beta_j = \beta_i
$$


To calculate the high Sobol bound, the coefficients are altered such that the coefficient of interest has a larger magnitude and the others have a smaller magnitude

$$
	\beta_{H_j} = \left\{
	  \begin{array}{lr} 
	      |\beta_j| - \Delta \beta_j \ , & |\beta_j| > \Delta \beta_j\\
	      0 \ , & |\beta_j| \leq \Delta \beta_j\\
	      \end{array}
	\right\} \ \ \text{if} \ \ \beta_j \leq \beta_i
$$

$$
	\beta_{H_j} = |\beta_j| + \Delta \beta_j \ \ \text{if} \ \ \beta_j = \beta_i
$$


The Sobol bounds are then calculated from the new coefficients using the same equation used to calculate the Sobol values

$$
	S_{L_i} = \frac{\beta_{L_i}^2 \langle\psi_{i}^{2}\rangle}
	{\sum_{j=1}^p \langle\psi_{j}^{2}\rangle \beta_{L_j}^2}
$$
	
$$
	S_{H_i} = \frac{\beta_{H_i}^2 \langle\psi_{i}^{2}\rangle}
	{\sum_{j=1}^p \langle\psi_{j}^{2}\rangle \beta_{H_j}^2}
$$

### Confidence Interval on the Variance

The bounds on the variance is calculated from the matrix coefficients and their corresponding uncertainties.

The minimum and maximum matrix coefficients are first calculated using the equations below

$$
	\beta_{L_j} = \left\{
	  \begin{array}{lr} 
	      |\beta_j| - \Delta \beta_j \ , & |\beta_j| > \Delta \beta_j\\
	      0 \ , & |\beta_j| \leq \Delta \beta_j\\
	      \end{array}
	\right\}
$$

$$
	\beta_{H_j} = |\beta_j| + \Delta \beta_j
$$

Using these altered coefficient values, the bounds of the variance are calculated using the below equations

$$
	\sigma^2_L = \sum_{i=1}^{p} \beta^{2}_{L_i} \langle\bf{\Psi}^{2}_i\rangle
$$

$$
	\sigma^2_H = \sum_{i=1}^{p} \beta^{2}_{H_i} \langle\bf{\Psi}^{2}_i\rangle
$$
	

### Confidence Interval on the Mean Response

The bounds on the mean response is calculated by finding the uncertainty of the mean of the point, and then adding and subtracting the uncertainty from the mean {cite:ps}`Montgomery2013`. This technique is also used to calculate the confidence intervals on the ProbabilityBox plot curves.

The uncertainty of the mean response can be calculated using equation

$$
	\Delta \mu = t_{\alpha/2, n-p} \sqrt{\hat{\sigma}^2 x_0'(X'X)^{-1}x_0}
$$

Where the unbiased estimate of error variance is given by

$$
	\hat{\sigma}^2 = \frac{SS_E}{n - p}
$$

Leading to the lower and upper bounds of the mean response, point $x_0$

$$
	\mu_L = \mu - \Delta \mu
$$
	
$$
	\mu_H = \mu + \Delta \mu
$$
	
{cite:ps}`Montgomery2013`


### Confidence Interval on a Predicted Response

The confidence intervals on the verification points are calculated using the below equation.

The prediction interval is calculated using the equation

$$
	\Delta \hat{y} = t_{\alpha/2, n-p} \sqrt{\hat{\sigma}^2 (1 + x_0'(X'X)^{-1})x_0}
$$

Where the unbiased estimate of error variance is given by

$$
	\hat{\sigma}^2 = \frac{SS_E}{n - p}
$$

{cite:ps}`Montgomery2013`

Leading to the confidence bounds for point $x_0$

$$
	\hat{y}_L = \hat{y} - \Delta \hat{y}
$$

$$
	\hat{y}_H = \hat{y} + \Delta \hat{y}
$$

With this applied to each point in each of the curves in the probability box, the confidence interval is calculated using the lower and upper bounds on the resampling points. When the ``--model-conf-int`` flag is used, this is the confidence interval output by UQPCE.


## Experimental Design

To select initial variable values and determine significant model terms more intelligently, UQPCE has additional capabilities that will be discussed below. The optimal designs are used to improve the model by more-strategically choosing input points.

### D-Optimal Design

D-optimal design minimizes the value
	
$$
	|(X'X)^{-1}|
$$
	
{cite:ps}`Montgomery2013`

Below is a visualization of how this affects the distribution of finite random variables.

```{figure} ../images/d_optimal_design.png
---
height: 400px
name: d_opt
---
Random samples versus D-optimal samples for a 2-dimensional design space with uniform variables. The boundaries of the variables are shown in black.
```


In this figure, we can see that both variables follow a uniform distribution. However, the D-optimal samples are spaced such that covariance of the sample space is minimized.


### A-Optimal Design

A-optimal design minimizes the value

$$
	tr((X'X)^{-1})
$$

Where $tr(\ )$ is the trace of the matrix.

{cite:ps}`Montgomery2013`

Below is a visualization of how this affects the distribution of finite random 
variables.

```{figure} ../images/a_optimal_design.png
---
height: 400px
name: a_opt
---
Random samples versus A-optimal samples for a 2-dimensional design space with uniform variables. The boundaries of the variables are shown in black.
```

In this figure, we can see that both variables follow a uniform distribution. However, the A-optimal samples are spaced such that average variance of the matrix coefficients are minimized.

<!-- ### S-Optimal Design

S-optimal design is a Sobol-weighted design that minimizes

$$
	tr\Big(\Big(\frac{1}{1-S}\Big) (X'X)^{-1}\Big)
$$

Where $S$ is the Sobol vector, $S_0 = 0$, and $tr(\ )$ is the trace of the matrix.

```{figure} ../images/s_optimal_design.png
---
height: 400px
name: s_opt
---
Random samples versus S-optimal samples for a 2-dimensional design space with uniform variables. The boundaries of the variables are shown in black.
```

In this figure, we can see that both variables follow a uniform distribution. However, the S-optimal samples are spaced such that Sobol-weighted average variance of the matrix coefficients are minimized. -->

## Stepwise Regression

Stepwise regression is used to intelligently build the surrogate model. Below, the equations used in stepwise regression will be discussed.

Stepwise regression uses an *f*-distributed variable to determine whether or not a term will be added to the model. The *f*-distribution used by UQPCE is shown below

$$
	F_{in} = F_{\alpha_{in}}(df_1 = 1, df_2 = n_{iter}-p_{iter})
$$

$$
	F_{out} = F_{\alpha_{out}}(df_1 = 1, df_2 = n_{iter}-p_{iter})
$$
		
where $\alpha_{in}$ and $\alpha_{out}$ are sensitivities to add and remove terms and $f_{in} \geq f_{out}$.

The difference in the sum of squares, $SS_R$, due to one model term is calculated for the model that includes the most terms and the model that omits term ``j``.

$$
	SS_R(\beta_j|\beta_i,\beta_0) = SS_R(\beta_i,\beta_j|\beta_0) - SS_R(\beta_i|\beta_0)
$$

The mean squared error, $MS_E$, is for the model that includes the most 
terms

$$
	MS_E(x_j, x_i) = \frac{y'y - \hat{\beta}X'y}{n_{iter}-p_{iter}}
$$


$$
	f_j = \frac{SS_R(\beta_j|\beta_i,\beta_0)}{MS_{E_{full}}}
$$

where $\beta_0$ is the intercept term, $\beta_i$ is the most correlated term, and $\beta_j$ is the $j^{th}$ remaining term.

Stepwise regression follows the following steps:

1. Build a model consisting of the intercept and the most-correlated term.
1. Calculate the partial *F*-statistic for the remaining terms.
1. Add the term that, when added, results in the largest partial *F*-statistic if $F_j > F_{in}$.
1. Calculate the partial *F*-statistic for removing each of the terms in the current model.
1. Remove the term with the smallest partial *F*-statistic if $F_j < F_{out}$.


{cite:ps}`Montgomery2007`


## Backward Elimination

Backward elimination is the process of removing model terms within some threshold, $\alpha_{out}$, to reduce model overfitting. This method begins by building a full PCE model, deleting the least significant term at each iteration until a stopping condition is satisfied. To do this, a partial $F$-distributed variable shown in Equation [](fdist) determines the significance of each term

$$
	F_{out} = F_{\alpha_{out}}(df_1 = 1, df_2 = n_{iter}-p_{iter})
$$ (fdist)

The difference in the sum of squares, $SS_R$, due to one model term is calculated for the model that includes the most terms and the model that omits term ``j``.

$$
	SS_R(\beta_j|\beta_i,\beta_0) = SS_R(\beta_i,\beta_j|\beta_0) - SS_R(\beta_i|\beta_0)
$$

The mean squared error, $MS_E$, is for the model that includes the most terms

$$
	MS_E(x_j, x_i) = \frac{y'y - \hat{\beta}X'y}{n_{iter}-p_{iter}}
$$

Eq. [](Fj) shows a ratio of the $SS_R$ of the model with term $j$ deleted and the $MS_E$ for the model that includes term $j$ is calculated to determine the effect of a term on the model. In the equation, $\beta_0$ is the intercept term and $\beta_j$ is the $j^{th}$ term.

$$
	F_j = \frac{SS_R(\beta_j|\beta_{0...p})}{MS_{E_{full}}}
$$ (Fj)

The following steps are followed to complete the backwards elimination process:

1. Build the full polynomial chaos expansion model.
1. Calculate the partial *F*-statistic for all terms.
1. Remove the term that, when removed, results in the smallest partial *F*-statistic if $F_j < F_{out}$.
1. Repeat this until $F_j >= F_{out}$.

{cite:ps}`Montgomery2007`


## Individual Variable Order

Individual variable order allows for one or more variables to have a different order than the other variables. Allowing for a higher order input for specified variables is intended for users that know which variables have a higher order in their model. 

### Example

To use this option, include ``order`` for the desired variables as shown below, where you can see that ``Variable 4`` has ``order: 2`` while the model order is ``order: 1``

```python
Variable 0:
    distribution: normal
    mean: 1
    stdev: 0.5
    type: aleatory
Variable 1:
    distribution: uniform
    interval_low: 1.75
    interval_high: 2.25
    type: aleatory
Variable 2:
    distribution: exponential
    lambda: 3
    type: aleatory
Variable 3:
    distribution: beta
    alpha: 0.5
    beta: 2.0
    type: aleatory
Variable 4:
    distribution: gamma
    alpha: 1.0
    theta: 0.5
    type: aleatory
    order: 2
    
Settings:
    order: 1
    version: true
    verbose: true
    plot: true
    plot_stand: true
    model_conf_int: true
    stats: true
    verify: true
```

This results in a model that includes terms 1st order terms for all variables and 2nd order terms for only variable ``x4`` as shown below:

```python
intercept
x0
x1
x2
x3
x4
x0*x4
x1*x4
x2*x4
x3*x4
x4^2
```

```{warning}
This option requires the user to know their data and analytical tool **well**. Do not use this option if you're unsure of which, if any, variables have a higher order term.
```