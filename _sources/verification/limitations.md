# Limitations

There are limitations to the method used to calculate the differentiable uncertain bound. Specifically, the `mu` parameter can be the difference between appropriate and inaccurate bounds on the uncertainty. To demonstrate this, two instances of executing the [beta distribution example](content:references:beta) with with different `mu` values will be discussed below.

In the beta verification case for solving the bound on uncertainty with parameter `mu = 0.1` 

    The analytical bound is 0.9999999999999958
    The interpolated bound is 0.9999999999999958
    The solved bound is 1.1456864325112104

```{figure} ../images/verification-conf_int/continuous_beta_fail.png
---
height: 400px
name: conf_int_cont_beta_fail
---
The figure of solving for the confidence interval using the activation function technique for a beta distribution.
```

This is the same example discussed in the [bound on uncertainty of a beta distribution](content:references:beta) with only the `mu` parameter changed. In this case, `mu=1e-12` was used to acheive an appropriate result. However, choosing an appropriate value is dependent on the values of which you are calculating the uncertain bound (i.e. a smaller `mu` is not necessarily better for your application).
