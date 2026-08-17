(uncertainty-interval)=

# Confidence Interval

Using `jax` to interpolate is the default method for calculating the confidence interval in `UQPCEGroup` as of v1.0.1 of `UQPCE`. This method is recommended.

The `UQPCEGroup` option `use_tanh_ci` can be set to `True` to use the prior arguments and support the prior method of calculating the confidence interval detailed [here]{https://doi.org/10.2514/1.C037976}. This is **not** recommended due to the nuance of choosing appropriate activation function parameters.