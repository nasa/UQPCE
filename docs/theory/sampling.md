
# Sample Generation
The techniques used to generate the samples when the ``--generate-samples`` flag is used are discussed below.

## Inverse Transform Sampling
For continuous and discrete variables that are not the user-input variable, the samples are generated using the respective  distribution, $f$, for $N$ number of samples.

$N$ random, uniformly distributed, well-spaced samples on interval [0, 1] are generated. These samples are input into the distribution's cumulative density function, $CDF$, to acquire samples according to distribution $f$ that are random and well-spaced.

$$
    x = uniform([0, 1], N)
$$

$$
	samples = f.CDF(x)
$$

```{figure} ../images/sample_dist.png
---
height: 400px
name: rej_ssamp_distamp
---
An exponential distribution with vertical black dashed lines at 10% intervals of the $CDF$. The green dots show the sample randomly generated on each interval.
```

Generating the samples according to the above figure ensures that the samples are random and well-spaced.

The ``ContinuousVariable`` class uses this method when the $CDF$ can be solved for. When this fails, the acceptance-rejection sampling method is used.

## Acceptance-Rejection Sampling
This method generates:

* An $x$ value over the variable's support range
* The corresponding $f(x)$
* A $y$ value over range [0, $max(f(x))$]

If $y \leq f(x)$, the $x$ value is accepted. If $y > f(x)$, the value is rejected. This process is repeated until all $N$ samples have been generated.

```{figure} ../images/rej_acc_samp.png
---
height: 400px
name: rej_samp
---
An exponential distribution with green upper triangles as the accepted points and the red upside down triangles as the rejected points.
```

The acceptance-rejection sampling method is often computationally expensive due to the large number of samples that are rejected, and the samples aren't as well-spaced as the inverse transform sampling method.

This is the sampling method used for both sample generation for the run matrix files and the resample values for the ProbabilityBox in the user-input ContinuousVariable.

## Sampling from a Set
This method uses the probabilities associated with a certain value and samples randomly from the given x-values and corresponding probabilities. This method is used for both sample generation for the run matrix files and the resample values for the ProbabilityBox in the user-input DiscreteVariable.