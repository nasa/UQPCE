
# Analytic Derivatives

In order to use PCE efficiently with optimization through OpenMDAO, UQPCE provides analytical derivatives. This allows users to optimize a system under uncertainty while avoiding the expense penalty of using derivatives with the finite difference method. This section will explain how derivatives for UQPCE are calculated.

## Implementation

The uncertain variables are constant throughout the optimization process and are unaffected by changes in the design variables. The only UQPCE inputs affected by the design variables are the responses that each PCE model is build on. Because of this, we follow the partial derivatives throughout UQPCE with respect to the response vector, $y$, to get the needed derivatives from UQPCE.

The derivative chain for the mean with respect to the responses is given by

$$
\frac{d \mu}{d y} = \frac{\partial \alpha}{\partial y} \cdot \frac{\partial \mu}{\partial \alpha} = ((X^T \cdot X)^{-1} \cdot X^T) \cdot [1, 0, ... , 0]
$$ (uqpce_deriv_mean)

The derivative chain for the variance with respect to the responses is given by

$$
\frac{d \sigma^2}{d y} = \frac{\partial \alpha}{\partial y} \cdot \frac{\partial \sigma^2}{\partial \alpha} = ((X^T \cdot X)^{-1} \cdot X^T) \cdot (2 \alpha \langle \Psi^2 \rangle)
$$ (uqpce_deriv_var)

Because the derivatives on the bound of uncertainty require driving a residual to zero, a framework like OpenMDAO must be used to accurately account for these derivatives. The needed derivatives for the uncertainty interval are given by the below equations

$$
\frac{\partial \alpha}{\partial y} = ((X^T \cdot X)^{-1} \cdot X^T)
$$ (palpha_py)

$$
\frac{\partial y_{resamp}}{\partial \alpha} =  X_{resamp}
$$ (pyresamp_palpha)

$$
\frac{\partial \mathcal{R}_{z}}{\partial y_{resamp}} = \frac{b-a}{2 n \omega \cosh^2(\frac{y_{resamp}-z}{\omega})}
$$ (pRci_pyresamp)

$$
\frac{\partial \mathcal{R}_{z}}{\partial z} = - \sum_{i=1}^{n} \frac{b-a}{2 n \omega \cosh^2(\frac{y_{resamp_{i}} - z}{\omega})}
$$ (pRci_pyci)
