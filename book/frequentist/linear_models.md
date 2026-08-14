# Linear Basis Function Models and Regularization

<iframe width="560" height="315" src="https://www.youtube.com/embed/FAyrV8P6Jx0?si=Ox8PTi-qp4t9r1lV" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Linear Basic Function Models
So far, we have been using a non-parametric model with k-nearest neighbors, meaning we needed access to the whole training dataset for each prediction. We will now focus on parametric models, namely linear models with basis functions. Parametric models are defined by a finite set of parameters calibrated in a training step. All we need for a prediction then are the parameter values. There is no longer a need to carry the whole dataset with us; the information used to make predictions is encoded in the model parameters. Once again, we will employ the simple sine function to demonstrate the concepts presented in this chapter.

```{figure} https://github.com/TUDelft-MUDE/source-files/raw/main/file/fig0.png
Ground truth vs noisy data
```

### Linear model

The key idea behind linear regression models is that they are linear in their parameters $\mathbf{w}$. They might be linear in their inputs as well, although this does not necessarily need to case, as we will see later on in this chapter. The simplest approach is to model our target function $y(x)$ as a linear combination of the coordinates $x$:

$$
y(x,\mathbf{w}) = w_0 + w_1 x_1
$$

In the one dimensional case, this is equivalent to fitting a straight line through our datapoints. The parameter $w_0$, also referred to as bias (**not to be confused with the model bias from the previous chapter**), determines the intercept, $w_1$ determines the slope. The introduction of a dummy input $x_0 = 1$ allows us to write the model in a more concise way:

$$
y(x,\mathbf{w}) = w_0 x_0 + w_1 x_1 = \mathbf{w}^T \mathbf{x}
$$

We will use the least squares error function from the previous chapter to fit our model, but will first show how this choice is motivated by a Maximum likelihood approach.

### Maximum Likelihood Estimation

Oftentimes, it is assumed that the target $t$ is given by a deterministic function $y(\mathbf{x}, \mathbf{w})$ with additive Gaussian noise so that

$$
t = y(\mathbf{x}, \mathbf w ) + \epsilon \hspace{0.6cm} \mathrm{with} \hspace{0.6cm} \epsilon \sim \mathcal{N} (0,\beta^{-1}),
$$

with precision $\beta$ (which is defined as $1/\sigma^2$). Note that this assumption is only justified in case of a unimodal conditional distribution for $t$. It can have grave influence on the model accuracy and its validity therefore must be carefully assessed.

As seen in the previous chapter, for the square loss function, the optimal prediction for some value $\mathbf{x}$ is given by the conditional mean of the target variable $t$. In this case, this conditional mean is given by:

$$
\mathbb{E}[t|\mathbf{x}] = \int t p(t|\mathbf{x}) \mathrm{d}t = y(\mathbf{x},\mathbf{w}).
$$

Given a new input $\mathbf{x}$, the distribution of $t$ that follows from our model is

$$
p (t | \mathbf{x}, \mathbf{w}, \beta) = \mathcal{N} (t | y (\mathbf{x}, \mathbf{w}), \beta^{-1}).
$$

Consider now a dataset $\mathcal{D}$ consisting of inputs $\mathbf{X} = \{ \mathbf{x}_1, \dots, \mathbf{x}_n \}$ and targets $\mathbf{t} = \{  t_1, \dots, t_n \}$. Assuming our datapoints are drawn independently from the same distribution (*i.i.d. assumption*), the likelihood of drawing this dataset from our model is

$$
p( \mathcal{D}|\mathbf{w}) =  \prod_{n = 1}^N \mathcal{N} (t_n | y (\mathbf{x}_n, \mathbf{w}), \beta^{-1}), 
$$

also referred to as the likelihood function. Taking the logarithm and expanding on the classic expression for a multivariate Gaussian distribution gives:

$$
\mathrm{ln} \, p( \mathcal{D}|\mathbf{w}) = \sum_{n = 1}^{N} \mathrm{ln} \, \mathcal{N} ( t_n | \mathbf{w}^T \mathbf{x}_n, \beta^{-1}) = \frac{N}{2} \mathrm{ln} \beta - \frac{N}{2}\mathrm{ln}(2 \pi) - \beta \, \underbrace{\frac{1}{2} \sum_{n = 1}^{N} ( t_n - \mathbf{w}^T \mathbf{x}_n)^2}_{E_\mathcal{D}}
$$

where we can identify our square-error loss function in the last term. Note that the first two terms are constant for a given dataset and have no influence on the parameter setting $\bar{\mathbf{w}}$ that maximizes the likelihood. Those optimal parameters values $\bar{\mathbf{w}}$ can be obtained by setting the gradient of our loss function w.r.t. $\mathbf{w}$ to zero and solving for $\mathbf{w}$.

$$
\nabla_{\mathbf{w}} E_{\mathcal{D}} = \frac{1}{N} \sum_{n=1}^N \left(t_n - \mathbf{w}^T \mathbf{x}_n \right) \mathbf{x}_n  \stackrel{!}{=} 0
$$

It is convenient to concatenate all inputs to a *design matrix* $\mathbf{X} = [\mathbf{x}_1^T, ..., \mathbf{x}_N^T]^T$. Solving for $\mathbf{w}$ gives

$$
\bar{\mathbf{w}} = \left(\mathbf{X}^T \mathbf{X} \right)^{-1} \mathbf{X}^T \mathbf{t}
$$

which is the classical expression for a least-squares solution you have by now seen many times during the course. 

### Data normalization

Before we move on to fitting the model, we normalize our data. **This step is recommended for most machine learning techniques and is often even necessary.** A non-normalized dataset almost always leads to a numerically more challenging optimization problem. Part of model selection is to ensure that our basis functions show the desired behavior in the relevant parts of the domain. We center and rescale our data, a process referred to as standardization, to operate in the vicinity of the origin only. The standardized dataset $\hat{\mathcal{D}} = ( \hat{x}, \hat{t} )$ is obtained by subtracting the sample mean $\mu$, and dividing by the sample standard deviation $\sigma$ of the data:

$$
\hat{x} = \frac{x - \mu_x}{\sigma_x}  \quad \mathrm{and} \quad \hat{t} = \frac{t - \mu_t}{\sigma_t}.
$$

Take a look below at the standardized and unstandardized data. Note that the standardization of the target $t$ has a marginal effect, as the sine function is already centered at 0 and almost shows a standard deviation of 1. A 4 by 4 square has been added to indicate the region from $-2 \hat{\sigma}$ to $2 \hat{\sigma}$. As you can see, all input and output variables fall roughly in this interval, and this property allows for more stability when applying numerical solvers to the problem.

Note that **there is no strictly correct way to shift and scale input data**. Depending on the distribution of the data, a min-max scaling or a quantile scaling might lead to a better numerical setup. The dataset's structure needs to be assessed carefully to make an informed decision on normalization.

```{figure} https://github.com/TUDelft-MUDE/source-files/raw/main/file/fig3.png
Unnormalized and normalized data
```

````{admonition} Coding in python
Different packages will offer distinct functionalities to perform the normalization. In the case of [`scikit-learn`](https://scikit-learn.org/stable/), the package [`sklearn.preprocessing`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) offers an wide range of normalizers:

- `sklearn.preprocessing.StandardScaler` implements the normalization scheme from our linear models page, based on sample mean and standard deviation;
- `sklearn.preprocessing.MinMaxScaler` uses dataset bounds to scale each feature based on its minimum and maximum value across the complete training dataset. This makes each separate input feature fall within a $[0,1]$ interval;
- `sklearn.preprocessing.MaxAbsScaler` scales each feature based only on their maximum value across the dataset.
- `sklearn.preprocessing.RobustScaler` implements a specialized normalization scheme that makes the trained models less sensitive to the presence of outliers in the training dataset

Different normalizers are more suitable for different applications. It is usual to experiment with different normalizers when looking at a new problem.
````

### Generalized linear models

Let us first train the model above on our nonlinear dataset:

```{figure} https://github.com/TUDelft-MUDE/source-files/raw/main/file/fig4.png
Linear prediction
```

It is clear from the plot that a linear model with linear features lacks the flexibility to fit the data well. A bias-variance decomposition analysis for this model would show that it has little variance but shows a strong bias. We now consider nonlinear functions of the input $x$ as features/regressors to increase the flexibility of our linear model. A common approach is to use a set of polynomial basis functions,

$$
\phi_j(x) = x^j,
$$

but numerous other choices are possible. The full formulation for a model with $M$ polynomial basis functions is thus

$$
y(x,\mathbf{w}) = w_0 x^0 + w_1 x^1 + w_2 x^2 + ... + w_M x^M,
$$

which shows how the model is still linear w.r.t. $\mathbf{w}$, even though it is no longer linear in the input parameters. The design matrix for this more general case reads

$$
\Phi_{ij} = \phi_j(x_i).
$$

As was the case for $\mathbf{X}$, we need to ensure that $\boldsymbol \Phi$ has full column rank. This is not always the case the case, for example if we have more basis functions that data points, or if our basis functions are not linearly independent.

```{figure} https://github.com/TUDelft-MUDE/source-files/raw/main/file/fig5.png
Polynomial basis functions
```

We obtain the linear model with nonlinear basis functions by replacing the coordinate vector $x$ with the  feature vector $\boldsymbol{\phi}(x)$

$$
y(x,\mathbf{w}) = \sum_{j=0}^M w_j \phi_j(x) = \mathbf{w}^T \boldsymbol{\phi} (x).
$$

The solution procedure remains the same, and we can solve for $\bar{\mathbf{w}}$ directly

$$
\bar{\mathbf{w}} = \left(\boldsymbol{\Phi}^T \boldsymbol{\Phi} \right)^{-1} \boldsymbol{\Phi}^T \mathbf{t}.
$$

Let's take a look at the linear model with polynomial regressors.

```{figure} https://github.com/TUDelft-MUDE/source-files/raw/main/file/fig6.png
Generalized linear regression for polynomials of degree $p$
```

That is looking much better already. However, the quality of the fit varies significantly with the degree of the polynomial basis. There seems to be an ideal model complexity for this specific problem. Try out the interactive tool below to get an idea of the interplay of the following variables:
- $p$, the degree of the polynomial basis
- $N$, the size of the training data set
- $freq$, the frequency of the underlying truth
- $\varepsilon$, the level of noise associated with the data
- The seed can be updated to generate new random data sets
- The truth can be hidden to simulate a situation that is closer to a practical setting

<iframe src="../_static/linear-models.html?mode=polynomial" width="100%" height="720" frameborder="0"></iframe>

A few questions that might have crossed your mind when playing with the tool:

- With a small amount of data ($N \leq 11$), what happens if we have as many data points as parameters? $(p + 1 = N)$

````{admonition} Answer
:class: dropdown
The model fits all data points exactly, since the least-squares problem is neither over- nor underdetermined. Note how the resulting model looks nothing like the ground truth, and how interpolating between data points usually gives low-quality predictions. These are signs of overfitting. Try it out for different noise levels to see something interesting!
````

- With a small amount of data ($N \leq 11$), what happens if we have more model parameters than data? $(p + 1 > N)$

````{admonition} Answer
:class: dropdown
The system is now underdetermined and therefore has no unique solution. The least-squares problem is also nearly singular and prone to numerical issues.
````

- We only have access to data in the interval $[0,2\pi]$. How well does our model extrapolate beyond the data range?

````{admonition} Answer
:class: dropdown
Unsurprisingly, higher-order models do not extrapolate well at all. In general it is a bad idea to use machine learning models in extrapolation. But try to increase the dataset size and select a more parsimonious model with lower $p$ and see what happens in extrapolation. There is still a lot we can do to carefully craft models that at the very least do not unreasonably deviate from the trends on the dataset when extrapolating.
````

### Other choices of basis functions

As mentioned previously, the polynomial basis is just one choice among many to define our model. Depending on the problem setting, a different set of basis functions might lead to better results. Another popular choice is the radial basis functions (also called Gaussian basis functions), given by

$$
\phi_j(x) = \exp\left\{-\frac{(x-\mu_j)^2}{2\ell^2}\right\} \quad \mathrm{for} \quad j=1,\dots,M
$$

where $\phi_j$ is centered around $\mu_j$, $l$ determines the width, and $M$ refers to the number of basis functions.

```{figure} https://github.com/TUDelft-MUDE/source-files/raw/main/file/fig8.png
Radial basis functions
```

One of the attributes of this model is the locality of its individual functions. This means data in one part of the domain will not impact predictions in other parts of the domain. Periodicity can be achieved with a Fourier basis. Wavelets are popular in signal processing since they are localized in both frequency and space. **It is up to the user to determine which basis function properties are desired for a given problem**, and this is an important part of model selection.

Let's see how well the linear model with radial basis functions performs on the sine wave problem. Keep in mind that the lengthscale parameter corresponds to the lengthscale in the standardized space.

```{figure} https://github.com/TUDelft-MUDE/source-files/raw/main/file/fig9.png
Generalized linear regression for radial basis functions with varying $M$ and $l$
```

The figure above shows four different combinations of the hyperparameters (number of basis functions and length scale). The quality of the fit depends strongly on the parameter setting, but a visual inspection indicates our model can replicate the general trend.

## Regularization 

We've seen that the least squares approach can lead to severe over-fitting if complex models are trained on small datasets. Our strategy of limiting the number of basis functions a priori to counter overfitting puts us at risk of missing critical trends in the data. Another way to control model flexibility is by introducing a regularization term. This additional term essentially puts a penalty on the weights and prevents them from taking too large values unless supported by the data. The concept of regularized least squares will be demonstrated on the usual sine function.

```{figure} https://github.com/TUDelft-MUDE/source-files/raw/main/file/fig12.png
Noisy data generated from a ground-truth sine wave
```

We extend the data-dependent error $E_D(\bf{w})$ with the regularization term $E_W(\bf{w})$:

$$
E (\mathbf{w}) = E_D (\mathbf{w}) + \lambda E_W (\mathbf{w})
$$

with regularization parameter $\lambda$ that controls the relative importance of two terms comprising the error function. A common choice for the regularizer is the sum-of-squares:

$$
E_W(\mathbf{w}) = \frac{1}{2} \mathbf{w}^T \mathbf{w}.
$$

The resulting model is known as L<sub>2</sub> regularization, ridge regression, or weight decay. The total error therefore becomes

$$
E (\mathbf{w}) = \frac{1}{2} \sum_{n = 1}^{N} \left( t_n - \mathbf{w}^T \boldsymbol{\phi} (x_n) \right)^2 + \frac{\lambda}{2} \mathbf{w}^T \mathbf{w}.
$$

A useful property of the L<sub>2</sub> regularization is that the error functions is still a quadratic function of $\mathbf{w}$, allowing for a closed form solution for its minimizer. Setting the gradient of the regularized error function with respect to $\mathbf{w}$ to $0$ gives

$$
\bar{\mathbf{w}} = \left(\boldsymbol{\Phi}^T \boldsymbol{\Phi} + \lambda \mathbf{I} \right)^{-1} \boldsymbol{\Phi}^T \mathbf{t}.
$$

### Determining the regularization parameter

We can use the formulation above in an interactive plot where you can control the dataset size, number of radial bases and the magnitude of the regularization parameter $\lambda$:

<iframe src="../_static/linear-models.html?mode=regularization" width="100%" height="700" frameborder="0"></iframe>

Play around with the number of radial basis functions and the regularization parameter. Try to answer the following questions:

- What happens for a low number of RBF, and what happens for a high number of RBF?

````{admonition} Answer
:class: dropdown
As you might expect, more radial basis functions means a more complex model, and therefore more prone to overfitting.
````

- How does this affect the value of $\lambda$ that yields the (visually) best fit?

````{admonition} Answer
:class: dropdown
Models with high variance will generally require more regularization to reach the same level of bias. So increasing the number of RBFs should require a higher value of $\lambda$ in order to reach the same level of model complexity.
````

- How does a larger dataset affect the optimal regularization parameter $\bar{\lambda}$?

````{admonition} Answer
:class: dropdown
As we increase $N$ and therefore let the model see more data, it becomes more tolerant to a higher range of values of $\lambda$. That is, even very low values of $\lambda$ will not lead to overfitting, as the data is now there to naturally limit model complexity. Again, overfitting is a problem which is much worse when we do not have a lot of data.
````

The pressing question is, of course: **how do we determine $\lambda$?** 

The problem of controlling model complexity has been shifted from choosing a suitable set of basis functions to determining the regularization parameter $\lambda$. **Optimizing for both $\mathbf{w}$ and $\lambda$ over the training dataset will always favor flexibility, leading to the trivial solution $\lambda = 0$** and, therefore, to the unregularized least squares problem. Instead, similar to what we did before, the data should be partitioned into a training set and a validation set. The training set is used to determine the parameters $\mathbf{w}$, and the validation set to optimize the model complexity through $\lambda$. 

For the visualization below we use a pragmatic approach and sample another dense validation set from our ground truth instead of splitting the original training dataset. We then loop over several values of $\lambda$ until we find the one corresponding to the minimum validation error. This is then our selected model. This leads to a best fit for $\lambda = 0.048301 $ with $ \text{MSE} = 0.267945$ 

```{figure} https://github.com/TUDelft-MUDE/source-files/raw/main/file/fig15.png
Minimum validation error for multiple values of $\lambda$ 
```

We then take a look at how the optimally regularized model looks like:

```{figure} https://github.com/TUDelft-MUDE/source-files/raw/main/file/fig16.png
Optimally regularized model
```

That looks much better than the solution of the unregularized problem. Luckily, this problem has a unique solution $\bar{\lambda}$. Can you explain why the error first decreases but keeps growing again at some point? Do you always expect this non-monotonic characteristic when looking at the **training** error as a function of the regularization parameter?