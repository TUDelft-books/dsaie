# Probabilistic PCA

Here we go from discrete to continuous latent variables. Now we assume our dataset follows a set of hidden patterns governed by a continuous latent variable $\bz$. In other words we assume that our data lies on a lower-dimensional continuous subspace that lives in the original data space (e.g. a line within a 2D space, a plane within a 3D space). Whatever is not explained by this subspace we attribute to noise in our dataset. We have motivated the need for continuous latents when introducing {doc}`unsupervised learning <../gen_discrete/unsupervised_learning>`.

We start with a latent variable model with **linear** mappings between data space and latent space. This leads to the so-called Principal Component Analysis (PCA) technique. We start right where we left off with Gaussian mixtures.

We start by just taking the graph model for a {doc}`gaussian mixture <../gen_discrete/gaussian_mixtures>` and making the latent variable $\bz$ **continuous**:

```{figure} ../figures/pcagraph1.svg
:scale: 75%
:name: pcagraph1

Graph model for PPCA, with a continous latent variable $\bz$.
```

and now instead of having a categorical distribution, we assume the latent $p(\bz)$ is Gaussian. This leads to a probabilistic model defined by the joint:

$$
p(\bx,\bz\vert\bs{\mu},\mbf{W},\sigma) = p(\bz)p(\bx\vert\bz,\bs\mu,\mbf{W},\sigma)
$$(ppca6)

and a combination of prior and conditional:

$$
 p(\bz) = \gauss\left(\bz\vert\mbf{0},\mbf{I}\right)
 \quad\quad
  p(\bx\vert\bz) = \gauss\left(\bx\vert\mbf{W}\bz + \bs\mu,\sigma^2\mbf{I}\right)
$$(ppca7)

where we define a $D\times M$ matrix $\mbf{W}$ that linearly decodes a lower-dimensional latent $\bz$ to the real data space, with added $\sigma^2$ noise. Using the standard expressions for {ref}`Bayes' Theorem with Gaussians <bayes-stdexpressions>`, we can get to a marginal over $\bx$:

$$
p(\bx) = \gauss\left(\bx\vert\bs\mu,\mbf{W}\mbf{W}^\T+\sigma^2\mbf    {I}\right)
$$(ppca8)

and a posterior conditional for $\bz$:

$$
p(\bz\vert\bx) = \gauss\left(\bz\vert\mbf{M}^{-1}\mbf{W}^\T\left(\bx-\bs\mu\right),\sigma^2\mbf{M}^{-    1}\right)\quad\text{where}\quad\mbf{M} = \mbf{W}^\T\mbf{W}+\sigma^2\mbf{I}
$$(ppca9)

The model is therefore a **constrained** version of the Gaussian distribution in the original $\bx$ space. It is also interesting to see how both $p(\bz)$ and $p(\bx\vert\bz)$ have diagonal covariances but the marginal $p(\bx)$ has a more complex correlation structure. Furthermore, by properly training the PPCA model we guarantee that the $M$ **most important** correlations between dimensions of $\bx$ are well captured while all remaining correlations are approximated by the isotropic noise in $p(\bx\vert\bz)$.

Note that PPCA also creates **a generative encoder-decoder** system: data can be mapped back and forth between the $\bx$ and $\bz$ spaces:

```{figure} ../figures/pcamappings.svg
:scale: 75%
:name: pcamappings

Forward and inverse mappings learned enabled by a PPCA model. All mappings are linear and all resultant distributions are Gaussian.
```

Be sure to check by yourself that these expressions are correct!

## Observing data

To train a PPCA model we observe samples from $p(\bx)$, just as we did for training Gaussian mixtures:

```{figure} ../figures/pcagraph2.svg
:scale: 75%
:name: pcagraph2

Graph model for PPCA, with $N$ observations of $\bx$.
```

Each sample $\bx_n$ has an associated latent variable $\bz_n$ which we of course cannot observe. Stacking all $\bx_n$ and $\bz_n$ into matrices, we have the following joint distribution:

$$
p(\mbf{X},\mbf{Z}) = p(\mbf{Z})p(\mbf{X}\vert\mbf{Z})
$$(ppca10)

Gathering all the variables we need to fit in a set $\bs\theta = \left[\mbf{W},\bs\mu,\sigma^2\right]$, using Bayes' Theorem and taking the natural logarithm on both sides we can state:

$$
\ln p(\mbf{X}\vert\bs\theta) = \ln p(\mbf{X},\mbf{Z}\vert\bs\theta) - \ln p(\mbf{Z}\vert\mbf{X},\bs\theta)
$$(ppca11)

which is the likelihood of our dataset, and therefore the quantity we would like to maximize. 

## The Expectation-Maximization algorithm

To make progress, we can follow the same approach as for the mixtures and come up with an EM solution for this problem. As with the mixtures, optimizing for $\bs\theta$ would be much easier if we knew $\mbf{Z}$ but we cannot observe them. We **also** do not know $p(\mbf{Z}\vert\mbf{X})$ because we do not know the parameters $\bs\theta$. Here we can only compute $p(\mbf{Z}\vert\mbf{X},\bs\theta)$, but that will only be the correct posterior if $\bs\theta$ is correct.

Let us therefore introduce an approximation $q(\mbf{Z})$ for $p(\mbf{Z}\vert\mbf{X})$. When $\bs\theta$ is correct and we compute the correct posterior, $q(\mbf{Z})=p(\mbf{Z}\vert\mbf{X})$, but that will only happen when we converge. We add and subtract it from the equation above:

$$
\ln p(\mbf{X}\vert\bs\theta) = \ln p(\mbf{X},\mbf{Z}\vert\bs\theta) - \ln p(\mbf{Z}\vert\mbf{X},\bs\theta) + \ln q(\mbf{Z}) - \ln q(\mbf{Z})
$$(ppca17)

We then rearrange the terms and take the expectation with respect to $q(\mbf{Z})$ on both sides:

$$
\displaystyle\ln p(\mbf{X}\vert\bs\theta) = \underbrace{\int_\mbf{Z} q(\mbf{Z})\ln\left[\frac{p(\mbf{X},\mbf{Z}\vert\bs\theta)}{q(\mbf{Z})}\right]\,\mrm{d}\mbf{Z}}_{\mathcal{L}(q,\bs\theta)} - \underbrace{\int_\mbf{Z}q(\mbf{Z})\ln\left[\frac{p(\mbf{Z}\vert\mbf{X},\bs\theta)}{q(\mbf{Z})}\right]\,\mrm{d}\mbf{Z}}_{\mrm{KL}(q\Vert p)}
$$(ppca18)

where we have used the fact that $\int_\mbf{Z} q(\mbf{Z}) \ln p(\mbf{X}\vert\bs\theta)\,\mrm{d}\mbf{Z}=\ln p(\mbf{X}\vert\bs\theta)$ since it does not depend on $\mbf{Z}$.

The second term in the right-hand side is the so-called **Kullback-Leibler Divergence**, a measure of similarity between two probability densities. The closer $q(\mbf{Z})$ is to $p(\mbf{Z}\vert\mbf{X},\bs\theta)$ the lower the value of $\mrm{KL}(q\Vert p)$ will be, with a minimum of zero when the two distributions are identical. Note that $\mrm{KL}(q\Vert p)$ can never be negative.

Given this property of the KL divergence, the term $\mathcal{L}(q,\bs\theta)$ must therefore be a lower bound for $p(\mbf{X}\vert\bs\theta)$. We therefore call it the **Evidence Lower Bound (ELBO)**. With this we can finally derive the EM algorithm for this problem. Remember that our goal is to **maximize the evidence $p(\mbf{X}\vert\bs\theta)$**, and since $p(\mbf{X}\vert\bs\theta)$ is at least as high as $\mathcal{L}(q)$ (its lower bound), we can just maximize $\mathcal{L}(q,\bs\theta)$ instead:

````{card}
Expectation (**E-step**)
^^^
Fix the model parameters $\bs\theta= \bs\theta_\mrm{old}$.

Maximize $\mathcal{L}(q,\bs\theta)$ with respect to $q(\mbf{Z})$. Since $p(\mbf{X}\vert\bs\theta_\mrm{old})$ depends **only** on $\bs\theta$ and that is fixed, the only freedom left is to do:

$$
p(\mbf{X}\vert\bs\theta_\mrm{old}) = \mathcal{L}(q,\bs\theta_\mrm{old}) \Leftrightarrow \mrm{KL}(q||p) = 0 \Rightarrow q_\mrm{new}(\mbf{Z}) = p(\mbf{Z}\vert\mbf{X},\bs\theta_\mrm{old})
$$

For PPCA this means computing the posteriors:

$$
p(\mbf{z}_n\vert\mbf{x}_n) = \gauss\left(\mbf{z}_n\vert\mbf{M}^{-1}\mbf{W}^\T(\mbf{x}-\bar{\bx}),\sigma^2\mbf{M}^{-1}\right)\quad\text{with}\quad\mbf{M} = \mbf{W}^\T\mbf{W} + \sigma^2\mbf{I}
$$

and since $\mathcal{L}$ involves an expectation over $\mbf{Z}$, we can already compute in anticipation of the M-step:

$$
\mathbb{E}[\bz_n] = \mbf{M}^{-1}\mbf{W}^\T(\bx_n-\bar{\bx})    
          \quad\quad    
          \mathbb{E}[\bz_n\bz_n^\T] = \sigma^2\mbf{M}^{-1} + \mathbb{E}[\bz_n]\mathbb{E}[\bz_n]^\T 
$$
````

````{card}
Maximization (**M-step**)
^^^
Fix the approximation $q(\mbf{Z})$.

Maximize $\mathcal{L}(q,\bs\theta)$ with respect to $\bs\theta$:

$$
\bs\theta_\mrm{new} = \underset{\overline{\bs\theta}}{\mrm{arg\,max}}\,\,\mathcal{L}(q_\mrm{new},\overline{\bs\theta})
$$

For PPCA this means computing new estimates for $\mbf{W}$ and $\sigma^2$:

$$
\mbf{W}_\mrm{new} = \displaystyle\left[\sum_{n=1}^N\left(\bx_n-\bar{\bx}\right)\mathbb{E}[\bz_n]^\T\right]\left[\sum_{n=1}^N\mathbb{E}[\bz_n\bz_n^\T]\right]^{-1}
$$

$$
\sigma^2_\mrm{new} = \displaystyle\frac{1}{ND}\sum_{n=1}^N\left[\lVert\bx_n-\bar{\bx}\rVert^2-2\mathbb{E}[\bz_n]^\T\mbf{W}_\mrm{new}^\T(\bx_n-\bar{\bx}) + \mrm{tr}\left(\mathbb{E}[\bz_n\bz_n^\T]\mbf{W}_\mrm{new}^\T\mbf{W}_\mrm{new}\right)\right]
$$

For PPCA, the mean $\bs\mu$ has a trivial solution:

$$
\bs\mu = \bar{\bx}
$$
````

Compare these two steps to what we did for {doc}`Gaussian mixtures<../gen_discrete/gaussian_mixtures>` and see how they are conceptually exactly the same. Compare them also with what we called E-step and M-step for {doc}`k-means clustering<../gen_discrete/kmeans>`. With this the whole content of this week comes together in a satisfying way.

Below is an example of applying PPCA to a two-dimensional dataset with a clear linear underlying pattern. Starting from randomized $D\times M$ weights $\mbf{W}$ and unit variance $\sigma^2$, we let the EM algorithm find the underlying linear correlation between $x_1$ and $x_2$:

```{figure} ../figures/ppca0.svg
:figwidth: 750px

Fitting a PPCA model from 2D to 1D with the EM algorithm. Initial situation
```

```{figure} ../figures/ppca1.svg
:figwidth: 750px

Fitting a PPCA model from 2D to 1D with the EM algorithm. Situation after convergence
```

Also note that in plotting the linear subspace we go through $\bz\in[-3,3]$ and transform it back to $\bx$ using $\mbf{W}$. This allows us to also plot $\bz$ with a colorbar.

## Generating samples

Since we made the latents $\bz$ probabilistic, PPCA can be used as a generative model. In the same way as we did for mixtures, we can generate fake data using ancestral sampling:

- Draw a latent sample $\widetilde{\bz}$ from the prior $p(\bz)=\gauss(\mbf{0},\mbf{I})$
- Transform it back to a sample in real space: $\bx=\mbf{W}\widetilde{\bz} + \bar{\bx}$
- Add some Gaussian noise sampled from $\gauss(\mbf{0},\sigma^2\mbf{I})$

Note how the two last steps together correspond to sampling from $p(\bx\vert\bz)$. You can find some convincing generations below:

```{figure} ../figures/ppca2.svg
:figwidth: 750px

Generating samples from a 2D to 1D PPCA.
```

```{figure} ../figures/ppca3.svg
:figwidth: 750px

Generating samples from a 2D to 1D PPCA, one more example.
```

```{figure} ../figures/ppca4.svg
:figwidth: 750px

Generating samples from a 2D to 1D PPCA, one last example.
```

```{admonition} Exam questions    
:class: danger    
You might find conceptual questions about PPCA in the exam, with focus on its relation to clustering and mixtures. Differences between PCA and PPCA are also important, as is the generative nature of PPCA. You will not be asked to derive the EM algorithm by hand, but the conceptual roles of the two right-hand side terms in Eq. {eq}`ppca18` are important. You might also see simple numerical questions, in which case you will not be asked to recall any formulas by heart.
+++    
{bdg-primary}`written-exam`    
```

## Extra: Deterministic PCA

In the deterministic view we look at projections of our original dataset (in $D$ dimensions) into a lower-dimensional subspace with dimensionality $M\ll D$. For PCA, we assume this subspace is **linear**, i.e. a line for $M=1$, a plane for $M=2$ or a hyperplane for $M>2$. To find the best possible subspace, we look at the projected data and either maximize its variance or minimize its reconstruction error. Both of these approaches lead to **the same optimum subspace**. We will briefly go through these options in the following. 

### Maximum variance

The goal is to find the subspace $\mbf{U}$ such that when we project our data to the subspace we obtain a dataset with the highest possible variance. Let the sample mean and sample covariance be defined by:

$$
\bar{\bx} = \displaystyle\frac{1}{N}\sum_{n=1}^N\bx_n
\quad\quad
\mbf{S} = \displaystyle\frac{1}{N}\sum_{n=1}^N\left(\bx_n-\bar{\bx}\right)\left(\bx_n-\bar{\bx}\right)^\T
$$(ppca1)

Projecting a data point $\bx$ onto the subspace means doing:

$$
 \bz = \left(\bx-\bar{\bx}\right)\mbf{U}
$$(ppca2)

where $\mbf{U}$ is a $D\times M$ matrix that defines the linear subspace. We then look at the projected data points:

```{figure} ../figures/pcavariance.svg
:scale: 75%
:name: pcavariance

Optimum Principal Component Analysis (PCA) subspace, maximizing variance of the projected dataset.
```

and try to find $\mbf{U}$ that maximizes the spread between the points. The solution is:

$$
\mbf{U} = \mrm{eigenvectors}\left(\mbf{S}\right)
$$(ppca3)

which makes the model quite simple. We just use one of many possible algorithms to find the eigenvectors of $\mbf{S}$ and we are done.

### Minimum error

The projected points can also be **brought back** into the original space. This leads to a **reconstructed** data point $\widetilde{\bx}$, and since $\mbf{U}$ is an orthogonal matrix we can arrive at:

$$
\widetilde{\bx}=\mbf{U}\bz + \bar{\bx}
$$(ppca4)

We can then compare $\widetilde{\bx}$ with its original version $\bx$, and it stands to reason that we should pick the subspace $\mbf{U}$ that minimizes this error we make when projecting to $\bz$ and back:

```{figure} ../figures/pcaerror.svg
:scale: 75%
:name: pcaerror

Optimum Principal Component Analysis (PCA) subspace, minimizing the error between original and projected datasets.
```

In other words, we minimize a distortion measure:

$$
J = \displaystyle\frac{1}{N}\sum_{n=1}^N\lVert\bx-\widetilde{\bx}\rVert^2
         \quad\Rightarrow\quad
         \mbf{U} = \underset{\overline{\mbf{U}}}{\mrm{arg}\,\mrm{min}}\,\,J\left(\overline{\mbf{U}}\right)
           \quad\Rightarrow\quad
           \mbf{U} = \mrm{eigenvectors}\left(\mbf{S}\right)
$$(ppca5)

So we get the same answer as before, a very satisfying result!

```{admonition} Further Reading    
:class: tip    
Here we skipped a few steps when getting to the final solutions for $\mbf{U}$. If you are curious you can check Sections 12.1.1 and 12.1.2 for more details.
+++                         
{bdg-danger}`bishop-prml`     
```

## Extra: PPCA with direct MLE

Just as with Gaussian mixtures, we can maximize $p(\mbf{X})$ in one go instead of using EM. Since we still have a simple linear-Gaussian system this is still possible, and will lead to a familiar result. Splitting $\ln p(\mbf{X})$ into a sum over the $N$ samples we can write:

$$
\ln p(\mbf{X}\vert\bs\theta) = \displaystyle - \frac{ND}{2}\ln(2\pi) - \frac{N}{2}\ln\lvert\mbf{C}\rvert - \frac{1}{2}\sum_{n=1}^N\left(\bx_n-\bs\mu\right)^\T\mbf{C}^{-1}\left(\bx_n-\bs\mu\right)
$$(ppca12)

with

$$
\mbf{C} = \mbf{W}\mbf{W}^\T + \sigma^2\mbf{I}
$$(ppca13)

We can maximize Eq. {eq}`ppca12` directly, giving:

$$
\bs\mu=\bar{\bx}
$$(ppca14)

$$
\mbf{W}_\mrm{MLE} = \mbf{U}_M\left(\mbf{L}_M-\sigma^2\mbf{I}\right)^{1/2}
$$(ppca15)

$$
\sigma^2_\mrm{MLE} = \displaystyle\frac{1}{D-M}\sum_{i=M+1}^{D}\lambda_i
$$(ppca20)

with $\mbf{U}_M$ being a $D\times M$ matrix with the $M$ first eigenvectors of $\mbf{S}$ and $\mbf{L}_M$ is a diagonal matrix with the first (highest) corresponding eigenvalues $\lambda_i$. We can also see that the variance coming from all remanining eigenvectors is just explained through the isotropic noise $\sigma^2_\mrm{MLE}$.

We therefore see a clear link with deterministic PCA, and it can be shown that PPCA collapses back into PCA when $\sigma^2\rightarrow 0$.

```{admonition} Further Reading    
:class: tip    
You can find an extended discussion on MLE for PPCA in Section 12.2.1, including the complementary roles of $\mbf{W}$ and the noise $\sigma^2$ in explaining correlations in the data.
+++                         
{bdg-danger}`bishop-prml`     
``` 