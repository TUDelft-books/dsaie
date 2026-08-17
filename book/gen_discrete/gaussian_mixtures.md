# Gaussian Mixtures

We will now treat the clustering problem in a **soft** way, and build an unsupervised model with consistent probabilistic foundation. We first set up a Gaussian mixture with $k$ components, and then observe some data with it.

## Setting up a mixture

In contrast to $k$-means, here we want to make our discrete latent variable appear explicitly so we can handle it. Consider the following graph:

```{figure} ../figures/mixgraph1.svg
:scale: 75%
:name: mixgraph1

Graph model for a mixture of Gaussians, with a multinomial latent variable $\bz$.
```

where $\bz$ is a {ref}`categorical variable<categoricaldist>` with a $1$-of-$K$ encoding (as we have seen before in {doc}`k-means<kmeans>` and {doc}`logistic regression<../../3-classification/lectures/logistic_regression>`) whose value indicates a component (i.e. cluster), and $\bx$ is a Gaussian distribution in the original data space. As usual, we can look at the graph and write its joint distribution:

$$
p(\bx,\bz) = p(\bz)p(\bx\vert\bz)
$$(gaussmixjoint)

with $p(\bz)$ being:

$$
p(\bz) = \displaystyle\prod_{k=1}^K\pi_k^{z_k}
$$(gaussmixzprior)

and the conditional Gaussian:

$$
p(\bx\vert\bz) = \displaystyle\prod_{k=1}^K
\gauss\left(\bx\vert\bs{\mu}_k,\bs{\Sigma}_k\right)^{z_k}
$$(gaussmixcond)

and since $z_k$ can only be either zero or one, it just acts as an indicator variable and the conditional is just a conventional multivariate Gaussian. With the usual marginalization we can get an expression for the marginal over $\bx$:

$$
p(\bx) = \displaystyle\sum_{\bz}p(\bz)p(\bx\vert\bz)
=
\sum_{k=1}^K\pi_k\gauss\left(\bx\vert\bs{\mu}_k,\bs{\Sigma}_k\right)
$$(gaussmixmarginal)

which is the classical expression for a **mixture** of Gaussians. We can intuitively look at it as a weighted average of the $K$ Gaussians, where the weights are the mixture coefficients $\pi_k$. We can sample from this density with the technique of {ref}`ancestral sampling<sec-ancestralsampling>` we introduced before: we first sample from $p(\bz)$ and see which $z_k$ is equal to one; then we sample from the Gaussian corresponding to that value of $k$. To get samples from the marginal $p(\bx)$ we do the same and then "forget" which values of $\bz$ are associated to each $\bx$:

```{figure} ../figures/gaussmix0.svg
:figwidth: 750px
:name: "figinit0"

A Gaussian mixture with three components. Joint distribution $p(\bx,\bz)$ on the left, marginal $p(\bx)$ on the right. Note that the resulting mixture is **not Gaussian**!
```

One last thing we can do with this graph is to employ Bayes' Theorem. Suppose we forgot which component $k$ generated a certain $\bx$ (i.e. we do not have its associated $\bz$). We can compute a posterior for $\bz$ as usual:

$$
p(z_k=1\vert\bx) = \displaystyle\frac
{p(z_k=1)p(\bx\vert z_k=1)}
{\sum_{k=1}^Kp(z_j=1)p(\bx\vert z_j=1)}
=
\frac
{\pi_k\gauss\left(\bx\vert\bs{\mu}_k,\bs{\Sigma}_k\right)}
{\sum_{k=1}^K\pi_j\gauss\left(\bx\vert\bs{\mu}_j,\bs{\Sigma}_j\right)}
$$(gaussmixbayes)

For the same samples we drew above, we can plot colors representing these posterior probabilities as follows:

```{figure} ../figures/gaussmix1.svg
:figwidth: 500px

A Gaussian mixture with two components. Posterior over the points after using Bayes' Theorem.
```

which you should compare with the left plot of {numref}`figinit0`. We can therefore have a good idea of which distribution generated each sample just by looking at the data. Of course, some points are a bit unclear (those with a mixed color) since more than one distribution is able to explain it well. The posterior is therefore a measure of how well each Gaussian component of the mixture is able to *explain* a data point. This is why Eq. {eq}`gaussmixbayes` is called the **responsibility** of component $z_k$.

## Observing data

In the above discussion we assume we know both $p(\bz)$ and $p(\bx\vert\bz)$. In practice we only have a few samples of $p(\bx)$ that we would like to cluster, and we have no a priori information of how many Gaussians we need and what their parameters are. Consider for instance the example we treated with {doc}`k-means<kmeans>`:

```{figure} ../figures/gaussmix2.svg
:figwidth: 500px

A set of unlabeled data points in two dimensions.
```

As always we define a new graph that reflects our observations:

```{figure} ../figures/mixgraph2.svg
:scale: 75%
:name: mixgraph2

Graph model for a mixture of Gaussians, with $N$ observations of $\bx$.
```

We therefore have some samples of $\bx$ and would like to calibrate an expression for $p(\bx)$. Assuming a mixture with $K$ components, the straightforward option is to go for a Maximum Likelihood approach. First we define the likelihood of our dataset $\mbf{X}$ (by stacking the $N$ samples of $\bx$ together):

$$
p\left(\mbf{X}\vert\bs{\pi},\bs{\mu},\bs{\Sigma}\right) = \displaystyle
\prod_{n=1}^N
\sum_{k=1}^K
\pi_k\gauss\left(\bx_n\vert\bs{\mu}_k,\bs{\Sigma}_k\right)
$$(likelihood)

The goal is therefore to obtain values for $\pi_k$, $\mu_k$ and $\bs{\Sigma}_k$ that maximize $p(\mbf{X})$. As always we first take the natural logarithm of this density:

$$
\ln p\left(\mbf{X}\vert\bs{\pi},\bs{\mu},\bs{\Sigma}\right) = \displaystyle
\sum_{n=1}^N\ln\left[
\sum_{k=1}^K\pi_k\gauss\left(\bx_n\vert\bs{\mu}_k\bs{\Sigma}_k\right)
\right]
$$(mixloglikelihood)

which turns the product over $N$ into another sum. The MLE estimate is obtained by differentiating this expression with respect to each parameter and setting it to zero:

$$
\displaystyle
\frac{\partial\ln p\left(\mbf{X}\vert\bs{\pi},\bs{\mu},\bs{\Sigma}\right)}{\partial\bs{\mu}_k}
= 0\,\Rightarrow\,\bs{\mu}_k^\mathrm{MLE}\left(\gamma_{nk}\right)
$$(gaussmixmle1)

$$
\frac{\partial\ln p\left(\mbf{X}\vert\bs{\pi},\bs{\mu},\bs{\Sigma}\right)}{\partial\pi_k}
= 0\,\Rightarrow\,\pi_k^\mrm{MLE}\left(\gamma_{nk}\right)
$$(gaussmixmle2)

$$
\frac{\partial\ln p\left(\mbf{X}\vert\bs{\pi},\bs{\mu},\bs{\Sigma}\right)}{\partial\bs{\Sigma}_k}
= 0\,\Rightarrow\,\bs{\Sigma}_k^\mrm{MLE}\left(\gamma_{nk},\bs{\mu}_k^\mrm{MLE}\right)
$$(gaussmixmle3)

We will show the full expressions below, but we already see that the quantities depend on the responsibilities which we now call $\gamma_{nk}$:

$$
\gamma_{nk} = \frac{\pi_k\gauss\left(\bx_n\vert\bs{\mu}_k,\bs{\Sigma}_k\right)}{\sum_j\pi_j\gauss\left(\bx_n\vert\bs{\mu}_j,\bs{\Sigma}_j\right)}
$$(gaussmixgamma)

Look again at the expressions above: all parameters depend on $\gamma_{nk}$, but the responsibilities already depend on $\mu_k$, $\bs{\Sigma}_k$ and $\pi_k$. So there is a circular dependency that makes the problem tricky to treat.

We can actually do this optimization in blocks, with the so-called **Expectation-Maximization (EM)** algorithm. This divides the work in two steps which are quite closely related to the ones we used for $k$-means clustering:

````{card}
Expectation (**E-step**)
^^^
For each data point $n$, evaluate the responsibility each component $k$ of the mixture has in explaining it:

$$
\gamma_{nk} = \frac{\pi_k\gauss\left(\bx_n\vert\bs{\mu}_k,\bs{\Sigma}_k\right)}{\sum_j\pi_j\gauss\left(\bx_n\vert\bs{\mu}_j,\bs{\Sigma}_j\right)}
$$

This is analogous to the cluster assignment step of $k$-means, but the points are only **softly** assigned. With this we also compute:

$$
N_k = \displaystyle\sum_{n=1}^N\gamma(z_{nk})
$$

which is the effective number of samples *"assigned"* to each component.
````

````{card}
Maximization (**M-step**)
^^^
Freeze the responsibilities $\gamma_{nk}$. Compute a Maximum Likelihood Estimate for all other parameters:

$$
\bs{\mu}_k^\mrm{new} = \displaystyle\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})\bx_n
$$

$$
\pi_k^\mrm{new} = \displaystyle\frac{N_k}{N}
$$

$$
\bs{\Sigma}_k^\mrm{new} = \displaystyle\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})\left(\bx_n-\bs{\mu}_k^\mrm{new}\right)\left(\bx_n-\bs{\mu}_k^\mrm{new}\right)^\T
$$

These new values will cause the responsibilities to change. We therefore go for the next iteration, back to the E-step.
````

In other words, in the **E-step** we compute a posterior over $\bz$, while in the **M-step** we maximize the marginal likelihood of the observed data. The steps are repeated until the increase of the marginal $p(\mbf{X})$ is below a tolerance value.

In the following we train a Gaussian mixture with two components to our dataset. The situations after the very first E- and M-steps are shown below:

```{figure} ../figures/gaussmix3.svg
:figwidth: 750px

Fitting a Gaussian mixture with two components to the dataset from before. Situation during the first iteration.
```

Note that we initialize the mixtures quite far away from the data clusters, which give rather inaccurate responsibilities. With those values of $\gamma_{nk}$ we maximize $p(\mbf{X})$ and that changes $\bs{\mu}_k$, $\bs{\pi}_k$ and $\bs{\Sigma}_k$. We keep going for a few more iterations:

```{figure} ../figures/gaussmix4.svg
:figwidth: 750px

Fitting a Gaussian mixture with two components to the dataset from before. Intermediate situation.
```

where we see that the clusters are starting to be corretly identified. Still, the variance of one of the mixtures is still too high, causing the red mixture to partially explain points from the other cluster. With this M-step we can already see it contracting slightly. Running the algorithm until convergence gives:

```{figure} ../figures/gaussmix5.svg
:figwidth: 750px

Fitting a Gaussian mixture with two components to the dataset from before. Situation at convergence.
```

which is a solution that nicely identifies the two data clusters and gives not just a cluster assignment but an uncertainty around it.

```{admonition} Exam questions    
:class: danger    
You might see conceptual questions about Gaussian mixtures on the written exam, as well as simple calculations related to the algorithm you see above. You do not need to memorize the formulas above. The relationship between Gaussian mixtures and the $k$-means algorithm from before is very important to keep in mind.
+++    
{bdg-primary}`written-exam`    
``` 