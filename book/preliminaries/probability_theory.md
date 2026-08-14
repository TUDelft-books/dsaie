# Probability Theory

Probability theory underpins every foundational machine learning development we will be treating in this course. We start by defining two simple and intuitive rules from which basically every other concept is derived from.

## Rules of probability

Consider we are picking at random a very large number $N$ of balls from a bag, from which two variables $X$ and $Y$ with possible values $x_i$ and $y_j$ can be observed (for instance the color and material of each ball). 

The grid below shows all possible "bins" our draws can fall into, with **five** possible values for $X$ and **four** possible values for $Y$:

```{figure} ../figures/probgrid.svg
:scale: 50%
:name: probgrid

Two random variables, with possible values on a grid and some sampled observations (gray)
```

You can each gray circle above to be one of the $N$ times we sample $X$ and $Y$. From this figure, we can define some very important concepts:

````{card}
**Joint probability**
^^^
$p(X=x_i, Y=y_j) = \displaystyle\frac{n_{ij}}{N}$

The probability of observing $x_i$ and $y_j$ **together**.
````

````{card}
**Marginal probability**
^^^
$p(X=x_i) = \displaystyle\sum_{j=1}^4 p(X=x_i, Y=y_j) = \displaystyle\frac{c_i}{N}\quad$ The probability of observing $x_i$ **regardless** of $Y$.

$p(Y=y_j) = \displaystyle\sum_{i=1}^5 p(X=x_i, Y=y_j) = \displaystyle\frac{r_j}{N}\quad$ The probability of observing $y_j$ **regardless** of $X$.
````

````{card}
**Conditional probability**
^^^
$p(Y=y_j\vert X=x_i) = \displaystyle\frac{n_{ij}}{c_i}\quad$ The probability of observing $y_j$ **given that** $X=x_i$.

$p(X=x_i\vert Y=y_j) = \displaystyle\frac{n_{ij}}{r_j}\quad$ The probability of observing $x_i$ **given that** $Y=y_j$.
````

Which can all be observed from the shaded regions in the figure above.

From these three observations, we can get to the two crucial rules of probability, namely the sum and the product rules:

````{card}
**Sum rule**
^^^
$p(X) = \displaystyle\sum_Yp(X,Y)$

From the joint, we can marginalize variables by summing over all other variables.
````

````{card}
**Product rule**
^^^
$p(X,Y) = p(Y\vert X)p(X)$

$p(X,Y) = p(X\vert Y)p(Y)$


We can recover the joint probability by combining a marginal and a conditional.
````

```{admonition} Further Reading    
:class: tip    
Read Section 1.2 up until 1.2.1 (pages 12-17). Try to see how the conditional and marginal histograms of Figure 1.11 arise from the points sampled for the two variables $X$ and $Y$ in that figure.    
+++                         
{bdg-danger}`bishop-prml`     
```

## Probability densities

We now move to expressing probabilities over continuous variables. In that case, it does not make sense anymore to have a probability associated with any given value of the variable, as there are infinitely many possibilities.

Instead, we can only compute the probability that a value falls somewhere within a range. To do this, we introduce a **probability density** $p(x)$ and express the probability that $x$ falls within a given interval $[a,b]$ as:

$$
p(x\in[a,b]) = \displaystyle\int_a^bp(x)\,\mathrm{d}x
$$(prelim_probdensity)

Furthermore, probability densities are always subject to the two constraints:

$$
p(x)\geq0
$$(prelim_probconstraint1)

$$
\int_{-\infty}^\infty p(x)\,\mathrm{d}x = 1
$$(prelim_probconstraint2)

which makes intuitive sense: probabilities cannot be negative, and once every possibility is accounted for they should sum up to 1.

The same rules of probability are also valid for probability densities, that is:

````{card}
**Rules of probability for densities**
^^^
$p(x) = \displaystyle\int p(x,y)\,\mathrm{d}y\quad$ Sum rule

$p(x,y) = p(y\vert x)p(x)\quad$ Product rule
````

## Expectations, Monte Carlo approximation

Consider a function $f(x)$ applied to a random variable with probability density $p(x)$. Since the value of $x$ is uncertain, it stands to reason that the value of $f$ should also vary. The average value of $f$ is defined as its **expectation**:

````{card}
**Expectation of a function**
^^^
$\mathbb{E}[f] = \displaystyle\int f(x)p(x)\,\mathrm{d}x$
````

Sometimes we do not have an exact expression for $p(x)$ but we can somehow get samples from it, think for instance of performing a series of experimental measurements. In this case we can approximate the expectation of $f$ in a **Monte Carlo** way:

````{card}
**Monte Carlo approximation**
^^^
$\mathbb{E}[f] \approx \displaystyle\frac{1}{N}\sum_i^N f(x_i)$
````

```{admonition} Further Reading    
:class: tip    
The Monte Carlo approximation is propagating the uncertainty over $x$ forward after the action performed by the function $f$. You can review this and other uncertainty propagation concepts by reviewing the contents of [Week 1.5 of MUDE](https://mude.citg.tudelft.nl/book/2025/propagation_uncertainty/2_Monte_Carlo_simulations.html).
+++  
{bdg-success}`MUDE`
```

## Variances and covariances

Again for a function $f(x)$ of a random variable $x$, it is also useful to compute how spread out the response is around the expected value. This is done by computing its variance:

````{card}
**Variance of a function**
^^^
$\mathrm{var}[f] = \mathbb{E}\left[\left(f(x)-\mathbb{E}\left[f(x)\right]\right)^2\right]$
````

Through a similar argument we can compute the covariance between two random variables $x$ and $y$:

````{card}
**Covariance between two variables**
^^^
$\mathrm{cov}[x,y] = \mathbb{E}\left[(x-\mathbb{E}[x])(y-\mathbb{E}[y])\right]$
````

which expresses how strong the correlation between $x$ and $y$ is. 

If the covariance between $x$ and $y$ is high, we can feel confident in observing only one of them and inferring the other since we can reasonably expect them to change together.

```{admonition} Exam questions    
:class: danger    
In the exam you might see questions with simple derivations using the rules of probability and the concepts of expectation and (co)variance. All the expressions needed for the derivations will be given, you do not need to memorize any of them.   
+++    
{bdg-primary}`written-exam`    
``` 