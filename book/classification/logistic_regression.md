# Logistic Regression

After trying out the shortcut of directly applying regression knowledge to classification problems, we come back to our discussion on proper {doc}`decision theory<decision_theory>` for classification in its probabilistic nature. We will see that we can derive a powerful probabilistic model that can conveniently be trained like a deterministic one.

Logistic Regression does not follow the strict generative approach of first inferring $p(\mbf{x}\vert\class)$ and then using Bayes' Theorem to get to the posterior $p(\class\vert\mbf{x})$. Here we will be jumping straight to inferring the posterior. But we shall do that in a careful way that is consistent with the generative approach.

## Back to Bayes

Suppose that we again have two classes $\class_1$ and $\class_2$. Going back to the Bayes' Theorem in Eq. {eq}`bayesclassification` and expanding $p(\mbf{x})$ with the Sum Rule we get for $\class_1$:

$$
p(\class_1\vert\mbf{x}) = 
\displaystyle\frac
{p(\mbf{x}\vert\class_1)p(\class_1)}
{p(\mbf{x}\vert\class_1)p(\class_1) + p(\mbf{x}\vert\class_2)p(\class_2)}
$$(bayestwoclasses)

We now define a quantity $a$ as:

$$
a = \ln\displaystyle\frac
{p(\mbf{x}\vert\class_1)p(\class_1)}
{p(\mbf{x}\vert\class_2)p(\class_2)}
$$(logistica)

Substituting $a$ back into Eq. {eq}`bayestwoclasses` we get:

$$
p(\class_1\vert\mbf{x}) =
\displaystyle\frac
{1}
{1+\exp(-a)}
=
\sigma(a)
$$(bayestwoclassessigmoid)

The function $\sigma(a)$ above is known as the **logistic sigmoid function**. You may remember it from {doc}`frequentist regression<../../1-regression/nn_interactive>`, where it was used as activation function for neural networks, and from {doc}`bayesian linear models<../../2-bayesregression/lectures/linear_models>` where it was used as basis function. The crucial point here is that $\sigma(a)$ is bounded to the interval $[0,1]$ since it represents a valid measure of probability.

## Towards a linear model

The discussion above is still general, which means that $a$ will have a different form depending on how we decide to model $p(\mbf{x}\vert\class)$ and $p(\class)$. If we assume both $p(\mbf{x}\vert\class_k)$ are Gaussian with independent means and a single shared covariance matrix, we get:

$$
a = \mbf{w}^\T\mbf{x} + w_0
\quad\Rightarrow\quad
p(\class_1\vert\mbf{x}) = \sigma\left(\mbf{w}^\T\mbf{x} + w_0\right)
$$(gaussiana)

where $\mbf{w}$ and the bias $w_0$ depend implicitly on the means and shared covariance of the Gaussians. Regardless, $a$ becomes a simple linear model! This means if we jump straight into fitting $\mbf{w}$ and $w_0$ directly from data, we can be sure we are still consistent with the generative approach.

But how do we fit this directly? Remember our posteriors represent a valid probability distribution:

$$
p(\class_1\vert\mbf{x}) = \sigma\left(\mbf{w}^\T\mbf{x} + w_0\right) \equiv y(\mbf{x})
\quad
p(\class_2\vert\mbf{x}) = 1 - p(\class_1\vert\mbf{x}) \equiv 1 - y(\mbf{x})
$$(2classlogisticposteriors)

and following {doc}`decision_theory`, we classify to $\class_1$ if $y(\bx)\geq 0.5$ and to $\class_2$ otherwise. We can use this to define a **likelihood function**: given a dataset $\dataset=\left\{\mbf{x}_n,t_n\right\}_N$ of size $N$ where we set $t_n=1$ for $\class_1$ and $t_n=0$ for $\class_2$, we can easily compute how likely our dataset is:

$$
p(\dataset\vert\mbf{w}) = \displaystyle
\prod_{n=1}^N
y_n^{t_n}\left[1-y_n\right]^{1-t_n}
$$(logisticlikelihood)

Be sure to check this expression by hand and see how the likelihood increases as $y_n$ gets closer to $t_n$ for both classes. As usual, we take the natural logarithm to turn the product into a sum and arrive at an **error function** we can minimize:

$$
E(\mbf{w}) = -\ln p(\dataset\vert\mbf{w}) = \displaystyle
-\sum_{n=1}^N\left[t_n\ln y_n + (1-t_n)\ln(1-y_n)\right]
$$(crossentropy)

where the minus sign appears because we are now trying to **minimize** an error instead of **maximizing** a likelihood. Eq. {eq}`crossentropy` is the famous **cross entropy** error function you will frequently come across when working with classification. We can summarize the approach as follows:

````{card}
**Linear Logistic Regression** (two-class)
^^^
Set $\class_1$ targets to $t=1$ and $\class_2$ targets to $t=0$. Given $\mbf{x}$, our classifier is:

$$
y(\mbf{x})=\sigma(\mbf{w}^\T\mbf{x}+w_0) \quad\Rightarrow\quad
\begin{cases}
\text{if }y(\mbf{x})\geq 0.5, & \mbf{x} \text{ belongs to } \class_1\\
\text{if }y(\mbf{x})<0.5, & \mbf{x} \text{ belongs to } \class_2
\end{cases}
$$

And $\mbf{w}$ and $w_0$ are obtained by minimizing the cross-entropy error function $E(\mbf{w}) = -\ln p(\dataset\vert\mbf{w}) = \displaystyle
-\sum_{n=1}^N\left[t_n\ln y_n + (1-t_n)\ln(1-y_n)\right]$.

The model implicitly fits Gaussian distributions to $p(\bx\vert\class_1)$ and $p(\bx\vert\class_2)$ with shared covariance.
````

## Beyond linearity

The model above has a clear probabilistic interpretation but its **decision boundaries are still linear**. This is however easy to address by using a set of basis functions $\basis(\bx)$. As always we incorporate the bias $w_0$ into $\basis$ and end up with a very similar model as above:

````{card}
**Basis-function Logistic Regression** (two-class)
^^^
Set $\class_1$ targets to $t=1$ and $\class_2$ targets to $t=0$. Given $\basis(\bx)$, our classifier is:

$$
y(\mbf{x})=\sigma(\mbf{w}^\T\basis(\bx)) \quad\Rightarrow\quad
\begin{cases}
\text{if }y(\mbf{x})\geq 0.5, & \mbf{x} \text{ belongs to } \class_1\\
\text{if }y(\mbf{x})<0.5, & \mbf{x} \text{ belongs to } \class_2
\end{cases}
$$

And $\mbf{w}$ are obtained by minimizing the cross-entropy error function $E(\mbf{w}) = -\ln p(\dataset\vert\mbf{w}) = \displaystyle
-\sum_{n=1}^N\left[t_n\ln y_n + (1-t_n)\ln(1-y_n)\right]$.

Now the distributions implicitly fitted for $p(\bx\vert\class)$ are **not Gaussian** anymore.
````

Now the decision boundary is **not linear** and the model is much more flexible (sometimes too flexible, see next section below). One interesting aspect is that the decision boundary is **still linear in the latent space of $\basis$**. In the figure below you see the same trained classifier but plotted in the original $\bx$ space and in the latent space:

```{figure} ../figures/logistic0.svg
:figwidth: 750px
:name: "2classlogisticexample"

The non-linearly separable example from before, now treated with logistic regression and basis functions. Note the shape of the decision boundary in the two spaces (original and latent).
```

Another way to interpret these results is by considering that $\basis$ is *"moving"* the original datapoints to positions that make it possible to create a linear decision boundary that perfectly splits the dataset in two. It is good to remember that after passing through $\basis$, the original points in $\dataset$ occupy different positions in latent space.

## Model selection

Even though logistic regression has sound probabilistic foundations, it is still an MLE model and can therefore suffer from severe overfitting. A Bayesian treatment similar to what we saw for {doc}`regression<../../2-bayesregression/lectures/linear_models>` is not easy to achieve here because $y(\bx,\bw)$ is not linear in $\bw$ anymore. 

The MLE approach is therefore quite popular in practice, but care must be taken to avoid overfitting the training data. Fortunately, we can adopt the same strategies from before in this case, for instance by adding an $L_2$ regularization term to the cross-entropy error function:

$$
E(\mbf{w},\lambda) = -\ln p(\dataset\vert\mbf{w}) {\color{red}+ E_\bw} = \displaystyle
-\sum_{n=1}^N\left[t_n\ln y_n + (1-t_n)\ln(1-y_n)\right] {\color{red}+ \frac{\lambda}{2}\bw^\T\bw}
$$(l2crossentropy)

and as always we calibrate the hyperparameter $\lambda$ by first splitting $\dataset$ into training, validation and test sets and minimizing Eq. {eq}`l2crossentropy` on the **validation set**.

```{admonition} Exam questions    
:class: danger    
You may encounter exam questions about logistic regression, with focus on the conceptual and theoretical aspects we discussed here. You do not need to memorize any mathematical expressions but the main concepts (and how they are arrived at) should be familiar to you.
+++    
{bdg-primary}`written-exam`    
```