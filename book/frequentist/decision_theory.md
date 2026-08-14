# Decision Theory

## Introduction

<iframe width="560" height="315" src="https://www.youtube.com/embed/7JHvoZfSFBY?si=9UvxfuVizLSxkBAy" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

In the previous chapter, we built a k-nearest neighbors model and observed the influence of choosing various values for k. We found that choosing k depends, among other things, on the specific dataset used and the noise in the data. By tweaking k, we could get a model which __qualitatively__ fits our data well.

In this chapter, we will try to __quantify__ how well a specific model performs for any dataset, which can help us choose the best one.

## Loss function
To quantify the "closeness" between predictions of some model $y(\mathbf{x})$ and the observed data $t$, we introduce the squared loss function:

$$
L ( t, y(\mathbf{x}))= (y(\mathbf{x}) - t)^2,
$$

where $t$ are the observed values corresponding to $x$. Squaring the difference gives a number of nice properties:
* The loss is always positive, regardless of whether we underestimate or overestimate a prediction.
* It is differentiable at $y(\mathbf{x})=t$, which is not true for all loss functions (e.g. the absolute loss)

Below we plot this loss for some data points. Notice that in this case we are comparing the observations $t$ to the ground truth $f(\mathbf{x})$ (as opposed to some fitted model). The error therefore comes solely from the noise in our observations.

```{figure} https://github.com/TUDelft-MUDE/source-files/raw/main/file/fig2.png
Ground truth vs noisy data
```

## Theory
To obtain our model, we want a single value that tells us how well it explains all the data. Therefore it is natural to compute the expected loss:

$$
\mathbb{E}[L]= \int \int (y(\mathbf{x})-t)^2 p( \mathbf{x},t) d\mathbf{x} dt 
$$

where $(y(\mathbf{x})-t)^2$ is the error term we have seen before, and we multiply it with $p( \mathbf{x},t)$, the probability of this particular point occurring. Integrating this over the complete probability density $p(\mathbf{x},t)$ results in a single scalar that summarizes our expected prediction error.

Naturally, our goal is to choose a model $y(\mathbf{x})$ so as to minimize $\mathbb{E}[L]$. We can do this using calculus of variations:

$$
\frac{\delta \mathbb{E}[L]}{\delta y(\mathbf{x})}= 2 \int (y(\mathbf{x})-t) p( \mathbf{x},t) dt = 0, 
$$

where we keep the detailed steps of the derivation out of the scope of this discussion. Solving for $y(\mathbf{x})$ we get:

$$
\begin{align}
\int y(\mathbf{x}) p( \mathbf{x},t) dt &= \int t p( \mathbf{x},t) dt \\
y(\mathbf{x}) p( \mathbf{x}) &= \int t p( \mathbf{x},t) dt \\
y(\mathbf{x}) &= \frac{\int t p( \mathbf{x},t) dt}{p(\mathbf{x})} = \int t p(t|\mathbf{x})dt = \mathbb{E}_t[t|\mathbf{x}]
\end{align},
$$

where the Sum Rule of probability $p(\mathbf{x}) = \int{p(\mathbf{x},t)dt}$ has been employed and we factorize the joint probability distribution as $p(\mathbf{x},t) = p(\mathbf{x})p(t\vert\mathbf{x})$ (Product Rule). From the result above, the model that minimizes the squared loss is therefore given by the mean of the conditional distribution $p(t|\mathbf{x})$. This important result can also be seen graphically for a regression model with a single input:

<center><img src="https://surfdrive.surf.nl/files/index.php/s/OGErUzhQqGl4BCh/download", width=300px/></center>

In practice we generally do not know $p( \mathbf{x},t)$ or $p(t|\mathbf{x})$ exactly. Instead, we can estimate the loss for any model by taking the average of the squared losses for each point in our limited dataset. We can then tweak $y(\mathbf{x})$ to minimize this loss.

$$
\int \int (y(\mathbf{x})-t)^2 p( \mathbf{x},t) d\mathbf{x} dt \approx \dfrac{1}{N} \sum_i^N (y(\mathbf{x}_i)-t_i)^2
$$

This is known as the Mean Squared Error (MSE) function. 

We now revisit our k-nearest neighbors model with the value of the training loss shown in the plot title:

<iframe src="../_static/decision_theory_knn.html?mode=training" width="100%" height="610" frameborder="0"></iframe>

## Model selection
For $k = 1$ our loss function goes to exactly zero, yet the resulting model will not yield good predictions for unseen data. Our model has fitted the noise in the dataset, and therefore does not generalize well. We say that we have __overfitted__ our model to the training data. Clearly, evaluating the squared loss solely on our training points does not suffice if our goal is to make an informed choice for $k$.

To solve this problem a validation set is introduced. This set consists of different observations from the process which are not included during training. We also define a test set with samples that can be used __at the very end__ of our model building process in order to assess in an unbiased way if our choice of $k$ has been well informed. We therefore have three separate loss values:

|   |   |
| :--- | :--- |
| **Training Loss:** | The value of the loss function based on all observations used to train a model |
| **Validation Loss:** | The value of the loss function based on observations not used during training, used for finding hyperparameters such as $k$ |
| **Test Loss:** | The value of the loss function based on observations not used during either training or validation, used for evaluating the fit of the final model (for example to compare to alternative models) |

We repeat the experiment above with an additional validation set for which we compute the validation loss. This validation error represents the error we obtain when making new predictions that were hidden from the model during training and is therefore a much better indication for assessing our model compared to the training error. Minimizing the error on this validation set allows us to find a good value for k.

<iframe src="../_static/decision_theory_knn.html?mode=validation" width="100%" height="610" frameborder="0"></iframe>

__Try it out above!__ How does this compare with your findings from the previous chapter? Note that we now leave 20% of data points out for validation, and that the trained function does not fit them exactly. For a given dataset size, move the $k$ slider until the validation loss is as low as possible. 

Having selected a value for $k$, we could then compute the loss on our test dataset as one final check of model validity. We leave that step to you when working on our real-world problems later on.

## Bias-variance tradeoff

As we determined earlier, the optimal prediction (in the sense that it minimizes the squared loss function), which we will denote as $h(\mathbf{x})$, is given by:

$$
h(\mathbf{x})= \mathbb{E}[t|\mathbf{x}] = \int t p(t|\mathbf{x})dt 
$$

Starting from the expression for the loss function you know, we add and subtract the equation above with no loss of generality. This allows us to split the loss expression in two important parts: 

$$
\begin{aligned}
    \mathbb{E}[L] &= \int\int \left(y(\mathbf{x}) - t\right)^2p(\mathbf{x},t) d\mathbf{x}dt \\
    &= \int\int \left(y(\mathbf{x}) - h(\mathbf{x}) + h(\mathbf{x}) - t \right)^2 p(\mathbf{x},t) d\mathbf{x}dt \\
    &= \int\int \left(y(\mathbf{x}) - h(\mathbf{x})\right)^2 p(\mathbf{x},t) d\mathbf{x}dt + \int\int \left(h(\mathbf{x}) - t \right)^2 p(\mathbf{x},t) d\mathbf{x}dt + 2 \int\int \left(y(\mathbf{x}) - h(\mathbf{x})\right)\left(h(\mathbf{x}) - t\right) p(\mathbf{x},t) d\mathbf{x}dt
\end{aligned}
$$

Now note that the first term does not depend on $t$ anymore (remember $h(x)$ is already an integral over $t$). This means we can simplify it to an expectation only over $\mathbf{x}$. Furthermore, the third term vanishes once we integrate over $t$. We therefore end up with the following very important decomposition:

$$
\mathbb{E}[L]= \underbrace{\int \left(y(\mathbf{x}) - h(\mathbf{x})\right)^2 p(\mathbf{x}) d\mathbf{x}}_{\text{reducible loss}} + \underbrace{\int\int \left(h(\mathbf{x}) - t \right)^2 p(\mathbf{x},t) d\mathbf{x}dt}_{\text{irreducible noise}}
$$

The irreducible noise term represents the variance that will always be present in our model because we make noisy observations. Even if we had the perfect model $h(\mathbf{x})$, we would still have an expected loss greater than zero due to the noise inherent to our data. For the kNN model above, this is a bit hidden because we have __exactly one observation__ for each value of $x$. This is the reason why we can still obtain zero loss with our kNN estimator, but in practice we would ideally have several observations for the same value of our inputs.

Regardless, focusing now on the first term, we might note that actually, our model $y(\mathbf{x})$ not only depends on $\mathbf{x}$ but also on which dataset $\mathcal{D}$ we used for training. So, the first integral should be replaced as follows:

$$
\int \left(y(\mathbf{x}) - h(\mathbf{x})\right)^2 p(\mathbf{x}) d\mathbf{x} \quad \Longrightarrow \quad \int \mathbb{E}_\mathcal{D}\left[\left(y(\mathbf{x},\mathcal{D}) - h(\mathbf{x})\right)^2\right] p(\mathbf{x}) d\mathbf{x} 
$$

Now, we do some rearrangement in the inner part of this expression:

$$
\begin{aligned}
    \left(y(\mathbf{x},\mathcal{D}) - h(\mathbf{x})\right)^2 &= \left(y(\mathbf{x},\mathcal{D}) - \mathbb{E}_\mathcal{D}\left[y(\mathbf{x},\mathcal{D})\right] + \mathbb{E}_\mathcal{D}\left[y(\mathbf{x},\mathcal{D})\right] - h(\mathbf{x})\right)^2 \\
    &= \left(y(\mathbf{x},\mathcal{D}) - \mathbb{E}_\mathcal{D}\left[y(\mathbf{x},\mathcal{D})\right]\right)^2 + \left(\mathbb{E}_\mathcal{D}\left[y(\mathbf{x},\mathcal{D})\right] - h(\mathbf{x})\right)^2 + 2 \left(y(\mathbf{x},\mathcal{D}) - \mathbb{E}_\mathcal{D}\left[y(\mathbf{x},\mathcal{D})\right]\right)\left(\mathbb{E}_\mathcal{D}\left[y(\mathbf{x},\mathcal{D})\right] - h(\mathbf{x})\right)
\end{aligned}
$$

and then taking the expectation over all possible datasets we get:

$$
\begin{aligned}
\mathbb{E}_\mathcal{D}\left[\left(y(\mathbf{x},\mathcal{D}) - h(\mathbf{x})\right)^2\right] &= \mathbb{E}_\mathcal{D}\left[\left(y(\mathbf{x},\mathcal{D}) - \mathbb{E}_\mathcal{D}\left[y(\mathbf{x},\mathcal{D})\right]\right)^2\right] + \mathbb{E}_\mathcal{D}\left[\left(\mathbb{E}_\mathcal{D}\left[y(\mathbf{x},\mathcal{D})\right] - h(\mathbf{x})\right)\right]^2 + 0 \\
    &= \underbrace{\mathbb{E}_\mathcal{D}\left[\left(y(\mathbf{x},\mathcal{D}) - \mathbb{E}_\mathcal{D}\left[y(\mathbf{x},\mathcal{D})\right]\right)^2\right]}_{\text{variance}} + \underbrace{\left\{\mathbb{E}_\mathcal{D}\left[y(\mathbf{x},\mathcal{D})\right] - h(\mathbf{x})\right\}^2}_{\text{bias}^2}
\end{aligned}
$$

Overall, the expected loss can be decomposed as follows:

$$
\text{expected loss} = \text{bias}^2 + \text{variance} + \text{noise}
$$

These terms can be understood as follows:

- __Bias__: Represents the mismatch between the expectaction of our model $\mathbb{E}_\mathcal{D}\left[y(\mathbf{x},\mathcal{D})\right]$ and the optimal prediction $h(\mathbf{x})$. An __unbiased__ model will on average give us the optimal prediction, whereas a __biased__ model will on average deviate from the optimal prediction.
- __Variance__: Represents the degree to which a given model $y(\mathbf{x},\mathcal{D})$ (trained with one specific realization of $\mathcal{D}$) deviates from the average model $\mathbb{E}_\mathcal{D}\left[y(\mathbf{x},\mathcal{D})\right]$. If the variance is large, it means that a new dataset would produce a very different model.
- __Noise__: Represents the inherent variation in our dataset. This term cannot be reduced because it stems from our data and not from our choice of model.

Generally speaking, there is a certain balance to strike between bias and variance. A more flexible model tends to target the optimal prediction $h(\mathbf{x})$ on average but can be significantly influenced by the dataset to which it is fitted. In other words, it has a lower bias but a higher variance. On the other hand, a less flexible model tends to be less affected by the dataset to which it is fitted, but this causes it to deviate from $h(\mathbf{x})$. In other words, it has a lower variance but a higher bias. This dilemma is referred to as the __bias-variance tradeoff__.

Looking back at the plot above, try to answer the following questions:
* When k is low, do we have a high or a low bias? And what about the variance?
* When k is high, do we have a high or a low bias? And what about the variance?

We will now quantify the bias and variance by creating several datasets and averaging over them, following the equations above. Notice that computing the expressions above is only possible here because we know the ground truth in this example, that is, we know what the perfect model $h(\mathbf{x})$ should look like.

<iframe src="../_static/decision_theory_knn.html?mode=biasvariance" width="100%" height="640" frameborder="0"></iframe>

## Playing around with the plot
Understanding bias and variance is critical for any method you use, and this will come back in many of the following lectures. Being able to answer some of the following questions can help you understand what is going on:

* Why is the variance small for a large $k$?

````{admonition} Answer
:class: dropdown
With higher values of $k$, we are averaging over a larger neighborhood. Intuitively, you can imagine that if our noise is high, generating multiple datasets of the same size will result in very different datasets, especially if $N$ is small. But over larger and larger neighborhoods, it makes sense that this noise we are adding tends to cancel out between points (remember the expected value of our noise is zero). It is therefore more likely that our resulting model will not change all that much for different datasets. This is the definition of a model with low variance (and therefore high bias!).
````

* Why is the bias not exactly zero everywhere when $k=1$?

````{admonition} Answer
:class: dropdown
The bias should be zero at the locations of every one of the $N$ data points and when we take expectations over an infinite number of datasets. Take a look at what happens with the bias between data point locations and for different numbers of datasets.
````

* Why does the bias increase when choosing a high frequency $freq$?

````{admonition} Answer
:class: dropdown
The complexity of our ground truth increases with frequency. So it makes sense that we would then need a more complex model to capture it properly. This in turn means models with the same $k$ will likely show an increase in bias. Note that this is not the case for $k=1$.
````

__TIP__: When playing with the plot, be aware that training the model with $2000$ different datasets might take a while! When moving the sliders set that to a lower number of datasets and only click the '2000 datasets' button when you are done with the sliders.

## Final remarks

So far, we have looked at our k-nearest neighbors regressor, which is relatively easy to understand and play around with. However, this method is not often used in practice, as it scales poorly for large datasets. Furthermore, computing the closest points becomes computationally expensive when considering multiple inputs (high dimensions). In a limited data setting, taking the average of the nearest neighbors will often lead to poor performance. k-nearest neighbors is also sensitive to outliers. In the following chapter, we will apply what we have learned to a different model, namely, linear regression.
