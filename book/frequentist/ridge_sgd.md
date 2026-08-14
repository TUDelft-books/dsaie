# Stochastic Gradient Descent

## Introduction

<iframe width="560" height="315" src="https://www.youtube.com/embed/Xy_1cRy9AOU?si=a0SnL6hmAtQHNyOU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

For now, we used the same dataset at once. In some situations, it might be beneficial or necessary to look at only a part of the dataset, e.g., when

- $N$ is too large and computing $( \boldsymbol{\Phi}^T \boldsymbol{\Phi} )^{-1}$ becomes prohibitively expensive
- the model is nonlinear in $\mathbf{w}$, and $\mathbf{w}_\mathrm{ML}$ does not have a closed-form solution
- the dataset is arriving sequentially (e.g., in real-time from a sensor)

Instead of solving for $\mathbf{w}$ directly, we could employ an iterative optimization strategy. Let's first take a look at the data part of the error function, and its gradient with respect to $\mathbf{w}$:

$$
E_D = \frac{1}{2N} \sum_{n = 1}^{N} \left( t_n - \mathbf{w}^T \boldsymbol{\phi}_n \right)^2 \quad \mathrm{with \, gradient} \quad \nabla E_D = - \frac{1}{N} \sum_{n=1}^N \left(t_n - \mathbf{w}^T \boldsymbol{\phi}_n \right) \boldsymbol{\phi}_n.
$$

The standard formulation does not include the division by the dataset size, the stepsize is purely regulated through the learning rate. However, normalizing the gradient with $N$ makes the influence of the learning rate more consistent when considering different datasets. With a standard gradient descent algorithm, the update rule for the weights is given by

$$
\mathbf{w}^{(\tau + 1)} = \mathbf{w}^{(\tau)} - \eta \nabla E_D
$$

with a fixed *learning rate* $\eta$. The costs for the gradient computations are independent of the dataset size $N$, when only considering subset $\mathcal{B}$ of our dataset with $N_{\mathcal{B}}$ data points. If we pick a random subset $\mathcal{B}$ for each iteration of the optimization scheme, we have derived the *stochastic gradient descent* algorithm. Together with its numerous variants, this algorithm forms the backbone of many machine learning techniques. Most deep learning libraries, such as [`TensorFlow`](https://www.tensorflow.org/) or [`PyTorch`](https://pytorch.org/) offer implementations of these algorithms.

We looked at the unregularized model to introduce SGD, but the extension to the regularized model is straightforward. Remember, in this case, the objective function is given by

$$
E (\mathbf{w}) = \frac{1}{2N} \sum_{n = 1}^{N} \left( t_n - \mathbf{w}^T \boldsymbol{\phi}_n \right)^2 + \frac{\lambda}{2N} \mathbf{w}^T \mathbf{w},
$$

and its gradient with respect to $\mathbf{w}$ reads

$$
\nabla E = \frac{1}{N} \left( - \sum_{n=1}^N \left(t_n - \mathbf{w}^T \boldsymbol{\phi}_n \right) \boldsymbol{\phi}_n + \lambda \, \mathbf{w} \right).
$$

When looking at the expression in the outer bracket, it becomes clear there are **two competing terms**: the first term pulls the weights towards the data, and the second term pulls them towards zero. Looking at the gradient, you can also see why ridge regression often is referred to as weight decay. The larger the weights become, the stronger the regularization term pulls them towards zero. If the data would not support a value for a certain weight, its presence in the gradient will lead to the weight decaying at a rate proportional to its magnitude.

```{figure} https://github.com/TUDelft-MUDE/source-files/raw/main/file/fig17.png
Training process of two models with different levels of regularization, predictions coming from model snapshots at different epochs.
```

You can see in the top figure that our predictions seem to converge towards a particular shape. The remaining discrepancy between our final model and the true function $f$ is due to our general model bias, and the particular dataset we drew.

You can see that our validation error increases at some point, indicating that overfitting might occur. Again, note how the training error cannot feel that, and just decreases monotonically. SGD with minibatches already has a slight regularizing effect. Other remedies include the L2-regularization technique discussed discusses previously, early stopping, or collecting more data.

Finally, it should be noted that the step size of the SGD must be chosen carefully. Try out for yourself what happens when you choose very small or large stepsizes by adapting the learning rate. Even though this optimization problem is well-defined and has a global minimum, SGD is not guaranteed to converge to it. Luckily, the most popular variants such as [AdaGrad][1], [RMSProp][2], and [Adam][3] feature some form of adaptive stepsize control, improving convergence rate and robustness. One usually starts with a larger stepsize to approach the minimum quickly. After that, the stepsize is reduced continuously to reliably uncover the exact location of the extremum.

[1]: https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
[2]: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
[3]: https://arxiv.org/abs/1412.6980

In the following animation you can see another example of extreme overfitting being observed during SGD iterations:

<center><img src="https://surfdrive.surf.nl/files/index.php/s/4Q8RDBb6ZKRgRBs/download" width="1000"></center>

Observing when the validation loss starts to increase is a useful sign of the onset of overfitting. Stopping the training process when this is detected is the core of so-called **early stopping** strategies implemented in popular machine learning packages.