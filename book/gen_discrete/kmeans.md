# k-Means Clustering

Here we go briefly over the popular $k$-means clustering technique. It is a powerful and flexible tool for finding discrete latent patterns in data, being employed in a wide range of practical applications.

The problem setting is the one from {ref}`before<clustering-unsupervised-clustering>`, but now we look at a two-dimensional version of it. Suppose we have some **unlabeled** data in 2D:

```{figure} ../figures/kmeans0.svg
:figwidth: 500px

A set of unlabeled data points in two dimensions.
```

There is a clear pattern in this data which we want to capture automatically with an algorithm. In $k$-means clustering, we assume *a priori* the existence of $K$ data clusters, with cluster means $\bs{\mu}_k$. Each cluster mean is considered a representative member of its cluster. We also assume the existence of a cluster assignment variable $r_{nk}\in\{0,1\}$, so if data point $n$ belongs to cluster 1, $r_{n1}=1$ and all other $r_{nk} = 0$. This is the same *1-of-K* coding scheme we used for {doc}`logistic regression<../classification/logistic_regression>`.

We then define a **distortion measure** we can minimize:

$$
J = \displaystyle\sum_{n=1}^N\sum_{k=1}^K r_{nk}\lVert\bx_n-\bs{\mu}_k\rVert^2
$$(kmeansJ)

For every point, minimizing $J$ means minimizing the distance between the point and the mean of the cluster we associate it with. As we can see there are two parts to $J$, a cluster assignment $r_{nk}$ and a distance that is a function of $\bs{\mu}_k$. Minimizing $J$ is best done in iterations of the two separate steps we see in the following, where we minimize with respect to the cluster assignment and the cluster means in turn:

````{card}
Cluster assignment (**E-step**)
^^^
We freeze the current means $\bs{\mu}_k$. We then loop through every point and assign it to the cluster whose mean is closest to its position:

$$
r_{nk} = 
\begin{cases}
1 & \text{if }k=\underset{j}{\mathrm{argmin}}\lVert\bx_n-\bs{\mu}_j\rVert^2\\
0 & \text{otherwise}
\end{cases}
$$
````

````{card}
Update cluster means (**M-step**)
^^^
We now freeze the cluster assignments $r_{nk}$. We then loop through every cluster and update $\bs{\mu}$ to the current average of all points belonging to it:

$$
\bs{\mu}_k = \displaystyle
\frac{\sum_nr_{nk}\bx_n}{\sum_nr_{nk}}
$$
````

We then repeat these two steps for a number of iterations until $r_{nk}$ stops changing. The reason to call them *E-step* and *M-step* will be clear when we talk about {doc}`Gaussian mixtures<gaussian_mixtures>`.

It is easier what the E-step and M-step are doing by looking at the algorithm in action. For the dataset above, we initialize two $\bs{\mu}$'s to random positions in our space. The situation after the first E-step is seen on the left plot below:

```{figure} ../figures/kmeans1.svg
:figwidth: 750px

Progress of $k$-means clustering on the dataset. Situation after the first complete iteration.
```

Now every point is associated with a cluster, and the dataset is chopped by a line perpendicular to the one that joins the two cluster centers. We then update the means with the M-step, leading to the new mean positions we can see on the right plot above.

We then repeat that for two more cycles, with the situation changing as we see in the figures below:

```{figure} ../figures/kmeans2.svg
:figwidth: 750px

Progress of $k$-means clustering on the dataset. Situation after the second complete iteration.
```

```{figure} ../figures/kmeans3.svg
:figwidth: 750px

Progress of $k$-means clustering on the dataset. Situation after the third complete iteration.
```

At the end, the algorithm clearly identified the two clusters we could already see intuitively, and the cluster means moved to representative positions within each cluster.

The $k$-means algorithm can be easily customized by similarity measure from the simple Euclidian distance we use here to some other application-specific metric. 

In any case, we can see that $k$-means assigns points to clusters in a **hard** way: each point is assigned to only one cluster with the same confidence regardless if the point is right next to its cluster mean or almost exactly midway between two different cluster means.

```{admonition} Exam questions    
:class: danger    
You might see conceptual questions about $k$-means on the written exam, as well as simple calculations related to the algorithm you see above. You do not need to memorize the formulas above, but the E- and M-steps are conceptually simple and you might be asked to reproduce them by hand.
+++    
{bdg-primary}`written-exam`    
``` 
