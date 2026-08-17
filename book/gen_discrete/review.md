# Review and Quiz

## Multiple choice questions

<iframe src="https://tudelft.h5p.com/content/1292036114919172137/embed" aria-label="Clustering mixtures" width="1088" height="637" frameborder="0" allowfullscreen="allowfullscreen" allow="autoplay *; geolocation *; microphone *; camera *; midi *; encrypted-media *"></iframe><script src="https://tudelft.h5p.com/js/h5p-resizer.js" charset="UTF-8"></script>

## In the exam you will also be tested on open questions. Here are a few example questions

`````{tab-set}

````{tab-item} Open question 1:
What are the key differences between supervised and unsupervised learning?
List two examples of each category.

Example answer:
```{toggle} Answer:
Supervised learning concerns labeled data, meaning that the data consists of input-output pairs and we try to learn a mapping from input to output.
Examples of supervised learning include linear regression, logistic regression, k-nearest neighbors and feed-forward neural networks.

Unsupervised learning concerns unlabeled data, meaning that the data only contains inputs, and try to learn patterns in the input data.
Examples of unsupervised learning include K-means clustering, Gaussian mixtures and autoencoders.
```
````

````{tab-item} Open question 2: 
We have seen how the EM-algorithm can be used to maximize the log likelihood of a Gaussian mixtures model. 
Describe what the EM-algorithm looks like for Gaussian mixtures.
Be specific on what happens in the E-step and M-step.

Example answer:
```{toggle} Answer:
In the *E-step*, the soft assignment of each data point to each cluster is performed. This gives us the *expectation* of the log-likelihood given the current parameter values.

In the *M-step*, the means, covariances and mixing coefficients of the clusters are recomputed given the current cluster assignment. This gives us a *maximization* of the expected log-likelihood from the E-step.

These steps are repeated until convergence is reached, in which case a local maximum of the likelihood function has been found.
```
````
`````