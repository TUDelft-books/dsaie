# Review and Quiz

## Multiple choice questions

<iframe src="https://tudelft.h5p.com/content/1292036114427895717/embed" aria-label="Kernel methods" width="1088" height="637" frameborder="0" allowfullscreen="allowfullscreen" allow="autoplay *; geolocation *; microphone *; camera *; midi *; encrypted-media *"></iframe><script src="https://tudelft.h5p.com/js/h5p-resizer.js" charset="UTF-8"></script>

`````{tab-set}
````{tab-item} Open question 1: 
Consider a case where your data follows a complex pattern and is noisy, as is shown below.

```{figure} ../figures/gpreview0.svg
:figwidth: 500px

Example data pattern
```

Would a Gaussian Process process be a good model to use? If so, how? If not, which model would be a better fit? Explain your answer in detail.

Example answer:
```{toggle} Answer:
We observe a regression problem with a relatively limited amount of data.
While the linear trend could be captured well by a linear model, there is a complex pattern in the fluctuations for which a more complex model is desirable. 
When using a GP, we can capture the linear trend by extending the kernel with a linear component.
In addition, the observed noise can also be captured well by a Gaussian Process.
Therefore, a Gaussian Process is expected to perform well.
```

````

````{tab-item} Open question 2: 
Explain how emperical bayes works and the role it plays for GPs.

Example answer:
```{toggle} Answer:
Kernels usually feature a number of hyperparameters that control their behavior and need to be determined. One method to fit these hyperparameters is Empirical Bayes.
Specifically, Emperical Bayes is an approximation to a fully Bayesian treatment of the hyperparameters, where they are set to their most likely values instead of being integrated out.

This can be achieved by finding the parameters which maximize the marginal likelihood (also known as the evidence).
```
````
`````