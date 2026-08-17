# Neural Networks

We wrap up our brief treatment of classification with a final jump from logistic regression with basis functions to neural networks. The idea is the same as what we did for {doc}`regression<../frequentist/nn>`: instead of having **fixed** basis functions $\basis$ we make them **adaptive** by using one or more fully-connected neuron layers. As before, we can see the second-to-last layer of the network as a new adaptive basis $\bar{\basis}(\bx)$, and the last layer represents the familiar linear combination $\bw^\T\bar{\basis}$.

## Two classes

We start from where we left off {doc}`previously<logistic_regression>`, going from $D$ input dimensions to a single output we can use for two-class problems. The resultant neural network is shown below:

```{figure} ../figures/nnclassification1.svg
:scale: 50%
:name: nnclassification1

A dense neural network going from $D$ inputs to a single output for two-class classification. Note the sigmoid activation of the last layer.
```

where the only change from the networks for regression is the activation of the output layer of the network with the logistic sigmoid function. We can also directly use the *cross-entropy* loss function we found for logistic regression. A summary of the model, also commonly referred to as **multilayer perceptron**, is given below:

````{card}
**Multilayer Perceptron (MLP)** with two classes
^^^
Set $\class_1$ targets to $t=1$ and $\class_2$ targets to $t=0$. Given a network architecture with weights $\bw$, the model becomes:

$$
y(\mbf{x})=y_\bw(\bx) \quad\Rightarrow\quad
\begin{cases}
\text{if }y_\bw(\mbf{x})\geq 0.5, & \mbf{x} \text{ belongs to } \class_1\\
\text{if }y_\bw(\mbf{x})<0.5, & \mbf{x} \text{ belongs to } \class_2
\end{cases}
$$

And $\mbf{w}$ is obtained by minimizing a loss function based on cross-entropy: 

$$
L(\bw,\lambda) = \displaystyle
-\frac{1}{N}\sum_{n=1}^N\left[t_n\ln y_n + (1-t_n)\ln(1-y_n)\right]
+\frac{\lambda}{2}\bw^\T\bw
$$

where the $L_2$ regularization parameter $\lambda$ should be adjusted with a validation dataset.
````

## Multiple classes

Up until now we have been working exclusively with two classes. Here we make one final extension to an arbitrary number $K$ of classes. To determine which output activation must be used to keep consistency with the original generative approach, we go back to Bayes' Theorem one last time:

$$
p(\class_k\vert\bx) = \displaystyle
\frac{p(\bx\vert\class_k)p(\class_k)}{\sum_jp(\bx\vert\class_j)p(\class_j)}
$$(multiclassbayes)

We now define $a_k=\ln\left(p(\bx\vert\class_k)p(\class_k)\right)$ and substitute above. The posterior we are trying to approximate then becomes:

$$
p(\class_k\vert\bx) = \displaystylecomputations
\frac{\exp(a_k)}{\sum_j\exp(a_j)} \equiv \mathrm{softmax}(\mbf{a},a_k)
$$(softmax)

which is known as the **softmax function**. This must therefore be our activation function of choice in this case. Note how softmax is not applied element-wise like other activation functions but actually depends on values from the whole layer. It is therefore a kind of layer-wise normalization that scales all values $a_k$ in in interval $[0,1]$ and ensures that $\sum_ja_j=1$. As a consequence, the decision step should have us pick the class $k$ with the highest value $a_k$ after activation. The resulting model is shown below:

```{figure} ../figures/nnclassification2.svg
:scale: 50%
:name: nnclassification2

A dense neural network going from $D$ inputs to $K$ outputs for K-class classification. Note the softmax activation of the last layer.
```

One final addition has to do with the targets $t$. We have been using a single target and setting it to either zero or one depending on the class. Now we adopt a so-called *1-of-$K$* encoding scheme: every target sample is now a vector with $K$ variables with a single entry being one and the rest being zero. For instance, for a problem with three classes, targets belonging to classes one, two and three would be, respectively, $t=[1\,0\,0]^\T$, $t=[0\,1\,0]^\T$ and $t=[0\,0\,1]^\T$.

We now have all the pieces we need. A summary of the resulting model is shown below:

````{card}
**Multilayer Perceptron (MLP)** for $K$ classes
^^^
Set targets in a 1-of-K scheme. Given a network architecture with weights $\bw$, the model becomes:

$$
\mathbf{y}(\mbf{x})=\mathbf{y}_\bw(\bx) \quad\Rightarrow\quad
\text{classify to class }\bar{k}=\underset{k}{\mathrm{argmax}}\,\bigl(y_k\bigr)
$$

And $\mbf{w}$ is obtained by minimizing a loss function based on cross-entropy: 

$$
L(\bw,\lambda) = \displaystyle
-\frac{1}{N}\sum_{n=1}^N\sum_{k=1}^K\left[t_{nk}\ln y_{nk}\right]
+\frac{\lambda}{2}\bw^\T\bw
$$

where the $L_2$ regularization parameter $\lambda$ should be adjusted with a validation dataset.
````

```{admonition} Exam questions    
:class: danger    
Conceptual questions about neural networks for classification may appear in the exam. You will not be expected to solve numerical questions on this topic as computations would be long and tedious. It is however important to grasp the main concepts on the architecture of MLPs for classification, appropriate loss functions and where they come from, what to do with the targets, etc.
+++    
{bdg-primary}`written-exam`    
``` 
