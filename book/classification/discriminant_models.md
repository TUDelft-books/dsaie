# Discriminant Models

Here we briefly look at a non-probabilistic treatment for classification problems. The idea is to altogether skip decision theory and directly train a function $y(\mathbf{x})$ that maps our features to a class label.

There are several flavors of discriminant models, but here we limit ourselves to a simple linear discriminant trained with least squares which will suffice in making a link with regression and making the case for better models.

```{admonition} Further Reading    
:class: tip    
Bishop covers two other types of linear discriminant. This includes the algorithm driving the Perceptron, a large machine that implemented the first trained neural network in 1958. If you are curious, read the complete Section 4.1.
+++                         
{bdg-danger}`bishop-prml`     
``` 

## A least-squares two-class discriminant

We start straight from what we learned in {doc}`regression problems<../../2-bayesregression/lectures/linear_models>`. There we have seen that maximizing the likelihood of a Gaussian observation model leads exactly to a least squares problem. We do the same here but for classification: given an input $\mathbf{x}$, we would like to know if it corresponds to one of two classes $\mathcal{C}_1$ or $\mathcal{C}_2$.

To train a model we are going to need some targets. For regression this was clear (the observed value of some continuous variable). Here the targets are not so trivial, as we must assign a numerical value to represent class attribution. Let us be practical and use $t=+1$ to indicate class $\mathcal{C}_1$ and $t=-1$ for $\mathcal{C}_2$. This means our whole dataset $\mathcal{D}$ of size $N$ is composed of **exclusively** $+1$'s and $-1$'s.

The goal is to minimize the distance between targets and model predictions, which gives the following least-squares error function:

$$
E_\mathcal{D} = \displaystyle
\frac{1}{2}
\sum_{n=1}^N
\left[
t_n - \mathbf{w}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x}_n)
\right]^2
$$(lsdiscriminanterror)

As always we arrange our targets in a vector $\mathbf{t}$ and features in a matrix $\boldsymbol{\Phi}$. Solving the least squares problem gives the following values for the weights:

$$
\mathbf{w}_\mathrm{LS} = \left(\boldsymbol{\Phi}^\mathrm{T}\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}^\mathrm{T}\mathbf{t}
$$(lsdiscriminantsolution)

and our estimator in its regression function form becomes:

$$
y(\mathbf{x}) = \mathbf{w}_\mathrm{LS}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x})
$$(lsdiscriminantpred)

And as a final step, we classify the point as being in $\mathcal{C}_1$ 
if $y\geq 0$ and in $\mathcal{C}_2$ if $y<0$.

We now show some simple examples of the approach. Let us for now assume our basis functions $\boldsymbol{\phi}(\mbf{x})=\mbf{x}$, which gives us a linear model both on $\mbf{x}$ and $\mbf{w}$. The figure below shows an example with a one-dimensional input feature $x$ and two classes. The trained model in its *regression form* $y(x)$ is shown in black. 

```{figure} ../figures/discriminant0.svg
:figwidth: 500px
:name: "discriminant1dexample"

A classification example with one feature and two classes. A linear discriminant provides the decision boundary.
```

In this case, the decision boundary corresponds to $y(x)=0$, so a single point. We still draw it as a vertical line to make it clearly visible. Points to the left of the boundary are classified as $\mathcal{C}_1$ and points to the right as $\mathcal{C}_2$. We can see that our simple linear model does a good job in this case, with every training sample being correctly classified. Note however that although we leave it out for this simple demonstration, the phenomenon of overfitting and the usual deterministic remedies to it (training/validation/test split, regularization) also apply to classification problems.

We now go for a similar example but in two dimensions. Now $\mbf{x}=[x_1\,\,x_2]$ and our observations are now scattered in a 2D space. Nothing else changes, however, and we can solve the problem in the same way. The figure below shows the observations for our two classes, with model predictions shown as shaded regions:

```{figure} ../figures/discriminant1.svg
:figwidth: 500px
:name: "discriminant2dexample"

A classification example with two features and two classes. A linear discriminant provides the decision boundary.
```

The transition between the two regions is our decision boundary, and again corresponds to $y(\mbf{x})=0$. As before, we classify to $\mathcal{C}_1$ if $y\geq 0$ and to $\mathcal{C}_2$ if $y<0$. For clarity, we do not show the actual function $y(\mbf{x})$ anymore but just the resulting decision regions.

A few important observations can be made from this figure:
- In a $D$-dimensional problem, the decision boundary is always a $D-1$-dimensional entity. So it was a single point in 1D and a line in 2D;
- Linear models result in linear decision boundaries. As you will see soon in an {doc}`exercise<../exercises-clean/discriminant>`, this observation will change when we use more complex basis functions $\boldsymbol{\phi}$;
- We can say that this particular problem is **linearly separable**, since a linear decision boundary can already do a good job.

On the figure below we see a case which is not linearly separable. Samples from $\mathcal{C}_1$ are spread out over $\mbf{x}$ in such a way that a straight decision boundary that perfectly splits the data is not possible anymore. 

```{figure} ../figures/discriminant2.svg
:figwidth: 500px
:name: "discriminant2dexample2"

A classification example with two features and two classes. The linear decision boundary from before is now not enough to properly separate the classes.
```

We could still get a good decision boundary in this case by using appropriate basis functions $\boldsymbol{\phi}$. But in any case we shall avoid non-probabilistic discriminant models moving forward, for a number of reasons:

- We implicitly assume our targets follow a Gaussian distribution when conditioned on $\mbf{x}$. This is inherited from the regression nature of the approach and is quite unrealistic;
- The link to the {doc}`decision theory<decision_theory>` we developed is completely lost;
- The outputs have no probabilistic interpretation. Even if we set our targets to be $t=0$ or $t=1$, there is no guarantee that $y$ will be bounded between $[0,1]$.

```{admonition} Exam questions    
:class: danger    
You might encounter conceptual questions about this subject in the written exam, but with focus on comparing discriminant models with other modeling approaches. The regression roots of the approach should be clear to you, and how those relate to the limitations of the method.
+++    
{bdg-primary}`written-exam`    
``` 