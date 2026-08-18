# Dropout Regularization

## Resources
[Slides](https://surfdrive.surf.nl/files/index.php/s/amyPZQpMv7URk4Z)

Chapter 7.11 and 7.12 until page 257 (stop at "More formally, ...") {bdg-muted}`goodfellow-dl`

Chapter 9.3.3 {bdg-warning}`prince-udl`

## Regularization
Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error. The goal of regularization is to improve the generalization performance of our model. This is achieved by preventing overfitting by discouraging the model from learning complex patterns that are specific to the training data and may not generalize well to unseen data.

Figure {numref}`overfitting_dropout_diagram` demonstrates how the generalization error rises as the model complexity increases due to the bias-variance tradeoff.

```{figure} ../figures/bias_variance_tradeoff.svg
:name: overfitting_dropout_diagram
:height: 300px
:align: center

Bias-Variance Trade-Off Error Graph
```

In traditional machine learning, the goal of regularization is often linked to the control of model complexity, where we try to find the appropriate number of parameters that allow it to capture complex patterns while not overfitting on specific training data points. There are two main types of regularization techniques: *L1 regularization* and *L2 regularization*.

**L1 Regularization (LASSO):** 
Adds the absolute values of the weights as a penalty term to the loss function, effectively performing feature selection by driving some weights to zero.

$$\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda \sum_{i=1}^n |w_i|$$

**L2 Regularization (Ridge):** 
Adds the squared values of the weights as a penalty term to the loss function, discouraging large weights uniformly across the model.

$$\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda \sum_{i=1}^n w_i^2$$

L1 and L2 regularization may not be effective for Deep Learning methods, mainly for the following reasons:

- **High Dimensionality:** Deep learning models often have millions of parameters, making L1 and L2 penalties less effective in controlling overfitting.

- **Non-convex Optimization:** Deep networks capture complex, non-linear relationships, which L1 and L2 regularization, suited for linear models, cannot adequately manage.

Instead, common approaches we already explored to prevent overfitting and improve generalization in deep learning are:
1. **Early Stopping:** Ceases training when validation loss worsens, avoiding overfitting due to learning the noise in the data.
2. **Data Augmentation:** Applies transformations to data, enhancing generalization by mimicking real-world input variability.
3. **Batch Normalization:** Stabilizes training and acts as a regularizer by normalizing layer inputs, aiding in smoother optimization.

## Ensembling as a Form of Regularization
Ensembling combines predictions from multiple independently trained models, effectively performing model averaging to reduce prediction variance. **Bagging** (bootstrap aggregating) is a specific ensembling technique where $k$ models are trained on different datasets, each constructed by sampling with replacement from the original training data.

Figure {numref}`bagging_diagram_probdl` demonstrates how using bagging we can have two models to together assign digits to handwritten numbers, where one model might learn that a top loop corresponds to an 8, whereas the second model might learn that a bottom loop corresponds to an 8. By combining the predictions of both models, we can correctly classify the digit as an 8.

```{figure} ../figures/bagging_example.png
:name: bagging_diagram_probdl
:height: 300px
:align: center

Bagging on MNIST example.
```
To understand ensembling's effectiveness, consider $k$ regression models where each makes an error $\epsilon_i$ with:

$$
\epsilon_i \sim N(0,v) \quad \text{with} \quad \text{Var}(\epsilon_i) = v  \quad \text{and} \quad \text{Cov}(\epsilon_i, \epsilon_j) = c
$$

The expected square error of their average prediction is:

$$
\text{MSE} = E \left[ \left( \frac{1}{k} \sum_i \epsilon_i \right)^2 \right] = \frac{v}{k} + \frac{(k-1)c}{k}
$$

When errors are perfectly correlated ($c=v$), $\text{MSE} = v$ and averaging doesn't help. However, with uncorrelated errors ($c=0$), $\text{MSE}=\frac{v}{k}$, showing error decreases linearly with ensemble size.

While effective, traditional ensembling becomes computationally expensive for large deep learning models. This limitation led to the development of **dropout** regularization, which approximates training an ensemble of exponentially many neural networks in a computationally efficient manner.

## Dropout Regularization
Dropout is a regularization technique that prevents overfitting by randomly deactivating neurons during training with a probability called the **dropout rate** (typically 20-50%). At each training iteration, different neurons are temporarily removed, effectively creating an ensemble of sub-networks. This process prevents neurons from co-adapting and forces the network to learn more robust features, as it cannot rely on specific neuron combinations. In figure {numref}`dropout_ensemble`, dropout is applied to the original neural network to showcase how the connections between neurons dissapear, forcing the network to not over-rely on specific neurons. 

```{figure} ../figures/dropout_rate_example.svg
:name: dropout_ensemble
:height: 450px
:align: center

Dropout regularization example
```

Dropout can be especially useful for improving generalization in model predictions. Consider a model with training data that is missing for certain inputs. The model would not be penalized for not generalizing during training, as the training loss only considers training data points. Figure {numref}`dropout_vis_1` demonstrates the predictions of a simple neural network with three hidden units (neurons), input $x$ and output $y$, where the model struggles to generalize its predictions (blue line) over the purple region with little training data points (red circles) compared to the ground truth (black line). 

```{figure} ../figures/dropout_vis_1.svg
:name: dropout_vis_1
:height: 300px
:align: center

Neural Network with Generalization Problem
```

The network has learned an undesirable "kink" pattern, marked by a spike, created by neuron 1, that neuron 2 tries to cancel out. While this achieves low training error (since there are no training points in the neighborhood of the spike), it's an unstable solution that won't generalize well. Figure {numref}`dropout_vis_2` demonstrates what happens when we turn off neuron 2: the prediction becomes extremely poor, revealing the network's over-reliance on the interaction between specific neurons. 

```{figure} ../figures/dropout_vis_2.svg
:name: dropout_vis_2
:height: 300px
:align: center

Neural Network with Deactivated Second Unit
```

Finally, Figure {numref}`dropout_vis_3` shows the result after training with dropout for many iterations: the network learns a smoother, more robust solution that better approximates the true function without artificial spikes. This demonstrates how dropout prevents co-adaptation between neurons, forcing the network to learn more robust features that don't rely on specific neuron interactions to cancel out each other's artifacts.

```{figure} ../figures/dropout_vis_3.svg
:name: dropout_vis_3
:height: 300px
:align: center

Neural Network after Many Iterations of Dropout Regularization
```

### Weight scaling inference rule
During training, dropout randomly deactivates a subset of neurons in each iteration, forcing the network to learn robust features without relying on specific neuron combinations. Since this subset changes at each iteration, the network learns to make predictions using different combinations of available neurons. During inference (testing), we want to use all neurons to maximize the network's capacity, but this creates a mismatch: the network now has more active units than it had during training iterations. To address this, we apply the **weight scaling inference rule**: multiply the weights by $1-p$ during inference, where $p$ is the dropout rate. This scaling ensures that the expected output of the network remains consistent between training and inference phases. In PyTorch, this adjustment is automatically handled by switching between `model.train()` and `model.eval()` functions.

## Dropout Variations Across Neural Architectures

While standard dropout is straightforward to implement in dense (fully connected) layers, different neural architectures require specialized dropout techniques to maintain their structural properties:

**CNNs and Spatial Dropout:**
- Standard dropout can disrupt spatial patterns in CNNs by randomly zeroing individual units
- This breaks spatial coherence and may remove critical parts of feature maps
- **Spatial Dropout (Dropout2D)** addresses this by dropping entire feature maps, preserving spatial relationships within remaining maps

**RNNs and Temporal Dropout:**
Dropout in recurrent networks can be applied in two ways:
1. **Input Dropout:** Randomly masks inputs to prevent over-reliance on specific features
   
   $$x_{\text{drop}}^{(t)} = x^{(t)} \odot r_x, \quad h^{(t)} = f(W_x x_{\text{drop}}^{(t)} + W_h h^{(t-1)} + b)$$

2. **Hidden State Dropout:** Randomly masks hidden states to regularize temporal dependencies
   
   $$h_{\text{drop}}^{(t-1)} = h^{(t-1)} \odot r_h, \quad h^{(t)} = f(W_x x^{(t)} + W_h h_{\text{drop}}^{(t-1)} + b)$$

A similar mechanism is implemented for gated RNNs.

These variations demonstrate how dropout must be adapted to respect the underlying architecture's structural assumptions while maintaining its regularizing effects.
