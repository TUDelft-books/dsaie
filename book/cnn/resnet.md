# Residual Networks, Batch Normalization and Transfer Learning

## Summary

In this lecture we will delve into some more advanced concepts of Deep Learning and CNNs.

We will start by addressing the challenges faced when training deeper CNNs, particularly the phenomenon known as *shattered gradients*. This term describes the issues that arise as networks become deeper, leading to gradients becoming unstable or ineffective during backpropagation. Then the focus will shift to the concept of **residual connections**, a groundbreaking solution to the shattered gradients problem. We will explain what residual connections are and why they are effective in maintaining the flow of gradients through deep networks. While residual connections offer a solution, they also introduce a new challenge: the potential for *variance explosion*. This issue may also lead to highly unstable training, where the network's learning becomes erratic and inefficient, often resulting in poor model convergence and decreased performance.

To address the variance explosion issue, we will introduce the **batch normalization**. As suggested by the name, batch normalization normalizes the inputs of each layer within a mini-batch. This process stabilizes the learning by maintaining the mean output close to 0 and the output standard deviation close to 1. By doing so, it mitigates the problem of exploding variance, ensuring smoother and more stable training progress, especially in deep networks. 

We will then explore **Residual Neural Networks** (ResNet), powerful CNN architectures that incorporate residual blocks with batch normalization to achieve better performances and allows training much deeper networks. Finally, the lecture will conclude with an exploration of **transfer learning**. It will explain how the knowledge gained from powerful architectures like ResNets, trained on extensive datasets such as ImageNet, can be repurposed or transferred to new tasks or datasets, significantly improving the efficiency and effectiveness of model training in various applications.

## Going deeper with CNNs

We've seen previously tha, in a CNN, each layer connects only to the previous and following layers. We refer to this as as a **sequential model** or a series of nested functions.

For example, {numref}`someneuralnetwork` illustrates a net with **4** layers. We can compute the output, $y$, as follows:

```{math}
:label: neural_network_formula

y = f_4[f_3[f_2[f_1[x, \phi_1], \phi_2], \phi_3], \phi_4],
```

where $x$ is the input, $f_i[x, \phi_i] = h_i$ represents an intermediate hidden layer, $\phi_i$ are the learned parameters at $i$th hidden layer and $f_i$ is the function performing the processing at $i$th hidden layer.

```{figure} ../figures/copyrighted/some_neural_network.svg
:scale: 50%
:name: someneuralnetwork

A network with 4 hidden layers. Prince, S.J., 2023. Understanding Deep Learning. MIT PRESS.
```

Additionally, you might remember about VGG or AlexNet (two CNNs architectures that played an important role in the development of CNNs). We've seen that **adding more layers** to a convolutional neural network **increases performance** (VGG outperformed AlexNet as VGG has 18 layers, while AlexNet has only 8 layers). However, this increase in the number of layers does not always lead to better performance. {numref}`performance_going_down` shows that a 20-layer network performs better (i.e. lower error) than a 56-layer network. Interestingly, we see the same behavior on the training set as well, indicating that the problem relates to training the original network rather than a failure to generalize to new data. 

```{figure} ../figures/copyrighted/cnn_lower_performance.svg
:scale: 30%
:name: performance_going_down

Decrease in performance when adding more convolutional layers. Prince, S.J., 2023. Understanding Deep Learning. MIT PRESS.
```

## Gradients

To train a neural network (NN), we need to perform two steps. First, compute the **error between the target and the prediction** of the NN. Then, we use the **Stochastic Gradient Descent (SGD)** to update the parameters by taking a step in the direction opposite to the gradient. As you might now, SGD is defined by {eq}`SGD`.

```{math}
:label: SGD

\phi_{t+1} \leftarrow \phi_t - \alpha \sum_{i \in \mathcal{B}_t} \frac{\partial l_i[\phi_t]}{\partial \phi},
```

where $\phi_{t, t+1}$ represent the learnable parameters and $l_i$ is the error for the $i$th input.

Yet, to compute the SGD, we can employ **backpropagation to efficiently compute the gradients** (i.e. the derivative of the loss with respect to the learnable parameter). To compute how a small change to a weight feeding into $h_1$ (the first hidden layer) changes the loss, we need to know how $h_1$ changes $h_2$ and how these changes propagate through the loss. A backward pass first computes the derivatives at the end of the network and then moves backwards to exploit the inherent redundancy of these computations. It uses the **chain rule** of derivatives to compute partial derivatives in linear time based on the graph size.

For example, considering the 4 layer network presented in {numref}`someneuralnetwork`, we can compute the derivative with respect to $f_1$ as follows:

```{math}
:label: simple_derivative

\frac{\partial y}{\partial f_1} = \frac{\partial f_4}{\partial f_3} \frac{\partial f_3}{\partial f_2} \frac{\partial f_2}{\partial f_1}.
```

```{admonition} Clarification    
:class: tip 

**Backpropagation** is an efficient method for **computing gradients** in directed graphs, such as neural networks, and serves as a computational technique rather than a learning method itself; it utilizes the chain rule of derivatives to compute partial derivatives in linear time based on the graph size. 

In contrast, **Stochastic Gradient Descent (SGD) is an optimization method** that **relies on analyzing the gradients** obtained from backpropagation to make efficient updates during the learning process.
```

## Shattered Gradients

Now that we saw how to compute the gradients, an interesting question might arise. Does the length of the network have any impact on the gradients?

To answer this question, consider the following three plots from {numref}`shattered_gradients`. For a shallow network, small changes in the input lead to **small adjustments in the output gradient** ({numref}`shattered_gradients`, left). In contrast, a minor change in the input of a deep network can produce a **drastically different gradient** ({numref}`shattered_gradients`, middle). This behavior is illustrated by the autocorrelation function of the gradient ({numref}`shattered_gradients`, right), where gradients in shallow networks show strong correlation, while this **correlation diminishes rapidly to zero** in deep networks. The rapid drop in gradient correlation in deep networks can lead to **unstable training, poor generalization, and inefficient learning**, ultimately affecting the model's performance. This phenomenon is known as **shattered gradients**.

```{figure} ../figures/copyrighted/shattered_gradients.svg
:name: shattered_gradients
:scale: 50%

For networks with a signle input and output, we see the values of the gradient on a shallow neural network (left), values of the gradient on a deeper neural network (middle) and the autocorrelation function of the gradient (right). Prince, S.J., 2023. Understanding Deep Learning. MIT PRESS.
```

## Residual Connections

A possible solution to the problem described above is represented by the use of **Residual** or **Skip** connections. They are branches in the computational path, where the input of each layer is added to the output. It is represented mathematically in {eq}`residual_connection`. For example, to compute $h_2$, we sum up the input that is fed in $h_2$, that is, $h_1$, and the output of $f_2$, as in the original forward pass.

```{math}
:label: residual_connection

\begin{array}{rl}
h_1 &= f_1[x, \phi_1] \\
h_2 &= f_2[h_1, \phi_2] \\
h_3 &= f_3[h_2, \phi_3] \\
y &= f_4[h_3, \phi_4] \\
\end{array}
\Longrightarrow
\begin{array}{rl}
h_1 &= x + f_1[x, \phi_1] \\
h_2 &= h_1 + f_2[h_1, \phi_2] \\
h_3 &= h_2 + f_3[h_2, \phi_3] \\
y &= h_3 + f_4[h_3, \phi_4] \\
\end{array}
```

If we put together the formulas presented in {eq}`residual_connection`, the output is computed as follows:
```{math}
:label: residual_connection_extended

\begin{aligned}
y &= x + f_1[x] \\
  & \quad + f_2[x + f_1[x]] \\
  & \quad + f_3[x + f_1[x] + f_2[x + f_1[x]]] \\
  & \quad + f_4[x + f_1[x] + f_2[x + f_1[x]] + f_3[x + f_1[x] + f_2[x + f_1[x]]]]
\end{aligned}
```

Alternatively, we can visualize {eq}`residual_connection_extended` in {numref}`residual_connection_extended_figure`. This figure might be a bit hard to comprehend, but follow along and you will understand. Let's say we want to compute the output the output for $f_2$. We see on the last line of the graph that $x + f_1[x]$ are fed into $f_2$. Yet, we are missing both $f_1[x]$ and $x$ from the final sum. Therefore, after $f_2$ on the last row, we observe that the output of $f_2$ is summed up with both $f_1[x]$ and $x$, resulting in the correct output. The same logic applies for the rest of the layers.

```{figure} ../figures/copyrighted/residual_connection_extended_figure.svg
:name: residual_connection_extended_figure
:scale: 50%

Residual connections in a 4 layer network. Prince, S.J., 2023. Understanding Deep Learning. MIT PRESS.
```

### Gradients in Residual Connections

We've seen previously in {eq}`simple_derivative` how to efficiently compute the derivative of the loss with respect to $f_1$. Yet, in the context of a network with residual connections, the formula is presented in {eq}`residual_connections_derivative`.

```{math}
:label: residual_connections_derivative

\frac{\partial y}{\partial f_1} = I + \frac{\partial f_2}{\partial f_1} + \left( \frac{\partial f_3}{\partial f_1} + \frac{\partial f_3}{\partial f_2} \frac{\partial f_2}{\partial f_1} \right) + \left( \frac{\partial f_4}{\partial f_1} + \frac{\partial f_4}{\partial f_2} \frac{\partial f_2}{\partial f_1} + \frac{\partial f_4}{\partial f_3} \frac{\partial f_3}{\partial f_1} +  \frac{\partial f_4}{\partial f_3} \frac{\partial f_3}{\partial f_2} \frac{\partial f_2}{\partial f_1} \right)
```

Generally, gradients that follow shorter paths tend to behave more predictably. Since both the identity term and various short chains of derivatives will contribute to the derivative for each layer, networks with residual links suffer less from shattered gradients.

## Residual Blocks

So far, we have implied that the additive functions $f(x)$ could be any valid network layer (e.g. fully connected or convolutional). This is mostly true, but the order of the operations in these functions is imperative. They must contain a nonlinear activation function like a ReLU, or the entire network will be linear. We identified 3 important orders of operations in residual blocks.

- First, we place the ReLU activation function at the end ({numref}`residual_blocks`, a), so the output will always be non-negative. However, if we adopt this convention, then each residual block can only increase the input values. 

- Given this impediment, we typically change the order of operations such that the network starts with a linear transformation, followed by a residual block, illustrated in {numref}`residual_blocks`, b. Moreover, in the remaining of the network, we first apply the activation function (e.g. ReLU), then another linear transformation. 

- Sometimes, there may be several layers of processing within the residual block ({numref}`residual_blocks`, c), but these usually end with a linear transformation. Lastly, we note that starting a residual block with a ReLU operation, then it will not do anything if the initial input is all negative, as the ReLU operation will transform the entire input to zero. Therefore, it's typically better to start a network with a linear transformation, as in {numref}`residual_blocks`.b, rather than a residual block.

```{figure} ../figures/copyrighted/residual_blocks.svg
:name: residual_blocks
:scale: 50%

In part (a) of the figure, the standard order of operations applies, where a linear transformation or convolution is followed by a ReLU activation. In part (b), the order is reversed, allowing both positive and negative quantities to be combined, although an initial linear transformation is necessary to handle potentially negative inputs. Part (c) illustrates that, a residual block typically consists of multiple network layers, enhancing its overall complexity and expressive capability. Adapted from Prince, S.J., 2023. Understanding Deep Learning. MIT PRESS.
```

## Batch Normalization

Residual connections help ensuring stable gradients during backpropagation. Yet, the residual connections may lead to **exponential growth of the variance**. Such exponential variance can trigger **numerical instability** and **exploding gradient** issues, disrupting model training. Fortunately, we can address it using **batch normalization**.

**Batch Normalization** (also known as BN) corrects layer output distributions **per batch** $\mathbf{\mathcal{B}}$. Below, we will illustrate how does BN work.

1. **Mean and standard deviation** of the batch are computed, as in {eq}`mean_std_bn`.

```{math}
:label: mean_std_bn

m_h = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} h_i \quad \quad s_h = \sqrt{\frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} (h_i - m_h)^2}
```

2. **Standardize** the batch activations to have **mean zero** and **unit variance** ({eq}`standardize_bn`).

```{math}
:label: standardize_bn

h_i \leftarrow \frac{h_i - m_h}{s_h + \epsilon} \quad \forall i \in \mathcal{B}
```

3. The normalized variable ($h_i$) is **scaled** by $\gamma$ and **shifted** by offset $\delta$.

```{math}
:label: scale_and_shift_bn

h_i \leftarrow \gamma h_i + \delta \quad \forall i \in \mathcal{B}
```

After these operations, the activations ($h_i$) have **mean** $\mathbf{\delta}$ and **standard deviation** $\mathbf{\gamma}$ across the batch. Both quantities ($\delta, \gamma$) are **learned during training**. Note that we apply BN independently to each hidden unit.

```{admonition} Number of learned parameters
:class: tip

A question that might come up is related to the number of additional learnable parameters. As a rule of thumb, for each hidden unit (node) we have **two additional learnable parameters**.

For example, in an **MLP** with $K$ layers, and $D$ hidden units in each layer, we will end up with $2 KD$ learnable parameters. In the case of a **CNN** with $K$ layers, each containing $C$ channels, we get $2 KC$ learnable parameters.
```

### Benefits of Batch Normalization

- **Maintains the variance of activations**, helping to avoid the disappearing or exploding gradient phenomena during training.

- **Combats the covariate shift** (different distributions in input data for different batches, which can confuse the learning model) by ensuring that each layer receives inputs with a consistent distribution.

- Allows for the **use of higher learning rates**, which can dramatically speed up network training.

### Cons of Batch Normalization

- Introduces **redundancy in weight parameters** and compensates with additional parameters, enlarging the model.

- **Relies on batch statistics**, which can vary, potentially affecting stability in smaller batches or at inference.

- **Adds complexity to the network's architecture**, requiring careful implementation, especially during the inference phase (i.e., testing). BN is typically applied after convolutional layers but before the non-linear activation functions. At inference time, BN relies on the entire population's statistics rather than individual mini-batch statistics for stability.

## ResNet

Residual connections are now a standard part of deep learning pipelines. We will discuss about **ResNet**, a common residual architecture.

In ResNets, each residual block includes a batch normalization operation, a ReLU activation function, and a convolutional layer. This sequence is repeated before the output is added back to the original input (see {numref}`resnet`.a). **Empirical evidence suggests that this order of operations is effective for image classification.**

In very deep networks, the **parameter count can grow excessively large**. Bottleneck residual blocks optimize parameter usage by employing **three convolutions**. The first convolution uses a $1 \times 1$ kernel to **reduce the number of channels**, followed by a standard $3 \times 3$ kernel, and finally another $1 \times 1$ kernel to restore the channel count to its original size ({numref}`resnet`.b). This approach allows us to aggregate information over a $3 \times 3$ pixel area while using fewer parameters.

```{figure} ../figures/copyrighted/resnet.svg
:name: resnet
:scale: 30%

ResNet blocks. a) A standard block in ResNet architecture and b) a bottleneck ResNet block, used for deep networks. Adapted from Prince, S.J., 2023. Understanding Deep Learning. MIT PRESS.
```

### ResNet - 200

A variation of the ResNet architecture with 200 layers. It is similar to AlexNet and VGG but employs bottleneck residual blocks in place of standard convolutional layers. These blocks are periodically integrated with reductions in spatial resolution and simultaneous increases in the number of channels. In terms of performance, it achieved a **4.8\%** top-5 error rate on ImageNet, **outperforming the human** (who achieved 5.1\% error).

## Changing the number of channels

We saw that ResNet, initially, reduces the number of channels in very deep networks by applying the $1 \times 1$ kernels. Each element of the output layer is calculated by taking a **weighted sum** of all channels at the corresponding position (see {numref}`apply_1_x_1_conv`). This process can be repeated multiple times with different weights to produce as many output channels as required. The corresponding convolution weights are sized $1 \times 1 \times C_i \times C_o$, which is why it is referred to as $1 \times 1$ convolution. **When combined with a bias and activation function, this is equivalent to applying the same fully connected network to the input channels at every position.**

```{figure} ../figures/copyrighted/apply_1_x_1_conv.svg
:name: apply_1_x_1_conv
:scale: 30%

$1 \times 1$ convolution. Adapted from Prince, S.J., 2023. Understanding Deep Learning. MIT PRESS.
```

### Advantages of using the $1 \times 1$ convolutions

- **Dimensionality Reduction**: Reduces depth, controls complexity, and minimizes parameters in CNNs.
- **Channel-wise Feature Integration**: Combines features across channels for enhanced feature representation.
- **Efficient Network Depth Increase**: Adds depth with minimal computational cost, improving learning capacity.

## Transfer Learning

It might happen that we want to train a model for some task, but we lack enough training data. By applying **transfer learning**, we can share the knowledge learned by powerful neural networks on other, larger datasets to solve a different but related problem.

A common architecture that makes use of transfer learning is depicted in {numref}`transfer_learning`, proposed by Tao, Wenjin, et al. Their observations suggest that, typically, the source dataset consists of a substantial amount of annotated data is used to train a deep learning model. For instance, a model features a series of convolutional layers that progressively extract the most discriminative features, followed by a series of dense layers that connect these features to the source labels. Once the source model is trained, part of its architecture along with the trained weights is frozen and transferred to a target domain. For the target model (i.e. the model used on a dataset with not enough training data), a new classifier, usually composed of dense layers, is required to adapt the source model to the target labels.

```{figure} ../figures/copyrighted/transfer_learning.svg
:name: transfer_learning
:scale: 30%

The architecture of a transfer learning model, proposed by Tao, Wenjin, et al. "Real-time assembly operation recognition with fog computing and transfer learning for human-centered intelligent manufacturing." Procedia Manufacturing 48 (2020): 926-931.
```

## Takeaways

- An indefinite increase in network depth can degrade both training and test performance in image classification.

- This degradation is primarily due to unpredictable changes in early network parameter gradients, referred to as **shattered gradients**.

- **Residual connections** reintegrate the processed output back into their input, smoothing the loss surface.

- Layers in residual networks contribute both directly and indirectly to the output, facilitating gradient propagation through many layers.

- Residual networks may experience **exponentially increasing activation variances** and **exploding gradients**.

- **Batch normalization** in residual networks mitigates variances by normalizing and rescaling based on learned parameters.

- Both residual connections and batch normalization help smooth the loss surface, enabling higher learning rates and providing regularization.

- **Transfer learning** utilizes existing pre-trained models to accelerate training for new, related tasks.

- This approach can significantly reduce training time and data requirements for effective model development.

- It is particularly beneficial when labeled data is scarce, leading to improved model accuracy and reliability.

- Transfer learning is widely applicable across various fields, including vision and language processing (e.g., ChatGPT's transfer from GPT), among others.

## Resources

[Slides](https://surfdrive.surf.nl/files/index.php/s/x3xoVWp93BfoOul)

Chapter 11 until 11.5.1 + 11.6 {bdg-warning}`prince-udl`