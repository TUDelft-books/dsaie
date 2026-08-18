# Curse of Dimensionality and Inductive Biases

This introductory lecture covers the basics of deep learning, focusing on the **curse of dimensionality** and how **inductive biases** help models handle high-dimensional data. It explains different types of inductive biases, such as *spatial*, *sequential*, *hierarchical*, *graph*, and *physics-informed* biases, and how they are implemented in various deep learning architectures like CNNs, RNNs, and Transformers. The lecture aims to provide a foundational understanding of deep learning models and their effectiveness.

**Resources** 

[Slides](https://surfdrive.surf.nl/files/index.php/s/B4SBNinra6farfD/download/)

```{admonition} Clarification    
:class: tip    
Inductive biases, which we'll explore in this lecture, should not be confused with other types of bias in machine learning and AI. You have already encountered the **bias term** in neural network layers (sometimes identified mathematically with *b*) that acts as an offset parameter in neurons, allowing the model to shift its activation function. Also, you might have heard of **societal bias** that refers to systematic unfairness or prejudice in AI systems, which can lead to discriminatory outcomes against certain groups. These distinct concepts serve different purposes and operate at different levels of machine learning systems.
+++                         
```

## The Curse of Dimensionality

One of the challenges in processing and learning from high-dimensional data is related to the amount of data needed to estimate a real function, known as the **curse of dimensionality**. More formally, the number of samples required to estimate a given function increases **exponentially** in relation to the number of input variables, that is, the function's dimensionality [1].

Let's consider an example. We examine the relationship between an input feature $x$ and an output variable $y$, which can be expressed as:

$$y = f(x)$$(realfunction)

In this equation, $f$ is the real function that maps the input x to the output y. This relationship can be approximated with a parametric model:

$$y \approx f'(\theta, x)$$(approxfunc)

where $f'(\theta, x)$ relies on the learnable parameter $θ$ and the input feature x. 

The plot below shows how the output (the green line) varies with different values of the input feature ($X$).

```{figure} ../figures/dl_example_plot.svg
:scale: 15%
:name: dlexampleplot

Relationship between input feature $X$ and output $y=f(x)$
```

Continuing, we can take 20% (or any amount) from the feature space to train our estimator. In the case of 1-dimensional data, 20% of the feature space represents 20% from the entire input space. However, let's see how the latter fraction **scales** as we **increase the number of dimmensions of the data**.

When we have **2-dimensional data**, sampling 20% from each feature represents at most **4%** from the input space, and this is only achieved in the ideal case where:all sampled points have different $x_1$ and $x_1$ coordinates. In practice, random sampling typically results in even less coverage of the input space due to coordinate overlaps and non-uniform distribution of points.

```{figure} ../figures/2d_cod.svg
:scale: 15%
:name: 2dcod

In a 2D space, sampling 20% from each feature represents only 4% from the entire input space.
```

In the case of a **3-dimensional data**, we see that sampling 20% from each feature only yields **0.8%** of all possible combinations in the input space. We need to sample $\mathbf{\sim 58\%}$ from each feature space to be able to represent 20% of the input space.

<!-- ````{card} -->
<!-- **Sampled Data vs. Input Space in High Dimensions** -->
<!-- ^^^ -->

These examples illustrate that as the number of dimensions increases, the amount of data required to adequately represent the entire **input space** grows **exponentially**. To quantify this relationship, we can use the following formula:  

$$
\text{sampled fraction (per feature)} = \text{input fraction}^{1/d}
$$

Here, the "input fraction" refers to the proportion of the total multidimensional input space we aim to cover, and $d$ denotes the number of dimensions (features) in the data. The "sampled fraction (per feature)" represents the fraction of each individual feature's range that must be sampled to achieve this coverage in the input space.  

Conversely, if we know the fraction of each individual feature range that we have sampled, we can calculate the total fraction of the input space that is covered by the sampled data as:  

$$
\text{input fraction} = (\text{sampled fraction (per feature)})^d
$$

The relationship can be visualized in the following plot. It demonstrates how, as the number of dimensions $d$ increases, the **sampled fraction (per feature)** (i.e., the "edge length" of the hypercube) must approach 1 in order to maintain a **fixed fraction of the input space**. This showcases the exponential relationship between the number of dimensions and the amount of data required to adequately represent the input space.  

```{figure} ../figures/cod_overall_plot.svg
:scale: 70%
:name: 2dcod
A plot showing the required fraction of each feature's range (Edge Length) to represent 20% of the input space (Fraction of Data) as a function of the number of features in the dataset.
```

## Consequences of the Curse of Dimensionality in Machine Learning

In the context of machine learning, increasing the number of features in a dataset introduces the **curse of dimensionality**, which significantly impacts model performance and computational efficiency. As the dimensionality grows, we require **disproportionately large amounts of data** to adequately represent the input space and train accurate models. This not only increases memory usage and computational costs but also exacerbates learning challenges.

High-dimensional data can lead to both **underfitting and overfitting**. **Underfitting** occurs when the model fails to capture important patterns in the high-dimensional space due to data sparsity—there simply isn't enough data to learn the underlying structure across all dimensions. **Overfitting** becomes particularly problematic in high dimensions because the vast feature space makes it easier for the model to find apparent patterns that are actually just noise. As the dimensionality increases, the available data becomes increasingly sparse relative to the size of the feature space, making it more likely for the model to identify spurious correlations that don't represent true relationships in the data. These spurious patterns **fail to generalize to unseen data**, resulting in poor test set performance.

To overcome the curse of dimensionality in deep learning, we rely on **inductive biases**—assumptions built into the model architecture that help constrain the learning process and make it more efficient in high-dimensional spaces.

## Inductive Bias(es)

Inductive biases are **built-in assumptions** that models make about the structure of data and the relationships they should learn. These biases effectively **reduce the hypothesis space**, making learning more tractable by guiding the model towards certain types of solutions. This is particularly important for overcoming the curse of dimensionality, as we just discussed.

The strength of an inductive bias significantly impacts model performance. A well-chosen, strong inductive bias can help the model:
- Learn efficiently from fewer samples
- Converge to better solutions
- Generalize more effectively to unseen data

In contrast, a weak or inappropriate inductive bias may lead to poor performance, making the model sensitive to initialization and prone to getting stuck in suboptimal solutions.

Continuing, we will discuss some interesting inductive biases that are used in **deep learning**.

### Spatial Inductive Bias
Spatial inductive bias refers to built-in assumptions about how spatial relationships in data should influence learning and predictions. Two key principles of spatial inductive bias are:
1. **Locality**: Nearby points are more likely to influence one another than distant points, reflecting how spatial relationships typically work in reality.
2. **Translation Invariance**: A model's predictions should remain consistent regardless of spatial position. For instance, a cat should be recognized as a cat whether it appears in the top-left or bottom-right of an image.

These principles are critical in domains like computer vision, where the spatial arrangement of features matters.

#### The Problem with Fully Connected Networks
Fully Connected Networks (e.g., MLPs) treat each input pixel **independently**, without considering spatial relationships. This means a simple shift or translation of an object in the image makes the network process it entirely differently, requiring the model to repeatedly learn the same features for every possible position. This is highly inefficient, as illustrated below:

```{figure} ../figures/starry_night_01.png
:scale: 50%
:name: starrymlp
MLPs process each pixel independently and treat translations differently, requiring the network to relearn the same features at every possible position.
```

#### Convolutional Layers
Convolutional layers address this inefficiency by leveraging **spatial inductive bias**. Instead of treating each pixel independently, they process an image by scanning (small) overlapping regions, learning local representations across the input data. This approach captures local patterns (such as edges and shapes) and applies the **same weights** across all positions, making the model inherently **translation invariant**. As a result, convolutional layers are far more efficient for image processing:

```{figure} ../figures/starry_night_0203.png
:scale: 40%
:name: starryconv
Left: A convolutional layer learns multiple filters, each focusing on different features of interest (e.g., "stars" or "swirls"). Right: The same filters are applied across the entire image, enabling the detection of these features at different positions, demonstrating translation invariance and efficient parameter sharing.
```


#### Convolutional Neural Networks (CNNs)
Convolutional Neural Networks (CNNs) build on the principles of convolutional layers to efficiently process structured grid data. By stacking multiple convolutional layers, CNNs create a hierarchical representation of features, capturing increasingly abstract patterns as the network depth increases. This architecture, combined with the spatial inductive biases of locality and translation invariance, enables CNNs to achieve remarkable performance in tasks like image recognition while using far fewer parameters than traditional neural networks.

```{figure} ../figures/cnn.svg
:scale: 60%
:name: cnnillustration
Architecture for a simple CNN. Used from *Murphy, K.P., 2022. Probabilistic machine learning: an introduction. MIT press*.
```

#### Sequential Inductive Bias
Sequential inductive bias incorporates assumptions about how elements in a sequence relate to each other, making it particularly valuable for processing **sequential data** such as time series or text. The key principle is that data points in a sequence are not independent: their order matters, nearby elements often share strong relationships, and patterns may repeat throughout the sequence.

This type of inductive bias is fundamental to **Recurrent Neural Networks (RNNs)**, which are designed to process sequential data. RNNs implement this bias through a hidden state that carries information from previous timesteps, allowing the network to capture temporal dependencies and recurring patterns:

```{figure} ../figures/copyrighted/simple_cell_rnn.svg
:scale: 100%
:name: rnnillustration
A Simple Recurrent Cell maintains a hidden state $h_t$ that captures information from previous timesteps, enabling the network to process sequential data. From Kratzert, Frederik, et al. "Rainfall–runoff modelling using long short-term memory (LSTM) networks." Hydrology and Earth System Sciences 22.11 (2018): 6005-6022.
```

#### Hierarchical Inductive Bias
Hierarchical inductive bias embodies the principle that higher-level features can be constructed through the composition of simpler, lower-level features. In deep learning, *consecutive layers capture features at various levels of abstraction*, enabling models to build increasingly complex representations of the data.

Returning to CNNs, which we discussed earlier, their multi-layer architecture provides an excellent example of hierarchical learning in action. The initial layers of a CNN process raw pixel values to identify basic elements like **edges** and **textures**. These simple features are then progressively combined in subsequent layers to recognize more complex patterns and **parts of objects**. As the network deepens, neurons can gather information from increasingly larger regions of the input, allowing the detection of more sophisticated, higher-order features. Through this hierarchical process, deeper layers learn to represent specific object parts, which are ultimately assembled in the final layers for instance to perform complete object recognition.

This hierarchical organization, combined with the spatial inductive bias we discussed earlier, makes CNNs particularly effective at learning meaningful representations from visual data. The network naturally builds up its understanding from simple to complex features, mirroring how humans often perceive and understand visual information.

```{figure} ../figures/copyrighted/multi_layer_cnn.svg
:scale: 60%
:name: multi_layer_cnn
The multi-layer architecture of CNNs illustrates hierarchical feature learning, from simple edges to complex object features. From *Katole, Atul Laxman, et al. "Hierarchical deep learning architecture for 10k objects classification." arXiv preprint arXiv:1509.01951 (2015)*.
```

#### Physics Inductive Bias
When modeling real-world physical systems, such as heat diffusion or electromagnetic fields, we often know the underlying scientific principles that govern their behavior. This knowledge can be incorporated directly into neural networks as an inductive bias, ensuring the model's predictions align with established physical laws.

**Physics-Informed Neural Networks (PINNs)** exemplify this approach. Unlike traditional neural networks that learn purely from data, PINNs combine data-driven learning with physical constraints derived from scientific equations. This hybrid approach offers two key advantages: it requires less training data since the model is already guided by physical principles, and it produces more interpretable results since its predictions must follow known physical laws. PINNs are typically implemented as MLPs and have proven particularly effective in solving complex problems described by partial differential equations.

```{figure} ../figures/copyrighted/pinn_framework.png
:scale: 60%
:name: intropinnframework
Framework for PINNs. From Cuomo, Salvatore, et al. "Scientific machine learning through physics–informed neural networks: Where we are and what’s next." Journal of Scientific Computing 92.3 (2022): 88.
```

#### Other Examples

##### Graph Inductive Bias
Unlike Euclidean data (e.g., images and sequences that exist in regular grids), graphs consist of nodes connected by edges, making their **relationships complex and irregular**. Graph-based inductive bias helps models understand local connectivity and **node relationships**, enabling them to capture patterns and interactions within graph structures. This is particularly valuable in applications involving social networks, molecular simulations, and recommendation systems (i.e., what Amazon or Netflix use to suggest new purchases by looking at your history). In our fields, graphs are particularly useful to model infrastructure networks or irregular meshes, typically used for computational mechanics and computational fluid dynamics.

##### Transformer Networks
Transformers revolutionized deep learning by introducing powerful inductive biases that were initially designed for natural language processing. The first step in a transformer involves **tokenization**—breaking down the input (originally text) into meaningful pieces. For text, these tokens might be words or subwords, each converted into a numerical representation. The **attention mechanism** then allows the model to focus on relevant parts of the input sequence, capturing relationships between tokens regardless of their distance. **Positional encoding** complements this by providing each token with position-specific information, helping maintain sequence order awareness.

The success of transformers has led to their adaptation for other types of data. For instance, in computer vision, images can be split into patches (acting as "tokens"), allowing transformers to process visual data effectively.

##### Combining Inductive Biases
Modern deep learning often benefits from combining multiple inductive biases to tackle complex problems. For instance, spatiotemporal learning combines spatial and sequential biases to analyze data that evolves both in space and time. Convolutional LSTMs (ConvLSTMs) exemplify this approach, using convolutional operations for spatial features while maintaining temporal dependencies through recurrent connections. This makes them particularly effective for tasks like video prediction or weather forecasting.

```{figure} ../figures/copyrighted/convlstm.png
:scale: 100%
:name: convlstm
Architecture of a ConvLSTM, combining spatial and sequential inductive biases. From https://cse.hkust.edu.hk/pg/research/projects/dyyeung/ml-hko/.
```

This integration of multiple inductive biases, along with hierarchical representations, enables models to capture increasingly complex patterns and relationships in data, leading to more robust and efficient learning systems.

## Conclusions
In this lecture, we explored how the **curse of dimensionality** poses fundamental challenges for machine learning in high-dimensional spaces. We saw how **deep learning** addresses these challenges through various inductive biases—assumptions that help models learn efficiently from data. From **spatial** biases in CNNs and **sequential** biases in RNNs to **physics-based** constraints in PINNs, each architecture leverages specific inductive biases to excel at particular types of problems. Modern architectures like **Transformers** demonstrate how powerful inductive biases (such as attention mechanisms) can be combined with flexible architectures to handle increasingly complex data relationships effectively.

```{admonition} Exam questions    
:class: danger    
The curse of dimensionality and inductive biases are foundational concepts in deep learning that you need to understand for the exam. You should be able to explain how the curse of dimensionality affects learning in high dimensions and why inductive biases are necessary for effective generalization. You do not need to memorize specific definitions or formulas, but should understand the key principles and their implications for machine learning models.
+++    
{bdg-primary}`written-exam`    
```

## References
[1]: Bellman R.E. Adaptive Control Processes. Princeton University Press, Princeton, NJ, 1961
