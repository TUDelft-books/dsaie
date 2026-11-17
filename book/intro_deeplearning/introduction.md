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

One of the challenges in processing and learning from high-dimensional data is related to the amount of data needed to estimate a real function, known as the **curse of dimensionality**. Without assumptions that exploit structure in the target function or data, the number of samples required can increase **exponentially** with the number of input variables, that is, the function's dimensionality ({ref}`Bellman, 1961 <introdl-ref-bellman>`).

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

Continuing, let's normalize each input feature's range to $[0, 1]$ and consider a region spanning 20% of each range. In one dimension, this is an interval of relative length 0.2. In $d$ dimensions, it is a **hypercube**, the higher-dimensional counterpart of a square or cube. Let's see how its **relative volume** changes as we **increase the number of dimensions**.

In **2 dimensions**, a square spanning 20% of each feature's range occupies $0.2^2 = 0.04$, or **4%**, of the total area.

```{figure} ../figures/2d_cod.svg
:scale: 15%
:name: 2dcod
:alt: A square spanning 20% of each of two feature ranges occupies 4% of the total area.

In a 2D space, a square with relative side length 0.2 occupies 4% of the total area.
```

In **3 dimensions**, a cube with the same relative side length occupies $0.2^3 = 0.008$, or **0.8%**, of the total volume. To occupy 20% of the volume, the cube must span $0.2^{1/3} \approx 58.5\%$ of each feature's range.

In $d$ dimensions, a hypercube with relative side length $a$ has relative volume $v = a^d$. As {numref}`cod-overall-plot` shows, occupying a fixed fraction of the volume requires a larger fraction of each feature's range as the number of dimensions increases.

```{figure} ../figures/cod_overall_plot.svg
:width: 560px
:name: cod-overall-plot
:alt: Relative side length versus relative volume for dimensions 1, 2, 3, 5, 10 and 100, with a vertical guide at volume 0.2.

Relative side length $a = v^{1/d}$ as a function of relative volume $v$ for different dimensions. The guide at $v = 0.2$ compares the side lengths needed to occupy 20% of the volume.
```

## Consequences of the Curse of Dimensionality in Machine Learning

The volume fractions above describe regions of the input space, rather than coverage by individual training samples. To connect this geometry to learning, divide each feature's range into $m$ intervals, creating $m^d$ grid cells. Obtaining examples throughout this grid can require exponentially many samples to maintain a fixed resolution, unless the function or data has structure we can exploit. Collecting and processing more data can increase memory usage and computational costs.

```{figure} ../figures/introdl_grid_samples.svg
:width: 480px
:name: introdl-grid-samples
:alt: Five intervals per feature create 5 cells in one dimension, 25 in two and 125 in three. The same five sample points occupy five cells in each view, leaving 20 empty cells in two dimensions and 120 in three. The 3D grid is shown as five layers along the third feature.

Five intervals per feature produce $5^d$ grid cells. The same five samples occupy five cells in each view; each dot marks an individual observation within a cell. The 3D grid is shown as five layers along $x_3$, with $x_1$ horizontal and $x_2$ vertical in each layer. Placing even one sample in every cell would require at least $5^d$ samples.
```

**Overfitting** becomes particularly problematic in high dimensions because the vast feature space makes it easier for the model to find apparent patterns that are actually just noise. As the dimensionality increases, the available data becomes increasingly sparse relative to the size of the feature space, making it more likely for the model to identify spurious correlations that don't represent true relationships in the data. These spurious patterns **fail to generalize to unseen data**, resulting in poor test set performance.

High dimensionality can also contribute to **underfitting**. To control overfitting with sparse data and keep computation manageable, we may simplify the model or strengthen regularization. If these restrictions prevent it from capturing important patterns even in the training data, it underfits.

To address these challenges in deep learning, we rely on well-chosen **inductive biases**—assumptions built into the model architecture that help constrain the learning process and make it more efficient in high-dimensional spaces.

## Inductive Bias(es)

Inductive biases are **built-in assumptions** that models make about the structure of data and the relationships they should learn. These biases effectively **reduce the hypothesis space**, making learning more tractable by guiding the model towards certain types of solutions. This is particularly important for overcoming the curse of dimensionality, as we just discussed.

The strength of an inductive bias significantly impacts model performance. A well-chosen, strong inductive bias can help the model:

- Learn efficiently from fewer samples
- Converge to better solutions
- Generalize more effectively to unseen data

In contrast, a weak or inappropriate inductive bias may lead to poor performance, making the model sensitive to initialization and prone to getting stuck in suboptimal solutions.

Continuing, we will discuss some interesting inductive biases that are used in **deep learning**.

### Spatial Inductive Bias
Spatial inductive bias refers to built-in assumptions about how spatial relationships in data should influence learning and predictions. Two key principles are:

1. **Locality**: Nearby points are more likely to influence one another than distant points, reflecting how spatial relationships typically work in reality.
2. **Weight sharing**: The same learned weights are used at every spatial position. Think of sliding a small edge detector across an image: it checks for the same pattern wherever it is applied, so the network can learn the detector once and reuse it throughout the image.

Sharing weights makes convolution **translation equivariant**: when an edge moves, the detector responds at its new location. The grid of responses, called a **feature map**, therefore shifts with the input.

For image classification, we often want **translation invariance**: a cat should be recognized whether it appears in the centre or the bottom-right corner of an image. **Pooling and aggregation** can help achieve this property; we will discuss them in greater detail in upcoming lectures.

#### The Problem with Fully Connected Networks
Fully Connected Networks (e.g., MLPs) can learn relationships between pixels, but do not build in **locality** or **share weights across spatial positions**. A shift of an object changes which input weights act on its pixels, so the network may need to learn similar feature detectors at different positions. This can be inefficient, as illustrated below:

```{figure} ../figures/starry_night_01.png
:scale: 50%
:name: starrymlp
:alt: Image pixels connected to an MLP, with different input weights acting on features at different spatial positions.

MLPs combine information across pixels but lack built-in locality and weight sharing across positions, so similar feature detectors may need to be learned at different positions.
```

#### Convolutional Layers
Convolutional layers implement these principles by applying learned **filters** to small, overlapping regions of an image. Each filter is a set of weights that acts as a local detector for a pattern, such as an edge. These weights are **shared across spatial positions**, so the same filter produces a feature map of its responses across the image:

```{figure} ../figures/starry_night_0203.png
:scale: 40%
:name: starryconv
:alt: Convolutional filters detect local image features using the same weights at different positions.

Left: A convolutional layer learns multiple filters, each focusing on different features of interest (e.g., "stars" or "swirls"). Right: The same filters are applied across the entire image. Shifting a feature shifts its response in the feature map, illustrating translation equivariance and parameter sharing.
```


#### Convolutional Neural Networks (CNNs)
Convolutional Neural Networks (CNNs) build on the principles of convolutional layers to efficiently process structured grid data. By stacking multiple convolutional layers, CNNs create a hierarchical representation of features, capturing increasingly abstract patterns as the network depth increases. This architecture, combined with the spatial inductive biases of locality and translation equivariance, enables CNNs to achieve remarkable performance in tasks like image recognition while using far fewer parameters than traditional neural networks ({ref}`LeCun et al., 1998 <introdl-ref-lecun>`).

```{figure} ../figures/cnn.svg
:scale: 60%
:name: cnnillustration
:alt: Architecture of a convolutional neural network with successive layers of feature maps.

Architecture of a simple CNN. From {ref}`Murphy (2022) <introdl-ref-murphy>`.
```

#### Sequential Inductive Bias
Sequential inductive bias incorporates assumptions about how elements in a sequence relate to each other, making it particularly valuable for processing **sequential data** such as time series or text. The key principle is that data points in a sequence are not independent: their order matters, nearby elements often share strong relationships, and patterns may repeat throughout the sequence.

This type of inductive bias is fundamental to **Recurrent Neural Networks (RNNs)**, which process sequential data through a hidden state that carries information from previous time steps ({ref}`Elman, 1990 <introdl-ref-elman>`).

RNNs use **parameter sharing across time**: the same weights update the hidden state at each successive time step. This parallels CNNs, which reuse the same convolutional filters across spatial locations. Both assume that useful patterns or relationships can recur at different positions in space or time. For example, an RNN modelling river discharge can apply the same learned relationship between recent rainfall and flow at different times:

```{figure} ../figures/introdl_rnn_cell.svg
:width: 480px
:name: rnnillustration
:alt: A recurrent cell combines the current input and previous hidden state. Three unrolled cells share parameters and pass a hidden state through time.

A simple recurrent cell combines input $x_t$ with the previous hidden state $h_{t-1}$ to produce $h_t$. Unrolled across time, the cells share parameters and pass a hidden state that summarizes previous inputs.
```

#### Hierarchical Inductive Bias
Hierarchical inductive bias embodies the principle that higher-level features can be constructed through the composition of simpler, lower-level features. In deep learning, *consecutive layers capture features at various levels of abstraction*, enabling models to build increasingly complex representations of the data.

Returning to CNNs, which we discussed earlier, their multi-layer architecture provides an excellent example of hierarchical learning in action. The initial layers of a CNN process raw pixel values to identify basic elements like **edges** and **textures**. These simple features are then progressively combined in subsequent layers to recognize more complex patterns and **parts of objects**. As the network deepens, neurons can gather information from increasingly larger regions of the input, allowing the detection of more sophisticated, higher-order features. Through this hierarchical process, deeper layers learn to represent specific object parts, which are ultimately assembled in the final layers for instance to perform complete object recognition.

This hierarchical organization, combined with the spatial inductive bias we discussed earlier, makes CNNs particularly effective at learning meaningful representations from visual data. The network naturally builds up its understanding from simple to complex features, mirroring how humans often perceive and understand visual information.

Hierarchical representations are not exclusive to CNNs. Although particularly prominent in deep CNNs and transformers, they can also be built into RNNs by stacking recurrent layers, so that each layer processes the representations learned by the previous one.

```{figure} ../figures/introdl_hierarchical_features.svg
:width: 460px
:name: multi_layer_cnn
:alt: An original building drawing progresses through illustrative edges and textures, object parts, and a high-level representation to a building prediction.

An illustrative feature hierarchy: successive layers combine simple features into more abstract representations that support a prediction. The examples are schematic, not measured activations; learned features need not correspond to recognizable object parts or follow this exact sequence.
```

#### Physics Inductive Bias
When modelling physical systems in engineering, we often know laws that govern their behaviour. For structural elements such as beams, **stress–strain relations** and **force balance** describe how they deform under loading. For flooding, the **shallow-water equations** can be derived from the Navier–Stokes equations under shallow-flow assumptions. Such governing laws can guide learning through the **loss function**, the **model architecture**, or other **constraints**.

**Physics-Informed Neural Networks (PINNs)** provide one example. They typically add physical-equation residuals, which measure how far predictions depart from the governing equations, as penalties in the training loss. Boundary and initial conditions, together with observations where available, also guide training. Often implemented as MLPs, PINNs can learn physical fields such as displacement, water depth or velocity. **Automatic differentiation** also provides derivatives of these predictions with respect to inputs such as position and time, for evaluating the governing equations. This approach can reduce dependence on labelled data and encourage physical consistency, although satisfying the physical laws is not guaranteed ({ref}`Cuomo et al., 2022 <introdl-ref-cuomo>`).

```{figure} ../figures/introdl_pinn_framework.svg
:width: 480px
:name: intropinnframework
:alt: A neural network predicts physical quantities such as displacement, water depth or velocity. Automatic differentiation provides their spatial and time derivatives for evaluating governing equations. Physics, boundary and initial conditions, and optional observations contribute to a weighted loss that guides parameter updates.

A loss-based approach to physics-informed learning. Predictions and their derivatives, obtained through automatic differentiation, are used to evaluate the governing equations. Boundary/initial conditions and observations where available also contribute to the loss. The resulting losses guide parameter updates and encourage physically consistent predictions.
```

#### Other Examples

##### Graph Inductive Bias
Unlike Euclidean data (e.g., images and sequences that exist in regular grids), graphs consist of nodes connected by edges, making their **relationships complex and irregular**. Graph-based inductive bias helps models understand local connectivity and **node relationships**, enabling them to capture patterns and interactions within graph structures ({ref}`Battaglia et al., 2018 <introdl-ref-battaglia>`). This is particularly valuable in applications involving social networks, molecular simulations, and recommendation systems (i.e., what Amazon or Netflix use to suggest new purchases by looking at your history). In our fields, graphs are particularly useful to model infrastructure networks or irregular meshes, typically used for computational mechanics and computational fluid dynamics.

##### Transformer Networks
Transformers process inputs as **tokens**, such as words, subwords or image patches, each represented numerically. The **attention mechanism** lets each token combine information from other tokens according to their relevance, while **positional encodings** provide information about order or location ({ref}`Vaswani et al., 2017 <introdl-ref-vaswani>`).

These mechanisms introduce inductive biases, but standard transformers impose less fixed spatial locality than CNNs and do not require the recurrent processing of RNNs. This flexibility allows them to learn complex relationships from data. Their ability to scale to large models and datasets has helped make them central to modern AI, particularly large language models used in systems such as ChatGPT ({ref}`OpenAI, 2023 <introdl-ref-openai>`). The strength and form of these biases vary across transformer designs.

In Civil Engineering, Environmental Engineering, and Applied Earth Sciences, we often work with much smaller datasets than those used to train large **foundation models**, which are pretrained on broad datasets for reuse across tasks. Remote sensing is an important exception, where large image archives can be available. With limited data, stronger inductive biases that match the problem can help models learn efficiently by incorporating relevant assumptions.

This is one reason why the course focuses primarily on CNNs and RNNs: they offer useful assumptions for our applications and a foundation for understanding deep learning. **Transformers will also be covered in the course.** No architecture is inherently superior; the choice depends on the problem, available data and appropriate assumptions.

##### Combining Inductive Biases
Combining inductive biases is already common in deep learning. A sufficiently deep CNN combines **spatial inductive bias** through convolution with **hierarchical inductive bias** through successive layers of learned representations.

More sophisticated models can integrate convolutional, recurrent, attention-based and physics-informed components to capture spatial, temporal, hierarchical and physical relationships within the same system. For example, **Convolutional LSTMs (ConvLSTMs)** combine convolutional operations with temporal recurrence. They were introduced for precipitation nowcasting—predicting rainfall in the near future—from sequences of radar images ({ref}`Shi et al., 2015 <introdl-ref-shi>`).

```{figure} ../figures/introdl_convlstm.svg
:width: 480px
:name: convlstm
:alt: Three ConvLSTM cells receive spatial input grids at successive timesteps. Convolutional gates act inside each cell, and separate hidden and cell feature maps pass to the next timestep with shared parameters.

ConvLSTM cells combine spatial processing with temporal recurrence. Convolutional transformations within each cell preserve spatial feature maps, while hidden states $H_t$ and cell states $C_t$ propagate through time. The cells share parameters across timesteps.
```

The aim is to choose and combine assumptions that fit the engineering problem and the available data.

## Conclusions
In this lecture, we explored how the **curse of dimensionality** poses fundamental challenges for machine learning in high-dimensional spaces. We saw how **deep learning** addresses these challenges through various inductive biases—assumptions that help models learn efficiently from data. From **spatial** biases in CNNs and **sequential** biases in RNNs to **physics-based** constraints in PINNs, each architecture leverages specific inductive biases to excel at particular types of problems. Modern architectures like **Transformers** demonstrate how powerful inductive biases (such as attention mechanisms) can be combined with flexible architectures to handle increasingly complex data relationships effectively.

```{admonition} Exam questions    
:class: danger    
The curse of dimensionality and inductive biases are foundational concepts in deep learning that you need to understand for the exam. You should be able to explain how the curse of dimensionality affects learning in high dimensions and why inductive biases are necessary for effective generalization. You do not need to memorize specific definitions or formulas, but should understand the key principles and their implications for machine learning models.
+++    
{bdg-primary}`written-exam`    
```

## References

(introdl-ref-battaglia)=
Battaglia, P. W., et al. (2018). [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261). *arXiv preprint*, arXiv:1806.01261.

(introdl-ref-bellman)=
Bellman, R. E. (1961). [*Adaptive control processes: A guided tour*](https://press.princeton.edu/books/ebook/9781400874668/adaptive-control-processes-pdf). Princeton University Press.

(introdl-ref-cuomo)=
Cuomo, S., Schiano Di Cola, V., Giampaolo, F., Rozza, G., Raissi, M., & Piccialli, F. (2022). [Scientific machine learning through physics-informed neural networks: Where we are and what's next](https://doi.org/10.1007/s10915-022-01939-z). *Journal of Scientific Computing, 92*, Article 88.

(introdl-ref-elman)=
Elman, J. L. (1990). [Finding structure in time](https://doi.org/10.1207/s15516709cog1402_1). *Cognitive Science, 14*(2), 179–211.

(introdl-ref-lecun)=
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). [Gradient-based learning applied to document recognition](https://doi.org/10.1109/5.726791). *Proceedings of the IEEE, 86*(11), 2278–2324.

(introdl-ref-murphy)=
Murphy, K. P. (2022). [*Probabilistic machine learning: An introduction*](https://probml.github.io/pml-book/book1.html). MIT Press.

(introdl-ref-openai)=
OpenAI. (2023). [GPT-4 technical report](https://arxiv.org/abs/2303.08774). *arXiv preprint*, arXiv:2303.08774.

(introdl-ref-shi)=
Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., & Woo, W.-C. (2015). [Convolutional LSTM network: A machine learning approach for precipitation nowcasting](https://proceedings.neurips.cc/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html). *Advances in Neural Information Processing Systems, 28*, 802–810.

(introdl-ref-vaswani)=
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention is all you need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html). *Advances in Neural Information Processing Systems, 30*, 5998–6008.
