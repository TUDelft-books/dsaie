# Convolution and Pooling

This lecture dives into the foundational building blocks of convolutional neural networks: **convolutions** and **pooling**. We'll begin by demystifying convolutions, starting with their one-dimensional form. These are useful for processing 1-D gridded data such as time series, audio signals, and any sequence data. We'll dissect key concepts like *kernels* (filters), *kernel size*, *stride*, and *dilation rate*. The role of "zero" *padding* and the distinction between 'same' and 'valid' convolutions will also be clarified. We'll observe how adding a non-linear activation function transforms a convolution into a convolutional layer and discuss how these layers are specialized cases of sparsely connected layers. The lecture will explain the concept of filter banks, and how learning more filters concurrently across multiple channels greatly increases the capabilities of convolutional neural networks. We will also explore *receptive fields* and their significance.

Then we transition to **2D convolutions**, the backbone of image processing with CNNs; we'll see how the principles from the 1D case extend naturally to 2D. In this context, we'll explore **pooling** operations like *max* and *average* pooling, discussing how they help in reducing the dimensionality of feature maps, leading to more abstract representations of data. Finally, we'll discuss the concepts of *upsampling* and *transposed convolutions*, which are important for tasks that require generating outputs images (or grids) from input ones. 

By the end of this session, you'll have a solid understanding of how these elements work together in deep learning models to handle various types of data efficiently.

**Resources**

[Slides](https://surfdrive.surf.nl/files/index.php/s/i6pLtXtmlc7rV6R/)

Chapter 10.2.1-6, 10.3, 10.4.1-2 {bdg-warning}`prince-udl`

## Convolutions

In the previous lecture, we briefly introduced the idea of Convolutional Neural Networks (CNNs). Today, we will continue on that and discuss **Convolution** and **Pooling** operations that enable CNNs to work effectively.

**Convolution** is an operation that applies a filter to an input to extract its features. This process is designed to capture local patterns within the input by utilizing shared weights, such that the efficiency and effectiveness of feature extraction is improved.

### 1D Convolutions

We will start by considering 1D convolutions, as they are easier to visualize and understand.

The main idea behind 1D convolutions is to transform an input vector $x$ into an output vector $z$, such that each output $z_i$ is a **weighted sum of nearby inputs**. The **same weights** are used at every position(shift) and are usually called **convolution kernel** or **filter**. The size of the region over which
inputs are combined is named the **kernel size**.

To compute $z_i$, we will use the following formula, with the note that $\text{kernel size} = 3$:

```{math}
:label: conv_3
z_i = \omega_1 x_{i-1} + \omega_2 x_{i} + \omega_3 x_{i+1}
```

#### Example
Given this introduction, let's consider an example. The figures below illustrate how $z_2$ ({numref}`conv_1`) and $z_3$ ({numref}`conv_2`) are computed based on input sequence $x$.

```{figure} ../figures/cnn_first_conv.svg
:scale: 50%
:name: conv_1

Process on how $z_2$ is computed based on $x_1$, $x_2$, $x_3$ and the shared weights $\omega_1$, $\omega_2$, $\omega_3$
```

We see that $z_2$ is computed as follows: $
z_2 = \omega_1 x_{1} + \omega_2 x_{2} + \omega_3 x_{3}
$.


```{figure} ../figures/cnn_second_conv.svg
:scale: 50%
:name: conv_2

Process on how $z_3$ is computed based on $x_2$, $x_3$, $x_4$ and the shared weights $\omega_1$, $\omega_2$, $\omega_3$
```

As before, $
z_3 = \omega_1 x_{2} + \omega_2 x_{3} + \omega_3 x_{4}
$. Note that the **weights** (kernel) **are the same** for both cases.

#### Zero Padding and Valid Convolutions

Some might wonder how to deal with the first output ($z_1$), where there is no input in the previous position (i.e., $x_0$), and the final output ($z_6$), where there is no subsequent input (i..e, $x_7$).

One common approach is to *pad* the edges of the input with new values and then proceed as usual. In the context of *zero padding*, we assume that the values outside the initial input are set to **0**, as illustrated in {numref}`zero_padding`. One disadvantage of this method is that it introduces some arbitrary information at the edges of the input.

```{figure} ../figures/cnn_zero_padding.svg
:scale: 50%
:name: zero_padding

For $z_1$, we consider $x_{-1}$ (based on Equation {eq}`conv_3`) as being 0, such that we can still keep the original size of the data
```

Alternatively, we can discard the output positions where the kernel exceeds the limits of the input ({numref}`valid_convolution`). These *valid convolutions* overcome the issues of zero padding, but have the disadvantage that the **output decreases in size**.

```{figure} ../figures/cnn_valid_convolution.svg
:scale: 50%
:name: valid_convolution

The size of the output is decreased, and $z_1$ is now computed using $x_1$, $x_2$, $x_3$ instead.
```

### Stride, kernel size, and dilation

The **kernel size** is the region over which the inputs are weighted and summed. Usually, we aim for an **odd number**, such that the kernel is centered around the current position we are computing.

**Striding** refers to the **step size** by which the kernel moves across the input. From {numref}`conv_1` to {numref}`conv_2`, notice that we only moved **one** position to compute $z_3$. This refers to a striding of one. Striding can also be seen as a subsampling operation, similar to pooling, that will be discussed later in the lecture.

We can use **dilation**, where we can combine information over a large area using fewer weights. In other words, dilation can be seen as the **spacing** between elements from a convolutional kernel. For example, we can turn a kernel of size *five* into a **dilated** kernal of size *three* by ignoring the second and the fourth element in the input, as illustrated in {numref}`dilation`. This refers to a dilation of size 1.

```{figure} ../figures/cnn_dilation.svg
:scale: 60%
:name: dilation

Illustration of a dilated convolution with a filter size of 3, a stride of 1, and a dilation rate of 1.
```

### Creating an 1D Convolutional Layer

As for a layer in a neural network, a convolutional layer consists of (i) convoluting the input, (ii) adding a bias $\beta$, and (iii) pass the result to a non-linear activation function ($a(*)$).

For example, using a $\text{kernel size} = 3$, $\text{stride} = 1$ and $\text{dilation} = 0$, the $i^{th}$ hidden unit, $h_i$ will be computed as below. Please note that the weights (**$\omega_1, \omega_2, \omega_3$**) and bias (**$\beta$**) are the **trainable parameters**.

```{math}
:label: conv-1d
h_i = a(\beta + \omega_1 x_{i-1} + \omega_2 x_{i} + \omega_3 x_{i+1}) = \\
= a(\beta + \sum_{j = 1}^{3} \omega_j x_{i + j - 2})
```

```{note}

Consider a **fully connected layer** that computes the $i^{th}$ hidden unit as:

$$
h_i = a(\beta_i + \sum_{j = 1}^{D} \omega_{ij} x_j)
$$

If there are $D$ inputs and $D$ hidden units, then this fully connected layer will produce $\mathbf{D^2}$ weights and $\mathbf{D}$ biases. On the other hand, an 1D convolutional layer will only have $\text{kernel size}$ weights and one bias. 

A fully connected layer can reproduce 1D convolutional layer exactly if most weights are set to zero and others are constrained to be identical.
```

### Channels

When applying a convolution, it might seem that some expressive power is lost since the operation reduces inputs to a weighted sum over a small neighborhood, combined with an activation function. However, convolutional layers circumvent this limitation in two key ways: first, by leveraging the *spatial inductive bias* of filters, which are well-suited for capturing local patterns like edges, textures, and shapes; and second, by *computing multiple convolutions in parallel*, enabling the network to learn diverse features simultaneously. Each convolution produces an output called a **feature map** or **channel**, which collectively form a richer, more expressive representation. 

For example, {numref}`cnn_channels` demonstrates two parallel convolutions applied to the same input. We have a convolution with weights $\omega_{1-3}$ and a bias that produces the first channel ({numref}`cnn_channels`, left). Then, we have another convolution with weights $\omega_{4-6}$ and a bias that creates the second channel ({numref}`cnn_channels`, right). In other words, the output is now 2 dimensional, each dimension containing the output from one convolution.

```{figure} ../figures/cnn_channels.svg
:scale: 60%
:name: cnn_channels

Two convolutions are applied to create hidden units $h_{1 - 6}$ (the first channel) and $h_{7 - 12}$ (the second channel), respectively. Both convolutions have $\text{kernel size} = 3$, different biasses, and $\omega_{1-3}$ belong to the first convolution, while $\omega_{4-6}$ are part of the second one.
```

Now that we've seen how to generate multiple output channels, how do we process inputs that themselves consist of multiple channels? This multi-channel structure appears naturally in many scenarios - both at the input level (e.g., processing concurrently multiple hydrometeorological variables like temperature, precipitation, and wind speed) and throughout the network, since each convolutional layer typically computes multiple feature channels to capture diverse patterns. 

For an input with $C_i$ channels and a kernel size $K$, we create each new output channel by computing a weighted sum over all $C_i$ input channels at their $K$ closest positions, as illustrated in {numref}`2inchannels`. The weight matrix has size $C_i \times K$ and there is only one bias term.

```{figure} ../figures/cnn_two_channels.svg
:scale: 50%
:name: 2inchannels

A convolution with two input channels. The 1D convolution defines a weighted sum over both
input channels at the three closest positions (as $\text{kernel size} = 3$) to create each new output channel.
```

Abstracting even further, instead of having one output channel, we have $C_o$ output channels, then we will need $\mathbf{C_o \times C_i \times \text{K}}$ weights and $C_o$ biases, on for each channel.

### Receptive Field

The receptive field refers to the portion of the input sequence that influences a particular unit in the network. For example, {numref}`receptive_field_1`(top) shows that starting with a kernel size of seven, each unit in the first hidden layer processes seven consecutive input values. As we move deeper into the network, the receptive field grows progressively larger - each unit aggregates information from an increasingly wider portion of the input sequence. The rate at which the receptive field expands depends on two key parameters: the kernel size (which determines how many adjacent positions are processed in each convolution) and the dilation rate (which controls the spacing between kernel elements). {numref}`receptive_field_1` illustrates how the receptive field expands across multiple layers of the network.

```{figure} ../figures/receptive_field_1.svg
:scale: 50%
:name: receptive_field_1
Illustration of how the receptive field grows as we stack more convolutional layers. Each subsequent layer processes a wider portion of the original input sequence.
```

### Convolutions in 2D

We saw that 1D convolutions work with 1D data, such as time series or signals. But, what can we do when we try to process an image, as images are usually stored in a 2D format? As you might have guessed, we need to use **2D convolutions**.

Fortunately, the two types of convolutions are very similar. Recall Equation {eq}`conv-1d`. Instead of iterating through only one dimension, we will need to iterate through both dimensions (identical to iterating through a 2D matrix). Moreover, instead of having $\text{kernel size}$ weights, we will have $\mathbf{\text{kernel size} \times \text{kernel size}}$ weights and still one bias (per each output channel). For a convolution with $\text{kernel size} = 3$, we can compute the hidden unit $h_{ij}$ as:

$$
h_{ij} = a(\beta + \sum_{m = 1}^{3} \sum_{n = 1}^{3} \omega_{mn} x_{i + m - 2, j + n - 2})
$$

Unsurprisingly, applying a *valid convolution* in 2D will have the same consequence to the output - reduced size on both dimensions. To overcome that, we can still use the *zero padding* technique, but apply it to both dimensions, as illustrated in {numref}`zero_padding_2d`.

```{figure} ../figures/zero_padding_2d.gif
:scale: 50%
:name: zero_padding_2d

Applying a convolution with zero padding on 2D data. Image credits: https://towardsdatascience.com/
```

#### Multiple input channels

So far we saw 2D convolutions with only one input channel. However, each pixel in an image contains three layers (red, green, blue - RGB). Therefore, applying convolutions to a RGB image translates to applying convolutions to a 2D input with three input channels corresponding to the RGB components.

For example, using a 3×3 kernel, the computation of each pre-activation (i.e. computing the matrix of elements that will be fed into the activation function) in the first hidden layer involves pointwise multiplying the kernel weights (3×3×3) with an RGB image patch (3×3) centered at the same location, followed by summation and the addition of the bias. To determine all pre-activations in the hidden layer, the kernel is slid over the image in both horizontal and vertical directions. The resulting output is a two-dimensional layer of hidden units. To generate multiple output channels, this process is repeated with different kernels, producing a three-dimensional tensor of hidden units in hidden layer $H_1$. This process is illustrated in {numref}`rgb_conv`.

```{figure} ../figures/cnn_rgb_conv.svg
:scale: 50%
:name: rgb_conv

Applying a convolution with $\text{kernel size} = 3$ on an input with three channels.

```

#### Computing the number of learnable parameters

We need $\text{kernel size}$ or $\text{kernel size} \times \text{kernel size}$ weights (depending on the nature of the input data) and also a bias for each convolution. Therefore, let's formalize the total number of learnable parameters (i.e. weights and biases) as follows:

```{math}
:label: number_of_params
\text{number of parameters} = \\ \text{output channels} \times (\text{input channels} \times \text{kernel size} \times \text{kernel size} + 1)
```

```{admonition} Formula in other words
:class: dropdown
We start with a kernel of size $\text{kernel size} \times \text{kernel size}$, which gives us $\text{kernel size}^2$ learnable parameters. Since we apply this kernel to each input channel, the total becomes $\text{input channels} \times \text{kernel size}^2$. To generate the first output channel, we will need to add one more parametes to the number of learnable parameters, representing the bias. Lastly, in case we have multiple output channels, we will only need to multiply the number of paramters so far by the number of output channels, achieving the formula described above.
```

```{admonition} Example!
:class: dropdown tip
For example, let's consider the following data:
* Kernel Height = 3
* Kernel Width = 3
* Input channels = 64
* Output channels = 128

The total number of learnable parameters is: $128 \times (64 * 3 * 3 + 1) = 73856$.
``` 

## Pooling

Is a **subsampling** operation that typically follows the convolutional process. Its primary purpose is to reduce the dimensions of feature maps, summarizing the presence of features within the data. This reduction improves computational efficiency and helps reducing potential overfitting.

Important to note is that the **pooling does not involve any learnable parameters**.

There exist two main types of pooling: average and max pooling.

##### Average Pooling
Given a region, it computes the average value of all the values within it and replaces the original data with this average value, as illustrated in {numref}`avg_pooling`.

```{figure} ../figures/cnn_avg_pooling.svg
:scale: 50%
:name: avg_pooling

Illustration for Average Pooling
```

##### Max Pooling
Given a region, it searches the maximum value of all the values within it and replaces the original data with this maximum value, as illustrated in {numref}`max_pooling`.

```{figure} ../figures/cnn_max_pooling.svg
:scale: 50%
:name: max_pooling

Illustration for Maximum Pooling.
```


#### Upsampling
Sometimes, we might need to scale up the representation size, also known as upsampling. There exist multiple strategies for achieving that:
1. To double the resolution, you can duplicate all the channels at each spatial position four times ({numref}`upsampling`, a)
2. **Max unpooling**: this is used where we have previously used a max pooling operation for downsampling, and we distribute the values to the positions they originated from ({numref}`upsampling`, b).
3. **Bilinear interpolation**: fill in the missing values between the points where we have samples ({numref}`upsampling`, c).
4. **Transposed convolution**: consider downsampling with kernel size three, stride two, and zero-padding. Each output is a weighted sum of three inputs (illustrated in {numref}`transposed_conv`, left, where arrows indicate weights). Applying transposed convolution, each input contributes three values to the output layer, which has twice as many outputs as inputs, as shown in {numref}`transposed_conv`, right.

```{figure} ../figures/cnn_upsampling.svg
:scale: 50%
:name: upsampling

Visualization of different upsampling techniques.
```

```{figure} ../figures/cnn_transposed_convolution.svg
:scale: 50%
:name: transposed_conv

Visualization of transposed convolution.
```

## Conclusions

Convolutional Neural Networks (CNNs) are powerful models designed to process structured data using specialized layers such as convolutions, pooling, and upsampling. Convolutions play a key role in feature extraction, detecting patterns in 1D temporal data or 2D spatial data like images. Concepts like kernel size, stride, dilation, and padding define the behavior of these filters, while pooling layers reduce dimensions to streamline computation without losing critical information. For tasks requiring output generation, techniques like upsampling and transposed convolutions expand spatial dimensions effectively. Together, these components enable CNNs to model complex data hierarchically, with deeper layers accessing larger receptive fields and capturing increasingly abstract features.