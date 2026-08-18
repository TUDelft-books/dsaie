# Inductive Biases of Convolutional Neural Networks

## Introduction

Convolutional Neural Networks (CNNs) are a type of deep learning model that are widely used for image classification and object detection tasks.
They have two main inductive biases: spatial and hierarchical. 

The spatial inductive bias refers to the ability of CNNs to learn spatial patterns in images, such as edges, corners, and textures. 
The hierarchical inductive bias refers to the ability of CNNs to learn complex representations by combining features from multiple layers. 

In this notebook, we will explore these inductive biases and demonstrate how they enable CNNs to learn spatial *invariance* and *equivariance*, as well as *hierarchical* representations.

### Invariance
We say that a model is *invariant* with respect to a transformation if that transformation, applied to the input, does not alter the output.
In equation form, this looks like:

$
f(T(x)) = f(x),
$

where $f$ is an invariant model and $T$ is a transformation.

This property allows a neural network to recognize patterns or features in the data regardless of their position or orientation in the input.
For example, a model that is trained to recognize a cat should be able to recognize a cat regardless of whether it is in the center or corner of the image, or whether it is facing left or right.
CNNs have invariance with respect to spatial translations.

```{figure} ../figures/spatial_invariance.png
:scale: 45%
:name: spatial_invariance

Example where translational invariance is useful.
```


### Equivariance
We say that a model is *equivariant* with respect to a transformation, such as translation, rotation, or scaling, if that transformation, applied to the input, alters the output of the same transformation.
In equation form, this looks like:

$
f(T(x)) = T(f(x)),
$

where $f$ is an equivariant model and $T$ is a transformation.

This property implies that when the input data changes, the learned features also change in a predictable way.
CNNs have equivariance with respect to spatial translations, which comes very handy, for example, for object segmentation.

```{figure} ../figures/spatial_equivariance.png
:scale: 45%
:name: spatial_equivariance

Example where translational equivariance is useful.
```

### Hierarchy
Hierarchical inductive biases learn complex representations by breaking down the problem into smaller and more manageable sub-problems.
This is achieved by using a hierarchical architecture, where each layer in the network learns to extract different levels of complexity from the input data.

Simple features (edges, corners) are detected in early layers.
Complex features (textures, object parts) are detected in deeper layers.

```{figure} ../figures/hierarchical_bias.png
:scale: 45%
:name: hierarchical_bias

Example of simple and complex features extracted by a CNN.
```