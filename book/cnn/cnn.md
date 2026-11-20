# Convolutional Neural Networks (CNNs)

To classify an image as a cat or a dog, a model needs features that help distinguish the two. In the [preceding lecture](convolution_and_pooling.md), we saw how a convolutional layer learns local filters and produces feature maps. In this lecture, we'll stack these layers with nonlinear activations to build **Convolutional Neural Networks (CNNs)** for **image classification** (IC).

We'll examine two main components: a feature extractor that learns representations from images, followed by a classification head that maps those features to class scores. Many CNNs use pooling or strided convolutions to reduce the spatial resolution along the way. We'll follow how features change with depth, then discuss **ImageNet** and landmark architectures such as **AlexNet**, **VGG** and **ResNet**.

**Resources**

- [Slides](https://surfdrive.surf.nl/files/index.php/s/YM965vSFvnM2vsX)
- Simon J. D. Prince, *Understanding Deep Learning* (MIT Press, 2023), section 10.5. [Book website](https://udlbook.github.io/udlbook/)

## Computer Vision

Computer Vision (CV) is a field of AI focused on enabling machines to interpret and understand visual data from the world. CV encompasses several key tasks:

* **Image Classification**: Assigns labels to the image as a whole (e.g., "this image contains a cat").
* **Object Detection**: Locates and identifies individual objects using bounding boxes (e.g., "there's a car at this position, with this width and height").
* **Semantic Segmentation**: Labels each pixel in the image according to which class it belongs to (e.g., "these pixels are road, these are sidewalk").
* **Instance Segmentation**: Identifies and segments individual instances of objects (e.g., "these pixels belong to person #1, these to person #2").

{numref}`cnn_vision_tasks` shows what each task predicts for the same street scene.

```{figure} ../figures/cnn_vision_tasks.svg
:name: cnn_vision_tasks
:alt: Four views of two cars show a single label for the image, one bounding box around each car, a shared class label for all car pixels, and a separate mask for each car.

Classification reports that the image contains a car. Detection draws a box around each car. Semantic segmentation assigns every pixel to car, road or background. The instance view overlays a separate mask for each car on the street scene.
```

```{admonition} CV beyond camera images
:class: tip
Computer vision methods can also process data beyond camera images. Radar reflectivity maps, satellite-derived flood extent maps and gridded model outputs can all be treated as images in which each pixel represents a measurement or prediction. CNNs can learn spatial patterns in these regular grids, with the inputs and outputs chosen to suit the prediction task.
```

## Image Classification

To introduce CNNs, we focus on **image classification**, where the goal is to predict labels for a whole image. The output layer depends on which combinations of labels the task allows:

1. **Binary Image Classification**
   * Assigns each image to one of two categories (e.g., land vs water in aerial images).
   * A single sigmoid output estimates the probability of one designated class, such as water. We compare it with a threshold, often 0.5, to choose between the two classes.

2. **Multiclass Image Classification**
   * Assigns each image to exactly one of several mutually exclusive categories (e.g., forest, urban or agricultural land).
   * A softmax output converts one score per class into probabilities that sum to one. We select the class with the largest probability.

3. **Multilabel Image Classification**
   * Identifies categories that can be present simultaneously (e.g., buildings, roads and water bodies in an aerial image).
   * One sigmoid output per label estimates its probability of being present. Applying a threshold to each output allows zero, one or several labels to be selected. These probabilities do not need to sum to one, and the labels can share features learned by the network.

These activations convert the network's raw scores, called **logits**, into probabilities. During training, some loss functions accept logits and perform the required conversion internally. For example, the land-use exercise uses PyTorch's `CrossEntropyLoss` directly on the logits for multiclass classification.

```{admonition} Multiclass vs. Multilabel
:class: tip

The same image can receive different labels depending on the task definition. Multiclass classification requires a **single**, exclusive category, while multilabel classification allows **multiple** categories. {numref}`imag_classif` illustrates the three output rules for the same inspection image.
```

```{figure} ../figures/imag_classif.svg
:scale: 15%
:name: imag_classif
:alt: One inspection image receives a binary condition label, one multiclass defect label, or several multilabel defect labels. Selected labels are bold and red.

Outputs of different image classification methods. Selected labels are shown in **bold** and red. Image from the [Sewer-ML dataset](https://vap.aau.dk/sewer-ml/).
```

### Learning features

How do we obtain the features that a classifier needs? Traditional computer vision systems often use **hand-crafted features**, such as measurements of edges, textures or shapes. A person designs the calculations, software applies them to each image, and a classifier is trained on the resulting measurements. The feature calculations remain fixed while the classifier is fitted, so their usefulness depends on the designer's choices.

A CNN enables **automatic feature extraction** by learning the filter weights along with the classification head. During end-to-end training, the classification loss measures the error against the image labels. Backpropagation computes gradients for both components, and the optimizer updates their parameters. We still choose the architecture and output labels, but do not have to specify the feature calculations in advance. {numref}`old_vs_new_cv` compares these two approaches.

```{figure} ../figures/old_vs_new_cv.svg
:name: old_vs_new_cv
:width: 100%
:alt: A shared input image feeds two routes to class scores. Hand-crafted measurements stay fixed while their classifier is trained. In the CNN route, the feature extractor and classification head are trained together.

Hand-crafted and learned features for image classification. In the left route, a designer chooses fixed measurements and training fits the classifier. In the CNN route, training adjusts both the filters that produce feature maps and the classification head using the same classification loss.
```

The local filters and shared weights introduced in the preceding lecture give CNNs a **spatial inductive bias**: nearby pixels are processed together, and the same pattern can be detected at different positions. Stacking layers with nonlinear activations gives a **hierarchical inductive bias**. Later filters combine responses across channels and, as their receptive fields grow, can use patterns spread over larger parts of the image.

To see this process, pass an image of a cat through a trained CNN and inspect its intermediate activations ({numref}`deep_maps`). Each tile shows one feature map, or output channel, for that same input. The rows show selected layers at increasing depths; only a subset of the channels is displayed.

In the early layers (block1), many maps still resemble the image because they respond to local edges and textures. Deeper layers combine these responses into more complex patterns. Some channels may respond to object parts, but an individual channel need not have a simple meaning such as an eye or an ear. Deep maps are often harder to interpret by inspection.

Pooling reduces the spatial resolution between blocks in the illustrated network. The maps are enlarged to comparable display sizes, so larger coloured cells in deeper rows correspond to a coarser grid. After feature extraction, the classification head uses the final maps to make an image-level prediction.

```{figure} ../figures/feature_maps_deep.png
:name: deep_maps
:scale: 40%
:alt: Selected feature maps for a cat image at increasing depths. Early maps preserve fine edges and textures; deeper maps show coarser activation patterns, enlarged for comparison.

Feature maps at different depths of a trained CNN. Each row shows selected channels for the same cat image. Source: [Arden Dertat, Applied Deep Learning — Convolutional Neural Networks](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2/).
```

### From feature maps to class scores

The CNN in {numref}`cnn_flatten_classifier` **flattens** its final feature maps into a vector and passes it to a fully connected layer. This layer combines the activations into class scores, which are converted to probabilities using the output rule for the task.

```{figure} ../figures/cnn_flatten_classifier.svg
:name: cnn_flatten_classifier
:alt: A grayscale image passes through convolution, ReLU and max pooling to give eight four-by-four feature maps. Flattening places their 128 activations in one vector. A fully connected layer produces three class scores.

Pooling reduces each of the eight maps to four rows and four columns. Flattening arranges their 128 values in a vector for the dense (fully connected) layer.
```

```{admonition} Meaning of the "flattened" features
:class: tip

Suppose the final tensor has shape $4\times4\times8$ in the channels-last convention: eight feature maps, each with four rows and four columns. Flattening rearranges all $4\times4\times8=128$ activations into a vector of length 128. It does not average or discard any values. Each entry still corresponds to one channel at one spatial location; the preceding layers have learned what that response represents.

For example, features useful for distinguishing a cat from a dog may respond to fur textures or combinations of shapes. In the land-use exercise, they may respond to patterns of buildings, vegetation or fields. The fully connected layers learn how to combine these responses for the chosen labels.
```

Other CNNs use **global average pooling** before a final classification layer. This averages each feature map across its spatial positions. In the same example, it produces eight values rather than 128: one mean response per channel. This reduces the number of inputs to the classifier but removes information about where within a map the responses occurred.

### ImageNet

To compare image classifiers, researchers need a shared set of images and labels. **ImageNet** is a large labelled image database; the widely used 1,000-class benchmark is a subset used in the **ImageNet Large-Scale Visual Recognition Challenge (ILSVRC)**. Its 2012 classification task has about 1.28 million training images, 50,000 validation images and 100,000 test images, covering object categories such as animals and everyday items. Images vary in scale, pose, lighting, background and object position {cite:p}`cnn-russakovsky2015imagenet`.

ILSVRC began in 2010 and provided a common benchmark for comparing models. Several influential architectures emerged from this challenge, showing how changes in feature extraction and network design could improve classification performance.

#### AlexNet
**AlexNet** won the 2012 ImageNet classification challenge by a large margin and helped establish deep CNNs as a practical approach to computer vision. It combined five convolutional layers with three fully connected layers, using ReLU activations, dropout regularization and GPU training. Its success helped popularize this combination of techniques {cite:p}`cnn-krizhevsky2012imagenet`.

#### VGG
The **VGGNet** family, developed for ILSVRC 2014 and published in 2015, used repeated $3\times3$ convolutions with nonlinear activations between them. Stacking these small filters lets later outputs depend on a larger region of the input while adding successive nonlinear transformations. VGG showed that increasing depth within this design could improve classification, though adding layers does not guarantee better performance for every network {cite:p}`cnn-simonyan2015very`.

#### ResNet
**Residual Networks (ResNets)** use shortcut connections that add a block's input to the output of its convolutional layers. These layers can then learn a change to the input, called a residual, instead of having to learn the entire mapping. This design made much deeper networks easier to optimize. ResNet-based models won ILSVRC 2015, and the paper was published in 2016 {cite:p}`cnn-he2016deep`. We'll study these connections in the [residual networks lecture](resnet.md).

(current-state-of-the-art)=
#### Vision Transformers

CNNs are not the only way to learn image features. A **Vision Transformer (ViT)** divides an image into patches, maps each patch to a vector and uses self-attention to combine information across patches. This offers a different approach to modelling relationships across an image from the local filters of a CNN {cite:p}`cnn-dosovitskiy2021image`.

```{admonition} Exam questions    
:class: danger    
For the exam, you should be able to explain how local filters with shared weights and successive layers let a CNN learn increasingly complex features. Relate this process to spatial and hierarchical inductive biases, explain the difference between hand-crafted and learned features, and describe how a classification head uses the final feature maps. You should understand what flattening does and how the output rule depends on the classification task. You won't need to memorize specific architectures.
+++    
{bdg-primary}`written-exam`    
```

## References

```{bibliography}
:filter: docname in docnames
:keyprefix: cnn-
```
