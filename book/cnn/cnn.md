# Convolutional Neural Networks (CNNs)

In this lecture, we explore how **Convolutional Neural Networks (CNNs)** are constructed by stacking convolutional and pooling layers to automatically extract features from images. We'll use **image classification** (IC) as our computer vision task, as it provides an ideal introduction to understanding how CNNs work.

We'll examine how modern CNN architectures for IC typically consist of two main components: an automatic feature extractor that learns hierarchical representations, followed by a classifier tailored to the specific classification task. As we delve deeper into these networks, we'll observe how features transform from simple edge detectors in early layers to more complex, abstract patterns in deeper layers.

We will discuss landmark architectures that shaped the field, starting with **AlexNet**, which sparked the *deep learning revolution*, and **VGG**, which demonstrated the power of deeper architectures. We'll also discuss the crucial role of **ImageNet**, the extensive dataset that served as a catalyst for these breakthroughs in computer vision, providing a standardized benchmark for evaluating model performance.


**Resources**

[Slides](https://surfdrive.surf.nl/files/index.php/s/YM965vSFvnM2vsX)

Chapter 10.5 {bdg-warning}`prince-udl`

## Computer Vision

Computer Vision (CV) is a field of AI focused on enabling machines to interpret and understand visual data from the world. CV encompasses several key tasks:

* **Image Classification**: Determines what category an image belongs to (e.g., "this sewer image shows a crack").
* **Object Detection**: Locates and identifies multiple objects in an image using bounding boxes (e.g., "there's a car at coordinates x,y with width w and height h").
* **Semantic Segmentation**: Labels each pixel in the image according to which class it belongs to (e.g., "these pixels are road, these are sidewalk").
* **Instance Segmentation**: Identifies and segments individual instances of objects (e.g., "these pixels belong to person #1, these to person #2").


```{admonition} CV beyond camera images
:class: tip
It's important to note that computer vision applications in our fields extend beyond traditional camera-based imagery. For instance, radar reflectivity maps in rainfall nowcasting, satellite-derived flood extent maps, or even gridded model outputs can all be treated as "images" where each pixel represents a measurement or prediction. Indeed, we can apply Deep Learning-based CV techniques to any variable that can be represented on a regular grid, from soil moisture maps to stresses in concrete.
```

## Image Classification

To introduce CNNs, we focus on **image classification**, where the goal is to identify categories in images. There are multiple types of image classification, depending on the complexity of the task:

1. **Binary Image Classification**
   * Classifies images into two categories (e.g., defective vs non-defective sewer pipe, land vs water in aerial images).
   * **How it works?** The model learns to distinguish between the two classes by extracting relevant features, using a single sigmoid activation in the output layer to produce a probability score for class membership.

2. **Multiclass Image Classification**
   * Assigns each image to exactly one category from multiple possible classes (e.g., crack/root/deposit in sewer pipes, or forest/urban/agricultural in aerial images).
   * **How it works?** The model learns to map extracted features to a single class using a softmax activation in the output layer, which produces a probability distribution over all possible classes.

3. **Multilabel Image Classification**
   * Identifies multiple categories that can be present simultaneously (e.g., multiple defect types in a sewer pipe, or the presence of buildings/roads/water bodies in aerial images).
   * **How it works?** The model uses multiple sigmoid activations in the output layer (one for each possible class) to independently predict the presence/absence of each category based on learned features.


```{admonition} Multiclass vs. Multilabel
:class: tip

It might look like the multiclass and multilabel classifications perform the same thing. However, the main difference between them is that multiclass classification assigns each instance to a **single**, exclusive category, while multilabel classification allows instances to be assigned to **multiple** categories. Figure {numref}`imag_classif` below illustrates classifications performed by each method.
```{figure} ../../../images/imag_classif.svg
:scale: 15%
:name: imag_classif

Outputs of different image classification methods. Class(es) bolded in <span style="color:red; font-weight:bold"> red</span> represent the classification output. Image from [SewerML dataset](https://vap.aau.dk/sewer-ml/).
``````

Early computer vision algorithms relied on manually extracting features from images, such as edges, corners, textures, or shapes, followed by training a simple classifier (e.g., a fully-connected network with a softmax activation) on these hand-crafted features. The performance of these systems heavily depended on domain expertise to design relevant and effective features appropriate for a specific task.

In contrast, CNNs revolutionize this approach by enabling **automatic feature extraction**. Due to their spatial and hierarchical inductive biases, CNNs learn complex features directly from raw image data, starting from low-level patterns (e.g., edges) to high-level, task-specific details (e.g., cracks, root intrusions, or urban buildings). These features are then fed into a classification head, typically using fully-connected layers with task-specific activation functions (e.g., softmax or sigmoid).

This automatic end-to-end learning process forms the foundation for modern computer vision systems and facilitates the extension of CV methods to data beyond conventional camera imagery, such as radar maps for rainfall nowcasting, gridded spatial flood extent maps, or other variables organized on a regular grid. A comparison of traditional CV systems and CNN-based systems is shown in {numref}`old_vs_new_cv`.

```{figure} ../figures/old_vs_new_cv.svg
:name: old_vs_new_cv
:scale: 20%
Architecture of the traditional CV (top) compared to that of a CNN (bottom).
```

To better understand how CNNs work, let's visualize the feature maps produced at different depths of the network. Figure {numref}`deep_maps` shows the activation patterns of different convolutional layers when processing an image of a cat. These feature maps are obtained by passing the input image through a trained CNN and extracting the outputs of specific convolutional layers. The visualization includes activation maps for only a subset of the filters in each layer, as the network typically has many filters at each level.

In the early layers (block1), the feature maps still resemble the input image, as they detect low-level features like edges and basic textures. As we move deeper into the network (block2-3), the features become more abstract, capturing more complex patterns and shapes. In the deeper layers (block4-5), the feature maps are highly abstract and specialized for the classification task, though they are no longer visually interpretable to humans.

Note that while the feature maps get progressively smaller through the network (due to pooling operations), they are shown at the same scale in the visualization for easier comparison. These final feature maps are ultimately flattened into a vector and fed into fully connected layers for the final classification decision.

This hierarchical feature extraction is what makes CNNs so powerful - they automatically learn to detect increasingly complex and task-relevant features, from simple edges to complex object parts, without requiring manual feature engineering.

```{figure} ../figures/feature_maps_deep.png
:name: deep_maps
:scale: 40%
Feature maps in Deep CNNs. Image credits: https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2.
```

```{admonition} Meaning of the "flattened" features
:class: tip
The flattened features produced by the final convolutional layers of a CNN represent highly abstract and task-specific information summarized from the input image. In the case of a generic classification task (i.e., ImageNet dataset below), these features might encode high-level representations such as the presence of fur patterns, eye shapes, or ear structures that distinguish a cat from a dog. For land-use detection from aerial images (i.e., the US Merced Dataset of the exercise), flattened features could represent abstract spatial patterns, such as grid-like textures of urban areas, irregular patterns of forests, or uniform patches of farmland. These condensed and meaningful representations are then used in the fully connected layers to make the final classification decision. Such adaptability enables CNNs to perform well across vastly different image classification tasks.
``````

### ImageNet 

**ImageNet** is a highly influential image database that has played a pivotal role in advancing computer vision over the last decade. It contains approximately 1.2 million labeled training images, 100,000 test images, and spans 1,000 diverse classes, ranging from animals and objects to scenes and everyday items. The ImageNet dataset is particularly challenging because of its high variability in scale, pose, lighting conditions, backgrounds, and object positions, making it a benchmark for assessing the robustness of image classification models.

The introduction of the **ImageNet Large-Scale Visual Recognition Challenge (ILSVRC)** in 2010 accelerated progress in image classification by providing a standardized framework for researchers to compare models. Several landmark models emerged from this challenge, which have significantly shaped the field of computer vision by driving advancements in architectures, feature extraction, and model scalability.

Below, several key models derived from ImageNet that have had a transformative impact on image classification tasks and deep learning/computer vision as a whole.

#### AlexNet
**AlexNet** (2012) was one of the first models to demonstrate the power of deep learning on the ImageNet challenge, achieving a breakthrough performance. It introduced key innovations such as ReLU activations, dropout regularization to reduce overfitting, and the use of GPUs for training, enabling deeper and more complex networks. 
> *Paper*: Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012).


#### VGG
The **VGGNet** (2014) family of models showed that stacking small 3×3 convolutional filters in a deep architecture could effectively capture complex patterns in the data. By increasing model depth, VGG achieved state-of-the-art performance while maintaining a relatively simple architecture. Its demonstration of the power of depth further popularized the design of deeper networks for robust feature extraction.
> *Paper*: Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

#### ResNet
The **Residual Network (ResNet)** (2015) introduced the concept of residual connections, which allowed networks to go significantly deeper by alleviating the vanishing gradient problem. These skip connections enabled the training of models with hundreds of layers, pushing performance to new heights and embedding ResNet as a milestone in model scalability.
> *Paper*: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

#### Current State-of-the-Art
More recently, **transformer-based architectures**, such as Vision Transformers (ViTs), have redefined state-of-the-art performance on ImageNet benchmarks. These models utilize self-attention mechanisms to identify and exploit meaningful relationships across the entire image, often outperforming traditional CNN-based models, especially when scaled with large datasets and computational resources. ViTs and hybrid transformer-CNN approaches represent an exciting frontier for computer vision.

```{admonition} Exam questions    
:class: danger    
Understanding how CNNs process images hierarchically is fundamental to computer vision applications. For the exam, you should be able to explain how features evolve from low-level to high-level representations as we move deeper in the network, and how this relates to the spatial and hierarchical inductive biases built into CNNs. You should also understand what flattened features represent and their role in classification. You won't need to memorize specific architectures, but should grasp these key principles and their implications for image understanding tasks.
+++    
{bdg-primary}`written-exam`    
```