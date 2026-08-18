# Invariance, Equivariance and Data Augmentation

## Summary

This lecture will try to unpack the concepts of **translational invariance** and **translational equivariance** — key properties that enable CNNs to understand images consistently, regardless of object placement. We'll explore why these properties are crucial for CNNs to function effectively. Despite their strengths, we'll see how basic CNN models lack inherent invariance and equivariance to transformations like *rotation* and *scaling*. To partially address this issue — without designing sophisticated CNNs that inherently account for such transformations — we can resort using to *data augmentation*.

Diving into **image augmentation**, we'll review the main *geometric* and *photometric transformations* applied in computer vision. We'll touch on how these techniques not only prevent overfitting and enhance model generalization but also grant improved invariance and equivariance properties to our CNNs. We will also see the downsides and challenges of data augmentation, acknowledging that while these techniques are powerful, they come with trade-offs such as potential data quality degradation, introduction of biases, and increased computational demands.

## Translational Invariance

CNNs are usually known as the **translational invariant** neural networks, due to their shared-weight architecture of the convolutional layers. Yet, what does translational invariance mean?

It's a property which ensures that the ability of a model to **recognize patterns or features** is **not affected by their location in the input space**. This property is crucial for generalizing learned knownledge across different spatial contexts.

````{admonition} Example
:class: tip

In image classification, if an object is in the corner or center of an image, the model will identify it correctly. {numref}`translational_invariance` illustrates this concept. Moreover, we can define this relation mathematically as follows:

```{math}
:label: translational_invariance_formula
f(t(x)) = f(x),
```

where $x$ is the input, $f$ is the classifier (as a function) and $t$ is the applied translation.

```{figure} ../figures/translational_invariance.svg
:scale: 40%
:name: translational_invariance

In the image on the left, the model correctly classifies the object as a "Plastic bottle". If we apply a translation to the object, the model classifies it again to the same class (right). Adapted from Bernhard Kainz – Deep Learning, https://www.youtube.com/watch?v=a4Quhf9NhMY.
```
````

## Translational Equivariance

Using the notions introduced for translational invariance, equivariance means **if the input is translated, the output shifts in the same way**. It is particularly important in several tasks, including image segmentation, where the model must maintain the spatial relationship between the input and output. This allows the model to preserve the geometric structure of the input data, ensuring accurate and consistent mapping of the features, no matter their position.

Let's consider again the example with the plastic bottle, illustrated in {numref}`translational_equivariance`. When the bottle is shifted to a different position in the image, the segmentation function should also adjust its output correspondingly, ensuring that the segment representing the bottle moves to the same location as the bottle itself. In other words, we can write it as follows:

```{math}
:label: translational_equivariance_formula 

f(t(x)) = t(f(x)),
```
where $x$ is the input, $f$ is the classifier (as a function) and $t$ is the applied translation.

```{figure} ../figures/translational_equivariance.svg
:scale: 40%
:name: translational_equivariance

The output of the function $f$ changes in the same way of the transformed input $x$. Adapted from Bernhard Kainz – Deep Learning, https://www.youtube.com/watch?v=a4Quhf9NhMY.
```

## Translational Invariance and Equivariance in different fields

We've seen so far the theory behind these two concepts, but it's also imperative to understand how they are used. The table below illustrates the importance of both translational invariance and equivariance in various fields.

```{list-table}
:header-rows: 1

* - Field
  - Task Description
  - Invariance
  - Equivariance
* - **Image Classification**
  - Identify the categories represented in an image.
  - Crucial for recognizing objects regardless of their location in the image.
  - Less critical, as exact positioning within the image is not the primary concern.
* - **Object Detection**
  - Locate and classify individual objects within an image.
  - Important for consistently identifying objects anywhere in the image.
  - Important to accurately locate and bound objects within the image.
* - **Semantic Segmentation**
  - Classify each pixel in an image into a predefined category.
  - Less crucial, as the focus is on pixel-level classification.
  - Crucial to maintain spatial accuracy of pixel classifications relative to object positions.
```

## Are CNNs rotation and scale invariant?

Until now, the discussion only included the **translation** invariant property of CNNs. What happens if we rotate or scale an object? Will the CNN still be able to accurately recognize or classify it?

However, the answer is ***no***. **CNNs are not scale or rotation invariant** (or we can say that **CNNs are scale and rotation variant**) because they rely on fixed filters that are trained on specific sizes and orientations, leading to difficulties in recognizing scaled or rotated objects due to the changed spatial relationships and potential loss of important features.

Fortunately, one way to address it is to make use of **data augmentation**. Of course, we can also consider other CNNs that can inherently account for such transformations, but that is outside the scope of this course.

## Data (Image) Augmentation

Data Augmentation is about altering the original training data to increase it and diversify it. This approach improves model generalization on new data and reduces the risk of overfitting. 

For images, we usually employ **geometric** and **photometric** (i.e. changes made to the brightness, contrast, etc of the image) transformations. {numref}`data_augmentation` illustrates different transformations applied to an input image.

```{figure} ../figures/data_augmentation.svg
:scale: 60%
:name: data_augmentation

The output of the function $f$ changes in the same way of the transformed input $x$. Adapted from https://pranjal-ostwal.medium.com/data-augmentation-for-computer-vision-b88b818b6010.
```

### Benefits of Image Augmentation
It **enhances model robustness to translation, scaling, rotation or other transformations**. By using this technique, we can achieve better invariance as the model becomes consistent across various inputs. Moreover, we get better equivariance, such that the model produces appropiate responses to input transformations.

### Challenges of Image Augmentation
1. One of the main challenges when using image augmentation is the **risk of degrading the quality of the data**. By employing improper augmentation techniques or an excessive augmentation, we might end up changing the target label. For example, {numref}`data_augmentation_challenge` illustrates the case where cropping the input image changes the target class (e.g. input image is classified to "plastic bag", but the cropped image cannot be classified to "plastic bag" anymore).
2. It might **introduce bias** in the data, reducing model's understanding. 
3. Adds significant **overhead to the computational resources** and time required for processing. 
4. It is **hard to choose an appropiate augmentation technique** for specific tasks.

```{figure} ../figures/data_augmentation_challenge.svg
:scale: 30%
:name: data_augmentation_challenge

Cropping compromises the correctness of the original label.
```

## Takeaways
- **Convolutional layers** facilitate feature detection regardless of location, providing **translational (or shift) equivariance**.

- **Pooling layers** reduce data size, offering robustness to changes in feature locations and approximating **translational (or shift) invariance**.

- **Equivariance** ensures that output changes predictably with translations, which is vital for **consistent pattern recognition** across different images.

- **Invariance** allows for reliable recognition despite shifts in object placement, making it crucial for **stable image classification**.

- Together, these properties enhance the **reliability** of CNNs across various visual scenarios.

- CNNs do not naturally maintain **invariance and equivariance** to other transformations, such as **rotations and scaling**.

- Some advanced models are designed to manage these transformations but tend to be **more complex**.

- **Image augmentation** can artificially introduce variations in rotation and scaling to help train traditional CNNs effectively.

- By applying augmentation techniques, CNNs can learn to recognize features across these transformations, overcoming their initial limitations.

- Augmentations typically fall into two categories: **geometric** (e.g., rotations, scaling, translations, cropping) and **photometric** (e.g. adjusting brightness, contrast).

- These augmentations **diversify** the dataset and expose CNNs to a variety of conditions, reducing **overfitting** and improving **generalization**.

- However, excessive augmentation may lead to **misalignment** with original labels, diminishing data relevance.

- Augmentation can introduce **biases**, increase **computational demands**, and unintentionally alter the **integrity** of the dataset.

## Resources

[Slides](https://surfdrive.surf.nl/files/index.php/s/Y78bhANWRq6IYjS)

Chapter 10.1 {bdg-warning}`prince-udl`