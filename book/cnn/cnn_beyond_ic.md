# CNNs Beyond Image Classification

## Summary

The lecture will begin with a brief overview of **object detection** and **semantic segmentation**. We will do so by describing specific architectures designed for these tasks. **YOLO** (You Only Look Once) will be introduced as an efficient and effective model for real-time object detection. In contrast, **UNet** will be presented as a powerful architecture for semantic segmentation, providing good performance when more detailed spatial understanding is needed. The lecture will further explore the versatility of UNet by introducing its application in image-to-image "transformation," where one image is transformed into another. This segment will emphasize how CNN architectures can be adept at handling tasks involving the generation of a regular grid from another image, showcasing the broader applicability of these models and leading to the following *Workshop*.

## Object Detection

Object detection is the field that deals with **detecting and identyfing objects** in images using **bounding boxes**. Therefore, we can see object detection as a dual-task:
1. **Regression**: Detect the object by predicting the coordinates (x, y) of the top-left corner of the object, together with the height and width of the object.
2. **Classification**: Identify the detected object.

```{figure} ../figures/object_detection.svg
:scale: 60%
:name: object_detection

Output of an object detection model. Each identified object is surrounded by a bounding box together with its confidence score and its classification. https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md.
```

## YOLO (You Only Look Once)

Unlike traditional methods that separate the detection and classification tasks, YOLO treats object detection as a **single regression problem**, analyzing the entire image in one pass to predict bounding boxes and class probabilities simultaneously. This leads to significantly **faster processing speeds**, making it suitable for applications that require real-time detection, such as autonomous driving and surveillance.

In terms of architecture, YOLO is composed of **24 convolutional layers**, with varying number of channels, as illustrated in {numref}`YOLO_architecture`. Yet, we see a rather interesting aspect: the **output is a tensor**, rather than a scalar. We will see next why is that the case.

```{figure} ../figures/copyrighted/YOLO_architecture.svg
:scale: 50%
:name: YOLO_architecture

Architecture of YOLO: 24 convolutional layers followed by 2 fully connected layers. Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016:
```

We will list below the main flow of YOLO, illustrated in {numref}`YOLO_example` (based on the observations from Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016).
1. The input **image is divided into an $\mathbf{S \times S}$ grid**. If the center of an object falls within a grid cell, that cell is assigned the responsibility of detecting the object.

2. Each grid cell predicts $\mathcal{B}$ **bounding boxes** along with **confidence scores** for these boxes. These confidence scores indicate the model's certainty that the box contains an object and its accuracy regarding the box's predictions. **Each bounding box consists of five predictions**: $ x $, $ y $, $ w $, $ h $, and *confidence*. The $ (x, y) $ coordinates signify the center of the box in relation to the grid cell's dimensions. The width $ w $ and height $ h $ are predicted relative to the entire image. Lastly, the confidence prediction represents the Intersection Over Union (IOU) between the predicted box and any corresponding ground truth box, combined with class probabilities for each bounding box.

3. The system **predicts the most likely class at each cell** in the grid.

4. After getting all the data, the system filters out low-confidence predictions with a threshold to reduct FP (False-Positives). Moreover, overlapping bounding boxes are pruned, keeping the highest confidence predictions.

5. This results in an output tensor of dimension:
```{math}
:label: YOLO_output_tensor
S \times S \times (\mathcal{B} \times 5 + C),
```
where $S$ is the length of the grid, $\mathcal{B}$ represents the number of bounding boxes for each cell, $5$ is the number of outputs per bounding box (*(x, y) coordinates, width, height and confidence*) and $C$ represents the number of classes.

```{admonition} Output tensor in our example
:class: tip

Recall {numref}`YOLO_architecture`. We see that the output tensor is of shape $7 \times 7 \times (2 \times 5 + 20) = 7 \times 7 \times 30$, since:
- $S = 7$
- $\mathcal{B} = 2$
- $C$ = 20.
```

```{figure} ../figures/copyrighted/YOLO_example.svg
:scale: 60%
:name: YOLO_example

The flow of YOLO. Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
```

## Semantic Segmentation

Semantic segmentation is a task that involves **classifying** each **pixel** in an image into **predefined categories** or **classes**. Unlike traditional image classification, which assigns a single label to an entire image, semantic segmentation provides a detailed understanding of the scene by labeling every pixel based on the object it belongs to. The output is typically a segmentation map where each pixel represents the class label corresponding to that part of the image.

### Semantic Segmentation with Encoder-Decoder

One of the early networks for semantic segmentation follows a similar structure to an encoder-decoder in the way that the first part of the network, the *encoder*, downsamples the data, while the second part upsamples the data, called the *decoder*, as illustrated in {numref}`semantic_segmentation_archi`.

```{figure} ../figures/copyrighted/semantic_segmentation_archi.svg
:scale: 60%
:name: semantic_segmentation_archi

A proposed architecture for a semantic segmentation network. Adapted from Prince, S.J., 2023. Understanding Deep Learning. MIT PRESS.
```

The first part of the network, is responsible for encoding the input image into a feature-rich, compact representation (known as the **latent space**). It **reduces the spatial dimensions** of the input image while **increasing the depth** through convolutions and pooling. Moreover, it **extracts and compresses the features**, capturing the context of the image.

The decoder, also called the "expensive path", reconstructs the output from the latent space. Yet, it makes use of **upsampling** (discussed next) to match the original image size.

## Upsampling

It's a technique used to **increase the spatial dimensions of feature maps**, enabling the reconstruction of output images or grids from lower-resolution inputs. There exist several methods for upsampling, including:
- Simple duplication of pixel values, which doubles the size by repeating each input multiple times.
- **Max unpooling**: reverses the max pooling operation by placing the maximum values back into their original positions in the output, while filling other positions with zeros to retain spatial information.
- **Bilinear Interpolation**: estimates new pixel values by weighing contributions from neighboring pixels based on their distances, resulting in smoother and more refined upscaled images.

```{figure} ../figures/copyrighted/upsampling.svg
:scale: 40%
:name: upsampling

Upsampling techniques: a) duplicate each input four times, b) max unpooling and c) bilinear interpolation. Adapted from Prince, S.J., 2023. Understanding Deep Learning. MIT PRESS.
```

## U-NET

The U-Net ({numref}`U-NET`) is an **encoder-decoder architecture** where the earlier representations are concatenated to the later ones. We will list below some of its important characteristics:

- There is a feature-rich latent space (the upsampling layer laying on the line separating the two parts of the model).

- **Skip connections**: Transfer spatial information from *encoder* to *decoder*, preserving fine details for segmentation or image reconstruction.

- **Dimension mismatch**: Original paper used valid convolutions (i.e. the spatial size decreases by $K + 1$ pixels each time a $K \times K$ convolutional layer is applied). **It can be avoided** by:
    - Using the *same* padding in convolutions.
    - Matching *pooling* and *upsampling* layers to reverse the dimension reduction.

- It is a **Fully Convolutional Neural Network (FCN)**, so it can be ran on an image of any size.

```{figure} ../figures/copyrighted/unet.svg
:scale: 20%
:name: U-NET

Architecture of U-Net. Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015.
```

## Takeaways

- **Object Detection** identifies objects within images using bounding boxes and corresponding classifications.

- **Semantic Segmentation** classifies every pixel in an image, enabling a detailed understanding of the scene.

- **YOLO** (You Only Look Once) is a fast, real-time object detection system that utilizes a single CNN for simultaneous localization and classification.

- **Encoder-decoder networks** are frequently employed for semantic segmentation. They classify each pixel while combining context capture with precise localization for comprehensive image analysis.

- **U-Net** and similar encoder-decoder architectures can be applied beyond semantic segmentation, specifically whenever reconstruction of a grid from another is required.

## Resources
[Slides](https://surfdrive.surf.nl/files/index.php/s/KQYkdD6UsDkUibt)

Chapter 10.5.2, 10.5.3, 11.5.3 but not the part on *Hourglass networks* {bdg-warning}`prince-udl`