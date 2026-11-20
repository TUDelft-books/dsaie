# Convolutional Neural Networks

Convolutional neural networks (CNNs) learn **local filters** and share their weights across positions. A filter trained to detect a boundary in one part of a satellite image can therefore detect the same pattern elsewhere. In one dimension, the same principle lets a filter detect an isolated peak wherever it occurs in a signal.

**Common design choices**

- **Stacked convolutional layers.** Each filter computes a simple local weighted sum. A CNN learns many filters and combines their responses across layers with nonlinear activations. In an aerial image, early layers might detect short edges, while later layers combine them into corners and building outlines.

- **Downsampling.** Many CNNs form a pyramid by reducing the spatial resolution in stages, using pooling or strided convolutions. Processing a smaller grid can save computation and discard spatial detail, such as a narrow watercourse. The effect on generalization depends on the task.

- **Output layers.** An image classifier often uses a fully connected layer to turn extracted features into class scores. A segmentation model produces a spatial grid of predictions so that each output can be associated with a location.

CNNs with bi-dimensional layers have a wide range of applications for gridded data, such as images. Common tasks include image classification, object detection, and image segmentation. On the other hand, their one-dimensional version can efficiently process sequential data, allowing for applications in time series analysis, signal processing, and natural language processing. CNNs have demonstrated exceptional performance in various tasks, including applications in our fields.

 In this series of lectures, we will explore the basics behind CNNs functioning and how to create basic CNNs by assembling these components together. We will apply them to a real-world case study from *earth observation*. We will then explore how we can improve the performances of these models by resorting to data augmentation and the very important concept of *transfer learning*, key to several developments in the field. Lastly, we will explore residual connections, and why they allow to improve CNN performance. We will also have a glimpse at typical computer vision tasks beyond image classification. We will also see how CNNs go beyond computer vision, with a practical application on *predicting flood arrival times* with *U-Net*, one of the most successful CNN architectures.

## Lectures

- {doc}`convolution_and_pooling`
- {doc}`cnn`
- {doc}`inv_eq_augmentation`
- {doc}`resnet`
- {doc}`cnn_beyond_ic`

## Exercises

- {doc}`exercises-clean/convolution_and_pooling`
- {doc}`exercises-clean/classification_metrics`
- {doc}`exercises-clean/landuse_cnn_part_1`
- {doc}`exercises-clean/data_augmentation`
- {doc}`exercises-clean/landuse_cnn_part_2`
- {doc}`exercises-clean/landuse_cnn_part_3`
- {doc}`exercises-clean/FAT_application`

## Quiz
- {doc}`review`
