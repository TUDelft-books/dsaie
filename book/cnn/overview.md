# Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are a type of deep learning neural network commonly used for image and video analysis. They are designed to automatically and adaptively learn spatial hierarchies of features through convolution layers, pooling layers, and fully connected layers. CNNs are particularly important in the field of computer vision, which aims to enable machines to see and understand the world like humans do.

Key aspects of CNNs include:

- **Hierarchical Feature Learning**: CNNs process input data, such as images, by applying a series of filters or kernels to extract features at different levels of complexity.

- **Convolution Layers**: These layers are responsible for learning and detecting features in the input data. They are typically organized in a pyramid structure, with each layer focusing on a specific level of detail.

- **Pooling Layers**: These layers are used to reduce the spatial dimensions of the input data, which helps to increase efficiencty, reduce overfitting and improve the network's performance.

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
- {doc}`exercises-clean/landuse_cnn_part_1`
- {doc}`exercises-clean/data_augmentation`
- {doc}`exercises-clean/landuse_cnn_part_2`
- {doc}`exercises-clean/landuse_cnn_part_3`

## Quiz
- {doc}`review`
