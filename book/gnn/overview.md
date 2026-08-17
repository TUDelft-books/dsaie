# Graph Neural Networks

Graph Neural Networks are a deep learning approach that can be imagined as a generalization of convolutional neural networks to non-Euclidean domains, such as graphs, over which they can model data.
The advanced topic builds up from graphs and graph signal processing until the most recent architectures involving graph neural networks.
You will also explore graphs and graph neural networks by working on two dedicated notebooks.

**Project integration**: Graph neural networks have proven to be valuable tools for surrogate modelling, especially if the data lies on a meshed domain. Moreover, they show excellent generalizability to unseen domains.

## Lectures:
- Introduction to graphs
- From graph shifting to graph neural networks
- Advanced GNN architectures

Lecture slides: [Slides](https://surfdrive.surf.nl/files/index.php/s/fhF4OB1LRgnbe1q)

## Exercises:
- {doc}`exercises-clean/EX1-graphs`
- {doc}`exercises-clean/EX2-pytorch_geometric`

## Installing the additional libraries

In the following notebooks, you will use two libraries that are not present in your environments.
To install them, go in your dsaie conda environment by running in the Anaconda prompt:

    conda activate dsaie

Then, you can install the PyTorch Geometric library as:

    pip install torch_geometric==2.4.0

In case there are compatibility issues with your current Pytorch version, please check the corresponding installation documentation at https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html or ask to a TA for help.