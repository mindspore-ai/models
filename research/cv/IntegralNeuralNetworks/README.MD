# Integral Neural Networks - MindSpore

## Overview

This repository will contain MindSpore implementation of the paper "Integral Neural Networks". 


## Table of contents

- [Integral Neural Networks - MindSpore](#integral-neural-networks---mindspore)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Credit](#credit)
        - [Integral Neural Networks](#integral-neural-networks )


## Credit

### Integral Neural Networks 
Kirill Solodskikh, Azim Kurbanov, Ruslan Aydarkhanov, Irina Zhelavskaya, Stamatios Lefkimmiatis, Yury Parfenov, Dehua Song

**Abstract** <br>

We introduce a new family of deep neural networks, where instead of the conventional representation of network layers as $N$-dimensional weight tensors, we use a continuous layer representation along the filter and channel dimensions. We call such networks Integral Neural Networks (INNs). In particular, the weights of INNs are represented as continuous functions defined on $N$-dimensional hypercubes, and the discrete transformations of inputs to the layers are replaced by continuous integration operations, accordingly. During the inference stage, our continuous layers can be converted into the traditional tensor representation via numerical integral quadratures. Such kind of representation allows the discretization of a network to an arbitrary size with various discretization intervals for the integral kernels. This approach can be applied to prune the model directly on an edge device while suffering only a small performance loss at high rates of structural pruning without any fine-tuning. To evaluate the practical benefits of our proposed approach, we have conducted experiments using various neural network architectures on multiple tasks. Our reported results show that the proposed INNs achieve the same performance with their conventional discrete counterparts, while being able to preserve approximately the same performance (2\% accuracy loss for ResNet18 on Imagenet) at a high rate (up to $30\%$) of structural pruning without fine-tuning, compared to 65\% accuracy loss of the conventional pruning methods under the same conditions.
