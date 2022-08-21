# Contents

- [AdaBin Description](#AdaBin-description)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)
- [Reference](#reference)

# [AdaBin Description](#contents)

This paper studies the Binary Neural Networks (BNNs) in which weights and activations are both binarized into 1-bit values, thus greatly reducing the memory usage and computational complexity. Since the modern deep neural networks are of sophisticated design with complex architecture for the accuracy reason, the diversity on distributions of weights and activations is very high. Therefore, the conventional sign function cannot be well used for effectively binarizing full-precision values in BNNs. To this end, we present a simple yet effective approach called AdaBin to adaptively obtain the optimal binary sets {b1, b2} (b1, b2 ∈ R) of weights and activations for each layer instead of a fixed set (i.e., {−1, +1}). In this way, the proposed method can better fit different distributions and increase the representation ability of binarized features. In practice, we use the center position and distance of 1-bit values to define a new binary quantization function. For the weights, we propose an equalization method to align the symmetrical center of binary distribution to real-valued distribution, and minimize the Kullback-Leibler divergence of them. Meanwhile, we introduce a gradient-based optimization method to get these two parameters for activations, which are jointly trained in an end-to-end manner. Experimental results on benchmark models and datasets demonstrate that the proposed AdaBin is able to achieve state-of-the-art performance. For instance, we obtain a 66.4% Top-1 accuracy on the ImageNet using ResNet-18 architecture, and a 69.4 mAP on PASCAL VOC using SSD300.

[Paper](https://arxiv.org/abs/2208.08084): Zhijun Tu, Xinghao chen, Pengju Ren, Yunhe Wang. AdaBin: Improving Binary Neural Networks with Adaptive Binary Sets. Accepted by ECCV 2022.

# [Dataset](#contents)

- Dataset used: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
    - Dataset size: 60000 colorful images in 10 classes
        - Train:  50000 images
        - Test: 10000 images
    - Data format: RGB images.
        - Note: Data will be processed in src/dataset.py

# [Features](#contents)

## [Mixed Precision(Ascend)](#contents)

The mixed precision training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend、GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/en/r0.5/index.html)
    - [MindSpore API](https://www.mindspore.cn/api/en/0.1.0-alpha/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```text
├── AdaBin
  ├── README.md       # readme
  ├── src
  │   ├──loss.py      # label smoothing cross-entropy loss
  │   ├──dataset.py   # creating dataset
  │   ├──resnet.py    # ResNet architecture
  │   ├──binarylib.py # binary quantizer  
  ├── eval.py         # evaluation script
```

## [Evaluation Process](#contents)

### Usage

After installing MindSpore via the official website, you can start evaluation as follows:

### Launch

```bash
# infer example
GPU: python eval.py --dataset_path path/to/cifar10 --platform GPU --checkpoint_path [CHECKPOINT_PATH]
```

checkpoint can be found at https://download.mindspore.cn/models/r1.8/adabin_ascend_v180_cifar10_research_cv_acc88.15.ckpt

### Result

```bash
result on cifar-10-verify-bin:
{'Validation-Loss': 1.5793, 'Top1-Acc': 0.9212, 'Top5-Acc': 0.9986}
result on complete cifar-10 test set:
{'Validation-Loss': 0.3264, 'Top1-Acc': 0.8815}
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### AdaBin on CIFAR-10

| Parameters                 |                                        |
| -------------------------- | -------------------------------------- |
| Model Version              | AdaBin         |
| uploaded Date              | 08/16/2022 (month/day/year)  ；                     |
|  Device                    | GPU |
| MindSpore Version          | 1.8.0                                                    |
| Dataset                    | CIFAR-10                                             |
| Input size   | 32x32                                       |
| Validation Loss | 0.326 |
| Training Time (min) | 350 |
| Training Time per step (s) | 0.18 |
| Accuracy (Top1) | 88.151 |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).

# [Reference](#reference)

[FDA-BNN](https://gitee.com/mindspore/models.git)

