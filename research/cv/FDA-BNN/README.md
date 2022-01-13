# Contents

- [FDA-BNN Description](#fdabnn-description)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
        - [Training Process](#training-process)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [FDA-BNN Description](#contents)

Binary neural networks (BNNs) represent original full-precision weights and activations into 1-bit with sign function. Since the gradient of the conventional sign function is almost zero everywhere which cannot be used for back-propagation, several attempts have been proposed to alleviate the optimization difficulty by using approximate gradient. However, those approximations corrupt the main direction of factual gradient. To this end, we propose to estimate the gradient of sign function in the Fourier frequency domain using the combination of sine functions for training BNNs, namely frequency domain approximation (FDA). The proposed approach does not affect the low-frequency information of the original sign function which occupies most of the overall energy, and high-frequency coefficients will be ignored to avoid the huge computational overhead. In addition, we embed a noise adaptation module into the training phase to compensate the approximation error. The experiments on several benchmark datasets and neural architectures illustrate that the binary network learned using our method achieves the state-of-the-art accuracy.

[Paper](https://arxiv.org/pdf/2103.00841.pdf): Yixing Xu,  Kai Han, Chang Xu, Yehui Tang, Chunjing Xu, Yunhe Wang. Learning Frequency Domain Approximation for Binary Neural Networks. Accepted by NeurIPS 2021.

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
├── FDA-BNN
  ├── README.md       # readme
  ├── src
  │   ├──loss.py      # label smoothing cross-entropy loss
  │   ├──dataset.py   # creating dataset
  │   ├──resnet.py    # ResNet architecture
  │   ├──quan.py      # quantization  
  ├── eval.py         # evaluation script
```

## [Eval process](#contents)

### Usage

After installing MindSpore via the official website, you can start evaluation as follows:

### Launch

```bash
# infer example
  GPU: python eval.py --dataset_path path/to/cifar10 --platform GPU --checkpoint_path [CHECKPOINT_PATH]
```

> checkpoint can be found at https://download.mindspore.cn/model_zoo/research/cv/FDA_BNN/fdabnn.ckpt

### Result

```bash
result: {'Validation-Loss': 1.4819902773851004, 'Top1-Acc': 0.8660857371794872, 'Top5-Acc': 0.9950921474358975}

```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### FDA-BNN on CIFAR-10

| Parameters                 |                                        |
| -------------------------- | -------------------------------------- |
| Model Version              | FDA-BNN         |
| uploaded Date              | 11/22/2021 (month/day/year)  ；                     |
|  | GPU |
| MindSpore Version          | 1.3                                                     |
| Dataset                    | CIFAR-10                                             |
| Input size   | 32x32                                       |
| Validation Loss | 1.482 |
| Training Time (min) | 350 |
| Training Time per step (s) | 0.27 |
| Accuracy (Top1) | 86.61 |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
