# Contents

- [Contents](#contents)
    - [AdderNGD Description](#AdderNGD-description)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Script description](#script-description)
        - [Script and sample code](#script-and-sample-code)
    - [Eval process](#eval-process)
        - [Usage](#usage)
        - [Launch](#launch)
        - [Result](#result)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [AdderNGD Description](#contents)

To achieve efficient inference with a hardware-friendly design, Adder Neural Networks (ANNs) are proposed to replace expensive multiplication operations in Convolutional Neural Networks (CNNs) with cheap additions through utilizing $\ell_1$-norm for similarity measurement instead of cosine distance. However, we observe that there exists an increasing gap between CNNs and ANNs with reducing parameters, which cannot be eliminated by existing algorithms. In this paper, we present a simple yet effective Norm-Guided Distillation (NGD) method for $\ell_1$-norm ANNs to learn superior performance from $\ell_2$-norm ANNs. Although CNNs achieve similar accuracy with $\ell_2$-norm ANNs, the clustering performance based on $\ell_2$-distance can be easily learned by $\ell_1$-norm ANNs compared with cross correlation in CNNs. The features in $\ell_2$-norm ANNs are encouraged to achieve intra-class centralization and inter-class decentralization to amplify this advantage. Furthermore, the roughly estimated gradients in vanilla ANNs are modified to a progressive approximation from $\ell_2$-norm to $\ell_1$-norm so that a more accurate optimization can be achieved. Extensive evaluations on several benchmarks demonstrate the effectiveness of NGD on lightweight networks. For example, our method improves ANN by $10.43\%$ with 0.25x GhostNet on CIFAR-100 and $3.1\%$ with 1.0x GhostNet on ImageNet.

*Minjing Dong, Xinghao Chen, Yunhe Wang, Chang Xu, Improving Lightweight AdderNet via Distillation from $\ell_2$ to $\ell_1$-Norm.*

## [Dataset](#contents)

Dataset used: [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)

- Dataset size 32*32 colorful images in 100 classes
    - Train：50,000 images  
    - Test： 10,000 images
- Data format：binary files

## [Environment Requirements](#contents)

- Hardware(Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [Script description](#contents)

### [Script and sample code](#contents)

```bash
AdderNGD
├── ckpt                             # Path of ckpt
├── README.md                        # Readme
└── src
    ├── adder.py                     # adder operation
    ├── res20_adder.py               # Adder ResNet-20
    └── eval.py                      # Inference entry


```

## [Eval process](#contents)

### Usage

After installing MindSpore via the official website, you can start evaluation as follows:

### Launch

```python
  cd ./src/
  python eval.py --checkpoint_file_path [path/to/checkpoint] --eval_dataset_path [path/to/data] --device_target [GPU/Ascend]
```

> checkpoint can be downloaded at https://download.mindspore.cn/model_zoo/research/cv/AdderNGD/.

### Result

```python
result: 0.7034 ckpt= adderngd_resnet20_cifar100_7034.ckpt
```

## [Description of Random Situation](#contents)

We set random norm in models. We also use random seed in attacks.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).