# Contents

- [Contents](#contents)
    - [SNN-MLP Description](#snn-mlp-description)
    - [Model architecture](#model-architecture)
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

## [SNN-MLP Description](#contents)

 To efficiently communicate between tokens, we incorporate the mechanism of LIF neurons into the MLP models, and achieve better accuracy without extra FLOPs.

[Paper](https://arxiv.org/pdf/2203.14679.pdf): Wenshuo Li, Hanting Chen, Jianyuan Guo, Ziyang Zhang, Yunhe Wang. Brain-inspired Multilayer Perceptron with Spiking Neurons. arxiv 2203.14679.

## [Model architecture](#contents)

A block of SNN-MLP is shown below:

![image-20211026160438718](./snnmlp.png)

## [Dataset](#contents)

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size 224*224 colorful images in 1000 classes
    - Train: 1,281,167 images  
    - Test: 50,000 images
- Data format: jpeg
    - Note: Data will be processed in dataset.py

## [Environment Requirements](#contents)

- Hardware(Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Script description](#contents)

### [Script and sample code](#contents)

```text

SNN-MLP
├── eval.py # inference entry
├── fig
│   └── snnmlp.png # the illustration of snn_mlp network
├── readme.md # Readme
└── src
    ├── dataset.py # dataset loader
    └── snn_mlp.py # snn_mlp network

```

## [Eval process](#contents)

### Usage

After installing MindSpore via the official website, you can start evaluation as follows:

### Launch

```bash

# infer example
  python eval.py --dataset_path [DATASET] --platform GPU --checkpoint_path [CHECKPOINT_PATH] --model [snnmlp_t|snnmlp_s|snnmlp_b] #GPU

```

> checkpoint can be downloaded at https://download.mindspore.cn/model_zoo/research/cv/snn_mlp/.

### Result

```text

result: {'acc': 0.8185} ckpt= ./SNNMLP_T.ckpt

```

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).