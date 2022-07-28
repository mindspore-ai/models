# Contents

- [Contents](#contents)
    - [Algorithm Introduction](#algorithm-introduction)
    - [Algorithm Description](#algorithm-description)
        - [Configuring](#configuring)
    - [Dataset](#dataset)
    - [Requirements](#requirements)
        - [Hardware (Ascend)](#hardware-ascend)
        - [Framework](#framework)
        - [For more information, please check the resources below](#for-more-information-please-check-the-resources-below)
    - [Script Description](#script-description)
        - [Scripts and Sample Code](#scripts-and-sample-code)
        - [Script Parameter](#script-parameter)
    - [Training Process](#training-process)
        - [For training](#for-training)
    - [Evaluation](#evaluation)
        - [Evaluation Process](#evaluation-process)
        - [Evaluation Result](#evaluation-result)
    - [ModelZoo Homepage](#modelzoo-homepage)

## Algorithm Introduction

The Davincit chips are used in many service scenarios for image and video classification, object detection, and segmentation. In these scenarios, there are various requirements for real-time performance or precision. The current model is optimized based on a backbone network. Therefore, it is important to find a backbone model that runs efficiently and with high accuracy on a D-chip.

## Algorithm Description

The current NAS method has two approaches:

a) Search for the Cell Internal Structure(Micro-level Structure). However, the stacking mode of cells is fixed. Such work has DARTS, PNAS, AmoebaNet.

b) Search for how to stack these Cell's Macro-level structure of channel. The structure of the cell is fixed such cell has Residual Bottleneck, ResNext Block, MobileNet Block. Such work has EffecientNet, RegNet, SM-NAS.

It can be seen that a neural network Micro-level Structure and Macro-level structure are very important for their performance. We expect to use the NAS method to obtain efficient cell designs on D chips and to use these efficient cells to construct an efficient entire network. Therefore, we adopt a two-stage search strategy:

Micro-level Search: Search for efficient cell designs on a D-chip.

Macro-Level Search: Use the efficient cell design obtained in the previous phase to construct an efficient entire network.

### Configuring

For details, see the configuration file src/dnet_nas.yml in the sample code.

```yaml
pipeline: [block_nas, net_nas]

block_nas:
    pipe_step:
        type: SearchPipeStep

    search_algorithm:
        type: DblockNas
        codec: DblockNasCodec
        range:
            max_sample: 100
            min_sample: 10

    search_space:
        hyperparameters:
            -   key: network.backbone.op_num
                type: CATEGORY
                range: [2, 3]
            -   key: network.backbone.skip_num
                type: CATEGORY
                range: [0, 1,]
            -   key: network.backbone.base_channel
                type: CATEGORY
                range:  [16, 32, 64]
                #range:  [16, 32, 48, 56, 64]
            -   key: network.backbone.doublechannel
                type: CATEGORY
                range: [3, 4]
            -   key: network.backbone.downsample
                type: CATEGORY
                range: [3, 4]
    model:
        model_desc:
            modules: ['backbone']
            backbone:
                type: DNet
                n_class: 1000

net_nas:
    search_algorithm:
        type: DnetNas
        codec: DnetNasCodec
        policy:
            num_mutate: 10
            random_ratio: 0.5
        range:
            max_sample: 100
            min_sample: 10

    search_space:
        hyperparameters:
            -   key: network.backbone.code_length
                type: INT
                range: [12,50]
            -   key: network.backbone.base_channel
                type: CATEGORY
                range:  [32, 64, 128]
            -   key: network.backbone.final_channel
                type: CATEGORY
                range:  [512, 1024, 2048]
            -   key: network.backbone.downsample
                type: CATEGORY
                range: [2, 3]
    model:
        models_folder: "{local_base_path}/output/block_nas/"
        model_desc:
            modules: ['backbone']
            backbone:
                type: DNet
                n_class: 1000
```

## Dataset

The benchmark datasets can be downloaded as follows:

[ImageNet2012](https://image-net.org/challenges/LSVRC/2012/).

After downloaded the correspond dataset to the target place, You can configure and use the dataset separately for train and test.

Dataset configuration parameters in src/dnet_nas.yml:

```yaml
        type: Imagenet
        common:
            data_path: "/cache/datasets/ILSVRC/Data/CLS-LOC"
            batch_size: 64
            n_class: 1000

```

## Requirements

### Hardware (Ascend)

> Prepare hardware environment with Ascend.

### Framework

> [MindSpore](https://www.mindspore.cn/install/en)

### For more information, please check the resources below

[MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
[MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## Script Description

### Scripts and Sample Code

```bash
dnet_nas
├── eval.py # inference entry
├── train.py # pre-training entry
├── README.md # Readme
├── scripts
│   ├── run_standalone.sh # shell script for standalone train on ascend
│   ├── run_distributed.sh # shell script for distributed train on ascend
└── src
    └── dnet_nas.yml # options/hyper-parameters of dnet_nas
```

### Script Parameter

> For details about hyperparameters, see src/dnet_nas.yml.

## Training Process

### For training

- Standalone Ascend Training:

```bash
sh scripts/run_standalone.sh
```

- Distributed Ascend Training:

```bash
sh scripts/run_distributed.sh  [RANK_TABLE_FILE]
```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.
`$RANK_TABLE_FILE` is needed when you are running a distribute task on ascend.

> Or one can run following script for all tasks.

```bash
python3 train.py
```

## Evaluation

### Evaluation Process

> Inference example:

Modify src/eval.yml:

```bash
models_folder: [CHECKPOINT_PATH]
```

```bash
python3 eval.py
```

### Evaluation Result

The result are evaluated by the value of accuracy, flops, params and inference time.

For details about inference performance, see noah-vega [modelzoo](https://github.com/huawei-noah/vega/blob/master/docs/model_zoo.md#dnet).

## ModelZoo Homepage

Please check the official [homepage](https://gitee.com/mindspore/models).

