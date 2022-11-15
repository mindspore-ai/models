# Contents

[查看中文](./README_CN.md)

<!-- TOC -->

- [Contents](#contents)
- [SDNE Description](#sdne-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick start](#quick-start)
    - [Script Description](#script-description)
        - [Scripts and sample code](#script-and-sample-code)
        - [Script parameters](#script-description)
        - [Training process](#train-process)
            - [Training](#train)
        - [Evaluation process](#evaluation-process)
            - [Evaluation](#evaluation)
        - [Export mindir](#export-mindir-model)\
            - [Usage](#usage)
            - [Result](#result)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Random state description](#description-of-random-state)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# SDNE Description

Network Embedding is an important method for learning low-dimensional representations of network vertices, whose purpose is to capture and preserve network structure.
Different from existing network embeddings, this paper presents a new deep learning network architecture, SDNE, which can effectively capture the highly nonlinear network structure while preserving the global and local structure of the original network.
This work has three main contributions:
（1）The authors propose a structured deep network embedding method that can map data into a highly nonlinear latent space.
（2）The authors propose a novel semi-supervised learning architecture that simultaneously learns the global and local structure of sparse networks.
（3）The authors use the method to evaluate on 5 datasets and apply it to 4 application scenarios with remarkable results.

[Paper](https://dl.acm.org/doi/10.1145/2939672.2939753) ： Wang D ,  Cui P ,  Zhu W. Structural Deep Network Embedding[C]// Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. August 2016.

# Dataset

Used dataset:

- [WIKI](https://github.com/shenweichen/GraphEmbedding/tree/master/data/wiki/) nodes: 2405 edges: 17981

- [CA-GRQC](https://github.com/suanrong/SDNE/tree/master/GraphData/) nodes: 5242 edges: 11496

# Environment Requirements

- Hardware（Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For more information about MindSpore, please check the resources below:
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)
- Other
    - networkx
    - numpy
    - tqdm

# Quick start

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:

```bash
# training on Ascend
bash scripts/run_standalone_train.sh dataset_name /path/data /path/ckpt epoch_num 0
# training on GPU
bash scripts/run_standalone_train_gpu.sh [dataset_name] [data_path] [ckpt_path] [epoch_num] [device_id]

# evaluation on Ascend
bash scripts/run_eval.sh dataset_name /path/data /path/ckpt 0
# evaluation on GPU
bash scripts/run_eval_gpu.sh [dataset_name] [data_path] [ckpt_path] [epoch_num] [device_id]
```

## Script Description

## Script and Sample Code

```text
└── SDNE  
 ├── README_CN.md
 ├── scripts
  ├── run_310_infer.sh
  ├── run_standalone_train.sh
  ├── run_eval.sh
  ├── run_standalone_train_gpu.sh
  └── run_eval_gpu.sh
 ├── ascend310_infer
  ├── inc
   └── utils.h
  ├── build.sh
  ├── convert_data.py                // 转换数据脚本
  └── CMakeLists.txt
 ├── src
  ├── __init__.py
  ├── loss.py
  ├── config.py
  ├── dataset.py
  ├── sdne.py
  ├── initializer.py
  ├── optimizer.py
  └── utils.py
 ├── export.py
 ├── eval.py
 └── train.py
```

## Script Parameters

```text
train.py中主要参数如下:

-- device_id: Device ID used for training or evaluation datasets. This parameter is ignored when using train.sh for distributed training.
-- device_target: Choose from ['Ascend', 'GPU']
-- data_url: dataset path.
-- ckpt_url: Path to store checkpoints.
-- dataset: dataset used. Choose from ['WIKI', 'GRQC']
-- epochs: number of iterations.
-- pretrained: whether to use pretrained parameters.
```

## Train process

### train

- Ascend

```shell
bash scripts/run_standalone_train.sh WIKI /path/wiki /path/ckpt 40 0
```

- GPU

```shell
bash scripts/run_standalone_train_gpu.sh WIKI /path/wiki /path/ckpt 40 0
```

The above shell script will run the training. The results can be viewed in the `train.log` file.
The loss value is as follows:

```text
...
epoch: 36 step: 1, loss is 31.026050567626953
epoch time: 1121.593 ms, per step time: 1121.593 ms
epoch: 37 step: 1, loss is 29.539968490600586
epoch time: 1121.818 ms, per step time: 1121.818 ms
epoch: 38 step: 1, loss is 27.804513931274414
epoch time: 1120.751 ms, per step time: 1120.751 ms
epoch: 39 step: 1, loss is 26.283227920532227
epoch time: 1121.551 ms, per step time: 1121.551 ms
epoch: 40 step: 1, loss is 24.820133209228516
epoch time: 1123.054 ms, per step time: 1123.054 ms
```

- Ascend

```shell
bash scripts/run_standalone_train.sh GRQC /path/grqc /path/ckpt 2 0
```

- GPU

```shell
bash scripts/run_standalone_train_gpu.sh GRQC /path/grqc /path/ckpt 2 0
```

The above shell script will run the training. The results can be viewed in the `train.log` file.
The loss value is as follows:

```text
...
epoch: 2 step: 157, loss is 607002.3125
epoch: 2 step: 158, loss is 638598.0625
epoch: 2 step: 159, loss is 485911.40625
epoch: 2 step: 160, loss is 774514.1875
epoch: 2 step: 161, loss is 733589.0625
epoch: 2 step: 162, loss is 504986.1875
epoch: 2 step: 163, loss is 416679.625
epoch: 2 step: 164, loss is 524830.75
epoch time: 14036.608 ms, per step time: 85.589 ms
```

## Evaluation process

### evaluation

- Ascend

```bash
bash scripts/run_eval.sh  WIKI /path/wiki /path/ckpt 0
```

- GPU

```bash
bash scripts/run_eval_gpu.sh WIKI /path/wiki /path/ckpt 0
```

The above command will run in the background and you can view the result through the `eval.log` file.
The accuracy of the test dataset is as follows:

```text
Reconstruction Precision K  [1, 10, 20, 100, 200, 1000, 2000, 6000, 8000, 10000]
Precision@K(1)= 1.0
Precision@K(10)=        1.0
Precision@K(20)=        1.0
Precision@K(100)=       1.0
Precision@K(200)=       1.0
Precision@K(1000)=      1.0
Precision@K(2000)=      1.0
Precision@K(6000)=      0.9986666666666667
Precision@K(8000)=      0.991375
Precision@K(10000)=     0.966
MAP :  0.6673926856547066
```

- Ascend

```bash
bash scripts/run_eval.sh GRQC /path/grqc /path/ckpt 0
```

- GPU

```bash
bash scripts/run_eval_gpu.sh GRQC /path/grqc /path/ckpt 0
```

The above command will run in the background and you can view the result through the `eval.log` file.
The accuracy of the test dataset is as follows:

```text
Reconstruction Precision K  [10, 100]
getting similarity...
Precision@K(10)=        1.0
Precision@K(100)=       1.0
```

## Export mindir model

```bash
python export.py --dataset [NAME] --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

Argument `ckpt_file` is required, `dataset` is a name of a dataset must be selected from [`WIKI`, `GRQC`], `FILE_FORMAT` must be selected from ["AIR", "MINDIR"].

### Usage

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the `mindir` file needs to be exported via `export.py`.

```bash
# Ascend310 inference
bash run_310_infer.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [DEVICE_ID]
```

`MINDIR_PATH` is the `mindir` file path, `DATASET_NAME` is the dataset name, `DATASET_PATH` is the path to the dataset file (eg `/datapath/sdne_wiki_dataset/WIKI/Wiki_edgelist.txt`).

### Result

The inference results are saved in the current path, and the final accuracy results can be seen in `acc.log`.

# Model description

## Performance

### Training performance

| Parameters        | Ascend                                        | GPU |
| ----------------- | --------------------------------------------- | ---- |
| Model             | SDNE                                          | SDNE |
| Environment       | Ascend 910; CPU: 2.60GHz, 192cores, RAM 755Gb | Ubuntu 18.04.6, GF RTX3090, CPU 2.90GHz, 64cores, RAM 252GB |
| Upload date       | 2021-12-31                                    | 2022-01-30 |
| MindSpore version | 1.5.0                                         | 1.5.0 |
| dataset           | wiki                                          | wiki |
| parameters        | lr=0.002, epoch=40                            | lr=0.002, epoch=40 |
| Optimizer         | Adam                                          | Adam |
| loss function     | SDNE Loss Function                            | SDNE Loss Function |
| output            | probability                                   | probability |
| loss              | 24.82                                         | 24.87 |
| speed             | 1p：1105 ms/step                              | 15 ms/step |
| total time        | 1p：44 sec                                    | 44 sec |
| Parameters(M)     | 1.30                                         | 1.30 |
| Checkpoints       | 15M （.ckpt file）                            | 15M （.ckpt file |
| Scripts           | [SDNE script](https://gitee.com/mindspore/models/tree/master/research/gnn/sdne) | [SDNE script](https://gitee.com/mindspore/models/tree/master/research/gnn/sdne) |

| Parameters          | Ascend                                         |
| ------------- | ----------------------------------------------- |
| Model      | SDNE                                  |
| Environment          | Ascend 910； CPU： 2.60GHz，192内核；内存，755G |
| Upload date      | 2022-4-7                                     |
| MindSpore version | 1.5.0                          |
| dataset        | CA-GRQC                                       |
| parameters      | lr=0.01                     |
| Optimizer        | RMSProp                                             |
| loss function      | SDNE Loss Function                       |
| output          | probability                                            |
| loss          | 736119.18                                            |
| speed | 1p：86ms/step |
| total time | 1p：28sec |
| Parameters(M) | 1.05 |
| Checkpoints | 13M （.ckpt file） |
| Scripts | [SDNE script](https://gitee.com/mindspore/models/tree/master/research/gnn/sdne) |

### Evaluation performance

| Parameters        | Ascend              | GPU |
| ----------------- | ------------------- | ------------------ |
| Model             | SDNE                | SDNE     |
| Environment       | Ascend 910          | Ubuntu 18.04.6, GF RTX3090, CPU 2.90GHz, 64cores, RAM 252GB |
| Upload date       | 2021-12-31          | 2022-01-30 |
| MindSpore version | MindSpore-1.3.0-c78 | 1.5.0 |
| dataset           | wiki                | wiki |
| output            | probability         | probability |
| MAP               | 1p: 66.74%          | 1p: 66.73% |

| Parameters          | Ascend            |
| ------------- | ------------------ |
| Model      | SDNE     |
| Environment          | Ascend 910         |
| Upload date      | 2022/4/7        |
| MindSpore version | MindSpore-1.3.0-c78      |
| dataset        | CA-GRQC          |
| output          | 概率               |
| MAP        | 1卡：1 |

# Description of Random State

Random seed fixed in `train.py` for python, numpy and mindspore.

# ModelZoo homepage  

Please check the official [homepage](https://gitee.com/mindspore/models).
