# Contents

<!-- TOC -->

[查看中文](./README_CN.md)

- [STGCN Description](#STGCN-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-2)
        - [Result](#result-2)
    - [Model Export](#model-export)
    - [Inference Process](#inference-process)
        - [Usage](#usage-3)
        - [Result](#result-3)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random State](#description-of-random-state)
- [ModelZoo Homepage](#modelzoo-homepage)

# [STGCN Description](#contents)

This novel deep learning framework, Spatio-temporal Graph Convolutional Network (STGCN), is proposed in article to solve the problem
of time series prediction in the general field. Authors formulate the problem on graphs and build the model with complete convolutional structures,
which enable much faster training speed with fewer parameters. STGCN effectively captures comprehensive spatio-temporal correlations through modeling
multi-scale traffic networks and consistently outperforms state-of-the-art baselines on various real-world traffic datasets.

[Paper](https://arxiv.org/abs/1709.04875): Bing yu, Haoteng Yin, and Zhanxing Zhu. "Spatio-Temporal Graph Convolutional Networks:
A Deep Learning Framework for Traffic Forecasting." Proceedings of the 27th International Joint Conference on Artificial Intelligence. 2017.

# [Model Architecture](#contents)

The STGCN model structure is composed of two spatio-temporal convolution blocks (ST-Conv blocks) and fully-connected output layer.
Each ST-Conv block contains two temporal gated convolution layers and one spatial graph convolution layer in the middle.
There are two different convolution methods for spatial convolution blocks: Cheb and GCN.

# [Dataset](#contents)

Dataset used:

- Only [PeMSD7-M](https://github.com/hazdzz/STGCN/tree/main/data/pemsd7-m) dataset is available for download. Adj_mat.csv can be found in older version [adj_mat.csv](https://github.com/hazdzz/STGCN/blob/3ca6c36f0e4b874976891d5b09d9d5b0858680d3/data/train/road_traffic/pemsd7-m/adj_mat.csv)
- BJER4 is not used, as it is a private dataset which is restricted by a confidentiality agreement.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information about MindSpore, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- Other
    - pandas
    - sklearn
    - easydict

# [Quick start](#contents)

After installing MindSpore through the official website, you can start training and evaluation through the following steps:

- running on Ascend with default parameters

```shell
# single card
python train.py --device_target="Ascend" --train_url="" --data_url="" --run_distribute=False --run_modelarts=False --graph_conv_type="chebconv" --n_pred=9

# multi card
bash scripts/run_distribute_train.sh train_code_path data_path n_pred graph_conv_type rank_table
```

- running on GPU with default parameters

```shell
# single card
python train.py --device_target="GPU" --train_url="" --data_url="" --run_distribute=False --run_modelarts=False --graph_conv_type="chebconv" --n_pred=9

# single card
bash scripts/run_single_train_gpu.sh data_path n_pred graph_conv_type device_id
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── STGCN
    ├── scripts
        ├── run_distribute_train.sh       # training on Ascend with 8P
        ├── run_single_train_gpu.sh       # training on GPU 1P
        ├── run_eval_ascend.sh            # testing on Ascend
        ├── run_eval_gpu.sh               # testing on GPU
    ├── src
        ├── model
            ├──layers.py                  # model layer
            ├──metric.py                  # network with losscell
            ├──models.py                  # network model
        ├──argparser.py                   # command line parameters
        ├──config.py                      # parameters
        ├──dataloder.py                   # creating dataset
        ├──utility.py                     # calculate laplacian matrix and evaluate metric
        ├──weight_init.py                 # layernorm weight init
    ├── train.py                          # training network
    ├── eval.py                           # testing network performance
    ├── export.py
    ├── postprocess.py                    # compute accuracy for ascend310
    ├── preprocess.py                     # process dataset for ascend310
    ├── README.md
    ├── README_CN.md
```

## [Script Parameters](#contents)

Training and evaluation parameters can be set in config.py

- config for STGCN

```text
    stgcn_chebconv_45min_cfg = edict({
    'learning_rate': 0.003,
    'n_his': 12,
    'n_pred': 9,
    'epochs': 50,
    'batch_size': 8,  # config.batch_size * int(8 / device_num)
    'decay_epoch': 10,
    'gamma': 0.7,
    'stblock_num': 2,
    'Ks': 3,
    'Kt': 3,
    'time_intvl': 5,
    'drop_rate': 0.5,
    'weight_decay_rate': 0.0005,
    'gated_act_func':"glu",
    'graph_conv_type': "chebconv",
    'mat_type': "wid_sym_normd_lap_mat",
    })
```

For more information, please check `config.py`.

## [Training process](#contents)

### Usage

- running on Ascend

```shell
# train single card
python train.py --device_target="Ascend" --train_url="" --data_url="" --run_distribute=False --run_modelarts=True --graph_conv_type="chebconv" --n_pred=9

# train 8 card
bash scripts/run_distribute_train.sh train_code_path data_path n_pred graph_conv_type rank_table
```

> Note: To train on 8p Ascend put `RANK_TABLE_FILE` in `scripts` folder. [How to generate RANK_TABLE_FILE](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

- running on GPU

```shell
# train single card
python train.py --device_target="GPU" --train_url="./checkpoint" --data_url="./data" --run_distribute=False --run_modelarts=False --graph_conv_type="chebconv" --n_pred=9

# train single card
bash scripts/run_single_train_gpu.sh data_path n_pred graph_conv_type device_id
```

### Result

During training epochs, steps and loss will be displayed in terminal:

```text
  epoch: 1 step: 139, loss is 0.429
  epoch time: 203885.163 ms, per step time: 1466.800 ms
  epoch: 2 step: 139, loss is 0.2097
  epoch time: 6330.939 ms, per step time: 45.546 ms
  epoch: 3 step: 139, loss is 0.4192
  epoch time: 6364.882 ms, per step time: 45.791 ms
  epoch: 4 step: 139, loss is 0.2917
  epoch time: 6378.299 ms, per step time: 45.887 ms
  epoch: 5 step: 139, loss is 0.2365
  epoch time: 6369.215 ms, per step time: 45.822 ms
  epoch: 6 step: 139, loss is 0.2269
  epoch time: 6389.238 ms, per step time: 45.966 ms
  epoch: 7 step: 139, loss is 0.3071
  epoch time: 6365.901 ms, per step time: 45.798 ms
  epoch: 8 step: 139, loss is 0.2336
  epoch time: 6358.127 ms, per step time: 45.742 ms
  epoch: 9 step: 139, loss is 0.2812
  epoch time: 6333.794 ms, per step time: 45.567 ms
  epoch: 10 step: 139, loss is 0.2622
  epoch time: 6334.013 ms, per step time: 45.568 ms
  ...
```

The checkpoint of this model is stored in the `train_url` path

## [Evaluation process](#contents)

### Usage

Use the PeMSD7-m test set for evaluation

- on Ascend

When using python to run, you need to input device, checkpoint path, spatial convolution method, and prediction period.

```shell
python eval.py --device_target="Ascend" --run_modelarts=False --run_distribute=False --device_id=0 --ckpt_url="" --graph_conv_type="" --n_pred=9

# using script to run
bash scripts/run_eval_ascend.sh [device] [data_path] [ckpt_url] [device_id] [graph_conv_type] [n_pred]
```

- on GPU

When using python to run, you need to input device, checkpoint path, spatial convolution method, and prediction period.

```shell
python eval.py --device_target="GPU" --run_modelarts=False --run_distribute=False --device_id=0 --ckpt_url="" --graph_conv_type="" --n_pred=9

# Using script to run
bash scripts/run_eval_gpu.sh [device] [data_path] [ckpt_url] [device_id] [graph_conv_type] [n_pred]
```

### Result

The above python command will run on the terminal, and you can view the result of this evaluation on the terminal. The accuracy of the test set will be presented as follows:

```text
MAE 3.23 | MAPE 8.32 | RMSE 6.06
```

## [Model Export](#contents)

```shell
python export.py --data_url [DATA_URL] --ckpt_file [CKPT_PATH] --n_pred [N_PRED] --graph_conv_type [GRAPH_CONV_TYPE] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

## [Inference Process](#contents)

**Before inference, please refer to [Environment Variable Setting Guide](https://gitee.com/mindspore/models/tree/master/utils/ascend310_env_set/README.md) to set environment variables.**

### Usage

Before performing inference, the minirir file must be exported by export.py. The input file must be in bin format

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
```

### Result

The inference result is saved in the current path, and you can find the result in the acc.log file

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

#### STGCN on PeMSD7-m (Cheb, n_pred=9)

| Parameters                 | Ascend 8p                                        | GPU 1p |
| -------------------------- | ------------------------------------------------ | ------ |
| Model                      | STGCN                                            | STGCN  |
| Environment                | ModelArts; Ascend 910; CPU 2.60GHz, 192cores, Memory, 755G | Ubuntu 18.04.6, 1pcs RTX3090, CPU 2.90GHz, 64cores, RAM 252GB |
| Uploaded Date (month/day/year) | 05/07/2021                                   | 17/01/2021 |
| MindSpore Version          | 1.2.0                                            | 1.5.0 |
| Dataset                    | PeMSD7-M                                         | PeMSD7-M |
| Training Parameters        | epoch=500, steps=139, batch_size=8, lr=0.003     | epoch=50, steps=139, batch_size=64, lr=0.003 |
| Optimizer                  | AdamWeightDecay                                  | AdamWeightDecay |
| Loss Function              | MSE Loss                                         | MSE Loss |
| Outputs                    | probability                                      | probability |
| Final loss                 | 0.183                                            | 0.23 |
| Speed                      | 45.601 ms/step                                   | 44 ms/step |
| Total time                 | 56min                                            | 5 min |
| Scripts                    | [STGCN script](https://gitee.com/mindspore/models/tree/master/research/cv/stgcn#https://arxiv.org/abs/1709.04875) | [STGCN script](https://gitee.com/mindspore/models/tree/master/research/cv/stgcn#https://arxiv.org/abs/1709.04875) | [STGCN script](https://gitee.com/mindspore/models/tree/master/research/cv/stgcn#https://arxiv.org/abs/1709.04875) |

### Evaluation Performance

#### STGCN on PeMSD7-m (Cheb, n_pred=9)

| Parameters          | Ascend                      | GPU 1P |
| ------------------- | --------------------------- | ------ |
| Model Version       | STGCN                       | STGCN  |
| Resource            | Ascend 910                  | Ubuntu 18.04.6, NVIDIA GeForce RTX3090, CPU 2.90GHz, 64cores, RAM 252GB |
| Uploaded Date       | 05/07/2021 (month/day/year) | 17/01/2021 |
| MindSpore Version   | 1.2.0                       | 1.5.0    |
| Dataset             | PeMSD7-M                    | PeMSD7-M |
| batch_size          | 8                           | 64        |
| outputs             | probability                 | probability |
| MAE                 | 3.23                        | 3.24 |
| MAPE                | 8.32                        | 8.25 |
| RMSE                | 6.06                        | 6.03 |
| Model for inference | about 6M(.ckpt fil)         | about 6M(.ckpt fil) |

# [Description of Random State](#contents)

Random seed is set in `train.py` script.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
