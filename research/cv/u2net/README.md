# Contents

- [U-2-Net Description](#u-2-net-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Pretrained model](#pretrained-model)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Run On Modelarts](#run-on-modelarts)
    - [Model Export](#model-export)
    - [Training Process](#training-process)
    - [Ascend310 Inference Process](#ascend310-inference-process)
- [Model Description](#model-description)
    - [Performance](#performance)
    - [Training Performance](#training-performance)
    - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [U-2-Net Description](#contents)

This is the Implementation of the Mindspore code of paper [**U<sup>2</sup>-Net: Going deeper with nested U-structure for
salient object detection** ](http://arxiv.org/abs/2005.09007)

Authors: Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin.

In this paper, we design a simple yet powerful deep network architecture, U<sup>2</sup>-Net, for salient object
detection (SOD). The architecture of our U<sup>2</sup>-Net is a two-level nested U-structure.

# [Model Architecture](#contents)

![](assets/network.png)

The architecture of our U<sup>2</sup>-Net is a two-level nested U-structure.

# [Dataset](#contents)

To train U<sup>2</sup>-Net, We use the dataset [DUTS-TR](http://saliencydetection.net/duts/download/DUTS-TR.zip)

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below£º
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
U-2-Net
 ├─  README.md # descriptions about U-2-Net
 ├─  scripts
     ├─ run_infer_310.sh        # 310 inference
     └─ run_distribute_train.sh # launch Ascend training (8 Ascend)
 ├─ assets # save pics for README.MD
 ├─ ckpts  # save ckpt  
 ├─ src
     ├─   data_loader.py # generate dataset for training
     ├─   loss.py        # loss function define
     └─   blocks.py      # U-2-Net model define
 ├─  train_modelarts.py  # train script for online train
 ├─  test.py  # generate detection images
 ├─  eval.py  # eval script
 ├─  train.py # train script
 ├─  ascend310_infer # 310 main
 ├─  export.py
 ├─  preprocess.py
 └─  postprocess.py
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training
cd scripts
./run_distribute_train.sh [/path/to/content] [/path/to/label] [/path/to/RANK_TABLE_FILE]

# standalone training
python train.py --content_path [/path/to/content] --label_path [/path/to/label]
```

### Training Result

Training result will be stored in './ckpts'. You can find checkpoint file.

### Evaluation Script Parameters

- Run `test.py` to generate semantically segmented pictures
- Run `eval.py` for evaluation.

```bash
# generate semantically segmented pictures
python test.py --content_path [/path/to/content] &>test.log &

# evaling
python eval.py --pred_dir [/path/to/pred_dir] --label_dir [/path/to/label] &>evaluation.log &
```

### [Run On Modelarts](#contents)

- Run `train.py` to train on modelarts

```bash
#In order to run on modelars, you should place data files like this:
# └─  dataset
#   ├─  pre_ckpt
#   └─  DUTS
#params:
#  run_distribute # Run distribute, default: false.
#  data_url       # path to data on obs default: None
#  train_url      # output path in obs default: None
#  ckpt_name      # prefix of ckpt files, default: u2net
#  run_online     # you should set run_online to 1 if you wnt to run online
#  is_load_pre    # whether use pretrained model, default: false.
```

- Run `test.py` to generate semantically segmented pictures on modelarts

```bash
#In order to run on modelars, you should place data files like this:
# └─  dataset
#   ├─  pre_ckpt
#   └─  content_dir
#params:
#  data_url       # path to data on obs default: None
#  run_online     # you should set run_online to 1 if you wnt to run online
```

- Run `eval.py` to evaluate on modelarts

```bash
#In order to run on modelars, you should place data files like this:
# └─  dataset
#   ├─  pre_ckpt
#   ├─  label_dir
#   └─  content_dir
#params:
#  data_url       # path to data on obs default: None
#  run_online     # you should set run_online to 1 if you wnt to run online
```

## [Training Process](#contents)

### [Training](#contents)

- Run `run_standalone_train_ascend.sh` for non-distributed training of U-2-Net model.

```bash
# standalone training
python train.py --content_path [/path/to/content] --label_path [/path/to/label]
```

### [Distributed Training](#contents)

- Run `run_distributed_train_ascend.sh` for distributed training of U-2-Net model.

```bash
# distributed training
cd scripts
bash run_distribute_train.sh [/path/to/content] [/path/to/label] [/path/to/RANK_TABLE_FILE]
```

- Notes

1. hccl.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it
   by using the [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).
## [Model Export](#contents)

```bash
python export.py --ckpt_dir [/path/to/ckpt_file]
```

## [Ascend310 Inference Process](#contents)

### Export MINDIR file

```bash
python  export.py --ckpt_file [/path/to/ckpt_file]
```

### Ascend310 Inference

- Run `run_infer_310.sh` for Ascend310 inference.

```bash
# infer
bash run_infer_310.sh [MINDIR_PATH] [CONTENT_PATH] [LABEL_PATH] [DEVICE_ID]
```

Semantically segmented pictures will be stored in the postprocess_Result path and the evaluation result will be stored in evaluation.log.

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 |                                                       |
| -------------------------- | ----------------------------------------------------- |
| Model Version              | v1                                                    |
| Resource                   | Red Hat 8.3.1; Ascend 910; CPU 2.60GHz; 192cores      |
| MindSpore Version          | 1.3.0                                                 |
| Dataset                    | DUTS-TR.                                              |
| Training Parameters        | epoch=1500, batch_size = 16                           |
| Optimizer                  | Adam                                                  |
| Loss Function              | BCELoss                                               |
| outputs                    | semantically segmented pictures                       |
| Speed                      | 8 Ascend: 440 ms/step; 1 Ascend: 303 ms/step;         |
| Total time                 | 8pcs: 15h                                             |
| Checkpoint for Fine tuning | 512M (.ckpt file)                                     |

### Picture Generating Performance

| Parameters        | single Ascend                                    |
| ----------------- | ------------------------------------------------ |
| Model Version     | U-2-Net                                          |
| Resource          | Red Hat 8.3.1; Ascend 910; CPU 2.60GHz; 192cores |
| MindSpore Version | 1.3.0                                            |
| Dataset           | content images                                   |
| batch_size        | 1                                                |
| outputs           | semantically segmented pictures                  |
| Speed             | 95 ms/pic                                        |

### Evaluation Performance

| Parameters        | single Ascend                                    |
| ----------------- | ------------------------------------------------ |
| Model Version     | U-2-Net                                          |
| Resource          | Red Hat 8.3.1; Ascend 910; CPU 2.60GHz; 192cores |
| MindSpore Version | 1.3.0                                            |
| Dataset           | DUTS-TE                                          |
| batch_size        | 1                                                |
| Accuracy          | 85.52%                                           |

# [Description of Random Situation](#contents)

We use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
