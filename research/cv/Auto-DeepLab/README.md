# Contents

- [Contents](#contents)
- [Auto-DeepLab Description](#auto-deeplab-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
    - [Download and unzip the dataset](#download-and-unzip-the-dataset)
    - [Prepare the mindrecord file](#prepare-the-mindrecord-file)
        - [ModelArts](#modelarts)
        - [Ascend](#ascend)
- [Environment Requirements](#environment-requirements)
    - [Dependencies](#dependencies)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training](#training)
        - [Usage](#usage)
            - [ModelArts](#modelarts)
            - [Ascend](#ascend)
        - [Result](#result)
            - [ModelArts](#modelarts-1)
            - [Ascend](#ascend-1)
    - [Evaluation](#evaluation)
    - [Export](#export)
    - [Inference](#inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Accuracy](#training-accuracy)
        - [Distributed Training Performance](#distributed-training-performance)
        - [Inference Performance on Ascend310](#inference-performance-on-ascend310)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Auto-DeepLab Description](#contents)

DeepLab is a state-of-art deep learning model for semantic image segmentation,  
where the goal is to assign semantic labels.  
we propose to search the network level structure in addition to the
cell level structure, which forms a hierarchical architecture
search space. We present a network level search space that
includes many popular designs, and develop a formulation
that allows efficient gradient-based architecture search.

[Paper](https://arxiv.org/abs/1901.02985v2): Chenxi Liu, Liang-Chieh Chen, Florian Schroff, Hartwig Adam, Wei Hua, Alan L. Yuille, Li Fei-Fei; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 82-92

# [Model Architecture](#contents)

Searched architecture as encoder, with ASPP and reuse Decoder from DeepLabV3+

The whole model is trained from scratch.

# [Dataset](#contents)

used Dataset :[Cityscape Dataset Website](https://www.cityscapes-dataset.com/) (please download 1st and 3rd zip)

It contains 5,000 finely annotated images split into training, validation and testing sets with 2,975, 500, and 1,525 images respectively.

## Download and unzip the dataset

```bash
├─ PATH_TO_DATASET
  ├─ cityscapes
    ├─ gtFine
    └─ leftImage8bit
```

## Prepare the mindrecord file

Using build_mindrecord.py to preprocess the dataset as follows.

```bash
├─ PATH_TO_OUTPUT_MINDRECORD
  ├─ cityscapes_train.mindrecord
  ├─ cityscapes_train.mindrecord.db
  ├─ cityscapes_val.mindrecord
  └─ cityscapes_val.mindrecord.db
```

### ModelArts

Training set

```bash
--train_url=/OBS/PATH/TO/OUTPUT_DIR \
--data_url=/OBS/PATH/TO/DATASET  \
--split=train  \
--modelArts_mode=True  \
```

Validation set

```bash
  --train_url=/OBS/PATH/TO/OUTPUT_DIR \
  --data_url=/OBS/PATH/TO/DATASET  \
  --split=val  \
  --modelArts_mode=True  \
```

### Ascend

Training set

```bash
python build_mindrecord.py --train_path=/PATH/TO/OUTPUT_DIR \
                           --data_path=/PATH/TO/DATASET  \
                           --split=train  \
                           --modelArts_mode=False
```

Validation set

```bash
python build_mindrecord.py --train_url=/PATH/TO/OUTPUT_DIR \
                           --data_url=/PATH/TO/DATASET  \
                           --split=val  \
                           --modelArts_mode=False
```

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Dependencies](#contents)

- opencv-python >= 4.1.2.30
- Python==3.7.5
- Mindspore==1.3

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:

```bash
├─ Auto-DeepLab
  ├─ scripts
    ├─ run_train.sh                      # launch ascend training(1 pcs)
    ├─ run_distribute_train.sh           # launch distributed ascend training(8 pcs)
    ├─ run_eval.sh                       # launch ascend eval
  ├─ src
    ├─ __init__.py                       # init file
    ├─ config.py                         # config file
    ├─ core
      ├─ __init__.py                     # init file
      ├─ aspp.py                         # aspp of Auto-DeepLab
      ├─ decoder.py                      # decoder of Auto-DeepLab
      ├─ encoder.py                      # encoder of Auto-DeepLab
      └─ model.py                        # The whole model of Auto-DeepLab
    ├─ modules
      ├─ __init__.py                     # init file
      ├─ bn.py                           # define bath normalization methods
      ├─ genotypes.py                    # collections of operations
      ├─ operations.py                   # define operations used in nas cells
      └─ schedule_drop_path.py           # define schedule drop path method
    └─ utils
      ├─ __init__.py                     # init file
      ├─ cityscapes.py                   # create cityscapes dataset
      ├─ dynamic_lr.py                   # learning rate decay strategy
      ├─ loss.py                         # implement of loss function
      └─ utils.py                        # utilities
  ├─ build_mindrecord.py                 # convert cityscapes dataset to mindrecord
  ├─ eval.py                             # evaluate code
  ├─ export.py                           # export mindir script
  ├─ train.py                            # train code
  ├─ requirements.txt
  └─ README.md                           # descriptions about Auto-DeepLab  
```

## [Script Parameters](#contents)

Major hyper-parameters are as follows:

```bash
"epochs": 4000                              # Number of epochs
"batch_size": 16                            # batch size
"seed": 0                                   # random seed
"modelArts": True                           # whether training on modelArts
"base_lr": 0.05                             # base learning rate
"warmup_start_lr": 5e-6                     # warm up start learning rate
"warmup_iters": 1000                        # Number of steps for warm up
"min_lr": None                              # Minimum value of learning rate
"weight_decay": 1e-4                        # L2 penalty of optimizer
"num_classes": 19                           # number of classes
"ignore_label": 255                         # background label
"crop_size": 769                            # image crop size, 769 in cityscapes
"workers": 4                                # number of data loading workers
"filter_multiplier": 20                     # filter multiplier
"block_multiplier": 5                       # block multiplier
"parallel": True                            # whether use SyncBatchNormal method
"use_ABN": True                             # whether use ABN
"affine": True                              # whether use affine in batch normalization
"bn_momentum": 0.995                        # the momentum for the running_mean and running_var computation
"bn_eps": 1e-5                              # eps for batch normalization
"drop_path_keep_prob": 1.0                  # keep probability in sdp, 1.0 means do not drop path
"net_arch": None                            # nas network architecture
"cell_arch": None                           # nas cell architecture
"criterion": ohemce                         # loss function 'ce', or 'ohemce'
"ohem_thresh": 0.7                          # topk present pixels used to compute loss
"initial-fm": None                          # init filter_multiplier
"ckpt_name": None                           # resume setting, checkpoint file name in obs
"save_epochs": 100                          # epoch gaps for saving model.
```

## [Training](#contents)

### Usage

#### ModelArts

launch distributed training on ModelArts (If you want to run in ModelArts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/)).

  ModelArts parameters settings：

```bash
    --train_url=/PATH/TO/OUTPUT_DIR \
    --data_url=/PATH/TO/MINDRECORD  \
    --modelArts_mode=True  \
    --epochs=4000 \
    --batch_size=16 \
    --bn_momentum=0.995 \
    --drop_path_keep_prob=1.0
```

#### Ascend

- launch training on Ascend with single device

```bash
bash scripts/run_train.sh [DEVICE_ID] [DATASET_PATH] [EPOCHS]
```

- launch distributed training on Ascend

```bash
bash scripts/run_distribute_train.sh [RANK_SIZE] [RANK_TABLE_FILE] [DATASET_PATH] [EPOCHS]
```

Note: To achieve the best mIOU with batch size = 16, using single Ascend 910 NPU is totally impractical,
due to time consumption and memory constraints. If you want to train from scratch,
please use the distributed training script.

### Result

#### ModelArts

- Ascend 910 * 8 on ModelArts

```bash
epoch: 1 step: 186, loss is 3.1355295
epoch time: 2939529.088 ms, per step time: 15803.920 ms
epoch: 2 step: 186, loss is 2.2626276
epoch time: 113427.926 ms, per step time: 609.828 ms
epoch: 3 step: 186, loss is 1.6811727
epoch time: 116354.789 ms, per step time: 625.563 ms
epoch: 4 step: 186, loss is 1.0462347
epoch time: 113887.618 ms, per step time: 612.299 ms
epoch: 5 step: 186, loss is 1.3679274
epoch time: 111030.954 ms, per step time: 596.941 ms
epoch: 6 step: 186, loss is 0.7847571
epoch time: 110045.064 ms, per step time: 591.640 ms
epoch: 7 step: 186, loss is 1.4403358
epoch time: 109904.390 ms, per step time: 590.884 ms
```

#### Ascend

- single Ascend 910, batch_size = 8

```bash
=> Trying build ohemceloss
Using Poly LR Scheduler!
epoch: 1 step: 1487, loss is 1.8537399
epoch time: 1582425.643 ms, per step time: 1064.173 ms
epoch: 2 step: 1487, loss is 1.0086884
epoch time: 648619.045 ms, per step time: 436.193 ms
epoch: 3 step: 1487, loss is 1.4574875
epoch time: 650850.311 ms, per step time: 437.694 ms
epoch: 4 step: 1487, loss is 1.7041992
epoch time: 652513.109 ms, per step time: 438.812 ms
epoch: 5 step: 1487, loss is 1.317204
epoch time: 652811.297 ms, per step time: 439.012 ms
epoch: 6 step: 1487, loss is 0.91216534
epoch time: 657633.585 ms, per step time: 442.255 ms
epoch: 7 step: 1487, loss is 1.667794
epoch time: 656531.723 ms, per step time: 441.514 ms
epoch: 8 step: 1487, loss is 1.2525432
epoch time: 656824.923 ms, per step time: 441.711 ms
epoch: 9 step: 1487, loss is 1.5522884
epoch time: 655388.507 ms, per step time: 440.745 ms
epoch: 10 step: 1487, loss is 0.935484
epoch time: 656751.908 ms, per step time: 441.662 ms
```

- distributed training (Ascend 910 * 8), batch_size = 16 (batch_size = 2 per device)

```bash
Using Poly LR Scheduler!
epoch: 1 step: 186, loss is 3.4315295
epoch time: 1846726.160 ms, per step time: 9928.635 ms
epoch: 2 step: 186, loss is 2.0722709
epoch time: 102844.377 ms, per step time: 552.927 ms
epoch: 3 step: 186, loss is 1.3147837
epoch time: 104283.465 ms, per step time: 560.664 ms
epoch: 4 step: 186, loss is 2.187191
epoch time: 103960.170 ms, per step time: 558.926 ms
epoch: 5 step: 186, loss is 1.5808569
epoch time: 99759.099 ms, per step time: 536.339 ms
epoch: 6 step: 186, loss is 0.8599842
epoch time: 99775.344 ms, per step time: 536.427 ms
epoch: 7 step: 186, loss is 2.6115859
epoch time: 98197.725 ms, per step time: 527.945 ms
epoch: 8 step: 186, loss is 0.95657295
epoch time: 100802.975 ms, per step time: 541.951 ms
epoch: 9 step: 186, loss is 0.94862676
epoch time: 97839.472 ms, per step time: 526.019 ms
epoch: 10 step: 186, loss is 1.0157641
epoch time: 98506.521 ms, per step time: 529.605 ms
```

## [Evaluation](#contents)

- running on ModelArts

```bash
    --train_url=/PATH/TO/OUTPUT_DIR \
    --data_url=/PATH/TO/MINDRECORD  \
    --modelArts_mode=True  \
    --ckpt_name=[CHECKPOINT_NAME] \
    --batch_size=1 \
    --split=val \
    --parallel=False \
    --ms_infer=False
```

- running on Ascend

```bash
bash scripts/run_eval.sh [DATASET_PATH] [CKPT_FILE] [OUTPUT_PATH]
```

## [Export](#contents)

- Export MINDIR

```bash
python export.py --filter_multiplier=20 --parallel=False --ckpt_name=[CKPT_NAME]
```

## [Inference](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

- Inference on Ascend310 device

```bash
cd /PATH/TO/Auto-DeepLab/scripts
bash run_infer_310.sh /PATH/TO/MINDIR/Auto-DeepLab-s.mindir /PATH/TO/DATASET/cityscapes/ 0
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Accuracy

Auto-DeepLab S

| **Itr** | mIOU  | mIOU in paper |
| :-----: | :-----: | :-------------: |
| 0.5M | 76.16 | 75.20 |
| 1.0M | 77.30 | 77.09 |
| 1.5M | 78.13 | 78.00 |

Note: In the paper, Itr-0.5M, 1M & 1.5M are correspondent to Epoch 1344, 2688, 4032, due to batch size = 8.
In order to reach such mIOU in paper, you should fine-tune Batch-Normalization parameters, which means batch size should
be 16 or larger. Simply, we set batch size = 16 and Epoch 1300, 2700, 4000 correspondent to Itr-0.5M, 1M & 1.5M in paper,
and the best bn_momentum corresponding to different epochs are shown below.

### Distributed Training Performance

| Parameters                 | Auto-DeepLab                                                |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910 * 8; CPU 2.60GHz, 192cores; Memory 755G          |
| uploaded Date              | 11/11/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | Cityscapes (cropped 769*769)                                |
| Training Parameters        | epoch=(1300, 2700, 4000), batch_size = 16, lr=0.05, bn_momentum=(0.995, 0.9, 0.99)  |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Cross Entropy with Online Hard Example Mining               |
| outputs                    | probability                                                 |
| Speed                      | 589.757 ms/step (8pcs)                                      |
| Total time                 | (42, 82, 125) hour (8pcs)                                   |
| Checkpoint                 | 85.37m (.ckpt file)                                         |

### Inference Performance on Ascend310

| Parameters                 | Auto-DeepLab                         |
| -------------------------- | ------------------------------------ |
| Resource                   | Ascend 310 * 1                       |
| uploaded Date              | 12/6/2021 (month/day/year)          |
| MindSpore Version          | 1.3.0                                |
| Dataset                    | Cityscapes (full image 1024*2048)    |
| Speed                      | 1677.48 ms/img                       |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
