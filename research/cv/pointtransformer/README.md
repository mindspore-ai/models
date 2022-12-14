# Contents

- [Contents](#contents)
- [Point Transformer Description](#point-transformer-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
- [Script and Sample Code](#script-and-sample-code)
- [Script Parameters](#script-parameters)
- [Training Process](#training-process)
- [Training Command](#training-command)
- [Evaluation Process](#evaluation-process)
- [Evaluation](#evaluation)
- [Model Description](#model-description)
- [Performance](#performance)
- [Training Performance](#training-performance)
- [Classification](#classification)
- [Segmentation](#segmentation)
- [Inference Performance](#inference-performance)
- [Classification](#classification-1)
- [Segmentation](#segmentation-1)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Point Transformer Description](#contents)

Point Transformer was proposed in 2021.It design a self-attention layers for point clouds and use these to construct
self-attention networks for tasks such as semantic scene segmentation, object part segmentation, and object classification.
The networks are based purely on self-attention and pointwise operations

[Paper](https://arxiv.org/abs/2012.09164): Zhao H, Jiang L, Jia J, et al. Point transformer[C]
Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 16259-16268.

# [Model Architecture](#contents)

The hierarchical structure of Point Transformer is composed by a number of *point transformer block* *transition down block*
and *transition up block*. Point Transformer Block is a residual block  with the point transformer layer at its core.The input
is a set of feature vectors $x$ with associated 3D coordinates $p$. The point transformer block facilitates information exchange
between these localized feature vectors, producing new feature vectors for all data points as its output. The transition down block
is to reduce the cardinality of the point set as required, for example from N to N/4 in the transition from the first to the second stage.
transition up block is to map features from the downsampled input point set P2 onto its superset. The feature encoder in point transformer
networks for semantic segmentation and classification has five stages that operate on progressively downsampled point sets.
The downsampling rates for the stages are [1, 4, 4, 4, 4], thus the cardinality of the point set produced by each stage is [N, N/4, N/16, N/64, N/256],
where N is thenumber of input points.

# [Dataset](#contents)

Dataset used: alignment [ModelNet40](<https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip>)

- Dataset size：6.48G，Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is
  zero-mean and normalized into a unit sphere.
    - Train：5.18G, 9843 point clouds
    - Test：1.3G, 2468 point clouds

- Data format：txt files

Dataset used: alignment [ShapeNetCore](<https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip>)

- Dataset size：2.48G. It consists of 16,880 models from 16 shape categories, with 14,006 3D models for training and 2,874 for testing.
- Data format：txt files

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- The third-party libraries
    - numpy==1.21.3
    - PyYAML==6.0
    - tqdm==4.64.0

    ```bash
        pip install -r requirements.txt
    ```

- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

1. Clone the MindSpore model repo

```bash
git clone https://gitee.com/mindspore/models.git

cd models/research/cv/pointtransformer
```

2. Create softlink for dataset

```bash
mkdir dataset && cd dataset

ln -s path-to-modelnet40_normal_resampled
ln -s path-to-shapenetcore_partanno_segmentation_benchmark_v0_normal
```

The struct of directory:

```text
└── dataset
    ├── modelnet40_normal_resampled
    └── shapenetcore_partanno_segmentation_benchmark_v0_normal
```

3. If you want to train and evaluate the classification ability:

```bash
# Run training for classification with ModelNet40
# Run distributed training
bash scripts/run_distribute_train.sh [DATASET_PATH] [LOG_PATH] [CONFIG] [DEVICE_NUM] [RANK_TABLE]
# example:
bash scripts/run_distribute_train.sh ./dataset log_cls config_cls.yaml 8 hccl_8p_01234567_127.0.0.1.json
# Run stand-alone training:
bash scripts/run_standalone_train.sh [DATASET_PATH] [LOG_PATH] [DEVICE_ID] [CONFIG]
# example:
bash scripts/run_standalone_train.sh ./dataset log_cls 2 config_cls.yaml
# Evaluate
bash scripts/run_standalone_eval.sh [DATA_PATH] [CONFIG] [CKPT_NAME]
# example:
bash scripts/run_standalone_eval.sh ./dataset config_cls.yaml best_model.ckpt
```

4. If you want to train and evaluate the segmentation ability :

```bash

# Run training for segmentation with ShapeNetCore
# Run distributed training
bash scripts/run_distribute_train.sh [DATASET_PATH] [LOG_PATH] [CONFIG] [DEVICE_NUM] [RANK_TABLE]
# example:
bash scripts/run_distribute_train.sh ./dataset log_seg config_seg.yaml 8 hccl_8p_01234567_127.0.0.1.json
# Run stand-alone training:
bash scripts/run_standalone_train.sh [DATASET_PATH] [LOG_PATH] [DEVICE_ID] [CONFIG]
# example:
bash scripts/run_standalone_train.sh ./dataset log_seg 2 config_seg.yaml
# Evaluate
bash scripts/run_standalone_eval.sh [DATA_PATH] [CONFIG] [CKPT_NAME]
# example:
bash scripts/run_standalone_eval.sh ./dataset config_seg.yaml best_model.ckpt
```

# [Script Description](#contents)

# [Script and Sample Code](#contents)

```bash
 ├── PointTransformer
    ├── eval_cls.py                      # eval ModelNet40
    ├── eval_seg.py                      # eval ShapeNetCore
    ├── export.py
    ├── scripts
    │   ├── run_eval_cls.sh              # launch evaluating for ModelNet40
    │   ├── run_eval_seg.sh              # launch evaluating for ShapeNetCore
    │   ├── run_train_cls.sh             # launch training for ModelNet40
    │   └── run_train_seg.sh             # launch training for ShapeNetCore
    ├── src
    │   ├── config
    │   │   ├── config_cls.yaml          # config file for ModelNet40
    │   │   ├── config_seg.yaml          # config file for ShapeNetCore
    │   │   └── default.py               # parse configuration
    │   ├── dataset
    │   │   ├── ModelNet.py              # data preprocessing
    │   │   └── ShapeNet.py              # data preprocessing
    │   ├── model
    │   │   ├── point_helper.py          # network definition utils
    │   │   ├── pointTransfomrer_cls.py  # network for classification
    │   │   ├── pointTransformer.py      # point transformer core
    │   │   └── pointTransformerSeg.py   # network for segmentation
    │   └── utils
    │       ├── callback.py              # network for segmentation
    │       ├── common.py                # common preprocessing
    │       ├── local_adapter.py         # warp modelarts processing
    │       ├── lr_scheduler.py          # dynamic learning rate
    │       └── metric.py                # IOU for pointcloud segmentation
    ├── train_cls.py                     # train net
    └── train_seg.py                     # train net
```

# [Script Parameters](#contents)

```bash
Major parameters in train.py are as follows:
--config_path          # Path to Configuration yaml
--batch_size           # Training batch size.
--epoch_size           # Total training epochs.
--learning_rate        # Training learning rate.
--weight_decay         # Optimizer weight decay.
--dataset_path         # The path to the train and evaluation datasets.
--save_checkpoint_path # The path to save files generated during training.
--use_normals          # Whether to use normals data in training.
--pretrained_ckpt      # The file path to load checkpoint.
--is_modelarts         # Whether to use modelarts.
```

# [Training Process](#contents)

## Training Command

```bash
# Distributed training
bash scripts/run_distribute_train.sh [DATASET_PATH] [LOG_PATH] [CONFIG] [DEVICE_NUM] [RANK_TABLE]
# Stand-alone training
bash scripts/run_standalone_train.sh [DATASET_PATH] [LOG_PATH] [DEVICE_ID] [CONFIG]
```

For example:

```bash
# Run distributed training for classification with ModelNet40
bash scripts/run_distribute_train.sh ./dataset log_cls config_cls.yaml 8 hccl_8p_01234567_127.0.0.1.json

# Run stand-alone training for classification with ModelNet40
bash scripts/run_standalone_train.sh ./dataset log_cls 2 config_cls.yaml

# Run distributed training for segmentation with ShapeNetCore
bash scripts/run_distribute_train.sh ./dataset log_seg config_seg.yaml 8 hccl_8p_01234567_127.0.0.1.json

# Run stand-alone training for classification with ShapeNetCore
bash scripts/run_standalone_train.sh ./dataset log_seg 2 config_seg.yaml
```

After training, the loss value will be achieved as follows:

```bash
# train log
epoch: 1 step: 94, loss is 2.88671875
epoch time: 951506.524 ms, per step time: 10122.410 ms
epoch: 2 step: 94, loss is 1.85546875
epoch time: 490416.995 ms, per step time: 5217.202 ms
...
```

The model checkpoint will be saved in the '[LOG_PATH/ckpt]' directory.

# [Evaluation Process](#contents)

## Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

```bash
# Evaluate
bash scripts/run_standalone_eval.sh [DATA_PATH] [CONFIG] [CKPT_NAME]
```

- Evaluate classification

```bash
# example:
bash scripts/run_standalone_eval.sh ./dataset config_cls.yaml best_model.ckpt
```

- Evaluate segmentation

```bash
# example:
bash scripts/run_standalone_eval.sh ./dataset config_seg.yaml best_model.ckpt
```

You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

```bash
# classification
'Accuracy': 0.9254
# segmentation
ins. mIoU is 0.83, cat. mIoU is 0.76

```

# [Model Description](#contents)

## [Performance](#contents)

## Training Performance

### Classification

| Parameters                 | Ascend                                            |
| -------------------------- | ------------------------------------------------- |
| Model Version              | Point Transformer                                 |
| Resource                   | Ascend 910; CPU 24cores; Memory 256G; OS Euler2.8 |
| uploaded Date              |                                                   |
| MindSpore Version          | 1.5.0                                             |
| Dataset                    | ModelNet40                                        |
| Training Parameters        | epoch=200, steps=7600, batch_size=32, lr=0.05     |
| Optimizer                  | SGD                                               |
| Loss Function              | CrossEntropy                                      |
| outputs                    | probability                                       |
| Loss                       | 0.05                                              |
| Speed                      | 3.8 s/step (8p)                                   |
| Total time                 | 18.3 h (8p)                                       |
| Checkpoint for Fine tuning | 81MB (.ckpt file)                                 |

### Segmentation

| Parameters                 | Ascend                                            |
| -------------------------- | ------------------------------------------------- |
| Model Version              | Point Transformer                                 |
| Resource                   | Ascend 910; CPU 24cores; Memory 256G; OS Euler2.8 |
| uploaded Date              |                                                   |
| MindSpore Version          | 1.5.0                                             |
| Dataset                    | ShapeNetCore                                      |
| Training Parameters        | epoch=200, steps=7600, batch_size=32, lr=0.05     |
| Optimizer                  | SGD                                               |
| Loss Function              | CrossEntropy                                      |
| outputs                    | probability                                       |
| Loss                       | 0.01                                              |
| Speed                      | 5.2 s/step (8p)                                   |
| Total time                 | 1days                                             |
| Checkpoint for Fine tuning | 136MB (.ckpt file)                                |

## Inference Performance

### Classification

| Parameters        | Ascend                                            |
| ----------------- | ------------------------------------------------- |
| Model Version     | Point Transformer                                 |
| Resource          | Ascend 910; CPU 24cores; Memory 256G; OS Euler2.8 |
| Uploaded Date     |                                                   |
| MindSpore Version | 1.5.0                                             |
| Dataset           | ModelNet40                                        |
| Batch_size        | 32                                                |
| Outputs           | probability                                       |
| Accuracy          | 92.5% (1p)                                        |
| Total time        |                                                   |

### Segmentation

| Parameters        | Ascend                                            |
| ----------------- | ------------------------------------------------- |
| Model Version     | Point Transformer                                 |
| Resource          | Ascend 910; CPU 24cores; Memory 256G; OS Euler2.8 |
| Uploaded Date     |                                                   |
| MindSpore Version | 1.5.0                                             |
| Dataset           | ModelNet40                                        |
| Batch_size        | 32                                                |
| Outputs           | probability                                       |
| mIoU              | 84% (1p)                                          |
| Total time        |                                                   |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
