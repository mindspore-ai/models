# Contents

- [Contents](#contents)
- [PointNet Description](#pointnet2-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [310 Inference Process](#310-infer-process)
        - [Export MindIR](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [PointNet Description](#contents)

PointNet was proposed in 2017, it is a hierarchical neural network that applies PointNet recursively on a nested partitioning of the input point set. The author of this paper proposes a method of applying deep learning model directly to point cloud data, which is called pointnet.

[Paper](https://arxiv.org/abs/1612.00593): Qi, Charles R., et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
 arXiv preprint arXiv:1612.00593 (2017).

# [Model Architecture](#contents)

For each n × 3 N\times 3N × 3 point cloud input, The network first aligns it spatially through a t-net (rotate to the front), map it to the 64 dimensional space through MLP, align it, and finally map it to the 1024 dimensional space. At this time, there is a 1024 dimensional vector representation for each point, and such vector representation is obviously redundant for a 3-dimensional point cloud. Therefore, the maximum pool operation is introduced at this time to keep only the maximum on all 1024 dimensional channels The one that gets 1 × 1024 1\times 1024 × 1 The vector of 1024 is the global feature of n nn point clouds.

# [Dataset](#contents)

Dataset used: Segmentation on A subset of shapenet [shapenet](<https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip>)

- Data format：txt files
    - Note：Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```shell
# Run stand-alone training
bash scripts/run_standalone_train.sh [DATA_PATH] [CKPT_PATH]
# example:
bash scripts/run_standalone_train.sh '/home/pointnet/shapenetcore_partanno_segmentation_benchmark_v0' '../results'

# Run distributed training
bash scripts/run_distributed_train.sh [RANK_TABLE_FILE] [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# example:
bash scripts/run_standalone_train.sh hccl_8p_01234567_127.0.0.1.json modelnet40_normal_resampled save pointnet2.ckpt

# Evaluate
bash scripts/run_standalone_eval.sh [DATA_PATH] [MODEL_PATH]
# example:
bash scripts/run_standalone_eval.sh '/home/pointnet/shapenetcore_partanno_segmentation_benchmark_v0' '../results/pointnet_network_epoch_10.ckpt'
```

# [Script Description](#contents)

# [Script and Sample Code](#contents)

```bash
├── .
    ├── pointnet
        ├── ascend310_infer
        │   ├── inc
        │   │   ├── utils.h
        │   ├── src
        │   │   ├── main.cc
        │   │   ├── utils.cc
        │   ├── build.sh
        ├── scripts
        │   ├── run_distribute_train.sh  # launch distributed training with ascend platform (8p)
        │   ├── run_standalone_eval.sh   # launch evaluating with ascend platform
        │   ├── run_infer_310.sh         # run 310 infer
        │   └── run_standalone_train.sh  # launch standalone training with ascend platform (1p)
        ├── src
        │   ├── misc                     # dataset part
        │   ├── dataset.py               # data preprocessing
        │   ├── export.py                # export model
        │   ├── loss.py                  # pointnet loss
        │   ├── network.py               # network definition
        │   └── preprocess.py            # data preprocessing for training
        ├── eval.py                      # eval net
        ├── postprocess.py               # 310 postprocess
        ├── preprocess.py                # 310 preprocess
        ├── README.md
        ├── requirements.txt
        └── train.py                     # train net
```

# [Script Parameters](#contents)

```bash
Major parameters in train.py are as follows:
--batchSize        # Training batch size.
--nepoch           # Total training epochs.
--learning_rate    # Training learning rate.
--device_id        # train on which device
--data_url         # The path to the train and evaluation datasets.
--loss_per_epoch   # The times to print loss value per epoch.
--train_url        # The path to save files generated during training.
--model            # The file path to load checkpoint.
--enable_modelarts # Whether to use modelarts.
```

# [Training Process](#contents)

## Training

- running on Ascend

```shell
# Run stand-alone training
bash scripts/run_standalone_train.sh [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# example:
bash scripts/run_standalone_train.sh modelnet40_normal_resampled save pointnet2.ckpt

# Run distributed training
bash scripts/run_distributed_train.sh [RANK_TABLE_FILE] [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# example:
bash scripts/run_standalone_train.sh hccl_8p_01234567_127.0.0.1.json modelnet40_normal_resampled save pointnet2.ckpt
```

Distributed training requires the creation of an HCCL configuration file in JSON format in advance. For specific
operations, see the instructions
in [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

After training, the loss value will be achieved as follows:

```bash
# train log
Epoch : 1/25  episode : 1/40   Loss : 1.3433  Accuracy : 0.489538 step_time: 1.4269
Epoch : 1/25  episode : 2/40   Loss : 1.2932  Accuracy : 0.541544 step_time: 1.4238
Epoch : 1/25  episode : 3/40   Loss : 1.2558  Accuracy : 0.567900 step_time: 1.4397
Epoch : 1/25  episode : 4/40   Loss : 1.1843  Accuracy : 0.654681 step_time: 1.4235
Epoch : 1/25  episode : 5/40   Loss : 1.1262  Accuracy : 0.726756 step_time: 1.4206
Epoch : 1/25  episode : 6/40   Loss : 1.1000  Accuracy : 0.736225 step_time: 1.4363
Epoch : 1/25  episode : 7/40   Loss : 1.0487  Accuracy : 0.814338 step_time: 1.4457
Epoch : 1/25  episode : 8/40   Loss : 1.0271  Accuracy : 0.782350 step_time: 1.4183
Epoch : 1/25  episode : 9/40   Loss : 0.9777  Accuracy : 0.831025 step_time: 1.4289

...
```

The model checkpoint will be saved in the 'SAVE_DIR' directory.

# [Evaluation Process](#contents)

## Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

```shell
# Evaluate
bash scripts/run_eval.sh [DATA_PATH] [CKPT_NAME]
# example:
bash scripts/run_eval.sh shapenetcore_partanno_segmentation_benchmark_v0 pointnet.ckpt
```

You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

```bash
# grep "mIOU " eval.log
'mIOU for class Chair: 0.869'
```

# [310 Inference Process](#310-infer-process)

## [Export MindIR](#evaluation)

```bash
python src/export.py --model [CKPT_PATH] --file_format [FILE_FORMAT]
```

FILE_FORMAT should be one of ['AIR','MINDIR'].

The MindIR model will be exported to './mindir/pointnet.mindir'

## [310 Infer](#evaluation)

before inferring in 310, the mindir model should be exported first. Then run the code below to infer:

```bash
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [LABEL_PATH] [DVPP] [DEVICE_ID]
# example:
bash run_infer_310.sh ./mindir/pointnet.mindir ../shapenetcore_partanno_segmentation_benchmark_v0 [LABEL_PATH] N 2
```

Here, DVPP should be 'N'!

## [Result](#evaluation)

```bash
'mIOU : 0.869 '
```

# [Model Description](#contents)

## [Performance](#contents)

## Training Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | PointNet                                                  |
| Resource                   | Ascend 910; CPU 24cores; Memory 256G; OS Euler2.8           |
| uploaded Date              | 11/30/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | A subset of ShapeNet                                                  |
| Training Parameters        | epoch=25, steps=83, batch_size=64, lr=0.005             |
| Optimizer                  | Adam                                                        |
| Loss Function              | NLLLoss                                                     |
| outputs                    | probability                                                 |
| Loss                       | 0.01                                                        |
| Speed                      | 1.5 s/step (1p)                                             |
| Total time                 | 0.3 h (1p)                                                 |
| Checkpoint for Fine tuning | 17 MB (.ckpt file)                                          |

## Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | PointNet                  |
| Resource            | Ascend 910; CPU 24cores; Memory 256G; OS Euler2.8 |
| Uploaded Date       | 11/30/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       |
| Dataset             | A subset of ShapeNet                 |
| Batch_size          | 64                          |
| Outputs             | probability                 |
| mIOU                | 86.3% (1p)                  |
| Total time          | 1 min                     |

# [Description of Random Situation](#contents)

We use random seed in train.py

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
