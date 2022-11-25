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
        - [ONNX Evaluation](#evaluation)
    - [310 Inference Process](#310-infer-process)
        - [Export MindIR](#evaluation)
        - [Export ONNX](#evaluation)
        - [310 Infer](#evaluation)
        - [Result](#evaluation)
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
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

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
        │   ├── run_distribute_ascend.sh        # launch distributed training with ascend platform (8p)
        │   ├── run_distribute_gpu.sh           # launch distributed training with gpu platform (8p)
        │   ├── run_standalone_eval_ascend.sh   # launch evaluating with ascend platform (1p)
        │   ├── run_standalone_eval_gpu.sh      # launch evaluating with gpu platform (1p)
        │   ├── run_standalone_eval_onnx_gpu.sh # launch evaluating onnx with gpu platform (1p)
        │   ├── run_infer_310.sh                # run 310 infer
        │   ├── run_standalone_train_ascend.sh  # launch standalone training with ascend platform (1p)
        │   └── run_standalone_train_gpu.sh     # launch standalone training with gpu platform (1p)
        ├── src
        │   ├── misc                     # dataset part
        │   ├── dataset.py               # data preprocessing
        │   ├── export.py                # export model
        │   ├── loss.py                  # pointnet loss
        │   ├── network.py               # network definition
        │   └── preprocess.py            # data preprocessing for training
        ├── eval.py                      # eval net
        ├── eval_onnx.py                 # eval onnx
        ├── postprocess.py               # 310 postprocess
        ├── preprocess.py                # 310 preprocess
        ├── README.md
        ├── requirements.txt
        └── train.py                     # train net
```

# [Script Parameters](#contents)

```bash
Major parameters in train.py are as follows:
--device_id        # train on which device
--batchSize        # Training batch size.
--nepoch           # Total training epochs.
--learning_rate    # Training learning rate.
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
    # Run stand-alone training for Ascend
    bash scripts/run_standalone_train_ascend.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID]
    # example:
    bash scripts/run_standalone_train_ascend.sh ../shapenetcore_partanno_segmentation_benchmark_v0 ./ckpts 1



    # Run distributed training for Ascend
    bash scripts/run_distribution_ascend.sh [RANK_TABLE_FILE] [CKPTS_DIR] [DATA_PATH]
    # example:
    bash scripts/run_distribution_ascend.sh [RANK_TABLE_FILE] ./ckpts ../shapenetcore_partanno_segmentation_benchmark_v0


    ```

- running on GPU

    ```shell
    # Run stand-alone training for GPU
    bash scripts/run_standalone_train_gpu.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID]
    # example:
    bash scripts/run_standalone_train_gpu.sh ../shapenetcore_partanno_segmentation_benchmark_v0 ./ckpts 1
    # Run distributed training for GPU
    bash scripts/run_distribute_gpu.sh [DATA_PATH] [CKPT_PATH]
    # example:
    bash scripts/run_distribute_gpu.sh ./ckpts ../shapenetcore_partanno_segmentation_benchmark_v0
    ```

Distributed training requires the creation of an HCCL configuration file in JSON format in advance. For specific
operations, see the instructions
in [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

After training, the loss value will be achieved as follows:

```bash
# train log
Epoch : 1/50  episode : 1/40   Loss : 1.3433  Accuracy : 0.489538 step_time: 1.4269
Epoch : 1/50  episode : 2/40   Loss : 1.2932  Accuracy : 0.541544 step_time: 1.4238
Epoch : 1/50  episode : 3/40   Loss : 1.2558  Accuracy : 0.567900 step_time: 1.4397
Epoch : 1/50  episode : 4/40   Loss : 1.1843  Accuracy : 0.654681 step_time: 1.4235
Epoch : 1/50  episode : 5/40   Loss : 1.1262  Accuracy : 0.726756 step_time: 1.4206
Epoch : 1/50  episode : 6/40   Loss : 1.1000  Accuracy : 0.736225 step_time: 1.4363
Epoch : 1/50  episode : 7/40   Loss : 1.0487  Accuracy : 0.814338 step_time: 1.4457
Epoch : 1/50  episode : 8/40   Loss : 1.0271  Accuracy : 0.782350 step_time: 1.4183
Epoch : 1/50  episode : 9/40   Loss : 0.9777  Accuracy : 0.831025 step_time: 1.4289

...
```

The model checkpoint will be saved in the 'SAVE_DIR' directory.

# [Evaluation Process](#contents)

## Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

    ```shell
    # Evaluate on ascend
    bash scripts/run_standalone_eval_ascend.sh [DATA_PATH] [MODEL_PATH] [DEVICE_ID]
    # example:
    bash scripts/run_standalone_eval_ascend.sh shapenetcore_partanno_segmentation_benchmark_v0 pointnet.ckpt 1
    ```

    You can view the results through the file "log_standalone_eval_ascend". The accuracy of the test dataset will be as follows:

    ```bash
    # grep "mIOU " log_standalone_eval_ascend
    'mIOU for class Chair: 0.869'
    ```

- running on GPU

    ```shell
    # Evaluate on GPU
    bash scripts/run_standalone_eval_gpu.sh [DATA_PATH] [MODEL_PATH] [DEVICE_ID]
    # example:
    bash scripts/run_standalone_eval_gpu.sh shapenetcore_partanno_segmentation_benchmark_v0 pointnet.ckpt 1
    ```

    You can view the results through the file "log_standalone_eval_gpu". The accuracy of the test dataset will be as follows:

    ```bash
    # grep "mIOU " log_standalone_eval_gpu
    'mIOU for class Chair: 0.869'
    ```

## ONNX Evaluation

- running on GPU

  ```shell
  # Evaluate on GPU
  bash scripts/run_standalone_eval_onnx_gpu.sh [DATA_PATH] [MODEL_PATH] [DEVICE_ID]
  # example
  bash scripts/run_standalone_eval_onnx_gpu.sh dataset/shapenetcore_partanno_segmentation_benchmark_v0 mindir/pointnet.onnx 1
  ```

  You can view the results through the file "log_standalone_eval_onnx_gpu". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "mIOU " log_standalone_eval_onnx_gpu
  'mIOU for class Chair: 0.869'
  ```

# [310 Inference Process](#310-infer-process)

## [Export MindIR/ONNX](#evaluation)

```bash
python src/export.py --model [CKPT_PATH] --file_format [FILE_FORMAT]
```

FILE_FORMAT should be one of ['AIR','MINDIR','ONNX'].

The MindIR/ONNX model will be exported to './mindir/pointnet.mindir'/'./mindir/pointnet.onnx'

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

| Parameters                 | Ascend                                            | GPU(V100(PCIE))                             |
| -------------------------- | ------------------------------------------------- | ------------------------------------------- |
| Model Version              | PointNet                                          | PointNet                                    |
| Resource                   | Ascend 910; CPU 24cores; Memory 256G; OS Euler2.8 | NVIDIA RTX Titan-24G                        |
| uploaded Date              | 11/30/2021 (month/day/year)                       | 4/19/2022 (month/day/year)                  |
| MindSpore Version          | 1.3.0                                             | 1.3.0 1.5.0 1.6.0                           |
| Dataset                    | A subset of ShapeNet                              | A subset of ShapeNet                        |
| Training Parameters        | epoch=50, steps=83, batch_size=64, lr=0.005       | epoch=50, steps=83, batch_size=64, lr=0.005 |
| Optimizer                  | Adam                                              | Adam                                        |
| Loss Function              | NLLLoss                                           | NLLLoss                                     |
| outputs                    | probability                                       | probability                                 |
| Loss                       | 0.01                                              | 0.01                                        |
| Speed                      | 1.5 s/step (1p)                                   | 0.19 s/step (1p)                            |
| Total time                 | 0.3 h (1p)                                        | 10 m (1p)                                   |
| Checkpoint for Fine tuning | 17 MB (.ckpt file)                                | 17 MB (.ckpt file)                          |

## Inference Performance

| Parameters        | Ascend                                            | GPU(V100(PCIE))            |
| ----------------- | ------------------------------------------------- | -------------------------- |
| Model Version     | PointNet                                          | PointNet                   |
| Resource          | Ascend 910; CPU 24cores; Memory 256G; OS Euler2.8 | NVIDIA RTX Titan-24G       |
| Uploaded Date     | 11/30/2021 (month/day/year)                       | 4/19/2022 (month/day/year) |
| MindSpore Version | 1.3.0                                             | 1.3.0 1.5.0 1.6.0          |
| Dataset           | A subset of ShapeNet                              | A subset of ShapeNet       |
| Batch_size        | 64                                                | 64                         |
| Outputs           | probability                                       | probability                |
| mIOU              | 86.3% (1p)                                        | 86.3% (1p)                 |
| Total time        | 1 min                                             | 1 min                      |

# [Description of Random Situation](#contents)

We use random seed in train.py

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).