# Contents

- [Contents](#contents)
- [MobileNetV3 Description](#mobilenetv3-description)
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
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# [MobileNetV3 Description](#contents)

MobileNetV3 combines hardware-aware neural network architecture search (NAS) and NetAdapt algorithm, which can already be transplanted to the mobile phone CPU to run, and will be further optimized and improved with the new architecture (November 20, 2019).

[Paper](https://arxiv.org/pdf/1905.02244)：Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al."Searching for mobilenetv3."In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324.2019.

# [Model Architecture](#contents)

The overall network architecture of MobileNetV3 is as follows:

[Link](https://arxiv.org/pdf/1905.02244)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: 146G, 1330k pictures, 1000 categories of color images
    - Train: 140G, 1280k pictures
    - Test: 6G, 50k pictures
- Data format: RGB
    - Note: The data is processed in src/dataset.py.

# [Environment Requirements](#contents)

- Hardware (Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend (examples)

```shell
# enter script dir, train MobileNet
bash run_standalone_train_ascend.sh /data/ImageNet/train 0
# enter script dir, evaluate MobileNet
bash run_eval_ascend.sh 0 /data/ImageNet/val train/checkpoint/model_1/mobilenetV3-35_2135.ckpt
# enter script dir, train MobileNet on 8 divices
bash run_distribute_train_ascend.sh ./hccl_8p_01234567_127.0.0.1.json /data/ImageNet/train

# Where:
#   - /data/ImageNet/train - path to the ImageNet dataset train folder
#   - /data/ImageNet/val - path to the ImageNet dataset val folder
#   - train/checkpoint/model_1/mobilenetV3-35_2135.ckpt - path to the saved checkpoint
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link: https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.

- Running on GPU (examples)

```shell
# enter script dir, train MobileNet
bash run_standalone_train_gpu.sh /data/ImageNet/train 0  # 0 - device_id of the GPU on which the training will be performed
# enter script dir, evaluate MobileNet
bash run_eval_gpu.sh /data/ImageNet/val train/checkpoint/model_1/mobilenetV3-35_2135.ckpt
# enter script dir, train MobileNet on 8 GPUs
bash run_distribute_train_gpu.sh /data/ImageNet/train 8  # 8 - the number of available GPUs

# Where:
#   - /data/ImageNet/train - path to the ImageNet dataset train folder
#   - /data/ImageNet/val - path to the ImageNet dataset val folder
#   - train/checkpoint/model_1/mobilenetV3-35_2135.ckpt - path to the saved checkpoint
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
├── mobilenetV3_small_x1_0
  ├── README_CN.md                          # MobileNetV3 description, CN version
  ├── README.md                             # MobileNetV3 description, EN version
  ├── scripts
  │   ├── run_distribute_train_ascend.sh    # launch Ascend distributed training (8 pcs)
  │   ├── run_distribute_train_gpu.sh       # launch GPU distributed training
  │   ├── run_eval_ascend.sh                # launch Ascend evaluation
  │   ├── run_eval_gpu.sh                   # launch GPU evaluation
  │   ├── run_standalone_train_ascend.sh    # launch Ascend standalone training (1 pcs)
  │   └── run_standalone_train_gpu.sh       # launch GPU standalone training (1 pcs)
  ├── src
  │   ├── config.py                         # parameters configuration
  │   ├── dataset.py                        # creating a dataset
  │   ├── loss.py                           # loss function
  │   ├── lr_generator.py                   # configure the learning rate
  │   ├── mobilenetV3.py                    # MobileNetV3 architecture
  │   └── monitor.py                        # monitoring network losses and other data
  ├── argparser.py                          # command line arguments parsing
  ├── eval.py                               # evaluation script
  ├── export.py                             # model format conversion script
  └── train.py                              # training script
```

## [Script Parameters](#contents)

Major parameters in train.py and config.py as follows:

```python
{
    'num_classes': 1000,                    # number of classes
    'image_height': 224,                    # input image height
    'image_width': 224,                     # input image width
    'batch_size': 256,                      # batch size
    'epoch_size': 370,                      # number of iterations (epochs)
    'warmup_epochs': 4,                     # warmup epoch number
    'lr': 0.05,                             # learning rate
    'momentum': 0.9,                        # momentum parameter
    'weight_decay': 4e-5,                   # weights decay rate
    'label_smooth': 0.1,                    # label smoothing
    'loss_scale': 1024,                     # loss scale
    'save_checkpoint': True,                # whether to save the ckpt
    'save_checkpoint_epochs': 1,            # save a ckpt file for the corresponding number of epochs
    'keep_checkpoint_max': 5,               # maximum number of saved checkpoints
    'save_checkpoint_path': "./checkpoint", # ckpt save path
    'export_file': "mobilenetv3_small",     # export file name
    'export_format': "MINDIR"               # export format
}
```

## [Training Process](#contents)

### [Training](#contents)

You can use python or shell scripts for training.

- Running on Ascend

```shell
# Training example
  # python:
    # Ascend single card training example:
      python train.py --device_id [DEVICE_ID] --dataset_path [DATASET_PATH] --device_target="Ascend" &> log &

  # shell:
    # Ascend single card training example:
      bash ./scripts/run_standalone_train_ascend.sh [DATASET_PATH] [DEVICE_ID]
    # Ascend eight-card parallel training:
      cd ./scripts/
      bash ./run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH]
```

- Running on GPU

```shell
# Training example
  # python:
    # single GPU training example:
      python train.py --device_id [DEVICE_ID] --dataset_path [DATASET_PATH] --device_target="GPU" &> log &

  # shell:
    # single GPU training example:
      cd ./scripts/
      bash ./scripts/run_standalone_train_gpu.sh [DATASET_PATH] [DEVICE_ID]
    # multiple GPU parallel training (DEVICE_NUM - number of available GPUs):
      cd ./scripts/
      bash ./run_distribute_train_gpu.sh [DATASET_PATH] [DEVICE_NUM]
```

Checkpoint file is stored in `./checkpoint` path, the training will be recorded to `log` file.

An example of a training log looks like this:

```shell
epoch 1: epoch time: 553262.126, per step time: 518.521, avg loss: 5.270
epoch 2: epoch time: 151033.049, per step time: 141.549, avg loss: 4.529
epoch 3: epoch time: 150605.300, per step time: 141.148, avg loss: 4.101
epoch 4: epoch time: 150638.805, per step time: 141.180, avg loss: 4.014
epoch 5: epoch time: 150594.088, per step time: 141.138, avg loss: 3.607
```

## [Evaluation Process](#contents)

### [Evaluation](#contents)

You can use python or shell script for evaluation.

Before running the command below, please check the checkpoint path used for evaluation.

- Running on Ascend

```shell
# Evaluation example
  # python:
      python eval.py --device_id [DEVICE_ID] --dataset_path [DATASET_PATH] --checkpoint_path [PATH_CHECKPOINT] --device_target="Ascend" &> eval.log &

  # shell:
      bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [DATASET_PATH] [PATH_CHECKPOINT]
```

- Running on GPU

```shell
# Evaluation example
  # python:
      python eval.py --device_id [DEVICE_ID] --dataset_path [DATASET_PATH] --checkpoint_path [PATH_CHECKPOINT] --device_target="GPU" &> eval.log &

  # shell:
      bash ./scripts/run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
```

> The ckpt file can be generated during the training process.

You can view the results through the file "eval.log".

```shell
result: {'Loss': 2.3101649037352554, 'Top_1_Acc': 0.6746546546546547, 'Top_5_Acc': 0.8722122122122122} ckpt= ./checkpoint/model_0/mobilenetV3-370_625.ckpt
```

# [Model Description](#contents)

## [Performance](#contents)

### [Evaluation Performance](#contents)

| Parameter                   | Ascend (8 pcs)                        | GPU Tesla V100 (1 pcs)                                          | GPU Tesla V100 (6 pcs) |
| --------------------------- | ------------------------------------- |-----------------------------------------------------------------|-----------------------------------------------------------------------|
| Model Version               | mobilenetV3 small                     | mobilenetV3 small                                               | mobilenetV3 small                                                     |
| Resource                    | HUAWEI CLOUD Modelarts                | Ubuntu 18.04.6, Tesla V100, CPU 2.70GHz, 32 cores, RAM 258 GB   | Ubuntu 18.04.6, Tesla V100 (6 pcs), CPU 2.70GHz, 32 cores, RAM 258 GB |
| Uploaded Date               | 03/25/2021 (month/day/year)           | 10/29/2021 (month/day/year)                                     | 10/29/2021 (month/day/year)                                           |
| MindSpore Version           | 1.3.0                                 | 1.5.0                                                           | 1.5.0                                                                 |
| Dataset                     | imagenet                              | imagenet                                                        | imagenet                                                              |
| Training Parameters         | src/config.py                         | epoch_size=370, batch_size=600, **lr=0.005**, other configs: src/config.py | epoch_size=370, batch_size=600, **lr=0.05**, other configs: src/config.py |
| Optimizer                   | RMSProp                               | RMSProp                                                         | RMSProp                            |
| Loss Function               | CrossEntropyWithLabelSmooth           | CrossEntropyWithLabelSmooth                                     | CrossEntropyWithLabelSmooth        |
| Outputs                     | Accuracy: Top1[67.5%], Top5[87.2%]    | Accuracy: Top1[67.3%], Top5[87.1%]                              | Accuracy: Top1[67.3%], Top5[87.1%] |
| Loss                        | 2.31                                  | 2.32                                                            | 2.32                               |
| Speed                       |                                       | 430 ms/step                                                     | 3500 ms/step                       |
| Total time                  | 16.4h (8 pcs)                         | 80h (1 pcs)                                                     | 117h (6 pcs)                       |
| Checkpoint size             | 36 M                                  | 36 M                                                            | 36 M                               |
| Scripts                     | [Link](https://gitee.com/mindspore/models/tree/master/research/cv/mobilenetV3_small_x1_0) | [Link](https://gitee.com/mindspore/models/tree/master/research/cv/mobilenetV3_small_x1_0) | [Link](https://gitee.com/mindspore/models/tree/master/research/cv/mobilenetV3_small_x1_0) |

# [Description of Random Situation](#contents)

Random seed is set in `dataset.py` and `train.py` scripts.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).