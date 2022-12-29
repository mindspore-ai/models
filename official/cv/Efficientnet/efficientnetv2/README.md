# Contents

<!-- TOC -->

- [Contents](#contents)
- [EfficientNetV2 Description](#efficientnetv2-description)
- [Dataset](#dataset)
- [Feature](#feature)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
- [Training and Testing](#training-and-testing)
    - [Export Process](#export-process)
        - [Export](#export)
    - [Inference Process](#inference-process)
        - [Inference](#inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [EfficientNetV2 on ImageNet-1k](#efficientnetv2-on-imagenet-1k)
        - [Inference Performance](#inference-performance)
            - [EfficientNetV2 on ImageNet-1k](#efficientnetv2-on-imagenet-1k-1)
- [ModelZoo Home Page](#modelzoo-home-page)

<!-- /TOC -->

# [EfficientNetV2 Description](#contents)

This document is an upgrade of EfficientNet by Google scholars MingxingTan and Quoc V. Le, aiming to improve the training speed while maintaining efficient use of parameters. Based on EfficientNet, Fused-MBConv is added into the search space, and an adaptive regular intensity adjustment mechanism is added for progressive learning. Based on the two improvements, EfficientNetV2 achieves SOTA performance on multiple benchmark datasets and trains faster. For example, EfficientNetV2's top 1 accuracy achieves 87.3% and the training speed is 5 to 11 times faster.

# [Dataset](#contents)

Used dataset: [ImageNet2012](http://www.image-net.org/)

- Dataset size: 224 x 224 colorful images of 1000 classes
    - Training set: 1,281,167 images
    - Test set: 50,000 images
- Data format: JPEG
    - Note: Data is processed in **dataset.py**.
- Download the dataset. The directory structure is as follows:

 ```text
└─dataset
    ├─train                 # Training set
    └─val                   # Validation set
```

# [Feature](#contents)

## Mixed Precision

[Mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html)
accelerates the training process of deep neural networks by using the single-precision (FP32) data and half-precision (FP16) data without compromising the precision of networks trained with single-precision (FP32) data. It not only accelerates the computing process and reduces the memory usage, but also supports a larger model or batch size to be trained on specific hardware.

# [Environment Requirements](#contents)

- Hardware
    - Set up the hardware environment with Ascend AI Processors.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the following resources:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#contents)

## Script and Sample Code

```bash
├── EfficientNetV2
  ├── README_CN.md                        // EfficientNetV2 description
  ├── ascend310_infer                     // File for inference on Ascend 310 AI Processor
  ├── scripts
      ├──run_standalone_train_ascend.sh   // Single-device Ascend 910 training script
      ├──run_distribute_train_ascend.sh   // Multi-device Ascend 910 training script
      ├──run_eval_ascend.sh               // Test script
      ├──run_infer_onnx.sh                // ONNX inference script
      ├──run_infer_310.sh                 // Script for inference on Ascend 310 AI Processor
  ├── src
      ├──configs                          // Configuration file of EfficientNetV2
      ├──data                             // Dataset configuration file
          ├──imagenet_finetune.py         // Configuration file of ImageNet
          ┕──data_utils                   // Data replication function at ModelArts runtime
  │ ├ ─ ─models                           // Definition file of EfficientNetV2
  │   ├──trainers                         // Custom TrainOneStep file
  │   ├──weight_conver                    // Checkpoint used to convert the TensorFlow weight into MindSpore
  │   ├──tools                            // Tool folder
          ├──callback.py                  // Custom callback function used for testing after training
          ├──cell.py                      // Cell-related tool functions
          ├──criterion.py                 // Loss function-related tool functions
          ├──get_misc.py                  // Other tool functions
          ├──optimizer.py                 // Functions related to optimizers and parameters
          ┕──schedulers.py                // Tool functions related to learning rate decay
  ├── train.py                            // Training file
  ├── eval.py                             // Evaluation file
  ├── export.py                           // File for exporting a model
  ├── infer_onnx.py                       // ONNX inference file
  ├── postprocess.py                      // File for obtaining accuracies after inference
  ├── preprocess.py                       // Preprocess images for inference.

```

## Script Parameters

You can configure both training and evaluation parameters in **config.py**.

- Configure EfficientNetV2 and ImageNet-1k datasets.

  ```python
    # Architecture
    arch: effnetv2_s                             # Model architecture
    # ===== Dataset ===== #
    data_url: ./imagenet                         # Dataset address
    set: ImageNetFinetune                        # Dataset type
    num_classes: 1000                           # Number of classes in the dataset
    interpolation: bilinear                      # Interpolation method
    # ===== Learning Rate Policy ======== #
    eps: 0.001                                   # Epsilon
    optimizer: rmsprop                           # Optimizer type
    base_lr: 0.0005                              # Base learning rate
    warmup_lr: 0                                # Warm-up learning rate
    min_lr: 0.                                   # Minimum learning rate
    lr_scheduler: constant_lr                    # Learning rate policy
    warmup_length: 1                             # Warm-up length of the learning rate.
    # ===== Network training config ===== #
    amp_level: O0                                # Mixed precision type
    clip_global_norm: True                       # Specifies whether to use global gradient clipping.
    clip_global_norm_value: 5                    # Global gradient clipping norm
    is_dynamic_loss_scale: True                 # Specifies whether to use dynamic loss scaling.
    epochs: 15                                   # Number of training epochs
    label_smoothing: 0.1                         # Label smoothing coefficient
    weight_decay: 0.00001                        # L2 weight decay coefficient
    decay: 0.9                                   # RMSProp decay coefficient
    momentum: 0.9                                # Momentum coefficient
    batch_size: 32                               # Batch size
    # ===== Hardware setup ===== #
    num_parallel_workers: 16                     # Number of data preprocessing threads
    device_target: Ascend                        # Device type
    # ===== Model config ===== #
    drop_path_rate: 0.2                          # Drop path rate
    drop_out_rate: 0.000001                      # Drop out rate
    image_size: 384                              # Image size
    pretrained: ./efficientnets_imagenet22k.ckpt # Pre-trained weight address
  ```

After installing MindSpore from the official website, you can perform the following steps for training and evaluation:

# [Training and Testing](#contents)

- Prepare for the running.

  ```bash
  # Download the ImageNet22k pre-trained weight at https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-s-21k.tgz.

  # Decompress the package and run the following command.
  python weight_convert.py --pretrained ./efficientnetv2-s-21k/model

  # After the weight is converted, configure the weight address in pretrained of efficientv2_s_finetune.
  ```

- On Ascend AI Processors

  ```bash
  # Run the Python command to start single-device training.
  python train.py --device_id 0 --device_target Ascend --config ./src/configs/effnetv2_s_finetune.yaml \
  > train.log 2>&1 &

  # Run the script to start single-device training.
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]

  # Run the script to start multi-device training.
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # Run the Python command to start single-device evaluation.
  python eval.py --device_id 0 --device_target Ascend --config ./src/configs/effnetv2_s_finetune.yaml \
  --pretrained ./ckpt_0/effnetv2_s.ckpt > ./eval.log 2>&1 &

  # Run the script to start single-device evaluation.
  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]

  # Inference example
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME(imagenet2012)] [DATASET_PATH] [DEVICE_ID(optional)]
  ```

  For distributed training, you need to create an HCCL configuration file in JSON format in advance.

  Follow the instructions in the following link:

[HCCL Tool](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

## Export Process

### Export

  ```shell
  python export.py --pretrained [CKPT_FILE] --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
  ```

`EXPORT_FORMAT`: ["AIR", "MINDIR", "ONNX"]
The exported model will be named by model structure and saved in the current directory.

## Inference Process

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

### Inference

Export the model before inference. MindIR models can be exported in any environment. AIR models can be exported only on Ascend 910 AI Processors. ONNX models can be exported in the CPU/GPU environment. The following shows an example of using the MindIR model to perform inference.

- Use the ImageNet-1k dataset for inference on Ascend 310 AI Processor.

  The inference results are saved in the **scripts** directory. You can find results similar to the following in the **acc.log** file:

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [DEVICE_ID]
  Top1 acc:  0.838
  Top5 acc:  0.96956
  ```

- Use the ImageNet-1k dataset for inference on a GPU or CPU.

  The inference results are saved in the home directory. You can find them in the **infer_onnx.log** file:

  ```shell
  bash run_infer_onnx.sh [ONNX_PATH] [CONFIG] [DEVICE_TARGET]
  ```

# [Model Description](#contents)

## Performance

### Evaluation Performance

#### EfficientNetV2 on ImageNet-1k

| Parameter                | Ascend                           |
| -------------------------- | ----------------------- |
|Model|EfficientNetV2|
| Model version             | EfficientNetV2-S     |
| Resources                  | Ascend 910               |
| Upload date             | 2021-12-19              |
| MindSpore version         | 1.3.0     |
| Dataset                   | ImageNet-1k Train (1,281,167 images)       |
| Training parameters       | epoch=15, batch_size=32 (16 devices)  |
| Optimizer                 | RMSProp         |
| Loss function             | CrossEntropySmooth   |
| Loss|0.687|
| Output                   | Probability               |
| Classification accuracies            | 16-device: top1: 83.778%; top5: 96.956%                  |
| Speed                     | 16-device: 582.105 ms/step                       |
| Training duration         |7 hours 25 minutes 15 seconds (run on ModelArts)|

### Inference Performance

#### EfficientNetV2 on ImageNet-1k

| Parameter                | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
|Model                |EfficientNetV2|
| Model version             | EfficientNetV2-S|                                                |
| Resources                  | Ascend 310               |
| Upload date             | 2021-12-19                                 |
| MindSpore version         | 1.3.0                                                 |
| Dataset                   | ImageNet-1k Val (50,000 images)                                                |
| Classification accuracies            | Top 1:83.8%; top 5: 96.956%                     |
| Speed                     | Average: 11.4918 ms/image|
| Inference duration| About 22 minutes|

# ModelZoo Home Page

For details, please go to the [official website](https://gitee.com/mindspore/models).
