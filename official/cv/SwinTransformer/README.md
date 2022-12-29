# Contents

<!-- TOC -->

- [Contents](#contents)
- [SwinTransformer Description](#swintransformer-description)
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
        - [Evaluating Performance](#evaluating-performance)
            - [SwinTransformer in ImageNet-1k](#swintransformer-in-imagenet-1k)
        - [Inference Performance](#inference-performance)
            - [SwinTransformer in ImageNet-1k](#swintransformer-in-imagenet-1k-1)
- [ModelZoo Home Page](#modelzoo-home-page)

<!-- /TOC -->

# [SwinTransformer Description](#contents)

SwinTransformer is a novel vision transformer that capably serves as a general-purpose backbone for computer vision. Challenges in adapting transformers from natural language processing to computer vision arise from differences between the two fields, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words.

To address these differences, the author proposes a hierarchical transformer whose representation is computed with shifted windowing. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size.

# [Dataset](#contents)

Used dataset: [ImageNet2012](http://www.image-net.org/)

- Dataset size: 224 x 224 colorful images of 1,000 classes
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

[Mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) accelerates the training process of deep neural networks by using the single-precision (FP32) data and half-precision (FP16) data without compromising the precision of networks trained with single-precision (FP32) data. It not only accelerates the computing process and reduces the memory usage, but also supports a larger model or batch size to be trained on specific hardware.

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
├── swin_transformer
  ├── README_CN.md                        // SwinTransformer description
  ├── ascend310_infer                     // File for inference on Ascend 310 AI Processor
  ├── scripts
      ├──run_standalone_train_ascend.sh   // Script for single-device training on Ascend 910 AI Processor
      ├──run_distribute_train_ascend.sh   // Script for multi-device training on Ascend 910 AI Processor
      ├──run_distribute_train_gpu.sh      // Script for multi-device training on GPU
      ├──run_eval_ascend.sh               // Script for testing on Ascend AI Processors
      ├──run_eval_gpu.sh                  // Script for testing on GPU
      ├──run_infer_310.sh                 // Script for inference on Ascend 310 AI Processor
  ├── src
      ├──configs                          // Configuration file of SwinTransformer
      ├──data                             // Dataset configuration file
          ├──imagenet.py                  // ImageNet configuration file
          ├──augment                      // Function file of data augmentation
          ┕──data_utils                   // Function file for dataset replication during ModelArts runtime
  │   ├──models                           // Model definition folder
          ┕──swintransformer              // SwinTransformer definition file
  │   ├──trainers                         // Custom TrainOneStep file
  │   ├──tools                            // Tools
          ├──callback.py                  // Customize the callback function and test it after training.
          ├──cell.py                      // Common tools related to cell
          ├──criterion.py                 // Tools related to the loss function
          ├──get_misc.py                  // Other tools
          ├──optimizer.py                 // Functions related to the optimizer and parameters
          ┕──schedulers.py                // Tools for learning rate decay
  ├── train.py                            // Training file
  ├── eval.py                             // Evaluation file
  ├── export.py                           // File for exporting a model
  ├── postprocess.py                      // File for obtaining accuracies after inference
  ├── preprocess.py                       // Preprocessed image file for inference

```

## Script Parameters

You can configure both training and evaluation parameters in **config.py**.

- Configure dataset for SwinTransformer and ImageNet-1k.

  ```python
    # Architecture
    arch: swin_tiny_patch4_window7_224 # SwinTransformer structure selection
    # ===== Dataset ===== #
    data_url: ./data                    # Dataset address
    set: ImageNet                       # Dataset name
    num_classes: 1000                   # Number of dataset classes
    mix_up: 0.8                         # MixUp data augmentation parameter
    cutmix: 1.0                         # CutMix data augmentation parameter
    auto_augment: rand-m9-mstd0.5-inc1 # Automatic augmentation parameter
    interpolation: bicubic              # Interpolation method for image scaling
    re_prob: 0.25                       # Data augmentation parameter
    re_mode: pixel                      # Data augmentation parameter
    re_count: 1                         # Data augmentation parameter
    mixup_prob: 1.                      # Data augmentation parameter
    switch_prob: 0.5                    # Data augmentation parameter
    mixup_mode: batch                   # Data augmentation parameter
    # ===== Learning Rate Policy ======== #
    optimizer: adamw                    # Optimizer type
    base_lr: 0.0005                     # Basic learning rate
    warmup_lr: 0.00000007               # Initial warmup learning rate
    min_lr: 0.000006                    # Minimum learning rate
    lr_scheduler: cosine_lr             # Learning rate decay policy
    warmup_length: 20                   # Warmup length
    nonlinearity: GELU                  # Activation function type
    # ===== Network training config ===== #
    amp_level: O2                       # Mixed precision policy
    beta: [ 0.9, 0.999 ]                # AdamW parameter
    clip_global_norm_value: 5.          # Global gradient norm clipping threshold
    is_dynamic_loss_scale: True         # Specifies whether to use dynamic scaling.
    epochs: 300                         # Number of training epochs
    label_smoothing: 0.1                # Label smoothing parameter
    weight_decay: 0.05                  # Weight decay parameter
    momentum: 0.9                       # Optimizer momentum
    batch_size: 128                     # Batch size
    # ===== Hardware setup ===== #
    num_parallel_workers: 16            # Number of threads for data processing
    # ===== Model config ===== #        # Model structure parameter
    drop_path_rate: 0.2
    embed_dim: 96
    depth: [ 2, 2, 6, 2 ]
    num_heads: [ 3, 6, 12, 24 ]
    window_size: 7
    image_size: 224                     # Image size
  ```

For details about configuration, see the `config.py`. After installing MindSpore from the official website, you can perform the following steps for training and evaluation:

# [Training and Testing](#contents)

- Running in the Ascend AI Processor Environment

  ```bash
  # Run the Python command to start single-device training.
  python train.py --device_id 0 --device_target Ascend --swin_config ./src/configs/swin_tiny_patch4_window7_224.yaml > train.log 2>&1 &

  # Run the script to start single-device training.
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]

  # Run the script to start multi-device training.
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # Run the Python command to start single-device evaluation.
  python eval.py --device_id 0 --device_target Ascend --swin_config ./src/configs/swin_tiny_patch4_window7_224.yaml --pretrained ./ckpt_0/swin_tiny_patch4_window7_224.ckpt > ./eval.log 2>&1 &

  # Run the script to start single-device evaluation.
  bash ./scripts/run_eval_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # Inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME(imagenet2012)] [DATASET_PATH] [DEVICE_ID(optional)]
  ```

  For distributed training, you need to create an HCCL configuration file in JSON format in advance.

  For details, please see [HCCL Tool](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

- Running in the GPU Environment

  ```bash
  # Run the script to start multi-device training.
  bash ./scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)]

  # Run the script to start single-device training.
  bash ./scripts/run_standalone_train_gpu.sh [CONFIG_PATH] [DEVICE_ID]

  # Run the script to start single-device evaluation.
  bash ./scripts/run_eval_gpu.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]
  ```

## Export Process

### Export

  ```shell
  python export.py --pretrained [CKPT_FILE] --swin_config [CONFIG_PATH] --device_target [DEVICE_TARGET]
  ```

The exported model will be named by model structure and saved in the current directory.

## Inference Process

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md).**

### Inference

Export the model before inference. MindIR models can be exported in any environment, but AIR models can be exported only in the Ascend 910 AI Processor environment. The following shows how to infer the MindIR model.

- Use the ImageNet-1k dataset for inference on Ascend 310 AI Processors.

  The inference results are saved in the **scripts** directory. You can find results similar to the following in the **acc.log** file:

  ```shell
  # Inference on Ascend 310 AI Processor
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [DEVICE_ID]
  Total data: 50000, top1 accuracy: 81.02%
  ```

# [Model Description](#contents)

## Performance

### Evaluating Performance

#### SwinTransformer in ImageNet-1k

| Parameter                | Ascend                                                      | GPU                                                    |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
|Model|SwinTransformerSwinTransformer| SwinTransformerSwinTransformer       |
| Model version             | swin_tiny_patch4_window7_224                                                | swin_tiny_patch4_window7_224                    |
| Resources                  | Ascend 910               | Gefore RTX 3090 * 8 |
| Upload date             | 2021-10-25                                 | 2022-5-28                        |
| MindSpore version         | 1.3.0                                                 | 1.6.1                                |
| Dataset                   | ImageNet-1k Train (1,281,167 images)                                             | ImageNet-1k Train (1,281,167 images)              |
| Training parameters       | epoch=300, batch_size=128            | epoch=300, batch_size=128 |
| Optimizer                 | AdamWeightDecay                                                    | AdamWeightDecay                                     |
| Loss function             | SoftTargetCrossEntropy                                       | SoftTargetCrossEntropy                 |
| Loss| 0.8279| |
| Output                   | Probability                                                | Probability                                              |
| Accuracy            | Eight devices: 81.07% (top 1) 95.31% (top 5)                  | Eight devices: 80.65% (top 1) 95.38% (top 5)|
| Speed                     | Eight devices: 624.124 ms/step                       | Eight devices: 4323 ms/step         |
| Training duration         |79 h 55 min 08s (run on ModelArts)|

### Inference Performance

#### SwinTransformer in ImageNet-1k

| Parameter                | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
|Model                |SwinTransformer|
| Model version             | swin_tiny_patch4_window7_224|                                                |
| Resources                  | Ascend 310 AI Processor              |
| Upload date             | 2021-10-25                                 |
| MindSpore version         | 1.3.0                                                 |
| Dataset                   | ImageNet-1k Val (50,000 images)                                                |
| Accuracy            | Top 1: 81.02%, top 5: 95.38%                     |
| Speed                     | The average time is 60.97 ms/image|
| Inference duration| About 51 minutes|

# ModelZoo Home Page

For details, please go to the [official website](https://gitee.com/mindspore/models).
