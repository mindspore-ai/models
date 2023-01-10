# Contents

<!-- TOC -->

- [Contents](#contents)
- [MAE Description](#mae-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [MAE-Vit-B-P16 on 1.2 Million ImageNet Images](#mae-vit-b-p16-on-12-million-imagenet-images)
            - [Finetune-Vit-B-P16 on 1.2 Million ImageNet Images](#finetune-vit-b-p16-on-12-million-imagenet-images)
            - [Vit-B-P16 on 1.2 Million ImageNet Images](#vit-b-p16-on-12-million-imagenet-images)
    - [How to Use](#how-to-use)
        - [Inference](#inference)
- [Random Seed Description](#random-seed-description)
- [ModelZoo Home Page](#modelzoo-home-page)

<!-- /TOC -->

# MAE Description

Masked Autoencoders, an MAE model proposed by Kaiming He, is a MindSpore implementation that applies the self-supervised pre-training mode in the NLP field to computer vision tasks. It builds a bridge between the NLP and CV fields with high performance. MAE is a simple self-coding method that can reconstruct the original signal with given partial observation. The observed signal is mapped to the potential representation by the encoder, and then the original signal is reconstructed from the potential representation by the decoder.

This is a MindSpore/NPU re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).

# Model Architecture

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>

During pre-training, a large proportion of random image block subsets (for example, 75%) are masked. The encoder is used to process a small subset of visible patches. Then, mask tags are introduced, and the complete set of code blocks and mask tags is processed by a small decoder that will reconstruct the original image in pixels. After pre-training, the decoder is discarded and the encoder is applied to the undamaged image to generate a representation of the recognition task.

# Dataset

Used dataset: [ImageNet2012](http://www.image-net.org/)

- Dataset size: 125 GB, 1.25 million color images of 1000 classes
    - Training set: 120 GB, 1.2 million images in total
    - Test set: 5 GB, 50,000 images in total
- Data format: RGB
    - Note: Data in the pre-training phase is processed in **src/datasets/imagenet.py**, and data in the finetune phase is processed in **src/datasets/dataset.py**.

 ```bash
└─dataset
    ├─train                # Training set. The file used for training on the cloud must be in the .tar format.
    └─val                  # Evaluation dataset
 ```

# Features

## Mixed Precision

[Mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) accelerates the training process of deep neural networks by using the single-precision (FP32) data and half-precision (FP16) data without compromising the precision of networks trained with single-precision (FP32) data. It not only accelerates the computing process and reduces the memory usage, but also supports a larger model or batch size to be trained on specific hardware.
Take the FP16 operator as an example. If the input data format is FP32, MindSpore automatically reduces the precision to process data. You can open the INFO log and search for the keyword "reduce precision" to view operators with reduced precision.

# Environment Requirements

- Hardware
    - Set up the hardware environment with Ascend AI Processors, GPUs or CPUs.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the following resources:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.html)

# Quick Start

After installing MindSpore from the official website, you can perform the following steps for training and evaluation:

- Run in the Ascend AI Processor environment.

  ```shell
  # Training (For details about the CONFIG_PATH configuration file, see the related files in the './config' directory.)
  # pretrain
  python pretrain.py --config_path=[CONFIG_PATH] --use_parallel False > train.log 2>&1 &
  # finetune
  python finetune.py --config_path=[CONFIG_PATH] --use_parallel False > train.log 2>&1 &

  # Distributed training
  cd scripts;
  # pretrain
  sh pretrain_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  # finetune
  sh finetune_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # Evaluation
  cd scripts;
  bash eval_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # Inference
  None
  ```

  For distributed training, you need to create an HCCL configuration file in JSON format in advance. For details, see [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

- Perform training on ModelArts. (If you want to run the training on ModelArts, see [modelarts](https://support.huaweicloud.com/modelarts/).)

    - Train the ImageNet dataset on ModelArts using multiple devices.

      ```text
      # (1) Set "config_path='/path_to_code/config/vit-base-p16.yaml'" on the web page.
      # (2) Perform step a or b.
      #       a. Set enable_modelarts to True in the .yaml file.
      #          Set output_path in the .yaml file.
      #          Set data_path to /cache/data/ImageNet/ in the .yaml file.
      #          Set other parameters in the .yaml file.
      # (3) Upload your compressed dataset to the S3 bucket. (You can also upload the original dataset, but that may take a long time.)
      # (4) Set your code path to /path/mae on the web page.
      # (5) Set the boot file to pretrain.py or finetune.py on the web page.
      # (6) Set the training set, training output file path, and job log path on the web page.
      # (7) Create a training job.
      ```

# Script Description

## Script and Sample Code

```text
├── model_zoo
    ├── README.md                            // Description of all models
    ├── mae
        ├── README.md                        // Description of the MAE model
        ├── scripts
        │   ├──pretrain_dist.sh              // Shell script for distributed training on Ascend AI Processors
        │   ├──finetune_dist.sh              // Shell script for single-device training on Ascend AI Processors
        │   ├──eval_dist.sh                  // Shell script for evaluation on Ascend AI Processors
        ├── src
        │   ├──datasets
        │       ├──auto_augment.py           // Automatic data augmentation policy
        │       ├──dataset.py                // Dataset for finetuning
        │       ├──image_policy.py           // Automatic data augmentation policy 2
        │       ├──imagenet.py               // Dataset for pre-training
        │       ├──mixup.py                  // Custom data augmentation policy
        │   ├──loss
        │       ├──loss.py                   // soft-target loss function
        │   ├──lr
        │       ├──lr_decay.py               // Layer-wise learning rate scheduler
        │       ├──lr_generator.py           // Learning rate policy
        │   ├──models
        │       ├──eval_engine.py            // Evaluation policy
        │       ├──metric.py                 // Method for calculating the evaluation result
        │       ├──modules.py                // Basic model module
        │       ├──mae_vit.py                // MAE model structure definition
        │       ├──vit.py                    // VIT model structure definition
        │   ├──monitors
        │       ├──monitor.py                // Function for monitoring the model status
        │   ├──trainer
        │       ├──ema.py                    // EMA method
        │       ├──trainer.py                // Definition of the training process
        │   ├──model_utils                   // Cloud-based training dependency
        ├── config
        │   ├──eval.yml                      // Evaluation configuration
        │   ├──vit-base-p16.yaml             // 64P training parameter configuration
        │   ├──mae-vit-base-p16.yaml         // 64P training parameter configuration
        │   ├──finetune-vit-base-16.yaml     // 32P training parameter configuration
        ├── pretrain.py                      // Pre-training script
        ├── finetune.py                      // Fine-tune the training script.
        ├── eval.py                          // Evaluation script
        ├── requirements.txt                 // Python dependency package
```

## Script Parameters

You can configure both training and evaluation parameters in **./config/.yaml**.

- Configure the MAE and ImageNet datasets.

  ```yaml
  # Configure the MAE model.
  encoder_layers: 12
  encoder_num_heads: 12
  encoder_dim: 768
  decoder_layers: 8
  decoder_num_heads: 16
  decoder_dim: 512
  mlp_ratio: 4
  masking_ratio: 0.75
  norm_pixel_loss: True

  # Initialize the Ascend training environment.
  seed: 2022
  context:
      mode: "GRAPH_MODE" #0--Graph Mode; 1--Pynative Mode
      device_target: "Ascend"
      max_call_depth: 10000
      save_graphs: False
      device_id: 0
  use_parallel: True
  parallel:
      parallel_mode: "DATA_PARALLEL"
      gradients_mean: True

  # Training set
  data_path: "/mnt/vision/ImageNet1K/CLS-LOC"
  img_ids: "tot_ids.json" # ImageNet index of data path
  num_workers: 8
  image_size: 224

  # Training configuration
  epoch: 800
  batch_size: 64
  patch_size: 16
  sink_mode: True
  per_step_size: 0
  use_ckpt: ""

  # Loss scale
  use_dynamic_loss_scale: True # default use FixLossScaleUpdateCell

  # Optimizer configuration
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.05

  # Learning rate configuration
  base_lr: 0.00015
  start_learning_rate: 0.
  end_learning_rate: 0.
  warmup_epochs: 40

  # EMA configuration
  use_ema: False
  ema_decay: 0.9999

  # Gradient configuration
  use_global_norm: False
  clip_gn_value: 1.0

  # Callback configuration
  cb_size: 1
  save_ckpt_epochs: 1
  prefix: "MaeFintuneViT-B-P16"

  # Save directory configuration
  save_dir: "./output/"
  ```

## Training Process

### Training

- Run in the Ascend AI Processor environment.

  ```shell
  # pretrain
  python pretrain.py --config_path=[CONFIG_PATH] --use_parallel False > train.log 2>&1 &
  # finetune
  python finetune.py --config_path=[CONFIG_PATH] --use_parallel False > train.log 2>&1 &
  ```

  The preceding Python command is executed in the backend. You can view the result in the **train.log** file.
  After the training is complete, you can find the checkpoint file in the default script folder. The following methods are used to achieve the loss value:

  ```shell
  # vim pretrain log
  To be updated
  ...
  # vim finetune log
  ```

  The model checkpoint is saved in the current directory.

### Distributed Training

- Run in the Ascend AI Processor environment.

  ```shell
  # Distributed training
  cd scripts;
  # pretrain
  sh pretrain_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  # finetune
  sh finetune_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  ```

  The preceding shell script will run distributed training in the backend. You can view the result in the **train_parallel[X]/log** file. The following methods are used to achieve the loss value:

  ```shell
  # vim train_parallel0/log
  To be updated
  ```

## Evaluation Process

### Evaluation

- Evaluate the ImageNet dataset in the Ascend environment.

  Check the checkpoint path used for evaluation before running the command below. Set the checkpoint path to an absolute full path, for example, **username/vit/vit_base_patch32.ckpt**.

  ```bash
  # Evaluation
  cd scripts;
  bash eval_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  ```

  The preceding Python command is executed in the backend. You can view the result in the **eval.log** file. The accuracy of the test dataset is as follows:

  ```bash
  # grep "accuracy=" eval0/log
  accuracy=0.81
  ```

  Note: For distributed post-training evaluation, set **checkpoint_path** to the checkpoint file saved by the user, for example, **username/mae/train_parallel0/outputs/finetune-vit-base-p16-300_312.ckpt**. The accuracy of the test dataset is as follows:

  ```bash
  # grep "accuracy=" eval0/log
  accuracy=0.81
  ```

# Model Description

## Performance

### Evaluation Performance

#### MAE-Vit-B-P16 on 1.2 Million ImageNet Images

| Parameter                      | Ascend                                                      |
| -------------------------- | -----------------------------------------------------------|
| Model version                  | MAE-Vit-Base-P16                                            |
| Resources                      | Ascend 910 AI Processor, 2.60 GHz CPU with 56 cores, 314 GB memory, and EulerOS 2.8     |
| Upload date                  | 03/30/2022                                               |
| MindSpore version             | 1.6.0                                                       |
| Dataset                    | 1.2 million images                                                 |
| Training parameters                  | epoch=800, steps=349*800, batch_size=64, base_lr=0.00015 |
| Optimizer                    | Adamw                                                       |
| Loss function                  | nMSE                              |
| Output                      | Probability                                                       |
| Loss                      | 0.19                                                      |
| Speed                      | 64-device: 481 ms/step (ModelArts training data)|
| Total duration                    | 64-device: 38 hours (ModelArts training data)                                |
| Finetuned checkpoint                | 1.34 GB (.ckpt)                                        |
| Script                   | [MAE script](https://gitee.com/mindspore/models/blob/master/official/cv/MAE/pretrain.py)                                            |

#### Finetune-Vit-B-P16 on 1.2 Million ImageNet Images

| Parameter         | Ascend                                                  |
| ------------- | ------------------------------------------------------- |
| Model version     | Finetune-Vit-Base-P16                                   |
| Resources         | Ascend 910 AI Processor, 2.60 GHz CPU with 56 cores, 314 GB memory, and EulerOS 2.8|
| Upload date     | 03/30/2022                                             |
| MindSpore version| 1.6.0                                                   |
| Dataset       | 1.2 million images                                            |
| Training parameters     | epoch=100, steps=312*100, batch_size=32, base_lr=0.001  |
| Optimizer       | Adamw                                                   |
| Loss function     | SoftTargetCrossEntropy                                  |
| Output         | Probability                                                   |
| Loss         | 2.5                                                     |
| Speed         | 32-device: 332 ms/step (ModelArts training data)                  |
| Total duration       | 32-device: 6 hours (ModelArts training data)                          |
| Top 1 accuracy   | 0.807                                                   |
| Finetuned checkpoint   | 1009 MB (.ckpt)                                      |
| Script         | [Finetune script](https://gitee.com/mindspore/models/blob/master/research/cv/squeezenet/finetune.py)                                       |

#### Vit-B-P16 on 1.2 Million ImageNet Images

| Parameter         | Ascend                                                  |
| ------------- | ------------------------------------------------------- |
| Model version     | Vit-Base-P16                                            |
| Resources         | Ascend 910 AI Processor, 2.60 GHz CPU with 56 cores, 314 GB memory, and EulerOS 2.8|
| Upload date     | 03/30/2022                                              |
| MindSpore version| 1.6.0                                                   |
| Dataset       | 1.2 million images                                            |
| Training parameters     | epoch=300, steps=312*300, batch_size=64, base_lr=0.001  |
| Optimizer       | Adamw                                                   |
| Loss function     | SoftTargetCrossEntropy                                  |
| Output         | Probability                                                   |
| Loss         | 2.5                                                     |
| Speed         | 64-device: 332 ms/step (ModelArts training data)                  |
| Total duration       | 64-device: 16 hours (ModelArts training data)                         |
| Top 1 accuracy   | 0.799                                                  |
| Finetuned checkpoint   | 1009 MB (.ckpt)                                      |
| Script         | [VIT script](https://gitee.com/mindspore/models/blob/master/official/cv/MAE/eval.py)                                            |

## How to Use

### Inference

If you want to use the training model for inference on multiple hardware platforms, such as GPU, Ascend 910 AI Processor, and Ascend 310 AI Processor, click [here](https://www.mindspore.cn/tutorials/experts/en/master/infer/inference.html). The following is an example of the operation procedure:

- Run in the Ascend AI Processor environment.

  ```python
  # Read the configuration file and generate parameters required for model training based on the configuration file.
  args.loss_scale = ...
  lrs = ...
  ...
  # Set the context.
  context.set_context(mode=context.GRAPH_HOME, device_target=args.device_target)
  context.set_context(device_id=args.device_id)

  # Load an unknown dataset for inference.
  dataset = dataset.create_dataset(args.data_path, 1, False)

  # Define the model.
  net = FinetuneViT(args.vit_config)
  opt = AdamW(filter(lambda x: x.requires_grad, net.get_parameters()), lrs, args.beta1, args.beta2, loss_scale=args.loss_scale, weight_decay=cfg.weight_decay)
  loss = CrossEntropySmoothMixup(smooth_factor=args.label_smooth_factor, num_classes=args.class_num)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # Load the pre-trained model.
  param_dict = load_checkpoint(args.pretrained)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # Perform the evaluation.
  acc = model.eval(dataset)
  print("accuracy: ", acc)
  ```

# Random Seed Description

The seed in the **create_dataset** function is set in **dataset.py**, and the random seed in **train.py** is also used.

# ModelZoo Home Page

 Fore details, please go to the [official website](https://gitee.com/mindspore/models).
