# Contents

- [Contents](#contents)
- [DDRNet Description](#ddrnet-description)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
- [Training and Evaluation Process](#training-and-evaluation-process)
    - [Running on Ascend](#running-on-ascend)
    - [Running on GPU](#running-on-gpu)
- [Inference Process](#inference-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DDRNet Description](#contents)

## Description

Semantic segmentation is a key technique for autonomous vehicles to understand the surrounding scene.
For actual autonomous vehicles, it is not desirable to spend a lot of reasoning time to obtain high-precision segmentation results.
Using a lightweight architecture (encoder decoder or dual channel) or inference on low-resolution images, recent methods have enabled very
fast scene resolution, even running at over 100 FPS on a single 1080Ti GPU. However, there is still a significant performance
gap between these real-time approaches and models based on the bulking backbone.

To solve this problem, inspired by HRNet, the authors proposed a deep double-resolution network
with deep high-resolution representation capabilities for real-time semantic segmentation of high-resolution images,
especially road driving images. The authors propose a new deep dual-resolution network for real-time semantic segmentation
of road scenes. DDRNet starts with one trunk and is then split into two parallel deep branches with different resolutions.
One deep branch generates a relatively high-resolution feature map, and the other extracts rich contextual information through
multiple downsampling operations. For efficient information fusion, multiple bilateral connections are bridged between the two branches.
In addition, we also propose a new module DAPPM, which greatly increases the number of accepted domains, extracting contextual
information more fully than ordinary PPM.

## Paper

[paper](https://arxiv.org/pdf/2101.06085):Hong, Yuanduo, et al. "Deep dual-resolution networks for real-time and accurate semantic segmentation of road scenes."

# [Dataset](#contents)

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images  
    - Test： 50,000 images
- Data format：jpeg
    - Note：Data will be processed in dataset.py

# [Features](#contents)

## Mixed Precision

The mixed precision training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── DDRNet
    ├── README_CN.md
    ├── README.md
    ├── ascend310_infer                     // Ascend310 infer
    ├── scripts
        ├──run_standalone_train_ascend.sh   // Ascend standalone train
        ├──run_distribute_train_ascend.sh   // Ascend distribute train
        ├──run_eval_ascend.sh               // Ascend evaluation
        ├──run_standalone_train_gpu.sh      // GPU standalone training
        ├──run_distribute_train_gpu.sh      // GPU distribute training
        ├──run_eval_gpu.sh                  // GPU evaluation
        └──run_infer_310.sh                 // 310 infer
    ├── src
        ├──configs                          // DDRNet configs
        ├──data                             // Dataset files
            ├──imagenet.py                  // imagenet dataset
            ├──augment                      // Augmentation
            └──data_utils                   // modelarts utils
        ├──models                           // DDRNet model
        ├──trainers                         // TrainOneStep interface
        ├──tools
            ├──callback.py                  // File with callbacks
            ├──cell.py                      // Cell to amp wrapper
            ├──criterion.py                 // Functions for criterion
            ├──get_misc.py                  // Interfaces for training
            ├──optimizer.py                 // Optimizer
            └──schedulers.py                // Shedulers
    ├── train.py                            // Training script
    ├── eval.py                             // Evaluation script
    ├── export.py                           // Export script
    ├── postprocess.py                      // Inference Computational Precision File
    └── preprocess.py                       // Inference preprocessing image files
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in yaml config in scr/configs.

```text
    arch: DDRNet23
    # ===== Dataset ===== #
    data_url: ./data/imagenet
    set: ImageNet
    num_classes: 1000
    mix_up: 0.8
    cutmix: 1.0
    color_jitter: 0.4
    auto_augment: rand-m9-mstd0.5-inc1
    interpolation: bicubic
    re_mode: pixel
    re_count: 1
    mixup_prob: 1.
    switch_prob: 0.5
    mixup_mode: batch
    mixup_off_epoch: 0.
    image_size: 224
    crop_pct: 0.875
    # ===== Learning Rate Policy ======== #
    optimizer: momentum
    use_nesterov: True
    base_lr: 0.1
    warmup_lr: 0.000001
    min_lr: 0.00001
    lr_scheduler: cosine_lr
    warmup_length: 10
    lr_adjust: 30 # for multistep lr
    # ===== Network training config ===== #
    amp_level: O2  
    keep_bn_fp32: True
    beta: [ 0.9, 0.999 ]
    clip_global_norm_value: 5.
    clip_global_norm: True
    is_dynamic_loss_scale: True
    epochs: 300
    label_smoothing: 0.1
    loss_scale: 1024
    weight_decay: 0.0001
    decay: 0.9 # for rmsprop
    momentum: 0.9
    batch_size: 512
    # ===== Hardware setup ===== #
    num_parallel_workers: 16
    device_target: Ascend
    # ===== Model config ===== #
    drop_path_rate: 0.1
```

## [Training and Evaluation Process](#contents)

### Running on Ascend

```bash
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]
  #OR
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]

  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME(imagenet2012)] [DATASET_PATH] [DEVICE_ID(optional)]
```

For distributed training, you need to create an hccl configuration file in JSON format in advance.
Please follow the instructions in the link below:
[hccl tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

### Running on GPU

```bash
  bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CONFIG_PATH]
  #OR
  bash ./scripts/run_distribute_train_gpu.sh [CONFIG_PATH]

  bash ./scripts/run_eval_gpu.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]
```

## [Inference Process](#contents)

### Export MindIR

```bash
python export.py --pretrained [CKPT_FILE] --ddr_config [CONFIG_PATH] --device_target [DEVICE_TARGET]
```

The following shows an example of using a mindir model to perform inference.

```bash
  # Ascend310 inference on ImageNet-1k
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [DEVICE_ID]
  Top1 acc: 0.76578
  Top5 acc: 0.9331
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

#### DDRNet on ImageNet2012

| Platform | GPU | Ascend |
| -------------------------- | ---------------------------------------------------------- | ------- |
| Arch | DDRNet-23 | DDRNet-23 |
| Resource | GPU: 8xRTX3090 24G; CPU: Intel(R) Xeon(R) Gold 6226R; RAM: 252G | Ascend 910 |
| Upload date | 2021-03-23 | 2021-12-04 |
| MindSpore version | 1.6.0 | 1.3.0 |
| Dataset | ImageNet-1k train | ImageNet-1k train |
| Training parameters | ddrnet23_imagenet_gpu.yaml | ddrnet23_imagenet_ascend.yaml |
| Parameters (M) | batch_size=256, epoch=300 | batch_size=512, epoch=300 |
| Optimizer | Momentum | Momentum |
| Loss function | SoftmaxCrossEntropy | SoftmaxCrossEntropy |
| Output | ckpt file | ckpt file |
| Final loss | 2.44 | - |
| Velocity | eight cards: mean 5000 ms/step | eight cards: mean 940 ms/step |
| Total time | eight cards: 240 h | eight cards: 35 h |

### Evaluation Performance

#### DDRNet on ImageNet2012

| Platform | GPU | Ascend |
| -------------------------- | ----------------- | ---------- |
| Arch | DDRNet-23 | DDRNet-23 |
| Resource |  GPU: 8xRTX3090 24G; CPU: Intel(R) Xeon(R) Gold 6226R; RAM: 252G  | Ascend 310 |
| Upload date | 2021-03-23 | 2021-12-04 |
| MindSpore version | 1.6.0 | 1.3.0 |
| Dataset | ImageNet-1k val | ImageNet-1k val |
| Eval loss | 1.2 | 1.313 |
| Accuracy | eight cards: top1:76.6% top5:93.4% | eight cards: top1:76.598% top5:93.312% |

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
