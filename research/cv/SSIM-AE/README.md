# Contents

<!-- TOC -->

- [Contents](#contents)
- [SSIM-AE Description](#ssim-ae-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Feature](#feature)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
  - [Training Process](#training-process)
      - [Training](#training)
      - [ModelArt Training](#modelart-training)
  - [Inference Process](#inference-process)
    - [Environment Inference on Ascend 910 AI Processor](#environment-inference-on-ascend-910-ai-processor)
    - [Export Process](#export-process)
    - [Environment Inference on Ascend 310 Processor](#environment-inference-on-ascend-310-processor)
- [Model Description](#model-description)
- [Random Seed Description](#random-seed-description)
- [ModelZoo Home Page](#modelzoo-home-page)

<!-- /TOC -->

# SSIM-AE Description

Autoencoder has emerged as a popular method for unsupervised defect detection. Usually, the image reconstructed by autoencoder is compared with the source image at the pixel level. If the distance is greater than a certain threshold, the source image is considered as a defective image. However, the distance-based loss function causes a large error when the reconstruction of some edge regions in the image is inaccurate. Moreover, when the defects are roughly the same in intensity but differ greatly in structure, the distance-based loss function cannot detect these defects. Given that even more advanced autoencoder cannot deal with these problems, this paper proposes to use a perceptual loss function based on structural similarity which examines inter-dependencies between pixels, taking into account luminance, contrast and structural information.

[Paper](https://www.researchgate.net/publication/326222902): Improving Unsupervised Defect Segmentation by Applying Structural Similarity To Autoencoders

# Model Architecture

SSIM-AE consists of a series of symmetric convolutional and transposed convolutional layers. The network structure is as follows.

| Layer      | Output Size | Kernel | Stride | Padding |
| ---------- | ----------- | :----: | ------ | ------- |
| Input      | 128 x 128 x 1 |        |        |         |
| Conv1      | 64 x 64 x 32  |  4 x 4  | 2      | 1       |
| Conv2      | 32 x 32 x 32  |  4 x 4  | 2      | 1       |
| Conv3      | 32 x 32 x 32  |  3 x 3  | 1      | 1       |
| Conv4      | 16 x 16 x 64  |  4 x 4  | 2      | 1       |
| Conv5      | 16 x 16 x 64  |  3 x 3  | 1      | 1       |
| Conv6      | 8 x 8 x 128   |  4 x 4  | 2      | 1       |
| Conv7      | 8 x 8 x 64    |  3 x 3  | 1      | 1       |
| Conv8      | 8 x 8 x 32    |  3 x 3  | 1      | 1       |
| Conv9      | 1 x 1 x d     |  8 x 8  | 1      | 0       |
| ConvTrans1 | 8 x 8 x 32    |  8 x 8  | 1      | 0       |
| Conv10     | 8 x 8 x 64    |  3 x 3  | 1      | 1       |
| Conv11     | 8 x 8 x 128   |  3 x 3  | 1      | 1       |
| ConvTrans2 | 16 x 16 x 64  |  4 x 4  | 2      | 1       |
| Conv12     | 16 x 16 x 64  |  3 x 3  | 1      | 1       |
| ConvTrans3 | 32 x 32 x 32  |  4 x 4  | 2      | 1       |
| Conv13     | 32 x 32 x 32  |  3 x 3  | 1      | 1       |
| ConvTrans4 | 64 x 64 x 32  |  4 x 4  | 2      | 1       |
| ConvTrans5 | 128 x 128 x 1 |  4 x 4  | 2      | 1       |

# Dataset

Used dataset: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

MVTec AD Dataset

- Description:
- Dataset size: 4.9 GB, 5354 high-resolution images in 15 classes
    - Training set: 3.4 GB, 3629 images
    - Test set: 1.5 GB, 1725 images
- Data format: binary file (PNG) and RGB
  The directory structure of a class in MVTec AD is as follows:

```bash
.
└─metal_nut
  └─train
    └─good
      └─000.png
      └─001.png
      ...
  └─test
    └─bent
      └─000.png
      └─001.png
       ...
    └─color
      └─000.png
      └─001.png
       ...
    ...
  └─ground_truth
    └─bent
      └─000_mask.png
      └─001_mask.png
      ...
    └─color
      └─000_mask.png
      └─001_mask.png
      ...
    ...
```

Non-detective images are stored in the **good** directory in the validation set.

We adopt pixel-level evaluation metrics for the woven texture dataset. The AUC values are used to determine whether the defect location is predicted correctly. We adopt image-level evaluation metrics for the MVTec AD dataset. The defective image is recognized if its defect location is predicted by the image-level prediction. **ok** indicates the correct rate of non-defective image prediction. **nok** indicates the correct rate of defective image prediction. **avg** indicates the correct rate of whole dataset prediction.

- Note: Data will be processed in **src/dataset.py**.

# Feature

## Mixed Precision

[Mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) accelerates the training process of deep neural networks by using the single-precision (FP32) data and half-precision (FP16) data without compromising the precision of networks trained with single-precision (FP32) data. It not only accelerates the computing process and reduces the memory usage, but also supports a larger model or batch size to be trained on specific hardware.
Take the FP16 operator as an example. If the input data format is FP32, MindSpore automatically reduces the precision to process data. You can open the INFO log and search for the keyword "reduce precision" to view operators with reduced precision.

# Environment Requirements

- Hardware (Ascend/CPU)
    - Set up the hardware environment with Ascend AI Processors or CPUs.
- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the following resources:
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# Quick Start

After installing MindSpore from the official website, you can perform the following steps for training and evaluation:

1. Modify the **yaml** file of the corresponding dataset in the **config** directory.

    ```yaml
    # The default parameter is located in the `ssim-ae/config/default_config.yaml` file.

    device_target: Ascend
    dataset: "none"
    dataset_path: ""
    aug_dir: ""   # If this parameter is empty, its default value is {dataset_path}/train_patches.
    distribute: False

    grayscale: False   # If grayscale is set to True, the read image is grayscale image.
    do_aug: True       # Specifies whether to enable data augmentation.
    online_aug: False  # Specifies whether to use online or offline augmentation.

    # Data augmentation parameters
    augment_num: 10000
    im_resize: 256
    crop_size: 128
    rotate_angle: 45.
    p_ratate: 0.3
    p_horizontal_flip: 0.3
    p_vertical_flip: 0.3

    # Training and model construction parameters
    z_dim: 100
    epochs: 200
    batch_size: 128
    lr: 2.0e-4
    decay: 1.0e-5
    flc: 32           # Number of channels at the first convolutional layer.
    stride: 32  
    load_ckpt_path: "" # Path for loading the checkpoint model. The model is loaded when the path is specified.

    # Inference-related parameters
    image_level: True     # Specifies whether to use the inference result of image level or pixel level. The `good` directory is required for image-level inference.
    ssim_threshold: -1.0  # If the value is less than 0, ssim_threshold is specified by the SSIM statistics at `percent` in the training set.
    l1_threshold: -1.0    # If the value is less than 0, l1_threshold is specified by the l1 statistics at `percent` in the training set.
    percent: 98
    checkpoint_path: ""   # Checkpoint path used for inference.
    save_dir: "./output"  # Path for saving images.
    ```

2. Start training.

- On Ascend AI Processors:

  ```shell
  # Training example:
  bash scripts/run_standalone_train.sh [CONFIG_PATH] [DEVICE_ID]

  # Inference example on Ascend 910
  bash scripts/run_eval.sh [CONFIG_PATH] [DEVICE_ID]

  # Inference example on Ascend 310
  bash scripts/run_infer_310.sh [MINDIR_PATH] [CONFIG_PATH] [SSIM_THRESHOLD] [L1_THRESHOLD] [DEVICE_ID]
  ```

## Training Process

### Training

- Single-device training on Ascend AI Processor

  ```bash
  # Set the parameters in config.yaml first.
  python train.py --config_path=[CONFIG_PATH]
  or
  bash scripts/run_standalone_train.sh [CONFIG_PATH] [DEVICE_ID]
  # example: bash scripts/run_standalone_train.sh config/bottle_config.yaml 0
  # The training runs at the backend, and the log file is stored in `./train.log`.
  ```

- Training on CPU

  ```bash
  # Set the parameters in config.yaml first.
  python train.py --config_path=[CONFIG_PATH] --device_target=CPU
  ```

  After the training is complete, you can find the checkpoint file in `./checkpoint`.

### ModelArt Training

- 8-device training on ModelArts

  ```python
  # (1) Set the code directory to "/[bucket name]/ssim-ae" on the web page. See the preceding steps to set the config.yaml file.
  # (2) Set the startup file on the web page to "/[bucket name]/ssim-ae/train.py" on the web page.
  # (3) Upload the dataset to the OBS bucket. See the preceding steps to set the directory structure of dataset.
  # (4) Set the data storage location to "/[bucket name]/ssim-ae/[data set name]" on the web page.
  # (5) Set the path for training output to "/[bucket name]/[expected output path]" on the web page.
  # (6) Set the following parameters on the web page:
  #     distribute = true
  #     model_arts = ture
  # (7) Create a training task.
  ```

- Single-device training on ModelArts

  ```python
  # (1) Set the code directory on the web page to "/[bucket name]/ssim-ae". See the preceding steps to set the config.yaml file.
  # (2) Set the startup file to "/[bucket name]/ssim-ae/train.py" on the web page.
  # (3) Upload the dataset to the OBS bucket. See the preceding steps to set the directory structure of dataset.
  # (4) Set the data storage location to "/[bucket name]/ssim-ae/[data set name]" on the web page.
  # (5) Set the path for training output to "/[bucket name]/[expected output path]" on the web page.
  # (6) Set the following parameters on the web page:
  #     model_arts = ture
  # (7) Create a training task.
  ```

After the training is complete, you can find the checkpoint file in `/[Bucket name]/result/checkpoint`.

## Inference Process

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

### Inference on Ascend 910 AI Processor

- Evaluation

  Check the checkpoint path used for inference before running the command below. Set the checkpoint path to the absolute path, for example, `username/ssim-ae/ssim_autocoder_22-257_8.ckpt`.

  ```bash
  # Set the checkpoint_path in config.yaml file.
  python eval.py --config_path=[CONFIG_PATH]
  or
  bash scripts/run_eval.sh [CONFIG_PATH] [DEVICE_ID]
  ```

  Note: For evaluation after distributed training , set checkpoint_path to the last saved checkpoint file, for example, `username/ssim-ae/ssim_autocoder_22-257_8.ckpt`. The accuracy of the test dataset is as follows:

  ```file
  # Single-device bottle dataset
  ok: 0.9, nok: 0.9841269841269841, avg: 0.963855421686747
  ```

### Export

```shell
python export.py --config_path=[CONFIG_PATH]
```

### Inference on Ascend 310 Processor

   Export the model before inference. AIR models can be exported only in the Ascend 910 environment. MindIR models can be exported in any environment. The value of **batch_size** can only be **1**.

- Infer the bottle dataset of MVTec AD on Ascend 310.

  Before running the following command, ensure that the configurations in the **config** file is the same as the training parameters. You need to manually add the values of **ssim_threshold** and **l1_threshold**. The values are better to be consistent with values automatically obtained on the Ascend 910.

  The inference result is saved in the current directory. In the **acc.log** file, you can find similar result below.

  ```shell
  # Inference on Ascend 310
  bash scripts/run_infer_310.sh [MINDIR_PATH] [CONFIG_PATH] [SSIM_THRESHOLD] [L1_THRESHOLD] [DEVICE_ID]
  # Example: bash scripts/run_infer_310.sh SSIM-AE-bottle.mindir config/bottle_config.yaml 0.777 0.3203 0
  # Result
  ok: 0.9, nok: 0.9841269841269841, avg: 0.963855421686747
  ```

# Model Description

| Parameter         | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| Model version     | SSIM-AE                                                      |
| Resources         | Ascend 910; 2.60 GHz CPU with 192 cores; 755 GB memory; EulerOS 2.8    |
| Upload date     | 2021-12-30                                                   |
| MindSpore version| 1.5.0                                                        |
| Script         | [ssim-ae script](https://gitee.com/mindspore/models/tree/master/research/cv/SSIM-AE)|

| Dataset   | Training Parameters| Speed (single device)| Total Duration| Loss Function| Accuracy| Checkpoint File Size|
| -------- |------- |----- |----- |-------- |------ |--------------- |
| MVTec AD bottle   | bottle_config.yaml | 354ms/step | 1.6 hours| SSIM | **ok**: 90%. **nok**: 98.4%. **avg**: 96.4%. (image level)| 32 MB|
| MVTec AD cable    | cable_config.yaml | 359 ms/step| 1.6 hours| SSIM | **ok**: 0%. **nok**: 100%. **avg**: 61.3%. (image level)| 32 MB|
| MVTec AD capsule  | capsule_config.yaml | 357 ms/step| 1.6 hours| SSIM | **ok**: 47.8%. **nok**: 91.7%. **avg**: 84.1%. (image level)| 32 MB|
| MVTec AD carpet   | carpet_config.yaml | 57 ms/step| 0.3 hours| SSIM | **ok**: 50%. **nok**: 98.8%. **avg**: 87.1%. (image level)| 13 MB|
| MVTec AD grid     | grid_config.yaml | 53 ms/step| 0.27 hours| SSIM | **ok**: 100%. **nok**: 94.7%. **avg**: 96.2%. (image level)| 13 MB|
| MVTec AD metal_nut   | metal_nut_config.yaml | 355 ms/step| 1.6 hours| SSIM | **ok**: 27.2%. **nok**: 91.4%. **avg**: 79.1%. (image level)| 32 MB|

# Random Seed Description

The seed in the create_dataset function is set in **dataset.py**, and the random seed in **train.py** is used.

# ModelZoo Home Page

For details, please go to the [official website](https://gitee.com/mindspore/models).
