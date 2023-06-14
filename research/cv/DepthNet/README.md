# Contents

<!-- TOC -->

- [Contents](#contents)
- [DepthNet Description](#depthnet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
        - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Export Process](#export-process)
        - [Model Export](#model-export)
    - [Inference Process](#inference-process)
        - [Inference](#inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [DepthNet on NYU](#depthnet-on-nyu)
        - [](#)
        - [Inference Performance](#inference-performance)
            - [DepthNet on NYU](#depthnet-on-nyu-1)
- [Random Seed Description](#random-seed-description)
- [ModelZoo Home Page](#modelzoo-home-page)

<!-- /TOC -->

# DepthNet Description

A main task of depth estimation is to obtain a corresponding depth map from a given RGB image, which is an important topic of 3D scenario understanding. However, monocular depth estimation is often more difficult due to factors such as uncertainty of scale. The classical monocular depth estimation work proposed by Eigen et al. from the New York University adopts the coarse-to-fine training strategy. First, a network is used to roughly estimate the depth map based on the global information of RGB images. Then, a network is used to precisely estimate the local depth information of the depth map.

[Paper](https://arxiv.org/abs/1406.2283): Depth Map Prediction from a Single Image using a Multi-Scale Deep Network.David Eigen, Christian Puhrsch, Rob Fergus.

# Model Architecture

Specifically, this monocular depth estimation network (Depth Net) work consists of two parts: CoarseNet and FineNet. First, an RGB image is input for CoarseNet. After passing through convolutional layers, the image passes through two fully-connected layers to output a coarse depth map (Coarse Depth). In the FineNet, after passing through a convolutional layer, the RGB image is stitched with the input Coarse Depth to form a new feature map. After passing through several convolution layers, a more refined depth map is obtained.

# Dataset

For reproduction and verification on MindSpore, the preprocessed [NYU dataset](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing) provided by [Junjie Hu](https://github.com/JunjH/Revisiting_Single_Depth_Estimation) is used, and the following NYU dataset preprocessing methods are used: [1](Structure-Aware Residual Pyramid Network for Monocular Depth Estimation. Xiaotian Chen, Xuejin Chen, Zheng-Jun Zha) and [2](Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps with Accurate Object Boundaries. Junjie Hu, Mete Ozay, Yan Zhang, Takayuki Okatani.). Data of 284 scenarios is used as the training set, and 654 images are used as the test set for validating the accuracy results.
Data files are stored in the following directory:

```text
├── NYU
    ├── Train
        ├── basement_0001a_out
            ├── 1.jpg
            ├── 1.png
            ├── 2.jpg
            ├── 2.png
              ....
        ├── basement_0001b_out
              ....
    ├── Test
        ├── 00000_colors.png
        ├── 00000_depth.png
        ├── 00001_colors.png
        ├── 00001_depth.png
              ....

```

In the training set, RGB images are stored in .jpg format, and depth map data is stored in .png format. Depth value z=pixel_value / 255.0 x 10.0 (m)
In the test set, RGB images and depth maps are stored in .png format. The depth value z = pixel_value/1000.0 (m). For details about how to read the dataset, refer to the **data_loader.py** file.

# Features

## Mixed Precision

[Mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) accelerates the training process of deep neural networks by using the single-precision (FP32) data and half-precision (FP16) data without compromising the precision of networks trained with single-precision (FP32) data. It not only accelerates the computing process and reduces the memory usage, but also supports a larger model or batch size to be trained on specific hardware.
Take the FP16 operator as an example. If the input data format is FP32, MindSpore automatically reduces the precision to process data. You can open the INFO log and search for the keyword "reduce precision" to view operators with reduced precision.

# Environment Requirements

- Hardware
    - Set up the hardware environment with Ascend AI Processors.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the following resources:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# Quick Start

Install MindSpore from the official website or use the configured development environment of ROMA.

- Prepare the data and model.

  ```bash
  # Start training.

  # Run the following command in the single-device training environment:
  ###Bash script commands
  cd scripts
  bash run_standalone_train_ascend.sh [DATASET_PATH] [DEVICE_ID]
  Example: bash run_standalone_train_ascend.sh ~/mindspore_dataset/NYU 0
  ### Alternatively, run the Python command.
  python train.py --data_url ~/mindspore_dataset/NYU --device_id 0 > train.log 2>&1 &

  # Model evaluation test, using the .ckpt model
  ## Bash script commands
  cd scripts
  bash run_eval.sh [DATASET_PATH] [COARSENET_MODEL_PATH] [FINENET_MODEL_PATH]
  Example: bash run_eval.sh ~/mindspore_dataset/NYU ~/Model/Ckpt/FinalCoarseNet.ckpt ~/Model/Ckpt/FinalFineNet.ckpt

  ## Alternatively, run the Python command.
  python eval.py --test_data ~/mindspore_dataset/NYU --coarse_ckpt_model ~/Model/Ckpt/FinalCoarseNet.ckpt --fine_ckpt_model ~/Model/Ckpt/FinalFineNet.ckpt> eval.log 2>&1 &

  # In the CoarseNet, convert the .ckpt models to the .mindir and .air formats.
  ## Bash script commands
  cd scripts
  bash run_export_coarse_model.sh
  ## Alternatively, run the Python command.
  python export.py --coarse_or_fine coarse

  # In the FineNet, convert the .ckpt models to the .mindir and .air formats.
  ## Bash script commands
  cd scripts
  bash run_export_fine_model.sh
  ## Alternatively, run the Python command.
  python export.py --coarse_or_fine fine

  # Model inference
  cd scripts
  bash run_infer_310.sh ../Model/MindIR/FinalCoarseNet.mindir ../Model/MindIR/FinalFineNet.mindir ../NYU/Test/ 0

  # 8-device distributed training:
  ###Bash script commands
  cd scripts
  bash run_distributed_train_ascend.sh [DATASET_PATH] [RANK_TABLE_FILE]
  Example: bash run_standalone_train_ascend.sh ~/mindspore_dataset/NYU ~/rank_table_8pcs.json

  # Model evaluation test, using the .ckpt model
  ## Bash script commands
  cd scripts
  bash run_eval.sh [DATASET_PATH] [COARSENET_MODEL_PATH] [FINENET_MODEL_PATH]
  Example: bash run_eval.sh ~/mindspore_dataset/NYU ~/Model/Ckpt/FinalCoarseNet_rank0.ckpt ~/Model/Ckpt/FinalFineNet_rank0.ckpt
  ## Alternatively, run the Python command.
  python eval.py --test_data ~/mindspore_dataset/NYU --coarse_ckpt_model ~/Model/Ckpt/FinalCoarseNet_rank0.ckpt --fine_ckpt_model ~/Model/Ckpt/FinalFineNet_rank0.ckpt > eval.log 2>&1 &
  ```

# Script Description

## Script and Sample Code

```text
├── ModelZoo_DepthNet_MS_MTI
        ├── ascend310_infer               // Model inference
            ├── CmakeLists.txt            // CMake list
            ├── build.sh                  // Build script
            ├── src
                ├── main.cc               // Main function for model inference
                ├── utils.cc              // File operation function
            ├── inc
                ├── utils.h               // File operation header file
        ├── scripts
            ├── run_eval.sh               // Script for running the evaluation
            ├── run_export_coarse_model.sh  // Script for exporting the Coarse model
            ├── run_export_fine_model.sh    // Script for exporting the Fine model
            ├── run_infer_310.sh            // Model inference script
            ├── run_standalone_train_ascend.sh  // Script for running the single-device training
            ├── run_distributed_train_ascend.sh  // Distributed training script in the 8-device environment
        ├── src
             ├── data_loader.py           // Read data.
             ├── loss.py                  // Define the loss function and evaluation metrics.
             ├── net.py                   // Define the network structure.
        ├── README.md                     // Description of DepthNet
        ├── eval.py                       // Evaluation test
        ├── export.py                     // Convert the .ckpt models to the .mindir and .air formats in a network.
        ├── postprocess.py                // Process the image after inference.
        ├── preprocess.py                 // Process images before inference.
        ├── train.py                      // Training file
```

### Training Process

- On Ascend AI Processors

  ```bash
  ### Bash script commands
  cd scripts
  bash run_standalone_train_ascend.sh [DATASET_PATH] [DEVICE_ID]
  Example: bash run_standalone_train_ascend.sh ~/mindspore_dataset/NYU 0
  ### Alternatively, run the Python command.
  python train.py --data_url ~/mindspore_dataset/NYU --device_id 0 > train.log 2>&1 &
  ```

  After running the preceding command, you can view the result in the `train.log` file.

  ```bash
  # python train.log
  traing coarse net, step: 0 loss:1.73914, time cost: 54.1325149361328
  traing coarse net, step: 10 loss:1.606946, time cost: 0.051651954650878906
  traing coarse net, step: 20 loss:1.5636182, time cost: 0.06647920608520508
  ...
  traing coarse net, step: 14150 loss:0.39416388, time cost: 0.04835963249206543
  traing coarse net, step: 14160 loss:0.38534725, time cost: 0.04690909385681152
  traing coarse net, step: 14170 loss:0.39199725, time cost: 0.04682588577270508
  ...
  ```

## Evaluation Process

### Evaluation

- Evaluate the NYU dataset in the Ascend environment.

  Name the finally trained models **FinalCoarseNet.ckpt** and **FinalFineNet.ckpt**, respectively, place them in the **./Model/Ckpt** folder, load the trained .ckpt model, and perform evaluation.

  ```bash
  ## Bash script commands
  cd scripts
  bash run_eval.sh [DATASET_PATH]
  Example: bash run_eval.sh ~/mindspore_dataset/NYU
  ## Alternatively, run the Python command.
  python eval.py --test_data ~/mindspore_dataset/NYU > eval.log 2>&1 &
  ```

## Export Process

### Model Export

```bash
# Bash script commands
## Export the Coarse model:
cd scripts
bash run_export_coarse_model.sh

## Export the Fine model:
cd scripts
bash run_export_fine_model.sh

# Alternatively, run the Python command.
## Export the Coarse model:
python export.py --coarse_or_fine coarse

## Export the Fine model:
python export.py --coarse_or_fine fine
```

## Inference Process

### Inference

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

- Evaluate the NYU dataset in the inference environment.

  Export the model before inference. Air models can be exported only in the Ascend 910 AI Processor environment. MindIR models can be exported in any environment.

- Go to the **scripts** directory and run the following command to perform model inference:

  Run the following command:

  ```bash
   bash run_infer_310.sh [MINDIR1_PATH] [MINDIR2_PATH] [DATA_PATH] [DEVICE_ID]
  ```

  **MINDIR1_PATH** indicates the CoarseNet path, **MINDIR2_PATH** indicates the FineNet path, and **DATA_PATH** indicates the test set path.

  ```bash
  cd scripts
  bash run_infer_310.sh ../Model/MindIR/FinalCoarseNet.mindir ../Model/MindIR/FinalFineNet.mindir ../NYU/Test/ 0
  ```

The inference result is saved in the current directory. The **preprocess_Result** folder stores the preprocessed images. The **result_Files** folder stores the image inference result of the model. You can find the result similar to the following in the **acc.log** file:

# Model Description

## Performance

### Evaluation Performance

#### DepthNet on NYU

| Parameter                | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model version             | DepthNet                                            |
| Resources                  | Ascend 910; 2.60 GHz CPU with 192 cores; 720 GB memory; EulerOS 2.8            |
| Upload date             | 2021-12-25                                 |
| MindSpore version         | 1.5.1                                                       |
| Dataset                   | NYU                                              |
| CoarseNet training parameters       | epoch=20, batch_size = 32, lr=0.0001    |
| FineNet training parameters       | epoch=10, batch_size = 32, lr=0.00001    |
| Optimizer                 | Adam                                                 |
| Loss function             | Combination of L2 Loss and ScaleInvariant Loss|
| Output                   | Panoramic depth map                                        |
| Loss| 0.2 |
| Speed| 640 batch/s (single device)|
| Total duration| 360min@1P(coarse) + 360min@1P(fine) |
| Parameter| 84.5 MB (.ckpt)|

####

### Inference Performance

#### DepthNet on NYU

| Parameter         | Ascend                      |
| ------------------- | --------------------------- |
| Model version      | DepthNet                |
| Resources           |  Ascend 310 AI Processor; Ubuntu 18.04.3 LTS 4.15.0-45.generic x86_64              |
| Upload date      | 2021-12-25 |
| MindSpore version  | 1.5.1                       |
| Dataset            | NYU test set|
| batch_size          | 1                      |
| Output            | delta1_loss, delta2_loss, delta3_loss, abs_relative_loss, sqr_relative_loss, rmse_linear_loss, rmse_log_loss|
| Metrics           | delta1_loss:  0.618 delta2_loss:  0.880 delta3_loss:  0.965 abs_relative_loss: 0.228 sqr_relative_loss:  0.224  rmse_linear_loss:  0.764 rmse_log_loss:  0.272
 |
| Inference models| coarse_net: 84 MB (.mindir); fine_net: 482 KB (.mindir)   |

# Random Seed Description

A random seed is set in **train.py**.

# ModelZoo Home Page

 For details, please go to the [official website](https://gitee.com/mindspore/models).
