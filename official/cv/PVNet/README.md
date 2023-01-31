# Contents <!-- TOC -->

- [Contents](#contents-)
- [PVNet Description](#pvnet-description)
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
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Export Process](#export-process)
        - [Export](#export)
    - [Inference Process](#inference-process)
        - [Inference](#inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [PVNet on LINEMOD](#pvnet-on-linemod)
            - [](#)
        - [Inference Performance](#inference-performance)
            - [PVNet on LINEMOD](#pvnet-on-linemod-1)
    - [Performance Description](#performance-description)
- [Random Seed Description](#random-seed-description)
- [ModelZoo Home Page](#modelzoo-home-page)

<!-- /TOC -->

# PVNet Description

PVNet is a CVPR oral paper in the 6D pose estimation field from the State Key Lab of CAD&CG of Zhejiang University in 2019. A 6D pose estimation task aims to detect the location and posture of an object in 3D space. With improvement of computer vision algorithms in recent years, detection of object status in 3D space attracts more and more attention, and the Best Paper Award at ECCV 2018 was also awarded to papers in the 6D pose estimation field. PVNet proposes a vector field-based voting method to predict the location of a key point. That is, each pixel predicts a direction vector pointing to a key point of an object. Compared with other methods, PVNet greatly improves the robustness of the prediction effect of occluded or truncated objects.

[Paper](https://zju3dv.github.io/pvnet/): Sida Peng, Y. Liu, Qixing Huang, Hujun Bao, Xiaowei Zhou."PVNet: Pixel-Wise Voting Network for 6DoF Pose Estimation."*2018, 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

# Model Architecture

PvNet adopts an Encoder-Decoder network structure. After an RGB graph is input, the semantic segmentation of the target object and the vector field pointing to the key points of the object are output. Then, the key points of the object are calculated from the direction vector field by using the RANSAC-based voting method.

# Dataset

Dataset used: [LINEMOD](https://zjueducn-my.sharepoint.com/personal/pengsida_zju_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fpvnet%2FLINEMOD%2Etar%2Egz&parent=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fpvnet)

- Dataset size: 1.8 GB, 13 objects in total

    - Training set/Test set: For details, see **train.txt**, **val.txt**, and **test.txt**. The number of images varies depending on the object.

Dataset used: [LINEMOD_ORIG](https://zjueducn-my.sharepoint.com/personal/pengsida_zju_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fpvnet%2FLINEMOD%5FORIG%2Etar%2Egz&parent=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fpvnet)

- Dataset size: 3.8 GB, 13 objects in total

    The synthetic dataset contains 10,000 images, which are included in the [LINEMOD_ORIG](https://zjueducn-my.sharepoint.com/personal/pengsida_zju_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fpvnet%2FLINEMOD%5FORIG%2Etar%2Egz&parent=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fpvnet) dataset.
    The rendering dataset contains 13 objects, and each object contains 10,000 images. To generate rendering data, perform the following steps:

    - [pvnet-rendering](https://github.com/zju3dv/pvnet-rendering)

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
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.html)

# Quick Start

After installing MindSpore from the official website, you can perform the following steps for training and evaluation:

- Prepare the data and model.

  ```python
  # The pickle files corresponding to the real data, rendering data, and synthetic data need to be generated for the original data.
  # For details about the conversion script, see model_utils/generateposedb.py.
  python model_utils/generateposedb.py

  # Original data needs to be converted into MindRecord data.
  # For details about the conversion script, see model_utils/data2mindrecord.py.
  python model_utils/data2mindrecord.py

  # Download the ResNet-18 pre-trained model and convert it to the MindSpore format.
  # For details about the PyTorch official version ResNet-18, please visit http://download.pytorch.org/models/resnet18-5c106cde.pth.
  # For details about the conversion script, see model_utils/pth2ms.py.
  python model_utils/pth2ms.py
  ```

- Run in the Ascend AI Processor environment.

  ```text
  # Add the dataset path. The following uses LINEMOD as an example.
  data_url:"/data/bucket-4609/dataset/pvnet/data2mindrecord/"

  # Set the path for saving the model file. The following uses LINEMOD as an example.
  train_url:"/data/bucket-4609/dataset/pvnet/trained/"

  # Add the training object name.
  cls_name:"cat"

  # Add the dataset name.
  dataset_name:"LINEMOD"

  # Add the pre-trained model.
  # The pre-trained model is stored in the root directory. If this parameter is set to None, no pre-trained model is used.
  pretrained_path:"./resnet18-5c106cde.ckpt"

  # Add the checkpoint file path before inference.
  ckpt_file:"./model/pvnet-199_681.ckpt"
  ```

  ```python
  # Training
  python train.py --cls_name=ape > train.log 2>&1 &

  # Distributed training
  export RANK_SIZE=8
  bash scripts/run_distribute.sh --cls_name ape --distribute 1 --data_url= ~/pvnet/data2mindrecord/ --train_url= ~/pvnet/trained/

  # Evaluation
  bash scripts/run_eval.sh

  # Inference
  bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [CLS_NAME] [DEVICE_ID]
  ```

By default, the LINEMOD dataset is used. For details, see the specified script.

- Perform training on ModelArts. (If you want to run the training on ModelArts, see [modelarts](https://support.huaweicloud.com/modelarts/).)

    - Train the LINEMOD dataset on ModelArts using eight devices.

      ```python
      # (1) Perform step a or b.
      #       a. In the pvnet_linemod_config.yaml file, set distribute to 1 and set other parameters as follows:
      #          cls_name,batch_size,data_url,train_url
      #       b. Set distribute to 1 on the web page.
      #          Set train_url to /bucket-xxxx/linemod/trained/ on the web page.
      #          Set data_url to /bucket-xxxx/linemod/dataset/ on the web page.
      #          Set other parameters on the web page.
      # (3) Upload your compressed dataset to the S3 bucket. (You can also upload the original dataset, but that may take a long time.)
      # (4) Set your code path to ~/pvnet on the web page.
      # (5) Set the boot file to ~/pvnet/train.py on the web page.
      # (6) Set the training set, training output file path, and job log path on the web page.
      # (7) Create a training job.
      ```

# Script Description

## Script and Sample Code

```text
├── model_zoo
    ├── README.md                            // Description of all models
    ├── pvnet
        ├── README.md                        // Description of PVNet
        ├── ascend310_infer                  // Implement the source code for inference on Ascend 310 AI Processor.

        ├── model_utils
        │   ├──config.py                     // Script for reading the configuration file
        │   ├──data_file_utils.py            // Utility script
        │   ├──data2mindrecord.py            // Script for converting the original data to the MindRecord format
        │   ├──data2mindrecord.py            // Script for generating pickle files based on original data
        │   ├──data2mindrecord.py            // Script for converting the pre-trained model to the MindSpore format

        ├── scripts
        │   ├──run_distribute.sh             // Shell script for distributed training on Ascend AI Processors
        │   ├──run_eval.sh                   // Shell script for evaluation on Ascend AI Processors
        │   ├──run_infer_310.sh              // Shell script for inference on Ascend AI Processors

        ├── src
        │   ├──lib
        │   │   ├──voting                   // Code related to RANSAC-based voting
        │   ├──dataset.py                   // Dataset script
        │   ├──evaluation_dataset.py        // Evaluation dataset script
        │   ├──evaluation_utils.py          // Evaluation utility script
        │   ├──loss_scale.py                // Dynamic loss_scale script
        │   ├──model_reposity.py            // Network model script
        │   ├──net_utils.py                 // Network utility script
        │   ├──resnet.py                    // Feature extraction network script

        ├── train.py                        // Training script
        ├── eval.py                         // Evaluation script
        ├── postprogress.py                 // Script for post-processing after inference on Ascend 310 AI Processors
        ├── export.py                       // Export the checkpoint file to MindIR.
        ├── pvnet_linemod_config.yaml       // YAML file for parameter configuration
        ├── requirements.txt                // Dependency description

```

## Script Parameters

You can configure both training and evaluation parameters in pvnet_linemod_config.yaml.

- Configure the PVNet and LINEMOD datasets.

  ```python
  'data_url': "./pvnet/"                           # Absolute full path of the training set
  'train_url': "./trained/"                        # Path for storing the trained model
  'group_size': 1                                  # Total number of training devices
  'rank': 0                                        # Current training device number
  'device_target': "Ascend"                        # Running device
  'distribute': False                              # Specifies whether distributed training is used.
  'cls_name': "cat"                                # Training object class
  'vote_num': 9                                    # Number of voting key points
  'workers_num': 16                                # Number of worker threads
  'batch_size': 16                                 # Training batch size
  'epoch_size': 200                                # Training epoch
  'learning_rate': 0.005                           # Training learning rate
  'learning_rate_decay_epoch': 20                  # Number of learning rate decay epochs
  'learning_rate_decay_rate': 0.5                  # Learning rate decay rate
  'pretrained_path': "./resnet18-5c106cde.ckpt"    # Path of the pretrained model
  'loss_scale_value': 1024                         # Initial value of dynamic loss scale
  'scale_factor': 2                                # Dynamic loss scale factor
  'scale_window': 1000                             # Dynamic loss scale update frequency
  'dataset_name': "LINEMOD"                        # Name of the LINEMOD training set
  'dataset_dir': "~/pvnet/data/"                   # LINEMOD dataset path
  'origin_dataset_name': "LINEMOD_ORIG"            # Original LINEMOD dataset name
  'img_width': 640                                 # Width of the dataset images
  'img_height': 480                                # Height of the dataset images
  'ckpt_file': "./train_cat-199_618.ckpt"          # File for saving models
  'eval_dataset': "./"                             # Evaluation dataset path
  'result_path': "./scripts/result_Files"          # Path for storing Ascend 310 AI Processor-based inference results
  'file_name': "pvnet"                             # Prefix of the generated MindIR file
  'file_format': "MINDIR"                          # Inference model conversion format
  'file_format': "MINDIR"                          # Maximum number of models that can be saved
  'img_crop_size_width': 480                       # Image width after data augmentation
  'img_crop_size_height': 360                      # Image height after data augmentation
  'rotation': True                                 # Specifies whether images are rotated after data augmentation.
  'rot_ang_min': -30                               # Rotation angle range of an object
  'rot_ang_max': 30
  'crop': True                                     # Specifies whether images are cropped after data augmentation.
  resize_ratio_min: 0.8                            # Image resizing ratio
  resize_ratio_max: 1.2
  overlap_ratio: 0.8                               # Object overlapping ratio
  brightness: 0.1                                  # Parameter for adjusting the image brightness
  contrast: 0.1                                    # Parameter for adjusting the image contrast
  saturation: 0.05                                 # Parameter for adjusting the image saturation
  hue: 0.05                                        # Parameter for adjusting the image hue
  ```

For more configuration details, see the `pvnet_linemod_config.yaml` file.

### Training Process

- Run in the Ascend AI Processor environment.

  ```bash
  # Standalone training
  python train.py >train.log
  ```

  You can modify related configurations, such as rank and cls_name, in the **pvnet_linemod_config.yaml** configuration file.

  After running the preceding Python command, you can view the result in the `train.log` file.

  ```bash
  # Distributed training
  Usage: bash scripts/run_distribute.sh --cls_name [cls_name] --distribute [distribute]
  #example: bash ./scripts/run_distribute.sh --cls_name ape --distribute 1
  ```

  You need to set **RANK_SIZE** in **run_distribute.sh** and modify other parameters in the **pvnet_linemod_config.yaml** configuration file.

  The preceding shell script will run distributed training in the backend. You can view the result in the **device[X]/train.log** file. The following methods are used to achieve the loss value:

  ```bash
  # grep "total" device[X]/train.log
  Rank:0/2, Epoch:[1/200], Step[80/308] cost:0.597510814666748.s total:0.28220612
  Rank:0/2, Epoch:[1/200], Step[160/308] cost:0.41454052925109863.s total:0.20701535
  Rank:0/2, Epoch:[1/200], Step[240/308] cost:0.2790074348449707.s total:0.15037575
  ...
  Rank:1/2, Epoch:[1/200], Step[80/308] cost:1.0746071338653564.s total:0.27446517
  Rank:1/2, Epoch:[1/200], Step[160/308] cost:1.1847755908966064.s total:0.20768473
  Rank:1/2, Epoch:[1/200], Step[240/308] cost:0.9300284385681152.s total:0.13899626
  ...
  ```

## Evaluation Process

### Evaluation

- Evaluate the LINEMOD dataset in the Ascend environment.

  Before running the following commands, check the parameters used for evaluation. The configuration items to be modified are cls_name and ckpt_file. Set the checkpoint file path to an absolute full path, for example, **/username/dataset/cat/train_cat-199_618.ckpt**.

  ```bash
  bash scripts/run_eval.sh
  ```

  The preceding Python command is executed in the backend. You can view the result in the eval.log file.

## Export Process

### Export

Before exporting the dataset, modify the **pvnet_linemod_config.yaml** configuration file corresponding to the dataset.
The configuration items to be modified are cls_name, file_name, and ckpt_file.

```shell
python export.py
```

## Inference Process

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

### Inference

Export the model before inference. AIR models can be exported only in the Ascend 910 AI Processor environment. MindIR models can be exported in any environment. The value of **batch_size** can only be **1**.

- Use the LINEMOD dataset for inference on Ascend 310 AI Processors.

  Before running the following commands, you need to modify the configuration file. The items to be modified include cls_name, eval_dataset, and result_path.

  The inference results are saved in the **scripts** directory. You can find results similar to the following in the **postprocess.log** file:

  ```shell
  # Run inference
  bash scripts/run_infer_cpp.sh [MODEL_PATH] [DATA_PATH] [CLS_NAME] [DEVICE_TYPE] [DEVICE_ID]
  # example:bash scripts/run_infer_cpp.sh ./can.mindir ./LINEMOD/can/JPEGImages/ can Ascend 0
  Processing object:can, 2D projection error:0.9960629921259843, ADD:0.8622047244094488
  ```

# Model Description

## Performance

### Evaluation Performance

#### PVNet on LINEMOD

| Parameter                   | Ascend                                                      |
| ----------------------- | ----------------------------------------------------------- |
| Model version               | PVNet                                                       |
| Resources                   | Ascend 910 AI Processor; 2.60 GHz CPU with 192 cores; 755 GB memory; EulerOS 2.8   |
| Upload date               | 2021-12-25                                                  |
| MindSpore version          | 1.5.0                                                       |
| Dataset                 | LINEMOD                                                     |
| Training parameters               | epoch=200, batch_size = 16, lr=0.0005                       |
| Optimizer                 | Adam                                                        |
| Loss function               | Smoth L1 Loss, SoftmaxCrossEntropyWithLogits               |
| Output                   | Segmentation probability and voting vector field                                       |
| Loss                   | 0.005                                                       |
| Speed                   | 990 ms/step (8 devices)                                          |
| Total duration                 | 547 minutes (8 devices)                                             |
| Parameter                   | 148.5 MB (.ckpt)                                          |

####

### Inference Performance

#### PVNet on LINEMOD

| Parameter               | Ascend                      |
| ------------------- | --------------------------- |
| Model version           | PVNet                       |
| Resources               |  Ascend 310 AI Processor, EulerOS 2.8 |
| Upload date           | 2021-12-25                  |
| MindSpore version     | 1.5.0                       |
| Dataset             | Four classes of objects, about 1000 images for each class  |
| batch_size          | 1                           |
| Output               | 2D projection fulfillment rate and ADD fulfillment rate|
| Accuracies             | 2D projection: 99.5% for a single device; 99.7% for eight devices. ADD: 70.4% for a single device; 66.7% for eight devices.|
| Inference model           | 49.6 MB (.mindir)        |

## Performance Description

This document describes how to verify the performance of four classes of objects (cat/ape/cam/can) in the LINEMOD dataset. The PVNet network has sufficient generalization. For other objects in the LINEMOD dataset, you can refer to the provided four classes of objects, modify the corresponding configuration parameters for training, such as the object name.

# Random Seed Description

A random seed is set in **train.py**.

# ModelZoo Home Page

 For details, please go to the [official website](https://gitee.com/mindspore/models).
