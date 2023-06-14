# Contents

<!-- TOC -->

- [Contents](#contents)
- [Semantic Human Matting Description](#semantic-human-matting-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
    - [Downloading a Dataset](#downloading-a-dataset)
    - [Creating a Dataset](#creating-a-dataset)
    - [Obtaining and Converting the Weight File of the Torch Network](#obtaining-and-converting-the-weight-file-of-the-torch-network)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Inference based on Ascend 310 AI Processors](#inference-based-on-ascend-310-ai-processors)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Dataset Creation and Configuration](#dataset-creation-and-configuration)
        - [Training Configuration](#training-configuration)
        - [Test, Inference, and Model Export Configurations](#test-inference-and-model-export-configurations)
    - [Dataset Creation Process](#dataset-creation-process)
        - [Creating a Dataset](#creating-a-dataset-1)
    - [Obtaining the Initialization Weight File](#obtaining-the-initialization-weight-file)
        - [Obtaining the Initialization Weight](#obtaining-the-initialization-weight)
    - [Training Process](#training-process)
        - [Training](#training-1)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation-1)
    - [Export Process](#export-process)
        - [Export](#export)
    - [Inference Process](#inference-process)
        - [Inference](#inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Random Seed Description](#random-seed-description)
- [ModelZoo Home Page](#modelzoo-home-page)

<!-- /TOC -->

# Semantic Human Matting Description

**Semantic Human Matting (SHM)** is a fully automatic method pioneered by Alibaba for extracting humans from natural images. It proposes a new algorithm that learns to jointly fit both semantic information and high quality details with deep networks. It can adaptively integrate rough semantic information and details on each pixel, which is critical to implementing end-to-end training. At the same time, a large-scale high-quality portrait dataset is created, which contains 35,513 human images and corresponding alpha matting results. The dataset can not only effectively train deep networks in SHM, but also facilitate its research on cutout. However, the dataset is not open-source.

Because the author does not open the source code and dataset, the [code implementation](https://github.com/lizhengwei1992/Semantic_Human_Matting) from **[lizhengwei1992]** and the [SHM cutout algorithm](https://blog.csdn.net/zzZ_CMing/article/details/109490676) from **zzZ_CMing** are used for reference.

Paper [Semantic Human Matting](https://arxiv.org/pdf/1809.01354.pdf): Quan Chen, Tiezheng Ge, Yanyu Xu, Zhiqiang Zhang, Xinxin Yang, Kun Gai.

# Model Architecture

**SHM** consists of three parts: **T-Net**, **M-Net**, and **Fusion Module**.
**T-Net**: adopts "MobileNetV2+UNet" and outputs a 3-channel feature map, which indicates the probability that each pixel belongs to its own category. This method is widely used in semantic segmentation.
**M-Net**: an encoder-decoder network (with the structure slightly different from that in the paper). The encoder network has four convolutional layers and four max-pooling layers. The decoder network has four convolutional layers and four transposed convolutional layers. M-Net adds a batch normalization layer (except for transposed convolution) after each convolutional layer to accelerate convergence.
**Fusion Module**: directly merges the outputs of T-Net and M-Net to generate the final alpha matte result.

# Dataset

Dataset used: [Matting Human Datasets](https://github.com/aisegmentcn/matting_human_datasets)

- Dataset size: 28.7 GB, including 137,706 images and corresponding matting results

  ```text
  ├── archive
      ├── clip_img                        // Half-body portrait image
      ├── matting                         // Matting image corresponding to clip_img
      ├── matting_human_half              // The directory contains the clip_img and matting subdirectories (not in use).
  ```

- Data format: RGB

# Features

## Mixed Precision

[Mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) accelerates the training process of deep neural networks by using the single-precision (FP32) data and half-precision (FP16) data. It not only accelerates the computing process and reduces the memory usage, but also supports a larger model or batch size to be trained on specific hardware. Take the FP16 operator as an example. If the input data format is FP32, MindSpore automatically reduces the precision to process data. You can open the INFO log and search for the keyword "`reduce precision`" to view operators with reduced precision.

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

For details about the configuration before the execution, see [Script Parameters](#script-parameters).

## Downloading a Dataset

  Download [Matting Human Datasets](<https://github.com/aisegmentcn/matting_human_datasets>), decompress it to the `/cache` directory, and change the name to `human_matting`. The directory and folder names can be user-defined.

## Creating a Dataset

  Before creating a dataset, you need to configure the `generate_data` field in `config.yaml`. For details, see [Dataset Creation Process](#dataset-creation-process).

  ```text
  # Run the following command to generate a dataset.
  python3 generate_datasets.py --yaml_path=../config.yaml
  ```

## Obtaining and Converting the Weight File of the Torch Network

  1. Obtain the source code of the Torch version from GitHub: https://github.com/lizhengwei1992/Semantic_Human_Matting.

  2. Copy the `get_init_weight.py` file in the `src` directory to the `root directory of the source code of the Torch version` and run the following command:

  ```text
  python3 get_init_weight.py
  ```

  3. After the command is executed, the `init_weight.ckpt` initialization weight file is generated in the `root directory of the source code of the Torch version`. You can copy the **initialization weight file** to the **root directory of the project**.

  For details, see [Obtaining the Initialization Weight File](#obtaining-the-initialization-weight-file).

## Training

  Before the execution, configure the `pre_train_t_net`, `pre_train_m_net`, `train_phase`, and other fields in `config.yaml`. For details, see `Training Configuration` and [Training Process](#training-process) under [Script Parameters](#script-parameters).

- Single-device training

  ```text
  # Example of single-device training
  python3 train.py --yaml_path=[YAML_PATH] --data_url=[DATASETS] --train_url=[OUTPUT] --init_weight=[INIT_WEIGHT][OPTIONAL] > train.log 2>&1 &
  # Example: python3 train.py --yaml_path=./config.yaml --data_url=/cache/datasets --train_url=/cache/output --init_weight=./init_weight.ckpt > train.log 2>&1 &
  ```

- Distributed training

  For distributed training, you need to create an HCCL configuration file in JSON format in advance. You can name the file `hccl_8p.json` and save it to the root directory of the current project.
  Follow the instructions in the following link: <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

  ```text
  # Distributed training
  bash run_train.sh [RANK_TABLE_FILE] [YAML_PATH] [DATASETS] [OUTPUT] [INIT_WEIGHT][OPTIONAL]
  # example: bash run_train.sh ../hccl_8p.json ../config.yaml /cache/datasets /cache/output ../init_weight.ckpt
  ```

## Evaluation

  Before execution, configure the `test` field in `config.yaml`. For details, see [Evaluation Process](#evaluation-process).

  ```text
  # Evaluation
  python3 eval.py --yaml_path=[YAML_PATH] > eval.log 2>&1 &
  # example: python3 eval.py --yaml_path=./config.yaml > eval.log 2>&1 &
  ```

  Or

  ```text
  bash run_eval.sh [DEVICE_ID][OPTIONAL]
  # example: bash run_eval.sh
  ```

## Inference based on Ascend 310 AI Processors

- **Model export**

    Before exporting a model, you need to modify the `export` parameters in the `config.yaml` file and configure the parameters related to model export. The current export format is `MINDIR`. For more configurations, see [Script Parameters](#script-parameters). For details about the export process, see [Export Process](#export-process).

    ```text
    python3 export.py --config_path=[CONFIG_PATH]
    # example: python3 export.py --config_path=./config.yaml
    ```

- **Inference**

    Before execution, you need to modify the `infer` parameters in the `config.yaml` file. `file_test_list` is an absolute path, which is the value of `test.txt` generated in section `Creating a Dataset`. The value of `size` is fixed to **320**. For more configurations, see [Script Parameters](#script-parameters). For details about the inference process, see [Inference Process](#inference-process).

    ```text
    # Inference
    bash run_infer_310.sh [MINDIR_PATH]
    # [MINDIR_PATH]: path of the exported model file configured in the previous step
    # example: bash run_infer_310.sh ../shm_export.mindir
    ```

# Script Description

## Script and Sample Code

```text
├── model_zoo
    ├── README.md                      // Description of all models
    ├── SemanticHumanMatting
        ├── README.md                    // SHM description
        ├── ascend310_infer                  // Implement the source code for inference on Ascend 310 AI Processor.
        ├── scripts
        │   ├──run_train.sh              // Shell script for distributed training on Ascend AI Processors
        │   ├──run_eval.sh               // Shell script for evaluation on Ascend AI Processors
        │   ├──run_infer_310.sh          // Shell script for inference on Ascend AI Processors
        ├── src
        │   ├──model
        │   │  ├──T_Net.py               // T network
        │   │  ├──M_Net.py               // M network
        │   │  ├──network.py             // End-to-end network
        │   ├──dataset.py                // Load the dataset.
        │   ├──loss.py                   // Loss
        │   ├──metric.py                 // Metrics
        │   ├──config.py                 // Configuration parsing
        │   ├──callback.py               // Callback function
        │   ├──generate_datasets.py      // Create a dataset.
        │   ├──load_model.py             // Load the model.
        │   ├──get_init_weight.py        // Obtain the initialization weight file.
        ├── train.py                     // Training script
        ├── eval.py                      // Evaluation script
        ├── export.py                    // Export the checkpoint file to a MindIR or AIR model.
        ├── preprocess.py                // Script for pre-processing before inference on Ascend 310 AI Processors
        ├── postprogress.py                 // Script for post-processing after inference on Ascend 310 AI Processors
        ├── config.yaml                  // Configuration file
```

## Script Parameters

### Dataset Creation and Configuration

Dataset configurations are stored in the `generate_data` field in the `config.yaml` file. The dataset configurations include:

1. Save path: `path_save`

2. Path of the debugging dataset: `path_debug`

3. `Proportions` of the training set, validation set, and test set: `6:2:2`. You can adjust the proportions as required.

4. Size of the generated debugging dataset: `debug_pic_nums`

5. For the initial running, set the following parameters to `True`. You can adjust the parameters as required.

    ```text
    copy_pic: True                               # Specifies whether to copy images from the downloaded dataset to the path for storing the generated dataset.
    generate_mask: True                          # Specifies whether to generate a mask image set.
    generate_txt: True                           # Specifies whether to generate a TXT file listing the paths of images used for training, evaluation, and testing.
    generate_trimap: True                        # Specifies whether to generate a TriMap image set.
    fixed_ksize: True                            # Specifies whether to set the kernel size to a fixed value or random disturbance when a TriMap image set is generated.
    generate_alpha: True                         # Specifies whether to generate an alpha image set.
    generate_debug: True                         # Specifies whether to generate a debugging image set.
    generate_mean_std: True                      # Specifies whether to calculate the mean value and variance of the training set and validation set.
    ```

6. `kernel size` policy: `kernel size` is the window kernel size when the `alpha` channel of the image is corroded and dilated during the generation of the `TriMap` image set. When `fixed_ksize` is set to `True`, `kernel size` is fixed. When `fixed_ksize` is set to `False`, `kernel size` changes randomly.

    The detailed configuration is as follows:

    ```text
    generate_data:
    path_mt_human: /cache/human_matting          # Downloaded dataset
    path_save: /cache/datasets                   # Path for storing the generated dataset
    path_debug: /cache/datasets_debug            # Path for storing the generated debugging dataset

    proportion: '6:2:2'                          # Proportions of the training set, validation set, and test set
    debug_pic_nums: 400                          # Total number of images in the debugging set
    copy_pic: True                               # Specifies whether to copy images from the downloaded dataset to the path for storing the generated dataset.
    generate_mask: True                          # Specifies whether to generate a mask image set.
    generate_txt: True                           # Specifies whether to generate a TXT file listing the paths of images used for training, evaluation, and testing.
    generate_trimap: True                        # Specifies whether to generate a TriMap image set.
    ksize: 10                                    # Kernel size
    fixed_ksize: True                            # Specifies whether to set the kernel size to a fixed value or random disturbance when a TriMap image set is generated.
    generate_alpha: True                         # Specifies whether to generate an alpha image set.
    generate_debug: True                         # Specifies whether to generate a debugging image set.
    generate_mean_std: True                      # Specifies whether to calculate the mean value and variance of the training set and validation set.

    # Add error files in the downloaded dataset here for filtering during code execution.
    list_error_files: ['/cache/human_matting/matting/1803201916/._matting_00000000',
                        '/cache/human_matting/clip_img/1803241125/clip_00000000/._1803241125-00000005.jpg']
    ```

### Training Configuration

Configure the training in the `config.yaml` file. There are different training phases, which can be configured in the `pre_train_t_net`, `pre_train_m_net`, and `end_to_end` fields.

```text
# Training configuration
seed: 9527                          # Set the random seed.
rank: 0                             # Serial number of the device used for training. (In the case of single-device training, the value is the configured value. In the case of multi-device training, the value is automatically set to the serial number of the current device.)
group_size: 8                       # Total number of devices used for distributed training
device_target: 'Ascend'             # Specify the device. Currently, only Ascend AI Processors are supported.
saveIRFlag: False                   # Save the IR image.
ckpt_version: ckpt_s2               # Version of the saved .ckpt file

pre_train_t_net:                    # T-Net training configuration
  rank: 0                           # Automatically updated. It comes from the preceding rank configuration to adapt to the rank configuration in the current T-Net training phase.
  group_size: 8                     # Automatically updated. It comes from the preceding group_size configuration to adapt to the group_size configuration in the current T-Net training phase.
  finetuning: True                  # Load the pre-trained model.
  nThreads: 4                       # Number of threads for loading datasets
  train_batch: 8                    # Batch size
  patch_size: 320                   # Patch size. The value is fixed to 320.
  lr: 1e-3                          # Learning rate
  nEpochs: 1000                     # Total number of epochs
  save_epoch: 1                     # Interval (in epochs) for saving a CKPT file
  keep_checkpoint_max: '10'         # Maximum number of CKPT files that can be saved. The value can be 'all', '0', '1', or '2'.
  train_phase: pre_train_t_net      # Current training phase

pre_train_m_net:                    # M-Net training configuration
  rank: 0
  group_size: 8
  finetuning: True
  nThreads: 1
  train_batch: 8
  patch_size: 320
  lr: 1e-4
  nEpochs: 200
  save_epoch: 1
  keep_checkpoint_max: '10'
  train_phase: pre_train_m_net

end_to_end:                         # End-to-end training configuration
  rank: 0
  group_size: 8
  finetuning: True
  nThreads: 1
  train_batch: 8
  patch_size: 320
  lr: 1e-4
  nEpochs: 200
  save_epoch: 1
  keep_checkpoint_max: '10'
  train_phase: end_to_end
```

### Test, Inference, and Model Export Configurations

You can also perform the following configurations in the `config.yaml` file:

- Testing: Configure the `test` field.

- Inference: Configure the `infer` field.

- Model export: Configure the `export` field.

The detailed configuration is as follows:

```text
# Testing Configuration
test:
  device_target: 'Ascend'                                                        # Specify the inference device.
  model: /cache/output/distribute/ckpt_s2/end_to_end/semantic_hm_best.ckpt       # Path of the checkpoint file after training is complete
  test_pic_path: /cache/datasets                                                 # Directory of the generated datasets
  output_path: /cache/output/distribute/test_result                              # Directory for storing test results
  size: 320

# Inference configuration
infer:
  file_test_list: /cache/datasets/test/test.txt                                  # text.txt file generated after the dataset is generated
  size: 320                                                                      # Patch size. The value is fixed to 320.

# Model export configuration
export:
  ckpt_file: /cache/output/distribute/ckpt_s2/end_to_end/semantic_hm_best.ckpt   # Path of the checkpoint file after training is complete
  file_name: shm_export                                                          # Name of the exported model file
  file_format: 'MINDIR'                                                          # Specify the model export format.
  device_target: 'Ascend'                                                        # Specify the device.
```

## Dataset Creation Process

### Creating a Dataset

- Download the dataset.

    [Matting Human Datasets](<https://github.com/aisegmentcn/matting_human_datasets>)

- Configure `config.yaml`.

    Set the `generate_data` field.

- Run the following command:

    ```bash
    python3 generate_datasets.py --yaml_path=../config.yaml
    ```

- View the output.

    - Output directory structure:

    ```text
    ├── /cache/datasets
        ├── train                        // Training set
            ├── alpha                    // Alpha image set
            ├── clip_img                 // Extract data from the clip_img directory of the downloaded dataset.
            ├── mask                     // Generated mask image set
            ├── matting                  // Extract data from the matting directory of the downloaded dataset.
            ├── trimap                   // TriMap image set
            ├── train.txt                // List the images used for training.
        ├── eval                         // Validation set, which is used for real-time evaluation after a training epoch is complete.
            ├── alpha
            ├── clip_img
            ├── mask
            ├── matting
            ├── trimap
            ├── eval.txt                  // List the images used for validation.
        ├── test                          // Test set
            ├── alpha
            ├── clip_img
            ├── mask
            ├── matting
            ├── trimap
            ├── test.txt                  // List the images used for testing.
    ```

    During training, **train** is used for training. After each training epoch is complete, **eval** is used for validation. (No validation is performed in the T-Net training phase. Validation is performed in the M-Net training phase (this phase is disabled by default during training) and End-to-End training phase.) Data in the **test** directory is used for testing the inference results.

    - Output logs:

    In addition to the training set, validation set, and test set, the following information is also printed:

    ```text
    Namespace(yaml_path='../config.yaml')
    Copying source files to train dir...
    Copying source files to eval dir...
    Copying source files to test dir...
    Generate mask ...
    Generate datasets txt ...
    Generate trimap ...
    Generate alpha ...
    Copying train mask into alpha...
    Copying eval mask into alpha...
    Copying test mask into alpha...
    Generate datasets_debug ...
    Generate datasets_debug txt ...
    Generate train and eval datasets mean/std ...
    Total images: 27540
    mean_clip: [0.40077, 0.43385, 0.49808] [102, 110, 127]
    std_clip: [0.24744, 0.24859, 0.26404] [63, 63, 67]
    mean_trimap: [0.56147, 0.56147, 0.56147] [143, 143, 143]
    std_trimap: [0.47574, 0.47574, 0.47574] [121, 121, 121]
    ```

## Obtaining the Initialization Weight File

### Obtaining the Initialization Weight

1. Obtain the source code of the Torch version from GitHub: https://github.com/lizhengwei1992/Semantic_Human_Matting.

2. Copy the `get_init_weight.py` file in the `src` folder to the `root directory of the source code of the Torch version`.

    ```text
    ├── shm_original_code         // Source code of the Torch version
        ├── data
        │   ├── data.py
        │   ├── gen_trimap.py
        │   ├── gen_trimap.sh
        │   ├── knn_matting.py
        │   ├── knn_matting.sh
        ├── model
        │   ├── M_Net.py
        │   ├── network.py
        │   ├── N_Net.py
        ├── network.png
        ├── README.md
        ├── test_camera.py
        ├── test_camera.sh
        ├── train.py
        ├── train.sh
        ├── get_init_weight.py    // Copy the file here.
    ```

3. Run the following command:

    ```text
    python3 get_init_weight.py
    ```

4. View the output.

    After the command is executed, the init_weight.ckpt initialization weight file is generated in the root directory of the source code of the Torch version. You can copy the **initialization weight file** to the **root directory of the project**.

## Training Process

### Training

1. Configure `config.yaml`.

    Set the `pre_train_t_net`, `pre_train_m_net`, `train_phase`, and other fields. For details, see `Training Configuration` in [Script Parameters](#script-parameters).

2. Run the following command:

    ```text
    python3 train.py --yaml_path=[YAML_PATH] --data_url=[DATASETS] --train_url=[OUTPUT] --init_weight=[INIT_WEIGHT][OPTIONAL] > train.log 2>&1 &
    # Example:  python3 train.py --yaml_path=./config.yaml --data_url=/cache/datasets --train_url=/cache/output --init_weight=./init_weight.ckpt > train.log 2>&1 &
    ```

    The preceding Python command is executed in the backend. You can view the result in the `train.log` file.

    During the training, the following information such as the loss value and speed (in seconds) is output:

    ```text
    train epoch: 553 step: 1, loss: 0.022731708, speed: 0.14632129669189453
    train epoch: 553 step: 2, loss: 0.016247142, speed: 0.22849583625793457
    train epoch: 553 step: 3, loss: 0.015720012, speed: 0.19010353088378906
    ...
    ```

    The model checkpoint file is saved in the `/cache/output/single/ckpt_s2 (ckpt_version option in the YAML configuration file)` directory. The final inference model is as follows:
    `/cache/output/single/ckpt_s2/end_to_end/semantic_hm_best.ckpt`

3. Structure of the training output directory:

    ```text
    ├── /cache/output/[single or distribute]/ckpt_s2/
        ├── pre_train_t_net                       // Directory for storing model files and logs in the T-Net training phase
        │   ├── log_best.txt                      // Not used
        │   ├── log_latest.txt                   // Log in the T-Net training phase, including the loss information
        │   ├── semantic_hm_latest_1.ckpt        // Checkpoint file saved after a specified number of epochs
        │   ├── semantic_hm_latest_2.ckpt
        │   ├── semantic_hm_latest_3.ckpt
        │   ├── ···
        ├── pre_train_m_net                       // Directory for storing model files and logs in the M-Net training phase. This directory is generated if M-Net is added for training.
        │   ├── log_best.txt                      // Log in the M-Net training phase, including information about the best accuracy compared with those of the preceding epochs
        │   ├── log_latest.txt                   // Log in the M-Net training phase, including the loss information
        │   ├── semantic_hm_best.ckpt             // Check point file with the best accuracy, as the previous checkpoint files are overwritten.
        │   ├── semantic_hm_latest_1.ckpt        // Checkpoint file saved after a specified number of epochs
        │   ├── semantic_hm_latest_2.ckpt
        │   ├── semantic_hm_latest_3.ckpt
        │   ├── ···
        ├── end_to_end                            // Directory for storing model files and logs in the End-to-End training phase
        │   ├── log_best.txt
        │   ├── log_latest.txt
        │   ├── semantic_hm_best.ckpt
        │   ├── semantic_hm_latest_1.ckpt
        │   ├── semantic_hm_latest_2.ckpt
        │   ├── semantic_hm_latest_3.ckpt
        │   ├── ···
        ├── log_best.txt                          // Log of all the training phases, including information about the best accuracy compared with those of the preceding epochs
        ├── log_latest.txt                        // Log of all the training phases, including the loss information
    ```

4. Notice

    In the `End-to-End` training phase (or `M-Net` training phase), when the `Sad` indicator in the output log is the `patch_size` (option configured in the YAML file), the predicted alpha and ground truth are used to calculate the `Sad` indicator. However, in the inference phase, the `Sad` indicator is calculated by resizing the image to its original size.

### Distributed Training

1. Generate the JSON configuration file for distributed training.

    For distributed training, you need to create an HCCL configuration file in JSON format in advance. You can name the file `hccl_8p.json` and save it to the root directory of the current project.
    Follow the instructions in the following link: <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

2. Configure `config.yaml`.

    Set the `pre_train_t_net`, `pre_train_m_net`, `train_phase`, and other fields. For details, see `Training Configuration` in [Script Parameters](#script-parameters).

3. Run the following command:

    ```bash
    bash run_train.sh [RANK_TABLE_FILE] [YAML_PATH] [DATASETS] [OUTPUT] [INIT_WEIGHT][OPTIONAL]
    # example: bash run_train.sh ../hccl_8p.json ../config.yaml /cache/datasets /cache/output ../init_weight.ckpt
    ```

4. View the output.

   The preceding shell script will run distributed training in the backend. The training output `OUTPUT/distribute/..` is similar to the training output `OUTPUT/single/..` in the previous section.

   - View logs.

   You can run the `tail -f device0/train.log` command to view the result. You can also open the `device[0-7]/train.log` file to view the information.

   - Find the generated model.

   The model checkpoint file is saved in the `/cache/output/distribute/ckpt_s2 (ckpt_version option in the YAML configuration file)` directory. The final inference model is as follows:
   `/cache/output/distribute/ckpt_s2/end_to_end/semantic_hm_best.ckpt`

## Evaluation Process

### Evaluation

1. Configure `config.yaml`.

    Set the `test` parameters.

2. Run the following command:

    ```bash
    python3 eval.py --yaml_path=[YAML_PATH] > eval.log 2>&1 &
    # example: python3 eval.py --yaml_path=./config.yaml > eval.log 2>&1 &
    ```

    Or

    ```bash
    bash run_eval.sh [DEVICE_ID][OPTIONAL]
    # example: bash run_eval.sh
    ```

3. View the output.

    The preceding Python command is executed in the backend. You can view the result in the **eval.log** file. The accuracy of the test dataset is as follows:

    ```text
    # grep "ave_sad: " ./eval.log
    ave_sad: 5.4309
    ```

## Export Process

### Export

Export the model before inference. `mindir` models can be exported in any environment. The value of `batch_size` can only be 1.

1. Configure `config.yaml`.
    Set the `export` parameters.

2. Run the following command:

    ```text
    python3 export.py --config_path=[CONFIG_PATH]
    # example: python3 export.py --config_path=./config.yaml
    ```

3. View the output.
    After the command is executed, the converted model file `shm_export.mindir` is generated in the current directory.

## Inference Process

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

### Inference

1. Configure `config.yaml`.

    Set the `infer` parameters. The items to be modified include `file_test_list` (test set file list, absolute path of the .txt file) and `size` (image size of the input network).

2. Run the following command:

    ```bash
    bash run_infer_310.sh [MINDIR_PATH]
    # [MINDIR_PATH]: path of the exported model file configured in the previous step
    # example: bash run_infer_310.sh ../shm_export.mindir
    ```

3. View the output.

    The inference result is saved in the current directory.

    ```text
    ├── scripts
        ├── preprocess_Result                                  // Directory for storing the preprocessing results
        │   ├── clip_data                                      // Directory for storing original images
        │   │   ├── matting_0000_1803280628-00000477.jpg       // Naming rule of the original images: matting_[Four-digit image ID]_[Original dataset image name]
        │   │   ├── matting_0001_1803280628-00000478.jpg
        │   │   ├── ···
        │   ├── img_data                                       // Directory for storing the bin data that can be input to the network after the original images are preprocessed
        │   │   ├── matting_0000_1803280628-00000477.bin
        │   │   ├── matting_0001_1803280628-00000478.bin
        │   │   ├── ···
        │   ├── label                                          // Label directory
        │   │   ├── matting_0000_1803280628-00000477.png
        │   │   ├── matting_0001_1803280628-00000478.png
        │   │   ├── ···
        ├── result_Files                                       // Output directory of model inference
        │   │   ├── matting_0000_1803280628-00000477_0.bin     // First value output by the model: trimap
        │   │   ├── matting_0000_1803280628-00000477_1.bin     // Second value output by the model: alpha
        │   │   ├── matting_0001_1803280628-00000478_0.bin
        │   │   ├── matting_0001_1803280628-00000478_1.bin
        │   │   ├── ···
        ├── postprocess_Result                                 // Post-processing output directory. You can view the model inference effect in this directory.
        │   │   ├── matting_0000_1803280628-00000477.jpg
        │   │   ├── matting_0001_1803280628-00000478.jpg
        │   │   ├── ···
        ├── time_Result                                        // Directory for saving the time consumed by inference
        │   │   ├── test_perform_static.txt
        ├── infer.log                                          // Log of the model inference process
        ├── infer.log                                          // Accuracy log
    ```

The time required for inference is recorded in the `test_perform_static.txt` file in the `time_Result` directory.

```bash
# grep "time" ./time_Result/test_perform_static.txt
NN inference cost average time: 102.869 ms of infer_count 6901
```

You can find results similar to the following in the `acc.log` file:

```bash
# grep "ave sad: " ./acc.log
Total images: 6901, total sad: 38133.463921038725, ave sad: 5.525788135203409
```

# Model Description

## Performance

The following performance is obtained when the initialization weight file `init_weight.ckpt` is loaded.

### Training Performance

| Parameter         | Ascend                                                                                                                                           |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| Model version     | Semantic Human Matting V1                                                                                                                        |
| Resources         | Ascend 910 AI Processor; 2.60 GHz CPU with 192 cores; 755 GB memory; EulerOS 2.8                                                                                        |
| Upload date     | 2022-01-10                                                                                                                                       |
| MindSpore version| 1.6.0                                                                                                                                            |
| Dataset       | human matting dataset                                                                                                                            |
| Training parameters     | T-Net: epoch=1000, steps=320, batch\_size = 8, lr=1e-3, nThreads=1; <br> End-to-End: epoch=200, steps=320, batch\_size = 8, lr=1e-4, nThreads=1; |
| Optimizer       | Adam                                                                                                                                             |
| Loss function     | Softmax cross entropy, absolute error                                                                                                                         |
| Output         | T-Net: probability, End-to-End: Sad indicator                                                                                                               |
| Speed         | 8-device: T-Net: 453.35 ms/step; End-to-End: 693.10 ms/step                                                                                        |
| Total duration       | 8-device: 52h37m12s                                                                                                                                  |
| Finetuned checkpoint   | 12.03 MB (.ckpt)                                                                                                                              |
| Inference model     | 16.56 MB (.mindir)                                                                                                                             |
| Script         |                                                                                                                                                  |

### Evaluation Performance

| Parameter          | Ascend                    |
|----------------|---------------------------|
| Model version      | Semantic Human Matting V1 |
| Resources          | Ascend 910                |
| Upload date      | 2022-01-10                |
| MindSpore version| 1.6.0                     |
| Dataset        | human matting dataset     |
| batch_size     | 8                         |
| Output          | Sad indicator of the matting graph       |
| Accuracy        | 8-device: 5.4309              |
| Inference model      | 16.56 MB (.mindir)     |

# Random Seed Description

- The seed in the `create_dataset` function is set in `dataset.py`, and the random seed in `train.py` is also used.

# ModelZoo Home Page

For details, please go to the [official website](https://gitee.com/mindspore/models).
