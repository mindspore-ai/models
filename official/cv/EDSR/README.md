# Contents

<!-- TOC -->

- [Contents](#contents)
- [EDSR Description](#edsr-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Export](#export)
        - [Export Script](#export-script)
    - [Inference Process](#inference-process)
        - [Inference](#inference)
            - [Use the DIV2K dataset for inference on Ascend 310 AI Processors.](#use-the-div2k-dataset-for-inference-on-ascend-310-ai-processors)
            - [Use another dataset for inference on Ascend 310 AI Processors.](#use-another-dataset-for-inference-on-ascend-310-ai-processors)
            - [Perform ONNX inference.](#perform-onnx-inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
            - [Train 2x/3x/4x super-resolution reconstruction EDSR on DIV2K.](#train-2x3x4x-super-resolution-reconstruction-edsr-on-div2k)
        - [Evaluation Performance](#evaluation-performance)
            - [Evaluate 2x/3x/4x super-resolution reconstruction EDSR on DIV2K.](#evaluate-2x3x4x-super-resolution-reconstruction-edsr-on-div2k)
        - [Inference Performance](#inference-performance)
            - [Infer 2x/3x/4x super-resolution reconstruction EDSR on DIV2K.](#infer-2x3x4x-super-resolution-reconstruction-edsr-on-div2k)
- [Random Seed Description](#random-seed-description)
- [ModelZoo Home Page](#modelzoo-home-page)

<!-- /TOC -->

# EDSR Description

EDSR is a single-image super-resolution reconstruction network proposed in 2017. It won the first place in NTIRE 2017 Challenge on Single-Image Super-Resolution. It expands the model size by deleting unnecessary modules (BatchNorm) in the traditional residual network, and optimizes the model by using a stable training method, thereby significantly improving performance.

Paper: [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf): Lim B, Son S, Kim H, et al.Enhanced Deep Residual Networks for Single Image Super-Resolution[C]// 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, 2017.

# Model Architecture

The EDSR is formed by connecting multiple optimized residual blocks in series. Compared with the original network, the BatchNorm layer and the last ReLU layer are deleted from the residual blocks in EDSR. Deleting BatchNorm reduces the network memory usage by 40% and achieves faster computing, thereby increasing the network depth and width. The trunk mode of the convolutional layer is formed by stacking 32 residual blocks, a quantity of convolution kernels of each convolutional layer is 256, residual scaling is 0.1, and a loss function is L1.

# Dataset

Used dataset: [DIV2K](<https://data.vision.ee.ethz.ch/cvl/DIV2K/>)

- Dataset size: 7.11 GB, 1000 color images (HR, LRx2, LRx3, and LRx4)
    - Training set: 6.01 GB, 800 images
    - Validation set: 783.68 MB, 100 images
    - Test set: 349.53 MB, 100 images (no HR images)
- Data format: PNG
    - Note: Data is processed in **src/dataset.py**.
- After downloading data from the official website, decompress the package. The data directory structure required for training and verification is as follows:

```shell
├─DIV2K_train_HR
│  ├─0001.png
│  ├─...
│  └─0800.png
├─DIV2K_train_LR_bicubic
│  ├─X2
│  │  ├─0001x2.png
│  │  ├─...
│  │  └─0800x2.png
│  ├─X3
│  │  ├─0001x3.png
│  │  ├─...
│  │  └─0800x3.png
│  └─X4
│     ├─0001x4.png
│     ├─...
│     └─0800x4.png
├─DIV2K_valid_LR_bicubic
│  ├─0801.png
│  ├─...
│  └─0900.png
└─DIV2K_valid_LR_bicubic
   ├─X2
   │  ├─0801x2.png
   │  ├─...
   │  └─0900x2.png
   ├─X3
   │  ├─0801x3.png
   │  ├─...
   │  └─0900x3.png
   └─X4
      ├─0801x4.png
      ├─...
      └─0900x4.png
```

# Features

## Mixed Precision

[Mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html?highlight=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6) accelerates the training process of deep neural networks by using the single-precision (FP32) data and half-precision (FP16) data without compromising the precision of networks trained with single-precision (FP32) data. It not only accelerates the computing process and reduces the memory usage, but also supports a larger model or batch size to be trained on specific hardware.
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

After installing MindSpore from the official website, you can perform the following steps for training and evaluation. For distributed training, you need to create an HCCL configuration file in JSON format in advance. Follow the instructions in the following link:
 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

- Training DIV2K on a single device (Ascend 910 AI Processor)

  ```python
  # Training example (EDSR (x2) in the Paper)
  python train.py --batch_size 16 --config_path DIV2K_config.yaml --scale 2 --data_path [DIV2K path] --output_path [path to save .ckpt] > train.log 2>&1 &
  # Training example (EDSR(x3) in the paper - from EDSR(x2))
  python train.py --batch_size 16 --config_path DIV2K_config.yaml --scale 3 --data_path [DIV2K path] --output_path [path to save .ckpt] --pre_trained [pre-trained EDSR_x2 model path] train.log 2>&1 &
  # Training example (EDSR(x4) in the paper - from EDSR(x2))
  python train.py --batch_size 16 --config_path DIV2K_config.yaml --scale 4 --data_path [DIV2K path] --output_path [path to save .ckpt] --pre_trained [pre-trained EDSR_x2 model path] train.log 2>&1 &
  ```

- Training DIV2K on 8 devices (Ascend 910 AI Processors)

  ```python
  # Distributed training example (EDSR (x2) in the Paper)
  bash scripts/run_train.sh rank_table.json --config_path DIV2K_config.yaml --scale 2 --data_path [DIV2K path] --output_path [path to save .ckpt]
  # Distributed training example (EDSR (x3) in the Paper)
  bash scripts/run_train.sh rank_table.json --config_path DIV2K_config.yaml --scale 3 --data_path [DIV2K path] --output_path [path to save .ckpt] --pre_trained [pre-trained EDSR_x2 model path]
  # Distributed training example (EDSR (x4) in the Paper)
  bash scripts/run_train.sh rank_table.json --config_path DIV2K_config.yaml --scale 4 --data_path [DIV2K path] --output_path [path to save .ckpt] --pre_trained [pre-trained EDSR_x2 model path]
  ```

- Evaluating DIV2K on a single device (Ascend 910 AI Processor)

  ```python
  # Evaluation example (EDSR (x2) in the Paper)
  python eval.py --config_path DIV2K_config.yaml --scale 2 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x2 model path] > train.log 2>&1 &
  # Evaluation example (EDSR (x3) in the Paper)
  python eval.py --config_path DIV2K_config.yaml --scale 3 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x3 model path] > train.log 2>&1 &
  # Evaluation example (EDSR (x4) in the Paper)
  python eval.py --config_path DIV2K_config.yaml --scale 4 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x4 model path] > train.log 2>&1 &
  ```

- Evaluating DIV2K on 8 devices (Ascend 910 AI Processors)

  ```python
  # Distributed evaluation example (EDSR (x2) in the Paper)
  bash scripts/run_eval.sh rank_table.json --config_path DIV2K_config.yaml --scale 2 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x2 model path]
  # Distributed evaluation example (EDSR (x3) in the Paper)
  bash scripts/run_eval.sh rank_table.json --config_path DIV2K_config.yaml --scale 3 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x3 model path]
  # Distributed evaluation example (EDSR (x4) in the Paper)
  bash scripts/run_eval.sh rank_table.json --config_path DIV2K_config.yaml --scale 4 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x4 model path]
  ```

- Evaluating benchmark on a single device (Ascend 910 AI Processor)

  ```python
  # Evaluation example (EDSR (x2) in the Paper)
  python eval.py --config_path benchmark_config.yaml --scale 2 --data_path [benchmark path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x2 model path] > train.log 2>&1 &
  # Evaluation example (EDSR (x3) in the Paper)
  python eval.py --config_path benchmark_config.yaml --scale 3 --data_path [benchmark path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x3 model path] > train.log 2>&1 &
  # Evaluation example (EDSR (x4) in the Paper)
  python eval.py --config_path benchmark_config.yaml --scale 4 --data_path [benchmark path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x4 model path] > train.log 2>&1 &
  ```

- Evaluating benchmark on 8 devices (Ascend 910 AI Processors)

  ```python
  # Distributed evaluation example (EDSR (x2) in the Paper)
  bash scripts/run_eval.sh rank_table.json --config_path benchmark_config.yaml --scale 2 --data_path [benchmark path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x2 model path]
  # Distributed evaluation example (EDSR (x3) in the Paper)
  bash scripts/run_eval.sh rank_table.json --config_path benchmark_config.yaml --scale 3 --data_path [benchmark path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x3 model path]
  # Distributed evaluation example (EDSR (x4) in the Paper)
  bash scripts/run_eval.sh rank_table.json --config_path benchmark_config.yaml --scale 4 --data_path [benchmark path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x4 model path]
  ```

- Evaluating DIV2K on a single device (Ascend 310 AI Processor)

  ```python
  # Inference command
  bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [SCALE] [LOG_FILE] [DEVICE_ID]
  # Inference example (EDSR (x2) in the Paper)
  bash scripts/run_infer_310.sh ./mindir/EDSR_x2_DIV2K-6000_50_InputSize1020.mindir ./DIV2K 2 ./infer_x2.log 0
  # Inference example (EDSR (x3) in the Paper)
  bash scripts/run_infer_310.sh ./mindir/EDSR_x3_DIV2K-6000_50_InputSize680.mindir ./DIV2K 3 ./infer_x3.log 0
  # Inference example (EDSR (x4) in the Paper)
  bash scripts/run_infer_310.sh ./mindir/EDSR_x4_DIV2K-6000_50_InputSize510.mindir ./DIV2K 4 ./infer_x4.log 0
  ```

- Training the DIV2K dataset on ModelArts

  If you want to operate on ModelArts, refer to the following document: [ModelArts](https://support.huaweicloud.com/modelarts/)

  ```python
  # (1) Upload code to the S3 bucket.
  #     Select the code directory /s3_path_to_code/EDSR/.
  #     Select the startup file /s3_path_to_code/EDSR/train.py.
  # (2) Set parameters on the web page. All parameters in DIV2K_config.yaml can be set on the web page.
  #     scale = 2
  #     config_path = /local_path_to_code/DIV2K_config.yaml
  #     enable_modelarts = True
  #     pre_trained = [model_s3_address] or [not set]
  #     [Other parameter] = [Value]
  # (3) Upload the DIV2K dataset to the S3 bucket and configure the training set path. If the dataset is not decompressed, configure the path in step (2).
  #     need_unzip_in_modelarts = True
  # (4) Set training output file path and job log path on the web page.
  # (5) Select an 8-device or single-device machine and create a training job.
  ```

# Script Description

## Script and Sample Code

```text
├── model_zoo
    ├── README.md                       // Description of all models
    ├── EDSR
        ├── README_CN.md                // EDSR Description
        ├── model_utils                 // Tool script for cloud migration
        ├── DIV2K_config.yaml           // EDSR parameters
        ├── scripts
        │   ├──run_train.sh             // Shell script for distributed training on Ascend AI Processors
        │   ├──run_eval.sh              // Shell script for evaluation on Ascend AI Processors
        │   ├──run_infer_310.sh         // Shell script for inference on Ascend 310 AI Processors
        │   └── run_eval_onnx.sh        // Shell script for ONNX evaluation
        ├── src
        │   ├──dataset.py               // Dataset creation
        │   ├──edsr.py                  // EDSR network architecture
        │   ├──config.py                // Parameter configuration
        │   ├──metric.py                // Evaluation metrics
        │   ├──utils.py                 // Common code segment of the train.py or eval.py
        ├── train.py                    // Training script
        ├── eval.py                     // Evaluation script
        ├── eval_onnx.py                // ONNX evaluation script
        ├── export.py                   // Export the checkpoint file to an ONNX, MindIR, or AIR model.
        ├── preprocess.py               // Script for preprocessing inference data on Ascend 310 AI Processors
        ├── ascend310_infer
        │   ├──src                      // Ascend 310 AI Processor inference source code
        │   ├──inc                      // Ascend 310 AI Processor inference source code
        │   ├──build.sh                 // Shell script for building the Ascend 310 inference program
        │   ├──CMakeLists.txt           // CMakeLists for building the Ascend 310 inference program
        ├── postprocess.py               // Script for postprocessing inference data on Ascend 310 AI Processors
```

## Script Parameters

You can configure both training and evaluation parameters in **DIV2K_config.yaml**. Parameters with the same name in **benchmark_config.yaml** have the same definition.

- You can run the following statement to print the configuration description:

  ```python
  python train.py --config_path DIV2K_config.yaml --help
  ```

- You can directly view the configuration description in **DIV2K_config.yaml**. The description is as follows:

  ```yaml
  enable_modelarts: "Set this parameter to True if the model is running in the cloud channel. The default value is False."

  data_url: "Cloud channel data path"
  train_url: "Cloud channel code path"
  checkpoint_url: "Path for storing the cloud channel"

  data_path: "Data path of the running machine, which is downloaded from the cloud channel data path by the script. The default value is /cache/data."
  output_path: "Output path of the running machine. The script is uploaded from the local host to checkpoint_url. The default value is /cache/train."
  device_target: "Value range: ['Ascend']. The default value is Ascend."

  amp_level: "Value range: ['O0', 'O2', 'O3', 'auto']. The default value is O3."
  loss_scale: "Loss scaling is performed for mixed precisions except O0. The default value is 1000.0."
  keep_checkpoint_max: "Maximum number of CKPTs that can be saved. The default value is 60."
  save_epoch_frq: "Interval (in epochs) for saving a checkpoint. The default value is 100."
  ckpt_save_dir: "Local relative path. The root directory is output_path. The default value is ./ckpt/."
  epoch_size: "Number of epochs to be trained. The default value is 6000."

  eval_epoch_frq: "Validation interval (in epochs) during training. The default value is 20."
  self_ensemble: "Execute self_ensemble during validation. Only used in eval.py. The default value is True."
  save_sr: "Save SR and HR images during validation. Only used in eval.py. The default value is True."

  opt_type: "Optimizer type. Value range: ['Adam']. The default value is Adam."
  weight_decay: "Optimizer weight decay parameter. The default value is 0.0."

  learning_rate: "Learning rate. The default value is 0.0001."
  milestones: "List of epoch nodes whose learning rate is decayed. The default value is [4000]."
  gamma: "Learning rate decay rate. The default value is 0.5."

  dataset_name: "Dataset name. The default value is DIV2K."
  lr_type: "lr graph degradation mode. Value range: ['bicubic', 'unknown']. The default value is bicubic."
  batch_size: "Recommended values: 2 for eight devices and 16 for a single device. The default value is 2."
  patch_size: "Size of the clipped HR image during training. The size of the clipped LR image is adjusted based on the scale. The default value is 192."
  scale: "Scale of the super-resolution reconstruction of the model. Value range: [2,3,4]. The default value is 4."
  dataset_sink_mode: "Data offloading mode used for training. The default value is True."
  need_unzip_in_modelarts: "Download data from s3 and compresses the data. The default value is False."
  need_unzip_files: "List of data to be decompressed. This parameter is valid only when need_unzip_in_modelarts is set to True."

  pre_trained: "Load the pre-trained model. x2, x3, and x4 can be loaded mutually. Value range: [s3 absolute address], [relative address in output_path], [absolute address of the local machine], '']. The default value is ''."
  rgb_range: "Image pixel range. The default value is 255."
  rgb_mean: "Average RGB value of an image. The default value is [0.4488, 0.4371, 0.4040]."
  rgb_std: "Image RGB variance. The default value is [1.0, 1.0, 1.0]."
  n_colors: "3-channel RGB image. The default value is 3."
  n_feats: "Number of output features of each convolutional layer. The default value is 256."
  kernel_size: "Convolution kernel size. The default value is 3."
  n_resblocks: "Number of residual blocks. The default value is 32."
  res_scale: "Coefficient of the res branch. The default value is 0.1."
  ```

## Export

Export the model before inference. Air models can be exported only in the Ascend 910 AI Processor environment. MindIR and ONNX models can be exported in any environment. The value of batch_size can only be 1.
Note: To export ONNX, change **file_format = 'MINDIR'** in the **export.py** code to **file_format = 'ONNX'**.

### Export Script

```shell
python export.py --config_path DIV2K_config.yaml --output_path [dir to save model] --scale [SCALE] --pre_trained [pre-trained EDSR_x[SCALE] model path]
```

## Inference Process

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

### Inference

#### Use the DIV2K dataset for inference on Ascend 310 AI Processors

- Inference script

  ```shell
  bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [SCALE] [LOG_FILE] [DEVICE_ID]
  ```

- Example

  ```shell
  # Inference example (EDSR (x2) in the Paper)
  bash scripts/run_infer_310.sh ./mindir/EDSR_x2_DIV2K-6000_50_InputSize1020.mindir ./DIV2K 2 ./infer_x2.log 0
  # Inference example (EDSR (x3) in the Paper)
  bash scripts/run_infer_310.sh ./mindir/EDSR_x3_DIV2K-6000_50_InputSize680.mindir ./DIV2K 3 ./infer_x3.log 0
  # Inference example (EDSR (x4) in the Paper)
  bash scripts/run_infer_310.sh ./mindir/EDSR_x4_DIV2K-6000_50_InputSize510.mindir ./DIV2K 4 ./infer_x4.log 0
  ```

- Inference metrics, which can be viewed in **infer_x2.log**, **infer_x3.log**, and **infer_x4.log**.

  ```python
  # EDSR(x2) in the Paper
  evaluation result = {'psnr': 35.068791459971266}
  # EDSR(x3) in the Paper
  evaluation result = {'psnr': 31.386362838892456}
  # EDSR(x4) in the Paper
  evaluation result = {'psnr': 29.38072897971985}
  ```

#### Use another dataset for inference on Ascend 310 AI Processors

- Inference Process

  ```bash
  # (1) Sort the dataset and pad the LR images to a fixed size. For details, see preprocess.py.
  # (2) Export a model based on a fixed size. For details, see export.py.
  # (3) Use build.sh to compile the inference program in the ascend310_infer folder to obtain ascend310_infer/out/main.
  # (4) Configure the dataset image path, model path, and output path, and use main inference to obtain the super-resolution reconstructed image.
  ./ascend310_infer/out/main --mindir_path=[model] --dataset_path=[read_data_path] --device_id=[device_id] --save_dir=[save_data_path]
  # (5) Post-process images. Remove the invalid padding area. The images are used together with the HR images to collect statistics on metrics. For details, see preprocess.py.
  ```

#### Perform ONNX inference

- Inference process

  ```bash
  # (1) Sort the dataset and pad the LR images to a fixed size. For details, see preprocess.py.
  # (2) Export a model based on a fixed size. For details, see export.py.
  # (3) Run the inference script.
  ```

- ONNX evaluation on GPUs

  ```bash
  # X2 evaluation example (EDSR (x2) in the Paper)
  bash scripts/run_eval_onnx.sh ./DIV2K_config.yaml  2  DIV2K path output_path  pre_trained_model_path  ONNX
  # X3 evaluation example (EDSR (x3) in the Paper)
  bash scripts/run_eval_onnx.sh ./DIV2K_config.yaml  3  DIV2K path output_path  pre_trained_model_path  ONNX
  # X4 evaluation example (EDSR (x4) in the Paper)
  bash scripts/run_eval_onnx.sh ./DIV2K_config.yaml  2  DIV2K path output_path  pre_trained_model_path  ONNX
  ```

  The preceding Python command is executed in the backend. You can view the result in the **eval_onnx.log** file. The accuracy of the test dataset is as follows:

  ```bash
  .....
  [100/100] rank = 0 result = {'psnr': 29.297856984107398, 'num_sr': 100.0, 'time': 5.842652082443237}
  evaluation result = {'psnr': 29.297856984107398, 'num_sr': 100.0, 'time': 2905.9808044433594}
  eval success
  ```

# Model Description

## Performance

### Training Performance

#### Train 2x/3x/4x super-resolution reconstruction EDSR on DIV2K

| Parameter| Ascend | Ascend | Ascend |
| --- | --- | --- | --- |
| Model version| EDSR (x2)| EDSR (x3)| EDSR (x4)|
| Resources| Ascend 910 AI Processor, 2.60 GHz CPU with 192 cores, 755 GB memory, and EulerOS 2.8| Ascend 910 AI Processor, 2.60 GHz CPU with 192 cores, 755 GB memory, and EulerOS 2.8| Ascend 910 AI Processor, 2.60 GHz CPU with 192 cores, 755 GB memory, and EulerOS 2.8|
| Upload date| 2021-09-01 | 2021-09-01 | 2021-09-01 |
| MindSpore version| 1.2.0 | 1.2.0 | 1.2.0 |
| Dataset| DIV2K | DIV2K | DIV2K |
| Training parameters| epoch=6000, total batch_size=16, lr=0.0001, patch_size=192| epoch=6000, total batch_size=16, lr=0.0001, patch_size=192| epoch=6000, total batch_size=16, lr=0.0001, patch_size=192|
| Optimizer| Adam | Adam | Adam |
| Loss function| L1 | L1 | L1 |
| Output| Super-resolution reconstruction of RGB images| Super-resolution reconstruction of RGB images| Super-resolution reconstruction of RGB images|
| Loss| 4.06 | 4.01 | 4.50 |
| Speed| Single device: 16.5 s/epoch; 8 devices: 2.76 s/epoch| Single device: 21.6 s/epoch; 8 devices: 1.8 s/epoch| Single device: 21.0 s/epoch; 8 devices: 1.8 s/epoch|
| Total duration| Single device: 1725 minutes; 8 devices: 310 minutes| Single device: 2234 minutes; 8 devices: 217 minutes| Single device: 2173 minutes; 8 devices: 210 minutes|
| Parameters (M)| 40.73M | 43.68M | 43.09M |
| Finetuned checkpoint| 467.28 MB (.ckpt)| 501.04 MB (.ckpt)| 494.29 MB (.ckpt)|

### Evaluation Performance

#### Evaluate 2x/3x/4x super-resolution reconstruction EDSR on DIV2K

| Parameter| Ascend | Ascend | Ascend |
| --- | --- | --- | --- |
| Model version| EDSR (x2)| EDSR (x3)| EDSR (x4)|
| Resources| Ascend 910 AI Processor; EulerOS 2.8| Ascend 910 AI Processor; EulerOS 2.8| Ascend 910 AI Processor; EulerOS 2.8|
| Upload date| 2021-09-01 | 2021-09-01 | 2021-09-01 |
| MindSpore version| 1.2.0 | 1.2.0 | 1.2.0 |
| Dataset| DIV2K, 100 images| DIV2K, 100 images| DIV2K, 100 images|
| self_ensemble | True | True | True |
| batch_size | 1 | 1 | 1 |
| Output| Super-resolution reconstruction of RGB images| Super-resolution reconstruction of RGB images| Super-resolution reconstruction of RGB images|
|     Set5 psnr | 38.275 db | 34.777 db | 32.618 db |
|    Set14 psnr | 34.059 db | 30.684 db | 28.928 db |
|     B100 psnr | 32.393 db | 29.332 db | 27.792 db |
| Urban100 psnr | 32.970 db | 29.019 db | 26.849 db |
|    DIV2K psnr | 35.063 db | 31.380 db | 29.370 db |
| Inference model| 467.28 MB (.ckpt)| 501.04 MB (.ckpt)| 494.29 MB (.ckpt)|

### Inference Performance

#### Infer 2x/3x/4x super-resolution reconstruction EDSR on DIV2K

| Parameter| Ascend | Ascend | Ascend |
| --- | --- | --- | --- |
| Model version| EDSR (x2)| EDSR (x3)| EDSR (x4)|
| Resources| Ascend 310 AI Processor; Ubuntu18.04| Ascend 310 AI Processor; Ubuntu18.04| Ascend 310 AI Processor; Ubuntu18.04|
| Upload date| 2021-09-01 | 2021-09-01 | 2021-09-01 |
| MindSpore version| 1.2.0 | 1.2.0 | 1.2.0 |
| Dataset| DIV2K, 100 images| DIV2K, 100 images| DIV2K, 100 images|
| self_ensemble | True | True | True |
| batch_size | 1 | 1 | 1 |
| Output| Super-resolution reconstruction of RGB images| Super-resolution reconstruction of RGB images| Super-resolution reconstruction of RGB images|
| DIV2K psnr | 35.068 db | 31.386 db | 29.380 db |
| Inference model| 156 MB (.mindir)| 167 MB (.mindir)| 165 MB (.mindir)|

# Random Seed Description

In **train.py** and **eval.py**, the mindspore.common.set_seed(2021) seed is set.

# ModelZoo Home Page

 For details, please go to the [official website](https://gitee.com/mindspore/models).
