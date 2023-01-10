Contents

<!-- TOC -->

- [Arcface Overview](#arcface-overview)
- [Datasets](#datasets)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
    - [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
        - [ONNX Evaluation](#onnx-evaluation)
    - [Exporting a MindIR Model](#exporting-a-mindir-model)
- [Inference Process](#inference-process)
    - [Usage](#usage)
        - [Results](#results)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Random Seed Description](#random-seed-description)
- [ModelZoo Home Page](#modelzoo-home-page)

<!-- /TOC -->

# Arcface Overview

One of the main challenges in feature learning using Deep Convolutional Neural Networks (DCNNs) for large-scale face recognition is the design of appropriate loss functions that enhance discriminative power. ArcFace outperforms SoftmaxLoss, Center Loss, A-Softmax Loss and Cosine Margin Loss in face recognition. ArcFace is improved in contrast to the conventional softmax. ArcFace has a clear geometric interpretation due to the exact correspondence to the geodesic distance on the hypersphere Based on experimental evaluation of all the recent state-of-the-art face recognition methods on over 10 benchmarks. It is proved that ArcFace consistently outperforms the state-of-the-art and can be easily implemented with negligible computational overhead.

[Paper](https://arxiv.org/pdf/1801.07698v3.pdf): Deng J ,  Guo J ,  Zafeiriou S . ArcFace: Additive Angular Margin Loss for Deep Face Recognition[J].  2018.

# Datasets

Training set: [MS1MV2](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)

Validation set: LFW, CFP-FP, AGEDB, CPLFW, CALFW, IJB-B, and IJB-C

Training set: 5,822,653 images of 85742 classes

```shell
#Convert the REC data format to JPG.
python src/rec2jpg_dataset.py --include rec/dataset/path --output output/path
```

Note: In the Arm environment, the [MXNet](https://mxnet.apache.org/versions/1.9.0/get_started/build_from_source.html) can run properly only after being built by the source code.

# Environment Requirements

- Hardware: Ascend AI Processors
    - Set up the hardware environment with Ascend AI Processors.

- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the following resources:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# Quick Start

After installing MindSpore from the official website, you can perform the following steps for training and evaluation:

- On Ascend AI Processors

  ```python
  # Example of distributed training
  bash scripts/run_distribute_train.sh rank_size /path/dataset

  # Example of standalone training
  bash scripts/run_standalone_train.sh /path/dataset device_id

  # Evaluation example
  bash scripts/run_eval.sh /path/evalset /path/ckpt
  ```

- On GPUs

  ```python
  # Example of distributed training
  bash scripts/run_distribute_train_gpu.sh  /path/dataset rank_size

  # Example of standalone training
  bash scripts/run_standalone_train_gpu.sh /path/dataset

  # Evaluation example
  bash scripts/run_eval_gpu.sh /path/evalset /path/ckpt
  ```

## Script Description

## Script and Sample Code

```path
└── Arcface  
 ├── README.md                           // Arcface description
 ├── ascend310_infer                     // Inference on Ascend 310 AI Processors
  ├── inc
   ├── utils.h
  ├── src
   ├── main.cc
   ├── utils.cc
  ├── build.sh
  └── CMakeLists.txt
 ├── scripts
  ├── run_310_infer.sh           // Shell script for inference on Ascend 310 AI Processors
  ├── run_distribute_train.sh    // Shell script for distributed training
  ├── run_standalone_train.sh    // Shell script for standalone training
  ├── run_eval_ijbc.sh           // Shell script for evaluation on the IJB-C dataset
  ├── run_eval.sh                // Shell script for evaluation
  ├── run_eval_ijbc_onnx.sh      // Shell script for ONNX evaluation on the IJB-C dataset
  └── run_eval_onnx.sh           // Shell script for ONNX evaluation
 ├──src
  ├── loss.py                         // Loss function
  ├── dataset.py                      // Dataset creation
  ├── iresnet.py                      // ResNet architecture
  ├── rec2jpg_dataset.py                  // Converting data from REC to JPG
 ├── val.py                            // Test script
 ├── train.py                          // Training script
 ├── export.py
 ├── requirements.txt
 ├── preprocess.py                    // Preprocessing inference data on Ascend 310 AI Processors
 ├── postprocess.py                    // Post-processing inference data on Ascend 310 AI Processors
 ├── eval_onnx.py                     // ONNX evaluation
 └── eval_ijbc_onnx.py                // ONNX evaluation on the IJB-C dataset

```

## Script Parameters

```python
Key parameters in train.py and val.py are as follows:

-- modelarts: Specifies whether to use the ModelArts platform for training. The options are True or False. The default value is False.
-- device_id: ID of the device used for dataset training or evaluation. When train.sh is used for distributed training, ignore this parameter.
-- device_num: Number of devices used for distributed training.
-- train_url: Output path of the checkpoint file.
-- data_url: Training dataset path.
-- ckpt_url: Checkpoint path.
-- eval_url: Validation dataset path.

```

## Training Process

### Distributed Training

- On Ascend AI Processors

  ```bash
  bash scripts/run_distribute_train.sh rank_size /path/dataset
  ```

  The preceding shell script will run distributed training in the backend. You can view the result in the `device[X]/train.log` file.
  The following methods are used to achieve the loss value:

  ```log
  epoch: 2 step: 11372, loss is 12.807039
  epoch time: 1104549.619 ms, per step time: 97.129 ms
  epoch: 3 step: 11372, loss is 9.13787
  ...
  epoch: 21 step: 11372, loss is 1.5028578
  epoch time: 1104673.362 ms, per step time: 97.140 ms
  epoch: 22 step: 11372, loss is 0.8846929
  epoch time: 1104929.793 ms, per step time: 97.162 ms
  ```

- On GPUs

  ```bash
  bash scripts/run_distribute_train_gpu.sh /path/dataset rank_size
  ```

  The preceding shell script will run distributed training in the backend. You can view the result in the `train_parallel/train.log` file.
  The following methods are used to achieve the loss value:

  ```log
  epoch: 2 step: 11372, loss is 10.572094
  epoch time: 1104549.619 ms, per step time: 991.390 ms
  epoch: 3 step: 11372, loss is 7.442794
  ...
  epoch: 21 step: 11372, loss is 0.8472798
  epoch time: 1104673.362 ms, per step time: 989.479 ms
  epoch: 22 step: 11372, loss is 0.5226351
  epoch time: 1104929.793 ms, per step time: 989.548 ms
  ```

## Evaluation Process

### Evaluation

- Evaluate the LFW, CFP-FP, AgeDB-30, CALFW, and CPLFW datasets on the Ascend AI Processors.

  Check the checkpoint path for evaluation before running the following command. Set the checkpoint path to an absolute full path, for example, `username/arcface/arcface-11372-1.ckpt`.

  ```bash
  bash scripts/run_eval.sh /path/evalset /path/ckpt
  ```

  The preceding Python command is executed in the backend. You can view the result in the **eval.log** file. The accuracy of the test set is as follows:

  ```bash
  [lfw]Accuracy-Flip: 0.99817+-0.00273
  [cfp_fp]Accuracy-Flip: 0.98000+-0.00586
  [agedb_30]Accuracy-Flip: 0.98100+-0.00642
  [calfw]Accuracy-Flip: 0.96150+-0.01099
  [cplfw]Accuracy-Flip: 0.92583+-0.01367
  ```

- Evaluate the IJB-B and IJB-C datasets on the Ascend AI Processors.

  Check the checkpoint path for evaluation before running the following command. Set the checkpoint path to an absolute full path, for example, `username/arcface/arcface-11372-1.ckpt`.

  Ensure that the input validation set path is `IJB_release/IJBB/` or `IJB_release/IJBC/`.

  ```bash
  bash scripts/run_eval_ijbc.sh /path/evalset /path/ckpt
  ```

  The preceding Python command is executed in the backend. You can view the result in the **eval.log** file. The accuracy of the test set is as follows:

  ```bash
  +-----------+-------+-------+--------+-------+-------+-------+
  |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
  +-----------+-------+-------+--------+-------+-------+-------+
  | ijbb-IJBB | 40.01 | 87.91 | 94.36  | 96.48 | 97.72 | 98.70 |
  +-----------+-------+-------+--------+-------+-------+-------+

  +-----------+-------+-------+--------+-------+-------+-------+
  |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
  +-----------+-------+-------+--------+-------+-------+-------+
  | ijbc-IJBC | 82.08 | 93.37 | 95.87  | 97.40 | 98.40 | 99.05 |
  +-----------+-------+-------+--------+-------+-------+-------+
  ```

- Evaluate the LFW, CFP-FP, AgeDB-30, CALFW, and CPLFW datasets on GPUs.

  Check the checkpoint path for evaluation before running the following command. Set the checkpoint path to an absolute full path, for example, `username/arcface/arcface-11372-1.ckpt`.

  ```bash
  bash scripts/run_eval_gpu.sh /path/evalset /path/ckpt
  ```

  The preceding Python command is executed in the backend. You can view the result in the **eval.log** file. The accuracy of the test set is as follows:

  ```bash
  [lfw]Accuracy-Flip: 0.99767+-0.00271
  [cfp_fp]Accuracy-Flip: 0.98414+-0.00659
  [agedb_30]Accuracy-Flip: 0.98033+-0.00878
  [calfw]Accuracy-Flip: 0.95983+-0.01141
  [cplfw]Accuracy-Flip: 0.92817+-0.01279
  ```

- Evaluate the IJB-B and IJB-C datasets on GPUs.

  Check the checkpoint path for evaluation before running the following command. Set the checkpoint path to an absolute full path, for example, `username/arcface/arcface-11372-1.ckpt`.

  Ensure that the input validation set path is `IJB_release/IJBB/` or `IJB_release/IJBC/`.

  ```bash
  bash scripts/run_eval_ijbc_gpu.sh /path/evalset /path/ckpt
  ```

  The preceding Python command is executed in the backend. You can view the result in the **eval.log** file. The accuracy of the test set is as follows:

  ```bash
  +-----------+-------+-------+--------+-------+-------+-------+
  |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
  +-----------+-------+-------+--------+-------+-------+-------+
  | ijbb-IJBB | 42.46 | 89.76 | 94.81  | 96.58 | 97.73 | 98.78 |
  +-----------+-------+-------+--------+-------+-------+-------+

  +-----------+-------+-------+--------+-------+-------+-------+
  |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
  +-----------+-------+-------+--------+-------+-------+-------+
  | ijbc-IJBC | 86.67 | 94.35 | 96.19  | 97.55 | 98.38 | 99.10 |
  +-----------+-------+-------+--------+-------+-------+-------+
  ```

### ONNX Evaluation

Click [here](https://www.mindspore.cn/resources/hub/details?MindSpore/1.6/arcface_ms1mv2) to obtain the CKPT file required for evaluation.

Before performing evaluation, you need to export the ONNX file using **export.py**.

'''python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --batch_size [BATCH_SIZE] --file_format ONNX
'''

- Use the ONNX file to evaluate the LFW, CFP-FP, AgeDB-30, CALFW, and CPLFW datasets on GPUs.

  ```bash
  bash run_eval_onnx.sh /path/evalset /path/onnx

  # Example of ONNX evaluation:
  python export.py --batch_size 64 --ckpt_file ./arcface_ascend_v160_ms1mv2_research_cv_IJBB97.67_IJBC98.36.ckpt --file_name onnx_64 --file_format ONNX --device_target GPU
  cd ./scripts/
  bash run_eval_onnx.sh ../dataset/ ../onnx_64.onnx
  ```

  The preceding Python command is executed in the backend. You can view the result in the **eval_onnx.log** file. The accuracy of the test set is as follows:

  ```bash
  [lfw]Accuracy-Flip: 0.99750+-0.00291
  [cfp_fp]Accuracy-Flip: 0.98500+-0.00643
  [agedb_30]Accuracy-Flip: 0.98050+-0.00778
  [calfw]Accuracy-Flip: 0.96117+-0.01118
  [cplfw]Accuracy-Flip: 0.92900+-0.01250
  ```

- Use th ONNX file to evaluate the IJB-B dataset on GPUs.

  Check the checkpoint file path used for evaluation before running the command below. Ensure that the input validation set path is `IJB_release/IJBB/`.

  ```bash
  bash run_eval_ijbc_onnx.sh /path/evalset path/onnx_bs path/onnx_rs IJBB

  # Example of ONNX evaluation:
  python export.py --batch_size 256 --ckpt_file ./arcface_ascend_v160_ms1mv2_research_cv_IJBB97.67_IJBC98.36.ckpt --file_name onnx_256 --file_format ONNX --device_target GPU
  python export.py --batch_size 92 --ckpt_file ./arcface_ascend_v160_ms1mv2_research_cv_IJBB97.67_IJBC98.36.ckpt --file_name onnx_92 --file_format ONNX --device_target GPU
  cd ./scripts/
  bash run_eval_ijbc_onnx.sh ../dataset/IJB_release/IJBB ../onnx_256.onnx ../onnx_92.onnx IJBB
  ```

  The preceding Python command is executed in the backend. You can view the result in the **eval_onnx_IJBB.log** file. The accuracy of the test set is as follows:

  ```bash
  +-----------+-------+-------+--------+-------+-------+-------+
  |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
  +-----------+-------+-------+--------+-------+-------+-------+
  | ijbb-IJBB | 46.81 | 89.89 | 94.81  | 96.54 | 97.68 | 98.70 |
  +-----------+-------+-------+--------+-------+-------+-------+
  ```

- Use th ONNX file to evaluate the IJB-C dataset on GPUs.

  Check the checkpoint file path used for evaluation before running the command below. Ensure that the input validation set path is `IJB_release/IJBC/`.

  ```bash
  bash run_eval_ijbc_onnx.sh /path/evalset path/onnx_bs path/onnx_rs IJBC

  # Example of ONNX evaluation:
  python export.py --batch_size 256 --ckpt_file ./arcface_ascend_v160_ms1mv2_research_cv_IJBB97.67_IJBC98.36.ckpt --file_name onnx_256 --file_format ONNX --device_target GPU
  python export.py --batch_size 254 --ckpt_file ./arcface_ascend_v160_ms1mv2_research_cv_IJBB97.67_IJBC98.36.ckpt --file_name onnx_254 --file_format ONNX --device_target GPU
  cd ./scripts/
  bash run_eval_ijbc_onnx.sh ../dataset/IJB_release/IJBC ../onnx_256.onnx ../onnx_254.onnx IJBC
  ```

  The preceding Python command is executed in the backend. You can view the result in the **eval_onnx_IJBC.log** file. The accuracy of the test set is as follows:

  ```bash
  +-----------+-------+-------+--------+-------+-------+-------+
  |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
  +-----------+-------+-------+--------+-------+-------+-------+
  | ijbc-IJBC | 88.76 | 94.39 | 96.28  | 97.52 | 98.36 | 99.12 |
  +-----------+-------+-------+--------+-------+-------+-------+
  ```

## Exporting a MindIR Model

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The `ckpt_file` parameter is mandatory, and the value range of `FILE_FORMAT` is ["AIR", "MINDIR"].

# Inference Process

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

## Usage

Before performing inference, you need to export the MINDIR file using **export.py**.

```bash
# Inference on Ascend 310 AI Processors
bash run_310_infer.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
```

The value range of `DEVICE_TARGET` is ['GPU', 'CPU', 'Ascend']. `NEED_PREPROCESS` specifies whether data needs to be preprocessed. The value can be **'y'** or **'n'**. In this example, **'y'** is set. `DEVICE_ID` is optional and the default value is **0**.

### Results

The inference result is saved in the current path. You can view the final accuracy result in **acc.log**.

# Model Description

## Performance

### Training Performance

| Parameter         | Arcface                                                      | GPU
| ------------- | ------------------------------------------------------------ | ------------------------------------------- |
| Model version     | arcface                                                        | arcface                                   |
| Resources         | Ascend 910 AI Processor; 2.60 GHz CPU with 192 cores; memory: 755 GB                  | GeForce RTX 3090; 2.90 GHz CPU with 64 cores; 755 GB memory                                          |
| Upload date     | 2021-05-30                                                     | 2021-11-12                               |
| MindSpore version| 1.2.0-c77-python3.7-aarch64                                   | 1.5.0                                     |
| Dataset       | MS1MV2                                                        | MS1MV2                                       |
| Training parameters     | lr=0.08; gamma=0.1                                             | lr=0.08; gamma=0.1                           |
| Optimizer       | SGD                                                           | SGD                                          |
| Loss function     | Arcface                                                      | Arcface                                        |
| Output         | Probability                                                        | Probability                                            |
| Loss         | 0.6                                                          | 0.7                                          |
| Speed         | Single device: 108 ms/step; 8 devices: 97 ms/step                             | 8 devices: 990 ms/step                                     |
| Total duration       | Single device: 65 hours; 8 devices: 8.5 hours                                   | 8 devices: 75 hours                                       |
| Parameters (M)      | 85.2                                                         | 85.2                                         |
| Finetuned checkpoint   | 1249 MB (.ckpt file)                                        | 1249 MB (.ckpt file)                           |
| Script         | [Link](https://gitee.com/mindspore/models/tree/master/official/cv/Arcface)| [Link](https://gitee.com/mindspore/models/tree/master/official/cv/Arcface)|

### Evaluation Performance

| Parameter         | Ascend                  | GPU                         |
| ------------- | ------------------------ | -------------------------  |
| Model version     | arcface                  | arcface                   |
| Resources         | Ascend 910               | GeForce RTX 3090                 |
| Upload date     | 2021/05/30               | 2021/11/12            |
| MindSpore version| 1.2.0-c77-python3.7-aarch64            |  1.5.0      |
| Datasets       | IJB-C, IJB-B, LFW, CFP-FP, AgeDB-30, CALFW, and CPLFW| IJB-C, IJB-B, LFW, CFP-FP, AgeDB-30, CALFW, and CPLFW|
| Output         | Probability                    | Probability                    |
| Accuracy       | lfw:0.998   cfp_fp:0.98   agedb_30:0.981   calfw:0.961   cplfw:0.926   IJB-B:0.943   IJB-C:0.958 | lfw:0.998   cfp_fp:0.984   agedb_30:0.9803   calfw:0.9598   cplfw:0.928   IJB-B:0.943   IJB-C:0.958 |

# Random Seed Description

The initial parameters of the network are initialized randomly.

# ModelZoo Home Page

For details, please go to the [official website](https://gitee.com/mindspore/models).
