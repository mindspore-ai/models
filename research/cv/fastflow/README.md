# FastFlow

<!-- TOC -->

- [Content](#content)
- [fastflow Description](#fastflow-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Characteristics](#characteristics)
    - [Mixed precision](#mixed-precision)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Scripts and sample code](#scripts-and-sample-code)
    - [Script parameters](#script-parameters)
    - [Training process](#training-process)
        - [Download pretrained weights](#download-pretrained-weights)
        - [Training](#training)
    - [Evaluation process](#evaluation)
        - [Evaluation](#evaluation)
    - [Export process](#export-process)
        - [Export](#export)
    - [Inference process](#inference-process)
        - [Inference](#inference)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training performance](#training-performance)
            - [Training FastFlow  on MVTec-AD](#training-fastflow-on-mvtec-ad)
        - [Evaluation performance](#evaluation-performance)
            - [Evaluating FastFlow  on MVTec-AD](#evaluating-fastflow-on-mvtec-ad)
        - [Inference performance](#inference-performance)
            - [Inference FastFlow on MVTec-AD](#inference-fastflow-on-mvtec-ad)
- [Random description](#random-description)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# FastFlow description

FastFlow is an industrial anomaly detection model based on a pre-trained neural network proposed in 2021, achieves 99.4% AUC on the MVTec-AD dataset with high inference efficiency. Implemented with 2D normalizing flows, FastFlow can be used as the probability distribution estimator, a plug-in module with arbitrary deep feature extractors such as ResNet and vision transformer for unsupervised anomaly detection and localization.In training phase, FastFlow learns to transform the input visual feature into a tractable distribution and obtains the likelihood to recognize anomalies in inference phase. Extensive experimental results on the MVTec AD dataset show that FastFlow surpasses previous state-of-the-art methods in terms of accuracy and inference efficiency with various backbone networks.

[FastFlow: Unsupervised Anomaly Detection and Localization
via 2D Normalizing Flows](https://arxiv.org/pdf/2111.07677v2.pdf)：Yu, Jiawei, Ye Zheng, Xiang Wang, Wei Li, Yushuang Wu, Rui Zhao, and Liwei Wu. "Fastflow: Unsupervised anomaly detection and localization via 2d normalizing flows." arXiv preprint arXiv:2111.07677 (2021).

# Model Architecture

![fastflow scheme](picture/fastflow.png)
FastFlow uses the pre-trained WideResNet50 as the Encoder and removes the layers after layer4.

# Dataset

Dataset used：[MVTec AD](<https://www.mvtec.com/company/research/datasets/mvtec-ad/>)

- Data set size: 4.9G, a total of 15 classes, 5354 images (size between 700x700~1024x1024)

    - Training set: 3629 in total

    - Test set: 1725 in total

- Data format: binary file

- Note: Data will be processed in src/dataset.py.

- Directory Structure:

  ```text
  data
  ├── bottle
  │   ├── bottle_test.json
  │   ├── bottle_train.json
  │   ├── ground_truth
  │   │   ├── broken_large
  │   │   │   ├── 000_mask.png
  │   │   │   └── ......
  │   │   ├── broken_small
  │   │   │   ├── 000_mask.png
  │   │       └── ......
  │   ├── test
  │   │   ├── broken_large
  │   │   │   ├── 000.png
  │   │   │   └── ......
  │   │   └── good
  │   │       ├── 000.png
  │   │       └── ......
  │   └── train
  │       └── good
  │           ├── 000.png
  │           └── ......
  ├── cable
  │   ├── cable_test.json
  │   ├── cable_train.json
  │   ├── ground_truth
  │   │   ├── bent_wire
  │   │   │   ├── 000_mask.png
  ......
  ```

# Characteristics

## Mixed precision

Training methods with [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/r1.6/enable_mixed_precision.html) use support for both single-precision and half-precision data to increase the training speed of deep learning neural networks, while maintaining the network accuracy that can be achieved with single-precision training. Mixed-precision training increases computational speed and reduces memory usage while enabling training of larger models on specific hardware or enabling larger batches of training. Taking the FP16 operator as an example, if the input data type is FP32, the MindSpore background will automatically reduce the precision to process the data. You can open the INFO log and search for "reduce precision" to view operators with reduced precision.

# Requirements

- Hardware
    - Use NVIDIA GPU or Ascend processor to build the hardware environment.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, see the following resources:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# Quick Start

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:

- Ascend processor environment running

  ```bash
  # Run training example
  python train.py --dataset_path /data/mvtec --device_id 0 --pre_ckpt_path pretrained/wide_resnet50_racm-8234f177.ckpt --category bottle > train.log 2>&1 &
  or
  cd scripts
  bash run_standalone_train_ascend.sh [dataset_path] [pre_ckpt_path] [category] [device_id]

  # Run evaluation example
  python eval.py --dataset_path /data/mvtec --device_id 0 --ckpt_path ckpt/fastflow_bottle.ckpt --category bottle > eval.log 2>&1 &
  or
  cd scripts
  bash run_eval_ascend.sh [dataset_path] [pre_ckpt_path] [category] [device_id]

  ```

- GPU environment running

  ```bash
  # Run training example
  cd scripts
  bash run_standalone_train_gpu.sh [dataset_path] [pre_ckpt_path] [category] [device_id]

  # Run evaluation example
  cd scripts
  bash run_eval_gpu.sh [dataset_path] [pre_ckpt_path] [category] [device_id]

  ```

# Script Description

## Scripts and sample code

```text

  ├── fastflow
      ├── README.md                               // FastFlow related instructions
      ├── ascend310_infer                         // 310 Infer
      ├── scripts
      │   ├── run_infer_310.sh                    // 310 inference script
      │   ├── run_eval_ascend.sh                  // evaluation script
      │   ├── run_eval_gpu.sh                     // evaluation script GPU
      │   ├── run_standalone_train_ascend.sh      // training script
      │   ├── run_standalone_train_gpu.sh         // training script GPU
      |   ├── run_all_mvtec_ascend.sh             // full MVTec-AD dataset train&eval
      |   └── run_all_mvtec_gpu.sh                // full MVTec-AD dataset train&eval GPU
      ├── src
      │   ├── anomaly_map.py                      // anomaly map genarator
      │   ├── cell.py                             // train one step cell
      │   ├── config.py                           // config file
      │   ├── dataset.py                          // data manipulation
      │   ├── fastflow.py                         // fastflow model
      │   ├── loss.py                             // fastflow loss
      │   ├── operator.py                         // operator for visible imgs
      │   ├── pthtockpt.py                        // pth2ckpt conversion
      │   ├── resnet.py                           // resnet feature extractor
      │   └── utils.py                            // utils to record
      ├── eval.py                                 // testing script
      ├── export.py                               // inference model export script
      ├─ preprocess.py                            // 310 preprocess
      ├─ postprocess.py                           // 310 postprocess
      └── train.py                                // training script
```

## Script parameters

  ```yaml
  --dataset_path: dataset path
  --pre_ckpt_path: pretrained feature extractor path
  --category: data category
  --device_id: device id
  ```

## Training process

### Download pretrained weights

pytorch's WideResNet50 pretrained model, [click](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth) to get

```bash
python src/pthtockpt.py --pth_path /path/wide_resnet50_racm-8234f177.pth
```

### Training

- Ascend

  ```bash
  python train.py --dataset_path /data/mvtec --pre_ckpt_path pretrained/wide_resnet50_racm-8234f177.ckpt --category bottle --device_id 0 > train.log 2>&1 &
  or
  cd scripts
  bash run_standalone_train_ascend.sh [dataset_path] [pre_ckpt_path] [category] [device_id]
  ```

  The above python commands will run in the background and you can view the results via the train.log file.

  For the MVTec-AD dataset, the training and inference of all categories of data in Mvtec can be performed by executing the following commands:

  ```bash
  cd scripts
  bash run_all_mvtec.sh [dataset_path] [pre_ckpt_path] [device_id]
  ```

- GPU

  ```bash
  cd scripts
  bash run_standalone_train_gpu.sh [dataset_path] [pre_ckpt_path] [category] [device_id]
  ```

  The above python commands will run in the background and you can view the results via the train.log file.

  For the MVTec-AD dataset, the training and inference of all categories of data in Mvtec can be performed by executing the following commands:

  ```bash
  cd scripts
  bash run_all_mvtec_ascend.sh [dataset_path] [pre_ckpt_path] [device_id]
  ```

## Evaluation process

### Evaluation

- Ascend

  ```text
  python eval.py --dataset_path /data/mvtec --device_id 0 --ckpt_path ckpt/fastflow_bottle.ckpt --category bottle > eval.log 2>&1 &
  or
  cd scripts
  bash run_eval_ascend.sh [dataset_path] [ckpt_path] [category] [device_id]
  ```

  The above python command will run in the background and you can view the result through the eval.log file. The accuracy of the test dataset is as follows:

  ```text
  # Reference precision of bottle class
  category is bottle
  pixel_auc: 0.98991
  ```

- GPU

  ```text
  cd scripts
  bash run_eval_gpu.sh [dataset_path] [ckpt_path] [category] [device_id]
  ```

  The above python command will run in the background and you can view the result through the eval.log file. The accuracy of the test dataset is as follows:

  ```text
  # Reference precision of bottle class
  category is bottle
  pixel_auc: 0.98991
  ```

## Export process

### Export

Export the checkpoint file to mindir format model, take checkpoint for bottle as an example:

  ```shell
  python export.py --device_id 0 --ckpt_file ckpt/fastflow_bottle.ckpt
  ```

## Inference process

### Inference

Before running inference we need to export the model. Air models can only be exported on the Ascend 910 environment, mindir can be exported on any environment.

- Inference on MVTec AD dataset using Ascend 310

  The command to perform inference is as follows, where ``MINDIR_PATH`` is the mindir file path;

  ``DATASET_PATH``is the path of the inference dataset, which is the parent directory of the data class (such as toothbrush);

  ``CATEGORY`` Indicates the data type, desirable: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper.

  ``DEVICE_ID`` optional, default value is 0;

  ```shell
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [CATEGORY] [DEVICE_ID]
  ```

  The inference accuracy results are saved in the acc_[CATEGORY].log log file.

# Model description

## Performance

### Training performance

#### Training FastFlow on MVTec-AD

| Parameter          | Ascend                            | GPU                            |
| ------------- | ---------------------------------------------|---------------------------------------------|
| Model      | FastFlow    |FastFlow|
| Environment   | Ascend 910；CPU 2.60GHz，192 cores；RAM 755G；OS Euler2.8       | NVIDIA RTX3090；CPU 2.90GHz，64 cores；RAM 251G；OS Ubuntu 18.04.6       |
| Upload date  | 2022-10-11         | 2022-10-11                                                   |
| MindSpore version | 1.8.1                   |1.8.1                    |
| Dataset        | MVTec AD   |MVTec AD   |
| Training params| epoch=500, steps depend on data type, batch_size = 32，optimizer=Adam，lr=1e-3，weight decay=1e-5 |epoch=500, steps depend on data type, batch_size = 32，optimizer=Adam，lr=1e-3，weight decay=1e-5                  |
| Speed          | 554 ms/step                                               | 557 ms/step                                                   |
| Total duration | 1-2h according to data type |1-2h according to data type |

### Evaluation performance

#### Evaluating FastFlow on MVTec-AD

| Parameter           | Ascend                           |GPU                           |
| ------------------- | --------------------------- |--------------------------- |
| Model       | FastFlow                |FastFlow                        |
| Environment           | Ascend 910；OS Euler2.8        |NVIDIA RTX3090；OS Ubuntu 18.04        |
| Upload date       | 2022-10-11                       |2022-10-11                       |
| MindSpore version | 1.8.1                           |1.8.1                            |
| Dataset         | MVTec AD                         |MVTec AD                         |
| batch_size     | 1                                |1                                |
| bottle_auc        | pixel_auc: 0.9900       | pixel_auc: 0.9878               |
| cable_auc         | pixel_auc: 0.9796       | pixel_auc: 0.9809               |
| capsule_auc       | pixel_auc: 0.9912       | pixel_auc: 0.9900               |
| carpet_auc        | pixel_auc: 0.9918       | pixel_auc: 0.9904               |
| grid_auc          | pixel_auc: 0.9929       | pixel_auc: 0.9924               |
| hazelnut_auc      | pixel_auc: 0.9804       | pixel_auc: 0.9816               |
| leather_auc       | pixel_auc: 0.9971       | pixel_auc: 0.9966               |
| metal_nut_auc     | pixel_auc: 0.9855       | pixel_auc: 0.9844               |
| pill_auc          | pixel_auc: 0.9802       | pixel_auc: 0.9792               |
| screw_auc         | pixel_auc: 0.9874       | pixel_auc: 0.9881               |
| tile_auc          | pixel_auc: 0.9736       | pixel_auc: 0.9739               |
| toothbrush_auc    | pixel_auc: 0.9816       | pixel_auc: 0.9834               |
| transistor_auc    | pixel_auc: 0.9804       | pixel_auc: 0.9824               |
| wood_auc          | pixel_auc: 0.9624       | pixel_auc: 0.9598               |
| zipper_auc        | pixel_auc: 0.9894       | pixel_auc: 0.9900               |
| **Average**       | **pixel_auc: ** 0.9842  | **pixel_auc: **0.9841           |

### Inference performance

#### Inference FastFlow on MVTec-AD

| Parameter           | Ascend                           |
| ------------------- | --------------------------- |
| Model       | FastFlow                        |
| Environment           | Ascend 310；OS Euler2.8        |
| Upload date       | 2022-10-11                       |
| MindSpore version | 1.8.1                            |
| Dataset         | MVTec AD                         |
| bottle_auc     | pixel_auc: 0.9892 |
| cable_auc      | pixel_auc: 0.9795 |
| capsule_auc    | pixel_auc: 0.9912 |
| carpet_auc     | pixel_auc: 0.9918 |
| grid_auc       | pixel_auc: 0.9929 |
| hazelnut_auc   | pixel_auc: 0.9806 |
| leather_auc    | pixel_auc: 0.9971 |
| metal_nut_auc  | pixel_auc: 0.9855 |
| pill_auc       | pixel_auc: 0.9802 |
| screw_auc      | pixel_auc: 0.9871 |
| tile_auc       | pixel_auc: 0.9738 |
| toothbrush_auc | pixel_auc: 0.9815 |
| transistor_auc | pixel_auc: 0.9803 |
| wood_auc       | pixel_auc: 0.9621 |
| zipper_auc     | pixel_auc: 0.9855 |

# Random description

In dataset.py, "shuffle=True" is set. In train.py, random seed is used.

# ModelZoo Homepage  

Please visit the official website [homepage](https://gitee.com/mindspore/models) .
