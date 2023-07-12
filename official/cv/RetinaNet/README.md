# Contents

<!-- TOC -->

- [RetinaNet Description](#retinanet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage)
        - [Running](#running)
        - [Results](#results)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
        - [Results](#results-1)
    - [Model Export](#model-export)
        - [Usage](#usage-2)
        - [Running](#running-1)
    - [Inference Process](#inference-process)
        - [Usage](#usage-3)
        - [Running](#running-2)
        - [Results](#results-2)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Random Seed Description](#random-seed-description)
- [ModelZoo Home Page](#modelzoo-home-page)
- [Transfer Learning](#transfer-learning)
    - [Transfer Learning Training Process](#transfer-learning-training-process)
        - [Dataset Preprocessing](#dataset-preprocessing)
        - [Transfer Learning Training Process](#transfer-learning-training-process-1)
        - [Transfer Learning Inference Process](#transfer-learning-inference-process)
        - [Transfer Learning Quick Start](#transfer-learning-quick-start)

<!-- /TOC -->

## RetinaNet Description

The RetinaNet algorithm is derived from the paper "Focal Loss for Dense Object Detection" of Facebook AI Research in 2018. The biggest contribution of this paper is that Focal Loss is proposed to solve the problem of class imbalance, thereby creating RetinaNet (one-stage object detection algorithm), an object detection network with accuracy higher than that of the classical two-stage Faster-RCNN.

[Paper](https://arxiv.org/pdf/1708.02002.pdf)
Lin T Y , Goyal P , Girshick R , et al. Focal Loss for Dense Object Detection[C]// 2017 IEEE International Conference on Computer Vision (ICCV). IEEE, 2017:2999-3007.

## Model Architecture

The following shows the overall network architecture of RetinaNet.

[Link](https://arxiv.org/pdf/1708.02002.pdf)

## Dataset

The following datasets are for reference.

COCO2017(https://cocodataset.org/)

- Dataset size: 19.3 GB, 123287 color images of 80 classes

    - Training set: 19.3 GB, 118287 images

    - Test set: 1814.3 MB, 5000 images

- Data format: RGB

    - Note: Data will be processed in **src/dataset.py**.

face-mask-detection (https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) (for transfer learning)

- Dataset size: 397.65 MB, 853 color images of three classes
- Data format: RGB

    - Note: Data will be processed in **src/dataset.py**.

## Environment Requirements

- Hardware
    - Set up the hardware environment with Ascend AI Processors.
- Architecture
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, check the following resources:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.html)

## Script Description

### Script and Sample Code

```retinanet
.
└─Retinanet
  ├─README.md
  ├─ascend310_infer                           # Implement the source code for inference on Ascend 310 AI Processor.
  ├─scripts
    ├─run_single_train.sh                     # Perform single-device training in the Ascend environment.
    ├─run_distribute_train.sh                 # Perform parallel 8-device training in the Ascend environment.
    ├─run_distribute_train_gpu.sh             # Perform parallel 8-device training in the GPU environment.
    ├─run_single_train_gpu.sh                 # Perform single-device training in the GPU environment.
    ├─run_infer_cpp.sh                        # Shell script for inference on Ascend AI Processors
    ├─run_eval.sh                        # Script for inference in the Ascend environment
    ├─run_eval_gpu.sh                         # Script for inference in the GPU environment
  ├─config
    ├─finetune_config.yaml                      # Transfer learning parameter configuration
    └─default_config.yaml                       # Parameter configuration
    └─default_config_gpu.yaml                    # Parameter configuration of GPU
  ├─src
    ├─dataset.py                              # Data preprocessing
    ├─retinanet.py                            # Network model definition
    ├─init_params.py                          # Parameter initialization
    ├─lr_generator.py                         # Learning rate generation function
    ├─coco_eval                               # COCO dataset evaluation
    ├─box_utils.py                            # Anchor box settings
    ├─_init_.py                               # Initialization
    ├──model_utils
      ├──config.py                            # Parameter configuration
      ├──device_adapter.py                    # Device-related information
      ├──local_adapter.py                     # Device-related information
      ├──moxing_adapter.py                    # Decorator (mainly used to copy ModelArts data)
  ├─train.py                                  # Network training script
  ├─export.py                                 # Script for exporting air and MindIR models
  ├─postprogress.py                           # Script for post-processing after inference on Ascend 310 AI Processors
  └─eval.py                                   # Network inference script
  └─create_data.py                            # Script for creating the MindRecord dataset
  └─data_split.py                             # Script for splitting the transfer learning dataset
  └─quick_start.py                            # Transfer learning visualization script
```

### Script Parameters

```default_config.yaml
The main parameters used in the script are as follows:
"img_shape": [600, 600],                                                                        # Image size
"num_retinanet_boxes": 67995,                                                                   # Total number of anchor boxes
"match_thershold": 0.5,                                                                         # Matching threshold
"nms_thershold": 0.6,                                                                           # Non-maximum suppression threshold
"min_score": 0.1,                                                                               # Minimum score
"max_boxes": 100,                                                                               # Maximum number of detection boxes
"lr_init": 1e-6,                                                                                # Initial learning rate
"lr_end_rate": 5e-3,                                                                            # Ratio of the final learning rate to the maximum learning rate
"warmup_epochs1": 2,                                                                            # Number of warm-up epochs in the first stage
"warmup_epochs2": 5,                                                                            # Number of warm-up epochs in the second stage
"warmup_epochs3": 23,                                                                           # Number of warm-up epochs in the third stage
"warmup_epochs4": 60,                                                                           # Number of warm-up epochs in the fourth stage
"warmup_epochs5": 160,                                                                          # Number of warm-up epochs in the fifth stage
"momentum": 0.9,                                                                                # momentum
"weight_decay": 1.5e-4,                                                                         # Weight decay rate
"num_default": [9, 9, 9, 9, 9],                                                                 # Number of anchor boxes in a single grid cell
"extras_out_channels": [256, 256, 256, 256, 256],                                               # Number of output channels at the feature layer
"feature_size": [75, 38, 19, 10, 5],                                                            # Feature size
"aspect_ratios": [[0.5,1.0,2.0], [0.5,1.0,2.0], [0.5,1.0,2.0], [0.5,1.0,2.0], [0.5,1.0,2.0]],   # Aspect ratios of the anchor box
"steps": [8, 16, 32, 64, 128],                                                                 # Steps of the anchor box
"anchor_size":[32, 64, 128, 256, 512],                                                          # Anchor size
"prior_scaling": [0.1, 0.2],                                                                    # Anchor scaling ratio
"gamma": 2.0,                                                                                   # Focal loss parameter
"alpha": 0.75,                                                                                  # Focal loss parameter
"mindrecord_dir": "/cache/MindRecord_COCO",                                                     # MindRecord file path
"coco_root": "/cache/coco",                                                                     # COCO dataset path
"train_data_type": "train2017",                                                                 # Training image directory
"val_data_type": "val2017",                                                                     # Testing image directory
"instances_set": "annotations_trainval2017/annotations/instances_{}.json",                      # Annotation file path
"coco_classes": ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',     # COCO dataset classes
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush'),
"num_classes": 81,                                                                              # Number of dataset classes
"voc_root": "",                                                                                 # VOC dataset path
"voc_dir": "",
"image_dir": "",                                                                                # Image path
"anno_path": "",                                                                                # Annotation file path
"save_checkpoint": True,                                                                        # Specifies whether checkpoints are saved.
"save_checkpoint_epochs": 1,                                                                    # Number of epochs saving checkpoints
"keep_checkpoint_max":10,                                                                       # Maximum number of checkpoints that can be saved
"save_checkpoint_path": "./ckpt",                                                              # Checkpoint file path
"finish_epoch":0,                                                                               # Number of epochs that have been finished
"checkpoint_path":"/home/hitwh1/1.0/ckpt_0/retinanet-500_458_59.ckpt"                           # Checkpoint file path for verification
```

### Training Process

#### Usage

Use the shell script for training. The usage of the shell script is as follows:

```Training
# 8-device parallel training:

Create RANK_TABLE_FILE.
bash scripts/run_distribute_train.sh DEVICE_NUM RANK_TABLE_FILE CONFIG_PATH MINDRECORD_DIR PRE_TRAINED(optional) PRE_TRAINED_EPOCH_SIZE(optional)

# Single-device training:

bash scripts/run_single_train.sh DEVICE_ID MINDRECORD_DIR CONFIG_PATH PRE_TRAINED(optional) PRE_TRAINED_EPOCH_SIZE(optional)

```

> Note:

  For details about RANK_TABLE_FILE, see [Link](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html). For details about how to obtain device IP address, see [Link](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

#### Running

```cocodataset
Dataset structure
└─cocodataset
  ├─train2017
  ├─val2017
  ├─test2017
  ├─annotations

```

```default_config.yaml
Before training, create a MindRecord file. Taking the COCO dataset as an example, configure the COCO dataset path and MindRecord storage path in the YAML file.
# your cocodataset dir
coco_root: /home/DataSet/cocodataset/
# mindrecord dataset dir
mindrecord_dr: /home/DataSet/MindRecord_COCO
```

```MindRecord
# Generate the training set.
python create_data.py --create_dataset coco --prefix retinanet.mindrecord --is_training True --config_path
(Example: python create_data.py  --create_dataset coco --prefix retinanet.mindrecord --is_training True --config_path /home/retinanet/config/default_config.yaml)

# Generate the test set.
python create_data.py --create_dataset coco --prefix retinanet_eval.mindrecord --is_training False --config_path
(Example: python create_data.py  --create_dataset coco --prefix retinanet.mindrecord --is_training False --config_path /home/retinanet/config/default_config.yaml)
```

```bash
Ascend:
# Example of 8-device parallel training (running in the RetinaNet directory):
bash scripts/run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [MINDRECORD_DIR] [CONFIG_PATH] [PRE_TRAINED(optional)] [PRE_TRAINED_EPOCH_SIZE(optional)]
# example: bash scripts/run_distribute_train.sh 8 ~/hccl_8p.json /home/DataSet/MindRecord_COCO/ /home/retinanet/config/default_config.yaml

# Example of single-device training (running in the RetinaNet directory):
bash scripts/run_single_train.sh [DEVICE_ID] [MINDRECORD_DIR] [CONFIG_PATH]
# Example: bash scripts/run_single_train.sh 0 /home/DataSet/MindRecord_COCO/ /home/retinanet/config/default_config.yaml
```

```bash
GPU:
# Example of 8-device parallel training (running in the RetinaNet directory):
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [MINDRECORD_DIR] [CONFIG_PATH] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [PRE_TRAINED(optional)] [PRE_TRAINED_EPOCH_SIZE(optional)]
# Example: bash scripts/run_distribute_train_gpu.sh 8 /home/DataSet/MindRecord_COCO/ /home/retinanet/config/default_config_gpu.yaml 0,1,2,3,4,5,6,7
```

#### Results

The training results are stored in the sample path. Checkpoints are stored in the `./ckpt` directory, and training logs are recorded in the `./log.txt` directory. The following is an example of training logs:

```Training logs
epoch: 2 step: 458, loss is 120.56251
lr:[0.000003]
Epoch time: 164034.415, per step time: 358.154
epoch: 3 step: 458, loss is 11.834166
lr:[0.000028]
Epoch time: 164292.012, per step time: 358.716
epoch: 4 step: 458, loss is 10.49008
lr:[0.000046]
Epoch time: 164822.921, per step time: 359.875
epoch: 5 step: 458, loss is 12.134182
lr:[0.000064]
Epoch time: 164531.610, per step time: 359.239
```

- If you want to train a model on ModelArts, perform model training and inference by referring to the [ModelArts official guide](https://support.huaweicloud.com/modelarts/). The procedure is as follows:

```ModelArts
#  Example of using distributed training on ModelArts:
#  Dataset structure

#  ├── MindRecord_COCO                                              # Directory
#    ├── annotations                                                # Annotation directory
#       ├── instances_val2017.json                                  # Annotation file
#    ├── checkpoint                                                 # Checkpoint directory
#    ├── pred_train                                                 # Pre-trained model directory
#    ├── MindRecord_COCO.zip                                        # Training MindRecord file and evaluation MindRecord file

# (1) Perform step a (modifying parameters in the YAML file) or b (creating a training job and modifying parameters on ModelArts).
#       a. Set enable_modelarts to True.
#          Set distribute to True.
#          Set keep_checkpoint_max to 5.
#          Set save_checkpoint_path to /cache/train/checkpoint.
#          Set mindrecord_dir to /cache/data/MindRecord_COCO.
#          Set epoch_size to 550.
#          Set modelarts_dataset_unzip_name to MindRecord_COCO.
#          Set pre_trained to /cache/data/train/train_predtrained/pred file name if pre-training weight is not set (pre_trained="").

#       b. Set enable_modelarts to True on the ModelArts page.
#          Set the parameters required by method a on the ModelArts page.
#          Note: Paths do not need to be enclosed in quotation marks.

# (2) Set the path of the network configuration file _config_path to /The path of config in default_config.yaml/.
# (3) Set the code path /path/retinanet on the ModelArts page.
# (4) Set the boot file train.py of the model on the ModelArts page.
# (5) On the ModelArts page, set the model data path to .../MindRecord_COCO (path of the MindRecord_COCO directory).
# Output file path and job log path of the model
# (6) Start model training.

# Example of model inference on ModelArts
# (1) Place the trained model to the corresponding position in the bucket.
# (2) Perform step a or b.
#        a. Set enable_modelarts to True.
#          Set mindrecord_dir to /cache/data/MindRecord_COCO.
#          Set checkpoint_path to /cache/data/checkpoint/checkpoint file name.
#          Set instance_set to /cache/data/MindRecord_COCO/annotations/instances_{}.json.

#       b. Set enable_modelarts to True on the ModelArts page.
#          Set the parameters required by method a on the ModelArts page.
#          Note: Paths do not need to be enclosed in quotation marks.

# (3) Set the path of the network configuration file _config_path to /The path of config in default_config.yaml/.
# (4) Set the code path /path/retinanet on the ModelArts page.
# (5) Set the boot file eval.py of the model on the ModelArts page.
# (6) On the ModelArts page, set the model data path to .../MindRecord_COCO (path of the MindRecord_COCO directory).
# Output file path and job log path of the model
# (7) Start model inference.
```

### Evaluation Process

#### Usage

Use the shell script for evaluation. The usage of the shell script is as follows:

```bash
Ascend:
bash scripts/run_eval.sh [DEVICE_ID] [DATASET] [MINDRECORD_DIR] [CHECKPOINT_PATH] [ANN_FILE PATH] [CONFIG_PATH]
# Example: bash scripts/run_eval.sh 0 coco /home/DataSet/MindRecord_COCO/ /home/model/retinanet/ckpt/retinanet_500-458.ckpt /home/DataSet/cocodataset/annotations/instances_{}.json /home/retinanet/config/default_config.yaml
```

```bash
GPU:
bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET] [MINDRECORD_DIR] [CHECKPOINT_PATH] [ANN_FILE PATH] [CONFIG_PATH]
# Example: bash scripts/run_eval_gpu.sh 0 coco /home/DataSet/MindRecord_COCO/ /home/model/retinanet/ckpt/retinanet_500-458.ckpt /home/DataSet/cocodataset/annotations/instances_{}.json /home/retinanet/config/default_config_gpu.yaml
```

> Checkpoints can be generated during training.

#### Results

The calculation results are stored in the sample path. You can view the results in `log.txt`.

```mAP
Ascend:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.503
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.134
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.414
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.152
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.598

========================================

mAP: 0.34747137754625645
```

```mAP
GPU:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.504
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.414
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.156
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.608

========================================

mAP: 0.34852168035724435
```

### Model Export

#### Usage

Before exporting a model, change the value of **checkpoint_path** in the **config.py** file to the checkpoint path.

```shell
python export.py --file_name [RUN_PLATFORM] --file_format[EXPORT_FORMAT] --checkpoint_path [CHECKPOINT PATH]
```

`EXPORT_FORMAT`: ["AIR", "MINDIR"]

#### Running

```Running
python export.py  --file_name retinanet --file_format MINDIR --checkpoint_path /cache/checkpoint/retinanet_550-458.ckpt
```

- Export MindIR models on ModelArts.

    ```ModelArts
    Example of exporting MindIR models on ModelArts
    # (1) Perform step a (modifying parameters in the YAML file) or b (creating a training job and modifying parameters on ModelArts).
    #       a. Set enable_modelarts to True.
    #          Set file_name to retinanet.
    #          Set file_format to MINDIR.
    #          Set checkpoint_path to /cache/data/checkpoint/checkpoint file name.

    #       b. Set enable_modelarts to True on the ModelArts page.
    #          Set the parameters required by method a on the ModelArts page.
    #          Note: Paths do not need to be enclosed in quotation marks.
    # (2) Set the path of the network configuration file _config_path to /The path of config in default_config.yaml/.
    # (3) Set the code path /path/retinanet on the ModelArts page.
    # (4) Set the boot file export.py of the model on the ModelArts page.
    # (5) On the ModelArts page, set the model data path to .../MindRecord_COCO (path of the MindRecord_COCO directory).
    # Output file path and job log path of the MindIR model
    ```

### Inference Process

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

#### Usage

Before inference, you need to export the model from the Ascend 910 AI Processor environment. During inference, images whose **iscrowd** is **true** must be excluded. The remaining image IDs are stored in the **ascend310_infer** directory.
You also need to modify the **coco_root**, **val_data_type** and **instances_set** configuration items in the **config.py** file. The values are the directory of the COCO dataset, the directory of the inference dataset, and the annotation file used for calculating the accuracy after inference. The value of **val_data_type** is a part of that of **instances_set**. Ensure that the file is correct and exists.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_ID]
```

#### Running

```Running
 bash run_infer_cpp.sh ./retinanet.mindir ./dataset/coco2017/val2017 Ascend 0
```

#### Results

The inference results are saved in the current directory. You can find results similar to the following in the **acc.log** file:

```mAP
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.350
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.509
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.139
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.509
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.155
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.608

========================================

mAP: 0.3499478734634595
```

## Model Description

### Performance

#### Training Performance

| Parameter                       | Ascend                                |GPU|
| -------------------------- | ------------------------------------- |------------------------------------- |
| Model name                   | RetinaNet                            |RetinaNet                            |
| Runtime environment                   | Ascend 910 AI Processor; 2.6 GHz CPU with 192 cores; 755 GB memory; EulerOS 2.8 | RTX 3090; 512 GB memory|
| Upload date                   | 10/01/2021                            |17/02/2022                            |
| MindSpore version            | 1.2.0                                 |1.5.0|
| Dataset                     | 123287 images                         |123287 images                         |
| Batch size                | 32                                   |32                                   |
| Training parameters                   | src/config.py                         |config/default_config_gpu.yaml
| Optimizer                     | Momentum                              |Momentum                              |
| Loss function                   | Focal loss                            |Focal loss                            |
| Final loss                   | 0.582                                  |0.57|
| Accuracy (8-device)                | mAP[0.3475]               |mAP[0.3499]               |
| Total duration (8-device)            | 23h16m54s                              |51h39m6s|
| Script                      | [Link](https://gitee.com/mindspore/models/tree/master/official/cv/RetinaNet)|[Link](https://gitee.com/mindspore/models/tree/master/official/cv/RetinaNet)|

#### Inference Performance

| Parameter                | Ascend                      |GPU|
| ------------------- | --------------------------- |--|
| Model name            | RetinaNet               |RetinaNet               |
| Runtime environment            | Ascend 910 AI Processor; 2.6 GHz CPU with 192 cores; 755 GB memory; EulerOS 2.8|RTX 3090; 512 GB memory|
| Upload date            | 10/01/2021                  |17/02/2022 |
| MindSpore version     | 1.2.0                        |1.5.0|
| Dataset             | 5000 images                  |5000 images                  |
| Batch size         | 32                          |32                          |
| Accuracy             | mAP[0.3475]                  |mAP[0.3499]               |
| Total duration             | 10 mins and 50 seconds       |13 mins and 40 seconds       |

## Random Seed Description

In the `dataset.py` script, a random seed is set in the `create_dataset` function. We also set a random seed in the `train.py` script.

## ModelZoo Home Page

For details, please visit the [official website](https://gitee.com/mindspore/models).

## Transfer Learning

### Transfer Learning Training Process

#### Dataset Preprocessing

[Dataset Download Address](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

Download the dataset, decompress it to the RetinaNet root directory, and use the data_split script to divide the dataset to two parts: 80% for the training set and 20% for the test set.

```bash
Example of running the script
python data_split.py
```

```text
Dataset structure
└─dataset
  ├─train
  ├─val
  ├─annotation

```

```text
Before training, create a MindRecord file. Taking the face_mask_detection dataset as an example, configure the facemask dataset path and MindRecord storage path in the YAML file.
# your dataset dir
dataset_root: /home/mindspore/retinanet/dataset/
# mindrecord dataset dir
mindrecord_dir: /home/mindspore/retinanet/mindrecord
```

```bash
# Generate the training set.
python create_data.py  --config_path
(Example: python create_data.py  --config_path  './config/finetune_config.yaml')

# Generate the test set.
The test set can be automatically generated by the eval script after the training is complete.
```

#### Transfer Learning Training Process

Download the pre-trained CKPT file from the [Mindspore Hub](https://www.mindspore.cn/resources/hub/details?MindSpore/1.8/retinanet_coco2017).

```text
# Set the CKPT file of the pre-trained model in finetune_config.yaml.
pre_trained: "/home/mindspore/retinanet/retinanet_ascend_v170_coco2017_official_cv_acc35.ckpt"
```

```bash
#Run the transfer learning training script.
python train.py --config_path  './config/finetune_config.yaml'
To save the log information, run the following command:
python train.py --config_path ./config/finetune_config.yaml > log.txt 2>&1
```

**Results**

The training results are stored in the sample path. You can view the checkpoint file in the `./ckpt` directory. The following is an example of the training loss output:

```text
epoch: 1 step: 42, loss is 4.347288131713867
lr:[0.000088]
Train epoch time: 992053.072 ms, per step time: 23620.311 ms
Epoch time: 164034.415, per step time: 358.154
epoch: 3 step: 42, loss is 1.8387094736099243
lr:[0.000495]
Train epoch time: 738396.280 ms, per step time: 17580.864 ms
epoch: 4 step: 42, loss is 1.3805917501449585
lr:[0.000695]
Train epoch time: 742051.709 ms, per step time: 17667.898 ms
```

#### Transfer Learning Inference Process

```bash
#Run the transfer learning training script.
python eval.py --config_path  './config/finetune_config.yaml'
```

**Results**

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.538
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.781
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.634
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.687
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.856
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.284
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.448
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.737
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.872

========================================

mAP: 0.5376701115352185

```

#### Transfer Learning Quick Start

After the eval script is executed, the `instances_val.json` and `predictions.json` files are generated. You need to change the paths of the `instances_val.json` and `predictions.json` files in the `quick_start.py` script before running the eval script.

```bash
# Example of running the quick start script
python quick_start.py --config_path './config/finetune_config.yaml'
```

**Results**

The meanings of the colors in the figure are as follows:

- Light blue: real label "mask_weared_incorrect"
- Light green: real label "with_mask"
- Light red: real label "without_mask"
- Blue: prediction label "mask_weared_incorrect"
- Green:  prediction label "with_mask"
- Red:  prediction label "without_mask"

