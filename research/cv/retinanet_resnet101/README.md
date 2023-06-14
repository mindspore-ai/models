# Contents

<!-- TOC -->

[查看中文](./README_CN.md)

- [Retinanet Description](#retinanet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#ccript-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage1)
        - [Run](#run1)
        - [Result](#result1)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage2)
        - [Run](#run2)
        - [Result](#result2)
    - [Inference Process](#inference-process)
        - [Usage](#usage4)
        - [Run](#run4)
        - [Result](#result4)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
- [Description of Random State](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- TOC -->

# [Retinanet Description](#content)

RetinaNet was proposed in "Focal Loss for Dense Object Detection" 2018 paper by Facebook AI Research.
The biggest contribution of this paper is to propose Focal Loss to solve the problem of category imbalance,
thus creating RetinaNet (One Stage target detection algorithm), a target detection network whose accuracy exceeds the classic Two Stage Faster-RCNN.

[Paper](https://arxiv.org/pdf/1708.02002.pdf)
Lin T Y , Goyal P , Girshick R , et al. Focal Loss for Dense Object Detection[C]// 2017 IEEE International Conference on Computer Vision (ICCV). IEEE, 2017:2999-3007.

# [Model Architecture](#content)

The overall network architecture of Retinanet is [here](https://arxiv.org/pdf/1708.02002.pdf)

# [Dataset](#content)

Dataset used (refer to paper): [COCO2017](https://cocodataset.org/#download)

- Dataset size: 19.3G, 123287 pcs 80 classes of colored images

    - [train](http://images.cocodataset.org/zips/train2017.zip): 19.3G, 118287 images
    - [val](http://images.cocodataset.org/zips/val2017.zip): 814.3M, 5000 images

- Data format: RGB

> Note: The data will be processed with src/dataset.py.

# [Environment Requirements](#content)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For more information about MindSpore, please check the resources below：
    - [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- Training in ModelArts (If you want to run on ModelArts, you can refer to the following [modelarts documents](https://support.huaweicloud.com/modelarts/))

    ```python
    # Train with 8 cards on ModelArts
    # (1) choose a or b
    #       a. in default_config.yaml set "enable_modelarts=True"
    #          in default_config.yaml set "distribute=True"
    #          in default_config.yaml set "coco_root='/cache/data'"
    #          in default_config.yaml set "epoch_size=500"
    #          (optional)in default_config.yaml set "checkpoint_path='s3://dir_to_your_pretrained/'"
    #          in default_config.yaml set other parameters
    #       b. Set on the web page  "enable_modelarts=True"
    #          Set on the web page "distribute=True"
    #          Set on the web page "coco_root=/cache/data"
    #          Set on the web page "epoch_size=500"
    #          (optional)Set on the web page "checkpoint_path='s3://dir_to_your_pretrained/'"
    #          Set on the web page other parameters
    # (2) Prepare model code
    # (3) If you choose to fine-tune your model, upload your pretrained model to the S3 bucket
    # (4) choose a or b (recommended choice a)
    #       a. First, compress the dataset into a ".zip" file.
    #          Second, upload your compressed dataset to the S3 bucket (you can also upload the uncompressed dataset, but that may be slow.)
    #       b. Upload the original dataset to the S3 bucket.
    #          (Dataset conversion occurs during the training process, which takes more time. It will be re-transformed every time you train.)
    # (5) Set your code path on the web page to "/path/retinanet"
    # (6) Set the startup file on the web page as "train.py"
    # (7) Set "training dataset", "training output file path", "job log path", etc. on the webpage
    # (8) Create a training job
    #
    # Train with 1 card on ModelArts
    # (1) choose a or b
    #       a. in default_config.yaml set "enable_modelarts=True"
    #          in default_config.yaml set "coco_root='/cache/data'"
    #          in default_config.yaml set "epoch_size=500"
    #          (optional)in default_config.yaml set "checkpoint_path='s3://dir_to_your_pretrained/'"
    #          in default_config.yaml set other parameters
    #       b. Set on the web page "enable_modelarts=True"
    #          Set on the web page "coco_root='/cache/data'"
    #          Set on the web page "epoch_size=500"
    #          (optional)Set on the web page "checkpoint_path='s3://dir_to_your_pretrained/'"
    #          Set on the web page other parameters
    # (2) Prepare model code
    # (3) If you choose to fine-tune your model, upload your pretrained model to the S3 bucket
    # (4) choose a or b (recommended choice a)
    #       a. First, compress the dataset into a ".zip" file.
    #          Second, upload your compressed dataset to the S3 bucket (you can also upload the uncompressed dataset, but that may be slow.)
    #       b. Upload the original dataset to the S3 bucket.
    #          (Data set conversion occurs during the training process, which takes more time. It will be re-transformed every time you train.)
    # (5) Set your code path on the web page to "/path/retinanet"
    # (6) Set the startup file on the web page as "train.py"
    # (7) Set "training dataset", "training output file path", "job log path", etc. on the webpage
    # (8) Create a training job
    #
    # Eval on ModelArts
    # (1) choose a or b
    #       a. in default_config.yaml set "enable_modelarts=True"
    #          in default_config.yaml set "checkpoint_path='s3://dir_to_your_trained_model/'"
    #          in default_config.yaml set "mindrecord_dir='./MindRecord_COCO'"
    #          in default_config.yaml set "coco_root='/cache/data'"
    #          in default_config.yaml set other parameters
    #       b. Set on the web page "enable_modelarts=True"
    #          Set on the web page "checkpoint_path='s3://dir_to_your_trained_model/'"
    #          Set on the web page "mindrecord_dir='./MindRecord_COCO'"
    #          Set on the web page "coco_root='/cache/data'"
    #          Set on the web page other parameters
    # (2) Prepare model code
    # (3) Upload your pretrained model to the S3 bucket
    # (4) choose a or b (recommended choice a)
    #       a. First, compress the dataset into a ".zip" file.
    #          Second, upload your compressed dataset to the S3 bucket (you can also upload the uncompressed dataset, but that may be slow.)
    #       b. Upload the original dataset to the S3 bucket.
    #          (Data set conversion occurs during the training process, which takes more time. It will be re-transformed every time you train.)
    # (5) Set your code path on the web page to "/path/retinanet"
    # (6) Set the startup file on the web page as "eval.py"
    # (7) Set "training dataset", "training output file path", "job log path", etc. on the webpage
    # (8) Create a training job
    ```

- Export in ModelArts (If you want to run on ModelArts, you can refer to the following [modelarts documents](https://support.huaweicloud.com/modelarts/))

    ```python
    # (1) choose a or b
    #       a. in default_config.yaml set "enable_modelarts=True"
    #          in default_config.yaml set "file_name='retinanet'"
    #          in default_config.yaml set "file_format='MINDIR'"
    #          in base_config.yaml set "checkpoint_path='/The path of checkpoint in S3/'"
    #          in base_config.yaml set other parameters
    #       b. Set on the web page "enable_modelarts=True"
    #          Set on the web page "file_name='retinanet'"
    #          Set on the web page "file_format='MINDIR'"
    #          Set on the web page "checkpoint_path='/The path of checkpoint in S3/'"
    #          Set on the web page other parameters
    # (2) Upload your pretrained model to the S3 bucket
    # (3) Set your code path on the web page to "/path/retinanet"
    # (4) Set the startup file on the web page as "export.py"
    # (5) Set "training dataset", "training output file path", "job log path", etc. on the webpage
    # (6) Create a training job
    ```

# [Script Description](#content)

## [Script and Sample Code](#content)

```shell
.
└─Retinanet_resnet101
  ├─README.md
  ├─ascend310_infer                           # inference in ascend310
  ├─scripts
    ├─run_single_train.sh                     # Use Ascend environment single card training
    ├─run_single_train_gpu.sh
    ├─run_distribute_train.sh                 # Parallel training with eight cards in the Ascend environment
    ├─run_distribute_train_gpu.sh
    ├─run_eval.sh                             # Use the Ascend environment to run inference scripts
    ├─run_eval_gpu.sh
  ├─src
    ├─backbone.py
    ├─bottleneck.py
    ├─dataset.py
    ├─retinahead.py
    ├─init_params.py
    ├─lr_generator.py
    ├─coco_eval
    ├─box_utils.py
    ├─_init_.py
    └──model_utils
       ├──config.py
       ├──device_adapter.py
       ├──local_adapter.py
       └──moxing_adapter.py
  ├─default_config.yaml
  ├─train.py
  └─eval.py
```

## [Script Parameters](#content)

```text
Main parameteres used in train.py and config.py:
"img_shape": [600, 600],                                                                        # image size
"num_retinanet_boxes": 67995,                                                                   # The total number of a priori boxes set
"match_thershold": 0.5,
"softnms_sigma": 0.5,
"nms_thershold": 0.6,
"min_score": 0.1,
"max_boxes": 100,                                                                               # Maximum number of detection frames
"global_step": 0,
"lr_init": 1e-6,
"lr_end_rate": 5e-3,                                                                            # The ratio of the final learning rate to the maximum learning rate
"warmup_epochs1": 2,                                                                            # Number of cycles of the 1st stage warmup
"warmup_epochs2": 5,                                                                            # Number of cycles of the 2d stage warmup
"warmup_epochs3": 23,                                                                           # Number of cycles of the 3d stage warmup
"warmup_epochs4": 60,                                                                           # Number of cycles of the 4th stage warmup
"warmup_epochs5": 160,                                                                          # Number of cycles of the 5th stage warmup
"momentum": 0.9,                                                                                # momentum
"weight_decay": 1.5e-4,
"num_default": [9, 9, 9, 9, 9],                                                                 # The number of a priori boxes in a single grid
"extras_out_channels": [256, 256, 256, 256, 256],                                               # Feature layer output channels
"feature_size": [75, 38, 19, 10, 5],
"aspect_ratios": [(0.5,1.0,2.0), (0.5,1.0,2.0), (0.5,1.0,2.0), (0.5,1.0,2.0), (0.5,1.0,2.0)],   # Priori box size change ratio
"steps": ( 8, 16, 32, 64, 128),                                                                 # Priori box setting step size
"anchor_size":(32, 64, 128, 256, 512),                                                          # A priori box size
"prior_scaling": (0.1, 0.2),                                                                    # Used to adjust the ratio of regression and regression in loss
"gamma": 2.0,                                                                                   # focal loss parameter
"alpha": 0.75,                                                                                  # focal loss parameter
"mindrecord_dir": "/opr/root/data/MindRecord_COCO",
"coco_root": "/opr/root/data/",
"train_data_type": "train2017",
"val_data_type": "val2017",
"instances_set": "annotations_trainval2017/annotations/instances_{}.json",
"coco_classes": ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
"num_classes": 81,
"voc_root": "",
"voc_dir": "",
"image_dir": "",
"anno_path": "",
"save_checkpoint": True,
"save_checkpoint_epochs": 1,
"keep_checkpoint_max":1,
"save_checkpoint_path": "./model",                                                              # Path to save checkpoints
"finish_epoch":0,                                                                               # number of epoch that have been run
"checkpoint_path":"/opr/root/reretina/retinanet2/LOG0/model/retinanet-400_458.ckpt"             # checkpoint path for evaluation
```

## [Training process](#content)

### Usage

You can use python or shell scripts for training. The usage of the shell script is as follows:

- Ascend:

```shell
# data and the path to store the mindrecord file are set in default_config.yaml

# before training convert dataset to MindRecord format
python train.py --only_create_dataset=True --run_platform="Ascend"

# Eight-card parallel training example：
# create RANK_TABLE_FILE
bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)

# Single card training example：
bash run_single_train.sh [DEVICE_ID] [EPOCH_SIZE] [LR] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

> Note: RANK_TABLE_FILE related reference materials see in this [link](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html), for details on how to get device_ip check this [link](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools).

- GPU

```shell
# data and the path to store the mindrecord file are set in default_config.yaml

# convert dataset to MindRecord format:
python train.py --only_create_dataset=True --run_platform="GPU"

# Eight-card parallel training example：
bash run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)

# Single card training example：
bash run_single_train_gpu.sh [DEVICE_ID] [EPOCH_SIZE] [LR] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

### Run

- Ascend

```shell
# Eight-card parallel training example (run in the retinanet directory):
bash run_distribute_train.sh 8 500 0.1 coco scripts/rank_table_8pcs.json /dataset/retinanet-322_458.ckpt 322

# Single card training example (run in the retinanet directory):
bash run_single_train.sh 0 500 0.1 coco /dataset/retinanet-322_458.ckpt 322
```

- GPU

```shell
# Eight-card parallel training example (run in the retinanet directory):
bash run_distribute_train_gpu.sh 8 400 0.025 coco /dataset/retinanet-322_1221.ckpt 322

# Single card training example (run in the retinanet directory):
bash run_single_train_gpu.sh 0 400 0.025 coco /dataset/retinanet-322_1221.ckpt 322
```

### Result

Paths are set in default_config.yaml. Checkpoints will be save in `./model`.
The training log will be recorded to `LOG/train.log`，an example of the training log is as follows:

```text
epoch: 397 step: 458, loss is 0.6153226
lr:[0.000598]
epoch time: 313364.642 ms, per step time: 684.202 ms
epoch: 398 step: 458, loss is 0.5491791
lr:[0.000544]
epoch time: 313486.094 ms, per step time: 684.467 ms
epoch: 399 step: 458, loss is 0.51681435
lr:[0.000511]
epoch time: 313514.348 ms, per step time: 684.529 ms
epoch: 400 step: 458, loss is 0.4305706
lr:[0.000500]
epoch time: 314138.455 ms, per step time: 685.892 ms
```

## [Evaluation process](#content)

### Usage

You can use python or shell scripts for training. The usage of the shell script is as follows:

- Ascend

```shell
bash scripts/run_eval.sh [DATASET] [DEVICE_ID]
```

- GPU

```shell
bash run_eval_gpu.sh [DATASET] [DEVICE_ID] [CHECKPOINT_PATH]
```

### Run

- Ascend:

```shell
bash run_eval.sh coco 0
```

- GPU

```shell
bash run_eval_gpu.sh coco 0 LOG/model/retinanet-500_610.ckpt
```

> Checkpoints can be generated during training

### Result

The calculation results will be stored in the example path, which you can view in `eval.log`.

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.517
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.143
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.394
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.455
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.172
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.680

========================================

mAP: 0.3710347196613514
```

## [Model Export](#content)

### Usage

Before exporting the model, modify the checkpoint_path configuration item in the default_config.yaml file. The value is the path of the checkpoint.

```shell
python export.py --file_name [FILE_NAME] --file_format [EXPORT_FORMAT] --checkpoint_path [CHECKPOINT_PATH]
```

`EXPORT_FORMAT` choose from ["AIR", "MINDIR"]

### Run

```shell
python export.py
```

## [Inference Process](#content)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Usage

Before inference, model needs to be exported to the Ascend 910 environment. Pictures with iscrowd set to true should be excluded.
The image id after removal is saved in the ascend310_infer directory.
You also need to modify the coco_root, val_data_type, and instances_set configuration items in the config.py file.
The values are respectively taken as the directory of the coco data set, the directory name of the data set used for inference, and the annotation file used to calculate the accuracy after the inference is completed.
The instances_set is spliced with val_data_type to ensure that the file is correct and exists.

```shell
# Ascend310 inference
sh run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_ID]
```

### Run

```bash
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

### Result

The result of the inference is saved in `acc.log` in the current directory, and the result similar to the following:

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.520
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.146
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.433
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.459
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.633
mAP: 0.36858371862143824
```

## [Model Description](#content)

### [Performance](#content)

#### Training Performance

| Parameters                     | Ascend (8 pcs)                                                                         | GPU (1 pcs)                                                                            | GPU (8 pcs)                                                                            |
| ------------------------------ |----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Model                          | Retinanet-resnet-101                                                                   | Retinanet-resnet-101                                                                   | Retinanet-resnet-101                                                                   |
| Environment                    | Huawei cloud Modelarts                                                                 | Ubuntu 18.04.6, 1pcs Tesla V100-PCIE 32G, CPU 2.90GHz, 64cores, RAM 252GB              | Ubuntu 18.04.6, 8pcs Tesla V100-PCIE 32G, CPU 2.90GHz, 64cores, RAM 252GB                |
| Uploaded Date                  | 10/03/2021                                                                             | 27/12/2021                                                                             | 27/12/2021                                                                             |
| MindSpore Version              | 1.0.1                                                                                  | 1.6.0                                                                                  | 1.6.0                                                                                  |
| Dataset                        | 118287 images                                                                          | 118287 images                                                                          | 118287 images                                                                          |
| Training Parameters            | batch_size=32                                                                          | batch_size=16, epochs=400, lr=0.025                                                    | batch_size=12, epochs=400, lr=0.025                                                    |
| Other training parameters      | default_config.yaml                                                                    | default_config.yaml                                                                    | default_config.yaml                                                                    |
| Optimizer                      | Momentum                                                                               | Momentum                                                                               | Momentum                                                                               |
| Loss function                  | Focal loss                                                                             | Focal loss                                                                             | Focal loss                                                                             |
| Final loss                     | 0.43                                                                                   | 0.49                                                                                   | 0.49                                                                                   |
| Speed                          |                                                                                        | 696 ms/step                                                                            | 881 ms/step                                                                            |
| Total training time            | 34h 50m 20s                                                                            | 708h                                                                                   | 120h                                                                                    |
| Script                         | [Link](https://gitee.com/mindspore/models/tree/r2.0/research/cv/retinanet_resnet101) | [Link](https://gitee.com/mindspore/models/tree/r2.0/research/cv/retinanet_resnet101) | [Link](https://gitee.com/mindspore/models/tree/r2.0/research/cv/retinanet_resnet101) |

#### Evaluation Performance

| Parameters          | Ascend                     | GPU                                                         |
| ------------------- | :------------------------- |-------------------------------------------------------------|
| Model               | Retinanet-resnet-101       | Retinanet-resnet-101                                        |
| Environment         | Huawei cloud Modelarts     | Ubuntu 18.04.6, Tesla V100-PCIE 32G, CPU 2.90GHz, 64cores, RAM 252GB |
| Uploaded Date       | 10/03/2021                 | 27/12/2021                                                  |
| MindSpore Version   | 1.0.1                      | 1.6.0                                                       |
| Dataset             | 5k images                  | 5k images                                                   |
| Batch_size          | 1                          | 1                                                           |
| Accuracy            | mAP[0.3710]                | mAP[0.3687]                                                 |
| Total time          | 10 min 50 seconds          | 10 min                                                      |

# [Description of Random State](#content)

Random seed is set in the `train.py` script.

# [ModelZoo Homepage](#content)

Please check the official [homepage](https://gitee.com/mindspore/models).

# FAQ

Refer to [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) to find some common public issues.

- **Q: What to do if memory overflow occurs when using PYNATIVE_MODE？**
- **A**：Memory overflow is usually because PYNATIVE_MODE requires more memory， Set the batch size to 16 to reduce memory consumption and allow network training.