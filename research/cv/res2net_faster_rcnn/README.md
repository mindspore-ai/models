# Contents

- [FasterRcnn Description](#fasterrcnn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Run in docker](#Run-in-docker)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training Process](#training-process)
        - [Training Usage](#usage)
        - [Training Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation Usage](#usage)
        - [Evaluation Result](#result)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# FasterRcnn Description

Before FasterRcnn, the target detection networks rely on the region proposal algorithm to assume the location of targets, such as SPPnet and Fast R-CNN. Progress has reduced the running time of these detection networks, but it also reveals that the calculation of the region proposal is a bottleneck.

FasterRcnn proposed that convolution feature maps based on region detectors (such as Fast R-CNN) can also be used to generate region proposals. At the top of these convolution features, a Region Proposal Network (RPN) is constructed by adding some additional convolution layers (which share the convolution characteristics of the entire image with the detection network, thus making it possible to make regions almost costlessProposal), outputting both region bounds and objectness score for each location.Therefore, RPN is a full convolutional network (FCN), which can be trained end-to-end, generate high-quality region proposals, and then fed into Fast R-CNN for detection.

[Paper](https://arxiv.org/abs/1506.01497):   Ren S , He K , Girshick R , et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 39(6).

# Model Architecture

FasterRcnn is a two-stage target detection network,This network uses a region proposal network (RPN), which can share the convolution features of the whole image with the detection network, so that the calculation of region proposal is almost cost free. The whole network further combines RPN and FastRcnn into a network by sharing the convolution features.

# Dataset

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2017](<https://cocodataset.org/>)

- Dataset size：19G
    - Train：18G，118000 images  
    - Val：1G，5000 images
    - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
    - Note：Data will be processed in dataset.py

# Environment Requirements

- Hardware（Ascend/GPU）

    - Prepare hardware environment with Ascend processor.

- Docker base image

    - [Ascend Hub](ascend.huawei.com/ascendhub/#/home)

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2017.

- We use COCO2017 as training dataset in this example by default, and you can also use your own datasets.

    1. If coco dataset is used. **Select dataset to coco when run script.**
        Install Cython and pycocotool, and you can also install mmcv to process data.

        ```bash
        pip install Cython

        pip install pycocotools

        pip install mmcv==0.2.14
        ```

        And change the COCO_ROOT and other settings you need in `default_config.yaml、default_config_101.yaml or default_config_152.yaml`. The directory structure is as follows:

        ```bash
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017

        ```

    2. If your own dataset is used. **Select dataset to other when run script.**
        Organize the dataset information into a TXT file, each row in the file is as follows:

        ```bash
        train2017/0000001.jpg 0,259,401,459,7,0 35,28,324,201,2,0 0,30,59,80,2,0
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box, class and if it is crowd information of the format [xmin,ymin,xmax,ymax,class,is_crowd]. We read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are setting in `default_config_50.yaml、default_config_101.yaml or default_config_152.yaml`.

# Quick Start

After installing MindSpore via the official website, you can start training and evaluation as follows:

Note:

1. the first run will generate the mindeocrd file, which will take a long time.
2. pretrained model is a res2net50 checkpoint that trained over ImageNet2012.you can train it with [res2net50]() scripts in modelzoo, and use src/convert_checkpoint.py to get the pretrain model.
3. BACKBONE_MODEL is a checkpoint file trained with [res2net50]() scripts in modelzoo.PRETRAINED_MODEL is a checkpoint file after convert.VALIDATION_JSON_FILE is label file. CHECKPOINT_PATH is a checkpoint file after trained.

## Run on Ascend

```bash

# convert checkpoint
python convert_checkpoint.py --ckpt_file=[BACKBONE_MODEL]

# standalone training
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# distributed training
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# eval
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE_PATH]
```

## Run on GPU

```bash

# convert checkpoint
python convert_checkpoint.py --ckpt_file=[BACKBONE_MODEL]

# standalone training
bash run_standalone_train_gpu.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# distributed training
bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# eval
bash run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)
```

## Run in docker

1. Build docker images

```bash
# build docker
docker build -t fasterrcnn:20.1.0 . --build-arg FROM_IMAGE_NAME=ascend-mindspore-arm:20.1.0
```

2. Create a container layer over the created image and start it

```bash
# start docker
bash scripts/docker_start.sh fasterrcnn:20.1.0 [DATA_DIR] [MODEL_DIR]
```

3. Train

```bash
# standalone training
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# distributed training
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)
```

4. Eval

```bash
# eval
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)
```

5. Inference

```bash
# inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "distribute=True" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "epoch_size: 20" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "dataset_path=/cache/data" on the website UI interface.
    #          Add "epoch_size: 20" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/faster_rcnn" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "epoch_size: 20" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add "epoch_size: 20" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/faster_rcnn" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on base_config.yaml file.
    #          Set "checkpoint='./faster_rcnn/faster_rcnn_trained.ckpt'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_trained_model/'" on the website UI interface.
    #          Add "checkpoint='./faster_rcnn/faster_rcnn_trained.ckpt'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/faster_rcnn" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Export s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```bash
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "file_name='faster_rcnn'" on base_config.yaml file.
    #          Set "file_format='MINDIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name='faster_rcnn'" on the website UI interface.
    #          Add "file_format='MINDIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/faster_rcnn" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

# Script Description

## Script and Sample Code

```bash
.
└─faster_rcnn
  ├─README.md                         // descriptions about fasterrcnn
  ├─ascend310_infer                   //application for 310 inference
  ├─scripts
    ├─run_standalone_train_ascend.sh  // shell script for standalone on ascend
    ├─run_standalone_train_gpu.sh     // shell script for standalone on GPU
    ├─run_distribute_train_ascend.sh  // shell script for distributed on ascend
    ├─run_distribute_train_gpu.sh     // shell script for distributed on GPU
    ├─run_infer_310.sh                // shell script for 310 inference
    └─run_eval_ascend.sh              // shell script for eval on ascend
    └─run_eval_gpu.sh                 // shell script for eval on GPU
  ├─src
    ├─FasterRcnn
      ├─__init__.py                   // init file
      ├─anchor_generator.py           // anchor generator
      ├─bbox_assign_sample.py         // first stage sampler
      ├─bbox_assign_sample_stage2.py  // second stage sampler
      ├─faster_rcnn_res2net.py         // fasterrcnn network
      ├─fpn_neck.py                   //feature pyramid network
      ├─proposal_generator.py         // proposal generator
      ├─rcnn.py                       // rcnn network
      ├─res2net.py                     // res2net backbone network
      ├─roi_align.py                  // roi align network
      └─rpn.py                        //  region proposal network
    ├─dataset.py                      // create dataset and process dataset
    ├─lr_schedule.py                  // learning ratio generator
    ├─network_define.py               // network define for fasterrcnn
    ├─util.py                         // routine operation
    └─model_utils
      ├─config.py                     // Processing configuration parameters
      ├─device_adapter.py             // Get cloud ID
      ├─local_adapter.py              // Get local ID
      └─moxing_adapter.py             // Parameter processing
  ├─default_config.yaml               // config for Res2Net50
  ├─default_config_101.yaml           // config for Res2Net101
  ├─default_config_152.yaml           // config for Res2Net152
  ├─export.py                         // script to export AIR,MINDIR,ONNX model
  ├─eval.py                           //eval scripts
  ├─postprogress.py                   // post process for 310 inference
  └─train.py                          // train scripts
```

```bash
`BACKBONE` should be in ["res2net_50", "res2net_101", "res2net_152"]

if backbone in ("res2net_50", "res2net_101", "res2net_152"):
    from src.FasterRcnn.faster_rcnn_res2net import Faster_Rcnn_Res2net
    "res2net_50" -> "./default_config.yaml"
    "res2net_101"  -> "./default_config_101.yaml"
    "res2net_152"  -> "./default_config_152.yaml"

```

## Training Process

### Usage

#### on Ascend

```bash
# standalone training on ascend
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# distributed training on ascend
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)
```

#### on GPU

```bash
# standalone training on gpu
bash run_standalone_train_gpu.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# distributed training on gpu
bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)
```

Notes:

1. Rank_table.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).
2. As for PRETRAINED_MODEL，it should be a trained Res2Net50 checkpoint. If you need to load Ready-made pretrained FasterRcnn checkpoint, you may make changes to the train.py script as follows.

```bash
# Comment out the following code
#   load_path = args_opt.pre_trained
#    if load_path != "":
#        param_dict = load_checkpoint(load_path)
#        for item in list(param_dict.keys()):
#            if not item.startswith('backbone'):
#                param_dict.pop(item)
#        load_param_into_net(net, param_dict)

# Add the following codes after optimizer definition since the FasterRcnn checkpoint includes optimizer parameters：
    lr = Tensor(dynamic_lr(config, rank_size=device_num), mstype.float32)
    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    if load_path != "":
        param_dict = load_checkpoint(load_path)
        for item in list(param_dict.keys()):
            if item in ("global_step", "learning_rate") or "rcnn.reg_scores" in item or "rcnn.cls_scores" in item:
                param_dict.pop(item)
        load_param_into_net(opt, param_dict)
        load_param_into_net(net, param_dict)
```

3. The original dataset path needs to be in the default_config_50.yaml、default_config_101.yaml、default_config_152.yaml,you can select "coco_root" or "image_dir".

### Result

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the following in loss_rankid.log.

```bash
# distribute training result(8p)
epoch: 1 step: 7393, rpn_loss: 0.12054, rcnn_loss: 0.40601, rpn_cls_loss: 0.04025, rpn_reg_loss: 0.08032, rcnn_cls_loss: 0.25854, rcnn_reg_loss: 0.14746, total_loss: 0.52655
epoch: 2 step: 7393, rpn_loss: 0.06561, rcnn_loss: 0.50293, rpn_cls_loss: 0.02587, rpn_reg_loss: 0.03967, rcnn_cls_loss: 0.35669, rcnn_reg_loss: 0.14624, total_loss: 0.56854
epoch: 3 step: 7393, rpn_loss: 0.06940, rcnn_loss: 0.49658, rpn_cls_loss: 0.03769, rpn_reg_loss: 0.03165, rcnn_cls_loss: 0.36353, rcnn_reg_loss: 0.13318, total_loss: 0.56598
...
epoch: 10 step: 7393, rpn_loss: 0.03555, rcnn_loss: 0.32666, rpn_cls_loss: 0.00697, rpn_reg_loss: 0.02859, rcnn_cls_loss: 0.16125, rcnn_reg_loss: 0.16541, total_loss: 0.36221
epoch: 11 step: 7393, rpn_loss: 0.19849, rcnn_loss: 0.47827, rpn_cls_loss: 0.11639, rpn_reg_loss: 0.08209, rcnn_cls_loss: 0.29712, rcnn_reg_loss: 0.18115, total_loss: 0.67676
epoch: 12 step: 7393, rpn_loss: 0.00691, rcnn_loss: 0.10168, rpn_cls_loss: 0.00529, rpn_reg_loss: 0.00162, rcnn_cls_loss: 0.05426, rcnn_reg_loss: 0.04745, total_loss: 0.10859
```

## Evaluation Process

### Usage

#### on Ascend

```bash
# eval on ascend
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)
```

#### on GPU

```bash
# eval on GPU
bash run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)
```

> checkpoint can be produced in training process.
>
> Images size in dataset should be equal to the annotation size in VALIDATION_JSON_FILE, otherwise the evaluation result cannot be displayed properly.

### Result

Eval result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.403
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.630
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.437
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.243
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.443
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.534
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.360
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.580
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.666
```

## Model Export

```bash
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT] --backbone [BACKBONE]
```

`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]
`BACKBONE` should be in ["res2net_50", "res2net_101", "res2net_152"]

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Usage

Before performing inference, the model file must be exported by export script on the Ascend910 environment.
The following example only supports mindir inference with batch_size=1.

```bash
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --config_path [CONFIG_PATH]
```

- The `ckpt_file` parameter is required.
- `file_format` should be in ["AIR", "MINDIR"].
- Default `config_path` is `default_config.yaml`.

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

- `DATA_PATH` is the path of test dataset.
- `ANNO_PATH` is the annotation file path of test dataset.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

# Model Description

## Performance

### Evaluation Performance

#### ResNet-50 backbone

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |
| uploaded Date              | 12/20/2020 (month/day/year)                                 |
| MindSpore Version          | 1.5.0                                                       |
| Dataset                    | COCO2017                                                    |
| Training Parameters        | epoch=12,  batch_size=2                                     |
| Optimizer                  | SGD                                                         |
| Loss Function              | Softmax Cross Entropy ,Sigmoid Cross Entropy,SmoothL1Loss   |
| Speed                      | 1pc: 17.36 ms/step;  8pcs: 127.30 ms/step                   |
| Parameters (M)             | 125.4                                                       |