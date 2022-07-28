# Contents

- [Contents](#contents)
- [TextFuseNet Description](#textfusenet-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Training Script Parameters](#training-script-parameters)
        - [Parameters Configuration](#parameters-configuration)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
        - [Training Result](#training-result)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
        - [Evaluation result](#evaluation-result)
    - [Model Export](#model-export)
    - [Inference Process](#inference-process)
        - [Usage](#usage)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [TextFuseNet Description](#contents)

Arbitrary shape text detection in natural scenes is an extremely challenging task. Unlike existing text detection approaches that only perceive texts based on limited feature representations, This paper proposes a novel framework, namely TextFuseNet, to exploit the use of richer features fused for text detection. More specifically, this paper propose to perceive texts from three levels of feature representations, i.e., character-, word- and global-level, and then introduce a novel text representation fusion technique to help achieve robust arbitrary text detection. The multi-level feature representation can adequately describe texts by dissecting them into individual characters while still maintaining their general semantics. TextFuseNet then collects and merges the texts’ features from different levels using a multi-path fusion architecture which can effectively align and fuse different representations. In practice, TextFuseNet can learn a more adequate description of arbitrary shapes texts, suppressing false positives and producing more accurate detection results. This paper proposed framework can also be trained with weak supervision for those datasets that lack character-level annotations

[Paper](https://www.ijcai.org/Proceedings/2020/72): Jian Ye, Zhe Chen, Juhua Liu, Bo Du. "TextFuseNet: Scene Text Detection with Richer Fused Features"

# [Dataset](#contents)

- [TotalText](https://github.com/cs-chan/Total-Text-Dataset/) is a comprehensive arbitrary shape text dataset for scene text reading. Total-Text contains 1255 training images and 300 test images. All images are annotated with polygons in word-level

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- Docker base image
    - [Ascend Hub](ascend.huawei.com/ascendhub/#/home)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

- third-party libraries

```bash
mindspore==1.5.0
mindspore_ascend==1.3.0
mmcv==0.2.14
numpy==1.21.0rc1
opencv_python==4.5.1.48
Pillow==8.4.0
pycocotools==2.0.0
PyYAML==6.0
Shapely==1.5.9
```

# [Quick Start](#contents)

1. Download the dataset TotalText.

2. Change the COCO_ROOT and other settings you need in `config.py`. The directory structure should look like the follows:

    ```text
    .
    └─data
      ├─annotations
        ├─instance_train.json
        └─instance_test.json
      ├─test
      └─train
    ```

3. Execute train script.
    After dataset preparation, you can start training as follows:

    ```text
    # distributed training
    bash run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_CKPT]
    # standalone training
    bash run_standalone_train.sh [PRETRAINED_CKPT]
    ```

    Note:
    1. To speed up data preprocessing, MindSpore provide a data format named MindRecord, hence the first step is to generate MindRecord files based on TotalText dataset before training.
    2. For distributed training, a [hccl configuration file](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools) with JSON format needs to be created in advance.
    3. PRETRAINED_CKPT is a resnet101 checkpoint that trained over ImageNet2012.you can train it with [resnet101](https://download.mindspore.cn/model_zoo/r1.3/resnet101_ascend_v130_imagenet2012_official_cv_bs32_top1acc78.55__top5acc94.34/) scripts in modelzoo, and use src/convert_checkpoint.py to get the pretrain checkpoint file.
    4. For character-level labeling, first use Synthtext dataset to train MaskRCNN to obtain model M, and then use model M to complete TotalText character-level labeling. Convert the dataset annotations to [COCO](https://cocodataset.org/) format
4. Execute eval script.
   After training, you can start evaluation as follows:

   ```shell
   # Evaluation
   bash run_eval.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
   ```

   Note:
   1. VALIDATION_JSON_FILE is a label json file for evaluation.

5. Execute inference script.
   After training, you can start inference as follows:

   ```shell
   # inference
   bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE_PATH]
   ```

   Note:
   1. MINDIR_PATH is a model file, exported by export script file on the Ascend910 environment.
   2. ANN_FILE_PATH is a annotation file for inference.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─TextFuseNet
  ├─README.md                             # README
  ├─ascend310_infer                       # application for 310 inference
  ├─scripts                               # shell script
    ├─run_standalone_train.sh             # training in standalone mode(1pcs)
    ├─run_distribute_train.sh             # training in parallel mode(8 pcs)
    ├─run_infer_310.sh                    # shell script for 310 inference
    └─run_eval.sh                         # evaluation
  ├─src
    ├─textfusenet
      ├─__init__.py
      ├─anchor_generator.py               # generate base bounding box anchors
      ├─bbox_assign_sample.py             # filter positive and negative bbox for the first stage learning
      ├─bbox_assign_sample_stage2.py      # filter positive and negative bbox for the second stage learning
      ├─text_fuse_net_r101.py             # main network architecture of textfusenet
      ├─fpn_neck.py                       # fpn network
      ├─proposal_generator.py             # generate proposals based on feature map
      ├─rcnn_cls.py                       # rcnn bounding box regression branch
      ├─rcnn_mask.py                      # rcnn mask branch
      ├─rcnn_seg.py                       # rcnn seg branch
      ├─mutil_path_fuse.py                # mutil path fuse branch
      ├─resnet101.py                      # backbone network
      ├─roi_align.py                      # roi align network
      └─rpn.py                            # reagion proposal network
    ├─convert_checkpoint.py               # convert resnet101 backbone checkpoint
    ├─dataset.py                          # dataset utils
    ├─lr_schedule.py                      # leanring rate geneatore
    ├─network_define.py                   # network define for textfusenet
    ├─util.py                             # routine operation
    └─model_utils
      ├─config.py                         # Processing configuration parameters
      ├─device_adapter.py                 # Get cloud ID
      ├─local_adapter.py                  # Get local ID
      └─moxing_adapter.py                 # Parameter processing
  ├─default_config.yaml                   # Training parameter profile
  ├─mindspore_hub_conf.py                 # mindspore hub interface
  ├─export.py                             # script to export AIR,MINDIR,ONNX model
  ├─eval.py                               # evaluation scripts
  ├─postprogress.py                       # post process for 310 inference
  └─train.py                              # training scripts
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]

# standalone training
Usage: bash run_standalone_train.sh [PRETRAINED_MODEL]
```

### [Parameters Configuration](#contents)

```bash
"img_width": 1280,          # width of the input images
"img_height": 768,          # height of the input images

# random threshold in data augmentation
"keep_ratio": True,
"flip_ratio": 0.5,
"expand_ratio": 1.0,

"max_instance_count": 128, # max number of bbox for each image
"mask_shape": (28, 28),    # shape of mask in rcnn_mask

# anchor
"feature_shapes": [(192, 320), (96, 160), (48, 80), (24, 40), (12, 20)], # shape of fpn feaure maps
"anchor_scales": [8],                                                    # area of base anchor
"anchor_ratios": [0.5, 1.0, 2.0],                                        # ratio between width of height of base anchors
"anchor_strides": [4, 8, 16, 32, 64],                                    # stride size of each feature map levels
"num_anchors": 3,                                                        # anchor number for each pixel

# resnet
"resnet_block": [3, 4, 6, 3],                                            # block number in each layer
"resnet_in_channels": [64, 256, 512, 1024],                              # in channel size for each layer
"resnet_out_channels": [256, 512, 1024, 2048],                           # out channel size for each layer

# fpn
"fpn_in_channels": [256, 512, 1024, 2048],                               # in channel size for each layer
"fpn_out_channels": 256,                                                 # out channel size for every layer
"fpn_num_outs": 5,                                                       # out feature map size

# rpn
"rpn_in_channels": 256,                                                  # in channel size
"rpn_feat_channels": 256,                                                # feature out channel size
"rpn_loss_cls_weight": 1.0,                                              # weight of bbox classification in rpn loss
"rpn_loss_reg_weight": 1.0,                                              # weight of bbox regression in rpn loss
"rpn_cls_out_channels": 1,                                               # classification out channel size
"rpn_target_means": [0., 0., 0., 0.],                                    # bounding box decode/encode means
"rpn_target_stds": [1.0, 1.0, 1.0, 1.0],                                 # bounding box decode/encode stds
"textfusenet_channels": 256,

# bbox_assign_sampler
"neg_iou_thr": 0.3,                                                      # negative sample threshold after IOU
"pos_iou_thr": 0.7,                                                      # positive sample threshold after IOU
"min_pos_iou": 0.3,                                                      # minimal positive sample threshold after IOU
"num_bboxes": 245520,                                                    # total bbox number
"num_gts": 128,                                                          # total ground truth number
"num_expected_neg": 256,                                                 # negative sample number
"num_expected_pos": 128,                                                 # positive sample number

# proposal
"activate_num_classes": 2,                                               # class number in rpn classification
"use_sigmoid_cls": True,                                                 # whethre use sigmoid as loss function in rpn classification

# roi_align
"roi_layer": dict(type='RoIAlign', out_size=7, mask_out_size=14, sample_num=2), # ROIAlign parameters
"roi_align_out_channels": 256,                                                  # ROIAlign out channels size
"roi_align_featmap_strides": [4, 8, 16, 32],                                    # stride size for different level of ROIAling feature map
"roi_align_finest_scale": 56,                                                   # finest scale ofr ROIAlign
"roi_sample_num": 640,                                                          # sample number in ROIAling layer

# bbox_assign_sampler_stage2                                                    # bbox assign sample for the second stage, parameter meaning is similar with bbox_assign_sampler
"neg_iou_thr_stage2": 0.5,
"pos_iou_thr_stage2": 0.5,
"min_pos_iou_stage2": 0.5,
"num_bboxes_stage2": 2000,
"num_expected_pos_stage2": 128,
"num_expected_neg_stage2": 512,
"num_expected_total_stage2": 512,

# rcnn                                                                          # rcnn parameter for the second stage, parameter meaning is similar with fpn
"rcnn_num_layers": 2,
"rcnn_in_channels": 256,
"rcnn_fc_out_channels": 1024,
"rcnn_mask_out_channels": 256,
"rcnn_loss_cls_weight": 1,
"rcnn_loss_reg_weight": 1,
"rcnn_loss_mask_fb_weight": 1,
"rcnn_target_means": [0., 0., 0., 0.],
"rcnn_target_stds": [0.1, 0.1, 0.2, 0.2],

# train proposal
"rpn_proposal_nms_across_levels": False,
"rpn_proposal_nms_pre": 2000,                                                  # proposal number before nms in rpn
"rpn_proposal_nms_post": 2000,                                                 # proposal number after nms in rpn
"rpn_proposal_max_num": 2000,                                                  # max proposal number in rpn
"rpn_proposal_nms_thr": 0.7,                                                   # nms threshold for nms in rpn
"rpn_proposal_min_bbox_size": 0,                                               # min size of box in rpn

# test proposal                                                                # part of parameters are similar with train proposal
"rpn_nms_across_levels": False,
"rpn_nms_pre": 1000,
"rpn_nms_post": 1000,
"rpn_max_num": 1000,
"rpn_nms_thr": 0.7,
"rpn_min_bbox_min_size": 0,
"test_roi_number": 100,
"test_score_thr": 0.05,                                                        # score threshold
"test_iou_thr": 0.5,                                                           # IOU threshold
"test_max_per_img": 100,                                                       # max number of instance
"test_batch_size": 2,                                                          # batch size

"rpn_head_use_sigmoid": True,                                                  # whether use sigmoid or not in rpn
"rpn_head_weight": 1.0,                                                        # rpn head weight in loss
"mask_thr_binary": 0.5,                                                        # mask threshold for in rcnn

# LR
"base_lr": 0.02,                                                               # base learning rate
"base_step": 58633,                                                            # bsae step in lr generator
"total_epoch": 13,                                                             # total epoch in lr generator
"warmup_step": 500,                                                            # warmp up step in lr generator
"warmup_ratio": 1/3.0,                                                         # warpm up ratio
"sgd_momentum": 0.9,                                                           # momentum in optimizer

# train
"batch_size": 2,
"loss_scale": 1,
"momentum": 0.91,
"weight_decay": 1e-4,
"pretrain_epoch_size": 0,                                                      # pretrained epoch size
"epoch_size": 12,                                                              # total epoch size
"save_checkpoint": True,                                                       # whether save checkpoint or not
"save_checkpoint_epochs": 1,                                                   # save checkpoint interval
"keep_checkpoint_max": 12,                                                     # max number of saved checkpoint
"save_checkpoint_path": "./",                                                  # path of checkpoint

"mindrecord_dir": "/home/textfusenet/MindRecord_TotalText_Train",                  # path of mindrecord
"coco_root": "/home/textfusenet/",                                                # path of totaltext root dateset
"train_data_type": "train2017",                                                # name of train dataset
"val_data_type": "val2017",                                                    # name of evaluation dataset
"instance_set": "annotations/instances_{}.json",                               # name of annotation
"coco_classes": ('background', 'text', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C','D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R','S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v','w', 'x', 'y', 'z'),
"num_classes": 64
```

## [Training Process](#contents)

- Set options in `config.py`, including loss_scale, learning rate and network hyperparameters. Click [here](https://www.mindspore.cn/docs/programming_guide/en/r1.3/dataset_sample.html) for more information about dataset.

### [Training](#content)

- Run `run_standalone_train.sh` for non-distributed training of TextFuseNet model.

```bash
# standalone training
bash run_standalone_train.sh [PRETRAINED_MODEL]
```

### [Distributed Training](#content)

- Run `run_distribute_train.sh` for distributed training of TextFuseNet model.

```bash
bash run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

- Notes
1. hccl.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).
2. As for PRETRAINED_MODEL，it should be a trained ResNet101 checkpoint. If not set, the model will be trained from the very beginning. If you need to load Ready-made pretrained TextFuseNet checkpoint, you may make changes to the train.py script as follows.

```python
    if load_path != "":
        param_dict = load_checkpoint(load_path)
        if config.pretrain_epoch_size == 0:
            for item in list(param_dict.keys()):
                if item in ("global_step", "learning_rate") or "rcnn.cls" in item or "rcnn.mask" in item:
                    param_dict.pop(item)
        load_param_into_net(net, param_dict)
        load_param_into_net(opt, param_dict)
```

3. This is processor cores binding operation regarding the `device_num` and total processor numbers. If you are not expect to do it, remove the operations `taskset` in `scripts/run_distribute_train.sh`

### [Training Result](#content)

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the following in loss_rankid.log.

```bash
# distribute training result(2p)
3893 epoch: 1 step: 151 total_loss: 2.65625
3923 epoch: 2 step: 151 total_loss: 2.40820
3953 epoch: 3 step: 151 total_loss: 2.44922
3983 epoch: 4 step: 151 total_loss: 2.48828
4013 epoch: 5 step: 151 total_loss: 1.35156
4043 epoch: 6 step: 151 total_loss: 1.79297
4073 epoch: 7 step: 151 total_loss: 2.24414
4102 epoch: 8 step: 151 total_loss: 1.33496
4132 epoch: 9 step: 151 total_loss: 0.67822
4162 epoch: 10 step: 151 total_loss: 1.76172
4192 epoch: 11 step: 151 total_loss: 0.90430
4222 epoch: 12 step: 151 total_loss: 1.92773
4252 epoch: 13 step: 151 total_loss: 1.85840
4282 epoch: 14 step: 151 total_loss: 1.33984
4312 epoch: 15 step: 151 total_loss: 1.61719
4342 epoch: 16 step: 151 total_loss: 1.52441
4372 epoch: 17 step: 151 total_loss: 1.34863
```

## [Evaluation Process](#contents)

### [Evaluation](#content)

- Run `run_eval.sh` for evaluation.
- Download the evaluation code [TIOU-metric-python3](https://github.com/PkuDavidGuan/TIoU-metric-python3.git)
1. Rename the code to eval_code and place it in the scripts directory
2. Place total-text-gt.zip in the code in the scripts directory
3. Comment out lines 232 and 233 in eval_code/curved_tiou/rrc_evaluation_funcs.py
4. Execute the following commands

```bash
# infer
bash run_eval.sh [VALIDATION_ANN_FILE_JSON] [CHECKPOINT_PATH]
```

> As for the TotalText dataset, VALIDATION_ANN_FILE_JSON is refer to the annotations/instances_val2017.json in the dataset directory.  
> checkpoint can be produced and saved in training process, whose folder name begins with "train/checkpoint" or "train_parallel*/checkpoint".
>
> Images size in dataset should be equal to the annotation size in VALIDATION_ANN_FILE_JSON, otherwise the evaluation result cannot be displayed properly.

### [Evaluation result](#content)

Inference result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.
You need to download the evaluation code before inference [TIOU-metric-python3](https://github.com/PkuDavidGuan/TIoU-metric-python3.git)
And place the evaluation code and total-text-gt.zip in the code under the scripts directory

```bash
num_gt, num_det:  2214 2430
Origin:
recall:  0.8071 precision:  0.8258 hmean:  0.8164
```

## Model Export

```shell
python export.py --config_path [CONFIG_FILE] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

## Inference Process

### Usage

Before performing inference, the air file must bu exported by export script on the 910 environment.
Current batch_ Size can only be set to 1.You need to download the evaluation code before inference [TIOU-metric-python3](https://github.com/PkuDavidGuan/TIoU-metric-python3.git)

1. Rename the code to eval_code and place it in the scripts directory
2. Place total-text-gt.zip in the code in the scripts directory
3. Comment out lines 232 and 233 in eval_code/curved_tiou/rrc_evaluation_funcs.py

```shell
# Ascend310 inference
sh run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_ID]
```

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
num_gt, num_det:  2214 2422
Origin:
recall:  0.8035 precision:  0.8282 hmean:  0.8157
```

# Model Description

## Performance

### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8             |
| uploaded Date              | 10/28/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | TotalText                                                    |
| Training Parameters        | epoch=200,  batch_size = 1                                  |
| Optimizer                  | SGD                                                         |
| Loss Function              | Softmax Cross Entropy, Sigmoid Cross Entropy, SmoothL1Loss  |
| Output                     | Probability                                                 |
| Loss                       | 0.39804                                                     |
| Speed                      | 1pc: 333 ms/step;  8pcs: 300 ms/step                        |
| Total time                 | 1pc: 22.4 hours;  8pcs: 2.1 hours                            |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | V1                          |
| Resource            | Ascend 910; OS Euler2.8                  |
| Uploaded Date       | 10/28/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       |
| Dataset             | TotalText                    |
| batch_size          | 1                           |
| outputs             | mAP                         |
| Accuracy            | IoU=0.50 (Mask 81.64%) |
| Model for inference | 563 (.ckpt file)           |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py for weight initialization.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
