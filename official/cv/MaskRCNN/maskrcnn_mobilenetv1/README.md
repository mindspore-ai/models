# Contents

- [Contents](#contents)
- [MaskRCNN Description](#maskrcnn-description)
- [Model Architecture](#model-architecture)
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
    - [Export Process](#export-process)
        - [Export](#export)
    - [Inference Process](#inference-process)
        - [Inference](#inference)
        - [Inference result](#inference-result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [MaskRCNN Description](#contents)

MaskRCNN is a conceptually simple, flexible, and general framework for object instance segmentation. The approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in
parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing to estimate human poses in the same framework.
It shows top results in all three tracks of the COCO suite of challenges, including instance segmentation, boundingbox object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners.

# [Model Architecture](#contents)

MaskRCNN is a two-stage target detection network. It extends FasterRCNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition.This network uses a region proposal network (RPN), which can share the convolution features of the whole image with the detection network, so that the calculation of region proposal is almost cost free. The whole network further combines RPN and mask branch into a network by sharing the convolution features.
This network uses MobileNetV1 as the backbone of the maskrcnn_mobilenetv1 network.

[Paper](http://cn.arxiv.org/pdf/1703.06870v3): Kaiming He, Georgia Gkioxari, Piotr Dollar and Ross Girshick. "MaskRCNN"

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- [COCO2017](https://cocodataset.org/) is a popular dataset with bounding-box and pixel-level stuff annotations. These annotations can be used for scene understanding tasks like semantic segmentation, object detection and image captioning. There are 118K/5K images for train/val.

- Dataset size: 19G
    - Train: 18G, 118000 images
    - Val: 1G, 5000 images
    - Annotations: 241M, instances, captions, person_keypoints, etc.

- Data format: image and json files
    - Note: Data will be processed in dataset.py

# [Environment Requirements](#contents)

- Hardware（Ascend/CPU）
    - Prepare hardware environment with Ascend or CPU processor or GPU.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

- third-party libraries

    ```bash
    pip install Cython
    pip install pycocotools
    pip install mmcv=0.2.14
    ```

# [Quick Start](#contents)

1. Download the dataset COCO2017.

2. Change the COCO_ROOT and other settings you need in `default_config.yaml`. The directory structure should look like the follows:

    ```
    .
    └─cocodataset
      ├─annotations
        ├─instance_train2017.json
        └─instance_val2017.json
      ├─val2017
      └─train2017
    ```

    If you use your own dataset to train the network, **Select dataset to other when run script.**
    Create a txt file to store dataset information organized in the way as shown as following:

    ```
    train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
    ```

    Each row is an image annotation split by spaces. The first column is a relative path of image, followed by columns containing box and class information in the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `IMAGE_DIR`(dataset directory) and the relative path in `ANNO_PATH`(the TXT file path), which can be set in `default_config.yaml`.

3. Execute train script.
    After dataset preparation, you can start training as follows:

    ```bash
    On Ascend:

    # distributed training
    bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [PRETRAINED_CKPT(optional)]
    # example: bash run_distribute_train.sh ~/hccl_8p.json /home/DataSet/cocodataset/

    # standalone training
    bash run_standalone_train.sh [DATA_PATH] [DEVICE_ID] [PRETRAINED_PATH](optional)
    # example: bash run_standalone_train.sh /home/DataSet/cocodataset/ 0

    On CPU:

    # standalone training
    bash run_standalone_train_cpu.sh [PRETRAINED_PATH](optional)

    On GPU:

    # distributed training
    bash run_distribute_train_gpu.sh [DATA_PATH] [PRETRAINED_PATH] (optional)
    ```

    Note:
    1. To speed up data preprocessing, MindSpore provide a data format named MindRecord, hence the first step is to generate MindRecord files based on COCO2017 dataset before training. The process of converting raw COCO2017 dataset to MindRecord format may take about 4 hours.
    2. For distributed training, a [hccl configuration file](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools) with JSON format needs to be created in advance.
    3. For large models like maskrcnn_mobilenetv1, it's better to export an external environment variable `export HCCL_CONNECT_TIMEOUT=600` to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size.

4. Execute eval script.

    After training, you can start evaluation as follows:

    ```bash
    # Evaluation on Ascend
    bash run_eval.sh [ANN_FILE] [CHECKPOINT_PATH] [DATA_PATH] [DEVICE_TARGET] [DEVICE_ID]
    # example: bash run_eval.sh /home/DataSet/cocodataset/annotations/instances_val2017.json /home/model/maskrcnn_mobilenetv1/ckpt/mask_rcnn-5_7393.ckpt /home/DataSet/cocodataset/ "Ascend" 0

    # Evaluation on CPU
    bash run_eval_cpu.sh [ANN_FILE] [CHECKPOINT_PATH]

    # Evaluation on GPU
    bash run_eval.sh [ANN_FILE] [CHECKPOINT_PATH] [DATA_PATH] [DEVICE_TARGET] [DEVICE_ID]
    ```

    Note:
    1. VALIDATION_JSON_FILE is a label json file for evaluation.

5. Execute inference script.

    After training, you can start inference as follows:

    ```shell
    # inference
    bash run_infer_310.sh [MODEL_PATH] [DATA_PATH] [ANN_FILE_PATH]
    ```

    Note:
    1. MODEL_PATH is a model file, exported by export script file.
    2. ANN_FILE_PATH is a annotation file for inference.

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "distribute=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='cocodataset'" on default_config.yaml file.
    #          Set "base_lr=0.02" on default_config.yaml file.
    #          Set "mindrecord_dir='./MindRecord_COCO'" on default_config.yaml file.
    #          Set "data_path='/cache/data'" on default_config.yaml file.
    #          Set "ann_file='./annotations/instances_val2017.json'" on default_config.yaml file.
    #          Set "epoch_size=12" on default_config.yaml file.
    #          Set "ckpt_path='./ckpt_maskrcnn/mask_rcnn-12_7393.ckpt'" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='cocodataset'" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "base_lr=0.02" on the website UI interface.
    #          Add "mindrecord_dir='./MindRecord_COCO'" on the website UI interface.
    #          Add "data_path='/cache/data'" on the website UI interface.
    #          Add "ann_file='./annotations/instances_val2017.json'" on the website UI interface.
    #          Add "epoch_size=12" on the website UI interface.
    #          Set "ckpt_path='./ckpt_maskrcnn/mask_rcnn-12_7393.ckpt'" on default_config.yaml file.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, run "train.py" like the following to create MindRecord dataset locally from coco2017.
    #             "python train.py --only_create_dataset=True --mindrecord_dir=$MINDRECORD_DIR --data_path=$DATA_PATH --ann_file=$ANNO_PATH"
    #          Second, zip MindRecord dataset to one zip file.
    #          Finally, Upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original coco dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/maskrcnn" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='cocodataset'" on default_config.yaml file.
    #          Set "mindrecord_dir='./MindRecord_COCO'" on default_config.yaml file.
    #          Set "data_path='/cache/data'" on default_config.yaml file.
    #          Set "ann_file='./annotations/instances_val2017.json'" on default_config.yaml file.
    #          Set "epoch_size=12" on default_config.yaml file.
    #          Set "ckpt_path='./ckpt_maskrcnn/mask_rcnn-12_7393.ckpt'" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='cocodataset'" on the website UI interface.
    #          Add "mindrecord_dir='./MindRecord_COCO'" on the website UI interface.
    #          Add "data_path='/cache/data'" on the website UI interface.
    #          Add "ann_file='./annotations/instances_val2017.json'" on the website UI interface.
    #          Add "epoch_size=12" on the website UI interface.
    #          Set "ckpt_path='./ckpt_maskrcnn/mask_rcnn-12_7393.ckpt'" on default_config.yaml file.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, run "train.py" like the following to create MindRecord dataset locally from coco2017.
    #             "python train.py --only_create_dataset=True --mindrecord_dir=$MINDRECORD_DIR --data_path=$DATA_PATH --ann_file=$ANNO_PATH"
    #          Second, zip MindRecord dataset to one zip file.
    #          Finally, Upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original coco dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/maskrcnn" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='cocodataset'" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on base_config.yaml file.
    #          Set "checkpoint_path='./ckpt_maskrcnn/mask_rcnn-12_7393.ckpt'" on default_config.yaml file.
    #          Set "mindrecord_file='/cache/data/cocodataset/MindRecord_COCO'" on default_config.yaml file.
    #          Set "data_path='/cache/data'" on default_config.yaml file.
    #          Set "ann_file='./annotations/instances_val2017.json'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='cocodataset'" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_trained_model/'" on the website UI interface.
    #          Add "checkpoint_path='./ckpt_maskrcnn/mask_rcnn-12_7393.ckpt'" on the website UI interface.
    #          Set "mindrecord_file='/cache/data/cocodataset/MindRecord_COCO'" on default_config.yaml file.
    #          Add "data_path='/cache/data'" on the website UI interface.
    #          Set "ann_file='./annotations/instances_val2017.json'" on default_config.yaml file.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Perform a or b. (suggested option a)
    #       a. First, run "eval.py" like the following to create MindRecord dataset locally from coco2017.
    #             "python eval.py --only_create_dataset=True --mindrecord_dir=$MINDRECORD_DIR --data_path=$DATA_PATH --ann_file=$ANNO_PATH \
    #              --checkpoint_path=$CHECKPOINT_PATH"
    #          Second, zip MindRecord dataset to one zip file.
    #          Finally, Upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original coco dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/maskrcnn" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Export s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "file_name='maskrcnn_mobilenetv1'" on base_config.yaml file.
    #          Set "file_format='MINDIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name='maskrcnn_mobilenetv1'" on the website UI interface.
    #          Add "file_format='MINDIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/maskrcnn_mobilenetv1" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─MaskRcnn_Mobilenetv1
  ├─README.md                             # README
  ├─ascend310_infer                       # application for 310 inference
  ├─scripts                               # shell script
    ├─run_standalone_train.sh             # training in standalone mode on Ascend(1pcs)
    ├─run_standalone_train_cpu.sh         # training in standalone mode on CPU(1pcs)
    ├─run_distribute_train_gpu.sh         # training in parallel mode on GPU(8pcs)
    ├─run_distribute_train.sh             # training in parallel mode on Ascend(8 pcs)
    ├─run_infer_310.sh                    # shell script for 310 inference
    ├─run_eval_cpu.sh                     # evaluation on CPU
    └─run_eval.sh                         # evaluation on Ascend or GPU
  ├─src
    ├─maskrcnn_mobilenetv1
      ├─__init__.py
      ├─anchor_generator.py               # generate base bounding box anchors
      ├─bbox_assign_sample.py             # filter positive and negative bbox for the first stage learning
      ├─bbox_assign_sample_stage2.py      # filter positive and negative bbox for the second stage learning
      ├─mask_rcnn_mobilenetv1.py          # main network architecture of maskrcnn_mobilenetv1
      ├─fpn_neck.py                       # fpn network
      ├─proposal_generator.py             # generate proposals based on feature map
      ├─rcnn_cls.py                       # rcnn bounding box regression branch
      ├─rcnn_mask.py                      # rcnn mask branch
      ├─mobilenetv1.py                    # backbone network
      ├─roi_align.py                      # roi align network
      └─rpn.py                            # reagion proposal network
    ├─util.py                             # routine operation
    ├─model_utils                         # network configuration
      ├─__init__.py
      ├─config.py                         # network configuration
      ├─device_adapter.py                 # Get cloud ID
      ├─local_adapter.py                  # Get local ID
      ├─moxing_adapter.py                 # Parameter processing
    ├─dataset.py                          # dataset utils
    ├─lr_schedule.py                      # leanring rate geneatore
    ├─network_define.py                   # network define for maskrcnn_mobilenetv1
    └─util.py                             # routine operation
  ├─default_config.yaml                   # default configuration settings
  ├─mindspore_hub_conf.py                 # mindspore hub interface
  ├─export.py                             #script to export AIR,MINDIR model
  ├─eval.py                               # evaluation scripts
  ├─postprogress.py                       #post process for 310 inference
  └─train.py                              # training scripts
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```bash
On Ascend:

# distributed training
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [PRETRAINED_CKPT(optional)]
# example: bash run_distribute_train.sh ~/hccl_8p.json /home/DataSet/cocodataset/

# standalone training
Usage: bash run_standalone_train.sh [DATA_PATH] [DEVICE_ID] [PRETRAINED_PATH](optional)
# example: bash run_standalone_train.sh /home/DataSet/cocodataset/ 0

On CPU:

# standalone training
Usage: bash run_standalone_train_cpu.sh [PRETRAINED_MODEL](optional)

On GPU:

# distributed training
Usage: bash run_distribute_train_gpu.sh [DATA_PATH] [PRETRAINED_PATH] (optional)
```

### [Parameters Configuration](#contents)

```default_config.yaml
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

# fpn
"fpn_in_channels": [128, 256, 512, 1024],                               # in channel size for each layer
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

# roi_alignj
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

"mindrecord_dir": "/home/maskrcnn/MindRecord_COCO2017_Train",                  # path of mindrecord
"coco_root": "/home/maskrcnn/",                                                # path of coco root dateset
"train_data_type": "train2017",                                                # name of train dataset
"val_data_type": "val2017",                                                    # name of evaluation dataset
"instance_set": "annotations/instances_{}.json",                               # name of annotation
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
"num_classes": 81
```

## [Training Process](#contents)

- Set options in `default_config.yaml`, including loss_scale, learning rate and network hyperparameters. Click [here](https://www.mindspore.cn/tutorials/en/master/advanced/dataset.html) for more information about dataset.

### [Training](#content)

- Run `run_standalone_train.sh` for non-distributed training of maskrcnn_mobilenetv1 model on Ascend.

    ```bash
    # standalone training
    bash run_standalone_train.sh [DATA_PATH] [DEVICE_ID] [PRETRAINED_PATH](optional)
    # example: bash run_standalone_train.sh /home/DataSet/cocodataset/ 0
    ```

- Run `run_standalone_train_cpu.sh` for non-distributed training of maskrcnn_mobilenetv1 model on CPU.

    ```bash
    # standalone training
    bash run_standalone_train_cpu.sh [PRETRAINED_MODEL](optional)
    ```

### [Distributed Training](#content)

- Run `run_distribute_train.sh` for distributed training of Mask model on Ascend.

    ```bash
    bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [PRETRAINED_MODEL(optional)]
    # example: bash run_distribute_train.sh ~/hccl_8p.json /home/DataSet/cocodataset/
    ```

> hccl.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools).
> As for PRETRAINED_MODEL, if not set, the model will be trained from the very beginning. Ready-made pretrained_models are not available now. Stay tuned.
> This is processor cores binding operation regarding the `device_num` and total processor numbers. If you are not expect to do it, remove the operations `taskset` in `scripts/run_distribute_train.sh`

- Run `run_distribute_train_gpu.sh` for distributed training of Mask model on GPU.

    ```bash
    bash run_distribute_train_gpu.sh [DATA_PATH] [PRETRAINED_PATH]
    ```

### [Training Result](#content)

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the following in loss_rankid.log.

```log
# distribute training result(8p)
2123 epoch: 1 step: 7393 ,rpn_loss: 0.24854, rcnn_loss: 1.04492, rpn_cls_loss: 0.19238, rpn_reg_loss: 0.05603, rcnn_cls_loss: 0.47510, rcnn_reg_loss: 0.16919, rcnn_mask_loss: 0.39990, total_loss: 1.29346
3973 epoch: 2 step: 7393 ,rpn_loss: 0.02769, rcnn_loss: 0.51367, rpn_cls_loss: 0.01746, rpn_reg_loss: 0.01023, rcnn_cls_loss: 0.24255, rcnn_reg_loss: 0.05630, rcnn_mask_loss: 0.21484, total_loss: 0.54137
5820 epoch: 3 step: 7393 ,rpn_loss: 0.06665, rcnn_loss: 1.00391, rpn_cls_loss: 0.04999, rpn_reg_loss: 0.01663, rcnn_cls_loss: 0.44458, rcnn_reg_loss: 0.17700, rcnn_mask_loss: 0.38232, total_loss: 1.07056
7665 epoch: 4 step: 7393 ,rpn_loss: 0.14612, rcnn_loss: 0.56885, rpn_cls_loss: 0.06186, rpn_reg_loss: 0.08429, rcnn_cls_loss: 0.21228, rcnn_reg_loss: 0.08105, rcnn_mask_loss: 0.27539, total_loss: 0.71497
...
16885 epoch: 9 step: 7393 ,rpn_loss: 0.07977, rcnn_loss: 0.85840, rpn_cls_loss: 0.04395, rpn_reg_loss: 0.03583, rcnn_cls_loss: 0.37598, rcnn_reg_loss: 0.11450, rcnn_mask_loss: 0.36816, total_loss: 0.93817
18727 epoch: 10 step: 7393 ,rpn_loss: 0.02379, rcnn_loss: 1.20508, rpn_cls_loss: 0.01431, rpn_reg_loss: 0.00947, rcnn_cls_loss: 0.32178, rcnn_reg_loss: 0.18872, rcnn_mask_loss: 0.69434, total_loss: 1.22887
20570 epoch: 11 step: 7393 ,rpn_loss: 0.03967, rcnn_loss: 1.07422, rpn_cls_loss: 0.01508, rpn_reg_loss: 0.02461, rcnn_cls_loss: 0.28687, rcnn_reg_loss: 0.15027, rcnn_mask_loss: 0.63770, total_loss: 1.11389
22411 epoch: 12 step: 7393 ,rpn_loss: 0.02937, rcnn_loss: 0.85449, rpn_cls_loss: 0.01704, rpn_reg_loss: 0.01234, rcnn_cls_loss: 0.20667, rcnn_reg_loss: 0.12439, rcnn_mask_loss: 0.52344, total_loss: 0.88387
```

## [Evaluation Process](#contents)

### [Evaluation](#content)

- Run `run_eval.sh` for evaluation.

    ```bash
    # infer
    bash run_eval.sh [VALIDATION_ANN_FILE_JSON] [CHECKPOINT_PATH] [DATA_PATH] [DEVICE_TARGET] [DEVICE_ID]
    # example: bash run_eval.sh /home/DataSet/cocodataset/annotations/instances_val2017.json /home/model/maskrcnn_mobilenetv1/ckpt/mask_rcnn-5_7393.ckpt /home/DataSet/cocodataset/ "Ascend" 0
    ```

> As for the COCO2017 dataset, VALIDATION_ANN_FILE_JSON is refer to the annotations/instances_val2017.json in the dataset directory.  
> Checkpoint can be produced and saved in training process, whose folder name begins with "train/checkpoint" or "train_parallel*/checkpoint".

- Run `run_eval.sh` for evaluation on gpu.

    ```bash
    # infer
    bash run_eval.sh [VALIDATION_ANN_FILE_JSON] [CHECKPOINT_PATH] [DATA_PATH] [DEVICE_TARGET] [DEVICE_ID]
    # example: bash run_eval.sh /home/DataSet/cocodataset/annotations/instances_val2017.json /home/model/maskrcnn_mobilenetv1/ckpt/mask_rcnn-12_7393.ckpt /home/DataSet/cocodataset/ "GPU" 0
    ```

### [Evaluation result](#content)

Inference result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

```log
Evaluate annotation type *bbox*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.227
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.145
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.240
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.283
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.239
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.390
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.440
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501

Evaluate annotation type *segm*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.166
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.185
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.388
```

## [Export Process](#contents)

### [Export](#content)

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format [EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

## [Inference Process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Inference](#content)

Before performing inference, we need to export model first. Air model can only be exported in Ascend 910 environment, mindir model can be exported in any environment.
Current batch_ Size can only be set to 1. The inference process needs about 600G hard disk space to save the reasoning results.

```shell
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_TYPE] [DEVICE_ID]
```

### [Inference result](#content)

Inference result is saved in current path, you can find result like this in acc.log file.

```log
Evaluate annotation type *bbox*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.227
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.145
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.240
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.283
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.239
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.390
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.440
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501

Evaluate annotation type *segm*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.166
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.185
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.388
```

# Model Description

## Performance

### Evaluation Performance

| Parameters                 | Ascend                                                      | GPU                                                         |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 | GPU(Tesla V100-PCIE); CPU 2.60 GHz, 26 cores; Memory 790G; OS Euler2.0             |
| uploaded Date              | 12/01/2020 (month/day/year)                                 | 09/21/2021 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                       | 1.5.0                                                       |
| Dataset                    | COCO2017                                                    | COCO2017                                                    |
| Training Parameters        | epoch=12,  batch_size = 2                                   | epoch=12,  batch_size = 2                                   |
| Optimizer                  | Momentum                                                    | Momentum                                                         |
| Loss Function              | Softmax Cross Entropy, Sigmoid Cross Entropy, SmoothL1Loss  | Softmax Cross Entropy, Sigmoid Cross Entropy, SmoothL1Loss  |
| Output                     | Probability                                                 | Probability                                                 |
| Loss                       | 0.88387                                                     | 0.60763                                                     |
| Speed                      | 8pcs: 249 ms/step                                           | 8pcs: 795.645 ms/step                                       |
| Total time                 | 8pcs: 6.23 hours                                            | 8pcs: 19.6 hours                                            |
| Scripts                    | [maskrcnn script](https://gitee.com/mindspore/models/tree/r2.0/official/cv/MaskRCNN/maskrcnn_mobilenetv1) |  [maskrcnn script](https://gitee.com/mindspore/models/tree/r2.0/official/cv/MaskRCNN/maskrcnn_mobilenetv1) |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | V1                          |
| Resource            | Ascend 910; OS Euler2.8                   |
| Uploaded Date       | 12/01/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       |
| Dataset             | COCO2017                    |
| batch_size          | 2                           |
| outputs             | mAP                         |
| Accuracy            | IoU=0.50:0.95 (BoundingBox 22.7%, Mask 17.6%) |
| Model for inference | 107M (.ckpt file)           |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py for weight initialization.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
