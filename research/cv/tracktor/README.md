# Contents

- [Tracktor Description](#fasterrcnn-description)
- [FasterRcnn Description](#fasterrcnn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training Process](#training-process)
        - [Training Usage](#usage)
        - [Training Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation Usage](#usage)
        - [Evaluation Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# Tracktor Description

We present a tracker (without bells and whistles) that accomplishes tracking without specifically targeting any of tracking tasks,
in particular, we perform no training or optimization on tracking data.
We exploit the bounding box regression of the Faster RCNN object detector to predict the position of an object in the next frame,
thereby converting a detector into a Tracktor.
[paper](https://arxiv.org/abs/1903.05625)

# FasterRcnn Description

Before FasterRcnn, the target detection networks rely on the region proposal algorithm to assume the location of targets, such as SPPnet and Fast R-CNN. Progress has reduced the running time of these detection networks, but it also reveals that the calculation of the region proposal is a bottleneck.

FasterRcnn proposed that convolution feature maps based on region detectors (such as Fast R-CNN) can also be used to generate region proposals. At the top of these convolution features, a Region Proposal Network (RPN) is constructed by adding some additional convolution layers (which share the convolution characteristics of the entire image with the detection network, thus making it possible to make regions almost costlessProposal), outputting both region bounds and objectness score for each location.Therefore, RPN is a full convolutional network (FCN), which can be trained end-to-end, generate high-quality region proposals, and then fed into Fast R-CNN for detection.

[Paper](https://arxiv.org/abs/1506.01497):   Ren S , He K , Girshick R , et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 39(6).

# Model Architecture

FasterRcnn is a two-stage target detection network,This network uses a region proposal network (RPN), which can share the convolution features of the whole image with the detection network, so that the calculation of region proposal is almost cost free. The whole network further combines RPN and FastRcnn into a network by sharing the convolution features.

# Dataset

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used for training: [MOT17](<https://motchallenge.net/data/MOT17Det/>)

- Dataset size：2G
    - Train：0.9G，5316 images
    - Annotations：1.7M，detection, ids
- Data format：image and txt files
    - Note：Data will be processed in dataset.py

# Environment Requirements

- Hardware（GPU）

    - Prepare hardware environment with GPU.

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset MOT17DET and MOT17 for validation.

- We use MOT17DET as training dataset in this example by default, and you can also use your own datasets.

Organize the dataset information into a TXT file, each row in the file is as follows:

```text
MOT17-05/img1/000056.jpg -9,-14,232,559,1 297,10,559,536,1
```

Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class information of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are setting in `default_config.yaml`.

# Quick Start

After installing MindSpore via the official website, you can start training and evaluation as follows:

Note:

1. the first run will generate the mindrecord file, which will take a long time.
2. pretrained model is a faster rcnn resnet50 checkpoint that trained over COCO. you can train it with [faster_rcnn](https://gitee.com/mindspore/models/tree/master/official/cv/faster_rcnn) scripts in modelzoo. Or you can download it from [hub](https://download.mindspore.cn/model_zoo/r1.3/fasterrcnnresnetv1550_ascend_v130_coco2017_official_cv_bs2_acc61.7/)

## Run on GPU

Unzip all datasets, and prepare annotation for training using

```bash
python prepare_detection_anno.py --dataset_path=PATH/TO/MOT17DET
```

Before running train scripts, you must specify all paths in `default_config.yaml`.

`pre_trained`, `image_dir`, `anno_path`, `mot_dataset_path`.

For evaluation specify path to trained checkpoints.

`checkpoint_path`, `ckpt_file`.

Note: `mot_dataset_path` is the path to MOT17 dataset, `image_dir` is the path to MOT17DET dataset for training.

```bash

# standalone training
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CONFIG_PATH]

# distributed training
bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM] [CONFIG_PATH] [LR]

# eval
bash scripts/run_eval_gpu.sh [DEVICE_ID] [MOT_DATASET_PATH] [CKPT_PATH]
```

# Script Description

## Script and Sample Code

```text
.
└─faster_rcnn
  ├─README.md                         // descriptions about fasterrcnn
  ├─scripts
    ├─run_standalone_train_gpu.sh     // shell script for standalone on GPU
    ├─run_distribute_train_gpu.sh     // shell script for distributed on GPU
    └─run_eval_gpu.sh                 // shell script for eval on GPU
  ├─src
    ├─FasterRcnn
      ├─__init__.py                   // init file
      ├─anchor_generator.py           // anchor generator
      ├─bbox_assign_sample.py         // first stage sampler
      ├─bbox_assign_sample_stage2.py  // second stage sampler
      ├─faster_rcnn.py                // fasterrcnn network
      ├─fpn_neck.py                   //feature pyramid network
      ├─proposal_generator.py         // proposal generator
      ├─rcnn.py                       // rcnn network
      ├─resnet.py                     // backbone network
      ├─resnet50v1.py                 // backbone network for ResNet50v1.0
      ├─roi_align.py                  // roi align network
      └─rpn.py                        // region proposal network
    ├─dataset.py                      // create dataset and process dataset
    ├─lr_schedule.py                  // learning ratio generator
    ├─network_define.py               // network define for fasterrcnn
    ├─tracker.py                      // tracker class for tracktor
    ├─tracking_utils.py               // tracker utils
    ├─util.py                         // routine operation
    └─model_utils
      ├─config.py                     // Processing configuration parameters
      ├─device_adapter.py             // Get cloud ID
      ├─local_adapter.py              // Get local ID
      └─moxing_adapter.py             // Parameter processing
  ├─default_config.yaml               // config for tracktor
  ├─export.py                         // script to export AIR,MINDIR,ONNX model
  ├─eval.py                           // eval scripts
  ├─eval_detector.py                  // helper scripts for evaluation detection metrics.
  └─train.py                          // train scripts
```

## Training Process

### Usage

#### on GPU

```bash
# standalone training
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CONFIG_PATH]

# distributed training
bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM] [CONFIG_PATH] [LR]
```

Before train you must unzip all datasets, and prepare annotation for training using

```bash
python prepare_detection_anno.py --dataset_path=PATH/TO/MOT17DET
```

Then you must specify all paths in `default_config.yaml`.

`pre_trained`, `image_dir`, `anno_path`, `mot_dataset_path`.

For evaluation specify path to trained checkpoints.

`checkpoint_path`, `ckpt_file`.

### Result

```text
# distribute training result(8p)
# loss logs
3406 epoch: 30 step: 285 total_loss: 0.09377
3406 epoch: 30 step: 286 total_loss: 0.09688
3406 epoch: 30 step: 287 total_loss: 0.06130
3407 epoch: 30 step: 288 total_loss: 0.03282
3407 epoch: 30 step: 289 total_loss: 0.19695
3407 epoch: 30 step: 290 total_loss: 0.14444
3408 epoch: 30 step: 291 total_loss: 0.06046
3408 epoch: 30 step: 292 total_loss: 0.05724
3408 epoch: 30 step: 293 total_loss: 0.09789
3409 epoch: 30 step: 294 total_loss: 0.07339
3409 epoch: 30 step: 295 total_loss: 0.05054
3409 epoch: 30 step: 296 total_loss: 0.09697
3410 epoch: 30 step: 297 total_loss: 0.03201
3410 epoch: 30 step: 298 total_loss: 0.04093
3410 epoch: 30 step: 299 total_loss: 0.04519
# performance log
epoch time: 111428.575 ms, per step time: 372.671 ms
epoch time: 111417.881 ms, per step time: 372.635 ms
epoch time: 111318.437 ms, per step time: 372.302 ms
epoch time: 111659.650 ms, per step time: 373.444 ms
epoch time: 111388.490 ms, per step time: 372.537 ms
epoch time: 111826.518 ms, per step time: 374.002 ms
epoch time: 111565.894 ms, per step time: 373.130 ms
epoch time: 111735.482 ms, per step time: 373.697 ms
epoch time: 106787.494 ms, per step time: 357.149 ms
epoch time: 106757.486 ms, per step time: 357.048 ms
epoch time: 106889.161 ms, per step time: 357.489 ms
epoch time: 106865.015 ms, per step time: 357.408 ms
epoch time: 107085.235 ms, per step time: 358.145 ms
epoch time: 106946.509 ms, per step time: 357.681 ms
epoch time: 106842.626 ms, per step time: 357.333 ms
epoch time: 106928.893 ms, per step time: 357.622 ms
```

## Evaluation Process

### Usage

#### on GPU

```bash
# eval on GPU
bash scripts/run_eval_gpu.sh [DEVICE_ID] [MOT_DATASET_PATH] [CKPT_PATH]
```

Note: `mot_dataset_path` is the path to MOT17 dataset, `image_dir` is the path to MOT17DET dataset for training.

### Result

Eval result will be printed in std out.

```text
                IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-FRCNN 42.0% 71.8% 29.7% 40.9% 99.1%  62   9  30  23   71 10973  66   67 40.2% 0.126  10  61   5
MOT17-04-FRCNN 69.9% 87.6% 58.2% 64.3% 96.8%  83  32  32  19 1013 16980  27   53 62.1% 0.129   8  21   2
MOT17-05-FRCNN 58.8% 78.3% 47.0% 57.1% 95.1% 133  32  67  34  205  2966  66   71 53.2% 0.167  13  62   9
MOT17-09-FRCNN 49.4% 63.9% 40.2% 61.6% 98.1%  26   9  14   3   65  2043  39   42 59.7% 0.165   9  33   3
MOT17-10-FRCNN 60.0% 70.1% 52.5% 72.0% 96.2%  57  29  25   3  364  3596  90  148 68.5% 0.178  20  72   4
MOT17-11-FRCNN 63.3% 76.2% 54.1% 68.8% 97.0%  75  26  31  18  201  2945  29   27 66.4% 0.093   5  26   3
MOT17-13-FRCNN 69.7% 80.6% 61.4% 73.9% 97.0% 110  61  38  11  267  3034 123  117 70.6% 0.175  35  99  11
OVERALL        62.5% 80.0% 51.3% 62.1% 97.0% 546 198 237 111 2186 42537 440  525 59.8% 0.141 100 374  37

```

## Model Export

```bash
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be in ["MINDIR"]

# Model Description

## Performance

### Evaluation Performance

| Parameters          | GPU                                                                                                      |
|---------------------|----------------------------------------------------------------------------------------------------------|
| Resource            | 8xRTX-3090                                                                                               |
| uploaded Date       | 02/08/2022 (month/day/year)                                                                              |
| MindSpore Version   | 1.5.0                                                                                                    |
| Dataset             | MOT17                                                                                                    |
| Training Parameters | epoch=30,  batch_size=2                                                                                  |
| Optimizer           | SGD                                                                                                      |
| Loss Function       | Softmax Cross Entropy, Sigmoid Cross Entropy,SmoothL1Loss                                                |
| Speed               | 1pcs 256.678 ms/step 8pcs: 357.489 ms/step                                                               |
| Total time          | 1pcs 5 hours 8pcs: 1 hour                                                                                |
| Parameters (M)      | 250                                                                                                      |
| Scripts             | [fasterrcnn script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/faster_rcnn) |

### Inference Performance

| Parameters          | GPU                         |
|---------------------|-----------------------------|
| Resource            | GPU                         |
| Uploaded Date       | 02/08/2022 (month/day/year) |
| MindSpore Version   | 1.5.0                       |
| Dataset             | MOT17                       |
| batch_size          | 2                           |
| outputs             | MOTA                        |
| Accuracy            | 59.8%                       |
| Model for inference | 250M (.ckpt file)           |

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
