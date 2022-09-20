# Contents

- [Contents](#contents)
- [Tracktor++ Description](#tracktor-description)
- [FasterRcnn Description](#fasterrcnn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
    - [Run on GPU](#run-on-gpu)
    - [Run on Ascend](#run-on-ascend)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training Process](#training-process)
        - [Usage](#usage)
            - [on GPU](#on-gpu)
            - [on Ascend](#on-ascend)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [on GPU and Ascend](#on-gpu-and-ascend)
        - [Result](#result-1)
    - [Model Export](#model-export)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# Tracktor++ Description

The problem of tracking multiple objects in a video se-quence poses several challenging tasks. For tracking-by-detection, these include object re-identification, motion pre-diction and dealing with occlusions.We present a tracker (without bells and whistles) that accomplishes tracking without specifically targeting any of tracking tasks,
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

Dataset used for training: [MOT17Det](<https://motchallenge.net/data/MOT17Det/>)
Dataset used for eval: [MOT17](<https://motchallenge.net/data/MOT17/>)

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
2. pretrained model is a faster rcnn resnet50 checkpoint that trained over COCO. you can train it with [faster_rcnn](https://gitee.com/mindspore/models/tree/master/official/cv/faster_rcnn) scripts in modelzoo. Or you can download it from [hub](https://download.mindspore.cn/model_zoo/r1.3/fasterrcnnresnetv1550_ascend_v130_coco2017_official_cv_bs2_acc61.7/).  And place it in the tracktor++ directory.
3. re-identification network model comes from the [official implementation model](https://vision.in.tum.de/webshare/u/meinhard/tracking_wo_bnw-output_v5.zip) of pytorch and converts it to mindspore's ckpt file. You can specify the pth file path in `pth2ckpt.py`, and run the program to get the ckpt file as:

```bash
python pth2ckpt.py
```

## Run on GPU

Unzip all datasets, and prepare annotation for training using

```bash
python prepare_detection_anno.py --dataset_path=PATH/TO/MOT17DET
```

Before running train scripts, you must specify all paths in `default_config.yaml`.

`pre_trained`, `image_dir`, `anno_path`, `mot_dataset_path`.

For evaluation specify path to trained checkpoints.

`checkpoint_path` (faster-rcnn ckpt file for tracktor++ eval), `ckpt_file` (ckpt file for model export), `reid_weight` (reid ckpt file for tracktor++ eval)

Note: `mot_dataset_path` is the path to MOT17 dataset, `image_dir` is the path to MOT17DET dataset for training.

```bash
# standalone training on gpu
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CONFIG_PATH]

# distributed training on gpu
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [CONFIG_PATH]

# eval on gpu
bash scripts/run_eval_gpu.sh [DEVICE_ID] [CONFIG_PATH]
```

## Run on Ascend

Unzip all datasets, and prepare annotation for training using

```bash
python prepare_detection_anno.py --dataset_path=PATH/TO/MOT17DET
```

```bash
# standalone training on ascend
bash scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]

# distributed training on ascend
bash scripts/run_distribute_train_ascend.sh [DEVICE_NUM] [CONFIG_PATH] [image_dir] [mindrecord_dir] [RANK_TABLE_FILE]

# eval on ascend
bash scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH]
```

# Script Description

## Script and Sample Code

```text
.
└─tracktor++
  ├─README.md                         // descriptions about fasterrcnn
  ├─scripts
    ├─run_standalone_train_gpu.sh     // shell script for standalone on GPU
    ├─run_distribute_train_gpu.sh     // shell script for distributed on GPU
    ├─run_standalone_train_ascend.sh     // shell script for standalone on Ascend
    ├─run_distribute_train_ascend.sh     // shell script for distributed Ascend
    ├─run_eval_gpu.sh                 // shell script for eval on GPU
    └─run_eval_ascend.sh              // shell script for eval Ascend
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
    ├─reid.py                         // reid class
    ├─tracker.py                      // tracker class
    ├─tracker_plus_plus.py                      // tracker++ class
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
    ├─prepare_detection_anno.py          // prepare annotations from dataset
    └─train.py                          // train scripts
```

## Training Process

### Usage

#### on GPU

```bash
# standalone training
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CONFIG_PATH]

# distributed training
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [CONFIG_PATH]
```

#### on Ascend

```bash
# standalone training
bash scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]

# distributed training
bash scripts/run_distribute_train_ascend.sh [DEVICE_NUM] [CONFIG_PATH] [image_dir] [mindrecord_dir] [RANK_TABLE_FILE]
```

Before train you must unzip all datasets, and prepare annotation for training using

```bash
python prepare_detection_anno.py --dataset_path=PATH/TO/MOT17DET
```

Then you must specify all paths in `default_config.yaml`.

`pre_trained`, `image_dir`, `anno_path`, `mot_dataset_path`.

For evaluation specify path to trained checkpoints.

`checkpoint_path` (faster-rcnn ckpt file for tracktor++ eval), `ckpt_file` (ckpt file for model export), `reid_weight` (reid ckpt file for tracktor++ eval)

### Result

```text
# distribute training result(8p)
# loss logs
4223 epoch: 30 step: 285 total_loss: 0.01451
4223 epoch: 30 step: 286 total_loss: 0.02001
4223 epoch: 30 step: 287 total_loss: 0.04003
4224 epoch: 30 step: 288 total_loss: 0.00887
4224 epoch: 30 step: 289 total_loss: 0.06539
4225 epoch: 30 step: 290 total_loss: 0.02722
4225 epoch: 30 step: 291 total_loss: 0.01804
4225 epoch: 30 step: 292 total_loss: 0.03315
4226 epoch: 30 step: 293 total_loss: 0.01891
4226 epoch: 30 step: 294 total_loss: 0.04566
4227 epoch: 30 step: 295 total_loss: 0.02666
4227 epoch: 30 step: 296 total_loss: 0.02234
4227 epoch: 30 step: 297 total_loss: 0.03690
4228 epoch: 30 step: 298 total_loss: 0.04819
4228 epoch: 30 step: 299 total_loss: 0.03124
# performance log(8p)
Train epoch time: 61636.463 ms, per step time: 206.142 ms
Train epoch time: 60531.407 ms, per step time: 202.446 ms
Train epoch time: 58972.855 ms, per step time: 197.234 ms
Train epoch time: 59358.096 ms, per step time: 198.522 ms
Train epoch time: 58845.790 ms, per step time: 196.809 ms
Train epoch time: 58206.283 ms, per step time: 194.670 ms
Train epoch time: 57716.250 ms, per step time: 193.031 ms
Train epoch time: 57889.853 ms, per step time: 193.612 ms
Train epoch time: 58644.023 ms, per step time: 196.134 ms
Train epoch time: 56841.141 ms, per step time: 190.104 ms
Train epoch time: 57365.054 ms, per step time: 191.856 ms
Train epoch time: 57290.224 ms, per step time: 191.606 ms
Train epoch time: 58077.257 ms, per step time: 194.238 ms
Train epoch time: 57222.631 ms, per step time: 191.380 ms
Train epoch time: 56907.141 ms, per step time: 190.325 ms
Train epoch time: 57316.618 ms, per step time: 191.694 ms
Train epoch time: 56885.291 ms, per step time: 190.252 ms
Train epoch time: 56845.306 ms, per step time: 190.118 ms
```

## Evaluation Process

### Usage

#### on GPU and Ascend

```bash
# eval on gpu
bash scripts/run_eval_gpu.sh [DEVICE_ID] [CONFIG_PATH]

# eval on ascend
bash scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH]
```

Note: set the device(GPU, Ascend) in default_config.yaml, `mot_dataset_path` is the path to MOT17 dataset, `image_dir` is the path to MOT17DET dataset for training.

### Result

Eval result will be printed in std out.

```text
                IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-FRCNN 43.4% 75.0% 30.5% 40.6% 99.7%  62   9  31  22   25 11041  60   63 40.1% 0.100   7  57   5
MOT17-04-FRCNN 71.5% 89.3% 59.6% 64.7% 97.0%  83  36  28  19  959 16794  22   37 62.6% 0.107   0  22   0
MOT17-05-FRCNN 59.6% 80.4% 47.3% 56.6% 96.1% 133  33  63  37  158  3004  65   68 53.3% 0.138  16  57  10
MOT17-09-FRCNN 59.4% 76.1% 48.8% 63.1% 98.5%  26  11  13   2   50  1964  28   29 61.7% 0.063   6  26   4
MOT17-10-FRCNN 60.4% 71.0% 52.6% 73.3% 99.1%  57  33  21   3   87  3424  85   81 72.0% 0.124  14  74   4
MOT17-11-FRCNN 64.6% 78.3% 55.0% 68.7% 97.7%  75  25  32  18  154  2956  23   25 66.8% 0.064   3  21   1
MOT17-13-FRCNN 73.6% 86.0% 64.4% 74.1% 99.0% 110  59  41  10   89  3018  65   73 72.8% 0.146  20  53   9
OVERALL        64.5% 82.8% 52.8% 62.4% 97.9% 546 206 229 111 1522 42201 348  376 60.8% 0.109  66 310  33
```

## Model Export

```bash
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be in ["MINDIR", "AIR"]

# Model Description

## Performance

### Training Performance

| Parameters          | Ascend                                                                                      | GPU                                                        |
|---------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------|
| Resource            | Ascend 910                                                                                  | V100                                                       |
| uploaded Date       | 08/12/2022 (month/day/year)                                                                 | 08/12/2022 (month/day/year)                                |
| MindSpore Version   | 1.8.0                                                                                       | 1.8.0                                                      |
| Dataset             | MOT17DET                                                                                    | MOT17DET                                                   |
| Training Parameters | epoch=30, batch_size=2                                                                      | epoch=30, batch_size=2                                     |
| Optimizer           | SGD                                                                                         | SGD                                                        |
| Loss Function       | Softmax Cross Entropy, Sigmoid Cross Entropy, SmoothL1Loss                                  | Softmax Cross Entropy, Sigmoid Cross Entropy, SmoothL1Loss |
| Speed               | 1pcs: 161.673 ms/step 8pcs: 226.333 ms/step                                                 | 1pcs: 357.115 ms/step 8pcs: 471.408 ms/step                |
| Total time          | 1pcs: 3.2 hours 8pcs: 0.5 hour                                                              | 1pcs: 7.1 hours  8pcs: 1.2 hours                           |
| Parameters (M)      | 473                                                                                         | 473                                                        |
| Scripts             | [fasterrcnn script](https://gitee.com/mindspore/models/tree/master/official/cv/faster_rcnn) | [fasterrcnn script](https://gitee.com/mindspore/models/tree/master/official/cv/faster_rcnn) |

### Evaluation Performance

| Parameters          | Ascend                      | GPU                         |
|---------------------|-----------------------------|-----------------------------|
| Resource            | Ascend 910                  | V100                        |
| Uploaded Date       | 08/12/2022 (month/day/year) | 08/12/2022 (month/day/year) |
| MindSpore Version   | 1.8.0                       | 1.8.0                       |
| Dataset             | MOT17                       | MOT17                       |
| batch_size          | 1                           | 1                           |
| outputs             | MOTA                        | MOTA                        |
| Accuracy            | 60.8%                       | 60.8%                       |
| Model for inference | 473M (.ckpt file)           | 473M (.ckpt file)           |

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/models/tree/master).
