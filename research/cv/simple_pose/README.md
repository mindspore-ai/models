# Contents

- [Description](#description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
    - [Dataset Preparation](#dataset-preparation)
    - [Model Checkpoints](#model-checkpoints)
    - [Running](#running)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Description](#contents)

SimplePoseNet is a convolution-based neural network for the task of human pose estimation and tracking. It provides baseline methods that are surprisingly simple and effective, thus helpful for inspiring and evaluating new ideas for the field. State-of-the-art results are achieved on challenging benchmarks. More detail about this model can be found in:

 B. Xiao, H. Wu, and Y. Wei, “Simple baselines for human pose estimation and tracking,” in Proc. Eur. Conf. Comput. Vis., 2018, pp. 472–487.

This repository contains a Mindspore implementation of SimplePoseNet based upon Microsoft's original Pytorch implementation (<https://github.com/microsoft/human-pose-estimation.pytorch>). The training and validating scripts are also included, and the evaluation results are shown in the [Performance](#performance) section.

# [Model Architecture](#contents)

The overall network architecture of SimplePoseNet is shown below:

[Link](https://arxiv.org/pdf/1804.06208.pdf)

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: COCO2017

- Dataset size:
    - Train: 19G, 118,287 images
    - Test: 788MB, 5,000 images
- Data format: JPG images
    - Note: Data will be processed in `src/dataset.py`
- Person detection result for validation: Detection result provided by author in the [repository](https://github.com/microsoft/human-pose-estimation.pytorch)

# [Features](#contents)

## [Mixed Precision](#contents)

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware. For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

To run the python scripts in the repository, you need to prepare the environment as follow:

- Hardware
    - Prepare hardware environment with Ascend.
- Python and dependencies
    - python 3.7
    - mindspore 1.2.0 (Ascend), mindspore-gpu 1.6.1 (GPU)
    - opencv-python 4.3.0.36
    - pycocotools 2.0
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

## [Dataset Preparation](#contents)

SimplePoseNet use COCO2017 dataset to train and validate in this repository. Download the dataset from [official website](https://cocodataset.org/). You can place the dataset anywhere and tell the scripts where it is by modifying the `DATASET.ROOT` setting in configuration file `src/config.py`. For more information about the configuration file, please refer to [Script Parameters](#script-parameters).

You also need the person detection result of COCO val2017 to reproduce the multi-person pose estimation results, as mentioned in [Dataset](#dataset). Please checkout the author's repository, download and extract them under `<ROOT>/experiments/`, and make them look like this:

```text
└─ <ROOT>
 └─ experiments
   └─ COCO_val2017_detections_AP_H_56_person.json
```

## [Model Checkpoints](#contents)

Before you start your training process, you need to obtain mindspore imagenet pretrained models. The model weight file can be obtained by running the Resnet training script in [official model zoo](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet). We also provide a pretrained model that can be used to train SimplePoseNet directly in [GoogleDrive](https://drive.google.com/file/d/1r3Hs0QNys0HyNtsQhSvx6IKdyRkC-3Hh/view?usp=sharing). The model file should be placed under `<ROOT>/models/` like this:

```text
└─ <ROOT>
 └─ models
  └─resnet50.ckpt
```

```dataset
 └─ coco2017
  └─train2017
  └─val2017
  └─annotations
```

```text
DATASET:
    ROOT:../coco2017
MODEL:
    PRETRAINED:./models/resnet50.ckpt

# Modify according to local path
```

In the example above, we put COCO2017 dataset in `../<ROOT>`.

## [Running](#contents)

- running on local Ascend

    To train the model on Ascend, run the shell script `scripts/run_standalone_train.sh` with the format below:

    ```shell
    bash scripts/run_standalone_train.sh [CKPT_SAVE_DIR] [DEVICE_ID] [BATCH_SIZE]
    # example: bash scripts/run_standalone_train.sh ./ckpt/ascend_standalone 0 128
    ```

    **Notice:** `[CKPT_SAVE_DIR]` is the relative path to `<ROOT>`.

    To validate the model on Ascend, change the settings in `default_config.yaml` to the path of the model you want to validate or setting that on the terminal. For example:

    ```text
    TEST:
        ...
        MODEL_FILE : ./ckpt/ascend_standalone/simplepose-140_1170.ckpt
    # Modify according to local path
    ```

    Then, run the shell script `scripts/run_eval.sh` with the format below:

    ```shell
    bash scripts/run_eval.sh [TEST_MODEL_FILE] [COCO_BBOX_FILE] [DEVICE_ID]
    # example: bash scripts/run_eval.sh ./ckpt/ascend_standalone/simplepose-140_1170.ckpt ./experiments/COCO_val2017_detections_AP_H_56_person.json 0
    ```

- running on local GPU

    To train the model on GPU, run the shell script `scripts/run_standalone_train_gpu.sh` with the format below:

    ```shell
    bash scripts/run_standalone_train_gpu.sh [CKPT_SAVE_DIR(relative)] [BATCH_SIZE] [END_EPOCH] [DEVICE_ID]
    # example: bash scripts/run_standalone_train_gpu.sh ./ckpt/gpu_standalone/ 64 185 0
    ```

    **Notice:** `[CKPT_SAVE_DIR]` is the relative path to `<ROOT>`.

    To validate the model on GPU, change the settings in `default_config.yaml` to the path of the model you want to validate or setting that on the terminal. For example:

    ```text
    TEST:
        ...
        MODEL_FILE : ./ckpt/gpu_standalone/simplepose-185_1170.ckpt
    # Modify according to local path
    ```

    Then, run the shell script `scripts/run_eval_gpu.sh` with the format below:

    ```shell
    bash scripts/run_eval_gpu.sh [TEST_MODEL_FILE] [COCO_BBOX_FILE] [DEVICE_ID]
    # example: bash scripts/run_eval_gpu.sh ./ckpt/gpu_standalone/simplepose-185_1170.ckpt ./experiments/COCO_val2017_detections_AP_H_56_person.json 0
    ```

- running on ModelArts

    If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows

    - Training with 8 cards on ModelArts

    ```python
    # (1) Upload the code folder to S3 bucket.
    # (2) Click to "create training task" on the website UI interface.
    # (3) Set the code directory to "/{path}/simple_pose" on the website UI interface.
    # (4) Set the startup file to /{path}/simple_pose/train.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/simple_pose/default_config.yaml.
    #         1. Set ”run_distributed: True“
    #         2. Set ”enable_modelarts: True“
    #         3. Set “batch_size: 64”(It's not necessary)
    #     b. adding on the website UI interface.
    #         1. Add ”run_distributed=True“
    #         2. Add ”enable_modelarts=True“
    #         3. Add “batch_size=64”(It's not necessary)
    # (6) Upload the dataset to S3 bucket.
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of 8 cards.
    # (10) Create your job.
    ```

    - evaluating with single card on ModelArts

    ```python
    # (1) Upload the code folder to S3 bucket.
    # (2) Click to "create training task" on the website UI interface.
    # (3) Set the code directory to "/{path}/simple_pose" on the website UI interface.
    # (4) Set the startup file to /{path}/simple_pose/eval.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/simple_pose/default_config.yaml.
    #         1. Set ”enable_modelarts: True“
    #         2. Set “eval_model_file: ./{path}/*.ckpt”('eval_model_file' indicates the path of the weight file to be evaluated relative to the file `eval.py`, and the weight file must be included in the code directory.)
    #         3. Set ”coco_bbox_file: ./{path}/COCO_val2017_detections_AP_H_56_person.json"(The same as 'eval_model_file')
    #     b. adding on the website UI interface.
    #         1. Add ”enable_modelarts=True“
    #         2. Add “eval_model_file=./{path}/*.ckpt”('eval_model_file' indicates the path of the weight file to be evaluated relative to the file `eval.py`, and the weight file must be included in the code directory.)
    #         3. Add ”coco_bbox_file=./{path}/COCO_val2017_detections_AP_H_56_person.json"(The same as 'eval_model_file')
    # (6) Upload the dataset to S3 bucket.
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (there is only data or zip package under this path).
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of a single card.
    # (10) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The structure of the files in this repository is shown below.

```text
└─ simple_pose
    ├─ scripts
    │  ├─ run_eval.sh                 // launch ascend evaluation
    │  ├─ run_eval_gpu.sh                 // launch gpu standalone evaluation
    │  ├─ run_distribute_train.sh    // launch ascend distributed training
    │  ├─ run_standalone_train.sh     // launch ascend standalone training
    │  ├─ run_distribute_train_gpu.sh    // launch gpu distributed training
    │  ├─ run_standalone_train_gpu.sh     // launch gpu standalone training
    │  └─ run_infer_310.sh
    ├─ src
    │  ├─ utils
    │  │  ├─ transform.py         // utils about image transformation
    │  │  └─ nms.py               // utils about nms
    │  ├─ evaluate
    │  │  └─ coco_eval.py         // evaluate result by coco
    │  ├─ model_utils
    │  │  ├── config.py           // parsing parameter configuration file of "*.yaml"
    │  │  ├── devcie_adapter.py   // local or ModelArts training
    │  │  ├── local_adapter.py    // get related environment variables in local training
    │  │  └── moxing_adapter.py   // get related environment variables in ModelArts training
    │  ├─ dataset.py              // dataset processor and provider
    │  ├─ model.py                // SimplePoseNet implementation
    │  ├─ network_define.py       // define loss
    │  └─ predict.py              // predict keypoints from heatmaps
    ├─ default_config.yaml        // parameter configuration
    ├─ eval.py                    // evaluation script
    ├─ train.py                   // training script
    └─ README.md                  // descriptions about this repository
```

## [Script Parameters](#contents)

Configurations for both training and evaluation are set in `default_config.yaml`. All the settings are shown following.

- config for SimplePoseNet on COCO2017 dataset:

```text
# These parameters can be modified at the terminal
ckpt_save_dir: 'checkpoints'                # the folder to save the '*.ckpt' file
batch_size: 128                             # TRAIN.BATCH_SIZE
run_distribute: False                       # training by several devices: "true"(training by more than 1 device) | "false", default is "false"
dataset_path: ''                            # DATASET.ROOT
end_epoch: 0                                # TRAIN.EPOCH
eval_model_file: ''                         # TEST.MODEL_FILE
coco_bbox_file: ''                          # TEST.COCO_BBOX_FILE
#pose_resnet-related
POSE_RESNET:
    NUM_LAYERS: 50                          # number of layers(for resnet)
    DECONV_WITH_BIAS: False                 # deconvolution bias
    NUM_DECONV_LAYERS: 3                    # the number of deconvolution layers
    NUM_DECONV_FILTERS: [256, 256, 256]     # the filter size of deconvolution layers
    NUM_DECONV_KERNELS: [4, 4, 4]           # kernel size of deconvolution layers
    FINAL_CONV_KERNEL: 1                    # final convolution kernel size
    TARGET_TYPE: 'gaussian'
    HEATMAP_SIZE: [48, 64]                  # heatmap size
    SIGMA: 2                                # Gaussian hyperparameter in heatmap generation
#network-related
MODEL:
    NAME: 'pose_resnet'                     # model name
    INIT_WEIGHTS: True                      # init model weights by resnet
    PRETRAINED: './resnet50.ckpt'           # pretrained model
    RESUME: False                           # whether resume training from pretrained checkpoint
    NUM_JOINTS: 17                          # the number of keypoints
    IMAGE_SIZE: [192, 256]                  # image size
#dataset-related
DATASET:
    ROOT: '../coco2017/'                 # coco2017 dataset root
    TEST_SET: 'val2017'                     # folder name of test set
    TRAIN_SET: 'train2017'                  # folder name of train set
    FLIP: True                              # random flip
    ROT_FACTOR: 40                          # random rotation
    SCALE_FACTOR: 0.3                       # random scale
#train-related
TRAIN:
    BATCH_SIZE: 64                          # batch size
    BEGIN_EPOCH: 0                          # begin epoch
    END_EPOCH: 140                          # end epoch
    PRINT_STEP: 100                         # print step
    SAVE_EPOCH: 5                           # number of epochs per saving
    KEEP_CKPT_MAX: 5                       # maximum number of checkpoint files can be saved
    LR: 0.001                               # initial learning rate
    LR_FACTOR: 0.1                          # learning rate reduce factor
    LR_STEP: [90, 120]                      # step to reduce lr
#eval-related
TEST:
    BATCH_SIZE: 32                          # batch size
    FLIP_TEST: True                         # flip test
    POST_PROCESS: True                      # post process
    SHIFT_HEATMAP: True                     # shift heatmap
    USE_GT_BBOX: False                      # use groundtruth bbox
    MODEL_FILE: ''                          # model file to test
    DATALOADER_WORKERS: 8
    COCO_BBOX_FILE: 'experiments/COCO_val2017_detections_AP_H_56_person.json'
#nms-related
    OKS_THRE: 0.9                           # oks threshold
    IN_VIS_THRE: 0.2                        # visible threshold
    BBOX_THRE: 1.0                          # bbox threshold
    IMAGE_THRE: 0.0                         # image threshold
    NMS_THRE: 1.0                           # nms threshold
```

**Notice:** When `TRAIN.RESUME=True`, the training script resume training from `TRAIN.PRETRAINED`.

## [Training Process](#contents)

### [Training](#contents)

#### Running on Ascend

Run `scripts/run_standalone_train.sh` to train the model standalone. The usage of the script is:

```shell
bash scripts/run_standalone_train.sh [CKPT_SAVE_DIR] [DEVICE_ID] [BATCH_SIZE]
# example: bash scripts/run_standalone_train.sh ./ckpt/ascend_standalone 0 128
```

The script will run training in the background, you can view the results through the file `train_log[X].txt` as follows:

```log
loading parse...
batch size :128
loading dataset from ../coco2017/train2017
loaded 149813 records from coco dataset.
loading pretrained model ./models/resnet50.ckpt
start training, epoch size = 140
epoch: 1 step: 1170, loss is 0.000699
Epoch time: 492271.194, per step time: 420.745
epoch: 2 step: 1170, loss is 0.000586
Epoch time: 456265.617, per step time: 389.971
...
```

The model checkpoint will be saved into `[CKPT_SAVE_DIR]`.

#### Running on GPU

Run `scripts/run_standalone_train_gpu.sh` to train the model standalone on GPU. The usage of the script is:

```shell
bash scripts/run_standalone_train_gpu.sh [CKPT_SAVE_DIR(relative)] [BATCH_SIZE] [END_EPOCH] [DEVICE_ID]
# example: bash scripts/run_standalone_train_gpu.sh ./ckpt/gpu_standalone/ 64 185 0
```

The script will run training in the background, you can view the results through the file `./logs/run_standalone_train_gpu.log` as follows:

```log
...
batch size :64
loading dataset from ../coco2017/train2017
loaded 149813 records from coco dataset.
use mindspore-style maxpool
loading pretrained model ./models/resnet50.ckpt
start training, epoch size = 200
epoch: 1 step: 100, loss is 0.001145
epoch: 1 step: 200, loss is 0.001206
...
epoch: 1 step: 2200, loss is 0.000607
epoch: 1 step: 2300, loss is 0.0007854
epoch time: 496858.148 ms, per step time: 212.333 ms
epoch: 2 step: 60, loss is 0.0005794
epoch: 2 step: 160, loss is 0.0006633
...
epoch: 2 step: 2160, loss is 0.000607
epoch: 2 step: 2260, loss is 0.0006447
epoch time: 491012.196 ms, per step time: 209.834 ms
...
```

The model checkpoint will be saved into `[CKPT_SAVE_DIR]`.

### [Distributed Training](#contents)

#### Running on Ascend

Run `scripts/run_distribute_train.sh` to train the model distributed. The usage of the script is:

```shell
bash scripts/run_distribute_train.sh [MINDSPORE_HCCL_CONFIG_PATH] [CKPT_SAVE_DIR] [RANK_SIZE]
# example: bash scripts/run_distribute_train.sh /root/hccl_8p_01234567_10.155.170.71.json ./checkpoint 8
```

The above shell script will run distribute training in the background. You can view the results through the file `train_parallel[X]/log.txt` as follows:

```log
loading parse...
batch size :64
loading dataset from /data/coco2017/train2017
loaded 149813 records from coco dataset.
loading pretrained model ./models/resnet50.ckpt
start training, epoch size = 140
epoch: 1 step: 585, loss is 0.0007944
Epoch time: 236219.684, per step time: 403.794
epoch: 2 step: 585, loss is 0.000617
Epoch time: 164792.001, per step time: 281.696
...
```

The model checkpoint will be saved into `[CKPT_SAVE_DIR]`.

#### Running on GPU

Run `scripts/run_distribute_train_gpu.sh` to train the model distributed. The usage of the script is:

```shell
bash run_distribute_train_gpu.sh [CKPT_SAVE_DIR(relative)] [BATCH_SIZE] [END_EPOCH] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)]
# example: bash scripts/run_distribute_train_gpu.sh ./ckpt/gpu_distributed/ 32 165 8 0,1,2,3,4,5,6,7
```

The above shell script will run distribute training in the background. You can view the results through the file `./logs/run_distribute_train_gpu.log` as follows:

```log
...
batch size : 32
Now we use 8 devices! This is No.0
Now we use 8 devices! This is No.4
Now we use 8 devices! This is No.6
Now we use 8 devices! This is No.7
Now we use 8 devices! This is No.1
Now we use 8 devices! This is No.2
Now we use 8 devices! This is No.5
Now we use 8 devices! This is No.3
loading dataset from ../coco2017/train2017
loading dataset from ../coco2017/train2017
...
epoch: 1 step: 400, loss is 0.0009723
epoch: 1 step: 500, loss is 0.0009146
epoch: 1 step: 500, loss is 0.00093
epoch: 1 step: 500, loss is 0.0007286
epoch: 1 step: 500, loss is 0.000901
epoch: 1 step: 500, loss is 0.0008855
epoch: 1 step: 500, loss is 0.0007744
epoch: 1 step: 500, loss is 0.0008564
epoch: 1 step: 500, loss is 0.0008945
epoch time: 398171.379 ms, per step time: 680.635 ms
epoch time: 398149.378 ms, per step time: 680.597 ms
epoch time: 398115.206 ms, per step time: 680.539 ms
epoch time: 398193.271 ms, per step time: 680.672 ms
epoch time: 398175.092 ms, per step time: 680.641 ms
epoch time: 398161.965 ms, per step time: 680.619 ms
epoch time: 397954.691 ms, per step time: 680.264 ms
epoch time: 398129.488 ms, per step time: 680.563 ms
...
```

The model checkpoint will be saved into `[CKPT_SAVE_DIR]`.

## [Evaluation Process](#contents)

### Running on Ascend

Run `scripts/run_eval.sh` to evaluate the model with one Ascend processor. The usage of the script is:

```shell
bash scripts/run_eval.sh [TEST_MODEL_FILE] [COCO_BBOX_FILE] [DEVICE_ID]
# example: bash scripts/run_eval.sh ./ckpt/ascend_standalone/simplepose-140_1170.ckpt ./experiments/COCO_val2017_detections_AP_H_56_person.json 0
```

The above shell command will run validation procedure in the background. You can view the results through the file `eval_log[X].txt`. The result will be achieved as follows:

```log
use flip test: True
loading model ckpt from results/distributed/sim-140_1170.ckpt
loading dataset from ../coco2017/val2017
loading bbox file from experiments/COCO_val2017_detections_AP_H_56_person.json
Total boxes: 104125
1024 samples validated in 18.133189916610718 seconds
2048 samples validated in 4.724390745162964 seconds
...
```

### Running on GPU

Run `scripts/run_eval_gpu.sh` to evaluate the model with one GPU. The usage of the script is:

```shell
bash scripts/run_eval_gpu.sh [TEST_MODEL_FILE] [COCO_BBOX_FILE] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)]
# example: bash scripts/run_eval_gpu.sh ./ckpt/gpu_distributed/simplepose-185_1170.ckpt ./experiments/COCO_val2017_detections_AP_H_56_person.json 0
```

The above shell command will run validation procedure in the background. You can view the results through the file `./logs/run_eval_gpu.log`. The result will be achieved as follows:

```log
use mindspore-style maxpool
loading model ckpt from ./ckpt/gpu_distributed/simplepose-165_585.ckpt
loading dataset from ../coco2017/val2017
loading bbox file from ./experiments/COCO_val2017_detections_AP_H_56_person.json
Total boxes: 104125
1024 samples validated in 17.805729389190674 seconds
...
103424 samples validated in 3.9891345500946045 seconds
(104125, 17, 3) (104125, 2) 104125
loading annotations into memory...
Done (t=0.27s)
creating index...
index created!
Loading and preparing results...
DONE (t=3.44s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *keypoints*
DONE (t=12.62s).
Accumulating evaluation results...
DONE (t=0.40s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.704
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.891
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.779
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.670
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.773
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.764
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.932
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.832
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.827
coco eval results saved to ckpt/gpu_dist/simplepose-165_585/results/keypoints_results.pkl
AP: 0.7043184629348278
```

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

- Export on local

```shell
python export.py
```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# (1) Upload the code folder to S3 bucket.
# (2) Click to "create training task" on the website UI interface.
# (3) Set the code directory to "/{path}/simple_pose" on the website UI interface.
# (4) Set the startup file to /{path}/simple_pose/export.py" on the website UI interface.
# (5) Perform a .
#     a. setting parameters in /{path}/simple_pose/default_config.yaml.
#         1. Set ”enable_modelarts: True“
#         2. Set “TEST.MODEL_FILE: ./{path}/*.ckpt”('TEST.MODEL_FILE' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
#         3. Set ”EXPORT.FILE_NAME: simple_pose“
#         4. Set ”EXPORT.FILE_FORMAT：MINDIR“
# (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (This step is useless, but necessary.).
# (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (9) Under the item "resource pool selection", select the specification of a single card.
# (10) Create your job.
# You will see simple_pose.mindir under {Output file path}.
```

The `TEST.MODEL_FILE` parameter is required
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
When the network processes datasets, if the last batch is insufficient, it will not be automatically supplemented, in a nutshell, batch_Size set to 1 will go better.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` means weather the dataset is processed in binary format, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```log
AP: 0.7036180026660003
```

# [Model Description](#contents)

## [Performance](#contents)

### SimplePoseNet on COCO2017 with detector

#### Ascend and GPU Performance parameters

| Parameters          | Ascend Standalone           | Ascend Distributed          | GPU Standalone              | GPU Distributed             |
| ------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| Model Version       | SimplePoseNet               | SimplePoseNet               | SimplePoseNet               | SimplePoseNet               |
| Resource            | Ascend 910; OS Euler2.8     | 4 Ascend 910 cards; OS Euler2.8  | Tesla V100-PCIE 32G; Ubuntu 18.04.3 LTS |8 Tesla V100-PCIE 32G cards; Ubuntu 18.04.3 LTS |
| Uploaded Date       | 12/18/2020 (month/day/year) | 12/18/2020 (month/day/year) | 10/13/2021 (month/day/year) | 10/29/2021 (month/day/year) |
| MindSpore Version   | 1.1.0                       | 1.1.0                       | 1.5.0rc1                    | 1.5.0rc1                    |
| Dataset             | COCO2017                    | COCO2017                    | COCO2017                    | COCO2017                    |
| Training Parameters | epoch=140, batch_size=128   | epoch=140, batch_size=64    | epoch=185, batch_size=64    | epoch=165, batch_size=32    |
| Optimizer           | Adam                        | Adam                        | Adam                        | Adam                        |
| Loss Function       | Mean Squared Error          | Mean Squared Error          | Mean Squared Error          | Mean Squared Error          |
| Outputs             | heatmap                     | heatmap                     | heatmap                     | heatmap                     |
| Train Performance   | mAP: 70.4                   | mAP: 70.4                   | mAP: 70.1                   | mAP: 70.5                   |
| Speed               | 1pc: 389.915 ms/step        | 4pc: 281.356 ms/step        | 1pc: 234.818 ms/step        | 8pc: 188.138 ms/step        |

#### Note

- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.
- The dataset preprocessing and general training configurations are shown in [Script Parameters](#script-parameters) section.

# [Description of Random Situation](#contents)

In `src/dataset.py`, we set the seed inside “create_dataset" function. We also use random seed in `src/model.py` to initial network weights.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
