# Contents

<!-- TOC -->

- [Contents](#contents)
- [YOLOX Description](#yolox-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Single-Device Training](#single-device-training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
            - [Start the Python command.](#start-the-python-command)
            - [Start the shell script.](#start-the-shell-script)
    - [Exporting MindIR Model](#exporting-mindir-model)
    - [Inference Process](#inference-process)
        - [Usage](#usage)
            - [Description](#description)
        - [Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Random Seed Description](#random-seed-description)
- [ModelZoo Home Page](#modelzoo-home-page)

<!-- /TOC -->

# YOLOX Description

YOLOX is an anchor-free version of the You Only Look Once (YOLO) series. Compared with YOLOv3 to YOLOv5, YOLOX features simpler network design and better performance. YOLOX is committed to connecting academic research and industry. For more details about the network, see Arxiv's paper.\
[Paper](https://arxiv.org/pdf/2107.08430.pdf): `YOLOX: Exceeding YOLO Series in 2021`

[Official code](https://github.com/Megvii-BaseDetection/YOLOX): <https://github.com/Megvii-BaseDetection/YOLOX>

# Model Architecture

As a new improvement to YOLO series in 2021, the backbone network of the YOLOX model adopts the DarkNet-53 of YOLOv3 and the CSP, Focus module, SPP (spatial pyramid pooling) module, and PANet path-aggregation neck of YOLOv4 to YOLOv5. For details about DarkNet53 and other network modules, see the design of YOLOv3, YOLOv4, and YOLOv5. To solve the conflict between classification and regression in target detection, YOLOX decouples the regression and classification branch in head, and adds the obj branch to the regression branch.

# Dataset

Used dataset: [COCO 2017](https://cocodataset.org/#download)

Supported datasets: COCO 2017 or datasets in the same format as MS COCO

Supported annotations: COCO 2017 or annotations in the same format as MS COCO

- You can define the directory and file names. The directory structure is as follows.

    ```ext

            ├── dataset
                ├── coco2017
                    ├── annotations
                    │   ├─ instances_train2017.json
                    │   └─ instances_val2017.json
                    ├─ train
                    │   ├─picture1.jpg
                    │   ├─ ...
                    │   └─picturen.jpg
                    └─ val
                        ├─picture1.jpg
                        ├─ ...
                        └─picturen.jpg

    ```

- To define a dataset, you need to convert the dataset format to COCO. In addition, the data in the JSON file should correspond to the image data.

# Environment Requirements

- Hardware
    - Set up the hardware environment with Ascend AI Processors.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the following resources:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# Quick Start

- After installing MindSpore from the official website, you can perform the following steps for training and evaluation:
- Before running the network, prepare and generate the **hccl_8p.json** file, and run the **model_utils/hccl_tools.****py** script.

    ```command
    python hccl_tools.py
    ```

- Select backbone: Support training yolox-darknet53 and yolox-x. Before training, you need to specify the backbone name. For example, set backbone to **yolox_darknet53** or **yolox_x** in the **default_config.yaml** file, you can also manually specify the backbone name in the command line, for example:

`python train.py --backbone="yolox_darknet53"`

- The training epochs are divided into the first 285 epochs and the last 15 epochs. The last 15 training epochs disable the data augmentation and use l1 loss function, to which the first 285 training epochs are contrary. If you do not terminate the training after 300 epochs, please decrease the no_aug_epochs parameter in the default_config.yaml file.
  **Note that you need to specify the checkpoint file of the 285th epoch as the network weight for the last 15 epochs to perform training. See the following example.**
- Local training

    ```shell
  # Single device training (first 285 epochs)
  python train.py --config_path=yolox_darknet53.yaml --data_aug=True --is_distributed=0 --backbone='yolox_darknet53'
    ```

  ```shell
  # Single device training (last 15 epochs)
  python train.py --config_path=yolox_darknet53.yaml --data_aug=False --is_distributed=0 --backbone='yolox_darknet53' --resume_yolox="your_285_ckpt_file.ckpt"
  ```

  ```shell
  # Run shell script for 8-device training (first 285 epochs)
  bash run_distribute_train.sh xxx/dataset/  rank_table_8pcs.json  yolox_darknet53
  ```

  ```shell
  # Run shell script for 8-device training (last 15 epochs)
  bash run_distribute_train.sh xxx/dataset/  rank_table_8pcs.json  yolox_darknet53  your_285_ckpt_file_path.ckpt
  ```

- Local evaluation

    ```shell
    python eval.py --config_path=yolox_darknet53.yaml --data_dir=./dataset/xxx --val_ckpt=your_val_ckpt_file_path --per_batch_size=8

    # Multi-device evaluation
    bash run_distribute_eval.sh xx/dataset/ your_val_ckpt_file_path yolox_darknet53 8 rank_table_8pcs.json
    ```

# Script Description

## Script and Sample Code

```text
    |----README_CN.md
    |----ascend310_infer
    |    |----build.sh
    |    |----CMakeLists.txt
    |    |----inc
    |    |    |----utils.h
    |    |----src
    |    |    |----main.cc
    |    |    |----utils.cc
    |----model_utils
    |    |----__init__.py
    |    |----config.py
    |    |----device_adapter.py
    |    |----hccl_tools.py
    |    |----local_adapter.py
    |    |----moxing_adapter.py
    |----scripts
    |    |----run_distribute_train.sh
    |    |----run_distribute_eval.sh
    |    |----run_infer_310.sh
    |    |----run_eval.sh
    |    |----run_standalone_train.sh
    |----src
    |    |----__init__.py
    |    |----boxes.py
    |    |----darknet.py
    |    |----initializer.py
    |    |----logger.py
    |    |----network_blocks.py
    |    |----transform.py
    |    |----util.py
    |    |----yolox.py
    |    |----yolox_dataset.py
    |    |----yolo_fpn.py
    |    |----yolo_pafpn.py
    |----third_party
    │    |----__init__.py
    │    |----build.sh
    │    |----cocoeval
    │    │    |----cocoeval.cpp
    │    │    |----cocoeval.h
    │    |----fast_coco_eval_api.py
    │    |----setup.py
    |----train.py
    |----eval.py
    |----export.py
    |----postprocess.py
    |----preprocess.py
    |----yolox_darknet53_config.yaml
    |----yolox_x_config.yaml
```

## Script Parameters

The main parameters in **train.py** are as follows:

```text

--backbone                  The backbone network for training. The default value is yolox_darknet53. You can also set it to yolox_x.
--data_aug                   Indicates whether to enable data augmentation. The default value is True, that means the data augmentation is enabled in the previous training epochs and is disabled in the later training epochs.
--device_target
                            Device that implements the code. The default value is 'Ascend'.
--outputs_dir               Directory of storage file for training information
--save_graphs               Indicates whether to save image files. The default value is False.
--aug_epochs                 Number of training epochs with data augmentation. The default value is 285.
--no_aug_epochs              Number of training epochs without data augmentation. The default value is 15.
--data_dir                  Dataset directory
--need_profiler
                            Indicates whether to use the profiler. 0: no; 1: yes. Default value: 0.
--per_batch_size            Batch size of training. The default value is 8.
--max_gt                    Maximum number of GTs in an image. The default value is 50.
--num_classes                Number of dataset classes. The default value is 80.
--input_size                The input size of the network. The default value is 640.
--fpn_strides               FPN scaling stride. The default value is 8, 16, and 32.
--use_l1                    Indicates whether to use l1 loss function. The value is True only in the training epochs where data augmentation is disabled. The default value is False.
--use_syc_bn                Indicates whether to enable BN synchronization. The default value is True.
--n_candidate_k             Number of IOUs in dynamic K. The default value is 10.
--lr                        Learning rate. The default value is 0.01.
--min_lr_ratio              Decay rate for learning rate. The default value is 0.05.
--warmup_epochs             Number of warm-up epochs. The default value is 5.
--weight_decay              Decay weight. The default value is 0.0005.
--momentum                  Momentum
--log_interval              Number of steps at the log recording interval
--ckpt_interval             Interval for saving a checkpoint. The default value is -1.
--is_save_on_master         Indicates whether to save the checkpoint on master or all ranks. 1 (default value): master; 0: all ranks.
--is_distributed            Indicates whether to enable distributed training. 1 (default value): yes; 0: no.
--rank                      Local rank in the distributed training. The default value is 0.
--group_size                Total number of device processes. The default value is 1.
--run_eval                  Indicates whether to enable inference during training. The default value is False.
--eval_parallel             Indicates whether to enable parallel inference. The default value is True, which is valid only when run_eval is set to True and is_distributed is set to 1.
```

## Training Process

Because YOLOX uses powerful data augmentation, the pre-trained model parameters on ImageNet are not required. Therefore, all training will start from scratch. The training consists of two steps. The first step is to train from scratch and enable data augmentation. The second step is to use the checkpoint file trained in the first step as the pre-training model and disable data augmentation.

### Single-Device Training

Run the Python script to start training on Ascend AI Processor (single device).

- Step 1
    Start the Python command.

    ```shell
    # Single-device training (first 285 epochs with data augmentation)
    python train.py --config_path=yolox_darknet53.yaml --data_dir=~/coco2017 --is_distributed=0 --backbone='yolox_darknet53'
    ```

    Start the shell script.

    ```shell
    bash run_standalone_train.sh  [DATASET_PATH] [BACKBONE]
    ```

    After the training in step 1 is complete, find the checkpoint file saved in the last epoch in the default folder and use the file path as the input parameter for the training in step 2.

- Step 2
    Start the Python command.

    ```shell
    # Single-device training (last 15 epochs without data augmentation)
    python train.py --config_path=yolox_darknet53.yaml --data_dir=~/coco2017 --is_distributed=0 --backbone='yolox_darknet53' --resume_yolox="your_285_ckpt_file_path.ckpt"
     ```

    Start the shell script.

    ```shell
    bash run_standalone_train.sh  [DATASET_PATH] [BACKBONE] [RESUME_CKPT]
    ```

### Distributed Training

Run the shell script to start distributed training on Ascend AI Processors (8 devices).

- Step 1

  ```shell

  # Run shell script for 8-device training (first 285 epochs).
  bash run_distribute_train.sh xxx/dataset/  rank_table_8pcs.json  yolox_darknet53

  ```

  After the training in step 1 is complete, find the checkpoint file saved in the last epoch in the default folder and use the file path as the input parameter for the training in step 2.
- Step 2

  ```shell

  # Run shell script for 8-device training (last 15 epochs).
  bash run_distribute_train.sh xxx/dataset/  rank_table_8pcs.json  yolox_darknet53  your_285_ckpt_file_path.ckpt

  ```

  The preceding shell script will run distributed training in the backend. You can view the result in the **train_parallel0/log.txt** file. The following loss values are obtained:

    ```log

    ...
    2022-10-10 11:43:14,405:INFO:epoch: [1/300] step: [150/1848], loss: 15.9977, lr: 0.000003, avg step time: 332.07 ms
    2022-10-10 11:43:37,711:INFO:epoch: [1/300] step: [160/1848], loss: 14.6404, lr: 0.000003, avg step time: 330.58 ms
    2022-10-10 11:44:41,012:INFO:epoch: [1/300] step: [170/1848], loss: 16.2315, lr: 0.000004, avg step time: 330.08 ms
    2022-10-10 11:43:44,326:INFO:epoch: [1/300] step: [180/1848], loss: 16.9418, lr: 0.000004, avg step time: 331.37 ms
    2022-10-10 11:43:47,646:INFO:epoch: [1/300] step: [190/1848], loss: 17.1101, lr: 0.000005, avg step time: 331.87 ms
    2022-10-10 11:43:50,943:INFO:epoch: [1/300] step: [200/1848], loss: 16.7288, lr: 0.000005, avg step time: 329.74 ms
    ...

    ```

## Evaluation Process

### Evaluation

#### Start the Python command

```shell
python eval.py --data_dir=./dataset/xxx --val_ckpt=your_val_ckpt_file_path --per_batch_size=8 --backbone=yolox_darknet53
```

If **backbone** is set to **yolox_darknet53** or **yolox_x**, the preceding python commands run in the backend. You can view the result in the `%Y- %m- %d_time_ %H_ %M_ %S.log` file.

Since the evaluation result program in the `pycocotools` is slow, it is advised to use the third-party library provided in the `third_party` folder to improve the result evaluation speed. Run the `bash build.sh` command in the `third_party` folder to build a dynamic link library. The program can automatically invoke this library.

#### Start the shell script

```shell
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [BACKBONE] [BATCH_SIZE]
```

```log

   ===============================coco eval result===============================
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.480
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.674
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.524
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.304
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.525
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.616
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.364
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.585
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.625
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.435
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.678
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.762

```

## Exporting MindIR Model

```shell

python export.py --config_path=yolox_darknet53.yaml --backbone=yolox_darknet53 --val_ckpt [CKPT_PATH] --file_format [MINDIR/AIR]

```

The `backbone` parameter specifies the backbone network, whose value **yolox_darknet53** and **yolox_x** are available for selection. The `val_ckpt` parameter specifies the checkpoint file of the model.

## Inference Process

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

### Usage

#### Description

- Run **export.py** to export the MindIR file. You can also specify the default backbone type in the configuration file.
- Run **preprocess.py** to convert dataset to binary file.
- Run **postprocess.py** to perform inference based on the output result of the MindIR network and save the evaluation results.

Execute the complete inference script as follows:

```shell

# Inference on Ascend 310 AI Processor
bash run_infer_310.sh [MINDIR_PATH] [DATA_DIR] [DEVICE_ID]
```

### Result

The inference result is saved in the current path. You can run the **cat acc.log** command to view the final accuracy.

```text

                            yolox-darknet53
=============================coco eval result==================================
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.480
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.674
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.524
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.304
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.525
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.616
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.585
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.678
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.762
                                    yolox-x
=============================coco eval result==================================
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.502
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.685
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.545
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.306
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.548
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.661
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.380
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.611
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.649
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.700
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.818

```

# Model Description

## Performance

### Evaluation Performance

YOLOX is applied to 118,000 images (the annotation and data format must be the same as those of COCO 2017).

|Parameter| YOLOX-darknet53                                                    |
| -------------------------- |--------------------------------------------------------------------|
|Resources| Ascend 910 AI Processor; 2.60 GHz CPU with 192 cores; 755 GB memory; EulerOS 2.8.              |
|Upload date| 2022-10-21                                                       |
| MindSpore version| 1.8.1-alpha                                                        |
|Dataset| coco2017                                                           |
|Training parameters| epoch=300, batch_size=8, lr=0.01,momentum=0.9                      |
| Optimizer                 | SGD                                                                |
|Loss function| Sigmoid Cross Entropy, Iou Loss, L1 Loss                           |
|Output| Framework and label                                                              |
|Speed| Single device: 25 FPS; 8 devices: 190 FPS (shape = 640)                                    |
|Total duration| 52 hours                                                              |
|Finetuned checkpoint| About 1000 MB (.ckpt file)                                                   |
|Script| <https://gitee.com/mindspore/models/tree/master/official/cv/YOLOX> |

|Parameter| YOLOX-X                                                           |
| -------------------------- |--------------------------------------------------------------------|
|Resources| Ascend 910 AI Processor; 2.60 GHz CPU with 192 cores; 755 GB memory; EulerOS 2.8.              |
|Upload date| 2022-3-11                                                        |
| MindSpore version| 1.3.0-alpha                                                        |
|Dataset| 118,000 images                                                         |
|Training parameters| epoch=300, batch_size=8, lr=0.04,momentum=0.9                      |
| Optimizer                 | Momentum                                                           |
|Loss function| Sigmoid Cross Entropy, Iou Loss, L1 Loss                           |
|Output| Frame and label                                                              |
|Loss| 50                                                                 |
|Speed| Single device: 12 FPS; 8 devices: 93 FPS (shape = 640)                                     |
|Total duration| 106 hours                                                             |
|Finetuned checkpoint| About 1100 MB (.ckpt file)                                                   |
|Script| <https://gitee.com/mindspore/models/tree/master/official/cv/YOLOX> |

### Inference Performance

YOLOX is applied to 118,000 images. (The annotation and data format must be the same as those of COCO test 2017.)

|Parameter| YOLOX-darknet53                     |
| -------------------------- |-------------------------------------|
| Resources                  | Ascend 910 AI Processor; 2.60 GHz CPU with 192 cores; 755 GB memory|
|Upload date| 2022-10-21                        |
| MindSpore version| 1.3.0-alpha                         |
|Dataset| 118,000 images                          |
|Batch size| 1                                   |
|Output| Boundary position, score, and probability                       |
|Accuracy| map = 48.0% (shape=640)             |
|Inference model| About 1000 MB (.ckpt file)                    |

|Parameter| YOLOX-X|
| -------------------------- | ----------------------------------------------------------- |
| Resources                  | Ascend 910 AI Processor; 2.60 GHz CPU with 192 cores; 755 GB memory            |
|Upload date|2020-10-16|
| MindSpore version|1.3.0-alpha|
|Dataset|118,000 images|
|Batch size|1|
|Output|Boundary position, score, and probability|
|Accuracy|map =50.2% (shape=640)|
|Inference model|About 1100 MB (.ckpt file)|

# Random Seed Description

The seed in the **create_dataset** function is set in **dataset.py**. Thr seed for weight initialization is set in **var_init.py**.

# ModelZoo Home Page

For details, please go to the [official website](https://gitee.com/mindspore/models).
