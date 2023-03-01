# Contents

<!-- TOC -->

- [Contents](#contents)
- [DeepLabV3+ Description](#deeplabv3-description)
    - [Description](#description)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage)
            - [On Ascend AI Processors](#on-ascend-ai-processors)
            - [On ModelArts](#on-modelarts)
            - [On GPUs](#on-gpus)
        - [Results](#results)
            - [On Ascend AI Processors](#on-ascend-ai-processors-1)
            - [On ModelArts](#on-modelarts-1)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [On Ascend AI Processors](#on-ascend-ai-processors-2)
            - [On GPUs](#on-gpus-1)
        - [Results](#results-1)
            - [Training Accuracies](#training-accuracies)
    - [Exporting a MindIR Model](#exporting-a-mindir-model)
    - [Inference Process](#inference-process)
        - [Usage](#usage-2)
        - [Results](#results-2)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [Running ONNX evaluation](#running-onnx-evaluation)
- [Random Seed Description](#random-seed-description)
- [ModelZoo Home Page](#modelzoo-home-page)

<!-- /TOC -->

# DeepLabV3+ Description

## Description

DeepLab is a series of semantic image segmentation models. DeepLabv3+ uses encoder-decoder to fuse multi-scale information and retains the original atrous convolution and atrous spatial pyramid pooling (ASPP) layer.
Its backbone network uses the ResNet model, which improves the robustness and running rate of semantic segmentation.

For more information about the network, see [paper][1].
`Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." Proceedings of the European conference on computer vision (ECCV). 2018.`

[1]: https://arxiv.org/abs/1802.02611

# Model Architecture

ResNet-101 is used as the backbone, encoder-decoder is used to perform multi-scale information fusion, and the atrous convolution is used to extract dense features.

# Datasets

[The PASCAL VOC dataset](https://host.robots.ox.ac.uk/pascal/VOC/) and [semantic boundaries dataset (SBD)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

- Download pretrain model

GPU: [click to download](https://download.mindspore.cn/models/r1.3/resnet101_ascend_v130_imagenet2012_official_cv_top1acc78.55_top5acc94.34.ckpt); Ascend: [clink to download](https://download.mindspore.cn/model_zoo/official/cv/deeplabv3p/resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78.ckpt)

- Download the segmentation dataset.

- Prepare a training data list to store the relative paths of images and annotations. For example:

     ```text
     VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg VOCdevkit/VOC2012/SegmentationClassGray/2007_000032.png
     VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg VOCdevkit/VOC2012/SegmentationClassGray/2007_000039.png
     VOCdevkit/VOC2012/JPEGImages/2007_000063.jpg VOCdevkit/VOC2012/SegmentationClassGray/2007_000063.png
     VOCdevkit/VOC2012/JPEGImages/2007_000068.jpg VOCdevkit/VOC2012/SegmentationClassGray/2007_000068.png
     ......
     ```

You can also run the script to automatically generate a data list: `python get_dataset_list.py --data_root=/PATH/TO/DATA`

- Configure and run **get_dataset_mindrecord.sh** to convert the dataset to MindRecord. Parameters in **scripts/get_dataset_mindrecord.sh**:

     ```
     --data_root                 Root directory of the training data
     --data_lst                  Training data list (prepared above)
     --dst_path                  Path of the MindRecord dataset
     --num_shards                Number of shards in the MindRecord dataset
     --shuffle                   Specifies whether to shuffle data.
     ```

# Features

## Mixed Precision

[Mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) accelerates the training process of deep neural networks by using the single-precision (FP32) data and half-precision (FP16) data without compromising the precision of networks trained with single-precision (FP32) data. It not only accelerates the computing process and reduces the memory usage, but also supports a larger model or batch size to be trained on specific hardware.
Take the FP16 operator as an example. If the input data format is FP32, MindSpore automatically reduces the precision to process data. You can open the INFO log and search for the keyword "reduce precision" to view operators with reduced precision.

# Environment Requirements

- Hardware
    - Set up the hardware environment with Ascend AI Processors.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the following resources:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- Install the python package in the **requirements.txt** file.
- Generate the **config.json** file for 8-device training.

    ```
    # Go to the project root directory.
    cd src/tools/
    python3 get_multicards_json.py 10.111.*.*
    # 10.111.*.* indicates the IP address of the computer.
    ```

# Quick Start

After installing MindSpore from the official website, you can perform the following steps for training and evaluation:

- On Ascend AI Processors

    Based on the original DeepLabV3+ paper, we conducted two training experiments using the VOCaug (also called trainaug) dataset and evaluation using the VOC VAL dataset.

    Run the following training script to configure single-device training parameters:

    ```bash
    run_alone_train.sh
    ```

Start 8-device training.

1. Use the VOCaug dataset to train the s16 model and fine-tune the ResNet-101 pre-trained model. The script is as follows:

    ```bash
    run_distribute_train_s16_r1.sh
    ```

2. Use the VOCaug dataset to train the s8 model and fine-tune the model in the previous step. The script is as follows:

    ```bash
    run_distribute_train_s8_r1.sh
    ```

3. Use the VOCtrain dataset to train the s8 model and fine-tune the model in the previous step. The script is as follows:

    ```bash
    run_distribute_train_s8_r2.sh
    ```

Start evaluation.

1. Use the VOC VAL dataset to evaluate the s16 model. The evaluation script is as follows:

    ```bash
    run_eval_s16.sh
    ```

2. Use the VOC VAL dataset to evaluate the multi-scale s16 model. The evaluation script is as follows:

    ```bash
    run_eval_s16_multiscale.sh
    ```

3. Use the VOC VAL dataset to evaluate the multi-scale flipping s16 model. The evaluation script is as follows:

    ```bash
    run_eval_s16_multiscale_flip.sh
    ```

4. Use the VOC VAL dataset to evaluate the s8 model. The evaluation script is as follows:

    ```bash
    run_eval_s8.sh
    ```

5. Use the VOC VAL dataset to evaluate the multi-scale s8 model. The evaluation script is as follows:

    ```bash
    run_eval_s8_multiscale.sh
    ```

6. Use the VOC VAL dataset to evaluate the multi-scale flipping s8 model. The evaluation script is as follows:

    ```bash
    run_eval_s8_multiscale_flip.sh
    ```

- On GPUs

    Start 8-device training.

    1. Use the VOCaug dataset to train the s16 model and fine-tune the ResNet-101 pre-trained model. The script is as follows:

    ```bash
    bash run_distribute_train_s16_r1_gpu.sh /PATH/TO/MINDRECORD_NAME /PATH/TO/PRETRAIN_MODEL
    ```

    2. Use the VOCaug dataset to train the s8 model and fine-tune the model in the previous step. The script is as follows:

    ```bash
    bash run_distribute_train_s8_r1_gpu.sh /PATH/TO/MINDRECORD_NAME /PATH/TO/PRETRAIN_MODEL
    ```

    3. Use the VOCtrain dataset to train the s8 model and fine-tune the model in the previous step. The script is as follows:

    ```bash
    bash run_distribute_train_s8_r2_gpu.sh /PATH/TO/MINDRECORD_NAME /PATH/TO/PRETRAIN_MODEL
    ```

Start evaluation.

1. Use the VOC VAL dataset to evaluate the s16 model. The evaluation script is as follows:

    ```bash
    bash run_eval_s16_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL_DIR DEVICE_ID
    ```

2. Use the VOC VAL dataset to evaluate the multi-scale s16 model. The evaluation script is as follows:

    ```bash
    bash run_eval_s16_multiscale_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL_DIR DEVICE_ID
    ```

3. Use the VOC VAL dataset to evaluate the multi-scale flipping s16 model. The evaluation script is as follows:

    ```bash
    bash run_eval_s16_multiscale_flip_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL_DIR DEVICE_ID
    ```

4. Use the VOC VAL dataset to evaluate the s8 model. The evaluation script is as follows:

    ```bash
    bash run_eval_s8_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL_DIR DEVICE_ID
    ```

5. Use the VOC VAL dataset to evaluate the multi-scale s8 model. The evaluation script is as follows:

    ```bash
    bash run_eval_s8_multiscale_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL_DIR DEVICE_ID
    ```

6. Use the VOC VAL dataset to evaluate the multi-scale flipping s8 model. The evaluation script is as follows:

    ```bash
    bash run_eval_s8_multiscale_flip_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL_DIR DEVICE_ID
    ```

# Script Description

## Script and Sample Code

```shell
.
└──deeplabv3plus
  ├── script
    ├── get_dataset_mindrecord.sh                 # Convert the original data into a MindRecord dataset
    ├── run_alone_train.sh                        # Start the Ascend standalone training (single-device).
    ├── run_distribute_train_s16_r1.sh            # Use the VOCaug dataset of the s16 model to start Ascend distributed training (8-device).
    ├── run_distribute_train_s8_r1.sh             # Use the VOCaug dataset of the s8 model to start Ascend distributed training (8-device).
    ├── run_distribute_train_s8_r2.sh             # Use the VOCtrain dataset of the s8 model to start Ascend distributed training (8-device).
    ├── run_eval_s16.sh                           # Use the s16 model to start Ascend evaluation.
    ├── run_eval_s16_multiscale.sh                # Use the multi-scale s16 model to start Ascend evaluation.
    ├── run_eval_s16_multiscale_filp.sh           # Use the multi-scale flipping s16 model to start Ascend evaluation.
    ├── run_eval_s8.sh                            # Use the s8 model to start Ascend evaluation.
    ├── run_eval_s8_multiscale.sh                 # Use the multi-scale s8 model to start Ascend evaluation.
    ├── run_eval_s8_multiscale_filp.sh            # Use the multi-scale flipping s8 model to start Ascend evaluation.
    ├── run_distribute_train_s16_r1_gpu.sh            # Use the VOCaug dataset of the s16 model to start GPU distributed training (8-device).
    ├── run_distribute_train_s8_r1_gpu.sh             # Use the VOCaug dataset of the s8 model to start GPU distributed training (8-device).
    ├── run_distribute_train_s8_r2_gpu.sh             # Use the VOCtrain dataset of the s8 model to start GPU distributed training (8-device).
    ├── run_eval_s16_gpu.sh                           # Use the s16 model to start GPU evaluation.
    ├── run_eval_s16_multiscale_gpu.sh                # Use the multi-scale s16 model to start GPU evaluation.
    ├── run_eval_s16_multiscale_filp_gpu.sh           # Use the multi-scale flipping s16 model to start GPU evaluation.
    ├── run_eval_s8_gpu.sh                            # Use the s8 model to start GPU evaluation.
    ├── run_eval_s8_multiscale_gpu.sh                 # Use the multi-scale s8 model to start GPU evaluation.
    ├── run_eval_s8_multiscale_filp_gpu.sh            # Use the multi-scale flipping s8 model to start GPU evaluation.
  ├── src
    ├── tools
        ├── get_dataset_list.py               # Obtain the data list.
        ├── get_dataset_mindrecord.py         # Obtain the MindRecord file.
        ├── get_multicards_json.py            # Obtain the rank table file.
        ├── get_pretrained_model.py           # Obtain the ResNet pre-trained model.
    ├── dataset.py                            # Preprocess data.
    ├── deeplab_v3plus.py                     # DeepLabv3+ network structure
    ├── learning_rates.py                     # Generate the learning rate.
    ├── loss.py                               # Definition of DeepLabV3+ loss
  ├── eval.py                                 # Evaluation network
  ├── train.py                                # Training network
  ├──requirements.txt                         # Requirements file
  └──README.md
```

## Script Parameters

Default configuration

```bash
"data_file":"/PATH/TO/MINDRECORD_NAME"            # Dataset path
"device_target":Ascend                            # Training device type
"train_epochs":300                                # Total number of epochs
"batch_size":32                                   # Batch size of the input tensor
"crop_size":513                                   # Cropping size
"base_lr":0.08                                    # Initial learning rate
"lr_type":cos                                     # Decay mode for generating the learning rate
"min_scale":0.5                                   # Minimum scale of data augmentation
"max_scale":2.0                                   # Maximum scale of data augmentation
"ignore_label":255                                # Number of ignored images
"num_classes":21                                  # Number of classes
"model":DeepLabV3plus_s16                         # Select a model.
"ckpt_pre_trained":"/PATH/TO/PRETRAIN_MODEL"      # Path for loading the pre-trained checkpoint
"is_distributed":                                 # Set this parameter to True for distributed training.
"save_steps":410                                  # Step interval for saving a model
"freeze_bn":                                      # Set freeze_bn to True.
"keep_checkpoint_max":200                         # Maximum number of checkpoints for saving a model
```

## Training Process

### Usage

#### On Ascend AI Processors

Based on the original DeepLab V3+ paper, we conducted two training experiments using the VOCaug (also called trainaug) dataset and evaluation using the VOC VAL dataset.

Run the following training script to configure single-device training parameters:

```bash
# run_alone_train.sh
python ${train_code_path}/train.py --data_file=/PATH/TO/MINDRECORD_NAME  \
                    --train_dir=${train_path}/ckpt  \
                    --train_epochs=200  \
                    --batch_size=32  \
                    --crop_size=513  \
                    --base_lr=0.015  \
                    --lr_type=cos  \
                    --min_scale=0.5  \
                    --max_scale=2.0  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=DeepLabV3plus_s16  \
                    --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                    --save_steps=1500  \
                    --keep_checkpoint_max=200 >log 2>&1 &
```

Start 8-device training.

1. Use the VOCaug dataset to train the s16 model and fine-tune the ResNet-101 pre-trained model. The script is as follows:

    ```bash
    # run_distribute_train_s16_r1.sh
    for((i=0;i<=$RANK_SIZE-1;i++));
    do
        export RANK_ID=$i
        export DEVICE_ID=`expr $i + $RANK_START_ID`
        echo 'start rank='$i', device id='$DEVICE_ID'...'
        mkdir ${train_path}/device$DEVICE_ID
        cd ${train_path}/device$DEVICE_ID
        python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                                --data_file=/PATH/TO/MINDRECORD_NAME  \
                                                --train_epochs=300  \
                                                --batch_size=32  \
                                                --crop_size=513  \
                                                --base_lr=0.08  \
                                                --lr_type=cos  \
                                                --min_scale=0.5  \
                                                --max_scale=2.0  \
                                                --ignore_label=255  \
                                                --num_classes=21  \
                                                --model=DeepLabV3plus_s16  \
                                                --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                                --is_distributed  \
                                                --save_steps=410  \
                                                --keep_checkpoint_max=200 >log 2>&1 &
    done
    ```

2. Use the VOCaug dataset to train the s8 model and fine-tune the model in the previous step. The script is as follows:

    ```bash
    # run_distribute_train_s8_r1.sh
    for((i=0;i<=$RANK_SIZE-1;i++));
    do
        export RANK_ID=$i
        export DEVICE_ID=`expr $i + $RANK_START_ID`
        echo 'start rank='$i', device id='$DEVICE_ID'...'
        mkdir ${train_path}/device$DEVICE_ID
        cd ${train_path}/device$DEVICE_ID
        python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                                --data_file=/PATH/TO/MINDRECORD_NAME  \
                                                --train_epochs=800  \
                                                --batch_size=16  \
                                                --crop_size=513  \
                                                --base_lr=0.02  \
                                                --lr_type=cos  \
                                                --min_scale=0.5  \
                                                --max_scale=2.0  \
                                                --ignore_label=255  \
                                                --num_classes=21  \
                                                --model=DeepLabV3plus_s8  \
                                                --loss_scale=2048  \
                                                --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                                --is_distributed  \
                                                --save_steps=820  \
                                                --keep_checkpoint_max=200 >log 2>&1 &
    done
    ```

3. Use the VOCtrain dataset to train the s8 model and fine-tune the model in the previous step. The script is as follows:

    ```bash
    # run_distribute_train_s8_r2.sh
    for((i=0;i<=$RANK_SIZE-1;i++));
    do
        export RANK_ID=$i
        export DEVICE_ID=`expr $i + $RANK_START_ID`
        echo 'start rank='$i', device id='$DEVICE_ID'...'
        mkdir ${train_path}/device$DEVICE_ID
        cd ${train_path}/device$DEVICE_ID
        python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                                --data_file=/PATH/TO/MINDRECORD_NAME  \
                                                --train_epochs=300  \
                                                --batch_size=16  \
                                                --crop_size=513  \
                                                --base_lr=0.008  \
                                                --lr_type=cos  \
                                                --min_scale=0.5  \
                                                --max_scale=2.0  \
                                                --ignore_label=255  \
                                                --num_classes=21  \
                                                --model=DeepLabV3plus_s8  \
                                                --loss_scale=2048  \
                                                --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                                --is_distributed  \
                                                --save_steps=110  \
                                                --keep_checkpoint_max=200 >log 2>&1 &
    done
    ```

#### On ModelArts

Set training parameters and start ModelArts training. The following is an example:

```shell
python  train.py    --train_url=/PATH/TO/OUTPUT_DIR \
                    --data_url=/PATH/TO/MINDRECORD  \
                    --model=DeepLabV3plus_s16  \
                    --modelArts_mode=True  \
                    --dataset_filename=MINDRECORD_NAME  \
                    --pretrainedmodel_filename=PRETRAIN_MODELNAME  \
                    --train_epochs=300  \
                    --batch_size=32  \
                    --crop_size=513  \
                    --base_lr=0.08  \
                    --lr_type=cos  \
                    --save_steps=410  \
```

#### On GPUs

For details about parameter settings, see the 8-device training script in [Quick Start](#quick-start).

### Results

#### On Ascend AI Processors

- Use the s16 model to train the VOCaug dataset.

    ```bash
    # Distributed training result (8-device)
    epoch: 1 step: 41, loss is 0.81338423
    epoch time: 202199.339 ms, per step time: 4931.691 ms
    epoch: 2 step: 41, loss is 0.34089813
    epoch time: 23811.338 ms, per step time: 580.764 ms
    epoch: 3 step: 41, loss is 0.32335973
    epoch time: 23794.863 ms, per step time: 580.363 ms
    epoch: 4 step: 41, loss is 0.18254203
    epoch time: 23796.674 ms, per step time: 580.407 ms
    epoch: 5 step: 41, loss is 0.27708685
    epoch time: 23794.654 ms, per step time: 580.357 ms
    epoch: 6 step: 41, loss is 0.37388346
    epoch time: 23845.658 ms, per step time: 581.601 ms
    ...
    ```

- Use the s8 model to train the VOCaug dataset.

    ```bash
    # Distributed training result (8-device)
    epoch: 1 step: 82, loss is 0.073864505
    epoch time: 226610.999 ms, per step time: 2763.549 ms
    epoch: 2 step: 82, loss is 0.06908825
    epoch time: 44474.187 ms, per step time: 542.368 ms
    epoch: 3 step: 82, loss is 0.059860937
    epoch time: 44485.142 ms, per step time: 542.502 ms
    epoch: 4 step: 82, loss is 0.084193744
    epoch time: 44472.924 ms, per step time: 542.353 ms
    epoch: 5 step: 82, loss is 0.072242916
    epoch time: 44466.738 ms, per step time: 542.277 ms
    epoch: 6 step: 82, loss is 0.04948996
    epoch time: 44474.549 ms, per step time: 542.373 ms
    ...
    ```

- Use the s8 model to train the VOCtrain dataset.

    ```bash
    # Distributed training result (8-device)
    epoch: 1 step: 11, loss is 0.0055908263
    epoch time: 183966.044 ms, per step time: 16724.186 ms
    epoch: 2 step: 11, loss is 0.008914589
    epoch time: 5985.108 ms, per step time: 544.101 ms
    epoch: 3 step: 11, loss is 0.0073758443
    epoch time: 5977.932 ms, per step time: 543.448 ms
    epoch: 4 step: 11, loss is 0.00677738
    epoch time: 5978.866 ms, per step time: 543.533 ms
    epoch: 5 step: 11, loss is 0.0053799236
    epoch time: 5987.879 ms, per step time: 544.353 ms
    epoch: 6 step: 11, loss is 0.0049248594
    epoch time: 5979.642 ms, per step time: 543.604 ms
    ...
    ```

#### On ModelArts

- Use the s16 model to train the VOCaug dataset.

    ```bash
    epoch: 1 step: 41, loss is 0.6122837
    epoch: 2 step: 41, loss is 0.4066103
    epoch: 3 step: 41, loss is 0.3504579
    ...
    ```

## Evaluation Process

### Usage

#### On Ascend AI Processors

Use **--ckpt_path** to configure checkpoints, run the script, and print mIOU in **eval_path/eval_log**.

```bash
./run_eval_s16.sh                     # Test the s16 model.
./run_eval_s16_multiscale.sh          # Test the multi-scale s16 model.
./run_eval_s16_multiscale_flip.sh     # Test the multi-scale flipping s16 model.
./run_eval_s8.sh                      # Test the s8 model.
./run_eval_s8_multiscale.sh           # Test the multi-scale s8 model.
./run_eval_s8_multiscale_flip.sh      #  Test the multi-scale flipping s8 model.
```

The following is an example of the test script:

```bash
python ${train_code_path}/eval.py --data_root=/PATH/TO/DATA  \
                    --data_lst=/PATH/TO/DATA_lst.txt  \
                    --batch_size=16  \
                    --crop_size=513  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=DeepLabV3plus_s8  \
                    --scales=0.5  \
                    --scales=0.75  \
                    --scales=1.0  \
                    --scales=1.25  \
                    --scales=1.75  \
                    --flip  \
                    --freeze_bn  \
                    --ckpt_dir=/PATH/TO/PRETRAIN_MODEL_DIR >${eval_path}/eval_log 2>&1 &
```

#### On GPUs

For details about parameter settings, see the evaluation script in [Quick Start](#quick-start).

### Results

Run the applicable training script to obtain the result. To get the same result, follow the steps in [Quick Start](#quick-start).

#### Training Accuracies

| **Network**| OS = 16 | OS = 8 | MS |Flip| mIOU |mIOU in the Paper|
| :----------: | :-----: | :----: | :----: | :-----: | :-----: | :-------------: |
| deeplab_v3+ | √     |      |      |       | 79.78 | 78.85    |
| deeplab_v3+ | √     |     | √    |       | 80.59 |80.09   |
| deeplab_v3+ | √     |     | √    | √     | 80.76 | 80.22        |
| deeplab_v3+ |       | √    |      |       | 79.56 | 79.35    |
| deeplab_v3+ |       | √    | √    |       | 80.43 |80.43   |
| deeplab_v3+ |       | √    | √    | √     | 80.69 | 80.57        |

Note: OS indicates the output stride, and MS indicates multi-scale.

## Exporting a MindIR Model

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The `ckpt_file` parameter is mandatory, and the value range of `EXPORT_FORMAT` is ["AIR", "MINDIR"].

## Inference Process

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

### Usage

```shell
# Inference on Ascend 310 AI Processors
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DATA_ROOT] [DATA_LIST] [DEVICE_ID]
```

`MINDIR_PATH`: Path of the MindIR model.

`DATA_PATH`: Actual file path of the test set (./VOCdevkit/VOC2012/JPEGImages/)

`DATA_ROOT`: Root directory of the PASCAL VOC and SBD datasets

`DATA_LIST`: Path of the data list **voc_val_lst.txt**, which can be obtained by running **get_dataset_list.py**.

`DEVICE_ID`: Optional. The default value is **0**.

### Results

The inference result is saved in the current path. You can view the final accuracy result in **acc.log**.

| **Network**    | OS = 16 | OS = 8|  mIOU  |
| :----------: | :-----: | :-----: | :-----: |
| deeplab_v3+ |  √    |      | 79.63 |
| deeplab_v3+ |       | √    | 79.33 |

# Model Description

## Performance

### Evaluation Performance

| Parameter| Ascend 910| GPU |
| -------------------------- | -------------------------------------- | -------------------------------------- |
| Model version| DeepLabV3+ | DeepLabV3+ |
| Resources| Ascend 910 |NV SMX2 V100-32G|
| Upload date| 2021-03-16 |2021-08-23|
| MindSpore version| 1.1.1 |1.4.0|
| Datasets| PASCAL VOC2012 + SBD | PASCAL VOC2012 + SBD |
| Training parameters| epoch = 300, batch_size = 32 (s16_r1)  epoch = 800, batch_size = 16 (s8_r1)  epoch = 300, batch_size = 16 (s8_r2) |epoch = 300, batch_size = 16 (s16_r1)  epoch = 800, batch_size = 8 (s8_r1)  epoch = 300, batch_size = 8 (s8_r2) |
| Optimizer| Momentum | Momentum |
| Loss function| Softmax cross entropy|Softmax cross entropy|
| Output| Probability|Probability|
| Loss| 0.0041095633 |0.003395824|
| Performance| 187736.386 ms (single device, s16)<br>  44474.187 ms (8 devices, s16)|  1080 ms/step (single device, s16)|  
| Finetuned checkpoint| 453 MB (.ckpt)| 454 MB (.ckpt)|
| Script| [Link](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabV3P)|[Link](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabV3P)|

#### Running ONNX evaluation

First, export your model:

```shell
python export.py \
  --checkpoint /path/to/checkpoint.ckpt \
  --filename /path/to/exported.onnx \
  --model [deeplab_v3_s16 or deeplab_v3_s8]  
```

Next, run evaluation:

```shell
python eval_onnx.py \
  --file_name /path/to/exported.onnx
  --data_root /path/to/VOC2012/
  --data_lst /path/to/VOC2012/voc_val_lst.txt \
  --device_target GPU \
  --batch_size [batch size]

or

bash run_eval_onnx.sh [DATA_ROOT] [DATA_LST] [FILE_NAME]
```

# Random Seed Description

The seed in the create_dataset function is set in **dataset.py**, and the random seed in **train.py** is used.

# ModelZoo Home Page

 For details, please go to the [official website](https://gitee.com/mindspore/models).
