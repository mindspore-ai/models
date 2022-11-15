# Contents

- [Contents](#contents)
- [DeepLabV3 Description](#deeplabv3-description)
    - [Description](#description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage)
            - [Running on Ascend](#running-on-ascend)
            - [Running on CPU](#running-on-cpu)
        - [Result](#result)
            - [Running on Ascend](#running-on-ascend-1)
            - [Running on CPU](#running-on-cpu-1)
            - [Transfer Training](#transfer-training)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Running on Ascend](#running-on-ascend-2)
        - [Result](#result-1)
            - [Training accuracy](#training-accuracy)
    - [Export MindIR](#export-mindir)
    - [Inference Process](#inference-process)
        - [Usage](#usage-2)
        - [result](#result-2)
    - [Post Training Quantization](#post-training-quantization)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
    - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DeepLabV3 Description](#contents)

## Description

DeepLab is a series of image semantic segmentation models, DeepLabV3 improves significantly over previous versions. Two keypoints of DeepLabV3: Its multi-grid atrous convolution makes it better to deal with segmenting objects at multiple scales, and augmented ASPP makes image-level features available to capture long range information.
This repository provides a script and recipe to DeepLabV3 model and achieve state-of-the-art performance.

Refer to [this paper][1] for network details.
`Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.`

[1]: https://arxiv.org/abs/1706.05587

# [Model Architecture](#contents)

Res2net101 as backbone, atrous convolution for dense feature extraction.

# [Dataset](#contents)

Pascal VOC datasets and Semantic Boundaries Dataset

- Download segmentation dataset.

- Prepare the training data list file. The list file saves the relative path to image and annotation pairs. Lines are like:

```shell
JPEGImages/00001.jpg SegmentationClassGray/00001.png
JPEGImages/00002.jpg SegmentationClassGray/00002.png
JPEGImages/00003.jpg SegmentationClassGray/00003.png
JPEGImages/00004.jpg SegmentationClassGray/00004.png
......
```

You can also generate the list file automatically by run script: `python get_dataset_lst.py --data_root=/PATH/TO/DATA`

- Configure and run build_data.sh to convert dataset to mindrecords. Arguments in scripts/build_data.sh:

 ```shell
 --data_root                 root path of training data
 --data_lst                  list of training data(prepared above)
 --dst_path                  where mindrecords are saved
 --num_shards                number of shards of the mindrecords
 --shuffle                   shuffle or not
 ```

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend）
- Prepare hardware environment with Ascend.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- Install python packages in requirements.txt
- Generate config json file for 8pcs training

     ```bash
     # From the root of this project
     cd src/tools/
     python3 get_multicards_json.py 10.111.*.*
     # 10.111.*.* is the computer's ip address.
     ```

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Prepare backbone

Download res2net101 pretrained ckpt.

- Running on Ascend

Based on original DeepLabV3 paper, we reproduce two training experiments on vocaug (also as trainaug) dataset and evaluate on voc val dataset.

For single device training, please config parameters, training script is:

```shell
Enter the shell script to modify the data_file and ckpt_pre_trained parameters
# example:
data_file=/home/DataSet/VOC2012/vocaug_mindrecords/vocaug.mindrecord0
ckpt_pre_trained=/home/model/deeplabv3/predtrained/res2net101_ascend.ckpt

bash run_standalone_train.sh
```

- For 8 devices training, training steps are as follows:

1. Train s16 with vocaug dataset, finetuning from res2net101 pretrained model, script is:

```shell
Enter the shell script to modify the data_file and ckpt_pre_trained parameters
# example:
data_file=/home/DataSet/VOC2012/vocaug_mindrecords/vocaug.mindrecord0
ckpt_pre_trained=/home/model/deeplabv3/predtrained/deeplab_v3_s16-300_41.ckpt

bash run_distribute_train_s16_r1.sh ~/hccl_8p.json
```

2. Train s8 with vocaug dataset, finetuning from model in previous step, training script is:

```shell
Enter the shell script to modify the data_file and ckpt_pre_trained parameters
# example:
data_file=/home/DataSet/VOC2012/vocaug_mindrecords/vocaug.mindrecord0
ckpt_pre_trained=/home/model/deeplabv3/predtrained/deeplab_v3_s8-300_11.ckpt

bash run_distribute_train_s8_r1.sh ~/hccl_8p.json
```

3. Train s8 with voctrain dataset, finetuning from model in previous step, training script is:

```shell
Enter the shell script to modify the data_file and ckpt_pre_trained parameters
Note: This training pre-training weight uses the weight file of the previous training, and the data set has also changed
# example:
data_file=/home/DataSet/VOC2012/voctrain_mindrecords/votrain.mindrecord0
ckpt_pre_trained=/home/model/deeplabv3/ckpt/deeplabv3-800_330.ckpt


bash run_distribute_train_s8_r2.sh ~/hccl_8p.json
```

- For evaluation, evaluating steps are as follows:

1. Enter the shell script to modify the data_file and ckpt_pre_trained parameters

```shell
modify the parameter according local path
# example:
data_root=/home/DataSet/VOC2012
data_lst=/home/DataSet/VOC2012/voc_val_lst.txt
ckpt_path=/home/model/deeplabv3/ckpt/deeplabv3-800_330.ckpt
```

2. Eval s16 with voc val dataset, eval script is:

```shell
# Modify the parameter ckpt_path to the path where deeplab_v3_s16-300_41.ckpt is located.
bash run_eval_s16.sh
```

3. Eval s8 with voc val dataset, eval script is:

```shell
# Modify the parameter ckpt_path to the path where deeplab_v3_s8-300_11.ckpt is located.
bash run_eval_s8.sh
```

4. Eval s8 multiscale with voc val dataset, eval script is:

```shell
# Modify the parameter ckpt_path to the path where deeplab_v3_s8-300_11.ckpt is located.
bash run_eval_s8_multiscale.sh
```

5. Eval s8 multiscale and flip with voc val dataset, eval script is:

```shell
# Modify the parameter ckpt_path to the path where deeplab_v3_s8-300_11.ckpt is located.
bash run_eval_s8_multiscale_flip.sh
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└──deeplabv3
  ├── README.md
  ├── scripts
    ├── build_data.sh                             # convert raw data to mindrecord dataset
    ├── run_distribute_train_s16_r1.sh            # launch ascend distributed training(8 pcs) with vocaug dataset in s16 structure
    ├── run_distribute_train_s8_r1.sh             # launch ascend distributed training(8 pcs) with vocaug dataset in s8 structure
    ├── run_distribute_train_s8_r2.sh             # launch ascend distributed training(8 pcs) with voctrain dataset in s8 structure
    ├── run_eval_s16.sh                           # launch ascend evaluation in s16 structure
    ├── run_eval_s8.sh                            # launch ascend evaluation in s8 structure
    ├── run_eval_s8_multiscale.sh                 # launch ascend evaluation with multiscale in s8 structure
    ├── run_eval_s8_multiscale_filp.sh            # launch ascend evaluation with multiscale and filp in s8 structure
    ├── run_standalone_train.sh                   # launch ascend standalone training(1 pc)
    ├── run_standalone_train_cpu.sh               # launch CPU standalone training
  ├── src
    ├── data
        ├── dataset.py                            # mindrecord data generator
        ├── build_seg_data.py                     # data preprocessing
        ├── get_dataset_lst.py                    # dataset list file generator
    ├── loss
       ├── loss.py                                # loss definition for deeplabv3
    ├── nets
       ├── deeplab_v3
          ├── deeplab_v3.py                       # DeepLabV3 network structure
       ├── net_factory.py                         # set S16 and S8 structures
    ├── tools
       ├── get_multicards_json.py                 # get rank table file
    └── utils
       └── learning_rates.py                      # generate learning rate
    └── convert_checkpoint.py                     # convert res2net101 as backbone
  ├── eval.py                                     # eval net
  ├── train.py                                    # train net
  └── requirements.txt                            # requirements file
```

## [Script Parameters](#contents)

Default configuration

```shell
"data_file":"/PATH/TO/MINDRECORD_NAME"            # dataset path
"device_target":Ascend                            # device target
"train_epochs":300                                # total epochs
"batch_size":32                                   # batch size of input tensor
"crop_size":513                                   # crop size
"base_lr":0.08                                    # initial learning rate
"lr_type":cos                                     # decay mode for generating learning rate
"min_scale":0.5                                   # minimum scale of data argumentation
"max_scale":2.0                                   # maximum scale of data argumentation
"ignore_label":255                                # ignore label
"num_classes":21                                  # number of classes
"model":deeplab_v3_s16                            # select model
"ckpt_pre_trained":"/PATH/TO/PRETRAIN_MODEL"      # path to load pretrain checkpoint
"is_distributed":                                 # distributed training, it will be True if the parameter is set
"save_steps":410                                  # steps interval for saving
"keep_checkpoint_max":200                         # max checkpoint for saving
```

## [Training Process](#contents)

### Usage

#### Running on Ascend

Based on original DeepLabV3 paper, we reproduce two training experiments on vocaug (also as trainaug) dataset and evaluate on voc val dataset.

For single device training, please config parameters, training script is as follows:

```shell
# run_standalone_train.sh
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
                    --model=deeplab_v3_s16  \
                    --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                    --save_steps=1500  \
                    --keep_checkpoint_max=200 >log 2>&1 &
```

For 8 devices training, training steps are as follows:

1. Train s16 with vocaug dataset, finetuning from resnet101 pretrained model, script is as follows:

    ```shell
    # run_distribute_train_s16_r1.sh
    for((i=0;i<=$RANK_SIZE-1;i++));
    do
        export RANK_ID=${i}
        export DEVICE_ID=$((i + RANK_START_ID))
        echo 'start rank='${i}', device id='${DEVICE_ID}'...'
        mkdir ${train_path}/device${DEVICE_ID}
        cd ${train_path}/device${DEVICE_ID} || exit
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
                                                --model=deeplab_v3_s16  \
                                                --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                                --is_distributed  \
                                                --save_steps=410  \
                                                --keep_checkpoint_max=200 >log 2>&1 &
    done
    ```

2. Train s8 with vocaug dataset, finetuning from model in previous step, training script is as follows:

    ```shell
    # run_distribute_train_s8_r1.sh
    for((i=0;i<=$RANK_SIZE-1;i++));
    do
        export RANK_ID=${i}
        export DEVICE_ID=$((i + RANK_START_ID))
        echo 'start rank='${i}', device id='${DEVICE_ID}'...'
        mkdir ${train_path}/device${DEVICE_ID}
        cd ${train_path}/device${DEVICE_ID} || exit
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
                                                --model=deeplab_v3_s8  \
                                                --loss_scale=2048  \
                                                --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                                --is_distributed  \
                                                --save_steps=820  \
                                                --keep_checkpoint_max=200 >log 2>&1 &
    done
    ```

3. Train s8 with voctrain dataset, finetuning from model in previous step, training script is as follows:

```shell
# run_distribute_train_s8_r2.sh
for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=${i}
    export DEVICE_ID=$((i + RANK_START_ID))
    echo 'start rank='${i}', device id='${DEVICE_ID}'...'
    mkdir ${train_path}/device${DEVICE_ID}
    cd ${train_path}/device${DEVICE_ID} || exit
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
                                               --model=deeplab_v3_s8  \
                                               --loss_scale=2048  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=110  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
```

#### Running on CPU

For CPU training, please config parameters, training script is as follows:

```shell
# run_standalone_train_cpu.sh
python ${train_code_path}/train.py --data_file=/PATH/TO/MINDRECORD_NAME  \
                    --device_target=CPU  \
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
                    --model=deeplab_v3_s16  \
                    --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                    --save_steps=1500  \
                    --keep_checkpoint_max=200 >log 2>&1 &
```

### Result

#### Running on Ascend

- Training vocaug in s16 structure

```shell
# distribute training result(8p)
[ModelZoo-compile_time:0:06:39.982534]
epoch: 1 step: 41, loss is 0.6358908
epoch time: 436009.555 ms, per step time: 10634.379 ms
[ModelZoo-compile_time:0:00:00.000175]
epoch: 2 step: 41, loss is 0.4275353
epoch time: 24787.156 ms, per step time: 604.565 ms
[ModelZoo-compile_time:0:00:00.000162]
epoch: 3 step: 41, loss is 0.33092758
epoch time: 24899.430 ms, per step time: 607.303 ms
[ModelZoo-compile_time:0:00:00.000153]
epoch: 4 step: 41, loss is 0.337021
epoch time: 24805.650 ms, per step time: 605.016 ms
[ModelZoo-compile_time:0:00:00.000102]
epoch: 5 step: 41, loss is 0.31007352
epoch time: 24783.147 ms, per step time: 604.467 ms
[ModelZoo-compile_time:0:00:00.000098]
epoch: 6 step: 41, loss is 0.29517695
epoch time: 24788.169 ms, per step time: 604.589 ms
[ModelZoo-compile_time:0:00:00.000126]
epoch: 7 step: 41, loss is 0.27527434
epoch time: 24783.927 ms, per step time: 604.486 ms
[ModelZoo-compile_time:0:00:00.000099]
epoch: 8 step: 41, loss is 0.24326736
epoch time: 24785.070 ms, per step time: 604.514 ms
[ModelZoo-compile_time:0:00:00.000095]
```

- Training vocaug in s8 structure

```shell
# distribute training result(8p)
[ModelZoo-compile_time:0:06:45.775471]
epoch: 1 step: 82, loss is 0.054066252
epoch time: 478000.375 ms, per step time: 5829.273 ms
[ModelZoo-compile_time:0:00:00.000133]
epoch: 2 step: 82, loss is 0.049781486
epoch time: 61610.573 ms, per step time: 751.348 ms
[ModelZoo-compile_time:0:00:00.000146]
epoch: 3 step: 82, loss is 0.05366816
epoch time: 61565.675 ms, per step time: 750.801 ms
[ModelZoo-compile_time:0:00:00.000686]
epoch: 4 step: 82, loss is 0.09842333
epoch time: 61697.758 ms, per step time: 752.412 ms
[ModelZoo-compile_time:0:00:00.000136]
epoch: 5 step: 82, loss is 0.050434023
epoch time: 61864.941 ms, per step time: 754.451 ms
[ModelZoo-compile_time:0:00:00.000813]
epoch: 6 step: 82, loss is 0.054736223
epoch time: 61603.703 ms, per step time: 751.265 ms
[ModelZoo-compile_time:0:00:00.000188]
epoch: 7 step: 82, loss is 0.03442829
epoch time: 61762.904 ms, per step time: 753.206 ms
[ModelZoo-compile_time:0:00:00.000172]
```

- Training voctrain in s8 structure

```shell
# distribute training result(8p)
[ModelZoo-compile_time:0:07:00.800941]
epoch: 1 step: 11, loss is 0.0031219586
epoch time: 439271.504 ms, per step time: 39933.773 ms
[ModelZoo-compile_time:0:00:00.000189]
epoch: 2 step: 11, loss is 0.006192135
epoch time: 8308.671 ms, per step time: 755.334 ms
[ModelZoo-compile_time:0:00:00.000110]
epoch: 3 step: 11, loss is 0.00744328
epoch time: 8397.222 ms, per step time: 763.384 ms
[ModelZoo-compile_time:0:00:00.000180]
epoch: 4 step: 11, loss is 0.00378249
epoch time: 8284.866 ms, per step time: 753.170 ms
[ModelZoo-compile_time:0:00:00.000106]
epoch: 5 step: 11, loss is 0.008109064
epoch time: 8291.124 ms, per step time: 753.739 ms
[ModelZoo-compile_time:0:00:00.000113]
epoch: 6 step: 11, loss is 0.008020833
epoch time: 8321.096 ms, per step time: 756.463 ms
[ModelZoo-compile_time:0:00:00.000667]
epoch: 7 step: 11, loss is 0.008300599
epoch time: 8454.503 ms, per step time: 768.591 ms
[ModelZoo-compile_time:0:00:00.000208]
epoch: 8 step: 11, loss is 0.005244185
epoch time: 8288.644 ms, per step time: 753.513 ms
[ModelZoo-compile_time:0:00:00.000085]
```

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

Configure checkpoint with --ckpt_path and dataset path. Then run script, mIOU will be printed in eval_path/eval_log.

```shell
./run_eval_s16.sh                     # test s16
./run_eval_s8.sh                      # test s8
./run_eval_s8_multiscale.sh           # test s8 + multiscale
./run_eval_s8_multiscale_flip.sh      # test s8 + multiscale + flip
```

Example of test script is as follows:

```shell
python ${train_code_path}/eval.py --data_root=/PATH/TO/DATA  \
                    --data_lst=/PATH/TO/DATA_lst.txt  \
                    --batch_size=16  \
                    --crop_size=513  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s8  \
                    --scales=0.5  \
                    --scales=0.75  \
                    --scales=1.0  \
                    --scales=1.25  \
                    --scales=1.75  \
                    --flip  \
                    --freeze_bn  \
                    --ckpt_path=/PATH/TO/PRETRAIN_MODEL >${eval_path}/eval_log 2>&1 &
```

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy

| **Network**    | OS=16 | OS=8 | MS   | Flip  | mIOU  |
| :----------: | :-----: | :----: | :----: | :-----: | :-----: |
| deeplab_v3 | √     |      |      |       | 78.90 |
| deeplab_v3 |       | √    |      |       | 80.04 |
| deeplab_v3 |       | √    | √    |       | 80.34 |
| deeplab_v3 |       | √    | √    | √     | 80.48 |

Note: There OS is output stride, and MS is multiscale.

# Export MindIR

```python
python export.py --ckpt_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --model [MODEL] --freeze_bn [MODE]
- The `ckpt_file` parameter is required
- The `file_format` should be in ["AIR", "MINDIR"]
- The `model` should be in ["deeplab_v3_s8", "deeplab_v3_s16"]
- The `freeze_bn` should be in [True, False]
```

# Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

## Usage

Before performing inference, the air file must be exported by `export.py`. Current batch_Size can only be set to 1. The precision calculation process needs about 70G+ memory space.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DATA_ROOT] [DATA_LIST] [DEVICE_ID]
```

DEVICE_ID is optional, default value is 0.

## result

Inference result is saved in current path, you can find result in acc.log file.

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend 910
| -------------------------- | -------------------------------------- |
| Model Version              | DeepLabV3
| Resource                   | Ascend 910; OS Euler2.8 |
| Uploaded Date              | 12/20/2021 (month/day/year)          |
| MindSpore Version          | 1.5.0                       |
| Dataset                    | PASCAL VOC2012 + SBD              |
| Training Parameters        | epoch = 300, batch_size = 32 (s16_r1) <br> epoch = 800, batch_size = 16 (s8_r1)   <br>    epoch = 300, batch_size = 16 (s8_r2) |
| Optimizer                  | Momentum                                 |
| Loss Function              | Softmax Cross Entropy                                  |
| Outputs                    | probability                                       |
| Loss                       | 0.006                                       |
| Speed                      | 606ms/step（8pcs, s16） <br> 752ms/step (8pcs, s8)      |  

