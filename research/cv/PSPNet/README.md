# Contents

- [Contents](#contents)
- [PSPNet Description](#PSPNet-description)
- [Model Architecture](#PSPNet-Architeture)
- [Dataset](#PSPNet-Dataset)
- [Environmental Requirements](#Environmental)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Pre-training](#pre-training)
        - [Training](#training)
        - [Training Results](#training-results)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
        - [Evaluation Result](#evaluation-resul)
    - [Export MindIR](#export-mindir)
    - [310 infer](#310-inference)
    - [ONNX CPU infer](#onnx-cpu-infer)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
    - [Distributed Training Performance](#distributed-training-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [PSPNet Description](#Contents)

PSPNet(Pyramid Scene Parsing Network) has great capability of global context information by different-region based context aggregation through the pyramid pooling module together.

[paper](https://arxiv.org/abs/1612.01105) from CVPR2017

# [Model Architecture](#Contents)

The pyramid pooling module fuses features under four different pyramid scales.For maintaining a reasonable gap in representation，the module is a four-level one with bin sizes of 1×1, 2×2, 3×3 and 6×6 respectively.

# [Dataset](#Content)

- [Semantic Boundaries Dataset](http://home.bharathh.info/pubs/codes/SBD/download.html)

 - It contains 11,355 finely annotated images split into training and testing sets with 8,498 and 2,857 images respectively.
 - The VOC2012-SBD is a scaling of the VOC2011 PASCAL segmentation subset to the full dataset, an increase by a factor of 5 in the number of images and objects.
 - The VOC2012-SBD directory structure is as follows:

    ```text
        ├── SBD
            ├─ cls
            │   ├─ 2008_00****.mat
            │   ├─ ...  
            │   └─ 2011_00****.mat
            ├─ img
            │   ├─ 2008_00****.jpg
            │   ├─ ...
            │   └─ 2011_00****.jpg
            ├─ inst
            │   ├─ 2008_00****.mat
            │   ├─ ...
            │   └─ 2011_00****.mat
            ├─ train.txt
            └─ val.txt
    ```

 - The path formats in train.txt and val.txt are partial. And the mat file in the cls needs to be converted to image. You can run preprocess_dataset.py to convert the mat file and generate train_list.txt and val_list.txt. As follow：

 ```python
 python src/dataset/preprocess_dataset.py --data_dir [DATA_DIR]
 ```

- [ADE20K Dataset Website](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
 - It contains 22,210 finely annotated images split into training and testing sets with 20,210 and 2,000 images respectively.
 - The ADE20k directory structure is as follows:

     ```text
         ├── ADE
             ├── annotations
                 ├─ training
                 │   ├─ADE_train_***.png
                 │   ├─ ...
                 │   └─ADE_train_***.png
                 └─ validation
                     ├─ADE_val_***.png
                     ├─ ...
                     └─ADE_val_***.png
             ├── images
                 ├─ training
                 │   ├─ADE_train_***.jpg
                 │   ├─ ...
                 │   └─ADE_train_***.jpg
                 └─ validation
                     ├─ADE_val_***.jpg
                     ├─ ...
                     └─ADE_val_***.jpg
     ```

 - After download dataset, you can run create_data_txt.py to generate train_list.txt and val_list.txt for ADE20K as follows:

 ```bash
  python src/dataset/create_data_txt.py --data_root [DATA_ROOT] --image_prefix [IMAGE_PREFIX] --mask_prefix [MASK_PREFIX] --output_txt [OUTPUT_TXT]
  example:
  python src/dataset/create_data_txt.py --data_root /root/ADE/ --image_prefix images/training --mask_prefix annotations/training --output_txt /root/ADE/training_list.txt
 ```

Datasets: attributes (names and colors) are needed, and please download as follows:

- [PASCAL VOC 2012 names.txt and colors.txt Website](https://github.com/hszhao/semseg/tree/master/data/voc2012)
- [ADE20K names.txt and colors.txt Website](https://github.com/hszhao/semseg/tree/master/data/ade20k)

VOC2012-SBD has the same categories as VOC2012 and therefore the same attributes.

# [Pretrained model](#contents)

[resnet50-imagenet pretrained model](https://download.mindspore.cn/thirdparty/pspnet/resnet_deepbase.ckpt)

# [Environmental requirements](#Contents)

- Hardware :(Ascend)
    - Prepare ascend processor to build hardware environment
- frame:
    - [Mindspore](https://www.mindspore.cn/install)
- For details, please refer to the following resources:
    - [MindSpore course](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [Scription Description](#Content)

## Script and Sample Code

```python
.
└─PSPNet
├── ascend310_infer
├── eval.py                                    # Evaluation python file for ADE20K/VOC2012-SBD
├── export.py                                  # export mindir
├── README.md                                  # descriptions about PSPnet
├── config                                     # the training config file
│   ├── ade20k_pspnet50.yaml
│   └── voc2012_pspnet50.yaml
├── src                                        # PSPNet
│   ├── dataset                          # data processing
│   │   ├── pt_dataset.py
│   │   ├── create_data_txt.py           # generate train_list.txt and val_list.txt
│   │   ├── create_voc_list.py           # generate train_list.txt and val_list.txt
│   │   └── pt_transform.py
│   ├── model                            # models for training and test
│   │   ├── PSPNet.py
│   │   ├── resnet.py
│   │   └── cell.py                # loss function
│   └── utils
│       ├── functions_args.py                  # test helper
│       ├── lr.py                              # learning rate
│       ├── metric_and_evalcallback.py         # evalcallback
│       ├── aux_loss.py                        # loss function helper
│       └── p_util.py                          # some functions
│
├── scripts
│   ├── run_distribute_train_ascend.sh         # multi cards distributed training in ascend
│   ├── run_train1p_ascend.sh                  # 1P training in ascend
│   ├── run_infer_310.sh                       # 310 infer
│   ├── run_eval_onnx_cpu.sh                   # ONNX infer
│   └── run_eval.sh                            # validation script
└── train.py                                         # The training python file for ADE20K/VOC2012-SBD
```

## Script Parameters

Set script parameters in ./config/ade20k_pspnet50.yaml and ./config/voc2012_pspnet50.yaml

### Model

```bash
name: "PSPNet"
backbone: "resnet50_v2"
base_size: 512   # based size for scaling
crop_size: 473
```

### Optimizer

```bash
init_lr: 0.005
momentum: 0.9
weight_decay: 0.0001
```

### Training

```bash
batch_size: 8    # batch size for training
batch_size_val: 8  # batch size for validation during training
ade_root: "./data/ADE/" # set dataset path
voc_root: "./data/SBD/"
epochs: 100/50 # ade/voc2012-sbd
pretrained_model_path: "./data/resnet_deepbase.ckpt"  
save_checkpoint_epochs: 10
keep_checkpoint_max: 10
```

## Training Process

### Training

- Train on a single card

```shell
    bash scripts/run_train1p_ascend.sh [YAML_PATH] [DEVICE_ID]
```

- Run distributed train in ascend processor environment

```shell
    bash scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [YAML_PATH]
```

- Train on CPU

You need to set 'device_target' in train.py to 'CPU' and run as follows:

```shell
    python3 train.py --config=[YAML_PATH] > train_log.txt 2>&1 &
```

### Training Result

The training results will be saved in the PSPNet path, you can view the log in the ./LOG/log.txt

```bash
# training result(1p)-voc2012-sbd
epoch: 1 step: 1063, loss is 0.62588865
epoch time: 493974.632 ms, per step time: 464.699 ms
epoch: 2 step: 1063, loss is 0.68774235
epoch time: 428786.495 ms, per step time: 403.374 ms
epoch: 3 step: 1063, loss is 0.4055968
epoch time: 428773.945 ms, per step time: 403.362 ms
epoch: 4 step: 1063, loss is 0.7540638
epoch time: 428783.473 ms, per step time: 403.371 ms
epoch: 5 step: 1063, loss is 0.49349666
epoch time: 428776.845 ms, per step time: 403.365 ms
```

## Evaluation Process

### Evaluation

Check the checkpoint path in config/ade20k_pspnet50.yaml and config/voc2012_pspnet50.yaml used for evaluation before running the following command.

#### Evaluation on gpu

```shell
    bash run_eval.sh [YAML_PATH] [DEVICE_ID]
```

#### Evaluation on cpu

You need to set 'device_target' to 'CPU' in eval.py, config/ade20k_pspnet50.yaml and config/voc2012_pspnet50.yaml before running the following command.

```shell
    python3 eval.py --config=[YAML_PATH] > eval_log.txt 2>&1 &
```

### Evaluation Result

The results at eval.log were as follows:

```bash
ADE20K:mIoU/mAcc/allAcc 0.4164/0.5319/0.7996.
VOC2012-SBD:mIoU/mAcc/allAcc 0.7380/0.8229/0.9293.
````

## [Export](#contents)

### Export MINDIR

```shell
python export.py --yaml_path [YAML_PTAH] --ckpt_file [CKPT_PATH]
```

### Export ONNX

```shell
python export.py --yaml_path [YAML_PTAH] --ckpt_file [CKPT_PATH] --file_format ONNX
```

The ckpt_file parameter and yaml_path are required.

## 310 infer

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

- Note: Before executing 310 infer, create the MINDIR/AIR model using "python export.py --ckpt_file [The path of the CKPT for exporting] --config [The yaml file]".

```shell
    bash run_infer_310.sh [MINDIR PTAH [YAML PTAH] [DATA PATH] [DEVICE ID]
```

## ONNX CPU infer

- Note: Before executing ONNX CPU infer, please export onnx model first.

```shell
    bash PSPNet/scripts/run_eval_onnx_cpu.sh PSPNet/config/voc2012_pspnet50.yaml
```

# [Model Description](#Content)

## Performance

### Distributed Training Performance

|Parameter              | PSPNet                                                   |
| ------------------- | --------------------------------------------------------- |
|resources              | Ascend 910；CPU 2.60GHz, 192core；memory：755G |
|Upload date            |2021.11.13                    |
|mindspore version      |mindspore1.3.0     |
|training parameter     |epoch=100,batch_size=8   |
|optimizer              |SGD optimizer，momentum=0.9,weight_decay=0.0001    |
|loss function          |SoftmaxCrossEntropyLoss   |
|training speed         |epoch time: 493974.632 ms, per step time: 464.699 ms(1p for voc2012-sbd), 485 ms(8p for voc2012-sbd), 998 ms(1p for ADE20K), 1050 ms(8p for ADE20K)|
|total time             |6h10m34s(1pcs)    |
|Script URL             |https://gitee.com/mindspore/models/tree/master/research/cv/PSPNet|
|Random number seed     |set_seed = 1234     |

## Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | PSPNet                |
| Resource            | Ascend 310; OS Euler2.8                   |
| Uploaded Date       | 12/22/2021 (month/day/year) |
| MindSpore Version   | 1.5.0                 |
| Dataset             | voc2012-sbd/ade20k    |
| outputs             | Miou/Acc                 |
| Accuracy            | 0.4164/0.7996.(ade20k) 0.7380/0.9293(voc2012-sbd) |

# [Description of Random Situation](#Content)

The random seed in `train.py`.

# [ModelZoo Homepage](#Content)

Please visit the official website [homepage](https://gitee.com/mindspore/models).
