# Contents

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
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
    - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [PSPNet Description](#Contents)

PSPNet(Pyramid Scene Parsing Network) has great capability of global context information by different-region based context aggregation through the pyramid pooling module together.

[paper](https://arxiv.org/abs/1612.01105) from CVPR2017

# [Model Architecture](#Contents)

The pyramid pooling module fuses features under four different pyramid scales.For maintaining a reasonable gap in representation，the module is a four-level one with bin sizes of 1×1, 2×2, 3×3 and 6×6 respectively.

# [Dataset](#Content)

- [PASCAL VOC 2012 and SBD Dataset Website](http://home.bharathh.info/pubs/codes/SBD/download.html)
 - It contains 11,357 finely annotated images split into training and testing sets with 8,498 and 2,857 images respectively.
- [ADE20K Dataset Website](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
 - It contains 22,210 finely annotated images split into training and testing sets with 20,210 and 2,000 images respectively.

# [Environmental requirements](#Contents)

- Hardware :(Ascend)
    - Prepare ascend processor to build hardware environment
- frame:
    - [Mindspore](https://www.mindspore.cn/install)
- For details, please refer to the following resources:
    - [MindSpore course](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# [Scription Description](#Content)

## Script and Sample Code

```python
.
└─PSPNet
├── ascend310_infer
├── eval.py                                    # Evaluation python file for ADE20K/VOC2012
├── export.py                                  # export mindir
├── README.md                                  # descriptions about PSPnet
├── src                                        # PSPNet
│   ├── config                           # the training config file
│   │   ├── ade20k_pspnet50.yaml
│   │   └── voc2012_pspnet50.yaml
│   ├── dataset                          # data processing
│   │   ├── dataset.py
│   │   └── transform.py
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
│   └── run_eval.sh                            # validation script
└── train.py                                         # The training python file for ADE20K/VOC2012
```

## Script Parameters

Set script parameters in src/config/ade20k_pspnet50.yaml and src/config/voc2012_pspnet50.yaml

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
voc_root: "./data/voc/voc"
epochs: 100/50 # ade/voc2012
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

### Training Result

The training results will be saved in the PSPNet path, you can view the log in the ./LOG/log.txt

```bash
# training result(1p)-voc2012
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

```shell
    bash run_eval.sh [YAML_PATH] [DEVICE_ID]
```

### Evaluation Result

The results at eval.log were as follows:

```bash
ADE20K:mIoU/mAcc/allAcc 0.4164/0.5319/0.7996.
VOC2012:mIoU/mAcc/allAcc 0.7380/0.8229/0.9293.
````

## [Export MindIR](#contents)

```shell
python export.py --yaml_path [YAML_PTAH] --ckpt_file [CKPT_PATH]
```

The ckpt_file parameter is required,

## 310 infer

- Note: Before executing 310 infer, create the MINDIR/AIR model using "python export.py --ckpt_file [The path of the CKPT for exporting] --config [The yaml file]".

```shell
    bash run_infer_310.sh [MINDIR PTAH [YAML PTAH] [DATA PATH] [DEVICE ID]
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
|training speed         |epoch time: 493974.632 ms, per step time: 464.699 ms(1p for voc2012)|
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
| Dataset             | voc2012/ade20k    |
| outputs             | Miou/Acc                 |
| Accuracy            | 0.4164/0.7996.(ade20k) 0.7380/0.9293(voc2012) |

# [Description of Random Situation](#Content)

The random seed in `train.py`.

# [ModelZoo Homepage](#Content)

Please visit the official website [homepage](https://gitee.com/mindspore/models).
