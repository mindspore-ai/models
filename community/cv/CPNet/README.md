# Contents

- [CPNet Description](#CPNet-description)
- [Model Architecture](#CPNet-Architeture)
- [Dataset](#CPNet-Dataset)
- [Environmental Requirements](#Environmental-Requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Model](#model)
        - [Optimizer](#optimizer)
        - [Training](#training)
    - [Training Process](#training-process)
        - [Pre-training](#pre-training)
        - [Training](#training)
        - [Training Results](#training-results)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
        - [Evaluation Result](#evaluation-result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
    - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CPNet Description](#Contents)

CPNet directly supervise the feature aggregation to distinguish the intra-class and inter-class context clearly  to achieve more accurate segmentation results.

[paper](https://arxiv.org/abs/2004.01547) from CVPR2020
Changqian Yu, Jingbo Wang, Changxin Gao, Gang Yu, Chunhua Shen, Nong Sang. "Context Prior for Scene Segmentation"

# [Model Architecture](#Contents)

CPNet develop a Context Prior with the supervision of the Affinity Loss. Given an input image and corresponding ground truth, Affinity Loss constructs an ideal affinity map to supervise the learning of Context Prior. The learned Context Prior extracts the pixels belonging to the same category, while the reversed prior focuses on the pixels of different classes. Embedded into a conventional deep CNN, the proposed Context Prior Layer can selectively capture the intra-class and inter-class contextual dependencies, leading to robust feature representation.

# [Dataset](#Content)

- [PASCAL VOC 2012 and SBD Dataset Website](http://home.bharathh.info/pubs/codes/SBD/download.html)
 - It contains 11,357 finely annotated images split into training and testing sets with 8,498 and 2,857 images respectively.  - After download dataset, you can run CPNet\src\tools\get_dataset_list.py to generate dataset list.

# [Environmental requirements](#Contents)

- Hardware :(GPU)
    - Prepare GPU processor to build hardware environment
- frame:
    - [Mindspore](https://www.mindspore.cn/install)
- For details, please refer to the following resources:
    - [MindSpore course](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on GPU

```bash
# run training example
# need to complete  CPNet/voc2012_cpnet50_gpu.yaml
python train.py > train.log

# run evaluation example
# need set config_path in config.py file and set data_path, checkpoint_file_path in yaml file
python eval.py > eval.log
```

# [Scription Description](#Content)

## Script and Sample Code

```python
.
└─CPNet
├── config                                     # the training config file
│   ├── voc2012_cpnet50.yaml
│   ├── voc2012_colors.txt
│   └── voc2012_names.txt
├── src                                        # CPNet
│   ├── dataset                          # data processing
│   │   ├── dataset.py
│   │   ├── pt_dataset.py
│   │   └── pt_transform.py
│   ├── model                            # models for training and test
│   │   ├── affinity_loss.py             # loss function
│   │   ├── all_loss.py                  # loss function
│   │   ├── aux_ce_loss.py               # loss function
│   │   ├── cpnet.py
│   │   ├── resnet.py
│   │   ├── stdresnet.py
│   │   └── utils.py                     # network utils
│   ├── model_utils
│   │   ├── config.py                    # Automatically get the yaml format config file in the CPNet/config/
│   ├── tools                            # get dataset list
│   │   ├── get_dataset_list.py
│   └── utils
│       ├── functions_args.py                  # test helper
│       ├── lr.py                              # learning rate
│       ├── metric_and_evalcallback.py         # evalcallback
│       ├── metrics.py                         # loss function helper
│       └── p_util.py                          # some functions
├── eval.py                                    # Evaluation python file for VOC2012
├── train.py                                   # The training python file for VOC2012
└── README.md                                  # descriptions about CPNet
```

## Script Parameters

Set script parameters in ./config/voc2012_cpnet50_gpu.yaml

### Model

```text
name: "CPNet"
backbone: "resnet50"
base_size: 512   # based size for scaling
crop_size: 473
```

### Optimizer

```text
init_lr: 0.005
momentum: 0.9
weight_decay: 0.0001
```

### Training

```text
batch_size: 6    # batch size for training
batch_size_val: 6  # batch size for validation during training\
epochs: 200 # voc2012
save_checkpoint_epochs: 20
```

## Training Process

### Training

python train.py >train.log

### Training Result

The training results will be saved in the CPNet path, you can view the log in the CPNet/train.log

```text
# training result
epoch: 200 step: 244, loss is 0.3580215275287628
epoch time: 128350.305 ms, per step time: 526.026 ms
```

## Evaluation Process

### Evaluation

Check the checkpoint path in config/voc2012_cpnet50_gpu.yaml used for evaluation before running the following command.

python eval.py >eval.log

### Evaluation Result

The results at eval.log were as follows:

```log
Eval result: mIoU/mAcc/allAcc 0.7052/0.7712/0.9347
````

# [Model Description](#Content)

## Performance

### Distributed Training Performance

|Parameter              | CPNet                                                   |
| ------------------- | --------------------------------------------------------- |
|resources              | GPU:NVIDIA V100; CPU: 3.0GHz; memory:64GB        |
|Upload date            |2022.8.16                    |
|mindspore version      |mindspore1.5.0     |
|training parameter     |epoch=200,batch_size=6   |
|optimizer              |SGD optimizer，momentum=0.9,weight_decay=0.0001    |
|loss function          |AffinityLoss,SoftmaxCrossEntropyLoss   |
|training speed         |epoch time: 128350.305 ms, per step time: 526.026 ms|
|total time             |10h58m26s(1pcs)    |
|Script URL             |https://gitee.com/mindspore/models/tree/master/community/cv/CPNet|
|Random number seed     |set_seed = 1234     |

## Inference Performance

| Parameters          | GPU                     |
| ------------------- | --------------------------- |
| Model Version       | CPNet                |
| Resource            | GPU:NVIDIA V100; CPU: 3.0GHz; memory:64GB          |
| Uploaded Date       | 8/16/2022 (month/day/year) |
| MindSpore Version   | 1.5.0                 |
| Dataset             | voc2012    |
| outputs             | mIoU/mAcc/allAcc                |
| Accuracy            | 0.7052/0.7712/0.9347 |

# [Description of Random Situation](#Content)

The random seed in `train.py`.

# [ModelZoo Homepage](#Content)

Please visit the official website [homepage](https://gitee.com/mindspore/models).