# Contents

- [Contents](#contents)
    - [SSD Description](#ssd-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
        - [Prepare the model](#prepare-the-model)
        - [Run the scripts](#run-the-scripts)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
            - [Training on Ascend](#training-on-ascend)
            - [Training on GPU](#training-on-gpu)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation on Ascend](#evaluation-on-ascend)
            - [Evaluation on GPU](#evaluation-on-gpu)
        - [Inference Process](#inference-process)
            - [Export MindIR](#export-mindir)
            - [Infer](#infer)
            - [Result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
            - [Inference Performance](#inference-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [SSD Description](#contents)

SSD discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape.Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes.

[Paper](https://arxiv.org/abs/1512.02325):   Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.European Conference on Computer Vision (ECCV), 2016 (In press).

## [Model Architecture](#contents)

The SSD approach is based on a feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections. The early network layers are based on a standard architecture used for high quality image classification, which is called the base network. Then add auxiliary structure to the network to produce detections.

- **ssd320**, reference from the paper. Using mobilenetv2 as backbone and the same bbox predictor as the paper present.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2017](<http://images.cocodataset.org/>)

- Dataset size：19G
    - Train：18G，118000 images  
    - Val：1G，5000 images
    - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
    - Note：Data will be processed in dataset.py

## [Environment Requirements](#contents)

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2017.

- We use COCO2017 as training dataset in this example by default, and you can also use your own datasets.
  First, install Cython ,pycocotool and opencv to process data and to get evaluation result.

    ```shell
    pip install Cython

    pip install pycocotools

    pip install opencv-python
    ```

    1. If coco dataset is used. **Select dataset to coco when run script.**

        Change the `coco_root` and other settings you need in `src/config.py`. The directory structure is as follows:

        ```shell
        .
        └─coco_dataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017
        ```

    2. If VOC dataset is used. **Select dataset to voc when run script.**
        Change `classes`, `num_classes`, `voc_json` and `voc_root` in `src/config.py`. `voc_json` is the path of json file with coco format for evaluation, `voc_root` is the path of VOC dataset, the directory structure is as follows:

        ```shell
        .
        └─voc_dataset
          └─train
            ├─0001.jpg
            └─0001.xml
            ...
            ├─xxxx.jpg
            └─xxxx.xml
          └─eval
            ├─0001.jpg
            └─0001.xml
            ...
            ├─xxxx.jpg
            └─xxxx.xml
        ```

    3. If your own dataset is used. **Select dataset to other when run script.**
        Organize the dataset information into a TXT file, each row in the file is as follows:

        ```shell
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are setting in `src/config.py`.

## [Quick Start](#contents)

### Prepare the model

Change the dataset config in the config.

### Run the scripts

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

```shell
# distributed training on Ascend
bash scripts/run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE]

# run eval on Ascend
bash scripts/run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]

# run inference on Ascend 310
bash scripts/run_infer_310.sh [CHECKPOINT_PATH](MINDIR) [DATASET PATH] [ANNOTATIONS PATH] [DEVICE ID](OPTION)

```

- running on GPU

```shell
# distributed training on GPU
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [EPOCH_SIZE] [DATASET]  [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)

# run eval on GPU
bash scripts/run_eval_gpu.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]

```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└─ cv
  └─ ssd_mobilenetV2
    ├─ README.md                      # descriptions about SSD
    ├─ scripts
      ├─ run_distribute_train_gpu.sh      # shell script for distributed on gpu
      ├─ run_distribute_train.sh      # shell script for distributed on ascend
      ├─ run_standalone_train_gpu.sh              # shell script for 1p on gpu
      ├─ run_1p_train.sh              # shell script for 1p on ascend
      ├─ run_eval.sh                  # shell script for eval on ascend
      ├─ run_eval_gpu.sh                  # shell script for eval on gpu
      └─ run_infer_310.sh             # shell script for inference on ascend 310
    ├─ src
      ├─ __init__.py                  # init file
      ├─ box_utils.py                 # bbox utils
      ├─ anchor_generator.py          # generate anchors
      ├─ eval_utils.py                # metrics utils
      ├─ config.py                    # total config
      ├─ dataset.py                   # create dataset and process dataset
      ├─ init_params.py               # parameters utils
      ├─ lr_schedule.py               # learning ratio generator
      ├─ mobilenet_v2_fpn.py          # extract features
      └─ ssd.py                       # ssd architecture
    ├─ ascend310_infer                # 310 inference
      ├─ inc
        └─ utils.h
      ├─ src
        ├─main.cc
        └─utils.cc
      ├─build.sh
      └─CMakeLists.txt
    ├─ eval.py                        # eval scripts
    ├─ train.py                       # train scripts
    ├─ export.py                      # transform ckpt into MINDIR, AIR or ONNX
    ├─ postprocess.py                 # 310 inference
    └─ requirements.txt                # python environment
```

### [Script Parameters](#contents)

  ```shell
  Major parameters in train.py and config.py as follows:

    "device_num": 1                                  # Use device nums
    "lr": 0.05                                       # Learning rate init value
    "dataset": coco                                  # Dataset name
    "epoch_size": 500                                # Epoch size
    "batch_size": 32                                 # Batch size of input tensor
    "pre_trained": None                              # Pretrained checkpoint file path
    "pre_trained_epoch_size": 0                      # Pretrained epoch size
    "save_checkpoint_epochs": 10                     # The epoch interval between two checkpoints. By default, the checkpoint will be saved per 10 epochs
    "loss_scale": 1024                               # Loss scale
    "filter_weight": False                           # Load parameters in head layer or not. If the class numbers of train dataset is different from the class numbers in pre_trained checkpoint, please set True.
    "freeze_layer": "none"                           # Freeze the backbone parameters or not, support none and backbone.

    "class_num": 81                                  # Dataset class number
    "image_shape": [320, 320]                        # Image height and width used as input to the model
    "mindrecord_dir": "/data/MindRecord_COCO"        # MindRecord path
    "coco_root": "/data/coco2017"                    # COCO2017 dataset path
    "voc_root": "/data/voc_dataset"                  # VOC original dataset path
    "voc_json": "annotations/voc_instances_val.json" # is the path of json file with coco format for evaluation
    "image_dir": ""                                  # Other dataset image path, if coco or voc used, it will be useless
    "anno_path": ""                                  # Other dataset annotation path, if coco or voc used, it will be useless

  ```

### [Training Process](#contents)

To train the model, run `train.py`. If the `mindrecord_dir` is empty, it will generate [mindrecord](https://www.mindspore.cn/tutorials/en/master/advanced/dataset/record.html) files by `coco_root`(coco dataset), `voc_root`(voc dataset) or `image_dir` and `anno_path`(own dataset). **Note if mindrecord_dir isn't empty, it will use mindrecord_dir instead of raw images.**

#### Training on Ascend

- Distribute mode

```shell
    bash scripts/run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

We need five or seven parameters for this scripts.

- `DEVICE_NUM`: the device number for distributed train.
- `EPOCH_NUM`: epoch num for distributed train.
- `LR`: learning rate init value for distributed train.
- `DATASET`：the dataset mode for distributed train.
- `RANK_TABLE_FILE :` the path of [rank_table.json](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools), it is better to use absolute path.
- `PRE_TRAINED :` the path of pretrained checkpoint file, it is better to use absolute path.
- `PRE_TRAINED_EPOCH_SIZE :` the epoch num of pretrained.

Training result will be stored in the current path, whose folder name begins with "LOG".  Under this, you can find checkpoint file together with result like the followings in log

```shell
epoch: 1 step: 458, loss is 2.329789
epoch time: 522433.474 ms, per step time: 1140.684 ms
epoch: 2 step: 458, loss is 2.1185513
epoch time: 32531.105 ms, per step time: 71.029 ms
epoch: 3 step: 458, loss is 1.9073256
epoch time: 32643.957 ms, per step time: 71.275 ms
...

epoch: 498 step: 458, loss is 0.6682728
epoch time: 31163.108 ms, per step time: 68.042 ms
epoch: 499 step: 458, loss is 0.8796004
epoch time: 31107.760 ms, per step time: 67.921 ms
epoch: 500 step: 458, loss is 0.7718496
epoch time: 32848.501 ms, per step time: 71.722 ms
```

- single mode

```shell
    bash scripts/run_1p_train.sh [DEVICE_ID] [EPOCH_SIZE] [LR] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

We need five or seven parameters for this scripts.

- `DEVICE_ID`: the device ID for train.
- `EPOCH_NUM`: epoch num for distributed train.
- `LR`: learning rate init value for distributed train.
- `DATASET`：the dataset mode for distributed train.
- `PRE_TRAINED :` the path of pretrained checkpoint file, it is better to use absolute path.
- `PRE_TRAINED_EPOCH_SIZE :` the epoch num of pretrained.

Training result will be stored in the current path, whose folder name begins with "LOG".  Under this, you can find checkpoint file together with result like the followings in log

```shell
epoch: 1 step: 3664, loss is 2.1746433
epoch time: 383006.976 ms, per step time: 104.532 ms
epoch: 2 step: 3664, loss is 2.1719098
epoch time: 227088.618 ms, per step time: 61.978 ms
```

#### Training on GPU

- Distribute mode

```shell
    bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
    For example:
    bash scripts/run_distribute_train_gpu.sh 4 500 coco

```

We need five or seven parameters for this scripts.

- `DEVICE_NUM`: the device number for distributed train.
- `EPOCH_NUM`: epoch num for distributed train.
- `DATASET`：the dataset mode for distributed train.
- `PRE_TRAINED :` the path of pretrained checkpoint file, it is better to use absolute path.
- `PRE_TRAINED_EPOCH_SIZE :` the epoch num of pretrained.

Training result will be stored in the current path, whose folder name begins with "LOG".  Under this, you can find checkpoint file together with result like the followings in log

```shell
epoch: 1 step: 916, loss is 2.1025786
epoch: 1 step: 916, loss is 2.0669649
epoch: 1 step: 916, loss is 2.1528106
epoch: 1 step: 916, loss is 2.1548476
epoch time: 359532.836 ms, per step time: 392.503 ms
epoch time: 359540.271 ms, per step time: 392.511 ms
epoch time: 359568.856 ms, per step time: 392.542 ms
epoch time: 359726.085 ms, per step time: 392.714 ms
...

epoch: 500 step: 916, loss is 0.7354241
epoch: 500 step: 916, loss is 0.77139974
epoch: 500 step: 916, loss is 0.8008499
epoch: 500 step: 916, loss is 0.8703647
epoch time: 249946.352 ms, per step time: 272.867 ms
epoch time: 249974.931 ms, per step time: 272.898 ms
epoch time: 249994.302 ms, per step time: 272.920 ms
epoch time: 250000.210 ms, per step time: 272.926 ms
```

- single mode

```shell
    bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [EPOCH_SIZE] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
    For example:
    bash scripts/run_standalone_train_gpu.sh 0 500 coco
```

We need five or seven parameters for this scripts.

- `DEVICE_ID`: the device ID for train.
- `EPOCH_NUM`: epoch num for distributed train.
- `DATASET`：the dataset mode for distributed train.
- `PRE_TRAINED :` the path of pretrained checkpoint file, it is better to use absolute path.
- `PRE_TRAINED_EPOCH_SIZE :` the epoch num of pretrained.

Training result will be stored in the current path, whose folder name begins with "LOG".  Under this, you can find checkpoint file together with result like the followings in log

```shell
epoch: 1 step: 3664, loss is 2.2511892
epoch time: 560109.006 ms, per step time: 152.868 ms

```

### [Evaluation Process](#contents)

#### Evaluation on Ascend

```shell
bash scripts/run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

We need two parameters for this scripts.

- `DATASET`：the dataset mode of evaluation dataset.
- `CHECKPOINT_PATH`: the absolute path for checkpoint file.
- `DEVICE_ID`: the device id for eval.

> checkpoint can be produced in training process.

Inference result will be stored in the example path, whose folder name begins with "eval". Under this, you can find result like the followings in log.

```shell
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.257
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.045
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.222
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.259
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.405
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.131
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.704

========================================

mAP: 0.2527925497483538
```

#### Evaluation on GPU

```shell
bash scripts/run_eval_gpu.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
For example:
bash scripts/run_eval_gpu.sh coco path/to/checkpoint.pth 0
```

We need two parameters for this scripts.

- `DATASET`：the dataset mode of evaluation dataset.
- `CHECKPOINT_PATH`: the absolute path for checkpoint file.
- `DEVICE_ID`: the device id for eval.

> checkpoint can be produced in training process.

Inference result will be stored in the example path, whose folder name begins with "eval". Under this, you can find result like the followings in log.

```shell
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.258
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.424
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.264
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.045
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.229
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.266
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.446
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.468
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.715

========================================

mAP: 0.25840622348226616
```

### Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

#### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

#### [Infer](#contents)

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1. The precision calculation process needs about 70G+ memory space, otherwise the process will be killed for execeeding memory limits.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_TYPE] [DEVICE_ID]
```

- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive. Note that the image shape of ssd_vgg16 inference is [300, 300], The DVPP hardware restricts width 16-alignment and height even-alignment. Therefore, the network needs to use the CPU operator to process images.
- `ANNO_PATH` is mandatory, and must specify annotation file path including file name.
- `DEVICE_ID` is optional, default value is 0.

#### [Result](#contents)

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.252
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.419
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.256
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.041
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.225
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.447
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.261
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.405
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.438
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.125
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.459
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.704
mAP:0.2522562031397977
```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters          | Ascend  | GPU  |  
| ------------------- | ------------------- | ------------------- |
| Model Version       | SSD mobielnetV2  | SSD mobielnetV2  |
| Resource            | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G| GeForce RTX 3090 ；CPU 2.90GHz，16cores；Memory，252G|
| uploaded Date       | 03/12/2021 (month/day/year)                    | 26/1/2022 (month/day/year)                 |
| MindSpore Version   | 1.1.1                                          | 1.5.0                                        |
| Dataset             | COCO2017          | COCO2017          |
| Training Parameters | epoch = 500,  batch_size = 32  | epoch = 500,  batch_size = 32  |
| Optimizer           | Momentum | Momentum |
| Loss Function       | Sigmoid Cross Entropy,SmoothL1Loss   | Sigmoid Cross Entropy,SmoothL1Loss   |
| Speed               | 8pcs: 80ms/step  |4pcs: 332ms/step  |
| Total time          | 8pcs: 4.67hours      | 4pcs: 21.9 hours      |
| Scripts             | <https://gitee.com/mindspore/models/tree/r2.0/research/cv/ssd_mobilenetV2> |

#### Inference Performance

| Parameters          | Ascend                      | GPU                      |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | SSD mobilenetV2             | SSD mobilenetV2             |
| Resource            | Ascend 910                  | GeForce RTX 3090            |
| Uploaded Date       | 03/12/2021 (month/day/year) | 26/1/2022 (month/day/year) |
| MindSpore Version   | 1.1.1                       | 1.5.0                       |
| Dataset             | COCO2017                    | COCO2017 |
| batch_size          | 1                           | 1 |
| outputs             | mAP                         | mAP |
| Accuracy            | IoU=0.50: 25.28%            | IoU=0.50: 25.8%            |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).  
