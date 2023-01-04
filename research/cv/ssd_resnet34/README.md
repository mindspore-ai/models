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
            - [Transfer Training](#transfer-training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation on Ascend](#evaluation-on-ascend)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer](#infer)
        - [result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Train Performance](#train-performance)
            - [Inference Performance](#inference-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [SSD Description](#contents)

SSD discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape.Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes.

[Paper](https://arxiv.org/abs/1512.02325):   Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.European Conference on Computer Vision (ECCV), 2016 (In press).

## [Model Architecture](#contents)

The SSD approach is based on a feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections. The early network layers are based on a standard architecture used for high quality image classification, which is called the base network. Then add auxiliary structure to the network to produce detections.

We present four different base architecture.

- **ssd300**, reference from the paper. Using mobilenetv2 as backbone and the same bbox predictor as the paper present.
- ***ssd-mobilenet-v1-fpn**, using mobilenet-v1 and FPN as feature extractor with weight-shared box predcitors.
- ***ssd-resnet50-fpn**, using resnet50 and FPN as feature extractor with weight-shared box predcitors.
- **ssd-vgg16**, reference from the paper. Using vgg16 as backbone and the same bbox predictor as the paper present.
- **ssd-resnet34**,reference from the paper. Using resnet34 as backbone and the same bbox predictor as the paper present.

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

       Change the `coco_root` and other settings you need in `src/config_xxx.py`. The directory structure is as follows:

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
       Change `classes`, `num_classes`, `voc_json` and `voc_root` in `src/config_xxx.py`. `voc_json` is the path of json file with coco format for evaluation, `voc_root` is the path of VOC dataset, the directory structure is as follows:

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

       Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are setting in `src/config_xxx.py`.

## [Quick Start](#contents)

### Prepare the model

1. Chose the model by changing the `using_model` in `src/config.py`. The optional models are: `ssd300`, `ssd_mobilenet_v1_fpn`, `ssd_vgg16`, `ssd_resnet50_fpn`,`ssd_resnet34`
2. Change the dataset config in the corresponding config. `src/config_xxx.py`, `xxx` is the corresponding backbone network name
3. If you are running with `ssd_mobilenet_v1_fpn` , `ssd_resnet50_fpn`or , `ssd_resnet34` , you need a pretrained model for `mobilenet_v1` ,`resnet50`or `resnet34`. Set the checkpoint path to `feature_extractor_base_param` in `src/config_xxx.py`. For more detail about training pre-trained model, please refer to the corresponding backbone network.

### Run the scripts

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

```shell
# distributed training on Ascend
sh scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH][PRE_TRAINED_PATH](optional)

# run eval on Ascend
sh scripts/run_eval.sh [DEVICE_ID] [DATASET] [DATASET_PATH] [CHECKPOINT_PATH] [MINDRECORD_PATH]

# run inference on Ascend310, MINDIR_PATH is the mindir model which you can export from checkpoint using export.py
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
  └─ ssd_resnet34
    ├─ ascend310_infer
    ├─ scripts
      ├─ run_distribute_train.sh      ## shell script for distributed on ascend 910
      ├─ run_eval.sh                  ## shell script for eval on ascend 910
      ├─ run_infer_310.sh             ## shell script for eval on ascend 310
      └─ run_standalone_train.sh      ## shell script for standalone on ascend 910
    ├─ src
      ├─ __init__.py                  ## init file
      ├─ anchor_generator.py          ## anchor generator
      ├─ box_util.py                  ## bbox utils
      ├─ callback.py                  ## the callback of train and evaluation
      ├─ config.py                    ## total config
      ├─ config_ssd_resnet34.py       ## ssd_resnet34 config
      ├─ dataset.py                   ## create dataset and process dataset
      ├─ eval_utils.py                ## eval utils
      ├─ lr_schedule.py               ## learning ratio generator
      ├─ init_params.py               ## parameters utils
      ├─ resnet34.py                  ## resnet34 architecture
      ├─ ssd.py                       ## ssd architecture
      └─ ssd_resnet34.py              ## ssd_resnet34 architecture
    ├─ eval.py                        ## eval script
    ├─ export.py                      ## export mindir script
    ├─ postprocess.py                 ## eval on ascend 310
    ├─ README.md                      ## English descriptions about SSD
    ├─ README_CN.md                   ## Chinese descriptions about SSD
    ├─ requirements.txt               ## Requirements
    └─ train.py                       ## train script
```

### [Script Parameters](#contents)

  ```shell
  Major parameters in train.py and config.py as follows:

    "device_num": 1                                  # Use device nums
    "lr": 0.075                                      # Learning rate init value
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
    "image_shape": [300, 300]                        # Image height and width used as input to the model
    "mindrecord_dir": "/data/MindRecord_COCO"        # MindRecord path
    "coco_root": "/data/coco2017"                    # COCO2017 dataset path
    "voc_root": "/data/voc_dataset"                  # VOC original dataset path
    "voc_json": "annotations/voc_instances_val.json" # is the path of json file with coco format for evaluation
    "image_dir": ""                                  # Other dataset image path, if coco or voc used, it will be useless
    "anno_path": ""                                  # Other dataset annotation path, if coco or voc used, it will be useless

  ```

### [Training Process](#contents)

To train the model, run `train.py`. If the `mindrecord_dir` is empty, it will generate [mindrecord](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/dataset/record.html) files by `coco_root`(coco dataset), `voc_root`(voc dataset) or `image_dir` and `anno_path`(own dataset). **Note if mindrecord_dir isn't empty, it will use mindrecord_dir instead of raw images.**

#### Training on Ascend

- Distribute mode

```shell
     sh scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH][PRE_TRAINED_PATH](optional)
```

- Standalone training

```shell
     sh scripts/run_standalone_train.sh [DEVICE_ID] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH][PRE_TRAINED_PATH](optional)
```

We need five or six parameters for this scripts.

- `RANK_TABLE_FILE :` the path of [rank_table.json](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools), it is better to use absolute path.
- `DATASET`：the dataset mode for distributed train.
- `DATASET_PATH`：the dataset path for distributed train.
- `MINDRECIRD_PATH`：the mindrecord path for distributed train.
- `TRAIN_OUT_PATH`：the output path of train for distributed train.
- `PRE_TRAINED_PATH :` the path of pretrained checkpoint file, it is better to use absolute path.

Training result will be stored in the train path, whose folder name  "log".  Under this, you can find checkpoint file together with result like the followings in log

```shell
epoch: 1 step: 458, loss is 4.185711
epoch time: 138740.569 ms, per step time: 302.927 ms
epoch: 2 step: 458, loss is 4.3121023
epoch time: 47116.166 ms, per step time: 102.874 ms
epoch: 3 step: 458, loss is 3.2209284
epoch time: 47149.108 ms, per step time: 102.946 ms
epoch: 4 step: 458, loss is 3.5159926
epoch time: 47174.645 ms, per step time: 103.001 ms
...
epoch: 497 step: 458, loss is 1.0916114
epoch time: 47164.002 ms, per step time: 102.978 ms
epoch: 498 step: 458, loss is 1.157409
epoch time: 47172.836 ms, per step time: 102.997 ms
epoch: 499 step: 458, loss is 1.2065268
epoch time: 47155.245 ms, per step time: 102.959 ms
epoch: 500 step: 458, loss is 1.1856415
epoch time: 47666.430 ms, per step time: 104.075 ms
```

#### Transfer Training

You can train your own model based on either pretrained classification model or pretrained detection model. You can perform transfer training by following steps.

1. Convert your own dataset to COCO or VOC style. Otherwise you have to add your own data preprocess code.
2. Change config_xxx.py according to your own dataset, especially the `num_classes`.
3. Prepare a pretrained checkpoint. You can load the pretrained checkpoint by `pre_trained` argument. Transfer training means a new training job, so just keep `pre_trained_epoch_size`  same as default value `0`.
4. Set argument `filter_weight` to `True` while calling `train.py`, this will filter the final detection box weight from the pretrained model.
5. Build your own bash scripts using new config and arguments for further convenient.

### [Evaluation Process](#contents)

#### Evaluation on Ascend

```shell
sh scripts/run_eval.sh [DEVICE_ID] [DATASET] [DATASET_PATH] [CHECKPOINT_PATH] [MINDRECORD_PATH]
```

We need five parameters for this scripts.

- `DEVICE_ID`: the device id for eval.
- `DATASET`：the dataset mode of evaluation dataset.
- `DATASET_PATH`：the dataset path for evaluation.
- `CHECKPOINT_PATH`: the absolute path for checkpoint file.
- `MINDRECIRD_PATH`：the mindrecord path for evaluation.

> checkpoint can be produced in training process.

Inference result will be stored in the eval path, whose folder name "log". Under this, you can find result like the followings in log.

```shell
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.240
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.258
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.016
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.229
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.446
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.256
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.389
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.077
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.439
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.734

========================================

mAP: 0.24011857000302622

```

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer

Before performing inference, the mindir file must bu exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1. The precision calculation process needs about 70G+ memory space, otherwise the process will be killed for execeeding memory limits.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [ANNO_FILE] [DEVICE_TYPE] [DEVICE_ID]
```

- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive. Note that the image shape of ssd_vgg16 inference is [300, 300], The DVPP hardware restricts width 16-alignment and height even-alignment. Therefore, the network needs to use the CPU operator to process images.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.250
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.266
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.018
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.241
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.260
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.399
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.090
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.739
0.249879750926743
```

## [Model Description](#contents)

### [Performance](#contents)

#### Train Performance

| Parameters          | Ascend                      |
| ------------------- | ----------------------------|
| Model Version       | SSD_ResNet34                |
| Resource            | Ascend 910；CPU 2.60GHz，192 cores；Memory 755 G|
| Uploaded Date       | 08/31/2021 (month/day/year) |
| MindSpore Version   | 1.3                       |
| Dataset             | COCO2017                    |
| Training Parameters | epoch = 500, batch_size = 32|
| Optimizer           | Momentum                    |
| Loss Function       | Sigmoid Cross Entropy,SmoothL1Loss|
| Speed               | 8pcs: 101ms/step            |
| Total time          | 8pcs: 8.34h                 |

#### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SSD_ResNet34                   |
| Resource            | Ascend 910                  |
| Uploaded Date       | 08/31/2021 (month/day/year) |
| MindSpore Version   | 1.2                       |
| Dataset             | COCO2017                    |
| outputs             | mAP                         |
| Accuracy            | IoU=0.50: 24.0%             |
| Model for inference | 98.77M(.ckpt file)             |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).  