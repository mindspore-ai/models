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
            - [Training on GPU](#training-on-gpu)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation on GPU](#evaluation-on-gpu)
            - [Performance](#performance)
        - [Export Process](#Export-process)
            - [Export](#Export)
        - [Inference Process](#Inference-process)
            - [Inference](#Inference)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [SSD Description](#contents)

SSD discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape.Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes.

[Paper](https://arxiv.org/abs/1512.02325):   Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.European Conference on Computer Vision (ECCV), 2016 (In press).

## [Model Architecture](#contents)

The SSD approach is based on a feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections. The early network layers are based on a standard architecture used for high quality image classification, which is called the base network. Then add auxiliary structure to the network to produce detections.

We present **ssd-resnet34**, reference from the paper. Using resnet34 as backbone and the same bbox predictor as the paper presents.

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
  First, install Cython, pycocotool and opencv to process data and to get evaluation result.

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

1. Change the dataset and model configuration in the corresponding config. `src/config_ssd_resnet34.py`.
2. You need a pretrained model for `resnet34` backbone ([download link](https://download.mindspore.cn/model_zoo/r1.3/resnet34_ascend_v130_imagenet2012_official_cv_bs256_top1acc73.83__top5acc91.61/resnet34_ascend_v130_imagenet2012_official_cv_bs256_top1acc73.83__top5acc91.61.ckpt)). Set the checkpoint path to `feature_extractor_base_param` in `src/config_ssd_resnet34.py`.

### Run the scripts

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on GPU

```shell
# distributed training on GPU
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH](optional) [PRE_TRAINED_EPOCH_SIZE](required if PRE_TRAINED_PATH is specified)

# run eval on GPU
bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET] [DATASET_PATH] [CHECKPOINT_PATH] [MINDRECORD_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
 └─ ssd_resnet_34
    ├─ scripts
    │ ├─ run_distribute_train_gpu.sh      ## shell 8P GPU training script
    │ ├─ run_eval_gpu.sh                  ## shell script for eval on GPU
    │ └─ run_standalone_train_gpu.sh      ## shell 1P GPU training script
    ├─ src
    │ ├─ __init__.py                      ## init file
    │ ├─ anchor_generator.py              ## anchor generator
    │ ├─ box_util.py                      ## bbox utils
    │ ├─ config.py                        ## total config
    │ ├─ config_ssd_resnet34.py           ## ssd-resnet34 config
    │ ├─ dataset.py                       ## create and process dataset
    │ ├─ eval_utils.py                    ## eval utils
    │ ├─ init_params.py                   ## parameters utils
    │ ├─ load_checkpoints.py              ## load checkpoints for resume
    │ ├─ lr_schedule.py                   ## learning ratio generator
    │ ├─ multi_box.py                     ## multi box utils
    │ ├─ resnet34.py                      ## resnet34 architecture
    │ └─ ssd_resnet34.py                  ## ssd-resnet34 architecture
    ├─ eval.py                            ## eval code
    ├─ export.py                          ## export mindir script
    ├─ README.md                          ## English descriptions about SSD
    └─ train.py                           ## train code
```

### [Script Parameters](#contents)

```text
Major parameters in train.py and config.py for Single GPU train:
  "device_num": 1                                  # Use device nums
  "lr": 0.075                                      # Learning rate init value
  "dataset": coco                                  # Dataset name
  "epoch_size": 1000                               # Epoch size
  "batch_size": 32                                 # Batch size of input tensor
  "pre_trained": None                              # Pretrained checkpoint file path
  "pre_trained_epochs": 0                          # Pretrained epoch size
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
Major parameters in train.py and config.py for Multi GPU train:
  "device_num": 1                                  # Use device nums
  "lr": 0.6                                        # Learning rate init value
  "dataset": coco                                  # Dataset name
  "epoch_size": 500                                # Epoch size
  "batch_size": 32                                 # Batch size of input tensor
  "pre_trained": None                              # Pretrained checkpoint file path
  "pre_trained_epochs": 0                          # Pretrained epoch size
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
  "anno_path": ""
```

### [Training Process](#contents)

To train the model, run `train.py`. If the `mindrecord_dir` is empty, it will generate [mindrecord](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/dataset/record.html) files by `coco_root`(coco dataset), `voc_root`(voc dataset) or `image_dir` and `anno_path`(own dataset). **Note if mindrecord_dir isn't empty, it will use mindrecord_dir instead of raw images.**

#### Training on GPU

- Distribute mode

```shell
     bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH](optional) [PRE_TRAINED_EPOCH_SIZE](required if PRE_TRAINED_PATH is specified)
```

- Standalone training

```shell
     bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH](optional) [PRE_TRAINED_EPOCH_SIZE](required if PRE_TRAINED_PATH is specified)
```

We need five or six parameters for this scripts.

- `DEVICE_NUM`: A number of the devices for the train.
- `DATASET`：the dataset mode for distributed train.
- `DATASET_PATH`：the dataset path for distributed train.
- `MINDRECIRD_PATH`：the mindrecord path for distributed train.
- `TRAIN_OUT_PATH`：the output path of train for distributed train.
- `PRE_TRAINED_PATH`: the path of pretrained checkpoint file, it is better to use absolute path.
- `PRE_TRAINED_EPOCH_SIZE`: number of epochs passed by checkpoint.

Training result will be stored in the train path, whose folder name  "log".  Under this, you can find checkpoint file together with result like the followings in log

```shell
Single GPU training:

epoch: 1 step: 3664, loss is 3.3759894
epoch time: 683788.272 ms, per step time: 187.570 ms
epoch: 2 step: 3664, loss is 3.212594
epoch time: 635788.033 ms, per step time: 176.004 ms
epoch: 3 step: 3664, loss is 3.0225306
epoch time: 628460.165 ms, per step time: 171.036 ms
epoch: 4 step: 3664, loss is 2.5885172
epoch time: 634952.848 ms, per step time: 173.512 ms
...
epoch: 997 step: 3664, loss is 1.1043124
epoch time: 636851.785 ms, per step time: 173.165 ms
epoch: 998 step: 3664, loss is 1.0603312
epoch time: 654803.247 ms, per step time: 178.656 ms
epoch: 999 step: 3664, loss is 1.2640748
epoch time: 629573.241 ms, per step time: 171.035 ms
epoch: 1000 step: 3664, loss is 1.1621332
epoch time: 635427.872 ms, per step time: 173.693 ms
```

#### Transfer Training

You can train your own model based on either pretrained classification model or pretrained detection model. You can perform transfer training by following steps.

1. Convert your own dataset to COCO or VOC style. Otherwise you have to add your own data preprocess code.
2. Change config_xxx.py according to your own dataset, especially the `num_classes`.
3. Prepare a pretrained checkpoint. You can load the pretrained checkpoint by `pre_trained` argument. Transfer training means a new training job, so just keep `pre_trained_epoch_size`  same as default value `0`.
4. Set argument `filter_weight` to `True` while calling `train.py`, this will filter the final detection box weight from the pretrained model.
5. Build your own bash scripts using new config and arguments for further convenient.

### [Evaluation Process](#contents)

#### Evaluation on GPU

```shell
bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET] [DATASET_PATH] [CHECKPOINT_PATH] [MINDRECORD_PATH]
```

We need five parameters for this script.

- `DEVICE_ID`: the device id for eval.
- `DATASET`：the dataset mode of evaluation dataset.
- `DATASET_PATH`：the dataset path for evaluation.
- `CHECKPOINT_PATH`: the absolute path for checkpoint file.
- `MINDRECORD_PATH`：the mindrecord path for evaluation.

> checkpoint can be produced in training process.

Inference result will be stored in the eval path, whose folder name "log". Under this, you can find result like the followings in log.

```shell
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.254
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.378
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.275
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.017
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.247
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.467
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.261
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.399
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.081
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.451
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.752

========================================

mAP: 0.2541605289643854

```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["MINDIR", "ONNX"]

## [Model Description](#contents)

### [Performance](#contents)

#### Inference Performance

| Parameters          | GPU                         |
| ------------------- | --------------------------- |
| Model Version       | SSD-ResNet34                |
| Resource            | 1x V100-PCIE, CPU Intel(R) Xeon(R) E5-2686 v4 @ 2.30GHz           |
| Uploaded Date       | 11/31/2021 (month/day/year) |
| MindSpore Version   | 1.5rc                       |
| Dataset             | COCO2017                    |
| Batch size          | 32                          |
| Outputs             | mAP (IoU=0.50:0.95)         |
| Accuracy            | 25.85%                      |
| Model for inference | 98.77M (.ckpt file)         |
| Total training time | 183 hours                   |
| FPS                 | 177                         |

| Parameters          | GPU                         |
| ------------------- | --------------------------- |
| Model Version       | SSD-ResNet34                |
| Resource            | 8x V100-PCIE, CPU Intel(R) Xeon(R) Gold 6226R @ 2.90GHz             |
| Uploaded Date       | 11/31/2021 (month/day/year) |
| MindSpore Version   | 1.5                         |
| Dataset             | COCO2017                    |
| Batch size          | 32                          |
| Outputs             | mAP (IoU=0.50:0.95)         |
| Accuracy            | 25.41%                      |
| Model for inference | 98.77M (.ckpt file)         |
| Total training time | 12.5 hours                  |
| FPS                 | 1292                        |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
