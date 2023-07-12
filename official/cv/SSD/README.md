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
            - [Evaluation while training](#evaluation-while-training)
            - [Transfer Training](#transfer-training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation on Ascend](#evaluation-on-ascend)
            - [Evaluation on GPU](#evaluation-on-gpu)
            - [ONNX Evaluation](#onnx-evaluation)
    - [Inference Process](#inference-process)
        - [Infer](#infer)
            - [Export MindIR](#export-mindir)
            - [Infer](#infer-1)
            - [result](#result)
        - [ONNX Infer](#onnx-infer)
            - [Export ONNX](#export-onnx)
            - [Infer](#infer-2)
            - [Result](#result-1)
    - [Model Description](#model-description)
        - [Performance](#performance)
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

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2017](<https://cocodataset.org/#download>)

- Dataset size：19G
    - Train：18G，118000 images
    - Val：1G，5000 images
    - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
    - Note：Data will be processed in dataset.py

[helmet](<https://osf.io/4pwj8>)

- Dataset size：526 MB
    - Train：325 MB，1500 images
    - Val：201 MB，1000 images

## [Environment Requirements](#contents)

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2017.

    `scripts` directory provides script `run_download_dataset.sh` for automatically downloading coco2017,
which can be used to automatically download and decompress data sets.

    At present, only downloading of coco2017 datasets is supported.

- We use COCO2017 as training dataset in this example by default, and you can also use your own datasets.
  First, install Cython ,pycocotool and opencv to process data and to get evaluation result.

    ```shell
    pip install Cython
    pip install pycocotools
    pip install opencv-python
    ```

    1. If coco dataset is used. **Select dataset to coco when run script.**

        Change the `coco_root` and other settings you need in `model_utils/ssd_xxx.yaml`. The directory structure is as follows:

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
        Change `classes`, `num_classes`, `voc_json` and `voc_root` in `model_utils/ssd_xxx.yaml`. `voc_json` is the path of json file with coco format for evaluation, `voc_root` is the path of VOC dataset, the directory structure is as follows:

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

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class infomations of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are setting in `model_utils/ssd_xxx.yaml`.

## [Quick Start](#contents)

### Prepare the model

1. Chose the model by changing the `using_model` in `model_utils/ssd_xxx.yaml`. The optional models are: `ssd300`, `ssd_mobilenet_v1_fpn`, `ssd_vgg16`, `ssd_resnet50_fpn`.
2. Change the dataset config in the corresponding config. `model_utils/ssd_xxx.yaml`, `xxx` is the corresponding backbone network name
3. If you are running with `ssd_mobilenet_v1_fpn` or `ssd_resnet50_fpn`, you need a pretrained model for `mobilenet_v1` or `resnet50`. Set the checkpoint path to `feature_extractor_base_param` in `model_utils/ssd_xxx.yaml`. For more detail about training pre-trained model, please refer to the corresponding backbone network.

### Run the scripts

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

```shell
# distributed training on Ascend
bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [CONFIG_PATH]

# run eval on Ascend
bash run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID] [CONFIG_PATH]

# run inference on Ascend310, MINDIR_PATH is the mindir model which you can export from checkpoint using export.py
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID] [CONFIG_PATH]
```

- running on GPU

```shell
# distributed training on GPU
bash run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [CONFIG_PATH]

# run eval on GPU
bash run_eval_gpu.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID] [CONFIG_PATH]
```

- running on CPU(support Windows and Ubuntu)

**CPU is usually used for fine-tuning, which needs pre_trained checkpoint.**

```shell
# training on CPU
python train.py --device_target=CPU --lr=[LR] --dataset=[DATASET] --epoch_size=[EPOCH_SIZE] --batch_size=[BATCH_SIZE] --config_path=[CONFIG_PATH] --pre_trained=[PRETRAINED_CKPT] --filter_weight=True --save_checkpoint_epochs=1

# run eval on GPU
python eval.py --device_target=CPU --dataset=[DATASET] --checkpoint_file_path=[PRETRAINED_CKPT] --config_path=[CONFIG_PATH]
```

- Run on docker

Build docker images(Change version to the one you actually used)

```shell
# build docker
docker build -t ssd:20.1.0 . --build-arg FROM_IMAGE_NAME=ascend-mindspore-arm:20.1.0
```

Create a container layer over the created image and start it

```shell
# start docker
bash scripts/docker_start.sh ssd:20.1.0 [DATA_DIR] [MODEL_DIR]
```

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```python
# run distributed training on modelarts example
# (1) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config directory to "config_path=/The path of config in S3/"
# (3) Set the code directory to "/path/ssd" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run evaluation on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a.  Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (3) Set the config directory to "config_path=/The path of config in S3/"
# (4) Set the code directory to "/path/ssd" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

Then you can run everything just like on ascend.

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└─ cv
  └─ ssd
    ├─ README.md                      ## descriptions about SSD
    ├─ ascend310_infer                ## source code of 310 inference
    ├─ scripts
      ├─ docker_start.sh              ## shell script for start docker container
      ├─ run_distribute_train.sh      ## shell script for distributed on ascend
      ├─ run_distribute_train_gpu.sh  ## shell script for distributed on gpu
      ├─ run_download_dataset.sh      ## shell script for downloading dataset
      ├─ run_eval.sh                  ## shell script for eval on ascend
      ├─ run_eval_onnx.sh             ## shell script for onnx model evaluation
      ├─ run_eval_gpu.sh              ## shell script for eval on gpu
      ├─ run_export.sh                ## shell script for exporting to MINDIR,AIR,ONNX
      └─ run_infer_cpp.sh             ## shell script for C++ inference
    ├─ src
      ├─ __init__.py                      ## init file
      ├─ anchor_generator.py              ## anchor generator
      ├─ box_util.py                      ## bbox utils
      ├─ dataset.py                       ## create dataset and process dataset
      ├─ download_dataset.py              ## download dataset
      ├─ eval_callback.py                 ## eval callback function definition
      ├─ eval_utils.py                    ## eval utils
      ├─ fpn.py                           ## feature pyramid network
      ├─ init_params.py                   ## parameters utils
      ├─ lr_schedule.py                   ## learning ratio generator
      ├─ mobilenet_v1.py                  ## network definition for mobilenet-v1
      ├─ resnet.py                        ## network definition for resnet
      ├─ ssd.py                           ## ssd architecture
      └─ vgg16.py                         ## network definition for vgg16
      ├── model_utils
      │   ├── config.py                   ## parameter configuration
      │   ├── device_adapter.py           ## device adapter
      │   ├── local_adapter.py            ## local adapter
      │   ├── moxing_adapter.py           ## moxing adapter
    ├─ config
        ├─ ssd_mobilenet_v1_300_config_gpu.yaml ## parameter configuration
        ├─ ssd_mobilenet_v1_fpn_config.yaml ## parameter configuration
        ├─ ssd_resnet50_fpn_config.yaml ## parameter configuration
        ├─ ssd_vgg16_config.yaml ## parameter configuration
        ├─ ssd300_config.yaml ## parameter configuration
        ├─ ssd_mobilenet_v1_fpn_config_gpu.yaml ## GPU parameter configuration
        ├─ ssd_resnet50_fpn_config_gpu.yaml ## GPU parameter configuration
        ├─ ssd_vgg16_config_gpu.yaml ## GPU parameter configuration
        ├─ ssd300_config_gpu.yaml ## GPU parameter configuration
        └─ ssd_mobilenet_v1_fpn_ONNX_config.yaml ## parameter configuration
    ├─ Dockerfile                         ## docker file
    ├─ download_dataset.py                ## download dataset script
    ├─ eval.py                            ## eval scripts
    ├─ eval_onnx.py                       ## eval onnx model
    ├─ export.py                          ## export mindir script
    ├─ postprocess.py                     ## post-processing script for 310 inference
    ├─ train.py                           ## train scripts
    └─ mindspore_hub_conf.py              ## mindspore hub interface
```

### [Script Parameters](#contents)

  ```shell
  Major parameters in the yaml config file as follows:

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
    "run_eval": False                                # Run evaluation when training
    "save_best_ckpt": True                           # Save best checkpoint when run_eval is True
    "eval_start_epoch": 40                           # Evaluation start epoch when run_eval is True
    "eval_interval": 1                               # valuation interval when run_eval is True

    "class_num": 81                                  # Dataset class number
    "img_shape": [300, 300]                        # Image height and width used as input to the model
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
    bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [CONFIG_PATH] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

- Standalone training

```shell
    bash run_standalone_train.sh [DEVICE_ID] [EPOCH_SIZE] [LR] [DATASET] [CONFIG_PATH] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

We need five or seven parameters for this scripts.

- `DEVICE_NUM`: the device number for distributed train.
- `EPOCH_NUM`: epoch num for distributed train.
- `LR`: learning rate init value for distributed train.
- `DATASET`：the dataset mode for distributed train.
- `RANK_TABLE_FILE :` the path of [rank_table.json](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools), it is better to use absolute path.
- `CONFIG_PATH`: parameter configuration.
- `PRE_TRAINED :` the path of pretrained checkpoint file, it is better to use absolute path.
- `PRE_TRAINED_EPOCH_SIZE :` the epoch num of pretrained.

Training result will be stored in the current path, whose folder name begins with "LOG".  Under this, you can find checkpoint file together with result like the followings in log

```shell
epoch: 1 step: 458, loss is 3.1681802
epoch time: 228752.4654865265, per step time: 499.4595316299705
epoch: 2 step: 458, loss is 2.8847265
epoch time: 38912.93382644653, per step time: 84.96273761232868
epoch: 3 step: 458, loss is 2.8398118
epoch time: 38769.184827804565, per step time: 84.64887516987896
...

epoch: 498 step: 458, loss is 0.70908034
epoch time: 38771.079778671265, per step time: 84.65301261718616
epoch: 499 step: 458, loss is 0.7974688
epoch time: 38787.413120269775, per step time: 84.68867493508685
epoch: 500 step: 458, loss is 0.5548882
epoch time: 39064.8467540741, per step time: 85.29442522723602
```

#### Training on GPU

- Distribute mode

```shell
    bash run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [CONFIG_PATH] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

We need five or seven parameters for this scripts.

- `DEVICE_NUM`: the device number for distributed train.
- `EPOCH_NUM`: epoch num for distributed train.
- `LR`: learning rate init value for distributed train.
- `DATASET`：the dataset mode for distributed train.
- `CONFIG_PATH`: parameter configuration.
- `PRE_TRAINED :` the path of pretrained checkpoint file, it is better to use absolute path.
- `PRE_TRAINED_EPOCH_SIZE :` the epoch num of pretrained.

Training result will be stored in the current path, whose folder name is "LOG".  Under this, you can find checkpoint files together with result like the followings in log

```shell
epoch: 1 step: 1, loss is 420.11783
epoch: 1 step: 2, loss is 434.11032
epoch: 1 step: 3, loss is 476.802
...
epoch: 1 step: 458, loss is 3.1283689
epoch time: 150753.701, per step time: 329.157
...
```

#### Evaluation while training

You can add `run_eval` to start shell and set it True, if you want evaluation while training. And you can set argument option: `save_best_ckpt`, `eval_start_epoch`, `eval_interval` when `run_eval` is True.

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
bash run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID] [CONFIG_PATH]
```

We need four parameters for this scripts.

- `DATASET`：the dataset mode of evaluation dataset.
- `CHECKPOINT_PATH`: the absolute path for checkpoint file.
- `DEVICE_ID`: the device id for eval.
- `CONFIG_PATH`: parameter configuration.

> checkpoint can be produced in training process.

Inference result will be stored in the example path, whose folder name begins with "eval". Under this, you can find result like the followings in log.

```shell
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.238
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.400
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.240
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.198
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.438
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.250
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.389
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.424
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.122
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434
Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.697

========================================

mAP: 0.23808886505483504
```

#### Evaluation on GPU

```shell
bash run_eval_gpu.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID] [CONFIG_PATH]
```

We need four parameters for this scripts.

- `DATASET`：the dataset mode of evaluation dataset.
- `CHECKPOINT_PATH`: the absolute path for checkpoint file.
- `DEVICE_ID`: the device id for eval.
- `CONFIG_PATH`: parameter configuration.

> checkpoint can be produced in training process.

Inference result will be stored in the example path, whose folder name begins with "eval". Under this, you can find result like the followings in log.

```shell
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.224
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.375
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.228
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.034
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.189
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.407
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.243
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.382
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.417
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.120
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.425
Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.686

========================================

mAP: 0.2244936111705981
```

#### ONNX Evaluation

- Export your model to ONNX:

  ```bash
  python export.py --checkpoint_file_path /path/to/ssd.ckpt --file_name /path/to/ssd.onnx --file_format ONNX --config_path config/ssd300_config_gpu.yaml --batch_size 1
  ```

- Run ONNX evaluation from ssd directory:

  ```bash
  bash scripts/run_eval_onnx.sh <DATA_DIR> <COCO_SUBDIR> <ONNX_MODEL_PATH> [<INSTANCES_SET>] [<DEVICE_TARGET>] [<CONFIG_PATH>]
  ```

  Results will be saved in eval.log and have the following form:

  ```log
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.239
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.398
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.242
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.035
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.198
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.436
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.251
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.388
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.423
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.117
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.435
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.688
  mAP: 0.23850595066045968
  ```

## Inference Process

### Infer

#### [Export MindIR](#contents)

Export MindIR on local

```shell
python export.py --checkpoint_file_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --config_path [CONFIG_PATH]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

You can also use the shell script in 'scripts' to export, just to give the configuration script path and export type.
The FILE_FORMAT is optional in `air`/`mindir`/`onnx`.

```shell
bash run_export.sh [CONFIG_FILE_PATH] [FILE_FORMAT]
```

Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# Export on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
#          Set "file_name='./ssd'" on default_config.yaml file.
#          Set "file_format: 'MINDIR'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
#          Add "file_name='./ssd'" on the website UI interface.
#          Add "file_format: 'MINDIR'" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config_path="/path/yaml file" on the website UI interface.
# (3) Set the code directory to "/path/ssd" on the website UI interface.
# (4) Set the startup file to "export.py" on the website UI interface.
# (5) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

#### Infer

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch size can only be set to 1. The precision calculation process needs about 70G+ memory space, otherwise the process will be killed for execeeding memory limits.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [CONFIG_PATH] [DEVICE_TYPE] [DEVICE_ID]
```

- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive. Note that the image shape of ssd_vgg16 inference is [300, 300], The DVPP hardware restricts width 16-alignment and height even-alignment. Therefore, the network needs to use the CPU operator to process images.
- `DEVICE_ID` is optional, default value is 0.

#### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.339
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.521
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.370
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.168
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.461
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.310
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.481
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.515
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.293
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.659
mAP: 0.33880018942412393
```

### ONNX Infer

#### [Export ONNX](#contents)

```shell
python export.py --checkpoint_file_path /path/to/ssd.ckpt --file_name /path/to/ssd.onnx --file_format ONNX --config_path config/ssd300_config_gpu.yaml --batch_size 1
```

Parameter ckpt_file is mandatory.

'FILE_FORMAT' select ONNX.

#### Infer

Currently, only inference with batch_Size equal to 1 is supported. Batch_Size is given in the inference script.

```shell
/bin/bash ./infer_ssd_mobilenet_v1_fpn_onnx.sh <DATA_PATH> <COCO_ROOT> <ONNX_MODEL_PATH> [<INSTANCES_SET>] [<DEVICE_TARGET>] [<CONFIG_PATH>]

# Example
/bin/bash ./infer_ssd_mobilenet_v1_fpn_onnx.sh ../cocodataset/ val2017 ../SSDMOBILE.onnx /home/workspace/ssd/cocodataset/annotations/instances_{}.json GPU ../config/ssd_mobilenet_v1_fpn_ONNX_config.yaml
```

#### Result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.351
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.522
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.353
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.548
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.702
mAP: 0.3510597974549167
```

## [Model Description](#contents)

### [Performance](#contents)

| Parameters          | Ascend                      | GPU                         | CPU                         |
| ------------------- | --------------------------- | --------------------------- | --------------------------- |
| Model Version       | SSD MobileNetV2             | SSD MobileNetV2             | SSD MobileNetV2             |
| Resource            | Ascend 910; OS Euler2.8     | GPU(Tesla V100 PCIE)，CPU 2.1GHz 64 cores，Memory 128G                         | CPU                         |
| Uploaded Date       | 07/05/2020 (month/day/year) | 09/24/2020 (month/day/year) | 11/05/2020 (month/day/year) |
| MindSpore Version   | 1.3.0                       | 1.3.0                       | 1.3.0                       |
| Dataset             | COCO2017                    | COCO2017                    | helmet                      |
| Training Parameters | epoch = 500,  batch_size = 32   | epoch = 800,  batch_size = 24(8ps)/32(1ps) | epoch = 40,  batch_size = 16 |
| Optimizer           | Momentum                    | Momentum                    | Momentum                    |
| Loss Function       | Sigmoid Cross Entropy,SmoothL1Loss  | Sigmoid Cross Entropy,SmoothL1Loss | Sigmoid Cross Entropy,SmoothL1Loss |
| Speed               | 8pcs: 90ms/step             | 8pcs: 121ms/step            | 1230ms/step                 |
| Total time          | 8pcs: 4.81hours             | 8pcs: 12.31hours            | 1.2h                        |
| outputs             | mAP                         | mAP                         | mAP                         |
| Accuracy            | IoU=0.50: 22%               | IoU=0.50: 22%               | IoU=0.50: 49.0%             |
| Model for inference | 34M(.ckpt file)             | 34M(.ckpt file)             | 34M(.ckpt file)             |
| configuration       | ssd300_config.yaml          |ssd300_config_gpu.yaml       | ssd300_config.yaml          |
| Scripts             | <https://gitee.com/mindspore/models/tree/master/official/cv/SSD> ||

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | SSD-MobileNetV1-FPN         | SSD-MobileNetV1-FPN         |
| Resource            | Ascend 910; OS Euler2.8     | GPU(Tesla V100 PCIE)，CPU 2.1GHz 64 cores，Memory 128G                         |
| Uploaded Date       | 11/14/2020 (month/day/year) | 07/23/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       | 1.3.0                       |
| Dataset             | COCO2017                    | COCO2017                    |
| Training Parameters | epoch = 60,  batch_size = 32   | epoch = 60,  batch_size = 16 |
| Optimizer           | Momentum                     | Momentum                   |
| Loss Function       | Sigmoid Cross Entropy,SmoothL1Loss  | Sigmoid Cross Entropy,SmoothL1Loss |
| Speed               | 8pcs: 408 ms/step             | 8pcs: 640 ms/step            |
| Total time          | 8pcs: 4.5 hours             | 8pcs: 9.7 hours            |
| outputs             | mAP                         | mAP                         |
| Accuracy            | IoU=0.50: 29.1 %             | IoU=0.50: 29.1 %             |
| Model for inference | 96M(.ckpt file)             | 96M(.ckpt file)             |
| configuration           | ssd_mobilenet_v1_fpn_config.yaml  |ssd_mobilenet_v1_fpn_config_gpu.yaml       |
| Scripts             | <https://gitee.com/mindspore/models/tree/master/official/cv/SSD> ||

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | SSD-Resnet50-FPN             | SSD-Resnet50-FPN             |
| Resource            | Ascend 910; OS Euler2.8     | GPU(Tesla V100 PCIE)，CPU 2.1GHz 64 cores，Memory 128G                         |
| Uploaded Date       | 03/10/2021 (month/day/year) | 07/23/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       | 1.3.0                       |
| Dataset             | COCO2017                    | COCO2017                    |
| Training Parameters | epoch = 60,  batch_size = 32   | epoch = 60,  batch_size = 16 |
| Optimizer           | Momentum                     | Momentum                   |
| Loss Function       | Sigmoid Cross Entropy,SmoothL1Loss  | Sigmoid Cross Entropy,SmoothL1Loss |
| Speed               | 8pcs: 345 ms/step             | 8pcs: 877 ms/step            |
| Total time          | 8pcs: 4.1 hours             | 8pcs: 12 hours            |
| outputs             | mAP                         | mAP                         |
| Accuracy            | IoU=0.50: 34.3%            | IoU=0.50: 34.3 %           |
| Model for inference | 255M(.ckpt file)             | 255M(.ckpt file)             |
| configuration           | ssd_resnet50_fpn_config.yaml | ssd_resnet50_fpn_config_gpu.yaml       |
| Scripts             | <https://gitee.com/mindspore/models/tree/master/official/cv/SSD> ||

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | SSD VGG16                   | SSD VGG16                   |
| Resource            | Ascend 910; OS Euler2.8     | GPU(Tesla V100 PCIE)，CPU 2.1GHz 64 cores，Memory 128G                         |
| Uploaded Date       | 03/27/2021 (month/day/year) | 07/23/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       | 1.3.0                       |
| Dataset             | COCO2017                    | COCO2017                    |
| Training Parameters | epoch = 150,  batch_size = 32   | epoch = 150,  batch_size = 32 |
| Optimizer           | Momentum                     | Momentum                   |
| Loss Function       | Sigmoid Cross Entropy,SmoothL1Loss  | Sigmoid Cross Entropy,SmoothL1Loss |
| Speed               | 8pcs: 117 ms/step             | 8pcs: 403 ms/step            |
| Total time          | 8pcs: 4.81hours             | 8pcs: 16.8 hours            |
| outputs             | mAP                         | mAP                         |
| Accuracy            | IoU=0.50: 23.2%               | IoU=0.50: 23.2%             |
| Model for inference | 186M(.ckpt file)             | 186M(.ckpt file)             |
| configuration           | ssd_vgg16_config.yaml      | ssd_vgg16_config_gpu.yaml      |
| Scripts             | <https://gitee.com/mindspore/models/tree/master/official/cv/SSD> ||

| Parameters          |                        GPU                         |
| ------------------- | ------------------------------------------------------ |
| Model Version       | SSD MobileNetV1          |
| Resource            | GPU(Tesla V100 PCIE)，CPU 2.1GHz 64 cores，Memory 128G   |
| Uploaded Date       | 03/03/2022 (month/day/year) |
| MindSpore Version   | 1.5.0                     |
| Dataset             | COCO2017                    |
| Training Parameters | epoch = 500,  batch_size = 32  |
| Optimizer           | Momentum                        |
| Loss Function       | Sigmoid Cross Entropy,SmoothL1Loss    |
| Speed               | 8pcs: 108 ms/step                     |
| Total time          | 8pcs: 6.87hours                      |
| outputs             | mAP                                          |
| Accuracy            | IoU=0.50: 21.5%                       |
| Model for inference | 88M(.ckpt file)                        |
| configuration           | ssd_mobilenet_v1_300_config_gpu.yaml    |
| Scripts             | <https://gitee.com/mindspore/models/tree/master/official/cv/SSD> |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
