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
            - [Evaluation while training](#evaluation-while-training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation on GPU](#evaluation-on-gpu)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [result](#result)
        - [Post Training Quantization](#post-training-quantization)
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

Inception_v2 is used as a backbone. The architecture of Inception_v2 is described in the article “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift”. It is a modification of InceptionV1 with Bath Normalization layers.

## [Dataset](#contents)

The [COCO2014](<http://images.cocodataset.org/>) dataset is used for training and validation.
We used the minival subset of the validation dataset. In the following sections, we will introduce how to run the scripts using the related dataset below.

- Dataset size：15G
    - Train: 13.5G, 82 783 images
    - Minival: 1.3G, 8 059 images
- Data format：image and json files
    - Note：Data will be processed in dataset.py

## [Environment Requirements](#contents)

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2014.

- We use COCO2014 as training dataset.
- First, install Cython, pycocotool and opencv to process data and to get evaluation result.

    ```shell
    pip install Cython

    pip install pycocotools

    pip install opencv-python
    ```

    1. Run script below to generate minival validation dataset for COCO14. Use [mininval_idx](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_minival_ids.txt) from OD API:

        ```shell
        python COCO14_minival_prepare.py \
        --val_annotation_json=$COCO14/annotations/instances_val2014.json \
        --minival_annotation_json=$COCO14/annotations/instances_minival2014.json \
        --mininval_idx=mscoco_minival_idx.txt \
        --val_images_folder=$COCO14/val2014 \
        --mininval_images_folder=$COCO14/minival2014
        ```

    2. Change the `coco_root` and other settings you need in `config/ssd_inception_v2_config_gpu.yaml`. The directory structure is as follows:

        ```shell
        .
        └─coco_dataset
          ├─annotations
            ├─instance_train2014.json
            └─instance_minival2014.json
          ├─minival2014
          └─train2014
        ```

## [Quick Start](#contents)

### Prepare the model

1. Change the dataset config in the config file `config/ssd_inception_v2_config_gpu.yaml`

### Run the scripts

After installing MindSpore via the official website, you can start training and evaluation with next parameters

- DEVICE_NUM = 8
- LR = 0.2
- DATASET = coco
- EPOCH_SIZE = 500
- BATCH_SIZE = 32

As follows:

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

# run eval on CPU
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

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└─ cv
  └─ ssd_inception_v2
    ├─ README.md                      ## descriptions about SSD
    ├─ scripts
      ├─ run_distribute_train_gpu.sh  ## shell script for distributed on gpu
      ├─ run_standalone_train_gpu.sh  ## shell script for standalone on gpu
      ├─ run_eval_gpu.sh              ## shell script for eval on gpu
    ├─ src
      ├─ __init__.py                      ## init file
      ├─ anchor_generator.py              ## anchor generator
      ├─ box_util.py                      ## bbox utils
      ├─ dataset.py                       ## create dataset and process dataset
      ├─ eval_callback.py                 ## eval callback function definition
      ├─ eval_utils.py                    ## eval utils
      ├─ feature_map_generators.py        ## utils to generate feature maps
      ├─ inception_v2.py                  ## backbone for detection model
      ├─ init_params.py                   ## parameters utils
      ├─ lr_schedule.py                   ## learning ratio generator
      ├─ ssd.py                           ## ssd architecture
      ├── model_utils
      │   ├── config.py                   ## parameter configuration
      │   ├── device_adapter.py           ## device adapter
      │   ├── local_adapter.py            ## local adapter
      │   ├── moxing_adapter.py           ## moxing adapter
    ├─ config
        ├─ ssd_inception_v2_config_gpu.yaml ## parameter configuration
    ├─ COCO14_minival_prepare.py          ## prepare minival dataset
    ├─ eval.py                            ## eval scripts
    ├─ export.py                          ## export mindir script
    ├─ train.py                           ## train scripts
```

### [Script Parameters](#contents)

  ```shell
  Major parameters in the ssd_inception_v2_config_gpu.yaml as follows:

    "device_num": 1                                  # Use device nums
    "lr": 0.2                                       # Learning rate init value
    "dataset": coco                                  # Dataset name
    "epoch_size": 500                                # Epoch size
    "batch_size": 32                                 # Batch size of input tensor
    "pre_trained": None                              # Pretrained checkpoint file path
    "pre_trained_epoch_size": 0                      # Pretrained epoch size
    "save_checkpoint_epochs": 1                      # The epoch interval between two checkpoints. By default, the checkpoint will be saved per 10 epochs
    "loss_scale": 1024                               # Loss scale
    "filter_weight": False                           # Load parameters in head layer or not. If the class numbers of train dataset is different from the class numbers in pre_trained checkpoint, please set True.
    "freeze_layer": "none"                           # Freeze the backbone parameters or not, support none and backbone.
    "run_eval": False                                # Run evaluation when training
    "save_best_ckpt": True                           # Save best checkpoint when run_eval is True
    "eval_start_epoch": 40                           # Evaluation start epoch when run_eval is True
    "eval_interval": 1                               # valuation interval when run_eval is True

    "class_num": 81                                  # Dataset class number
    "image_shape": [300, 300]                        # Image height and width used as input to the model
    "mindrecord_dir": "/data/MindRecord_COCO"        # MindRecord path
    "coco_root": "/data/coco2014"                    # COCO2014 dataset path
  ```

### [Training Process](#contents)

To train the model, run `train.py`. If the `mindrecord_dir` is empty, it will generate [mindrecord](https://www.mindspore.cn/tutorials/en/master/advanced/dataset/record.html) files by `coco_root`(coco dataset). **Note if mindrecord_dir isn't empty, it will use mindrecord_dir instead of raw images.**

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
epoch: 1 step: 320, loss is 4.008658
epoch time: 220911.886 ms, per step time: 690.350 ms
epoch: 2 step: 320, loss is 3.6807125
epoch time: 133212.011 ms, per step time: 416.288 ms
epoch: 3 step: 320, loss is 3.52385
epoch time: 129957.381 ms, per step time: 406.117 ms
...


epoch: 498 step: 320, loss is 1.3933899
epoch time: 129433.331 ms, per step time: 404.479 ms
epoch: 499 step: 320, loss is 1.3042376
epoch time: 129900.291 ms, per step time: 405.938 ms
epoch: 500 step: 320, loss is 1.4773071
epoch time: 130173.260 ms, per step time: 406.791 ms
```

#### Evaluation while training

You can add `run_eval` to start shell and set it True, if you want evaluation while training. And you can set argument option: `save_best_ckpt`, `eval_start_epoch`, `eval_interval` when `run_eval` is True.

### [Evaluation Process](#contents)

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
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.344
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.238
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.028
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.160
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.423
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.215
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.277
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.277
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.200
Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.521

========================================

mAP: 0.22372216552930263
```

## Inference Process

### [Export MindIR](#contents)

Export MindIR on local

```shell
python export.py --checkpoint_file_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --config_path [CONFIG_PATH]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters          | GPU                                                                           | GPU                                                                          |
| ------------------- | ----------------------------------------------------------------------------- |------------------------------------------------------------------------------|
| Model Version       | SSD-Inception_v2                                                              |SSD-Inception_v2                                                              |
| Resource            | V100-PICE                                                                     |V100-PICE                                                                     |
| uploaded Date       | (month/day/year)                                                              |(month/day/year)                                                              |
| MindSpore Version   | 1.5.0                                                                         |1.5.0                                                                         |
| Dataset             | COCO2014                                                                      |COCO2014                                                                      |
| Training Parameters | epoch = 500,  batch_size = 32                                                 |epoch = 500,  batch_size = 32                                                 |
| Optimizer           | Momentum                                                                      |Momentum                                                                      |
| Loss Function       | Sigmoid Cross Entropy,SmoothL1Loss                                            |Sigmoid Cross Entropy,SmoothL1Loss                                            |
| Speed               | 1pcs: 141 ms/step                                                             |8pcs: 383 ms/step                                                             |
| Total time          | 1pcs: 50.4 hours                                                              |8pcs: 17hours                                                                 |
| Parameters (M)      | 14M                                                                           |14M                                                                           |
| Scripts             | <https://gitee.com/mindspore/models/tree/master/research/cv/ssd_inception_v2>  | <https://gitee.com/mindspore/models/tree/master/research/cv/ssd_inception_v2> |

#### Inference Performance

| Parameters          | GPU                         | GPU                      |
| ------------------- | --------------------------- |--------------------------|
| Model Version       | SSD-Inception_v2             | SSD-Inception_v2          |
| Resource            | GPU                         | 8 x GPU                  |
| Uploaded Date       | (month/day/year)            | (month/day/year)         |
| MindSpore Version   | 1.5.0                       | 1.5.0                    |
| Dataset             | COCO2014                    | COCO2014                 |
| batch_size          | 1                           | 1                        |
| outputs             | mAP                         | mAP                      |
| Accuracy            | IoU=0.50: 21%               | Iout=0.50: 22.4%         |
| Model for inference | 14M(.ckpt file)             | 14M(.ckpt file)          |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
