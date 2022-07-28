# Contents

- [Contents](#contents)
    - [DETR description](#detr-description)
    - [Model architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment requirements](#environment-requirements)
    - [Quick start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
        - [Evaluation Process](#evaluation-process)
        - [Export MINDIR](#export-mindir)
    - [Model Description](#model-description)
        - [Training Performance on GPU](#training-performance-gpu)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [DETR description](#contents)

Detection transformer or DETR is a new method that views object detection as a direct set prediction problem. This
approach streamlines the detection pipeline, effectively removing the need for many hand-designed components
like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task.
The main ingredients of DETR are a set-based global loss that forces unique predictions via bipartite matching,
and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons
about the relations of the objects and the global image context to directly output the final set of predictions
in parallel.

> [Paper](https://arxiv.org/abs/2005.12872):  End-to-End Object Detection with Transformers.
> Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko, 2020.

## [Model architecture](#contents)

DETR contains three main components: a CNN backbone to extract a compact feature representation,
an encoder-decoder transformer, and a simple feed forward network (FFN) that makes the final detection prediction.

## [Dataset](#contents)

Dataset used: [COCO2017](https://cocodataset.org/#download)

- Dataset size：~19G
    - [Train](http://images.cocodataset.org/zips/train2017.zip) - 18G，118000 images
    - [Val](http://images.cocodataset.org/zips/val2017.zip) - 1G，5000 images
    - [Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) - 241M，instances，captions，person_keypoints etc
- Data format：image and json files
  - The directory structure is as follows:

  ```text
    .
    ├── annotations  # annotation jsons
    └── images
      ├── train2017    # train dataset
      └── val2017      # val dataset
  ```

## [Environment requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- Download [COCO2017](https://cocodataset.org/#download)

## [Quick start](#contents)

After preparing the dataset you can start training and evaluation as follows：

### [Running on GPU](#contents)

#### Train

```shell
# standalone train
bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CFG_PATH] [SAVE_PATH] [BACKBONE_PRETRAIN] [DATASET_PATH]

# distribute train
bash ./scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [CFG_PATH] [SAVE_PATH] [BACKBONE_PRETRAIN] [DATASET_PATH]
```

Example:

```shell
# standalone train
# DEVICE_ID - device number for training
# CFG_PATH - path to config
# SAVE_PATH - path to save logs and checkpoints
# BACKBONE_PRETRAIN - path to pretrained backbone
# DATASET_PATH - path to COCO dataset
bash ./scripts/run_standalone_train_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/resnet50_pretrain.ckpt /path/to/coco

# distribute train (8p)
# DEVICE_NUM - number of devices for training
# other parameters as for standalone train
bash ./scripts/run_distribute_train_gpu.sh 8 ./default_config.yaml /path/to/output /path/to/resnet50_pretrain.ckpt /path/to/coco
```

#### Evaluate

```shell
# evaluate
bash ./scripts/run_eval_gpu.sh [DEVICE_ID] [CFG_PATH] [SAVE_PATH] [CKPT_PATH] [DATASET_PATH]
```

Example:

```shell
# evaluate
# DEVICE_ID - device number for evaluating
# CFG_PATH - path to config
# SAVE_PATH - path to save logs
# CKPT_PATH - path to ckpt for evaluation
# DATASET_PATH - path to COCO dataset
bash ./scripts/run_eval_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/ckpt /path/to/coco  
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
.
└── detr
    ├── model_utils
    │   ├── __init__.py                         # init file
    │   └── config.py                           # parse arguments
    ├── scripts
    │   ├── run_distribute_train_gpu.sh         # launch distributed training(8p) on GPU
    │   ├── run_eval_gpu.sh                     # launch evaluating on GPU
    │   ├── run_export_gpu.sh                   # launch export mindspore model to mindir
    │   └── run_standalone_train_gpu.sh         # launch standalone traininng(1p) on GPU
    ├── src
    │   ├── __init__.py                         # init file
    │   ├── backbone.py                         # resnet50 model
    │   ├── box_ops.py                          # bounding box ops
    │   ├── coco_eval.py                        # coco evaluator for validate mindspore model
    │   ├── criterion.py                        # bipartite matching loss
    │   ├── dataset.py                          # coco dataset
    │   ├── detr.py                             # detr model
    │   ├── grad_ops.py                         # calculate gradients for GIOU and l1 loss
    │   ├── init_weights.py                     # model weights initialization
    │   ├── matcher.py                          # hungarian matcher for bipartite matching loss
    │   ├── position_encoding.py                # position encoding
    │   ├── transformer.py                      # transformer model
    │   ├── transforms.py                       # image augmentations
    │   └── utils.py                            # utils
    ├── __init__.py                             # init file
    ├── default_config.yaml                     # config file
    ├── eval.py                                 # evaluate mindspore model
    ├── export.py                               # export mindspore model to mindir format
    ├── README.md                               # readme file
    ├── requirements.txt                        # requirements
    └── train.py                                # train mindspore model
```

### [Script Parameters](#contents)

Training parameters can be configured in `default_config.yaml`

```text
"lr": 0.0001,                                   # learning rate
"lr_backbone": 0.00001,                         # learning rate for pretrained backbone
"epochs": 300,                                  # number of training epochs
"lr_drop": 200,                                 # epoch`s number for decay lr
"weight_decay": 0.0001,                         # weight decay
"batch_size": 4,                                # batch size
"clip_max_norm": 0.1,                           # max norm of gradients
"img_scales": [480, 512, 544, 576, 608, 640],   # scale of min(image_w, image_h) for resize with save aspect ratios
"max_img_size": 906                             # max image size
```

For more parameters refer to the contents of `default_config.yaml`.

### [Training Process](#contents)

#### [Run on GPU](#contents)

##### Standalone training

```shell
# DEVICE_ID - device number for training (0)
# CFG_PATH - path to config (./default_config.yaml)
# SAVE_PATH - path to save logs and checkpoints (/path/to/output)
# BACKBONE_PRETRAIN - path to pretrained backbone (/path/to/resnet50_pretrain.ckpt)
# DATASET_PATH - path to COCO dataset (/path/to/coco)
bash ./scripts/run_standalone_train_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/resnet50_pretrain.ckpt /path/to/coco
```

Logs will be saved to `/path/to/output/log.txt`

##### Distribute training (8p)

```shell
# DEVICE_NUM - number of devices for training (8)
# other parameters as for standalone train
bash ./scripts/run_distribute_train_gpu.sh 8 ./default_config.yaml /path/to/output /path/to/resnet50_pretrain.ckpt /path/to/coco
```

Logs will be saved to `/path/to/output/log.txt`

Result:

```text
...
DATE TIME epoch:0, iter:1000, loss:29.56565077591855, fps:46.35 imgs/sec
DATE TIME epoch:0, iter:1100, loss:22.849745480075857, fps:46.21 imgs/sec
DATE TIME epoch:0, iter:1200, loss:22.51765294163112, fps:46.32 imgs/sec
DATE TIME epoch:0, iter:1300, loss:26.24592643387885, fps:46.41 imgs/sec
DATE TIME epoch:0, iter:1400, loss:19.893030108011338, fps:46.35 imgs/sec
DATE TIME epoch:0, iter:1500, loss:23.800480195525267, fps:46.39 imgs/sec
DATE TIME epoch:0, iter:1600, loss:25.929977759383746, fps:46.35 imgs/sec
DATE TIME epoch:0, iter:1700, loss:25.51915783277105, fps:46.31 imgs/sec
DATE TIME epoch:0, iter:1800, loss:19.74402231897486, fps:46.21 imgs/sec
DATE TIME epoch:0, iter:1900, loss:23.77527821792139, fps:46.37 imgs/sec
DATE TIME epoch:0, iter:2000, loss:21.950026585584542, fps:46.54 imgs/sec
...
```

### [Evaluation Process](#contents)

#### GPU

```shell
bash ./scripts/run_eval_gpu.sh [DEVICE_ID] [CFG_PATH] [SAVE_PATH] [CKPT_PATH] [DATASET_PATH]
```

Example:

```shell
# DEVICE_ID - device number for evaluating (0)
# CFG_PATH - path to config (./default_config.yaml)
# SAVE_PATH - path to save logs (/path/to/output)
# CKPT_PATH - path to ckpt for evaluation (/path/to/ckpt)
# DATASET_PATH - path to COCO dataset (/path/to/coco)
bash ./scripts/run_eval_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/ckpt /path/to/coco
```

Logs will be saved to `/path/to/output/log_eval.txt`.

Result:

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.393
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.592
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.412
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.159
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.427
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.602
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.259
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.599
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.793
```

### [Export MINDIR](#contents)

If you want to infer the network on Ascend 310, you should convert the model to MINDIR.

#### GPU

```shell
bash ./scripts/run_export_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH]
```

Example:

```shell
# DEVICE_ID - device number (0)
# CFG_PATH - path to config (./default_config.yaml)
# CKPT_PATH - path to ckpt for evaluation (/path/to/ckpt)
bash ./scripts/run_export_gpu.sh 0 ./default_config.yaml /path/to/ckpt
```

Logs will be saved to parent dir of ckpt, converted model will have the same name as ckpt except extension.

## [Model Description](#contents)

### [Training Performance on GPU](#contents)

| Parameter           | DETR (8p)                                                                    |
|---------------------|------------------------------------------------------------------------------|
| Resource            | 8x Nvidia RTX 3090                                                           |
| Uploaded date       | 15.03.2022                                                                   |
| Mindspore version   | 1.5.0                                                                        |
| Dataset             | COCO2017                                                                     |
| Training parameters | epoch=300, lr=0.0001, lr_backbone=0.00001, weight_decay=0.0001, batch_size=4 |
| Optimizer           | AdamWeightDecay                                                              |
| Loss function       | L1, GIOU loss, SoftmaxCrossEntropyWithLogits                                 |
| Speed               | 46.4 fps                                                                     |
| mAP0.5:0.95         | 39.3                                                                         |
| mAP0.5              | 59.2                                                                         |

## [Description of Random Situation](#contents)

`train.py` script use mindspore.set_seed() to set global random seed, which can be modified.  

## [ModelZoo Homepage](#contents)

Please visit the official website [homepage](https://gitee.com/mindspore/models).
