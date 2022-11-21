# Contents

- [Contents](#contents)
- [Faster R-CNN-DCN description](#faster-r-cnn-dcn-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Quick start](#quick-start)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Training process](#training-process)
        - [Usage](#usage)
        - [Result](#result)
    - [Evaluation process](#evaluation-process)
        - [Usage](#usage)
        - [Result](#result)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training performance](#training-performance)
        - [Evaluation performance](#evaluation-performance)
- [ModelZoo Home page](#modelzoo-home-page)

<!-- /TOC -->

# Faster R-CNN-DCN description

Before Faster R-CNN, the target detection network relied on region proposal algorithms to hypothesize the location of the target, such as SPPNet, Fast R-CNN, etc. The research results show that the inference time of these detection networks is shortened, but the calculation of the region proposals is still a bottleneck.

Faster R-CNN proposes that convolution feature maps based on region detectors (such as Fast R-CNN) can also be used to generate region candidates. Building a region proposal network (RPN) on top of these convolutional features requires adding some additional convolutional layers (sharing the convolutional features of the full image with the detection network, which can perform region proposals almost at no cost), while outputting the bounding boxes coordinates and objectivity scores. Therefore, RPN is a fully convolutional network that can be trained end-to-end to generate high-quality region proposals, and then sent to Fast R-CNN for detection.

[Paper](https://arxiv.org/abs/1506.01497):   Ren S , He K , Girshick R , et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 39(6).

# Deformable Convolution description

In recent years, convolutional neural networks fast progress in the field of computer vision, and have many applications in the field of image recognition, semantic segmentation, and object detection. However, due to the fixed geometric structure of convolutional neural networks, the geometric deformation to model is limited, so Deformable Convolution is proposed.

Deformable convolution adds extra offsets to the spatial sampling positions of the convolution kernel in the convolutional layer, and learns the offsets from the target task without additional supervision. Since deformable convolution makes the shape of the convolution kernel not only a rectangular frame, but also closer to the feature extraction target, it can extract the features we want more accurately.

The V2 version of deformable convolution is used in this network.

[Paper](https://arxiv.org/pdf/1811.11168):   Zhu X, Hu H, Lin S, et al. Deformable convnets v2: More deformable, better results[C].Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 9308-9316.

# Model architecture

Faster R-CNN-DCN is a two-stage target detection network. The network uses RPN, which can share the convolutional features of the entire image with the detection network, and can perform region candidate calculations almost cost-free. The entire network further merges RPN and Fast R-CNN into one network by sharing convolutional features.

By adding a deformable convolutional network, the convolutional layer in the 3-5 stage of resnet is replaced with a deformable convolutional layer, so that the shape of the convolution kernel is closer to the feature, and the desired feature can be extracted more accurately.

# Dataset

Dataset used: [COCO 2017](<https://cocodataset.org/>)

- Dataset size: 19G
    - Training set: 18G，118,000 Images
    - Validation set: 1G，5000 Images
    - Labels set: 241M，Instances，Captions，person_keypoints class
- Data Format: Images and json files
    - Note: The data is processed in dataset.py.

# Environmental requirements

- Hardware（Ascend/GPU）
    - Use the Ascend processor to build the hardware environment.
- Get the Docker image
    - [Ascend Hub](https://ascend.huawei.com/ascendhub/#/home)

- Install[MindSpore](https://www.mindspore.cn/install).

- Download the data set COCO 2017.

- This example uses COCO 2017 as the training data set by default, but you can also use your own data set.

    1. If the COCO data set is used, **select the data set COCO when executing the script.**
        Install Cython and pycocotool.

        ```python
        pip install Cython

        pip install pycocotools
        ```

        Change COCO_ROOT and other required settings in `default_config.yaml` or `default_config_gpu.yaml` according to the running needs of the model. The directory structure is as follows:

        ```path
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017

        ```

    2. If you use your own data set, **select the data set as other when executing the script.**
       Organize the data set information into a TXT file, the content of each line is as follows:

        ```txt
        train2017/0000001.jpg 0,259,401,459,7,0 35,28,324,201,2,0 0,30,59,80,2,0
        ```

        Each row is an image and a label divided by space, the first column is the relative path of the image, and the rest are boxes in the format of [xmin,ymin,xmax,ymax,class,is_crowd], the class and the information about whether it is a group of objects. Read the image from the image path of `image_dir` (data set directory) and the relative path of `anno_path` (TXT file path). `image_dir` and `anno_path` can be set in `config_50.yaml, config_101.yaml or config_152.yaml`.

# Quick start

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:

Notice:

1. It takes a long time to generate the MindRecord file for the first run.
2. The pre-trained model is a ResNet-50 checkpoint trained on ImageNet2012. You can use the [resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) script in ModelZoo to train.
3. BACKBONE_MODEL is trained through the ResNet-50 [resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) script in modelzoo.
4. PRETRAINED_MODEL is the converted weight file. VALIDATION_JSON_FILE is a label file. CHECKPOINT_PATH is the checkpoint file after training.

> For GPU training please use [GPU pretrained ResNet-50 model](https://download.mindspore.cn/model_zoo/r1.3/resnet50_gpu_v130_imagenet_official_cv_bs32_acc0/) (resnet50_gpu_v130_imagenet_official_cv_bs32_acc0)

## Run on Ascend

```shell

# Stand-alone training
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# Distributed training
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# Evaluation
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# Inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

## Run on GPU

Use [pretrained ResNet-50 model](https://download.mindspore.cn/model_zoo/r1.3/resnet50_gpu_v130_imagenet_official_cv_bs32_acc0/) (resnet50_gpu_v130_imagenet_official_cv_bs32_acc0)

```shell

# Stand-alone training
bash run_standalone_train_gpu.sh [PRETRAINED_MODEL] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](option)

# Distributed training
bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL] [COCO_ROOT] [MINDRECORD_DIR](option)

# Evaluation
bash run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](option)

```

- ModelArts for training (If you want to run on modelarts, you can refer to the following documents [modelarts](https://support.huaweicloud.com/modelarts/))

    ```python
    # Using 8 card training on ModelArts
    # (1) Execute a or b
    #       a. Set "enable_modelarts=True" in the default_config.yaml file
    #          Set "distribute=True" in the default_config.yaml file
    #          Set "dataset_path='/cache/data'" in the default_config.yaml file
    #          Set "epoch_size: 20" in the default_config.yaml file
    #          (Optional) Set "checkpoint_url='s3://dir_to_your_pretrained/'" in the default_config.yaml file
    #          Set other parameters in the default_config.yaml file
    #       b. Set "enable_modelarts=True" on the web page
    #          Set "distribute=True" on the web page
    #          Set "dataset_path=/cache/data" on the web page
    #          Set "epoch_size: 20" on the web page
    #          (Optional) Set "checkpoint_url='s3://dir_to_your_pretrained/'" on the web page
    #          Set other parameters on the web page
    # (2) Prepare model code
    # (3) If you choose to fine-tune your model, please upload your pre-trained model to the S3 bucket
    # (4) Perform a or b (recommended to choose a)
    #       a. First, compress the data set into a ".zip" file.
    #          Second, upload your compressed data set to the S3 bucket (You can also upload uncompressed data sets, but that may be very slow.)
    #       b. Upload the original data set to the S3 bucket.
    #           (Data set conversion occurs during the training process, which takes more time. The conversion will be performed every time you train.)
    # (5) Set your code path on the web page to "/path/faster_rcnn"
    # (6) Set the startup file to "train.py" on the web page
    # (7) Set "training data set", "training output file path", "job log path", etc. on the web page
    # (8) Start training
    #
    # Use single card training on ModelArts
    # (1) Execute a or b
    #       a. Set "enable_modelarts=True" in the default_config.yaml file
    #          Set "dataset_path='/cache/data'" in the default_config.yaml file
    #          Set "epoch_size: 20" in the default_config.yaml file
    #          (Optional) Set "checkpoint_url='s3://dir_to_your_pretrained/'" in the default_config.yaml file
    #          Set other parameters in the default_config.yaml file
    #       b. Set "enable_modelarts=True" on the web page
    #          Set "dataset_path='/cache/data'" on the web page
    #          Set "epoch_size: 20" on the web page
    #          (Optional) Set "checkpoint_url='s3://dir_to_your_pretrained/'" on the web page
    #          Set other parameters on the web page
    # (2) Prepare model code
    # (3) If you choose to fine-tune your model, upload your pre-trained model to the S3 bucket
    # (4) Perform a or b (recommended to choose a)
    #       a. First, compress the data set into a ".zip" file.
    #          Second, upload your compressed data set to the S3 bucket (You can also upload uncompressed data sets, but that may be very slow.)
    #       b. Upload the original data set to the S3 bucket.
    #           (Data set conversion occurs during the training process, which takes more time. The conversion will be performed every time you train.)
    # (5) Set your code path on the web page to "/path/faster_rcnn"
    # (6) Set the startup file to "train.py" on the web page
    # (7) Set "training data set", "training output file path", "job log path", etc. on the web page
    # (8) Create training job
    #
    # Use single card evaluation on ModelArts
    # (1) Execute a or b
    #       a. Set "enable_modelarts=True" in the default_config.yaml file
    #          Set "checkpoint_url='s3://dir_to_your_trained_model/'" in the default_config.yaml file
    #          Set "checkpoint='./faster_rcnn/faster_rcnn_trained.ckpt'" in the default_config.yaml file
    #          Set "dataset_path='/cache/data'" in the default_config.yaml file
    #          Set other parameters in the default_config.yaml file
    #       b. Set "enable_modelarts=True" on the web page
    #          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on the webpage
    #          Set "checkpoint='./faster_rcnn/faster_rcnn_trained.ckpt'" on the webpage
    #          Set "dataset_path='/cache/data'" on the web page
    #          Set other parameters on the web page
    # (2) Prepare model code
    # (3) Upload your trained model to the S3 bucket
    # (4) Perform a or b (recommended to choose a)
    #       a. First, compress the data set into a ".zip" file.
    #          Second, upload your compressed data set to the S3 bucket (You can also upload uncompressed data sets, but that may be very slow.)
    #       b. Upload the original data set to the S3 bucket.
    #           (Data set conversion occurs during the training process, which takes more time. The conversion will be performed every time you train.)
    # (5) Set your code path on the web page to "/path/faster_rcnn"
    # (6) Set the startup file to "eval.py" on the web page
    # (7) Set "training data set", "training output file path", "job log path", etc. on the web page
    # (8) Create training job
    ```

# Script description

## Script and sample code

```shell
.
└─faster_rcnn_dcn
  ├─README.md                        // Faster R-CNN related instructions
  ├─ascend310_infer                  // Implement 310 inference source code
  ├─scripts
    ├─run_standalone_train_ascend.sh // Ascend stand-alone shell script
    ├─run_standalone_train_gpu.sh    // GPU stand-alone shell script
    ├─run_distribute_train_ascend.sh // Ascend distributed shell script
    ├─run_distribute_train_gpu.sh    // GPU distributed shell script
    ├─run_infer_310.sh               // Ascend distributed shell script
    ├─run_eval_ascend.sh             // Ascend distributed shell script
    └─run_eval_gpu.sh                // GPU distributed shell script
  ├─src
    ├─FasterRcnn
      ├─__init__.py                  // init file
      ├─anchor_generator.py          // Anchor generator
      ├─bbox_assign_sample.py        // First stage sampler
      ├─bbox_assign_sample_stage2.py // Second stage sampler
      ├─dcn_v2.py                    // Variable convolutional V2 network
      ├─faster_rcnn_resnet50.py      // Faster R-CNN network with Resnet50 as the backbone
      ├─fpn_neck.py                  // Feature Pyramid Network
      ├─proposal_generator.py        // Candidate generator
      ├─rcnn.py                      // R-CNN network
      ├─resnet.py                    // Backbone network
      ├─roi_align.py                 // ROI alignment network
      └─rpn.py                       // Regional candidate network
    ├─dataset.py                     // Create and process the data set
    ├─lr_schedule.py                 // Learning rate generator
    ├─network_define.py              // Faster R-CNN network definition
    ├─util.py                        // Evaluation related operations
    └─model_utils
      ├─config.py                    // Get .yaml configuration parameters
      ├─device_adapter.py            // Get the id on the cloud
      ├─local_adapter.py             // Get local id
      └─moxing_adapter.py            // Data preparation on the cloud
  ├─default_config.yaml              // Resnet50 related configuration for Ascend
  ├─default_config_gpu.yaml          // Resnet50 related configuration for GPU
  ├─export.py                        // Script to export AIR, MINDIR model
  ├─eval.py                          // Evaluation script
  ├─postprogress.py                  // 310 inference post-processing script
  └─train.py                         // Training script
```

## Training process

### Usage

#### Run on Ascend

```shell
# Ascend stand-alone training
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [COCO_ROOT] [MINDRECORD_DIR](option)

# Ascend distributed training
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [COCO_ROOT] [MINDRECORD_DIR](option)
```

#### Run on GPU

```shell
# GPU stand-alone training
bash run_standalone_train_gpu.sh [PRETRAINED_MODEL] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](option)

# GPU distributed training
bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL] [COCO_ROOT] [MINDRECORD_DIR](option)
```

Notes:

1. The rank_table.json specified by RANK_TABLE_FILE is required to run distributed tasks. You can use [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools) to generate this file.
2. PRETRAINED_MODEL should be a trained ResNet-50 checkpoint. If you need to load the checkpoints of the trained FasterRcnn, you need to modify train.py as follows:

```python
# Comment out the following code
#   load_path = args_opt.pre_trained
#    if load_path != "":
#        param_dict = load_checkpoint(load_path)
#        for item in list(param_dict.keys()):
#            if not item.startswith('backbone'):
#                param_dict.pop(item)
#        load_param_into_net(net, param_dict)

# When loading the trained FasterRcnn checkpoint, you need to load the network parameters and optimizer to the model, so you can add the following code after defining the optimizer:
    lr = Tensor(dynamic_lr(config, rank_size=device_num), mstype.float32)
    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    if load_path != "":
        param_dict = load_checkpoint(load_path)
        for item in list(param_dict.keys()):
            if item in ("global_step", "learning_rate") or "rcnn.reg_scores" in item or "rcnn.cls_scores" in item:
                param_dict.pop(item)
        load_param_into_net(opt, param_dict)
        load_param_into_net(net, param_dict)
```

3. defaule_config.yaml contains the original data set path, you can choose "coco_root" or "image_dir".

### Result

The training results are saved in the example path, and the folder name starts with "train" or "train_parallel". You can find the checkpoint file and results in loss_rankid.log, as shown below.

```log
# Distributed training results（8P）
339 epoch: 1 step: 1 total_loss: 5.00443
340 epoch: 1 step: 2 total_loss: 1.09367
340 epoch: 1 step: 3 total_loss: 0.90158
...
346 epoch: 1 step: 15 total_loss: 0.31314
347 epoch: 1 step: 16 total_loss: 0.84451
347 epoch: 1 step: 17 total_loss: 0.63137
```

## Evaluation process

### Usage

#### Run on Ascend

```shell
# Ascend evaluation
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [COCO_ROOT] [MINDRECORD_DIR](option)
```

> Generate checkpoints during training.
>
> The number of images in the data set must be the same as the number of tags in the VALIDATION_JSON_FILE file, otherwise the accuracy result display format may be abnormal.

#### Run on GPU

```shell
# GPU evaluation
bash run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](option)
```

> Generate checkpoints during training.
>
> The number of images in the data set must be the same as the number of tags in the VALIDATION_JSON_FILE file, otherwise the accuracy result display format may be abnormal.

### Result on Ascend

The evaluation result will be saved in the example path, the folder name is "eval". Under this folder, you can find results similar to the following in the log.

```log
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.406
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.624
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.441
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.264
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.439
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.533
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.517
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.541
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.384
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.577
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.675
```

### Result on GPU

The evaluation result will be saved in the example path, the folder name is "eval". Under this folder, you can find results similar to the following in the log.

```log
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.402
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.615
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.434
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.256
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.429
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.522
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.331
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.516
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.540
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.374
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.570
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.670
```

## Model export

```shell
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` Optional ["AIR", "MINDIR"]

## Inference process

### Instructions

It is necessary to complete the export of the model in the Shengteng 910 environment before inference. The following example only supports mindir inference with batch_size=1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

### Result on Ascend

The result of the inference is saved in the current directory, and the result similar to the following can be found in the acc.log log file.

```log
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.403
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.620
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.434
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.252
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.436
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.523
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.328
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.513
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.536
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.370
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.573
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.667

```

### Result on GPU

The result of the inference is saved in the current directory, and the result similar to the following can be found in the acc.log log file.

```log
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.402
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.615
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.434
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.256
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.429
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.522
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.331
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.516
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.540
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.374
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.570
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.670

```

# Model description

## Performance

### Training performance

| parameter |Ascend | GPU |
| -------------------------- | -----------------------------------------------------------|------------------------------------------------------------|
| Model version              | V1                                                         | V1                                                         |
| resource                   | Ascend 910；CPU 2.60GHz, 192 cores；RAM：755G               | GeForce RTX 3090;CPU 2.90GHz, 64 cores;RAM:252G            |
| Upload date                | 2021/11/5                                                  | 2021/11/13                                                 |
| MindSpore Version          | 1.3.0                                                      | 1.5.0rc1                                                   |
| Dataset                    | COCO 2017                                                  | COCO 2017                                                  |
| Training parameters        | epoch=70, batch_size=2                                     | epoch=72, batch_size=2                                     |
| Optimizer                  | SGD                                                        | SGD                                                        |
| Loss function              | Softmax Cross entropy, Sigmoid Cross entropy, SmoothL1Loss | Softmax Cross entropy, Sigmoid Cross entropy, SmoothL1Loss |
| speed                      | 8 cards:448 milliseconds/step                              | 8 cards:655 milliseconds/step                              |
| total time                 | 8 cards:66.2 hours                                         | 8 cards:96,8 hours                                         |
| parameters(M)              | 486                                                        | 486                                                        |

### Evaluation performance

| parameter | Ascend | GPU |
| ------------------- | ----------------- | ----------------- |
| Model version       | V1                | V1                |
| resource            | Ascend 910        | GeForce RTX 3090  |
| Upload date         | 2021/11/5         | 2021/11/13        |
| MindSpore Version   | 1.3.0             | 1.5.0rc1          |
| Dataset             | COCO2017          | COCO2017          |
| batch_size          | 2                 | 2                 |
| Output              | mAP               | mAP               |
| Accuracy            | IoU=0.50:62.0%    | IoU=0.50:61.5%    |
| Evaluation model    | 486M (.ckpt file) | 486M (.ckpt file) |

# ModelZoo home page

Please visit the official website [homepage](https://gitee.com/mindspore/models).
