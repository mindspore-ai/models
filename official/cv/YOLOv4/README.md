# Contents

- [Contents](#contents)
- [YOLOv4 Description](#yolov4-description)
- [Model Architecture](#model-architecture)
- [Pretrain Model](#pretrain-model)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
        - [Transfer Training](#transfer-training)
    - [Evaluation Process](#evaluation-process)
        - [Valid](#valid)
        - [Test-dev](#test-dev)
    - [Convert Process](#convert-process)
        - [Convert](#convert)
    - [Inference Process](#inference-process)
        - [Usage](#usage)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [YOLOv4 Description](#contents)

YOLOv4 is a state-of-the-art detector which is faster (FPS) and more accurate (MS COCO AP50...95 and AP50) than all available alternative detectors.
YOLOv4 has verified a large number of features, and selected for use such of them for improving the accuracy of both the classifier and the detector.
These features can be used as best-practice for future studies and developments.

[Paper](https://arxiv.org/pdf/2004.10934.pdf):
Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection[J]. arXiv preprint arXiv:2004.10934, 2020.

# [Model Architecture](#contents)

YOLOv4 choose CSPDarknet53 backbone, SPP additional module, PANet path-aggregation neck, and YOLOv4 (anchor based) head as the architecture of YOLOv4.

# [Pretrain Model](#contents)

YOLOv4 needs a CSPDarknet53 backbone to extract image features for detection. The pretrained checkpoint trained with ImageNet2012 can be downloaded at [hear](https://download.mindspore.cn/model_zoo/r1.2/cspdarknet53_ascend_v120_imagenet2012_official_cv_bs64_top1acc7854_top5acc9428/cspdarknet53_ascend_v120_imagenet2012_official_cv_bs64_top1acc7854_top5acc9428.ckpt).

# [Dataset](#contents)

Dataset used: [COCO2017](https://cocodataset.org/#download)
Dataset support: [COCO2017] or datasetd with the same format as MS COCO
Annotation support: [COCO2017] or annotation as the same format as MS COCO

- The directory structure is as follows, the name of directory and file is user define:

    ```text
        ├── dataset
            ├── YOLOv4
                ├── annotations
                │   ├─ train.json
                │   └─ val.json
                ├─train
                │   ├─picture1.jpg
                │   ├─ ...
                │   └─picturen.jpg
                ├─ val
                    ├─picture1.jpg
                    ├─ ...
                    └─picturen.jpg
    ```

we suggest user to use MS COCO dataset to experience our model,
other datasets need to use the same format as MS COCO.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

- After installing MindSpore via the official website, you can start training and evaluation as follows:
- Prepare the CSPDarknet53.ckpt and hccl_8p.json files, before run network.
    - Please refer to [Pretrain Model]

    - Genatating hccl_8p.json, Run the script of utils/hccl_tools/hccl_tools.py.
      The following parameter "[0-8)" indicates that the hccl_8p.json file of cards 0 to 7 is generated.

      ```
      python hccl_tools.py --device_num "[0,8)"
      ```

- Run on local

  ```text
  # The parameter of training_shape define image shape for network, default is
                     [416, 416],
                     [448, 448],
                     [480, 480],
                     [512, 512],
                     [544, 544],
                     [576, 576],
                     [608, 608],
                     [640, 640],
                     [672, 672],
                     [704, 704],
                     [736, 736].
  # It means use 11 kinds of shape as input shape, or it can be set some kind of shape.

  #run training example(1p) by python command (Training with a single scale)
  python train.py \
      --data_dir=./dataset/xxx \
      --pretrained_backbone=cspdarknet53_backbone.ckpt \
      --is_distributed=0 \
      --lr=0.1 \
      --t_max=320 \
      --max_epoch=320 \
      --warmup_epochs=4 \
      --training_shape=416 \
      --lr_scheduler=cosine_annealing > log.txt 2>&1 &

  # standalone training example(1p) by shell script (Training with a single scale)
  bash run_standalone_train.sh dataset/xxx cspdarknet53_backbone.ckpt 0

  # For Ascend device, distributed training example(8p) by shell script (Training with multi scale)
  bash run_distribute_train.sh dataset/xxx cspdarknet53_backbone.ckpt rank_table_8p.json

  # run evaluation by python command
  python eval.py \
      --data_dir=./dataset/xxx \
      --pretrained=yolov4.ckpt \
      --testing_shape=608 > log.txt 2>&1 &

  # run evaluation by shell script
  bash run_eval.sh dataset/xxx checkpoint/xxx.ckpt 0
  ```

- Train on [ModelArts](https://support.huaweicloud.com/modelarts/)

  ```python
  # Train 8p with Ascend
  # (1) Perform a or b.
  #       a. Set "enable_modelarts=True" on base_config.yaml file.
  #          Set "data_dir='/cache/data/coco/'" on base_config.yaml file.
  #          Set "checkpoint_url='s3://dir_to_your_pretrain/'" on base_config.yaml file.
  #          Set "pretrained_backbone='/cache/checkpoint_path/cspdarknet53_backbone.ckpt'" on base_config.yaml file.
  #          Set other parameters on base_config.yaml file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/coco/" on the website UI interface.
  #          Add "checkpoint_url=s3://dir_to_your_pretrain/" on the website UI interface.
  #          Add "pretrained_backbone=/cache/checkpoint_path/cspdarknet53_backbone.ckpt" on the website UI interface.
  #          Add other parameters on the website UI interface.
  # (3) Upload or copy your pretrained model to S3 bucket.
  # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (5) Set the code directory to "/path/yolov4" on the website UI interface.
  # (6) Set the startup file to "train.py" on the website UI interface.
  # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (8) Create your job.
  #
  # Train 1p with Ascend
  # (1) Perform a or b.
  #       a. Set "enable_modelarts=True" on base_config.yaml file.
  #          Set "data_dir='/cache/data/coco/'" on base_config.yaml file.
  #          Set "checkpoint_url='s3://dir_to_your_pretrain/'" on base_config.yaml file.
  #          Set "pretrained_backbone='/cache/checkpoint_path/cspdarknet53_backbone.ckpt'" on base_config.yaml file.
  #          Set "is_distributed=0" on base_config.yaml file.
  #          Set "warmup_epochs=4" on base_config.yaml file.
  #          Set "training_shape=416" on base_config.yaml file.
  #          Set other parameters on base_config.yaml file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/coco/" on the website UI interface.
  #          Add "checkpoint_url=s3://dir_to_your_pretrain/" on the website UI interface.
  #          Add "pretrained_backbone=/cache/checkpoint_path/cspdarknet53_backbone.ckpt" on the website UI interface.
  #          Add "is_distributed=0" on the website UI interface.
  #          Add "warmup_epochs=4" on the website UI interface.
  #          Add "training_shape=416" on the website UI interface.
  #          Add other parameters on the website UI interface.
  # (3) Upload or copy your pretrained model to S3 bucket.
  # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (5) Set the code directory to "/path/yolov4" on the website UI interface.
  # (6) Set the startup file to "train.py" on the website UI interface.
  # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (8) Create your job.
  #
  # Eval 1p with Ascend
  # (1) Perform a or b.
  #       a. Set "enable_modelarts=True" on base_config.yaml file.
  #          Set "data_dir='/cache/data/coco/'" on base_config.yaml file.
  #          Set "checkpoint_url='s3://dir_to_your_trained_ckpt/'" on base_config.yaml file.
  #          Set "pretrained='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
  #          Set "is_distributed=0" on base_config.yaml file.
  #          Set "per_batch_size=1" on base_config.yaml file.
  #          Set other parameters on base_config.yaml file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/coco/" on the website UI interface.
  #          Add "checkpoint_url=s3://dir_to_your_trained_ckpt/" on the website UI interface.
  #          Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
  #          Add "is_distributed=0" on the website UI interface.
  #          Add "per_batch_size=1" on the website UI interface.
  #          Add other parameters on the website UI interface.
  # (3) Upload or copy your trained model to S3 bucket.
  # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (5) Set the code directory to "/path/yolov4" on the website UI interface.
  # (6) Set the startup file to "eval.py" on the website UI interface.
  # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (8) Create your job.
  #
  # Test 1p with Ascend
  # (1) Perform a or b.
  #       a. Set "enable_modelarts=True" on base_config.yaml file.
  #          Set "data_dir='/cache/data/coco/'" on base_config.yaml file.
  #          Set "checkpoint_url='s3://dir_to_your_trained_ckpt/'" on base_config.yaml file.
  #          Set "pretrained='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
  #          Set "is_distributed=0" on base_config.yaml file.
  #          Set "per_batch_size=1" on base_config.yaml file.
  #          Set "test_nms_thresh=0.45" on base_config.yaml file.
  #          Set "test_ignore_threshold=0.001" on base_config.yaml file.
  #          Set other parameters on base_config.yaml file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/coco/" on the website UI interface.
  #          Add "checkpoint_url=s3://dir_to_your_trained_ckpt/" on the website UI interface.
  #          Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
  #          Add "is_distributed=0" on the website UI interface.
  #          Add "per_batch_size=1" on the website UI interface.
  #          Add "test_nms_thresh=0.45" on the website UI interface.
  #          Add "test_ignore_threshold=0.001" on the website UI interface.
  #          Add other parameters on the website UI interface.
  # (3) Upload or copy your trained model to S3 bucket.
  # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (5) Set the code directory to "/path/yolov4" on the website UI interface.
  # (6) Set the startup file to "test.py" on the website UI interface.
  # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (8) Create your job.
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
└─yolov4
  ├─README.md
  ├─README_CN.md
  ├─mindspore_hub_conf.py             # config for mindspore hub
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in ascend
    ├─run_distribute_train.sh         # launch distributed training(8p) in ascend
    └─run_eval.sh                     # launch evaluating in ascend
    ├─run_test.sh                     # launch testing in ascend
  ├─src
    ├─__init__.py                     # python init file
    ├─config.py                       # parameter configuration
    ├─cspdarknet53.py                 # backbone of network
    ├─distributed_sampler.py          # iterator of dataset
    ├─export.py                       # convert mindspore model to air model
    ├─initializer.py                  # initializer of parameters
    ├─logger.py                       # log function
    ├─loss.py                         # loss function
    ├─lr_scheduler.py                 # generate learning rate
    ├─transforms.py                   # Preprocess data
    ├─util.py                         # util function
    ├─yolo.py                         # yolov4 network
    ├─yolo_dataset.py                 # create dataset for YOLOV4
  ├─eval.py                           # evaluate val results
  ├─test.py#                          # evaluate test results
  └─train.py                          # train net
```

## [Script Parameters](#contents)

Major parameters train.py as follows:

```text
optional arguments:
  -h, --help            show this help message and exit
  --device_target       device where the code will be implemented: "Ascend", default is "Ascend"
  --data_dir DATA_DIR   Train dataset directory.
  --per_batch_size PER_BATCH_SIZE
                        Batch size for Training. Default: 8.
  --pretrained_backbone PRETRAINED_BACKBONE
                        The ckpt file of CspDarkNet53. Default: "".
  --resume_yolov4 RESUME_YOLOV4
                        The ckpt file of YOLOv4, which used to fine tune.
                        Default: ""
  --lr_scheduler LR_SCHEDULER
                        Learning rate scheduler, options: exponential,
                        cosine_annealing. Default: exponential
  --lr LR               Learning rate. Default: 0.001
  --lr_epochs LR_EPOCHS
                        Epoch of changing of lr changing, split with ",".
                        Default: 220,250
  --lr_gamma LR_GAMMA   Decrease lr by a factor of exponential lr_scheduler.
                        Default: 0.1
  --eta_min ETA_MIN     Eta_min in cosine_annealing scheduler. Default: 0
  --t_max T_MAX         T-max in cosine_annealing scheduler. Default: 320
  --max_epoch MAX_EPOCH
                        Max epoch num to train the model. Default: 320
  --warmup_epochs WARMUP_EPOCHS
                        Warmup epochs. Default: 0
  --weight_decay WEIGHT_DECAY
                        Weight decay factor. Default: 0.0005
  --momentum MOMENTUM   Momentum. Default: 0.9
  --loss_scale LOSS_SCALE
                        Static loss scale. Default: 64
  --label_smooth LABEL_SMOOTH
                        Whether to use label smooth in CE. Default:0
  --label_smooth_factor LABEL_SMOOTH_FACTOR
                        Smooth strength of original one-hot. Default: 0.1
  --log_interval LOG_INTERVAL
                        Logging interval steps. Default: 100
  --ckpt_path CKPT_PATH
                        Checkpoint save location. Default: outputs/
  --ckpt_interval CKPT_INTERVAL
                        Save checkpoint interval. Default: None
  --is_save_on_master IS_SAVE_ON_MASTER
                        Save ckpt on master or all rank, 1 for master, 0 for
                        all ranks. Default: 1
  --is_distributed IS_DISTRIBUTED
                        Distribute train or not, 1 for yes, 0 for no. Default:
                        1
  --rank RANK           Local rank of distributed. Default: 0
  --group_size GROUP_SIZE
                        World size of device. Default: 1
  --need_profiler NEED_PROFILER
                        Whether use profiler. 0 for no, 1 for yes. Default: 0
  --training_shape TRAINING_SHAPE
                        Fix training shape. Default: ""
  --resize_rate RESIZE_RATE
                        Resize rate for multi-scale training. Default: 10
  --transfer_train TRANSFER_TRAIN
                        Transfer training on other dataset, if set it True set filter_weight True. Default: False
```

## [Training Process](#contents)

YOLOv4 can be trained from the scratch or with the backbone named cspdarknet53.
Cspdarknet53 is a classifier which can be trained on some dataset like ImageNet(ILSVRC2012).
It is easy for users to train Cspdarknet53. Just replace the backbone of Classifier Resnet50 with cspdarknet53.
Resnet50 is easy to get in mindspore model zoo.

### Training

For Ascend device, standalone training example(1p) by shell script

```bash
bash run_standalone_train.sh dataset/coco2017 cspdarknet53_backbone.ckpt 0
```

```text
python train.py \
    --data_dir=/dataset/xxx \
    --pretrained_backbone=cspdarknet53_backbone.ckpt \
    --is_distributed=0 \
    --lr=0.1 \
    --t_max=320 \
    --max_epoch=320 \
    --warmup_epochs=4 \
    --training_shape=416 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

The python command above will run in the background, you can view the results through the file log.txt.

After training, you'll get some checkpoint files under the outputs folder by default. The loss value will be achieved as follows:

```text

# grep "loss:" train/log.txt
2020-10-16 15:00:37,483:INFO:epoch[0], iter[0], loss:8248.610352, 0.03 imgs/sec, lr:2.0466639227834094e-07
2020-10-16 15:00:52,897:INFO:epoch[0], iter[100], loss:5058.681709, 51.91 imgs/sec, lr:2.067130662908312e-05
2020-10-16 15:01:08,286:INFO:epoch[0], iter[200], loss:1583.772806, 51.99 imgs/sec, lr:4.1137944208458066e-05
2020-10-16 15:01:23,457:INFO:epoch[0], iter[300], loss:1229.840823, 52.75 imgs/sec, lr:6.160458724480122e-05
2020-10-16 15:01:39,046:INFO:epoch[0], iter[400], loss:1155.170310, 51.32 imgs/sec, lr:8.207122300518677e-05
2020-10-16 15:01:54,138:INFO:epoch[0], iter[500], loss:920.922433, 53.02 imgs/sec, lr:0.00010253786604152992
2020-10-16 15:02:09,209:INFO:epoch[0], iter[600], loss:808.610681, 53.09 imgs/sec, lr:0.00012300450180191547
2020-10-16 15:02:24,240:INFO:epoch[0], iter[700], loss:621.931513, 53.23 imgs/sec, lr:0.00014347114483825862
2020-10-16 15:02:39,280:INFO:epoch[0], iter[800], loss:527.155985, 53.20 imgs/sec, lr:0.00016393778787460178
...
```

### Distributed Training

For Ascend device, distributed training example(8p) by shell script

```bash
bash run_distribute_train.sh dataset/coco2017 cspdarknet53_backbone.ckpt rank_table_8p.json
```

The above shell script will run distribute training in the background. You can view the results through the file train_parallel[X]/log.txt. The loss value will be achieved as follows:

```text
# distribute training result(8p, dynamic shape)
...
2020-10-16 20:40:17,148:INFO:epoch[0], iter[800], loss:283.765033, 248.93 imgs/sec, lr:0.00026233625249005854
2020-10-16 20:40:43,576:INFO:epoch[0], iter[900], loss:257.549973, 242.18 imgs/sec, lr:0.00029508734587579966
2020-10-16 20:41:12,743:INFO:epoch[0], iter[1000], loss:252.426355, 219.43 imgs/sec, lr:0.00032783843926154077
2020-10-16 20:41:43,153:INFO:epoch[0], iter[1100], loss:232.104760, 210.46 imgs/sec, lr:0.0003605895326472819
2020-10-16 20:42:12,583:INFO:epoch[0], iter[1200], loss:236.973975, 217.47 imgs/sec, lr:0.00039334059692919254
2020-10-16 20:42:39,004:INFO:epoch[0], iter[1300], loss:228.881298, 242.24 imgs/sec, lr:0.00042609169031493366
2020-10-16 20:43:07,811:INFO:epoch[0], iter[1400], loss:255.025714, 222.19 imgs/sec, lr:0.00045884278370067477
2020-10-16 20:43:38,177:INFO:epoch[0], iter[1500], loss:223.847151, 210.76 imgs/sec, lr:0.0004915939061902463
2020-10-16 20:44:07,766:INFO:epoch[0], iter[1600], loss:222.302487, 216.30 imgs/sec, lr:0.000524344970472157
2020-10-16 20:44:37,411:INFO:epoch[0], iter[1700], loss:211.063779, 215.89 imgs/sec, lr:0.0005570960929617286
2020-10-16 20:45:03,092:INFO:epoch[0], iter[1800], loss:210.425542, 249.21 imgs/sec, lr:0.0005898471572436392
2020-10-16 20:45:32,767:INFO:epoch[1], iter[1900], loss:208.449521, 215.67 imgs/sec, lr:0.0006225982797332108
2020-10-16 20:45:59,163:INFO:epoch[1], iter[2000], loss:209.700071, 242.48 imgs/sec, lr:0.0006553493440151215
...
```

### Transfer Training

You can train your own model based on either pretrained classification model or pretrained detection model. You can perform transfer training by following steps.

1. Convert your own dataset to COCO style. Otherwise you have to add your own data preprocess code.
2. Change `default_config.yaml`:
   1) Set argument `labels` according to your own dataset.
   2) Set argument `transfer_train` to `True`, start to transfer training on other dataset.
   3) `pretrained_checkpoint` is the path of pretrained checkpoint, it will download pretrained checkpoint automatically if not set it.
   4) Set argument `run_eval` to `True` if you want get eval dataset mAP when training.
3. Build your own bash scripts using new config and arguments for further convenient.

## [Evaluation Process](#contents)

### Valid

```bash
python eval.py \
    --data_dir=./dataset/coco2017 \
    --pretrained=yolov4.ckpt \
    --testing_shape=608 > log.txt 2>&1 &
OR
bash run_eval.sh dataset/coco2017 checkpoint/yolov4.ckpt 0
```

The above python command will run in the background. You can view the results through the file "log.txt". The mAP of the test dataset will be as follows:

```text
# log.txt
=============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.635
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.479
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.717
```

Furthermore, set `result_view` as true, it will draw the result bbox, categories and score values to the original image and save them to the current eval path. Also can set `recommend_threshold` as true, which enables map calculation of every categories, recommends the best score threshold, and distinguishes between saving correct and negative prediction result.

### Test-dev

```bash
python test.py \
    --data_dir=./dataset/coco2017 \
    --pretrained=yolov4.ckpt \
    --testing_shape=608 > log.txt 2>&1 &
OR
bash run_test.sh dataset/coco2017 checkpoint/yolov4.ckpt 0
```

The predict_xxx.json will be found in test/outputs/%Y-%m-%d_time_%H_%M_%S/.
Rename the file predict_xxx.json to detections_test-dev2017_yolov4_results.json and compress it to detections_test-dev2017_yolov4_results.zip
Submit file detections_test-dev2017_yolov4_results.zip to the MS COCO evaluation server for the test-dev2019 (bbox) <https://competitions.codalab.org/competitions/20794#participate>
You will get such results in the end of file View scoring output log.

```text
overall performance
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.642
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.487
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.267
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.627
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.711
```

## [Convert Process](#contents)

### Convert

If you want to infer the network, you should convert the model to MINDIR:

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --keep_detect [Bool]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "ONNX", "MINDIR"]
`keep_detect` keep the detect module or not, default: True

## [Inference Process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Usage

Before performing inference, the mindir file must be exported by export script on the 910 environment.
Current batch_Size can only be set to 1. The precision calculation process needs about 70G+ memory space.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID] [ANN_FILE]
```

`DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```text
=============coco eval reulst=========
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.438
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.630
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.475
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.272
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.330
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.542
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.588
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.410
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.636
Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.716
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

YOLOv4 on 118K images(The annotation and data format must be the same as coco2017)

| Parameters                 | YOLOv4                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G; System, Euleros 2.8;|
| uploaded Date              | 10/16/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0-alpha                                                 |
| Dataset                    | 118K images                                                 |
| Training Parameters        | epoch=320, batch_size=8, lr=0.012,momentum=0.9              |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Sigmoid Cross Entropy with logits, Giou Loss                |
| outputs                    | boxes and label                                             |
| Loss                       | 50                                                          |
| Speed                      | 1p 53FPS 8p 390FPS(shape=416) 220FPS(dynamic shape)         |
| Total time                 | 48h(dynamic shape)                                          |
| Checkpoint for Fine tuning | about 500M (.ckpt file)                                     |
| Scripts                    | <https://gitee.com/mindspore/models/tree/r2.0/official/cv/YOLOv4> |

### Inference Performance

YOLOv4 on 20K images(The annotation and data format must be the same as coco test2017 )

| Parameters                 | YOLOv4                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 10/16/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0-alpha                                                 |
| Dataset                    | 20K images                                                  |
| batch_size                 | 1                                                           |
| outputs                    | box position and sorces, and probability                    |
| Accuracy                   | map >= 43.8%(shape=608)                                     |
| Model for inference        | about 500M (.ckpt file)                                     |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.
In var_init.py, we set seed for weight initialization

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
