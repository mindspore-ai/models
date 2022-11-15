# Contents

- [Contents](#contents)
    - [PointPillars description](#pointpillars-description)
    - [Model architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment requirements](#environment-requirements)
    - [Quick start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Dataset Preparation](#dataset-preparation)
        - [Training Process](#training-process)
        - [Evaluation Process](#evaluation-process)
        - [Export MINDIR](#export-mindir)
    - [Model Description](#model-description)
        - [Training Performance on GPU](#training-performance-gpu)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [PointPillars description](#contents)

PointPillars is a method for object detection in 3D that enables end-to-end learning with only 2D convolutional layers.
PointPillars uses a novel encoder that learn features on pillars (vertical columns) of the point cloud to predict 3D oriented boxes for objects.
There are several advantages of this approach.
First, by learning features instead of relying on fixed encoders, PointPillars can leverage the full information represented by the point cloud.
Further, by operating on pillars instead of voxels there is no need to tune the binning of the vertical direction by hand.
Finally, pillars are highly efficient because all key operations can be formulated as 2D convolutions which are extremely efficient to compute on a GPU.
An additional benefit of learning features is that PointPillars requires no hand-tuning to use different point cloud configurations.
For example, it can easily incorporate multiple lidar scans, or even radar point clouds.

> [Paper](https://arxiv.org/abs/1812.05784):  PointPillars: Fast Encoders for Object Detection from Point Clouds.
> Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom, 2018.

## [Model architecture](#contents)

The main components of the network are a Pillar Feature Network, Backbone, and SSD detection head.
The raw point cloud is converted to a stacked pillar tensor and pillar index tensor.
The encoder uses the stacked pillars to learn a set of features that can be scattered back to a 2D pseudo-image for a convolutional neural network.
The features from the backbone are used by the detection head to predict 3D bounding boxes for objects.

## [Dataset](#contents)

Dataset used: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

Data was collected with using a standard station wagon with two high-resolution color and grayscale video cameras.
Accurate ground truth is provided by a Velodyne laser scanner and a GPS localization system.
Dataset was captured by driving around the mid-size city of Karlsruhe, in rural areas and on highways.
Up to 15 cars and 30 pedestrians are visible per image. The 3D object detection benchmark consists of 7481 images.

## [Environment requirements](#contents)

- Hardware（GPU/Ascend）

    - Prepare hardware environment with GPU processor.
    - Prepare hardware environment with Ascend processor.
- Framework

    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), data from [ImageSets](https://github.com/traveller59/second.pytorch/tree/master/second/data/ImageSets), put files from `ImageSets` into `pointpillars/src/data/ImageSets/`

  For the dataset preparation instructions see the [Dataset preparation](#dataset-preparation) section.

## [Quick start](#contents)

After preparing the dataset you can start training and evaluation as follows：

### [Running on GPU](#contents)

#### Train

```shell
# standalone train
bash ./scripts/run_standalone_train_gpu.sh [CFG_PATH] [SAVE_PATH] [DEVICE_ID]

# distribute train
bash ./scripts/run_distribute_train_gpu.sh [CFG_PATH] [SAVE_PATH] [DEVICE_NUM]
```

Example:

```shell
# standalone train
bash ./scripts/run_standalone_train_gpu.sh ./configs/car_xyres16.yaml ./experiments/car/ 0

# distribute train (8p)
bash ./scripts/run_distribute_train_gpu.sh ./configs/car_xyres16.yaml ./experiments/car/ 8
```

#### Evaluate

```shell
# evaluate
bash ./scripts/run_eval_gpu.sh [CFG_PATH] [CKPT_PATH] [DEVICE_ID]
```

Example:

```shell
# evaluate
bash ./scripts/run_eval_gpu.sh ./configs/car_xyres16.yaml ./experiments/car/poinpitllars-160_37120.ckpt 0
```

### [Running on Ascend](#contents)

#### Train

```shell
# standalone train
bash ./scripts/run_standalone_train.sh [CFG_PATH] [SAVE_PATH] [DEVICE_ID]

# distribute train
bash ./scripts/run_distribute_train.sh [CFG_PATH] [SAVE_PATH] [RANK_SIZE] [RANK_TABLE]
```

Example:

```shell
# standalone train
bash ./scripts/run_standalone_train.sh ./configs/car_xyres16.yaml ./experiments/car/ 0

# distribute train (8p)
bash ./scripts/run_distribute_train.sh ./configs/car_xyres16.yaml ./experiments/car/ 8 /home/hccl_8p_01234567_192.168.88.13.json
```

#### Evaluate

```shell
# evaluate
bash ./scripts/run_eval.sh [CFG_PATH] [CKPT_PATH] [DEVICE_ID]
```

Example:

```shell
# evaluate
bash ./scripts/run_eval.sh ./configs/car_xyres16.yaml ./experiments/car/poinpitllars-160_37120.ckpt 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
.
└── pointpillars
    ├── ascend310_infer
    │    ├── inc
    │         └── utils.h
    │    ├── src
    │         ├── main.cc
    │         └── utils.cc
    │    ├── build.sh
    │    └── CMakeLists.txt
    ├── configs
    │    ├── car_xyres16.yaml  # config for car detection
    │    └── ped_cycle_xyres16.yaml  # config for pedestrian and cyclist detection
    ├── scripts
    │    ├── run_distribute_train_gpu.sh  # launch distributed training(8p) on GPU
    │    ├── run_eval_gpu.sh              # launch evaluating on GPU
    │    ├── run_export_gpu.sh            # launch export mindspore model to mindir
    │    ├── run_standalone_train_gpu.sh  # launch standalone traininng(1p) on GPU
    │    ├── run_distribute_train.sh
    │    ├── run_eval.sh
    │    └── run_standalone_train.sh
    ├── src
    │    ├── builder
    │    │    ├── __init__.py                       # init file
    │    │    ├── anchor_generator_builder.py       # builder for anchor generator
    │    │    ├── box_coder_builder.py              # builder for box coder
    │    │    ├── dataset_builder.py                # builder for dataset
    │    │    ├── dbsampler_builder.py              # builder for db sampler
    │    │    ├── model_builder.py                  # builder for model
    │    │    ├── preprocess_builder.py             # builder for preprocess
    │    │    ├── similarity_calculator_builder.py # builder for similarity calculator
    │    │    ├── target_assigner_builder.py        # builder for target assigner
    │    │    └── voxel_builder.py                  # builder for voxel generator
    │    ├── core
    │    │    ├── point_cloud
    │    │    │    ├── __init__.py                  # init file
    │    │    │    ├── bev_ops.py                   # ops for bev
    │    │    │    └── point_cloud_ops.py           # ops for point clouds
    │    │    ├── __init__.py                       # init file
    │    │    ├── anchor_generator.py               # anchor generator
    │    │    ├── box_coders.py                     # box coders
    │    │    ├── box_np_ops.py                     # box ops with numpy
    │    │    ├── box_ops.py                        # box ops with mindspore
    │    │    ├── einsum.py                         # einstein sum
    │    │    ├── eval_utils.py                     # utils for evaluate
    │    │    ├── geometry.py                       # geometry
    │    │    ├── losses.py                         # losses
    │    │    ├── nms.py                            # nms
    │    │    ├── preprocess.py                     # preprocess operations
    │    │    ├── region_similarity.py              # region similarity calculator
    │    │    ├── sample_ops.py                     # ops for sample data
    │    │    ├── target_assigner.py                # target assigner
    │    │    └── voxel_generator.py                # voxel generator
    │    ├── data
    │    │    ├── ImageSets                         # splits for train and val
    │    │    ├── __init__.py                       # init file
    │    │    ├── dataset.py                        # kitti dataset
    │    │    ├── kitti_common.py                   # auxiliary file for kitti
    │    │    └── preprocess.py                     # preprocess dataset
    │    ├── __init__.py                            # init file
    │    ├── create_data.py                         # create dataset for train model
    │    ├── pointpillars.py                        # pointpillars model
    │    ├── utils.py                               # utilities functions
    │    └── predict.py                             # postprocessing pointpillars`s output
    ├── __init__.py                                 # init file
    ├── eval.py                                     # evaluate mindspore model
    ├── export.py                                   # convert mindspore model to mindir
    ├── README.md                                   # readme file
    ├── requirements.txt                            # requirements
    └── train.py                                    # train mindspore model
```

### [Script Parameters](#contents)

Training parameters can be configured in `car_xyres16.yaml` for car detection or `ped_cycle_xyres16.yaml` for pedestrian
and cyclist detection.

```text
"initial_learning_rate": 0.0002,        # learning rate
"max_num_epochs": 160,                  # number of training epochs
"weight_decay": 0.0001,                 # weight decay
"batch_size": 2,                        # batch size
"max_number_of_voxels": 12000,          # mux number of voxels in one pillar
"max_number_of_points_per_voxel": 100   # max number of points per voxel
```

For more parameters refer to the contents of `car_xyres16.yaml` or `ped_cycle_xyres16.yaml`.

### [Dataset Preparation](#contents)

1. Add `/path/to/pointpillars/` to your `PYTHONPATH`

```text
export PYTHONPATH=/path/to/pointpillars/:$PYTHONPATH
```

2. Download KITTI dataset into one folder:
- [Download left color images of object data set (12 GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)
- [Download camera calibration matrices of object data set (16 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
- [Download training labels of object data set (5 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)
- [Download Velodyne point clouds, if you want to use laser information (29 GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)
3. Unzip all downloaded archives.
4. Directory structure is as follows:

```text
└── KITTI
       ├── training
       │   ├── image_2 <-- for visualization
       │   ├── calib
       │   ├── label_2
       │   ├── velodyne
       │   └── velodyne_reduced <-- create this empty directory
       └── testing
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- create this empty directory
```

5. Download [ImageSets](https://github.com/traveller59/second.pytorch/tree/master/second/data/ImageSets), put files from `ImageSets` into `pointpillars/src/data/ImageSets/`

6. Create KITTI infos:

```shell
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
```

7. Create reduced point cloud:

```shell
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
```

8. Create groundtruth-database infos:

```shell
python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
```

9. The config file `car_xyres16.yaml` or `ped_cycle_xyres16.yaml` needs to be edited to point to the above datasets:

```text
train_input_reader:
  ...
  database_sampler:
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
...
eval_input_reader:
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
```

### [Training Process](#contents)

#### [Run on GPU](#contents)

##### Standalone training

```shell
bash ./scripts/run_standalone_train_gpu.sh ./configs/car_xyres16.yaml ./experiments/cars/ 0
```

Logs will be saved to `./experiments/cars/log.txt`

Result:

```text
2022-01-18 11:29:13 epoch:0, iter:1000, loss:1.1359277, fps:14.38 imgs/sec
2022-01-18 11:29:20 epoch:0, iter:1050, loss:1.0492299, fps:13.57 imgs/sec
2022-01-18 11:29:27 epoch:0, iter:1100, loss:0.884439, fps:14.24 imgs/sec
2022-01-18 11:29:34 epoch:0, iter:1150, loss:1.0804198, fps:14.34 imgs/sec
2022-01-18 11:29:41 epoch:0, iter:1200, loss:0.92863345, fps:14.19 imgs/sec
2022-01-18 11:29:49 epoch:0, iter:1250, loss:0.8504363, fps:13.51 imgs/sec
2022-01-18 11:29:56 epoch:0, iter:1300, loss:1.0816091, fps:13.89 imgs/sec
2022-01-18 11:30:03 epoch:0, iter:1350, loss:0.98323077, fps:13.51 imgs/sec
2022-01-18 11:30:10 epoch:0, iter:1400, loss:0.824274, fps:14.33 imgs/sec
2022-01-18 11:30:18 epoch:0, iter:1450, loss:0.9153076, fps:13.77 imgs/sec
```

##### Distribute training (8p)

```shell
bash ./scripts/run_distribute_train_gpu.sh ./confgs/car_xyres16.yaml ./experiments/cars/ 8
```

Logs will be saved to `./experiments/cars/log.txt`

Result:

```text
2022-01-18 08:30:39 epoch:1, iter:300, loss:0.9105564, fps:66.5 imgs/sec
2022-01-18 08:30:51 epoch:1, iter:350, loss:1.0566418, fps:68.85 imgs/sec
2022-01-18 08:31:03 epoch:1, iter:400, loss:0.98004365, fps:65.54 imgs/sec
2022-01-18 08:31:14 epoch:1, iter:450, loss:0.9133666, fps:69.86 imgs/sec
2022-01-18 08:31:27 epoch:2, iter:500, loss:0.8083154, fps:64.09 imgs/sec
2022-01-18 08:31:39 epoch:2, iter:550, loss:0.75948864, fps:65.78 imgs/sec
2022-01-18 08:31:51 epoch:2, iter:600, loss:1.0096964, fps:66.61 imgs/sec
2022-01-18 08:32:03 epoch:2, iter:650, loss:0.86279136, fps:66.55 imgs/sec
2022-01-18 08:32:15 epoch:3, iter:700, loss:0.89273417, fps:65.99 imgs/sec
2022-01-18 08:32:27 epoch:3, iter:750, loss:0.90943766, fps:67.59 imgs/sec
```

#### [Run on Ascend](#contents)

##### Standalone training

```shell
bash ./scripts/run_standalone_train.sh ./configs/car_xyres16.yaml ./output/car/ 0
```

Logs will be saved to `./experiments/cars/log.txt`

Result:

```text
2022-08-10 15:26:18 epoch:1, iter:3450,  loss:0.58619386, fps:8.42 imgs/sec,  step time: 0.006396604252272639 ms
2022-08-10 15:26:30 epoch:1, iter:3500,  loss:0.67319953, fps:8.5 imgs/sec,  step time: 0.006337071544137494 ms
2022-08-10 15:26:42 epoch:1, iter:3550,  loss:0.5983803, fps:8.21 imgs/sec,  step time: 0.006562378385971333 ms
2022-08-10 15:26:53 epoch:1, iter:3600,  loss:0.6749635, fps:8.87 imgs/sec,  step time: 0.0060733427004567506 ms
2022-08-10 15:27:05 epoch:1, iter:3650,  loss:0.56281704, fps:8.43 imgs/sec,  step time: 0.0063904304185817985 ms
2022-08-10 15:27:17 epoch:1, iter:3700,  loss:0.7617205, fps:8.32 imgs/sec,  step time: 0.006478644907474518 ms
```

##### Distribute training (8p)

```shell
bash ./scripts/run_distribute_train.sh /home/group1/pointpillars/
configs/car_xyres16.yaml ./output/dist/ 8 /home/hccl_8p_01234567_192.168.88.13.json
```

Logs will be saved to `./experiments/cars/log.txt`

Result:

```text
2022-08-10 15:48:11 epoch:4, iter:1000,  loss:0.74541646, fps:67.11 imgs/sec,  step time: 0.051382866399041535 ms
2022-08-10 15:48:24 epoch:4, iter:1050,  loss:0.87863845, fps:61.0 imgs/sec,  step time: 0.05652883032272602 ms
2022-08-10 15:48:37 epoch:4, iter:1100,  loss:0.52257985, fps:62.88 imgs/sec,  step time: 0.05484302907154478 ms
2022-08-10 15:48:49 epoch:4, iter:1150,  loss:0.5654994, fps:66.59 imgs/sec,  step time: 0.05178481545941583 ms
2022-08-10 15:49:01 epoch:5, iter:1200,  loss:0.5621812, fps:64.18 imgs/sec,  step time: 0.05373023707291175 ms
2022-08-10 15:49:13 epoch:5, iter:1250,  loss:0.5237954, fps:67.22 imgs/sec,  step time: 0.05129807262585081 ms
```

### [Evaluation Process](#contents)

#### GPU

```shell
bash ./scripts/run_eval_gpu.sh [CFG_PATH] [CKPT_PATH] [DEVICE_ID]
```

Example:

```shell
bash ./scripts/run_eval_gpu.sh ./configs/car_xyres16.yaml ./experiments/car/pointpillars-160_37120.ckpt 0
```

Result:

Here is model for cars detection as an example，you can view the result in log file `./experiments/car/log_eval.txt`：

```text
        Easy   Mod    Hard
Car AP@0.70, 0.70, 0.70:
bbox AP:90.71, 89.17, 87.81
bev  AP:89.81, 86.99, 84.99
3d   AP:86.90, 76.88, 72.76
aos  AP:90.43, 88.47, 86.77
```

Here is result for pedestrian and cyclist detection：

```text
        Easy   Mod    Hard
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:84.63, 65.24, 62.17
bev  AP:83.25, 63.01, 59.65
3d   AP:80.13, 60.26, 56.07
aos  AP:83.79, 63.86, 60.78
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:66.80, 63.98, 60.59
bev  AP:72.15, 67.50, 62.51
3d   AP:66.48, 60.61, 55.87
aos  AP:46.26, 44.97, 43.03
```

Mod. mAP is calculated as mean of metric values for corresponding rows of each classes.

For example mod. mAP for BEV detection benchmark is calculated as `(86.99 + 63.01 + 67.50) / 3`

So mod. mAP for BEV detection benchmark is `72.5`, mod. mAP for 3D detection benchmark is `65.91`

Also, the article contains information about splits for experimental studies and for test submission,
but authors provided splits only for experimental studies.
Results in the article are presented for test submission splits.

```text
The samples are originally divided into 7481 training and 7518 testing samples.
For experimental studies we split the official training into 3712 training samples
and 3769 validation samples, while for our test submission we created a minival
set of 784 samples from the validation set and trained on the remaining 6733 samples.
```

#### Ascend

```shell
bash ./scripts/run_eval.sh [CFG_PATH] [CKPT_PATH] [DEVICE_ID]
```

Example:

```shell
bash ./scripts/run_eval.sh ./configs/car_xyres16.yaml ./experiments/car/pointpillars-160_37120.ckpt 0
```

Result:

Here is model for cars detection as an example，you can view the result in log file `./experiments/car/log_eval.txt`：

```text
        Easy   Mod    Hard
Car AP@0.70, 0.70, 0.70:
bbox AP:90.47, 88.53, 87.31
```

Here is result for pedestrian and cyclist detection：

```text
        Easy   Mod    Hard
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:84.23, 64.64, 61.48
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:65.55, 62.05, 58.94
```

### [Export MINDIR](#contents)

If you want to infer the network on Ascend 310, you should convert the model to MINDIR.

#### GPU

```shell
bash ./scripts/run_export_gpu.sh [CFG_PATH] [CKPT_PATH] [FILE_NAME] [DEVICE_ID] [FILE_FORMAT](optional)
```

- CFG_PATH - path to the configuration file
- CKPT_PATH - path to the checkpoint file, containing weights of a trained Pointpillars model
- FILE_NAME - name of the output file for the exported model.
- DEVICE_ID - ID of the device, which will be used for the export.
- FILE_FORMAT (optional) - format of the exported model. ("AIR" or "MINDIR"). Default: "MINDIR".

Example:

```shell
bash ./scripts/run_export_gpu.sh ./configs/car_xyres16.yaml ./experiments/car/pointpillars.ckpt pointpillars 0
```

or

```shell
bash ./scripts/run_export_gpu.sh ./configs/car_xyres16.yaml ./experiments/car/pointpillars.ckpt pointpillars 0 MINDIR
```

The output should look as following:

```text
pointpillars.mindir exported successfully!
```

#### Ascend

```shell
python export.py --cfg_path=cfg_path.yaml --ckpt_path=ckpt_path.ckpt --file_name=file_name --file_format=MINDIR
```

Example:

```shell
python export.py --cfg_path=./configs/car_xyres16.yaml --ckpt_path=./experiments/car/pointpillars.ckpt --file_name=pointpillars --file_format=MINDIR
```

The output should look as following:

```text
pointpillars.mindir exported successfully!
```

### [310 Infer](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported by export.py script. We only provide an example of inference using MINDIR model.

- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n', if you choose y, the kitti dataset will be preprocessd. (Recommend to use 'y' at the first running.)

#### Ascend

```shell
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [CONFIG_PATH] [DEVICE_ID]
```

Example:

```shell
bash run_infer_310.sh pointpillars.mindir y ../configs/car_xyres16.yaml 0
```

## [Model Description](#contents)

### [Training Performance](#contents)(再加两列Ascend的结果)

| Parameter           | PointPillars (1p)                                            | PointPillars (8p)                                            | PointPillars (1p)                                            | PointPillars (8p)                                            |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Resource            | 1x Nvidia RTX 3090                                           | 8x Nvidia RTX 3090                                           | Ascend                                                       | Ascend                                                       |
| Uploaded date       | -                                                            | -                                                            | -                                                            | -                                                            |
| Mindspore version   | 1.5.0                                                        | 1.5.0                                                        | 1.8.0                                                        | 1.8.0                                                        |
| Dataset             | KITTI                                                        | KITTI                                                        | KITTI                                                        | KITTI                                                        |
| Training parameters | epoch=160, lr=0.0002, weight_decay=0.0001, batch_size=2      | epoch=160, lr=0.0016, weight_decay=0.0001, batch_size=2      | poch=160, lr=0.0002, weight_decay=0.0001, batch_size=2       | epoch=160, lr=0.0016, weight_decay=0.0001, batch_size=2      |
| Optimizer           | Adam                                                         | Adam                                                         | Adam                                                         |                                                              |
| Loss function       | WeightedSmoothL1LocalizationLoss, WeightedSoftmaxClassificationLoss, SigmoidFocalClassificationLoss | WeightedSmoothL1LocalizationLoss, WeightedSoftmaxClassificationLoss, SigmoidFocalClassificationLoss | WeightedSmoothL1LocalizationLoss, WeightedSoftmaxClassificationLoss, SigmoidFocalClassificationLoss | WeightedSmoothL1LocalizationLoss, WeightedSoftmaxClassificationLoss, SigmoidFocalClassificationLoss |
| Speed               | Car: 14 img/s; Pedestrian + Cyclist: 11 img/s                | Car: 66 img/s; Pedestrian + Cyclist: 64 img/s                | Car: 8.4 img/s;                                              | Car: 64 img/s;                                               |
| mAP (BEV detection) | <table> <tr> <td></td> <td>Easy</td> <td>Moderate</td> <td>Hard</td> </tr> <tr> <td>Car</td> <td>89.81</td> <td>86.99</td> <td>85.0</td> </tr> <tr> <td>Pedestrian</td> <td>72.16</td> <td>67.5</td> <td>62.51</td> </tr> <tr> <td>Cyclist</td> <td>83.25</td> <td>63.02</td> <td>59.65</td> </tr> <tr> <td>mAP</td> <td>81.74</td> <td>**72.4**</td> <td>69.05</td> </tr> </table> | <table> <tr> <td></td> <td>Easy</td> <td>Moderate</td> <td>Hard</td> </tr> <tr> <td>Car</td> <td>89.75</td> <td>87.14</td> <td>84.58</td> </tr> <tr> <td>Pedestrian</td> <td>68.21</td> <td>62.83</td> <td>58.06</td> </tr> <tr> <td>Cyclist</td> <td>81.63</td> <td>62.75</td> <td>59.21</td> </tr> <tr> <td>mAP</td> <td>79.86</td> <td>**70.91**</td> <td>67.28</td> </tr> </table> | -                                                            | -                                                            |
| mAP (3D detection)  | <table> <tr> <td></td> <td>Easy</td> <td>Moderate</td> <td>Hard</td> </tr> <tr> <td>Car</td> <td>86.90</td> <td>76.89</td> <td>72.79</td> </tr> <tr> <td>Pedestrian</td> <td>66.48</td> <td>60.62</td> <td>55.87</td> </tr> <tr> <td>Cyclist</td> <td>80.13</td> <td>60.26</td> <td>56.07</td> </tr> <tr> <td>mAP</td> <td>77.83</td> <td>**65.92**</td> <td>61.58</td> </tr> </table> | <table> <tr> <td></td> <td>Easy</td> <td>Moderate</td> <td>Hard</td> </tr> <tr> <td>Car</td> <td>85.96</td> <td>76.33</td> <td>69.84</td> </tr> <tr> <td>Pedestrian</td> <td>57.60</td> <td>53.37</td> <td>48.35</td> </tr> <tr> <td>Cyclist</td> <td>79.53</td> <td>60.70</td> <td>56.96</td> </tr> <tr> <td>mAP</td> <td>74.36</td> <td>**63.47**</td> <td>58.38</td> </tr> </table> | -                                                            | -                                                            |
| mAOS                | <table> <tr> <td></td> <td>Easy</td> <td>Moderate</td> <td>Hard</td> </tr> <tr> <td>Car</td> <td>90.43</td> <td>88.47</td> <td>86.78</td> </tr> <tr> <td>Pedestrian</td> <td>46.27</td> <td>44.96</td> <td>43.03</td> </tr> <tr> <td>Cyclist</td> <td>83.79</td> <td>63.86</td> <td>60.78</td> </tr> <tr> <td>mAP</td> <td>73.5</td> <td>**65.76**</td> <td>63.53</td> </tr> </table> | <table> <tr> <td></td> <td>Easy</td> <td>Moderate</td> <td>Hard</td> </tr> <tr> <td>Car</td> <td>90.46</td> <td>88.27</td> <td>86.52</td> </tr> <tr> <td>Pedestrian</td> <td>40.06</td> <td>39.23</td> <td>37.29</td> </tr> <tr> <td>Cyclist</td> <td>83.17</td> <td>63.21</td> <td>61.08</td> </tr> <tr> <td>mAP</td> <td>71.23</td> <td>**63.57**</td> <td>61.63</td> </tr> </table> | -                                                            | -                                                            |

For cars required a bounding box overlap of 70%, while for pedestrians and cyclists required a bounding box overlap of 50%.

Difficulties are defined as follows:

1. Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
2. Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
3. Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %

## [Description of Random Situation](#contents)

`train.py` script use mindspore.set_seed() to set global random seed, which can be modified.

## [ModelZoo Homepage](#contents)

Please visit the official website [homepage](https://gitee.com/mindspore/models).
