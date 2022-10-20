# Contents

- [Contents](#contents)
    - [JDE Description](#jde-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
            - [Standalone Training](#standalone-training)
            - [Distribute Training](#distribute-training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation)
        - [Inference Process](#inference-process)
            - [Usage](#usage)
            - [Result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [JDE Description](#contents)

Paper with introduced JDE model is dedicated to the improving efficiency of an MOT system.
It's introduce an early attempt that Jointly learns the Detector and Embedding model (JDE) in a single-shot deep network.
In other words, the proposed JDE employs a single network to simultaneously output detection results and the corresponding appearance embeddings of the detected boxes.
In comparison, SDE methods and two-stage methods are characterized by re-sampled pixels (bounding boxes) and feature maps, respectively.
Both the bounding boxes and feature maps are fed into a separate re-ID model for appearance feature extraction.
Method is near real-time while being almost as accurate as the SDE methods.

[Paper](https://arxiv.org/pdf/1909.12605.pdf):  Towards Real-Time Multi-Object Tracking. Department of Electronic Engineering, Tsinghua University

## [Model Architecture](#contents)

Architecture of the JDE is the Feature Pyramid Network (FPN).
FPN makes predictions from multiple scales, thus bringing improvement in pedestrian detection where the scale of targets varies a lot.
An input video frame first undergoes a forward pass through a backbone network to obtain feature maps at three scales, namely, scales with 1/32, 1/16 and 1/8 down-sampling rate, respectively.
Then, the feature map with the smallest size (also the semantically strongest features) is up-sampled and fused with the feature map from the second smallest scale by skip connection, and the same goes for the other scales.
Finally, prediction heads are added upon fused feature maps at all the three scales.
A prediction head consists of several stacked convolutional layers and outputs a dense prediction map of size (6A + D) × H × W, where A is the number of anchor templates assigned to this scale, and D is the dimension of the embedding.

## [Dataset](#contents)

Used a large-scale training set by putting together six publicly available datasets on pedestrian detection, MOT and person search.

These datasets can be categorized into two types: ones that only contain bounding box annotations, and ones that have both bounding box and identity annotations.
The first category includes the ETH dataset and the CityPersons (CP) dataset. The second category includes the CalTech (CT) dataset, MOT16 (M16) dataset, CUHK-SYSU (CS) dataset and PRW dataset.
Training subsets of all these datasets are gathered to form the joint training set, and videos in the ETH dataset that overlap with the MOT-16 test set are excluded for fair evaluation.

Datasets preparations are described in [DATASET_ZOO.md](DATASET_ZOO.md).

Datasets size: 134G, 1 object category (pedestrian).

Note: `--dataset_root` is used as an entry point for all datasets, used for training and evaluating this model.

Organize your dataset structure as follows:

```text
.
└─dataset_root/
  ├─Caltech/
  ├─Cityscapes/
  ├─CUHKSYSU/
  ├─ETHZ/
  ├─MOT16/
  ├─MOT17/
  └─PRW/
```

Information about train part of dataset.

| Dataset | ETH |  CP |  CT | M16 |  CS | PRW | Total |
| :------:|:---:|:---:|:---:|:---:|:---:|:---:|:-----:|
| # img   |2K   |3K   |27K  |53K  |11K  |6K   |54K    |
| # box   |17K  |21K  |46K  |112K |55K  |18K  |270K   |
| # ID    |-    |-    |0.6K |0.5K |7K   |0.5K |8.7K   |

## [Environment Requirements](#contents)

- Hardware（GPU）
- Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore through the official website, you can follow the steps below for training and evaluation,
in particular, before training, you need to install `requirements.txt` by following command `pip install -r requirements.txt`.

> If an error occurred, update pip by `pip install --upgrade pip` and try again.
> If it didn't help install packages manually by using `pip install {package from requirements.txt}`.

Note: The PyTorch is used only for checkpoint conversion.

All trainings will starts from pre-trained backbone,
[download](https://drive.google.com/file/d/1keZwVIfcWmxfTiswzOKUwkUz2xjvTvfm/view) and convert the pre-trained on
ImageNet backbone with commands below:

```bash
# From the root model directory run
python -m src.convert_checkpoint --ckpt_url [PATH_TO_PYTORCH_CHECKPOINT]
```

- PATH_TO_PYTORCH_CHECKPOINT - Path to the downloaded darknet53 PyTorch checkpoint.

After converting the checkpoint and installing the requirements.txt, you can run the training scripts:

```bash
# Run standalone training example
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT]

# Run distribute training example
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT]
```

- DEVICE_ID - Device ID
- LOGS_CKPT_DIR - path to the directory, where the training results will be stored.
- CKPT_URL - Path to the converted pre-trained DarkNet53 backbone.
- DATASET_ROOT - Path to the dataset root directory (containing all dataset parts, described in [DATASET_ZOO.md](DATASET_ZOO.md))

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
.
└─JDE
  ├─data
  │ └─prepare_mot17.py                 # MOT17 data preparation script
  ├─cfg
  │ ├─ccmcpe.json                      # paths to dataset schema (defining relative paths structure)
  │ └─config.py                        # parameter parser
  ├─scripts
  │ ├─run_distribute_train_gpu.sh      # launch distribute train on GPU
  │ ├─run_eval_gpu.sh                  # launch evaluation on GPU
  │ └─run_standalone_train_gpu.sh      # launch standalone train on GPU
  ├─src
  │ ├─__init__.py
  │ ├─convert_checkpoint.py            # backbone checkpoint converter (torch to mindspore)
  │ ├─darknet.py                       # backbone of network
  │ ├─dataset.py                       # create dataset
  │ ├─evaluation.py                    # motmetrics evaluator
  │ ├─io.py                            # MOT evaluation utils
  │ ├─kalman_filter.py                 # kalman filter script
  │ ├─log.py                           # logger script
  │ ├─model.py                         # create model script
  │ ├─timer.py                         # timer script
  │ ├─utils.py                         # utilities used in other scripts
  │ └─visualization.py                 # visualization for inference
  ├─tracker
  │ ├─__init__.py
  │ ├─basetrack.py                     # base class for tracking
  │ ├─matching.py                      # matching for tracking script
  │ └─multitracker.py                  # tracker init script
  ├─DATASET_ZOO.md                     # dataset preparing description
  ├─README.md
  ├─default_config.yaml                # default configs
  ├─eval.py                            # evaluation script
  ├─eval_detect.py                     # detector evaluation script
  ├─export.py                          # export to MINDIR script
  ├─infer.py                           # inference script
  ├─requirements.txt
  └─train.py                           # training script
```

### [Script Parameters](#contents)

```text
Parameters in config.py and default_config.yaml.
Include arguments for Train/Evaluation/Inference.

--config_path             Path to default_config.yaml with hyperparameters and defaults
--data_cfg_url            Path to .json with paths to datasets schemas
--momentum                Momentum for SGD optimizer
--decay                   Weight_decay for SGD optimizer
--lr                      Init learning rate
--epochs                  Number of epochs to train
--batch_size              Batch size per one device'
--num_classes             Number of object classes
--k_max                   Max predictions per one map (made for optimization of FC layer embedding computation)
--img_size                Size of input images
--track_buffer            Tracking buffer
--keep_checkpoint_max     Keep saved last N checkpoints
--backbone_input_shape    Input filters of backbone layers
--backbone_shape          Input filters of backbone layers
--backbone_layers         Output filters of backbone layers
--out_channel             Number of channels for detection
--embedding_dim           Number of channels for embeddings
--iou_thres               IOU thresholds
--conf_thres              Confidence threshold
--nms_thres               Threshold for Non-max suppression
--min_box_area            Filter out tiny boxes
--anchor_scales           12 predefined anchor boxes. Different 4 per each of 3 feature maps
--col_names_train         Names of columns for training GeneratorDataset
--col_names_val           Names of columns for validation GeneratorDataset
--is_distributed          Distribute training or not
--dataset_root            Path to datasets root folder
--device_target           Device GPU or any
--device_id               Device id of target device
--device_start            Start device id
--ckpt_url                Location of checkpoint
--logs_dir                Dir to save logs and ckpt
--input_video             Path to the input video
--output_format           Expected output format
--output_root             Expected output root path
--save_images             Save tracking results (image)
--save_videos             Save tracking results (video)
```

### [Training Process](#contents)

#### Standalone Training

Note: For all trainings necessary to use pretrained backbone darknet53.

```bash
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT]
```

- DEVICE_ID - device ID
- LOGS_CKPT_DIR - path to the directory, where the training results will be stored.
- CKPT_URL - Path to the converted pre-trained DarkNet53 backbone.
- DATASET_ROOT - Path to the dataset root directory (containing all dataset parts, described in [DATASET_ZOO.md](DATASET_ZOO.md))

The above command will run in the background, you can view the result through the generated standalone_train.log file.
After training, you can get the training loss and time logs in chosen logs_dir.

The model checkpoints will be saved in LOGS_CKPT_DIR directory.

#### Distribute Training

```bash
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT]
```

- DEVICE_ID - device ID
- LOGS_CKPT_DIR - path to the directory, where the training results will be stored.
- CKPT_URL - Path to the converted pre-trained DarkNet53 backbone.
- DATASET_ROOT - Path to the dataset root directory (containing all dataset parts, described in [DATASET_ZOO.md](DATASET_ZOO.md))

The above shell script will run the distributed training in the background.
Here is the example of the training logs:

```text
epoch: 30 step: 1612, loss is -4.7679796
epoch: 30 step: 1612, loss is -5.816874
epoch: 30 step: 1612, loss is -5.302864
epoch: 30 step: 1612, loss is -5.775913
epoch: 30 step: 1612, loss is -4.9537477
epoch: 30 step: 1612, loss is -4.3535285
epoch: 30 step: 1612, loss is -5.0773625
epoch: 30 step: 1612, loss is -4.2019467
epoch time: 2023042.925 ms, per step time: 1209.954 ms
epoch time: 2023069.500 ms, per step time: 1209.970 ms
epoch time: 2023097.331 ms, per step time: 1209.986 ms
epoch time: 2023038.221 ms, per step time: 1209.951 ms
epoch time: 2023098.113 ms, per step time: 1209.987 ms
epoch time: 2023093.300 ms, per step time: 1209.984 ms
epoch time: 2023078.631 ms, per step time: 1209.975 ms
epoch time: 2017509.966 ms, per step time: 1206.645 ms
train success
train success
train success
train success
train success
train success
train success
train success
```

### [Evaluation Process](#contents)

#### Evaluation

Tracking ability of the model is tested on the train part of the MOT16 dataset (doesn't use during training).

To start tracker evaluation run the command below.

```bash
bash scripts/run_eval_gpu.sh [DEVICE_ID] [CKPT_URL] [DATASET_ROOT]
```

- DEVICE_ID - Device ID.
- CKPT_URL - Path to the trained JDE model.
- DATASET_ROOT - Path to the dataset root directory (containing all dataset parts, described in [DATASET_ZOO.md](DATASET_ZOO.md)).

> Note: the script expects that the DATASET_ROOT directory contains the MOT16 sub-folder.

The above python command will run in the background. The validation logs will be saved in "eval.log".

For more details about `motmetrics`, you can refer to [MOT benchmark](https://motchallenge.net/).

```text
DATE-DATE-DATE TIME:TIME:TIME [INFO]: Time elapsed: 240.54 seconds, FPS: 22.04
          IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT ML   FP    FN  IDs    FM  MOTA  MOTP IDt IDa IDm
MOT16-02 45.1% 49.9% 41.2% 71.0% 86.0%  54  17  31  6 2068  5172  425   619 57.0% 0.215 239  68  14
MOT16-04 69.5% 75.5% 64.3% 80.6% 94.5%  83  45  24 14 2218  9234  175   383 75.6% 0.184  98  28   3
MOT16-05 63.6% 68.1% 59.7% 82.0% 93.7% 125  67  49  9  376  1226  137   210 74.5% 0.203 113  40  40
MOT16-09 55.2% 60.4% 50.8% 78.1% 92.9%  25  16   8  1  316  1152  108   147 70.0% 0.187  76  15  11
MOT16-10 57.1% 59.9% 54.5% 80.1% 88.1%  54  28  26  0 1337  2446  376   569 66.2% 0.228 202  66  16
MOT16-11 75.0% 76.4% 73.7% 89.6% 92.9%  69  50  16  3  626   953   78   137 81.9% 0.159  49  24  12
MOT16-13 64.8% 69.9% 60.3% 78.5% 90.9% 107  58  43  6  900  2463  272   528 68.3% 0.223 200  59  48
OVERALL  63.2% 68.1% 58.9% 79.5% 91.8% 517 281 197 39 7841 22646 1571  2593 71.0% 0.196 977 300 144
```

To evaluate detection ability (get mAP, Precision and Recall metrics) of the model, run command below.

```bash
python eval_detect.py --device_id [DEVICE_ID] --ckpt_url [CKPT_URL] --dataset_root [DATASET_ROOT]
```

- DEVICE_ID - Device ID.
- CKPT_URL - Path to the trained JDE model.
- DATASET_ROOT - Path to the dataset root directory (containing all dataset parts, described in [DATASET_ZOO.md](DATASET_ZOO.md)).

Results of evaluation will be visualized at command line.

```text
      Image      Total          P          R        mAP
       4000      30353      0.829      0.778      0.765      0.426s
       8000      30353      0.863      0.798      0.788       0.42s
      12000      30353      0.854      0.815      0.802      0.419s
      16000      30353      0.857      0.821      0.809      0.582s
      20000      30353      0.865      0.834      0.824      0.413s
      24000      30353      0.868      0.841      0.832      0.415s
      28000      30353      0.874      0.839       0.83      0.419s
mean_mAP: 0.8225, mean_R: 0.8325, mean_P: 0.8700
```

### [Inference Process](#contents)

#### Usage

To compile video from frames with predicted bounding boxes, you need to install `ffmpeg` by using
`sudo apt-get install ffmpeg`. Video compiling will happen automatically.

```bash
python infer.py --device_id [DEVICE_ID] --ckpt_url [CKPT_URL] --input_video [INPUT_VIDEO]
```

- DEVICE_ID - Device ID.
- CKPT_URL - Path to the trained JDE model.
- INPUT_VIDEO - Path to the input video to tracking.

#### Result

Results of the inference will be saved into default `./results` folder, logs will be shown at command line.

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters                 | GPU (8p)                                                                            |
| -------------------------- |-----------------------------------------------------------------------------------  |
| Model                      | JDE (1088*608)                                                                      |
| Hardware                   | 8 Nvidia RTX 3090, Intel Xeon Gold 6226R CPU @ 2.90GHz                              |
| Upload Date                | 02/02/2022 (day/month/year)                                                         |
| MindSpore Version          | 1.5.0                                                                               |
| Dataset                    | Joint Dataset (see `DATASET_ZOO.md`)                                                |
| Training Parameters        | epoch=30, batch_size=4 (per device), lr=0.01, momentum=0.9, weight_decay=0.0001     |
| Optimizer                  | SGD                                                                                 |
| Loss Function              | SmoothL1Loss, SoftmaxCrossEntropyWithLogits (and apply auto-balancing loss strategy)|
| Outputs                    | Tensor of bbox cords, conf, class, emb                                              |
| Speed                      | Eight cards: ~1206 ms/step                                                          |
| Total time                 | Eight cards: ~17 hours                                                              |

#### Evaluation Performance

| Parameters          | GPU (1p)                                               |
| ------------------- |--------------------------------------------------------|
| Model               | JDE (1088*608)                                         |
| Resource            | 1 Nvidia RTX 3090, Intel Xeon Gold 6226R CPU @ 2.90GHz |
| Upload Date         | 02/02/2022 (day/month/year)                            |
| MindSpore Version   | 1.5.0                                                  |
| Dataset             | MOT-16                                                 |
| Batch_size          | 1                                                      |
| Outputs             | Metrics, .txt predictions                              |
| FPS                 | 22.04                                                  |
| Metrics             | mAP 82.2, MOTA 71.0%                                   |

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
