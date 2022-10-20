# Contents

- [Contents](#contents)
- [DeepSort description](#deepsort-description)
    - [Description](#description)
    - [Paper](#paper)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
    - [MOT16](#mot16)
    - [Market-1501](#market-1501)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage)
            - [Running on GPU](#running-on-gpu)
            - [Running on Ascend](#running-on-ascend)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Running on GPU](#running-on-gpu-1)
            - [Running on Ascend](#running-on-ascend-1)
        - [Result](#result-1)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DeepSort description](#contents)

## Description

DeepSort is a multi-target tracking algorithm proposed in 2017. The network won the championship at MOT16, which not only improved the accuracy, but also the speed was 20 times faster than before.

## Paper

Nicolai Wojke, Alex Bewley, Dietrich Paulus. ["SIMPLE ONLINE AND REALTIME TRACKING WITH A DEEP ASSOCIATION METRIC"](https://arxiv.org/abs/1602.00763). Presented at ICIP 2016.

# [Model architecture](#contents)

DeepSort consists of a feature extractor, a Kalman filter and a Hungarian algorithm. The feature extractor is used to extract the feature information of the person in the frame, the Kalman filter predicts the location of the person in the current frame based on the information in the previous frame, and the Hungarian algorithm is used to match the predicted information with the detected location information of the person.

# [Datasets](#contents)

## [MOT16](<https://motchallenge.net/data/MOT16.zip>)

- Data set size: 1.9G, a total of 14 video frame sequences
    - test: 7 video sequence frames
    - train: 7 sequence frames
- Data format (a train video frame sequence):
    - det: Information such as character coordinates and confidence in the video sequence
    - gt: Video tracking tag information
    - img1: All frame sequences in the video
    - seqinfo.ini: file with dataset properties  

Note: the [npy](https://gitee.com/link?target=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp) files with detections provided by the author are used.

## [Market-1501](<http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html>)

This dataset contains 32,668 annotated bounding boxes of 1,501 identities. If the ratio of the overlapping area to the union area is larger than 50%, the bounding box is marked as "good"; if the ratio is smaller than 20%, the bounding box is marked as "distractor"; otherwise, it is marked as "junk", meaning that this image is of zero influence to the re-identification accuracy.

# [Environmental requirements](#contents)

- Hardware（GPU / Ascend）
- Framework
    - [MindSpore](https://www.mindspore.cn/install)

For details, please refer to the following resources:

- [MindSpore](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [Quick start](#contents)

- Running on GPU

```bash
# Extract detections information by path in the script
python process-npy.py
# Pre-process Market-1501 by path in the script
python prepare.py
# Train DeepSort feature extractor on single GPU
bash run_standalone_train_gpu.sh DATA_PATH
# Generate features information
python generater-detection.py --data_url="" --train_url="" --det_url="" --ckpt_url="" --model_name=""
# Generate tracking result
python evaluate_motchallenge.py --data_url="" --train_url="" --detection_url=""
# Generate metrics for MOT16 Challenge using https://github.com/cheind/py-motmetrics
python -m motmetrics.apps.eval_motchallenge <groundtruths> <tests>
```

- Running on Ascend

```bash
# Extract detections information by path in the script
python process-npy.py
# Pre-process Market-1501 by path in the script
python prepare.py
# Train DeepSort feature extractor on Ascend
python src/deep/train.py --run_modelarts=False --run_distribute=True --data_url="" --train_url=""
# Generate features information
python generater_detection.py --run_modelarts=False --run_distribute=True --data_url="" --train_url="" --det_url="" --ckpt_url="" --model_name=""
# Generate tracking result
python evaluate_motchallenge.py --data_url="" --train_url="" --detection_url=""
# Generate metrics for MOT16 Challenge using https://github.com/cheind/py-motmetrics
python -m motmetrics.apps.eval_motchallenge <groundtruths> <tests>
```

# [Script description](#contents)

## [Script and Sample Code](#contents)

```text
├── DeepSort
    ├── ascend310_infer
    ├── infer
    ├── modelarts
    ├── scripts #scripts for training
    │   ├──run_standalone_train_gpu.sh
    │   ├──run_distributed_train_ascend.sh
    │   ├──run_infer_310.sh
    │   ├──docker_start.sh
    ├── src
    │   │   ├── application_util
    │   │   │   ├──image_viewer.py
    │   │   │   ├──preprocessing.py
    │   │   │   ├──visualization.py
    │   │   ├──deep #features extractor code
    │   │   │   ├──feature_extractor.py
    │   │   │   ├──config.py
    │   │   │   ├──market1501_standalone_gpu.yaml #parameters for 1P GPU training
    │   │   │   ├──original_model.py
    │   │   │   ├──train.py
    │   │   ├──sort
    │   │   │   ├──detection.py
    │   │   │   ├──iou_matching.py
    │   │   │   ├──kalman_filter.py
    │   │   │   ├──linear_assignment.py
    │   │   │   ├──nn_matching.py
    │   │   │   ├──track.py
    │   │   │   ├──tracker.py
    ├── deep_sort_app.py #auxiliary module
    ├── Dockerfile
    ├── evaluate_motchallenge.py #script for generating tracking result
    ├── export.py
    ├── generate_videos.py
    ├── generater-detection.py #script for generating features information
    ├── preprocess.py #auxiliary module
    ├── prepare.py #script to prepare market-1501 dataset
    ├── process-npy.py #script to extract and prepare MOT detections
    ├── show_results.py
    ├── pipeline.sh #example of calling all scripts in a sequence
    ├── README.md
```

## [Script Parameters](#contents)

```text
generater_detection.py evaluate_motchallenge.py:

--data_url: path to dataset (MOT / Market)
--train_url: output path
--ckpt_url: path to checkpoint
--model_name: name of the checkpoint
--det_url: path to detection files
--detection_url:  path to features files
```

## [Training Process](#contents)

### Running on GPU

  ```bash
  #standalone
  Usage:
  bash run_standalone_train_gpu.sh DATA_PATH
  ```

Example of output in log file：

  ```bash
  epoch: 1 step: 809, loss is 4.4773345
  epoch time: 14821.373 ms, per step time: 18.321 ms
  epoch: 2 step: 809, loss is 3.3706033
  epoch time: 9110.971 ms, per step time: 11.262 ms
  epoch: 3 step: 809, loss is 3.000544
  epoch time: 9131.733 ms, per step time: 11.288 ms
  epoch: 4 step: 809, loss is 1.196707
  epoch time: 8973.570 ms, per step time: 11.092 ms
  epoch: 5 step: 809, loss is 1.0504937
  epoch time: 9051.383 ms, per step time: 11.188 ms
  epoch: 6 step: 809, loss is 0.7604818
  epoch time: 9384.670 ms, per step time: 11.600 ms
  ...
  ```

### Running on Ascend

  ```bash
  bash scripts/run_distributed_train_ascend.sh train_code_path RANK_TABLE_FILE DATA_PATH
  ```

Example of output in log file：

  ```bash
  epoch: 1 step: 3984, loss is 6.4320717
  epoch: 1 step: 3984, loss is 6.414733
  epoch: 1 step: 3984, loss is 6.4306755
  epoch: 1 step: 3984, loss is 6.4387856
  epoch: 1 step: 3984, loss is 6.463995
  ...
  epoch: 2 step: 3984, loss is 6.436552
  epoch: 2 step: 3984, loss is 6.408932
  epoch: 2 step: 3984, loss is 6.4517527
  epoch: 2 step: 3984, loss is 6.448922
  epoch: 2 step: 3984, loss is 6.4611588
  ...
  ```

## [Evaluation Process](#contents)

### Running on GPU

```bash
  # Generate features information
  python generater-detection.py --data_url="" --train_url="" --det_url="" --ckpt_url="" --model_name=""
  # Generate tracking result
  python evaluate_motchallenge.py --data_url="" --train_url="" --detection_url=""
  # Generate metrics for MOT16 Challenge using https://github.com/cheind/py-motmetrics
  python -m motmetrics.apps.eval_motchallenge <groundtruths> <tests>
  ```

### Running on Ascend

```bash
  # Generate features information
  python generater-detection.py --data_url="" --train_url="" --det_url="" --ckpt_url="" --model_name="" --device="Ascend"
  # Generate tracking result
  python evaluate_motchallenge.py --data_url="" --train_url="" --detection_url=""
  # Generate metrics for MOT16 Challenge using https://github.com/cheind/py-motmetrics
  python -m motmetrics.apps.eval_motchallenge <groundtruths> <tests>
  ```

### Result

#### MOT16 Challenge train set's metrics for 1P GPU

| Seq | MOTA | MOTP| MT | ML| IDs | FM | FP | FN |
| -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -----------------------------------------------------------
| MOT16-02 | 29.0% | 0.207 | 12 | 10| 167 | 247 | 4212 | 8285 |
| MOT16-04 | 58.7% | 0.168| 42 | 15| 58 | 254 | 6268 | 13328 |
| MOT16-05 | 51.9% | 0.215| 31 | 27| 62 | 112 | 643 | 2577 |
| MOT16-09 | 64.4% | 0.162| 13 | 1| 42 | 57 | 313 | 1519 |
| MOT16-10 | 48.7% | 0.228| 24 | 1| 220 | 301 | 3183 | 2922 |
| MOT16-11 | 65.3% | 0.153| 29 | 9| 57 | 95 | 927 | 2195 |
| MOT16-13 | 44.3% | 0.237| 62 | 6| 328 | 332 | 3784 | 2264 |
| overall | 51.7% | 0.190| 211 | 69| 934 | 1398 | 19330 | 33090 |

#### MOT16 Challenge train set's metrics for Ascend

| Seq | MOTA | MOTP| MT | ML| IDs | FM | FP | FN |
| -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -----------------------------------------------------------
| MOT16-02 | 29.0% | 0.207 | 11 | 11| 159 | 226 | 4151 | 8346 |
| MOT16-04 | 58.6% | 0.167| 42 | 14| 62 | 242 | 6269 | 13374 |
| MOT16-05 | 51.7% | 0.213| 31 | 27| 68 | 109 | 630 | 2595 |
| MOT16-09 | 64.3% | 0.162| 12 | 1| 39 | 58 | 309 | 1537 |
| MOT16-10 | 49.2% | 0.228| 25 | 1| 201 | 307 | 3089 | 2915 |
| MOT16-11 | 65.9% | 0.152| 29 | 9| 54 | 99 | 907 | 2162 |
| MOT16-13 | 45.0% | 0.237| 61 | 7| 269 | 335 | 3709 | 2251 |
| overall | 51.9% | 0.189| 211 | 70| 852 | 1376 | 19094 | 33190 |

## [Inference Process](#contents)

### Export mindir model

```shell
python export.py --device_id [DEVICE_ID] --ckpt_file [CKPT_PATH]
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| parameter | | |
| -------------------------- | -----------------------------------------------------------|-------------------------------------------------------|
| Resources | GPU Tesla V100-PCIE 32G| Ascend 910 CPU 2.60GHz, 192 cores：755G |
| Upload date | 2022-01-08 | 2021-08-12 |
| MindSpore | 1.6.0 |  1.2.0 |
| Dataset | MOT16 Market-1501 | MOT16 Market-1501 |
| Training parameters | epoch=24, batch_size=16, lr=0.01 | epoch=100, step=191, batch_size=8, lr=0.1 |
| Optimizer | SGD |
| Loss function | SoftmaxCrossEntropyWithLogits | SoftmaxCrossEntropyWithLogits |
| Final loss | 0.04 | 0.03 |
| Speed | 12.8 ms/step | 9.804 ms/step |
| Time to train | 3 min | 10 min |
| Checkpoint size | 23.4 Mb | 40 Mb |

### Evaluation Performance

| parameter          || |
| ------------------- | --------------------------- | --- |
| Resources | GPU Tesla V100-PCIE 32G| Ascend 910 CPU 2.60GHz, 192 cores：755G |
| MindSpore version    | 1.6.0      |  1.2.0 |
| Dataset | MOT16 Market-1501 | MOT16 Market-1501 |
| MOTA/MOTP | 51.7%/0.190                | 51.9%/0.189 |
| Checkpoint size | 23.4 Mb | 40 Mb |

# [Description of Random Situation](#contents)

We set random seed inside train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).