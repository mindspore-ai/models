# Contents

- [Contents](#contents)
    - [OctSqueeze Description](#octsqueeze-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
            - [Training](#training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Train network with fixed batch\_size(Can skip, if already  done)](#train-network-with-fixed-batch_sizecan-skip-if-already--done)
        - [Generate input dat a for network](#generate-input-dat-a-for-network)
        - [Export MindIR](#export-mindir)
        - [Infer](#infer)
        - [Result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
            - [Take 000000.bin~001023.bin in KITTI to generate training dataset and train the network](#take-000000bin001023bin-in-kitti-to-generate-training-dataset-and-train-the-network)
        - [Inference Performance](#inference-performance)
            - [Take 007000.bin~007099.bin in KITTI as test date, evaluation result as follow](#take-007000bin007099bin-in-kitti-as-test-date-evaluation-result-as-follow)
    - [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

## [OctSqueeze Description](#contents)

OctSqueeze is a network for spare point cloud compression. It was proposed by Uber ATG at CVPR2020 oral presentation. Follow conventional octree compression pipline, by using network to predict context, it successfully outperforms existing methods, including MPEG G-PCC and Google Draco.

[论文](https://arxiv.org/abs/2005.07178): Huang L, Wang S, Wong K, Liu J, Urtasun R. Octsqueeze: Octree-structured entropy model for lidar compression. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2020

## [Model Architecture](#contents)

OctSqueeze network have two parts: feature extraction and feature fusion. Feature extraction part consists of 5 mlp layers, which is in charge of extracting hidden feature for each node. Then a wavenet-like feature fusion part is used to combine the feature from different depth. In the end, after a softmax layer, we can get the predicted distribution of octree partition.

## [Dataset](#contents)

Dataset used: [KITTI](<http://www.cvlibs.net/datasets/kitti/>)

- Dataset size: 13.2G, totally 7481 point cloud frames
    - training set and test dataset can be decided arbitrarily
- Data format: binary files
    - Note: OctSqueeze is a hybrid compression method. We need to convert point cloud into octree format, then summary the explicit feature of each node and sent then in to network. Therefore, we need to run run_process_data.sh first to process point cloud and generate the feature which can be sent into network. Then we can use these features to train our network.

- Download the dataset (only .bin files under velodyne folder are needed), the directory structure is as follows:

```text
├─ImageSets
|
└─object
  ├─training
    ├─calib
    ├─image_2
    ├─label_2
    └─velodyne
      ├─000000.bin
      ├─000001.bin
      ...
      └─007480.bin
```

## [Environment Requirements](#contents)

- Hardware(Ascend)
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```python
# enter script dir, generate training dataset
bash run_process_data.sh [POIND_CLOUD_PATH] [OUTPUT_PATH] [MIN_ID] [MAX_ID] [MODE]
# example: bash run_process_data.sh ./KITTI/object/training/velodyne/ /home/ma-user/work/training_dataset/ 0 1000 train (a.k.a 000000.bin ~ 001000.bin are used to generate training dataset)

# train OctSqueeze (1P)
bash run_train_standalone.sh [TRAINING_DATASET_PATH] [DEVICE] [CHECKPOINT_SAVE_PATH] [batch_size]
# example: bash run_train_standalone.sh /home/ma-user/work/training_dataset/ Ascend ./ckpt/ 0

# or tain OctSqueeze parallelly (8P)
bash run_train_distribute.sh [TRAINING_DATASET_PATH] [CHECKPOINT_SAVE_PATH] [RANK_TABLE_FILE]
# example: bash run_train_distribute.sh /home/ma-user/work/training_dataset/ ./ckpt/ /path/hccl_8p.json

# evaluate OctSqueeze
bash run_eval.sh [TEST_DATASET_PATH] [COMPRESSED_DATA_PATH] [RECONSTRUCTED_DATA_PATH] [MODE] [DEVICE]
# example: bash run_eval.sh /home/ma-user/work/test_dataset/ ./com/ ./recon/ /home/ma-user/work/checkpoint/CKP-199_1023.ckpt Ascend
```

- Running on [modelarts](https://support.huaweicloud.com/modelarts/))

   ```bash
      # Train 8p with Ascend
      # (1) Add "enable_modelarts=True" on the website UI interface.
      #     Add "distribute=True" on the website UI interface.
      #     Add "dataset_path=/cache/data" on the website UI interface.
      #     Add "ckpt_path='/cache/train'" on the website UI interface.
      #     Add other parameters on the website UI interface.
      # (2) Prepare model code
      # (3) Upload or copy training dataset generated by run_process_data.sh to S3 bucket
      # (4) Set the code directory to "/path/octsqueeze" on the website UI interface.
      # (5) Set the startup file to "train.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
      #
      # Train 1p with Ascend
      # (1) Add "enable_modelarts=True" on the website UI interface.
      #     Add "dataset_path=/cache/data" on the website UI interface.
      #     Add "ckpt_path='/cache/train'" on the website UI interface.
      #     Add other parameters on the website UI interface.
      # (2) Prepare model code
      # (3) Upload or copy training dataset generated by run_process_data.sh to S3 bucket
      # (4) Set the code directory to "/path/octsqueeze" on the website UI interface.
      # (5) Set the startup file to "train.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
   ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```python
├── cv
    ├── octsqueeze
        ├── README_CN.md                 // descriptions about octsqueeze in Chinese
        ├── README.md                    // descriptions about octsqueeze in English
        ├── requirements.txt             // required packages
        ├── scripts
        │   ├──run_eval.sh                    // evaluate in Ascend
        │   ├──run_proccess_data.sh           // generate training dataset from KITTI .bin files
        │   ├──run_train_standalone.sh                // train in Ascend 1P
        │   ├──run_train_distribute.sh                // train in Ascend 8P
        ├── src
        │   ├──network.py               // octsqueeze network architecture
        │   ├──dataset.py               // read dataset
        |   └──tools
        |      ├──__init__.py
        |      ├──utils.py                    // save .ply file, calculate entropy, error, etc.
        |      └──octree_base.py              // octree module
        ├── third_party
        │   └──arithmetic_coding_base.py   // arithmetic module
        ├── ascend310_infer                // For inference on Ascend310 (C++)
        ├── train.py                 // train in Ascend main program
        ├── process_data.py          // generate training dataset from KITTI .bin files main program
        ├── eval.py                  // evaluation main program
```

### [Script Parameters](#contents)

```python
Major parameters in process_data.py as follows:

--input_route: The path to the folder which contain point cloud (.bin file)
--output_route: Output path, aka generated training dataset path
--min_file: Min index of point cloud frame which is used to generate training dataset
--max_file: Max index of point cloud frame which is used to generate training dataset
--mode: Mode can be "train"、"inference", "inference" mode is only for Ascend310 inference. When generate training dataset, you should choose "train"

Major parameters in train.py:

--batch_size: Training batch size(set batch_size=0 means use dynamic batch_size, aka input all nodes of a point cloud into network. We recommend to do so for training. When training for 310 inference, we should set a fix batch_size which is bigger than 0)
--train: The path to training dataset(recommend to use absolute path for 8P training), training dataset is generated by process_data.py and point cloud in KITTI dataset
--max_epochs: Total training epochs
--device_target: Device where the code will be implemented. Only "Ascend" is supported now.
--checkpoint: The path to the checkpoint file saved after training.(recommend )

Major parameters in eval.py:

--test_dataset: The path to test dataset, test data are point cloud(.bin)chosen from KITTI data
--compression: The path to compressed file
--recon: The path to decompressed / reconstructed file
--model: The path to the checkpoint file which needs to be loaded
--device_target: Device where the code will be implemented. Optional values are "Ascend", "GPU", "CPU"
```

### [Training Process](#contents)

#### Training

- Running on Ascend

  ```bash
  python train.py --train=[TRAINING_DATASET_PATH] --device_target=[DEVICE] --checkpoint=[CHECKPOINT_SAVE_PATH] --batch_size=[batch_size] --is_distributed=0
  # or enter script dir, run 1P training script
  bash bash run_train_standalone.sh /home/ma-user/work/training_dataset/ Ascend ./ckpt/ 0
  # or enter script dir, run 8P training script
  bash run_train_distribute.sh /home/ma-user/work/training_dataset/ ./ckpt/ /path/hccl_8p.json
  ```

  After training, the loss value will be achieved as follows:

  ```bash
  epoch: 1 step: 1, loss is 2.2791853
  ...
  epoch: 10 step: 1023, loss is 2.7296906
  epoch: 11 step: 1023, loss is 2.7205226
  epoch: 12 step: 1023, loss is 2.7087197
  ...
  ```

  The model checkpoint will be saved in the specified directory.

### [Evaluation Process](#contents)

#### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

  ```bash
  python eval.py --test_dataset=[TEST_DATASET_PATH] --compression=[COMPRESSED_DATA_PATH] --recon=[RECONSTRUCTED_DATA_PATH] --model=[MODE] --device_target=[DEVICE]
  # or enter script dir, run evaluation script
  bash run_eval.sh /home/ma-user/work/test_dataset/ ./com/ ./recon/ /home/ma-user/work/checkpoint/CKP-199_1023.ckpt Ascend
  ```

  Take 007000.bin~007099.bin in KITTI as test date, evaluation result as follow:

  ```python
  bpip and chamfer distance at different bitrates:
  [[7.04049839 0.0191274 ]
   [4.47647693 0.03779216]
   [2.51015919 0.07440553]
   [1.2202392  0.14568059]]
  ```

## [Inference Process](#contents)

### Train network with fixed batch_size(Can skip, if already  done)

```shell
python train.py --train=[TRAINING_DATASET_PATH] --device_target=[DEVICE] --checkpoint=[CHECKPOINT_SAVE_PATH] --batch_size=[batch_size] --is_distributed=0
# or enter script dir, run 1P script
bash bash run_train_standalone.sh /home/ma-user/work/training_dataset/ Ascend ./ckpt/ 98304
```

### Generate input dat a for network

```shell
bash bash run_process_data.sh [POIND_CLOUD_PATH] [OUTPUT_PATH] [MIN_ID] [MAX_ID] [MODE]
# or enter script dir, run run_process_data.sh script
bash run_process_data.sh ./KITTI/object/training/velodyne/ /home/ma-user/work/infernece_dataset/ 7000 7099 inference
```

### Export MindIR

```shell
python export.py ----ckpt_file=[CKPT_PATH] --batch_size=[BATCH_SIZE] --file_name=[FILE_NAME]
# Example:
python export.py --ckpt_file='/home/ma-user/work/AM_OctSqueeze/checkpoint/CKP-196_1024.ckpt' --batch_size=98304 --file_name=octsqueeze
```

The ckpt_file parameter is required

### [Infer](#contents)

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```bash
bash run_cpp_infer.sh [MINDIR_PATH] [DATA_PATH] [BATCH_SIZE] [DEVICE_TYPE] [DEVICE_ID]
```

- `MODEL_PATH` Path of mindir
- `DATASETS_DIR` Path of input data(Under this path, there should be several subfolders like ./0.01 ./0.02 corresponds to different precisions)
- `BATCH_SIZE` batch_size of MindIR

### Result

Inference result is saved in ./logs, you can find result in test_performance.txt.

```bash
At precision 0.01: bpip =  7.03702; each frame cost 149717 ms
At precision 0.02: bpip =  4.46777; each frame cost 101945 ms
At precision 0.04: bpip =  2.50509; each frame cost 56277 ms
At precision 0.08: bpip =  1.22112; each frame cost 53100 ms
```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

#### Take 000000.bin~001023.bin in KITTI to generate training dataset and train the network

| Parameters    | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| Network Name  | OctSqueeze                                                   |
| Resource      | Ascend 910; CPU 10 core; Momory 120G;                        |
| Uploaded Date | TBD                                                          |
| MindSpore Version | 1.3.0                                                    |
| Dataset       | 7481 point cloud frame                                       |
| Training Parameters | epoch=200, batch_size=0 (dynamic batch size), lr=0.001 |
| Optimizer     | Adam                                                         |
| Loss Function | SoftmaxCrossEntropy                                          |
| Output        | distribution                                                 |
| Loss          | 1.0                                                          |
| Speed         | 1P: 58ms/step;  8P: 80ms/step                                |
| Total Time    | 1P: 3.3h；8P: 0.6h                                           |
| parameters(M) | 0.34M                                                        |
| Checkpoint for Fine tuning | 3M (.ckpt file)                                 |
| Scripts       | [octsqueeze script](https://gitee.com/mindspore/models/tree/master/research/cv/OctSqueeze) |

### Inference Performance

#### Take 007000.bin~007099.bin in KITTI as test date, evaluation result as follow

| Parameters        | Ascend                                                       |
| ----------------- | ------------------------------------------------------------ |
| Network Name      | OctSqueeze                                                   |
| Resource          | Ascend 910                                                   |
| Uploaded Date     | TBD                                                          |
| MindSpore Version | 1.3.0                                                        |
| Dataset           | 7481 point cloud frame                                       |
| batch_size        | 1                                                            |
| Output            | average bpip and corresponding chamfer distance, at 4 different bitrate |
| Accuracy          | [[7.04049839 0.0191274 ]<br/> [4.47647693 0.03779216]<br/> [2.51015919 0.07440553]<br/> [1.2202392  0.14568059]] |

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
