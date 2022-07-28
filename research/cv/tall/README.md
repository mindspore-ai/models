# Contents

- [TALL](#tall)
- [Dataset](#dataset)
- [Environmental Requirements](#environmental-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Ascend Standalone training](#ascend-standalone-training)

        - [Ascend Distribute training](#ascend-distribute-training)

        - [Output](#output)
    - [Evaluation Process](#evaluation-process)
        - [Run](#Run)

        - [Output](#Output)
    - [Inference Process](#inference-process)
    - [Export MindIR/AIR](#export-mindIR/AIR)
    - [Inference on Ascend310](#inference-on-ascend310)
    - [Inference with SDK](#Inference-with-SDK)
- [Model Description](#model-description)
- [Performance](#performance)
     - [Training performance](#training-performance)

     - [Inference performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

## [TALL](#contents)

TALL: Temporal Activity Localization via Language Query MindSpore framework implementation.

TALL is a model for locating video clips through sentence questions. It combines the characteristics of language modality and visual modality to perform regression positioning on video clips. Paper link:[TALL](https://openaccess.thecvf.com/content_iccv_2017/html/Gao_TALL_Temporal_Activity_ICCV_2017_paper.html)。

This implementation refers to the original author's [TensorFlow version](https://github.com/jiyanggao/TALL) and iworldtong's [Pytorch version](https://github.com/iworldtong/TALL.pytorch).

## [Dataset](#contents)

This project uses the TACoS data set experiment.

First download the [training set](https://drive.google.com/file/d/1zQp0aYGFCm8PqqHOh4UtXfy2U3pJMBeu/view?usp=sharing) and [test set](https://drive.google.com/file/d/1zC-UrspRf42Qiu5prQw4fQrbgLQfJN-P/view?usp=sharing) of the C3D feature of TACoS, unzip it after downloading, and remember the data set storage path [data_path];

Then download the Skip-thought sentence embeddings of the TACoS data set [here](https://drive.google.com/file/d/1HF-hNFPvLrHwI5O7YvYKZWTeTxC5Mg1K/view?usp=sharing), create an exp_data folder in [data_path], and unzip it into this folder.

The directory structure of the unzipped and organized data set should be as follows:

```shell
.
├── exp_data
│   └── TACoS
│      ├── TACoS_test_samples.txt
│      ├── TACoS_test_videos.txt
│      ├── TACoS_train_samples.txt
│      ├── TACoS_train_videos.txt
│      ├── TACoS_val_samples.txt
│      ├── TACoS_val_videos.txt
│      ├── test_clip-sentvec.pkl
│      ├── train_clip-sentvec.pkl
│      └── val_clip-sentvec.pkl
├── Interval128_256_overlap0.8_c3d_fc6
└── Interval64_128_256_512_overlap0.8_c3d_fc6
```

## [Environmental Requirements](#contents)

- Hardware

    - Ascend

- Software

    ```
    mindspore == 1.2.0
    numpy >= 1.17.0
    protobuf >= 3.13.0
    asttokens >= 1.1.13
    pillow >= 6.2.0
    scipy >= 1.5.2
    cffi >= 1.12.3
    wheel >= 0.32.0
    decorator >= 4.4.0
    setuptools >= 40.8.0
    matplotlib >= 3.1.3         # for ut test
    opencv-python >= 4.1.2.30   # for ut test
    sklearn >= 0.0              # for st test
    pandas >= 1.0.2             # for ut test
    astunparse >= 1.6.3
    packaging >= 20.0
    pycocotools >= 2.0.2        # for st test
    tables >= 3.6.1             # for st test
    easydict >= 1.9             # for st test
    psutil >= 5.7.0
    ```

- For details, please refer to the following resources：

    - [MindSpore tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)

    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [Quick Start](#contents)

After installing MindSpore through the official website and following the above steps to download the dataset, you can follow the steps below to train and evaluation:

```shell script
# Ascend standalone training
cd ./model_zoo/research/cv/tall
bash scripts/run_standalone_train.sh [TRAIN_DATA_DIR] [DEVICE_ID]
```

```shell script
# Ascend evaluation
cd ./model_zoo/research/cv/tall
bash scripts/run_eval.sh [CHECKPOINT_PATH] [EVAL_DATA_DIR] [DEVICE_ID]
```

If you need distributed parallel training, you can use [the tool](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools) to generate the configuration file of your server, and then follow the steps below to train:

```shell script
# Ascend Distribute training
cd ./model_zoo/research/cv/tall
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell script
.tall
├── ascend310_infer
│   ├── build.sh                            # build ascend310 infer script
│   ├── CMakeLists.txt                      # cmake configuration
│   ├── inc
│   │   └── utils.h                         # utils.cc's head file
│   └── src
│       ├── main.cc                         # ascend310 infer script
│       └── utils.cc                        # ascend310 infer common utils
├── eval.py                                 # Test set accuracy evaluation, called by run_eval.sh
├── export.py                               # Export mindspore models to other formats
├── README.md
├── scripts
│   ├── run_310_infer.sh                    # Ascend310 evaluation script
│   ├── run_distribute_train.sh             # Distributed training script
│   ├── run_eval.sh                         # Evaluation script
│   └── run_standalone_train.sh             # Training script
├── src
│   ├── config.py                           # Model configuration file
│   ├── ctrl.py                             # CTRL model building
│   ├── dataset.py                          # Dataset loading
│   └── utils.py                            # common utils
├── train.py                                # Model training code
├── get_310_eval_dataset.py                 # Get ascend310 evaluation dataset
└── postprocess.py                          # Ascend310 evaluation postprocess
```

### [Script Parameters](#contents)

The main parameters of train.py and config.py and their meaning

```shell
-----------------train.py--------------------------
"device_target": Ascend  # Operating platform, the default is Ascend
"device_id": 0           # Device ID, default is device 0
"run_distribute"         # Whether to Distribute Parallel Training
"train_data_dir"         # The data set storage directory, which is [data_path] above
"check_point_dir"        # check_point save path
-----------------config.py-------------------------
"mode": "GRAPH_MODE"     # Operating mode
"train_csv_path"         # Sentence feature train set path
"valid_csv_path"         # Sentence feature validation set path
"test_csv_path"          # Sentence feature test set path
"train_feature_dir"      # Video feature train set path
"test_feature_dir"       # Video feature test set path
"nIoL": 0.15             # nIoL threshold in sample selection
"IoU": 0.5               # IoU threshold in sample selection
"max_epoch": 3           # Total number of training rounds
"batch_size": 64         # The batch size of the incoming model for each training
"test_batch_size": 128   # The maximum batch size passed into the model during the evaluation test
"optimizer": "Adam"      # Optimizer
"lr": 2e-5               # Learning rate
"visual_dim": 4096 * 3   # Visual feature dimension
"sentence_embed_dim": 4800 # Sentence feature dimension
"semantic_dim": 1024     # Semantic feature dimension
"middle_layer_dim": 1024 # Middle layer feature dimension
"lambda_reg": 0.01       # Balance factor of location loss and alignment loss
----------------run_standalone_train.sh--------------
$1:[TRAIN_DATA_DIR]      # The data set storage directory, which is [data_path] above
$2:[DEVICE_ID]           # Device id of Ascend.
----------------run_distribute_train.sh-----------------
$1:[RANK_TABLE_FILE]     # Parallel training configuration file path
$2:[TRAIN_DATA_DIR]      # The data set storage directory, which is [data_path] above
----------------run_eval.sh---------------------------
$1:[CHECKPOINT_PATH]     # Model checkpoint path
$2:[EVAL_DATA_DIR]       # The data set storage directory, which is [data_path] above
$3:[DEVICE_ID]           # Device id of Ascend.
```

### [Training Process](#contents)

#### [Ascend Standalone training](#contents)

Run run_standalone_train.sh for standalone training

```shell
bash scripts/run_standalone_train.sh [TRAIN_DATA_DIR] [DEVICE_ID]
```

This script requires one parameters.

- `TRAIN_DATA_DIR`：The dataset path, which is [data_path] above.
- `DEVICE_ID`：Device id of Ascend.

#### [Ascend Distribute training](#contents)

Run the distributed training script run_distribute_train.sh to start training

```shell script
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]
```

This script requires two parameters.

- `RANK_TABLE_FILE`：The path of [rank_table.json](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools). It is best to use absolute paths.
- `TRAIN_DATA_DIR`：The dataset path, which is [data_path] above.

#### [Output](#contents)

The checkpoint file will be stored in the current path, and the training results and error messages will be output to train.log by default. Examples are as follows:

```shell script
Set Context...
Done.
Get Dataset...
Done.
Get Model...
Done.
Train Model...
epoch: 1 step: 422, loss is 0.21251486
epoch time: 132006.143 ms, per step time: 312.811 ms
epoch: 2 step: 422, loss is 0.26855457
epoch time: 76715.097 ms, per step time: 181.789 ms
[...]
Done.
End.
```

### [Evaluation Process](#contents)

#### [Run](#contents)

Run the run_eval.sh to evaluate the model trained in the previous step.

```bash
bash scripts/run_eval.sh [CHECKPOINT_PATH] [EVAL_DATA_DIR] [DEVICE_ID]
```

This script requires three parameters.

- `CHECKPOINT_PATH`：The absolute path of the checkpoint file. The checkpoint file during training is saved by default in`./logs/datetime/best.ckpt`
- `EVAL_DATA_DIR`：The dataset path, which is [data_path] above.
- `DEVICE_ID`：Device id of Ascend.

#### [Output](#contents)

```shell script
Set Context...
Done.
Get Dataset...
Done.
Get Model...
loading /home/yuanyibo/mindspore/model_zoo/research/nlp/tall/logs/08-17_21-21/best.ckpt...
Done.
Start eval...
Test movie: s27-d50.avi....loading movie data
sentences: 99
clips: 136
s27-d50.avi IoU=0.1, R@10: 1.0; IoU=0.1, R@5: 0.9595959595959596; IoU=0.1, R@1: 0.494949494949495
s27-d50.avi IoU=0.3, R@10: 0.98989898989899; IoU=0.3, R@5: 0.8585858585858586; IoU=0.3, R@1: 0.3838383838383838
s27-d50.avi IoU=0.5, R@10: 0.7272727272727273; IoU=0.5, R@5: 0.6666666666666666; IoU=0.5, R@1: 0.3434343434343434
Test movie: s30-d41.avi....loading movie data
[...]
```

### [Inference Process](#contents)

#### [Export MindIR/AIR](#contents)

You can use the export.py script to export the model

```shell script
python export.py --checkpoint_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The parameter checkpoint_path is required, and `EXPORT_FORMAT`must be selected in ["AIR", "MINDIR"].

#### [Inference on Ascend310](#contents)

Run the run_310_infer.sh to inference on ascend310.

```bash
bash scripts/run_310_infer.sh [MINDIR_PATH] [EVAL_DATA_DIR] [DEVICE_ID]
```

This script requires three parameters.

- `CHECKPOINT_PATH`：The absolute path of the mindir file.
- `EVAL_DATA_DIR`：The dataset path, which is [data_path] above.
- `DEVICE_ID`：Device id of Ascend310.

#### [Inference with SDK](#contents)

1.Convert the AIR model to the OM model ：execute the script air2om.sh to convert the OM.

```bash
bash infer/convert/air2om.sh [AIR_PATH] [OUTPUT_MODEL_NAME]
```

This script requires two parameters.

- `AIR_PATH`：The absolute path of the air file.

- `OUTPUT_MODEL_NAME`：The name of om file.

2.Then write the absolute path of the om model to the corresponding location of sdk/config/tall.pipeline. In the following example, the location of the om model is [OM_PATH]

```json
"mxpi_tensorinfer0": {
                "props": {
                    "dataSource": "appsrc0",
                    "modelPath": "[OM_PATH]",
                    "waitingTime": "30000",
                    "singleBatchInfer": "1"
                },
                "factory": "mxpi_tensorinfer",
                "next": "appsink0"
}
```

3.Run the run_sdk_infer.sh to inference on ascend310.

```bash
cd scripts
bash run_sdk_infer.sh [EVAL_DATA_DIR]
```

This script requires one parameters.

- `EVAL_DATA_DIR`：The dataset path, which is [data_path] above.

## [Model Description](#contents)

### [Performance](#contents)

#### [Training performance](#contents)

| Parameters          | Ascend                                           |
| ------------------- | ------------------------------------------------ |
| Model Version       | TALL                                             |
| Resource            | Ascend 910；CPU：2.60GHz, 192 cores; RAM: 755 GB |
| uploaded Date       | 2021-08-19                                       |
| MindSpore Version   | 1.2.0                                            |
| Dataset             | TACoS                                            |
| Training Parameters | See config.py for details                        |
| Optimizer           | Adam                                             |
| Loss Function       | AlignLoss，Smooth L1                             |
| Speed               | 190.731 ms / step                                |
| Total time          | 5.2 min                                          |
| R@5 IoU=0.1         | 49.77%                                           |
| Parameters          | 22750211                                         |

#### [Inference performance](#contents)

the performance of R@5 IoU=0.1 is 49.33%.

## [Description of Random Situation](#contents)

config.py can set a random seed to avoid randomness in the training process.

## [ModelZoo Homepage](#contents)

Please visit the official website [homepage](https://gitee.com/mindspore/models).
