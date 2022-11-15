
- [Directory](#directory)
- [DLRM overview](#dlrm-overview)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Quickstart](#Quickstart)
- [Script description](#script-description)
    - [Script and Sample Code](#Script-and-Sample-Code)
    - [Script parameters](#script-parameters)
    - [Training process](#training-process)
        - [Training](#training)
        - [Distributed training](#distributed-training)
        - [Training result](#training-result)
    - [Evaluation process](#evaluation-process)
        - [Assessment](#assessment)
        - [Assessment result](#assessment-result)
    - [Inference process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Execute reasoning at Ascend310](#at-ascend310-execute-reasoning)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training performance](#training-performance)
        - [Inference performance](#inference-performance)
- [Random Situation Description](#Random-Situation-Description)
- [ModelZoo homepage](#modelzoo-homepage)

## DLRM overview

In the click-through rate prediction model, the input features usually contain
a large number of sparse categorical features and some numerical features. DLRM uses embedding technology
to process categorical features, MLP to process numerical features, and then features interaction through an explicit
dot product feature interaction layer. Another top MLP produces CTR predictions. Usually,
the amount of data in the recommendation system is huge, and there are many category features,
which leads to a huge amount of parameters in the recommendation model, and the embedded table accounts for the main part.
DLRM improves the efficiency of the model through a novel hybrid parallel mechanism, which divides the embedding tables of
different feature domains and the bottom MLP into each parallel GPU for model parallelism, while the top interaction layer
and MLP perform data parallelism.

Paper：Naumov M, Mudigere D, et al. Deep learning recommendation model for personalization and recommendation systems.

## Model Architecture

DLRM can be divided into bottom and top parts. The bottom contains an MLP for processing numerical features and an embedding layer
, which then generates feature interactions through an interaction layer.
At the top is an MLP that finally produces predictions through a sigmoid activation function.

## Dataset

- [Criteo Kaggle Display Advertising Challenge Dataset](http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz)

## Environmental requirements

- Hardware（Ascend or GPU)
- Framework
    - [MindSpore](https://www.mindspore.cn/install) (1.3 / 1.5 / 1.6)
- Additional libraries

  ```bash
  pip install sklearn
  pip install pandas
  pip install pyyaml
  ```

## Quick Start

- Dataset Preprocessing

  ```bash
  #Download dataset
  mkdir -p data/origin_data && cd data/origin_data
  wget DATA_LINK
  tar -zxvf criteo.tar.gz

  python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --train_line_count=45840617
  ```

- Ascend

  ```bash
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target=Ascend \
    --do_eval=True > ms_log/output.log 2>&1 &
  OR
  bash scripts/run_standalone_train_ascend.sh DEVICE_ID/CUDA_VISIBLE_DEVICES DEVICE_TARGET DATASET_PATH

  OR
  bash scripts/run_distribute_train_ascend.sh 8 /dataset_path /rank_table_8p.json

  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/dlrm.ckpt' \
    --device_target=Ascend > ms_log/eval_output.log 2>&1 &
  OR
  bash scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/dlrm.ckpt
  ```

  In distributed training, the HCCL configuration file in JSON format needs to be created in advance.

  For specific operations, see:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

## Script description

### Scripts and sample code

```dlrm
└─dlrm
  ├─ascend310_info
  ├─README.md
  ├─mindspore_hub_conf.md
  ├─scripts
    ├─run_standalone_train_ascend.sh
    ├─run_distribute_train_ascend.sh
    ├─run_standalone_train_gpu.sh
    ├─run_eval.sh
    ├─run_eval_gpu.sh
    └─run_infer_310.sh
  ├─src
    ├─model_utils
      ├─__init__.py
      ├─config.py
      ├─device_target.py
      ├─local_adapter.py
      └─moxing_adapter.py
    ├─__init__.py
    ├─callback.py
    ├─dlrm.py
    ├─dataset.py
    └─preprocess_data.py
  ├─npu_config.yaml
  ├─gpu_config.yaml
  ├─eval.py
  ├─export.py
  ├─preprocess.py
  ├─postprocess.py
  └─train.py
```

### Script parameters

Both training parameters and evaluation parameters can be configured in config.py.

- Training parameters

  ```text
  optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Dataset path
  --ckpt_path CKPT_PATH
                        Checkpoint path
  --eval_file_name EVAL_FILE_NAME
                        Auc log file path. Default: "./auc.log"
  --loss_file_name LOSS_FILE_NAME
                        Loss log file path. Default: "./loss.log"
  --do_eval DO_EVAL     Do evaluation or not. Default: True
  --device_target DEVICE_TARGET
                        Ascend or GPU. Default: Ascend
  ```

- Evaluation parameters

  ```text
  optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        Checkpoint file path
  --dataset_path DATASET_PATH
                        Dataset path
  --device_target DEVICE_TARGET
                        Ascend or GPU. Default: Ascend
  ```

### Training process

#### Training

- Ascend

  ```bash
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target=Ascend \
    --do_eval=True > ms_log/output.log 2>&1 &
  ```

  The above python commands will run in the background and you can view the results via the `ms_log/output.log` file.

  After training, you can find checkpoint files in the default folder `./checkpoint`. Loss values are saved in the loss.log file.

- GPU

  ```bash
  bash scripts/run_standalone_train_gpu.sh DEVICE_ID DEVICE_TARGET DATASET_PATH
  ```

#### Distributed training

- Ascend

  ```bash
  bash scripts/run_distribute_train.sh 8 /dataset_path /rank_table_8p.json
  ```

  The above shell script will run distributed training in the background. Please see the results in the `log[X]/output.log` file.
  Loss values are saved in the `loss.log` file.

#### Training results

The training results will be saved in the example path, checkpoint and output log as described above, and the loss values
during training will be saved in the `loss.log` file.

```result
2021-11-08 21:35:30 epoch: 1, step: 76742, loss is 0.3277069926261902...
```

### Evaluation process

#### Evaluate

- Ascend

  Before running the following commands, please check the dataset and checkpoint paths used for evaluation.

  ```bash
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/dlrm.ckpt' \
    --device_target=Ascend > ms_log/eval_output.log 2>&1 &
  OR
  bash scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/dlrm.ckpt
  ```

  The above python command will run in the background, please check the result under eval_output.log path.
  Accuracy rates are saved in the acc.log file.

- GPU

  ```bash
  bash scripts/run_eval.sh DEVICE_ID DEVICE_TARGET DATASET_PATH CHECKPOINT_PATH
  ```

#### Evaluation result

Evaluations are saved in the `acc.log` file.

```result
2021-11-08 21:51:14 EvalCallBack metric {'acc': 0.787175917087641}; eval_time 894s
```

### Inference process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

#### Export MindIR

Please modify the three parameters of checkpoint_path, file_name, file_format in npu_config.yaml, and then execute.

```shell
python export.py --config_path [/path/to/npu_config.yaml]
```

`file_format` must be selected in ["AIR", "MINDIR"]

#### Perform inference in Ascend310

Inference has not completed because the model is too large to load.

Before performing inference, the mindir file must be exported via the `export.py` script.

```shell
# Ascend310
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` indicates whether the data needs to be preprocessed, the value range is 'y' or 'n'.
- `DEVICE_ID` is optional, the default value is 0.

## Model description

### Performance

#### Training performance

| Platform                  | Ascend                                                      | GPU |
| -------------------------- | ----------------------------------------------------------- |----|
| Network              | DLRM                                                  | DLRM   |
| Resource                   |Ascend 910；CPU 2.60GHz，192 cores; memory 755G; system Euler2.8              | CPU Intel(R) Xeon(R) Gold 6226R; RAM 252G GPU RTX3090 24G |
| Upload date              | 2021-11-09                           | 2022-02-23 |
| MindSpore version          | 1.3.0/1.5.0                                           | 1.6.0 |
| Dataset                    | Criteo                                           | Criteo |
| Training parameters        | epoch=1,  batch_size=128, lr=0.15                        |epoch=1,  batch_size=1280, lr=0.15  |
| Optimizer                   | SGD                                                      |SGD   |
| Loss function              | Sigmoid Cross Entropy With Logits                           |Sigmoid Cross Entropy With Logits                           |
| Output                    | ckpt file                                                  |ckpt file |
| Final loss                 | 0.3277069926261902                                                    | 0.452256|
| Speed | single card ：144 ms/step ;                                      | single card ：38 ms/step;|
| Total duration | single card: 9 hours ;                                               | single card：20 min;|
| Parameters(M)             | 540                                                        | 540 |
| Checkpoints | 6.1G (.ckpt)                                     | 6.1G (.ckpt)  

#### Inference performance

| Platform          | Ascend                      | GPU |
| ------------------- | --------------------------- | --- |
| Network        | DLRM                | DLRM   |
| Resource            | Ascend 910；CPU 2.60GHz，192 cores; memory 755G; system Euler2.8                  | CPU: Intel(R) Xeon(R) Gold 6226R RAM: 252G GPU: RTX3090 24G |
| Upload date       | 2021-11-09 | 2022-02-23 |
| MindSpore version    | 1.3.0/1.5.0      |  1.6.0 |
| Dataset           | Criteo                    | Criteo |
| Batch_size          | 16384                        | 16384  |
| Output              | Accuracy                    | Accuracy |
| Total time            | 1H50min                  | 155 s|
| Accuracy | 0.787175917087641                | 0.7876784245770677 |
| Checkpoints | 6.1G (.ckpt)           | 6.1G (.ckpt) |

## Random Situation Description

Set the random seed for `MindSpore` in `train.py`.

## ModelZoo

Please visit the official website [homepage](https://gitee.com/mindspore/models)
