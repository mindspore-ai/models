# Contents

- [Contents](#contents)
    - [TBNet Description](#tbnet-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Inference Process](#inference-process)
            - [Export MindIR](#export-mindir)
            - [Infer on Ascend310](#infer-on-ascend310)
            - [Result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
            - [Inference and Explanation Performance](#inference-explanation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

# [TBNet Description](#contents)

TB-Net is a knowledge graph based explainable recommender system.

Paper: Shendi Wang, Haoyang Li, Xiao-Hui Li, Caleb Chen Cao, Lei Chen. Tower Bridge Net (TB-Net): Bidirectional Knowledge Graph Aware Embedding Propagation for Explainable Recommender Systems

# [Model Architecture](#contents)

TB-Net constructs subgraphs in knowledge graph based on the interaction between users and items as well as the feature of items, and then calculates paths in the graphs using bidirectional conduction algorithm. Finally we can obtain explainable recommendation results.

# [Dataset](#contents)

[Interaction of users and games](https://www.kaggle.com/tamber/steam-video-games), and the [games' feature data](https://www.kaggle.com/nikdavis/steam-store-games?select=steam.csv) on the game platform Steam are public on Kaggle.

Dataset directory: `./data/{DATASET}/`, e.g. `./data/steam/`.

- train: train.csv, evaluation: test.csv

Each line indicates a \<user\>, an \<item\>, the user-item \<rating\> (1 or 0), and PER_ITEM_NUM_PATHS paths between the item and the user's \<hist_item\> (\<hist_item\> is the item whose the user-item \<rating\> in historical data is 1).

```text
#format:user,item,rating,relation1,entity,relation2,hist_item,relation1,entity,relation2,hist_item,...,relation1,entity,relation2,hist_item  # module [relation1,entity,relation2,hist_item] repeats PER_ITEM_NUM_PATHS times
```

- infer and explain: infer.csv

Each line indicates the \<user\> and \<item\> to be inferred, \<rating\>, and PER_ITEM_NUM_PATHS paths between the item and the user's \<hist_item\> (\<hist_item\> is the item whose the user-item \<rating\> in historical data is 1).
Note that the \<item\> needs to traverse candidate items (all items by default) in the dataset. \<rating\> can be randomly assigned (all values are assigned to 0 by default) and is not used in the inference and explanation phases.

```text
#format:user,item,rating,relation1,entity,relation2,hist_item,relation1,entity,relation2,hist_item,...,relation1,entity,relation2,hist_item  # module [relation1,entity,relation2,hist_item] repeats PER_ITEM_NUM_PATHS times
```

We have to download the data package and put it underneath the current project path。

```bash
wget https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/xai/tbnet_data.tar.gz
tar -xf tbnet_data.tar.gz
```

# [Environment Requirements](#contents)

- Hardware（NVIDIA GPU or Ascend NPU）
    - Prepare hardware environment with NVIDIA GPU or Ascend NPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Data preprocessing

Download the data package(e.g. 'steam' dataset) and put it underneath the current project path.

```bash
wget https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/xai/tbnet_data.tar.gz
tar -xf tbnet_data.tar.gz
```

and then run code as follows.

- Training

```bash
bash scripts/run_standalone_train.sh [DATA_NAME] [DEVICE_ID] [DEVICE_TARGET]
```

Example:

```bash
bash scripts/run_standalone_train.sh steam 0 Ascend
```

- Evaluation

Evaluation model on test dataset.

```bash
bash scripts/run_eval.sh [CHECKPOINT_ID] [DATA_NAME] [DEVICE_ID] [DEVICE_TARGET]
```

Argument `[CHECKPOINT_ID]` is required.

Example:

```bash
bash scripts/run_eval.sh 19 steam 0 Ascend
```

- Inference and Explanation

Recommende items to user acrodding to `user`, the number of items is determined by `items`.

```bash
python infer.py \
  --dataset [DATASET] \
  --checkpoint_id [CHECKPOINT_ID] \
  --user [USER] \
  --items [ITEMS] \
  --explanations [EXPLANATIONS] \
  --csv [CSV] \
  --device_target [DEVICE_TARGET]
```

Arguments `--checkpoint_id` and `--user` are required.

Example:

```bash
python infer.py \
  --dataset steam \
  --checkpoint_id 19 \
  --user 2 \
  --items 1 \
  --explanations 3 \
  --csv test.csv \
  --device_target Ascend
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
└─tbnet
  ├─README.md
  ├── scripts
      ├─run_infer_310.sh                  # Ascend310 inference script
      ├─run_standalone_train.sh           # NVIDIA GPU or Ascend NPU training script
      └─run_eval.sh                       # NVIDIA GPU or Ascend NPU evaluation script
  ├─data
    ├─steam
        ├─config.json               # data and training parameter configuration
        ├─src_infer.csv             # inference and explanation dataset
        ├─src_test.csv              # evaluation dataset
        ├─src_train.csv             # training dataset
        └─id_maps.json              # explanation configuration
  ├─src
    ├─utils
        ├─__init__.py               # init file
        ├─device_adapter.py         # Get cloud ID
        ├─local_adapter.py          # Get local ID
        ├─moxing_adapter.py         # Parameter processing
        └─param.py                  # parse args
    ├─aggregator.py                 # inference result aggregation
    ├─config.py                     # parsing parameter configuration
    ├─dataset.py                    # generate dataset
    ├─embedding.py                  # 3-dim embedding matrix initialization
    ├─metrics.py                    # model metrics
    ├─steam.py                      # 'steam' dataset text explainer
    └─tbnet.py                      # TB-Net model
  ├─export.py                       # export mindir script
  ├─preprocess_dataset.py           # dataset preprocess script
  ├─preprocess.py                   # inference data preprocess script
  ├─postprocess.py                  # inference result calculation script
  ├─eval.py                         # evaluation
  ├─infer.py                        # inference and explanation
  └─train.py                        # training
```

## [Script Parameters](#contents)

The entire code structure is as following:

```python
data_path: "."                      # The location of input data
load_path: "./checkpoint"           # file path of stored checkpoint file in training
checkpoint_id: 19                   # checkpoint id
same_relation: False                # only generate paths that relation1 is same as relation2
dataset: "steam"                    # dataset name
train_csv: "train.csv"              # the train csv datafile inside the dataset folder
test_csv: "test.csv"                # the test csv datafile inside the dataset folder
infer_csv: "infer.csv"              # the infer csv datafile inside the dataset folder
device_id: 0                        # Device id
device_target: "GPU"                # device id of GPU or Ascend
run_mode: "graph"                   # run code by GRAPH mode or PYNATIVE mode
epochs: 20                          # number of training epochs
```

- preprocess_dataset.py parameters

```text
--dataset         'steam' dataset is supported currently
--device_target   run code on GPU or Ascend NPU
--same_relation   only generate paths that relation1 is same as relation2
```

- train.py parameters

```text
--dataset         'steam' dataset is supported currently
--train_csv       the train csv datafile inside the dataset folder
--test_csv        the test csv datafile inside the dataset folder
--device_id       device id
--epochs          number of training epochs
--device_target   run code on GPU or Ascend NPU
--run_mode        run code by GRAPH mode or PYNATIVE mode
```

- eval.py parameters

```text
--dataset         'steam' dataset is supported currently
--csv             the csv datafile inside the dataset folder (e.g. test.csv)
--checkpoint_id   use which checkpoint(.ckpt) file to eval
--device_id       device id
--device_target   run code on GPU or Ascend NPU
--run_mode        run code by GRAPH mode or PYNATIVE mode
```

- infer.py parameters

```text
--dataset         'steam' dataset is supported currently
--csv             the csv datafile inside the dataset folder (e.g. infer.csv)
--checkpoint_id   use which checkpoint(.ckpt) file to infer
--user            id of the user to be recommended to
--items           no. of items to be recommended
--reasons         no. of recommendation reasons to be shown
--device_id       device id
--device_target   run code on GPU or Ascend NPU
--run_mode        run code by GRAPH mode or PYNATIVE mode
```

## [Inference Process](#contents)

### [Export MindIR](#contents)

```shell
python export.py --config_path [CONFIG_PATH] --checkpoint_path [CKPT_PATH] --device_target [DEVICE] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

- `CKPT_PATH` parameter is required.
- `CONFIG_PATH` is `config.json` file, data and training parameter configuration.
- `DEVICE` should be in ['Ascend', 'GPU'].
- `FILE_FORMAT` should be in ['MINDIR', 'AIR'].

Example：

```bash
python export.py \
  --config_path ./data/steam/config.json \
  --checkpoint_path ./checkpoints/tbnet_epoch19.ckpt \
  --device_target Ascend \
  --file_name model \
  --file_format MINDIR
```

### [Infer on Ascend310](#contents)

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
cd scripts
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` specifies path of used "MINDIR" model.
- `DATA_PATH` specifies path of test.csv.
- `DEVICE_ID` is optional, default value is 0.

Example：

```bash
cd scripts
bash run_infer_310.sh ../model.mindir ../data/steam/test.csv 0
```

### [Result](#contents)

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
auc: 0.8251359368836292
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | GPU                                                                                        | Ascend NPU                                   |
| -------------------------- |--------------------------------------------------------------------------------------------| ---------------------------------------------|
| Model Version              | TB-Net                                                                                     | TB-Net                                       |
| Resource                   | NVIDIA RTX 3090                                                                            | Ascend 910                                   |
| Uploaded Date              | 2022-07-14                                                                                 | 2022-06-30                                   |
| MindSpore Version          | 1.6.1                                                                                      | 1.6.1                                        |
| Dataset                    | steam                                                                                      | steam                                        |
| Training Parameter         | epoch=20, batch_size=1024, lr=0.001                                                        | epoch=20, batch_size=1024, lr=0.001          |
| Optimizer                  | Adam                                                                                       | Adam                                         |
| Loss Function              | Sigmoid Cross Entropy                                                                      | Sigmoid Cross Entropy                        |
| Outputs                    | AUC=0.8573，Accuracy=0.7733                                                                 | AUC=0.8592，准确率=0.7741                      |
| Loss                       | 0.57                                                                                       | 0.59                                         |
| Speed                      | 1pc: 90ms/step                                                                             | 单卡：80毫秒/步                                |
| Total Time                 | 1pc: 297s                                                                                  | 单卡：336秒                                    |
| Checkpoint for Fine Tuning | 686.3K (.ckpt file)                                                                       | 671K (.ckpt 文件)                             |
| Scripts                    | [TB-Net scripts](https://gitee.com/mindspore/models/tree/master/official/recommend/tbnet)  |

### Evaluation Performance

| Parameters                | GPU                        | Ascend NPU                    |
| ------------------------- |----------------------------| ----------------------------- |
| Model Version             | TB-Net                     | TB-Net                        |
| Resource                  | NVIDIA RTX 3090            | Ascend 910                    |
| Uploaded Date             | 2022-07-14                 | 2022-06-30                    |
| MindSpore Version         | 1.3.0                      | 1.5.1                         |
| Dataset                   | steam                      | steam                         |
| Batch Size                | 1024                       | 1024                          |
| Outputs                   | AUC=0.8487，Accuracy=0.7699 | AUC=0.8486，Accuracy=0.7704    |
| Total Time                | 1pc: 5.7s                  | 1pc: 1.1秒                    |

### Inference and Explanation Performance

| Parameters                | GPU                                   |
| --------------------------| ------------------------------------- |
| Model Version             | TB-Net                                |
| Resource                  | Tesla V100-SXM2-32GB                  |
| Uploaded Date             | 2021-08-01                            |
| MindSpore Version         | 1.3.0                                 |
| Dataset                   | steam                                 |
| Outputs                   | Recommendation Result and Explanation |
| Total Time                | 1pc: 3.66s                            |

# [Description of Random Situation](#contents)

- Initialization of embedding matrix in `tbnet.py` and `embedding.py`.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).