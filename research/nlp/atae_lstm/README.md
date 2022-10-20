# Content

<!-- TOC -->

- [Content](#content)
- [AttentionLSTM description](#attentionlstm-description)]
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
    - [Requirements Installation](#requirements-installation)
    - [Dataset preprocessing](#dataset-preprocessing)
    - [Running](#running)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Script parameters](#script-parameters)
    - [Training process](#training-process)
        - [Train](#train)
    - [Evaluation process](#evaluation-process)
        - [evaluate](#evaluate)
    - [Export process](#export-process)
        - [Export](#export)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Inference performance](#inference-performance)
        - [Transfer learning](#transfer-learning)
- [Random](#random)
- [ModelZoo Homepage](#modelzoo-homepage)  

<!-- /TOC -->

# AttentionLSTM Description

AttentionLSTM can also be referred to as ATAE_LSTM. The paper mainly proposes a network model suitable for fine-grained text emotional polarity analysis.We know that some hotels or commodity comments are often more than a good aspects of goods, for example, "Pizza's taste is great! But the service is really poor!" The review involved "Food", "Service" evaluationThe ordinary LSTM model cannot capture information about aspects, so it will only produce one result regardless of which aspects.The ATTENTION-based LSTM with Aspect Embedding model uses aspect vector and the Attention mechanism provides a good solution to this sub-evaluation problem.

[Paper](https://www.aclweb.org/anthology/D16-1058.pdf)：Wang, Y. , et al. "Attention-based LSTM for Aspect-level Sentiment Classification." Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing 2016.

# Model Architecture

![62486723912](https://gitee.com/honghu-zero/mindspore/raw/atae_r1.2/model_zoo/research/nlp/atae_lstm/src/model_utils/ATAE_LSTM.png)

The input of the AttentionLSTM model consists of an ASPECT and WORD versions, and the input section Enter a single layer LSTM network to obtain a status vector, then connect the status vector to the aspect, calculate the Attention weight, and finally use Attention weight and state vectors to obtain an emotional polarity classification.

# Dataset

Used data set：[SemEval 2014 task4](https://alt.qcri.org/semeval2014/task4) Restaurant (aspect category)

- Dataset size:
    - Training set: 2990 sentences, each sentence corresponds to an Aspect and a polar classification
    - Test set: 973 sentences, each sentence corresponds to an Aspect and a polar classification
- Data format: XML or COR file
    - Note: The data will be processed in create_dataset.py to convert to the MindRecord format.

# Characteristics

## Mixed Precision

**Mixed Precision:** the training method uses support for single-precision and semi-precision data to increase the training speed of deep learning neural network while maintaining the network accuracy that single-precision training can achieve. Mixed precision training increases the calculation speed, reducing memory usage, supports a larger training on a particular hardware or a larger batch training. Taking the FP16 operator as an example, if the input data type is FP32, the MINDSPORE background will automatically reduce precision to process data. Users can open the INFO log, search "Reduce Precision" to see the accuracy reduced operator.

# Environment Requirements

- Hardware (ASCEND of GPU)
    - Use the Ascend processor of GPU to build a hardware environment.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For details, please refer to the following resources:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# Quick Start

## Dataset preprocessing

To convert data to MindRecord format, GloVe file is needed. Global Vector or GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Preprocessing can be made with shell script `convert_dataset.sh`

Link to GloVe file: [glove.840B.300d] (http://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip)
Link to folder with COR files used during training: [SemEval 2014 task4 cor files](https://github.com/TianHongZXY/pytorch-atae-lstm/tree/master/data)

```bash
bash scripts/convert_dataset.sh [PATH_TO_DATA_FOLDER] [PATH_TO_GLOVE_FILE]
```

- Example of command for handling the original data set：

  ```bash
  bash scripts/convert_dataset.sh \
      /home/workspace/atae_lstm/data \
      /home/workspace/atae_lstm/data/glove.840B.300d.txt
  ```

  The above command will convert data set to MindRecord format. It will appear in `train.mindrecord` and `test.mindrecord` files. Also file `weight.npz` will be created, it is necessary for training process.

## Running

After installing MINDSPORE through the official website, you can follow the steps:

- Ascend processor environment operation

  ```bash
  # Run training example
  bash scripts/run_train_ascend.sh [DATA_DIR]

  # evaluation
  bash scripts/run_eval.sh [DEVICE] [DATA_DIR] [CKPT_FILE]
  ```

- GPU environment

  ```bash
  # Standalone training
  bash scripts/run_standalone_train_gpu.sh [DATA_DIR]

  # Distributed training
  bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATA_DIR]

  # evaluation
  bash scripts/run_eval.sh [DEVICE] [DATA_DIR] [CKPT_FILE]
  ```

# Script Description

## Script and sample code

```text
├── atae_lstm
    ├── README.md        // AttentionLSTM related instructions
    ├── ascend310_infer  // Application for 310 inference
    ├── modelarts        // Folder for modelarts mode
    ├── infer            // Conponents for Ascend inference
    ├── scripts
    │   ├──convert_dataset.sh           // Shell script for converting original data
    │   ├──run_distribute_train_gpu.sh  // Shell script for distributed training on GPU
    │   ├──run_eval.sh                  // Shell script evaluated on Ascend or GPU
    │   ├──run_standalone_train_gpu.sh  // Shell script for standalone training on GPU
    │   ├──run_train_ascend.sh          // Shell script for training on Ascend
    ├── src
    │   ├──model_utils
    │   │   ├──my_utils.py   // LSTM related components
    │   │   ├──rnn_cells.py  // LSTM unit
    │   │   ├──rnns.py       // LSTM
    │   ├──config.py          // Parameter generation
    │   ├──load_dataset.py    // Data loader
    │   ├──model.py           // Model file
    │   ├──atae_for_train.py  // Model training file
    │   ├──atae_for_test.py   // Model assessment file
    ├── create_dataset.py  // Dataset preprocessing
    ├── eval.py            // Evaluation script
    ├── export.py          // Export checkpoint files to Air/mindir
    ├── postprocess.py     // 310 post-processing script
    ├── preprocess.py      // 310 pre-processing script
    ├── README_CN.md       // Chinese readme
    ├── README.md          // English readme
    ├── requirements.txt   // pip requirements
    ├── train.py           // Training script
```

## Script parameters

Training parameters and evaluation parameters can be simultaneously configured in default_config.yaml.

```text
batch_size: 25          # training batch size
epoch_size: 25          # Total number of training epochs
momentum: 0.9           # momentum
weight_decay: 0.001     # Weight decay value
dim_hidden: 300         # hidden layer dimension
rseed: 4373337
dim_word: 300           # word vector dimension
dim_aspect: 300         # aspect vector dimension
optimizer: 'Adagrad'    # optimizer type
vocab_size: 5177        # vocabulary size
dropout_prob: 0.6       # dropout probability
aspect_num: 5           # number of aspect words
grained: 3              # Number of polar classifications
lr: 0.01                # learning rate
lr_word_vector: 0.001   # learning rate for word vector
```

For more configuration details, please refer to the script `default_config.yaml`.

## Training process

### Train

- Training on Ascend

  ```bash
  bash scripts/run_train_ascend.sh  \
      /home/workspace/atae_lstm/data/  \
      /home/workspace/atae_lstm/train/
  ```

  The command of the above training network will run in the background, results will appear in net_log.log.

  After training, you can find checkpoint files under the default script folder. The loss value is achieved in the following ways:

  ```text
  # grep "loss is " net.log
  epoch:1 step:2990, loss is 1.4842823
  epcoh:2 step:2990, loss is 1.0897788
  ...
  ```

  Model checkpoints are saved in the current directory.

  After training, you can find the checkpoint file in the default `./train/` script folder.

- Training on GPU

  ```bash
  # Standalone training
  bash scripts/run_standalone_train_gpu.sh /home/workspace/atae_lstm/data/

  # Distributed training
  bash scripts/run_distribute_train_gpu.sh 8 /home/workspace/atae_lstm/data/
  ```

## Evaluation process

### Evaluation

- Evaluation on Ascend or GPU

  ```bash
    bash scripts/run_eval.sh [DEVICE] \
        /home/workspace/atae_lstm/data/ \
        /home/workspace/atae_lstm/train/atae-lstm_max.ckpt
  ```

  The above Python command will run in the background, you can view the results via the `eval/eval.log` file. The accuracy of the test data set is as follows:

  ```text
  # grep "accuracy:" eval.log
  accuracy:{'acc':0.8253}
  ```

## Export process

### Export

You can use the following command to export the mindir file

```bash
python export.py --existed_ckpt="./train/atae-lstm_max.ckpt" \
                 --word_path="./data/weight.npz"
```

## Inference process

### Usage

Before performing inference, you need to export the mindir file via export.py. Convert the file to bin format.

```bash
# 文件预处理
python preprocess.py --data_path="./data/test.mindrecord"

# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_TARGET] [DEVICE_ID]
```

`DEVICE_TARGET` Optional value range：['GPU', 'CPU', 'Ascend']；
`DEVICE_ID` Optional, the default is 0.

### Result

The reasoning result is saved in the current path, and the final accuracy result can be seen in acc.log.

# Model description

|     parameter     |                          Ascend                       |    GPU              |
| :---------------: | :---------------------------------------------------: | :-----------------: |
|     resource      | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 | 8x RTX3090          |
|   upload date     |                        2021-12-11                     | 2022-05-17          |
| MindSpore version |                          1.5.0                        | 1.6.1               |
|    dataset        |                    SemEval 2014 task4                 | SemEval 2014 task4  |
| training parameters |      epoch=25, batch_size=1, lr=0.0125              | epoch=25, batch_size=25, lr=0.01 |
| optimizer         |                     Momentum                          | Adagrad             |
| Accuracy          |   0.8253                                              | 0.8336              |
|   parameter (M)   |                        2.68                           | 2.68                |
| Fine-tune checkpoints |                    26M                            | 26M                 |
|  inference model  |            9.9M(.air文件)、11M(.mindir文件)             | 9.9M(.air文件)、11M(.mindir文件) |

# Random

In train.py and eval.py, we set a random seed.

# ModelZoo Homepage

 Please browse official website [Homepage](https://gitee.com/mindspore/models).
