# Contents

- [Contents](#contents)
- [Model Description](#model-description)
    - [Description](#description)
    - [Paper](#paper)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Data Preprocessing](#data-preprocessing)
    - [Training Process](#training-process)
        - [Usage](#usage)
            - [Running on Ascend](#running-on-ascend)
            - [Running on GPU](#running-on-gpu)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Running on Ascend](#running-on-ascend-1)
            - [Running on GPU](#running-on-gpu-1)
        - [Result](#result-1)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result-2)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Model Description](#contents)

## Description

CTCModel uses the CTC criterion to train the RNN model to complete the morpheme labeling task. The full name of CTC is
Connectionist Temporal Classification, and the Chinese name is "Connection Time Series Classification". This method
mainly solves the problem that the neural network label and output are not aligned. Supervise the label sequence to
train. CTC is widely used in speech recognition, OCR and other tasks, and has achieved remarkable results.

## Paper

[Paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf): Alex Graves, Santiago Fernández, Faustino J. Gomez, Jürgen
Schmidhuber:
"Connectionist temporal classification": labelling unsegmented sequence data with recurrent neural networks. ICML 2006:
369-376

# [Model Architecture](#contents)

The model includes a two-layer bidirectional LSTM model with an input dimension of 39, which is the dimension of the
extracted speech features, a fully connected layer with an output dimension of 62, the number of labels + 1, and 61
represents a blank symbol.

# [Dataset](#contents)

The dataset used is: [TIMIT](<https://catalog.ldc.upenn.edu/docs/LDC93S1/TIMIT.html>), which includes four formats of
WAV, WRD, TXT, PHN. The official website charges for the TIMIT dataset. At the same time, the .WAV file in the original
TIMIT dataset is not a real .wav file, but a .sph file. The file cannot be used directly and needs to be converted into
a .wav file. Here is the [download link](https://1drv.ms/u/s!AhFKCvZorXL2pneof_90OJZx-cyh?e=51YAIc)
of the converted TIMIT dataset. Preprocessing of the downloaded and decompressed data:

- Read voice data and tag data, extract voice signal features through mfcc and second-order difference
- Fill in the processed data and convert the processed data into MindRecord format
- The data preprocessing script preprocess_data.sh is provided here, which will be described in detail in the data
  preprocessing script section later.
- The length of the training set after preprocessing is 4620, and the length of the test set is 1680

# [Features](#contents)

## Mixed Precision

The mixed precision training method accelerates the deep learning neural network training process by using both the
single-precision and half-precision data types,
and maintains the network precision achieved by the single-precision training at the same
time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or
batch size to be trained on specific hardware. For FP16 operators, if the input data type is FP32, the backend of
MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by
enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend, GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)
- See the `requirements.txt` file, the usage is as follows:

```bash
pip install -r requirements.txt
```

# [Quick Start](#contents)

After installing MindSpore via the official website and [Data Preprocessing](#data-preprocessing), you can start
training and evaluation as follows:

- Running on Ascend

```bash
# distributed training
Usage: bash scripts/train_distributed.sh [TRAIN_PATH] [TEST_PATH] [SAVE_DIR] [RANK_TABLE_FILE]

# standalone training
Usage: bash scripts/train_alone.sh [TRAIN_PATH] [TEST_PATH] [SAVE_DIR] [DEVICE_ID]

# run evaluation example
Usage: bash scripts/eval.sh [TEST_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
```

- Running on GPU

```bash
# distributed training
Usage: bash run_standalone_train_gpu.sh [TRAIN_PATH] [TEST_PATH] [SAVE_DIR]

# standalone training
Usage: bash run_standalone_train_gpu.sh [TRAIN_PATH] [TEST_PATH] [SAVE_DIR] [DEVICE_ID]

# run evaluation
Usage: bash run_eval_gpu.sh [TEST_PATH] [CHECKPOINT_PATH] [DEVICE_ID]

# Description:
# TRAIN_PATH - Training set MindRecord file, eg: /data/Datasets/TIMIT/dataset/train.mindrecord0
# TEST_PATH  - Test set MindRecord file, eg: /data/Datasets/TIMIT/dataset/test.mindrecord0
# SAVE_DIR   - Path to the directory where you need to save the best checkpoint, eg: ./saved_ckpt
```

If you want to run in modelarts, please check the official documentation
of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
├── scripts
│ ├── eval.sh                       # launch Ascend evaluation
│ ├── preprocess_data.sh            # data preprocess script
│ ├── run_distribute_train_gpu.sh   # launch GPU distributed training
│ ├── run_eval_gpu.sh               # launch GPU evaluation
│ ├── run_infer_310.sh              # launch Ascend infer
│ ├── run_standalone_train_gpu.sh   # launch GPU standalone training (1 pcs)
│ ├── train_alone.sh                # launch Ascend standalone training (1 pcs)
│ └── train_distributed.sh          # launch Ascend distributed training
├── src
│ ├── dataset.py                    # data preprocessing
│ ├── eval_callback.py              # evaluation callback while training
│ ├── loss.py                       # custom loss function
│ ├── metric.py                     # custom metrics
│ ├── model.py                      # backbone model
│ ├── model_for_eval.py             # custom network evaluation
│ ├── model_for_train.py            # custom network training
│ └── model_utils
│   ├── config.py                   # parse configuration document
│   ├── device_adapter.py           # distinguish local /modelarts files
│   ├── local_adapter.py            # obtain device information for local training
│   └── moxing_adapter.py           # modelarts configuration, exchange files
├── README.md                       # README EN
├── README_CN.md                    # README CN
├── default_config.yaml             # parameters configuration file
├── eval.py                         # network evaluation
├── export.py                       # export MINDIR format
├── postprocess.py                  # results postprocessing
├── preprocess.py                   # preprocessing script
├── preprocess_data.py              # data preprocessing
├── requirements.txt                # additional required modules
└── train.py                        # training script
```

## [Script Parameters](#contents)

The relevant parameters of data preprocessing, training, and evaluation are in the `default_config.yaml` file.

- Data preprocessing related parameters

```text
dataset_dir     # The directory where the MindRecord files obtained by preprocessing are saved
train_dir       # The directory of the original training data before preprocessing
test_dir        # The directory of the original test data before preprocessing
train_name      # The name of the preprocessed training MindRecord file
test_name       # The preprocessed Test the name of the MindRecord file
```

- Model related parameters

```text
feature_dim         # Input feature dimension, which is consistent with the preprocessed data dimension, 39
batch_size          # batch size
hidden_size         # hidden layer dimension
n_class             # number of labels, dimension of the final output of the model, 62
n_layer             # LSTM layer number
max_sequence_length # maximum sequence length, all sequences are padded to This length, 1555
max_label_length    # The maximum length of the label, all labels are padded to this length, 75
```

- Training related parameters

```text
train_path              # Training set MindRecord file
test_path               # Test set MindRecord file
save_dir                # Directory where the model is saved
epoch                   # Number of iteration rounds
lr_init                 # Initial learning rate
clip_value              # Gradient clipping threshold
save_check              # Whether to save the model
save_checkpoint_steps   # The number of steps to save the model
keep_checkpoint_max     # The maximum number of saved models
train_eval              # Whether to train while testing
interval                # How many steps to test
run_distribute          # Is distributed training
dataset_sink_mode       # Whether data sinking is enabled

```

- Evaluation related parameters

```text
test_path         # Test set MindRecord file
checkpoint_path   # Model save path
test_batch_size   # Test set batch size
beam              # Greedy decode (False) or prefix beam decode (True), the default is greedy decode
```

- Export related parameters

```text
file_name   # export file name
file_format # export file format, MINDIR
```

- Configure related parameters

```text
enable_modelarts  # Whether training on modelarts, default: False
device_target     # Ascend or GPU
device_id         # Device number
```

## [Data Preprocessing](#contents)

Before data preprocessing, please make sure to install the `python-speech-features` library and run the example:

```bash
python preprocess_data.py \
       --dataset_dir ./dataset \
       --train_dir /data/TIMIT/TRAIN \
       --test_dir /data/TIMIT/TEST \
       --train_name train.mindrecord \
       --test_name test.mindrecord
```

```text
Parameters:
    --dataset_dir The path to store the processed MindRecord file, the default is ./dataset, it will be automatically created
    --train_dir The directory where the original training set data is located
    --test_dir The directory where the original test set data is located
    --train_name The name of the training file generated, the default is train.mindrecord
    --test_name The name of the generated test file, the default is test.mindrecord
    Other parameters can be set through the default_config.yaml file
```

Or you can run the script:

```bash
bash scripts/preprocess_data.sh [DATASET_DIR] [TRAIN_DIR] [TEST_DIR]
```

All three parameters are required, respectively corresponding to the above `--dataset_dir`, `--train_dir`, `--test_dir`.

Data preprocessing process is slow, it takes about ten minutes.

## [Training Process](#contents)

### Usage

#### Running on Ascend

- Standalone training

Run the example:

```bash
python train.py \
       --train_path ./dataset/train.mindrecord0 \
       --test_path ./dataset/test.mindrecord0 \
       --save_dir ./save \
       --epoch 120 \
       --train_eval True \
       --interval 5 \
       --device_id 0 > train.log 2>&1 &
```

```text
parameters:
    --train_path training set file path
    --test_path test set file path
    --save_dir model save path
    --epoch iteration rounds
    --train_eval whether to test while training
    --interval Test interval
    --device_id device number
    Other parameters can be set through the default_config.yaml file
```

Or you can run the script:

```bash
bash scripts/train_alone.sh [TRAIN_PATH] [TEST_PATH] [SAVE_DIR] [DEVICE_ID]
```

All four parameters are required, corresponding to the above `--train_path`, `--test_path`, `--save_dir`, `--device_id`.

Commands will run in the background, you can view the results through `train.log`.

The first epoch operator takes a long time to compile, about 60 minutes, and each epoch after that is about 7 minutes.

- Distributed training

The distributed training script is as follows

```bash
bash scripts/train_distributed.sh [TRAIN_PATH] [TEST_PATH] [SAVE_DIR] [RANK_TABLE_FILE]
```

The four parameters are all required, which are the training set mindrecord file path, the test set mindrecord file
path, the model saving path, and the distributed configuration file path.

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

- ModelArts training

```text
modelarts8 card training
(1) Upload the code to the bucket
(2) Upload the processing data to the bucket
(3) Set the code directory, startup file, data set, training output location, job log path
(4) Set parameters: set parameters
    on the web page enable_modelarts=True
    set the parameter run_distribute=True
    on the webpage to set the parameter local_train_path corresponding to the training file path in the container, such as /cache/dataset/train.mindrecord0
    Set the parameter local_test_path on the webpage to correspond to the test file path in the container, such as /cache/dataset/test .mindrecord0
(5) Set up the node
(6) Create a training job
```

#### Running on GPU

```bash
# standalone training
bash run_standalone_train_gpu.sh [TRAIN_PATH] [TEST_PATH] [SAVE_DIR] [DEVICE_ID]

# distributed training
bash run_distribute_train_gpu.sh [TRAIN_PATH] [TEST_PATH] [SAVE_DIR]
```

The four parameters correspond to the above (`--train_path`, `--test_path`, `--save_dir`, `--device_id`).

### Result

```text
# Standalone training results
epoch: 1 step: 72, loss is 139.60443115234375
epoch time: 55464.999 ms, per step time: 770.347 ms
epoch: 2 step: 72, loss is 104.76423645019531
epoch time: 39097.865 ms, per step time: 543.026 ms
epoch: 3 step: 72, loss is 89.34956359863281
epoch time: 38863.849 ms, per step time: 539.776 ms
epoch: 4 step: 72, loss is 76.7833023071289
epoch time: 38395.734 ms, per step time: 533.274 ms
epoch: 5 step: 72, loss is 67.78717041015625
epoch time: 38008.225 ms, per step time: 527.892 ms
epoch: 5, ler: 0.7277136554647986
update best result: 0.7277136554647986
...
```

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

Make sure to install the edit-distance library before evaluating and run the example:

```bash
python eval.py \
       --test_path ./dataset/test.mindrecord0 \
       --checkpoint_path ./save/best.ckpt \
       --beam False \
       --device_id 0 > eval.log 2>&1 &
```

```text
parameters:
    --test_path test Set file path
    --checkpoint_path path to load model
    --device_id device number
    --beam greedy decoding or prefix beam decoding
    Other parameters can be set through the default_config.yaml file
```

Or you can run the script:

```bash
bash scripts/eval.sh [TEST_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
```

The 3 parameters are all required, respectively corresponding to the above `--test_path`, `--checkpoint_path`,
`--device_id`.

The above command runs in the background, you can view the results through `eval.log`.

The default beam_size of prefix beam search is 5, and the result is slightly better, but because it is implemented by
itself, the speed is very slow. The evaluation takes about 1 hour. It is recommended to use greedy, and the evaluation
takes about 5 minutes.

#### Running on GPU

```bash
bash run_eval_gpu.sh [TEST_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
```

The 3 parameters are all required, respectively corresponding to the above.

### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like
the following in log.

- Ascend

```text
greedy decode run result
{'ler': 0.3038}

prefix beam search decode run result
{'ler': 0.3005}
```

- GPU

```text
{'ler': 0.3110710076038643}
```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --checkpoint_path="./save/best.ckpt"
```

Export on ModelArts (If you want to run in modelarts, please check the official documentation
of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

### Infer on Ascend310

Before performing inference, the mindir file needs to be exported via export.py. The input data file is in bin format.

```shell
bash scripts/run_infer_310.sh [MINDIR_PATH] [TEST_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

The four parameters represent the mindir file address, the test set data storage path, whether to preprocess the data,
and the device number.

### result

The inference result is saved in the current path, and the final accuracy result can be seen in acc.log.

```text
READ:{'read': 0.3038}
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters          | Ascend 910 (8 pcs)                                                             | GPU (1 pcs)                                                                    | GPU (8 pcs)                                                                    |
|---------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Model Version       | CTCModel                                                                       | CTCModel                                                                       | CTCModel                                                                       |
| Resource            | Ascend 910                                                                     | Ubuntu 18.04.5, GF RTX 3090, CPU 2.90 GHz, 64 cores, RAM 252 GB                | Ubuntu 18.04.5, GF RTX 3090, CPU 2.90 GHz, 64 cores, RAM 252 GB                |
| Uploaded Date       | 2021-11-03                                                                     | 2022-01-17                                                                     | 2022-01-17                                                                     |
| MindSpore Version   | 1.3.0                                                                          | 1.6.0                                                                          | 1.6.0                                                                          |
| Dataset             | TIMIT, training set length 4620                                                | TIMIT, training set length 4620                                                | TIMIT, training set length 4620                                                |
| Training Parameters | 8p, epoch=300, batch_size=64, lr_init=0.01, clip_value=5.0                     | epoch=300, batch_size=64, lr_init=0.005, clip_value=5.0                        | epoch=500, batch_size=128, lr_init=0.005, clip_value=5.0                       |
| Optimizer           | Adam                                                                           | Adam                                                                           | Adam                                                                           |
| Loss Function       | CTCLoss                                                                        | CTCLoss                                                                        | CTCLoss                                                                        |
| Outputs             | LER                                                                            | LER                                                                            | LER                                                                            |
| Loss                | 24.5                                                                           | 15.87                                                                          | 24.4                                                                           |
| Speed               | 6299.475 ms/step                                                               | 530 ms/step                                                                    | 1060 ms/step                                                                   |
| Total time          | about 7h                                                                       | 190 mins                                                                       | 35 mins                                                                        |
| Checkpoint          | 7.2M (.ckpt file)                                                              | 7.2M (.ckpt file)                                                              | 7.2M (.ckpt file)                                                              |
| Script              | [Link](https://gitee.com/mindspore/models/tree/master/research/audio/ctcmodel) | [Link](https://gitee.com/mindspore/models/tree/master/research/audio/ctcmodel) | [Link](https://gitee.com/mindspore/models/tree/master/research/audio/ctcmodel) |

**LER** - edit distance between predicted label sequence and true label sequence, the smaller the better.

### Inference Performance

| Parameters          | Ascend                                                      | GPU (1 pcs)                                                 | GPU (8 pcs)                                                 |
| ------------------- |-------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------|
| Model Version       | CTCModel                                                    | CTCModel                                                    | CTCModel                                                    |
| Resource            | Ascend 910                                                  | GF RTX 3090                                                 | GF RTX 3090                                                 |
| Uploaded Date       | 2021-11-03                                                  | 2022-01-17                                                  | 2022-01-17                                                  |
| MindSpore Version   | 1.3.0                                                       | 1.6.0                                                       | 1.6.0                                                       |
| Dataset             | TIMIT, test set size 1680                                   | TIMIT, test set size 1680                                   | TIMIT, test set size 1680                                   |
| batch_size          | 1                                                           | 1                                                           | 1                                                           |
| outputs             | LER:0.3038 (greedy decode), LER:0.3005 (prefix beam decode) | LER:0.3111 (greedy decode), LER:0.3081 (prefix beam decode) | LER:0.3162 (greedy decode), LER:0.3074 (prefix beam decode) |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset“ function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).

# FAQ

Refer to the [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) for some common question.