# Directory

[View Chinese](./README_CN.md)

- [Directory](#directory)
- [Soft-Masked BERT](#soft-maskedbert)
- [Model Architecture](#model architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment requirements)
- [Quick Start](#quick start)
- [Script description](#script description)
    - [Script parameters](#script parameters)
    - [Training process](#training process)
        - [Single player training](#single player training)
        - [Distributed training](#distributed training)
    - [Inference](#inference)
        - [Reasoning process](#reasoning process)
- [Model Description](#model description)
    - [Performance](#performance)
        - [Training Performance](#training performance)
        - [Inference Performance](#inference performance)
- [Contribution Guide](#contribution guide)
    - [Contributors](#contributors)
- [ModeZoo Homepage](#modezoo homepage)

<TOC>

# Soft-Masked BERT

[paper](https://arxiv.org/pdf/2005.07421v1.pdf)：Zhang S, Huang H, Liu J, et al. Spelling error correction with soft-masked BERT[J]. arXiv preprint arXiv:2005.07421, 2020.

## Model Architecture

> Soft-masked BERT consists of a detection network based on BI-GRU and a correction network based on BERT. The probability of network prediction error is detected and the probability of network prediction error correction is corrected, while the detection network transmits the prediction results to the correction network by soft masking.

## Dataset

1. Download [SIGHAN dataset](http://nlp.ee.ncu.edu.tw/resource/csc.html)
1. Unpack the dataset above and copy all the ".sgml "files in the folder to the datasets/csc/directory
1. Copy 'sighan15_csc_testInt. TXT' and 'sighan15_csc_testtrut. TXT' to the datasets/csc/directory
1. [download] (https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml) to datasets/csc directory
1. Ensure that the following files are in datasets/csc

```text
train.sgml
B1_training.sgml
C1_training.sgml
SIGHAN15_CSC_A2_Training.sgml
SIGHAN15_CSC_B2_Training.sgml
SIGHAN15_CSC_TestInput.txt
SIGHAN15_CSC_TestTruth.txt
```

6. Preprocess the data(Please refer to the requirement.txt installation for the dependency package required to run the script.)

```python
python preprocess_dataset.py
```

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend/GPU/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)
- Dependencies
    - The installation depends on
      > pip install -r requirements.txt
- version problem
    - If the GLIBC version is too late, install an earlier version of openCC (e.g. 1.1.0).

## Quick Start

1. Store preprocessed data in the datasets directory.
2. Download [bert-base-chinese-vocab.txt](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt) and put it in the src/ file.
3. Download[pre-trained model](https://download.mindspore.cn/models/r1.3/bertbase_ascend_v130_cnnews128_official_nlp_loss1.5.ckpt) and put it in the weight/ file。
4. Execute the training script.
- Train on offline servers

```python
# Distributed training
bash scripts/run_distribute_train.sh [RANK_SIZE] [RANK_START_ID] [RANK_TABLE_FILE] [BERT_CKPT]
BERT_CKPT:Pre-trained BERT file name (for example bert_base.ckpt)

# Single training
bash scripts/run_standalone_train.sh [BERT_CKPT] [DEVICE_ID] [PYNATIVE]
BERT_CKPT:Pre-trained BERT file name (for example bert_base.ckpt)
DEVICE_ID:ID of the running machine
PYNATIVE:Whether to run in PYNATIVE mode (default False)
```

- while the ModelArts for training (if you want to run on ModelArts, can refer to the following document [ModelArts] (https://support.huaweicloud.com/modelarts/))

```text
# (1) Go to [code warehouse](https://git.openi.org.cn/OpenModelZoo/SoftMaskedBert) and create a training task.
# (2) Set "enable_modelarts=True; bert_ckpt=bert_base.ckpt"
# (3) If running in Pynative mode, set "pynative=True"
# (4) Set dataset "softmask.zip" on the web page
# (5) Set the startup file to "train.py"
# (6) run training task
```

5. Execute the evaluation script.

After the training, follow these steps to initiate the evaluation:

```python
# assessment
bash scripts/run_eval.sh [BERT_CKPT_NAME] [CKPT_DIR]
```

## Script description

```text
├ ─ ─ model_zoo
├─ Readme.md // All model related instructions
├ ─ ─ soft - maksed - Bert
├─ Readme.md // Googlenet
├── Ascend310_infer // Implement 310 inference source code
├ ─ ─ scripts
│ ├─ Run_Train. Sh // Distributed to Ascend shell script
│ ├─ Run_eval. sh // Ascend evaluation shell script
│ ├─ Run_INFER_310.sh // Ascend Reasoning shell Script
├ ─ ─ the SRC
│ ├─ Soft Maksed Bert // Soft Maksed Bert
├─ Train.py //
├─ Eval. Py // Evaluation script
├─ Postprogress.py // 310 Reasoning Postprocessing script
├─ export.py // Checkpoint file export

├── model_zoo
    ├── README.md                          // All model related instructions
    ├── soft-maksed-bert
        ├── README.md                    // softmasked-BERT related instructions
        ├── README_CN.md             // softmasked-BERT related instructions in Chinese
        ├── ascend310_infer              // Implement 310 inference source code
        ├── scripts
        │   ├──run_distribute_train.sh             // Distributed to Ascend shell script
        │   ├──run_standalone_train.sh          // Ascend single machine training shell script
        │   ├──run_eval.sh                  // Ascend evaluation shell script
        │   ├──run_infer_310.sh         // Ascend inferences shell scripts
        │   ├──run_preprocess.sh      // Run a shell script for data preprocessing
        ├── src
        │   ├──soft_masked_bert.py           //  soft-maksed bert architecture
        │   ├──bert_model.py                    //  BERT architecture
        │   ├──dataset.py                          //   Data set processing
        │   ├──finetune_config.py             //   Model's hyperparameter
        │   ├──gru.py                               //   GRU architecture
        │   ├──tokenization.py                 //   Words tokenizer
        │   ├──util.py                               //   tools
        ├── train.py               // Training script
        ├── eval.py               // Evaluation of the script
        ├── postprogress.py       // 310 Inference postprocessing scripts
        ├── export.py            // Export the checkpoint file
        ├── preprocess_dataset.py            // Data preprocessing
```

### Script parameters

```python
'Batch size':36 # batch size
'epoch':100 # Total training epoch number
'Learning rate':0.0001 # Initial learning rate
'Loss function':'BCELoss' # Loss function used for training
'Optimizer ':AdamWeightDecay # Activate function
```

## Training process

### Single player training

- Ascend runs in the processor environment

```python
bash scripts/run_standalone_train.sh [BERT_CKPT] [DEVICE_ID] [PYNATIVE]
```

After the training, you can find the checkpoint file in the default scripts folder. The operation process is as follows:

```python
Epoch: 1 Step: 152, loss is 3.3235654830932617
Epoch: 1 Step: 153, loss is 3.6958463191986084
Epoch: 1 Step: 154, loss is 3.585498571395874
Epoch: 1 Step: 155, loss is 3.276094913482666
```

## Distributed training

- Ascend runs in the processor environment

```python
Bash run_distribute_train_smb.sh [RANK_SIZE] [RANK_START_ID] [RANK_TABLE_FILE] [BERT_CKPT]
```

The shell script above runs the distributed training in the background.

```python
Epoch: 1 Step: 12, Loss is 7.957302093505859
Epoch: 1 Step: 13, loss is 7.886098861694336
Epoch: 1 Step: 14, Loss is 7.781495094299316
Epoch: 1 Step: 15, Loss is 7.755488395690918

```

### [Inference](#contents)

#### Reasoning process

Before performing inference, the mindir file must be exported by export.py. Input files must be in bin format.

```python
# Export mindir file
python export.py --bert_ckpt [BERT_CKPT] --ckpt_dir [CKPT_DIR]
# Ascend310 inference
bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_FILE_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

`BERT_CKPT` means the pre-train BERT filename. (e.g. bert_base.ckpt)
`CKPT_DIR` means the trained ckpt file path. (e.g. ./checkpoint/SoftMaskedBert-100_874.ckpt)
`MINDIR_PATH` means the directory of the model file.
`DATA_FILE_PATH` means the directory of the input data.
`NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
`DEVICE_ID` is optional, default value is 0.

#### result

Inference result is saved in the project's main path, you can find result in acc.log file.

```eval log
1 The detection result is precision=0.6733436055469953, recall=0.6181046676096181 and F1=0.6445427728613569
2 The correction result is precision=0.8260869565217391, recall=0.7234468937875751 and F1=0.7713675213675213
3 Sentence Level: acc:0.606364, precision:0.650970, recall:0.433579, f1:0.520487
```

# Model Description

## Performance

### Training Performance

| Parameters                   | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | BERT-base                                                |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8            |
| uploaded Date              | 2022-06-28                                 |
| MindSpore版本          | 1.6.0                                                       |
| Dataset                    | SIGHAN                                                    |
| Training Parameters        | epoch=100, steps=6994, batch_size = 36, lr=0.0001              |
| Optimizer                  | AdamWeightDecay                                                    |
| Loss Function              | BCELoss                                       |
| Loss                       | 0.0016                                                      |
| Speed                      | 1p：349.7ms/step;  8p：314.7ms/step                          |
| Total time                 | 1p：4076mins;  8p：458mins                          |
| Checkpoint for Fine tuning | 459M (.ckpt文件)                                         |
| Scripts                    | [link](https://gitee.com/rafeal8830/soft-maksed-bert/edit/master/README_TEMPLATE_CN.md) |

### Inference Performance

> Provide the detail of evaluation performance including latency, accuracy and so on.

e.g. you can reference the following template

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet18                    |
| Resource            | Ascend 910; OS Euler2.8     |
| Uploaded Date       | 02/25/2021 (month/day/year) |
| MindSpore Version   | 1.7.0                       |
| Dataset             | CIFAR-10                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 94.02%                      |
| Model for inference | 43M (.air file)             |

## Contribution Guide

If you want to contribute, please review the [contribution guidelines](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING.md) and [how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

### Contributors

* [c34](https://gitee.com/c_34) (Huawei)

## ModeZoo Homepage

Please check the official [homepage](https://gitee.com/mindspore/models).