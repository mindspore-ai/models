
# Contents

[查看中文](./README_CN.md)

<!-- TOC -->

- [Contents](#contents)
- [KT-NET Description](#ktnet-description)
- [Model Architecture](#model-architecture)
- [Knowledge Bases](#knowledge-bases)
- [Datasets](#datasets)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Training](#training)
        - [Evaluation](#evaluation)
    - [Training Process](#training-process)
        - [Ascend](#ascend)
        - [GPU](#gpu)
    - [Evaluation Process](#evaluation-process)
        - [Ascend](#ascend-evaluation)
        - [GPU](#gpu-evaluation)
    - [Inference Process](#inference-process)
        - [Usage](#usage)
        - [Results](#results)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# KTNET Description

Knowledge and Text fusion Network is MRC (Machine Reading Comprehension) model, it integrates knowledge from KB (Knowledge Bases) into pre-trained contextual expression. This model is proposed in a paper, and aims to enhance pre-trained language expression with rich knowledge to improve machine reading comprehension ability.

[Paper](https://www.aclweb.org/anthology/P19-1226/):  Yang A ,  Wang Q ,  Liu J , et al. Enhancing Pre-Trained Language Representations with Rich Knowledge for Machine Reading Comprehension[C]// Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.

# Model Architecture

KT-NET consists of four major modules:

- Layer 1: BERT Encoding layer，compute deep, context-aware representations for question and passage
- Layer 2: Knowledge Integration layer，select the corresponding knowledge vector from KBs (knowledge bases) and integrate them with BERT representations
- Layer 3：Self-Matching layer，fuse BERT and KB representations
- Layer 4：Output layer，predict the final answer

# Knowledge Bases

- Before training the model, relevant knowledge should be retrieved and coded. In this project, we used two kb: WordNet and NELL. WordNet records the relationships between words, and NELL stores beliefs about entities. The following procedure describes how to retrieve related WordNet synsets and NELL concepts for MRC samples.

  ```bash
  curl -O https://raw.githubusercontent.com/bishanyang/kblstm/master/embeddings/wn_concept2vec.txt
  curl -O https://raw.githubusercontent.com/bishanyang/kblstm/master/embeddings/nell_concept2vec.txt
  mv wn_concept2vec.txt nell_concept2vec.txt data/KB_embeddings
  ```

- retrieve_nell
  [Retrieve NELL](https://baidu-nlp.bj.bcebos.com/KTNET_preprocess_nell_concepts.tar.gz)
  Please unzip the downloaded file and place it into the data/retrieve_nell/ directory of this repository.
  Or just run the following command inside the data/retrieve_nell/ directory:

  ```bash
  wget -c https://baidu-nlp.bj.bcebos.com/KTNET_preprocess_nell_concepts.tar.gz -O - | tar -xz
  ```

- retrieve_wordnet
  [Retrieve WordNet](https://baidu-nlp.bj.bcebos.com/KTNET_preprocess_wordnet_concepts.tar.gz)
  Please unzip the downloaded file and place it into the data/retrieve_wordnet/ directory of this repository.
  Or just run the following command inside the data/retrieve_wordnet/ directory:

  ```bash
  wget -c https://baidu-nlp.bj.bcebos.com/KTNET_preprocess_wordnet_concepts.tar.gz -O - | tar -xz
  ```

- tokenization_record
  [Tokenization record](https://baidu-nlp.bj.bcebos.com/KTNET_preprocess_tokenize_result_record.tar.gz)
  Please unzip the downloaded file and place it into the data/tokenization_record/ directory of this repository.
  Or just run the following command inside the data/tokenization_record/ directory:

  ```bash
  wget -c https://baidu-nlp.bj.bcebos.com/KTNET_preprocess_tokenize_result_record.tar.gz -O - | tar -xz
  ```

- tokenization_squad
  [Tokenization squad](https://baidu-nlp.bj.bcebos.com/KTNET_preprocess_tokenize_result_squad.tar.gz)
  Please unzip the downloaded file and place it into the data/tokenization_squad/ directory of this repository.
  Or just run the following command inside the data/tokenization_squad/ directory:

  ```bash
  wget -c https://baidu-nlp.bj.bcebos.com/KTNET_preprocess_tokenize_result_squad.tar.gz -O - | tar -xz
  ```

# Datasets

- **ReCoRD**（read-understanding with Commonsense Reasoning Dataset）It is a large-scale MRC data set that requires common sense reasoning. The official data set in JSON format, you can use the following link to download
    - [train](https://drive.google.com/file/d/1PoHmphyH79pETNws8kU2OwuerU7SWLHj/view) - 216 MB, 100000
    - [dev](https://drive.google.com/file/d/1WNaxBpXEGgPbymTzyN249P4ub-uU5dkO/view) - 24,3 MB, 10000

    Please download train.json and dev.json and place it into the data/ReCoRD/ directory of this repository

- **SQuAD v1.1** is a well-known extractive MRC dataset that consists of questions created by crowdworkers for Wikipedia articles.
  Please run the following command to download the official dataset and evaluation script.

  ```bash
  cd data/SQuAD
  curl -O https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
  curl -O https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
  ```

  Please download train-v1.1.json and dev-v1.1.json and place it into the data/SQuAD/ directory of this repository

- Prepare BERT checkpoint (this step is required to convert dataset to mindspore format - vocab.txt is needed)

  ```bash
  cd data
  wget https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip --no-check-certificate
  tar xvf cased_L-24_H-1024_A-16.tar.gz
  rm -rf cased_L-24_H-1024_A-16.tar.gz
  ```

- Run the following command to convert the two datasets ReCoRD and SQuAD into mindrecord format
  (this step requires ~ 250GB RAM)

  ```bash
  python data_convert.py --data_url=./data
  ```

  The parameter data_url represents the path of the data folder, and the default is ./data. After running successfully, both datasets will be automatically converted to mindrecord format and stored in the data/ReCoRD/ directory and data/SQuAD/ directory respectively.

- convert BERT checkpoint to the format corresponding to Mindspore (require mindspore and paddle environment)

  ```bash
  python src/bert_ms_format.py --data_url=./data
  ```

  The parameter data_url represents the path to data，the default is ./data. After running successfully, it will generate BERT checkpoint in data/cased_L-24_H-1024_A-16/roberta.ckpt.

The overall directory structure of the data files is as follows:

```shell
.
└─data
  ├─KB_embeddings                       # Embedded data in the knowledge base
    ├─nell_concept2vec.txt
    ├─wn_concept2vec.txt
  ├─ReCoRD                              # ReCoRD dataset
    ├─dev.json
    ├─train.json
    ├─dev.mindrecord
    ├─dev.mindrecord.db
    ├─train.mindrecord
    ├─train.mindrecord.db
  ├─SQuAD                               # SQuAD dataset
    ├─dev-v1.1.json
    ├─train-v1.1.json
    ├─dev.mindrecord
    ├─dev.mindrecord.db
    ├─train.mindrecord
    ├─train.mindrecord.db
  ├─retrieve_nell                       # Retrieving data from the NELL knowledge base
    ├─output_record
      ├─dev.retrieved_nell_concepts.data
      ├─train.retrieved_nell_concepts.data
    ├─output_squad
      ├─dev.retrieved_nell_concepts.data
      ├─train.retrieved_nell_concepts.data
  ├─retrieve_wordnet                    # Retrieving data from the WordNet knowledge base
    ├─output_record
      ├─retrived_synsets.data
    ├─output_squad
      ├─retrived_synsets.data
  ├─tokenization_record                 # ReCoRD dataset tokenization
    ├─tokens
      ├─dev.tokenization.cased.data
      ├─dev.tokenization.uncased.data
      ├─train.tokenization.cased.data
      ├─train.tokenization.uncased.data
  ├─tokenization_squad                  # SQuAD dataset tokenization
    ├─tokens
      ├─dev.tokenization.cased.data
      ├─dev.tokenization.uncased.data
      ├─train.tokenization.cased.data
      ├─train.tokenization.uncased.data
  ├─cased_L-24_H-1024_A-16              # BERT
    ├─params
    ├─bert_config.json
    ├─vocab.txt
    ├─roberta.ckpt
```

# Environment Requirements

- Hardware (GPU/CPU/Ascend)
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- Other requirements
    - python >= 3.7
    - mindspore 1.6.0.20211129
    - paddlepaddle 2.0
    - NLTK >= 3.3 (with WordNet 3.0)

- For more information about MindSpore, please check the resources below:
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# Script Description

## Script and Sample Code

```shell
.
└─KTNET
  ├─README.md
  ├─README_CN.md
  ├─scripts
    ├─__init__.py
    ├─export.sh                           # Model output script
    ├─run_infer_310.sh
    ├─run_record_twomemory.sh             # Stand-alone training shell script on the Ascend device（record dataset）
    ├─run_record_twomemory_gpu.sh
    ├─run_record_twomemory_distribute.sh  # 8-cards training shell script on Ascend device（record dataset）
    ├─run_record_twomemory_distribute_gpu.sh
    ├─run_squad_twomemory.sh              # Stand-alone training shell script on the Ascend device（squad dataset）
    ├─run_squad_twomemory_gpu.sh
    ├─run_squad_twomemory_distribute.sh   # 8-cards training shell script on Ascend device（squad dataset）
    ├─run_squad_twomemory_distribute_gpu.sh
    ├─run_squad_eval.sh                   # Stand-alone evaluation shell script on Ascend device（record dataset）
    ├─run_squad_eval_gpu.sh
    ├─run_record_eval.sh                  # Stand-alone evaluation shell script on Ascend device（squad dataset）
    ├─run_record_eval_gpu.sh
  ├─src
    ├─reader                              # Data preprocessing
      ├─__init__.py
      ├─batching_twomemory.py
      ├─record_official_evaluate.py
      ├─record_twomemory.py
      ├─squad_twomemory.py
      ├─squad_v1_official_evaluate.py
      ├─tokenization.py
    ├─__init__.py
    ├─bert_ms_format.py                   # convert BERT checkpoint to the format corresponding to Mindspore
    ├─KTNET.py                            # Network backbone coding
    ├─KTNET_eval.py                       # Evaluation method of the evaluation process
    ├─bert.py                             # Network backbone coding
    ├─layers.py                           # Network backbone coding
    ├─dataset.py                          # Read mindrecord format data
    ├─data_convert.py                     # Process data into mindrecord format
  ├─utils
    ├─__init__.py
    ├─args.py
    ├─util.py
  ├─ascend310_infer
    ├─inc
      ├─utils.h
    ├─src
      ├─main.cc
      ├─utils.cc
    ├─build.sh
    ├─CMakeLists.txt
  ├─run_KTNET_squad.py                    # train（squad dataset）
  ├─run_KTNET_squad_eval.py               # evaluate（squad dataset）
  ├─run_KTNET_record.py                   # train（record dataset）
  ├─run_KTNET_record_eval.py              # evaluate（record dataset）
  ├─export.py
  ├─postprocess.py
  ├─preprocess.py
```

## Script Parameters

### Training

- To train model on SQuAD dataset:

  ```bash
  python run_KTNET_squad.py  [--device_target DEVICE_TARGET] [--device_id N] [batch_size N] [--do_train True] [--do_predict False] [--do_lower_case False] [--init_pretraining_params INIT_PRETRAINING_PARAMS] [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH] [--load_checkpoint_path LOAD_CHECKPOINT_PATH] [--train_file TRAIN_FILE] [--predict_file PREDICT_FILE] [--train_mindrecord_file TRAIN_MINDRECORD_FILE] [--predict_mindrecord_file PREDICT_MINDRECORD_FILE] [-vocab_path VOCAB_PATH] [--bert_config_path BERT_CONFIG_PATH] [ --freeze False] [--save_steps N] [--weight_decay F] [-warmup_proportion F] [--learning_rate F] [--epoch N] [--max_seq_len N] [--doc_stride N] [--wn_concept_embedding_path WN_CONCEPT_EMBEDDING_PATH] [--nell_concept_embedding_path NELL_CONCEPT_EMBEDDING_PATH] [--use_wordnet USE_WORDNET] [--use_nell True] [--random_seed N]  [--is_modelarts True] [--checkpoints CHECKPOINT]  
  ```

  ```shell
  Options：
      --device_target                 code implementation device, the options are Ascend or CPU. The default is Ascend
      --device_num
      --device_id                     ID of the device on which the task is running (not required for distributed gpu train)
      --batch_size                    the batch size of the input data set
      --do_train                      whether to start training based on the training set, the options are true or false
      --do_predict                    whether to start the evaluation based on the development set, the options are true or false
      --do_lower_case
      --init_pretraining_params       initial checkpoint
      --load_pretrain_checkpoint_path initial checkpoint
      --train_file                    dataset for training
      --predict_file                  dataset for evaluation
      --train_mindrecord_file         mindrecord dataset for training
      --predict_mindrecord_file       mindrecord dataset for evaluation
      --vocab_path                    vocabulary list for BERT model training
      --bert_config_path              bert's parameter path
      --freeze                        default is false
      --save_steps                    number of checkpoints saved
      --warmup_proportion
      --learning_rate
      --epoch                         total number of training rounds
      --max_seq_len
      --doc_stride
      --wn_concept_embedding_path     path to load wordnet embedding
      --nell_concept_embedding_path   path to load nell embedding
      --use_wordnet                   whether to use wordnet, default is true
      --use_nell                      whether to use nell, default is true
      --random_seed
      --save_finetune_checkpoint_path training checkpoint save path
      --is_modelarts                  whether to run tasks on modelarts
      --is_distribute                 whether it is distributed training
      --save_url                      data save path when running on modelarts
      --log_url                       log save path when running on modelarts
      --checkpoints output            file where to save training logs
  ```

- To train model on ReCoRD dataset change `python run_KTNET_squad.py` to `python run_KTNET_record.py`

### Evaluation

- To evaluate model on SQuAD dataset:

```bash
    python run_KTNET_squad_eval.py   [--device_target DEVICE_TARGET] [--device_id N] [batch_size N] [--do_train True] [--do_predict False] [--do_lower_case False][--init_pretraining_params INIT_PRETRAINING_PARAMS] [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH] [--load_checkpoint_path LOAD_CHECKPOINT_PATH][--train_file TRAIN_FILE] [--predict_file PREDICT_FILE] [--train_mindrecord_file TRAIN_MINDRECORD_FILE] [--predict_mindrecord_file PREDICT_MINDRECORD_FILE][-vocab_path VOCAB_PATH] [--bert_config_path BERT_CONFIG_PATH] [ --freeze False] [--save_steps N] [--weight_decay F] [-warmup_proportion F] [--learning_rate F][--epoch N] [--max_seq_len N] [--doc_stride N] [--wn_concept_embedding_path WN_CONCEPT_EMBEDDING_PATH] [--nell_concept_embedding_path NELL_CONCEPT_EMBEDDING_PATH][--use_wordnet USE_WORDNET] [--use_nell True] [--random_seed N]  [--is_modelarts True] [--checkpoints CHECKPOINT]
```

```shell
Options：
    --device_target                 code implementation device, the options are Ascend or GPU. The default is Ascendc
    --device_id                     ID of the device on which the task is running
    --batch_size                    the batch size of the input data set
    --do_train                      whether to start training based on the training set, the options are true or false
    --do_predict                    whether to start the evaluation based on the development set, the options are true or false
    --do_lower_case
    --init_pretraining_params       initial checkpoint
    --load_pretrain_checkpoint_path initial checkpoint
    --load_checkpoint_path          path to saved checkpoints from training
    --train_file                    dataset for training
    --predict_file                  dataset for evaluation
    --train_mindrecord_file         mindrecord dataset for training
    --predict_mindrecord_file       mindrecord dataset for evaluation
    --vocab_path                    vocabulary list for BERT model training
    --bert_config_path              bert parameters path
    --freeze                        default is false
    --save_steps                    number of steps from which checkpoint is saved
    --weight_decay
    --warmup_proportion
    --learning_rate
    --epoch                         total number of training rounds
    --max_seq_len
    --doc_stride
    --wn_concept_embedding_path     path to load wordnet embedding
    --nell_concept_embedding_path   path to load nell embedding
    --use_wordnet                   whether to use wordnet, default is true
    --use_nell                      whether to use nell, default is true
    --random_seed
    --save_finetune_checkpoint_path training checkpoint save path
    --data_url                      path to data
    --checkpoints                   file where to save evaluation logs
```

- To eval model on ReCoRD dataset change `python run_KTNET_squad_eval.py` to `python run_KTNET_record_eval.py`

## Training process

### ascend

#### squad dataset

```bash
# Standalone
bash run_squad_twomemory.sh [DATAPATH]
# Distributed training 8pcs
bash run_squad_twomemory_distribute.sh [DATAPATH] [RANK_TABLE_FILE]
```

DATAPATH is a required option, which is the path where the data file is stored.
Check logs in output/train_squad.log. After training，you can find the checkpoint file in the script folder under the default script path，get the following loss value:

```shell
# train_squad.log
epoch: 0.0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0856101e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.000, step: 2, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0821701e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

```bash
python run_KTNET_squad.py
```

#### record dataset

```bash
# Standalone
bash run_record_twomemory.sh [DATAPATH]
# Distributed 8pcs
bash run_record_twomemory_distribute.sh [DATAPATH] [RANK_TABLE_FILE]
```

DATAPATH is a required option, which is the path where the data file is stored.
The above command runs in the background, you can view the training log in output/train_record.log.

```bash
python run_KTNET_record.py
```

### GPU

#### squad dataset

```bash
# Standalone
bash run_squad_twomemory_gpu.sh [DATAPATH]
# Distributed training 8pcs
bash run_squad_twomemory_distribute_gpu.sh [DATAPATH] [DEVICE_NUM]
```

DATAPATH is a required option, which is the path where the data file is stored.
Check logs in train_squad/train_squad.log or in train_parallel_squad/train_squad.log.
After training，you can find the checkpoint file in the output/finetune_checkpoint/ ，get the following loss value:

```text
# train_squad.log
epoch: 1 step: 1, loss is 5.964628
epoch: 1 step: 1, loss is 6.228141
...
```

#### record dataset

```bash
# Standalone
bash run_record_twomemory_gpu.sh [DATAPATH]
# Distributed 8 devices
bash run_record_twomemory_distribute_gpu.sh [DATAPATH] [DEVICE_NUM]
```

DATAPATH is a required option, which is the path where the data file is stored.
The above command runs in the background, you can view the training log in train/train_record.log or train_parallel/train_record.log.
After training，you can find the checkpoint files in output/finetune_checkpoint/

```shell
# train_record.log
epoch: 1 step: 2, loss is 6.11811
epoch: 1 step: 2, loss is 5.9109883
...
```

## Evaluation Process

### Ascend evaluation

#### squad dataset

Before running the following command, make sure that the loading and training checkpoint path has been set. Please set the checkpoint path to an absolute full path.

```bash
bash run_squad_eval.sh [DATAPATH] [CHECKPOINT_PATH]
```

DATAPATH is a required option, which is the path where the data file is stored.
CHECKPOINT_PATH is a required option, which is the path where the ckpt file is stored.
The above command runs in the background, you can view the training log in eval_squad.log.
The following results can be obtained on Ascend:

```text
"exact_match": 71.00,
"f1": 71.62
```

```bash
python run_KTNET_squad_eval.py
```

#### record dataset

```bash
bash run_record_eval.sh [DATAPATH] [CHECKPOINT_PATH]
```

DATAPATH is a required option, which is the path where the data file is stored.
CHECKPOINT_PATH is a required option, which is the path where the ckpt file is stored.
The above command runs in the background, you can view the training log in eval_squad.log.

```text
"exact_match": 69.00,
"f1(macro-averaged?)": 70.62
```

```bash
python run_KTNET_record_eval.py
```

### GPU evaluation

#### squad dataset

Before running the following command, make sure that the loading and training checkpoint path has been set. Please set the checkpoint path to an absolute full path

```bash
bash run_squad_eval_gpu.sh [DATAPATH] [CHECKPOINT_PATH]
```

DATAPATH is a required option, which is the path where the data file is stored.
CHECKPOINT_PATH is a required option, which is the path where the ckpt file is stored.
The above command runs in the background, you can view the training log in eval_squad.log.
The following results can be obtained on GPU:

```text
"exact_match": 84.24,
"f1": 91.06
```

#### record dataset

```bash
bash run_record_eval_gpu.sh [DATAPATH] [CHECKPOINT_PATH]
```

DATAPATH is a required option, which is the path where the data file is stored.
CHECKPOINT_PATH is a required option, which is the path where the ckpt file is stored.
The above command runs in the background, you can view the training log in eval_squad.log.

```text
"exact_match": 68.95,
"f1": 70.86
```

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Usage

Before performing inference, you need to export the mindir file through export.sh

```bash
bash script/export.sh [RECORD_CKPT] [SQUAD_CKPT]
```

After running successfully, you can get the mindir file with training results on two datasets (ReCoRD and SQuAD) stored in the mindir folder.
The input data file is in bin format.

```bash
# Ascend310推理
bash script/run_infer_310.sh [MINDIR_PATH] [DATA_FILE_PATH] [NEED_PREPROCESS] [DATASET] [DATA_URL] [DEVICE_ID]
```

NNEED_PREPROCESS is a required option, and the value in [y|n] indicates whether the data is preprocessed into bin format.
DATASET is a required option, and the value in [record|squad] indicates the dataset selection for inference.
DATA_URL is a required option, indicating the path where the data is stored.

### Results

After running successfully, you can check the final accuracy result in acc.log.

```shell
"exact_match": 69.00,
"f1(macro-averaged?)": 70.62
```

## Performance

### Training performance

- ReCoRD dataset

| Parameters          | Ascend                                                           | GPU (1pcs)                  | GPU (8pcs)                  |
| --------------------| ---------------------------------------------------------------- |---------------------------- | --------------------------- |
| Model Version       | KTNET                                                            | KTNET                       | KTNET                       |
| Resource            | Ascend 910；CPU 2.60GHz，192cores；Memory 755GB；System Euler2.8  | GPU(Tesla V100-PCIE 32G)；CPU：2.60GHz 52cores ；RAM：754G; Mindspore 1.6.0.20211129 | GPU(Tesla V100-PCIE 32G)；CPU：2.60GHz 52cores ；RAM：754G; Mindspore 1.6.0.20211129 |
| Uploaded Date       | 2021-05-12                                                       | 2021-10-29                  | 2021-10-29                  |
| Dataset             | ReCoRD                                                           | ReCoRD                      | ReCoRD                      |
| Training Parameters | epochs=4, batch_size=12*8, lr=7e-5                               | epochs=4, batch_size=12, lr=6e-5   | epochs=4, batch_size=12*8, lr=6e-5   |
| Optimizer           | Adam                                                             | Adam                        | Adam                        |
| Loss function       | SoftmaxCrossEntropy                                              | SoftmaxCrossEntropy         | SoftmaxCrossEntropy         |
| Loss                | 0.31248128                                                       | 0.2                         | 0.11                        |
| Speed               | 428ms/step                                                       | 668.1 ms/step               | 960 ms/step                 |
| Total time          | 2.5h                                                             | 6h 14min                    | 1h 7min                     |

- SQuaD dataset

| Parameters          | Ascend                                                          | GPU (1pcs)                   | GPU (8pcs)                   |
| --------------------| --------------------------------------------------------------- | ---------------------------- | ---------------------------- |
| Model Version       | KTNET                                                           | KTNET | KTNET |
| Resource            | Ascend 910；CPU 2.60GHz，192cores；RAM 755GB；System Euler2.8    | GPU(Tesla V100-PCIE 32G)；CPU：2.60GHz 52cores ；RAM：754G; Mindspore 1.6.0.20211129 | GPU(Tesla V100-PCIE 32G)；CPU：2.60GHz 52cores ；RAM：754G ; Mindspore 1.6.0.20211129 |
| Uploaded Date       | 2021-05-12                                                      | 2021-10-29 | 2021-10-29 |
| Dataset             | SQuAD                                                           | SQuAD | SQuAD |
| Training Parameters | epochs=3, batch_size=8*8, lr=4e-5                               | epochs=3, batch_size=8, lr=4e-5 | epochs=3, batch_size=8*8, lr=4e-5 |
| Optimizer           | Adam                                                            | Adam                | Adam                |
| Loss function       | SoftmaxCrossEntropy                                             | SoftmaxCrossEntropy | SoftmaxCrossEntropy |
| Loss                | 0.35267675                                                      | 0.3                 | 0.336353            |
| Speed               | 338ms/step                                                      | 474.5 ms/step       | 760 ms/step         |
| Total time          | 1h                                                              | 4h 20min            | 52min               |

### Evaluation performance

| Parameters           | Ascend         |  GPU          | Ascend        | GPU          |
| ---------------------| ---------------| ------------- | --------------| ------------ |
| Model Version        | KTNET          | KTNET         | KTNET         | KTNET        |
| Dataset              | ReCoRD         | ReCoRD        | SQuAd         | SQuAd        |
| Uploaded Date        | 2021-05-12     | 2021-10-29    | 2021-05-12    | 2021-10-29   |
| F1(macro-averaged)   | 70.62          | 70.58         | 71.62         | 91.42        |
| exact_match          | 69.00          | 68.63         | 71.00         | 84.67        |
| Total time           | 15min          | 15min         | 15min         | 15min        |

### Inference performance

| Parameters           | Ascend         | Ascend        |
| ---------------------| ---------------| --------------|
| Model Version        | KTNET          | KTNET         |
| Dataset              | ReCoRD         | SQuAd        |
| Uploaded Date        | 2021-05-12     | 2021-05-12    |
| F1(macro-averaged)   | 71.48          | 91.31         |
| exact_match          | 69.61          | 84.38         |
| Total time           | 15min          | 15min         |

# ModelZoo homepage

Please check the official [homepage](https://gitee.com/mindspore/models).