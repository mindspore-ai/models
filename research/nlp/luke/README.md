# Contents

- [Contents](#contents)
- [Luke overview](#luke-overview)
- [Model architecture](#model-architecture)
- [Pretrained model](#pretrained-model)
    - [Pretrained model preprocessing](#pretrained-model-preprocessing)
    - [Pretrained model structure](#pretrained-model-structure)
- [Dataset](#dataset)
    - [Dataset downloading](#dataset-downloading)
    - [Data preprocessing](#data-preprocessing)
    - [Dataset structure](#dataset-structure)
- [Environmental requirement](#environmental-requirement)
- [Quick start](#quick-start)
- [Script description](#script-description)
    - [Scripts and sample code](#scripts-and-sample-code)
    - [Script parameters](#script-parameters)
- [Training process](#training-process)
- [Evaluate](#evaluate)
- [Export process](#export-process)
- [Performance](#performance)
- [ModelZoo homepage](#modelzoo-homepage)

# Luke overview

LUKE (Language Understanding with Knowledge-based Embeddings) is a new pre-trained transformer-based contextual
representation of words and entities. It achieves state-of-the-art results on important NLP benchmarks,
including SQuAD v1.1 (extractive question answering), CoNLL-2003 (named entity recognition), ReCoRD (cloze-style
question answering), TACRED (relation classification), and Open Entity (entity type).

Paper: <https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/2020.emnlp-main.523.pdf>

Source code: <https://link.zhihu.com/?target=https%3A//github.com/studio-ousia/luke>

# Model architecture

1. The author proposes a new contextual representation method LUKE (Language Understanding with Knowledge-based
   Embeddings) specially designed to deal with entity-related tasks. LUKE utilizes a large corpus of entity annotations
   obtained from Wikipedia to predict random masked words and entities.

2. The author proposes an entity-aware
   self-attention mechanism, which is an effective extension to the original attention mechanism of the transformer,
   which takes into account the type of token (word or entity) when calculating the attention score.

# Pretrained model

- Download luke-large model: [(download link)](https://drive.google.com/file/d/1S7smSBELcZWV7-slfrb94BKcSCCoxGfL/view?usp=sharing)

- Extract it into `pre_luke/luke_large` directory

- Modify the `convert_luke.py` file: set `model_type = 'luke-large'`

## Pretrained model preprocessing

Convert pretrained model to the `.ckpt` format.

- For SQUAD training

    ```bash
    python convert_luke.py
    ```

- For TACRED training:

    ```bash
    python convert_luke_tacred.py
    ```

## Pretrained model structure

### TACRED luke-large model structure after downloading

```text
       │──luke_large_500k
          │──metadata.json
          │──entity_vocab.tsv
          │──pytorch_model.bin
```

### TACRED luke-large model structure after preprocessing

```text
       │──luke_large_500k
          │──metadata.json
          │──entity_vocab.tsv
          │──pytorch_model.bin
          │──luke.ckpt
```

# Dataset

## Dataset downloading

- SQUAD dataset

    - Download the `squad1.1.zip` file [(download link)](https://data.deepai.org/squad1.1.zip)

    - Extract it into the `squad_data/squad` directory

- TACRED dataset

    - Download the TACRED dataset [(webpage link)](https://nlp.stanford.edu/projects/tacred/)

    - Extract it into the `/TACRED_LDC2018T24/` directory

- Wikipedia dataset

    - Download the `luke_squad_wikipedia_data.tar.gz` file
    [(download link)](https://drive.google.com/file/d/129tDJ3ev6IdbJiKOmO6GTgNANunhO_vt/view?usp=sharing)

    - Extract it into `/enwiki_dataset/` directory

The overall directory structure of the data files and pretraining files is shown below.

## Dataset preprocessing

- Data processing SQUAD

    ```bash
    bash scripts/run_squad_data_process.sh ./squad_data ./pre_luke/luke_large/ ./enwiki_dataset
    # or
    python create_squad_data.py --data ./squad_data --model_file ./pre_luke/luke_large/ --wikipedia ./enwiki_dataset
    ```

- Data processing TACRED

    ```bash
    python create_tacred_data.py --data /path/to/tacred_LDC2018T24/tacred/data/json/ --model_file /path/to/luke_large_500k/
    ```

## Dataset structure

### SQUAD

```text
     │──squad_data
        │──squad                      # squadv1.1
           │──train-v1.1.json
           │──dev-v1.1.json
        │──squad_change               # processed dataset
        │──mindrecord                 # processed mindrecord
     │──pre_luke
        │──luke_large                 # pretrained luke model
           │──metadata.json
           │──entity_vocab.tsv
           │──luke.ckpt
     │──enwiki_dataset                # enwiki dataset
           │──enwiki_20160305_redirects.pkl
           │──enwiki_20160305.pkl
           │──enwiki_20181220_redirects.pkl
 ```

### TACRED after downloading

```text
   └─TACRED_LDC2018T24
      └──tacred
         │──docs
         │──tools
         └──data
            │──conll
            │──gold
            └─json
              │──dev.json
              │──test.json
              │──train.json
 ```

### TACRED after preprocessing

```text
    └───TACRED_LDC2018T24
       └───tacred
          │──docs
          │──tools
          └──data
            │──conll
            │──gold
            └──json
              │──mindrecord
              │──tacred_change
              │──dev.json
              │──test.json
              │──train.json
```

# Environmental requirements

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- Additional Python packages
    - ujson
    - marisa-trie
    - wikipedia2vec==1.0.5
    - transformers==2.3.0
    - torch==1.8.1

    Install additional packages manually or using `pip install -r requirements.txt` command in the model directory.

# Quick start

It is recommended to use an absolute path to run the scripts.

- Runs SQUAD train on an Ascend processor

  ```bash
  # processing the pretrained model
  python convert_luke.py
  # processing the SQUAD dataset
  bash scripts/run_squad_data_process.sh [DATA] [MODEL_FILE] [WIKIPEDIA]
  # run the training example
  bash scripts/run_squad_standalone_train.sh [DATA] [MODEL_FILE]
  # running the distributed training example
  bash scripts/run_squad_distribute_train.sh [RANK_TABLE] [DATA] [MODEL_FILE]
  # run the evaluation example
  bash scripts/run_squad_eval.sh [DATA] [MODEL_FILE] [CHECKPOINT_FILE]
  ```

- Runs SQUAD train on an GPU

  ```bash
  # processing the pretrained model
  python convert_luke.py
  # processing the SQUAD dataset
  bash scripts/run_squad_data_process.sh [DATA] [MODEL_FILE] [WIKIPEDIA]
  # run the training example
  bash scripts/run_squad_standalone_train.sh [DATA] [MODEL_FILE]
  # running the distributed training example
  bash scripts/run_squad_distribute_train_gpu.sh [DATA] [MODEL_FILE] [TRAIN_BATCH_SIZE]
  # run the evaluation example
  bash scripts/run_squad_eval.sh [DATA] [MODEL_FILE] [CHECKPOINT_FILE]
  ```

  ```text
  Options:
  [DATA]              dataset address(/home/ma-user/work/squad_data)
  [MODEL_FILE]        pre-trained model address(/home/ma-user/work/pre_luke/luke_large)
  [WIKIPEDIA]         wikipedia data address(/home/ma-user/work/enwiki_dataset)
  [CHECKPOINT_FILE]   Trained model address(/home/ma-user/work/output/luke_squad-2_10973.ckpt)
  [RANK_TABLE]        hccl configuration file
  ```

- Runs TACRED on an GPU

  ```bash
  # processing the pretrained model
  python convert_luke_tacred.py
  # processing the TACRED dataset
  python create_tacred_data.py --data [DATA] --model_file [MODEL_FILE]
  # run the training example
  bash scripts/run_tacred_standalone_train.sh [DATA] [MODEL_FILE] [TRAIN_BATCH_SIZE]
  # running the distributed training example
  bash scripts/run_tacred_distribute_train.sh [DATA] [MODEL_FILE] [TRAIN_BATCH_SIZE]
  # run the evaluation example
  bash scripts/run_tacred_eval.sh [CHECKPOINT_FILE] [MODEL_FILE] [EVAL_BATCH_SIZE]
  ```

  ```text
  Options:
  [DATA]              dataset address (/home/ma-user/work/tacred_LDC2018T24/tacred/data/json/)
  [MODEL_FILE]        pre-trained model address(/home/ma-user/work/pre_luke/luke_large)
  [CHECKPOINT_FILE]   Trained model address (folder or file: /home/ma-user/work/output/luke_tacred-2_4000.ckpt)
  [TRAIN_BATCH_SIZE]  Training batch size
  [EVAL_BATCH_SIZE]   Evaluation batch size
  ```

For distributed training, you need to create an hccl configuration file in JSON format in advance.
Follow the instructions at the following link:
<https://gitee.com/mindspore/models/tree/r1.5/utils/hccl_tools>

- Training with 8 cards on ModelArt

  ```text
  # (1) Upload your code and the converted luke model to the s3 bucket
  # (2) Create a training task on ModelArts
  # (3) Select code directory /{path}/luke
  # (4) Select startup file /{path}/luke/run_squad_train.py
  # (5) do a or b:
  #     a. Set parameters in pretrain_main file
  #         1. set up ”modelArts=True“
  #         3. set up “checkpoint_url=<location of luke pretrained model>”
  #         2. Set other parameters, other parameter configuration can refer to `src/model_utils/config.yaml`
  #     b. Set up on the webpage
  #         1. add to ”modelArts = True“
  #         2. add to ”checkpoint_url = obs://jeffery/luke_pre/luke_large/“
  #         3. Add other parameters, other parameter configuration can refer to `src/model_utils/config.yaml`
  # (6) Upload your data to the s3 bucket
  # (7) Check the data storage location on the web page and set the "training data set" path
  # (8) Set the "training output file path" and "job log path" on the web page
  # (9) Under the 'Resource Pool Selection' item on the web page, select the resource with the 8-card specification
  # (10) Create a training job
  # After training, the training weights will be saved under 'training output file path'
  ```

# Script description

## Scripts and sample code

```text
|--luke
    |--scripts
    |      |--run_squad_data_process.sh         # process the Squad dataset script
    |      |--run_squad_eval.sh                 # squad evaluation script
    |      |--run_squad_standalone_train.sh     # squad training script
    |      |--run_squad_distribute_train.sh     # distributed Squad training script
    |      |--run_squad_distribute_train_gpu.sh  # distributed Squad training GPU script
    |      |--run_tacred_distribute_train_gpu.sh # distributed GPU Tacred training script
    |      |--run_tacred_eval.sh                 # tacred evaluation script
    |--src
    |      |--luke                              # luke model related files
    |      |      |--config.py                  # configuration file
    |      |      |--model.py                   # luke model file
    |      |      |--roberta.py                 # roberta file
    |      |      |--tacred_model.py            # luke tacred model file
    |      |      |--tacred_robert.py           # luke tacred roberta
    |      |--model_utils
    |      |      |--local_adapter.py           # local adaptation
    |      |      |--moxing_adapter.py          # mox upload
    |      |      |--config.yaml                # configuration file
    |      |      |--config_args.py             # config read file
    |      |--reading_comprehension             # reading comprehension task
    |      |      |--dataLoader.py              # data load file
    |      |      |--dataProcessing.py          # data processing file
    |      |      |--dataset.py                 # dataset file
    |      |      |--model.py                   # model file
    |      |      |--squad_get_predictions.py   # squad get prediction file
    |      |      |--squad_postprocess.py       # squad processing files
    |      |      |--train.py                   # squad training file
    |      |      |--wiki_link_db.py            # wiki link database file
    |      |--relation_classification           # relation classification task
    |      |      |--main.py                    # main script file
    |      |      |--model.py                   # model file
    |      |      |--preprocess_data.py         # data processing file
    |      |      |--train.py                   # tacred training file
    |      |      |--utils.py                   # toolkit
    |      |      |--requirements.py            # requirements file
    |      |--utils
    |      |      |--entity_vocab.py            # entity vocab file
    |      |      |--interwiki_db.py            # wiki db file
    |      |      |--model_utils.py             # model toolkit
    |      |      |--sentence_tokenizer.py      # sentence tokenizer
    |      |      |--utils.py                   # toolkit
    |      |      |--word_tokenizer.py          # word segmentation
    |--convert_luke.py                          # convert luke model
    |--convert_luke_tacred.py                   # convert luke tacred model
    |--create_squad_data.py                     # create the Squad dataset
    |--create_tacred_data.py                    # create the Tacred dataset
    |--export.py                                # export script
    |--run_tacred_export.py                     # tacred export script
    |--README_CN.md                             # readme in Chinese
    |--README.md                                # readme in English
    |--requirements.txt                         # requirements file
    |--run_squad_eval.py                        # eval squad
    |--run_squad_train.py                       # run squad
    |--run_tacred_train.py                      # run tacred
    |--run_tacred_eval.py                       # eval tacred
```

## Script parameters

Parameters can be modified in run_squad_train.py.

```text
train_batch_size
num_train_epochs
decay_steps
learning_rate
warmup_proportion
weight_decay
eps
seed
data
with_negative (Is the dataset square1.1 or square2.0)

For other parameters, see src/model_utils/config.yaml
```

# Training process

- Run SQUAD on the Ascend processor
    - Standalone training

        ```bash
        bash scripts/run_squad_standalone_train.sh [DATA] [MODEL_FILE]
        # or
        python run_squad_train.py --warmup_proportion 0.09 --num_train_epochs 2 --model_file ./pre_luke/luke_large --data ./squad_data --train_batch_size 8 --learning_rate 12e-6 --dataset_sink_mode False
        ```

    - Distribute training

        ```bash
        bash scripts/run_squad_distribute_train.sh [RANK_TABLE] [DATA] [MODEL_FILE]
        ```

- Run SQUAD on the GPU
    - Standalone training

        ```bash
        bash scripts/run_squad_standalone_train.sh [DATA] [MODEL_FILE]
        # or
        python run_squad_train.py --warmup_proportion 0.09 --num_train_epochs 2 --model_file ./pre_luke/luke_large --data ./squad_data --train_batch_size 8 --learning_rate 12e-6 --dataset_sink_mode False
        ```

    - Distribute training

        ```bash
        bash scripts/run_squad_distribute_train_gpu.sh [DATA] [MODEL_FILE] [TRAIN_BATCH_SIZE]
        ```

- Run TACRED on the GPU
    - Standalone training

        ```bash
        bash scripts/run_tacred_standalone_train.sh [DATA] [MODEL_FILE] [TRAIN_BATCH_SIZE]
        # or
        python run_tacred_train.py --data /path/to/tacred_LDC2018T24/tacred/data/json/ --model_file /path/to/luke_large_500k/ --train_batch_size 2 --num_train_epochs 5 --learning_rate 0.00001 --distribute=False
        ```

    - Distribute training

        ```bash
        bash scripts/run_tacred_distribute_train.sh [DATA] [MODEL_FILE] [TRAIN_BATCH_SIZE]
        ```

```text
Options:

--model_file              pre-trained model address
--data                    dataset address
--warmup_proportion       warm-up learning ratio
--num_train_epochs        epoch number
--train_batch_size        bach size
--dataset_sink_mode       data sink mode
--distribute              flag for distributed GPU training

Script parameters:

[DATA]                    dataset address (`/home/ma-user/work/tacred_LDC2018T24/tacred/data/json/`)
[MODEL_FILE]              pre-trained model address (`/home/ma-user/work/pre_luke/luke_large`)
[RANK_TABLE]              hccl configuration file
[TRAIN_BATCH_SIZE]        training batch size (`8`)
```

- Modelarts

    - You can view the quickstart guide

# Evaluation

Put the trained ckpt into output

- SQUAD startup script

    ```bash
    bash scripts/run_squad_eval.sh [DATA] [MODEL_FILE] [CHECKPOINT_FILE]
    # or
    python run_squad_eval.py --data ./squad_data --eval_batch_size 8  --checkpoint_file ./output/luke_squad-2_10973.ckpt --model_file ./pre_luke/luke_large/
    ```

    Where checkpoint_file refers to the location where the model was trained.

    Result:

    ```text
    single card: {"exact_match": 89.50804162724693, "f1": 95.01586892223338}
    doka: {"exact_match": 89.33774834437087, "f1": 94.89125713127684}
    ```

- TACRED startup script

    ```bash
    bash scripts/run_tacred_eval.sh [DATA] [CHECKPOINT_FILE] [MODEL_FILE] [EVAL_BATCH_SIZE]
    # or
    python run_tacred_eval.py --data /path/to/tacred_LDC2018T24/tacred/data/json --eval_batch_size 32  --checkpoint_file ./output/luke_tacred-2_4000.ckpt --model_file /path/to/luke_large_500k/
    ```

    Result:

    ```text
    distributed {'precision': 0.7031615925058547, 'recall': 0.7239300783604581, 'f1': 0.7133947133947133}
    ```

```text
Options:

--data                    dataset address
--eval_batch_size         evaluation batch size
--checkpoint_file         trained checkpoint path
--model_file              pre-trained model address

Script parameters:

[DATA]                    dataset address (`/home/ma-user/work/tacred_LDC2018T24/tacred/data/json/`)
[CHECKPOINT_FILE]         trained model address (folder or file: /home/ma-user/work/output/luke_tacred-2_4000.ckpt)
[MODEL_FILE]              pre-trained model address (`/home/ma-user/work/pre_luke/luke_large`)
[EVAL_BATCH_SIZE]         evaluation batch size (`32`)
```

# Export process

## Usage

- Ascend processor environment running

  ```bash
  python export.py --export_batch_size 8 --model_file ./pre_luke/luke_large/ --checkpoint_file ./output/luke_squad-2_10973.ckpt
  ```

- GPU environment running (for TACRED dataset)

  ```bash
  python run_tacred_export.py --export_batch_size 1 --model_file /path/to/luke_large_500k/ --checkpoint_file ./output/luke_tacred-5_4258.ckpt --file_name luke_tacred
  ```

# Performance

## Training performance

| device              | Ascend                                                            | GPU                                                         |
|---------------------|-------------------------------------------------------------------|-------------------------------------------------------------|
| model               | Luke                                                              | Luke( relation classification)                              |
| resource            | Ascend 910; CPU 2.60GHz，192 cores; memory 755GB; system Euler2.8  | GeForce RTX 3090; Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz |
| upload date         | 2022-01-28                                                        | 2022-03-14                                                  |
| dataset             | SQuAD                                                             | TACRED                                                      |
| training_parameters | src/model_utils/config.yaml                                       | src/model_utils/config.yaml                                 |
| learning rate       | 12e-6                                                             | 10e-5                                                       |
| optimizer           | AdamWeightDecay                                                   | AdamWeightDecay                                             |
| loss function       | SoftmaxCrossEntropy                                               | SoftmaxCrossEntropy                                         |
| epochs              | 2                                                                 | 5                                                           |
| batch size          | 2*8                                                               | 2*8                                                         |
| loss                | 0.8173281                                                         | 0.189                                                       |
| speed               | 359 ms/step                                                       | 1813 ms/step                                                |
| total duration      | 1.5 hours                                                         | 10.7 hours                                                  |

## Inference Performance

| Parameters  | GPU                                                         |
|-------------|-------------------------------------------------------------|
| task        | Relation classification                                     |
| resource    | GeForce RTX 3090; Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz |
| upload date | 2022-03-14                                                  |
| dataset     | TACRED                                                      |
| batch size  | 32                                                          |
| time        | 16 min                                                      |
| metrics     | precision - 70.3%, recall - 71.3%,  F1 -71.34%              |

# ModelZoo homepage

Please visit the official website [homepage](https://gitee.com/mindspore/models).
