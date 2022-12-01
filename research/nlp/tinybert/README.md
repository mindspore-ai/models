﻿# Contents

- [Contents](#contents)
- [TinyBERT Description](#tinybert-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [General Distill](#general-distill)
        - [Task Distill](#task-distill)
    - [Options and Parameters](#options-and-parameters)
        - [Options](#options)
        - [Parameters](#parameters)
    - [Training Process](#training-process)
        - [Training](#training)
            - [running on Ascend](#running-on-ascend)
            - [running on GPU](#running-on-gpu)
        - [Distributed Training](#distributed-training)
            - [running on Ascend](#running-on-ascend-1)
            - [running on GPU](#running-on-gpu-1)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
            - [evaluation on SST-2 dataset](#evaluation-on-sst-2-dataset)
            - [evaluation on MNLI dataset](#evaluation-on-mnli-dataset)
            - [evaluation on QNLI dataset](#evaluation-on-qnli-dataset)
            - [evaluation on TNEWS dataset](#evaluation-on-tnews-dataset)
            - [evaluation on CLUENER dataset](#evaluation-on-cluener-dataset)
    - [ONNX Export And Evaluation](#onnx-export-and-evaluation)
        - [ONNX Export](#onnx-export)
        - [ONNX Evaluation](#onnx-evaluation)
            - [ONNX evaluation on SST-2 dataset](#onnx-evaluation-on-sst-2-dataset)
            - [ONNX evaluation on MNLI dataset](#onnx-evaluation-on-mnli-dataset)
            - [ONNX evaluation on QNLI dataset](#onnx-evaluation-on-qnli-dataset)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
    - [Model Description](#model-description)
    - [Performance](#performance)
        - [training Performance](#training-performance)
            - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [TinyBERT Description](#contents)

[TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) is 7.5x smalller and 9.4x faster on inference than [BERT-base](https://github.com/google-research/bert) (the base version of BERT model) and achieves competitive performances in the tasks of natural language understanding. It performs a novel transformer distillation at both the pre-training and task-specific learning stages.

[Paper](https://arxiv.org/abs/1909.10351):  Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351). arXiv preprint arXiv:1909.10351.

# [Model Architecture](#contents)

The backbone structure of TinyBERT is transformer, the transformer contains four encoder modules, one encoder contains one selfattention module and one selfattention module contains one attention module.  

# [Dataset](#contents)

- Create dataset for general distill phase
    - Download the [enwiki](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2) dataset for pre-training.
    - Extract and refine texts in the dataset with [WikiExtractor](https://github.com/attardi/wikiextractor). The commands are as follows:
        - pip install wikiextractor
        - python -m wikiextractor.WikiExtractor [Wikipedia dump file] -o [output file path] -b 2G
    - Download [BERT](https://github.com/google-research/bert), and download [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip), it contains `vocab.txt`, `bert_config.json`, and pretrain model.
    - Use `create_pretraining_data.py`, transform data to tfrecord, please refer to readme file, if AttributeError: module 'tokenization' has no attribute 'FullTokenizer' occur, please install bert-tensorflow.
    - Transform tensorflow model to mindspore model

        ```bash
        cd scripts/ms2tf
        python ms_and_tf_checkpoint_transfer_tools.py --tf_ckpt_path=PATH/model.ckpt　\
            --new_ckpt_path=PATH/ms_model_ckpt.ckpt　\
            --tarnsfer_option=tf2ms
        # Attention，tensorflow model include 3 parts，data, index and meta，the input of tf_ckpt_path is *.ckpt
        ```

- Create dataset for task distill phase
    - Download [GLUE](https://github.com/nyu-mll/GLUE-baselines) dataset for task distill phase, use `download_glue_data.py` to download sst2, mnli, qnli dataset, Chinese Named Entity Recognition[CLUENER](https://github.com/CLUEbenchmark/CLUENER2020), Chinese sentences classification[TNEWS](https://github.com/CLUEbenchmark/CLUE)
    - Transform dataset to TFRecord format. Use `run_classifier.py` in [BERT](https://github.com/google-research/bert), referring to readme. Besides, transforming sst2 dataset, you should add code in [PR:327](https://github.com/google-research/bert/pull/327); transforming qnli dataset, you should refer sst2 add add follow code. Parts of code, such as training, evaling, predicting, are useless, you can comment them and only left necessary code. task_name should be SST2, ber_config_files should be `bert_config.json`, max_seq_length should be 64. For ClUENER and TNEWS dataset, please refer to official/nlp/Bert.

    ```python
    ...
    class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")),
                               "dev_matched")

        def get_labels(self):
            """See base class."""
            return ["entailment", "not_entailment"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, line[0])
                text_a = line[1]
                text_b = line[2]
                label = line[-1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples
    ...
    "qnli": QnliProcessor,
    ...
    ```

    - download task distill phase pretrain models for following datasets:
        [sst2 pretrain model](https://download.mindspore.cn/models/r1.9/tinybert_ascend_v190_enwiki128_sst2_official_nlp_acc90.28.ckpt)
        [mnli pretrain model](https://download.mindspore.cn/models/r1.9/tinybert_ascend_v190_enwiki128_mnli_official_nlp_acc81.31.ckpt)
        [qnli pretrain model](https://download.mindspore.cn/models/r1.9/tinybert_ascend_v190_enwiki128_qnli_official_nlp_acc88.86.ckpt)

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
    - scipy>=1.7
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

- running on local

  After installing MindSpore via the official website, you can start general distill, task distill and evaluation as follows:

    ```text
    # run standalone general distill example
    bash scripts/run_standalone_gd.sh

    Before running the shell script, please set the `load_teacher_ckpt_path`, `data_dir`, `schema_dir` and `dataset_type` in the run_standalone_gd.sh file first. If running on GPU, please set the `device_target=GPU`.

    # For Ascend device, run distributed general distill example
    bash scripts/run_distributed_gd_ascend.sh 8 1 /path/hccl.json

    Before running the shell script, please set the `load_teacher_ckpt_path`, `data_dir`, `schema_dir` and `dataset_type` in the run_distributed_gd_ascend.sh file first.

    # For GPU device, run distributed general distill example
    bash scripts/run_distributed_gd_gpu.sh 8 1 /path/data/ /path/schema.json /path/teacher.ckpt

    # run task distill and evaluation example
    bash scripts/run_standalone_td.sh {path}/*.yaml

    Before running the shell script, please set the `task_name`, `load_teacher_ckpt_path`, `load_gd_ckpt_path`, `train_data_dir`, `eval_data_dir`, `schema_dir` and `dataset_type` in the run_standalone_td.sh file first.
    If running on GPU, please set the `device_target=GPU`.
    ```

    For distributed training on Ascend, a hccl configuration file with JSON format needs to be created in advance.
    Please follow the instructions in the link below:
    https:gitee.com/mindspore/models/tree/master/utils/hccl_tools.

    For dataset, if you want to set the format and parameters, a schema configuration file with JSON format needs to be created, please refer to [tfrecord](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.TFRecordDataset.html) format.

    ```text
    For general task, schema file contains ["input_ids", "input_mask", "segment_ids"].

    For task distill and eval phase, schema file contains ["input_ids", "input_mask", "segment_ids", "label_ids"].

    `numRows` is the only option which could be set by user, the others value must be set according to the dataset.

    For example, the dataset is cn-wiki-128, the schema file for general distill phase as following:
    {
        "datasetType": "TF",
        "numRows": 7680,
        "columns": {
            "input_ids": {
                "type": "int64",
                "rank": 1,
                "shape": [256]
            },
            "input_mask": {
                "type": "int64",
                "rank": 1,
                "shape": [256]
            },
            "segment_ids": {
                "type": "int64",
                "rank": 1,
                "shape": [256]
            }
        }
    }
    ```

- running on ModelArts

  If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows

    - general_distill with 8 cards on ModelArts

    ```python
    # (1) Upload the code folder to S3 bucket.
    # (2) Click to "create training task" on the website UI interface.
    # (3) Set the code directory to "/{path}/tinybert" on the website UI interface.
    # (4) Set the startup file to /{path}/tinybert/run_general_distill.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/tinybert/gd_config.yaml.
    #         1. Set ”enable_modelarts=True“
    #         2. Set other parameters('config_path' cannot be set here), other parameter configuration can refer to `./scripts/run_distributed_gd_ascend.sh`
    #     b. adding on the website UI interface.
    #         1. Add ”enable_modelarts=True“
    #         3. Add other parameters, other parameter configuration can refer to `./scripts/run_distributed_gd_ascend.sh`
    #     Note that 'data_dir' and 'schema_dir' fill in the relative path relative to the path selected in step 7.
    #     Add "config_path=../../gd_config.yaml" on the webpage ('config_path' is the path of the'*.yaml' file relative to {path}/tinybert/src/model_utils/config.py, and'* .yaml' file must be in {path}/bert/)
    # (6) Upload the dataset to S3 bucket.
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (there is only data or zip package under this path).
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of 8 cards.
    # (10) Create your job.
    # After training, the '*.ckpt' file will be saved under the'training output file path'
    ```

    - Running task_distill with single card on ModelArts

    ```python
    # (1) Upload the code folder to S3 bucket.
    # (2)  Click to "create training task" on the website UI interface.
    # (3) Set the code directory to "/{path}/tinybert" on the website UI interface.
    # (4) Set the startup file to /{path}/tinybert/run_ner.py"(or run_pretrain.py or run_squad.py) on the website UI interface.
    # (5) Perform a or b.
    #     Add "config_path=../../td_config/td_config_sst2.yaml" on the web page (select the *.yaml configuration file according to the distill task)
    #     a. setting parameters in task_ner_config.yaml(or task_squad_config.yaml or task_classifier_config.yaml under the folder `/{path}/bert/`
    #         1. Set ”enable_modelarts=True“
    #         2. Set "task_name=SST-2" (depending on the task, select from ["SST-2", "QNLI", "MNLI", "TNEWS", "CLUENER"])
    #         3. Set other parameters, other parameter configuration can refer to './scripts/run_standalone_td.sh'.
    #     b. adding on the website UI interface.
    #         1. Add ”enable_modelarts=True“
    #         2. Add "task_name=SST-2" (depending on the task, select from ["SST-2", "QNLI", "MNLI", "TNEWS", "CLUENER"])
    #         3. Add other parameters, other parameter configuration can refer to './scripts/run_standalone_td.sh'.
    #     Note that 'load_teacher_ckpt_path', 'train_data_dir', 'eval_data_dir' and 'schema_dir' fill in the relative path relative to the path selected in step 7.
    #     Note that 'load_gd_ckpt_path' fills in the relative path relative to the path selected in step 3.
    # (6) Upload the dataset to S3 bucket.
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of a single card.
    # (10) Create your job.
    # After training, the '*.ckpt' file will be saved under the'training output file path'.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─bert
  ├─README.md
  ├─README_CN.md
  ├─scripts
    ├─run_distributed_gd_ascend.sh       # shell script for distributed general distill phase on Ascend
    ├─run_distributed_gd_gpu.sh          # shell script for distributed general distill phase on GPU
    ├─run_infer_310.sh                   # shell script for 310 infer
    ├─run_standalone_gd.sh               # shell script for standalone general distill phase
    ├─run_standalone_td.sh               # shell script for standalone task distill phase
    ├─run_eval_onnx.sh                   # shell script for exported onnx model eval
  ├─src
    ├─model_utils
      ├── config.py                      # parse *.yaml parameter configuration file
      ├── devcie_adapter.py              # distinguish local/ModelArts training
      ├── local_adapter.py               # get related environment variables in local training
      └── moxing_adapter.py              # get related environment variables in ModelArts training
    ├─__init__.py
    ├─assessment_method.py               # assessment method for evaluation
    ├─dataset.py                         # data processing
    ├─tinybert_for_gd_td.py              # backbone code of network
    ├─tinybert_model.py                  # backbone code of network
    ├─utils.py                           # util function
  ├─td_config                            # folder where *.yaml files of different distillation tasks are located
    ├── td_config_15cls.yaml
    ├── td_config_mnli.py
    ├── td_config_ner.py
    ├── td_config_qnli.py
    └── td_config_stt2.py
  ├─__init__.py
  ├─export.py                            # export scripts
  ├─gd_config.yaml                       # parameter configuration for general_distill
  ├─mindspore_hub_conf.py                # Mindspore Hub interface
  ├─postprocess.py                       # scripts for 310 postprocess
  ├─preprocess.py                        # scripts for 310 preprocess
  ├─run_general_distill.py               # train net for general distillation
  ├─run_eval_onnx.py                     # eval exported onnx model
  └─run_task_distill.py                  # train and eval net for task distillation
```

## [Script Parameters](#contents)

### General Distill

```text
usage: run_general_distill.py   [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                                [--device_target DEVICE_TARGET] [--do_shuffle DO_SHUFFLE]
                                [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                                [--save_ckpt_path SAVE_CKPT_PATH]
                                [--load_teacher_ckpt_path LOAD_TEACHER_CKPT_PATH]
                                [--save_checkpoint_step N] [--max_ckpt_num N]
                                [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR] [--dataset_type DATASET_TYPE] [train_steps N]

options:
    --device_target            device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --distribute               pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --epoch_size               epoch size: N, default is 1
    --device_id                device id: N, default is 0
    --device_num               number of used devices: N, default is 1
    --save_ckpt_path           path to save checkpoint files: PATH, default is ""
    --max_ckpt_num             max number for saving checkpoint files: N, default is 1
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --data_sink_steps          set data sink steps: N, default is 1
    --save_checkpoint_step     steps for saving checkpoint files: N, default is 1000
    --load_teacher_ckpt_path   path to load teacher checkpoint files: PATH, default is ""
    --data_dir                 path to dataset directory: PATH, default is ""
    --schema_dir               path to schema.json file, PATH, default is ""
    --dataset_type             the dataset type which can be tfrecord/mindrecord, default is tfrecord
```

### Task Distill

```text
usage: run_general_task.py  [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [--do_eval DO_EVAL]
                            [--td_phase1_epoch_size N] [--td_phase2_epoch_size N]
                            [--device_id N] [--do_shuffle DO_SHUFFLE]
                            [--enable_data_sink ENABLE_DATA_SINK] [--save_ckpt_step N]
                            [--max_ckpt_num N] [--data_sink_steps N]
                            [--load_teacher_ckpt_path LOAD_TEACHER_CKPT_PATH]
                            [--load_gd_ckpt_path LOAD_GD_CKPT_PATH]
                            [--load_td1_ckpt_path LOAD_TD1_CKPT_PATH]
                            [--train_data_dir TRAIN_DATA_DIR]
                            [--eval_data_dir EVAL_DATA_DIR] [--task_type TASK_TYPE]
                            [--task_name TASK_NAME] [--schema_dir SCHEMA_DIR] [--dataset_type DATASET_TYPE]
                            [--assessment_method ASSESSMENT_METHOD]

options:
    --device_target            device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --do_train                 enable train task: "true" | "false", default is "true"
    --do_eval                  enable eval task: "true" | "false", default is "true"
    --td_phase1_epoch_size     epoch size for td phase1: N, default is 10
    --td_phase2_epoch_size     epoch size for td phase2: N, default is 3
    --device_id                device id: N, default is 0
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --save_ckpt_step           steps for saving checkpoint files: N, default is 1000
    --max_ckpt_num             max number for saving checkpoint files: N, default is 1
    --data_sink_steps          set data sink steps: N, default is 1
    --load_teacher_ckpt_path   path to load teacher checkpoint files: PATH, default is ""
    --load_gd_ckpt_path        path to load checkpoint files which produced by general distill: PATH, default is ""
    --load_td1_ckpt_path       path to load checkpoint files which produced by task distill phase 1: PATH, default is ""
    --train_data_dir           path to train dataset directory: PATH, default is ""
    --eval_data_dir            path to eval dataset directory: PATH, default is ""
    --task_type                task type: "classification" | "ner", default is "classification"
    --task_name                classification or ner task: "SST-2" | "QNLI" | "MNLI" | "TNEWS", "CLUENER", default is ""
    --assessment_method        assessment method to do evaluation: acc | f1
    --schema_dir               path to schema.json file, PATH, default is ""
    --dataset_type             the dataset type which can be tfrecord/mindrecord, default is tfrecord
```

## Options and Parameters

`gd_config.yaml` and `td_config/*.yaml` contain parameters of BERT model and options for optimizer and lossscale.

### Options

```text
batch_size                          batch size of input dataset: N, default is 16
Parameters for lossscale:
    loss_scale_value                initial value of loss scale: N, default is 2^8
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 50

Parameters for optimizer:
    learning_rate                   value of learning rate: Q
    end_learning_rate               value of end learning rate: Q, must be positive
    power                           power: Q
    weight_decay                    weight decay: Q
    eps                             term added to the denominator to improve numerical stability: Q
```

### Parameters

```text
Parameters for bert network:
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, must be consistent with the dataset you use. Default is 30522
                                    Usually, we use 21128 for CN vocabs and 30522 for EN vocabs according to the origin paper. Default is 30522
    hidden_size                     size of bert encoder layers: N
    num_hidden_layers               number of hidden layers: N
    num_attention_heads             number of attention heads: N, default is 12
    intermediate_size               size of intermediate layer: N
    hidden_act                      activation function used: ACTIVATION, default is "gelu"
    hidden_dropout_prob             dropout probability for BertOutput: Q
    attention_probs_dropout_prob    dropout probability for BertAttention: Q
    max_position_embeddings         maximum length of sequences: N, default is 512
    save_ckpt_step                  number for saving checkponit: N, default is 100
    max_ckpt_num                    maximum number for saving checkpoint: N, default is 1
    type_vocab_size                 size of token type vocab: N, default is 2
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    use_relative_positions          use relative positions or not: True | False, default is False
    dtype                           data type of input: mstype.float16 | mstype.float32, default is mstype.float32
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float16
```

## [Training Process](#contents)

### Training

#### running on Ascend

Before running the command below, please check `load_teacher_ckpt_path`, `data_dir` and `schma_dir` has been set. Please set the path to be the absolute full path, e.g:"/username/checkpoint_100_300.ckpt".

```bash
bash scripts/run_standalone_gd.sh
```

The command above will run in the background, you can view the results the file log.txt. After training, you will get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```text
# grep "epoch" log.txt
epoch: 1, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, 28.2093), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 2, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, 30.1724), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **Attention** This will bind the processor cores according to the `device_num` and total processor numbers. If you don't expect to run pretraining with binding processor cores, remove the operations about `taskset` in `scripts/run_distributed_gd_ascend.sh`

#### running on GPU

Before running the command below, please check `load_teacher_ckpt_path`, `data_dir` `schma_dir` and `device_target=GPU` has been set. Please set the path to be the absolute full path, e.g:"/username/checkpoint_100_300.ckpt".

```bash
bash scripts/run_standalone_gd.sh
```

The command above will run in the background, you can view the results the file log.txt. After training, you will get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```text
# grep "epoch" log.txt
epoch: 1, step: 100, outputs are 28.2093
...
```

### Distributed Training

#### running on Ascend

Before running the command below, please check `load_teacher_ckpt_path`, `data_dir` and `schma_dir` has been set. Please set the path to be the absolute full path, e.g:"/username/checkpoint_100_300.ckpt".

```bash
bash scripts/run_distributed_gd_ascend.sh 8 1 /path/hccl.json
```

The command above will run in the background, you can view the results the file log.txt. After training, you will get some checkpoint files under the LOG* folder by default. The loss value will be achieved as follows:

```text
# grep "epoch" LOG*/log.txt
epoch: 1, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, 28.1478), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 1, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, 30.5901), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

#### running on GPU

Please input the path to be the absolute full path, e.g:"/username/checkpoint_100_300.ckpt".

```bash
bash scripts/run_distributed_gd_gpu.sh 8 1 /path/data/ /path/schema.json /path/teacher.ckpt
```

The command above will run in the background, you can view the results the file log.txt. After training, you will get some checkpoint files under the LOG* folder by default. The loss value will be achieved as follows:

```text
# grep "epoch" LOG*/log.txt
epoch: 1, step: 1, outputs are 63.4098
...
```

## [Evaluation Process](#contents)

### Evaluation

If you want to after running and continue to eval, please set `do_train=true` and `do_eval=true`, If you want to run eval alone, please set `do_train=false` and `do_eval=true`. If running on GPU, please set `device_target=GPU`.

#### evaluation on SST-2 dataset  

```bash
bash scripts/run_standalone_td.sh {path}/*.yaml
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```bash
# grep "The best acc" log.txt
The best acc is 0.872685
The best acc is 0.893515
The best acc is 0.899305
...
The best acc is 0.902777
...
```

#### evaluation on MNLI dataset

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:"/username/pretrain/checkpoint_100_300.ckpt".

```bash
bash scripts/run_standalone_td.sh {path}/*.yaml
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```text
# grep "The best acc" log.txt
The best acc is 0.803206
The best acc is 0.803308
The best acc is 0.810355
...
The best acc is 0.813929
...
```

#### evaluation on QNLI dataset

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:"/username/pretrain/checkpoint_100_300.ckpt".

```bash
bash scripts/run_standalone_td.sh {path}/*.yaml
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```text
# grep "The best acc" log.txt
The best acc is 0.870772
The best acc is 0.871691
The best acc is 0.875183
...
The best acc is 0.891176
...
```

#### evaluation on TNEWS dataset

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:"/username/pretrain/checkpoint_100_300.ckpt".

```bash
bash scripts/run_standalone_td.sh {path}/*.yaml # on CPU device
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```text
# grep "The best acc" log.txt
The best acc is 0.506787
The best acc is 0.515646
The best acc is 0.518760
...
The best acc is 0.534121
...
```

#### evaluation on CLUENER dataset

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:"/username/pretrain/checkpoint_100_300.ckpt".

```bash
bash scripts/run_standalone_td.sh {path}/*.yaml # on CPU device
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```text
# grep "The best acc" log.txt
The best acc is 0.889724
The best acc is 0.894650
The best acc is 0.900675
...
The best acc is 0.919423
...
```

## [ONNX Export And Evaluation](#contents)

### ONNX Export

```bash
python export.py --ckpt_file [CKPT_PATH] --task_name [TASK_NAME] --file_name [FILE_NAME] --file_format "ONNX" --config_path [CONFIG_PATH]
# example:python export.py --ckpt_file tinybert_ascend_v170_enwiki128_sst2_official_nlp_acc90.28.ckpt --task_name SST-2 --file_name 'sst2_tinybert' --file_format "ONNX" --config_path td_config/td_config_sst2.yaml
```

### ONNX Evaluation

#### ONNX evaluation on SST-2 dataset  

Before running the command below, please check the onnx path has been set. Please set the onnx path to be the absolute full path, e.g:"/home/username/models/research/nlp/tinybert/SST-2.onnx".

```bash
bash scripts/run_eval_onnx.sh {path}/*.yaml
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```text
=================================================================
============== acc is 0.8862132352941177
=================================================================
```

#### ONNX evaluation on MNLI dataset

Before running the command below, please check the onnx path has been set. Please set the onnx path to be the absolute full path, e.g:"/home/username/models/research/nlp/tinybert/MNLI.onnx".

```bash
bash scripts/run_eval_onnx.sh {path}/*.yaml
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```text
=================================================================
============== acc is 0.8862132352941177
=================================================================
```

#### ONNX evaluation on QNLI dataset

Before running the command below, please check the onnx path has been set. Please set the onnx path to be the absolute full path, e.g:"/home/username/models/research/nlp/tinybert/QNLI.onnx".

```bash
bash scripts/run_eval_onnx.sh {path}/*.yaml
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```text
=================================================================
============== acc is 0.8862132352941177
=================================================================
```

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

- Export on local

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
#example:
python export.py --ckpt_file ./2021-09-03_time_16_00_12/tiny_bert_936_100.ckpt --file_name SST-2 --file_format MINDIR --config_path ./td_config/td_config_sst2.yaml
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# (1) Upload the code folder to S3 bucket.
# (2) Click to "create training task" on the website UI interface.
# (3) Set the code directory to "/{path}/tinybert" on the website UI interface.
# (4) Set the startup file to /{path}/tinybert/export.py" on the website UI interface.
# (5) Perform a or b.
#     a. Set parameters in a *.yaml file under /path/tinybert/td_config/
#         1. Set ”enable_modelarts: True“
#         2. Set “ckpt_file: ./{path}/*.ckpt”('ckpt_file' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
#         3. Set ”file_name: bert_ner“
#         4. Set ”file_format：MINDIR“
#     b. Adding on the website UI interface.
#         1. Add ”enable_modelarts=True“
#         2. Add “ckpt_file=./{path}/*.ckpt”('ckpt_file' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
#         3. Add ”file_name=tinybert_sst2“
#         4. Add ”file_format=MINDIR“
#     Finally, "config_path=../../td_config/*.yaml" must be added on the web page (select the *.yaml configuration file according to the downstream task)
# (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.(Although it is useless, but to do)
# (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (9) Under the item "resource pool selection", select the specification of a single card.
# (10) Create your job.
# You will see tinybert_sst2.mindir under {Output file path}.
```

### Infer on Ascend310

#### code supplement  

NLP infer using a few opensource code, those code save as another repo->[NLP_infer_opensource](https://gitee.com/Stan.Xu/bert_tokenizer.git),supplement those code before you run infer code.step like below:  

```shell
# 1. mkdir ./infer/opensource  
# 2. downloading code form https://gitee.com/Stan.Xu/bert_tokenizer.git  
# 3. all code file in this repo, move to ./infer/opensource  
```  

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [SCHEMA_DIR] [DATASET_TYPE] [TASK_NAME] [ASSESSMENT_METHOD] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
=================================================================
============== acc is 0.8862132352941177
=================================================================
```

## [Model Description](#contents)

## [Performance](#contents)

### training Performance

| Parameters                 | Ascend                                                     | GPU                       |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| Model Version              | TinyBERT                                                   | TinyBERT                           |
| Resource                   |Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8           | NV SMX2 V100-32G, cpu:2.10GHz 64cores,  memory:251G         |
| uploaded Date              | 07/05/2021                                                 | 07/05/2021                |
| MindSpore Version          | 1.3.0                                                      | 1.3.0                     |
| Dataset                    | en-wiki-128                                                | en-wiki-128               |
| Training Parameters        | src/gd_config.yaml                                           | src/gd_config.yaml          |
| Optimizer                  | AdamWeightDecay                                            | AdamWeightDecay           |
| Loss Function              | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| outputs                    | probability                                                | probability               |
| Loss                       | 6.541583                                                   | 6.6915                    |
| Speed                      | 35.4ms/step                                                | 98.654ms/step             |
| Total time                 | 17.3h(3poch, 8p)                                           | 48h(3poch, 8p)            |
| Params (M)                 | 15M                                                        | 15M                       |
| Checkpoint for task distill| 74M(.ckpt file)                                            | 74M(.ckpt file)           |
| Scripts                    | [TinyBERT](https://gitee.com/mindspore/models/tree/master/research/nlp/tinybert) |          |

#### Inference Performance

> SST2 dataset

| Parameters                 | Ascend                        | GPU                       |
| -------------------------- | ----------------------------- | ------------------------- |
| Model Version              |                               |                           |
| Resource                   | Ascend 910; OS Euler2.8       | NV SMX2 V100-32G          |
| uploaded Date              | 07/05/2021                    | 07/05/2021                |
| MindSpore Version          | 1.3.0                         | 1.3.0                     |
| Dataset                    | SST-2,                        | SST-2                     |
| batch_size                 | 32                            | 32                        |
| Accuracy                   | 0.902777                      | 0.9086                    |
| Model for inference        | 74M(.ckpt file)               | 74M(.ckpt file)           |

> QNLI dataset

| Parameters                 | Ascend                        | GPU                       |
| --------------             | ----------------------------- | ------------------------- |
| Model Version              |                               |                           |
| Resource                   | Ascend 910; OS Euler2.8       | NV SMX2 V100-32G          |
| uploaded Date              | 2021-12-17                    | 2021-12-17                |
| MindSpore Version          | 1.5.0                         | 1.5.0                     |
| Dataset                    | QNLI                          | QNLI                      |
| batch_size                 | 32                            | 32                        |
| Accuracy                   | 0.8860                        | 0.8755                    |
| Model for inference        | 74M(.ckpt file)               | 74M(.ckpt file)           |

> MNLI dataset

| Parameters                 | Ascend                        | GPU                       |
| --------------             | ----------------------------- | ------------------------- |
| Model Version              |                               |                           |
| Resource                   | Ascend 910；系统 Euler2.8      | NV SMX2 V100-32G          |
| uploaded Date              | 2021-12-17                    | 2021-12-17                |
| MindSpore Version          | 1.5.0                         | 1.5.0                     |
| Dataset                    | QNLI                          | QNLI                      |
| batch_size                 | 32                            | 32                        |
| Accuracy                   | 0.8116                        | 0.9086                    |
| Model for inference        | 74M(.ckpt file)               | 74M(.ckpt file)           |

> TNEWS dataset

| Parameters                 | CPU                        |
| --------------             | ----------------------------- |
| Model Version              |                               |
| Resource                   | CPU, 192cores                 |
| uploaded Date              | 2021-12-17                    |
| MindSpore Version          | 1.5.0                         |
| Dataset                    | QNLI                          |
| batch_size                 | 32                            |
| Accuracy                   | 0.53                          |
| Model for inference        | 74M(.ckpt file)               |

> CLUENER dataset

| Parameters                 | CPU                        |
| --------------             | ----------------------------- |
| Model Version              |                               |
| Resource                   | CPU, 192cores      |
| uploaded Date              | 2021-12-17                    |
| MindSpore Version          | 1.5.0                         |
| Dataset                    | QNLI                          |
| batch_size                 | 32                            |
| Accuracy                   | 0.9150                        |
| Model for inference        | 74M(.ckpt file)               |

# [Description of Random Situation](#contents)

In run_standaloned_td.sh, we set do_shuffle to shuffle the dataset.

In gd_config.yaml and td_config/*.yaml, we set the hidden_dropout_prob and attention_pros_dropout_prob to dropout some network node.

In run_general_distill.py, we set the random seed to make sure distribute training has the same init weight.

If accuracy < standard, may be scipy version < 1.7.

if this error occurs, `connect p2p timeout, timeout: 120s.`, please add `export HCCL_CONNECT_TIMEOUT=600` in shell to resolve it.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
