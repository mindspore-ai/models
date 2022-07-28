# Contents

- [Contents](#contents)
- [ALBERT Description](#albert-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Pre-Training](#pre-training)
        - [Fine-Tuning and Evaluation](#fine-tuning-and-evaluation)
    - [Options and Parameters](#options-and-parameters)
        - [Options:](#options)
        - [Parameters:](#parameters)
    - [Training Process](#training-process)
        - [Training](#training)
            - [Running on Ascend](#running-on-ascend)
            - [Running on GPU](#running-on-gpu)
    - [Distributed Training](#distributed-training)
        - [Running on Ascend](#running-on-ascend-1)
        - [Running on GPU](#running-on-gpu)
    - [Evaluation Process](#evaluation-process)
    - [Evaluation](#evaluation)
        - [evaluation on SST-2 dataset when running on Ascend](#evaluation-on-cola-dataset-when-running-on-ascend)
        - [evaluation on squad v1.1 dataset when running on Ascend](#evaluation-on-squad-v11-dataset-when-running-on-ascend)
        - [Export MindIR](#export-mindir)
        - [Inference Process](#inference-process)
            - [Usage](#usage)
            - [result](#result)
    - [Model Description](#model-description)
    - [Performance](#performance)
        - [Pretraining Performance](#pretraining-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [ALBERT Description](#contents)

The ALBERT network was proposed by Google in 2019. The network has made a breakthrough in the field of NLP. The network uses pre-training to achieve a large network structure without modifying, and only by adding an output layer to achieve multiple text-based tasks in fine-tuning. The ALBERT present two parameter reduction techniques (Factorized embedding parameterization and Cross-layer parameter sharing) to lower memory consumption and increase the training
speed of BERT. The attention mechanism is introduced to enable the output layer to capture high-latitude global semantic information. The pre-training uses denoising and self-encoding tasks, namely MLM(Masked Language Model) and SOP( sentence-order prediction). No need to label data, pre-training can be performed on massive text data, and only a small amount of data to fine-tuning downstream tasks to obtain good results.

- [Paper](https://arxiv.org/abs/1909.11942):  Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2020). [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations]((https://arxiv.org/abs/1909.11942)). ArXiv, abs/1909.11942.

- [Paper](https://arxiv.org/abs/1810.04805):  Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding]((https://arxiv.org/abs/1810.04805)). arXiv preprint arXiv:1810.04805.

# [Model Architecture](#contents)

The backbone of the ALBERT architecture is similar to BERT in that it uses a transformer encoder with GELU nonlinearities.

# [Dataset](#contents)

- Download the zhwiki or enwiki dataset for pre-training. Extract and refine texts in the dataset with [WikiExtractor](https://github.com/attardi/wikiextractor). Convert the dataset to TFRecord format. Please refer to create_pretraining_data.py file in [ALBERT](https://github.com/google-research/albert) repository.
- Download dataset for fine-tuning and evaluation such as [GLUE](https://gluebenchmark.com/tasks), SQuAD v1.1, etc. Convert dataset files from JSON(or TSV) format to MINDRECORD format, please run ```bash convert_finetune_datasets_[DOWNSTREAM_TASK].sh``` in /path/albert/scripts as follows:

Before running the command below, please check the vocab and spm model file path has been set. Please set the path to be the absolute full path, e.g:

--vocab_path="albert_base/30k-clean.vocab" \

--spm_model_file="albert_base/30k-clean.model"

You can download from [ALBERT](https://github.com/google-research/albert) or [ALBERT_BASE](https://storage.googleapis.com/albert_models/albert_base_v1.tar.gz).

```bash
# convert classifier task dataset from tsv to mindrecord, [TASK_NAME] is dataset name
bash scripts/convert_finetune_datasets_classifier.sh

# convert squad task dataset from JSON to mindrecord
bash scripts/convert_finetune_datasets_squad.sh
```

You can convert different dataset by changing [TASK_NAME], e.g.:TASK_NAME="MNLI"

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend/GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)

- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

- Convert pre-training model

For the Tensorflow albert pre-train model, you can download from [Albert](https://github.com/google-research/albert),
or [ALBERT_BASE](https://storage.googleapis.com/albert_models/albert_base_v1.tar.gz).

Firstly, convert model.ckpt-best into npy files. You need tensorflow environment, run read_weight_tf.py as follows:

```text
python convert_tf_ckpt/read_weight_tf.py --ckpt_file_path=albert_base/model.ckpt-best --output_path=path/output/
```

Secondly, convert all npy files into one MindSpore ckpt file. You need MindSpore environment, run save_weight_ms.py as follows:

```text
python convert_tf_ckpt/save_weight_ms.py --load_dir=npy_file_path/ --output_file_name=path/ms_albert_pretrain.ckpt
```

*load_dir equal to the output_path in first step.

After installing MindSpore via the official website, you can start pre-training, fine-tuning and evaluation as follows:

- Running on Ascend

```bash
# run standalone pre-training example
bash scripts/run_standalone_pretrain_ascend.sh 0 1 /path/cn-wiki-128

# run distributed pre-training example
bash scripts/run_distributed_pretrain_ascend.sh /path/cn-wiki-128 /path/hccl.json

# run fine-tuning and evaluation example
- If you are going to run a fine-tuning task, please prepare a checkpoint generated from pre-training.
- Set albert network config and optimizer hyperparameters in `task_[DOWNSTREAM_TASK]_config.yaml`.

- Classification task: Set task related hyperparameters in scripts/run_classifier.sh.
- Run `bash scripts/run_classifier.sh` for fine-tuning of AlBERT-base model.

  bash scripts/run_classifier.sh

- SQuAD task: Set task related hyperparameters in scripts/run_squad.sh.
- Run `bash scripts/run_squad.sh` for fine-tuning of ALBERT-base model.

  bash scripts/run_squad.sh
```

- Running on ModelArts

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows

- Pretraining with 8 cards on ModelArts

```python
# (1) Upload the code folder to S3 bucket.
# (2) Click to "create training task" on the website UI interface.
# (3) Set the code directory to "/{path}/albert" on the website UI interface.
# (4) Set the startup file to /{path}/albert/run_pretrain.py" on the website UI interface.
# (5) Perform a or b.
#     a. setting parameters in /{path}/albert/pretrain_config.yaml.
#         1. Set ”enable_modelarts=True“
#         2. Set other parameters, other parameter configuration can refer to `./scripts/run_distributed_pretrain_ascend.sh`
#     b. adding on the website UI interface.
#         1. Add ”enable_modelarts=True“
#         3. Add other parameters, other parameter configuration can refer to `./scripts/run_distributed_pretrain_ascend.sh`
# (6) Upload the dataset to S3 bucket.
# (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (there is only data or zip package under this path).
# (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (9) Under the item "resource pool selection", select the specification of 8 cards.
# (10) Create your job.
# After training, the '*.ckpt' file will be saved under the'training output file path'
```

- Running downstream tasks with single card on ModelArts

```python
# (1) Upload the code folder to S3 bucket.
# (2) Click to "create training task" on the website UI interface.
# (3) Set the code directory to "/{path}/albert" on the website UI interface.
# (4) Set the startup file to /{path}/albert/run_classifier.py"(run_squad_v1.py) on the website UI interface.
# (5) Perform a or b.
#     a. setting parameters in task_classifier_config.yaml(task_squad_config.yaml under the folder `/{path}/albert/`
#         1. Set ”enable_modelarts=True“
#         2. Set other parameters, other parameter configuration can refer to `task_classifier_config.sh`(or run_squad.sh) under the folder '{path}/albert/scripts/'.
#     b. adding on the website UI interface.
#         1. Add ”enable_modelarts=True“
#         2. Set other parameters, other parameter configuration can refer to `task_classifier_config.sh`(or run_squad.sh) under the folder '{path}/albert/scripts/'.
#     Note that vocab_file_path, train_data_file_path, eval_data_file_path, schema_file_path fill in the relative path relative to the path selected in step 7.
#     Finally, "config_path=/path/*.yaml" must be added on the web page (select the *.yaml configuration file according to the downstream task)
# (6) Upload the dataset to S3 bucket.
# (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (there is only data or zip package under this path).
# (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (9) Under the item "resource pool selection", select the specification of a single card.
# (10) Create your job.
# After training, the '*.ckpt' file will be saved under the'training output file path'
```

For distributed training, an hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link below:
[https://gitee.com/mindspore/models/tree/master/utils/hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

```text
For pretraining, schema file contains ["input_ids", "input_mask", "segment_ids", "next_sentence_labels", "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"].

For ner or classification task, schema file contains ["input_ids", "input_mask", "segment_ids", "label_ids"].

For squad task, training: schema file contains ["start_positions", "end_positions", "input_ids", "input_mask", "segment_ids"], evaluation: schema file contains ["input_ids", "input_mask", "segment_ids"].

`numRows` is the only option which could be set by user, other values must be set according to the dataset.

For example, the schema file of cn-wiki-128 dataset for pretraining shows as follows:
{
    "datasetType": "TF",
    "numRows": 7680,
    "columns": {
        "input_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [128]
        },
        "input_mask": {
            "type": "int64",
            "rank": 1,
            "shape": [128]
        },
        "segment_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [128]
        },
        "next_sentence_labels": {
            "type": "int64",
            "rank": 1,
            "shape": [1]
        },
        "masked_lm_positions": {
            "type": "int64",
            "rank": 1,
            "shape": [20]
        },
        "masked_lm_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [20]
        },
        "masked_lm_weights": {
            "type": "float32",
            "rank": 1,
            "shape": [20]
        }
    }
}
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─albert
  ├─ascend310_infer
  ├─README.md
  ├──convert_tf_ckpt
        ├──read_weight_tf.py                # read tensorflow pretrain model
        ├──trans_dict.py                    # pretrain model params dict
        ├──save_weight_ms.py                # generate mindspore ckpt
  ├─scripts
    ├─ascend_distributed_launcher
        ├─__init__.py
        ├─hyper_parameter_config.ini          # hyper parameter for distributed pretraining
        ├─get_distribute_pretrain_cmd.py      # script for distributed pretraining
        └─README.md
    ├─convert_finetune_datasets_classifier.sh # shell script for converting dataset files from TSV format to MINDRECORD format
    ├─convert_finetune_datasets_squad.sh      # shell script for converting dataset files from JSON format to MINDRECORD format
    ├─run_classifier.sh                       # shell script for standalone classifier task on ascend or gpu
    ├─run_distributed_classiffier_ascend.sh   # shell script for distributed classifier task on ascend
    ├─run_distributed_pretrain_ascend.sh      # shell script for distributed pretrain on ascend
    ├─run_distributed_pretrain_gpu.sh         # shell script for distributed pretrain on gpu
    ├─run_distributed_squad_ascend.sh         # shell script for distributed SQUAD task on ascend
    ├─run_squad.sh                            # shell script for standalone SQUAD task on ascend or gpu
    ├─run_standalone_pretrain_ascend.sh       # shell script for standalone pretrain on ascend
    └─run_standaloned_pretrain_gpu.sh         # shell script for distributed pretrain on gpu
  ├─src
    ├─model_utils
      ├──config.py                            # parse *.yaml parameter configuration file
      ├──device_adapter.py                    # distinguish local/ModelArts training
      ├──local_adapter.py                     # get related environment variables in local training
      └──moxing_adapter.py                    # get related environment variables in ModelArts training
    ├─__init__.py
    ├─Albert_Callback.py                      # SQUAD task callback
    ├─Albert_Callback_class.py                # classifier task callback
    ├─albert_for_finetune.py                  # backbone code of network
    ├─albert_for_pre_training.py              # backbone code of network
    ├─albert_model.py                         # backbone code of network
    ├─assessment_method.py                    # assessment method for evaluation
    ├─classifier_utils.py                     # classifier task util function
    ├─dataset.py                              # data preprocessing
    ├─finetune_eval_model.py                  # backbone code of network
    ├─squad_get_predictions.py                # SQUAD task predictions
    ├─squad_postprocess.py                    # SQUAD task postprocess
    ├─squad_utils.py                          # SQUAD task utils
    ├─tokenization.py                         # tokenization for albert and downstream task
    ├─utils.py                                # util function
  ├─pretrain_config.yaml                      # parameter configuration for pretrain
  ├─pretrain_eval.py                          # train and eval net
  ├─run_classifier.py                         # finetune and eval net for classifier task
  ├─run_pretrain.py                           # train net for pretraining phase
  ├─run_squad_v1.py                           # finetune and eval net for squad task
  ├─task_classifier_config.yaml               # parameter configuration for downstream_task_classifier
  └─task_squad_config.yaml                    # parameter configuration for downstream_task_squad
```

## [Script Parameters](#contents)

### Pre-Training

```text
usage: run_pretrain.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                        [--enable_save_ckpt ENABLE_SAVE_CKPT] [--device_target DEVICE_TARGET]
                        [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                        [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                        [--accumulation_steps N]
                        [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                        [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                        [--save_checkpoint_steps N] [--save_checkpoint_num N]
                        [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR] [train_steps N]

options:
    --device_target            device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --distribute               pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --epoch_size               epoch size: N, default is 1
    --device_num               number of used devices: N, default is 1
    --device_id                device id: N, default is 0
    --enable_save_ckpt         enable save checkpoint: "true" | "false", default is "true"
    --enable_lossscale         enable lossscale: "true" | "false", default is "true"
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --data_sink_steps          set data sink steps: N, default is 1
    --accumulation_steps       accumulate gradients N times before weight update: N, default is 1
    --save_checkpoint_path     path to save checkpoint files: PATH, default is ""
    --load_checkpoint_path     path to load checkpoint files: PATH, default is ""
    --save_checkpoint_steps    steps for saving checkpoint files: N, default is 1000
    --save_checkpoint_num      number for saving checkpoint files: N, default is 1
    --train_steps              Training Steps: N, default is -1
    --data_dir                 path to dataset directory: PATH, default is ""
    --schema_dir               path to schema.json file, PATH, default is ""
```

### Fine-Tuning and Evaluation

```text
usage: run_squad_v1.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--device_id N] [--epoch_num N] [--num_class N]
                    [--vocab_file_path VOCAB_FILE_PATH]
                    [--eval_json_path EVAL_JSON_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
                    [--spm_model_file SPM_MODEL_FILE]
                    [--predict_feature_left_file PREDICT_FEATURE_LEFT_FILE]
options:
    --device_target                   device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --do_train                        whether to run training on training set: true | false
    --do_eval                         whether to run eval on dev set: true | false
    --device_id                       device id to run task
    --epoch_num                       total number of training epochs to perform
    --num_class                       number of classes to classify, usually 2 for squad task
    --train_data_shuffle              Enable train data shuffle, default is true
    --eval_data_shuffle               Enable eval data shuffle, default is true
    --vocab_file_path                 the vocabulary file that the BERT model was trained on
    --save_finetune_checkpoint_path   path to save generated finetuning checkpoint
    --load_pretrain_checkpoint_path   initial checkpoint (usually from a pre-trained ALBERT model)
    --load_finetune_checkpoint_path   give a finetuning checkpoint path if only do eval
    --train_data_file_path            squad mindrecord for training. E.g., train1.1.mindrecord
    --eval_json_path                  squad eval json file for predictions. E.g., dev1.1.json
    --schema_file_path                path to datafile schema file
    --spm_model_file                  path to datafile spm file
    --predict_feature_left_file       path to datafile predict file

usage: run_classifier.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                         [--assessment_method ASSESSMENT_METHOD] [--device_id N] [--epoch_num N] [--num_class N]
                         [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                         [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                         [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                         [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                         [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                         [--train_data_file_path TRAIN_DATA_FILE_PATH]
                         [--eval_data_file_path EVAL_DATA_FILE_PATH]
                         [--schema_file_path SCHEMA_FILE_PATH]
                         [--spm_model_file SPM_MODEL_FILE]
options:
    --device_target                   targeted device to run task: Ascend | GPU
    --do_train                        whether to run training on training set: true | false
    --do_eval                         whether to run eval on dev set: true | false
    --assessment_method               assessment method to do evaluation: accuracy | f1 | mcc | spearman_correlation
    --device_id                       device id to run task
    --epoch_num                       total number of training epochs to perform
    --num_class                       number of classes to do labeling
    --train_data_shuffle              Enable train data shuffle, default is true
    --eval_data_shuffle               Enable eval data shuffle, default is true
    --save_finetune_checkpoint_path   path to save generated finetuning checkpoint
    --load_pretrain_checkpoint_path   initial checkpoint (usually from a pre-trained ALBERT model)
    --load_finetune_checkpoint_path   give a finetuning checkpoint path if only do eval
    --train_data_file_path            mindrecord for training. E.g., train.mindrecord
    --eval_data_file_path             mindrecord for predictions. E.g., dev.mindrecord
    --schema_file_path                path to datafile schema file
    --spm_model_file                  path to datafile spm file
```

## Options and Parameters

Parameters for training and downstream task can be set in yaml config file respectively.

### Options

```text
config for lossscale and etc.
    bert_network                    version of ALBERT model: base | large | xlarge | xxlarge, default is base
    batch_size                      batch size of input dataset: N, default is 32
    loss_scale_value                initial value of loss scale: N, default is 2^32
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 1000
    optimizer                       optimizer used in the network: AdamWeightDecay | Lamb | Momentum, default is "AdamWeightDecay"
```

### Parameters

```text
Parameters for dataset and network (Pre-Training/Fine-Tuning/Evaluation):
    seq_length                      length of input sequence: N, default is 512
    vocab_size                      size of each embedding vector: N, must be consistent with the dataset you use. Default is 30000
    hidden_size                     size of bert encoder layers: N, default is 768
    num_hidden_layers               number of hidden layers: N, default is 12
    num_attention_heads             number of attention heads: N, default is 12
    intermediate_size               size of intermediate layer: N, default is 3072
    hidden_act                      activation function used: ACTIVATION, default is "gelu"
    hidden_dropout_prob             dropout probability for BertOutput: Q, default is 0.1
    attention_probs_dropout_prob    dropout probability for BertAttention: Q, default is 0.1
    max_position_embeddings         maximum length of sequences: N, default is 512
    type_vocab_size                 size of token type vocab: N, default is 2
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    use_relative_positions          use relative positions or not: True | False, default is False
    dtype                           data type of input: mstype.float16 | mstype.float32, default is mstype.float32
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float16
    num_hidden_groups               group(s) of hidden layers: Q, default is 1
    inner_group_num                 group(s) of attention: Q, default is 1
Parameters for optimizer:
    AdamWeightDecay:
    decay_steps                     steps of the learning rate decay: N
    learning_rate                   value of learning rate: Q
    end_learning_rate               value of end learning rate: Q, must be positive
    power                           power: Q
    warmup_steps                    steps of the learning rate warm up: N
    weight_decay                    weight decay: Q
    eps                             term added to the denominator to improve numerical stability: Q

    Lamb:
    decay_steps                     steps of the learning rate decay: N
    learning_rate                   value of learning rate: Q
    end_learning_rate               value of end learning rate: Q
    power                           power: Q
    warmup_steps                    steps of the learning rate warm up: N
    weight_decay                    weight decay: Q

    Momentum:
    learning_rate                   value of learning rate: Q
    momentum                        momentum for the moving average: Q
```

## [Training Process](#contents)

### Training

#### Running on Ascend

```bash
bash scripts/run_standalone_pretrain_ascend.sh 0 1 /path/cn-wiki-128
```

The command above will run in the background, you can view training logs in pretraining_log.txt. After training finished, you will get some checkpoint files under the script folder by default. The loss values will be displayed as follows:

```text
# grep "epoch" pretraining_log.txt
epoch: 0.0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0856101e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.000, step: 2, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0821701e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

#### Running on GPU

```bash
bash scripts/run_standalone_pretrain_for_gpu.sh 0 1 /path/cn-wiki-128
```

The command above will run in the background, you can view the results the file pretraining_log.txt. After training, you will get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```text
# grep "epoch" pretraining_log.txt
epoch: 0.0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0856101e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.000, step: 2, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0821701e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **Attention** If you are running with a huge dataset, it's better to add an external environ variable to make sure the hccl won't timeout.
>
> ```bash
> export HCCL_CONNECT_TIMEOUT=600
> ```
>
> This will extend the timeout limits of hccl from the default 120 seconds to 600 seconds.
> **Attention** If you are running with a big bert model, some error of protobuf may occurs while saving checkpoints, try with the following environ set.
>
> ```bash
> export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
> ```

### Distributed Training

#### Running on Ascend

```bash
bash scripts/run_distributed_pretrain_ascend.sh /path/cn-wiki-128 /path/hccl.json
```

The command above will run in the background, you can view training logs in pretraining_log.txt. After training finished, you will get some checkpoint files under the LOG* folder by default. The loss value will be displayed as follows:

```text
# grep "epoch" LOG*/pretraining_log.txt
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08209e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07566e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08218e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07770e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

#### Running on GPU

```bash
bash scripts/run_distributed_pretrain_for_gpu.sh /path/cn-wiki-128
```

The command above will run in the background, you can view the results the file pretraining_log.txt. After training, you will get some checkpoint files under the LOG* folder by default. The loss value will be achieved as follows:

```text
# grep "epoch" LOG*/pretraining_log.txt
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08209e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07566e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08218e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07770e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **Attention** This will bind the processor cores according to the device_num and total processor numbers. If you don't expect to run pretraining with binding processor cores, remove the operations about taskset in scripts/ascend_distributed_launcher/get_distribute_pretrain_cmd.py

## [Evaluation Process](#contents)

### Evaluation

#### evaluation on SST-2 dataset when running on Ascend

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_albert-20000_1.ckpt" \

--train_data_file_path="/data/sst2/train.mindrecord" \

--eval_data_file_path="/data/sst2/dev.mindrecord"

```bash
bash scripts/run_classifier.sh
```

The command above will run in the background, you can view training logs in classfier_log.txt in albert root path.
"load_pretrain_checkpoint_path" is mindspore ckpt file.

If you choose accuracy as assessment method, the result will be as follows:

```text
acc_num XXX , total_num XXX, accuracy 0.904817
```

#### evaluation on squad v1.1 dataset when running on Ascend

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:

--vocab_file_path="/data/albert_base/30k-clean.vocab" \

--load_pretrain_checkpoint_path="/data/scripts/albert_converted.ckpt" \

--train_data_file_path="/data/squad/train.mindrecord" \

--eval_json_path="/data/squad/dev-v1.1.json" \

```bash
bash scripts/run_squad.sh
```

The command above will run in the background, you can view training logs in squad_log.txt.
The result will be as follows:

```text
{"exact_match": 82.25165562913908, "f1": 89.41021197411379}
```

### [Export MindIR](#contents)

- Export on local

We only support export with fine-tuned downstream task model and yaml config file, because the pretrained model is useless in inferences task.

```shell
python export.py --config_path [/path/*.yaml] --export_ckpt_file [CKPT_PATH] --export_file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

### [Inference Process](#contents)

#### Usage

Before performing inference, the mindir file must be exported by export.py. Input files must be in bin format.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [LABEL_PATH] [DATA_FILE_PATH] [DATASET_FORMAT] [NEED_PREPROCESS] [TASK_TYPE] [EVAL_JSON_PATH] [DEVICE_ID]

# An example of task classifier (MNLI dataset):
bash run_infer_310.sh /your/path/mnli.mindir /your/path/mnli_dev.mindrecord y mnli 2
```

`NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.

`EVAL_JSON_PATH` is original eval dataset of task squadv1, must be provided if task type is squadv1.

`DEVICE_ID` is optional, it can be set by environment variable device_id, otherwise the value is zero.

#### result

Inference result is saved in current path, you can find result in acc.log file.

```eval log
acc_num 8096 , total_num 9815, accuracy 0.824860
```

## [Model Description](#contents)

## [Performance](#contents)

### Pretraining Performance

| Parameters                 | Ascend                                                     | GPU                       |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| Model Version              | BERT_base                                                  | BERT_base                 |
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8             | NV SMX2 V100-16G, cpu: Intel(R) Xeon(R) Platinum 8160 CPU @2.10GHz, memory: 256G         |
| uploaded Date              | 07/05/2021                                                 | 07/05/2021                |
| MindSpore Version          | 1.3.0                                                      | 1.3.0                     |
| Dataset                    | cn-wiki-128(4000w)                                         | cn-wiki-128(4000w)        |
| Training Parameters        | pretrain_config.yaml                                       | pretrain_config.yaml      |
| Optimizer                  | Lamb                                                       | AdamWeightDecay           |
| Loss Function              | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| outputs                    | probability                                                | probability               |
| Epoch                      | 40                                                         | 40                        |
| Batch_size                 | 256*8                                                      | 32*8                      |
| Loss                       | 1.7                                                        | 1.7                       |
| Speed                      | 284ms/step                                                 | 180ms/step                |
| Total time                 | 63H                                                        | 610H                      |
| Params (M)                 | 110M                                                       | 110M                      |
| Checkpoint for Fine tuning | 1.2G(.ckpt file)                                           | 1.2G(.ckpt file)          |
| Scripts                    | [BERT_base](https://gitee.com/mindspore/models/tree/master/official/nlp/bert)  | [BERT_base](https://gitee.com/mindspore/models/tree/master/official/nlp/bert)     |

| Parameters                 | Ascend                                                     |
| -------------------------- | ---------------------------------------------------------- |
| Model Version              | BERT_NEZHA                                                 |
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8              |
| uploaded Date              | 07/05/2021                                                 |
| MindSpore Version          | 1.3.0                                                      |
| Dataset                    | cn-wiki-128(4000w)                                         |
| Training Parameters        | src/config.py                                              |
| Optimizer                  | Lamb                                                       |
| Loss Function              | SoftmaxCrossEntropy                                        |
| outputs                    | probability                                                |
| Epoch                      | 40                                                         |
| Batch_size                 | 96*8                                                       |
| Loss                       | 1.7                                                        |
| Speed                      | 320ms/step                                                 |
| Total time                 | 180h                                                       |
| Params (M)                 | 340M                                                       |
| Checkpoint for Fine tuning | 3.2G(.ckpt file)                                           |
| Scripts                    | [BERT_NEZHA](https://gitee.com/mindspore/models/tree/master/official/nlp/bert)  |

#### Inference Performance

| Parameters                 | Ascend                        |
| -------------------------- | ----------------------------- |
| Model Version              |                               |
| Resource                   | Ascend 910; OS Euler2.8                     |
| uploaded Date              | 07/05/2021                    |
| MindSpore Version          | 1.3.0                         |
| Dataset                    | cola, 1.2W                    |
| batch_size                 | 32(1P)                        |
| Accuracy                   | 0.588986                      |
| Speed                      | 59.25ms/step                  |
| Total time                 | 15min                         |
| Model for inference        | 1.2G(.ckpt file)              |

# [Description of Random Situation](#contents)

In run_standalone_pretrain.sh and run_distributed_pretrain.sh, we set do_shuffle to True to shuffle the dataset by default.

In run_classifier.sh, and run_squad.sh, we set train_data_shuffle and eval_data_shuffle to True to shuffle the dataset by default.

In pretrain_config.yaml, task_squad_config.yaml, and task_classifier_config.yaml, we set the hidden_dropout_prob and attention_pros_dropout_prob to 0.1 to dropout some network node by default.

In run_pretrain.py, we set a random seed to make sure that each node has the same initial weight in distribute training.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).