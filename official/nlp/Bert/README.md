# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [BERT Description](#bert-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Pretrained models](#pretrained-models)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Pre-Training](#pre-training)
        - [Fine-Tuning and Evaluation](#fine-tuning-and-evaluation)
    - [Options and Parameters](#options-and-parameters)
        - [Options](#options)
        - [Parameters](#parameters)
        - [schema\_file](#schema_file)
    - [Training Process](#training-process)
        - [Training](#training)
            - [Running on Ascend](#running-on-ascend)
            - [running on GPU](#running-on-gpu)
        - [Distributed Training](#distributed-training)
            - [Running on Ascend](#running-on-ascend-1)
            - [running on GPU](#running-on-gpu-1)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
            - [evaluation on tnews dataset when running on Ascend](#evaluation-on-tnews-dataset-when-running-on-ascend)
            - [evaluation on tnews dataset when running on CPU](#evaluation-on-tnews-dataset-when-running-on-cpu)
            - [evaluation on cluener dataset when running on Ascend](#evaluation-on-cluener-dataset-when-running-on-ascend)
            - [evaluation on cluener dataset when running on CPU](#evaluation-on-cluener-dataset-when-running-on-cpu)
            - [evaluation on chineseNer dataset when running on Ascend](#evaluation-on-chinesener-dataset-when-running-on-ascend)
            - [evaluation on chineseNer dataset when running on CPU](#evaluation-on-chinesener-dataset-when-running-on-cpu)
            - [evaluation on msra dataset when running on Ascend](#evaluation-on-msra-dataset-when-running-on-ascend)
            - [evaluation on squad v1.1 dataset when running on Ascend](#evaluation-on-squad-v11-dataset-when-running-on-ascend)
            - [evaluation on squad v1.1 dataset when running on CPU](#evaluation-on-squad-v11-dataset-when-running-on-cpu)
        - [Export MindIR](#export-mindir)
        - [Inference Process](#inference-process)
            - [Usage](#usage)
            - [result](#result)
        - [Export ONNX model and inference](#export-onnx-model-and-inference)
    - [Model Description](#model-description)
    - [Performance](#performance)
        - [Pretraining Performance](#pretraining-performance)
            - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)
- [FAQ](#faq)

# [BERT Description](#contents)

The BERT network was proposed by Google in 2018. The network has made a breakthrough in the field of NLP. The network uses pre-training to achieve a large network structure without modifying, and only by adding an output layer to achieve multiple text-based tasks in fine-tuning. The backbone code of BERT adopts the Encoder structure of Transformer. The attention mechanism is introduced to enable the output layer to capture high-latitude global semantic information. The pre-training uses denoising and self-encoding tasks, namely MLM(Masked Language Model) and NSP(Next Sentence Prediction). No need to label data, pre-training can be performed on massive text data, and only a small amount of data to fine-tuning downstream tasks to obtain good results. The pre-training plus fune-tuning mode created by BERT is widely adopted by subsequent NLP networks.

[Paper](https://arxiv.org/abs/1810.04805):  Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding]((https://arxiv.org/abs/1810.04805)). arXiv preprint arXiv:1810.04805.

[Paper](https://arxiv.org/abs/1909.00204):  Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen, Qun Liu. [NEZHA: Neural Contextualized Representation for Chinese Language Understanding](https://arxiv.org/abs/1909.00204). arXiv preprint arXiv:1909.00204.

# [Model Architecture](#contents)

The backbone structure of BERT is transformer. For BERT_base, the transformer contains 12 encoder modules, each module contains one self-attention module and each self-attention module contains one attention module. For BERT_NEZHA, the transformer contains 24 encoder modules, each module contains one self-attention module and each self-attention module contains one attention module. The difference between BERT_base and BERT_NEZHA is that BERT_base uses absolute position encoding to produce position embedding vector and BERT_NEZHA uses relative position encoding.

# [Dataset](#contents)

- Create pre-training dataset
    - Download the [zhwiki](https://dumps.wikimedia.org/zhwiki/) or [enwiki](https://dumps.wikimedia.org/enwiki/) dataset for pre-training.
    - Extract and refine texts in the dataset with [WikiExtractor](https://github.com/attardi/wikiextractor). The commands are as follows:
        - pip install wikiextractor
        - python -m wikiextractor.WikiExtractor -o <output file path> -b <output file size> <Wikipedia dump file>
    - Extracted text data from `WikiExtractor` cannot be trained directly, you have to preprocess the data and convert the dataset to TFRecord format. Please refer to create_pretraining_data.py file in [BERT](https://github.com/google-research/bert#pre-training-with-bert) repository and download vocab.txt here, if AttributeError: module 'tokenization' has no attribute 'FullTokenizer' occur, please install bert-tensorflow.
- Create fine-tune dataset
    - Download dataset for fine-tuning and evaluation such as Chinese Named Entity Recognition[CLUENER](https://github.com/CLUEbenchmark/CLUENER2020), Chinese sentences classification[TNEWS](https://github.com/CLUEbenchmark/CLUE), Chinese Named Entity Recognition[ChineseNER](https://github.com/zjy-ucas/ChineseNER), English question and answering[SQuAD v1.1 train dataset](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json), [SQuAD v1.1 eval dataset](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json), package of English sentences classification[GLUE](https://gluebenchmark.com/tasks).
    - We haven't provide the scripts to create tfrecord yet, while converting dataset files from JSON format to TFRECORD format, please refer to run_classifier.py or run_squad.py file in [BERT](https://github.com/google-research/bert) repository or the CLUE official repository [CLUE](https://github.com/CLUEbenchmark/CLUE/blob/master/baselines/models/bert/run_classifier.py) and [CLUENER](https://github.com/CLUEbenchmark/CLUENER2020/tree/master/tf_version)
- Create MindRecord dataset
    - Generate pretrain mindrecord
        - If you download original dataset, extract and refine texts in the dataset with WikiExtractor follow the instructions above, you can generate mindrecord as follows:

        ```bash
           bash ./generate_pretrain_mindrecords.sh INPUT_FILES_PATH OUTPUT_FILES_PATH VOCAB_FILE
           for example:
           bash ./generate_pretrain_mindrecords.sh /path/wiki-clean-aa /path/output/ /path/bert-base-uncased-vocab.txt
        ```

        - If you have converted dataset from JSON format to tfrecord format, you can generate mindrecord as follows:

        ```python
            python parallel_tfrecord_to_mindrecord.py --input_tfrecord_dir /path/tfrecords_path --output_mindrecord_dir /path/save_mindrecord_path
        ```

        - you can visualize the dataset as follows:

        ```python
            python vis_tfrecord_or_mindrecord.py --file_name /path/train.mindrecord --vis_option vis_mindrecord > mindrecord.txt
            `vis_option` should be in ["vis_tfrecord", "vis_mindrecord"]
            Notice, before run vis_tfrecord_or_mindrecord.py, need to install tensorflow==1.15.0
        ```

    - Generate CLUENER and ChineseNER mindrecord for ner task
        Before generate mindrecord files, you need download original dataset follow the instructions above, and download [vocab.txt](https://github.com/CLUEbenchmark/CLUENER2020/blob/master/tf_version/vocab.txt)
        - Generate CLUENER mindrecord for ner task

            ```python
                python generate_cluener_mindrecord.py --data_dir /path/ClueNER/cluener_public/ --vocab_file /path/vocab.txt --output_dir /path/ClueNER/
            ```

        - Generate ChineseNER mindrecord for ner task

            ```python
                python generate_chinese_mindrecord.py --data_dir /path/ChineseNER/data/ --vocab_file /path/vocab.txt --output_dir /path/ChineseNER/
            ```

        - Generate ChineseNER mindrecord for ner task on CPU

            ```python
                python generate_chinese_mindrecord.py --data_dir /path/ChineseNER/data/ --vocab_file /path/vocab.txt --output_dir /path/ChineseNER/ --max_seq_length 128
            ```

    - Generate SQuAD v1.1 mindrecord for squad task
        Before generate mindrecord files, you need download original dataset follow the instructions above, and download [vocab.txt](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/blob/master/pretrained_model/uncased_L-12_H-768_A-12/vocab.txt)
        - Generate SQuAD v1.1 mindrecord for squad task

            ```python
                python generate_squad_mindrecord.py --vocab_file /path/squad/vocab.txt --train_file /path/squad/train-v1.1.json --predict_file /path/squad/dev-v1.1.json --output_dir /path/squad
            ```

    - Generate tnews mindrecord for classifier task
        Before generate mindrecord files, you need download original dataset follow the instructions above, and download [vocab.txt](https://github.com/CLUEbenchmark/CLUENER2020/blob/master/tf_version/vocab.txt)
        - Generate tnews mindrecord for classifier task

            ```python
                python generate_tnews_mindrecord.py --data_dir /path/tnews/ --task_name tnews --vocab_file /path/tnews/vocab.txt --output_dir /path/tnews
            ```

# [Pretrained models](#contents)

We have provided several kinds of pretrained checkpoint.

- [Bert-base-zh](https://download.mindspore.cn/models/r1.9/bert_base_ascend_v190_zhwiki_official_nlp_bs256_acc91.72_recall95.06_F1score93.36.ckpt), trained on zh-wiki datasets with 128 length.
- [Bert-large-zh](https://download.mindspore.cn/model_zoo/r1.3/bert_large_ascend_v130_zhwiki_official_nlp_bs3072_loss0.8/), trained on zh-wiki datasets with 128 length.
- [Bert-large-en](https://download.mindspore.cn/model_zoo/r1.3/bert_large_ascend_v130_enwiki_official_nlp_bs768_loss1.1/), tarined on en-wiki datasets with 512 length.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend/GPU/CPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start pre-training, fine-tuning and evaluation as follows:

- Running on Ascend

```bash
# run standalone pre-training example
bash scripts/run_standalone_pretrain_ascend.sh 0 1 /path/cn-wiki-128

# run distributed pre-training example
bash scripts/run_distributed_pretrain_ascend.sh /path/cn-wiki-128 /path/hccl.json

# run the evaluation for pre-training example
# Modify the `eval_ckpt` and `eval_data_dir` in pretrain_config.yaml
bash scripts/run_pretrain_eval_ascend.sh

# run fine-tuning and evaluation example
- If you are going to run a fine-tuning task, please prepare a checkpoint generated from pre-training.
- Set bert network config and optimizer hyperparameters in `task_[DOWNSTREAM_TASK]_config.yaml`.

- Classification task: Set task related hyperparameters in scripts/run_classifier.sh.
- Run `bash scripts/run_classifier.sh [DEVICE_ID]` for fine-tuning of BERT-base and BERT-NEZHA model.

  bash scripts/run_classifier.sh DEVICE_ID(optional)

- NER task: Set task related hyperparameters in scripts/run_ner.sh.
- Run `bash scripts/run_ner.sh [DEVICE_ID]` for fine-tuning of BERT-base and BERT-NEZHA model.

  bash scripts/run_ner.sh DEVICE_ID(optional)

- SQuAD task: Set task related hyperparameters in scripts/run_squad.sh.
- Run `bash scripts/run_squad.sh [DEVICE_ID]` for fine-tuning of BERT-base and BERT-NEZHA model.

  bash scripts/run_squad.sh DEVICE_ID(optional)
```

- Running on GPU

```bash
# run standalone pre-training example
bash run_standalone_pretrain_for_gpu.sh 0 1 /path/cn-wiki-128

# run distributed pre-training example
bash scripts/run_distributed_pretrain_for_gpu.sh 8 40 /path/cn-wiki-128

# run fine-tuning and evaluation example
- If you are going to run a fine-tuning task, please prepare a checkpoint generated from pre-training.
- Set bert network config and optimizer hyperparameters in `task_[DOWNSTREAM_TASK]_config.yaml`.

- Classification task: Set task related hyperparameters in scripts/run_classifier_gpu.sh.
- Run `bash scripts/run_classifier_gpu.sh [DEVICE_ID]` for fine-tuning of BERT-base and BERT-NEZHA model.

  bash scripts/run_classifier_gpu.sh DEVICE_ID(optional)

- NER task: Set task related hyperparameters in scripts/run_ner_gpu.sh.
- Run `bash scripts/run_ner_gpu.sh [DEVICE_ID]` for fine-tuning of BERT-base and BERT-NEZHA model.

  bash scripts/run_ner_gpu.sh DEVICE_ID(optional)

- SQuAD task: Set task related hyperparameters in scripts/run_squad_gpu.sh.
- Run `bash scripts/run_squad_gpu.py [DEVICE_ID]` for fine-tuning of BERT-base and BERT-NEZHA model.

  bash scripts/run_squad_gpu.sh DEVICE_ID(optional)
```

- running on ModelArts

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows

- Pretraining with 8 cards on ModelArts

  ```python
  # (1) Upload the code folder to S3 bucket.
  # (2) Click to "create training task" on the website UI interface.
  # (3) Set the code directory to "/{path}/bert" on the website UI interface.
  # (4) Set the startup file to /{path}/bert/train.py" on the website UI interface.
  # (5) Perform a or b.
  #     a. setting parameters in /{path}/bert/pretrain_config.yaml.
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
  # (2)  Click to "create training task" on the website UI interface.
  # (3) Set the code directory to "/{path}/bert" on the website UI interface.
  # (4) Set the startup file to /{path}/bert/run_ner.py"(or run_pretrain.py or run_squad.py) on the website UI interface.
  # (5) Perform a or b.
  #     a. setting parameters in task_ner_config.yaml(or task_squad_config.yaml or task_classifier_config.yaml under the folder `/{path}/bert/`
  #         1. Set ”enable_modelarts=True“
  #         2. Set other parameters, other parameter configuration can refer to `run_ner.sh`(or run_squad.sh or run_classifier.sh) under the folder '{path}/bert/scripts/'.
  #     b. adding on the website UI interface.
  #         1. Add ”enable_modelarts=True“
  #         2. Set other parameters, other parameter configuration can refer to `run_ner.sh`(or run_squad.sh or run_classifier.sh) under the folder '{path}/bert/scripts/'.
  #     Note that vocab_file_path, label_file_path, train_data_file_path, eval_data_file_path, schema_file_path fill in the relative path relative to the path selected in step 7.
  #     Finally, "config_path=/path/*.yaml" must be added on the web page (select the *.yaml configuration file according to the downstream task)
  # (6) Upload the dataset to S3 bucket.
  # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (there is only data or zip package under this path).
  # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
  # (9) Under the item "resource pool selection", select the specification of a single card.
  # (10) Create your job.
  # After training, the '*.ckpt' file will be saved under the'training output file path'
  ```

For distributed training on Ascend, an hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link below to create an hccl.json file in need:
[https://gitee.com/mindspore/models/tree/master/utils/hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

For distributed training among multiple machines, training command should be executed on each machine in a small time interval. Thus, an hccl.json is needed on each machine. [merge_hccl](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools#merge_hccl) is a tool to create hccl.json for multi-machine case.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─bert
  ├─ascend310_infer
  ├─README.md
  ├─README_CN.md
  ├─scripts
    ├─ascend_distributed_launcher
        ├─__init__.py
        ├─hyper_parameter_config.ini          # hyper parameter for distributed pretraining
        ├─get_distribute_pretrain_cmd.py          # script for distributed pretraining
        ├─README.md
    ├─run_classifier.sh                       # shell script for standalone classifier task on ascend
    ├─run_classifier_gpu.sh                   # shell script for standalone classifier task on gpu
    ├─run_ner.sh                              # shell script for standalone NER task on ascend
    ├─run_ner_gpu.sh                          # shell script for standalone SQUAD task on gpu
    ├─run_squad.sh                            # shell script for standalone SQUAD task on ascend
    ├─run_squad_gpu.sh                        # shell script for standalone SQUAD task on gpu
    ├─run_standalone_pretrain_ascend.sh       # shell script for standalone pretrain on ascend
    ├─run_distributed_pretrain_ascend.sh      # shell script for distributed pretrain on ascend
    ├─run_distributed_pretrain_gpu.sh         # shell script for distributed pretrain on gpu
    └─run_standaloned_pretrain_gpu.sh         # shell script for distributed pretrain on gpu
  ├─src
    ├─generate_mindrecord
      ├── generate_chinesener_mindrecord.py   # generate mindrecord for ChineseNER dataset for ner task
      ├── generate_cluener_mindrecord.py      # generate mindrecord for CLUENER dataset for ner task
      ├── generate_pretrain_mindrecord.py     # generate mindrecord for pretrain dataset for pretrain
      ├── generate_pretrain_mindrecords.sh    # generate mindrecord for pretrain dataset parallel
      ├── generate_squad_mindrecord.py        # generate mindrecord for SquadV1.1 dataset for squad task
      └── generate_tnews_mindrecord.py        # generate mindrecord for tnews dataset for classifier task
    ├─model_utils
      ├── config.py                           # parse *.yaml parameter configuration file
      ├── devcie_adapter.py                   # distinguish local/ModelArts training
      ├── local_adapter.py                    # get related environment variables in local training
      └── moxing_adapter.py                   # get related environment variables in ModelArts training
    ├─tools
      ├── parallel_tfrecord_to_mindrecord.py  # multi pool for converting tfrecord to mindrecord
      └── vis_tfrecord_or_mindrecord.py       # visualize tfrecord or mindrecord dataset
    ├─__init__.py
    ├─assessment_method.py                    # assessment method for evaluation
    ├─bert_for_finetune.py                    # backbone code of network
    ├─bert_for_finetune_cpu.py                # backbone code of network
    ├─bert_for_pre_training.py                # backbone code of network
    ├─bert_model.py                           # backbone code of network
    ├─finetune_data_preprocess.py             # data preprocessing
    ├─cluner_evaluation.py                    # evaluation for cluner
    ├─cluner_evaluation_cpu.py                # evaluation for cluner on cpu
    ├─CRF.py                                  # assessment method for clue dataset
    ├─dataset.py                              # data preprocessing
    ├─finetune_eval_model.py                  # backbone code of network
    ├─sample_process.py                       # sample processing
    ├─utils.py                                # util function
  ├─pretrain_config.yaml                      # parameter configuration for pretrain
  ├─task_ner_config.yaml                      # parameter configuration for downstream_task_ner
  ├─task_ner_cpu_config.yaml                  # parameter configuration for downstream_task_ner on cpu
  ├─task_classifier_config.yaml               # parameter configuration for downstream_task_classifier
  ├─task_classifier_cpu_config.yaml           # parameter configuration for downstream_task_classifier on cpu
  ├─task_squad_config.yaml                    # parameter configuration for downstream_task_squad
  ├─pretrain_eval.py                          # train and eval net  
  ├─quick_start.py                            # quick start  
  ├─run_classifier.py                         # finetune and eval net for classifier task
  ├─run_ner.py                                # finetune and eval net for ner task
  ├─run_pretrain.py                           # train net for pretraining phase
  └─run_squad.py                              # finetune and eval net for squad task
```

## [Script Parameters](#contents)

### Pre-Training

```text
usage: run_pretrain.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                        [--enable_save_ckpt ENABLE_SAVE_CKPT] [--device_target DEVICE_TARGET]
                        [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                        [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                        [--accumulation_steps N]
                        [--allreduce_post_accumulation ALLREDUCE_POST_ACCUMULATION]
                        [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                        [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                        [--save_checkpoint_steps N] [--save_checkpoint_num N]
                        [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR] [train_steps N]

options:
    --device_target                device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --distribute                   pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --epoch_size                   epoch size: N, default is 1
    --device_num                   number of used devices: N, default is 1
    --device_id                    device id: N, default is 0
    --enable_save_ckpt             enable save checkpoint: "true" | "false", default is "true"
    --enable_lossscale             enable lossscale: "true" | "false", default is "true"
    --do_shuffle                   enable shuffle: "true" | "false", default is "true"
    --enable_data_sink             enable data sink: "true" | "false", default is "true"
    --data_sink_steps              set data sink steps: N, default is 1
    --accumulation_steps           accumulate gradients N times before weight update: N, default is 1
    --allreduce_post_accumulation  allreduce after accumulation of N steps or after each step: "true" | "false", default is "true"
    --save_checkpoint_path         path to save checkpoint files: PATH, default is ""
    --load_checkpoint_path         path to load checkpoint files: PATH, default is ""
    --save_checkpoint_steps        steps for saving checkpoint files: N, default is 1000
    --save_checkpoint_num          number for saving checkpoint files: N, default is 1
    --train_steps                  Training Steps: N, default is -1
    --data_dir                     path to dataset directory: PATH, default is ""
    --dataset_format               dataset format, support mindrecord or tfrecord, default is mindrecord
    --schema_dir                   path to schema.json file, PATH, default is ""
```

### Fine-Tuning and Evaluation

```text
usage: run_ner.py   [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--assessment_method ASSESSMENT_METHOD] [--use_crf USE_CRF] [--with_lstm WITH_LSTM]
                    [--device_id N] [--epoch_num N] [--vocab_file_path VOCAB_FILE_PATH]
                    [--label2id_file_path LABEL2ID_FILE_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --do_train                        whether to run training on training set: true | false
    --do_eval                         whether to run eval on dev set: true | false
    --assessment_method               assessment method to do evaluation: f1 | clue_benchmark
    --use_crf                         whether to use crf to calculate loss: true | false
    --with_lstm                       Whether to use LSTM subnet after the Bert network: true | false
    --device_id                       device id to run task
    --epoch_num                       total number of training epochs to perform
    --train_data_shuffle              Enable train data shuffle, default is true
    --eval_data_shuffle               Enable eval data shuffle, default is true
    --vocab_file_path                 the vocabulary file that the BERT model was trained on
    --label2id_file_path              label to id file, each label name must be consistent with the type name labeled in the original dataset file
    --save_finetune_checkpoint_path   path to save generated finetuning checkpoint
    --load_pretrain_checkpoint_path   initial checkpoint (usually from a pre-trained BERT model)
    --load_finetune_checkpoint_path   give a finetuning checkpoint path if only do eval
    --train_data_file_path            ner tfrecord for training. E.g., train.tfrecord
    --eval_data_file_path             ner tfrecord for predictions if f1 is used to evaluate result, ner json for predictions if clue_benchmark is used to evaluate result
    --dataset_format                  dataset format, support mindrecord or tfrecord, default is mindrecord
    --schema_file_path                path to datafile schema file

usage: run_squad.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--device_id N] [--epoch_num N] [--num_class N]
                    [--vocab_file_path VOCAB_FILE_PATH]
                    [--eval_json_path EVAL_JSON_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
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
    --eval_json_path                  path to squad dev json file
    --save_finetune_checkpoint_path   path to save generated finetuning checkpoint
    --load_pretrain_checkpoint_path   initial checkpoint (usually from a pre-trained BERT model)
    --load_finetune_checkpoint_path   give a finetuning checkpoint path if only do eval
    --train_data_file_path            squad tfrecord for training. E.g., train1.1.tfrecord
    --eval_data_file_path             squad tfrecord for predictions. E.g., dev1.1.tfrecord
    --dataset_format                  dataset format, support mindrecord or tfrecord, default is mindrecord
    --schema_file_path                path to datafile schema file

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
    --load_pretrain_checkpoint_path   initial checkpoint (usually from a pre-trained BERT model)
    --load_finetune_checkpoint_path   give a finetuning checkpoint path if only do eval
    --train_data_file_path            tfrecord for training. E.g., train.tfrecord
    --eval_data_file_path             tfrecord for predictions. E.g., dev.tfrecord
    --dataset_format                  dataset format, support mindrecord or tfrecord, default is mindrecord
    --schema_file_path                path to datafile schema file
```

## Options and Parameters

Parameters for training and downstream task can be set in yaml config file respectively.

### Options

```text
config for lossscale and etc.
    bert_network                    version of BERT model: base | nezha, default is base
    batch_size                      batch size of input dataset: N, default is 32
    loss_scale_value                initial value of loss scale: N, default is 2^32
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 1000
    optimizer                       optimizer used in the network: AdamWerigtDecayDynamicLR | Lamb | Momentum, default is "Lamb"
```

### Parameters

```text
Parameters for dataset and network (Pre-Training/Fine-Tuning/Evaluation):
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, must be consistent with the dataset you use. Default is 21128. Usually, we use 21128 for CN vocabs and 30522 for EN vocabs according to the origin paper.
    hidden_size                     size of bert encoder layers: N, default is 768
    num_hidden_layers               number of hidden layers: N, default is 12
    num_attention_heads             number of attention heads: N, default is 12
    intermediate_size               size of intermediate layer: N, default is 3072
    hidden_act                      activation function used: ACTIVATION, default is "gelu"
    hidden_dropout_prob             dropout probability for BertOutput: Q, default is 0.1
    attention_probs_dropout_prob    dropout probability for BertAttention: Q, default is 0.1
    max_position_embeddings         maximum length of sequences: N, default is 512
    type_vocab_size                 size of token type vocab: N, default is 16
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    use_relative_positions          use relative positions or not: True | False, default is False
    dtype                           data type of input: mstype.float16 | mstype.float32, default is mstype.float32
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float16

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

### schema_file

When using TFRecord as training or evaluation data, you need to use `schema_file` data, which needs to be generated by yourself, and its specific meaning is as follows:

For pre-training, the schema file contains `["input_ids", "input_mask", "segment_ids", "next_sentence_labels", "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"]`.

For ner or classification tasks, the schema file contains `["input_ids", "input_mask", "segment_ids", "label_ids"]`.

For the squad task, the schema file used for training contains `["start_positions", "end_positions", "input_ids", "input_mask", "segment_ids"]`, and the schema file used for evaluation contains `["input_ids", "input_mask" ", "segment_ids"]`.

When dataset_format is tfrecord, `numRows` is the only option in the schema file that can be set by the user, other values ​​must be set according to the dataset.

When dataset_format is mindrecord, `num_samlpes` is the only user-settable option in the yaml file, other values ​​must be set according to the dataset.

For example, the schema file for the pre-trained cn-wiki-128 dataset is shown below:

```json
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

Alternatively, the schema file for the SQuAD dataset is shown below:

```json
{
  "datasetType": "TF",
  "numRows": 1000000,
  "columns": {
    "input_ids": {
      "type": "int64",
      "rank": 1,
      "shape": [384]
    },
    "input_mask": {
      "type": "int64",
      "rank": 1,
      "shape": [384]
    },
    "segment_ids" : {
      "type": "int64",
      "rank": 1,
      "shape": [384]
    },
    "unique_ids": {
      "type": "int64",
      "rank": 1,
      "shape": [1]
    },
    "start_positions": {
      "type": "int64",
      "rank": 1,
      "shape": [1]
    },
    "end_positions": {
      "type": "int64",
      "rank": 1,
      "shape": [1]
    },
    "is_impossible": {
      "type": "int64",
      "rank": 1,
      "shape": [1]
    }
  }
}
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

#### running on GPU

```bash
bash scripts/run_standalone_pretrain_for_gpu.sh 0 1 /path/cn-wiki-128
```

The command above will run in the background, you can view the results the file pretraining_log.txt. After training, you will get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```bash
# grep "epoch" pretraining_log.txt
epoch: 0.0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0856101e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.000, step: 2, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0821701e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **Attention** If you are running with a huge dataset on Ascend, it's better to add an external environ variable to make sure the hccl won't timeout.
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

Before distribute pretrain on ascend, you need to generate distributed_cmd.sh as follows:

```python
python scripts/ascend_distributed_launcher/get_distribute_pretrain_cmd.py --run_script_dir ./scripts/run_distributed_pretrain_ascend.sh --hyper_parameter_config_dir ./scripts/ascend_distributed_launcher/hyper_parameter_config.ini --data_dir /path/data_dir/ --hccl_config /path/hccl.json --cmd_file ./distributed_cmd.sh
```

```bash
bash scripts/run_distributed_pretrain_ascend.sh /path/cn-wiki-128 /path/hccl.json
```

The command above will run in the background, you can view training logs in pretraining_log.txt. After training finished, you will get some checkpoint files under the LOG* folder by default. The loss value will be displayed as follows:

```bash
# grep "epoch" LOG*/pretraining_log.txt
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08209e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07566e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08218e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07770e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

#### running on GPU

```bash
bash scripts/run_distributed_pretrain_for_gpu.sh /path/cn-wiki-128
```

The command above will run in the background, you can view the results the file pretraining_log.txt. After training, you will get some checkpoint files under the LOG* folder by default. The loss value will be achieved as follows:

```bash
# grep "epoch" LOG*/pretraining_log.txt
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08209e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07566e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08218e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07770e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **Attention** This will bind the processor cores according to the `device_num` and total processor numbers. If you don't expect to run pretraining with binding processor cores, remove the operations about `taskset` in `scripts/ascend_distributed_launcher/get_distribute_pretrain_cmd.py`

## [Evaluation Process](#contents)

### Evaluation

> **Attention** If the evaluate dataset used is mindrecord/tfrecord format, the 'dataset_format' in *yaml file need to change to 'mindrecord'/'tfrecord'. Parameters of 'train_data_file_path' and 'eval_data_file_path' need to modify, and no need to set 'schema_file_path' in *sh files

#### evaluation on tnews dataset when running on Ascend

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_bert-20000_1.ckpt" \

--train_data_file_path="/data/tnews/train.tf_record" \

--eval_data_file_path="/data/tnews/dev.tf_record" \

--schema_file_path="/data/tnews/dataset.json"

```bash
bash scripts/run_classifier.sh
```

The command above will run in the background, you can view training logs in classfier_log.txt.

If you choose accuracy as assessment method, the result will be as follows(accuracy: 0.55~0.56):

```text
acc_num XXX, total_num XXX, accuracy 0.555500
```

#### evaluation on tnews dataset when running on CPU

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_bert-20000_1.ckpt" \

--train_data_file_path="/data/tnews/train.mindrecord" \

--eval_data_file_path="/data/tnews/dev.mindrecord"

```bash
python run_classifier.py --config_path=../../task_classifier_cpu_config.yaml --device_target CPU --do_train=true --do_eval=true --num_class=15 --train_data_file_path="" --eval_data_file_path="" --load_pretrain_checkpoint_path=""
```

The command above will run in the background, you can view training logs in classfier_log.txt.

If you choose accuracy as assessment method, the result will be as follows(accuracy: 0.55~0.56)：:

```text
acc_num XXX, total_num XXX, accuracy 0.554200
```

#### evaluation on cluener dataset when running on Ascend

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:

--label_file_path="/data/cluener/label_file" \

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_bert-20000_1.ckpt" \

--train_data_file_path="/data/cluener/train.tf_record" \

--eval_data_file_path="/data/cluener/dev.tf_record" \

--schema_file_path="/data/cluener/dataset.json"

```bash
bash scripts/run_ner.sh
```

The command above will run in the background, you can view training logs in ner_log.txt.

If you choose F1 as assessment method, the result will be as follows:

```text
Precision 0.868245
Recall 0.865611
F1 0.866926
```

#### evaluation on cluener dataset when running on CPU

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:

--label_file_path="/data/cluener/label_file" \

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_bert-20000_1.ckpt" \

--train_data_file_path="/data/cluener/train.mindrecord" \

--eval_data_file_path="/data/cluener/dev.mindrecord"

```bash
python run_ner.py --config_path=../../task_ner_cpu_config.yaml --device_target CPU --do_train=true --do_eval=true --assessment_method=Accuracy --use_crf=false --with_lstm=false --label_file_path="" --train_data_file_path="" --eval_data_file_path="" --load_pretrain_checkpoint_path=""
```

The command above will run in the background, you can view training logs in ner_log.txt.

If you choose accuracy as assessment method, the result will be as follows:

```text
acc_num XXX, total_num XXX, accuracy 0.916855
```

#### evaluation on chineseNer dataset when running on Ascend

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:

--label_file_path="/data/chineseNer/label_file" \

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_bert-20000_1.ckpt" \

--train_data_file_path="/data/chineseNer/train.tf_record" \

--eval_data_file_path="/data/chineseNer/dev.tf_record" \

--schema_file_path="/data/chineseNer/dataset.json"

```bash
bash scripts/run_ner.sh
```

The command above will run in the background, you can view training logs in ner_log.txt.

If you choose F1 as assessment method, the result will be as follows:

```text
F1 0.986526
```

#### evaluation on chineseNer dataset when running on CPU

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:

--label_file_path="/data/chineseNer/label_file" \

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_bert-20000_1.ckpt" \

--train_data_file_path="/data/chineseNer/train.mindrecord" \

--eval_data_file_path="/data/chineseNer/dev.mindrecord"

```bash
python run_ner.py --config_path=../../task_ner_cpu_config.yaml --device_target CPU --do_train=true --do_eval=true --assessment_method=BF1 --use_crf=true --with_lstm=true --label_file_path="" --train_data_file_path="" --eval_data_file_path="" --load_pretrain_checkpoint_path=""
```

The command above will run in the background, you can view training logs in ner_log.txt.

If you choose BF1 as assessment method, the result will be as follows:

```text
Precision 0.983121
Recall 0.978546
F1 0.980828
```

#### evaluation on msra dataset when running on Ascend

For preprocess, you can first convert the original txt format of MSRA dataset into mindrecord by run the command as below (please keep in mind that the label names in label2id_file should be consistent with the type names labeled in the original msra_dataset.xml dataset file):

```python
python src/finetune_data_preprocess.py --data_dir=/path/msra_dataset.xml --vocab_file=/path/vacab_file --save_path=/path/msra_dataset.mindrecord --label2id=/path/label2id_file --max_seq_len=seq_len --class_filter="NAMEX" --split_begin=0.0 --split_end=1.0
```

For finetune and evaluation, just do

```bash
bash scripts/run_ner.sh DEVICE_ID(optional)
```

The command above will run in the background, you can view training logs in ner_log.txt.

If you choose MF1(F1 score with multi-labels) as assessment method, the result will be as follows if evaluation is done after finetuning 10 epoches:

```text
F1 0.931243
```

#### evaluation on squad v1.1 dataset when running on Ascend

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:

--vocab_file_path="/data/squad/vocab_bert_large_en.txt" \

--load_pretrain_checkpoint_path="/data/scripts/bert_converted.ckpt" \

--train_data_file_path="/data/squad/train.tf_record" \

--eval_json_path="/data/squad/dev-v1.1.json" \

```bash
bash scripts/squad.sh
```

The command above will run in the background, you can view training logs in squad_log.txt.
The result will be as follows:

```text
{"exact_match": 80.3878923040233284, "f1": 87.6902384023850329}
```

#### evaluation on squad v1.1 dataset when running on CPU

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:

--vocab_file_path="/data/squad/vocab_bert_large_en.txt" \

--load_pretrain_checkpoint_path="/data/scripts/bert_converted.ckpt" \

--train_data_file_path="/data/squad/train.mindrecord" \

--eval_json_path="/data/squad/dev-v1.1.json" \

```bash
python run_squad.py --config_path=../../task_squad_config.yaml --device_target CPU --do_train=true --do_eval=true --vocab_file_path="" --train_data_file_path="" --eval_json_path="" --dataset_format tfrecord --load_pretrain_checkpoint_path=""
```

The command above will run in the background, you can view training logs in squad_log.txt.
The result will be as follows:

```text
{"exact_match": 79.62157048249763, "f1": 87.24089125977054}
```

### [Export MindIR](#contents)

- Export on local

We only support export with fine-tuned downstream task model and yaml config file, because the pretrained model is useless in inferences task.

```shell
python export.py --config_path [/path/*.yaml] --export_ckpt_file [CKPT_PATH] --export_file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# (1) Upload the code folder to S3 bucket.
# (2) Click to "create training task" on the website UI interface.
# (3) Set the code directory to "/{path}/bert" on the website UI interface.
# (4) Set the startup file to /{path}/bert/export.py" on the website UI interface.
# (5) Perform a or b.
#     a. setting parameters in task_ner_config.yaml(or task_squad_config.yaml or task_classifier_config.yaml under the folder `/{path}/bert/`
#         1. Set ”enable_modelarts: True“
#         2. Set “export_ckpt_file: ./{path}/*.ckpt”('export_ckpt_file' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
#         3. Set ”export_file_name: bert_ner“
#         4. Set ”file_format：MINDIR“
#         5. Set ”label_file_path：{path}/*.txt“('label_file_path' refers to the relative path relative to the folder selected in step 7.)
#     b. adding on the website UI interface.
#         1. Add ”enable_modelarts=True“
#         2. Add “export_ckpt_file=./{path}/*.ckpt”('export_ckpt_file' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
#         3. Add ”export_file_name=bert_ner“
#         4. Add ”file_format=MINDIR“
#         5. Add ”label_file_path：{path}/*.txt“('label_file_path' refers to the relative path relative to the folder selected in step 7.)
#     Finally, "config_path=/path/*.yaml" must be added on the web page (select the *.yaml configuration file according to the downstream task)
# (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
# (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (9) Under the item "resource pool selection", select the specification of a single card.
# (10) Create your job.
# You will see bert_ner.mindir under {Output file path}.
```

The `export_ckpt_file` parameter is required, and `file_format` should be in ["AIR", "MINDIR"]

### [Inference Process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

#### Usage

Before performing inference, the mindir file must be exported by export.py. Input files must be in bin format.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [LABEL_PATH] [DATA_FILE_PATH] [DATASET_FORMAT] [SCHEMA_PATH] [TASK] [VOCAB_FILE_PATH] [EVAL_JSON_PATH] NEED_PREPROCESS] [DEVICE_TYPE] [DEVICE_ID]
```

`NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
`TASK` is mandatory, and must choose from [ner|ner_crf|classifier|sqaud].
`DEVICE_ID` is optional, default value is 0.

#### result

Inference result is saved in current path, you can find result in acc.log file.

```eval log
Classifier: Accuracy=0.5539  NER: F1=0.931243
```

### [Export ONNX model and inference](#contents)

Currently, the ONNX model of Bert classification task can be exported, and third-party tools such as ONNXRuntime can be used to load ONNX for inference.

- export ONNX

```shell
python export.py --config_path [/path/*.yaml] --file_format ["ONNX"] --export_ckpt_file [CKPT_PATH] --num_class [NUM_CLASS] --export_file_name [EXPORT_FILE_NAME]
```

'CKPT_PATH' is mandatory, it is the path of the CKPT file that has been trained for a certain classification task model.
'NUM_CLASS' is mandatory, it is the number of categories in the classification task model.
'EXPORT_FILE_NAME' is optional, it is the name of the exported ONNX model. If not set, the ONNX model will be saved in the current directory with the default name.

After running, the ONNX model of Bert will be saved in the current file directory.

- Load ONNX and inference

```shell
python run_eval_onnx.py --config_path [/path/*.yaml] --eval_data_file_path [EVAL_DATA_FILE_PATH] -export_file_name [EXPORT_FILE_NAME]
```

'EVAL_DATA_FILE_PATH' is mandatory, it is the eval data of the dataset used by the classification task.
'EXPORT_FILE_NAME' is optional, it is the model name of the ONNX in the step of export ONNX, which is used to load the specified ONNX model for inference.

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
| Scripts                    | [BERT_base](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert)  | [BERT_base](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert)     |

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
| Scripts                    | [BERT_NEZHA](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert)  |

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

In run_classifier.sh, run_ner.sh and run_squad.sh, we set train_data_shuffle and eval_data_shuffle to True to shuffle the dataset by default.

In config.py, we set the hidden_dropout_prob and attention_pros_dropout_prob to 0.1 to dropout some network node by default.

In run_pretrain.py, we set a random seed to make sure that each node has the same initial weight in distribute training.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).

# FAQ

Refer to the [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) for some common question.

- **Q: How to resolve the continually overflow?**

  **A**: Continually overflow is usually caused by using too high learning rate.
  You could try lower `learning_rate` to use lower base learning rate or higher `power` to make learning rate decrease faster in config yaml.

- **Q: Why the training process failed with error for the shape can not match?**

  **A**: This is usually caused by the config `seq_length` of model can't match the dataset. You could check and modified the `seq_length` in yaml config according to the dataset you used.
  The parameter of model won't change with `seq_length`, the shapes of parameter only depends on model config `max_position_embeddings`.

- **Q: Why the training process failed with error about operator `Gather`?**

  **A**: Bert use operator `Gather` for embedding. The size of vocab is configured by `vocab_size` in yaml config file, usually 21128 for CN vocabs, 30522 for EN vocabs. If the vocab used to construct the dataset is larger than config, the operator will failed for the violation access.

- **Q: Why the modification in yaml config file doesn't take effect?**

  **A**: Configuration is defined by both `yaml` file and `command line arguments`, additionally with the `ini` file if you are using `ascend_distributed_launcher`. The priority of these configuration is **command line arguments > ini file > yaml file**.

- **Q: How to resolve the RuntimeError by get_dataset_size error?**

  **A**: Configuration of 'dataset_format' is defined in `yaml` file. If get_dataset_size error,  you need to check whether the dataset format set in the 'sh' file is the same as the dataset format set in the yaml file. The current dataset format only supports [tfrecord, mindrecord].
