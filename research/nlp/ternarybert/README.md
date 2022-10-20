
# Contents

- [Contents](#contents)
- [TernaryBERT Description](#ternarybert-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Train](#train)
        - [Eval](#eval)
    - [Options and Parameters](#options-and-parameters)
        - [Parameters](#parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
            - [evaluation on STS-B dataset](#evaluation-on-sts-b-dataset)
    - [Model Description](#model-description)
    - [Performance](#performance)
        - [training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [TernaryBERT Description](#contents)

[TernaryBERT](https://arxiv.org/abs/2009.12812) ternarizes the weights in a fine-tuned [BERT](https://arxiv.org/abs/1810.04805) or [TinyBERT](https://arxiv.org/abs/1909.10351) model and achieves competitive performances in natural language processing tasks. TernaryBERT outperforms the other BERT quantization methods, and even achieves comparable performance as the full-precision model while being 14.9x smaller

[Paper](https://arxiv.org/abs/2009.12812): Wei Zhang, Lu Hou, Yichun Yin, Lifeng Shang, Xiao Chen, Xin Jiang and Qun Liu. [TernaryBERT: Distillation-aware Ultra-low Bit BERT](https://arxiv.org/abs/2009.12812). arXiv preprint arXiv:2009.12812.

# [Model Architecture](#contents)

The backbone structure of TernaryBERT is transformer, the transformer contains six encoder modules, one encoder contains one self-attention module and one self-attention module contains one attention module. The pretrained teacher model and student model are provided [here](https://download.mindspore.cn/model_zoo/research/nlp/ternarybert/).

# [Dataset](#contents)

- Download glue dataset for task distillation. Convert dataset files from json format to tfrecord format, please refer to data_transfer.py which in [BERT](https://gitee.com/slyang2021/bert-multi-gpu#convert-sts_b-dataset-to-tf_record) repository.

- Note that when the parameter is passed in, the directory is the upper level directory of sts-b

```text

└─data_dir
  ├─sts-b
    ├─eval.tf_record
    ├─predict.tf_record
    ├─train.tf_record

```

# [Environment Requirements](#contents)

- Hardware（Ascend or GPU）
    - Prepare hardware environment with GPU processor or Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)
- Software：
    - sklearn

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash

# run training example

bash scripts/run_standalone_train_ascend.sh [TASK_NAME] [DEVICE_TARGET] [TEACHER_MODEL_DIR] [STUDENT_MODEL_DIR] [DATA_DIR]

Before running the shell script, please set the `task_name`, `device_target`, `teacher_model_dir`, `student_model_dir` and `data_dir` in the run_standalone_train_ascend.sh file first.

# run evaluation example

bash scripts/run_standalone_eval_ascend.sh [TASK_NAME] [DEVICE_TARGET] [MODEL_DIR] [DATA_DIR]

Before running the shell script, please set the `task_name`, `device_target`, `model_dir` and `data_dir` in the run_standalone_eval_ascend.sh file first.
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text

.
└─ternarybert
  ├─ascend310_infer
  ├─README.md
  ├─scripts
    ├─run_standalone_train_ascend.sh                  # shell script for training phase
    ├─run_standalone_eval_ascend.sh                   # shell script for evaluation phase
    ├─run_infer_310.sh              # shell script for 310infer
  ├─src
    ├─__init__.py
    ├─assessment_method.py          # assessment method for evaluation
    ├─cell_wrapper.py               # cell for training
    ├─config.py                     # parameter configuration for training and evaluation phase
    ├─dataset.py                    # data processing
    ├─quant.py                      # function for quantization
    ├─tinybert_model.py             # backbone code of network
    ├─utils.py                      # util function
  ├─train.py                        # train net for task distillation
  ├─eval.py                         # evaluate net after task distillation
  ├─export.py                       # export scripts
  ├─preprocess.py                   # 310推理前处理脚本
  ├─postprocess.py                  # 310推理后处理脚本
  ├─mindspore_hub_conf.py           # Mindspore Hub接口
```

## [Script Parameters](#contents)

### Train

```text

usage: train.py    [--h] [--device_target GPU] [--do_eval {true,false}] [--epoch_size EPOCH_SIZE]
                   [--device_id DEVICE_ID] [--do_shuffle {true,false}] [--enable_data_sink {true,false}] [--save_ckpt_step SAVE_CKPT_STEP]
                   [--max_ckpt_num MAX_CKPT_NUM] [--data_sink_steps DATA_SINK_STEPS]
                   [--teacher_model_dir TEACHER_MODEL_DIR] [--student_model_dir STUDENT_MODEL_DIR] [--data_dir DATA_DIR]
                   [--output_dir OUTPUT_DIR] [--task_name {sts-b,qnli,mnli}] [--dataset_type DATASET_TYPE] [--seed SEED]
                   [--train_batch_size TRAIN_BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE]

options:
    --device_target                 Device where the code will be implemented: "Ascend"
    --do_eval                       Do eval task during training or not: "true" | "false", default is "true"
    --epoch_size                    Epoch size for train phase: N, default is 5
    --device_id                     Device id: N, default is 0
    --do_shuffle                    Enable shuffle for train dataset: "true" | "false", default is "true"
    --enable_data_sink              Enable data sink: "true" | "false", default is "true"
    --save_ckpt_step                If do_eval is false, the checkpoint will be saved every save_ckpt_step: N, default is 50
    --max_ckpt_num                  The number of checkpoints will not be larger than max_ckpt_num: N, default is 50
    --data_sink_steps               Sink steps for each epoch: N, default is 1
    --teacher_model_dir             The checkpoint directory of teacher model: PATH, default is ""
    --student_model_dir             The checkpoint directory of student model: PATH, default is ""
    --data_dir                      Data directory: PATH, default is ""
    --output_dir                    The output checkpoint directory: PATH, default is "./"
    --task_name                     The name of the task to train: "sts-b" | "qnli" | "mnli", default is "sts-b"
    --dataset_type                  The name of the task to train: "tfrecord" | "mindrecord", default is "tfrecord"
    --seed                          The random seed: N, default is 1
    --train_batch_size              Batch size for training: N, default is 16
    --eval_batch_size               Eval Batch size in callback: N, default is 32
    --file_name                     The output filename of export, default is "ternarybert"
    --file_format                   The output format of export, default is "MINDIR"
    --enable_modelarts              Do modelarts or not. (Default: False)
    --data_url                      Real input file path
    --train_url                     Real output file path include .ckpt and .air
    --modelarts_data_dir            Modelart input path
    --modelarts_result_dir          Modelart output path
    --result_dir                    Output path
```

### Eval

```text

usage: eval.py    [--h] [--device_target GPU] [--device_id DEVICE_ID] [--model_dir MODEL_DIR] [--data_dir DATA_DIR]
                  [--task_name {sts-b,qnli,mnli}] [--dataset_type DATASET_TYPE] [--batch_size BATCH_SIZE]

options:
    --device_target                 Device where the code will be implemented: "GPU"
    --device_id                     Device id: N, default is 0
    --model_dir                     The checkpoint directory of model: PATH, default is ""
    --data_dir                      Data directory: PATH, default is ""
    --task_name                     The name of the task to train: "sts-b" | "qnli" | "mnli", default is "sts-b"
    --dataset_type                  The name of the task to train: "tfrecord" | "mindrecord", default is "tfrecord"
    --batch_size                    Batch size for evaluating: N, default is 32

```

## Parameters

`config.py`contains parameters of glue tasks, train, optimizer, eval, teacher BERT model and student BERT model.

```text

Parameters for glue task:
    num_labels                      the numbers of labels: N.
    seq_length                      length of input sequence: N
    task_type                       the type of task: "classification" | "regression"
    metrics                         the eval metric for task: Accuracy | F1 | Pearsonr | Matthews

Parameters for train:
    batch_size                      batch size of input dataset: N, default is 16
    loss_scale_value                initial value of loss scale: N, default is 2^16
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 50

Parameters for optimizer:
    learning_rate                   value of learning rate: Q, default is 5e-5
    end_learning_rate               value of end learning rate: Q, must be positive, default is 1e-14
    power                           power: Q, default is 1.0
    weight_decay                    weight decay: Q, default is 1e-4
    eps                             term added to the denominator to improve numerical stability: Q, default is 1e-6
    warmup_ratio                    the ratio of warmup steps to total steps: Q, default is 0.1

Parameters for eval:
    batch_size                      batch size of input dataset: N, default is 32

Parameters for teacher bert network:
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, must be consistent with the dataset you use. Default is 30522
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
    initializer_range               initialization value of Normal: Q, default is 0.02
    use_relative_positions          use relative positions or not: True | False, default is False
    dtype                           data type of input: mstype.float16 | mstype.float32, default is mstype.float32
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float32

Parameters for student bert network:
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, must be consistent with the dataset you use. Default is 30522
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
    initializer_range               initialization value of Normal: Q, default is 0.02
    use_relative_positions          use relative positions or not: True | False, default is False
    dtype                           data type of input: mstype.float16 | mstype.float32, default is mstype.float32
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float32
    do_quant                        do activation quantilization or not: True | False, default is True
    embedding_bits                  the quant bits of embedding: N, default is 2
    weight_bits                     the quant bits of weight: N, default is 2
    cls_dropout_prob                dropout probability for BertModelCLS: Q
    activation_init                 initialization value of activation quantilization: Q, default is 2.5
    is_lgt_fit                      use label ground truth loss or not: True | False, default is False

```

## [Training Process](#contents)

### Training

Before running the command below, please check `teacher_model_dir`, `student_model_dir` and `data_dir` has been set. Please set the path to be the absolute full path, e.g:"/home/xxx/model_dir/".

```text

python
    python train.py --task_name='sts-b' --device_target="Ascend" --teacher_model_dir='/home/xxx/model_dir/' --student_model_dir='/home/xxx/model_dir/' --data_dir='/home/xxx/data_dir/'
shell
    bash scripts/run_standalone_train_ascend.sh [TASK_NAME] [DEVICE_TARGET] [DEVICE_ID] [TEACHER_MODEL_DIR] [STUDENT_MODEL_DIR] [DATA_DIR]

```

The shell command above will run in the background, you can view the results the file log.txt. The python command will run in the console, you can view the results on the interface. After training, you will get some checkpoint files under the script folder by default. The eval metric value will be achieved as follows:

```text

train dataset size: 359
eval dataset size: 47
epoch: 1 step: 359, loss is 3.5354042053222656
epoch time: 2039458.960 ms, per step time: 5680.944 ms
epoch: 2 step: 359, loss is 1.4011192321777344
epoch time: 1955723.881 ms, per step time: 5447.699 ms
epoch: 3 step: 359, loss is 1.2592418193817139
epoch time: 1955666.337 ms, per step time: 5447.539 ms
epoch: 4 step: 359, loss is 0.7391554713249207
epoch time: 1955738.087 ms, per step time: 5447.738 ms
epoch: 5 step: 359, loss is 0.5966147184371948
epoch time: 1955702.814 ms, per step time: 5447.640 ms
===========training success================
===========Done!!!!!================

```

### Training on ModelArts

Upload weight file and sts-b file (data.zip), the directory structure is as follows

```text
.
└─data
  ├─sts-b
    ├─eval.tf_record
    ├─predict.tf_record
    ├─train.tf_record
  ├─weight
    ├─student_model
        ├─sts-b
            ├─eval_model.ckpt
    ├─teacher_model
        ├─sts-b
            ├─eval_model.ckpt
```

Select startup file train.py, dataset data.zip, training parameters is as follows

```text
enable_modelarts         True
data_dir                 data
student_model_dir        data/weight/student_model
teacher_model_dir        data/weight/teacher_model
```

## [Evaluation Process](#contents)

### Evaluation

If you want to after running and continue to eval.

#### evaluation on STS-B dataset

```text

python
    python eval.py --task_name='sts-b' --device_target="Ascend" --model_dir='/home/xxx/model_dir/' --data_dir='/home/xxx/data_dir/'
shell
    bash scripts/run_standalone_eval_ascend.sh [TASK_NAME] [DEVICE_TARGET] [DEVICE_ID] [MODEL_DIR] [DATA_DIR]


```

The shell command above will run in the background, you can view the results the file log.txt. The python command will run in the console, you can view the results on the interface. The metric value of the test dataset will be as follows:

```text

eval step: 0, Pearsonr: 96.91109003302263
eval step: 1, Pearsonr: 95.6800637493701
eval step: 2, Pearsonr: 94.23823082886167
...
The best Pearsonr: 87.58388835685437

```

## [Evaluation on Ascend 310](#contents)

### Export MINDIR

```text
python export.py --task_name [TASK_NAME] --file_name [FILE_NAME] --ckpt_file [CKPT_FILE]
#example
python export.py --task_name sts-b --file_name ternarybert --ckpt_file ./output/sts-b/eval_model.ckpt
```

### Evaluation

Before performing inference, the mindir file must be exported through the export.py script. The following shows an example of using the mindir model to perform inference.

```text
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DATASET_TYPE] [TASK_NAME] [ASSESSMENT_METHOD] [NEED_PREPROCESS] [DEVICE_ID]
# example
bash run_infer_310.sh ../ternarybert.mindir ../data/sts-b/eval.tf_record tfrecord sts-b pearsonr y 0

```

## [Model Description](#contents)

## [Performance](#contents)

### training Performance

| Parameters                                                                    | Ascend                       |
| -------------------------- | ------------------------- |
| Model Version              | TernaryBERT                           |
| Resource                   | Ascend 910, ARM CPU 2.60GHz, cores 192, mem 755G, os Euler2.8         |
| Date              | 2021-6-10      |
| MindSpore Version          | 1.7.0                     |
| Dataset                    | STS-B              |
| batch_size                    | 16              |
| Metric value                 | 87.5                       |

# [Description of Random Situation](#contents)

In train.py, we set do_shuffle to shuffle the dataset.

In config.py, we set the hidden_dropout_prob, attention_pros_dropout_prob and cls_dropout_prob to dropout some network node.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
