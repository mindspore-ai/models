# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
    - [Transformer Description](#transformer-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
            - [Training Script Parameters](#training-script-parameters)
            - [Running Options](#running-options)
            - [Network Parameters](#network-parameters)
    - [Dataset Preparation](#dataset-preparation)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
        - [ONNX Evaluation](#onnx-evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer](#infer)
        - [result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)
    - [FAQ](#faq)

## [Transformer Description](#contents)

Transformer was proposed in 2017 and designed to process sequential data. It is adopted mainly in the field of natural language processing(NLP), for tasks like machine translation or text summarization. Unlike traditional recurrent neural network(RNN) which processes data in order, Transformer adopts attention mechanism and improve the parallelism, therefore reduced training times and made training on larger datasets possible. Since Transformer model was introduced, it has been used to tackle many problems in NLP and derives many network models, such as BERT(Bidirectional Encoder Representations from Transformers) and GPT(Generative Pre-trained Transformer).

[Paper](https://arxiv.org/abs/1706.03762):  Ashish Vaswani, Noam Shazeer, Niki Parmar, JakobUszkoreit, Llion Jones, Aidan N Gomez, Ł ukaszKaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS 2017, pages 5998–6008.

## [Model Architecture](#contents)

Specifically, Transformer contains six encoder modules and six decoder modules. Each encoder module consists of a self-attention layer and a feed forward layer, each decoder module consists of a self-attention layer, a encoder-decoder-attention layer and a feed forward layer.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- *WMT English-German(https://nlp.stanford.edu/projects/nmt/)* for training.
- *WMT newstest2014(https://nlp.stanford.edu/projects/nmt/)* for evaluation.

## [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After dataset preparation, you can start training and evaluation as follows:

In Ascend environment

```bash
# run training example
bash scripts/run_standalone_train.sh Ascend [DEVICE_ID] [EPOCH_SIZE] [GRADIENT_ACCUMULATE_STEP] [DATA_PATH]
# EPOCH_SIZE recommend 52, GRADIENT_ACCUMULATE_STEP recommend 8 or 1

# run distributed training example
bash scripts/run_distribute_train_ascend.sh [DEVICE_NUM] [EPOCH_SIZE] [DATA_PATH] [RANK_TABLE_FILE] [CONFIG_PATH]
# EPOCH_SIZE recommend 52

# run evaluation example
bash scripts/run_eval.sh Ascend [DEVICE_ID] [MINDRECORD_DATA] [CKPT_PATH] [CONFIG_PATH]
# CONFIG_PATH　should be consistent with training
```

In GPU environment

```bash
# run training example
bash scripts/run_standalone_train.sh GPU [DEVICE_ID] [EPOCH_SIZE] [GRADIENT_ACCUMULATE_STEP] [DATA_PATH]
# EPOCH_SIZE recommend 52, GRADIENT_ACCUMULATE_STEP recommend 8 or 1

# run distributed training example
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [DATA_PATH] [CONFIG_PATH]
# EPOCH_SIZE recommend 52

# run evaluation example
bash scripts/run_eval.sh GPU [DEVICE_ID] [MINDRECORD_DATA] [CKPT_PATH] [CONFIG_PATH]
# CONFIG_PATH　should be consistent with training
```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config_large.yaml file.
    #          Set "distribute=True" on default_config_large.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config_large.yaml file.
    #          Set "epoch_size: 52" on default_config_large.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config_large.yaml file.
    #          Set other parameters on default_config_large.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "dataset_path=/cache/data" on the website UI interface.
    #          Add "epoch_size: 52" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/transformer" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config_large.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config_large.yaml file.
    #          Set "epoch_size: 52" on default_config_large.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config_large.yaml file.
    #          Set other parameters on default_config_large.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add "epoch_size: 52" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/transformer" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config_large.yaml file.
    #          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on base_config.yaml file.
    #          Set "checkpoint='./transformer/transformer_trained.ckpt'" on default_config_large.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config_large.yaml file.
    #          Set other parameters on default_config_large.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_trained_model/'" on the website UI interface.
    #          Add "checkpoint='./transformer/transformer_trained.ckpt'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/transformer" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Export s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "file_name='transformer'" on base_config.yaml file.
    #          Set "file_format='MINDIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name='transformer'" on the website UI interface.
    #          Add "file_format='MINDIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/transformer" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└─Transformer
  ├─README_CN.md
  ├─README.md
  ├─ascend310_infer
    ├─build.sh
    ├─CMakeLists.txt
    ├─inc
    │ └─utils.h
    └─src
      ├─main.cc
      └─utils.cc
  ├─scripts
    ├─process_output.sh
    ├─replace-quote.perl
    ├─run_distribute_train_ascend_multi_machines.sh
    ├─run_distribute_train_ascend.sh
    ├─run_distribute_train_gpu.sh
    ├─run_eval_onnx.sh                                      // bash script for ONNX evaluation
    ├─run_eval.sh
    ├─run_infer_310.sh
    └─run_standalone_train.sh
  ├─src
    ├─__init__.py
    ├─beam_search.py
    ├─dataset.py
    ├─lr_schedule.py
    ├─process_output.py
    ├─tokenization.py
    ├─transformer_for_train.py
    ├─transformer_model.py
    ├─weight_init.py
    └─model_utils
      ├─__init__.py
      ├─config.py
      ├─device_adapter.py
      ├─local_adapter.py
      └─moxing_adapter.py
  ├─create_data.py
  ├─default_config_large_gpu.yaml
  ├─default_config_large.yaml
  ├─default_config.yaml
  ├─eval_onnx.py                                            // script for ONNX evaluation
  ├─eval.py
  ├─export.py
  ├─mindspore_hub_conf.py
  ├─postprocess.py
  ├─preprocess.py
  ├─requirements.txt
  └─train.py
```

### [Script Parameters](#contents)

#### Training Script Parameters

```text
usage: train.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                 [--enable_save_ckpt ENABLE_SAVE_CKPT]
                 [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                 [--save_checkpoint_steps N] [--save_checkpoint_num N]
                 [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                 [--data_path DATA_PATH] [--bucket_boundaries BUCKET_LENGTH]

options:
    --distribute               pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --epoch_size               epoch size: N, default is 52
    --device_num               number of used devices: N, default is 1
    --device_id                device id: N, default is 0
    --enable_save_ckpt         enable save checkpoint: "true" | "false", default is "true"
    --enable_lossscale         enable lossscale: "true" | "false", default is "true"
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --checkpoint_path          path to load checkpoint files: PATH, default is ""
    --save_checkpoint_steps    steps for saving checkpoint files: N, default is 2500
    --save_checkpoint_num      number for saving checkpoint files: N, default is 30
    --save_checkpoint_path     path to save checkpoint files: PATH, default is "./checkpoint/"
    --data_path                path to dataset file: PATH, default is ""
    --bucket_boundaries        sequence lengths for different bucket: LIST, default is [16, 32, 48, 64, 128]
```

#### Running Options

```text
default_config_large.yaml:
    transformer_network             version of Transformer model: base | large, default is large
    init_loss_scale_value           initial value of loss scale: N, default is 2^10
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 2000
    optimizer                       optimizer used in the network: Adam, default is "Adam"
    data_file                       data file: PATH
    model_file                      checkpoint file to be loaded: PATH
    output_file                     output file of evaluation: PATH
```

#### Network Parameters

```text
Parameters for dataset and network (Training/Evaluation):
    batch_size                      batch size of input dataset: N, default is 96
    seq_length                      max length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, default is 36560
    hidden_size                     size of Transformer encoder layers: N, default is 1024
    num_hidden_layers               number of hidden layers: N, default is 6
    num_attention_heads             number of attention heads: N, default is 16
    intermediate_size               size of intermediate layer: N, default is 4096
    hidden_act                      activation function used: ACTIVATION, default is "relu"
    hidden_dropout_prob             dropout probability for TransformerOutput: Q, default is 0.3
    attention_probs_dropout_prob    dropout probability for TransformerAttention: Q, default is 0.3
    max_position_embeddings         maximum length of sequences: N, default is 128
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    label_smoothing                 label smoothing setting: Q, default is 0.1
    beam_width                      beam width setting: N, default is 4
    max_decode_length               max decode length in evaluation: N, default is 80
    length_penalty_weight           normalize scores of translations according to their length: Q, default is 1.0
    compute_type                    compute type in Transformer: mstype.float16 | mstype.float32, default is mstype.float16

Parameters for learning rate:
    learning_rate                   value of learning rate: Q
    warmup_steps                    steps of the learning rate warm up: N
    start_decay_step                step of the learning rate to decay: N
    min_lr                          minimal learning rate: Q
```

## [Dataset Preparation](#contents)

- You may use this [shell script](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh) to download and preprocess WMT English-German dataset. Assuming you get the following files:
    - train.tok.clean.bpe.32000.en
    - train.tok.clean.bpe.32000.de
    - vocab.bpe.32000
    - newstest2014.tok.bpe.32000.en
    - newstest2014.tok.bpe.32000.de
    - newstest2014.tok.de

- Convert the original data to mindrecord for training:

    - The 'bucket' parameter is configured through the yaml file

    ``` bash
    paste train.tok.clean.bpe.32000.en train.tok.clean.bpe.32000.de > train.all
    python create_data.py --input_file train.all --vocab_file vocab.bpe.32000 --output_file /path/ende-l128-mindrecord --max_seq_length 128
    ```

- Convert the original data to mindrecord for evaluation:

    - The 'bucket' parameter is configured as [128] through the yaml file

    ``` bash
    paste newstest2014.tok.bpe.32000.en newstest2014.tok.bpe.32000.de > test.all
    python create_data.py --input_file test.all --vocab_file vocab.bpe.32000 --output_file /path/newstest2014-l128-mindrecord --num_splits 1 --max_seq_length 128 --clip_to_max_len True
    ```

## [Training Process](#contents)

- Set options in `default_config_large.yaml`, including loss_scale, learning rate and network hyperparameters. Click [here](https://www.mindspore.cn/tutorials/en/master/advanced/dataset.html) for more information about dataset.

- Run `run_standalone_train.sh` for non-distributed training of Transformer model.

    ``` bash
    bash scripts/run_standalone_train.sh [DEVICE_TARGET] [DEVICE_ID] [EPOCH_SIZE] [GRADIENT_ACCUMULATE_STEP] [DATA_PATH]
    ```

- Run `run_distribute_train_ascend.sh` for distributed training of Transformer model.

    ``` bash
    # Ascend environment
    bash scripts/run_distribute_train_ascend.sh [DEVICE_NUM] [EPOCH_SIZE] [DATA_PATH] [RANK_TABLE_FILE] [CONFIG_PATH]
    # GPU environment
    bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [DATA_PATH] [CONFIG_PATH]
    ```

**Attention**: data sink mode can not be used in transformer since the input data have different sequence lengths.

## [Evaluation Process](#contents)

- Set options in [CONFIG_PATH], that should be consistent with training. Make sure the 'device_target', 'data_file', 'model_file' and 'output_file' are set to your own path.

- Run `eval.py` for evaluation of Transformer model.

    ```bash
    python eval.py --config_path=[CONFIG_PATH]
    ```

- Run `process_output.sh` to process the output token ids to get the real translation results.

    ```bash
    bash scripts/process_output.sh REF_DATA EVAL_OUTPUT VOCAB_FILE
    ```

    You will get two files, REF_DATA.forbleu and EVAL_OUTPUT.forbleu, for BLEU score calculation.

- Calculate BLEU score, you may use this [perl script](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) and run following command to get the BLEU score.

    ```bash
    perl multi-bleu.perl REF_DATA.forbleu < EVAL_OUTPUT.forbleu
    ```

### [ONNX Evaluation](#contents)

- Export your model to ONNX:

  ```bash
  python export.py --device_target GPU --config default_config_large.yaml --model_file /path/to/transformer.ckpt --file_name /path/to/transformer.onnx --file_format ONNX
  ```

- Run ONNX evaluation:

  ```bash
  bash run_eval_onnx.sh <ONNX_MODEL> <MINDRECORD_DATA> [<CONFIG_PATH>] [<DEVICE_TARGET>] [<DEVICE_ID>]
  ```

- Run `process_output.sh` to process the output token ids to get the real translation results.

  ```bash
  bash scripts/process_output.sh REF_DATA EVAL_OUTPUT VOCAB_FILE
  ```

  You will get two files, REF_DATA.forbleu and EVAL_OUTPUT.forbleu, for BLEU score calculation.

- Calculate BLEU score, you may use this [perl script](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) and run following command to get the BLEU score.

  ```bash
  perl multi-bleu.perl REF_DATA.forbleu < EVAL_OUTPUT.forbleu
  ```

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

```shell
python export.py --model_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --config_path [CONFIG_PATH]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_TYPE] [DEVICE_ID] [CONFIG_PATH]
```

- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.
- `CONFIG_PATH` is optional, default value is '../default_config_large.yaml'"

### result

Inference result is saved in current path, 'output_file' will generate in path specified, For details about how to get BLEU score, see [Evaluation Process](#evaluation-process).

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters                 | Ascend                                                                                         | GPU                             |
| -------------------------- |------------------------------------------------------------------------------------------------| --------------------------------|
| Resource                   | Ascend 910; OS Euler2.8                                                                        | GPU(Tesla V100 SXM2)            |
| uploaded Date              | 07/05/2021 (month/day/year)                                                                    | 12/21/2021 (month/day/year)     |
| MindSpore Version          | 1.3.0                                                                                          | 1.5.0                           |
| Dataset                    | WMT Englis-German                                                                              | WMT Englis-German               |
| Training Parameters        | epoch=52, batch_size=96                                                                        | epoch=52, batch_size=32         |
| Optimizer                  | Adam                                                                                           | Adam                            |
| Loss Function              | Softmax Cross Entropy                                                                          | Softmax Cross Entropy           |
| BLEU Score                 | 28.7                                                                                           | 24.4                            |
| Speed                      | 400ms/step (8pcs)                                                                              | 337 ms/step (8pcs)              |
| Loss                       | 2.8                                                                                            | 2.9                             |
| Params (M)                 | 213.7                                                                                          | 213.7                           |
| Checkpoint for inference   | 2.4G (.ckpt file)                                                                              | 2.4G (.ckpt file)               |
| Scripts                    | [Transformer scripts](https://gitee.com/mindspore/models/tree/master/official/nlp/Transformer) |
| Model Version       | large                                                                                          |large|

#### Evaluation Performance

| Parameters        | Ascend                      | GPU                         |
| ----------------- | --------------------------- | --------------------------- |
| Resource          | Ascend 910; OS Euler2.8     | GPU(Tesla V100 SXM2)        |
| Uploaded Date     | 07/05/2021 (month/day/year) | 12/21/2021 (month/day/year) |
| MindSpore Version | 1.3.0                       | 1.5.0                       |
| Dataset           | WMT newstest2014            | WMT newstest2014            |
| batch_size        | 1                           | 1                           |
| outputs           | BLEU score                  | BLEU score                  |
| Accuracy          | BLEU=28.7                   | BLEU=24.4                   |
| Model Version     | large                       | large                       |

## [Description of Random Situation](#contents)

There are three random situations:

- Shuffle of the dataset.
- Initialization of some model weights.
- Dropout operations.

Some seeds have already been set in train.py to avoid the randomness of dataset shuffle and weight initialization. If you want to disable dropout, please set the corresponding dropout_prob parameter to 0 in default_config_large.yaml.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).

## FAQ

Refer to the [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) for some common question.

- **Q: Why the last checkpoint I got can't reach the accuracy?**

  **A**: At the end stage of training, the model accuracy usually drifts irregularly. Because we have to use a third-party perl scripts for evaluation, we can't find the best checkpoint as soon as the training process finished.
  You can try to evaluate the last several checkpoints to find the best one.

- **Q: why the shape match error such as "For 'Add', x.shape and y.shape need to broadcast." occurs?**

  **A**: because all of the parameters is supported by the dataset in readme. if users use new datasets, please modify parameters in the same time.
