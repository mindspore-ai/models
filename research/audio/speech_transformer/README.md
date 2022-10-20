# Contents

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
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [Speech Transformer Description](#contents)

The standard transformer sequence2sequence (encoder, decoder) model architecture is used to solve the speech2text problem.

[Paper](https://ieeexplore.ieee.org/document/8682586):  Yuanyuan Zhao, Jie Li, Xiaorui Wang, and Yan Li. "The SpeechTransformer for Large-scale Mandarin Chinese Speech Recognition." ICASSP 2019.

## [Model Architecture](#contents)

Specifically, Transformer contains six encoder modules and six decoder modules. Each encoder module consists of a self-attention layer and a feed forward layer, each decoder module consists of a self-attention layer, a encoder-decoder-attention layer and a feed forward layer.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using AISHELL dataset.

You can download dataset by this [link](http://www.openslr.org/33/)

## [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After dataset preparation, you can start training and evaluation as follows:

(Note that you must specify dataset path and path to char dictionary in `default_config.yaml`)

```bash
# run training example
bash scripts/run_standalone_train_gpu.sh 0 150 1

# run distributed training example
bash scripts/run_distribute_train_gpu.sh 8 150 ./default_config.yaml

# run evaluation example
bash scripts/run_eval_gpu.sh 0 /your/path/data.json /your/path/checkpoint_file ./default_config.yaml
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└── speech_transformer
    ├── README.md
    ├── default_config.yaml
    ├── eval.py
    ├── evaluate_cer.py
    ├── export.py
    ├── requirements.txt
    ├── prepare_aishell_data
    │   ├── README.md
    │   └── convert_kaldi_bins_to_pickle.py
    ├── scripts
    │   ├── run_distribute_train_gpu.sh
    │   ├── run_eval_gpu.sh
    │   └── run_standalone_train_gpu.sh
    ├── src
    │   ├── beam_search.py
    │   ├── dataset.py
    │   ├── __init__.py
    │   ├── lr_schedule.py
    │   ├── model_utils
    │   │   ├── config.py
    │   │   ├── device_adapter.py
    │   │   ├── __init__.py
    │   │   ├── local_adapter.py
    │   │   └── moxing_adapter.py
    │   ├── transformer_for_train.py
    │   ├── transformer_model.py
    │   └── weight_init.py
    └── train.py
```

### [Script Parameters](#contents)

#### Training Script Parameters

```text
usage: train.py  [--distribute DISTRIBUTE] [--epoch_size N] [--device_num N] [--device_id N]
                 [--enable_save_ckpt ENABLE_SAVE_CKPT]
                 [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                 [--save_checkpoint_steps N] [--save_checkpoint_num N]
                 [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                 [--data_json_path DATA_PATH]

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
    --data_json_path           path to dataset file: PATH, default is ""
```

#### Running Options

```text
default_config.yaml:
    transformer_network             version of Transformer model: base | large, default is base
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
    batch_size                      batch size of input dataset: N, default is 32
    seq_length                      max length of input sequence: N, default is 512
    input_feature_size              Input feature size: N, default is 320
    vocab_size                      size of each embedding vector: N, default is 4233
    hidden_size                     size of Transformer encoder layers: N, default is 1024
    num_hidden_layers               number of hidden layers: N, default is 6
    num_attention_heads             number of attention heads: N, default is 8
    intermediate_size               size of intermediate layer: N, default is 2048
    hidden_act                      activation function used: ACTIVATION, default is "relu"
    hidden_dropout_prob             dropout probability for TransformerOutput: Q, default is 0.3
    attention_probs_dropout_prob    dropout probability for TransformerAttention: Q, default is 0.3
    max_position_embeddings         maximum length of sequences: N, default is 512
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    label_smoothing                 label smoothing setting: Q, default is 0.1
    beam_width                      beam width setting: N, default is 5
    max_decode_length               max decode length in evaluation: N, default is 100
    length_penalty_weight           normalize scores of translations according to their length: Q, default is 1.0
    compute_type                    compute type in Transformer: mstype.float16 | mstype.float32, default is mstype.float16

Parameters for learning rate:
    learning_rate                   value of learning rate: Q
    warmup_steps                    steps of the learning rate warm up: N
    start_decay_step                step of the learning rate to decay: N
    min_lr                          minimal learning rate: Q
    lr_param_k                      scale factor for learning rate
```

## [Dataset Preparation](#contents)

Detailed instruction for dataset preparation is described in [prepare_aishell_data/README.md](prepare_aishell_data/README.md)

Dataset preprocessed using reference [implementation](https://github.com/kaituoxu/Speech-Transformer).
Dataset is preprocessed using `Kaldi` and converts kaldi binaries into Python pickle objects.

## [Training Process](#contents)

- Set options in `default_config.yaml`, including loss_scale, learning rate and network hyperparameters. Click [here](https://www.mindspore.cn/tutorials/en/master/advanced/dataset.html) for more information about dataset.

- Run `run_standalone_train_gpu.sh` for non-distributed training of Transformer model.

    ``` bash
    bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [EPOCH_SIZE] [GRADIENT_ACCUMULATE_STEP]
    ```

- Run `run_distribute_train_gpu.sh` for distributed training of Transformer model.

    ``` bash
    bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [CONFIG_PATH]
    ```

**Attention**: data sink mode can not be used in transformer since the input data have different sequence lengths.

## [Evaluation Process](#contents)

- Set options in `default_config.yaml`. Make sure the 'data_file', 'model_file' and 'output_file' are set to your own path.

- Run `bash scripts/run_eval_gpu.sh` for evaluation of Transformer model.

    ```bash
    bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATA_JSON_PATH] [CKPT_PATH] [CONFIG_PATH]
    ```

- Calculate Character Error Rate

    ```bash
    python evaluate_cer.py
    ```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --model_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in "MINDIR"

### result

Inference result is saved in 'output_eval.json'

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters                 | GPU                                                            |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | 8x Nvidia V100-PCIE                                            |
| uploaded Date              | 07/05/2021 (month/day/year)                                    |
| MindSpore Version          | 1.5.0rc1                                                       |
| Dataset                    | AISHELL Train                                                  |
| Training Parameters        | epoch=150, batch_size=32                                       |
| Optimizer                  | Adam                                                           |
| Loss Function              | Softmax Cross Entropy                                          |
| Speed                      | 321ms/step (8pcs)                                              |
| Total Training time        | 6.2 hours (8pcs)                                               |
| Loss                       | 1.24                                                           |
| Params (M)                 | 49.6                                                           |
| Checkpoint for inference   | 557Mb (.ckpt file)                                             |
| Scripts                    | [Speech Transformer scripts](scripts) |

#### Evaluation Performance

| Parameters          | GPU                         |
| ------------------- | --------------------------- |
| Resource            | Nvidia V100-PCIE            |
| Uploaded Date       | 07/05/2021 (month/day/year) |
| MindSpore Version   | 1.5.0rc1                    |
| Dataset             | AISHELL Test                |
| batch_size          | 1                           |
| outputs             | Character Error Rate        |
| Accuracy            | CER=12.21                    |

## [Description of Random Situation](#contents)

There are three random situations:

- Shuffle of the dataset.
- Initialization of some model weights.
- Dropout operations.

Some seeds have already been set in train.py to avoid the randomness of dataset shuffle and weight initialization. If you want to disable dropout, please set the corresponding dropout_prob parameter to 0 in default_config.yaml.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
