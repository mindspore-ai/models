# Contents

- [Tacotron2 Description](#tacotron2-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Inference Process](#inference-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Tacotron2 Description](#contents)

Tacotron2 is a TTS models. It contaion two phases, in first phase it use sequence to sequence method to predict mel spectrogram from text sequence,
in second phase it apply WaveNet as vocoder to convert mel spectrogram to waveform. We support training and evaluation tacotron2 model on Ascend platform.

[Paper](https://arxiv.org/abs/1712.05884): Jonathan, et al. Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions.

# [Model Architecture](#contents)

Tacotron2 substantially is a sequence to sequence model which contain an encoder and a decoder, the encoder is implemented by three conv layers and one BiLSTM layer, and the decoder use  two LSTM layers to decode next state, a location-aware attention is applied between encoder and decoder, then the decoded state is fed into postnet which is implemented by five conv layers to predict mel spectrogram, finally the predicted mel spectrogram features is fed into WaveNet vocoder to synthesis speech signal.

# [Dataset](#contents)

In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [The LJ Speech Dataset](<https://keithito.com/LJ-Speech-Dataset>)

- Dataset size：2.6G
- Data format：audio clips(13100) and transcription

- The dataset structure is as follows:

    ```text
    .
    └── LJSpeech-1.1
        ├─ wavs                  //audio clips files
        └─ metadata.csv           //transcripts
    ```

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

  ```python
  # install python3 package
  pip install -r requirements.txt
  # generate hdf5 file from dataset, the output file is ljspeech.h5 in current directory.
  python generate_hdf5 --data_path /path/to/LJSpeech-1.1
  ```

  ```shell
  cd scripts
  # run standalone training
  bash run_standalone_train.sh [DATASET_PATH] [DEVICE_ID]
  # example: bash run_standalone_train.sh /path/ljspeech.hdf5 0

  # run distributed training
  bash run_distributed_train.sh [DATASET_PATH] [RANK_TABLE_PATH] [DATANAME] [RANK_SIZE] [DEVICE_BEGIN]
  # example: bash run_distributed_train.sh /path/ljspeech.h5 ../hccl_8p_01234567_127.0.0.1.json 8 0

  # run evaluation
  bash run_eval.sh [OUTPUT_PATH] [MODEL_CKPT] [DEVICE_ID] text is set in config.py( can modify text of ljspeech_config.yaml)
  # example: bash run_eval.sh /path/output /path/model.ckpt 0
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    - Standalone training example on ModelArts

      ```python
      # run standalone training example

      # (1) Add "config_path='/path_to_code/[DATASET_NAME]_config.yaml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=True" on [DATASET_NAME]_config.yaml file.
      #          Set "dataset_path='/cache/data/[DATASET_NAME]'" on [DATASET_NAME]_config.yaml file.
      #          Set "data_name='[DATASET_NAME]'" on [DATASET_NAME]_config.yaml file.
      #          (option)Set other parameters on [DATASET_NAME]_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add "dataset_path='/cache/data/[DATASET_NAME]'" on the website UI interface.
      #          Add "data_name='[DATASET_NAME]'" on the website UI interface.
      #          (option)Add other parameters on the website UI interface.
      # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
      # (4) Set the code directory to "/path/to/tacotron2" on the website UI interface.
      # (5) Set the startup file to "train.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
      ```

    - Distributed Training example on Modelarts

      ```python
      # run distributed training example

      # (1) Add "config_path='/path_to_code/[DATASET_NAME]_config.yaml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=True" on [DATASET_NAME]_config.yaml file.
      #          Set "run_distribute=True" on [DATASET_NAME]_config.yaml file.
      #          Set "dataset_path='/cache/data/[DATASET_NAME]'" on [DATASET_NAME]_config.yaml file.
      #          Set "data_name='[DATASET_NAME]'" on [DATASET_NAME]_config.yaml file.
      #          (option)Set other parameters on [DATASET_NAME]_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add "run_distribute=True" on the website UI interface.
      #          Add "dataset_path='/cache/data/[DATASET_NAME]'" on the website UI interface.
      #          Add "data_name='[DATASET_NAME]'" on the website UI interface.
      #          (option)Add other parameters on the website UI interface.
      # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
      # (4) Set the code directory to "/path/to/tacotron2" on the website UI interface.
      # (5) Set the startup file to "train.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
      ```

    - Eval on ModelArts

      ```python
      # run eval example

      # (1) Add "config_path='/path_to_code/[DATASET_NAME]_config.yaml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=True" on [DATASET_NAME]_config.yaml file.
      #          Set "data_name='[DATASET_NAME]'" on [DATASET_NAME]_config.yaml file.
      #          Set "model_ckpt='/cache/checkpoint_path/model.ckpt'" on [DATASET_NAME]_config.yaml file.
      #          Set "text='text to synthesize'" on [DATASET_NAME]_config.yaml file.
      #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on [DATASET_NAME]_config.yaml file.
      #          (option)Set other parameters on [DATASET_NAME]_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add "data_name='[DATASET_NAME]'" on the website UI interface.
      #          Add "model_ckpt=/cache/checkpoint_path/model.ckpt" on the website UI interface.
      #          Add "text='text to synthesize'" on the website UI interface.
      #          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
      #          (option)Add other parameters on the website UI interface.
      # (3) Upload or copy your pretrained model to S3 bucket.
      # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
      # (5) Set the code directory to "/path/to/tacotron2" on the website UI interface.
      # (6) Set the startup file to "eval.py" on the website UI interface.
      # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (8) Create your job.
      ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```path

tacotron2/
├── eval.py                             //  evaluate entry
├── generate_hdf5.py                    // generate hdf5 file from dataset
├── ljspeech_config.yaml
├── model_utils
│  ├── config.py                       // Parse arguments
│  ├── device_adapter.py               // Device adapter for ModelArts
│  ├── __init__.py                     // init file
│  ├── local_adapter.py                // Local adapter
│  └── moxing_adapter.py               // Moxing adapter for ModelArts
├── README.md                           // descriptions about Tacotron2
├── requirements.txt                // reqired package
├── scripts
│  ├── run_distribute_train.sh         // launch distributed training
│  ├── run_eval.sh                     // launch evaluate
│  └── run_standalone_train.sh         // launch standalone training
├── src
│  ├── callback.py                     // callbacks to monitor the training
│  ├── dataset.py                      // define dataset and sampler
│  ├── hparams.py                      // Tacotron2 configs
│  ├── rnn_cells.py                    // rnn cells implementations
│  ├── rnns.py                         // lstm implementations with length mask
│  ├── tacotron2.py                    // Tacotron2 networks
│  ├── text
│  │  ├── cleaners.py                  // clean text sequence
│  │  ├── cmudict.py                   // define cmudict
│  │  ├── __init__.py                  // processing text sequunce
│  │  ├── numbers.py                   // normalize numbers
│  │  └── symbols.py                   // symbols for encoding
│  └── utils
│      ├── audio.py                     // extract audio feature
│      └── convert.py                   // normalize mel spectrogram by meanvar
└── train.py                            // training entry

```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in [DATASET]_config.yaml

- config for LJSpeech-1.1

  ```python
  'pretrain_ckpt': '/path/to/model.ckpt'# use pretrained ckpt at training phase
  'model_ckpt': '/path/to/model.ckpt'   # use pretrained ckpt at inference phase
  'lr': 0.002                           # initial learning rate
  'batch_size': 16                      # training batch size
  'epoch_num': 2000                     # total training epochs
  'warmup_epochs': 30                   # warmpup lr epochs
  'save_ckpt_dir:' './ckpt'             # specify ckpt saving dir
  'keep_checkpoint_max': 10             # only keep the last keep_checkpoint_max checkpoint

  'text': 'text to synthesize'          # specify text to synthesize at inference
  'dataset_path': '/dir/to/hdf5'        # specify dir to hdf5 file
  'data_name': 'ljspeech'               # specify dataset name
  'audioname': 'text2speech'            # specify filename for generated audio
  'run_distribute': False               # whether distributed training
  'device_id': 0                        # specify which device to use
  ```

### [Training Process](#content)

- Running on Ascend

    - Start task training on a single device and run the shell script

        ```bash
        cd scripts
        bash run_standalone_train.sh [DATASET_PATH] [DEVICE_ID] [DATANAME]
        ```

    - Running scripts for distributed training of Tacotron2. Task training on multiple device and run the following command in bash to be executed in `scripts/`:

        ```bash
        cd scripts
        bash run_distributed_train.sh [DATASET_PATH] [RANK_TABLE_PATH] [DATANAME] [RANK_SIZE] [DEVICE_BEGIN]
        ```

    Note: `DATASET_PATH` is the directory contains hdf5 file.

### [Inference Process](#content)

- Running on Ascend

    - Running scripts for evaluation of Tacotron2. The commdan as below.

        ```bash
        cd scripts
        bash run_eval.sh [OUTPUT_PATH] [DATANAME] [MODEL_CKPT] [DEVICE_ID]
        ```

    Note: The `OUTPUT_PATH` is the directory to save evaluate outputs

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Tacotron2                                                      |
| -------------------------- | ---------------------------------------------------------------|
| Resource                   | Ascend 910; OS Euler2.8              |
| uploaded Date              | 12/20/2021 (month/day/year)                                    |
| MindSpore Version          | 1.3.0                                                          |
| Dataset                    | LJSpeech-1.1                                                 |
| Training Parameters        | 8p, epoch=2000, batch_size=16  |
| Optimizer                  | Adam                                                           |
| Loss Function              | BinaryCrossEntropy, MSE                                |
| outputs                    | mel spectrogram                                                     |
| Loss                       | 0.33                                                        |
| Speed|1264ms/step|
| Total time: training       | 8p: 24h/19m/41s;;                                  |
| Checkpoint                 | 328.9M (.ckpt file)                                              |
| Scripts                    | [Tacotron2 script](https://gitee.com/mindspore/models/tree/master/official/audio/Tacotron2) |

### Inference Performance

| Parameters                 | Tacotron2                                                       |
| -------------------------- | ----------------------------------------------------------------|
| Resource                   | Ascend 910; OS Euler2.8                   |
| uploaded Date              | 12/20/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                           |
| Dataset                    | LJSpeech-1.1                         |
| batch_size                 | 1                                                               |
| outputs                    | mel spectrogram                       |
| Speed       | 1p: cost 125s synthesize 6s mel spectrogram|

## [Random Situation Description](#content)

There only one random situation.

- Initialization of some model weights.

Some seeds have already been set in train.py to avoid the randomness of weight initialization.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
