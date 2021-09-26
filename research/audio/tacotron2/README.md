# Contents

- [Tacotron2 Description](#CenterNet-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training  Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Tacotron2 Description](#contents)

Tacotron2 is a TTS models. It contaion two phases, in first phase it use sequence to sequence method to predict mel spectrogram from text sequence,
in second phase it apply WaveNet as vocoder to convert mel spectrogram to waveform. We support training and evaluation tacotron2 model on Ascend platform.
[Paper](https://arxiv.org/abs/1712.05884): Jonathan, et al. Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions.

# [Dataset](#contents)

In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [The LJ Speech Dataset](<https://keithito.com/LJ-Speech-Dataset>)

- Dataset size：2.6G
- Data format：audio clips(13100) and transcription

- The dataset structure is as follows:

    ```path
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
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```path
tacotron2
├── eval.py                             //  testing and evaluation outputs
├── generate_hdf5.py                    // generate hdf5 file from dataset
├── requirements.txt                    // reqired package
├── README.md                           // descriptions about Tacotron2
├── scripts
│   ├── run_distribute_train_npu.sh    // launch distributed training with ascend platform
│   ├── run_standalone_eval_npu.sh     // launch standalone evaling with ascend platform
│   └── run_standalone_train_npu.sh    // launch standalone training with ascend platform
├── src
│   ├── callback.py                     // callbacks to monitor the training
│   ├── dataset.py                      // generate dataset and sampler
│   ├── hparams.py                      // Tacotron2 configs
│   ├── rnn_cells.py                    // rnn cells implements
│   ├── rnns.py                         // lstm cell with length mask implements
│   ├── tacotron2.py                    // Tacotron2 networks
│   ├── text
│   │   ├── cleaners.py                 // clean text sequence
│   │   ├── cmudict.py                  // symbols for encoding
│   │   ├── __init__.py                 // preprocessing and postprocessing text sequunce
│   │   ├── numbers.py                  // normalize numbers
│   │   └── symbols.py                  // symbols for encoding
│   └── utils
│       ├── audio.py                    // extract audio feature
│       └── convert.py                  // normalize mel spectrogram by meanvar for WaveNet  
└── train.py                            // training scripts

```

## [Script Parameters](#contents)

### Training

```text
usage: train.py  [--data_dir DATA_DIR]
                 [--ckpt_dir CKPT_DIR]
                 [--ckpt_pth CKPT_PTH]
                 [--is_distributed IS_DISTRIBUTED]
                 [--device_id DEVICE_ID]
                 [--workers WORKERS]
                 [--pretrained_model PRETRAINED_MODEL]

options:
    --pretrained_model          pretrained checkpoint path, default is ''
    --wokers                    num parallel workers, default is 8
```

### Evaluation

```text
usage: eval.py  [--ckpt_pth CKPT_PTH]
                [--out_dir OUT_DIR]
                [--fname FNAME]
                [--device_id DEVICE_ID]
                [--text TEXT]

options:
    --out_dir                    dirs to save outputs
    --fname                      filename to save outputs
```

# [Training Process](#contents)

Before training, the dataset should be processed. We use the scripts provided by [keithito](https://github.com/keithito/tacotron) to cleaning the text sequence and encoding symbols to number. In the meantime we extract mel spectrogram from the audio waveform, then pack the text sequence and corresponding mel spectrogram into a hdf5 file, you can run the following command line to install requirements and generate hdf5 file:

```shell
pip install -r requirements.txt

python3 generate_hdf5 --data_path path/to/LJSpeech-1.1
```

After preprocessing the dataset, you can get hdf5 file which contain text  sequence and mel spectrogram, then you run the following command line to train the network:

```shell

# standalone training ascend

bash scripts/run_distribute_train_npu.sh [relative path to hdf5 file] [device id]

# distributed training ascend

bash scripts/run_standalone_train_npu.sh [relative path to hdf5 file] [relative path to init config file] [rank size] [device id to begin]
```

# [Evaluation Process](#contents)

The following script is used to evaluate the model. You should specify text to synthesize. We use '~' as stop token, so the specified text should end with stop token. You can run the following command line to eval the network:

```shell
# standalone eval ascend

bash scripts/run_standalone_eval_npu.sh [ckpt_pth] [text to synthesis] [save dir] [filename] [device_id]
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | DeepSpeech                                                      |
| -------------------------- | ---------------------------------------------------------------|
| Resource                   | Ascend 910; OS Euler2.8              |
| uploaded Date              | 9/27/2021 (month/day/year)                                    |
| MindSpore Version          | 1.3.0                                                          |
| Dataset                    | LJSpeech-1.1                                                 |
| Training Parameters        | 8p, epoch=2000, batch_size=32  |
| Optimizer                  | Adam                                                           |
| Loss Function              | BinaryCrossEntropy, MSE                                |
| outputs                    | mel spectrogram                                                     |
| Loss                       | 0.33                                                        |
| Total time: training       | 8p: around 2 week;                                  |
| Checkpoint                 | 328.9M (.ckpt file)                                              |
| Scripts                    | [Tacotron2 script](https://gitee.com/mindspore/models/tree/master/research/audio/tacotron2) |

### Inference Performance

| Parameters                 | DeepSpeech                                                       |
| -------------------------- | ----------------------------------------------------------------|
| Resource                   | Ascend 910; OS Euler2.8                   |
| uploaded Date              | 9/27/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                           |
| Dataset                    | LJSpeech-1.1                         |
| batch_size                 | 1                                                               |
| outputs                    | mel spectrogram                       |
| Speed       | 1p: cost 125s synthesize 6s mel spectrogram|

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).