# Contents

- [Contents](#contents)
    - [FastSpeech Description](#fastspeech-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
            - [Standalone Training](#standalone-training)
            - [Distribute Training](#distribute-training)
        - [Evaluation Process](#evaluation-process)
            - [Checkpoints preparation](#checkpoints-preparation)
            - [Evaluation](#evaluation)
        - [Model Export](#model-export)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [FastSpeech Description](#contents)

Neural network based end-to-end text to speech (TTS) has significantly improved
the quality of synthesized speech. TTS methods usually first generate mel-spectrogram from text,
and then synthesize speech from the mel-spectrogram using vocoder such as WaveNet (WaveGlow in that work).
Compared with traditional concatenative and statistical parametric approaches, neural network based end-to-end models suffer from slow inference speed, and the synthesized speech is
usually not robust (i.e., some words are skipped or repeated) and lack of controllability (voice speed or prosody control).
In this work, we use feed-forward network based on Transformer to generate mel-spectrogram in parallel for TTS. Specifically, we use previously extracted attention alignments from an encoder-decoder
based teacher model for phoneme duration prediction, which is used by a length regulator to expand the source phoneme sequence to match the length of the target
mel-spectrogram sequence for parallel mel-spectrogram generation. Experiments on the LJSpeech dataset show that parallel model matches autoregressive models in terms of speech quality, nearly eliminates the problem of word skipping and
repeating in particularly hard cases, and can adjust voice speed smoothly.

[Paper](https://arxiv.org/pdf/1905.09263v5.pdf): FastSpeech: Fast, Robust and Controllable Text to Speech.

## [Model Architecture](#contents)

The architecture for FastSpeech is a feed-forward structure based on self-attention in Transformer
and 1D convolution. This structure is called Feed-Forward Transformer (FFT). Feed-Forward Transformer stacks multiple FFT blocks for phoneme to mel-spectrogram
transformation, with N blocks on the phoneme side, and N blocks on the mel-spectrogram side, with
a length regulator in between to bridge the length gap between the phoneme and mel-spectrogram sequence.
Each FFT block consists of a self-attention and 1D convolutional network.
The self-attention network consists of a multi-head attention to extract the cross-position information.
Different from the 2-layer dense network in Transformer, FastSpeech uses a 2-layer 1D convolutional network with ReLU activation.
The motivation is that the adjacent hidden states are more closely related in the character/phoneme and mel-spectrogram sequence in speech tasks.

## [Dataset](#contents)

We use LJSpeech-1.1 dataset and previously extracted alignments by teacher model.

Dataset description: 3.8 Gb of the .wav files with the annotated text (contains English speech only).

- [Download](https://keithito.com/LJ-Speech-Dataset/) LJSpeech and extract it into your `datasets` folder.
- [Download](https://github.com/xcmyz/FastSpeech/blob/master/alignments.zip) alignments and unzip into extracted LJSpeech dataset folder.

> Original LJSpeech-1.1 dataset not split into train/test parts.
> We manually split it into 13000/100 (train/test) by select 100 test indices stored into preprocess.py.
> We fixed indices, so you can reproduce our results.
> Also, you can select indices independently if you want and put it into _INDICES_FOR_TEST into preprocess.py.

The original dataset structure is as follows:

```text
.
└── LJSpeech-1.1
    ├─ alignments/
    ├─ wavs/
    └─ metadata.csv
```

Note: Before pre-processing the dataset you need to prepare the environment and install the requirements.
Preprocess script uses ~3.5 Gb video memory, thus you can specify the visible GPU devices if necessary.

Run (from the project folder) `preprocess.py` script located into `data` folder with following command:

```bash
python -m data.preprocess --dataset_path [PATH_TO_DATASET_FOLDER]
```

- PATH_TO_DATASET_FOLDER - path to the dataset root.

Processed data will be also saved into the PATH_TO_DATASET_FOLDER folder.

After pre-precessing the data, the dataset structure should be as follows:

```text
.
└── LJSpeech-1.1
    ├─ alignments/
    ├─ mels/
    ├─ metadata.csv
    ├─ metadata.txt
    ├─ train_indices.txt
    ├─ validation.txt
    └─ wavs/
```

## [Environment Requirements](#contents)

- Hardware (GPU).
- Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

Note: We use MindSpore 1.6.0 GPU, thus make sure that you install > 1.6.0 version.

## [Quick Start](#contents)

After installing MindSpore through the official website, you can follow the steps below for training and evaluation,
in particular, before training, you need to install `requirements.txt` by following
command `pip install -r requirements.txt`.

Then run training script as shown below.

```example
# Run standalone training example
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [DATASET_ROOT]

# Run distribute training example
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [LOGS_CKPT_DIR] [DATASET_ROOT]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```contents
.
└─FastSpeech
  ├─README.md
  ├─requirements.txt
  ├─data
  │ └─preprocess.py                    # data preprocessing script
  ├─scripts
  │ ├─run_distribute_train_gpu.sh      # launch distribute train on GPU
  │ ├─run_eval_gpu.sh                  # launch evaluation on GPU
  │ └─run_standalone_train_gpu.sh      # launch standalone train on GPU
  ├─src
  │ ├─audio
  │ │ ├─__init__.py
  │ │ ├─stft.py                        # audio processing scripts
  │ │ └─tools.py                       # audio processing tools
  │ ├─cfg
  │ │ ├─__init__.py
  │ │ └─config.py                      # config parser
  │ ├─deepspeech2
  │ │ ├─__init__.py
  │ │ ├─dataset.py                     # audio parser script for DeepSpeech2
  │ │ └─model.py                       # model scripts
  │ ├─import_ckpt
  │ │ ├─__init__.py
  │ │ ├─import_deepspeech2.py          # importer for DeepSpeech2 from < 1.5 MS versions
  │ │ └─import_waveglow.py             # importer for WaveGlow from .pickle
  │ ├─text
  │ │ ├─__init__.py
  │ │ ├─cleaners.py                    # text cleaners scripts
  │ │ ├─numbers.py                     # numbers to text preprocessing scripts
  │ │ └─symbols.py                     # symbols dictionary
  │ ├─transformer
  │ │ ├─__init__.py
  │ │ ├─constants.py                   # constants for transformer
  │ │ ├─layers.py                      # layers initialization
  │ │ ├─models.py                      # model blocks
  │ │ ├─modules.py                     # model modules
  │ │ └─sublayers.py                   # model sublayers
  │ ├─waveglow
  │ │ ├─__init__.py
  │ │ ├─layers.py                      # model layers
  │ │ ├─model.py                       # model scripts
  │ │ └─utils.py                       # utils tools
  │ ├─__init__.py
  │ ├─dataset.py                       # create dataset
  │ ├─metrics.py                       # metrics scripts
  │ ├─model.py                         # model scripts
  │ ├─modules.py                       # model modules
  │ └─utils.py                         # utilities used in other scripts
  ├─default_config.yaml                # default configs
  ├─eval.py                            # evaluation script
  ├─export.py                          # export to MINDIR script
  └─train.py                           # training script
```

### [Script Parameters](#contents)

```parameters
all parameters and descriptions, except --config_path, stored into default_config.yaml

usage: train.py [--config_path CONFIG_PATH]
                [--device_target DEVICE_TARGET]
                [--device_id DEVICE_ID]
                [--logs_dir LOGS_DIR]
                [--dataset_path DATASET_PATH]
                [--epochs EPOCHS]
                [--lr_scale LR_SCALE]
```

### [Training Process](#contents)

#### Standalone Training

```bash
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [DATASET_PATH]
```

The above command will run in the background, you can view the result through the generated standalone_train.log file.
After training, you can get the training loss and time logs in chosen logs dir:

```log
epoch: 200 step: 406, loss is 0.8701540231704712
epoch time: 168215.485 ms, per step time: 413.072 ms
```

The model checkpoints will be saved in logs outputs directory.

#### Distribute Training

```bash
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [LOGS_CKPT_DIR] [DATASET_PATH]
```

The above shell script will run distributed training in the background.
After training, you can get the training results:

```log
epoch: 200 step: 50, loss is 0.9151536226272583
epoch: 200 step: 50, loss is 0.9770485162734985
epoch: 200 step: 50, loss is 0.9304656982421875
epoch: 200 step: 50, loss is 0.8000383377075195
epoch: 200 step: 50, loss is 0.8380972146987915
epoch: 200 step: 50, loss is 0.854132890701294
epoch: 200 step: 50, loss is 0.8262668251991272
epoch: 200 step: 50, loss is 0.8031083345413208
epoch time: 25208.625 ms, per step time: 504.173 ms
epoch time: 25207.587 ms, per step time: 504.152 ms
epoch time: 25206.404 ms, per step time: 504.128 ms
epoch time: 25210.164 ms, per step time: 504.203 ms
epoch time: 25210.281 ms, per step time: 504.206 ms
epoch time: 25210.364 ms, per step time: 504.207 ms
epoch time: 25210.161 ms, per step time: 504.203 ms
epoch time: 25059.312 ms, per step time: 501.186 ms
```

Note: It was just examples of logs, values may vary.

### [Evaluation Process](#contents)

#### Checkpoints preparation

Before starting evaluation process, you need to import WaveGlow vocoder (generate audio from FastSpeech output mel-spectrograms) and DeepSpeech2 (to evaluate metrics) checkpoints.

- [Download](https://download.mindspore.cn/model_zoo/r1.3/deepspeech2_gpu_v130_librispeech_research_audio_bs20_avgwer11.34_avgcer3.79/) DeepSpeech2 checkpoint (version < 1.5, not directly load to new MindSpore versions).

  To import checkpoints follow steps below:
- Run `import_deepspeech2.py`. Converted checkpoint will be saved at the same directory as original and named `DeepSpeech2.ckpt`.

```bash
# from project root folder
python -m src.import_ckpt.import_deepspeech2 --ds_ckpt_url [CKPT_URL] # weights of .ckpt format
```

- To get WaveGlow take the following steps. We convert checkpoint from original [checkpoint](https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view) to the .pickle format with numpy weights, by using code below (previously we download `glow.py` model from original WaveGlow [implementation](https://github.com/NVIDIA/waveglow)).

```python
# script in same dir with glow.py
import torch

waveglow = torch.load(checkpoint_url)['model']  # ckpt_url for original .pt object
waveglow = waveglow.remove_weightnorm(waveglow)
numpy_weights = {key: value.detach().numpy() for key, value in waveglow.named_parameters()}
# save numpy_weights as .pickle format
```

Note: The original checkpoint is stored in the PyTorch format (.pth). You need to install PyTorch first, before running the code above.

- To import .pickle WaveGlow checkpoint run `import_waveglow.py`. Converted checkpoint will be saved at the same directory as original and named `WaveGlow.ckpt`.

```bash
# from project root folder
python -m src.import_ckpt.import_waveglow --wg_ckpt_url [CKPT_URL] # weights of .pickle format
```

#### Evaluation

Before evaluation make sure that you have trained FastSpeech.ckpt, converted WaveGlow.ckpt, and converted DeepSpeech2.ckpt.
To start evaluation run the command below.

```bash
bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET_PATH] [FS_CKPT_URL] [WG_CKPT_URL] [DS_CKPT_URL]
```

The above python command will run in the background. You can view the results through the file "eval.log".

```text
==========Evaluation results==========
Mean Frechet distance 201.42256
Mean Kernel distance 0.02357
Generated audios stored into results
```

### [Model Export](#contents)

You can export the model to mindir format by running the following python script:

```bash
python export.py --fs_ckpt_url [FS_CKPT_URL]
```

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters                 | GPU (1p)                                                   | GPU (8p)                                                              |
| -------------------------- |----------------------------------------------------------- |---------------------------------------------------------------------- |
| Model                      | FastSpeech                                                 | FastSpeech                                                            |
| Hardware                   | 1 Nvidia Tesla V100-PCIE, CPU @ 3.40GHz                    | 8 Nvidia RTX 3090, Intel Xeon Gold 6226R CPU @ 2.90GHz                |
| Upload Date                | 14/03/2022 (day/month/year)                                | 14/03/2022 (day/month/year)                                           |
| MindSpore Version          | 1.6.0                                                      | 1.6.0                                                                 |
| Dataset                    | LJSpeech-1.1                                               | LJSpeech-1.1                                                          |
| Training Parameters        | epochs=200, batch_size=32, warmup_steps=5000, lr_scale=1   | epochs=300, batch_size=32 (per device), warmup_steps=5000, lr_scale=2 |
| Optimizer                  | Adam (beta1=0.9, beta2=0.98, eps=1e-9)                     | Adam (beta1=0.9, beta2=0.98, eps=1e-9)                                |
| Loss Function              | MSE, L1                                                    | MSE, L1                                                               |
| Speed                      | ~412 ms/step                                               | ~504 ms/step                                                          |
| Total time                 | ~9.3 hours                                                 | ~2.1 hours                                                            |

Note: lr scheduler was taken from [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) paper.

#### Evaluation Performance

| Parameters          | Trained on GPU (1p)                                        | Trained on GPU (8p)                                        |
| ------------------- |--------------------------------------------------------    |----------------------------------------------------------- |
| Model               | FastSpeech                                                 | FastSpeech                                                 |
| Resource            | 1 Nvidia Tesla V100-PCIE, CPU @ 3.40GHz                    | 1 Nvidia Tesla V100-PCIE, CPU @ 3.40GHz                    |
| Upload Date         | 14/03/2022 (day/month/year)                                | 14/03/2022 (day/month/year)                                |
| MindSpore Version   | 1.6.0                                                      | 1.6.0                                                      |
| Dataset             | LJSpeech-1.1                                               | LJSpeech-1.1                                               |
| Batch_size          | 1                                                          | 1                                                          |
| Outputs             | Mel-spectrogram, mel duration                              | Mel-spectrogram, mel duration                              |
| Metric              | (classifier distances) Frechet 201.42256, Kernel 0.02357   | (classifier distances) Frechet 203.89236, Kernel 0.02386   |

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
