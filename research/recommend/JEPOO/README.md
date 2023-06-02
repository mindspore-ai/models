# Contents

- [Contents](#contents)
- [JEPOO Description](#JEPOO-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [JEPOO Description](#contents)

We propose a highly accurate method for joint estimation of pitch, onset and offset, named JEPOO.
We address the challenges of joint learning optimization and handling both single-pitch and
multi-pitch data through novel model design and a new optimization technique named Pareto modulated
loss with loss weight regularization. This is the first method that can accurately handle both
single-pitch and multi-pitch music data, and even a mix of them.

JEPOO: Highly Accurate Joint Estimation of Pitch, Onset and Offset for Music Information Retrieval

IJCAI 2023

# [Dataset](#contents)

- [MDB-stem-synth](https://zenodo.org/record/1481172#.ZFYvB3ZByUk)
- [MAPS](http://www.tsi.telecom-paristech.fr/aao/en/category/database/)
- [MAESTRO-V1.0.0](https://magenta.tensorflow.org/datasets/maestro)

# [Environment Requirements](#contents)

- Hardware（CPU and GPU）
    - Prepare hardware environment with CPU processor and GPU of Nvidia.
- Framework
    - [MindSpore-1.7.0](https://www.mindspore.cn/install/en)
- Requirements
  - numpy
  - tqdm
  - sacred
  - mindspore==1.7.0
  - mir_eval
  - librosa==0.8.0
  - mir_eval
  - soundfile
  - mido
- For more information, please check the resources below:
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Training and Evaluating

  ```shell
  # train and evaluate Naive joint learning of pitch, onset, offset
  python train.py
  # train and evaluate JEPOO with naive optimization
  python train_Naive.py
  # train and evaluate JEPOO
  python train_PML+LWR.py
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

  ```text
  .
  └─JEPOO
    ├─README.md               # descriptions of JEPOO
    ├─model                   # code for JEPOO model
      ├─__init__.py
      ├─dataset.py            # create dataset for JEPOO
      ├─decoding.py           # extract notes from prediction
      ├─loss.py               # the loss function used for JEPOO
      ├─lstm.py               # the model structure of LSTM
      ├─mel.py                # extract mel-spectrogram from raw audio
      ├─midi.py               # open and save midi files
      ├─min_norm_solvers.py   # Praeto optimization
      ├─model.py              # the overall model structure of JEPOO
      ├─utils.py              # some function for training JEPOO
    ├─evaluate.py             # evaluate function
    ├─train.py                # train and evaluate Naive joint learning of pitch, onset, offset
    ├─train_naive.py          # train and evaluate JEPOO with naive optimization
    ├─train_pml_lwr.py        # train and evaluate JEPOO
  ```

## [Script Parameters](#contents)

- Parameters that can be modified at the terminal

  ```text
  # Train
  path_dataset: './dataset/MAPS'  # dataset path
  alpha_FL: 5                     # the hyper-parameter to balance categories.
  gamma_FL: 0.2                   # the hyper-parameter to focus on hard samples
  logdir: "./runs"                # file path to record training log
  hop_length: 512                 # the hop length of audio frames
  batch_size: 8                   # training batch size.
  checkpoint_interval: 500        # the interval to save a checkpoint
  iterations: 300000              # training iterations
  w_re: 0.04                      # the weight of regularization
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters          | GPU                                                                                                                         |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Resource            | AMD Ryzen 2990WX 32-Core Processor;256G Memory;NVIDIA GeForce 2080Ti                                                        |
| uploaded Date       | 01/15/2023 (month/day/year)                                                                                                 |
| MindSpore Version   | 1.7.0                                                                                                                       |
| Dataset             | MAPS                                                                                       |
| Training Parameters | iterations=300000, batch_size=8, lr=1e-3                                                                                           |
| Optimizer           | Adam                                                                                                                        |
| Loss Function       | PML with LWR                                                                                                         |
| Outputs             | F1                                                                                                                     |
| Per Step Time       | 1.15s                                                                                                                    |

### Inference Performance

| Parameters        | GPU                                                                                                                         |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Resource          | AMD Ryzen 2990WX 32-Core Processor;256G Memory;NVIDIA GeForce 2080Ti                                                        |
| uploaded Date     | 01/15/2023 (month/day/year)                                                                                                 |
| MindSpore Version | 1.7.0                                                                                                                     |
| Dataset           | MAPS                                                                                             |
| Outputs           | F1                                                                                                                      |
| Per Step Time     | 1.54s                                                                                                      |

# [Description of Random Situation](#contents)

- We set the random seed before training in train.py, train_Naive.py or train_PML+LWR.py

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models)

