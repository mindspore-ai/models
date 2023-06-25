# Contents

- [Contents](#contents)
    - [LPCNet Description](#lpcnet-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
            - [Training](#training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Generate input data for network](#generate-input-data-for-network)
        - [Export MindIR](#export-mindir)
        - [Inference](#inference)
        - [Result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
    - [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

## [LPCNet Description](#contents)

LPCNet is a lowbitrate neural vocoder based on linear prediction and sparse recurrent networks.

[Article](https://jmvalin.ca/papers/lpcnet_codec.pdf): J.-M. Valin, J. Skoglund, A Real-Time Wideband Neural Vocoder at 1.6 kb/s Using LPCNet, Proc. INTERSPEECH, arxiv:1903.12087, 2019.

## [Model Architecture](#contents)

LPCNet has two parts: frame rate network and sample rate newtwork. Frame rate network consists of two convolutional layers and two fully connected layers, this network extracts features from 5-frames context. Sample rate network consists of two GRU layers and dual fully connected layer (modification of fully connected layer). The first GRU layer waights are sparcified. Sample rate network gets features extracted by frame rate network along with linear prediction for current timestep, sample end excitation for previous timestep and predicts current excitation via sigmoid function and binary tree probability representation.

## [Dataset](#contents)

Dataset used: [LibriSpeech](<https://www.openslr.org/12/>)

- Dataset size：6.3 GB
    - training set is train-clean-100.tar.gz and test dataset is test-clean.tar.gz
- Data format：binary files
    - Note：LPCNet is a hybrid compression method. The compression is made by quantizing 18 Bark-scale cepstral coefficients and 2 pitch paramteters. Network reconstructs original audio from this quantized features.

- Download the dataset (only .flac files are needed), the directory structure is as follows:

    ```train-clean-100
    ├─README.TXT
    ├─READERS.TXT
    ├─CHAPTERS.TXT
    ├─BOOKS.TXT
    └─train-clean-100
    ├─19
        ├─198
        ├─19-198.trans.txt
        ├─19-198-0001.flac
        ├─19-198-0002.flac
        ...
        └─19-198-0014.flac
    ```

## [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
# enter script dir, compile code for feature quantization
bash run_compile.sh

# generate training data (flac and sox should be installed sequentially before running the command)
bash run_process_train_data.sh [TRAIN_DATASET_PATH] [OUTPUT_PATH]
# example: bash run_process_train_data.sh ./train-100-clean ~/dataset_path/training_dataset/

# train LPCNet (1P)
bash run_standalone_train_ascend.sh [PREPROCESSED_TRAINING_DATASET_PATH] [CHECKPOINT_SAVE_PATH]
# example: bash run_standalone_train_ascend.sh ~/dataset_path/training_dataset/ ./ckpt/

# or train LPCNet parallelly (2P, 4P, 8P)
bash run_distribute_train_ascend.sh [PREPROCESSED_TRAINING_DATASET_PATH] [RANK_SIZE] [RANK_TABLE]
# example: bash run_distribute_train_ascend.sh ~/dataset_path/training_dataset/ 8 ./hccl_8p.json

# generate test data (10 files are selected from test-clean for evaluation)
bash run_process_eval_data.sh [EVAL_DATASET_PATH] [OUTPUT_PATH]
# example: bash run_process_eval_data.sh ./dataset_path/test_dataset ~/dataset_path/test_features/

# evaluate LPCNet
bash run_eval_ascend.sh [TEST_DATASET_PATH] [OUTPUT_PATH] [CHECKPOINT_SAVE_PATH]
# example: bash run_eval_ascend.sh ~/dataset_path/test_features/ ./eval_results/ ./ckpt/lpcnet-4_37721.ckpt
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── audio
    ├── LPCNet
        ├── README.md                          // description of LPCNet in English
        ├── requirements.txt                   // required packages
        ├── scripts
        │   ├──run_compile.sh                  // compile code feature extraction and quantization code
        │   ├──run_infer_310.sh                // 310 inference
        │   ├──run_proccess_train_data.sh      // generate training dataset from .flac files
        |   ├──run_process_eval_data.sh        // generate eval dataset from .flac files
        │   ├──run_stanalone_train_ascend.sh   // train in Ascend 1P
        │   ├──run_distribute_train_ascend.sh  // train in Ascend 2P, 4P or 8P
        ├── src
        │   ├──rnns                            // dynamic GRU implementation
        │   ├──dataloader.py                   // dataloader for model training
        │   ├──lossfuncs.py                    // loss function
        │   ├──lpcnet.py                       // lpcnet implementation
        │   ├──mdense.py                       // dual fully connected layer implementation
        │   ├──train_lpcnet_parallel.py        // distributed training
        |   └──ulaw.py                         // u-law qunatization
        ├── third_party                        // feature extraction and quantization (C++)
        ├── ascend310_infer                    // for inference on Ascend310 (C++)
        ├── train.py                           // train in Ascend main program
        ├── process_data.py                    // generate training dataset from KITTI .bin files main program
        ├── eval.py                            // evaluation main program
        ├──export.py                           // exporting model for infer
```

### [Script Parameters](#contents)

```text
Major parameters in train.py：
features: Path to binary features
data: Path to 16-bit PCM aligntd with features
output: Path where .ckpt stored
--batch-size：Training batch size
--epochs：Total training epochs
--device：Device where the code will be implemented. Optional values are "Ascend", "GPU", "CPU"
--checkpoint：The path to the checkpoint file saved after training.（recommend ）

Major parameters in eval.py：
test_data_path: path to test dataset，test data is features extracted and quantized by run_process_eval_data.sh
output_path: The path where decompressed / reconstructed files stored
model_file: The path to the checkpoint file which needs to be loaded
```

### [Training Process](#contents)

#### Training

- Running on Ascend

  ```bash
  python train.py [FEATURES_FILE] [AUDIO_DATA_FILE] [CHECKPOINT_SAVE_PATH] --device=[DEVICE_TARGET] --batch-size=[batch-size]
  # or enter script dir, run 1P training script
  bash run_stanalone_train_ascend.sh ~/dataset_path/training_dataset/ ./ckpt/
  # or enter script dir, run 2P, 4P or 8P training script
  bash run_distribute_train_ascend.sh ~/dataset_path/training_dataset/ 8 ./hccl_8p.json
  ```

  After training, the loss value will be achieved as follows:

  ```bash
  epoch: 1 step: 37721, loss is 4.2791853
  ...
  epoch: 4 step: 37721, loss is 3.7296906
  ...
  ```

  The model checkpoint will be saved in the specified directory.

### [Evaluation Process](#contents)

#### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

  ```bash
  python eval.py [TEST_DATA_PATH] [OUTPUT_PATH] [CHECKPOINT_SAVE_PATH]
  # or enter script dir, run evaluation script
  bash run_eval_ascend.sh [TEST_DATASET_PATH] [OUTPUT_PATH] [CHECKPOINT_SAVE_PATH]
  ```

## [Inference Process](#contents)

### Generate input data for network

```shell
# Enter script dir, run run_process_data.sh script
bash run_process_eval_data.sh [EVAL_DATASET_PATH] [OUTPUT_PATH]
# example: bash run_process_eval_data.sh ./dataset_path/test_dataset ~/dataset_path/test_features/
```

### Export MindIR

```shell
python export.py --ckpt_file=[CKPT_PATH] --max_len=[MAX_LEN] --out_file=[OUT_FILE]
# Example:
python export.py --ckpt_file='./checkpoint/ms-4_37721.ckpt'  --out_file=lpcnet --max_len 500
# NOTE: max_len is the max number of 10 ms frames which can be processed, audios longer will be truncated
```

The ckpt_file parameter is required

### [Inference](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```bash
# Enter script dir, run run_infer_310.sh script
bash run_infer_cpp.sh [ENCODER_PATH] [DECODER_PATH] [DATA_PATH] [DEVICE_TYPE] [DEVICE_ID] (optional)
```

- `ENCODER_PATH` Absolute path of \*_enc mindir
- `DECODER_PATH` Absolute path of \*_dec mindir
- `DATA_PATH` Absolute path of input data（Under this path, there should be one \*.f32 file extracted using run_process_eval_data.sh）
- `DEVICE_TYPE` Device type can be chosen from [Ascend, GPU, CPU]
- `DEVICE_ID` Device id where the code will be run

### Result

Inference result is saved in ./infer.log.

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters          | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| Network Name | LPCNet                                                   |
| Resource  | Ascend 910；CPU 191 core；Memory 755G;                            |
| Uploaded Date | TBD                                                          |
| MindSpore Version | 1.5.0                                                        |
| Dataset | 6.16 GB of clean speech                               |
| Training Parameters | epoch=4, batch_size=64 , lr=0.001 |
| Optimizer | Adam                                                         |
| Loss Function | SparseCategoricalCrossentropy                                          |
| Output   | distribution                                         |
| Loss      | 3.24957                                                          |
| Speed     | 1P：182 samples/sec;  8P：154 samples/sec             |
| Total Time | 1P：17.5h；8P：2.5h                               |
| parameters(M) | 1.2M                                                        |
| Checkpoint for Fine tuning | 15.7M (.ckpt file)                                               |
| Scripts   | [lpcnet script](https://gitee.com/mindspore/models/tree/master/official/audio/LPCNet) |

### Inference Performance

| Parameters        | Ascend                                                       |
| ----------------- | ------------------------------------------------------------ |
| Network Name      | LPCNet                                                   |
| Resource          | Ascend 910                                                   |
| Uploaded Date     | TBD                                                          |
| MindSpore Version | 1.5.0                                                        |
| Dataset           | Binary feature files constructed by 10 Files from LibriSpeach test-clean                                       |
| batch_size        | 1                                                            |
| Output            | Reconstructed 16-bit PCM audio|
| Accuracy         | 0.004 (MSE)|

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
