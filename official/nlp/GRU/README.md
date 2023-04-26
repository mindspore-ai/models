
<!-- TOC -->

- [GRU](#gru)
    - [Paper](#paper)
- [Model Structure](#model-structure)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
    - [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Dataset Preparation](#dataset-preparation)
    - [Configuration File](#configuration-file)
    - [Training Process](#training-process)
    - [Inference Process](#inference-process)
    - [Export MindIR](#export-mindir)
    - [ONNX Export And Evaluation](#onnx-export-and-evaluation)
        - [ONNX Export](#onnx-export)
        - [ONNX Evaluation](#onnx-evaluation)
    - [Inference Process](#inference-process-1)
        - [Usage](#usage)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Random Situation Description](#random-situation-description)
- [Others](#others)
- [ModelZoo HomePage](#modelzoo-homepage)

<!-- /TOC -->

# [GRU](#contents)

GRU(Gate Recurrent Unit) is a kind of recurrent neural network algorithm, just like the LSTM(Long-Short Term Memory). It was proposed by Kyunghyun Cho, Bart van Merrienboer etc. in the article "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" in 2014. In this paper, it proposes a novel neural network model called RNN Encoder-Decoder that consists of two recurrent neural networks (RNN).To improve the effect of translation task, we also refer to "Sequence to Sequence Learning with Neural Networks" and "Neural Machine Translation by Jointly Learning to Align and Translate".

## Paper

1.[Paper](https://arxiv.org/abs/1406.1078): "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", 2014, Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio

2.[Paper](https://arxiv.org/pdf/1409.3215.pdf): "Sequence to Sequence Learning with Neural Networks", 2014, Ilya Sutskever, Oriol Vinyals, Quoc V. Le

3.[Paper](): "Neural Machine Translation by Jointly Learning to Align and Translate", 2014, Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio

# [Model Structure](#contents)

The GRU model mainly consists of an Encoder and a Decoder.The Encoder is constructed with a bidirection GRU cell.The Decoder mainly contains an attention and a GRU cell.The input of the net is sequence of words (text or sentence), and the output of the net is the probability of each word in vocab, and we choose the maximum probability one as our prediction.

# [Dataset](#contents)

In this model, we use the Multi30K dataset as our train and test dataset.As training dataset, it provides 29,000 respectively, each containing an German sentence and its English translation.For testing dataset, it provides 1000 German and English sentences.We also provide a preprocess script to tokenize the dataset and create the vocab file.

# [Environment Requirements](#content)

- Hardware（Ascend or GPU）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## Requirements

```txt
nltk
numpy
onnxruntime-gpu
```

To install nltk, you should install nltk as follow:

```bash
pip install nltk
```

Then you should download extra packages as follow:

```python
import nltk
nltk.download()
```

# [Quick Start](#content)

- Running on local with Ascend,GPU or CPU

    After dataset preparation, you can start training and evaluation as follows:

    ```bash
    cd ./scripts
    # download dataset
    bash download_dataset.sh

    # preprocess dataset
    bash preprocess.sh [DATASET_PATH]

    # create mindrecord
    bash create_dataset.sh [DATASET_PATH] [OUTPUT_PATH]

    # run training example
    bash run_standalone_train_{platform}.sh [TRAIN_DATASET_PATH] [DEVICE_ID]
    # platform: ascend or gpu
    python train.py --config_path=[CPU_CONFIG_PATH] --dataset_path=[TRAIN_DATASET_PATH]
    # platform: cpu

    # run distributed training example
    bash run_distribute_train_{platform}.sh [RANK_TABLE_FILE] [TRAIN_DATASET_PATH]
    # platform: ascend or gpu
    # do not need [RANK_TABLE_FILE] if you use GPU
    # do not need this step if you use CPU

    # run evaluation example
    bash run_eval_{platform}.sh [CKPT_FILE] [DATASET_PATH]
    # platform: ascend or gpu
    python eval.py --dataset_path=[DATASET_PATH] --ckpt_file=[CKPT_FILE] --device_target=CPU
    # platform: cpu
    ```

    After dataset preparation and training, you can run quick_start.py showing the results of training.

    ```bash
    # run quick_start.py
    python quick_start.py --dataset_path=[DATASET_PATH] --ckpt_file=[CKPT_FILE] --device_target=CPU
    # platform: cpu
    # example
    python quick_start.py --dataset_path=./data/mindrecord/multi30k_test_mindrecord_32 --ckpt_file=./ckpt_0/0-20_1807.ckpt --device_target=CPU
    ```

- Running on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    ```python
    # Train 8p on ModelArts
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "run_distribute=True" on default_config.yaml file.
    #          Set "dataset_path='/cache/data/mindrecord/multi30k_train_mindrecord_32_0'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "run_distribute=True" on the website UI interface.
    #          Add "dataset_path=/cache/data/mindrecord/multi30k_train_mindrecord_32_0" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (3) Set the code directory to "/path/gru" on the website UI interface.
    # (4) Set the startup file to "train.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    #
    # Train 1p on ModelArts
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "dataset_path='/cache/data/mindrecord/multi30k_train_mindrecord_32_0'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "dataset_path=/cache/data/mindrecord/multi30k_train_mindrecord_32_0" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (3) Set the code directory to "/path/gru" on the website UI interface.
    # (4) Set the startup file to "train.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    #
    # Eval 1p on ModelArts
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data/mindrecord/multi30k_train_mindrecord_32_0'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "ckpt_file=/cache/checkpoint_path/model.ckpt" on the website UI interface.
    #          Add "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface.
    #          Add "dataset_path=/cache/data/mindrecord/multi30k_train_mindrecord_32" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (3) Set the code directory to "/path/gru" on the website UI interface.
    # (4) Set the startup file to "eval.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

# [Script Description](#content)

The GRU network script and code result are as follows:

```text
├── gru
  ├── README.md                              // Introduction of GRU model.
  ├── model_utils
  │   ├──__init__.py                         // module init file
  │   ├──config.py                           // Parse arguments
  │   ├──device_adapter.py                   // Device adapter for ModelArts
  │   ├──local_adapter.py                    // Local adapter
  │   ├──moxing_adapter.py                   // Moxing adapter for ModelArts
  ├── src
  │   ├──create_data.py                      // Dataset preparation.
  │   ├──dataset.py                          // Dataset loader to feed into model.
  │   ├──gru_for_infer.py                    // GRU eval model architecture.
  │   ├──gru_for_train.py                    // GRU train model architecture.
  │   ├──loss.py                             // Loss architecture.
  │   ├──lr_schedule.py                      // Learning rate scheduler.
  │   ├──parse_output.py                     // Parse output file.
  │   ├──preprocess.py                       // Dataset preprocess.
  |   ├──rnn_cells.py                        // rnn cell architecture.
  |   ├──rnns.py                             // rnn layer architecture.
  │   ├──seq2seq.py                          // Seq2seq architecture.
  |   ├──utils.py                            // utils for rnn.
  │   ├──tokenization.py                     // tokenization for the dataset.
  │   ├──weight_init.py                      // Initialize weights in the net.
  ├── scripts
  │   ├──create_dataset.sh                   // shell script for create dataset.
  │   ├──download_dataset.sh                 // shell script for download dataset.
  │   ├──parse_output.sh                     // shell script for parse eval output file to calculate BLEU.
  │   ├──preprocess.sh                       // shell script for preprocess dataset.
  │   ├──run_distributed_train_ascend.sh     // shell script for distributed train on ascend.
  │   ├──run_distributed_train_gpu.sh        // shell script for distributed train on gpu.
  │   ├──run_eval_ascend.sh                  // shell script for standalone eval on ascend.
  │   ├──run_eval_gpu.sh                     // shell script for standalone eval on gpu.
  │   ├──run_eval_onnx_gpu.sh                // shell script for standalone onnx model eval on gpu.
  │   ├──run_infer_310.sh                    // shell script for 310 inference.
  │   ├──run_standalone_train_ascend.sh      // shell script for standalone eval on ascend.
  │   ├──run_standalone_train_gpu.sh         // shell script for standalone eval on gpu.
  ├── default_config.yaml                    // Configurations.
  ├── cpu_config.yaml                        // Configurations for cpu.
  ├── postprocess.py                         // GRU postprocess script.
  ├── preprocess.py                          // GRU preprocess script.
  ├── export.py                              // Export API entry.
  ├── eval.py                                // Infer API entry.
  ├── eval_onnx.py                           // ONNX infer API entry.
  ├── quick_start.py                         // GRU quickstart script.
  ├── requirements.txt                       // Requirements of third party package.
  ├── train.py                               // Train API entry.
```

## [Dataset Preparation](#content)

Firstly, we should download the dataset from the WMT16 official net.

```bash
cd scripts
bash download_dataset.sh
```

After downloading the Multi30k dataset file, we get six dataset file, which is show as below.And we should in put the in same directory.

```text
train.de
train.en
val.de
val.en
test.de
test.en
```

Then, we can use the scripts/preprocess.sh to tokenize the dataset file and get the vocab file.

```bash
bash preprocess.sh [DATASET_PATH]
```

After preprocess, we will get the dataset file which is suffix with ".tok" and two vocab file, which are nameed vocab.de and vocab.en.
Then we provided scripts/create_dataset.sh to create the dataset file which format is mindrecord.

```bash
bash create_dataset.sh [DATASET_PATH] [OUTPUT_PATH]
```

Finally, we will get multi30k_train_mindrecord_0 ~ multi30k_train_mindrecord_8 as our train dataset, and multi30k_test_mindrecord as our test dataset.

## [Configuration File](#content)

Parameters for both training and evaluation can be set in config.py. All the datasets are using same parameter name, parameters value could be changed according the needs.

- Network Parameters for Ascend and GPU

  ```text
    "batch_size": 16,                  # batch size of input dataset.
    "src_vocab_size": 8154,            # source dataset vocabulary size.
    "trg_vocab_size": 6113,            # target dataset vocabulary size.
    "encoder_embedding_size": 256,     # encoder embedding size.
    "decoder_embedding_size": 256,     # decoder embedding size.
    "hidden_size": 512,                # hidden size of gru.
    "max_length": 32,                  # max sentence length.
    "num_epochs": 30,                  # total epoch.
    "save_checkpoint": True,           # whether save checkpoint file.
    "ckpt_epoch": 1,                   # frequence to save checkpoint file.
    "target_file": "target.txt",       # the target file.
    "output_file": "output.txt",       # the output file.
    "keep_checkpoint_max": 30,         # the maximum number of checkpoint file.
    "base_lr": 0.001,                  # init learning rate.
    "warmup_step": 300,                # warmup step.
    "momentum": 0.9,                   # momentum in optimizer.
    "init_loss_scale_value": 1024,     # init scale sense.
    'scale_factor': 2,                 # scale factor for dynamic loss scale.
    'scale_window': 2000,              # scale window for dynamic loss scale.
    "warmup_ratio": 1/3.0,             # warmup ratio.
    "teacher_force_ratio": 0.5         # teacher force ratio.
  ```

- Network Parameters for Ascend and GPU

  ```text
    "batch_size": 16,                  # batch size of input dataset.
    "src_vocab_size": 8154,            # source dataset vocabulary size.
    "trg_vocab_size": 6113,            # target dataset vocabulary size.
    "encoder_embedding_size": 256,     # encoder embedding size.
    "decoder_embedding_size": 256,     # decoder embedding size.
    "hidden_size": 512,                # hidden size of gru.
    "max_length": 32,                  # max sentence length.
    "num_epochs": 13,                  # total epoch.
    "save_checkpoint": True,           # whether save checkpoint file.
    "ckpt_epoch": 1,                   # frequence to save checkpoint file.
    "target_file": "target.txt",       # the target file.
    "output_file": "output.txt",       # the output file.
    "keep_checkpoint_max": 5,          # the maximum number of checkpoint file.
    "base_lr": 0.001,                  # init learning rate.
    "warmup_step": 300,                # warmup step.
    "momentum": 0.9,                   # momentum in optimizer.
    "init_loss_scale_value": 1024,     # init scale sense.
    'scale_factor': 2,                 # scale factor for dynamic loss scale.
    'scale_window': 2000,              # scale window for dynamic loss scale.
    "warmup_ratio": 1/3.0,             # warmup ratio.
    "teacher_force_ratio": 0.5         # teacher force ratio.
  ```

## [Training Process](#content)

- Start task training on a single device.Run the shell script if you use ascend or gpu, and run the python file if you use cpu.

    ```bash
    cd ./scripts
    # platform: ascend or gpu
    bash run_standalone_train_{platform}.sh [DATASET_PATH] [DEVICE_ID]
    # example：
    bash run_standalone_train_ascend.sh /Muti30k/mindrecord/multi30k_train_mindrecord_32_0 0

    # platform: cpu
    python train.py --config_path=[CPU_CONFIG_PATH] --dataset_path=[TRAIN_DATASET_PATH] --device_target=CPU
    # example：
    python train.py --config_path=cpu_config.yaml --dataset_path=./data/mindrecord/multi30k_train_mindrecord_32_0 --device_target=CPU
    ```

- Running scripts for distributed training of GRU. Task training on multiple device and run the following command in bash to be executed in `scripts/`:

    ``` bash
    cd ./scripts
    bash run_distributed_train_{platform}.sh [RANK_TABLE_PATH] [DATASET_PATH]
    # platform: ascend or gpu
    # do not need [RANK_TABLE_FILE] if you use GPU
    # do not need this step if you use CPU
    ```

## [Inference Process](#content)

- Running scripts for evaluation of GRU. The command as below.

    ``` bash
    cd ./scripts
    # platform: ascend or gpu
    bash run_eval_{platform}.sh [CKPT_FILE] [DATASET_PATH]
    # example:
    bash run_eval_ascend.sh /data/ckpt_0/0-20_1807.ckpt /data/mindrecord/multi30k_test_mindrecord_32

    # platform: cpu
    python eval.py --dataset_path=[DATASET_PATH] --ckpt_file=[CKPT_FILE] --device_target=CPU
    # example:
    python eval.py --dataset_path=./data/mindrecord/multi30k_test_mindrecord_32 --ckpt_file=./ckpt_0/0-20_1807.ckpt --device_target=CPU
    ```

- After evaluation, we will get eval/target.txt and eval/output.txt.Then we can use parse_output.sh to get the translation.

    ``` bash
    cp eval/*.txt ./
    bash parse_output.sh target.txt output.txt /path/vocab.en
    ```

    Extra: We recommend doing this locally, but you can also do it on modelarts by running a python script with the following command "os.system("bash parse_output.sh target.txt output.txt /path/vocab.en")".

- After parse output, we will get target.txt.forbleu and output.txt.forbleu.To calculate BLEU score, you may use this [perl script](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) and run following command to get the BLEU score.

    ```bash
    perl multi-bleu.perl target.txt.forbleu < output.txt.forbleu
    ```

    Extra: We recommend doing this locally, but you can also do it on modelarts by running a python script with the following command "os.system("perl multi-bleu.perl target.txt.forbleu < output.txt.forbleu")".

Note: The `DATASET_PATH` is path to mindrecord. eg. train: /dataset_path/multi30k_train_mindrecord_0  eval: /dataset_path/multi30k_test_mindrecord

## [Export MindIR](#contents)

- Export on local

    ```python
    # The ckpt_file parameter is required, `EXPORT_FORMAT` should be in ["AIR", "MINDIR"]
    python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

    ```python
    # Eval 1p on ModelArts
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
    #          Set "file_name='./gru'" on default_config.yaml file.
    #          Set "file_format='MINDIR'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
    #          Add "file_name='./gru'" on the website UI interface.
    #          Add "file_format='MINDIR'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Set the code directory to "/path/gru" on the website UI interface.
    # (3) Set the startup file to "export.py" on the website UI interface.
    # (4) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (5) Create your job.
    ```

## [ONNX Export And Evaluation](#contents)

### ONNX Export

```bash
python export.py --device_target="GPU" --file_format="ONNX" --ckpt_file [CKPT_PATH]
# example:python export.py --device_target="GPU" --file_format="ONNX" --ckpt_file models/official/nlp/GRU/0-25_1807.ckpt
```

### ONNX Evaluation

- Running scripts for onnx evaluation of GRU. The command as below.

    ``` bash
    cd ./scripts
    bash run_eval_onnx_gpu.sh [ONNX_CKPT_FILE] [DATASET_PATH]
    # platform: gpu
    # example:
    bash run_eval_onnx_gpu.sh gru.onnx /data/mindrecord/multi30k_test_mindrecord_32
    ```

- After evaluation, we will get eval/target.txt and eval/output.txt.Then we can use parse_output.sh to get the translation.

    ``` bash
    cp eval/*.txt ./
    bash parse_output.sh target.txt output.txt /path/vocab.en
    ```

- After parse output, we will get target.txt.forbleu and output.txt.forbleu. To calculate BLEU score, you may use this [perl script](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) and run following command to get the BLEU score.

    ```bash
    perl multi-bleu.perl target.txt.forbleu < output.txt.forbleu
    ```

## [Inference Process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Usage

Before performing inference, the mindir file must be exported by export.py. Input files must be in bin format.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TYPE] [DEVICE_ID]
```

`NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
`DEVICE_ID` is optional, default value is 0.

### result

we will get target.txt and output.txt.Then we can use parse_output.sh to get the translation.

``` bash
bash parse_output.sh target.txt output.txt /path/vocab.en
```

After parse output, we will get target.txt.forbleu and output.txt.forbleu.To calculate BLEU score, you may use this [perl script](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) and run following command to get the BLEU score.

```bash
perl multi-bleu.perl target.txt.forbleu < output.txt.forbleu
```

# [Model Description](#content)

## [Performance](#content)

### Training Performance

| Parameters                 | Ascend                        | GPU                       | CPU |
| -------------------------- | ----------------------------- |---------------------------| -------------------------- |
| Resource                   | Ascend 910; OS Euler2.8       | GTX1080Ti, Ubuntu 18.04   | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz,Ubuntu 18.04 |
| uploaded Date              | 06/05/2021 (month/day/year)   | 06/05/2021 (month/day/year) | 09/28/2022 (month/day/year) |
| MindSpore Version          | 1.2.0                         |1.2.0                      | 1.2.0 |
| Dataset                    | Multi30k Dataset              | Multi30k Dataset          | Multi30k Dataset |
| Training Parameters        | epoch=30, batch_size=16       | epoch=30, batch_size=16   | epoch=13, batch_size=16 |
| Optimizer                  | Adam                          | Adam                      | Adam |
| Loss Function              | NLLLoss                       | NLLLoss                   | NLLLoss |
| outputs                    | probability                   | probability               | probability |
| Speed                      | 35ms/step (1pcs)              | 200ms/step (1pcs)         | 1465ms/step (1pcs) |
| Epoch Time | 64.4s (1pcs)                                                 | 361.5s (1pcs) | 2640s (1pcs) |
| Loss | 3.86888 | 2.533958 | 2.9340835 |
| Params (M) | 21 | 21 | 21 |
| Checkpoint for inference | 272M (.ckpt file) | 272M (.ckpt file) | 321M(.ckpt file) |
| Scripts | [gru](https://gitee.com/mindspore/models/tree/master/official/nlp/GRU) | [gru](https://gitee.com/mindspore/models/tree/master/official/nlp/GRU) | [gru](https://gitee.com/mindspore/models/tree/master/official/nlp/GRU) |

### Inference Performance

| Parameters          | Ascend                      | GPU | CPU |
| ------------------- | --------------------------- |---------------------------| ------------------- |
| Resource            | Ascend 910; OS Euler2.8     | GTX1080Ti, Ubuntu 18.04   | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz,Ubuntu 18.04 |
| Uploaded Date       | 06/05/2021 (month/day/year) | 06/05/2021 (month/day/year)| 09/28/2022 (month/day/year) |
| MindSpore Version   | 1.2.0                       | 1.2.0                     | 1.2.0 |
| Dataset             | Multi30K                    | Multi30K                  | Multi30K |
| batch_size          | 1                           | 1                         | 1 |
| outputs             | label index                 | label index               | label index |
| Accuracy            | BLEU: 31.26                 | BLEU: 29.30               | BLEU: 30.19 |
| Model for inference | 272M (.ckpt file)           | 272M (.ckpt file)         | 321M(.ckpt file) |

# [Random Situation Description](#content)

There only one random situation.

- Initialization of some model weights.

Some seeds have already been set in train.py to avoid the randomness of weight initialization.

# [Others](#others)

This model has been validated in the Ascend environment and is not validated on the CPU and GPU.

# [ModelZoo HomePage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models)
