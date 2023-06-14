# FastText

<!-- TOC -->

- [FastText](#fasttext)
- [Model Structure](#model-structure)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Dataset Preparation](#dataset-preparation)
    - [Configuration File](#configuration-file)
    - [Training Process](#training-process)
    - [Inference Process](#inference-process)
    - [ONNX Export And Evaluation](#onnx-export-and-evaluation)
        - [ONNX Export](#onnx-export)
        - [ONNX Evaluation](#onnx-evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Random Situation Description](#random-situation-description)
- [Others](#others)
- [ModelZoo HomePage](#modelzoo-homepage)

<!-- /TOC -->

## [FastText](#contents)

FastText is a fast text classification algorithm, which is simple and efficient. It was proposed by Armand
Joulin, Tomas Mikolov etc. in the article "Bag of Tricks for Efficient Text Classification" in 2016. It is similar to
CBOW in model architecture, where the middle word is replace by a label. FastText adopts ngram feature as addition feature
to get some information about words. It speeds up training and testing while maintaining high precision, and widly used
in various tasks of text classification.

[Paper](https://arxiv.org/pdf/1607.01759.pdf): "Bag of Tricks for Efficient Text Classification", 2016, A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov

## [Model Structure](#contents)

The FastText model mainly consists of an input layer, hidden layer and output layer, where the input is a sequence of words (text or sentence).
The output layer is probability that the words sequence belongs to different categories. The hidden layer is formed by average of multiple word vector.
The feature is mapped to the hidden layer through linear transformation, and then mapped to the label from the hidden layer.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network
architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- [AG's news topic classification dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- [DBPedia Ontology Classification Dataset](https://emilhvitfeldt.github.io/textdata/reference/dataset_dbpedia.html)
- [Yelp Review Polarity Dataset](https://www.kaggle.com/datasets/irustandi/yelp-review-polarity)

## [Environment Requirements](#content)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [Quick Start](#content)

After dataset preparation, you can start training and evaluation as follows:

- Running on Ascend

    ```bash
    # run training example
    cd ./scripts
    bash run_standalone_train.sh [TRAIN_DATASET] [DEVICEID] [DATANAME]

    # run evaluation example
    bash run_eval.sh [EVAL_DATASET_PATH] [DATASET_NAME] [MODEL_CKPT] [DEVICEID]
    ```

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    ```python
    # run standalone training example
    # (1) Add "config_path='/path_to_code/[DATASET_NAME]_config.yaml'" on the website UI interface.
    # (2) Perform a or b.
    #       a. Set "enable_modelarts=True" on [DATASET_NAME]_config.yaml file.
    #          Set "dataset_path='/cache/data/[DATASET_NAME]'" on [DATASET_NAME]_config.yaml file.
    #          Set "data_name='[DATASET_NAME]'" on [DATASET_NAME]_config.yaml file.
    #          (option)Set "device_target=GPU" on [DATASET_NAME]_config.yaml file if run with GPU.
    #          (option)Set other parameters on [DATASET_NAME]_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "dataset_path='/cache/data/[DATASET_NAME]'" on the website UI interface.
    #          Add "data_name='[DATASET_NAME]'" on the website UI interface.
    #          (option)Set "device_target=GPU" on the website UI interface if run with GPU.
    #          (option)Set other parameters on the website UI interface.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/fasttext" on the website UI interface.
    # (5) Set the startup file to "train.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    #
    # run evaluation example
    # (1) Add "config_path='/path_to_code/[DATASET_NAME]_config.yaml'" on the website UI interface.
    # (2) Perform a or b.
    #       a. Set "enable_modelarts=True" on [DATASET_NAME]_config.yaml file.
    #          Set "dataset_path='/cache/data/[DATASET_NAME]'" on [DATASET_NAME]_config.yaml file.
    #          Set "data_name='[DATASET_NAME]'" on [DATASET_NAME]_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on [DATASET_NAME]_config.yaml file.
    #          Set "model_ckpt='/cache/checkpoint_path/model.ckpt'" on [DATASET_NAME]_config.yaml file.
    #          (option)Set "device_target=GPU" on [DATASET_NAME]_config.yaml file if run with GPU.
    #          (option)Set other parameters on [DATASET_NAME]_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "dataset_path='/cache/data/[DATASET_NAME]'" on the website UI interface.
    #          Add "data_name='[DATASET_NAME]'" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
    #          Add "model_ckpt='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          (option)Set "device_target=GPU" on the website UI interface if run with GPU.
    #          (option)Set other parameters on the website UI interface.
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (5) Set the code directory to "/path/fasttext" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # run export example
    # (1) Add "config_path='/path_to_code/[DATASET_NAME]_config.yaml'" on the website UI interface.
    # (2) Perform a or b.
    #       a. Set "enable_modelarts=True" on [DATASET_NAME]_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on [DATASET_NAME]_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on [DATASET_NAME]_config.yaml file.
    #          Set other parameters on [DATASET_NAME]_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (5) Set the code directory to "/path/fasttext" on the website UI interface.
    # (6) Set the startup file to "export.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

## [Script Description](#content)

The FastText network script and code result are as follows:

```text
├── fasttext
  ├── README.md                              // Introduction of FastText model.
  ├── model_utils
  │   ├──__init__.py                        // module init file
  │   ├──config.py                          // Parse arguments
  │   ├──device_adapter.py                  // Device adapter for ModelArts
  │   ├──local_adapter.py                   // Local adapter
  │   ├──moxing_adapter.py                  // Moxing adapter for ModelArts
  ├── src
  │   ├──create_dataset.py                   // Dataset preparation.
  │   ├──fasttext_model.py                   // FastText model architecture.
  │   ├──fasttext_train.py                   // Use FastText model architecture.
  │   ├──load_dataset.py                     // Dataset loader to feed into model.
  │   ├──lr_scheduler.py                     // Learning rate scheduler.
  ├── scripts
  │   ├──create_dataset.sh                   // shell script for Dataset preparation.
  │   ├──run_eval.sh                         // shell script for standalone eval on ascend.
  │   ├──run_standalone_train.sh             // shell script for standalone eval on ascend.
  │   ├──run_distribute_train_gpu.sh        // shell script for distributed train on GPU.
  │   ├──run_eval_gpu.sh                     // shell script for standalone eval on GPU.
  │   ├──run_eval_onnx_gpu.sh                // shell script for standalone eval_onnx on GPU.
  │   ├──run_standalone_train_gpu.sh         // shell script for standalone train on GPU.
  ├── ag_config.yaml                         // ag dataset arguments
  ├── dbpedia_config.yaml                    // dbpedia dataset arguments
  ├── yelp_p_config.yaml                      // yelpp dataset arguments
  ├── mindspore_hub_conf.py                  // mindspore hub scripts
  ├── export.py                              // Export API entry.
  ├── eval.py                                // Infer API entry.
  ├── eval_onnx.py                           // Infer onnx API entry.
  ├── requirements.txt                       // Requirements of third party package.
  ├── train.py                               // Train API entry.
```

### [Dataset Preparation](#content)

- Download the AG's News Topic Classification Dataset, DBPedia Ontology Classification Dataset and Yelp Review Polarity Dataset. Unzip datasets to any path you want.

- Run the following scripts to do data preprocess and convert the original data to mindrecord for training and evaluation.

    ``` bash
    cd scripts
    bash creat_dataset.sh [SOURCE_DATASET_PATH] [DATASET_NAME]
    ```

    example:bash create_dataset.sh your_path/fasttext/dataset/ag_news_csv ag
### [Configuration File](#content)

Parameters for both training and evaluation can be set in config.py. All the datasets are using same parameter name, parameters value could be changed according the needs.

- Network Parameters

    ```text
      vocab_size               # vocabulary size.
      buckets                  # bucket sequence length.
      test_buckets             # test dataset bucket sequence length
      batch_size               # batch size of input dataset.
      embedding_dims           # The size of each embedding vector.
      num_class                # number of labels.
      epoch                    # total training epochs.
      lr                       # initial learning rate.
      min_lr                   # minimum learning rate.
      warmup_steps             # warm up steps.
      poly_lr_scheduler_power  # a value used to calculate decayed learning rate.
      pretrain_ckpt_dir        # pretrain checkpoint direction.
      keep_ckpt_max            # Max ckpt files number.
    ```

### [Training Process](#content)

- Running on Ascend

    - Start task training on a single device and run the shell script

        ```bash
        cd ./scripts
        bash run_standalone_train.sh [DATASET_PATH] [DEVICE_ID] [DATANAME]
        ```

- Running on GPU

    - Start task training on a single device and run the shell script

        ```bash
        cd ./scripts
        bash run_standalone_train_gpu.sh [DATASET_PATH] [DATANAME]
        ```

### [Inference Process](#content)

- Running on Ascend

    - Running scripts for evaluation of FastText. The commdan as below.

        ```bash
        cd ./scripts
        bash run_eval.sh [DATASET_PATH] [DATASET_NAME] [MODEL_CKPT]
        ```

  Note: The `DATASET_PATH` is path to mindrecord. eg. `/dataset_path/`

- Running on GPU

    - Running scripts for evaluation of FastText. The commdan as below.

        ```bash
        cd ./scripts
        bash run_eval_gpu.sh [DATASET_PATH] [DATASET_NAME] [MODEL_CKPT]
        ```

  Note: The `DATASET_PATH` is path to mindrecord. eg. `/dataset_path/`

### [ONNX Export And Evaluation](#content)

Note that run all onnx concerned scripts on GPU.

- ONNX Export

    - The command below will produce lots of fasttext onnx files, named different input shapes due to different input shapes of evaluation data.

        ```bash
        python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [ONNX] --config_path [CONFIG_PATH]
        --onnx_path [ONNX_PATH] --dataset_path [DATASET_PATH]
        ```

    example:python export_onnx.py --ckpt_file ./checkpoint/fasttext_ascend_v170_dbpedia_official_nlp_acc98.62.ckpt --file_name fasttext --file_format ONNX --config_path ./dbpedia_config.yaml
    --onnx_path ./fasttext --dataset_path ./fasttext/scripts/ag/

- ONNX Evaluation
    - Note that ONNX_PATH should be the absolute directory to the exported onnx files, such as: '/home/mindspore/ls/models/research/nlp/fasttext'.

        ```bash
        cd ./scripts
        bash run_eval_onnx_gpu.sh DATASET_PATH DATASET_NAME ONNX_PATH
        ```

    example:bash run_eval_onnx_gpu.sh /home/mindspore/fasttext/scripts/dbpedia/ dbpedia  /home/mindspore/research/nlp/fasttext

You can view the results through the file "eval_onnx.log".

## [Model Description](#content)

### [Performance](#content)

#### Training Performance

| Parameters               | Ascend                                                       | GPU                                                          |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Resource                 | Ascend 910; OS Euler2.8                                                 | NV SMX3 V100-32G                                             |
| uploaded Date            | 12/21/2020 (month/day/year)                                  | 1/29/2021 (month/day/year)                                   |
| MindSpore Version        | 1.1.0                                                        | 1.1.0                                                        |
| Dataset                  | AG's News Topic Classification Dataset                       | AG's News Topic Classification Dataset                       |
| Training Parameters      | epoch=5, batch_size=512                                      | epoch=5, batch_size=512                                      |
| Optimizer                | Adam                                                         | Adam                                                         |
| Loss Function            | Softmax Cross Entropy                                        | Softmax Cross Entropy                                        |
| outputs                  | probability                                                  | probability                                                  |
| Speed                    | 10ms/step (1pcs)                                             | 11.91ms/step(1pcs)                                           |
| Epoch Time               | 2.36s (1pcs)                                                 | 2.815s(1pcs)                                                 |
| Loss                     | 0.0067                                                       | 0.0085                                                       |
| Params (M)               | 22                                                           | 22                                                           |
| Checkpoint for inference | 254M (.ckpt file)                                            | 254M (.ckpt file)                                            |
| Scripts                  | [fasttext](https://gitee.com/mindspore/models/tree/r2.0/research/nlp/fasttext) | [fasttext](https://gitee.com/mindspore/models/tree/r2.0/research/nlp/fasttext) |

| Parameters               | Ascend                                                       | GPU                                                          |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Resource                 | Ascend 910; OS Euler2.8                                                  | NV SMX3 V100-32G                                             |
| uploaded Date            | 11/21/2020 (month/day/year)                                  | 1/29/2020 (month/day/year)                                   |
| MindSpore Version        | 1.1.0                                                        | 1.1.0                                                        |
| Dataset                  | DBPedia Ontology Classification Dataset                      | DBPedia Ontology Classification Dataset                      |
| Training Parameters      | epoch=5, batch_size=4096                                     | epoch=5, batch_size=4096                                     |
| Optimizer                | Adam                                                         | Adam                                                         |
| Loss Function            | Softmax Cross Entropy                                        | Softmax Cross Entropy                                        |
| outputs                  | probability                                                  | probability                                                  |
| Speed                    | 58ms/step (1pcs)                                             | 34.82ms/step(1pcs)                                           |
| Epoch Time               | 8.15s (1pcs)                                                 | 4.87s(1pcs)                                                  |
| Loss                     | 2.6e-4                                                       | 0.0004                                                       |
| Params (M)               | 106                                                          | 106                                                          |
| Checkpoint for inference | 1.2G (.ckpt file)                                            | 1.2G (.ckpt file)                                            |
| Scripts                  | [fasttext](https://gitee.com/mindspore/models/tree/r2.0/research/nlp/fasttext) | [fasttext](https://gitee.com/mindspore/models/tree/r2.0/research/nlp/fasttext) |

| Parameters               | Ascend                                                       | GPU                                                          |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Resource                 | Ascend 910; OS Euler2.8                                                 | NV SMX3 V100-32G                                             |
| uploaded Date            | 11/21/2020 (month/day/year)                                  | 1/29/2020 (month/day/year)                                   |
| MindSpore Version        | 1.1.0                                                        | 1.1.0                                                        |
| Dataset                  | Yelp Review Polarity Dataset                                 | Yelp Review Polarity Dataset                                 |
| Training Parameters      | epoch=5, batch_size=2048                                     | epoch=5, batch_size=2048                                     |
| Optimizer                | Adam                                                         | Adam                                                         |
| Loss Function            | Softmax Cross Entropy                                        | Softmax Cross Entropy                                        |
| outputs                  | probability                                                  | probability                                                  |
| Speed                    | 101ms/step (1pcs)                                            | 30.54ms/step(1pcs)                                           |
| Epoch Time               | 28s (1pcs)                                                   | 8.46s(1pcs)                                                  |
| Loss                     | 0.062                                                        | 0.002                                                        |
| Params (M)               | 103                                                          | 103                                                          |
| Checkpoint for inference | 1.2G (.ckpt file)                                            | 1.2G (.ckpt file)                                            |
| Scripts                  | [fasttext](https://gitee.com/mindspore/models/tree/r2.0/research/nlp/fasttext) | [fasttext](https://gitee.com/mindspore/models/tree/r2.0/research/nlp/fasttext) |

#### Inference Performance

| Parameters          | Ascend                      | GPU |
| ------------------- | --------------------------- | ------------------- |
| Resource            | Ascend 910; OS Euler2.8                 | NV SMX3 V100-32G |
| Uploaded Date       | 12/21/2020 (month/day/year) | 1/29/2020 (month/day/year) |
| MindSpore Version   | 1.1.0                       | 1.1.0 |
| Dataset             | AG's News Topic Classification Dataset            | AG's News Topic Classification Dataset |
| batch_size          | 512                         | 128 |
| Epoch Time          | 2.36s                       | 2.815s(1pcs) |
| outputs             | label index                 | label index |
| Accuracy            | 92.53                        | 92.58 |
| Model for inference | 254M (.ckpt file)           | 254M (.ckpt file) |

| Parameters          | Ascend                      | GPU |
| ------------------- | --------------------------- | ------------------- |
| Resource            | Ascend 910; OS Euler2.8                | NV SMX3 V100-32G |
| Uploaded Date       | 12/21/2020 (month/day/year) | 1/29/2020 (month/day/year) |
| MindSpore Version   | 1.1.0                       | 1.1.0 |
| Dataset             | DBPedia Ontology Classification Dataset            | DBPedia Ontology Classification Dataset |
| batch_size          | 4096                         | 4096 |
| Epoch Time          | 8.15s                          | 4.87s |
| outputs             | label index                 | label index |
| Accuracy            | 98.6                        | 98.49 |
| Model for inference | 1.2G (.ckpt file)           | 1.2G (.ckpt file) |

| Parameters          | Ascend                      | GPU |
| ------------------- | --------------------------- | ------------------- |
| Resource            | Ascend 910; OS Euler2.8                 | NV SMX3 V100-32G |
| Uploaded Date       | 12/21/2020 (month/day/year) | 12/29/2020 (month/day/year) |
| MindSpore Version   | 1.1.0                       | 1.1.0 |
| Dataset             | Yelp Review Polarity Dataset            | Yelp Review Polarity Dataset |
| batch_size          | 2048                         | 2048 |
| Epoch Time          | 28s                         | 8.46s |
| outputs             | label index                 | label index |
| Accuracy            | 95.7                        | 95.7 |
| Model for inference | 1.2G (.ckpt file)           | 1.2G (.ckpt file) |

## [Random Situation Description](#content)

There only one random situation.

- Initialization of some model weights.

Some seeds have already been set in train.py to avoid the randomness of weight initialization.

## [Others](#others)

This model has been validated in the `context.GRAPH_MODE` on Ascend environment and is not validated on the CPU and GPU.

This model can not be run in `context.PYNATIVE_MODE`.

## [ModelZoo HomePage](#contents)

Note: This model will be move to the `/models/research/` directory in r1.8.

Please check the official [homepage](https://gitee.com/mindspore/models)
