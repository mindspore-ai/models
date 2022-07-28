# Contents

- [Contents](#contents)
- [TransX Models Description](#transx-models-description)
- [Models architecture](#models-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage)
            - [Launch](#launch)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Launch](#launch-1)
            - [Result](#result)
    - [Model Export](#model-export)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
            - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [TransX Models Description](#contents)

TransE, TransH, TransR, TransD are models for Knowledge Graph Embeddings. The "Knowledge" for this model is represented
as a triple (head, relation, tail) where the head and tail are entities.

The basic idea of the **TransE** model is making the sum of the head vector and relation vector as close as possible with the tail vector.
The distance is calculated using L1 or L2 norm. The loss function used for training this model is the margin loss
calculated over scores for positive and negative samples.
The negative sampling is performed by replacing head or tail entities in the original triple. This model is good for managing one-to-one relations.

**TransH** allows us to tackle the problem of one-to-many, many-to-one and many-to-many relations.
Its basic idea is to reinterpret relations as the translations on a hyperplane.

The idea of **TransR** is that the entity and relations can have different semantic spaces.
It uses the trainable projection matrix to project the entities into the multi-relational space.
It also has some shortages. For example, the projection matrix is determined only by the relation, and the
heads and tails are assumed to be from the same semantic space. Moreover, the **TransR** model has a much larger number of parameters,
which is not suitable for large-scale tasks.

**TransD** compensates for the flaws of the **TransR** model by using the dynamic mapping of the heads and tails entities.
The projection matrices for heads and tails are calculated from the head-relation and tail-relation pairs correspondingly.

- [Paper TransE](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
  Translating Embeddings for Modeling Multi-relational Data（2013)
- [Paper TransH](https://persagen.com/files/misc/wang2014knowledge.pdf)
  Knowledge Graph Embedding by Translating on Hyperplanes（2014)
- [Paper TransR (download)](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwicz7i6hvfzAhVEmYsKHR8qClYQFnoECAQQAQ&url=https%3A%2F%2Fwww.aaai.org%2Focs%2Findex.php%2FAAAI%2FAAAI15%2Fpaper%2Fdownload%2F9571%2F9523&usg=AOvVaw07cpMPMew9IF8Yn5iZDvCu)
  Learning Entity and Relation Embeddings for Knowledge Graph Completion（2015
- [Paper TransD](https://aclanthology.org/P15-1067.pdf)
  Knowledge Graph Embedding via Dynamic Mapping Matrix（2015

# [Models architecture](#contents)

The base elements of all models are trainable lookup tables for entities and relations which produce the embeddings.

# [Dataset](#contents)

We use Wordnet and Fresbase datasets for training the models.

The preprocessed data files are available here:

- [WN18RR (Wordnet)](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/WN18RR)
    - Size: 3.7 MB
    - Number of entities: 40943
    - Number of relations: 11
    - Number of train triplets: 86835
    - Number of test triplets: 3134
- [FB15K237 (Freebase)](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/FB15K237)
    - Size: 5.5 MB
    - Number of entities: 14541
    - Number of relations: 237
    - Number of train triplets: 272115
    - Number of test triplets: 28466

# [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```text
./transX
├── configs  # models configuration files
│   ├── default_config.yaml
│   ├── transD_fb15k237_1gpu_config.yaml
│   ├── transD_fb15k237_8gpu_config.yaml
│   ├── transD_wn18rr_1gpu_config.yaml
│   ├── transD_wn18rr_8gpu_config.yaml
│   ├── transE_fb15k237_1gpu_config.yaml
│   ├── transE_fb15k237_8gpu_config.yaml
│   ├── transE_wn18rr_1gpu_config.yaml
│   ├── transE_wn18rr_8gpu_config.yaml
│   ├── transH_fb15k237_1gpu_config.yaml
│   ├── transH_fb15k237_8gpu_config.yaml
│   ├── transH_wn18rr_1gpu_config.yaml
│   ├── transH_wn18rr_8gpu_config.yaml
│   ├── transR_fb15k237_1gpu_config.yaml
│   ├── transR_fb15k237_8gpu_config.yaml
│   ├── transR_wn18rr_1gpu_config.yaml
│   └── transR_wn18rr_8gpu_config.yaml
├── model_utils  # Model Arts utilities
│   ├── config.py
│   ├── device_adapter.py
│   ├── __init__.py
│   ├── local_adapter.py
│   └── moxing_adapter.py
├── scripts  # Shell scripts for training and evaluation
│   ├── run_distributed_train_gpu.sh
│   ├── run_eval_gpu.sh
│   ├── run_export_gpu.sh
│   └── run_standalone_train_gpu.sh
├── src
│   ├── base  # C++ backend code for the dataset
│   │   ├── Base.cpp
│   │   ├── CMakeLists.txt
│   │   ├── Corrupt.h
│   │   ├── Random.h
│   │   ├── Reader.h
│   │   ├── Setting.h
│   │   └── Triple.h
│   ├── dataset_lib  # Compiled dataset tools
│   │   └── train_dataset_lib.so
│   ├── utils
│   │   └── logging.py  # Logging utilities
│   ├── dataset.py
│   ├── loss.py
│   ├── make.sh
│   ├── metric.py
│   ├── model_builder.py  # Convenient scripts building models
│   ├── trans_x.py  # Models definitions
│   └── __init__.py
├── eval.py  # Script for evaluation of the trained model
├── export.py  # Script for exporting the trained model
├── requirements.txt  # Additional dependencies
├── train.py  # Script for start the training process
└── README.md  # Documentation in English
```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be provided via a \*.yaml configuration files
or by directly providing the arguments to the train.py, eval.py and export.y scripts.

```yaml
device_target: "GPU"         # tested with GPUs only
is_train_distributed: False  # Whether to use the NCCL for multi-GPU training
group_size: 1                # Number of the devices
device_id: 0                 # Device ID (only for a single GPU training)
seed: 1                      # Random seed

# Model options
model_name: "TransE"         # Name of the model (TransE / TransH / TransR / TransD)
dim_e: 50                    # Embeddings size for entities
dim_r: 50                    # Embeddings size for relations

# Dataset options
dataset_root: "/path/to/dataset/root"
train_triplet_file_name: "train2id.txt"
eval_triplet_file_name: "test2id.txt"
filter_triplets_files_names:  # Files with positive triplets samples
  - "train2id.txt"
  - "valid2id.txt"
  - "test2id.txt"
entities_file_name: "entity2id.txt"
relations_file_name: "relation2id.txt"
negative_sampling_rate: 1    # The number of negative samples per a single positive sample.
train_batch_size: 868

# Logging options
train_output_dir: "train-outputs/"
eval_output_dir: "eval-output/"
export_output_dir: "export-output/"
ckpt_save_interval: 5
ckpt_save_on_master_only: True
keep_checkpoint_max: 10
log_interval: 100

# Training options
pre_trained: ""              # Path to the pre-trained model (necessary for TransR)
lr: 0.5                      # Learning rate
epochs_num: 1000             # Number of epochs
weight_decay: 0.0            # Weight decay
margin: 6.0                  # Parameters of the Margin loss
train_use_data_sink: False

# Evaluation and export options
ckpt_file: "/path/to/trained/checkpoint"
file_format: "MINDIR"
eval_use_data_sink: False
export_batch_size: 1000      # The batch size of the exported model
```

## [Training Process](#contents)

### Before training

You need to compile the library for generating the corrupted triplets.

The SOTA implementation uses triplets filtering to ensure that the corrupted triplets are actually not presented among the original triplets.
This filtering process is difficult to vectorize in order to effectively implement in in Python, so we use our custom **\*.so** library.

To build the library go to the **./transX/src** directory and run

```shell script
bash make.sh
```

After build is successfully finished, **train_dataset_lib.so** appears in **./transX/src/dataset_lib**.

### Usage

You can start the single GPU training process by running the python script:

- Without pre-trained model

  ```shell script
  python train.py --config_path=/parth/to/model_config.yaml --dataset_root=/path/to/dataset
  ```

- With pre-trained model

  ```shell script
  python train.py --config_path=/parth/to/model_config.yaml --dataset_root=/path/to/dataset --pre_trained=/path/to/pretrain.ckpt
  ```

or by running the shell script:

- Without pre-trained model

  ```shell script
  bash scripts/run_standalone_train_gpu.sh [DATASET_ROOT] [DATASET_NAME] [MODEL_NAME]
  ```

- With pre-trained model

  ```shell script
  bash scripts/run_standalone_train_gpu.sh [DATASET_ROOT] [DATASET_NAME] [MODEL_NAME] [PRETRAIN_CKPT]
  ```

You can start the 8-GPU training by running the following shell script

- Without pre-trained model

  ```shell script
  bash scripts/run_distributed_train_gpu.sh [DATASET_ROOT] [DATASET_NAME] [MODEL_NAME]
  ```

- With pre-trained model

  ```shell script
  bash scripts/run_distributed_train_gpu.sh [DATASET_ROOT] [DATASET_NAME] [MODEL_NAME] [PRETRAIN_CKPT]
  ```

> DATASET_NAME must be "wn18rr" or "fb15k237"
>
> MODEL_NAME must be "transE", "transH", "transR" or "transD"
>
> Using this names the corresponding configuration file in ./configs directory will be selected.

The train results will be stored in the **./train-outputs** directory.
If shell scripts are used, the logged information will be redirected to the **./train-logs** directory.

## [Evaluation Process](#contents)

### Usage

You can start evaluation by running the following python script:

```shell script
python eval.py --config_path=/parth/to/model_config.yaml --dataset_root=/path/to/dataset --ckpt_file=/path/to/trained.ckpt
```

or shell script:

```shell script
bash scripts/run_eval_gpu.sh [DATASET_ROOT] [DATASET_NAME] [MODEL_NAME] [CKPT_PATH]
```

> DATASET_NAME must be "wn18rr" or "fb15k237"
>
> MODEL_NAME must be "transE", "transH", "transR" or "transD"
>
> Using this names the corresponding configuration file in ./configs directory will be selected.

#### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.

The evaluation results will be stored in the **./eval-output** directory.
If the shell script is used, the logged information will be redirected to the **./eval-logs** directory.

Example of the evaluation output:

```text
...
[DATE/TIME]:INFO:start evaluation
[DATE/TIME]:INFO:evaluation finished
[DATE/TIME]:INFO:Result: hit@10 = 0.5056 hit@3 = 0.3623 hit@1 = 0.0490
```

## [Model Export](#contents)

You can export the model by running the following python script:

```shell script
python export.py --config_path=/parth/to/model_config.yaml --dataset_root=/path/to/dataset --ckpt_file=/path/to/trained.ckpt
```

or by running the shell script:

```shell script
bash scripts/run_export_gpu.sh [DATASET_ROOT] [DATASET_NAME] [MODEL_NAME] [CKPT_PATH]
```

> DATASET_NAME must be "wn18rr" or "fb15k237"
>
> MODEL_NAME must be "transE", "transH", "transR" or "transD"
>
> Using this names the corresponding configuration file in ./configs directory will be selected.

The tested formats for export are: **MINDIR**.

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

> For training the TransR models we used corresponding trained TransE models!
> You need to train TransE models first in order to get the better performance of the TransR model.

**1 GPU Training**

| Parameters                 |            |            |            |            |            |            |            |            |
| -------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Resource                   | 1x V100    | 1x V100    | 1x V100    | 1x V100    | 1x V100    | 1x V100    | 1x V100    | 1x V100    |
| uploaded Date (mm/dd/yyy)  | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 |
| MindSpore Version          | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      |
| Model                      | TransE     | TransH     | TransR     | TransD     | TransE     | TransH     | TransR     | TransD     |
| Dataset                    | Wordnet    | Wordnet    | Wordnet    | Wordnet    | Freebase   | Freebase   | Freebase   | Freebase   |
| Batch size                 | 868        | 868        | 868        | 868        | 2721       | 2721       | 453        | 2721       |
| Learning rate              | 0.5        | 0.5        | 0.05       | 0.5        | 1          | 0.5        | 0.16667    | 1          |
| Epochs                     | 1000       | 300        | 250        | 200        | 1000       | 1000       | 1000       | 1000       |
| Accuracy (Hit@10)          | 0.511      | 0.504      | 0.516      | 0.508      | 0.476      | 0.481      | 0.509      | 0.483      |
| Total time                 | 3m 0s      | 1m 22s     | 1m 10s     | 1m         | 19m 32s    | 34m 21s    | 7h 34m 16s | 33m 22s    |

**8 GPU Training**

| Parameters                 |            |            |            |            |            |            |            |            |
| -------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Resource                   | 8x V100    | 8x V100    | 8x V100    | 8x V100    | 8x V100    | 8x V100    | 8x V100    | 8x V100    |
| uploaded Date (mm/dd/yyy)  | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 |
| MindSpore Version          | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      |
| Model                      | TransE     | TransH     | TransR     | TransD     | TransE     | TransH     | TransR     | TransD     |
| Dataset                    | Wordnet    | Wordnet    | Wordnet    | Wordnet    | Freebase   | Freebase   | Freebase   | Freebase   |
| Batch size                 | 868        | 868        | 868        | 868        | 2721       | 2721       | 453        | 2721       |
| Learning rate              | 0.5        | 0.5        | 0.05       | 0.5        | 8          | 4          | 1.3333     | 8          |
| Epochs                     | 1000       | 300        | 250        | 200        | 1000       | 1000       | 1000       | 1000       |
| Accuracy (Hit@10)          | 0.511      | 0.507      | 0.512      | 0.514      | 0.475      | 0.483      | 0.509      | 0.481      |
| Total time                 | 1m 17s     | 31s        | 27s        | 52s        | 3m 51s     | 5m 41s     | 1h 24m 18s | 6m 32s     |

#### Inference Performance

| Parameters                |            |            |            |            |            |            |            |            |
| ------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Resource                  | GPU V100   | GPU V100   | GPU V100   | GPU V100   | GPU V100   | GPU V100   | GPU V100   | GPU V100   |
| uploaded Date (mm/dd/yyy) | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 | 02/06/2022 |
| MindSpore Version         | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      | 1.5.0      |
| Model                     | TransE     | TransH     | TransR     | TransD     | TransE     | TransH     | TransR     | TransD     |
| Dataset                   | Wordnet    | Wordnet    | Wordnet    | Wordnet    | Freebase   | Freebase   | Freebase   | Freebase   |
| batch_size                | 1          | 1          | 1          | 1          | 1          | 1          | 1          | 1          |
| outputs                   | Scores     | Scores     | Scores     | Scores     | Scores     | Scores     | Scores     | Scores     |
| Hit@10                    | 0.511      | 0.507      | 0.512      | 0.514      | 0.475      | 0.483      | 0.509      | 0.481      |

# [Description of Random Situation](#contents)

We also use random seed in train.py and provide the random seed into the C++ backend of the dataset generator.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
