# Contents

- [Contents](#contents)
- [FiBiNET Description](#FiBiNET-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Training Script Parameters](#training-script-parameters)
        - [Preprocess Script Parameters](#preprocess-script-parameters)
    - [Dataset Preparation](#dataset-preparation)
        - [Process the Criteo Dataset](#process-the-real-world-data)
        - [Process the Synthetic Data](#generate-and-process-the-synthetic-data)
    - [Training Process](#training-process)
        - [SingleDevice](#singledevice)
    - [Evaluation Process](#evaluation-process)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Performance Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [FiBiNET Description](#contents)

FiBiNET (Feature Importance and Bilinear feature Interaction NETwork) is a deep learning-based advertising recommendation algorithm proposed by Sina Weibo in 2019. See the paper in [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf).

# [Model Architecture](#contents)

The FiBiNET model contains a wide linear model and a deep learning neural network, and on the basis of Wide&Deep, the SENET module (Squeeze-and-Excitation Network) which dynamically learns the feature importance as well as the Bilinear-Interaction module that learns the feature interaction are added to the neural network part.

# [Dataset](#contents)

- [Criteo Kaggle Display Advertising Challenge Dataset](http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz)

# [Environment Requirements](#contents)

- Hardware (GPU)
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore), for more information, please check the resources below:
        - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
        - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

1. Clone the Code

```bash
git clone https://gitee.com/mindspore/models.git
cd models/research/recommend/fibinet
```

2. Download the Dataset

  > Please refer to [1](#dataset) to obtain the download link

```bash
mkdir -p data/origin_data && cd data/origin_data
wget DATA_LINK
tar -zxvf dac.tar.gz
```

3. Use this script to preprocess the data. This may take about one hour and the generated mindrecord data is under data/mindrecord.

```bash
python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
```

4. Start Training

Once the dataset is ready, the model can be trained and evaluated by the command as follows:

```bash
# Python command
python train.py --data_path=./data/mindrecord --device_target=GPU --eval_while_train=True

# Shell command
bash ./script/run_train_gpu.sh './data/mindrecord/' 1 GPU True
```

To evaluate the model, command as follows:

```bash
# Python command
python eval.py  --data_path=./data/mindrecord --dataset_type=mindrecord --device_target=GPU

# Shell command
bash ./script/run_eval_gpu.sh './data/mindrecord/' 1 GPU
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```markdown
└── fibinet
    ├── README.md                                 # FiBiNET related instructions and tutorials
    ├── requirements.txt                          # python environment
    ├── script
    │   ├── common.sh
    │   ├── run_train_gpu.sh                      # Shell script of GPU single-device training
    │   └── run_eval_gpu.sh                       # Shell script of GPU single-device evaluation
    ├──src
    │   ├── callbacks.py
    │   ├── datasets.py                           # Create dataset
    │   ├── generate_synthetic_data.py            # Generate synthetic data
    │   ├── __init__.py
    │   ├── metrics.py                            # Script of metrics
    │   ├── preprocess_data.py                    # Preprocess on dataset
    │   ├── process_data.py
    │   ├── fibinet.py                            # FiBiNET model script
    │   └── model_utils
    │       ├── __init__.py
    │       ├── config.py                         # Processing configuration parameters
    │       └── moxing_adapter.py                 # Parameter processing
    ├── default_config.yaml                       # Training parameter profile, it is recommended to
    │                                             # modify any model related parameters here
    ├── train.py                                  # Python script of training
    ├── eval.py                                   # Python script of evaluation
    └── export.py
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```markdown

Used by: train.py

Arguments:


  --device_target                     Device where the code will be implemented, only support GPU currently. (Default:GPU)
  --data_path                         Where the preprocessed data is put in
  --epochs                            Total train epochs. (Default:10)
  --full_batch                        Enable loading the full batch. (Default:False)
  --batch_size                        Training batch size.(Default:1000)
  --eval_batch_size                   Eval batch size.(Default:1000)
  --line_per_sample                   The number of sample per line, must be divisible by batch_size.(Default:10)
  --field_size                        The number of features.(Default:39)
  --vocab_size                        The total features of dataset.(Default:200000)
  --emb_dim                           The dense embedding dimension of sparse feature.(Default:10)
  --deep_layer_dim                    The dimension of all deep layers.(Default:[400,400,400])
  --deep_layer_act                    The activation function of all deep layers.(Default:'relu')
  --keep_prob                         The keep rate in dropout layer.(Default:0.5)
  --dropout_flag                      Enable dropout.(Default:0)
  --output_path                       Deprecated
  --ckpt_path                         The location of the checkpoint file. If the checkpoint file
                                      is a slice of weight, multiple checkpoint files need to be
                                      transferred. Use ';' to separate them and sort them in sequence
                                      like "./checkpoints/0.ckpt;./checkpoints/1.ckpt".
                                      (Default:"./ckpt/")
  --eval_file_name                    Eval output file.(Default:eval.og)
  --loss_file_name                    Loss output file.(Default:loss.log)
  --dataset_type                      The data type of the training files, chosen from [tfrecord, mindrecord, hd5].(Default:mindrecord)
  --vocab_cache_size                  Enable cache mode.(Default:0)
  --eval_while_train                  Whether to evaluate after training each epoch
```

### [Preprocess Script Parameters](#contents)

```markdown

used by: generate_synthetic_data.py

Arguments:
  --output_file                        The output path of the generated file.(Default: ./train.txt)
  --label_dim                          The label category. (Default:2)
  --number_examples                    The row numbers of the generated file. (Default:4000000)
  --dense_dim                          The number of the continue feature.(Default:13)
  --slot_dim                           The number of the category features.(Default:26)
  --vocabulary_size                    The vocabulary size of the total dataset.(Default:400000000)
  --random_slot_values                 0 or 1. If 1, the id is generated by the random. If 0, the id is set by the row_index mod
                                       part_size, where part_size is the vocab size for each slot
```

```markdown

usage: preprocess_data.py

  --preprocess_data_path              Where the origin sample data is put in (i.e. where the file origin_data is put in)
  --dense_dim                         The number of your continues fields.(default: 13)
  --slot_dim                          The number of your sparse fields, it can also be called category features.(default: 26)
  --threshold                         Word frequency below this value will be regarded as OOV. It aims to reduce the vocab size.(default: 100)
  --train_line_count                  The number of examples in your dataset.
  --skip_id_convert                   0 or 1. If set 1, the code will skip the id convert, regarding the original id as the final id.(default: 0)
  --eval_size                         The percent of eval samples in the whole dataset.(default: 0.1)
  --line_per_sample                   The number of sample per line, must be divisible by batch_size.
```

## [Dataset Preparation](#contents)

### [Process the Criteo Data](#content)

1. Download the Dataset and place the raw dataset under a certain path, such as: ./data/origin_data

```bash
mkdir -p data/origin_data && cd data/origin_data
wget DATA_LINK
tar -zxvf dac.tar.gz
```

> Please refer to [1](#dataset) to obtain the download link

2. Use this script to preprocess the data

```bash
python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
```

### [Process the Synthetic Data](#content)

1. The following command will generate 40 million lines of click data, in the format of

> "label\tdense_feature[0]\tdense_feature[1]...\tsparse_feature[0]\tsparse_feature[1]...".

```bash
mkdir -p syn_data/origin_data
python src/generate_synthetic_data.py --output_file=syn_data/origin_data/train.txt --number_examples=40000000 --dense_dim=13 --slot_dim=51 --vocabulary_size=2000000000 --random_slot_values=0
```

2. Preprocess the generated data

```bash
python src/preprocess_data.py --data_path=./syn_data/  --dense_dim=13 --slot_dim=51 --threshold=0 --train_line_count=40000000 --skip_id_convert=1
```

## [Train Process](#contents)

To train the model, command as follows:

```bash
python train.py --data_path=./data/mindrecord --dataset_type=mindrecord --device_target=GPU

# Or

bash ./script/run_train_gpu.sh './data/mindrecord/' 1 GPU False
```

## [Evaluation Process](#contents)

To evaluate the model, command as follows:

```bash
python eval.py --data_path=./data/mindrecord --dataset_type=mindrecord --device_target=GPU --ckpt_path=./ckpt/fibinet_train-10_41265.ckpt

# Or

bash ./script/run_eval_gpu.sh './data/mindrecord/' 1 GPU
```

## Inference Process

### [Export MindIR](#contents)

```bash
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --device_target [DEVICE_TARGET] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

### Performance Result

Inference result is saved in current path, you can find result like this in eval_output.log file.

```markdown
auc : 0.7814143582416716
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters        | FiBiNET                  |
| ----------------- | --------------------------- |
| Resource          |A100-SXM4-40GB   |
| Uploaded Date     | 07/29/2022 |
| MindSpore Version | 1.9                       |
| Dataset           | [1]                        |
| Batch Size        | 1000                       |
| Epoch           | 10                         |
| Learning rate   | 0.0001               |
| Optimizer           | FTRL,Adam                         |
| Loss Function     | Sigmoid cross entropy    |
| Loss           | 0.4702615                       |
| Speed           | 15.588 ms/step                         |
| Outputs           | AUC                         |
| Accuracy          | AUC= 0.7814143582416716                   |

# [Description of Random Situation](#contents)

There are three random situations:

- Shuffle of the dataset.
- Initialization of some model weights.
- Dropout operations.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
