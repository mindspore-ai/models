# Contents

- [Contents](#contents)
    - [Signed Graph Convolution Network description](#sgcn-description)
    - [Model architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment requirements](#environment-requirements)
    - [Quick start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
        - [Evaluation Process](#evaluation-process)
        - [Export MINDIR](#export-mindir)
        - [Ascend310 Inference](#ascend310-inference)
    - [Model Description](#model-description)
        - [Training Performance on Ascend](#training-performance-ascend)
        - [Training Performance on GPU](#training-performance-gpu)
        - [Ascend310 Performance](#ascend310-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [Signed Graph Convolution Network description](#contents)

Signed Graph Convolution Network was proposed in 2018 to learn the structured data of the signed graph. Authors redesigned the
GCN model defined balanced and unbalanced paths according to the balance theory.

> [Paper](https://arxiv.org/abs/1808.06354):  Signed Graph Convolutional Network. Tyler Derr, Yao Ma, and Jiliang Tang ICDM, 2018.

## [Model architecture](#contents)

SGCN contains three graph convolutional layers according to the positive and negative connections. Each layer takes the
corresponding positive/negative connection edge as input data. Loss calculation of the network consists of three
parts: loss of the positive connection, loss of the negative connection, and regression loss of both.

## [Dataset](#contents)

[Bitcoin-Alpha and Bitcoin-OTC](https://github.com/benedekrozemberczki/SGCN/tree/master/input)

Data were taken from the real world related to Bitcoin. Both of these datasets were received from websites where users
can use bitcoin to buy and sell things. Since bitcoin accounts are anonymous, users of the website can evaluate positively
or negatively other users, which helps solve possible fraud problems in the transaction. In the experiment, 80% of data are
used for train and 20% for test.

| Dataset  | Users | Positive connections | Negative connections |
| -------  | ---------------:|-----:| ----:|
| Bitcoin-Alpha |3784 | 12729  | 1416  |
| Bitcoin-OTC| 5901 |18390  | 3132  |

## [Environment requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- Download [Bitcoin-Alpha and Bitcoin-OTC](https://github.com/benedekrozemberczki/SGCN/tree/master/input) and put them
  into the input folder under the root directory.

## [Quick start](#contents)

After installing [MindSpore](https://www.mindspore.cn/install/en)，[download data](https://github.com/benedekrozemberczki/SGCN/tree/master/input)，
you can organize the dataset in the following structure：

```text
.
└─input
    ├─bitcoin_alpha.csv
    └─bitcoin_otc.csv
```

After preparing the dataset you can start training and evaluation as follows：

### [Running on Ascend](#contents)

#### Train：

```shell
# standalone training
bash ./scripts/run_standalone_train.sh [DEVICE_ID] [dataset]

# distributed training
bash ./scripts/run_distributed_train.sh [RANK_TABLE] [RANK_SIZE] [DEVICE_START] [DATA_PATH] [DISTRIBUTED]
```

Example：

```shell
# standalone training
bash ./scripts/run_standalone_train.sh 0 ./input/bitcoin_otc.csv

# distributed training (8p)
bash ./scripts/run_distributed_train.sh ./rank_table_8pcs.json 8 0 ./input/bitcoin_otc.csv True
```

#### Evaluate：

```shell
# evaluate
bash ./scripts/run_eval.sh [checkpoint_auc] [checkpoint_f1] [dataset]
```

Example：

```shell
# evaluate
bash ./scripts/run_eval.sh sgcn_otc_auc.ckpt sgcn_otc_f1.ckpt ./input/bitcoin_otc.csv
```

### [Running on GPU](#contents)

#### Train

```shell
# standalone train
bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID] [EDGE_PATH] [CKPT_NAME]

# distribute train
bash ./scripts/run_distribute_train_gpu.sh [EDGE_PATH] [CKPT_NAME] [DEVICE_NUM]
```

Example:

```shell
# standalone train
bash ./scripts/run_standalone_train_gpu.sh 0 /input/bitcoin_otc.csv bitcoin_otc

# distribute train (8p)
bash ./scripts/run_distribute_train_gpu.sh ./input/bitcoin_otc.csv bitcoin_otc 8
```

#### Evaluate

```shell
# evaluate
bash ./scripts/run_eval_gpu.sh [CKPT_AUC] [CKPT_F1] [EDGE_PATH]
```

Example:

```shell
# evaluate
bash ./scripts/run_eval_gpu.sh standalone_bitcoin_otc_auc.ckpt standalone_bitcoin_otc_f1.ckpt ./input/bitcoin_otc.csv
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
.
└── sgcn
    └── ascend310_infer
        ├── build.sh                            # build bash
        ├── CMakeLists.txt                      # CMakeLists
        ├── inc
        │   └── utils.h                         # utils head
        └── src
            ├── main.cc                         # main function of ascend310_infer
            └── utils.cc                        # utils function of ascend310_infer
    ├── eval.py                                 # evaluate val results
    ├── export.py                               # convert mindspore model to minddir model
    ├── postprocess.py                          # postprocessing
    ├── preprocess.py                           # preprocessing
    ├── README_CN.md
    ├── README.md
    ├── requirements.txt
    └── scripts
        ├── run_distributed_train_gpu.sh        # launch distributed training(8p) on GPU
        ├── run_distributed_train.sh            # launch distributed training(8p) on Ascend
        ├── run_eval_gpu.sh                     # launch evaluating on GPU
        ├── run_eval.sh                         # launch evaluating in Ascend
        ├── run_export.sh                       # launch export mindspore model to mindir model
        ├── run_infer_310.sh                    # launch evaluating in 310
        ├── run_standalone_train_gpu.sh         # launch standalone traininng(1p) on GPU
        └── run_standalone_train.sh             # launch standalone training(1p) on Ascend
    └── src
        ├── metrics.py                          # calculate loss and backpropogation
        ├── ms_utils.py                         # utils function
        ├── param_parser.py                     # parameter configuration
        ├── sgcn.py                             # SGCN network and trainer
        └── signedsageconvolution.py            # Definition graph convolutional layers
    └── train.py                                # train
```

### [Script Parameters](#contents)

Training parameters can be configured in `param_parser.py`

```text
"learning-rate": 0.01,            # learning rate
"epochs": 500,                    # number of training epochs
"lamb": 1.0,                      # embedding regularization parameter
"weight_decay": 1e-5,             # weight decay for conv layer parameters
"test-size": 0.2,                 # test set ratio
```

Model parameters can be configured in `param_parser.py`

```text
"norm": True,                     # normalize dense features by number of elements
"norm-embed": True,               # normalize embedding or not
"bias"" True,                     # add bias or not
```

For more parameter information refer to the contents of `param_parser.py`

### [Training Process](#contents)

#### Run on Ascend

```shell
# standalone training
bash ./scripts/run_standalone_train.sh 0 ./input/bitcoin_otc.csv

# distribute trainng
bash ./scripts/run_distributed_train.sh ./rank_table_8pcs.json 8 0 ./input/bitcoin_otc.csv True
```

During training information such as current epoch, loss value and running time of each epoch will be displayed in the
following form and running logs will be saved to `logs/train.log`

Result:

```text
=========================================================================
Epoch: 0494 train_loss= 0.6885321 time= 0.3938899040222168 auc= 0.8568888790661333 f1= 0.8595539481615432
Epoch: 0494 sample_time= 0.09816598892211914 train_time= 0.03619980812072754 test_time= 0.2595179080963135 save_time= 0.0002493858337402344
=========================================================================
Epoch: 0495 train_loss= 0.6892552 time= 0.3953282833099365 auc= 0.8591453682601632 f1= 0.8528564934080921
Epoch: 0495 sample_time= 0.1002199649810791 train_time= 0.03614640235900879 test_time= 0.2589559555053711 save_time= 0.00026345252990722656
=========================================================================
Epoch: 0496 train_loss= 0.6864389 time= 0.3973879814147949 auc= 0.8581971834403941 f1= 0.7663798808735937
Epoch: 0496 sample_time= 0.09870719909667969 train_time= 0.03621697425842285 test_time= 0.26245594024658203 save_time= 0.0003509521484375
=========================================================================
Epoch: 0497 train_loss= 0.68468577 time= 0.3998579978942871 auc= 0.8540750929442135 f1= 0.6958808063102541
Epoch: 0497 sample_time= 0.10423851013183594 train_time= 0.03621530532836914 test_time= 0.2593989372253418 save_time= 0.00024199485778808594
=========================================================================
Epoch: 0498 train_loss= 0.6862765 time= 0.3946268558502197 auc= 0.8611026245391791 f1= 0.8313908313908315
Epoch: 0498 sample_time= 0.10092616081237793 train_time= 0.03557133674621582 test_time= 0.25812458992004395 save_time= 0.00023102760314941406
=========================================================================
Epoch: 0499 train_loss= 0.6885195 time= 0.3965325355529785 auc= 0.8558373386341545 f1= 0.8473539308657082
Epoch: 0499 sample_time= 0.10099625587463379 train_time= 0.03545022010803223 test_time= 0.26008152961730957 save_time= 0.00026535987854003906
=========================================================================
Epoch: 0500 train_loss= 0.692978 time= 0.3948099613189697 auc= 0.8620984786140553 f1= 0.8376332457902056
Epoch: 0500 sample_time= 0.10086441040039062 train_time= 0.03542065620422363 test_time= 0.25852012634277344 save_time= 0.0002288818359375
=========================================================================
Training fished! The best AUC and F1-Score is: 0.8689866859770485 0.9425843754201964 Total time: 41.48991870880127
******************** finish training! ********************
```

#### [Run on GPU](#contents)

##### Standalone training

```shell
bash ./scripts/run_standalone_train_gpu.sh 0 ./input/bitcoin_otc.csv bitcoin_otc
```

Logs will be saved to `logs/standalone_train_bitcoin_otc.log`

Result:

```text
Epoch: 0493 train_loss= 0.7017297 time= 0.12964224815368652 auc= 0.8721751091084246 f1= 0.8090532355753906
Epoch: 0494 train_loss= 0.703571 time= 0.13238096237182617 auc= 0.8585092003829141 f1= 0.6739480752014324
Epoch: 0495 train_loss= 0.6995465 time= 0.12849903106689453 auc= 0.8547010769351442 f1= 0.7463390001683218
Epoch: 0496 train_loss= 0.69432104 time= 0.12692952156066895 auc= 0.8497748927766777 f1= 0.9010514186950888
Epoch: 0497 train_loss= 0.6988386 time= 0.12778067588806152 auc= 0.8678389403211261 f1= 0.8050414805360561
Epoch: 0498 train_loss= 0.69004244 time= 0.12807798385620117 auc= 0.8684280889785649 f1= 0.7031086273140063
Epoch: 0499 train_loss= 0.6941365 time= 0.1282806396484375 auc= 0.868635233559849 f1= 0.7985831589116085
Epoch: 0500 train_loss= 0.6938346 time= 0.12740111351013184 auc= 0.8662704051496662 f1= 0.8446288612263717
Training fished! The best AUC and F1-Score is: 0.8737889300722362 0.9388881587593639 Total time: 66.13358068466187
******************** finish training! ********************
```

##### Distribute training (8p)

```shell
bash ./scripts/run_distribute_train_gpu.sh ./input/bitcoin_otc.csv bitcoin_otc 8
```

Logs will be saved to `logs/ditributed_train_bitcoin_otc.log`

Result:

```text
Epoch: 0493 train_loss= 0.68585926 time= 0.2384045124053955 auc= 0.853910200657229 f1= 0.8114014251781473
Epoch: 0494 train_loss= 0.68601215 time= 0.17083501815795898 auc= 0.8589302065768837 f1= 0.8144053072184488
Epoch: 0495 train_loss= 0.68679065 time= 0.29767560958862305 auc= 0.8525771949416853 f1= 0.8135271807838179
Epoch: 0496 train_loss= 0.6887768 time= 0.25383973121643066 auc= 0.8605643953133322 f1= 0.7908782144590005
Epoch: 0497 train_loss= 0.6939055 time= 0.2060248851776123 auc= 0.8528348255976966 f1= 0.8598187311178249
Epoch: 0498 train_loss= 0.70239216 time= 0.27203822135925293 auc= 0.8638813087550653 f1= 0.7749138072566081
Epoch: 0499 train_loss= 0.69806135 time= 0.16673588752746582 auc= 0.8591195835267189 f1= 0.8028146489684952
Epoch: 0500 train_loss= 0.6878193 time= 0.280958890914917 auc= 0.8542807207764715 f1= 0.8447509578544062
Training fished! The best AUC and F1-Score is: 0.8725269948824887 0.9388881587593639 Total time: 128.41520500183105
******************** finish training! ********************
```

### [Evaluation Process](#contents)

#### Ascend

```shell
bash ./scripts/run_eval.sh sgcn_otc_auc.ckpt sgcn_otc_f1.ckpt ./input/bitcoin_otc.csv
```

Result:

Here is Bitcoin-OTC dataset as an example，you can view the result in log file `./logs/eval.log`：

```text
=====Evaluation Results=====
AUC: 0.866983
F1-Score: 0.930903
============================
```

#### GPU

```shell
bash ./scripts/run_eval_gpu.sh [CKPT_AUC] [CKPT_F1] [EDGE_PATH]
```

Example:

```shell
bash ./scripts/run_eval_gpu.sh bitcoin_otc_auc.ckpt bitcoin_otc_f1.ckpt ./input/bitcoin_otc.csv
```

Result:

Here is Bitcoin-OTC dataset as an example，you can view the result in log file `./logs/eval.log`：

```text
=====Evaluation Results=====
AUC: 0.873789
F1-Score: 0.938888
============================
```

### [Export MINDIR](#contents)

If you want to infer the network on Ascend 310, you should convert the model to MINDIR.

#### Ascend

```shell
bash ./scripts/run_export.sh 0 ./input/bitcoin_otc.csv
```

Result:

Logs will be saved to `./logs/export.log`：

```text
==========================================
sgcn.mindir exported successfully!
==========================================
```

#### GPU

```shell
bash ./scripts/run_export_gpu.sh [DEVICE_ID] [CKPT_PATH] [EDGE_PATH] [OUTPUT_FILE_NAME]
```

Example:

```shell
bash ./scripts/run_export_gpu.sh 0 standalone_bitcoin_otc_auc.ckpt ./input/bitcoin_otc.csv standalone_bitcoin_otc_auc
```

Result:

Logs will be saved to `./logs/export.log`：

```text
==========================================
standalone_bitcoin_otc_auc.mindir exported successfully!
==========================================
```

### [Ascend310 Inference](#contents)

#### Run

Note that when inferring on different datasets, you need to modify the corresponding `checkpoint` parameters in the
`postprocess.py` file.

```shell
bash ./scripts/run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `DATASET_NAME` indicates dataset name： 'bitcoin-alpha', 'bitcoin-otc'.
- `NEED_PREPROCESS` indicates whether data needs to be processed: y or n.
- `DEVICE_ID` optional, device id, default value is 0.

Result:

Inference result saved in `acc.log`，here is bitcoin-otc dataset as an example.

```text
==========================================
Test set results: auc= 0.87464 f1= 0.93635
==========================================
```

## [Model Description](#contents)

### [Training Performance on Ascend](#contents)

| Parameter            | SGCN (Ascend)                                                            |
| -------------------- | ------------------------------------------------------------------------ |
| Resource             | Ascend 910；CPU 2.60GHz，192 cores；Memory 755G； OS Euler2.8                |
| Uploaded date        | 2021-10-01                                                               |
| MindSpore version    | 1.3.0                                                                    |
| Dataset              | Bitcoin-OTC / Bitcoin-Alpha                                              |
| Training parameters  | epoch=500, lr=0.01, weight_decay=1e-5                                    |
| Optimizer            | Adam                                                                     |
| Loss function        | SoftmaxCrossEntropyWithLogits                                            |
| Speed                | 0.128 s/epoch (Bitcoin-OTC) / 0.118 s (Bitcoin-Alpha)                      |
| Total time           | 8pcs: 64 s (Bitcoin-OTC) / 60 s (Bitcoin-Alpha)                          |
| AUC                  | 0.8663 / 0.7979                                                          |
| F1-Score             | 0.9309 / 0.9527                                                          |
| Scripts              | [SGCN](https://gitee.com/mindspore/models/tree/master/research/gnn/sgcn) |

### [Training Performance on GPU](#contents)

| Parameter            | SGCN (1p)                                          | SGCN (8p)                                         |
| -------------------- | -------------------------------------------------- | ------------------------------------------------- |
| Resource             | 1x Nvidia V100-PCIE                          | 8x Nvidia V100-PCIE                         |
| Uploaded date        | -                                                  | -                                                 |
| Mindspore version    | 1.5.0rc1                                           | 1.5.0rc1                                          |
| Dataset              | Bitcoin-OTC / Bitcoin-Alpha                        | Bitcoin-OTC / Bitcoin-Alpha                       |
| Training parameters  | epoch=500, lr=0.01, weight_decay=1e-5              | epoch=500, lr=0.01, weight_decay=1e-5             |
| Optimizer            | Adam                                               | Adam                                              |
| Loss function        | SoftmaxCrossEntropyWithLogits                      | SoftmaxCrossEntropyWithLogits                     |
| AUC                  | <table> <tr> <td></td> <td>Bitcoin-OTC</td> <td>Bitcoin-Alpha</td></tr> <tr> <td>norm=True</td> <td>0.8649</td> <td>0.8097</td> </tr> <tr> <td>norm=False</td> <td>0.8738</td> <td>0.8286</td> </tr> </table> | <table> <tr> <td></td> <td>Bitcoin-OTC</td> <td>Bitcoin-Alpha</td></tr> <tr> <td>norm=True</td> <td>0.8656</td> <td>0.8058</td> </tr> <tr> <td>norm=False</td> <td>0.8725</td> <td>0.8271</td> </tr> </table> |
| F1-Score             | <table> <tr> <td></td> <td>Bitcoin-OTC</td> <td>Bitcoin-Alpha</td></tr> <tr> <td>norm=True</td> <td>0.9302</td> <td>0.9536</td> </tr> <tr> <td>norm=False</td> <td>0.9389</td> <td>0.9555</td> </tr> </table> | <table> <tr> <td></td> <td>Bitcoin-OTC</td> <td>Bitcoin-Alpha</td></tr> <tr> <td>norm=True</td> <td>0.9333</td> <td>0.9550</td> </tr> <tr> <td>norm=False</td> <td>0.9389</td> <td>0.9555</td> </tr> </table> |

### [Ascend310 Performance](#contents)

| Parameter             | SGCN                                                                     |
| --------------------- | ------------------------------------------------------------------------ |
| Resource              | Ascend 310                                                               |
| Upload Date           | 2021-11-01                                                               |
| MindSpore version     | 1.3.0                                                                    |
| Dataset               | Bitcoin-OTC / Bitcoin-Alpha                                              |
| Training parameters   | epoch=500, lr=0.01, weight_decay=1e-5                                    |
| Optimizer             | Adam                                                                     |
| Loss function         | SoftmaxCrossEntropyWithLogits                                            |
| AUC                   | 0.8746 / 0.8227                                                          |
| F1-Score              | 0.9363 / 0.9543                                                          |
| Scripts               | [SGCN](https://gitee.com/mindspore/models/tree/master/research/gnn/sgcn) |

## [Description of Random Situation](#contents)

`train.py` and `eval.py` scripts use mindspore.set_seed() to set global random seed, which can be modified in the corresponding
parser. `sgcn.py` set random_state in `train_test_split` function for split dataset into train and test.

## [ModelZoo Homepage](#contents)

Please visit the official website [homepage](https://gitee.com/mindspore/models).
