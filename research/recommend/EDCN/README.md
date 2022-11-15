# Contents

- [Contents](#contents)
- [EDCN Description](#EDCN-description)
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
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [EDCN Description](#contents)

Effectively modeling feature interactions is crucial for CTR prediction in industrial recommender systems. The state-of-the-art  deep CTR models with parallel structure (e.g., DCN) learn explicit and implicit feature interactions through independent parallel networks. However, these models suffer from trivial sharing issues, namely insufficient sharing in hidden layers and excessive sharing in network input, limiting the model’s expressiveness and effectiveness.
Therefore, to enhance information sharing between explicit and implicit feature interactions, we propose a novel deep CTR model EDCN. EDCN introduces two advanced modules, namely bridge module and regulation module, which work collaboratively to capture the layer-wise interactive signals and learn discriminative feature distributions for each hidden layer of the parallel networks. Furthermore, two modules are lightweight and model-agnostic, which can be generalized well to mainstream parallel deep CTR models.

[Paper](https://dl.acm.org/doi/abs/10.1145/3459637.3481915): Bo Chen*, Yichao Wang*, Zhirong Liu, Ruiming Tang, Wei Guo, Hongkun Zheng, Weiwei Yao, Muyu Zhang, Xiuqiang He. Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models

# [Model Architecture](#contents)

Specifically, in EDCN, two novel modules, namely bridge module and regulation module, are introduced to tackle the insufficient sharing in hidden layers and excessive sharing in network input, respectively. On the one hand, bridge module performs dense fusion by building connections between cross and deep networks, so as to capture the layer-wise interactive signals between parallel networks and enhance the feature interactions. On the other hand, regulation module is designed to learn discriminative feature distributions for different networks by a field-wise gating network in a soft selection manner. Moreover, regulation module is also able to work jointly with bridge module to further learn reasonable inputs for each hidden layer, making two parallel networks learn explicit and implicit feature interactions collaboratively.

# [Dataset](#contents)

- [1] A dataset Criteo used in  Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction[J]. 2017. [download](http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz)

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- preprocess dataset

  ```shell
  # download dataset
  # Please refer to [1] to obtain the download link
  mkdir -p data/origin_data && cd data/origin_data
  wget DATA_LINK
  tar -zxvf dac.tar.gz

  #preprocess dataset
  python -m src.preprocess_data  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0 --device_target=Ascend

  # OR
  python -m src.preprocess_data  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0 --device_target=GPU
  ```

- running on Ascend

  ```shell
  # run training example
  python train.py \
    --train_data_dir='./data/mindrecord' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='Ascend' \
    --epochs=10 \
    --do_eval=True > ms_log/output.log 2>&1 &

  # run evaluation example
  python eval.py \
    --test_data_dir='./data/mindrecord' \
    --checkpoint_path='./checkpoint/EDCN.ckpt' \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &

  # OR
  bash scripts/run_eval.sh 0 Ascend /test_data_dir /checkpoint_path/edcn.ckpt
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

- running on ModelArts

  If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows

    - Training with single cards on ModelArts

    ```python
    # (1) Upload the code folder to S3 bucket.
    # (2) Click to "create training task" on the website UI interface.
    # (3) Set the code directory to "/{path}/EDCN" on the website UI interface.
    # (4) Set the startup file to /{path}/EDCN/train.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/EDCN/default_config.yaml.
    #         1. Set "enable_modelarts: True"
    #     b. adding on the website UI interface.
    #         1. Add "enable_modelarts=True"
    # (6) Upload the dataset to S3 bucket.
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of single cards.
    # (10) Create your job.
    ```

    - evaluating with single card on ModelArts

    ```python
    # (1) Upload the code folder to S3 bucket.
    # (2) Click to "create training task" on the website UI interface.
    # (3) Set the code directory to "/{path}/EDCN" on the website UI interface.
    # (4) Set the startup file to /{path}/EDCN/eval.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/EDCN/default_config.yaml.
    #         1. Set "enable_modelarts: True"
    #         2. Set "checkpoint_path: ./{path}/*.ckpt"('checkpoint_path' indicates the path of the weight file to be evaluated relative to the file `eval.py`, and the weight file must be included in the code directory.)
    #     b. adding on the website UI interface.
    #         1. Add "enable_modelarts=True"
    #         2. Add "checkpoint_path=./{path}/*.ckpt"('checkpoint_path' indicates the path of the weight file to be evaluated relative to the file `eval.py`, and the weight file must be included in the code directory.)
    # (6) Upload the dataset to S3 bucket.
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (there is only data or zip package under this path).
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of a single card.
    # (10) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
└─EDCN
  ├─README.md                         # descriptions of warpctc
  ├─ascend310_infer                   # application for 310 inference
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in Ascend or GPU
    ├─run_infer_310.sh                # launch 310infer
    └─run_eval.sh                     # launch evaluating in Ascend or GPU
  ├─src
    ├─model_utils
      ├─config.py                     # parsing parameter configuration file of "*.yaml"
      ├─device_adapter.py             # local or ModelArts training
      ├─local_adapter.py              # get related environment variables in local training
      └─moxing_adapter.py             # get related environment variables in ModelArts training
    ├─__init__.py                     # python init file
    ├─callback.py                     # define callback function
    ├─edcn.py                         # EDCN network
    └─dataset.py                      # create dataset for EDCN
  ├─default_config.yaml               # parameter configuration
  ├─eval.py                           # eval script
  ├─export.py                         # export checkpoint file into air/mindir
  ├─mindspore_hub_conf.py             # mindspore hub interface
  ├─postprocess.py                    # 310infer postprocess script
  ├─preprocess.py                     # 310infer preprocess script
  └─train.py                          # train script
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in `default_config.yaml`

- Parameters that can be modified at the terminal

  ```text
  # Train
  train_data_dir: ''                  # train dataset path
  ckpt_path: 'ckpts'                  # the folder path to save '*.ckpt' files. Relative path.
  eval_file_name: "./auc.log"         # file path to record accuracy
  loss_file_name: "./loss.log"        # file path to record loss
  epochs: 10                          # train epochs
  do_eval: "True"                     # whether do eval while training, default is 'True'.
  # Test
  test_data_dir: ''                   # test dataset path
  checkpoint_path: ''                 # the path of the weight file to be evaluated relative to the file `eval.py`, and the weight file must be included in the code directory.
  # Export
  batch_size: 16000                   # batch_size for exported model.
  ckpt_file: ''                       # the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.
  file_name: "edcn"                   # output file name.
  file_format: "MINDIR"                  # output file format, you can choose from AIR or MINDIR, default is MINDIR"
  ```

## [Training Process](#contents)

### Training

- running on Ascend

  ```shell
  python train.py \
    --train_data_dir='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='Ascend' \
    --epochs=10 \
    --do_eval=True > ms_log/output.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `ms_log/output.log`.

  After training, you'll get some checkpoint files under `./checkpoint` folder by default. The loss value are saved in loss.log file.

  ```txt
  2021-12-03 05:16:47 epoch: 1 step: 41257, loss is 0.4894940256821349
  2021-12-03 05:35:22 epoch: 2 step: 41257, loss is 0.4524501245203546
  ...
  ```

  The AUC value are saved in auc.log file.

  ```txt
  2021-12-03 05:17:57 EvalCallBack metricdict_values([0.8058029115087255]); eval_time61s
  2021-12-03 05:36:12 EvalCallBack metricdict_values([0.8095880231992393]); eval_time48s
  ...
  ```

  The model checkpoint will be saved in the current directory.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation.

  ```shell
  python eval.py \
    --test_data_dir='./data/mindrecord' \
    --checkpoint_path='./checkpoint/edcn.ckpt' \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &
  # OR
  bash scripts/run_eval.sh 0 Ascend /test_data_dir /checkpoint_path/edcn.ckpt
  ```

  The above python command will run in the background. You can view the results through the file "eval_output.log". The accuracy is saved in auc.log file.

  ```txt
  {'result': {'AUC': 0.8111802356798972, 'eval_time': 80.33271026615487s}}
  ```

## Inference Process

### [Export MindIR](#contents)

- Export on local

  ```shell
  # The ckpt_file parameter is required, `EXPORT_FORMAT` should be in ["AIR", "MINDIR"]
  python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
  ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

  ```python
  # (1) Upload the code folder to S3 bucket.
  # (2) Click to "create training task" on the website UI interface.
  # (3) Set the code directory to "/{path}/EDCN" on the website UI interface.
  # (4) Set the startup file to /{path}/EDCN/export.py" on the website UI interface.
  # (5) Perform a or b.
  #     a. setting parameters in /{path}/EDCN/default_config.yaml.
  #         1. Set "enable_modelarts: True"
  #         2. Set "ckpt_file: ./{path}/*.ckpt"('ckpt_file' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
  #         3. Set "file_name: edcn"
  #         4. Set "file_format='MINDIR'"(you can choose from AIR or MINDIR)
  #     b. adding on the website UI interface.
  #         1. Add "enable_modelarts=True"
  #         2. Add "ckpt_file=./{path}/*.ckpt"('ckpt_file' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
  #         3. Add "file_name=edcn"
  #         4. Add "file_format='MINDIR'"(you can choose from AIR or MINDIR)
  # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (This step is useless, but necessary.).
  # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
  # (9) Under the item "resource pool selection", select the specification of a single card.
  # (10) Create your job.
  # You will see EDCN.air under "Output file path".
  ```

### Infer on Ascend310

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

`NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
`DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result in acc.log file.

```text
auc: 0.6288036416334053
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | EDCN                                                        |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G              |
| uploaded Date              | 12/03/2021 (month/day/year)                                 |
| MindSpore Version          | 1.5.1                                                       |
| Dataset                    | [1]                                                         |
| Training Parameters        | epoch=10, batch_size=1000, lr=3e-4                          |
| Optimizer                  | Adam                                                        |
| Loss Function              | Sigmoid Cross Entropy With Logits                           |
| outputs                    | Accuracy                                                    |
| Loss                       | 0.44                                                        |
| Per Step Time              | 25.22 ms                                                        |

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | EDCN                        |
| Resource            | Ascend 910                  |
| Uploaded Date       | 12/03/2021 (month/day/year) |
| MindSpore Version   | 1.5.1                       |
| Dataset             | [1]                         |
| batch_size          | 1000                        |
| outputs             | accuracy                    |
| AUC                 | 1pc: 0.8110;                |

# [Description of Random Situation](#contents)

We set the random seed before training in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
