
# CTPN

<!-- TOC -->

- [CTPN](#ctpn)
- [CTPN Description](#ctpn-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
- [Environment Requirements](#environment-requirements)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Training process](#training-process)
        - [Dataset](#dataset-1)
        - [Usage](#usage)
        - [Launch](#launch)
        - [Result](#result)
    - [Eval process](#eval-process)
        - [Usage](#usage-1)
        - [Evaluation while training](#evaluation-while-training)
        - [Result](#result-1)
    - [Model Export](#model-export)
    - [Inference process](#inference-process)
        - [Usage](#usage-2)
        - [Result](#result-2)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
            - [Inference Performance](#inference-performance)
            - [Training performance results](#training-performance-results)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CTPN Description](#contents)

CTPN is a text detection model based on object detection method. It improves Faster R-CNN and combines with bidirectional LSTM, so ctpn is very effective for horizontal text detection. Another highlight of ctpn is to transform the text detection task into a series of small-scale text box detection.This idea was proposed in the paper "Detecting Text in Natural Image with Connectionist Text Proposal Network".

[Paper](https://arxiv.org/pdf/1609.03605.pdf) Zhi Tian, Weilin Huang, Tong He, Pan He, Yu Qiao, "Detecting Text in Natural Image with Connectionist Text Proposal Network", ArXiv, vol. abs/1609.03605, 2016.

# [Model architecture](#contents)

The overall network architecture contains a VGG16 as backbone, and use bidirection lstm to extract context feature of the small-scale text box, then it used the RPN(RegionProposal Network) to predict the boundding box and probability.

[Link](https://arxiv.org/pdf/1605.07314v1.pdf)

# [Dataset](#contents)

Here we used 6 datasets for training, and 1 datasets for Evaluation.

- Dataset1: [ICDAR 2013: Focused Scene Text](https://rrc.cvc.uab.es/?ch=2&com=tasks):
    - Train: 142MB, 229 images
    - Test: 110MB, 233 images
- Dataset2: [ICDAR 2011: Born-Digital Images](https://rrc.cvc.uab.es/?ch=1&com=tasks):
    - Train: 27.7MB, 410 images
- Dataset3: [ICDAR 2015: Incidental Scene Text](https://rrc.cvc.uab.es/?ch=4&com=tasks):
    - Train：89MB, 1000 images
- Dataset4: [SCUT-FORU: Flickr OCR Universal Database](https://github.com/yan647/SCUT_FORU_DB_Release):
    - Train: 388MB, 1715 images
- Dataset5: [CocoText v2(Subset of MSCOCO2017)](https://rrc.cvc.uab.es/?ch=5&com=tasks):
    - Train: 13GB, 63686 images
- Dataset6: [SVT(The Street View Dataset)](https://www.kaggle.com/datasets/nageshsingh/the-street-view-text-dataset):
    - Train: 115MB, 349 images

We use [ICDAR 2017: ICDAR2017 Competition on Multi-lingual scene text detection and script identification](https://rrc.cvc.uab.es/?ch=8&com=tasks) for multilingual detection training.

This dataset consists of 9000 natural scene images annotated in multiple mixed languages (Chinese, Japanese, Korean, English, French, Arabic, Italian, German, and Hindi, with 7200 trained and 1800 tested). The annotation format is a four point annotation with a grid like clockwise coordinate.

# [Features](#contents)

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend/GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
└─ ctpn
  ├── README.md                             # network readme
  ├── ascend310_infer                        #application for 310 inference
  ├── eval.py                               # eval net
  ├── scripts
  │   ├── eval_res.sh                       # calculate precision and recall
  │   ├── run_distribute_train_ascend.sh    # launch distributed training with ascend platform(8p)
  │   ├── run_distribute_train_gpu.sh       # launch distributed training with gpu platform(8p)
  │   ├── run_eval_ascend.sh                # launch evaluating with ascend platform
  │   ├── run_eval_gpu.sh                   # launch evaluating with gpu platform
  │   ├── run_infer_310.sh                  # shell script for 310 inference
  │   ├── run_standalone_train_gpu.sh       # launch standalone training with gpu platform(1p)
  │   └── run_standalone_train_ascend.sh    # launch standalone training with ascend platform(1p)
  ├── src
  │   ├── CTPN
  │   │   ├── BoundingBoxDecode.py          # bounding box decode
  │   │   ├── BoundingBoxEncode.py          # bounding box encode
  │   │   ├── __init__.py                   # package init file
  │   │   ├── anchor_generator.py           # anchor generator
  │   │   ├── bbox_assign_sample.py         # proposal layer
  │   │   ├── proposal_generator.py         # proposla generator
  │   │   ├── rpn.py                        # region-proposal network
  │   │   └── vgg16.py                      # backbone
  │   ├── model_utils
  │   │   ├── config.py             // Parameter config
  │   │   ├── moxing_adapter.py     // modelarts device configuration
  │   │   ├── device_adapter.py     // Device Config
  │   │   ├── local_adapter.py      // local device config
  │   ├── convert_icdar2015.py              # convert icdar2015 dataset label
  │   ├── convert_svt.py                    # convert svt label
  │   ├── create_dataset.py                 # create mindrecord dataset
  │   ├── ctpn.py                           # ctpn network definition
  │   ├── dataset.py                        # data proprocessing
  │   ├── eval_callback.py                  # evaluation callback while training
  │   ├── eval_utils.py                     # evaluation function
  │   ├── lr_schedule.py                    # learning rate scheduler
  │   ├── weight_init.py                    # lstm initialization
  │   ├── network_define.py                 # network definition
  │   └── text_connector
  │       ├── __init__.py                   # package init file
  │       ├── connect_text_lines.py         # connect text lines
  │       ├── detector.py                   # detect box
  │       ├── get_successions.py            # get succession proposal
  │       └── utils.py                      # some functions which is commonly used
  ├── postprogress.py                        # post process for 310 inference
  ├── export.py                              # script to export AIR,MINDIR model
  ├── requirements.txt                       # requirements file
  ├── train.py                               # train net
  └── default_config.yaml                    # config file

```

## [Training process](#contents)

### Dataset

To create dataset, download the dataset first and deal with it.We provided src/convert_svt.py and src/convert_icdar2015.py to deal with svt and icdar2015 dataset label.For svt dataset, you can deal with it as below:

```shell
    python convert_svt.py --dataset_path=/path/img --xml_file=/path/train.xml --location_dir=/path/location
```

For ICDAR2015 dataset, you can deal with it

```shell
    python convert_icdar2015.py --src_label_path=/path/train_label --target_label_path=/path/label
```

Then modify the src/config.py and add the dataset path.For each path, add IMAGE_PATH and LABEL_PATH into a list in config.An example is show as blow:

```python
    # create dataset
    "coco_root": "/path/coco",
    "coco_train_data_type": "train2017",
    "cocotext_json": "/path/cocotext.v2.json",
    "icdar11_train_path": ["/path/image/", "/path/label"],
    "icdar13_train_path": ["/path/image/", "/path/label"],
    "icdar15_train_path": ["/path/image/", "/path/label"],
    "icdar13_test_path": ["/path/image/", "/path/label"],
    "flick_train_path": ["/path/image/", "/path/label"],
    "svt_train_path": ["/path/image/", "/path/label"],
    "pretrain_dataset_path": "",
    "finetune_dataset_path": "",
    "test_dataset_path": "",
```

Then you can create dataset with src/create_dataset.py with the command as below:

```shell
python src/create_dataset.py
```

### Usage

- Ascend:

    ```default_config.yaml
    if pretraining set pretraining_dataset_file: /home/DataSet/ctpn_dataset/pretrain/ctpn_pretrain.mindrecord0
    if finetune set pretraining_dataset_file: /home/DataSet/ctpn_dataset/finetune/ctpn_finetune.mindrecord0
    img_dir:/home/DataSet/ctpn_dataset/ICDAR2013/test

    Modify the parameters according to the actual path
    ```

    ```bash
    # distribute training
    bash scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [TASK_TYPE] [PRETRAINED_PATH]
    # example: bash scripts/run_distribute_train_ascend.sh /home/hccl_8p_01234567_10.155.170.71.json Pretraining(or Finetune) \
    # /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt

    # standalone training
    bash scrpits/run_standalone_train_ascend.sh [TASK_TYPE] [PRETRAINED_PATH] [DEVICE_ID]
    example: bash scrpits/run_standalone_train_ascend.sh Pretraining(or Finetune) /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt 0

    # evaluation:
    bash scripts/run_eval_ascend.sh [IMAGE_PATH] [DATASET_PATH] [CHECKPOINT_PATH]
    # example: bash script/run_eval_ascend.sh /home/DataSet/ctpn_dataset/ICDAR2013/test \
    # /home/DataSet/ctpn_dataset/ctpn_final_dataset/test/ctpn_test.mindrecord /home/model/cv/ctpn/train_parallel0/ckpt_0/
    ```

- GPU:

    ```bash
    # distribute training
    bash scripts/run_distribute_train_gpu.sh [TASK_TYPE] [PRETRAINED_PATH] [CONFIG_PATH](optional)
    # example: bash scripts/run_distribute_train_gpu.sh Pretraining(or Finetune) \
    # /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt

    # standalone training
    bash scrpits/run_standalone_train_gpu.sh [TASK_TYPE] [PRETRAINED_PATH] [DEVICE_ID] [CONFIG_PATH](optional)
    example: bash scrpits/run_standalone_train_gpu.sh Pretraining(or Finetune) /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt 0

    # evaluation:
    bash scripts/run_eval_gpu.sh [IMAGE_PATH] [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH](optional)
    # example: bash script/run_eval_gpu.sh /home/DataSet/ctpn_dataset/ICDAR2013/test \
    # /home/DataSet/ctpn_dataset/ctpn_final_dataset/test/ctpn_test.mindrecord /home/model/cv/ctpn/train_parallel0/ckpt_0/
    ```

The `pretrained_path` should be a checkpoint of vgg16 trained on Imagenet2012. The name of weight in dict should be totally the same, also the batch_norm should be enabled in the trainig of vgg16, otherwise fails in further steps.COCO_TEXT_PARSER_PATH coco_text.py can refer to [Link](https://github.com/andreasveit/coco-text).To get the vgg16 backbone, you can use the network structure defined in src/CTPN/vgg16.py.To train the backbone, copy the src/CTPN/vgg16.py under modelzoo/official/cv/VGG/vgg16/src/, and modify the vgg16/train.py to suit the new construction.You can fix it as below:

```python
...
from src.vgg16 import VGG16
...
network = VGG16(num_classes=cfg.num_classes)
...

```

To train a better model, you can modify some parameter in modelzoo/official/cv/VGG/vgg16/src/config.py, here we suggested you modify the "warmup_epochs" just like below, you can also try to adjust other parameter.

```python

imagenet_cfg = edict({
    ...
    "warmup_epochs": 5
    ...
})

```

Then you can train it with ImageNet2012.
> Notes:
> RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html) , and the device_ip can be got as [Link](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools). For large models like InceptionV4, it's better to export an external environment variable `export HCCL_CONNECT_TIMEOUT=600` to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size.
>
> This is processor cores binding operation regarding the `device_num` and total processor numbers. If you are not expect to do it, remove the operations `taskset` in `scripts/run_distribute_train.sh`
>
> TASK_TYPE contains Pretraining and Finetune. For Pretraining, we use ICDAR2013, ICDAR2015, SVT, SCUT-FORU, CocoText v2. For Finetune, we use ICDAR2011,
ICDAR2013, SCUT-FORU to improve precision and recall, and when doing Finetune, we use the checkpoint training in Pretrain as our PRETRAINED_PATH.
> COCO_TEXT_PARSER_PATH coco_text.py can refer to [Link](https://github.com/andreasveit/coco-text).
>

### Launch

```bash
# training example
  shell:
    Ascend:
      # distribute training example(8p)
      bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [TASK_TYPE] [PRETRAINED_PATH] [CONFIG_PATH](optional)
      # example: bash scripts/run_distribute_train_ascend.sh /home/hccl_8p_01234567_10.155.170.71.json Pretraining(or Finetune) /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt

      # standalone training
      bash run_standalone_train_ascend.sh [TASK_TYPE] [PRETRAINED_PATH] [CONFIG_PATH](optional)
      # example: bash scrpits/run_standalone_train_ascend.sh Pretraining(or Finetune) /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt 0

  shell:
    GPU:
      # distribute training example(8p)
      bash run_distribute_train_gpu.sh [TASK_TYPE] [PRETRAINED_PATH] [CONFIG_PATH](optional)
      # example: bash scripts/run_distribute_train_gpu.sh Pretraining(or Finetune) /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt

      # standalone training
      bash run_standalone_train_gpu.sh [TASK_TYPE] [PRETRAINED_PATH] [CONFIG_PATH](optional)
      # example: bash scrpits/run_standalone_train_gpu.sh Pretraining(or Finetune) /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt 0
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `ckpt_path` by default, and training log  will be redirected to `./log`, also the loss will be redirected to `./loss_0.log` like followings.

```python
377 epoch: 1 step: 229 ,rpn_loss: 0.00355
399 epoch: 2 step: 229 ,rpn_loss: 0.00327
424 epoch: 3 step: 229 ,rpn_loss: 0.00910
```

- running on ModelArts
- If you want to train the model on modelarts, you can refer to the [official guidance document] of modelarts (https://support.huaweicloud.com/modelarts/)

```python
#  Example of using distributed training dpn on modelarts :
#  Data set storage method

#  ├── ctpn_dataset              # dir
#    ├──train                         # train dir
#      ├── pretrain               # pretrain dataset dir
#      ├── finetune               # finetune dataset dir
#      ├── backbone               # predtrained dir if exists
#    ├── eval                    # eval dir
#      ├── ICDAR2013              # ICDAR2013 img dir
#      ├── checkpoint           # ckpt files dir
#      ├── test                  # ckpt files dir
#          ├── ctpn_test.mindrecord       # test img of mindrecord
#          ├── ctpn_test.mindrecord.db    # test img of mindrecord.db

# (1) Choose either a (modify yaml file parameters) or b (modelArts create training job to modify parameters) 。
#       a. set "enable_modelarts=True" 。
#          set "run_distribute=True"
#          set "save_checkpoint_path=/cache/train/checkpoint/"
#          set "finetune_dataset_file=/cache/data/finetune/ctpn_finetune.mindrecord0"
#          set "pretrain_dataset_file=/cache/data/finetune/ctpn_pretrain.mindrecord0"
#          set "task_type=Pretraining" or task_type=Finetune
#          set "pre_trained=/cache/data/backbone/pred file name" Without pre-training weights  pre_trained=""
#
#       b. add "enable_modelarts=True" Parameters are on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted

# (2) Set the path of the network configuration file  "_config_path=/The path of config in default_config.yaml/"
# (3) Set the code path on the modelarts interface "/path/ctpn"。
# (4) Set the model's startup file on the modelarts interface "train.py" 。
# (5) Set the data path of the model on the modelarts interface ".../ctpn_dataset/train"(choices ctpn_dataset/train Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path" 。
# (6) start trainning the model。

# Example of using model inference on modelarts
# (1) Place the trained model to the corresponding position of the bucket。
# (2) chocie a or b。
#       a. set "enable_modelarts=True" 。
#          set "dataset_path=/cache/data/test/ctpn_test.mindrecord"
#          set "img_dir=/cache/data/ICDAR2013/test"
#          set "checkpoint_path=/cache/data/checkpoint/checkpoint file name"

#       b. Add "enable_modelarts=True" parameter on the interface of modearts。
#          Set the parameters required by method a on the modelarts interface
#          Note: The path parameter does not need to be quoted

# (3) Set the path of the network configuration file "_config_path=/The path of config in default_config.yaml/"
# (4) Set the code path on the modelarts interface "/path/ctpn"。
# (5) Set the model's startup file on the modelarts interface "eval.py" 。
# (6) Set the data path of the model on the modelarts interface ".../ctpn_dataset/eval"(choices FSNS/eval Folder path) ,
# The output path of the model "Output file path" and the log path of the model "Job log path"  。
# (7) Start model inference。
```

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

    ```bash
    bash run_eval_ascend.sh [IMAGE_PATH] [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH](optional)
    # example: bash script/run_eval_ascend.sh /home/DataSet/ctpn_dataset/ICDAR2013/test /home/DataSet/ctpn_dataset/ctpn_final_dataset/test/ctpn_test.mindrecord /home/model/cv/ctpn/train_parallel0/ckpt_0/
    ```

- GPU:

    ```bash
    bash run_eval_gpu.sh [IMAGE_PATH] [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH](optional)
    # example: bash script/run_eval_gpu.sh /home/DataSet/ctpn_dataset/ICDAR2013/test /home/DataSet/ctpn_dataset/ctpn_final_dataset/test/ctpn_test.mindrecord /home/model/cv/ctpn/train_parallel0/ckpt_0/
    ```

After eval, you can get serval archive file named submit_ctpn-xx_xxxx.zip, which contains the name of your checkpoint file.To evalulate it, you can use the scripts provided by the ICDAR2013 network, you can download the Deteval scripts from the [link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=2&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9zdGFuZGFsb25lcy9zY3JpcHRfdGVzdF9jaDJfdDFfZTItMTU3Nzk4MzA2Ny56aXA=)
After download the scripts, unzip it and put it under ctpn/scripts and use eval_res.sh to get the result.You will get files as below:

```text
gt.zip
readme.txt
rrc_evalulation_funcs_1_1.py
script.py
```

Then you can run the scripts/eval_res.sh to calculate the evalulation result.

```base
bash eval_res.sh
```

### Evaluation while training

You can add `run_eval` to start shell and set it True, if you want evaluation while training. And you can set argument option: `eval_dataset_path`, `save_best_ckpt`, `eval_start_epoch`, `eval_interval` when `run_eval` is True.

### Result

Evaluation result will be stored in the example path, you can find result like the followings in `log`.

```text
{"precision": 0.90791, "recall": 0.86118, "hmean": 0.88393}
```

Evaluation result on GPU will be as follows:

```text
{"precision": 0.9346, "recall": 0.8621, "hmean": 0.8969}
```

## Transfer learning on Multilingual Datasets

We use the ICDAR 2017 MLT dataset as the dataset for transfer learning. The dataset contains annotation data in 9 languages, including Chinese, Japanese, Korean, English, French, Arabic, Italian, German and Hindi. Because the dataset does not only have horizontal labels.

1. Process the dataset:

```shell
python src/convert_icdar2015.py --src_label_path=/path/train_gt --target_label_path=/path/train_gt_convert
python src/convert_icdar2015.py --src_label_path=/path/val_gt --target_label_path=/path/val_gt_convert
```

2. Change `default_cn_finetune_config.yaml`：

```text
icdar17_mlt_train_path: ["icdar17_train_img_dir_path", "icdar17_train_gt_txt_dir_path"]
icdar17_mlt_test_path: ["icdar17_val_img_dir_path", "icdar17_val_gt_txt_dir_path"]
icdar17_mlt_prefix: "gt_"  # The prefix of gt_txt name compared to img name
finetune_dataset_path: "/data/ctpn_mindrecord_ic17/finetune"  # Path to generate finetune mindrecord
test_dataset_path: "/data/ctpn_mindrecord_ic17/test"          # Path to generate test mindrecord

# training dataset
finetune_dataset_file: "/data/ctpn_mindrecord_ic17/finetune/ctpn_finetune.mindrecord0"  # Mindrecord path generated by training set
test_dataset_file: "/data/ctpn_mindrecord_ic17/test/ctpn_test.mindrecord"      # Mindrecord path generated by val set
img_dir: "icdar17_val_img_dir_path"                # The original dataset path used for inference
```

3. Generate mindrecord：

Since this process does not involve a pre-trained dataset, it is necessary to comment out `create_train_dataset("pretraining")` in `src/create_dataset.py`:

```python
if __name__ == "__main__":
    # create_train_dataset("pretraining")
    create_train_dataset("finetune")
    create_train_dataset("test")
```

And run:

```shell
python src/create_dataset.py --config_path=default_cn_finetune_config.yaml
```

If you encounter src path issues, you need to add the root directory of the CTPN network to `PYTHONPATH`:

```shell
export PYTHONPATH=/data/models/official/cv/CTPN:$PYTHONPATH
```

You can generate mindrecord files under the configuration above `default_cn_finetune_config.yaml`.

4. Finetune

Download the trained [parameter file](https://download.mindspore.cn/models/r1.9/ctpn_pretrain_ascend_v190_icdar2013_official_cv_acc87.69.ckpt), the training method of transfer learning is the same as that of training, such as:

```shell
bash scripts/run_distribute_train_ascend.sh /home/hccl_8p_01234567_10.155.170.71.json Finetune /home/DataSet/ctpn_dataset/ctpn_pretrain_ascend_v190_icdar2013_official_cv_acc87.69.ckpt /CTPN/default_cn_finetune_config.yaml
```

5. Evaluation

The inference process is consistent with the training inference process, please note add the config path：

```shell
bash scripts/run_eval_ascend.sh icdar17_val_img_dir_path /data/ctpn_mindrecord_ic17/test/ctpn_test.mindrecord train_parallel0/ckpt_0/ default_cn_finetune_config.yaml
```

6. Because the ICDAR 2017 MLT does not provide offline evaluation packages, we use the processing script of ICDAR 2013 and change gt.zip to ICDAR 2017 MLT.

Package the txt file processed in the first step and replace it with [link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=2&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9zdGFuZGFsb25lcy9zY3JpcHRfdGVzdF9jaDJfdDFfZTItMTU3Nzk4MzA2Ny56aXA=) gt.zip：

```shell
cd /path/val_gt_convert
zip -r gt.zip *.txt
mv gt.zip ctpn_code_path   # ctpn_code_path is the root directory of the code, which contains the submit_ *. generated by eval zip file
bash scripts/eval_res.sh
```

get

```text
eval result for submit_ctpn-50_1548.zip
Calculated!{"precision": 0.7585255767301913, "recall": 0.6783185026081612, "hmean": 0.7161833921945736}.
```

## Model Export

```shell
python export.py --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

- Export MindIR on Modelarts

    ```Modelarts
    Export MindIR example on ModelArts
    Data storage method is the same as training
    # (1) Choose either a (modify yaml file parameters) or b (modelArts create training job to modify parameters)。
    #       a. set "enable_modelarts=True"
    #          set "file_name=ctpn"
    #          set "file_format=MINDIR"
    #          set "ckpt_file=/cache/data/checkpoint file name"

    #       b. Add "enable_modelarts=True" parameter on the interface of modearts。
    #          Set the parameters required by method a on the modelarts interface
    #          Note: The path parameter does not need to be quoted
    # (2)Set the path of the network configuration file "_config_path=/The path of config in default_config.yaml/"
    # (3) Set the code path on the modelarts interface "/path/ctpn"。
    # (4) Set the model's startup file on the modelarts interface "export.py" 。
    # (5) Set the data path of the model on the modelarts interface ".../ctpn_dataset/eval/checkpoint"(choices CNNCTC_Data/eval/checkpoint Folder path) ,
    # The output path of the model "Output file path" and the log path of the model "Job log path"  。
    ```

    `EXPORT_FORMAT` should be in ["AIR",  "MINDIR"]

## [Inference process](#contents)

### Usage

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the air file must bu exported by export script on the Ascend910 environment.

```shell
bash run_infer_cpp.sh [MODEL_PATH] [DATA_PATH] [LABEL_PATH] [DEVICE_TYPE] [DEVICE_ID]
```

After inference, you can get a archive file named submit.zip.To evalulate it, you can use the scripts provided by the ICDAR2013 network, you can download the Deteval scripts from the [link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=2&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9zdGFuZGFsb25lcy9zY3JpcHRfdGVzdF9jaDJfdDFfZTItMTU3Nzk4MzA2Ny56aXA=)
After download the scripts, unzip it and put it under ctpn/scripts and use eval_res.sh to get the result.You will get files as below:

```text
gt.zip
readme.txt
rrc_evalulation_funcs_1_1.py
script.py
```

Then you can run the scripts/eval_res.sh to calculate the evalulation result.

```base
bash eval_res.sh
```

### Result

Evaluation result will be stored in the example path, you can find result like the followings in `log`.

```text
{"precision": 0.88913, "recall": 0.86082, "hmean": 0.87475}
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend                                                       | GPU                                              |
| -------------------------- | ------------------------------------------------------------ |------------------------------------------------------------ |
| Model Version              | CTPN                                                         | CTPN                                                     |
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8  | Tesla V100 PCIE 32GB; CPU 2.60GHz; 104cores; Memory 790G; EulerOS 2.0     |
| uploaded Date              | 02/06/2021                                                   | 09/20/2021                                                   |
| MindSpore Version          | 1.1.1                                                        | 1.5.0                                                        |
| Dataset                    | 16930 images                                                 | 16930 images                                                 |
| Batch_size                 | 2                                                            | 2                                                            |
| Training Parameters        | default_config.yaml                                          | default_config.yaml                                          |
| Optimizer                  | Momentum                                                     | Momentum                                                     |
| Loss Function              | SoftmaxCrossEntropyWithLogits for classification, SmoothL2Loss for bbox regression| SoftmaxCrossEntropyWithLogits for classification, SmoothL2Loss for bbox regression|
| Loss                       | ~0.04                                                        | ~0.04                                                       |
| Total time (8p)            | 6h                                                           | 11h                                                           |
| Scripts                    | [ctpn script](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv1) | [ctpn script](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv1)     |

#### Inference Performance

| Parameters          | Ascend                                        | GPU                 |
| ------------------- | --------------------------------------------- | --------------------------- |
| Model Version       | CTPN                                          | CTPN                 |
| Resource            | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8   | Tesla V100 PCIE 32GB; CPU 2.60GHz; 104cores; Memory 790G; EulerOS 2.0 |
| Uploaded Date       | 02/06/2021                                    | 09/20/2021                 |
| MindSpore Version   | 1.1.1                                         | 1.5.0              |
| Dataset             | 229 images                                    |229 images                  |
| Batch_size          | 1                                             |1                         |
| Accuracy            | precision=0.9079, recall=0.8611 F-measure:0.8839 | precision=0.9346, recall=0.8621 F-measure:0.8969 |
| Total time          | 1 min                                         |1 min                      |
| Model for inference | 135M (.ckpt file)                             | 135M (.ckpt file)           |

#### Training performance results

| **Ascend** | train performance |
| :--------: | :---------------: |
|     1p     |     10 img/s      |

| **Ascend** | train performance |
| :--------: | :---------------: |
|     8p     |     84 img/s     |

| **GPU** | train performance |
| :--------: | :---------------: |
|     1p     |     6 img/s      |

| **GPU** | train performance |
| :--------: | :---------------: |
|     8p     |     52 img/s     |

# [Description of Random Situation](#contents)

We set seed to 1 in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
