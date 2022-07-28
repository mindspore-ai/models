# DLinkNet

<!-- TOC -->

- [DLinkNet](#DLinkNet)
    - [D-LinkNet Description](#dlinknet-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [running on Ascend](#running-on-ascend)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Infer Performance](#infer-performance)
        - [How to use](#how-to-use)
            - [Inference](#inference)
                - [Running on Ascend 310](#running-on-ascend-310)
    - [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

## dlinknet-description

D-Linknet model is constructed based on LinkNet architecture. This implementation is as described  in the original paper [D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.html).
The model performed best in the 2018 DeepGlobe Road Extraction Challenge. The network uses encoder-decoder structure, cavity convolution and pre-trained encoder to extract road.

[D-LinkNet Paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf): chen Zhou, Chuang Zhang, Ming Wu; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2018, pp. 182-186

You can get the weight files in. CKPT format that have been trained and conform to the accuracy of dlinknet34 model in [this Baidu Netdisk link](https://pan.baidu.com/s/1KAPPfkSbe5T4wdwLngcIhw?pwd=hkju)

Note: Before executing the python command or bash command, please remember to complete the required parameters in the dlinknet_config.yaml file.

## model-architecture

In the DeepGlobe Road Extraction Challenge, the raw size of the images and masks provided is 1024×1024, and the roads in most images span the entire image. Still, roads have some natural properties, such as connectivity, complexity, etc. With these attributes in mind, D-Linknet is designed to receive 1024×1024 images as input and retain detailed spatial information. D-linknet can be divided into A, B, C three parts, called encoder, central part and decoder respectively.

D-linknet uses ResNet34, pre-trained on the ImageNet dataset, as its encoder. ResNet34 was originally designed for the classification of 256×256 medium resolution images, but in this challenge the task was to segment roads from 1024×1024 high resolution satellite images. Considering narrowness, connectivity, complexity, and long road spans, it is important to increase the perceived range of features of the central part of the network and retain details. Pooling layer can multiply the felt range of features, but may reduce the resolution of the central feature map and reduce the spatial information. The empty convolution layer may be an ideal alternative to the pooling layer. D-linknet uses several empty convolution layers with skip-connection in the middle.

The code for the ResNet34 model under the MindSpore framework used in this project comes from [this website](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo/official/cv/resnet). The corresponding pre-training weights can be found [here](https://download.mindspore.cn/model_zoo/r1.3/)

Empty convolution can be stacked in cascading mode. As shown in the figure above, if the expansion coefficients of the stacked cavity convolution layers are 1, 2, 4, 8 and 16 respectively, then the acceptance fields of each layer will be 3, 7, 15, 31 and 63. The encoder part (ResNet34) has five down-sampling layers. If the image of size 1024×1024 passes through the encoder part, the size of the output feature map will be 32×32. In this case, D-Linknet uses hollow convolution layers with expansion coefficients of 1, 2, 4 and 8 in the central part, so the feature points on the last central layer will see 31×31 points on the first central feature map, covering the main part of the first central feature map. Nevertheless, d-Linknet takes advantage of multi-resolution capabilities, and the central part of D-Linknet can be seen as parallel mode.

The decoder for D-Linknet is the same as the original LinkNet, which is computationally valid. In the decoder part, the transpose convolution layer is used for up-sampling, and the resolution of the feature map is restored from 32×32 to 1024×1024.

## dataset

Dataset used： [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/balraj98/deepglobe-road-extraction-dataset)

- Description: The dataset consisted of 6226 training images, 1243 validation images and 1101 test images. The resolution of each image is 1024×1024. The dataset is represented as a dichotomous segmentation problem, where roads are marked as foreground and other objects as background.
- Dataset size: 3.83 GB

    - Train: 2.79 GB, 6226 images, including the corresponding label image, the original image named 'xxx_sat.jpg', the corresponding label image named 'xxx_mask.png'.
    - Val: 552 MB, 1243 images, no corresponding label image, original image named 'xxx_sat.jpg'.
    - Test: 511 MB, 1101 images, no corresponding label image, original image named 'xxx_sat.jpg'.

- Note: since this data set is used for competition, the label images of the verification set and test set will not be disclosed. I have adopted the method of dividing the training set by one tenth as the verification set to verify the training accuracy of the model.
- The data set shown above is linked to the Kaggle community and can be downloaded directly.

- If you don't want to divide the training set by yourself, you can just download this [baiduNetDisk link](https://pan.baidu.com/s/1DofqL6P13PEDGUvNMPo-1Q?pwd=5rp1) , which contains three folders:

    - train: file used for training script, 5604 images, including the corresponding label image, the original image is named `xxx_sat.jpg`, the corresponding label image is named `xxx_mask.png`.
    - valid: file used for the test script. 622 images, not containing the corresponding label image. The original image is named `xxx_sat.jpg`.
    - valid_mask: file used for the eval script. 622 images are the label image corresponding to the valid image named `xxx_mask.png`.

## environment-requirements

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## quick-start

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Select the network and dataset to use

    If you use other parameters, you can also refer to the yaml file under 'dlinknet/' and choose the network structure to use by setting ' 'model_name' 'to' 'dinknet34' 'or' 'dinknet50' '.
    Note that when different network structures are used offline or in the cloud, the address path of the pre-training weight model of the corresponding network needs to be modified in the YAML file.

- Run on Ascend

    Note that before the offline machine runs, make sure that the `enable_Modelarts` parameter in the `dlinknet_config.yaml` file is set to `False`.
    Also, before running the training and evaluation scripts, make sure you download the resnet34 pre-training weights file [here](https://download.mindspore.cn/model_zoo/r1.3/resnet34_ascend_v130_imagenet2012_official_cv_bs256_top1acc73.83__top5acc91.61/) and set the `pretrained_ckpt` parameter in the `dlinknet_config.yaml` file to its absolute path.

  ```shell
  # run training example
  python train.py --data_path=/path/to/data/ --config_path=/path/to/yaml > train.log 2>&1 &
  OR
  bash scripts/run_standalone_train.sh [DATASET] [CONFIG_PATH]

  # run distributed training example
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [CONFIG_PATH]

  # run evaluation example
  python eval.py --data_path=$DATASET --label_path=$LABEL_PATH --trained_ckpt=$CHECKPOINT --predict_path=$PREDICT_PATH --config_path=$CONFIG_PATH > eval.log 2>&1 &
  OR
  bash scripts/run_standalone_eval.sh [DATASET] [LABEL_PATH] [CHECKPOINT] [PREDICT_PATH] [CONFIG_PATH]

  # run export
  python export.py --config_path=[CONFIG_PATH] --trained_ckpt=[model_ckpt_path] --file_name=[model_name] --file_format=MINDIR --batch_size=1
  ```

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```text
# run distributed training on modelarts example
# (1) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config directory to "config_path=/The path of config in S3/"
# (3) Set the code directory to "/path/dlinknet" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run evaluation on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a.  Set "enable_modelarts=True" on yaml file.
#          Set "trained_ckpt=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "trained_ckpt=/The path of checkpoint in S3/" on the website UI interface.
# (3) Set the config directory to "config_path=/The path of config in S3/"
# (4) Set the code directory to "/path/dlinknet" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

## script-description

### Script and Sample Code

```text
├── model_zoo
    ├── README.md                           // descriptions about all the models
    ├── dlinknet
        ├── README.md                       // descriptions about DLinknet
        ├── README_CN.md                    // chinese descriptions about DLinknet
        ├── ascend310_infer                 // code of infer on ascend 310
        ├── scripts
        │   ├──run_disribute_train.sh       // shell script for distributed on Ascend
        │   ├──run_standalone_train.sh      // shell script for standalone on Ascend
        │   ├──run_standalone_eval.sh       // shell script for evaluation on Ascend
        │   ├──run_infer_310.sh             // shell script for infer on ascend 310
        ├── src
        │   ├──__init__.py
        │   ├──callback.py                  // custom Callback
        │   ├──data.py                      // data processing
        │   ├──loss.py                      // loss
        │   ├──resnet.py                    // resnet network structure (reference to intra-site ModelZoo)
        │   ├──dinknet.py                   // dlinknet model
        │   ├──model_utils
                ├──__init__.py
                ├──config.py                // parameter configuration
                ├──device_adapter.py        // device adapter
                ├──local_adapter.py         // local adapter
                └──moxing_adapter.py        // moxing adapter
        ├── dlinknet_config.yaml            // parameter configuration
        ├── train.py                        // training script
        ├── eval.py                         // evaluation script
        ├── export.py                       // export script
        ├── postprocess.py                  // dlinknet 310 infer postprocess
        └── requirements.txt                // Requirements of third party package.
```

### script-parameters

Parameters for both training , evaluation and export can be set in *.yaml

- D-LinkNet配置，DeepGlobe Road Extraction Dataset

  ```yaml
  enable_modelarts: True              # whether train on ModelArts
  data_url: ""                        # no need to fill, the data path for training or evaluation on the cloud
  train_url: ""                       # no need to fill, the output path of training or evaluation on the cloud
  data_path: "/cache/data"            # data path for training locally
  output_path: "/cache/train"         # output path for training locally
  device_target: "Ascend"             # type of the target device
  epoch_num: 300                      # total training epochs when running 1p
  run_distribute: "False"             # distributed training or not
  distribute_epoch_num: 1200          # total training epochs when running 8p
  pretrained_ckpt: '~/resnet34.ckpt'  # pretrained model path
  log_name: "weight01_dink34"         # save name of the model weight
  batch_size: 4                       # training batch size
  learning_rate: 0.0002               # learning rate
  model_name: "dlinknet34"             # model name
  scale_factor: 2                     # loss scale: scale_factor
  scale_window: 1000                  # loss scale: scale_window
  init_loss_scale: 16777216           # loss scale: init_loss_scale
  trained_ckpt: '~/dinknet34.ckpt'    # model weight path for evaluation and export
  label_path: './'                    # standard label path for evaluation
  predict_path: './'                  # predicted label path for evaluation
  num_channels: 3                     # number of image channels used for export
  width: 1024                         # width of picture used for export
  height: 1024                        # height of picture used for export
  file_name: "dinknet34"              # file name used for export
  file_format: "MINDIR"               # file format used for export
  ```

## training-process

- Note that before the offline machine runs, make sure that the `enable_Modelarts` parameter in the `dlinknet_config.yaml` file is set to `False`.

- Also, before running the training and evaluation scripts, make sure you download the resnet34 pre-training weights file [here](https://download.mindspore.cn/model_zoo/r1.3/resnet34_ascend_v130_imagenet2012_official_cv_bs256_top1acc73.83__top5acc91.61/) and set the `pretrained_ckpt` parameter in the `dlinknet_config.yaml` file to its absolute path.

### running-on-ascend

#### running on Ascend

  ```shell
  python train.py --data_path=/path/to/data/ --config_path=/path/to/yaml > train.log 2>&1 &
  OR
  bash scripts/run_standalone_train.sh [DATASET] [CONFIG_PATH]
  ```

  The path to the `[DATASET]` parameter is the train file extracted from the DATASET. Please remember to draw a tenth of it for subsequent and validation of the IOU.
  If you download the partitioned data set, set `[DATASET]` to the absolute path to the `train` file in it.

  The python command above will run in the background, you can view the results through the file `train.log`.
  Model checkpoints and logs are stored in the `'./output'` path.

### distributed-training

#### running on Ascend

```shell
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [CONFIG_PATH]
```

  The path to the `[DATASET]` parameter is the train file extracted from the DATASET. Please remember to draw a tenth of it for subsequent and validation of the IOU.
  If you download the partitioned data set, set `[DATASET]` to the absolute path to the `train` file in it.

  The above shell script will run distribute training in the background.   You can view the results through the file `LOG[X]/log.log`.
  Model checkpoints and logs are stored in the `'LOG[X]/output'` path.

## evaluation-process

### evaluation

#### running evaluation on Ascend

  ```shell
  python eval.py --data_path=$DATASET --label_path=$LABEL_PATH --trained_ckpt=$CHECKPOINT --predict_path=$PREDICT_PATH --config_path=$CONFIG_PATH > eval.log 2>&1 &
  OR
  bash scripts/run_standalone_eval.sh [DATASET] [LABEL_PATH] [CHECKPOINT] [PREDICT_PATH] [CONFIG_PATH] [DEVICE_ID](option, default is 0)
  ```

  The `[DATASET]` parameter corresponds to the path of the image part of the train file we previously mapped.
  If you download the partitioned data set, set `[DATASET]` to the absolute path to the `valid` file in it.

  The `[LABEL_PATH]` argument corresponds to the path of the label part of the train file that we drew earlier.
  If you download the partitioned data set, set `[LABEL_PATH]` to the absolute path to the `valid_mask` file in it.

  The path corresponding to the `[CHECKPOINT]` parameter is the trained model CHECKPOINT path.

  The path of the parameter `[PREDICT_PATH]` corresponds to the output path of the label of the validation set through the model prediction. If the path already exists, it will be deleted and created again.

  The python commands above run in the background. The results can be viewed in the "eval.log" file.

# model-description

## performance

### training-performance

| Parameters                 | Ascend     |
| -------------------------- | ------------------------------------------------------------ |
| Model Version | D-LinkNet(DinkNet34) |
| Resource | Ascend 910 ; CPU 2.60GHz,192cores; Memory,755G; OS Euler2.8 |
| uploaded Date | 01/22/2022 (month/day/year)  |
| MindSpore Version  | 1.3.0 |
| Dataset             | DeepGlobe Road Extraction Dataset|
| Training Parameters    | 1pc: epoch=300, total steps=1401, batch_size = 4, lr=0.0002  |
| Optimizer | ADAM |
| Loss Function              | Dice Bce Loss|
| outputs | probability |
| Loss | 0.249542944|
| Speed | 1pc：407 ms/step; 8pc：430 ms/step |
| Total training time | 1pc：25.30h; 8pc：6.27h |
| Accuracy | IOU 98% |
| Parameters (M)  | 31M|
| Checkpoint for Fine tuning | 118.70M (.ckpt file)|
| configuration | dlinknet_config.yaml |
| Scripts| [D-LinkNet scripts](https://gitee.com/mindspore/models/tree/master/research/cv/dlinknet) |

### infer-performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version             | D-LinkNet(DinkNet34)                |
| Resource                 | Ascend 310；OS Euler2.8                  |
| uploaded Date       | 02/11/2022 (month/day/year)  |
| MindSpore Version   | 1.5.0                 |
| Dataset             | DeepGlobe Road Extraction Dataset    |
| batch size          | 1                         |
| Accuracy            | acc: 98.13% <br>  acc_cls: 87.19% <br>  iou: 0.9807  |
| Inference model | 118M (.mindir file)         |

### how-to-use

#### inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as Ascend 910 or Ascend 310, you
can refer to this [Link](https://www.mindspore.cn/tutorials/experts/en/master/infer/inference.html). Following
the steps below, this is a simple example:

##### running-on-ascend-310

Export MindIR on local

Before exporting, you need to modify the parameter in the configuration — trained_ckpt and batch_ Size . trained_ckpt is the CKPT file path, batch_ Size is set to 1.
You only need to use at least one of the two options to either (1) modify the above two parameters in the configuration file or (2) execute python statements with these two parameters.
The exported files will be generated directly in the same path as export.py.

```shell
python export.py --config_path=[CONFIG_PATH] --trained_ckpt=[trained_ckpt_path] --file_name=[model_name] --file_format=MINDIR --batch_size=1
```

ModelArtsExport on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```text
# Export on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "trained_ckpt='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "file_name='./dlinknet'" on default_config.yaml file.
#          Set "file_format='MINDIR'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "trained_ckpt='s3://dir_to_trained_ckpt/'" on the website UI interface.
#          Add "file_name='./dlinknet'" on the website UI interface.
#          Add "file_format='MINDIR'" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config_path="/path/yaml file" on the website UI interface.
# (3) Set the code directory to "/path/dlinknet" on the website UI interface.
# (4) Set the startup file to "export.py" on the website UI interface.
# (5) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

Before performing inference, the MINDIR file must be exported by export script on the 910 environment.
310 Inference Currently, only batch_Size 1 can be processed.

```shell
# Ascend310 inference
bash run_infer_310.sh [DATA_PATH] [LABEL_PATH] [MINDIR_PATH] [DEVICE_ID]
```

`DATA_PATH` is the image data folder path.
`LABEL_PATH` is the label image folder path.
`MINDIR_PATH` is the exported mindir model path.
`DEVICE_ID` is optional. The default value is 0.

The inference result is saved in the current path, and the final accuracy result can be seen in acc.log.

```text
acc:   0.9813138557017042
acc cls:   0.8719874771543723
iou:   0.980713602209491
```

## modelzoo-homepage

Please check the official [homepage](https://gitee.com/mindspore/models).
