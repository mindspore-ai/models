# Contents

- [Contents](#contents)
- [DeepLabV3 Description](#deeplabv3-description)
    - [Description](#description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage)
            - [Running on Ascend](#running-on-ascend)
            - [Running on CPU](#running-on-cpu)
        - [Result](#result)
            - [Running on Ascend](#running-on-ascend-1)
            - [Running on CPU](#running-on-cpu-1)
            - [Transfer Training](#transfer-training)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Running on Ascend](#running-on-ascend-2)
            - [Running ONNX evaluation](#running-onnx-evaluation)
        - [Result](#result-1)
            - [Training accuracy](#training-accuracy)
    - [Export MindIR](#export-mindir)
    - [Inference Process](#inference-process)
        - [Usage](#usage-2)
        - [result](#result-2)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
    - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DeepLabV3 Description](#contents)

## Description

DeepLab is a series of image semantic segmentation models, DeepLabV3 improves significantly over previous versions. Two keypoints of DeepLabV3: Its multi-grid atrous convolution makes it better to deal with segmenting objects at multiple scales, and augmented ASPP makes image-level features available to capture long range information.
This repository provides a script and recipe to DeepLabV3 model and achieve state-of-the-art performance.

Refer to [this paper][1] for network details.
`Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.`

[1]: https://arxiv.org/abs/1706.05587

# [Model Architecture](#contents)

Resnet101 as backbone, atrous convolution for dense feature extraction.

# [Dataset](#contents)

Pascal VOC datasets [link](https://host.robots.ox.ac.uk/pascal/VOC/), and Semantic Boundaries Dataset: [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)

- Download segmentation dataset.

- Prepare the training data list file. The list file saves the relative path to image and annotation pairs. Lines are like:

```shell
JPEGImages/00001.jpg SegmentationClassGray/00001.png
JPEGImages/00002.jpg SegmentationClassGray/00002.png
JPEGImages/00003.jpg SegmentationClassGray/00003.png
JPEGImages/00004.jpg SegmentationClassGray/00004.png
......
```

You can also generate the list file automatically by run script: `python get_dataset_lst.py --data_root=/PATH/TO/DATA`

- Configure and run build_data.sh to convert dataset to mindrecords. Arguments in scripts/build_data.sh:

 ```shell
 --data_root                 root path of training data
 --data_lst                  list of training data(prepared above)
 --dst_path                  where mindrecords are saved
 --num_shards                number of shards of the mindrecords
 --shuffle                   shuffle or not
 ```

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend）
- Prepare hardware environment with Ascend.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- Install python packages in requirements.txt
- Generate config json file for 8pcs training

     ```bash
     # From the root of this project
     cd src/tools/
     python3 get_multicards_json.py 10.111.*.*
     # 10.111.*.* is the computer's ip address.
     ```

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Prepare backbone

Download resnet101 for [here](https://download.mindspore.cn/model_zoo/r1.2/resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78/resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78.ckpt).

- Running on Ascend

Based on original DeepLabV3 paper, we reproduce two training experiments on vocaug (also as trainaug) dataset and evaluate on voc val dataset.

For single device training, please config parameters, training script is:

```shell
Enter the shell script to modify the data_file and ckpt_pre_trained parameters
# example:
data_file=/home/DataSet/VOC2012/vocaug_mindrecords/vocaug.mindrecord0
ckpt_pre_trained=/home/model/deeplabv3/predtrained/resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78.ckpt

bash run_standalone_train.sh
```

- For 8 devices training, training steps are as follows:

1. Train s16 with vocaug dataset, finetuning from resnet101 pretrained model, script is:

```shell
Enter the shell script to modify the data_file and ckpt_pre_trained parameters
# example:
data_file=/home/DataSet/VOC2012/vocaug_mindrecords/vocaug.mindrecord0
ckpt_pre_trained=/home/model/deeplabv3/predtrained/resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78.ckpt

bash run_distribute_train_s16_r1.sh ~/hccl_8p.json
```

2. Train s8 with vocaug dataset, finetuning from model in previous step, training script is:

```shell
Enter the shell script to modify the data_file and ckpt_pre_trained parameters
# example:
data_file=/home/DataSet/VOC2012/vocaug_mindrecords/vocaug.mindrecord0
ckpt_pre_trained=/home/model/deeplabv3/predtrained/resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78.ckpt

bash run_distribute_train_s8_r1.sh ~/hccl_8p.json
```

3. Train s8 with voctrain dataset, finetuning from model in previous step, training script is:

```shell
Enter the shell script to modify the data_file and ckpt_pre_trained parameters
Note: This training pre-training weight uses the weight file of the previous training, and the data set has also changed
# example:
data_file=/home/DataSet/VOC2012/voctrain_mindrecords/votrain.mindrecord0
ckpt_pre_trained=/home/model/deeplabv3/ckpt/deeplabv3-800_330.ckpt


bash run_distribute_train_s8_r2.sh ~/hccl_8p.json
```

- For evaluation, evaluating steps are as follows:

1. Enter the shell script to modify the data_file and ckpt_pre_trained parameters

```shell
modify the parameter according local path
# example:
data_root=/home/DataSet/VOC2012
data_lst=/home/DataSet/VOC2012/voc_val_lst.txt
ckpt_path=/home/model/deeplabv3/ckpt/deeplabv3-800_330.ckpt
```

2. Eval s16 with voc val dataset, eval script is:

```shell
bash run_eval_s16.sh
```

3. Eval s8 with voc val dataset, eval script is:

```shell
bash run_eval_s8.sh
```

4. Eval s8 multiscale with voc val dataset, eval script is:

```shell
bash run_eval_s8_multiscale.sh
```

5. Eval s8 multiscale and flip with voc val dataset, eval script is:

```shell
bash run_eval_s8_multiscale_flip.sh
```

- Train on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

1. Train s16 with vocaug dataset on modelarts, finetuning from resnet101 pretrained model, training steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "data_file='/cache/data/vocaug/vocaug_mindrecord/vocaug_mindrecord0'" on base_config.yaml file.
    #          Set "checkpoint_url=/The path of checkpoint in S3/" on beta_config.yaml file.
    #          Set "ckpt_pre_trained=/cache/checkpoint_path/path_to_pretrain/resnet101.ckpt" on base_config.yaml file.
    #          Set "base_lr=0.08" on base_config.yaml file.
    #          Set "is_distributed=True" on base_config.yaml file.
    #          Set "save_steps=410" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "data_file=/cache/data/vocaug/vocaug_mindrecord/vocaug_mindrecord0" on the website UI interface.
    #          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
    #          Add "ckpt_pre_trained=/cache/checkpoint_path/path_to_pretrain/resnet101.ckpt" on the website UI interface.
    #          Add "base_lr=0.08" on the website UI interface.
    #          Add "is_distributed=True" on the website UI interface.
    #          Add "save_steps=410" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your pretrained model to S3 bucket.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/deeplabv3" on the website UI interface.
    # (5) Set the startup file to "train.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    ```

2. Train s8 with vocaug dataset on modelarts, finetuning from model in previous step, training steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "model='deeplab_v3_s8'" on base_config.yaml file.
    #          Set "train_epochs=800" on base_config.yaml file.
    #          Set "batch_size=16" on base_config.yaml file.
    #          Set "base_lr=0.02" on base_config.yaml file.
    #          Set "loss_scale=2048" on base_config.yaml file.
    #          Set "data_file='/cache/data/vocaug/vocaug_mindrecord/vocaug_mindrecord0'" on base_config.yaml file.
    #          Set "checkpoint_url=/The path of checkpoint in S3/" on beta_config.yaml file.
    #          Set "ckpt_pre_trained=/cache/checkpoint_path/path_to_pretrain/deeplab_v3_s16-300_41.ckpt" on base_config.yaml file.
    #          Set "is_distributed=True" on base_config.yaml file.
    #          Set "save_steps=820" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "model='deeplab_v3_s8'" on the website UI interface.
    #          Add "train_epochs=800" on the website UI interface.
    #          Add "batch_size=16" on the website UI interface.
    #          Add "base_lr=0.02" on the website UI interface.
    #          Add "loss_scale=2048" on the website UI interface.
    #          Add "data_file='/cache/data/vocaug/vocaug_mindrecord/vocaug_mindrecord0'" on the website UI interface.
    #          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
    #          Add "ckpt_pre_trained=/cache/checkpoint_path/path_to_pretrain/deeplab_v3_s16-300_41.ckpt" on the website UI interface.
    #          Add "is_distributed=True" on the website UI interface.
    #          Add "save_steps=820" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your pretrained model to S3 bucket.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/deeplabv3" on the website UI interface.
    # (5) Set the startup file to "train.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    ```

3. Train s8 with voctrain dataset on modelarts, finetuning from model in previous step, training steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "model='deeplab_v3_s8'" on base_config.yaml file.
    #          Set "batch_size=16" on base_config.yaml file.
    #          Set "base_lr=0.008" on base_config.yaml file.
    #          Set "loss_scale=2048" on base_config.yaml file.
    #          Set "data_file='/cache/data/vocaug/voctrain_mindrecord/voctrain_mindrecord00'" on base_config.yaml file.
    #          Set "checkpoint_url=/The path of checkpoint in S3/" on beta_config.yaml file.
    #          Set "ckpt_pre_trained=/cache/checkpoint_path/path_to_pretrain/deeplab_v3_s8-800_82.ckpt" on base_config.yaml file.
    #          Set "is_distributed=True" on base_config.yaml file.
    #          Set "save_steps=110" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "model='deeplab_v3_s8'" on the website UI interface.
    #          Add "batch_size=16" on the website UI interface.
    #          Add "base_lr=0.008" on the website UI interface.
    #          Add "loss_scale=2048" on the website UI interface.
    #          Add "data_file='/cache/data/vocaug/voctrain_mindrecord/voctrain_mindrecord00'" on the website UI interface.
    #          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
    #          Add "ckpt_pre_trained=/cache/checkpoint_path/path_to_pretrain/deeplab_v3_s8-800_82.ckpt" on the website UI interface.
    #          Add "is_distributed=True" on the website UI interface.
    #          Add "save_steps=110" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your pretrained model to S3 bucket.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/deeplabv3" on the website UI interface.
    # (5) Set the startup file to "train.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    ```

- Eval on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Eval s16 with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "model='deeplab_v3_s16'" on base_config.yaml file.
    #          Set "batch_size=32" on base_config.yaml file.
    #          Set "scales_type=0" on base_config.yaml file.
    #          Set "freeze_bn=True" on base_config.yaml file.
    #          Set "data_root='/cache/data/vocaug'" on base_config.yaml file.
    #          Set "data_lst='/cache/data/vocaug/voc_val_lst.txt'" on base_config.yaml file.
    #          Set "checkpoint_url=/The path of checkpoint in S3/" on beta_config.yaml file.
    #          Set "ckpt_path='/cache/checkpoint_path/path_to_pretrain/deeplab_v3_s16-300_41.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "model=deeplab_v3_s16" on the website UI interface.
    #          Add "batch_size=32" on the website UI interface.
    #          Add "scales_type=0" on the website UI interface.
    #          Add "freeze_bn=True" on the website UI interface.
    #          Add "data_root=/cache/data/vocaug" on the website UI interface.
    #          Add "data_lst=/cache/data/vocaug/voc_val_lst.txt" on the website UI interface.
    #          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
    #          Add "ckpt_path=/cache/checkpoint_path/path_to_pretrain/deeplab_v3_s16-300_41.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your pretrained model to S3 bucket.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/deeplabv3" on the website UI interface.
    # (5) Set the startup file to "eval.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    ```

2. Eval s8 with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "model='deeplab_v3_s8'" on base_config.yaml file.
    #          Set "batch_size=16" on base_config.yaml file.
    #          Set "scales_type=0" on base_config.yaml file.
    #          Set "freeze_bn=True" on base_config.yaml file.
    #          Set "data_root='/cache/data/vocaug'" on base_config.yaml file.
    #          Set "data_lst='/cache/data/vocaug/voc_val_lst.txt'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_path='/cache/checkpoint_path/path_to_pretrain/deeplab_v3_s8-300_11.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "model=deeplab_v3_s8" on the website UI interface.
    #          Add "batch_size=16" on the website UI interface.
    #          Add "scales_type=0" on the website UI interface.
    #          Add "freeze_bn=True" on the website UI interface.
    #          Add "data_root=/cache/data/vocaug" on the website UI interface.
    #          Add "data_lst=/cache/data/vocaug/voc_val_lst.txt" on the website UI interface.
    #          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
    #          Add "ckpt_path=/cache/checkpoint_path/path_to_pretrain/deeplab_v3_s8-300_11.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your pretrained model to S3 bucket.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/deeplabv3" on the website UI interface.
    # (5) Set the startup file to "eval.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    ```

3. Eval s8 multiscale with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "model='deeplab_v3_s8'" on base_config.yaml file.
    #          Set "batch_size=16" on base_config.yaml file.
    #          Set "scales_type=1" on base_config.yaml file.
    #          Set "freeze_bn=True" on base_config.yaml file.
    #          Set "data_root='/cache/data/vocaug'" on base_config.yaml file.
    #          Set "data_lst='/cache/data/vocaug/voc_val_lst.txt'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_path='/cache/checkpoint_path/path_to_pretrain/deeplab_v3_s8-300_11.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "model=deeplab_v3_s8" on the website UI interface.
    #          Add "batch_size=16" on the website UI interface.
    #          Add "scales_type=1" on the website UI interface.
    #          Add "freeze_bn=True" on the website UI interface.
    #          Add "data_root=/cache/data/vocaug" on the website UI interface.
    #          Add "data_lst=/cache/data/vocaug/voc_val_lst.txt" on the website UI interface.
    #          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
    #          Add "ckpt_path=/cache/checkpoint_path/path_to_pretrain/deeplab_v3_s8-300_11.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your pretrained model to S3 bucket.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/deeplabv3" on the website UI interface.
    # (5) Set the startup file to "eval.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    ```

4. Eval s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "model='deeplab_v3_s8'" on base_config.yaml file.
    #          Set "batch_size=16" on base_config.yaml file.
    #          Set "scales_type=1" on base_config.yaml file.
    #          Set "freeze_bn=True" on base_config.yaml file.
    #          Set "flip=True" on base_config.yaml file.
    #          Set "data_root='/cache/data/vocaug'" on base_config.yaml file.
    #          Set "data_lst='/cache/data/vocaug/voc_val_lst.txt'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_path='/cache/checkpoint_path/path_to_pretrain/deeplab_v3_s8-300_11.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "model=deeplab_v3_s8" on the website UI interface.
    #          Add "batch_size=16" on the website UI interface.
    #          Add "scales_type=1" on the website UI interface.
    #          Add "freeze_bn=True" on the website UI interface.
    #          Add "flip=True" on the website UI interface.
    #          Add "data_root=/cache/data/vocaug" on the website UI interface.
    #          Add "data_lst=/cache/data/vocaug/voc_val_lst.txt" on the website UI interface.
    #          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
    #          Add "ckpt_path=/cache/checkpoint_path/path_to_pretrain/deeplab_v3_s8-300_11.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your pretrained model to S3 bucket.
    # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
    # (4) Set the code directory to "/path/deeplabv3" on the website UI interface.
    # (5) Set the startup file to "eval.py" on the website UI interface.
    # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (7) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Export s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "export_model='deeplab_v3_s8'" on base_config.yaml file.
    #          Set "export_batch_size=1" on base_config.yaml file.
    #          Set "file_name='deeplabv3'" on base_config.yaml file.
    #          Set "file_format='MINDIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "export_model='deeplab_v3_s8'" on the website UI interface.
    #          Add "export_batch_size=1" on the website UI interface.
    #          Add "file_name='deeplabv3'" on the website UI interface.
    #          Add "file_format='MINDIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/deeplabv3" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└──deeplabv3
  ├── README.md
  ├── scripts
    ├── build_data.sh                             # convert raw data to mindrecord dataset
    ├── run_distribute_train_s16_r1.sh            # launch ascend distributed training(8 pcs) with vocaug dataset in s16 structure
    ├── run_distribute_train_s8_r1.sh             # launch ascend distributed training(8 pcs) with vocaug dataset in s8 structure
    ├── run_distribute_train_s8_r2.sh             # launch ascend distributed training(8 pcs) with voctrain dataset in s8 structure
    ├── run_eval_s16.sh                           # launch ascend evaluation in s16 structure
    ├── run_eval_s8.sh                            # launch ascend evaluation in s8 structure
    ├── run_eval_s8_multiscale.sh                 # launch ascend evaluation with multiscale in s8 structure
    ├── run_eval_s8_multiscale_filp.sh            # launch ascend evaluation with multiscale and filp in s8 structure
    ├── run_eval_onnx.sh                          # launch onnx evaluation with multiscale and flip
    ├── run_standalone_train.sh                   # launch ascend standalone training(1 pc)
    ├── run_standalone_train_cpu.sh               # launch CPU standalone training
  ├── src
    ├── data
        ├── dataset.py                            # mindrecord data generator
        ├── build_seg_data.py                     # data preprocessing
        ├── get_dataset_lst.py                    # dataset list file generator
    ├── loss
       ├── loss.py                                # loss definition for deeplabv3
    ├── nets
       ├── deeplab_v3
          ├── deeplab_v3.py                       # DeepLabV3 network structure
       ├── net_factory.py                         # set S16 and S8 structures
    ├── tools
       ├── get_multicards_json.py                 # get rank table file
    └── utils
       └── learning_rates.py                      # generate learning rate
  ├── convert_resnet101.py                        # convert resnet101 as backbone
  ├── eval.py                                     # eval net
  ├── train.py                                    # train net
  └── requirements.txt                            # requirements file
```

## [Script Parameters](#contents)

Default configuration

```shell
"data_file":"/PATH/TO/MINDRECORD_NAME"            # dataset path
"device_target":Ascend                            # device target
"train_epochs":300                                # total epochs
"batch_size":32                                   # batch size of input tensor
"crop_size":513                                   # crop size
"base_lr":0.08                                    # initial learning rate
"lr_type":cos                                     # decay mode for generating learning rate
"min_scale":0.5                                   # minimum scale of data argumentation
"max_scale":2.0                                   # maximum scale of data argumentation
"ignore_label":255                                # ignore label
"num_classes":21                                  # number of classes
"model":deeplab_v3_s16                            # select model
"ckpt_pre_trained":"/PATH/TO/PRETRAIN_MODEL"      # path to load pretrain checkpoint
"is_distributed":                                 # distributed training, it will be True if the parameter is set
"save_steps":410                                  # steps interval for saving
"keep_checkpoint_max":200                         # max checkpoint for saving
```

## [Training Process](#contents)

### Usage

#### Running on Ascend

Based on original DeepLabV3 paper, we reproduce two training experiments on vocaug (also as trainaug) dataset and evaluate on voc val dataset.

For single device training, please config parameters, training script is as follows:

```shell
# run_standalone_train.sh
python ${train_code_path}/train.py --data_file=/PATH/TO/MINDRECORD_NAME  \
                    --train_dir=${train_path}/ckpt  \
                    --train_epochs=200  \
                    --batch_size=32  \
                    --crop_size=513  \
                    --base_lr=0.015  \
                    --lr_type=cos  \
                    --min_scale=0.5  \
                    --max_scale=2.0  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s16  \
                    --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                    --save_steps=1500  \
                    --keep_checkpoint_max=200 >log 2>&1 &
```

For 8 devices training, training steps are as follows:

1. Train s16 with vocaug dataset, finetuning from resnet101 pretrained model, script is as follows:

    ```shell
    # run_distribute_train_s16_r1.sh
    for((i=0;i<=$RANK_SIZE-1;i++));
    do
        export RANK_ID=${i}
        export DEVICE_ID=$((i + RANK_START_ID))
        echo 'start rank='${i}', device id='${DEVICE_ID}'...'
        mkdir ${train_path}/device${DEVICE_ID}
        cd ${train_path}/device${DEVICE_ID} || exit
        python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                                --data_file=/PATH/TO/MINDRECORD_NAME  \
                                                --train_epochs=300  \
                                                --batch_size=32  \
                                                --crop_size=513  \
                                                --base_lr=0.08  \
                                                --lr_type=cos  \
                                                --min_scale=0.5  \
                                                --max_scale=2.0  \
                                                --ignore_label=255  \
                                                --num_classes=21  \
                                                --model=deeplab_v3_s16  \
                                                --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                                --is_distributed  \
                                                --save_steps=410  \
                                                --keep_checkpoint_max=200 >log 2>&1 &
    done
    ```

2. Train s8 with vocaug dataset, finetuning from model in previous step, training script is as follows:

    ```shell
    # run_distribute_train_s8_r1.sh
    for((i=0;i<=$RANK_SIZE-1;i++));
    do
        export RANK_ID=${i}
        export DEVICE_ID=$((i + RANK_START_ID))
        echo 'start rank='${i}', device id='${DEVICE_ID}'...'
        mkdir ${train_path}/device${DEVICE_ID}
        cd ${train_path}/device${DEVICE_ID} || exit
        python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                                --data_file=/PATH/TO/MINDRECORD_NAME  \
                                                --train_epochs=800  \
                                                --batch_size=16  \
                                                --crop_size=513  \
                                                --base_lr=0.02  \
                                                --lr_type=cos  \
                                                --min_scale=0.5  \
                                                --max_scale=2.0  \
                                                --ignore_label=255  \
                                                --num_classes=21  \
                                                --model=deeplab_v3_s8  \
                                                --loss_scale=2048  \
                                                --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                                --is_distributed  \
                                                --save_steps=820  \
                                                --keep_checkpoint_max=200 >log 2>&1 &
    done
    ```

3. Train s8 with voctrain dataset, finetuning from model in previous step, training script is as follows:

```shell
# run_distribute_train_s8_r2.sh
for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=${i}
    export DEVICE_ID=$((i + RANK_START_ID))
    echo 'start rank='${i}', device id='${DEVICE_ID}'...'
    mkdir ${train_path}/device${DEVICE_ID}
    cd ${train_path}/device${DEVICE_ID} || exit
    python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH/TO/MINDRECORD_NAME  \
                                               --train_epochs=300  \
                                               --batch_size=16  \
                                               --crop_size=513  \
                                               --base_lr=0.008  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=deeplab_v3_s8  \
                                               --loss_scale=2048  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=110  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
```

#### Running on CPU

For CPU training, please config parameters, training script is as follows:

```shell
# run_standalone_train_cpu.sh
python ${train_code_path}/train.py --data_file=/PATH/TO/MINDRECORD_NAME  \
                    --device_target=CPU  \
                    --train_dir=${train_path}/ckpt  \
                    --train_epochs=200  \
                    --batch_size=32  \
                    --crop_size=513  \
                    --base_lr=0.015  \
                    --lr_type=cos  \
                    --min_scale=0.5  \
                    --max_scale=2.0  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s16  \
                    --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                    --save_steps=1500  \
                    --keep_checkpoint_max=200 >log 2>&1 &
```

### Result

#### Running on Ascend

- Training vocaug in s16 structure

```shell
# distribute training result(8p)
epoch: 1 step: 41, loss is 0.8319108
epoch time: 213856.477 ms, per step time: 5216.012 ms
epoch: 2 step: 41, loss is 0.46052963
epoch time: 21233.183 ms, per step time: 517.883 ms
epoch: 3 step: 41, loss is 0.45012417
epoch time: 21231.951 ms, per step time: 517.852 ms
epoch: 4 step: 41, loss is 0.30687785
epoch time: 21199.911 ms, per step time: 517.071 ms
epoch: 5 step: 41, loss is 0.22769661
epoch time: 21240.281 ms, per step time: 518.056 ms
epoch: 6 step: 41, loss is 0.25470978
...
```

- Training vocaug in s8 structure

```shell
# distribute training result(8p)
epoch: 1 step: 82, loss is 0.024167
epoch time: 322663.456 ms, per step time: 3934.920 ms
epoch: 2 step: 82, loss is 0.019832281
epoch time: 43107.238 ms, per step time: 525.698 ms
epoch: 3 step: 82, loss is 0.021008959
epoch time: 43109.519 ms, per step time: 525.726 ms
epoch: 4 step: 82, loss is 0.01912349
epoch time: 43177.287 ms, per step time: 526.552 ms
epoch: 5 step: 82, loss is 0.022886964
epoch time: 43095.915 ms, per step time: 525.560 ms
epoch: 6 step: 82, loss is 0.018708453
epoch time: 43107.458 ms per step time: 525.701 ms
...
```

- Training voctrain in s8 structure

```shell
# distribute training result(8p)
epoch: 1 step: 11, loss is 0.00554624
epoch time: 199412.913 ms, per step time: 18128.447 ms
epoch: 2 step: 11, loss is 0.007181881
epoch time: 6119.375 ms, per step time: 556.307 ms
epoch: 3 step: 11, loss is 0.004980865
epoch time: 5996.978 ms, per step time: 545.180 ms
epoch: 4 step: 11, loss is 0.0047651967
epoch time: 5987.412 ms, per step time: 544.310 ms
epoch: 5 step: 11, loss is 0.006262637
epoch time: 5956.682 ms, per step time: 541.517 ms
epoch: 6 step: 11, loss is 0.0060750707
epoch time: 5962.164 ms, per step time: 542.015 ms
...
```

#### Running on CPU

- Training voctrain in s16 structure

```bash
epoch: 1 step: 1, loss is 3.655448
epoch: 2 step: 1, loss is 1.5531876
epoch: 3 step: 1, loss is 1.5099041
...
```

#### Transfer Training

You can train your own model based on pretrained model. You can perform transfer training by following steps.

1. Convert your own dataset to Pascal VOC datasets. Otherwise you have to add your own data preprocess code.
2. Set argument `filter_weight` to `True`, `ckpt_pre_trained` to pretrained checkpoint and `num_classes` to the classes of your dataset while calling `train.py`, this will filter the final conv weight from the pretrained model.
3. Build your own bash scripts using new config and arguments for further convenient.

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

Configure checkpoint with --ckpt_path and dataset path. Then run script, mIOU will be printed in eval_path/eval_log.

```shell
./run_eval_s16.sh                     # test s16
./run_eval_s8.sh                      # test s8
./run_eval_s8_multiscale.sh           # test s8 + multiscale
./run_eval_s8_multiscale_flip.sh      # test s8 + multiscale + flip
```

Example of test script is as follows:

```shell
python ${train_code_path}/eval.py --data_root=/PATH/TO/DATA  \
                    --data_lst=/PATH/TO/DATA_lst.txt  \
                    --batch_size=16  \
                    --crop_size=513  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s8  \
                    --scales=0.5  \
                    --scales=0.75  \
                    --scales=1.0  \
                    --scales=1.25  \
                    --scales=1.75  \
                    --flip  \
                    --freeze_bn  \
                    --ckpt_path=/PATH/TO/PRETRAIN_MODEL >${eval_path}/eval_log 2>&1 &
```

#### Running ONNX evaluation

First, export your model:

```shell
python export.py \
  --freeze_bn False \
  --export_model [deeplab_v3_s16 or deeplab_v3_s8] \
  --ckpt_file /path/to/checkpoint.ckpt \
  --device_target GPU \
  --file_name /path/to/exported.onnx \
  --export_batch_size [batch size]
  --file_format ONNX
```

Next, run evaluation:

```shell
python eval_onnx.py \
  --file_name /path/to/exported.onnx
  --data_root /path/to/VOC2012/
  --data_lst /path/to/VOC2012/voc_val_lst.txt \
  --device_target GPU \
  --batch_size [batch size]

or

bash run_eval_onnx.sh [DATA_ROOT] [DATA_LST] [FILE_NAME]
```

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy

| **Network**    | OS=16 | OS=8 | MS   | Flip  | mIOU  | mIOU in paper |
| :----------: | :-----: | :----: | :----: | :-----: | :-----: | :-------------: |
| deeplab_v3 | √     |      |      |       | 77.37 | 77.21    |
| deeplab_v3 |       | √    |      |       | 78.84 | 78.51    |
| deeplab_v3 |       | √    | √    |       | 79.70 |79.45   |
| deeplab_v3 |       | √    | √    | √     | 79.89 | 79.77        |

Note: There OS is output stride, and MS is multiscale.

## [Export MindIR](#contents)

Currently, batchsize can only set to 1.

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --freeze_bn True
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

## [Inference Process](#contents)

### Usage

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the air file must bu exported by export script on the 910 environment.
Current batch_Size can only be set to 1. The precision calculation process needs about 70G+ memory space.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [DATA_ROOT] [DATA_LIST] [DEVICE_TYPE] [DEVICE_ID]
```

`DEVICE_TYPE` can choose from [Ascend, GPU, CPU].
`DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result in acc.log file.

| **Network**    | OS=16 | OS=8 | MS   | Flip  | mIOU  | mIOU in paper |
| :----------: | :-----: | :----: | :----: | :-----: | :-----: | :-------------: |
| deeplab_v3 |       | √    |      |       | 78.84 | 78.51    |

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend 910
| -------------------------- | -------------------------------------- |
| Model Version              | DeepLabV3
| Resource                   | Ascend 910; OS Euler2.8 |
| Uploaded Date              | 07/05/2021 (month/day/year)          |
| MindSpore Version          | 1.3.0                       |
| Dataset                    | PASCAL VOC2012 + SBD              |
| Training Parameters        | epoch = 300, batch_size = 32 (s16_r1) <br> epoch = 800, batch_size = 16 (s8_r1)   <br>    epoch = 300, batch_size = 16 (s8_r2) |
| Optimizer                  | Momentum                                 |
| Loss Function              | Softmax Cross Entropy                                  |
| Outputs                    | probability                                       |
| Loss                       | 0.0065883575                                       |
| Speed                      | 60 fps（1pc, s16）<br> 480 fps（8pcs, s16） <br> 244 fps (8pcs, s8)      |
| Total time                 | 8pcs: 706 mins                     |
| Parameters (M)             | 58.2                                       |
| Checkpoint for Fine tuning | 443M (.ckpt file)                       |
| Model for inference        | 223M (.air file)                     |
| Scripts                    | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/DeepLabV3P) |

## Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | DeepLabV3 V1                |
| Resource            | Ascend 910; OS Euler2.8                   |
| Uploaded Date       | 07/05/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                 |
| Dataset             | VOC datasets    |
| batch_size          | 32 (s16); 16 (s8)                         |
| outputs             | probability                 |
| Accuracy            | 8pcs: <br> s16: 77.37 <br>  s8: 78.84% <br>  s8_multiscale: 79.70% <br> s8_Flip: 79.89%  |
| Model for inference | 443M (.ckpt file)         |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
