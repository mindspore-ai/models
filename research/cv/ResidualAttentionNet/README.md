# Contents

[查看中文](./README_CN.md)

<!-- TOC -->

- [Contents](#contents)
- [ResidualAttentionNet_Description](#residualattentionnet-description)
- [Model_Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Export Process](#export-process)
        - [Export](#export)
    - [Inference Process](#inference-process)
        - [Inference](#inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
            - [ResidualAttentionNet train on CIFAR-10](#residualattentionnet-train-on-cifar-10)
            - [ResidualAttentionNet train on ImageNet2012](#residualattentionnet-train-on-imagenet2012)
        - [Inference Performance](#inference-performance)
            - [ResidualAttentionNet infer on CIFAR-10](#residualattentionnet-infer-on-cifar-10)
            - [ResidualAttentionNet infer on ImageNet2012](#residualattentionnet-infer-on-imagenet2012)
    - [How to use](#how-to-use)
        - [Inference](#inference-1)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
        - [Transfer Learning](#transfer-learning)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# [ResidualAttentionNet_Description](#contents)

In the work of Residual_Attention_Net, the residual attentional network is proposed, which uses the convolutional neural network of attentional mechanism. It can be combined with the advanced feed forward network architecture in the end-to-end training mode. The residual attention network is composed of the superposition of attention modules which generate the feature of attentional perception. The attentional perception characteristics of different modules change adaptively with the deepening of the level. In each attention module, a bottom-up and top-down feed forward structure is adopted to expand the attention process of feed forward and feedback into a single feed forward process. Importantly, using attentional residuals to train very deep residual attentional networks can easily scale to hundreds of layers.

[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Residual_Attention_Network_CVPR_2017_paper.pdf)：Residual Attention Network for Image Classification (CVPR-2017 Spotlight) By Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Chen Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang

# [Model Architecture](#contents)

The model is stacked with multiple residual modules and attention modules. ResidualAttentionModel_92 is stacked by AttentionModule_stage1x1, AttentionModule_stage2x2, AttentionModule_stage3x3 and multiple residual structures。ResidualAttentionModel_56 is stacked by AttentionModule_stage1x1, AttentionModule_stage2x1, AttentionModule_stage3x1 and multiple residual structures.

# [Dataset](#contents)

Dataset used：[CIFAR-10](https://gitee.com/link?target=http%3A%2F%2Fwww.cs.toronto.edu%2F~kriz%2Fcifar.html)

- Dataset size：175M，60,000 32*32 colorful images in 10 classes
    - Train：146M，50,000 images  
    - Test：29M，10,000 images
- Data format：binary files
    - Note：Data will be processed in src/dataset.py, the data_path must be specified in config.

Dataset used：[ImageNet2012](https://gitee.com/link?target=http%3A%2F%2Fwww.image-net.org%2F)

- Dataset size: 224*224 colorful images in 1000 classes
    - Train：1,281,167 images  
    - Test： 50,000 images
- Data format：jpeg
    - Note: Data will be processed in src/dataset.py, the data_path must be specified in config.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend/GPU/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

  ```bash
  # Add data set path, take training cifar10 as an example
  data_path:/home/DataSet/cifar10/
  # Add config path parameters before inference
  config_path:/home/config.yaml
  # Add checkpoint path parameters before inference
  checkpoint_file_path:path_to_ckpt/cifar10-300.ckpt
  ```

  ```bash
  # run training example
  python train.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] > train.log &
  # example: python train.py --config_path config/cifar10_Ascend_1p_config.yaml --data_path /data/cifar10/ > train.log &
  or
  bash scripts/run_standalone_train.sh [DATA_PATH] [CONFIG_PATH]
  # example: bash scripts/run_standalone_train.sh data/cifar10/ ../config/cifar10_Ascend_1p_config.yaml
  # run distributed training example
  bash scripts/run_distribute8_train.sh [RANK_TABLE_FILE] [DATA_PATH] [CONFIG_PATH]
  # example: bash scripts/run_standalone_train.sh rank_table.json /data/cifar10/ ../config/cifar10_Ascend_8p_config.yaml
  # run evaluation example
  python eval.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] > eval.log &
  # example: python  eval.py --config_path config/cifar10_Ascend_1p_config.yaml --data_path /data/cifar10/ > eval.log &
  # run inferenct example
  bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [CONFIG_PATH] [DEVICE_ID]
  # example: bash run_infer_310.sh cifar10-300.mindir cifar10 data/cifar10/ ../config/cifar10_Ascend_1p_config.yaml 0
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below: https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.

- running on CPU

  For running on CPU, please change `device_target` from `Ascend` to `CPU` in configuration file [dataset]_config.yaml

  ```bash
  # run training example
  python train.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] > train.log &
  # example: python train.py --config_path config/cifar10_CPU_config.yaml --data_path /data/cifar10/ > train.log &
  or
  bash scripts/run_standalone_train.sh [DATA_PATH] [CONFIG_PATH]
  # example: bash scripts/run_standalone_train.sh /data/cifar10/ ../config/cifar10_CPU_config.yaml
  # run evaluation example
  python eval.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] > eval.log &
  # example: python  eval.py --config_path config/cifar10_CPU_config.yaml --data_path /data/cifar10/ > eval.log &
  ```

We use CIFAR-10 dataset by default. Your can also pass `$dataset` to the scripts so that select different datasets. For more details, please refer the specify script.

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    - Train imagenet 8p on ModelArts

      ```bash
      # (1) Set the startup file to "train.py" on the website UI interface.
      # (2) Set the dataset to "imagenet.zip" on the website UI interface.
      # (3) Add "config_path='/path_to_code/imagenet2012_Modelart_config.yaml'" on the website UI interface.
      # (4) Add "lr_init"、"epoch_size"、"lr_decay_mode" on the website UI interface.
      # (5) Set 8p on the website UI interface.
      # (6) Create your job.
      ```

    - Eval imagenet on ModelArts

      ```bash
      # (1) Set the startup file to "eval.py" on the website UI interface.
      # (2) Set the dataset to "imagenet.zip" on the website UI interface.
      # (3) Add "config_path='/path_to_code/imagenet2012_Modelart_config.yaml'" on the website UI interface.
      # (4) Add "chcekpoint_file_path='path_to_ckpt/imagenet2012-300.ckpt'" on the website UI interface.
      # (5) Create your job.
      ```

    - Train cifar10 8p on ModelArts

      ```bash
      # (1) Set the startup file to "train.py" on the website UI interface.
      # (2) Set the dataset to "cifar10-bin.zip" on the website UI interface.
      # (3) Add "config_path='/path_to_code/cifar10_Modelart_8p_config.yaml'" on the website UI interface.
      # (4) Add "lr" on the website UI interface.
      # (5) Set 8p on the website UI interface.
      # (6) Create your job.
      ```

    - Eval cifar10 on ModelArts

      ```bash
      # (1) Set the startup file to "eval.py" on the website UI interface.
      # (2) Set the dataset to "cifar10-bin.zip" on the website UI interface.
      # (3) Add "config_path='/path_to_code/cifar10_Modelart_1p_config.yaml'" on the website UI interface.
      # (4) Add "chcekpoint_file_path='path_to_ckpt/cifar10-300.ckpt'" on the website UI interface.
      # (5) Create your job.
      ```

    - Export on ModelArts

      ```bash
      # (1) Set the startup file to "export.py" on the website UI interface.
      # (2) Set the dataset to "cifar10-bin.zip" on the website UI interface.
      # (3) Add "config_path='/path_to_code/cifar10_Modelart_1p_config.yaml'" on the website UI interface.
      # (4) Add "ckpt_file='path_to_ckpt/cifar10-300.ckpt'" on the website UI interface.
      # (5) Create your job.
      ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
├── model_zoo
    ├── README.md                          // descriptions about all the models
    ├── residual_attention_net
        ├── README.md                    // descriptions about residual_attention_net
        ├── ascend310_infer              // application for 310 inference
        ├── config
        │   ├──cifar10_Ascend_1p_config.yaml      // shell script of cifar10 dataset of 1p on Ascend
        │   ├──cifar10_Ascend_8p_config.yaml      // shell script of cifar10 dataset of 8p on Ascend
        │   ├──cifar10_Modelart_1p_confip_g.yaml  // shell script of cifar10 dataset of 1p on Modelart
        │   ├──cifar10_Modelart_8p_config.yaml    // shell script of cifar10 dataset of 8p on Modelart
        │   ├──cifar10_CPU_config.yaml            // shell script of cifar10 dataset on CPU
        │   ├──imagenet2012_Ascend_config.yaml    // shell script of imagenet2012 dataset on Ascend
        │   ├──imagenet2012_Modelart_config.yaml  // shell script of imagenet2012 dataset on Modelart
        ├── model                                 // model architecture
        │   ├──attention_module.py
        │   ├──basic_layers.py
        │   ├──residual_attention_network.py
        ├── scripts
        │   ├──run_distribute_train.sh        // shell script of distributed 8P to Ascend
        │   ├──run_standalone_train.sh         // shell script of single card to Ascend
        │   ├──run_infer_310.sh                // shell script of Ascend310 infer
        ├── src
        │   ├──model_utils                    // related configuration
        │   ├──conv2d_ops.py                  // Convolution operator transformation
        │   ├──cross_entropy_loss_ops.py      // Loss function
        │   ├──dataset.py                     // creating dataset
        │   ├──eval_callback.py               // eval_callback setting
        │   ├──local_adapter.py               // local setting
        │   ├──lr_generator.py                // lr setting
        │   ├──moxing_adapter.py              // moxing setting
        ├── train.py               // training script
        ├── eval.py                // evaluation script
        ├── postprogress.py        // post process for 310 inference
        ├── export.py              // export checkpoint files into air/mindir
        ├── preprocess.py          // pre process for 310 inference
        ├── create_imagenet2012_label.py    // 310 Inference imagenet2012 data label processing
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- 1p config for ResidualAttentionNet, CIFAR-10 dataset

  ```bash
  'lr':0.1                 # initial learning rate
  'batch_size':64          # training batchsize
  'epoch_size':220         # total training epochs
  'momentum':0.9           # momentum
  'weight_decay':1e-4      # weight decay value
  'image_height':32        # image height used as input to the model
  'image_width':32         # image width used as input to the model
  'data_path':'./data'     # absolute full path to th train and evalluaton datasets
  'device_target':'Ascend' # device running the program
  'device_id':4            # device ID used to train or evaluate the dataset. Ignore it when you use run_train.sh for distributed training
  'keep_checkpoint_max':10 # only keep the last keep_checkpoint_max checkpoint
  'checkpoint_file_path':path_to_ckpt/cifar10-300.ckpt   # the absolute full path to save the checkpoint file
  ```

- 8p config for ResidualAttentionNet, CIFAR-10 dataset

  ```bash
  'lr':0.24                # initial learning rate
  'batch_size':64          # training batchsize
  'epoch_size':220         # total training epochs
  'momentum':0.9           # momentum
  'weight_decay':1e-4      # weight decay value
  'image_height':32        # image height used as input to the model
  'image_width':32         # image width used as input to the model
  'data_path':'./data'     # absolute full path to th train and evalluaton datasets
  'device_target':'Ascend' # device running the program
  'device_id':4            # device ID used to train or evaluate the dataset. Ignore it when you use run_train.sh for distributed training
  'keep_checkpoint_max':10 # only keep the last keep_checkpoint_max checkpoint
  'checkpoint_file_path':path_to_ckpt/cifar10-300.ckpt   # the absolute full path to save the checkpoint file
  ```

- 8p config for ResidualAttentionNet, ImageNet2012 dataset

  ```bash
  'lr_init': 0.24           # initial learning rate
  'batch_size': 32          # training batchsize
  'epoch_size': 60          # total training epochs
  'momentum': 0.9           # momentum
  'weight_decay': 1e-4      # weight decay value
  'image_height': 224       # image height used as input to the model
  'image_width': 224        # image width used as input to the model
  'data_path':'./data'      # absolute full path to th train and evalluaton datasets
  'device_target': 'Ascend' # device running the program
  'device_id': 0            # device ID used to train or evaluate the dataset. Ignore it when you use run_train.sh for distributed training
  'keep_checkpoint_max': 10 # only keep the last keep_checkpoint_max checkpoint
  'checkpoint_file_path':path_to_ckpt/imagenet2012-300.ckpt  # the absolute full path to save the checkpoint file
  'lr_scheduler': 'exponential'     # learning rate scheduler
  'warmup_epochs': 0         # warmup epoch
  'loss_scale': 1024         # loss scale
  ```

For more configuration details, please refer the script `config.py`.

## [Training Process](#contents)

### [Training](#contents)

- running on Ascend

  ```bash
  # run training example
  python train.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] > train.log &
  # example: python train.py --config_path config/cifar10_Ascend_1p_config.yaml --data_path /data/cifar10/ > train.log &
  or
  bash scripts/run_standalone_train.sh [DATA_PATH] [CONFIG_PATH]
  # example: bash scripts/run_standalone_train.sh data/cifar10/ ../config/cifar10_Ascend_1p_config.yaml
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```text
  # grep "loss is " train.log
  epoch:1 step:97, loss is 1.4842823
  epcoh:2 step:97, loss is 1.0897788
  ...
  ```

  The model checkpoint will be saved in the current directory.

- running on CPU

  ```bash
  # run training example
  python train.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] > train.log &
  # example: python train.py --config_path config/cifar10_CPU_config.yaml --data_path /data/cifar10/ > train.log &
  or
  bash scripts/run_standalone_train.sh [DATA_PATH] [CONFIG_PATH]
  # example: bash scripts/run_standalone_train.sh /data/cifar10/ ../config/cifar10_CPU_config.yaml
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the folder defined in config.yaml.

### [Distributed Training](#contents)

- running on Ascend

  ```bash
  # run distributed training example
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [CONFIG_PATH]
  # example: bash scripts/run_standalone_train.sh rank_table.json /data/cifar10/ ../config/cifar10_Ascend_8p_config.yaml
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `log`. The loss value will be achieved as follows:

  ```text
  log:epoch:1 step:48, loss is 1.4302931
  log:epcoh:2 step:48, loss is 1.4023874
  ...
  ```

## [Evaluation Process](#contents)

### [Evaluation](#contents)

- evaluation on CIFAR-10 dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set  the absolute full path of checkpoint_file_path，eg: “username/RedidualAttentionNet/mindspore_cifar10.ckpt”

  ```bash
  python eval.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] --checkpoint_file_path [CHECKPOINT_FILE_PATH] > eval.log &
  # example: python eval.py --config_path config/cifar10_Ascend_1p_config.yaml --data_path /data/cifar10/ --checkpoint_file_path cifar10-1p.ckpt > eval.log &
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```bash
  grep "accuracy" eval.log
  # accuracy：{'top_1_accuracy':0.9952 'top_5_accuracy:0.9978'}
  ```

## [Export Process](#contents)

### [Export](#content)

Before export model, you must modify the config file, Cifar10 config file is config/cifar10_Ascend_config.yam and imagenet config file is config/imagenet2012_Ascend_config.yaml.
The config items you should modify are file_name and ckpt_file.

```bash
python export.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] --ckpt_file [CKPT_FILE] --file_name [FILE_NAME]
# python export.py --config_path config/cifar10_Ascend_1p_config.yaml --data_path /data/cifar10/ --ckpt_file cifar10-1p.ckpt --file_name ResidualAttentionNet92-cifar10_1
```

## [Inference Process](#contents)

### [Inference](#content)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, we need to export model first. Air model can only be exported in Ascend 910 environment, mindir model can be exported in any environment.
Current batch_ Size can only be set to 1.

- inference on CIFAR-10 dataset when running on Ascend 310

  ```bash
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [CONFIG_PATH] [DEVICE_ID]
  # example: bash run_infer_310.sh cifar10-300.mindir cifar10 /data/cifar10/ ../config/cifar10_Ascend_1p_config.yaml 0
  ```

- Inference result will be stored in the example path, you can find result like the followings in acc.log.

  ```bash
  Total data:10000, top1 accuracy:0.9514, top5 accuracy:0.9978.
  ```

# [Model Description](#contents)

## [Performance](#contents)

### [Training Performance](#contents)

#### [ResidualAttentionNet train on CIFAR-10](#contents)

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | V1                                                           |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |
| uploaded Date              | 2022-05-29                                                   |
| MindSpore Version          | 1.5.1                                                        |
| Dataset                    | CIFAR-10                                                     |
| Training Parameters        | epoch=220, steps=97, batch_size = 64, lr=0.24(8p)            |
| Optimizer                  | Momentum                                                     |
| Loss Function              | Softmax Cross Entropy                                        |
| outputs                    | probobility                                                  |
| Loss                       | 0.0003                                                       |
| Speed                      | 8pcs: 72 ms/step                                             |
| Total time                 | 8pcs: 46 mins                                                |
| Parameters (M)             | 51.3                                                         |
| Checkpoint for Fine tuning | 153M (.ckptfile)                                             |
| Scripts                    | [residual_attention_net script](https://gitee.com/fuguidan/models/tree/master/research/cv/ResidualAttentionNet) |

#### [ResidualAttentionNet train on ImageNet2012](#contents)

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | V1                                                           |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |
| uploaded Date              | 2022-05-29                                                   |
| MindSpore Version          | 1.5.0                                                        |
| Dataset                    | ImageNet2012                                                 |
| Training Parameters        | epoch=60, steps=5004, batch_size=32, lr=0.24(8p)             |
| Optimizer                  | Momentum                                                     |
| Loss Function              | Softmax Cross Entropy                                        |
| outputs                    | probobility                                                  |
| Loss                       | 0.5                                                          |
| Speed                      | 8pcs: 109 ms/step                                            |
| Total time                 | 8pcs: 10.16 hours                                            |
| Parameters (M)             | 31.9                                                         |
| Checkpoint for Fine tuning | 657M (.ckptfile)                                             |
| Scripts                    | [residual_attention_net script](https://gitee.com/fuguidan/models/tree/master/research/cv/ResidualAttentionNet) |

### [Inference Performance](#contents)

#### [ResidualAttentionNet infer on CIFAR-10](#contents)

| Parameters        | Ascend                  |
| ----------------- | ----------------------- |
| Model Version     | V1                      |
| Resource          | Ascend 910; OS Euler2.8 |
| Uploaded Date     | 2022-05-29              |
| MindSpore Version | 1.5.1                   |
| Dataset           | CIFAR-10, 10,000 images |
| batch_size        | 64                      |
| outputs           | probability             |
| Accuracy          | 1pc: 95.4%; 8pcs：95.2% |

#### [ResidualAttentionNet infer on ImageNet2012](#contents)

| Parameters        | Ascend                  |
| ----------------- | ----------------------- |
| Model Version     | V1                      |
| Resource          | Ascend 910; OS Euler2.8 |
| Uploaded Date     | 2022-05-29              |
| MindSpore Version | 1.5.1                   |
| Dataset           | ImageNet2012            |
| batch_size        | 32                      |
| outputs           | probability             |
| Accuracy          | 8pcs: 77.5%             |

## [How to use](#contents)

### [Inference](#contents)

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/docs/zh-CN/r1.7/migration_guide/inference.html). Following the steps below, this is a simple example:

- Running on Ascend

  ```python
  # Set context
  context.set_context(mode=context.GRAPH_HOME, device_target=config.device_target)
  context.set_context(device_id=config.device_id)
  # Load unseen dataset for inference
  dataset = dataset.create_dataset(config.data_path)
  # Define model
  net = ResidualAttentionNet()
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01,
                 config.momentum, weight_decay=config.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean',
                                          is_grad=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})
  # Load pre-trained model
  param_dict = load_checkpoint(cfg.checkpoint_path)
  load_param_into_net(net, param_dict)
  net.set_train(False)
  # Make predictions on the unseen dataset
  acc = model.eval(dataset)
  print("accuracy:", acc)
  ```

### [Continue Training on the Pretrained Model](#contents)

- running on Ascend

  ```python
  # Load dataset
  dataset = create_dataset(config.data_path)
  batch_num = dataset.get_dataset_size()
  # Define model
  net = ResidualAttentionNet()
  # Continue training if set pre_trained to be True
  if config.pre_trained:
      param_dict = load_checkpoint(config.checkpoint_path)
      load_param_into_net(net, param_dict)
  lr = lr_steps(0, lr_max=config.lr_init, total_epochs=config.epoch_size,
                steps_per_epoch=batch_num)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 Tensor(lr), config.momentum, weight_decay=config.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean', is_grad=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=None)
  # Set callbacks
  config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5,
                               keep_checkpoint_max=cfg.keep_checkpoint_max)
  time_cb = TimeMonitor(data_size=batch_num)
  ckpoint_cb = ModelCheckpoint(prefix="train_cifar10", directory="./",
                               config=config_ck)
  loss_cb = LossMonitor()
  # Start training
  model.train(config.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
  print("train success")
  ```

# [Description of Random Situation](#contents)

# [Contribution to guide](#contents)

If you want to be a part of this effort,Please read[Ascend Contribution to guide](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING_CN.md)和[how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/models).  