# Contents

- [RetinaFace Description](#retinaface-description)
- [Model Architecture](#model-architecture)
- [Pretrain Model](#pretrain-model)
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
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
    - [How to use](#how-to-use)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [RetinaFace Description](#contents)

Retinaface is a face detection model, which was proposed in 2019 and achieved the best results on the wideface dataset at that time. Retinaface, the full name of the paper is retinaface: single stage dense face localization in the wild. Compared with s3fd and mtcnn, it has a significant improvement, and has a higher recall rate for small faces. It is not good for multi-scale face detection. In order to solve these problems, retinaface feature pyramid structure is used for feature fusion between different scales, and SSH module is added.

[Paper](https://arxiv.org/abs/1905.00641v2):  Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, Stefanos Zafeiriou. "RetinaFace: Single-stage Dense Face Localisation in the Wild". 2019.

# [Pretrain Model](#contents)

Retinaface needs a resnet50 backbone to extract image features for detection. You could get resnet50 train script from our modelzoo and modify the pad structure of resnet50 according to resnet in ./src/network.py, Final train it on imagenet2012 to get resnet50 pretrain model.
Steps:

1. Get resnet50 train script from our modelzoo.
2. Modify the resnet50 architecture according to resnet in ```./src/network.py```.(You can also leave the structure of a unchanged, but the accuracy will be 2-3 percentage points lower.)
3. Train resnet50 on imagenet2012.

# [Model Architecture](#contents)

Specifically, the retinaface network is based on retinanet. The feature pyramid structure of retinanet is used in the network, and SSH structure is added. Besides the traditional detection branch, the prediction branch of key points and self-monitoring branch are added in the network. The paper indicates that the two branches can improve the performance of the model. Here we do not implement the self-monitoring branch.

# [Dataset](#contents)

Dataset used: [WIDERFACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)

Dataset acquisition:

1. Get the dataset and annotations from [here](https://github.com/peteryuX/retinaface-tf2).
2. Get the eval ground truth label from [here](https://github.com/peteryuX/retinaface-tf2/tree/master/widerface_evaluate/ground_truth).

- Dataset size：3.42G，32,203 colorful images
    - Train：1.36G，12,800 images
    - Val：345.95M，3,226 images
    - Test：1.72G，16,177 images

# [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website and download the dataset, you can start training and evaluation as follows:

- running on GPU

  ```python
  # run training example
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &

  # run distributed training example
  bash scripts/run_distribute_gpu_train.sh DEVICE_NUM CUDA_VISIBLE_DEVICES
  example: bash scripts/run_distribute_gpu_train.sh 4 0,1,2,3

  # run exporting ONNX model example
  python export.py

  # run evaluation example
  export CUDA_VISIBLE_DEVICES=0
  python eval.py > eval.log 2>&1 &
  OR
  bash run_standalone_gpu_eval.sh CUDA_VISIBLE_DEVICES
  example: run_standalone_gpu_eval.sh 0

  # run evaluation of ONNX model example
  python eval_onnx.py
  OR
  bash scripts/run_onnx_eval.sh
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md                          // descriptions about all the models
    ├── RetinaFace_ResNet50
        ├── README.md                    // descriptions about googlenet
        ├── scripts
        │   ├──run_distribute_gpu_train.sh         // shell script for distributed on GPU
        │   ├──run_standalone_gpu_eval.sh         // shell script for evaluation on GPU
        │   ├──run_onnx_eval.sh                  // shell script for ONNX model evaluation on GPU or CPU
        ├── src
        │   ├── __init__.py            // init file
        │   ├──dataset.py             // creating dataset
        │   ├──network.py            // retinaface architecture
        │   ├──config.py            // parameter configuration
        │   ├──augmentation.py     // data augment method
        │   ├──loss.py            // loss function
        │   ├──utils.py          // data preprocessing
        │   ├──lr_schedule.py   // learning rate schedule
        ├── train.py               // training script
        ├── eval.py               //  evaluation script
        ├── export.py            // ONNX model exporting script
        ├── eval_onnx.py        // ONNX model evaluation script
        ├── requirements.txt   // requirements file
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for RetinaFace, WIDERFACE dataset

  ```python
    'variance': [0.1, 0.2],                                   # Variance
    'clip': False,                                            # Clip
    'loc_weight': 2.0,                                        # Bbox regression loss weight
    'class_weight': 1.0,                                      # Confidence/Class regression loss weight
    'landm_weight': 1.0,                                      # Landmark regression loss weight
    'batch_size': 8,                                          # Batch size of train
    'num_workers': 8,                                         # Num worker of dataset load data
    'num_anchor': 29126,                                      # Num of anchor boxes, it depends on the image size
    'ngpu': 4,                                                # Num gpu of train
    'epoch': 100,                                             # Training epoch number
    'decay1': 70,                                             # Epoch number of the first weight attenuation
    'decay2': 90,                                             # Epoch number of the second weight attenuation
    'image_size': 840,                                        # Training image size
    'match_thresh': 0.35,                                     # Threshold for match box
    'optim': 'sgd',                                           # Optimizer type
    'warmup_epoch': 5,                                        # Warmup size, 0 means no warm-up
    'initial_lr': 0.01,                                       # Learning rate
    'momentum': 0.9,                                          # Momentum for Optimizer
    'weight_decay': 5e-4,                                     # Weight decay for Optimizer
    'gamma': 0.1,                                             # Attenuation ratio of learning rate
    'ckpt_path': './checkpoint/',                             # Model save path
    'save_checkpoint_steps': 2000,                            # Save checkpoint steps
    'keep_checkpoint_max': 1,                                 # Number of reserved checkpoints
    'resume_net': None,                                       # Network for restart, default is None
    'training_dataset': '',                                   # Training dataset label path, like 'data/widerface/train/label.txt'
    'pretrain': True,                                         # Whether training based on the pre-trained backbone
    'pretrain_path': './data/res50_pretrain.ckpt',            # Pre-trained backbone checkpoint path
    'seed': 1,                                                # Setup train seed
    'lr_type': 'dynamic_lr',                                  # Learning rate decline function type, set dynamic_lr or standard_lr
  # val
    'val_model': './checkpoint/ckpt_0/RetinaFace-100_536.ckpt',   # Validation model path
    'val_dataset_folder': './data/widerface/val/',                # Validation dataset path
    'val_origin_size': False,                                     # Is full size verification used
    'val_confidence_threshold': 0.02,                             # Threshold for val confidence
    'val_nms_threshold': 0.4,                                     # Threshold for val NMS
    'val_iou_threshold': 0.5,                                     # Threshold for val IOU
    'val_save_result': False,                                     # Whether save the resultss
    'val_predict_save_folder': './widerface_result',              # Result save path
    'val_gt_dir': './data/ground_truth/',                         # Path of val set ground_truth
    # onnx
    'ckpt_model': '../ckpt/retinaface.ckpt',                  # path of ckpt file to be exported
    'onnx_model': '../ckpt/retinaface.onnx',                  # path of the ONNX model to be evaluated
    'device': 'CPU',                                          # device type: CPU or GPU
  ```

## [Training Process](#contents)

### Training

- running on GPU

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the folder `./checkpoint/` by default.

### Distributed Training

- running on GPU

  ```bash
  bash scripts/run_distribute_gpu_train.sh 4 0,1,2,3
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `train/train.log`.

  After training, you'll get some checkpoint files under the folder `./checkpoint/ckpt_0/` by default.

## ONNX EXPORTING

- **preparation**：Modify the parameter  `device` in the `src/config.py` file to select the type of device: `CPU` or `GPU`; then modify the parameter  `ckpt_model` to specify the path of ckpt file that is uesd to export to onnx model.

- **run script**：Run the following command to export the `ONNX` model and it will be saved in the current directory.

  ```SHELL
  python export.py
  ```

## [Evaluation Process](#contents)

### Evaluation

- evaluation on WIDERFACE dataset when running on GPU

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path in src/config.py, e.g., "username/retinaface/checkpoint/ckpt_0/RetinaFace-100_402.ckpt".

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python eval.py > eval.log 2>&1 &  
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The result of the test dataset will be as follows:

  ```text
  # grep "Val AP" eval.log
  Easy   Val AP : 0.9422
  Medium Val AP : 0.9325
  Hard   Val AP : 0.8900
  ```

  OR,

  ```bash
  bash run_standalone_gpu_eval.sh 0
  ```

  The above python command will run in the background. You can view the results through the file "eval/eval.log". The result of the test dataset will be as follows:

  ```text
  # grep "Val AP" eval.log
  Easy   Val AP : 0.9422
  Medium Val AP : 0.9325
  Hard   Val AP : 0.8900
  ```

## Evaluation for ONNX MODEL

- **preparation**：Modify the following parameters in the `src/config.py` file according to the actual situation:

    - `device`：type of device `CPU` 或 `GPU`；
    - `onnx_model`：path of onnx model；
    - `val_dataset_folder`：path of validation dataset;
    - `val_gt_dir`：path of validation dataset ground_truth label.

- **run script**：Run the following command to evaluate the `ONNX` model:

  ```bash
  export CUDA_VISIBLE_DEVICES="$1"
  python eval_onnx.py
  or
  bash scripts/run_onnx_eval.sh CUDA_VISIBLE_DEVICES
  example: bash scripts/run_onnx_eval.sh 0
  ```

The evaluation results can be viewed in the window：

  ```python
  Easy   Val AP : 0.9390
  Medium Val AP : 0.9306
  Hard   Val AP : 0.8886
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | GPU                                                           |
| -------------------------- | --------------------------------------------------------------|
| Model Version              | RetinaFace + Resnet50                                         |
| Resource                   | NV SMX2 V100-16G                                              |
| uploaded Date              | 01/04/2022 (month/day/year)                                   |
| MindSpore Version          | 1.6.0                                                         |
| Dataset                    | WIDERFACE                                                     |
| Training Parameters        | epoch=100, steps=402, batch_size=8, lr=0.01                   |
| Optimizer                  | SGD                                                           |
| Loss Function              | MultiBoxLoss + Softmax Cross Entropy                          |
| outputs                    | bounding box + confidence + landmark                          |
| Loss                       | 1.200                                                         |
| Speed                      | 4pcs: 560 ms/step                                             |
| Total time                 | 4pcs: 6.4 hours                                               |
| Parameters (M)             | 27.29M                                                        |
| Checkpoint for Fine tuning | 336.3M (.ckpt file)                                           |
| Scripts                    | [retinaface script](https://gitee.com/mindspore/models/tree/master/official/cv/RetinaFace_ResNet50) |

### Inference Performance

| Parameters          | GPU                         |
| ------------------- | --------------------------- |
| Model Version       | RetinaFace + Resnet50       |
| Resource            | NV SMX2 V100-16G            |
| Uploaded Date       | 01/04/2022 (month/day/year) |
| MindSpore Version   | 1.6.0                       |
| Dataset             | WIDERFACE                   |
| batch_size          | 1                           |
| outputs             | bounding box + confidence + landmark |
| Accuracy            | 4pcs:                       |
|                     | Easy   Val AP : 0.9422      |
|                     | Medium Val AP : 0.9325      |
|                     | Hard   Val AP : 0.8900      |

## [How to use](#contents)

### Continue Training on the Pretrained Model

- running on GPU

  ```python
  # Load dataset
  ds_train = create_dataset(training_dataset, cfg, batch_size, multiprocessing=True, num_worker=cfg['num_workers'])

  # Define model
  multibox_loss = MultiBoxLoss(num_classes, cfg['num_anchor'], negative_ratio, cfg['batch_size'])
  lr = adjust_learning_rate(initial_lr, gamma, stepvalues, steps_per_epoch, max_epoch, warmup_epoch=cfg['warmup_epoch'])
  opt = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=momentum,
              weight_decay=weight_decay, loss_scale=1)
  backbone = resnet50(1001)
  net = RetinaFace(phase='train', backbone=backbone)

  # Continue training if resume_net is not None
  pretrain_model_path = cfg['resume_net']
  param_dict_retinaface = load_checkpoint(pretrain_model_path)
  load_param_into_net(net, param_dict_retinaface)

  net = RetinaFaceWithLossCell(net, multibox_loss, cfg)
  net = TrainingWrapper(net, opt)

  model = Model(net)

  # Set callbacks
  config_ck = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_steps'],
                                 keep_checkpoint_max=cfg['keep_checkpoint_max'])
  ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)
  time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
  callback_list = [LossMonitor(), time_cb, ckpoint_cb]

  # Start training
  model.train(max_epoch, ds_train, callbacks=callback_list,
                dataset_sink_mode=False)
  ```

# [Description of Random Situation](#contents)

In train.py, we set the seed with setup_seed function.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).  
