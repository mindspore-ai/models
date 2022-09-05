# Contents

- [Contents](#contents)
- [Flownet2 Description](#flownet2-description)
- [Model Architecture](#model-architecture)
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
        - [Training Performance](#training-performance)
            - [FlowNet2 train on FlyingChairs](#flownet2-train-on-flyingchairs)
        - [Inference Performance](#inference-performance)
            - [FLowNet2 infer on MpiSintelClean](#flownet2-infer-on-mpisintelclean)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Flownet2 Description](#contents)

FlowNet2.0, a deep network proposed in 2017, which performs end-to-end learning on optical flow data.
It is optimized based on the FlowNet network , The large improvements in quality and
speed are caused by three major contributions: first, it
focus on the training data and show that the schedule of
presenting data during training is very important. Second,
it develop a stacked architecture that includes warping
of the second image with intermediate optical flow. Third,
it elaborate on small displacements by introducing a subnetwork specializing on small motions.

Compared with the FLownet network,  FlowNet 2.0 is only
marginally slower than the original FlowNet but decreases
the estimation error by more than 50%.

[FlowNet2 paper](https://arxiv.org/abs/1612.01925 )：Eddy Ilg, Nikolaus Mayer, Tonmoy Saikia, Margret Keuper, Alexey Dosovitskiy, Thomas Brox

[FlowNet paper](https://arxiv.org/abs/1504.06852 )：Philipp Fischer, Alexey Dosovitskiy, Eddy Ilg, Philip Häusser, Caner Hazırbaş, Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, Thomas Brox

# [Model Architecture](#contents)

The FlowNet2 network is stacked by multiple flownet sub-modules. After the output of the previous network is processed by warp, it is used as the input of the second network.

The model structure is flowNet2CSS and FlowNet2SD two sub-networks fuse the output through the FlownetFusion network, and the entire large network structure formed is FLowNet2
The FlowNet2CSS network is a stack of FLowNet2C and two FLowNet2S. The specific structure can be further understood according to the paper

This source code provides the following model structure, which can be configured and used in the yaml file:

- FlowNet2S
- FlowNet2C
- FlowNet2CS
- FlowNet2CSS
- FlowNet2SD
- FlowNet2

# [Dataset](#contents)

Dataset used: [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)

- Dataset size：31GB，22,872 pairs 512*384 colorful images
- Data format：PPM
    - Note：Data will be processed in src/dataset.py
- you can download here [dataset package](https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip)

Dataset used: [ChairsSDHom](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)

- Dataset size: 51GB, 21,668 pairs 512*384 colorful images
    - Train：20,965 pairs image
    - Test： 703 pairs image
- Data format：PNG
    - Note: Data will be processed in src/dataset.py
- you can download here [dataset package](https://lmb.informatik.uni-freiburg.de/data/FlowNet2/ChairsSDHom/ChairsSDHom.tar.gz)

Dataset used: [MpiSintel](http://sintel.cs.washington.edu)

- Dataset size: 536M, 1024 x 436 colorful images in 23 classes
    - MpiSintelClean：1150 images  
    - MpiSintelFinal： 1150 images
- Data format：PNG
    - Note: Data will be processed in src/dataset.py
- you can download here [dataset package](http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip)

# [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- download pretrained parameter

    FlowNet2 [620MB](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)

    FlowNet2-C [149MB](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view?usp=sharing)

    FlowNet2-CS [297MB](https://drive.google.com/file/d/1iBJ1_o7PloaINpa8m7u_7TsLCX0Dt_jS/view?usp=sharing)

    FlowNet2-CSS [445MB](https://drive.google.com/file/d/157zuzVf4YMN6ABAQgZc8rRmR5cgWzSu8/view?usp=sharing)

    FlowNet2-CSS-ft-sd [445MB](https://drive.google.com/file/d/1R5xafCIzJCXc8ia4TGfC65irmTNiMg6u/view?usp=sharing)

    FlowNet2-S [148MB](https://drive.google.com/file/d/1V61dZjFomwlynwlYklJHC-TLfdFom3Lg/view?usp=sharing)

    FlowNet2-SD [173MB](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)

- convert pretrained parameter (from pytorch pretrained parameter to mindspore pretained parameter,so the env should both installed torch and mindspore)
    convert pytorch pretrained parameter to mindspore pretrained parameter
    the pytorch pretrained parameter are supposed to be downloaded by above link

    ```text
    bash scripts/run_ckpt_convert.sh [PYTORCH_FILE_PATH] [MINDSPORE_FILE_PATH]
    # example:
    bash scripts/run_ckpt_convert.sh /path/to/FlowNet2_checkpoint.pth.tar /path/to/flownet2.ckpt
    ```

- compile custom operation Correlation and Resample2d
  after execution,you can check the whether generate correlation.so and resample2d.so under path src/submodels/custom_ops/

    ```text
    bash scripts/run_compile_custom_ops.sh
    ```

- config pretrained parameter path in yaml file

  ```text
  pre_trained:     # whether use pretrained parameter file 1 or 0
  pre_trained_ckpt_path: # pretrained checkpoint file path
  # 实例：
  pre_trained: 1
  pre_trained_ckpt_path: /path/checkpoint/flownet2.ckpt
  ```

- config dataset name and path in yaml file

    ```text
    train_data: [DATASET_NAME]  # Name of dataset, 'FlyingChairs' or 'MpiSintelFinal' or 'MpiSintelClean'
    train_data_path：[DATASET_PATH] # path of dataset
    # example：
    train_data: FlyingChairs
    train_data_path: /path/to/FlyingChairs_release/data
    ```

- running on GPU

  ```python
  # run training example
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &

  # run distributed training example
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7

  # run evaluation example
  python eval.py --eval_checkpoint_path=[EVAL_CHECKPOINT_PATH] > eval.log 2>&1 &  
  OR
  bash scripts/run_eval_gpu.sh [MpiSintelClean/MpiSintelFinal] [DATA_PATH] [MODEL_NAME] [CKPT_PATH] [DEVICE_ID]
  ```

We use FlyingChairs dataset by default. Your can also pass `$dataset_type` to the scripts so that select different datasets. For more details, please refer the specify script.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md                          // descriptions about all the models
    ├── flownet2
        ├── README.md                      // descriptions about flownet2
        ├── scripts
        │   ├── run_ckpt_convert.sh        // shell script for converting pytorch ckpt file to pickle file on GPU
        │   ├── run_compile_custom_ops.sh  // shell script for compile ops
        │   ├── run_eval_gpu.sh            // shell script for eval on GPU
        │   └── run_train_gpu.sh           // shell script for training on GPU
        ├── src
        │   ├── dataset.py                 // creating dataset
        │   ├── eval_callback.py           // eval callback when training
        │   ├── metric.py                  // metric to calculate mean error
        │   ├── model_utils
        │   │   ├── ckpt_convert.py         // convert pytorch ckpt file to pickle file
        │   │   ├── config.py               // parameter configuration
        │   │   ├── device_adapter.py       // device adapter
        │   │   ├── local_adapter.py        // local adapter
        │   │   ├── moxing_adapter.py       // moxing adapter
        │   │   ├── frame_utils.py          // utils to read files of dataset
        │   │   └── tools.py                // tools to match class with paratmeter from config
        │   ├── models.py                   // FlowNet2/FlowNet2CSS/FlowNet2CS/FlowNet2C/FlowNet2S/FlowNet2SD model
        │   └── submodels
        │       ├── custom_ops
        │       │    ├── correlation.cu        // cuda file for operation correlation
        │       │    ├── resample2d.cu         // cuda file for operation resample2d
        │       │    └── custom_ops.py         // definition of correlation and resample2d
        │       ├── FlowNetC.py             // FlowNetC model
        │       ├── FlowNetFusion.py        // FlowNetFusion model
        │       ├── FlowNetS.py             // FlowNetS model
        │       ├── FlowNetSD.py            // FlowNetSD model
        │       └── submodules.py           // submodules used in flownet model
        ├── default_config.yaml             // parameter configuration
        ├── requirements.txt                // requirements configuration
        ├── eval.py                         // evaluation script
        └── train.py                        // training script
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for FLowNet2

  ```text
    # ==============================================================================
    # Device
    device_target:          "GPU"
    device_id:              0

    # Dataset Setup
    crop_type:              Random                   # Type of cropping operation (Random and Center)
    crop_size:              [384, 512]               # (Height, Width) of image when training
    eval_size:              [256, 256]               # (Height, Width) of image when eval

    # Experiment Setup
    model:                  "FlowNet2"                # Name of model to be loaded
    rgb_max:                255                       # rgb channel used
    batchNorm:              False                     # boolean switch to whether add batchnorm before conv
    lr:                     1e-6                      # Learning rate
    num_parallel_workers:   2                         # Number of CPU worker used to load data
    max_rowsize:            2                         # Number of max rowsize used to load data
    batch_size:             2                         # Numbers of image pairs in a mini-batch
    epoch_size:             20                        # Total number of epochs
    pre_trained:            1                         # Load pretrained network
    pre_trained_ckpt_path:  "/path/flownet2.ckpt"     # Pretrained ckpt path
    seed:                   1                         # Seed for reproducibility
    is_dynamicLoss_scale:   0                         # Using dynamicLoss scale or fix scale
    scale:                  1024                      # Fix scale value
    weight_decay:           0.00001                   # Weight decay
    train_data:             "FlyingChairs"            # Train Dataset name
    train_data_path:        "/path/ds/FlyingChairs_release/data"       # Train Dataset path

    # Train Setup
    run_distribute:         1                         # Distributed training or not
    is_save_on_master:      1                         # Only save ckpt on master device
    save_checkpoint:        1                         # Is save ckpt while training
    save_ckpt_interval:     1                         # Saving ckpt interval
    keep_checkpoint_max:    5                         # Max ckpt file number
    save_checkpoint_path:   "/path/ckpt/"             # Ckpt save path

    # eval Setup
    eval_data:              "MpiSintelClean"           # Eval Dataset name
    eval_data_path:         "/home/shm/ds/training"    # Eval Dataset path
    eval_checkpoint_path:   "/path/flownet2.ckpt"      # Ckpt path used to eval
    run_evalCallback:       1                          # Is run evalCallBack while training
    eval_start_epoch:       1                          # EvalCallback start epoch
    eval_interval:          5                          # EvalCallback running interval
    save_best_ckpt:         1                          # Is save best ckpt

    # Export Setup
    mindir_file_name:        "Flownet2"                 # Save file path
    file_format:             "MINDIR"                   # Save file format

    # Modelarts Setup
    enable_modelarts:       0                           # Is training on modelarts
  ```

For more configuration details, please refer the script `config.py`.

## [Training Process](#contents)

### Training

- running on GPU

  ```python
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &
  ```

  ```bash
  bash scripts/run_train_gpu.sh 1 0
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the folder `${save_checkpoint_path}/ckpt_0/` by default.

- train.log for flyingchairs

```text
epoch: 1 step: 2859, loss is 1.0592992305755615
epoch time: 2454542.145 ms, per step time: 858.532 ms
epoch: 2 step: 2859, loss is 1.074428915977478
epoch time: 2416319.469 ms, per step time: 845.162 ms
epoch: 3 step: 2859, loss is 0.6141664981842041
epoch time: 2412936.084 ms, per step time: 843.979 ms
```

- train.log for MpiSintel

```text
epoch: 1 step: 131, loss is 0.3894098699092865
epoch time: 114087.253 ms, per step time: 870.895 ms
epoch: 2 step: 131, loss is 1.822862982749939
epoch time: 93423.045 ms, per step time: 713.153 ms
epoch: 3 step: 131, loss is 0.06125941127538681
epoch time: 93837.971 ms, per step time: 716.320 ms
```

### Distributed Training

- running on GPU

  ```bash
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `train.log`.

- train.log for flyingchairs

```text
epoch: 1 step: 358, loss is 1.1717915534973145
epoch: 1 step: 358, loss is 0.6347103118896484
epoch: 1 step: 358, loss is 1.4680955410003662
epoch: 1 step: 358, loss is 1.7656424045562744
epoch: 1 step: 358, loss is 1.1760812997817993
epoch: 1 step: 358, loss is 0.8203185200691223
epoch: 1 step: 358, loss is 2.2942874431610107
epoch: 1 step: 358, loss is 1.3205347061157227
epoch time: 858929.203 ms, per step time: 2399.244 ms
epoch time: 859414.930 ms, per step time: 2400.600 ms
epoch time: 859515.190 ms, per step time: 2400.880 ms
epoch time: 859614.460 ms, per step time: 2401.158 ms
epoch time: 859695.493 ms, per step time: 2401.384 ms
epoch time: 859799.146 ms, per step time: 2401.674 ms
epoch time: 859995.238 ms, per step time: 2402.221 ms
epoch time: 860035.718 ms, per step time: 2402.334 ms
```

## [Evaluation Process](#contents)

### Evaluation

- evaluation on MpiSintelClean dataset when running on GPU

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "path/flownet2/ckpt/flownet2-125_390.ckpt".

  ```python
  python eval.py --eval_data=[DATASET_NAME] --eval_data_path=[DATASET_PATH]/
  --model=[MODEL_NAME] --eval_checkpoint_path=[CHECKPOINT_PATH] > eval.log 2>&1 &  
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "mean error: " eval.log
  flownet2 mean error:  {'flownetEPE': 2.112366}
  ```

  OR,

  ```bash
  bash scripts/run_eval_gpu.sh [MpiSintelClean/MpiSintelFinal] [DATA_PATH] [MODEL_NAME] [CKPT_PATH] [DEVICE_ID]
  ```

  The above python command will run in the background. You can view the results through the file "eval/eval.log". The accuracy of the test dataset will be as follows:

  ```text
  # grep "mean error: " eval.log
  flownet2 mean error:  {'flownetEPE': 2.112366}
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

#### FlowNet2 train on FlyingChairs

| Parameters                 | GPU                                                                                               |
|----------------------------|---------------------------------------------------------------------------------------------------|
| Model Version              | Inception V1                                                                                      |
| Resource                   | NV SMX2 V100-32G                                                                                  |
| uploaded Date              | 04/05/2021 (month/day/year)                                                                       |
| MindSpore Version          | 1.7.0                                                                                             |
| Dataset                    | FlyingChairs                                                                                      |
| Training Parameters        | epoch=50, steps=2800, batch_size=8, lr=1e-6                                                       |
| Optimizer                  | Adam                                                                                              |
| Loss Function              | L1loss                                                                                            |
| outputs                    | flow                                                                                              |                                                                                             |
| Speed                      | 1pc: 152 ms/step;  8pcs: 171 ms/step                                                              |
| Total time                 | 8pcs: 8.8 hours                                                                                   |
| Parameters                 | 162,518,834                                                                                       |
| Checkpoint for Fine tuning | 260M (.ckpt file)                                                                                 |
| Scripts                    | [flownet2 script](https://gitee.com/mindspore/models/tree/master/research/cv/flownet2) |

### Inference Performance

#### FlowNet2 infer on MpiSintelClean

| Parameters        | GPU                         |
|-------------------|-----------------------------|
| Model Version     | Inception V1                |
| Resource          | NV SMX2 V100-32G            |
| Uploaded Date     | 04/05/2022 (month/day/year) |
| MindSpore Version | 1.7.0                       |
| Dataset           | MpiSintelClean              |
| batch_size        | 8                           |
| outputs           | flow                        |
| Mean Error        | 2.10                        |

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/models).  
