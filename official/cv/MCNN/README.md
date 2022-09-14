# Contents

- [MCNN Description](#mcnn-description)
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
    - [ONNX Export And Evaluation](#onnx-export-and-evaluation)
        - [ONNX Export](#onnx-export)
        - [ONNX Evaluation](#onnx-evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [MCNN Description](#contents)

MCNN was a Multi-column Convolution Neural Network which can estimate crowd number accurately in a single image from almost any perspective.

[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf): Yingying Zhang, Desen Zhou, Siqin Chen, Shenghua Gao, Yi Ma. Single-Image Crowd Counting via Multi-Column Convolutional Neural Network.

# [Model Architecture](#contents)

MCNN contains three parallel CNNs whose filters are with local receptive fields of different sizes. For simplification, we use the same network structures for all columns (i.e.,conv–pooling–conv–pooling) except for the sizes and numbers of filters. Max pooling is applied for each 2×2 region, and Rectified linear unit (ReLU) is adopted as the activation function because of its good performance for CNNs.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [ShanghaitechA](<https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0>)

```text
├─data
    ├─formatted_trainval
        ├─shanghaitech_part_A_patches_9
            ├─train
            ├─train-den
            ├─val
            ├─val-den
    ├─original
        ├─shanghaitech
            ├─part_A_final
                ├─train_data
                    ├─images
                    ├─ground_truth
                ├─test_data
                    ├─images
                    ├─ground_truth
                    ├─ground_truth_csv
```

# [Environment Requirements](#contents)

- Hardware (Ascend/GPU)
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

```bash
# enter script dir, train MCNN
bash run_standalone_train_ascend.sh [DATA_PATH] [CKPT_SAVE_PATH]
# enter script dir, evaluate MCNN
bash run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_NAME]
```

- running on GPU

For running on GPU, please change device_target from Ascend to GPU in train.py

```bash
# enter script dir, train MCNN
bash run_train_gpu.sh
# enter script dir, evaluate MCNN
bash run_eval.sh [DATA_PATH] [CKPT_NAME]
```

# [Ascend310 Inference](#contents)

- Generate .mindir file

```bash
python export.py
```

- Execute 310 inference script

```bash

bash run_infer_310.sh [model] [data_path] [label_path] [device_id]

Example:

bash run_infer_310.sh ../mcnn.mindir ../test_data/images ../test_data/ground_truth_csv 0
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── cv
    ├── MCNN
        ├── README.md                    // descriptions about MCNN
        ├── ascend310_infer              // 310 reasoning source code
        ├── infer
        ├── modelarts
        ├── scripts
        │   ├──run_distribute_train.sh             // train in distribute
        │   ├──run_eval.sh                         // eval in ascend
        │   ├──run_eval_onnx_gpu.sh                // exported onnx eval in gpu
        │   ├──run_infer_310.sh                    // infer in 310
        │   ├──run_standalone_train.sh             // train in standalone
        │   ├──run_train_gpu.sh                    // train on GPU
        ├── src
        │   ├──dataset.py             // creating dataset
        │   ├──mcnn.py               // mcnn architecture
        │   ├──config.py            // parameter configuration
        │   ├──data_loader.py            // prepare dataset loader(GREY)
        │   ├──data_loader_3channel.py            // prepare dataset loader(RGB)
        │   ├──evaluate_model.py            // evaluate model
        │   ├──generator_lr.py            // generator learning rate
        │   ├──Mcnn_Callback.py            // Mcnn Callback
        ├── train.py                // training script
        ├── eval.py                 //  evaluation script
        ├── eval_onnx.py            //  exported onnx evaluation script
        ├── export.py               //  export script
        ├── export_onnx.py          //  export onnx format script
```

## [Script Parameters](#contents)

``` python
Major parameters in train.py and config.py as follows:

--data_path: The absolute full path to the train and evaluation datasets.
--epoch_size: Total training epochs.
--batch_size: Training batch size.
--device_target: Device where the code will be implemented. Optional values are "Ascend", "GPU".
--ckpt_path: The absolute full path to the checkpoint file saved after training.
--onnx_path: The absolute full path to the exported onnx file.
--train_path: Training dataset's data
--train_gt_path: Training dataset's label
--val_path: Testing dataset's data
--val_gt_path: Testing dataset's label
```

## [Training Process](#contents)

### Training

- running on Ascend

  ```bash
  # python train.py
  # or enter script dir, and run the distribute script
  bash run_distribute_train.sh 8 /home/wks/hccl_8p_01234567_127.0.0.1.json
  # or enter script dir, and run the standalone script
  bash run_standalone_train.sh 1
  ```

  After training, the loss value will be achieved as follows:

  ```text
  # grep "loss is " log
  epoch: 1 step: 305, loss is 0.00041025918
  epoch: 2 step: 305, loss is 3.7117527e-05
  ...
  epoch: 798 step: 305, loss is 0.000332611
  epoch: 799 step: 305, loss is 2.6959011e-05
  epoch: 800 step: 305, loss is 5.6599742e-06
  ...
  ```

  The model checkpoint will be saved in the current directory.

- running on GPU

  ```bash
  # python train.py
  # or enter script dir, and run the distribute script
  bash run_train_gpu.sh 8 0,1,2,3,4,5,6,7
  # or enter script dir, and run the standalone script
  bash run_train_gpu.sh
  ```

  After training, the loss value will be achieved as follows:

  ```text
  # grep "loss is " log
  epoch: 1 step: 305, loss is 0.00041025918
  epoch: 2 step: 305, loss is 3.7117527e-05
  ...
  epoch: 798 step: 305, loss is 0.000332611
  epoch: 799 step: 305, loss is 2.6959011e-05
  epoch: 800 step: 305, loss is 5.6599742e-06
  ...
  ```

  The model checkpoint will be saved in the current directory.

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

  ```bash
  # python eval.py
  # or enter script dir, and run the script
  bash run_eval.sh
  ```

  You can view the results through the file "eval_log". The accuracy of the test dataset will be as follows:

  ```text
  # grep "MAE: " eval_log
  MAE: 105.87984801910736 MSE: 161.6687899899305
  ```

- running on GPU

  ```bash
  # python eval.py
  # or enter script dir, and run the script
  bash run_eval.sh
  ```

  You can view the results through the file "eval_log". The accuracy of the test dataset will be as follows:

  ```text
  # grep "MAE: " eval_log
  MAE: 105.87984801910736 MSE: 161.6687899899305
  ```

## [ONNX Export And Evaluation](#contents)

Note that run all onnx concerned scripts on GPU.

### ONNX Export

The command below will produce lots of MCNN onnx files, named different input shapes due to different input shapes of evaluation data.

```bash
python export_onnx.py --ckpt_file [CKPT_PATH] --val_path [VAL_PATH] --val_gt_path [VAL_GT_PATH]
# example: python export_onnx.py --ckpt_file mcnn_ascend_v170_shanghaitecha_official_cv_MAE112.11.ckpt  --val_path /data0/dc/mcnn/models/official/cv/MCNN/data/original/shanghaitech/part_A_final/test_data/images/  --val_gt_path /data0/dc/mcnn/models/official/cv/MCNN/data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/
```

### ONNX Evaluation

Note that ONNX_PATH should be the absolute directory to the exported onnx files, such as: '/data0/dc/mcnn/models/official/cv/MCNN/'.

```bash
bash run_eval_onnx_gpu.sh [VAL_PATH] [VAL_GT_PATH] [ONNX_PATH]
# example: bash run_eval_onnx_gpu.sh  /data0/dc/mcnn/models/official/cv/MCNN/data/original/shanghaitech/part_A_final/test_data/images/ /data0/dc/mcnn/models/official/cv/MCNN/data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/ /data0/dc/mcnn/models/official/cv/MCNN/
 ```

You can view the results through the file "log_onnx". The accuracy of the test dataset will be as follows:

```text
MAE: 112.11429375868578   MSE: 172.62108098880813
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                      |GPU                                                          |
| -------------------------- | ------------------------------------------------------------| ------------------------------------------------------------|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             | TITAN Xp; NVIDIA-SMI 455.28                                 |
| uploaded Date              | 03/29/2021 (month/day/year)                                 | 10/28/2021 (month/day/year)                                 |
| MindSpore Version          | 1.1.0                                                       | 1.3.0                                                       |
| Dataset                    | ShanghaitechA                                               | ShanghaitechA                                               |
| Training Parameters        | steps=2439, batch_size = 1                                  | steps=2439, batch_size = 1                                  |
| Optimizer                  | Momentum                                                    | Momentum                                                    |
| outputs                    | probability                                                 | probability                                                 |
| Speed                      | 5.79 ms/step                                                | 23.54 ms/step                                               |
| Total time                 | 23 mins                                                     | 94 mins                                                     |
| Checkpoint for Fine tuning | 500.94KB (.ckpt file)                                       | 500.94KB (.ckpt file)                                       |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models/).
