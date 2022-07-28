# BRDNet

<!-- TOC -->

- [BRDNet](#brdnet)
- [Introduction to BRDNet](#introduction-to-brdnet)
- [Model structure](#model-structure)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Quick start](#quick-start)
- [Script description](#script-description)
    - [Directory structure](#directory-structure)
    - [Script parameters](#script-parameters)
    - [Training process](#training-process)
        - [train](#train)
    - [Evaluation process](#evaluation-process)
        - [eval](#eval)
    - [310 Inference](#310-inference)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Assess performance](#assess-performance)
- [Random description](#random-description)
- [ModelZoo homepage](#modelzoo-homepage)

<!-- /TOC -->

## Introduction to BRDNet

BRDNet was published in the artificial intelligence journal Neural Networks in 2020, and was rated as one of the most downloaded papers in 2019/2020 by the journal. This paper proposes a scheme to deal with the unevenness of data under the condition of limited hardware resources. At the same time, for the first time, the idea of using a two-way network to extract complementary information for image denoising is proposed, which can effectively remove synthetic noise and real noise.

[Paper](https://www.sciencedirect.com/science/article/pii/S0893608019302394)：Ct A ,  Yong X ,  Wz C . Image denoising using deep CNN with batch renormalization[J]. Neural Networks, 2020, 121:461-473.

## Model structure

BRDNet contains two branches, the upper and lower branches. The upper branch only includes residual learning and BRN; the lower branch includes BRN, residual learning, and expanded convolution.

The upper branch network contains two different types of layers: Conv+BRN+ReLU and Conv. Its depth is 17, the first 16 layers are Conv+BRN+ReLU, and the last layer is Conv. The number of feature map channels is 64, and the convolution kernel size is 3.
The lower branch network also contains 17 layers, but its 1, 9, 16 layers are Conv+BRN+ReLU, 2-8, 10-15 are dilated convolutions, and the last layer is Conv. The size of the convolution kernel is 3 and the number of channels is 64.

The two branch results are combined by `Concat` and the noise is obtained by Conv, and finally the original input is used to subtract the noise to obtain a clear and noise-free image. The entire BRDNet contains a total of 18 layers, which are relatively shallow and will not cause gradient disappearance and gradient explosion problems.

## Dataset

Training dataset：[color noisy](<https://pan.baidu.com/s/1cx3ymsWLIT-YIiJRBza24Q>)

When removing Gaussian noise, the dataset uses 3,859 images from the Waterloo Exploration Database and preprocesses them into 50x50 small images, totaling 1,166,393 images.

Test dataset: CBSD68, Kodak24, and McMaster

## Environmental requirements

- Hardware（Ascend/GPU/ModelArts）
    - Prepare the Ascend, GPU or ModelArts processor to build the hardware environment
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For details, please refer to the following resources:
    - [MindSpore](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html)

## Quick start

After installing MindSpore through the official website, you can follow the steps below to train and evaluate:

```shell
# Start single-card Ascend training through the bash command
cd scripts
bash run_standalone_train_ascend.sh [config_file] [dataset_path]

# Or single-card GPU training
cd scripts
bash run_standalone_train_gpu.sh [config_file] [dataset_path]

# Distributed Ascend training
cd scripts
bash run_distribute_train_ascend.sh [train_code_path] [train_data] [batch_size] [sigma] [channel] [epoch] [lr] [rank_table_file_path]

# Distributed GPU training
cd scripts
bash run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [CONFIG_FILE] [DATASET_PATH]

# Start Ascend eval with through the bash command. (There is no requirement for the path format of parameters such as test_dir, it will be automatically converted to an absolute path and ending with "/")
cd scripts
bash run_eval_ascend.sh [config_file] [test_dataset_path] [pretrain_path] [ckpt_name]

# Start GPU eval
cd scripts
bash run_eval_gpu.sh [config_file] [test_dataset_path] [pretrain_path] [ckpt_name]
```

Ascend training: generation of [RANK_TABLE_FILE](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

## Script description

### Directory structure

```text
├── model_zoo
    ├── README.md                              // Documentation for all models
    ├── brdnet
        ├── README.md                          // brdnet Documentation
        ├── ascend310_infer                    // 310 Inference code directory（C++）
        │   ├──inc
        │   │   ├──utils.h                     // Toolkit header file
        │   ├──src
        │   │   ├──main.cc                     // 310 Inference code
        │   │   ├──utils.cc                    // Toolkit
        │   ├──build.sh                        // Code compilation script
        │   ├──CMakeLists.txt                  // Code compilation settings
        ├── scripts
        │   ├──run_distribute_train_ascend.sh  // Ascend 8-card training script
        │   ├──run_distribute_train_gpu.sh     // GPU distributed training script
        │   ├──run_eval_ascend.sh              // Ascend model evaluation script
        │   ├──run_eval_gpu.sh                 // GPU model evaluation script
        │   ├──run_standalone_train_ascend.sh  // Ascend training startup script
        │   ├──run_standalone_train_gpu.sh     // GPU training startup script
        │   ├──run_infer_310.sh                // Start 310 inference script
        │   ├──docker_start.sh                 // Docker startup script when using MindX inference
        ├── src
        │   ├──config.py                       // Configuration processing
        │   ├──dataset.py                      // Dataset processing
        │   ├──logger.py                       // Log print file
        │   ├──models.py                       // Model structure
        ├── export.py                          // Export weight files to scripts in MINDIR and other formats
        ├── train.py                           // Training script
        ├── eval.py                            // Inference script
        ├── cal_psnr.py                        // 310 Script to calculate the final PSNR value during inference
        ├── preprocess.py                      // 310 Script to add noise to the test picture during inference
        ├─�requirements.txt                   // requirements for pip
        ├── default_config.yaml                // Default configuration file
```

### Script parameters

Parameters in ```default_config.yaml```:

```text
batch_size: 32
train_data: '../dataset/waterloo5050step40colorimage/'
sigma: 15
channel: 3
epoch: 50
lr: 0.001
save_every: 1

resume_path: ''
resume_name: ''

use_modelarts: False
train_url: 'train_url/'
data_url: 'data_url/'
output_path: './output/'
outer_path: 's3://output/'

test_dir: ''
pretrain_path: ''
ckpt_name: ''

device_target: 'GPU'
is_distributed: False
rank: 0
group_size: 1
is_save_on_master: True
ckpt_save_max: 5
```

### Training process

#### train

- Ascend training

  ```shell
  # Start single-card Ascend training through the bash command
  cd scripts
  bash run_standalone_train_ascend.sh [config_file] [dataset_path]

  # Distributed Ascend training
  cd scripts
  bash run_distribute_train_ascend.sh [train_code_path] [train_data] [batch_size] [sigma] [channel] [epoch] [lr] [rank_table_file_path]
  ```

- GPU training
    - Single GPU:

      ```shell
      cd scripts
      bash run_standalone_train_gpu.sh [config_file] [dataset_path]
      ```

    - Multiple GPU:

      ```shell
      cd scripts
      bash run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [CONFIG_FILE] [DATASET_PATH]
      ```

**Note**: The first time it runs, it may stay in the following interface for a long time. This is because the log will be printed after an epoch is completed. Please wait patiently.

The first epoch is estimated to take 20 to 30 minutes when running on a single card.

  ```log
  2021-05-16 20:12:17,888:INFO:Args:
  2021-05-16 20:12:17,888:INFO:--> batch_size: 32
  2021-05-16 20:12:17,888:INFO:--> train_data: ../dataset/waterloo5050step40colorimage/
  2021-05-16 20:12:17,889:INFO:--> sigma: 15
  2021-05-16 20:12:17,889:INFO:--> channel: 3
  2021-05-16 20:12:17,889:INFO:--> epoch: 50
  2021-05-16 20:12:17,889:INFO:--> lr: 0.001
  2021-05-16 20:12:17,889:INFO:--> save_every: 1
  2021-05-16 20:12:17,889:INFO:--> pretrain: None
  2021-05-16 20:12:17,889:INFO:--> use_modelarts: False
  2021-05-16 20:12:17,889:INFO:--> train_url: train_url/
  2021-05-16 20:12:17,889:INFO:--> data_url: data_url/
  2021-05-16 20:12:17,889:INFO:--> output_path: ./output/
  2021-05-16 20:12:17,889:INFO:--> outer_path: s3://output/
  2021-05-16 20:12:17,889:INFO:--> device_target: Ascend
  2021-05-16 20:12:17,890:INFO:--> is_distributed: 0
  2021-05-16 20:12:17,890:INFO:--> rank: 0
  2021-05-16 20:12:17,890:INFO:--> group_size: 1
  2021-05-16 20:12:17,890:INFO:--> is_save_on_master: 1
  2021-05-16 20:12:17,890:INFO:--> ckpt_save_max: 5
  2021-05-16 20:12:17,890:INFO:--> rank_save_ckpt_flag: 1
  2021-05-16 20:12:17,890:INFO:--> logger: <LOGGER BRDNet (NOTSET)>
  2021-05-16 20:12:17,890:INFO:
  ```

  After the training is completed, you can find the saved weight file in the directory specified by the --output_path parameter. The convergence of some losses during the training process is as follows:

  ```log
  # grep "epoch time:" log.txt
  epoch time: 1197471.061 ms, per step time: 32.853 ms
  epoch time: 1136826.065 ms, per step time: 31.189 ms
  epoch time: 1136840.334 ms, per step time: 31.190 ms
  epoch time: 1136837.709 ms, per step time: 31.190 ms
  epoch time: 1137081.757 ms, per step time: 31.197 ms
  epoch time: 1136830.581 ms, per step time: 31.190 ms
  epoch time: 1136845.253 ms, per step time: 31.190 ms
  epoch time: 1136881.960 ms, per step time: 31.191 ms
  epoch time: 1136850.673 ms, per step time: 31.190 ms
  epoch: 10 step: 36449, loss is 103.104095
  epoch time: 1137098.407 ms, per step time: 31.197 ms
  epoch time: 1136794.613 ms, per step time: 31.189 ms
  epoch time: 1136742.922 ms, per step time: 31.187 ms
  epoch time: 1136842.009 ms, per step time: 31.190 ms
  epoch time: 1136792.705 ms, per step time: 31.189 ms
  epoch time: 1137056.362 ms, per step time: 31.196 ms
  epoch time: 1136863.373 ms, per step time: 31.191 ms
  epoch time: 1136842.938 ms, per step time: 31.190 ms
  epoch time: 1136839.011 ms, per step time: 31.190 ms
  epoch time: 1136879.794 ms, per step time: 31.191 ms
  epoch: 20 step: 36449, loss is 61.104546
  epoch time: 1137035.395 ms, per step time: 31.195 ms
  epoch time: 1136830.626 ms, per step time: 31.190 ms
  epoch time: 1136862.117 ms, per step time: 31.190 ms
  epoch time: 1136812.265 ms, per step time: 31.189 ms
  epoch time: 1136821.096 ms, per step time: 31.189 ms
  epoch time: 1137050.310 ms, per step time: 31.196 ms
  epoch time: 1136815.292 ms, per step time: 31.189 ms
  epoch time: 1136817.757 ms, per step time: 31.189 ms
  epoch time: 1136876.477 ms, per step time: 31.191 ms
  epoch time: 1136798.538 ms, per step time: 31.189 ms
  epoch: 30 step: 36449, loss is 116.179596
  epoch time: 1136972.930 ms, per step time: 31.194 ms
  epoch time: 1136825.174 ms, per step time: 31.189 ms
  epoch time: 1136798.900 ms, per step time: 31.189 ms
  epoch time: 1136828.101 ms, per step time: 31.190 ms
  epoch time: 1136862.983 ms, per step time: 31.191 ms
  epoch time: 1136989.445 ms, per step time: 31.194 ms
  epoch time: 1136688.820 ms, per step time: 31.186 ms
  epoch time: 1136858.111 ms, per step time: 31.190 ms
  epoch time: 1136822.853 ms, per step time: 31.189 ms
  epoch time: 1136782.455 ms, per step time: 31.188 ms
  epoch: 40 step: 36449, loss is 70.95368
  epoch time: 1137042.689 ms, per step time: 31.195 ms
  epoch time: 1136797.706 ms, per step time: 31.189 ms
  epoch time: 1136817.007 ms, per step time: 31.189 ms
  epoch time: 1136861.577 ms, per step time: 31.190 ms
  epoch time: 1136698.149 ms, per step time: 31.186 ms
  epoch time: 1137052.034 ms, per step time: 31.196 ms
  epoch time: 1136809.339 ms, per step time: 31.189 ms
  epoch time: 1136851.343 ms, per step time: 31.190 ms
  epoch time: 1136761.354 ms, per step time: 31.188 ms
  epoch time: 1136837.762 ms, per step time: 31.190 ms
  epoch: 50 step: 36449, loss is 87.13184
  epoch time: 1137022.554 ms, per step time: 31.195 ms
  2021-05-19 14:24:52,695:INFO:training finished....
  ...
  ```

  Convergence of loss when 8 cards are in parallel:

  ```log
  epoch time: 217708.130 ms, per step time: 47.785 ms
  epoch time: 144899.598 ms, per step time: 31.804 ms
  epoch time: 144736.054 ms, per step time: 31.768 ms
  epoch time: 144737.085 ms, per step time: 31.768 ms
  epoch time: 144738.102 ms, per step time: 31.769 ms
  epoch: 5 step: 4556, loss is 106.67432
  epoch time: 144905.830 ms, per step time: 31.805 ms
  epoch time: 144736.539 ms, per step time: 31.768 ms
  epoch time: 144734.210 ms, per step time: 31.768 ms
  epoch time: 144734.415 ms, per step time: 31.768 ms
  epoch time: 144736.405 ms, per step time: 31.768 ms
  epoch: 10 step: 4556, loss is 94.092865
  epoch time: 144921.081 ms, per step time: 31.809 ms
  epoch time: 144735.718 ms, per step time: 31.768 ms
  epoch time: 144737.036 ms, per step time: 31.768 ms
  epoch time: 144737.733 ms, per step time: 31.769 ms
  epoch time: 144738.251 ms, per step time: 31.769 ms
  epoch: 15 step: 4556, loss is 99.18075
  epoch time: 144921.945 ms, per step time: 31.809 ms
  epoch time: 144734.948 ms, per step time: 31.768 ms
  epoch time: 144735.662 ms, per step time: 31.768 ms
  epoch time: 144733.871 ms, per step time: 31.768 ms
  epoch time: 144734.722 ms, per step time: 31.768 ms
  epoch: 20 step: 4556, loss is 92.54497
  epoch time: 144907.430 ms, per step time: 31.806 ms
  epoch time: 144735.713 ms, per step time: 31.768 ms
  epoch time: 144733.781 ms, per step time: 31.768 ms
  epoch time: 144736.005 ms, per step time: 31.768 ms
  epoch time: 144734.331 ms, per step time: 31.768 ms
  epoch: 25 step: 4556, loss is 90.98991
  epoch time: 144911.420 ms, per step time: 31.807 ms
  epoch time: 144734.535 ms, per step time: 31.768 ms
  epoch time: 144734.851 ms, per step time: 31.768 ms
  epoch time: 144736.346 ms, per step time: 31.768 ms
  epoch time: 144734.939 ms, per step time: 31.768 ms
  epoch: 30 step: 4556, loss is 114.33954
  epoch time: 144915.434 ms, per step time: 31.808 ms
  epoch time: 144737.336 ms, per step time: 31.769 ms
  epoch time: 144733.943 ms, per step time: 31.768 ms
  epoch time: 144734.587 ms, per step time: 31.768 ms
  epoch time: 144735.043 ms, per step time: 31.768 ms
  epoch: 35 step: 4556, loss is 97.21166
  epoch time: 144912.719 ms, per step time: 31.807 ms
  epoch time: 144734.795 ms, per step time: 31.768 ms
  epoch time: 144733.824 ms, per step time: 31.768 ms
  epoch time: 144735.946 ms, per step time: 31.768 ms
  epoch time: 144734.930 ms, per step time: 31.768 ms
  epoch: 40 step: 4556, loss is 82.41978
  epoch time: 144901.017 ms, per step time: 31.804 ms
  epoch time: 144735.060 ms, per step time: 31.768 ms
  epoch time: 144733.657 ms, per step time: 31.768 ms
  epoch time: 144732.592 ms, per step time: 31.767 ms
  epoch time: 144731.292 ms, per step time: 31.767 ms
  epoch: 45 step: 4556, loss is 77.92129
  epoch time: 144909.250 ms, per step time: 31.806 ms
  epoch time: 144732.944 ms, per step time: 31.768 ms
  epoch time: 144733.161 ms, per step time: 31.768 ms
  epoch time: 144732.912 ms, per step time: 31.768 ms
  epoch time: 144733.709 ms, per step time: 31.768 ms
  epoch: 50 step: 4556, loss is 85.499596
  2021-05-19 02:44:44,219:INFO:training finished....
  ```

### Evaluation process

#### eval

Before running the following command, please check whether the weight file path used for inference evaluation is correct.

- Ascend processor environment operation

  ```shell
  cd scripts
  bash run_eval_ascend.sh [config_file] [test_dataset_path] [pretrain_path] [ckpt_name]
  ```

  ```log
  2021-05-17 13:40:45,909:INFO:Start to test on ./Test/CBSD68/
  2021-05-17 13:40:46,447:INFO:load test weights from channel_3_sigma_15_rank_0-50_227800.ckpt
  2021-05-17 13:41:52,164:INFO:Before denoise: Average PSNR_b = 24.62, SSIM_b = 0.56;After denoise: Average PSNR = 34.05, SSIM = 0.94
  2021-05-17 13:41:52,207:INFO:testing finished....
  ```

- GPU evaluation

  ```shell
  cd scripts
  bash run_eval_gpu.sh [config_file] [test_dataset_path] [pretrain_path] [ckpt_name]
  ```

After the evaluation is completed, you can find the image after adding Gaussian noise and the image after removing Gaussian noise by the model in the directory specified by the --output_path parameter. The image naming method represents the processing result. For example, 00001_sigma15_psnr24.62.bmp is the picture after adding noise (psnr=24.62 after adding noise), and 00001_psnr31.18.bmp is the picture after removing noise (after denoising psnr=31.18).

In addition, the metrics.csv file in this folder records the processing results of each test picture in detail, as shown below, psnr_b is the psnr value before denoising, and psnr is the psnr value after denoising; the same is true for ssim indicators.

|      | name    | psnr_b      | psnr        | ssim_b      | ssim        |
| ---- | ------- | ----------- | ----------- | ----------- | ----------- |
| 0    | 1       | 24.61875916 | 31.17827606 | 0.716650724 | 0.910416007 |
| 1    | 2       | 24.61875916 | 35.12858963 | 0.457143694 | 0.995960176 |
| 2    | 3       | 24.61875916 | 34.90437698 | 0.465185702 | 0.935821533 |
| 3    | 4       | 24.61875916 | 35.59785461 | 0.49323535  | 0.941600204 |
| 4    | 5       | 24.61875916 | 32.9185257  | 0.605194688 | 0.958840668 |
| 5    | 6       | 24.61875916 | 37.29947662 | 0.368243992 | 0.962466478 |
| 6    | 7       | 24.61875916 | 33.59238052 | 0.622622728 | 0.930195987 |
| 7    | 8       | 24.61875916 | 31.76290894 | 0.680295587 | 0.918859363 |
| 8    | 9       | 24.61875916 | 34.13358688 | 0.55876708  | 0.939204693 |
| 9    | 10      | 24.61875916 | 34.49848557 | 0.503289104 | 0.928179622 |
| 10   | 11      | 24.61875916 | 34.38597107 | 0.656857133 | 0.961226702 |
| 11   | 12      | 24.61875916 | 32.75747299 | 0.627940595 | 0.910765707 |
| 12   | 13      | 24.61875916 | 34.52487564 | 0.54259634  | 0.936489582 |
| 13   | 14      | 24.61875916 | 35.40441132 | 0.44824928  | 0.93462956  |
| 14   | 15      | 24.61875916 | 32.72385788 | 0.61768961  | 0.91652298  |
| 15   | 16      | 24.61875916 | 33.59120178 | 0.703662276 | 0.948698342 |
| 16   | 17      | 24.61875916 | 36.85597229 | 0.365240872 | 0.940135658 |
| 17   | 18      | 24.61875916 | 37.23021317 | 0.366332233 | 0.953653395 |
| 18   | 19      | 24.61875916 | 33.49061584 | 0.546713233 | 0.928890586 |
| 19   | 20      | 24.61875916 | 33.98015213 | 0.463814735 | 0.938398063 |
| 20   | 21      | 24.61875916 | 32.15977859 | 0.714740098 | 0.945747674 |
| 21   | 22      | 24.61875916 | 32.39984512 | 0.716880679 | 0.930429876 |
| 22   | 23      | 24.61875916 | 34.22258759 | 0.569748521 | 0.945626318 |
| 23   | 24      | 24.61875916 | 33.974823   | 0.603115499 | 0.941333234 |
| 24   | 25      | 24.61875916 | 34.87198639 | 0.486003697 | 0.966141582 |
| 25   | 26      | 24.61875916 | 33.2747879  | 0.593207896 | 0.917522907 |
| 26   | 27      | 24.61875916 | 34.67901611 | 0.504613101 | 0.921615481 |
| 27   | 28      | 24.61875916 | 37.70562363 | 0.331322074 | 0.977765024 |
| 28   | 29      | 24.61875916 | 31.08887672 | 0.759773433 | 0.958483219 |
| 29   | 30      | 24.61875916 | 34.48878479 | 0.502000451 | 0.915705442 |
| 30   | 31      | 24.61875916 | 30.5480938  | 0.836367846 | 0.949165165 |
| 31   | 32      | 24.61875916 | 32.08041382 | 0.745214283 | 0.941719413 |
| 32   | 33      | 24.61875916 | 33.65553284 | 0.556162357 | 0.963605523 |
| 33   | 34      | 24.61875916 | 36.87154388 | 0.384932011 | 0.93150568  |
| 34   | 35      | 24.61875916 | 33.03263474 | 0.586027861 | 0.924151421 |
| 35   | 36      | 24.61875916 | 31.80633736 | 0.572878599 | 0.84426564  |
| 36   | 37      | 24.61875916 | 33.26797485 | 0.526310742 | 0.938789487 |
| 37   | 38      | 24.61875916 | 33.71062469 | 0.554955184 | 0.914420724 |
| 38   | 39      | 24.61875916 | 37.3455925  | 0.461908668 | 0.956513464 |
| 39   | 40      | 24.61875916 | 33.92232895 | 0.554454744 | 0.89727515  |
| 40   | 41      | 24.61875916 | 33.05244827 | 0.590977669 | 0.931611121 |
| 41   | 42      | 24.61875916 | 34.60203552 | 0.492371827 | 0.927084684 |
| 42   | 43      | 24.61875916 | 35.20042419 | 0.535991669 | 0.949365258 |
| 43   | 44      | 24.61875916 | 33.47367096 | 0.614959836 | 0.954348624 |
| 44   | 45      | 24.61875916 | 37.65309143 | 0.363631308 | 0.944297135 |
| 45   | 46      | 24.61875916 | 31.95152092 | 0.709732175 | 0.924522877 |
| 46   | 47      | 24.61875916 | 31.9910202  | 0.70427531  | 0.932488263 |
| 47   | 48      | 24.61875916 | 34.96608353 | 0.585813344 | 0.969006479 |
| 48   | 49      | 24.61875916 | 35.39241409 | 0.388898522 | 0.923762918 |
| 49   | 50      | 24.61875916 | 32.11050415 | 0.653521299 | 0.938310325 |
| 50   | 51      | 24.61875916 | 33.54981995 | 0.594990134 | 0.927819192 |
| 51   | 52      | 24.61875916 | 35.79096603 | 0.371685684 | 0.922166049 |
| 52   | 53      | 24.61875916 | 35.10015869 | 0.410564244 | 0.895557165 |
| 53   | 54      | 24.61875916 | 34.12319565 | 0.591762364 | 0.925524533 |
| 54   | 55      | 24.61875916 | 32.79537964 | 0.653338313 | 0.92444253  |
| 55   | 56      | 24.61875916 | 29.90909004 | 0.826190114 | 0.943322361 |
| 56   | 57      | 24.61875916 | 33.23035812 | 0.527200282 | 0.943572938 |
| 57   | 58      | 24.61875916 | 34.56663132 | 0.409658968 | 0.898451686 |
| 58   | 59      | 24.61875916 | 34.43690109 | 0.454208463 | 0.904649734 |
| 59   | 60      | 24.61875916 | 35.0402565  | 0.409306735 | 0.902388573 |
| 60   | 61      | 24.61875916 | 34.91940308 | 0.443635762 | 0.911728501 |
| 61   | 62      | 24.61875916 | 38.25325394 | 0.42568323  | 0.965163887 |
| 62   | 63      | 24.61875916 | 32.07671356 | 0.727443576 | 0.94306612  |
| 63   | 64      | 24.61875916 | 31.72690964 | 0.671929657 | 0.902075231 |
| 64   | 65      | 24.61875916 | 33.47768402 | 0.533677042 | 0.922906399 |
| 65   | 66      | 24.61875916 | 42.14694977 | 0.266868591 | 0.991976082 |
| 66   | 67      | 24.61875916 | 30.81770706 | 0.84768647  | 0.957114518 |
| 67   | 68      | 24.61875916 | 32.24455261 | 0.623004258 | 0.97843051  |
| 68   | Average | 24.61875916 | 34.05390495 | 0.555872787 | 0.935704286 |

### 310 Inference

- Run in Ascend 310 processor environment

  ```shell
  # Start inference by bash command
  bash run_infer_310.sh [model_path] [data_path] [noise_image_path] [sigma] [channel] [device_id]
  # The above command will complete all the work required for inference. After the execution is completed, three log files, preprocess.log, infer.log, and psnr.log, will be generated.
  # If you need to execute each part of the code separately, you can refer to the process in run_infer_310.sh for compilation, image preprocessing, inference, and PSNR calculation. Please check the parameters required for each part!
  ```

## Model description

### performance

#### Assess performance

BRDNet on “waterloo5050step40colorimage”

| Parameters                 | BRDNet                                                                         | BRDNet |
| -------------------------- | ------------------------------------------------------------------------------ | ------ |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G                                 | GPU: 8p Tesla V100-PCIE 32G |
| uploaded Date              | 5/20/2021 (month/day/year)                                                     | 12/24/2021 |
| MindSpore Version          | master                                                                         | 1.5.0 |
| Dataset                    | waterloo5050step40colorimage                                                   | waterloo5050step40colorimage |
| Training Parameters        | epoch=50, batch_size=32, lr=0.001                                              | epoch=50, batch_size=32, lr=0.001 |
| Optimizer                  | Adam                                                                           | Adam |
| Loss Function              | MSELoss(reduction='sum')                                                       | MSELoss(reduction='sum')  |
| outputs                    | denoised images                                                                | denoised images |
| Loss                       | 80.839773                                                                      | 78.04561 |
| Speed                      | 8p about 7000FPS to 7400FPS                                                    | 8p 78ms/step |
| Total time                 | 8p  about 2h 14min                                                             | 8p 5h |
| Checkpoint for Fine tuning | 8p: 13.68MB , 1p: 19.76MB (.ckpt file)                                         | 14M (.ckpt file) |
| Scripts                    | [BRDNet](https://gitee.com/mindspore/models/tree/master/official/cv/brdnet) | [BRDNet](https://gitee.com/mindspore/models/tree/master/official/cv/brdnet) |

## Random description

A random seed is set in train.py.

## ModelZoo主页

Please visit the official website [Home page](https://gitee.com/mindspore/models) .
