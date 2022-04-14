# [Contents](#content)

- [Content](#content)
- [LightCNN description](#lightcnn_description)
- [Model architecture](#modelarchitecture)
- [Dataset](#Dataset)
- [Training method](#trainingmethod)
    - [Mixed precision](#mixedprecision)
- [Environmental requirements](#environmentalrequirements)
- [Quick Start](#Quickstart)
- [Script Description](#scriptdescription)
    - [Script and Sample Code](#script_sample)
    - [Scripts Parameters](#scripts_parameters)
    - [Training process](#training_process)
        - [Training](#training)
            - [Train on Ascend](#ascend_training)
            - [Train on GPU](#gpu_training)
    - [Evaluation process](#evaluation_process)
        - [Ascend](#ascend)
        - [GPU](#gpu)
        - [Results](#results)
            - [Training accuracy](#training_accuracy)
- [Model description](#model_description)
    - [Performance](#performance)
- [ModelZoo homepage](#modelzoo)

# [LightCNN description](#lightcnn_description)

LightCNN is suitable for noisy face recognition datasets. The architecture exploits the proposed maxout named
Max-Feature-Map (MFM). It uses multiple feature maps to approximate the linear approximation of any convex activation
function with maxout. MFM uses a competitive relationship to select the convex activation function, which can separate
noise from useful information, and can also perform feature selection between two feature maps.

[Paper](https://arxiv.org/pdf/1511.02683.pdf): Wu, Xiang, et al. "A light cnn for deep face representation with noisy
labels." IEEE Transactions on Information Forensics and Security 13.11 (2018): 2884-2896.

# [Model architecture](#modelarchitecture)

The lightweight CNN network structure can learn face recognition tasks on training samples that contain a lot of noise:

- The concept of maxout activation is introduced in each convolutional layer of CNN, and a Max-Feature-Map (MFM) with a
  small number of parameters is obtained. Unlike ReLU, which inhibits neurons through threshold or bias, MFM inhibits
  through competition. Not only can the noise signal be separated from the useful signal, but it can also play a key
  role in feature selection.
- The network is based on MFM and has 5 convolutional layers and 4 Network in Network (NIN) layers. The small
  convolution kernel and NIN are to reduce parameters and improve performance.
- A method of semantic bootstrapping by pre-training the model is adopted to improve the stability of the model in noise
  samples. Wrong samples can be detected by the predicted probability.
- Experiments show that the network can train a lightweight model on training samples containing a lot of noise, and a
  single model outputs a 256-dimensional feature vector, achieving a state-of-art effect on a five-face test set. The
  speed on the CPU is 67ms.

# [Dataset](#dataset)

Training set: Microsoft face recognition database (MS-Celeb-1M). The original MS-Celeb-1M dataset contains more than 8
million images. The original author of LightCNN provided a cleaned file list MS-Celeb-1M_clean_list.txt, which contains
79077 people and 5049824 face images. The original dataset was officially deleted by Microsoft due to infringement
issues, and an available third-party download link is provided (you can access
it [here](https://hyper.ai/datasets/5543)). After downloading the dataset, the aligned data should be used which is
provided in the `FaceImageCroppedWithAlignment.tsv` file.

Training set list: The authors of the paper uploaded the file `MS-Celeb-1M_clean_list.txt` with the training list
to [Baidu Yun](http://pan.baidu.com/s/1gfxB0iB)
and [Google](https://drive.google.com/file/d/0ByNaVHFekDPRbFg1YTNiMUxNYXc/view?usp=sharing).

Test set: LFW face data set (Labeled Faces in the Wild). The LFW dataset contains 13,233 face images from 5749 people.
The aligned test set link provided by the original author of LightCNN and it is
available [here](https://pan.baidu.com/s/1eR6vHFO).

Test set list: The original author did not provide a test set list, so the test set list was inferred on the basis of
the results given by the original author. To get the test set list, firslty, download the blufr official test set (it is
available [here](http://www.cbsr.ia.ac.cn/users/scliao/projects/blufr/BLUFR.zip)) and the test results (they are
available [here](https://github.com/AlfredXiangWu/face_verification_experiment/blob/master/results/LightenedCNN_B_lfw.mat))
. Then, unzip the `BLUFR.zip` file and put it in the same folder with the `LightenedCNN_B_lfw.mat` file. After this, run
the `LightCNN/src/get_list.py` file which should be placed in the same directory. It will generate two test set
lists: `image_list_for_lfw.txt` and `image_list_for_blufr.txt`. Instructions:

- Download training set, training set list, test set and generate test set list.

- Convert the downloaded training set (tsv file) into a picture set. Run the
  script:  `bash scripts/convert.sh FILE_PATH OUTPUT_PATH` where `FILE_PATH` is the location of the tsv file
  and `OUTPUT_PATH` is the output folder, which needs to be created by the user. The recommended name
  is `FaceImageCroppedWithAlignment`.

Dataset structure:

```shell
.
└──MS-Celeb-1M
    ├── mat_files
    │   ├── LightenedCNN_B_lfw.mat
    │   ├── blufr_lfw_config.mat                        # lfw 6,000 pairs test configuration file
    │   └── lfw_pairs.mat                               # lfw BLUFR protocols test configuration file
    ├── FaceImageCroppedWithAlignment               # Training dataset MS-Celeb-1M
    │   ├── m.0_0zl
    │   ├── m.0_0zy
    │   ├── m.01_06j
    │   ├── m.0107_f
    │   ...
    │
    ├── lfw                                         # Test dataset LFW
    │   ├── image
    │   │   ├── Aaron_Eckhart
    │   │   ├── Aaron_Guiel
    │   │   ├── Aaron_Patterson
    │   │   ├── Aaron_Peirsol
    │   │   ├── Aaron_Pena
    │   │   ...
    │
    ├── BLUFR
    │   ├── ...
    │
    ├── image_list_for_blufr.txt                # BLUFR protocols test set list, need to be generated by the user, see above for the method
    ├── image_list_for_lfw.txt                  # lfw 6,000 pairs test set list, need to be generated by the user, see above for the method
    └── MS-Celeb-1M_clean_list.txt                  # Cleaned training set list
```

# [Training method](#trainingmethod)

## [Mixed Precision](#mixedprecision)

The [mixed-precision](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html) training
method uses single-precision and half-precision data to improve the training speed of deep learning neural networks,
while maintaining the network accuracy that can be achieved by single-precision training. Mixed-precision training
increases computing speed and reduces memory usage, while supporting training larger models or achieving larger batches
of training on specific hardware. Taking the FP16 operator as an example, if the input data type is FP32, the MindSpore
background will automatically reduce the accuracy to process the data. Users can open the INFO log and search for "
reduce precision" to view the operators with reduced precision.

# [Environmental requirements](#environmentalrequirements)

- Hardware (Ascend/GPU)
    - Prepare the Ascend/GPU processor to build the hardware environment.
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For details, please refer to the following resources:
    - [MindSpore tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)
- Generate config json file for 8-card training
    - [Simple tutorial](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)
    - For detailed configuration method, please refer to
      the [official website tutorial](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#configuring-distributed-environment-variables).

# [Quick start](#Quickstart)

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:

- Preparation before operation Modify the configuration file `src/config.py`, especially to choose the correct dataset
  path.

```python
from easydict import EasyDict as edict

lightcnn_cfg = edict({
    # training setting
    'network_type': 'LightCNN_9Layers',
    'epochs': 80,
    'lr': 0.01,
    'num_classes': 79077,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'batch_size': 128,
    'image_size': 128,
    'save_checkpoint_steps': 60000,
    'keep_checkpoint_max': 40,
    # train data location
    'data_path': '/data/MS-Celeb-1M/FaceImageCroppedWithAlignment',  # Absolute path (need to be modified)
    'train_list': '/data/MS-Celeb-1M_clean_list.txt',  # Absolute path (requires modification)
    # test data location
    'root_path': '/data/lfw/image',  # Absolute path (requires modification)
    'lfw_img_list': 'image_list_for_lfw.txt',  # Filename
    'lfw_pairs_mat_path': 'mat_files/lfw_pairs.mat',  # The relative path
    'blufr_img_list': 'image_list_for_blufr.txt',  # Filename
    'blufr_config_mat_path': 'mat_files/blufr_lfw_config.mat'  # The relative path
})

```

Based on the original LightCNN paper, we conducted training experiments on the MS-Celeb-1M dataset and evaluated on the
LFW dataset.

- Running on Ascend

Run the following training script to configure the training parameters of a single card:

```bash
# Enter the root directory
cd LightCNN/

# Start the training
# DEVICE_ID: Ascend processor id, user needs to specify
bash scripts/train_standalone.sh DEVICE_ID
```

Run the following training script to configure the multi-card training parameters:

```bash
cd LightCNN/scripts

# Running 2-cards or 4-cards training
# hccl.json: Ascend configuration information, which needs to be configured by the user. It is different from eight cards. Please refer to the tutorial on the official website for details.
# DEVICE_NUM should be the same as the number of cards in the train_distribute.sh. Modify device_ids=(id1 id2) or device_ids=(id1 id2 id3 id4)
bash train_distribute.sh hccl.json DEVICE_NUM

# Running 8-cards training
# hccl.json: Ascend configuration information, users need to configure it by themselves
bash train_distribute_8p.sh hccl.json
```

The evaluation steps are the following:

```bash
# Enter the root directory
cd LightCNN/

# Evaluate the performance of LightCNN on lfw 6,000 pairs
# DEVICE_ID: Ascend processor id
# CKPT_FILE: checkpoint file
bash scripts/eval_lfw.sh DEVICE_ID CKPT_FILE

# Evaluate the performance of LightCNN on lfw BLUFR protocols
# DEVICE_ID: Ascend processor id
# CKPT_FILE: checkpoint weight file
bash scripts/eval_blufr.sh DEVICE_ID CKPT_FILE
```

- Running on GPU

Run the following training script to configure the training parameters of a single card:

```bash
# Enter the root directory
cd LightCNN/scripts

# Start the training
# DEVICE_ID: GPU processor id, user needs to specify
bash run_standalone_train_gpu.sh [DATASET_PATH] [DEVICE_ID]
```

Run the following training script to configure the multi-card training parameters:

```bash
cd LightCNN/scripts

# Running 8-cards training
bash run_distribute_train_gpu.sh [DATASET_PATH]
```

The evaluation steps are the following:

```bash
# Enter the root directory
cd LightCNN/scripts

# Evaluate the performance of LightCNN on lfw 6,000 pairs
# DEVICE_ID: GPU processor id
# CKPT_FILE: checkpoint file
bash run_eval_lfw_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]

# Evaluate the performance of LightCNN on lfw BLUFR protocols
# DEVICE_ID: GPU processor id
# CKPT_FILE: checkpoint weight file
bash sh run_eval_blufr_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
```

# [Script description](#scriptdescription)

## [Script and sample code](script_sample)

```shell
.
├── ascend310_infer
│   └── ...
├── scripts
│   ├── eval_blufr.sh                               # lfw BLUFR protocols ascend test script
│   ├── run_eval_blufr_gpu.sh                       # lfw BLUFR protocols gpu test script
│   ├── eval_lfw.sh                                 # lfw 6,000 pairs ascend test script
│   ├── run_eval_lfw_gpu.sh                         # lfw 6,000 pairs gpu test script
│   ├── convert.sh                                  # Training dataset format conversion
│   ├── run_infer_310.sh
│   ├── train_distribute_8p.sh                      # 8-card parallel training
│   ├── run_distribute_train_gpu.sh                 # 8-card gpu parallel training
│   ├── train_distribute.sh                         # Multi-card (2 cards/4 cards) Parallel training
│   ├── train_standalone.sh                         # Single card training script
│   └── run_standalone_train_gpu.sh                 # Single card gpu training script
├── src
│   ├── config.py                                   # training parameter configuration file
│   ├── convert.py                                  # training dataset conversion script
│   ├── dataset.py                                  # training dataset loader
│   ├── get_list.py                                 # obtain the test set list
│   ├── lightcnn.py                                 # LightCNN model file
│   └── lr_generator.py                             # dynamic learning rate generated script
├── eval_blufr.py                                   # LFW BLUFR Protocols test script
├── eval_lfw.py                                     # LFW 6,000 pairs test script
├── train.py                                        # Training script
├── export.py
├── postprocess.py
├── preprocess.py
├── README_CN.md
└── README_EN.md
```

Note: `mat_files` The two mat files in the folder need to be downloaded by the user. `blufr_lfw_config.mat`is available
in the [BLUFR zip file](http://www.cbsr.ia.ac.cn/users/scliao/projects/blufr/BLUFR.zip). Once the archive is
decompressed, the file may be found in the `/BLUFR/config/lfw` folder.
`lfw_pairs.mat` is provided in the original repository, you can find
it [here](https://github.com/AlfredXiangWu/face_verification_experiment/blob/master/code/lfw_pairs.mat).

# [Script parameters](#script_parameters)

Default training configuration

```bash
'network_type': 'LightCNN_9Layers',                 # model name
'epochs': 80,                                       # total number of training epochs
'lr': 0.01,                                         # training learning rate
'num_classes': 79077,                               # number of classes
'momentum': 0.9,                                    # momentum
'weight_decay': 1e-4,                               # attention weights
'batch_size': 128,                                  # batch size
'image_size': 128,                                  # image size
'save_checkpoint_steps': 60000,                     # save checkpoint interval隔step数
'keep_checkpoint_max': 40,                          # maximal number of checpoints to store
```

## [Training process](#training_process)

### [Training](#training)

#### [Train on Ascend](#ascend_training)

```bash
# train_standalone.sh
python3 train.py \
          --device_target Ascend \
          --device_id "$DEVICE_ID" \
          --ckpt_path ./ckpt_files > train_standalone_log.log 2>&1 &
```

```bash
# train_distribute_8p.sh
for ((i = 0; i < ${DEVICE_NUM}; i++)); do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env >env.log
    python3 train.py \
        --device_target Ascend \
        --device_id "$DEVICE_ID" \
        --run_distribute 1 \
        --ckpt_path ./ckpt_files > train_distribute_8p.log 2>&1 &
    cd ..
done
```

```bash
# train_distribute.sh

# distributed devices id
device_ids=(0 1 2 3)

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
    export DEVICE_ID=${device_ids[i]}
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env >env.log
    python3 train.py \
        --device_target Ascend \
        --device_id $DEVICE_ID \
        --run_distribute 1 \
        --ckpt_path ./ckpt_files > train_distribute.log 2>&1 &
  cd ..
done
```

After the training the timing and the loss can be found in the `train_standalone_log.log`, `train_distribute_8p.log`
, `train_distribute.log` file depending on the script.

```bash
# Single card training result
epoch: 1 step: 39451, loss is 4.6629214
epoch time: 4850141.061 ms, per step time: 122.941 ms
epoch: 2 step: 39451, loss is 3.6382508
epoch time: 4148247.801 ms, per step time: 105.149 ms
epoch: 3 step: 39451, loss is 2.9592063
epoch time: 4146129.041 ms, per step time: 105.096 ms
epoch: 4 step: 39451, loss is 3.6300964
epoch time: 4128986.449 ms, per step time: 104.661 ms
epoch: 5 step: 39451, loss is 2.9682
epoch time: 4117678.376 ms, per step time: 104.374 ms
epoch: 6 step: 39451, loss is 3.2115498
epoch time: 4139044.713 ms, per step time: 104.916 ms
...
```

```bash
# Distributed training results (8P)
epoch: 1 step: 4931, loss is 8.716646
epoch time: 1215603.837 ms, per step time: 246.523 ms
epoch: 2 step: 4931, loss is 3.6822505
epoch time: 1038280.276 ms, per step time: 210.562 ms
epoch: 3 step: 4931, loss is 1.8040423
epoch time: 1033455.542 ms, per step time: 209.583 ms
epoch: 4 step: 4931, loss is 1.6634097
epoch time: 1047134.763 ms, per step time: 212.357 ms
epoch: 5 step: 4931, loss is 1.369437
epoch time: 1053151.674 ms, per step time: 213.578 ms
epoch: 6 step: 4931, loss is 1.3599608
epoch time: 1064338.712 ms, per step time: 215.846 ms
...
```

#### [Train on GPU](#gpu_training)

```bash
# run_standalone_train_gpu.sh
python train.py \
          --device_target="GPU" \
          --device_id=$DEVICE_ID \
          --dataset_path=$DATASET_PATH &> log &
```

```bash
# run_distribute_train_gpu.sh
mpirun --allow-run-as-root -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout \
       python train.py --run_distribute=True --device_target="GPU" --dataset_path=$DATASET_PATH &> log &

```

After the training the timing and the loss can be found in the `train_standalone_log.log`, `train_distribute_8p.log`
, `train_distribute.log` file depending on the script.

```bash
# Distributed training results (8P)
epoch: 80 step: 4931, loss is 0.10246127
epoch: 80 step: 4931, loss is 0.15914159
epoch: 80 step: 4931, loss is 0.14224662
epoch: 80 step: 4931, loss is 0.21186805
epoch: 80 step: 4931, loss is 0.20936105
epoch: 80 step: 4931, loss is 0.15212367
epoch: 80 step: 4931, loss is 0.1972757
epoch: 80 step: 4931, loss is 0.18706863
epoch time: 1042074.953 ms, per step time: 211.331 ms
epoch time: 1042075.095 ms, per step time: 211.331 ms
epoch time: 1042075.139 ms, per step time: 211.331 ms
epoch time: 1042075.017 ms, per step time: 211.331 ms
epoch time: 1042075.134 ms, per step time: 211.331 ms
epoch time: 1042075.437 ms, per step time: 211.331 ms
epoch time: 1042075.175 ms, per step time: 211.331 ms
epoch time: 1042075.380 ms, per step time: 211.331 ms
...
```

## [Evaluation](#evaluation)

### [Ascend](#ascend)

```bash
# Enter the root directory
cd LightCNN/

# Evaluate the performance of LightCNN on lfw 6,000 pairs
# DEVICE_ID: Ascend processor id
# CKPT_FILE: checkpoint file
bash scripts/eval_lfw.sh DEVICE_ID CKPT_FILE

# Evaluate the performance of LightCNN on lfw BLUFR protocols
# DEVICE_ID: Ascend processor id
# CKPT_FILE: checkpoint file
bash scripts/eval_blufr.sh DEVICE_ID CKPT_FILE
```

An example of the test script:

```bash
# eval_lfw.sh
# ${DEVICE_ID}: Ascend processor id
# ${ckpt_file}: checkpoint file, input by the user
# eval_lfw.log: saved test results
python3 eval_lfw.py \
            --device_target Ascend \
            --device_id "${DEVICE_ID}" \
            --resume "${ckpt_file}" > eval_lfw.log 2>&1 &
```

```bash
# eval_blufr.sh
# ${DEVICE_ID}: Ascend processor id
# ${ckpt_file}: checkpoint file, input by the user
# eval_blufr.log: saved test results
# Tips: In eval_blufr.py, you can use the numba library to accelerate the calculation. If the numba library is introduced, you can use the'@jit' syntactic sugar for acceleration, just remove the comment
python3 eval_blfur.py \
          --device_target Ascend \
          --device_id "${DEVICE_ID}" \
          --resume "${ckpt_file}" > eval_blufr.log 2>&1 &
```

### [GPU](#gpu)

```bash
# Enter the root directory
cd LightCNN/scripts

# Evaluate the performance of LightCNN on lfw 6,000 pairs
# DEVICE_ID: GPU processor id
# CKPT_FILE: checkpoint file
bash run_eval_lfw_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]

# Evaluate the performance of LightCNN on lfw BLUFR protocols
# DEVICE_ID: GPU processor id
# CKPT_FILE: checkpoint file
bash run_eval_blufr_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
```

An example of the test script:

```bash
# run_eval_lfw_gpu.sh
# ${DATASET_PATH}: path to dataset
# ${DEVICE_ID}: GPU processor id
# ${CHECKPOINT_PATH}: checkpoint file, input by the user
# eval_lfw.log: saved test results
python eval_lfw.py --dataset_path=$PATH1 --resume=$PATH2 \
                   --device_id=$DEVICE_ID --device_target="GPU" > log 2>&1 &
```

```bash
# run_eval_blufr_gpu.sh
# ${DATASET_PATH}: path to dataset
# ${DEVICE_ID}: GPU processor id
# ${CHECKPOINT_PATH}: checkpoint file, input by the user
# eval_blufr.log: saved test results
# Tips: In eval_blufr.py, you can use the numba library to accelerate the calculation. If the numba library is introduced, you can use the'@jit' syntactic sugar for acceleration, just remove the comment
python eval_blufr.py --dataset_path=$PATH1 --resume=$PATH2 \
                   --device_id=$DEVICE_ID --device_target="GPU" > log 2>&1 &
```

### [Results]

Run the applicable training script to get the results. To get the same result, follow the steps in the quick start.

#### [Training accuracy](#training_accuracy)

> Note: This section shows the results of Ascend single card training.

- Evaluation result on lfw 6,000 pairs

| **Model** | 100% - EER | TPR@RAR=1% | TPR@FAR=0.1% | TPR@FAR|
| :----------: | :-----: | :----: | :----: | :-----:|
| LightCNN-9(MindSpore version)| 98.57%| 98.47%  | 95.5% | 89.87% |
| LightCNN-9(PyTorch version)| 98.70%| 98.47%  | 95.13% | 89.53% |

- Evaluation results on lfw BLUFR protoclos

| **Model** | VR@FAR=0.1% | DIR@RAR=1% |
| :----------: | :-----: | :----: |
| LightCNN-9(MindSpore version) | 96.26% | 81.66%|
| LightCNN-9(PyTorch version) | 96.80% | 83.06%|

# [Model description](#model_description)

## [Performance](#performance)

### Training Performance

| Parameters | Ascend 910| GPU Tesla V100 (single) | GPU Tesla V100 (8 pcs) |
| -------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| Model | LightCNN |LightCNN|LightCNN|
| Environment | Ascend 910 |GPU(Tesla V100-PCIE 32G)；CPU：2.70GHz 52cores ；RAM：1.5T|GPU(Tesla V100-PCIE 32G)；CPU：2.70GHz 52cores ；RAM：1.5T|
| Upload date | 2021-05-16 |2021-11-09|2021-11-09|
| MindSpore version | 1.1.1 |1.6.0.20211125|1.6.0.20211125|
| Dataset | MS-Celeb-1M, LFW | MS-Celeb-1M, LFW | MS-Celeb-1M, LFW|
| Training Parameters | epoch = 80, batch_size = 128, lr = 0.01 |epoch = 80, batch_size = 128, lr = 0.01|epoch = 80, batch_size = 128, lr = 0.01|
| Optimizer | SGD |SGD|SGD|
| Loss function | Softmax Cross Entropy |Softmax Cross Entropy|Softmax Cross Entropy|
| Output | Probability |Probability|Probability|
| Loss | 0.10905003 |0.10246127|0.10246127|
| Speed | - | 147.1 ms/step |191.5 ms/step|
| Performance | 103h（1 card）<br>  24h（8 cards） |129h|21h|
| Script | [Link](https://gitee.com/mindspore/models/tree/master/research/cv/LightCNN) |[Link](https://gitee.com/mindspore/models/tree/master/research/cv/LightCNN)|[Link](https://gitee.com/mindspore/models/tree/master/research/cv/LightCNN)|

### Evaluation Performance

- on lfw 6,000 pairs

| Parameters | Ascend 910| GPU GeForce RTX 3090 |
| ---------- |---------- | -------------------- |
| Model | LightCNN |LightCNN|
| Environment | Ascend 910 | Ubuntu 18.04.6, GF RTX3090, CPU 2.90GHz, 64cores, RAM 252GB|
| Upload date | 2021-05-16 |2021-11-09|
| MindSpore version | 1.1.1 |1.5.0|
| Dataset | MS-Celeb-1M, LFW (6,000 pairs) |MS-Celeb-1M, LFW (6,000 pairs) |
| Batch_size | 1 |1|
| 100% - EER | 98.57% |98.13%|
| TPR@RAR=1% | 98.47% |97.73%|
| TPR@FAR=0.1% | 95.5% |93.07%|
| TPR@FAR=0% | 89.87% |78.60%|
| Total time || 5min |

- on lfw BLUFR protoclos

| Parameters | Ascend 910| GPU GeForce RTX 3090 |
| ---------- |---------- | -------------------- |
| Model | LightCNN |LightCNN|
| Environment | Ascend 910 | Ubuntu 18.04.6, GF RTX3090, CPU 2.90GHz, 64cores, RAM 252GB|
| Upload date | 2021-05-16 |2021-11-09|
| MindSpore version | 1.1.1 |1.5.0|
| Dataset | MS-Celeb-1M, LFW (BLUFR protoclos) |MS-Celeb-1M, LFW (BLUFR protoclos) |
| Batch_size | 1 |1|
| VR@FAR=0.1% | 96.26% |94.13%|
| DIR@RAR=1% | 81.66% |74.80%|
| Total time || 10min |

# [ModelZoo homepage](#modelzoo)

Please check the official [homepage](https://gitee.com/mindspore/models).

[1]: https://arxiv.org/pdf/1511.02683

[2]: http://pan.baidu.com/s/1gfxB0iB

[3]: https://drive.google.com/file/d/0ByNaVHFekDPRbFg1YTNiMUxNYXc/view?usp=sharing

[4]: https://hyper.ai/datasets/5543

[5]: https://pan.baidu.com/s/1eR6vHFO

[6]: https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html

[7]: http://www.cbsr.ia.ac.cn/users/scliao/projects/blufr/BLUFR.zip

[8]: https://github.com/AlfredXiangWu/face_verification_experiment/blob/master/code/lfw_pairs.mat

[9]: https://github.com/AlfredXiangWu/face_verification_experiment/blob/master/results/LightenedCNN_B_lfw.mat
