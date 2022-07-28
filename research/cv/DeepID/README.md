# Contents

- [DeepID Description](#DeepID-description)
- [Dataset](#Dataset)
    - [Dataset Process](#Dataset-Process)
        - [Crop images](#Crop-images)
        - [Split images](#Split-images)
- [Environment Requirements](#Environment-requirements)
- [Script Description](#Script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training Process](#training-process)
    - [Prediction Process](#prediction-process)
    - [Export MindIR](#export-mindir)
- [Result](#Result)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# [DeepID-description](#Contents)

[论文](https://openaccess.thecvf.com/content_cvpr_2014/papers/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf)：  Sun Y, Wang X, Tang X. Deep learning face representation from predicting 10,000 classes[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2014: 1891-1898.

>This paper proposes to learn a set of high-level feature representations through deep learning, referred to as Deep hidden IDentity features (DeepID), for face verification. We argue that DeepID can be effectively learned through challenging multi-class face identification tasks, whilst they can be generalized to other tasks (such as verification) and new identities unseen in the training set. Moreover, the generalization capability of DeepID increases as more face classes are to be predicted at training. DeepID features are taken from the last hidden layer neuron activations of
deep convolutional networks (ConvNets). When learned as classifiers to recognize about 10; 000 face identities in the training set and configured to keep reducing the neuron numbers along the feature extraction hierarchy, these deep
ConvNets gradually form compact identity-related features in the top layers with only a small number of hidden neurons. The proposed features are extracted from various face regions to form complementary and over-complete mrepresentations. Any state-of-the-art classifiers can be learned based on these high-level representations for face verification. 97:45% verification accuracy on LFW is achieved with only weakly aligned faces.

# [Dataset](#Contents)

[YouTube Faces](<http://www.cs.tau.ac.il/~wolf/ytfaces/>)

The data set contains 3,425 videos of 1,595 different people. All the videos were downloaded from YouTube. An average of 2.15 videos are available for each subject. The shortest clip duration is 48 frames, the longest clip is 6,070 frames, and the average length of a video clip is 181.3 frames.

## [Dataset Process](#Contents)

Used to get the face out of the img. Face in youtube data has been aligned into the center of the img. So this programme aims to increase the ratio of the face in the whole img and resize the img into (47,55), which is the input size for the DeepID.

### [Crop images](#Contents)

```shell
cd ./src
```

Change the **aligned_db_folder** and **result_folder** in crop.py.

``` python
python crop.py
```

### [Split images](#Contents)

After cropping images, change **src_folder** in split.py and split data into two set, One is for train and one is for valid.

```bash
python split.py
```

# [Environment Requirements](#Contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#Contents)

## [Script and Sample Code](#Contents)

```shell
.
└─ DeepID
  ├── ascend310_infer                 // 310 infer directory
  ├─ README.md                        // Descriptions about DeepID
  ├─ scripts
    ├─ run_standalone_train_ascend.sh // Train standalone
    ├─ run_distribute_train_ascend.sh // Train distribute
    ├─ run_eval_ascend.sh             // Evaluation
    ├─ run_infer_310.sh               // 310 inference
    ├─ eval_ascend.sh             // Evaluation
    ├─ run_standalone_train_gpu.sh // Train standalone
    ├─ run_distribute_train_gpu.sh // Train distribute
    └─ eval_gpu.sh             // Evaluation
  ├─src
    ├─ dataset.py                     // Prepare dataset
    ├─ loss.py                        // Loss function
    ├─ model.py                       // Define subnetwork about DeepID
    ├─ util.py                        // Utils for DeepID
    ├─ reporter.py                    // Reporter class
    ├─ cell.py                        // StarGAN model define
    ├─ crop.py                        // Crop images
    ├─ split.py                       // Split train and valid dataset
  ├─ eval.py                          // Evaluation script
  ├─ train.py                         // Train script
  ├─ export.py                        // Export mindir script
  ├─ preprocess.py                    // Convert images and labels to bin
  ├─ postprocess.py                   // Calculate accuracy
```

## Script parameters

```bash
'data_url':'./data/'      # Dataset path
'epochs':200              # Total epochs
'lr':1e-4                 # Learning rate
'batch_size':2048         # Batch size
'input_dim':125           # Image dim
'ckpt_path':''            # Checkpoint saving path
'run_distribute':0        # Run distribute, default: 0
'device_target':'Ascend'  # Device target
'device_id':4             # Device id, default: 0
'device_num':1            # Number of device, default: 1
'rank_id':0               # Rank id, default: 0
'modelarts':0             # Running on modelarts
'train_url':'None'        # Train output path in modelarts
```

After installing MindSpore, follow these steps:

## [Training Process](#Contents)

Train on Ascend

```shell
# train standalone
bash run_standalone_train_ascend.sh [DEVICE_NUM] [DEVICE_ID]

# train distribute
bash run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE]
```

Train on GPU

```shell
# train standalone
bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_DIR]
for example: bash run_standalone_train_gpu.sh 0 ./data

# train distribute
bash run_distribute_train_gpu.sh [DEVICE_NUM] [CUDA_VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [DATA_DIR]
for example: bash run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7 /home/DeepID/data
```

## [Prediction Process](#Contents)

Eval on Ascend

```shell
# Evaluation on Ascend
sh eval_ascend.sh [DEVICE_NUM] [DEVICE_ID]
```

## [Ascen 310 infer](#contents)

### Export MindIR

```bash
python export.py
```

### Infer on Ascend 310

```bash
cd scripts
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` Directionary of MINDIR
- `DATA_PATH` Directionary of dataset
- `DEVICE_ID` Optional, default 0

=======
Eval on GPU

```shell
# Evaluation on GPU
bash eval_gpu.sh [DEVICE_ID] [CHECKPOINTPATH] [DATA_DIR] [MODE(valid,test)]
for example: bash eval_gpu.sh 0 /home/DeepID.ckpt /home/DeepID/data valid
```

# [Result](#Contents)

The evaluation results will be saved in the sample path in a log file named "log_eval.txt". You can find results similar to the following in the log.

```log
Valid dataset accuracy: 0.9683
```

# [Model Description](#Contents)

## [Performance](#Contents)

### Training Performance

| Parameters                 | Ascend 910                                                  |GPU|
| -------------------------- | ----------------------------------------------------------- |---|
| Model Version              | DeepID                                                      |DeepID|
| Resource                   | Ascend                                                      |GPU|
| uploaded Date              | 11/04/2021 (month/day/year)                                 |11/18/2021|
| MindSpore Version          | 1.3.1                                                       |1.5.0|
| Dataset                    | Youtube Face                                                |Youtube Face|
| Training Parameters        | epochs=200, batch_size=1024, lr=0.0001                      |epochs=200, batch_size=2048, lr=0.0001|
| Optimizer                  | Adam                                                        |Adam|
| outputs                    | Accuracy = 99.18%                                           |Accuracy = 99.24%|
| Speed                      | 1pc: 900 ms/step;                                           |1pc: 1250 ms/step 8pc:290 ms/step|
| Total time                 | 1pc: 3h6s;                                                  |1pc: 4:44:52 8pc:0:50:23|
| Parameters (M)             | 4.56  M                                                     |4.56  M   |
| Checkpoint for Fine tuning | 2.03  M (.ckpt file)                                        |2.03  M (.ckpt file)   |

### Inference Performance

| Parameters          | Ascend 910                      |GPU|
| ------------------- | ---------------------------     |---|
| Model Version       | DeepID                          |DeepID|
| Resource            | Ascend                          |GPU|
| Uploaded Date       | 11/04/2021 (month/day/year)     |11/18/2021|
| MindSpore Version   | 1.3.1                           |1.5.0|
| Dataset             | Youtube Face                    |Youtube Face|
| batch_size          | 512                             |256|
| outputs             | Accuracy = 96.83%               |Accuracy = 95.05%|

| Parameters          | Ascend 310                      |
| ------------------- | ---------------------------     |
| Model Version       | DeepID                          |
| Resource            | Ascend                          |
| Uploaded Date       | 11/30/2021 (month/day/year)     |
| MindSpore Version   | 1.3.1                           |
| Dataset             | Youtube Face                    |
| batch_size          | 1                               |
| outputs             | Accuracy = 96.83%               |

# [ModelZoo Homepage](#Contents)

Please check the official [homepage](https://gitee.com/mindspore/models).