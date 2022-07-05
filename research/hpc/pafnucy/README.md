# Contents

- [Contents](#contents)
- [Pafnucy Description](#pafnucy-description)
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
            - [Pafnucy train on pdbbind v2016](#pafnucy-train-on-pdbbind-v2016)
        - [Inference Performance](#inference-performance)
            - [Pafnucy infer on PDBBindv2016](#pafnucy-infer-on-pdbbind-v2016)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Pafnucy Description](#contents)

Pafnucy is a deep convolutional neural network that predicts binding affinity for protein-ligand complexes. The complex is represented with a 3D grid, and the model utilizes a 3D convolution to produce a feature map of this representation, treating the atoms of both proteins and
ligands in the same manner. The model discovers patterns that are encoded by the filters in the convolutional layer and creates a feature map with spatial occurrences for each pattern in the data.

[Paper](https://watermark.silverchair.com/bty374.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAt8wggLbBgkqhkiG9w0BBwagggLMMIICyAIBADCCAsEGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMlQsvKbTH5dfxUn0PAgEQgIICkkrW7CAf1uMTk_v4Y7Q1Ye9-sBPbjrZSRUKkj-Rs5pWdDx9rrD9NwkaXZ886O2JNA6G8-WG1R74vcz6yQLolrf02TcbdPJY4LdglhqwDZAZEjeMb-TMPtyn_G9a0YSO7Z6LBibfO9FScM3X9VPP2pA9_Qo4Yz70idqciaP-rssYZm5xKyatns6mGyJUDl-H1kzgmaZYyrTL1K68Aic787un4r5GeaqmJDz3HTqlM8RAJRSa78FfouHfiWUNF6W0pGpV0NhR0mgjH5AQEfMYzY2M9tNlRz5fGvuZqdz4rljk0AMcSRWaIRzdP_MRBoWKTFoAkWOAEJmfs6Ql5gLYXthzdTGiJDUoMXWrr3EJ3xg9NNQoxQjuxWVIOurqTAf7Wy6l6KAaKTsWQ_ldzOUu5l2vsOqNz0VtLdlwjQ64RC9_6x8m7u_4Txk2UuGoWhxRWExaYyZq-5DHU_OWmvWLFRQymOFOKCPBzSDF9l7-yEDLRpmUcKsvJmrHpmzSpYZ5iz3aeNMEbk1W8OVDmRrmJ3pOmgmIKe_03BcM2Dc5db37RTinw0FUpUitsNC3R6tl2z2xUQGxNixoaBgmZ4Avcuo74SjUYKPpkSV5sKTO7X3sWoRT0qMBdwgTqOun__NRlo4ynPk0oXICGZNSlsvn69MBSRDsNkBiJDetoslVjj5YytlAMGeuYnRM4H8_dLHCdsTcwrzb6gEaNth3d8zwc06pQtjd6JLCnHUu3xMK8WgmuP1hoKKkqrf0PSUB4XHTrt7mHUMSPkxUcfa5VcRosUGS893wBeekVsBolrckUu7blZDQHgA8KlLHOF14vj0dV_spL1pUgNnEU2SUlF-6BoKpoRYMtUfXQ-idVRYj12SO_T1E): Marta M. Stepniewska-Dziubinska, Piotr Zielenkiewicz and, Pawel Siedlecki. "Development and evaluation of a deep learning model for protein–ligand binding affinity prediction", published in Bioinformatics.

# [Model Architecture](#contents)

Pafnucy model consists of two parts: the convolutional and  dense parts, with different types of connections between layers. The input is processed by a block of 3D convolutional layers combined with a max pooling layer. Pafnucy uses 3 convolutional layers with 64, 128 and 256 filters, and is followed by a max pooling layer. The result of the last convolutional layer is flattened and used as input for a block of fully connected layers.  

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [PDBbind v2016](http://www.pdbbind.org.cn/download.php)

- Dataset size：2.82G.
    - Protein-ligand complex(general set minus refined set): includes 9228 complexes in total
    - Protein-ligand complex(refined set): includes 4057 complexes in total
    - Ligand molecules in general set(Mol2 format)
    - Ligand molecules in general set(SDF format)

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorials/experts/en/master/others/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend/GPU/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- dataset preprocess

  ```python
  #before prepare the data, please check you have install openbabel. The version of openbable we used is v2.4.1
  #first, you can run the following example, to preprocess pdbbind dataset
  python process_pdbbind_data.py --data_path pdbbind/dataset/path/

  #then, use split_dataset.py script to split it into 3 subsets
  python src/split_dataset.py -i processed/data/path -o output/path

  #finally, convert to mindrecord
  python src/create_mindrecord.py --data_path  ~/splited_data/path --mindrecord_path output/path
  ```

- running on Ascend

  ```yaml
  # Add data set path
  mindrecord_path:/home/DataSet/mindrecord_path/

  # Add checkpoint path parameters before inference
  chcekpoint_path:/home/model/pafnucy/ckpt/pafnucy.ckpt
  ```

  ```python
  # run training example
  python train.py > train.log 2>&1 &

  # run distributed training example
  bash run_distribution_train.sh [MINDRECORD_PATH] [RANKTABLE_PATH] [DEVICE_NUM]
  # example: bash run_distribution_train.sh ~/mindrecord_path/ ~/hccl_8p.json 8

  #run standalone training example
  bash run_standalone_train.sh [MINDRECORD_PATH] [DEVICE_ID]

  # run evaluation example
  python eval.py > eval.log 2>&1 &
  OR
  bash run_eval.sh [MINDRECORD_PATH] [CKPT_PATH] [DEVICE_ID]
  # example: bash run_eval.sh ~/mindrecord_path/ ~/pafnucy.ckpt 1

  #predict process, which can be used to score molecular complexes. As input it takes 3D grids, with each grid point described with 19 atomic features.
  #first, you can create grids from molecular structures using following command.
  python prepare.py -l ligand1.mol2 -p pocket.mol2 -o complexes.hdf

  #then, when complexes are prepared, you can score them with the trained network.
  python predict.py --predict_input /path/to/complexes.hdf --ckpt_file /path/to/pafnucy.ckpt
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md                          // descriptions about all the models
    ├── pafnucy
        ├── README.md                    // descriptions about pafnucy
        ├── scripts
        │   ├──run_distribution_train.sh             // shell script for distributed on Ascend
        │   ├──run_standalone_train.sh         // shell script for standalone on Ascend
        │   ├──run_eval.sh              // shell script for evaluation on Ascend
        ├── src
        │   ├──data.py             // creating dataset
        │   ├──dataloader.py             // creating dataset
        │   ├──logger.py          // logger module
        │   ├──net.py            // pafnucy architecture
        │   ├──split_dataset.py             // split dataset
        ├── train.py               // training script
        ├── process_pdbbind_data.py               // dataset preprocess
        ├── prepare.py               // prepare complexes
        ├── predict.py               // score complexes
        ├── eval.py               //  evaluation script
        ├── default_config.yaml       // config file
        ├── export.py             // export checkpoint files into air/mindir
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for Pafnucy

  ```python
  'grid_spacing': 1.0       # distance between grid points
  'lr': 1e-5                # learning rate
  'momentum': 0.9           # momentum
  'weight_decay': 0.001     # weight decay
  'epoch_size': 20          # epoch number
  'batch_size': 20           # batch size
  'max_dist': 10.0          # max distance from complex center
  'conv_patch': 5           # kernel size for convolutional layers
  'pool_patch': 2           # kernel size for pooling layers
  'conv_channels': [64, 128, 256] # number of fileters in convolutional layers
  'dense_sizes': [1000, 500, 200] # number of neurons in dense layers
  'keep_prob': 0.5          # dropout rate
  'rotations': 24           # rotations to perform
  ```

For more configuration details, please refer the script `default_config.yaml`.

## [Training Process](#contents)

### Training

- running on Ascend

  ```python
  python train.py > train.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```text
  # grep "loss is " train.log
  epoch[0], train error: [0.0004304781323298812], Validation error: [2.4519902387857444], Validation RMSE: [1.5658832136483694]
  epoch[1], train error: [0.001452913973480463], Validation error: [2.4301812992095946], Validation RMSE: [1.5589038774759638]
  ...
  ```

  The model checkpoint will be saved in the current directory.

### Distributed Training

- running on Ascend

  ```bash
  bash run_distribution_train.sh [MINDRECORD_PATH] [RANKTABLE_PATH] [DEVICE_NUM]
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `train_parallel[X]/year-mouth-day-time-*-rank-[rank_id].log`. The loss value will be achieved as follows:

  ```text
  # grep "result: " train_parallel*/year-mouth-day-time-*-rank-[rank_id].log
  train_parallel0/log:epoch: 1 step: 48, loss is 1.4302931
  train_parallel0/log:epcoh: 2 step: 48, loss is 1.4023874
  ...
  train_parallel1/log:epoch: 1 step: 48, loss is 1.3458025
  train_parallel1/log:epcoh: 2 step: 48, loss is 1.3729336
  ...
  ...
  ```

## [Evaluation Process](#contents)

### Evaluation

- evaluation on pdbbind validation dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/pafnucy/pafnucy.ckpt".

  ```python
  python eval.py > eval.log 2>&1 &  
  OR
  bash run_eval.sh ~/mindrecord_path/ ~/pafnucy.ckpt 1
  ```

  The above python command will run in the background. You can view the results through the file "year-mouth-day-time-*-rank-[rank_id].log". The accuracy of the test dataset will be as follows:

  ```text
  # grep "Validation RMSE: " year-mouth-day-time-*-rank-[rank_id].log
  2022-07-06 11:53:08,535:INFO:Validation RMSE: [1.4378043893221764]
  ```

  Note that for evaluation after distributed training, please set the checkpoint_path to be the last saved checkpoint file such as "scripts/train_parallel0/pafnucy.ckpt". The accuracy of the test dataset will be as follows:

  ```text
  # grep "Validation RMSE: " eval/year-mouth-day-time-*-rank-[rank_id].log
  2022-07-06 11:53:08,535:INFO:Validation RMSE: [1.4378043893221764]
  ```

## [Export Process](#contents)

### [Export](#content)

Before export model, you must modify the config file, Cifar10 config file is cifar10_config.yaml and imagenet config file is imagenet_config.yaml.
The config items you should modify are batch_size and ckpt_file.

```shell
python export.py --ckpt_file path/to/checkpoint --file_format file_format
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

#### Pafnucy train on pdbbind v2016

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | Pafnucy                                               |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8             |
| uploaded Date              | 07/06/2022 (month/day/year)                                 |
| MindSpore Version          | 1.6.0                                                       |
| Dataset                    | PDBBind v2016                                                    |
| Training Parameters        | epoch=20, batch_size = 20, lr=1e-5              |
| Optimizer                  | Adam                                                    |
| Loss Function              | MSELoss                                      |
| Loss                       | 0.0016                                                      |
| Speed                      | 1pc: 13 ms/step;  8pcs: 12 ms/step                          |
| Total time                 | 1pc: 99 mins;  8pcs: 25 mins                          |
| Parameters (M)             | 13.0                                                        |
| Checkpoint for Fine tuning | 147M (.ckpt file)                                         |
| Model for inference        | 49M (.mindir file),  49M(.air file)                     |
| Scripts                    | [Pafnucy script](https://gitee.com/mindspore/models/tree/master/research/hpc/pafnucy) |

### Inference Performance

#### Pafnucy infer on PDBBindv2016

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | Pafnucy                |
| Resource            | Ascend 910; OS Euler2.8                  |
| Uploaded Date       | 07/06/2022 (month/day/year) |
| MindSpore Version   | 1.6.1                       |
| Dataset             | PDBBind v2016                |
| batch_size          | 20                         |
| outputs             | probability                 |
| Accuracy            | 1pcs: 1.44                |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/models).  

[]: #contents

[]: #Pafnucy-description