# Contents

- [Relation Network Description](#relation-network-description)
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
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Relation Network Description](#contents)

[Relation Network](https://arxiv.org/abs/1711.06025) was propsed in 2018, a conceptually simple, flexible, and general framework for few-shot learning. It was used to learn to recognise new classes given only few examples from each. Once trained, a RN is able to classify images of new classes by computing relation scores between query images and the few examples of each new class without further updating the network.

[Paper](https://arxiv.org/abs/1711.06025): Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip H.S. Torr, Timothy M. Hospedales. Learning to Compare: Relation Network for Few-Shot Learning. 2018.

# [Model Architecture](#contents)

Relation-Net contains 2 parts named Encoder and Relation. The former one has 4 convolution layers, the latter one has 2 convolution layers and 2 linear layers.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [omniglot](https://github.com/brendenlake/omniglot)

- Dataset size 4.02M，32462 28*28 in 1622 classes
    - Train 1,200 classes  
    - Test 422 classes
- Data format .png files
    - Note Data has been processed in omniglot_resized

- The directory structure is as follows:

```shell
└─Data
    ├─miniImagenet
    │
    └─omniglot_resized
           Alphabet_of_the_Magi
           Angelic
```

# [Environment Requirements](#contents)

- Hardware (Ascend/GPU)
    - Prepare hardware environment with Ascend/GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

```shell
# enter script dir, train RelationNet
bash run_standalone_train_ascend.sh ./ckpt /data/omniglot_resized 0
# enter script dir, evaluate RelationNet
bash run_standalone_eval_ascend.sh ./ckpt/omniglot_encoder_relation_network5way_1shot.ckpt /data/omniglot_resized 0
# enter script dir, train RelationNet on 8 divices
bash run_distribution_ascend.sh ./hccl_8p_01234567_127.0.0.1.json  ./ckpt /data/omniglot_resized
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link: https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.

- Running on GPU

```shell
# enter script dir, train RelationNet
bash run_standalone_train_gpu.sh ./ckpt /data/omniglot_resized 0
# enter script dir, evaluate RelationNet
bash run_standalone_eval_gpu.sh ./ckpt/omniglot_encoder_relation_network5way_1shot.ckpt /data/omniglot_resized 0
# enter script dir, train RelationNet on 8 GPUs
bash run_distribution_gpu.sh /data/omniglot_resized 8
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
├── cv
    ├── relationnet
        ├── README.md                             # description
        ├── scripts
        │   ├──run_distribution_ascend.sh         # launch ascend distributed training (8 pcs)
        │   ├──run_distribution_gpu.sh            # launch GPU distributed training (8 pcs)
        │   ├──run_infer_310.sh                   # launch infer on Ascend310
        │   ├──run_standalone_eval_ascend.sh      # launch ascend evaluation
        │   ├──run_standalone_eval_gpu.sh         # launch gpu evaluation
        │   ├──run_standalone_train_ascend.sh     # launch ascend standalone training (1 pcs)
        │   └──run_standalone_train_gpu.sh        # launch gpu standalone training (1 pcs)
        ├── src
        │   ├──config.py                          # parameter configuration
        │   ├──dataset.py                         # creating dataset
        │   ├──lr_generator.py                    # generate lr
        │   ├──relationnet.py                     # relationnet architecture
        │   └──net_train.py                       # train model
        ├── ascend310_infer                       # Source code for Ascend310 Infer
            ├──inc
                └──utils.h
            └──src
                ├──main.cc
                └──utils.cc
        ├── train.py                              # training script
        ├── eval.py                               # evaluation script
        ├── export.py                             # export model
        └── argparser.py                          # command line arguments parsing
```

## [Script Parameters](#contents)

Major parameters in train.py and config.py as follows:

```shell
--class_num: the number of class we use in one step.
--sample_num_per_class: the number of quert data we extract from one class.
--batch_num_per_class: the number of support data we extract from one class.
--data_path: The absolute full path to the train and evaluation datasets.
--episode: Total training epochs.
--test_episode: Total testing episodes
--learning_rate: Learning rate
--device_target: Device where the code will be implemented.
--save_dir: The absolute full path to the checkpoint file saved after training.
--data_path: Path where the dataset is saved
```

## [Training Process](#contents)

### Training

- Running on Ascend

```bash
python train.py --ckpt_dir ./ckpt --data_path /data/omniglot_resized --device_id 0 --device_target Ascend &> train.log &
# or enter script dir, and run the script
bash run_standalone_train_ascend.sh ./ckpt /data/omniglot_resized 0
```

- Running on GPU

```bash
python train.py --ckpt_dir ./ckpt --data_path /data/omniglot_resized --device_id 0 --device_target GPU &> train.log &
# or enter script dir, and run the script
bash run_standalone_train_gpu.sh ./ckpt /data/omniglot_resized 0
```

After training, the loss value will be achieved as follows:

```shell
# grep train.log
...
init data folders
init neural networks
init optim,loss
init loss function and grads
==========Training==========
-----Episode 100/1000000-----
Episode: 100 Train, Loss(MSE): 0.16057138
-----Episode 200/1000000-----
Episode: 200 Train, Loss(MSE): 0.16390544
-----Episode 300/1000000-----
Episode: 300 Train, Loss(MSE): 0.1247341
...
```

The model checkpoint will be saved in the current directory.

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- Running on Ascend

```bash
python eval.py --ckpt_dir ./ckpt/omniglot_encoder_relation_network5way_1shot.ckpt --data_path /data/omniglot_resized \
               --device_target Ascend --device_id 0 &> eval.log &
# or enter script dir, and run the script
bash run_standalone_eval_ascend.sh ./ckpt/omniglot_encoder_relation_network5way_1shot.ckpt /data/omniglot_resized 0
```

- Running on GPU

```bash
python eval.py --ckpt_dir ./ckpt/omniglot_encoder_relation_network5way_1shot.ckpt --data_path /data/omniglot_resized \
               --device_target GPU --device_id 0 &> eval.log &
# or enter script dir, and run the script
bash run_standalone_eval_gpu.sh ./ckpt/omniglot_encoder_relation_network5way_1shot.ckpt /data/omniglot_resized 0
```

You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

- Ascend (8 pcs)

```shell
grep "Accuracy: " log
'Accuracy': 0.9842
```

- GPU (single)

```shell
grep "Average accuracy:" eval.log
Average accuracy: 0.9925
```

- GPU (8 pcs)

```shell
grep "Average accuracy:" eval.log
Average accuracy: 0.9911
```

### Evaluation On Ascend310

Enter ./scripts and execute run_infer_310.sh

```shell
bash run_infer_310.sh [path_to_mindir] [path_to_dataset_directory]
```

And we will get the result as followed

```shell
NN inference cost average time: 1.10642ms of infer_count 1000
aver_accuracy: 0.9926
```

### [Export MindIR](#contents)

```shell
# export model
python export.py --ckpt_file ./scripts/ckpt/omniglot_encoder_relation_network5way_1shot.ckpt --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend 910 (8 pcs)                                          | GPU Tesla V100 (single)                                    | GPU GeForce RTX 3090 (8 pcs)                                     |
| -------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| Resource                   | CentOs8.2, Ascend 910, CPU 2.60GHz, 192 cores, RAM 755 GB   | Ubuntu 18.04.6, Tesla V100, CPU 3.00GHz, 8 cores, RAM 32 GB| Ubuntu 18.04.5, GF RTX 3090, CPU 2.90 GHz, 64 cores, RAM 252 GB  |
| uploaded Date              | 08/24/2021 (month/day/year)                                 | 10/15/2021 (month/day/year)                                | 10/19/2021 (month/day/year) |
| MindSpore Version          | 1.2.0                                                       | 1.3.0                                                      | 1.5.0 |
| Dataset                    | OMNIGLOT                                                    | OMNIGLOT                                                   | OMNIGLOT |
| Training Parameters        | episode=1000000, class_num = 5, lr=0.001                    | episode=370000, class_num = 5, lr=0.0005                   | episode=125000, class_num = 5, lr=0.0005 |
| Optimizer                  | Adam                                                        | Adam                                                       | Adam |
| Loss Function              | MSE                                                         | MSE                                                        | MSE |
| outputs                    | Accuracy                                                    | Accuracy                                                   | Accuracy |
| Loss                       | 0.002                                                       | 0.002                                                      | 0.002 |
| Speed                      | 70 ms/episode                                               | 41 ms/episode                                              | 58 ms/episode |
| Total time                 | 4.5h (8 pcs)                                                | 4.5h (1 pcs)                                               | 2h (8 pcs) |
| Checkpoint for Fine tuning | 875k (.ckpt file)                                           | 905K (.ckpt file)                                          | 905K (.ckpt file) |
| Scripts                    | [Link](https://gitee.com/mindspore/models/tree/master/research/cv/relationnet) | [Link](https://gitee.com/mindspore/models/tree/master/research/cv/relationnet) |[Link](https://gitee.com/mindspore/models/tree/master/research/cv/relationnet) |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```omniglot_character_folders``` function.
In net_train.py, we set the random.choice inside ```train``` function.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).  
