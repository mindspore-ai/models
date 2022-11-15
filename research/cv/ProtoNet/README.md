# Contents

- [Prototypical-Network Description](#protonet-description)
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

# [protonet-Description](#contents)

PyTorch code for NeuralIPS 2017 paper: [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)

# [Model Architecture](#contents)

Proto-Net contains 2 parts named Encoder and Relation. The former one has 4 convolution layers, the latter one has 2 convolution layers and 2 linear layers.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Download dataset: [train](https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip), [evaluation](https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip)

Download split txt: [link](https://github.com/jakesnell/prototypical-networks/tree/master/data/omniglot/splits/vinyals)

- Dataset size 4.02M，32462 28*28 in 1622 classes
    - Train 1,200 classes  
    - Test 422 classes
- Data format .png files
    - Note Data has been processed in omniglot_resized

- The directory structure is as follows:

```shell
└─Data
    ├─raw
    ├─spilts
    │     vinyals
    │         test.txt
    │         train.txt
    │         val.txt
    │         trainval.txt
    └─data
           Alphabet_of_the_Magi
           Angelic
```

# [Environment Requirements](#contents)

- Hardware(Ascend&GPU)
    - Prepare hardware environment with Ascend.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```python
# enter script dir, train ProtoNet
sh run_standalone_train_ascend.sh "../dataset" 1 60 500
# enter script dir, evaluate ProtoNet
sh run_standalone_eval_ascend.sh "../dataset" "./output/best_ck.ckpt" 1 5
# enter script dir, train ProtoNet distributed
sh run_distribution_ascend.sh "./rank_table.json" "../dataset" 60 500
# enter script dir, train ProtoNet in GPU.
sh run_standalone_train_gpu.sh "../dataset" 1 60 500
# enter script dir, evaluate ProtoNet
sh run_standalone_eval_gpu.sh "../dataset" "./output/best_ck.ckpt" 1 5
# enter script dir, train ProtoNet distributed
sh run_distribution_gpu.sh "../dataset" 60 500
```

## [Script and Sample Code](#contents)

```shell
├── cv
    ├── ProtoNet
        ├── requirements.txt  
        ├── README.md                    // descriptions about lenet
        ├── scripts
        │   ├──run_standalone_train_gpu.sh          // train in GPU
        │   ├──run_standalone_eval_gpu.sh          //  evaluate in GPU
        │   ├──run_distribution_gpu.sh          //  distribution in GPU
        │   ├──run_standalone_train_ascend.sh          // train in ascend
        │   ├──run_standalone_eval_ascend.sh          //  evaluate in ascend
        │   ├──run_distribution_ascend.sh          //  distribution in ascend
        ├── src
        │   ├──parser_util.py            // parameter configuration
        │   ├──dataset.py               // creating dataset
        │   ├──IterDatasetGenerator.py // generate dataset
        │   ├──protonet.py             // relationnet architecture
        │   ├──PrototypicalLoss.py    // loss function
        ├── train.py                // training script
        ├── eval.py                 //  evaluation script  
```

## [Script Parameters](#contents)

```python
Major parameters in train.py and config.py as follows:

--class_num: the number of class we use in one step.
--sample_num_per_class: the number of quert data we extract from one class.
--batch_num_per_class: the number of support data we extract from one class.
--data_path: The absolute full path to the train and evaluation datasets.
--episode: Total training epochs.
--test_episode: Total testing episodes
--learning_rate: Learning rate
--device_target: Device where the code will be implemented.
--save_dir: The absolute full path to the checkpoint file saved
                   after training.
--data_path: Path where the dataset is saved
```

## [Training Process](#contents)

### Training

```bash
bash run_standalone_train_ascend.sh "../dataset" 1 60 500
bash run_standalone_train_gpu.sh "../dataset" 1 60 500
```

The model checkpoint will be saved in the current directory.

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

```bash
sh run_standalone_eval_ascend.sh "../dataset" "./output/best_ck.ckpt" 1 5
sh run_standalone_eval_gpu.sh "../dataset" "./output/best_ck.ckpt" 1 5
```

```shell

Test Acc in Ascend: 0.9954400658607483  Loss: 0.02102319709956646
Test Acc in GPU: 0.996999979019165  Loss: 0.013885765336453915
```

## [Inference Process](#contents)

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### [Infer on Ascend310](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` specifies path of used "MINDIR" OR "AIR" model.
- `DATASET_PATH` specifies path of omniglot datasets  
- `DEVICE_ID` is optional, default value is 0.

### [Result](#contents)

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'acc': 0.9956
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | ProtoNet(Ascend)                                                   |ProtoNet(GPU)                                                   |
| -------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| Resource                   | CentOs 8.2; Ascend 910 ; CPU 2.60GHz，192cores；Memory 755G             | ubuntu 18.04; Tesla V100 ; CPU 2.60GHz             |
| uploaded Date              | 03/26/2021 (month/day/year)                                 | 08/28/2021 (month/day/year)                                 |
| MindSpore Version          | 1.1.1                                                      | 1.3.0                                                      |
| Dataset                    | OMNIGLOT                                                    |OMNIGLOT                                                    |
| Training Parameters        | episode=500, class_num = 5, lr=0.001, classes_per_it_tr=60, num_support_tr=5, num_query_tr=5, classes_per_it_val=20, num_support_val=5, num_query_val=15         | episode=500, class_num = 5, lr=0.001, classes_per_it_tr=60, num_support_tr=5, num_query_tr=5, classes_per_it_val=20, num_support_val=5, num_query_val=15         |
| Optimizer                  | Adam                                                         | Adam                                                         |
| Loss Function              | Prototypicalloss                                             | Prototypicalloss                                             |
| outputs                    | Accuracy                                                 | Accuracy                                                 |
| Loss                       | 0.002                                                      | 0.002                                                      |
| Speed                      | 215 ms/step                          | 144 ms/step                          |
| Total time                 | 3 h 23m (8p)                | 2 h 48m (8p)                |
| Checkpoint for Fine tuning | 440 KB (.ckpt file)                                         | 441 KB (.ckpt file)                                         |
| Scripts                    | <https://gitee.com/mindspore/models/tree/master/research/cv/ProtoNet> |<https://gitee.com/mindspore/models/tree/master/research/cv/ProtoNet> |

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).  
