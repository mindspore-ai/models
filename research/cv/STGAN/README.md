# Contents

- [Contents](#contents)
    - [STGAN Description](#stgan-description)
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
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [STGAN Description](#contents)

STGAN was proposed in CVPR 2019, one of the facial attributes transfer networks using Generative Adversarial Networks (GANs). It introduces a new Selective Transfer Unit (STU) to get better facial attributes transfer than others.

[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_STGAN_A_Unified_Selective_Transfer_Network_for_Arbitrary_Image_Attribute_CVPR_2019_paper.pdf): Liu M, Ding Y, Xia M, et al. STGAN: A Unified Selective Transfer Network for Arbitrary Image
Attribute Editing[C]. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
IEEE, 2019: 3668-3677.

## [Model Architecture](#contents)

STGAN composition consists of Generator, Discriminator and Selective Transfer Unit. Using Selective Transfer Unit can help networks keep more attributes in the long term of training.

## [Dataset](#contents)

In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

- Dataset size：1011M，202,599 128*128 colorful images, marked as 40 attributes
    - Train：182,599 images
    - Test：18,800 images
- Data format：binary files
    - Note：Data will be processed in celeba.py
- Download the dataset, the directory structure is as follows:

```bash
├── dataroot
    ├── anno
        ├── list_attr_celeba.txt
    ├── image
        ├── 000001.jpg
        ├── ...
```

## [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend processor.It also supports the use of GPU processor to prepare the hardware environment.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

    ```python
    # train STGAN
    bash scripts/run_standalone_train.sh [DATA_PATH] [EXPERIMENT_NAME] [DEVICE_ID]
    # distributed training
    bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [EXPERIMENT_NAME] [DATA_PATH]
    # evaluate STGAN
    bash scripts/run_eval.sh [DATA_PATH] [EXPERIMENT_NAME] [DEVICE_ID] [CHECKPOINT_PATH]
    ```

- running on GPU

    ```python
    # train STGAN
    bash scripts/run_standalone_train_gpu.sh [DATA_PATH] [EXPERIMENT_NAME] [DEVICE_ID]
    # distributed training
    bash scripts/run_distribute_train_gpu.sh [EXPERIMENT_NAME] [DATA_PATH]
    # evaluate STGAN, if you want to evaluate distributed training result, you should enter ./train_parallel
    bash scripts/run_eval_gpu.sh [DATA_PATH] [EXPERIMENT_NAME] [DEVICE_ID] [CHECKPOINT_PATH]
    ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```bash
├── cv
    ├── STGAN
        ├── README.md                    // descriptions about STGAN
        ├── requirements.txt             // package needed
        ├── scripts
        │   ├──docker_start.sh                  // start docker container
        │   ├──run_standalone_train.sh          // train in ascend
        │   ├──run_eval.sh                      //  evaluate in ascend
        │   ├──run_distribute_train.sh          // distributed train in ascend
        │   ├──run_standalone_train_gpu.sh      // train in GPU
        │   ├──run_eval_gpu.sh                  //  evaluate in GPU
        │   ├──run_distribute_train_gpu.sh      // distributed train in GPU
        ├── src
            ├── dataset
                ├── datasets.py                 // creating dataset
                ├── celeba.py                   // processing celeba dataset
                ├── distributed_sampler.py      // distributed sampler
            ├── models
                ├── base_model.py
                ├── losses.py                   // loss models
                ├── networks.py                 // basic models of STGAN
                ├── stgan.py                    // executing procedure
            ├── utils
                ├── args.py                     // argument parser
                ├── tools.py                    // simple tools
        ├── train.py               // training script
        ├── eval.py               //  evaluation script
        ├── export.py               //  model-export script
```

### [Script Parameters](#contents)

```python
Major parameters in train.py and utils/args.py as follows:

--dataroot: The relative path from the current path to the train and evaluation datasets.
--n_epochs: Total training epochs.
--batch_size: Training batch size.
--image_size: Image size used as input to the model.
--device_target: Device where the code will be implemented. Optional value is "Ascend" or "GPU".
```

### [Training Process](#contents)

#### Training

- running on Ascend

  ```bash
  python train.py --dataroot ./dataset --experiment_name 128 > log 2>&1 &
  # or run the script
  bash scripts/run_standalone_train.sh ./dataset 128 0
  # distributed training
  bash scripts/run_distribute_train.sh ./config/rank_table_8pcs.json 128 /data/dataset
  ```

- running on GPU

  ```bash
  python train.py --dataroot ./dataset --experiment_name 128 --platform="GPU" > log 2>&1 &
  # or run the script
  bash scripts/run_standalone_train_gpu.sh ./dataset 128 0
  # distributed training
  bash scripts/run_distribute_train_gpu.sh 128 /data/dataset
  ```

  After training, the loss value will be achieved as follows:

  ```bash
  # grep "loss is " log
  epoch: 1 step: 1, loss is 2.2791853
  ...
  epoch: 1 step: 1536, loss is 1.9366643
  epoch: 1 step: 1537, loss is 1.6983616
  epoch: 1 step: 1538, loss is 1.0221305
  ...
  ```

  The model checkpoint will be saved in the output directory.

### [Evaluation Process](#contents)

#### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

  ```bash
  python eval.py --dataroot ./dataset --experiment_name 128 > eval_log.txt 2>&1 &
  # or run the script
  bash scripts/run_eval.sh ./dataset 128 0 ./ckpt/generator.ckpt
  ```

- running on GPU

  ```bash
  python eval.py --dataroot ./dataset --experiment_name 128 --platform="GPU" > eval_log.txt 2>&1 &
  # or run the script (if you want to evaluate distributed training result, you should enter ./train_parallel, then run the script)
  bash scripts/run_eval_gpu.sh ./dataset 128 0 ./ckpt/generator.ckpt
  ```

  You can view the results in the output directory, which contains a batch of result sample images.

### Model Export

```shell
python export.py --ckpt_path [CHECKPOINT_PATH] --platform [PLATFORM] --file_format[EXPORT_FORMAT]
```

If you want to infer the network on Ascend 310, `EXPORT_FORMAT` should be "MINDIR"

## Inference Process

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model. The batch_size can only be set to 1.

```bash
# Ascend310 inference
bash run_infer_310.sh [GEN_MINDIR_PATH] [DATA_PATH][NEED_PREPROCESS] [DEVICE_ID]
```

- `GEN_MINDIR_PATH` specifies path of used "MINDIR" model.
- `DATA_PATH` specifies path of dataset, dataset structure must be

    ```bash
    ├── dataroot
        ├── anno
            ├── list_attr_celeba.txt
        ├── image
            ├── 000001.jpg
            ├── ...
    ```

    above `list_attr_celeba.txt` records attributes of all images, you can refer to the list_attr_celeba.txt from dataset CelebA.
- `NEED_PREPROCESS`  means weather need preprocess or not, it's value is 'y' or 'n'. This step will process the image and label into .bin file and put them in the `process_Data` folder.
- `DEVICE_ID` is optional, it can be set by environment variable device_id, otherwise the value is zero.

### Result

Inference result is saved in `result_Files/` in current path, Inference time result is saved in `time_Result/`. The edited picture is saved as xxx.jpg format, such as `183800.jpg`.

```bash
# time result
NN inference cost average time: 9.98606 ms of infer_count 10
```

## Model Description

### Performance

#### Evaluation Performance

| Parameters                 | Ascend                                                      | GPU |
| -------------------------- | ----------------------------------------------------------- | --- |
| Model Version              | V1                                                          | V1 |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             | RTX-3090 |
| uploaded Date              | 05/07/2021 (month/day/year)                                 | 11/23/2021 (month/day/year) |
| MindSpore Version          | 1.2.0                                                       | 1.5.0rc1 |
| Dataset                    | CelebA                                                      | CelebA |
| Training Parameters        | epoch=100,  batch_size = 128                                | epoch=100, batch_size=64 |
| Optimizer                  | Adam                                                        | Adam |
| Loss Function              | Loss                                                        | Loss |
| Output                     | predict class                                               | image |
| Loss                       | 6.5523                                                      | 31.23 |
| Speed                      | 1pc: 400 ms/step;  8pcs:  143 ms/step                       | 1pc: 369 ms/step;  8pcs:  68 ms/step |
| Total time                 | 1pc: 41:36:07                                               | 1pc: 29:15:09 |
| Checkpoint for Fine tuning | 170.55M(.ckpt file)                                         | 283.76M(.ckpt file) |
| Scripts                    | [STGAN script](https://gitee.com/mindspore/models/tree/master/research/cv/STGAN) | [STGAN script](https://gitee.com/mindspore/models/tree/master/research/cv/STGAN) |

## [Model Description](#contents)

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).

## FAQ

Please refer to [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) to get some common FAQ.

- **Q**: Get "out of memory" error in PYNATIVE_MODE.
  **A**: You can set smaller batch size, e.g. 32, 16.
