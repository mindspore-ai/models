# Contents

- [SiamFC Description](#SiamFC-Description)
- [Model Architecture](#SiamFC-Architecture)
- [Dataset](#SiamFC-dataset)
- [Environmental requirements](#Environmental)
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

# [SiamFC Description](#Contents)

Siamfc proposes a new full convolution twin network as the basic tracking algorithm, which is trained end-to-end on ilsvrc15 target tracking video data set. Our tracker exceeds the real-time requirement in frame rate. Although it is very simple, it achieves the best performance on multiple benchmarks.

[paper](https://arxiv.org/pdf/1606.09549.pdf)  Luca Bertinetto Jack Valmadre Jo˜ao F. Henriques Andrea Vedaldi Philip H. S. Torr
Department of Engineering Science, University of Oxford

# [Model Architecture](#Contents)

Siamfc first uses full convolution alexnet for feature extraction online and offline, and uses twin network to train the template and background respectively. On line, after getting the box of the first frame, it carries out centrrop, and then loads checkpoint to track the subsequent frames. In order to find the box, it needs to carry out a series of penalties on the score graph, Finally, the final prediction point is obtained by twice trilinear interpolation.

# [Dataset](#Contents)

used Dataset :[ILSVRC2015-VID](http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz)

- Dataset size : 85GB ,total 30 type
    - Training set: a total of 3862 videos and their corresponding frame pictures and box positions
    - Verification set: 555 videos and corresponding pictures and box locations
    - Test set: a total of 973 videos and corresponding pictures and box locations
- Data format: the image is in h*w*C format, the box position includes the coordinates of the lower left corner and the upper right corner, the format is XML, and the XML needs to be parsed

# [Environmental requirements](#Contents)

- Hardware :(Ascend/GPU)
    - Prepare Ascend/GPU processor to build hardware environment
- frame:
    - [Mindspore](https://www.mindspore.cn/install/en)
- For details, please refer to the following resources:
    - [MindSpore course](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)
- more API
    - got10k toolkit
    - opencv
    - lmdb

# [quick start](#Contents)

After installing mindspree through the official website, you can follow the following steps to train and evaluate:

- Run the python script to preprocess the data set

  ```bash
  # create dataset
  python src/create_dataset_ILSVRC.py --d data_dir --o output_dir

  # Run Python script to create LMDB
  python src/create_lmdb.py --d data_dir --o output_dir

  for example:
  data_dir = '/data/VID/ILSVRC_VID_CURATION_train'
  output_dir = '/data/VID/ILSVRC_VID_CURATION_train.lmdb'

  __Remarks:The encrypted pathname is used as the index.Therefore,you cannot change the location of the dataset
  after creating it, because you need to find the corresponding image according to the index.__
  ```

- Run on Ascend

  ```bash
  # run training example
  python train.py --device_id [DEVICE_ID] --data_path [DATA_PATH] > log.txt 2>&1 &

  # run bash training example
  bash scripts/run_standalone_train_ascend.sh [Device_ID] [Dataset_path]
  Remarks:For the training set position after preprocessing

  # more
  This example is single card training.

  # Run the python script for evaluation
  python eval.py --device_id [DEVICE_ID] --model_path [MODEL_PATH] --dataset_path [DATASET_PATH] > log.txt 2>&1 &

  # Run the bash script for evaluation
  bash scripts/run_eval_ascend.sh DEVICE_ID MODEL_PATH DATASET_PATH
  ```

- Run on GPU

  ```bash
  # run training example.
  python train.py --device_id [DEVICE_ID] --data_path [DATA_PATH] --eval_data [EVAL_DATA_PATH] > log.txt 2>&1 &

  # run bash training example.
  bash scripts/run_standalone_train_gpu.sh [Device_ID] [Dataset_path] [Eval_Data_Path]
  Remarks:For the training set position after preprocessing

  # more
  This example is single card training.

  # Run the python script for evaluation.
  python eval.py --device_id [DEVICE_ID] --model_path [MODEL_PATH] --dataset_path [DATASET_PATH] --device_target [DEVICE_TARGET] > log.txt 2>&1 &

  # Run the bash script for evaluation.
  bash scripts/run_eval_gpu.sh DEVICE_ID MODEL_PATH DATASET_PATH
  ```

# [Script description](#Contents)

## Script and sample code

```python
    ├── SiamFC
        ├── README.md                    // Notes on siamfc
        ├── scripts
        │   ├──ma-pre-start.sh          // Create environment before modelarts training
        │   ├──run_standalone_train_ascend.sh             // Single card training in ascend
        │   ├──run_distribution_ascend.sh          // Multi card distributed training in ascend
        ├── src
        │   ├──alexnet.py             // Create dataset
        │   ├──config.py              // Alexnet architecture
        │   ├──custom_transforms.py   //Data set processing
        │   ├──dataset.py            //GeneratorDataset
        │   ├──Groupconv.py        //Mindpore does not support group convolution at present. This is an alternative
        │   ├──lr_generator.py       //Dynamic learning rate
        │   ├──tracker.py           //Trace script
        │   ├──utils.py             // utils
        │   ├──create_dataset_ILSVRC.py     // Create dataset
        │   ├──create_lmdb.py               //Create LMDB
        ├── train.py               // Training script
        ├── eval.py               //  Evaluation script
```

## Script parameters

python train.py and config.py The main parameters are as follows:

- data_path：An absolutely complete path to training and evaluation data sets.
- epoch_size：Total training rounds
- batch_size：Training batch size.
- image_height：The image height is used as the model input.
- image_width：The image width is used as the model input.
- exemplar_size：Template size
- instance_size：Sample size.
- lr：Learning rate.
- frame_range：Select the frame interval of the template and sample.
- response_scale：Scaling factor of score chart.

## Training process

### Training

- Running in Ascend processor environment

  ```bash
    python train.py  --device_id=${DEVICE_ID} --data_path=${DATASET_PATH}
  ```

  # After training, the loss value is as follows

  ```bash
    grep "loss is " log
    epoch: 1 step: 1, loss is 1.14123213
    ...
    epoch: 1 step: 1536, loss is 0.5234123
    epoch: 1 step: 1537, loss is 0.4523326
    epoch: 1 step: 1538, loss is 0.6235748
   ...
  ```

  # Model checkpoints are saved in the current directory

  # After training, the loss value is as follows

  ```bash
    grep "loss is " log:
    epoch: 30 step: 1, loss is 0.12534634
    ...
    epoch: 30 step: 1560, loss is 0.2364573
    epoch: 30 step: 1561, loss is 0.156347
    epoch: 30 step: 1561, loss is 0.173423
  ```

- Running in GPU processor environment

  ```bash
    python train.py  --device_id=${DEVICE_ID} --data_path=${DATASET_PATH} --device_target "GPU"
  ```

  # After training, the loss value is as follows

  ```bash
    grep "loss is " log
    epoch: 1 step: 6650, loss is 0.3076
    ...
    epoch: 2 step: 6650, loss is 0.1665
    epoch: 3 step: 6650, loss is 0.44
    epoch: 4 step: 6650, loss is 0.3164
   ...
  ```

  # Model checkpoints are saved in the current directory

  # After training, the loss value is as follows

  ```bash
    grep "loss is " log:
    epoch: 20 step: 6650, loss is 0.1871
    ...
    epoch: 48 step: 6650, loss is 0.1437
    epoch: 49 step: 6650, loss is 0.1476
    epoch: 50 step: 6650, loss is 0.1718
  ```

## Evaluation process

Check the checkpoint path used for evaluation before running the following command.

- Running in Ascend processor environment

  ```bash
    python eval.py  --device_id=${DEVICE_ID} --model_path=${MODEL_PATH}
  ```

    The results were as follows:

  ```bash
    SiamFC_159_50_6650.ckpt -prec_score:0.777 -succ_score:0.589 _succ_rate:0.754
  ```

- Running in GPU processor environment

  ```bash
    python eval.py  --device_id=${DEVICE_ID} --model_path=${MODEL_PATH} --device_target "GPU"
  ```

    The results were as follows:

  ```bash
    SiamFC-50_6650.ckpt -prec_score:0.791 -succ_score:0.591 -succ_rate:0.761
  ```

# [Model description](#Contents)

## performance

### Evaluate performance

|parameter   | Ascend        | GPU                                                                  |
| -------------------------- | ---------------------------------------------- |---------------------|
|resources     | Ascend 910；CPU 2.60GHz, 192core；memory：755G | GPU GTX3090： CPU 2.90GHz, 64core；memory：251G |
|Upload date   |2021.5.20         | 2021.11.19        |
|mindspore version   |mindspore1.2.0     |    mindspore 1.5        |
|training parameter | epoch=50,step=6650,batch_size=8,lr_init=1e-2,lr_endl=1e-5   | epoch=50,step=6650,batch_size=8,lr_init=3.5e-2,lr_endl=1e-5 |
|optimizer     |SGD optimizer，momentum=0.0,weight_decay=0.0    |  SGD optimizer，momentum=0.0,weight_decay=0.0   |
|loss function     |BCEWithLogits   |     BCEWithLogits     |
|training speed    | epoch time：285693.557 ms per step time :42.961 ms |  epoch time：122536.251 per step time :18.427 ms   |
|total time        |about 5 hours    |    about 1.5 hours
|Script URL        |[SiamFC](https://gitee.com/mindspore/models/tree/master/research/cv/SiamFC)  |  [SiamFC](https://gitee.com/mindspore/models/tree/master/research/cv/SiamFC)|
|Random number seed         |set_seed = 1234     |  set_seed = 1234     |
