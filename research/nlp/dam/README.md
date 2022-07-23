# Contents

- [DAM Description](#DAM-description)
- [Model Architecture](#Model-Architecture)
- [Dataset](#Dataset)
    - [Download and unzip the dataset](#Download-and-unzip-the-dataset)
    - [Prepare the mindrecord file](#Prepare-the-mindrecord-file)
- [Environmental Requirements](#Environmental-requirements)
- [Script Description](#Script-Description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Training Results](#Training-Result)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
        - [Evaluation Result](#evaluation-result)
- [Model Description](#model-description)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DAM Description](#Contents)

Human generates responses relying on semantic and functional dependencies, including coreference relation, among dialogue elements and their context. In this paper, we investigate matching a response with its multi-turn context using dependency information based entirely on attention. Our solution is inspired by the recently proposed Transformer in machine translation (V aswani et al., 2017) and we extend the attention mechanism in two ways. First, we construct representations of text segments at different granularities solely with stacked self-attention. Second, we try to extract the truly matched segment pairs with attention across the context and response. We jointly introduce those two kinds of attention in one uniform neural network. Experiments on two large-scale multi-turn response selection tasks show that our proposed model significantly outperforms the state-of-the-art models.

[Paper](https://aclanthology.org/P18-1103.pdf): Zhou, Xiangyang , et al. "Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network." Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) 2018.

# [Model Architecture](#Contents)

The pyramid pooling module fuses features under four different pyramid scales.For maintaining a reasonable gap in representation，the module is a four-level one with bin sizes of 1×1, 2×2, 3×3 and 6×6 respectively.

# [Dataset](#Content)

Train DAM Dataset used: [Ubuntu Corpus and Douban Corpus](https://pan.baidu.com/s/1hakfuuwdS8xl7NyxlWzRiQ "data")

- Note: The package contains the Ubuntu and Douban datasets
- Note: Data will be processed in src/data2mindrecord.py

Both the Ubuntu and Douban training set contains 0.5 million multiturn contexts, and each context has one positive response and one negative response. Both valiation and testing sets of Ubuntu Corpus have 50k contexts, where each context is provided with one positive response and nine negative replies. The validation set of Douban corpus contains 50K instances, and the test set contains 10K instances.

## Download and unzip the dataset

```bash
├── data
  ├── douban
  └── ubuntu
```

## Prepare the mindrecord file

Using `data2mindrecord.py` to preprocess the dataset as follows.

```bash
├─ PATH_TO_OUTPUT_MINDRECORD
  ├─ douban
  │  ├─ data_train.mindrecord
  │  ├─ data_train.mindrecord.db
  │  ├─ data_val.mindrecord
  │  ├─ data_val.mindrecord.db
  │  ├─ data_test.mindrecord
  │  └─ data_test.mindrecord.db
  └─ ubuntu
     ├─ data_train.mindrecord
     ├─ data_train.mindrecord.db
     ├─ data_val.mindrecord
     ├─ data_val.mindrecord.db
     ├─ data_test.mindrecord
     └─ data_test.mindrecord.db
```

- Training set

```bash
python data2mindrecord.py --device_id=[DEVICE_ID] \
                          --data_name=ubuntu \
                          --data_root=[DATA_ROOT] \
                          --raw_data=data.pkl \
                          --mode=train \
                          --print_data=True
```

- Validation set

```bash
python data2mindrecord.py --device_id=[DEVICE_ID] \
                          --data_name=ubuntu \
                          --data_root=[DATA_ROOT] \
                          --raw_data=data.pkl \
                          --mode=val \
                          --print_data=True
```

- Testing set

```bash
python data2mindrecord.py --device_id=[DEVICE_ID] \
                          --data_name=ubuntu \
                          --data_root=[DATA_ROOT] \
                          --raw_data=data.pkl \
                          --mode=test \
                          --print_data=True
```

# [Environmental requirements](#Contents)

- Hardware (Ascend)
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#Content)

## [Script and Sample Code](#contents)

```bash
├─ dam
  ├─ README.md                      # descriptions about DAM
  ├─ requirements.txt
  ├─ data2mindrecord.py             # convert dataset to mindrecord
  ├─ scripts
  │  ├─ run_distribute_train.sh     # launch distributed ascend training(8 pcs)
  │  ├─ run_train.sh                # launch ascend training(1 pcs)
  │  └─ run_eval.sh                 # launch ascend eval
  ├─ src
  │  ├─ __init__.py                 # init file
  │  ├─ callback.py                 # define callback function
  │  ├─ config.py                   # config file
  │  ├─ douban_evaluation.py        # evaluation function of Douban data
  │  ├─ dynamic_lr.py               # generative learning rate
  │  ├─ ubuntu_evaluation.py        # evaluation function of Ubuntu data
  │  ├─ layers.py                   # network module of DAM
  │  ├─ metric.py                   # verify the model
  │  ├─ net.py                      # DAM network
  │  └─ utils.py                    # network utils file
  ├─ train.py                       # train DAM script
  ├─ eval.py                        # evaluate script
  └─ export.py                      # export mindir script
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in `config.py`.

Major hyper-parameters are as follows:

```bash
"seed": 1                                   # random seed
"parallel": False                           # Whether to use parallel mode for training.
"do_eval": True                             # infer while training
"max_turn_num": 9                           # Maximum number of utterances in context.
"max_turn_len": 50                          # Maximum length of setences in turns.
"stack_num": 5                              # The number of stacked attentive modules in network.
"attention_type": "dot"                     # attention type
"is_emb_init":False                         # Whether to use a pre-trained embedding file.
"vocab_size": 434512                        # The size of vocabulary. --172130 for douban data--
"emb_size": 200                             # The dimension of word embedding.
"channel1_dim": 32                          # he channels' number of the 1st conv3d layer's output. --16 for douban data--
"channel2_dim": 16                          # The channels' number of the 2nd conv3d layer's output.
"is_mask": True                             # use mask
"is_layer_norm": True                       # use layer normal
"is_positional": False                      # use positional code
"drop_attention": None                      # attention module use dropout
"batch_size": 256                           # Batch size for training.
"eval_batch_size": 200                      # Batch size for testing.
"learning_rate": 1e-3                       # Learning rate used to train.
"decay_rate": 0.9                           # learning rate decay rate
"decay_steps": 405                          # learning rate decay step
"loss_scale": 1                             # loss scale
"epoch_size": 2                             # Number of pass for training.
"modelArts"': False                         # whether training on modelArts
```

## [Training Process](#contents)

### Training

- launch distributed training on ModelArts (If you want to run in ModelArts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/)).

  ModelArts parameters settings：

```bash
    --train_url=/PATH/TO/OUTPUT_DIR \
    --data_url=/PATH/TO/MINDRECORD  \
    --modelArts_mode=True  \
    --model_name=DAM_ubuntu  \
    --epoch_sie=2 \
    --batch_size=256 \
    --learning_rate=0.001 \
    --decay_steps=405 \
```

- launch training on Ascend with single device

```bash
bash scripts/run_train.sh [DEVICE_ID] [MODEL_NAME ] [BATCH_SIZE] [EPOCH_SIZE] [LEARNING_RATE] [DECAY_STEPS] [DATA_ROOT] [OUTPUT_PATH]
```

- launch distributed trainning on Ascend (8pcs)

```bash
bash scripts/run_distribute_train.sh [RANK_SIZE] [RANK_TABLE_FILE] [MODEL_NAME ] [BATCH_SIZE] [EPOCH_SIZE] [LEARNING_RATE] [DECAY_STEPS] [DATA_ROOT]
```

### Training Result

The python command above will run in the background, you can view the results through the file `eval.log`.

After training, you'll get some checkpoint files under `./output/ubuntu/` folder by default. The loss value are saved in `loss.log` file.

```bash
# training result(8p)-ubuntu
step time 197.62039184570312
step time 197.71790504455566
epoch: 1 step: 3853 global_step: 3853, loss is 0.21962138
epoch: 1 step: 3853 global_step: 3853, loss is 0.21994969
epoch: 1 step: 3853 global_step: 3853, loss is 0.32234603
epoch: 1 step: 3853 global_step: 3853, loss is 0.37376451
epoch: 1 step: 3853 global_step: 3853, loss is 0.5122621
epoch: 1 step: 3853 global_step: 3853, loss is 0.20732686
step time 197.07393646240234
step time 197.10779190063477
step time 197.42536544799805
step time 197.4952220916748
step time 197.47066497802734
epoch: 1 step: 3854 global_step: 3854, loss is 0.2575438
epoch: 1 step: 3854 global_step: 3854, loss is 0.29517844
epoch: 1 step: 3854 global_step: 3854, loss is 0.17604485
epoch: 1 step: 3854 global_step: 3854, loss is 0.22759959
epoch: 1 step: 3854 global_step: 3854, loss is 0.43964553
step time 197.8461742401123
step time 198.20117950439453
step time 198.29559326171875
epoch: 1 step: 3854 global_step: 3854, loss is 0.2520399
epoch: 1 step: 3854 global_step: 3854, loss is 0.3967452
epoch: 1 step: 3854 global_step: 3854, loss is 0.3976175
```

## [Evaluation Process](#contents)

### Evaluation

- Evaluation on dataset when running on Ascend.

  Before running the command below, please check the checkpoint path used for evaluation.

```bash
bash scripts/run_eval.sh [DEVICE_ID] [MODEL_NAME ] [EVAL_BATCH_SIZE] [DATA_ROOT] [CKPT_PATH] [CKPT_NAME] [OUTPUT_PATH]
```

### Evaluation Result

The results were as follows:

```bash
Ubuntu:R2@1/R10@1/R10@2/R10@5  0.937/0.765/0.870/0.968
Douban:MAP/MRR/P@1/R10@1/R10@2/R10@5  0.550/0.601/0.427/0.254/0.410/0.757
````

## Model Export

```shell
python export.py --model_name [MODEL_NAME] --ckpt_path [CKPT_PATH] --ckpt_name [CKPT_NAME] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT] --batch_size [BATCH_SIZE]
```

`MODEL_NAME` can be in ["DAM_ubuntu", "DAM_douban"]
`EXPORT_FORMAT` can be in ["AIR", "MINDIR"]
`BATCH_SIZE` default value of ubuntu is 200， default value of douban is 256.

## Inference Process

### Usage

Before performing inference, the model file must be exported by export script on the Ascend910 environment.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [MODEL_NAME] [DATA_FILE_PATH] [EVAL_BATCH_SIZE] [NEED_PREPROCESS] [DEVICE_ID]（optional）
```

- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```log
Ubuntu: accuracy: (0.93736, 0.75458, 0.87044, 0.96802)
Douban: accuracy: (0.5479, 0.5967, 0.4227, 0.2504, 0.4167, 0.7552)
```

# [Model Description](#contents)

## [Performance](#contents)

### Distributed Training Performance

| Parameters          | DAM                                                          | DAM                                                          |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Resource            | Ascend 910 * 8; CPU 2.60GHz, 192cores; Memory 755G           | Ascend 910 * 8; CPU 2.60GHz, 192cores; Memory 755G           |
| MindSpore Version   | 1.3.0                                                        | 1.3.0                                                        |
| Dataset             | Ubuntu                                                       | Douban                                                       |
| Training Parameters | epoch=2, batch_size = 256, learning_rate=1e-3, decay_steps=405 | epoch=2, batch_size = 256, learning_rate=1e-3, decay_steps=405 |
| Optimizer           | Adam                                                         | Adam                                                         |
| Loss Function       | SigmoidCrossEntropyWithLogits                                | SigmoidCrossEntropyWithLogits                                |
| Outputs             | score                                                        | score                                                        |
| Accuracy            | 0.937/0.765/0.870/0.968 (Ubuntu)                             | 0.550/0.601/0.427/0.254/0.410/0.757 (Douban)                 |
| Speed               | 197.425 ms/step (8pcs);                                      | 197.425 ms/step (8pcs);                                      |
| Total time          | 4h52min (8pcs)                                               | 13h4m (8pcs)                                                 |
| Checkpoint          | 1010.56 M (.ckpt file)                                       | 410 M (.ckpt file)                                           |

# [Description of Random Situation](#Content)

The random seed in `train.py`.

# [ModelZoo Homepage](#Content)

Please visit the official website [homepage](https://gitee.com/mindspore/models).