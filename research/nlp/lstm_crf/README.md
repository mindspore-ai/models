# Contents

- [Contents](#contents)
- [LSTM Description](#lstm-crf-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [LSTM-CRF Description](#contents)

This example is for LSTM_CRF model training and evaluation.

[Paper](https://arxiv.org/abs/1508.01991):  Zhiheng Huang, Wei Xu, Kai Yu. [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991).

# [Model Architecture](#contents)

LSTM contains embeding, encoder and decoder modules. Encoder module consists of LSTM layer. Decoder module consists of fully-connection layer. Take the full-connection layer output as the input of CRF.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

- CoNLL2000 for training evaluation.[CoNLL 2000 chunking](https://www.clips.uantwerpen.be/conll2000/chunking/)
- GloVe: Vector representations for words.[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

# [Environment Requirements](#contents)

- Hardware（CPU/Ascend）
    - Prepare hardware environment with Ascend or CPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

- bulid_data

  ```bash
  # run bulid_data example
  bash run_bulid_data.sh ..data/CoNLL2000 ../data/glove
  ```

- running on Ascend

  ```bash
  # run training example
  bash run_train_ascend.sh 0 ..data/CoNLL2000

  # run evaluation example
  bash run_eval_ascend.sh 0 ..data/CoNll200 lstm-20_446.ckpt
  ```

- running on CPU

  ```bash
  # run training example
  bash run_train_cpu.sh ..data/CoNLL2000

  # run evaluation example
  bash run_eval_cpu.sh ..data/CoNll200 lstm-20_446.ckpt
  ```

# [Script Description](#contents)

```shell
.
├── lstm_crf
    ├── README.md               # descriptions about LSTM
    ├── script
    │   ├── run_bulid_data.sh   # shell script for create data
    │   ├── run_eval_ascend.sh  # shell script for evaluation on Ascend
    │   ├── run_eval_cpu.sh     # shell script for evaluation on CPU
    │   ├── run_train_ascend.sh # shell script for training on Ascend
    │   ├── run_train_cpu.sh    # shell script for training on CPU
    ├── src
    │   ├── lstm.py             # lstm model
    │   ├── lstm_crf.py         # lstm_crf model
    │   ├── dataset.py          # dataset preprocess
    │   ├── imdb.py             # imdb dataset read script
    │   ├── util.py             # utils script
    │   └─model_utils
    │     ├── config.py               # Processing configuration parameters
    │     ├── device_adapter.py       # Get cloud ID
    │     ├── local_adapter.py        # Get local ID
    │     ├── moxing_adapter.py       # Parameter processing
    ├── default_config.yaml           # Training parameter profile(cpu/ascend)
    ├── eval.py                 # evaluation script on CPU and Ascend
    └── train.py                # training script on CPU and Ascend
    └── export.py                # export script on CPU and Ascend
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | LSTM_CRF (Ascend)          | LSTM_CRF (CPU)             |
| -------------------------- | -------------------------- | -------------------------- |
| Resource                   | Ascend 910                 | windows10 i7-9700-32G      |
| uploaded Date              | 12/28/2021 (month/day/year)| 12/28/2021 (month/day/year)|
| MindSpore Version          | 1.6.0                      | 1.6.0                      |
| Dataset                    | CoNLL2000                  | CoNLL2000                  |
| Training Parameters        | epoch=20, batch_size=20    | epoch=20, batch_size=20    |
| Optimizer                  | AdamWeightDecay            |AdamWeightDecay             |
| Loss Function              | CRF LOSS                   | CRF LOSS                   |
| Checkpoint for inference   | 64.9M (.ckpt file)         | 64.9M (.ckpt file)         |
| Scripts                    | [lstm script](https://gitee.com/mindspore/models/tree/master/research/nlp/lstm_crf) | [lstm script](https://gitee.com/mindspore/models/tree/master/research/nlp/lstm_crf) |

### Evaluation Performance

| Parameters          | LSTM_CRF (Ascend)            | LSTMLSTM_CRF (CPU)           |
| ------------------- | ---------------------------- | ---------------------------- |
| Resource            | Ascend 910                   | Ubuntu X86-i7-8565U-16GB     |
| uploaded Date       | 12/28/2021 (month/day/year)  | 12/28/2021 (month/day/year)  |
| MindSpore Version   | 1.6.0                        | 1.6.0                        |
| Dataset             | CoNLL2000                    | CoNLL2000                    |
| batch_size          | 20                           | 20                           |
| F1                  | 93.48%                       | 93.13%                       |

# [Description of Random Situation](#contents)

There are three random situations:

- Shuffle of the dataset.
- Initialization of some model weights.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
