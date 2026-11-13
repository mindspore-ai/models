# PAT-LLM

> MindSpore implementation of **PAT-LLM: Textualized Multi-Patch Attention
> for Multivariate Time Series Forecasting with LLMs**.

[中文](README_CN.md)

## Contents

- [Overview](#overview)
- [Authors and Affiliations](#authors-and-affiliations)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment](#environment)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Overview

PAT-LLM forecasts strongly coupled industrial multivariate time series using
a frozen large language model. It keeps the complete multivariate input window
and combines two learned structural components:

1. **Multi-Patch Attention (MPA)** aggregates global cross-variable
   interactions and then refines patch-level temporal dependencies.
2. **Attention-as-Prompt (AaP)** converts influential variables and important
   temporal regions from the attention maps into natural-language context.

The prompts and reprogrammed time-series embeddings are concatenated and sent
to a frozen LLM. The patch encoder, MPA, prototype mapping, reprogramming layer,
and forecast head are trainable.

## Authors and Affiliations

**Xiaofeng Yuan**, **Zhiyuan Liu**, **Zijian Xu**, **Kai Wang**,
**Yalin Wang**, **Chunhua Yang**, **Weihua Gui**, **Feifan Shen**, and
**Lingjian Ye**.

- Xiaofeng Yuan, Zhiyuan Liu, Zijian Xu, Kai Wang, Yalin Wang, Chunhua Yang,
  and Weihua Gui are with the School of Automation, Central South University,
  Changsha 410083, China.
- Lingjian Ye is with the School of Engineering, Huzhou University,
  Huzhou 313000, China.
- Feifan Shen is with the School of Information Science and Engineering,
  NingboTech University, Ningbo 315000, China.

IEEE membership: Xiaofeng Yuan and Yalin Wang are Senior Members; Chunhua Yang
is a Fellow; Kai Wang, Weihua Gui, Feifan Shen, and Lingjian Ye are Members.

Contact: yuanxf@csu.edu.cn, 254612061@csu.edu.cn, 254603027@csu.edu.cn,
219083@csu.edu.cn, ylwang@csu.edu.cn, ychh@csu.edu.cn, gwh@csu.edu.cn,
ffshen@nbt.edu.cn, and lingjian.ye@gmail.com.

## Model Architecture

```text
multivariate history
        |
patch embedding
        |
two-stage Multi-Patch Attention
        |-----------------------------|
        |                             |
reprogramming tokens          Attention-as-Prompt
        |                             |
        |                    tokenizer + embeddings
        |                             |
        +-------------+---------------+
                      |
                  frozen LLM
                      |
                forecast head
```

The manuscript uses a 96-step look-back window, patch length 8,
`d_model=64`, three attention heads, and forecast horizons 8, 18, 32, and 72.
The custom projected multi-head attention supports the non-divisible
`64 / 3` setting by projecting each head to `floor(d_model / n_heads)`.

## Dataset

The manuscript evaluates two industrial hydrocracking datasets for C4 and C5
quality forecasting. These datasets are not distributed in this directory.
Prepare each dataset as a CSV file with this schema:

```text
date,process_variable_1,...,process_variable_n,C5
2025-01-01 00:00:00,...,...,...
```

- `date` is required and must be parseable by pandas.
- The target name is selected with `target` in the YAML configuration.
- Other numeric columns are treated as process variables.
- Missing numeric values are interpolated, forward-filled, then back-filled.
- Samples are split chronologically into 70% train, 20% validation, and 10%
  test partitions. Scaling statistics are fitted only on the training data.

For C4 forecasting, change `data_path` and `target` to the corresponding CSV
and target column. Do not commit confidential industrial data to the model
repository.

## Environment

The code targets Python 3.10/3.11, MindSpore 2.7.x, and MindNLP 0.6.x.
Install the MindSpore package suitable for the target hardware, then install
the remaining dependencies:

```bash
pip install -r requirements.txt
```

Download or convert a MindNLP-compatible Llama 2 model locally. The default
configuration disables remote downloads and remote model code.

## Quick Start

Run from the `research/nlp/pat_llm` directory.

### Standalone training on Ascend

```bash
bash scripts/run_standalone_train_ascend.sh \
  /absolute/path/to/dataset \
  /absolute/path/to/llama2-7b
```

Optional third and fourth arguments select the relative CSV path and target:

```bash
bash scripts/run_standalone_train_ascend.sh \
  /absolute/path/to/dataset \
  /absolute/path/to/llama2-7b \
  C4/C4.csv C4
```

### Four manuscript horizons

```bash
bash scripts/run_all_horizons_ascend.sh \
  /absolute/path/to/dataset \
  /absolute/path/to/llama2-7b
```

### Evaluation

```bash
bash scripts/run_eval_ascend.sh \
  /absolute/path/to/dataset \
  /absolute/path/to/llama2-7b \
  /absolute/path/to/pat_llm.ckpt
```

The evaluation command writes `metrics.json` and `predictions.npz` under
`output_dir`.

## Configuration

Defaults are stored in `default_config.yaml`. Override individual options
without editing the file:

```bash
python train.py --config_path default_config.yaml \
  --set device_target=Ascend \
  --set device_id=0 \
  --set llm_model_path=/models/llama2-7b \
  --set pred_len=32
```

`features=MS` trains from all variables and evaluates the final target
channel. `features=M` forecasts all channels, while `features=S` uses only the
target series.

## Limitations

- Attention-as-Prompt performs tensor-to-text conversion and tokenisation on
  the Python host. The network therefore uses `PYNATIVE_MODE` and currently
  does not support MindSpore static graph export.
- The LLM backbone and tokenizer must be compatible with MindNLP.
- Distributed training and Ascend 310 offline inference are not yet included.
- Before a model-zoo pull request, add reproducible MindSpore performance,
  execution logs, hardware details, and a public or reviewable dataset access
  procedure.

## Acknowledgements

This work was supported in part by the National Natural Science Foundation of
China under Grants 92267205 and 62573432; in part by the Foundation of Hunan,
China under Grants 2024RC1020 and 2025JJ10007; and in part by the CAAI-MindSpore
Open Fund under Grant CAAIXSJLJJ 2025 MindSpore 10. The work was developed on
the OpenI Community.

## Citation

PAT-LLM is currently provided as a manuscript. Volume, issue, page, and DOI
information should be added after publication.

```bibtex
@article{yuan2026patllm,
  title={{PAT-LLM}: Textualized Multi-Patch Attention for Multivariate
         Time Series Forecasting with LLMs},
  author={Yuan, Xiaofeng and Liu, Zhiyuan and Xu, Zijian and Wang, Kai and
          Wang, Yalin and Yang, Chunhua and Gui, Weihua and Shen, Feifan and
          Ye, Lingjian},
  journal={IEEE Transactions on Industrial Informatics},
  year={2026},
  note={Manuscript}
}
```

The implementation also builds on the LLM reprogramming paradigm from
Time-LLM:

```bibtex
@inproceedings{jin2023time,
  title={{Time-LLM}: Time Series Forecasting by Reprogramming Large Language Models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and
          Zhang, James Y. and Shi, Xiaoming and Chen, Pin-Yu and
          Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
