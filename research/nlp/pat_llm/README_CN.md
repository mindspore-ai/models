# PAT-LLM

> 论文 **PAT-LLM: Textualized Multi-Patch Attention for Multivariate Time
> Series Forecasting with LLMs** 的 MindSpore 实现。

[English](README.md)

## 模型简介

PAT-LLM 面向强耦合工业多变量时间序列预测。模型保留完整的多变量输入，
通过两阶段 Multi-Patch Attention 建模变量间关系和跨时间片依赖，再通过
Attention-as-Prompt 将重要变量与时间区域转换成自然语言提示。提示嵌入与
重编程后的时间序列嵌入共同输入冻结的大语言模型，最后由预测头输出结果。

论文默认设置为：历史窗口 96、patch 长度 8、`d_model=64`、注意力头数 3，
预测长度为 8、18、32 和 72。实现中的自定义投影式多头注意力支持 64 无法被
3 整除的参数组合。

## 作者与单位

作者依次为：Xiaofeng Yuan、Zhiyuan Liu、Zijian Xu、Kai Wang、Yalin Wang、
Chunhua Yang、Weihua Gui、Feifan Shen 和 Lingjian Ye。

- Xiaofeng Yuan、Zhiyuan Liu、Zijian Xu、Kai Wang、Yalin Wang、Chunhua Yang
  和 Weihua Gui：中南大学自动化学院，长沙 410083。
- Lingjian Ye：湖州学院工学院，湖州 313000。
- Feifan Shen：浙大宁波理工学院信息科学与工程学院，宁波 315000。

IEEE 会籍：Xiaofeng Yuan、Yalin Wang 为 Senior Member，Chunhua Yang 为
Fellow，Kai Wang、Weihua Gui、Feifan Shen、Lingjian Ye 为 Member。

联系邮箱：yuanxf@csu.edu.cn、254612061@csu.edu.cn、254603027@csu.edu.cn、
219083@csu.edu.cn、ylwang@csu.edu.cn、ychh@csu.edu.cn、gwh@csu.edu.cn、
ffshen@nbt.edu.cn、lingjian.ye@gmail.com。

## 目录结构

```text
pat_llm/
├── default_config.yaml
├── train.py
├── eval.py
├── requirements.txt
├── scripts/
└── src/
    ├── config.py
    ├── dataset.py
    ├── engine.py
    ├── layers.py
    ├── metrics.py
    └── pat_llm.py
```

## 数据集

代码要求 CSV 至少包含 `date`、若干数值过程变量及目标列。例如：

```text
date,variable_1,...,variable_n,C5
2025-01-01 00:00:00,...,...,...
```

数据按照时间顺序划分为 70% 训练、20% 验证和 10% 测试；标准化统计量只从
训练集计算。论文使用的工业 C4/C5 数据没有放入投稿目录，禁止将保密工业
数据直接提交到公共仓库。

## 环境与训练

先安装与硬件匹配的 MindSpore 2.7.x，再执行：

```bash
pip install -r requirements.txt
bash scripts/run_standalone_train_ascend.sh \
  /absolute/path/to/dataset \
  /absolute/path/to/llama2-7b
```

运行四个论文预测长度：

```bash
bash scripts/run_all_horizons_ascend.sh \
  /absolute/path/to/dataset \
  /absolute/path/to/llama2-7b
```

评估已有权重：

```bash
bash scripts/run_eval_ascend.sh \
  /absolute/path/to/dataset \
  /absolute/path/to/llama2-7b \
  /absolute/path/to/pat_llm.ckpt
```

## 当前限制

- AaP 在 Python 侧完成张量转文字和分词，因此使用 `PYNATIVE_MODE`，暂不支持
  静态图导出。
- 尚未加入分布式训练与 Ascend 310 离线推理。
- 当前电脑不具备运行环境，不能把论文表格数值标记为该 MindSpore 实现的
  已复现结果。
- 正式发起模型仓 Pull Request 前，还需要补充 MindSpore 实测结果、运行日志、
  硬件信息和可审查的数据获取方案。

## 致谢

本工作得到国家自然科学基金项目 92267205、62573432，湖南省相关基金项目
2024RC1020、2025JJ10007，以及 CAAI-MindSpore Open Fund 项目
CAAIXSJLJJ 2025 MindSpore 10 的部分资助，并基于 OpenI 启智社区完成开发。

## 引用

论文目前为 manuscript，正式发表后需要补充卷期、页码和 DOI。

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
