# Contents

- [Contents](#contents)
- [SRC Description](#SRC-description)
- [Dataset](#dataset)
- [Simulator](#simulator)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [SRC Description](#contents)

we propose a novel framework named Set-to-Sequence Ranking-based Concept-aware Learning Path Recommendation (SRC), which formulates the recommendation task under a set-to-sequence paradigm. Specifically, we first design a concept-aware encoder module which can capture the correlations among the input learning concepts. 
The outputs are then fed into a decoder module that sequentially generates a path through an attention mechanism that handles correlations between the learning and target concepts. Our recommendation policy is optimized by policy gradient. 
In addition, we also introduce an auxiliary module based on knowledge tracing to enhance the model’s stability by evaluating students’ learning effects on learning concepts

Set-to-Sequence Ranking-based Concept-aware Learning Path Recommendation

AAAI2023

# [Dataset](#contents)

- [assist09](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data)
- [junyi](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy)

# [Simulator](#contents)

- [DKT](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)
- [CoKT](https://dl.acm.org/doi/10.1145/3488560.3498374)

# [Environment Requirements](#contents)

- Hardware（CPU and GPU）
    - Prepare hardware environment with CPU processor and GPU of Nvidia.
- Framework
    - [MindSpore-1.9.0](https://www.mindspore.cn/install/en)
- Requirements
  - numpy
  - tqdm
  - longling
  - mindspore==1.9.0
  - scikit-learn
  - elasticsearch
  
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on GPU

  ```shell
  # Build Simulator first
  python trainDKT.py -d dataset -m model
  # After installing MindSpore via the official website, you can start training and evaluation as follows:
  python trainSRC.py -d dataset -p path_type 
  ```
# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text 
.
└─SRC
  ├─README.md             # descriptions of SRC
  ├─KTScripts             # KT module for simulator, SRC, MPC, etc.
    ├─__init__.py
    ├─BackModels.py       # Some basic model
    ├─DataLoader.py       # Dataset file
    ├─options.py          # Parameter option
    ├─PredictModel.py     # The framework of the prediction model
    ├─utils.py            # Some utility functions
  ├─Scripts               # Various simulators, baseline models, SRC.
    ├─__init__.py
    ├─Agent               
      ├─__init__.py
      ├─MPC.py            # MPC model
      ├─SRC.py            # SRC model
      ├─utils.py          # Some utility functions
    ├─Envs               
      ├─__init__.py
      ├─KES
        ├─Env.py          # Simulator based on DKT
        ├─EnvCo.py        # Simulator based on CoKT
        ├─utils.py        # Some utility functions
    ├─Optimizer.py        # Definition of ModelWithLoss and ModelWithOptimizer
    ├─options.py          # Parameter option
    ├─utils.py            # Some utility functions
  ├─generate_data.py      # generate the datasets required for CoKT
  ├─testRule.py           # The performance of three baselines -- random, rule-based, and GRU4Rec -- was tested.
  ├─trainDKT.py           # Train simulator, GRU4Rec 
  ├─trainMPC.py           # Train and validate MPC
  ├─trainSRC.py           # Train and validate SRC
```
## [Script Parameters](#contents)

- Parameters of trainDKT.py

See [KTScripts/options.py](./KTScripts/options.py) for detailed parameters and explanations

- Parameters of trainSRC.py and trainMPC.py

See [Scripts/options.py](./Scripts/options.py) for detailed parameters and explanations


# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters          | GPU                                                                                                                         |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Resource            | AMD Ryzen 2990WX 32-Core Processor;256G Memory;NVIDIA GeForce 2080Ti                                                        |
| uploaded Date       | 01/15/2023 (month/day/year)                                                                                                 |
| MindSpore Version   | 1.9.0                                                                                                                       |
| Dataset             | assist09, junyi                                                                                                             |
| Simulator           | DKT, CoKT                                                                                                                   |
| Training Parameters | epoch=10, batch_size=256, lr=1e-3                                                                                           |
| Optimizer           | Adam                                                                                                                        |
| Loss Function       | Policy Gradient                                                                                                             |
| Outputs             | Reward                                                                                                                      |
| Results             | Based on simulator training, it has strong randomness. The comparison of different baselines can be referred to the paper.  |
| Per Step Time       | 54.97 ms                                                                                                                    |

### Inference Performance

| Parameters        | GPU                                                                                                                         |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Resource          | AMD Ryzen 2990WX 32-Core Processor;256G Memory;NVIDIA GeForce 2080Ti                                                        |
| uploaded Date     | 01/15/2023 (month/day/year)                                                                                                 |
| MindSpore Version | 1.9.10                                                                                                                      |
| Dataset           | assist09, junyi                                                                                                             |
| Simulator         | DKT, CoKT                                                                                                                   |
| Outputs           | Reward                                                                                                                      |
| Results           | Based on simulator training, it has strong randomness. The comparison of different baselines can be referred to the paper.  |
| Per Step Time     | 40.61 ms                                                                                                                    |

# [Description of Random Situation](#contents)

- Simulator Training.
- Random initialization of model weights.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models)