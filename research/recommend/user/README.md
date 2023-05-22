# Contents

- [USER Description](#USER-description)
- [Contributions](#contributions)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Detailed Description](#detailed-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training & Evaluation Process](#training-&-evaluation-process)
        - [Training](#training)
        - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
    - [Performance](#performance)
- [Extension](#extension)
- [ModelZoo Homepage](#modelzoo-homepage)

# [USER Description](#contents)

USER is a causality enhanced framework to generate unbiased explanations. More specifically, USER firstly defines an ideal unbiased learning objective, and then derive a tractable loss for the observational data based on the inverse propensity score (IPS), where the key is a sample re-weighting strategy for equalizing the loss and ideal objective in expectation. Considering that the IPS estimated from the sparse and noisy recommendation datasets can be inaccurate, it introduces a fault tolerant mechanism by minimizing the maximum loss induced by the sample weights near the IPS. For more comprehensive modeling, USER method further analyzes and infers the potential latent confounders induced by the complex and diverse user personalities.

Paper: Zhang J, Chen X, et al. Recommendation with Causality enhanced Natural Language Explanations. In TheWebConf 2023.

# [Contributions](#contents)

The main contributions of this paper can be concluded as follows:

(1) we propose to build an unbiased explainable recommender framework based on causal inference, which, to the best of our knowledge, is the first time in the recommendation domain.

(2) To achieve the above idea, we design a framework to jointly correct the item- and feature-level biases, where we propose a fault tolerant IPS estimation strategy.

(3) For more comprehensive modeling, We further analyzes and infers the potential latent confounders induced by the complex and diverse user personalities.

# [Dataset](#contents)

We use three real-world datasets from different scenarios, including *TripAdvisor-HongKong*, *Amazon-Movie&TV* and *Yelp Challenge 2019*. All the datasets are available at this [link](https://drive.google.com/file/d/1sW3K-qdcXn27r5zWlItLAGxyvhxsAnrP/view?usp=sharing). The following is a detailed description to the datasets.

## TripAdvisor-HongKong

TripAdvisor-HongKong (TA-HK) dataset contains 169,389 interaction records of approximately 6,280 hotels made by 9,765 users. The data format is as follows:

```cpp
  UserID::ItemID::Rating::Review::FeatureID
```

- UserIDs range between 0 and 9764.
- MovieIDs range between 0 and 6279.
- Ratings are made on a 5-star scale (whole-star ratings only).
- Review are written by users.
- FeatureID indicates a representative feature extracted from review.
- To test the code  quickly, we extract a very small-scale dataset (named *small*) and submit it to this repository.

## Amazon-Movie&TV

Amazon-Movie&TV (AZ-MT) dataset belongs to E-commerce domain and it contains 235,459 interaction records of approximately 7,360 items made by 7,506 users. The data format is as follows:

```cpp
  UserID::ItemID::Rating::Review::FeatureID
```

- UserIDs range between 0 and 7505.
- MovieIDs range between 0 and 7359.
- Ratings are made on a 5-star scale (whole-star ratings only).
- Review are written by users.
- FeatureID indicates a representative feature extracted from review.

## Yelp Challenge 2019

Yelp Challenge 2019 (Yelp) dataset contains 676,433 interaction records of approximately 20,265 restaurants made by 27,147 users. The data format is as follows:

```cpp
  UserID::ItemID::Rating::Review::FeatureID
```

- UserIDs range between 0 and 27146.
- MovieIDs range between 0 and 20264.
- Ratings are made on a 5-star scale (whole-star ratings only).
- Review are written by users.
- FeatureID indicates a representative feature extracted from review.

# [Environment Requirements](#contents)

- Hardware(GPU/CPU）
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

- Install MindSpore via the official website.

- Download the codes and datasets.

- Quick run the models

  For example: run NETE_USER model

```bash
#run NETE_USER
python run_nete_user.py --dataset [dataset_name] --lr [learning_rate]
```

# [Detailed Description](#contents)

## [Script and Sample Code](#contents)

```text
├── USER
    ├── README.md                   // descriptions about USER
    ├── dataset                     // dataset
    │   ├──small                    // a small-scale set of TA-HK
    │       ├──reviews.pickle       // total records
    │       ├──train.csv            // trainging data
    │       ├──valid.csv            // validation data
    │       ├──test.csv             // testing data
    ├── init_ips                    // initialized ips (predicted using MF)
    │   ├──mf_ui_small.ckpt         // p(i|u)
    │   ├──mf_fui_small.ckpt        // p(f|u,i)
    ├── metrics                     // evaluation metrics
    │   ├──bleu.py                  // BLEU metric
    │   ├──rouge.py                 // ROUGE metric
    │   ├──metrics.py               // RMSE & MAE metric
    ├── model                       // implemented models
    │   ├──__init__.py              // import class
    │   ├──mf_ui.py                 // MF architecture for predicting p(i|u)
    │   ├──mf_fui.py                // MF architecture for predicting p(f|u,i)
    │   ├──nete_user.py             // NETE_USER architecture
    ├── run_nete_user.py            // training & evaluation script for NETE_USER
    ├── nete_small.ckpt             // trained NETE model
    ├── utils.py                    // data loader & data processor & generic functions
```

## [Script Parameters](#contents)

An introduction of key parameters for both training and evaluation.

```text
* `--emisze`                        // Embedding size.
* `--dataset`                       // The dataset name to be used.
* `--epochs`                        // Total train epochs.
* `--batch_size`                    // Training batch size.
* `--lr`                            // Training learning rate.
* `--hidden_size`                   // The hidden size for MLP.
* `--nlayers`                       // The number of hidden layers for MLP.
* `--dropout_prob`                  // The dropout probability.
* `--rating_reg`                    // The weight of rating prediction loss.
* `--pui_reg`                       // The weight of constrains.
* `--alternate_num`                 // The training number of maximization.
```

## [Training & Evaluation Process](#contents)

We unify the training process and evaluation process into one execution file, and evaluate the performance of the best learned model after the training is completed, which is also the mode usually used by most developers.
Next we will give an illustration by running NETE model to demonstrate the flow of training, evaluation and inference.

- Execute running file on command line

```bash
python run_nete_user.py --dataset=small --emsize=32 --epochs=30 --lr=0.001 --hidden_size=256 --nlayers=2 --dropout_prob=0.8 &>running.log
```

### Training

The python command above will run in the background, you can view the results through the file `running.log`. After training, you'll get a checkpoint file containing the parameters of the best model under the script folder by default. The training process will be printed as follows:

```log
# running.log
[2023-03-08 10:53:11]: NETE multi-task learning
[2023-03-08 10:53:11]: epoch 1
[2023-03-08 10:53:17]: rating loss 18.0558 | text loss 9.9066 | total loss 9.9066 on train
[2023-03-08 10:53:23]: rating loss 16.7847 | text loss 9.9074 | total loss 26.6921 on valid
[2023-03-08 10:53:23]: Save the best model./nete/nete-Mar-08-2023_10-53-23.ckpt
[2023-03-08 10:53:23]: epoch 2
[2023-03-08 10:53:24]: rating loss 18.0560 | text loss 9.9066 | total loss 9.9066 on train
[2023-03-08 10:53:26]: rating loss 16.7849 | text loss 9.9074 | total loss 26.6923 on valid
[2023-03-08 10:53:26]: Endured 1 time(s)
[2023-03-08 10:53:26]: epoch 3
[2023-03-08 10:53:27]: rating loss 18.0557 | text loss 9.9066 | total loss 9.9066 on train
[2023-03-08 10:53:28]: rating loss 16.7849 | text loss 9.9074 | total loss 26.6923 on valid
[2023-03-08 10:53:28]: Endured 2 time(s)
[2023-03-08 10:53:28]: Cannot endure it anymore | Exiting from early stop
```

The model checkpoint will be saved in the directory named by model name.

### Evaluation

We use BLEU and ROUGE to evaluate the quality of generated text and use RMSE and MAE to evaluate the performance of rating prediction. We can selectively print the required metric values.

```log
# running.log
Best epoch:1
Best test: RMSE  3.1810 | MAE  3.0462
Best test: BLEU1  0.0402 | BLEU4  0.0000
Best test: rouge_1/f_score  0.0973
Best test: rouge_1/r_score  0.0720
Best test: rouge_1/p_score  0.2008
```

## Inference Process

At the same time, we implement the inferring module, which can directly load the pre-trained model to conduct the corresponding prediction tasks, and only need to execute the `infer.py`

```log
python infer.py --dataset=small --model_path='nete_small.ckpt'
```

Information about model loading and inference is displayed in the log file：

```log
[2023-03-08 15:35:05]: Loading dataset: small
[2023-03-08 15:35:09]: Load the pre-trained model: nete_small.ckpt
[2023-03-08 15:35:09]: Run on test set:
[2023-03-08 15:35:35]: RMSE  3.1578
[2023-03-08 15:35:35]: MAE  3.0145
[2023-03-08 15:35:35]: BLEU-1  0.0375
[2023-03-08 15:35:35]: BLEU-4  0.0000
[2023-03-08 15:35:35]: rouge_1/f_score  0.0864
[2023-03-08 15:35:35]: rouge_1/r_score  0.0622
[2023-03-08 15:35:35]: rouge_1/p_score  0.1677
```

## Performance

| Parameters          | GPU                                | CPU                                       |
| ------------------- | ---------------------------------- | ----------------------------------------- |
| Model Version       | NETE                               | NETE                                      |
| Resource            | NVIDIA-SMI 510.108.03     CUDA11.6 | Intel(R) Xeon(R) Gold 5318Y CPU @ 2.10GHz |
| uploaded Date       | 03/27/2023 (month/day/year)        | 03/27/2023 (month/day/year)               |
| MindSpore Version   | 2.0.0                              | 2.0.0                                     |
| Dataset             | TA-HK (small-scale)                | TA-HK (small-scale)                       |
| Training Parameters | emsize=32, batch_size=128, lr=0.01 | emsize=32, batch_size=128, lr=0.01        |
| Optimizer           | Adam                               | Adam                                      |
| Loss Function       | MSEloss and NLLloss                | MSEloss and NLLloss                       |
| outputs             | Rating and Review                  | Rating and Review                         |
| Speed               | 1pc: 10 s/epoch                    | 1pc: 10 s/epoch                           |

# [Extension](#contents)

In order to facilitate the use of our proposed method, we will also provide an implementation version using pytorch.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).