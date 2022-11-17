# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
eval model
"""
import os
from datetime import datetime
import pickle
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from mindspore import Tensor
from mindspore import context, load_checkpoint, load_param_into_net
from src.ecapa_tdnn import ECAPA_TDNN
from src.reader import DatasetGenerator
from src.metrics import get_EER_from_scores
from src.model_utils.config import config as hparams

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

excluded_set = {2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315,
                2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329,
                2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343,
                2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357,
                2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371,
                2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385,
                2386, 2387, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981,
                2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995,
                2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009,
                3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023,
                3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037,
                4442, 4443, 4444, 4445, 4446, 4447, 4448, 4449, 4450, 4451, 4452, 4453, 4454, 4455,
                4456, 4457, 4458, 4459, 4460, 4461, 4462, 4463, 4464, 4465, 4466, 4467, 4468, 4469,
                4470, 4471, 4472, 4473, 4474, 4475, 4476, 4477, 4478, 4479, 4480, 4481, 4482, 4483,
                4484, 4485, 4486, 4487, 4488, 4489, 4490, 4491, 4492, 4639, 4640, 4641, 4642, 4643}


def evaluate(spk2emb, utt2emb, trials):
    # Evaluate EER given utterance to embedding mapping and trials file
    scores, labels = [], []
    with open(trials, "r") as f:
        for trial in f:
            trial = trial.strip()
            label, spk, test = trial.split(" ")
            spk = spk[:-4]
            if label == '1':
                labels.append(1)
            else:
                labels.append(0)
            enroll_emb = spk2emb[spk]
            test_emb = utt2emb[test[:-4]]
            scores.append(1 - cosine(enroll_emb, test_emb))

    return get_EER_from_scores(scores, labels)[0]


def evaluate2(spk2emb, utt2emb, norm_dict, params, trials):
    # Evaluate EER given utterance to embedding mapping and trials file
    train_cohort = None
    if norm_dict is not None:
        train_cohort = norm_dict
        print("train_cohort shape:", train_cohort.shape)
    positive_scores = []
    negative_scores = []
    with open(trials, "r") as f:
        lines = f.readlines()
        print_dur = 100
        for idx_c, trial in enumerate(lines):
            if idx_c % print_dur == 0:
                print(f'{datetime.now()}, processing {idx_c}/{len(lines)}')
            trial = trial.strip()
            label, spk_utt, test_utt = trial.split(" ")
            spk_utt = spk_utt[:-4]
            test_utt = test_utt[:-4]
            enrol = (spk2emb[spk_utt])
            test = (utt2emb[test_utt])
            if train_cohort is not None:
                score_e_c = cosine_similarity(
                    enrol.reshape(1, -1), train_cohort)
                score_e_c = np.squeeze(score_e_c)
                if hasattr(params, 'cohort_size'):
                    score_e_c = np.partition(
                        score_e_c, kth=-params.cohort_size
                    )[-params.cohort_size:]
                mean_e_c = np.mean(score_e_c)
                std_e_c = np.std(score_e_c)
                # Getting norm stats for test impostors
                score_t_c = cosine_similarity(
                    test.reshape(1, -1), train_cohort)
                score_t_c = np.squeeze(score_t_c)
                if hasattr(params, 'cohort_size'):
                    score_t_c = np.partition(
                        score_t_c, kth=-params.cohort_size
                    )[-params.cohort_size:]
                mean_t_c = np.mean(score_t_c)
                std_t_c = np.std(score_t_c)
            # Compute the score for the given sentence
            score = cosine_similarity(enrol.reshape(
                1, -1), test.reshape(1, -1)).item()
            # Perform score normalization
            if hasattr(params, 'score_norm'):
                if params.score_norm == "z-norm":
                    score = (score - mean_e_c) / std_e_c
                elif params.score_norm == "t-norm":
                    score = (score - mean_t_c) / std_t_c
                elif params.score_norm == "s-norm":
                    score_e = (score - mean_e_c) / std_e_c
                    score_t = (score - mean_t_c) / std_t_c
                    score = 0.5 * (score_e + score_t)
            if label == '1':
                positive_scores.append(score)
            else:
                negative_scores.append(score)
    return positive_scores, negative_scores


def EER(pos_arr, neg_arr):
    thresholds = np.sort(np.concatenate((pos_arr, neg_arr)))
    thresholds = np.unique(thresholds)

    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds = np.sort(np.concatenate((thresholds, interm_thresholds)))
    pos_scores = np.repeat(np.expand_dims(pos_arr, 0), len(thresholds), axis=0)
    pos_scores_threshold = np.transpose(pos_scores) <= thresholds
    FRR = (pos_scores_threshold.sum(0)) / pos_scores.shape[1]
    del pos_scores
    del pos_scores_threshold

    neg_scores = np.repeat(np.expand_dims(neg_arr, 0), len(thresholds), axis=0)
    neg_scores_threshold = np.transpose(neg_scores) > thresholds
    FAR = (neg_scores_threshold.sum(0)) / neg_scores.shape[1]
    del neg_scores
    del neg_scores_threshold
    # Finding the threshold for EER
    min_index = np.argmin(np.absolute(FAR - FRR))
    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    equal_error_rate = (FAR[min_index] + FRR[min_index]) / 2
    return equal_error_rate


def emb_mean(g_mean, increment, emb_dict):
    emb_dict_mean = dict()
    for utt in emb_dict:
        if increment == 0:
            g_mean = emb_dict[utt]
        else:
            weight = 1 / (increment + 1)
            g_mean = (
                1 - weight
            ) * g_mean + weight * emb_dict[utt]
        emb_dict_mean[utt] = emb_dict[utt] - g_mean
        increment += 1
        if increment % 3000 == 0:
            print('processing ', increment)
    return emb_dict_mean, g_mean, increment


def compute_embeddings(embedder, dataloader, startidx=0, dur=50000, exc_set=None):
    # Compute embeddings for utterances from dataloader
    embedder.set_train(False)
    utt2emb = dict()
    print("Compute embeddings, num to process:", len(dataloader))

    for index in range(startidx, startidx + dur):
        if index >= len(dataloader):
            print("exceed data size")
            return utt2emb
        batchdata = dataloader[index][0]
        if hparams.cut_wav:
            batchdata = batchdata[:, :301, :]
        if exc_set is not None and index in exc_set:
            continue
        if index % 1000 == 0:
            print(f"{datetime.now()}, iter-{index}")
        wavs = Tensor(batchdata)
        embs = embedder(wavs)
        utt2emb[dataloader[index][1]] = embs.asnumpy()
    return utt2emb


if __name__ == "__main__":
    context.set_context(device_id=hparams.device_id)
    in_channels = hparams.in_channels
    channels = hparams.channels
    emb_size = hparams.emb_size
    model = ECAPA_TDNN(in_channels, channels=(channels, channels, channels, channels, channels * 3),
                       lin_neurons=emb_size)

    eval_data_path = hparams.eval_data_path
    dataset_enroll = DatasetGenerator(eval_data_path, False)
    steps_per_epoch_enroll = len(dataset_enroll)
    print("size of enroll, test:", steps_per_epoch_enroll)
    model_path = os.path.join(hparams.model_path)
    print(model_path)
    param_dict = load_checkpoint(model_path)
    # load parameter to the network
    load_param_into_net(model, param_dict)
    veri_file_path = hparams.veri_file_path
    if not os.path.exists(os.path.join(hparams.npy_file_path)):
        os.makedirs(hparams.npy_file_path, exist_ok=False)
    fpath = os.path.join(hparams.npy_file_path,
                         f"enroll_dict_bleeched.npy")
    if os.path.isfile(fpath):
        print(f'find cache file:{fpath}, continue')
        enroll_dict = pickle.load(open(fpath, "rb"))
    else:
        enroll_dict = compute_embeddings(
            model, dataset_enroll, dur=len(dataset_enroll), exc_set=excluded_set)
        pickle.dump(enroll_dict, open(fpath, "wb"))
    eer = evaluate(enroll_dict, enroll_dict, veri_file_path)
    print("eer baseline:", eer)

    print("Sub mean...")
    glob_mean = Tensor([0])
    cnt = 0
    enroll_dict_mean, glob_mean, cnt = emb_mean(
        glob_mean, cnt, enroll_dict)
    enroll_dict_mean, glob_mean, cnt = emb_mean(
        glob_mean, cnt, enroll_dict)
    enroll_dict_mean, glob_mean, cnt = emb_mean(
        glob_mean, cnt, enroll_dict)
    eer = evaluate(enroll_dict_mean, enroll_dict_mean, veri_file_path)
    print("eer with sub mean:", eer)

    if hasattr(hparams, 'score_norm') and hparams.cut_wav is not True:
        train_norm_path = hparams.train_norm_path
        dataset_train = DatasetGenerator(train_norm_path, False)
        steps_per_epoch_train = len(dataset_train)
        print("steps_per_epoch_train:", steps_per_epoch_train)
        start_idx = 0
        for start in range(start_idx, len(dataset_train), 50000):
            end = start + 50000
            if end > len(dataset_train):
                end = len(dataset_train)
            print("start end:", start, end)
            fpath = os.path.join(hparams.npy_file_path,
                                 f"train_dict_{start}_{end}.npy")
            if os.path.isfile(fpath):
                print(f'find cache file:{fpath}, continue')
                continue
            train_dict = compute_embeddings(
                model, dataset_train, startidx=start, dur=50000)
            pickle.dump(train_dict, open(fpath, "wb"))

        dict_lst = []
        for idx in range(0, 5):
            dict_lst.append(pickle.load(open(os.path.join(
                hparams.npy_file_path, f"train_dict_{idx*50000}_{(idx+1)*50000}.npy"), "rb")))
        dict_lst.append(pickle.load(open(os.path.join(
            hparams.npy_file_path, f"train_dict_250000_{len(dataset_train)}.npy"), "rb")))
        train_dict = dict()
        for dicti in dict_lst:
            train_dict.update(dicti)
        print('norm data len:', len(train_dict))
        train_dict_mean, glob_mean, cnt = emb_mean(
            glob_mean, cnt, train_dict)
        items = list(train_dict_mean.values())
        train_arr = np.asarray(items)
        pos_score, neg_score = evaluate2(
            enroll_dict_mean, enroll_dict_mean, train_arr, hparams, veri_file_path)

        eer = EER(np.array(pos_score), np.array(neg_score))
        print("EER with norm:", eer)
