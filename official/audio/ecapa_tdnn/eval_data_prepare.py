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

"""Recipe for prepare evaluation data.
"""
import os
import sys
import logging
import datetime
import numpy as np
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from src.voxceleb_prepare import prepare_voxceleb
from src.reader import DatasetGenerator

# bad utterances
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

def compute_feat_loop(data_loader, save_dir):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """

    print("compute_feat_loop")
    with open(os.path.join(save_dir, "fea.lst"), 'w') as fea_fp, \
        open(os.path.join(save_dir, "label.lst"), 'w') as label_fp:
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(params["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig
            ct = datetime.datetime.now()
            ts = ct.timestamp()
            wavs, lens = wavs.to(params["device"]), lens.to(params["device"])
            feats = params["compute_features"](wavs)
            feats = params["mean_var_norm"](feats, lens)
            feat_mvn = feats
            save_fea_name = str(ts) + "_fea_mvn.npy"
            save_label_name = str(ts) + "_label.npy"
            np.save(os.path.join(save_dir, save_fea_name), feat_mvn.cpu().numpy())
            np.save(os.path.join(save_dir, save_label_name), seg_ids)
            fea_fp.write(save_fea_name + "\n")
            label_fp.write(save_label_name + "\n")


def dataio_prep(hparams):
    "Creates the dataloaders and their data processing pipelines."
    data_folder = hparams["data_folder"]
    # 1. Declarations:
    # Train data (used for normalization)
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"], replacements={"data_root": data_folder},
    )
    train_data = train_data.filtered_sorted(
        sort_key="duration", select_n=hparams["n_train_snts"]
    )

    # Enrol data
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["enrol_data"], replacements={"data_root": data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, enrol_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, _ = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # 4 Create dataloaders
    train_dataloader_ = sb.dataio.dataloader.make_dataloader(
        train_data, **hparams["train_dataloader_opts"]
    )
    enrol_dataloader_ = sb.dataio.dataloader.make_dataloader(
        enrol_data, **hparams["enrol_dataloader_opts"]
    )
    test_dataloader_ = sb.dataio.dataloader.make_dataloader(
        test_data, **hparams["test_dataloader_opts"]
    )

    return train_dataloader_, enrol_dataloader_, test_dataloader_


if __name__ == "__main__":
    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    print('param, overrides:', params_file, type(overrides))
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin)

    # Download verification list (to exclude verification sentences from train)
    veri_file_path = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file"])
    )
    download_file(params["verification_file"], veri_file_path)

    # Prepare data from dev of Voxceleb1
    prepare_voxceleb(
        data_folder=params["data_folder"],
        save_folder=params["save_folder"],
        verification_pairs_file=veri_file_path,
        splits=("train", "dev", "test"),
        split_ratio=(90, 10),
        seg_dur=3.0,
        source=params["voxceleb_source"]
        if "voxceleb_source" in params
        else None,
        skip_prep=params["skip_prep"]
    )

    if not os.path.exists(params["feat_eval_folder"]):
        os.makedirs(params["feat_eval_folder"], exist_ok=False)

    if not os.path.exists(params["feat_norm_folder"]):
        os.makedirs(params["feat_norm_folder"], exist_ok=False)

    # here we create the datasets
    train_dataloader, enrol_dataloader, test_dataloader = dataio_prep(params)

    compute_feat_loop(enrol_dataloader, params["feat_eval_folder"])
    compute_feat_loop(train_dataloader, params["feat_norm_folder"])

    # test eval data & generate bleeched verification file
    eval_dataset = DatasetGenerator(params["feat_eval_folder"])
    excluded_utt_set = set()
    for idx in range(len(eval_dataset)):
        if idx in excluded_set:
            excluded_utt_set.add(eval_dataset[idx][1])
    with open(params["verification_file"], 'r') as fp, \
         open(os.path.join(params["verification_file_bleeched"]), 'w') as fpOut:
        for line in fp:
            tokens = line.strip().split(" ")
            if tokens[1][:-4] in excluded_utt_set:
                continue
            if  tokens[2][:-4] in excluded_utt_set:
                continue
            fpOut.write(line)
