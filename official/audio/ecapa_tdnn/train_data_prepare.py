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
prepare train data
"""
import os
import sys
import random
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.sampler import ReproducibleRandomSampler
from src.voxceleb_prepare import prepare_voxceleb

def dataio_prep(params):
    "Creates the datasets and their data processing pipelines."

    data_folder = params["data_folder"]

    # 1. Declarations:
    train_data_reader = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data_reader = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data_reader, valid_data_reader]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(params["sample_rate"] * params["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if params["random_chunk"]:
            duration_sample = int(duration * params["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample - 1)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, _ = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data_reader], output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data_reader, valid_data_reader, label_encoder


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print('parse parameters done')
    print("start load hyper param")
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    print("download verification file")
    # Download verification list (to exclude verification sentences from train)
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    download_file(hparams["verification_file"], veri_file_path)
    print("data_prep")
    # Dataset prep (parsing VoxCeleb and annotation into csv files)

    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "verification_pairs_file": veri_file_path,
            "splits": ("train", "dev"),
            "split_ratio": (90, 10),
            "seg_dur": hparams["sentence_len"],
            "skip_prep": hparams["skip_prep"]
        },
    )
    print("data io prep")
    if not os.path.exists(os.path.join(hparams["feat_folder"])):
        os.makedirs(os.path.join(hparams["feat_folder"]), exist_ok=False)
    save_dir = os.path.join(hparams["feat_folder"])
    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, _ = dataio_prep(hparams)
    print("len of train:", len(train_data))
    loader_kwargs = hparams["dataloader_options"]
    sampler = None
    if loader_kwargs.get("shuffle", False) is True:
        sampler = ReproducibleRandomSampler(train_data)
        loader_kwargs["sampler"] = sampler
        del loader_kwargs["shuffle"]
    dataloader = sb.dataio.dataloader.make_dataloader(
        train_data, **loader_kwargs
    )
    fea_fp = open(os.path.join(save_dir, "fea.lst"), 'w')
    label_fp = open(os.path.join(save_dir, "label.lst"), 'w')
    for epoch in range(hparams["number_of_epochs"]):
        sampler.set_epoch(epoch)
        cnt = 0
        for batch in tqdm(dataloader):
            batch = batch.to('cpu')
            wavs, lens = batch.sig
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(hparams["augment_pipeline"]):
                # Apply augment
                wavs_aug = augment(wavs, lens)
                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if hparams["concat_augment"]:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * n_augment)
            # Feature extraction and normalization
            feats = hparams["compute_features"](wavs)
            feats = hparams["mean_var_norm"](feats, lens)
            ct = datetime.now()
            ts = ct.timestamp()
            id_save_name = str(ts) + "_id.npy"
            fea_save_name = str(ts) + "_fea.npy"
            spkid = batch.spk_id_encoded.data
            spkid = torch.cat([batch.spk_id_encoded.data] * n_augment, dim=0)
            np.save(os.path.join(save_dir, id_save_name), spkid.numpy())
            np.save(os.path.join(save_dir, fea_save_name), feats.numpy())
            label_fp.write(id_save_name + "\n")
            fea_fp.write(fea_save_name + "\n")
            cnt += 1
