# Copyright 2020 Huawei Technologies Co., Ltd
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
UNITER pre-training
"""
import argparse
import json
import os
from os.path import join
from time import time

import mindspore
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm

from src.data import data_column, create_audio_dataset, get_batch_data_audio
from src.fastspeech2_ms import hifigan
from src.tools.logger import LOGGER, add_log_to_file
from src.tools import parse_with_config, set_random_seed
from src.tools.save import save_training_meta


def get_vocoder(speaker):
    '''
    mindspore;done;useful
    '''
    # name = "HiFi-GAN"
    with open("src/fastspeech2_ms/hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    if speaker == "LJSpeech":
        ckpt = load_checkpoint("fastspeech2_ms/hifigan/generator_LJSpeech.ckpt")
    elif speaker == "universal":
        ckpt = load_checkpoint("fastspeech2_ms/hifigan/generator_universal.ckpt")
    load_param_into_net(vocoder, ckpt, strict_load=True)
    return vocoder


def main(opts):
    context.set_context(mode=context.PYNATIVE_MODE,
                        save_graphs=False,
                        device_target="CPU")

    set_random_seed(opts.seed)

    opts.rank = 0

    if opts.rank == 0:
        save_training_meta(opts)
        log_dir = join(opts.output_dir, 'log')
        add_log_to_file(join(log_dir, 'log.txt'))
    else:
        LOGGER.disabled = True

    LOGGER.info("Create Dataset")

    LOGGER.info("create_two_dataloaders val_dataloaders")
    ds = create_audio_dataset(opts, column_name=data_column,
                              token_size=opts.train_batch_size, full_batch=opts.full_batch)
    opts.val_len = len(json.load(open(opts.ids_val_path)))

    # Prepare model
    if opts.checkpoint:
        LOGGER.info("Load checkpoint %s", opts.checkpoint)
    else:
        LOGGER.info("No checkpoint")

    LOGGER.info("UniterThreeForPretrainingForAd.inference")

    LOGGER.info('start validation')
    validate_ad(None, ds, opts, task='adText')


def validate_ad(model, ds, opts, task):
    """Validate audio"""
    MAX_WAV_VALUE = 32768.0

    expand_dims = mindspore.ops.ExpandDims()

    vocoder = get_vocoder("universal")

    ids_val_path = json.load(open(opts.ids_val_path))

    pbar = tqdm(total=len(ids_val_path))

    LOGGER.info("start running Audio Decoder validation...")
    st = time()

    wav_path = join(opts.output_dir, 'wav')
    if not os.path.exists(wav_path):
        os.mkdir(wav_path)

    mel_path = join(opts.output_dir, 'mel')
    if not os.path.exists(mel_path):
        os.mkdir(mel_path)

    for batch in ds.create_dict_iterator():
        (input_ids, position_ids, attention_mask,
         mel_targets, duration_targets, speakers, texts, src_lens, mel_lens,
         audio_max_text_len, audio_max_mel_len, pitch_targets, energy_targets) = get_batch_data_audio(batch)
        mels, mel_lens = model(input_ids, position_ids, attention_mask,
                               mel_targets, duration_targets, speakers, texts, src_lens, mel_lens,
                               audio_max_text_len, audio_max_mel_len, pitch_targets, energy_targets, compute_loss=False)

        for k in range(mels.shape[0]):
            mel = mels[k]
            mel = expand_dims(mel.transpose(1, 0), 0)
            y_g_hat = vocoder(mel)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.asnumpy().astype('int16')
            key = batch['ids'][k]

            if mel_lens is not None:
                mel_len = mel_lens[k]
                audio = audio[:mel_len * 256]

            full_mel_path = join(mel_path, key.replace("/", "_") + ".npz")
            np.savez(full_mel_path, feat=mel.cpu().numpy())

            full_path = join(wav_path, key.replace("/", "_") + ".wav")
            wavfile.write(full_path, 22050, audio)
            pbar.update(1)

    tot_time = time() - st
    val_log = {'tot_time': tot_time}
    LOGGER.info("validation finished in %d seconds, ", int(tot_time))

    return val_log


def str2bool(b):
    assert b.lower() in ["false", "true"]
    if b.lower() in ["false"]:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',
                        default="./config/pretrain_three_modal_audio_local_config.json",
                        help='JSON config files')

    parser.add_argument('--use_txt_out', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_parallel', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--data_type', default=2, type=int, help='use txt out')

    parser.add_argument('--audio_dim', default=1024, type=int, help='use txt out')
    parser.add_argument('--img_dim', default=2048, type=int, help='use txt out')
    parser.add_argument('--use_data_fix', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--use_mask_fix', default=True, type=str2bool, help='use txt out')

    parser.add_argument('--name_txt', default="id2len_three.json", type=str, help='use txt out')
    parser.add_argument('--name_img', default="img2len_three.json", type=str, help='use img out')
    parser.add_argument('--name_audio', default="audio2len_three.json", type=str, help='use audio out')

    parser.add_argument("--checkpoint", default=None, type=str, help="")
    parser.add_argument("--full_batch", default=False, type=bool, help="")
    parser.add_argument("--use_moe", default=False, type=bool, help="use moe")

    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
