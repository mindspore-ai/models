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
""" model """

import os
import json

import mindspore as ms
import numpy as np

from fastspeech2_ms import hifigan
from fastspeech2_ms.model import FastSpeech2, ScheduledOptim


def get_model(args, configs, train=False):
    """ get_model """

    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = ms.load_checkpoint(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    """ get_param_num """

    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config):
    """ get_vocoder """

    speaker = config["vocoder"]["speaker"]

    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    if speaker == "LJSpeech":
        ckpt = ms.load_checkpoint("hifigan/generator_LJSpeech.pth.tar")
    elif speaker == "universal":
        ckpt = ms.load_checkpoint("hifigan/generator_universal.pth.tar")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    """ vocoder_infer """

    name = model_config["vocoder"]["model"]

    if name == "MelGAN":
        wavs = vocoder.inverse(mels / np.log(10))
    elif name == "HiFi-GAN":
        wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
