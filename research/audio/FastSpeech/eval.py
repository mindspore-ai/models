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
"""Evaluation script."""
import os
from pathlib import Path

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import load_checkpoint
from mindspore.common import set_seed
from scipy.io.wavfile import write

from src.cfg.config import config as hp
from src.dataset import get_val_data
from src.deepspeech2.dataset import LoadAudioAndTranscript
from src.deepspeech2.model import DeepSpeechModel
from src.metrics import frechet_classifier_distance_from_activations
from src.metrics import kernel_classifier_distance_and_std_from_activations
from src.model import FastSpeech
from src.model import FastSpeechEval
from src.waveglow.model import WaveGlow

set_seed(1)


def save_audio(audio, audio_length, save_root_dir, name, audio_cfg):
    """Process raw audio and save as .wav audio file."""
    audio_length = int(audio_length.asnumpy())
    audio = audio[:, :audio_length] * audio_cfg['wav_value']
    audio = (audio.asnumpy().squeeze()).astype('int16')

    audio_path = os.path.join(save_root_dir, name + '_synth.wav')
    write(audio_path, audio_cfg['sampling_rate'], audio)

    return audio_path


def get_waveglow(ckpt_url):
    """
    Init WaveGlow vocoder model with weights.
    Used to generate realistic audio from mel-spectrogram.
    """
    wn_config = {
        'n_layers': hp.wg_n_layers,
        'n_channels': hp.wg_n_channels,
        'kernel_size': hp.wg_kernel_size
    }

    audio_config = {
        'wav_value': hp.wg_wav_value,
        'sampling_rate': hp.wg_sampling_rate
    }

    model = WaveGlow(
        n_mel_channels=hp.wg_n_mel_channels,
        n_flows=hp.wg_n_flows,
        n_group=hp.wg_n_group,
        n_early_every=hp.wg_n_early_every,
        n_early_size=hp.wg_n_early_size,
        wn_config=wn_config
    )

    load_checkpoint(ckpt_url, model)
    model.set_train(False)

    return model, audio_config


def get_deepspeech(ckpt_url):
    """
    Init DeepSpeech2 model with weights.
    Used to get activations from lstm layers to compute metrics.
    """
    spect_config = {
        'sampling_rate': hp.ds_sampling_rate,
        'window_size': hp.ds_window_size,
        'window_stride': hp.ds_window_stride,
        'window': hp.ds_window
    }

    model = DeepSpeechModel(
        batch_size=1,
        rnn_hidden_size=hp.ds_hidden_size,
        nb_layers=hp.ds_hidden_layers,
        labels=hp.labels,
        rnn_type=hp.ds_rnn_type,
        audio_conf=spect_config,
        bidirectional=True
    )

    load_checkpoint(ckpt_url, model)
    model.set_train(False)

    return model, spect_config


def get_fastspeech(ckpt_url):
    """
    Init FastSpeech model with weights.
    Used to generate mel-spectrogram from sequence (text).
    """
    model = FastSpeech()

    load_checkpoint(ckpt_url, model)
    model.set_train(False)

    return model


def activation_from_audio(loader, model, path):
    """
    Compute activations of audio to get metric.

    Args:
        loader (class): Audio loader.
        model (nn.Cell): DeepSpeech2 model.
        path (str): Path to the audio.

    Returns:
         activation (np.array): Activations from last lstm layer.
    """
    metric_mel = loader.parse_audio(audio_path=path)
    metric_mel_len = Tensor([metric_mel.shape[1]], mstype.float32)
    metric_mel_padded = np.pad(metric_mel, (0, hp.mel_val_len - metric_mel.shape[1]))[:metric_mel.shape[0], :]
    metric_mel_padded = Tensor(np.expand_dims(np.expand_dims(metric_mel_padded, 0), 0), mstype.float32)

    _, output_length, activation = model(metric_mel_padded, metric_mel_len)
    output_length = int(output_length.asnumpy())

    activation = activation.asnumpy().transpose((1, 0, 2)).squeeze()
    clear_activation = activation[:output_length, :]

    return clear_activation


def main(args):
    fastspeech = get_fastspeech(args.fs_ckpt_url)
    waveglow, audio_config = get_waveglow(args.wg_ckpt_url)
    deepspeech, spect_config = get_deepspeech(args.ds_ckpt_url)

    audio_loader = LoadAudioAndTranscript(spect_config)

    model = FastSpeechEval(
        mel_generator=fastspeech,
        vocoder=waveglow,
        config=args
    )

    data_list = get_val_data(hp.dataset_path)

    if not os.path.exists(hp.output_dir):
        os.makedirs(hp.output_dir, exist_ok=True)

    frechet, kernel = [], []

    for sequence, src_pos, target_audio_path in data_list:
        raw_audio, audio_len = model.get_audio(sequence, src_pos)

        audio_path = save_audio(
            audio=raw_audio,
            audio_length=audio_len,
            save_root_dir=args.output_dir,
            audio_cfg=audio_config,
            name=Path(target_audio_path).stem
        )

        activation = activation_from_audio(audio_loader, deepspeech, audio_path)
        activation_target = activation_from_audio(audio_loader, deepspeech, target_audio_path)

        frechet_distance = frechet_classifier_distance_from_activations(
            activations1=activation,
            activations2=activation_target,
        )

        kernel_distance, _ = kernel_classifier_distance_and_std_from_activations(
            activations1=activation,
            activations2=activation_target,
        )

        frechet.append(frechet_distance)
        kernel.append(kernel_distance)

    print('=' * 10 + 'Evaluation results' + '=' * 10)
    print(f'Mean Frechet distance {round(float(np.mean(np.array(frechet))), 5)}')
    print(f'Mean Kernel distance {round(float(np.mean(np.array(kernel))), 5)}')
    print(f'Generated audios stored into {args.output_dir}')


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=hp.device_target)
    context.set_context(device_id=hp.device_id)
    main(hp)
