# Copyright 2021 Huawei Technologies Co., Ltd
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
''' generate hdf5 file '''
import os
import argparse
import random
import h5py
from tqdm import tqdm

import numpy as np
import librosa
from src.utils.audio import load_wav, melspectrogram
from src.hparams import hparams as hps
from src.text import text_to_sequence

from src.utils import audio

random.seed(0)


def files_to_list(fdir):
    ''' collect text and filepath to list'''
    f_list = []
    with open(os.path.join(fdir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(fdir, 'wavs', '%s.wav' % parts[0])
            f_list.append([wav_path, parts[1]])
    return f_list


def get_mel_text_pair(filename_and_text):
    '''preprocessing mel and text '''
    filename, text = filename_and_text[0], filename_and_text[1]
    text += '~'
    text = get_text(text)
    mel = produce_mel_features(filename)
    print(mel.shape)
    return (text, mel)


def get_text(text):
    '''encode text to sequence'''
    return text_to_sequence(text, hps.text_cleaners)


def get_mel(filename):
    '''extract mel spectrogram'''
    wav = load_wav(filename)
    trim_wav, _ = librosa.effects.trim(
        wav, top_db=60, frame_length=2048, hop_length=512)
    wav = np.concatenate(
        (trim_wav,
         np.zeros(
             (5 * hps.hop_length),
             np.float32)),
        0)
    mel = melspectrogram(wav).astype(np.float32)
    return mel


def produce_mel_features(filename):
    '''produce Mel-Frequency features'''
    wav, fs = librosa.load(filename, sr=22050)
    wav = librosa.resample(wav, fs, 16000)

    # between audio and mel-spectrogram
    wav = audio.wav_padding(wav, hps)
    assert len(wav) % hps.hop_size == 0
    # Pre-emphasize
    preem_wav = audio.preemphasis(wav, hps.preemphasis, hps.preemphasize)

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.mel_spectrogram(preem_wav, hps).astype(np.float32)
    mel = (mel_spectrogram + hps.max_abs_value) / (2 * hps.max_abs_value)
    return mel.astype(np.float32)


def generate_hdf5(fdir):
    '''generate hdf5 file'''
    f_list = files_to_list(fdir)
    random.shuffle(f_list)

    max_text, max_mel = 0, 0
    for idx, filename_and_text in tqdm(enumerate(f_list)):
        text, mel = get_mel_text_pair(filename_and_text)
        max_text = max(max_text, len(text))
        max_mel = max(max_mel, mel.shape[1])

        with h5py.File('ljdataset.hdf5', 'a') as hf:
            hf.create_dataset('{}_mel'.format(idx), data=mel)
            hf.create_dataset('{}_text'.format(idx), data=text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='',
        help='Path to LJSpeech-1.1')
    args = parser.parse_args()
    generate_hdf5(args.data_path)
