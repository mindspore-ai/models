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
'''
Process audio files for generating training and evaluating data.
'''
import os
from os.path import basename, splitext, join
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob
import audio
import librosa
from nnmnkwii import preprocessing as P
from hparams import hparams
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw
import numpy as np
sys.path.append('.')


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''processing audio files'''
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    src_files = sorted(glob(join(in_dir, "*.wav")))
    for wav_path in src_files:
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, index, wav_path, "dummy")))
        index += 1

    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
    '''processing audio'''
    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    # Trim begin/end silences
    # NOTE: the threshold was chosen for clean signals
    wav, _ = librosa.effects.trim(wav, top_db=60, frame_length=2048, hop_length=512)

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = audio.logmelspectrogram(wav).astype(np.float32).T

    # Mu-law quantize
    if is_mulaw_quantize(hparams.input_type):
        # Trim silences in mul-aw quantized domain
        silence_threshold = 0
        if silence_threshold > 0:
            # [0, quantize_channels)
            out = P.mulaw_quantize(wav, hparams.quantize_channels - 1)
            start, end = audio.start_and_end_indices(out, silence_threshold)
            wav = wav[start:end]
        constant_values = P.mulaw_quantize(0, hparams.quantize_channels - 1)
        out_dtype = np.int16
    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        constant_values = P.mulaw(0.0, hparams.quantize_channels - 1)
        out_dtype = np.float32
    else:
        # [-1, 1]
        constant_values = 0.0
        out_dtype = np.float32

    if hparams.global_gain_scale > 0:
        wav *= hparams.global_gain_scale

    # Clip
    if np.abs(wav).max() > 1.0:
        print("""Warning: abs max value exceeds 1.0: {}""".format(np.abs(wav).max()))
        # ignore this sample
        return ("dummy", "dummy", -1, "dummy")

    # Set waveform target (out)
    if is_mulaw_quantize(hparams.input_type):
        out = P.mulaw_quantize(wav, hparams.quantize_channels - 1)
    elif is_mulaw(hparams.input_type):
        out = P.mulaw(wav, hparams.quantize_channels - 1)
    else:
        out = wav

    # zero pad
    # this is needed to adjust time resolution between audio and mel-spectrogram
    l, r = audio.pad_lr(out, hparams.fft_size, audio.get_hop_size())
    if l > 0 or r > 0:
        out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
    N = mel_spectrogram.shape[0]
    assert len(out) >= N * audio.get_hop_size()

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:N * audio.get_hop_size()]
    assert len(out) % audio.get_hop_size() == 0

    # Write the spectrograms to disk:
    name = splitext(basename(wav_path))[0]
    audio_filename = '%s-wave.npy' % name
    mel_filename = '%s-feats.npy' % name
    np.save(os.path.join(out_dir, audio_filename),
            out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return (audio_filename, mel_filename, N, text)
