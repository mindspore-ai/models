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
'''audio'''

import librosa
import librosa.filters
import numpy as np
import scipy
from scipy.io import wavfile

from src.hparams import hparams as hps


def load_wav(path):
    ''' load wav '''
    _, wav = wavfile.read(path)
    signed_int16_max = 2**15
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / signed_int16_max

    wav = wav / np.max(np.abs(wav))
    return wav


def save_wav(wav, path):
    ''' save wav'''
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hps.sample_rate, wav.astype(np.int16))


def preemphasis(x):
    ''' preemphasis '''
    return scipy.signal.lfilter([1, -hps.preemphasis], [1], x)


def inv_preemphasis(x):
    ''' inv preemphasis '''
    return scipy.signal.lfilter([1], [1, -hps.preemphasis], x)


def spectrogram(y):
    ''' extract spectrogram '''
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - hps.ref_level_db
    return _normalize(S)


def inv_spectrogram(spec):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spec) + hps.ref_level_db)
    return inv_preemphasis(_griffin_lim(S ** hps.power))


def melspectrogram(y):
    '''extract normalized mel spectrogram'''
    D = _stft(y)
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hps.ref_level_db
    return _normalize(S)


def inv_melspectrogram(spec):
    '''convert mel spectrogram to waveform '''
    mel = _db_to_amp(_denormalize(spec) + hps.ref_level_db)
    S = _mel_to_linear(mel)
    return _griffin_lim(S ** hps.power)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
    ''' find endpoint '''
    window_length = int(hps.sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x + window_length]) < threshold:
            return x + hop_length
    return len(wav)


def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for _ in range(hps.gl_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _stft(y):
    ''' stft using librosa '''
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        pad_mode='reflect')


def _istft(y):
    ''' istft using librosa '''
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
    '''  get stft parameters'''
    n_fft = (hps.num_freq - 1) * 2
    hop_length = hps.hop_length
    win_length = hps.win_length
    return n_fft, hop_length, win_length


_mel_basis = None


def _linear_to_mel(spec):
    ''' linear spectrogram to mel spectrogram'''
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spec)


def _mel_to_linear(spec):
    ''' mel spectrogram to linear spectrogram '''
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    inv_mel_basis = np.linalg.pinv(_mel_basis)
    inverse = np.dot(inv_mel_basis, spec)
    inverse = np.maximum(1e-10, inverse)
    return inverse


def _build_mel_basis():
    ''' build mel filters '''
    n_fft = (hps.num_freq - 1) * 2
    return librosa.filters.mel(
        hps.sample_rate,
        n_fft,
        fmin=hps.fmin,
        fmax=hps.fmax,
        n_mels=hps.num_mels)


def _amp_to_db(x):
    ''' amp to db'''
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    ''' db to amp '''
    return np.power(10.0, x * 0.05)


def _normalize(S):
    ''' normalize '''
    return np.clip((S - hps.min_level_db) / -hps.min_level_db, 0, 1)


def _denormalize(S):
    '''denormalize '''
    return (np.clip(S, 0, 1) * -hps.min_level_db) + hps.min_level_db
