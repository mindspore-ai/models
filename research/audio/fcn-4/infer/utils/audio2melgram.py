# coding=utf-8

"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import librosa
import numpy as np


def compute_melgram(audio_source_path, _save_path):
    """
    extract melgram feature from the audio and save as numpy array

    Args:
        audio_path (str): path to the audio clip.
        melgram_save_path (str): path to save the numpy array of melgram feature data.

    Returns:
        numpy array.

    """
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame.
    try:
        src, _ = librosa.load(audio_source_path, sr=SR)  # whole signal
    except EOFError:
        print("file was damaged:", audio_source_path)
        print("now skip it!")
        return
    except FileNotFoundError:
        print("cant load file:", audio_source_path)
        print("now skip it!")
        return
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)
    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]
    logam = librosa.core.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS))
    save_path = _save_path[:-4] + '.txt'
    np.savetxt(save_path, np.array(ret, dtype=np.float32).reshape(1, -1), fmt='%s', delimiter=' ')


if __name__ == "__main__":
    arg_nums = len(sys.argv)
    if arg_nums == 1:
        root_audio_path = "../data/audio/"
        melgram_save_path = "../data/melgram/"
        print("Two arguments (root_audio_path, melgram_save_path) are required, command should be like:")
        print(" python audio2melgram.py  root_audio_path melgram_save_path")
        print("And now, default path will be used, e.g. execute command:")
        print(" python audio2melgram.py  ../data/audio/ ../data/melgram/")
    elif arg_nums == 2:
        root_audio_path = sys.argv[1]
        melgram_save_path = "../data/melgram/"
    else:
        root_audio_path = sys.argv[1]
        melgram_save_path = sys.argv[2]
    dir_names = os.listdir(root_audio_path)
    for _dir in dir_names:
        print("start to process source audio in path:\n", os.path.join(root_audio_path, _dir))
        file_names = os.listdir(os.path.join(root_audio_path, _dir))
        for f in file_names:
            compute_melgram(os.path.join(root_audio_path, _dir, f),
                            os.path.join(melgram_save_path, f))
        print("successfully save melgram feature in file path:\n", melgram_save_path)
