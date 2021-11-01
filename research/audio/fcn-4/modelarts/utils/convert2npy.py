# coding: utf-8
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
'''python prepare_dataset.py'''

import os
import pandas as pd
import numpy as np
import librosa


def compute_melgram(audio_path, save_path='', filename='', save_npy=True):
    """
    extract melgram feature from the audio and save as numpy array
    Args:
        audio_path (str): path to the audio clip.
        save_path (str): path to save the numpy array.
        filename (str): filename of the audio clip.
        save_npy (bool): weather to save data in npy file.
    Returns:
        ret: numpy array.
    """
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..
    ret = None
    try:
        src, _ = librosa.load(audio_path, sr=SR)  # whole signal
    except EOFError:
        print("file was damaged:", audio_path)
        print("now skip it!")
        return ret
    except FileNotFoundError:
        print("cant load file:", audio_path)
        print("now skip it!")
        return ret
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]
    logam = librosa.core.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS))
    ret = ret[np.newaxis, np.newaxis, :]
    if save_npy:
        save_path = os.path.join(save_path, filename[:-4] + '.npy')
        np.save(save_path, ret)
    return ret

def convert(s):
    if s.isdigit():
        return int(s)
    return s

def GetLabel(info_path, info_name):
    """
    separate dataset into training set and validation set

    Args:
        info_path (str): path to the information file.
        info_name (str): name of the information file.

    """
    T = []
    with open(info_path + '/' + info_name, 'rb') as info:
        data = info.readline()
        while data:
            T.append([
                convert(i[1:-1])
                for i in data.strip().decode('utf-8').split("\t")
            ])
            data = info.readline()

    annotation = pd.DataFrame(T[1:], columns=T[0])
    count = []
    for i in annotation.columns[1:-2]:
        count.append([annotation[i].sum() / len(annotation), i])
    count = sorted(count)
    full_label = []
    for i in count[-50:]:
        full_label.append(i[1])
    out = []
    for i in T[1:]:
        index = [k for k, x in enumerate(i) if x == 1]
        label = [T[0][k] for k in index]
        L = [str(0) for k in range(50)]
        L.append(i[-1])
        for j in label:
            if j in full_label:
                ind = full_label.index(j)
                L[ind] = '1'
        out.append(L)
    out = np.array(out)

    Train = []
    Val = []

    for i in out:
        if np.random.rand() > 0.2:
            Train.append(i)
        else:
            Val.append(i)
    np.savetxt("{}/music_tagging_train_tmp.csv".format(info_path),
               np.array(Train),
               fmt='%s',
               delimiter=',')
    np.savetxt("{}/music_tagging_val_tmp.csv".format(info_path),
               np.array(Val),
               fmt='%s',
               delimiter=',')

if __name__ == "__main__":
    num_classes = 50
    base_info_path = "../../modelarts/config/"
    info_file = "annotations_final.csv"
    root_audio_path = "../data/audio/"
    melgram_save_path = "../../modelarts/npy_path/"
    GetLabel(base_info_path, info_file)
    if not os.path.isdir(melgram_save_path):
        os.mkdir(melgram_save_path)
    dir_names = os.listdir(root_audio_path)
    for _dir in dir_names:
        print("-"*40)
        print("audio_path: ", os.path.join(root_audio_path, _dir))
        file_names = os.listdir(os.path.join(root_audio_path, _dir))
        if not os.path.isdir(os.path.join(melgram_save_path, _dir)):
            os.mkdir(os.path.join(melgram_save_path, _dir))
            print("save_path: ", os.path.join(melgram_save_path, _dir))
        for f in file_names:
            compute_melgram(os.path.join(root_audio_path, _dir, f),
                            os.path.join(melgram_save_path, _dir), f)
        print("successfully save npy files in:\n", os.path.join(melgram_save_path, _dir))
