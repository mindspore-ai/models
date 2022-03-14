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
"""Dataset preprocess script."""
import os
from pathlib import Path

import numpy as np

from src.audio.tools import get_mel
from src.cfg.config import config as hp

# Original dataset contains 13100 samples and not splited into parts.
# We manually selected 100 test indices and fixed it to be able to reproduce results.
_INDICES_FOR_TEST = (
    3667, 8887, 10353, 7657, 1498, 2758, 4913, 1697, 5653, 1911,
    12032, 8925, 11517, 5881, 6575, 120, 6232, 11680, 8433, 1728,
    12771, 11738, 6574, 12918, 9836, 7556, 2231, 7916, 5985, 3148,
    2596, 1709, 5841, 5383, 6248, 9831, 7667, 10944, 2833, 614,
    11990, 6894, 12645, 5422, 12015, 447, 7108, 2973, 9937, 11938,
    3626, 11406, 2853, 6379, 1621, 3981, 5486, 3902, 10925, 4249,
    6518, 3376, 1998, 10250, 10145, 7325, 2665, 61, 2709, 11683,
    8776, 10979, 8834, 4805, 4565, 2577, 9369, 4422, 8212, 5871,
    10721, 6046, 5129, 9610, 821, 4378, 693, 10500, 5027, 1663,
    6946, 2460, 6068, 4329, 11001, 10122, 9154, 6990, 8908, 2530,
)


def preprocess_ljspeech(root_dir):
    """Preprocess LJSpeech dataset."""
    in_dir = root_dir
    out_dir = os.path.join(in_dir, 'mels')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    metadata = build_from_path(in_dir, out_dir)
    write_metadata(metadata, in_dir)
    train_test_split(in_dir)


def write_metadata(metadata, out_dir):
    """Write clear metadata."""
    with Path(out_dir, 'metadata.txt').open('w', encoding='utf-8') as file:
        for m in metadata:
            file.write(m + '\n')


def build_from_path(in_dir, out_dir):
    """Get text and preprocess .wavs to mels."""
    index = 1
    texts = []

    with Path(in_dir, 'metadata.csv').open('r', encoding='utf-8') as file:
        for line in file.readlines():
            if index % 100 == 0:
                print("{:d} Done".format(index))

            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            texts.append(_process_utterance(out_dir, index, wav_path, text))

            index = index + 1

    return texts


def _process_utterance(out_dir, index, wav_path, text):
    """Preprocess .wav to mel and save."""
    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = get_mel(wav_path)

    # Write the spectrograms to disk:
    mel_filename = 'ljspeech-mel-%05d.npy' % index
    np.save(
        os.path.join(out_dir, mel_filename),
        mel_spectrogram.T,
        allow_pickle=False
    )

    return text


def train_test_split(folder_path):
    """Prepare data for training and validation format."""
    test_indices = np.array(_INDICES_FOR_TEST)

    with Path(folder_path, 'metadata.csv').open('r') as file:
        metadata = file.readlines()
        dataset_size = len(metadata)

    test_metadata = []
    all_indices = np.arange(dataset_size)
    train_indices = np.delete(all_indices, test_indices)

    with Path(folder_path, 'train_indices.txt').open('w') as file:
        for i in train_indices:
            file.write(f'{i}\n')

    for i, line in enumerate(metadata):
        if i in test_indices:
            wav_name, _, text = line.strip().split('|')
            test_data = f'{wav_name}|{text}\n'
            test_metadata.append(test_data)

    with Path(folder_path, 'validation.txt').open('w') as file:
        for line in test_metadata:
            file.write(line)


def main():
    preprocess_ljspeech(hp.dataset_path)


if __name__ == "__main__":
    main()
