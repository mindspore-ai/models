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
'''proposessing mel spectrogram synthesized by Tacotron2 to make it suitable for Wavenet inference,
the statistic data(meanvar.joblib) is attained from Wavenet(ascend version) dataset proposessing phase.
'''
import os
import os.path
import argparse
import numpy as np

import joblib

from audio import inv_melspectrogram

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_pth', type=str, default='',
                        required=True, help='path to load checkpoints')
    args = parser.parse_args()
    mel = np.load(args.file_pth)['arr_0']

    dir_name, fname = os.path.split(args.file_pth)

    wav = inv_melspectrogram(mel)

    scalar = joblib.load('meanvar.joblib')
    mel = scalar.transform(mel)

    np.save(
        os.path.join(
            dir_name,
            'output-feats.npy'),
        mel.T,
        allow_pickle=False)
    np.save(os.path.join(dir_name, 'output-wave.npy'), wav, allow_pickle=False)
