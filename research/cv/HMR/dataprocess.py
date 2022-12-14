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
"""
Add notes.
"""
import os
import argparse
import numpy as np
import h5py

parser = argparse.ArgumentParser()
parser.description = 'please enter datasat path'
parser.add_argument(
    "--data_path",
    type=str,
    required=True,
    help="Datasat path")
args = parser.parse_args()


def WriteHuman36m(file):

    gt2d = []
    gt3d = []
    pose = []
    shape = []
    imagename = []
    file_npz = file
    file_h5 = file.replace('annots.npz', 'annot.h5')
    annots = np.load(file_npz, allow_pickle=True)['annots'][()]
    for i in annots.keys():
        _ = np.ones(32)
        _tmp_ = np.insert(annots[i]['kp2d'], 2, _, axis=1)
        gt2d.append(_tmp_)
        gt3d.append(annots[i]['kp3d'])
        pose.append(annots[i]['poses'])
        shape.append(annots[i]['betas'])
        imagename.append(i.encode())

    with h5py.File(file_h5, 'w') as hf:
        hf.create_dataset('gt2d', data=np.array(gt2d))
        hf.create_dataset('gt3d', data=np.array(gt3d))
        hf.create_dataset('pose', data=np.array(pose))
        hf.create_dataset('shape', data=np.array(shape))
        hf.create_dataset('imagename', data=np.array(imagename))


def WriteMpi(file):

    gt2d = []
    gt3d = []
    imagename = []
    file_npz = file
    file_h5 = file.replace('annots.npz', 'annot.h5')
    annots = np.load(file_npz, allow_pickle=True)['annots'][()]
    for i in annots.keys():
        _ = np.ones(28)
        _tmp_ = np.insert(annots[i]['kp2d'], 2, _, axis=1)
        gt2d.append(_tmp_)
        gt3d.append(annots[i]['kp3d'])
        imagename.append(i.encode())

    with h5py.File(file_h5, 'w') as hf:
        hf.create_dataset('gt2d', data=np.array(gt2d))
        hf.create_dataset('gt3d', data=np.array(gt3d))
        hf.create_dataset('imagename', data=np.array(imagename))


if __name__ == '__main__':
    PathHum36m = os.path.join(args.data_path, 'human3.6', 'annots.npz')
    PathMpi = os.path.join(args.data_path, 'mpii_3d', 'annots.npz')
    WriteHuman36m(PathHum36m)
    WriteMpi(PathMpi)
