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
"""
If you need to use n4correction.py code, you need to copy it to the bin directory where antsRegistration etc are
located. Then run python n4correction.py
python n4correction.py
"""
from __future__ import division
import os
import sys
import glob
from multiprocessing import Pool, cpu_count


def n4_correction(im_input):
    """ n4 correction """
    command = 'N4BiasFieldCorrection -d 3 -i ' + im_input + ' ' + ' -s 3 -c [50x50x30x20] -b [300] -o ' + \
              im_input.replace('.nii.gz', '_corrected.nii.gz')
    os.system(command)


def batch_works(k):
    """ batch works """
    if k == n_processes - 1:
        paths = all_paths[k * int(len(all_paths) / n_processes):]
    else:
        paths = all_paths[k * int(len(all_paths) / n_processes): (k + 1) * int(len(all_paths) / n_processes)]

    for path in paths:
        n4_correction(glob.glob(os.path.join(path, '*_t1.nii.gz'))[0])
        n4_correction(glob.glob(os.path.join(path, '*_t1ce.nii.gz'))[0])
        n4_correction(glob.glob(os.path.join(path, '*_t2.nii.gz'))[0])
        n4_correction(glob.glob(os.path.join(path, '*_flair.nii.gz'))[0])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Need at least the input data directory")
    input_path = sys.argv[1]

    all_paths = []
    for dirpath, dirnames, files in os.walk(input_path):
        if os.path.basename(dirpath)[0:7] == 'Brats17':
            all_paths.append(dirpath)

    n_processes = cpu_count()
    pool = Pool(processes=n_processes)
    pool.map(batch_works, range(n_processes))
