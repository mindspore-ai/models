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
"""Prepare data."""
import argparse
import os
import os.path as osp
import shutil
from pathlib import Path

import numpy as np


def prepare(seq_root):
    """Prepare MOT17 dataset for JDE training."""
    label_root = str(Path(Path(seq_root).parents[0], 'labels_with_ids', 'train'))
    seqs = [s for s in os.listdir(seq_root) if s.endswith('SDP')]

    tid_curr = 0
    tid_last = -1

    for seq in seqs:
        with open(osp.join(seq_root, seq, 'seqinfo.ini')) as file:
            seq_info = file.read()

        seq_width = int(seq_info[seq_info.find('imWidth=') + 8: seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9: seq_info.find('\nimExt')])

        gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

        seq_label_root = osp.join(label_root, seq, 'img1')
        if not osp.exists(seq_label_root):
            os.makedirs(seq_label_root)

        for fid, tid, x, y, w, h, mark, label, _ in gt:
            if mark == 0 or not label == 1:
                continue
            fid = int(fid)
            tid = int(tid)
            if tid != tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)

        old_path = str(Path(seq_root, seq))
        new_path = str(Path(Path(seq_root).parents[0], 'images', 'train'))

        if not osp.exists(new_path):
            os.makedirs(new_path)

        shutil.move(old_path, new_path)

    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_root", required=True, help='Path to root dir of sequences')

    args = parser.parse_args()
    prepare(args.seq_root)
