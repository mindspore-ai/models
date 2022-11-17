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
rec format to jpg
"""
import os
import argparse
from skimage import io
import mxnet as mx
from mxnet import recordio
from tqdm import tqdm


def main(dataset_path, output_dir):
    path_imgrec = os.path.join(dataset_path, 'train.rec')
    path_imgidx = os.path.join(dataset_path, 'train.idx')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    print('max_idx:', max_idx)
    for i in tqdm(range(max_idx)):
        header, s = recordio.unpack(imgrec.read_idx(i + 1))
        img = mx.image.imdecode(s).asnumpy()
        label = str(header.label)
        ids = str(i)

        label_dir = os.path.join(output_dir, label)
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
        fname = 'Figure_{}.png'.format(ids)
        fpath = os.path.join(label_dir, fname)
        io.imsave(fpath, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do dataset merge')
    # general
    parser.add_argument('--include', default='', type=str, help='')
    parser.add_argument('--output', default='', type=str, help='')
    args = parser.parse_args()
    main(args.include, args.output)
