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
import cv2
import mxnet as mx

def main(input_args):
    """
    trans rec format to jpg
    :param args: inputs arguments
    :return:
    """
    include_datasets = input_args.include.split(',')
    rec_list = []
    for ds in include_datasets:
        path_imgrec = os.path.join(ds, 'train.rec')
        path_imgidx = os.path.join(ds, 'train.idx')
        imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        rec_list.append(imgrec)
    if not os.path.exists(input_args.output):
        os.makedirs(input_args.output)

    imgid = 0
    for ds_id in range(len(rec_list)):
        imgrec = rec_list[ds_id]
        s = imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        assert header.flag > 0
        seq_identity = range(int(header.label[0]), int(header.label[1]))

        for identity in seq_identity:
            s = imgrec.read_idx(identity)
            header, _ = mx.recordio.unpack(s)
            for _idx in range(int(header.label[0]), int(header.label[1])):
                s = imgrec.read_idx(_idx)
                _header, _img = mx.recordio.unpack(s)
                label = int(_header.label[0])
                class_path = os.path.join(args.output, "%d" % label)
                if not os.path.exists(class_path):
                    os.makedirs(class_path)
                _img = mx.image.imdecode(_img).asnumpy()[:, :, ::-1]  # to bgr
                image_path = os.path.join(class_path, "%d_%d.jpg" % (label, imgid))
                cv2.imwrite(image_path, _img)
                imgid += 1
                if imgid % 10000 == 0:
                    print(imgid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do dataset merge')
    # general
    parser.add_argument('--include', default='', type=str, help='')
    parser.add_argument('--output', default='', type=str, help='')
    args = parser.parse_args()
    main(args)
