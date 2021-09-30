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
"""pre process for 310 inference"""
import numpy
from mindspore.common import set_seed
from src.utils import get_args
from src.dataset import CelebADataLoader

set_seed(1)

def create_test_label(c_org, selected_attrs):
    """Generate target domain labels for debugging and testing."""
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in [
                'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'
        ]:
            hair_color_indices.append(i)
    c_trg = numpy.copy(c_org)
    for i in range(len(selected_attrs)):
        if i in hair_color_indices:
            c_trg[:, i] = 1.
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0.
                    c_trg[:, j].astype(numpy.float32)
        else:
            c_trg[:, i] = (c_trg[:, i] == 0).astype(numpy.float32)
    return c_trg

def preprocess_data():
    """ preprocess data """
    args = get_args("test")
    data_loader = CelebADataLoader(args.dataroot,
                                   mode=args.phase,
                                   selected_attrs=args.attrs,
                                   batch_size=1,
                                   image_size=args.image_size)
    iter_per_epoch = len(data_loader)
    args.dataset_size = iter_per_epoch
    for _ in range(iter_per_epoch):
        data = next(data_loader.test_loader)
        filename = data_loader.test_set.get_current_filename()
        data_image = data['image']
        data_label = data['label']
        temp = data_label.asnumpy()
        c_trg = create_test_label(temp, args.attrs)
        attr_diff = c_trg - temp if args.attr_mode == 'diff' else c_trg
        attr_diff = attr_diff * args.thres_int
        filenum = filename.split('.')[0]
        file_name = args.dataroot + "/preprocess_Data/data/" + filenum + ".bin"
        label_name = args.dataroot + "/preprocess_Data/label/" + filenum + ".bin"
        data_image.asnumpy().tofile(file_name)
        attr_diff.tofile(label_name)


if __name__ == '__main__':
    preprocess_data()
