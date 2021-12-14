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
"""build feature dataset for retrieval"""
import os
import time
import argparse
import numpy as np
from annoy import AnnoyIndex

import dataset
import feature_io

parser = argparse.ArgumentParser(description='MindSpore delf Example')

parser.add_argument('--ann_file', type=str, default="")
parser.add_argument('--index_features_dir', type=str, default="")
parser.add_argument('--image_path', type=str, default="")
parser.add_argument('--gt_path', type=str, default="")
parser.add_argument('--ann_path', type=str, default="")

args = parser.parse_known_args()[0]

_STATUS_CHECK_LOAD_ITERATIONS = 50


def _ReadDELFDescriptors(input_dir, image_list):
    """extract index features"""
    num_images = len(image_list)
    print('Starting to collect descriptors for %d images...' % num_images)
    start = time.time()

    descriptors = np.zeros([num_images * 1000, 40], dtype=np.float32)
    locations = np.zeros([num_images * 1000, 2], dtype=np.float32)
    for i in range(num_images):
        if i > 0 and i % _STATUS_CHECK_LOAD_ITERATIONS == 0:
            elapsed = (time.time() - start)
            print('Reading descriptors for image %d out of %d, last %d '
                  'images took %f seconds' %
                  (i, num_images, _STATUS_CHECK_LOAD_ITERATIONS, elapsed))
            start = time.time()

        descriptor_fullpath = os.path.join(input_dir, image_list[i])
        descriptor_t, location_t = feature_io.ReadFeature(descriptor_fullpath)
        if descriptor_t.shape[0] != 1000:
            pad_num = 1000 - descriptor_t.shape[0]
            descriptor_t = np.pad(
                descriptor_t, ((0, pad_num), (0, 0)), 'constant', constant_values=np.inf)
            location_t = np.pad(location_t, ((0, pad_num), (0, 0)),
                                'constant', constant_values=(-1, -1))
            print('not 1000 feature, pad num:', pad_num)
            print(descriptor_t)
            print(location_t)
        descriptors[1000*i:1000*(i+1),] = descriptor_t
        locations[1000*i:1000*(i+1),] = location_t

    return descriptors, locations


def main():
    if not os.path.exists(args.ann_path):
        os.makedirs(args.ann_path)
    print('Parsing dataset...')
    _, index_list, _ = dataset.read_ground_truth(
        args.gt_path, args.image_path)

    num_index_images = len(index_list)

    print('done! Found %d index images' % (num_index_images))

    index_global_features, index_location = _ReadDELFDescriptors(
        args.index_features_dir, index_list)

    np.savez(args.index_features_dir+'/index.location', index_location)

    f = 40
    t = AnnoyIndex(f, 'euclidean')
    for i in range(num_index_images*1000):
        t.add_item(i, index_global_features[i].tolist())

    t.build(10, n_jobs=8)  # 10 trees
    t.save(os.path.join(args.ann_path, args.ann_file))


if __name__ == '__main__':
    main()
