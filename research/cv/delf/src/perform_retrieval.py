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
"""perform retrieval"""
import os
import time
import argparse
import numpy as np
from skimage import measure
from skimage import transform
from annoy import AnnoyIndex

import dataset
import feature_io

parser = argparse.ArgumentParser(description='MindSpore delf Example')

parser.add_argument('--worker_id', type=int, default=0)
parser.add_argument('--worker_num', type=int, default=1)
parser.add_argument('--threshold', type=int, default=4)
parser.add_argument('--ann_file', type=str, default="")
parser.add_argument('--query_features_dir', type=str, default="")
parser.add_argument('--index_features_dir', type=str, default="")
parser.add_argument('--image_path', type=str, default="")
parser.add_argument('--gt_path', type=str, default="")

parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--rank_file', type=str, default="")


args = parser.parse_known_args()[0]

# Pace to log.
_STATUS_CHECK_LOAD_ITERATIONS = 50

# extract query features


def _Read_query_Descriptors(input_dir, image_list):
    """read query images' features"""
    num_images = len(image_list)
    print('Starting to collect descriptors for %d images...' % num_images)
    start = time.time()

    query_images = []

    for i in range(num_images):
        image_dict = {}
        if i > 0 and i % _STATUS_CHECK_LOAD_ITERATIONS == 0:
            elapsed = (time.time() - start)
            print('Reading descriptors for image %d out of %d, last %d '
                  'images took %f seconds' %
                  (i, num_images, _STATUS_CHECK_LOAD_ITERATIONS, elapsed))
            start = time.time()

        descriptor_fullpath = os.path.join(input_dir, image_list[i])
        descriptor_t, location_t = feature_io.ReadFeature(descriptor_fullpath)
        image_dict['descriptor'] = descriptor_t.tolist()
        image_dict['location'] = location_t

        query_images.append(image_dict)

    return query_images


def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Parse dataset to obtain query/index images, and ground-truth.
    print('Parsing dataset...')
    query_list, index_list, _ = dataset.read_ground_truth(
        args.gt_path, args.image_path)

    num_query_images = len(query_list)
    num_index_images = len(index_list)

    print('done! Found %d queries and %d index images' %
          (num_query_images, num_index_images))

    query_images = _Read_query_Descriptors(args.query_features_dir, query_list)

    index_location = np.load(args.index_features_dir +
                             '/index.location'+'.npz')['arr_0']

    f = 40
    t = AnnoyIndex(f, 'euclidean')
    t.load(args.ann_file)  # super fast, will just mmap the file

    print('success build tree!')

    uplimit = (args.worker_id + 1) * num_query_images // args.worker_num
    downlimit = args.worker_id * num_query_images // args.worker_num

    for i in range(num_query_images)[downlimit:uplimit]:
        indices = []
        for feature in query_images[i]['descriptor']:
            one_indice = t.get_nns_by_vector(feature, 60)
            indices.append(one_indice)
        query_images[i]['indices'] = indices
        query_indice = [[val for _ in range(60)]for val in range(
            len(query_images[i]['descriptor']))]
        query_images[i]['query_indice'] = query_indice

    ranks = np.zeros([uplimit - downlimit, num_index_images], dtype='int32')

    for i in range(num_query_images)[downlimit:uplimit]:
        score = np.zeros([num_index_images], dtype='int32')
        print('Performing retrieval with query %d (%s)...' %
              (i, query_list[i]))
        start_t = time.time()

        sort_query = sum(query_images[i]['query_indice'], [])
        sort_index = sum(query_images[i]['indices'], [])
        sort_index, sort_query = zip(*sorted(zip(sort_index, sort_query)))
        value = sort_index[0] // 1000
        cur_j = 1
        start = 0
        end = 1
        #print('sort_index: ', len(sort_index))
        while cur_j < len(query_images[i]['descriptor'])*60:
            new_value = sort_index[cur_j] // 1000
            if new_value == value:
                end += 1
                cur_j += 1
            else:
                # start match
                location_q_i = list(sort_query[start:end])
                location_i_i = list(sort_index[start:end])
                location_1_to_use = query_images[i]['location'][location_q_i]
                location_2_to_use = index_location[location_i_i]

                old_value = value
                value = new_value
                start = cur_j
                end = cur_j + 1
                cur_j += 1

                if location_1_to_use.shape[0] < args.threshold:
                    continue
                _, inliers = measure.ransac((location_1_to_use, location_2_to_use),
                                            transform.AffineTransform,
                                            min_samples=3,
                                            residual_threshold=20,
                                            max_trials=1000)
                if inliers is None:
                    continue

                print(f'i: {i}, j: {old_value}, sum: , {sum(inliers)}')

                score[old_value] = int(sum(inliers))

        ranks[i-downlimit] = np.argsort(-score)
        elapsed = (time.time() - start_t)
        print('done! Retrieval for query %d took %f seconds' % (i, elapsed))

    np.savez(os.path.join(args.output_dir, args.rank_file) +
             str(args.worker_id), ranks)
    print('save success!')


if __name__ == '__main__':
    main()
