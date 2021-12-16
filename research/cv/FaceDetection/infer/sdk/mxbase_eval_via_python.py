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
"""eval the mxbase infer result via python"""
import os
import sys
import numpy as np

from util.data_preprocess import SingleScaleTrans
from util.eval import gen_eval_result, eval_according_output
from util.eval_util import prepare_file_paths, get_data


def run_pipeline(dataset_path, mxbase_result_path):
    """
        enable comparison graph output
        :param: the path of dataset, the path of mxbase result
        :returns: null
        Output: the figure of accuracy
    """

    # record the eval times
    eval_times = 0
    # record the infer result
    det = {}
    # record the image size
    img_size = {}
    # record the image label
    img_anno = {}

    # loop through the image, start to output the results, and evaluate the accuracy
    print('=============FaceDetection start evaluating==================')
    image_files, anno_files, image_names = prepare_file_paths(dataset_path)
    dataset_size = len(anno_files)
    assert dataset_size == len(image_files)
    assert dataset_size == len(image_names)
    data_set = []
    for i in range(dataset_size):
        data_set.append(get_data(image_files[i], anno_files[i], image_names[i]))
    for data in data_set:
        single_trans = SingleScaleTrans()
        pre_data = single_trans.deal_ann_size_name_to_array(data['annotation'], data['image_name'], data['image_size'])
        labels, image_name, image_size = pre_data[0:3]
        # Judge the input picture whether is a jpg format

        # bin files directory for storing inference results
        file_path = os.path.join(os.path.join(mxbase_result_path, image_name[0]), 'output')
        # take 6 outcomes from the bin file: coords_0, cls_scores_0, coords_1, cls_scores_1, coords_2, cls_scores_2
        coords_0 = np.fromfile(file_path + '_0.bin', dtype=np.float32).reshape((1, 4, 84, 4))
        cls_scores_0 = np.fromfile(file_path + '_1.bin', dtype=np.float32).reshape((1, 4, 84))
        coords_1 = np.fromfile(file_path + '_2.bin', dtype=np.float32).reshape((1, 4, 336, 4))
        cls_scores_1 = np.fromfile(file_path + '_3.bin', dtype=np.float32).reshape((1, 4, 336))
        coords_2 = np.fromfile(file_path + '_4.bin', dtype=np.float32).reshape((1, 4, 1344, 4))
        cls_scores_2 = np.fromfile(file_path + '_5.bin', dtype=np.float32).reshape((1, 4, 1344))

        eval_according_output(coords_0, cls_scores_0, coords_1, cls_scores_1, coords_2, cls_scores_2, det,
                              img_anno, img_size, labels, image_name, image_size)
        eval_times += 1

    # generate the eval result
    gen_eval_result(eval_times, det, img_size, img_anno)


if __name__ == '__main__':
    arg_arr_input_output_path = []
    if len(sys.argv) != 3:
        print('Wrong parameter setting.')
        exit()
    run_pipeline(sys.argv[1], sys.argv[2])
