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
network config setting, will be used in train.py and eval.py
"""
import os


class Data:
    """
    Data
    """
    # the path to voc dataset
    __root_voc = os.path.abspath(os.path.dirname(__file__) + "/../data/VOCdevkit/VOC2007")
    # the path to voc test dataset
    __root_voc_test = os.path.abspath(os.path.dirname(__file__) + "/../data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007")
    # the path to store data process result
    __root_tmp = os.path.abspath(os.path.dirname(__file__) + "/../data")

    # image id list
    # train image list
    image_id_list_train = __root_voc + '/ImageSets/Main/train.txt'
    # val image list
    image_id_list_val = __root_voc + '/ImageSets/Main/val.txt'
    # test image list
    image_id_list_test = __root_voc_test + '/ImageSets/Main/test.txt'
    # for each class
    image_id_list_test_all_class = __root_voc_test + '/ImageSets/Main/'

    # jpeg root path
    jpeg = __root_voc + '/JPEGImages'
    # test image path
    jpeg_test = __root_voc_test + '/JPEGImages'

    # annotation path
    annotation = __root_voc + '/Annotations'
    # test annotation path
    annotation_test = __root_voc_test + '/Annotations'

    # the root path to store selective search result
    ss_root = __root_tmp + '/ss_result'
    # finetune dataset
    finetune = __root_tmp + '/finetune'
    # svm dataset
    svm = __root_tmp + '/svm'
    # regression dataset
    regression = __root_tmp + '/regression'


class Model:
    """
    Model
    """
    # base path
    __root = os.path.dirname(__file__) + "/../models"
    # pre trained model
    pretrained_alexnet = __root + '/checkpoint_alexnet-150_625.ckpt'
    # the finetune ckpt used for svm and regression
    finetune = __root + '/finetune_best.ckpt'
    # the svm ckpt used for eval.py
    svm = __root + '/svm_best.ckpt'
    # the regression ckpt used for eval.py
    regression = __root + '/regression_best.ckpt'
    # the path to store generated ckpt
    save_path = __root + "/"


class Log:
    """
    Log
    """
    root = os.path.dirname(__file__) + "/../logs"
