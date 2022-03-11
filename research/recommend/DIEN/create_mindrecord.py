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
'''
create dataset with mindrecord format
'''
import os
from mindspore import context
from src.dataset import DataIterator, create_mindrecord
from src.config import parse_args

args_opt = parse_args()

if __name__ == '__main__':
    output_path = args_opt.mindrecord_path
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    device_id = args_opt.device_id
    context.set_context(device_id=device_id)

    dataset_file_path = args_opt.dataset_file_path
    train_file = os.path.join(dataset_file_path, "local_train_splitByUser")
    test_file = os.path.join(dataset_file_path, "local_test_splitByUser")
    uid_voc = os.path.join(dataset_file_path, "uid_voc.pkl")
    mid_voc = os.path.join(dataset_file_path, "mid_voc.pkl")
    cat_voc = os.path.join(dataset_file_path, "cat_voc.pkl")
    meta_path = os.path.join(dataset_file_path, "item-info")
    review_path = os.path.join(dataset_file_path, "reviews-info")
    batch_size = args_opt.batch_size
    maxlen = args_opt.max_len
    train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, meta_path, review_path, batch_size, maxlen,
                              shuffle_each_epoch=False)

    # Convert books train dataset to mindrecord format
    train_name = "{0}_train.mindrecord".format(args_opt.dataset_type)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    MINDRECORD_TRAIN_FILE = os.path.join(output_path, train_name)

    create_mindrecord(train_data, maxlen, MINDRECORD_TRAIN_FILE, mode="train")

    test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, meta_path, review_path, batch_size, maxlen)
    # Convert books test dataset to mindrecord format

    test_name = "{0}_test.mindrecord".format(args_opt.dataset_type)
    MINDRECORD_TEST_FILE = os.path.join(output_path, test_name)
    create_mindrecord(test_data, maxlen, MINDRECORD_TEST_FILE)
