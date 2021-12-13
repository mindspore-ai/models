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

"""creat_mindrecord"""

from mindspore import context

from src.dataset_train import DataIterator, create_mindrecord
from src.dataset_test import DataIterator as DI_test
from src.config import parse_args

args_opt = parse_args()

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    device_id = args_opt.device_id
    context.set_context(device_id=device_id)

    dataset_file_path = args_opt.dataset_file_path
    test_file = os.path.join(dataset_file_path, "local_test_splitByUser")
    uid_voc = os.path.join(dataset_file_path, "uid_voc.pkl")
    mid_voc = os.path.join(dataset_file_path, "mid_voc.pkl")
    cat_voc = os.path.join(dataset_file_path, "cat_voc.pkl")
    meta_path = os.path.join(dataset_file_path, "item-info")
    review_path = os.path.join(dataset_file_path, "reviews-info")
    batch_size = args_opt.batch_size
    maxlen = args_opt.max_len
    train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, meta_path, review_path)
    test_data = DI_test(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, meta_path, review_path)

    # Convert books train dataset to mindrecord format
    train_mindrecord_file = args_opt.train_mindrecord_path
    create_mindrecord(train_data, maxlen, train_mindrecord_file)

    # Convert books test dataset to mindrecord format
    test_mindrecord_file = args_opt.test_mindrecord_path
    create_mindrecord(test_data, maxlen, test_mindrecord_file)
