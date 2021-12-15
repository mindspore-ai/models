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

"""preprocess dataset"""
import os

from mindspore import context
import mindspore.dataset as ds

from src.config import parse_args

args_opt = parse_args()

if __name__ == "__main__":
    target = args_opt.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target)
    context.set_context(device_id=args_opt.device_id)
    if target == "Ascend":
        dataset_type = args_opt.dataset_type
        if dataset_type == 'Books':
            test_mindrecord_path = os.path.join(args_opt.mindrecord_path, 'Books_test.mindrecord')
        elif dataset_type == 'Electronics':
            test_mindrecord_path = os.path.join(args_opt.mindrecord_path, 'Electronics_test.mindrecord')

        ds_test = ds.MindDataset(test_mindrecord_path)

        dataset_path = os.path.join(args_opt.binary_files_path, dataset_type + "_data")
        mid_mask_path = os.path.join(dataset_path, "mid_mask")
        uids_path = os.path.join(dataset_path, "uids")
        mids_path = os.path.join(dataset_path, "mids")
        cats_path = os.path.join(dataset_path, "cats")
        mid_his_path = os.path.join(dataset_path, "mid_his")
        cat_his_path = os.path.join(dataset_path, "cat_his")
        noclk_mids_path = os.path.join(dataset_path, "noclk_mids")
        noclk_cats_path = os.path.join(dataset_path, "noclk_cats")
        target_path = os.path.join(dataset_path, 'target')
        if os.path.exists(dataset_path) is False:
            os.makedirs(dataset_path)
            os.makedirs(mid_mask_path)
            os.makedirs(uids_path)
            os.makedirs(mids_path)
            os.makedirs(cats_path)
            os.makedirs(mid_his_path)
            os.makedirs(cat_his_path)
            os.makedirs(noclk_mids_path)
            os.makedirs(noclk_cats_path)
            os.makedirs(target_path)
        for i, d in enumerate(ds_test.create_dict_iterator(output_numpy=True)):
            file_name = "DIEN_data_bs" + dataset_type + str(i) + ".bin"
            d['mid_mask'].tofile(os.path.join(mid_mask_path, file_name))
            d['uids'].tofile(os.path.join(uids_path, file_name))
            d['mids'].tofile(os.path.join(mids_path, file_name))
            d['cats'].tofile(os.path.join(cats_path, file_name))
            d['mid_his'].tofile(os.path.join(mid_his_path, file_name))
            d['cat_his'].tofile(os.path.join(cat_his_path, file_name))
            d['noclk_mids'].tofile(os.path.join(noclk_mids_path, file_name))
            d['noclk_cats'].tofile(os.path.join(noclk_cats_path, file_name))
            d['target'].tofile(os.path.join(target_path, file_name))
        print('=' * 20, "export bin files finished", '=' * 20)
