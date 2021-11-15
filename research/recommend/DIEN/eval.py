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

"""eval"""
import time
import os

from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
import mindspore.dataset as ds

from src.dataset_test import DataIterator
from src.model import DIEN, Accuracy
from src.utils import calc_auc
from src.config import parse_args

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0

args_opt = parse_args()


def test(ds_test, save_checkpoint_path,
         test_file,
         uid_voc,
         mid_voc,
         cat_voc,
         meta_path,
         review_path,
         batch_size,
         maxlen):
    """test data"""
    test_data = DataIterator(test_file, uid_voc, mid_voc,
                             cat_voc, meta_path, review_path, batch_size, maxlen)

    n_uid, n_mid, n_cat = test_data.get_n()
    model = DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    NetAcc = Accuracy()

    param_dict = load_checkpoint(save_checkpoint_path)
    load_param_into_net(model, param_dict)

    steps = 0
    stored_arr = []
    acc_sum = 0
    time_start = time.time()
    for d in ds_test.create_dict_iterator():
        y_hat, aux_loss = model(d['mid_mask'], d['uids'], d['mids'], d['cats'], d['mid_his'], d['cat_his'],
                                d['noclk_mids'], d['noclk_cats'])
        y_hat_1 = y_hat.asnumpy()
        aux_loss = aux_loss
        target_1 = d['target'].asnumpy()
        y_hat_1 = y_hat_1[:, 0].tolist()
        target_2 = target_1[:, 0].tolist()
        for y, t in zip(y_hat_1, target_2):
            stored_arr.append([y, t])
        acc = NetAcc(y_hat, d['target'])
        acc_sum += acc
        steps += 1
    time_end = time.time()
    test_acc = acc_sum / steps
    test_auc = calc_auc(stored_arr)
    print('acc:{0}  test_auc:{1}'.format(test_acc, test_auc))
    print('spend_time:', time_end - time_start)


def main():
    if args_opt.device_target == 'Ascend':
        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
        context.set_context(device_id=args_opt.device_id)

        dataset_file_path = args_opt.dataset_file_path
        test_file = os.path.join(dataset_file_path, "local_test_splitByUser")
        uid_voc = os.path.join(dataset_file_path, "uid_voc.pkl")
        mid_voc = os.path.join(dataset_file_path, "mid_voc.pkl")
        cat_voc = os.path.join(dataset_file_path, "cat_voc.pkl")
        meta_path = os.path.join(dataset_file_path, "item-info")
        review_path = os.path.join(dataset_file_path, "reviews-info")
        save_checkpoint_path = args_opt.save_checkpoint_path
        batch_size = args_opt.batch_size
        max_len = args_opt.max_len
        if args_opt.dataset_type == 'Books':
            test_mindrecord_path = os.path.join(args_opt.mindrecord_path, 'Books_test.mindrecord')
        elif args_opt.dataset_type == 'Electronics':
            test_mindrecord_path = os.path.join(args_opt.mindrecord_path, 'Electronics_test.mindrecord')
        ds_test = ds.MindDataset(test_mindrecord_path)
        test(ds_test, save_checkpoint_path, test_file, uid_voc, mid_voc, cat_voc, meta_path, review_path,
             batch_size=batch_size, maxlen=max_len)


if __name__ == '__main__':
    main()
