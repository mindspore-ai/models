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

"""export"""
import os
import numpy as np

from mindspore import Tensor, export, load_checkpoint
from mindspore import context

from src.dien import DIEN
from src.dataset import DataIterator
from src.config import parse_args

args_opt = parse_args()

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0


def export_DIEN(train_file, uid_voc, mid_voc, cat_voc, meta_path, review_path, batch_size, max_len, ckpt_path):
    """export"""

    train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, meta_path, review_path, batch_size, maxlen,
                              shuffle_each_epoch=False)

    n_uid, n_mid, n_cat = train_data.get_n()

    model = DIEN(n_uid, n_mid, n_cat, embedding_size=EMBEDDING_DIM)

    # load the parameter into net
    load_checkpoint(ckpt_path, net=model)
    uids = Tensor(np.random.rand(batch_size).astype(np.int32))
    mids = Tensor(np.random.rand(batch_size).astype(np.int32))
    cats = Tensor(np.random.rand(batch_size).astype(np.int32))
    mid_his = Tensor(np.random.rand(batch_size, max_len).astype(np.int32))
    cat_his = Tensor(np.random.rand(batch_size, max_len).astype(np.int32))
    mid_mask = Tensor(np.random.rand(batch_size, max_len).astype(np.int32))
    noclk_mids = Tensor(np.random.rand(batch_size, max_len, 5).astype(np.int32))
    noclk_cats = Tensor(np.random.rand(batch_size, max_len, 5).astype(np.int32))
    export(model, mid_mask, uids, mids, cats, mid_his, cat_his,
           noclk_mids, noclk_cats, file_name=args_opt.MINDIR_file_name, file_format='MINDIR')


def main():
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

    device_id = args_opt.device_id

    context.set_context(device_id=device_id)

    dataset_file_path = args_opt.dataset_file_path
    train_file = args_opt.train_mindrecord_path
    uid_voc = os.path.join(dataset_file_path, "uid_voc.pkl")
    mid_voc = os.path.join(dataset_file_path, "mid_voc.pkl")
    cat_voc = os.path.join(dataset_file_path, "cat_voc.pkl")
    meta_path = os.path.join(dataset_file_path, "item-info")
    review_path = os.path.join(dataset_file_path, "reviews-info")
    batch_size = args_opt.batch_size
    max_len = args_opt.max_len
    ckpt_path = args_opt.save_checkpoint_path
    export_DIEN(train_file, uid_voc, mid_voc, cat_voc, meta_path, review_path, batch_size, max_len, ckpt_path)
    print('export success!')


if __name__ == "__main__":
    main()
