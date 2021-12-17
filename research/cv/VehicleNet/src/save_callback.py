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
"""infer while train"""
import time
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import save_checkpoint
from mindspore.train.callback import Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.re_ranking import re_ranking

def fliplr(x):
    """flip horizontally
    """
    for i in range(x.shape[0]):
        x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x

def extract_feature(model, dataset):
    """feature extract
    """
    norm = nn.Norm(axis=1, keep_dims=True)
    div = ops.Div()

    image_size = dataset.get_dataset_size()
    features = np.zeros((image_size, 512), dtype=float)
    label = []
    camera = []

    for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=False)):
        img = data['image']
        label.append(data['label'].asnumpy()[0])
        camera.append(data['camera'].asnumpy()[0])

        n = img.shape[0]
        ff = Tensor(np.zeros((n, 512)), mindspore.float32)
        for i in range(2):
            if i == 1:
                img = img.asnumpy()
                img = fliplr(img)
                img = Tensor.from_numpy(img)

            outputs = model(img)
            ff += outputs

        fnorm = norm(ff)
        ff = div(ff, fnorm.expand_as(ff))
        features[idx] = ff.asnumpy()

    return features, label, camera

def calculate_result_rerank(test_feature, test_label, test_camera, query_feature, query_label, query_camera, k1=100, k2=15, lambda_value=0):
    """accuracy calculation
    """
    CMC = np.zeros((len(test_label)), dtype=float)
    AP = 0.0

    since = time.time()
    q_t_dist = np.matmul(query_feature, test_feature.transpose((1, 0)))
    q_q_dist = np.matmul(query_feature, query_feature.transpose((1, 0)))
    t_t_dist = np.matmul(test_feature, test_feature.transpose((1, 0)))

    re_rank = re_ranking(q_t_dist, q_q_dist, t_t_dist, k1, k2, lambda_value)
    time_elapsed = time.time() - since
    print('Reranking complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    for i in range(len(query_label)):
        AP_tmp, CMC_tmp = evaluate(re_rank[i, :], query_label[i], query_camera[i], test_label, test_camera)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        AP += AP_tmp

    CMC = CMC / len(query_label)
    str_result = 'Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f\n' % (CMC[0], CMC[4], CMC[9], AP / len(query_label))
    print(str_result)


def evaluate(score, query_label, query_camera, test_label, test_camera):
    """evaluate
    """
    index = np.argsort(score)  # from small to large (1 * 11579)

    query_index = np.argwhere(test_label == query_label)  # test_label: == query_label index (1 * n1)
    camera_index = np.argwhere(test_camera == query_camera)  # test_camera: == query_camera index (1 * n2)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)  # in query_index, not in camera_index
    junk_index1 = np.argwhere(test_label == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)  # in query_index, and in camera_index
    junk_index = np.append(junk_index2, junk_index1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)

    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    """compute mAP
    """
    ap = 0
    cmc = np.zeros((len(index)), dtype=int)  # 11579

    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)  # different is True, same is False
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)  # different is False, same is True
    rows_good = np.argwhere(np.equal(mask, True))
    rows_good = rows_good.flatten()

    for i in range(len(cmc)):
        if i >= rows_good[0]:
            cmc[i] = 1

    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

def get_base_param(load_ckpt_path):
    par_dict = load_checkpoint(load_ckpt_path)
    new_params_dict = {}
    for name in par_dict:
        if 'classifier' not in name:
            new_params_dict[name] = par_dict[name]
    return new_params_dict

class SaveCallback(Callback):
    """SaveCallback"""
    def __init__(self, net, test_dataset, query_dataset, epoch_per_eval, cfg):
        super(SaveCallback, self).__init__()
        self.net = net
        self.test_dataset = test_dataset
        self.query_dataset = query_dataset
        self.save_path = cfg.checkpoint_dir
        self.epoch_per_eval = epoch_per_eval
        self.cfg = cfg

    def epoch_end(self, run_context):
        """epoch_end"""
        t1 = time.time()

        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num

        if cur_epoch % self.epoch_per_eval == 0:
            file_name = self.save_path + str(cur_epoch) + ".ckpt"
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)

            param_dict = get_base_param(file_name)
            load_param_into_net(self.net, param_dict)

            test_feature, test_label, test_camera = extract_feature(self.net, self.test_dataset)
            query_feature, query_label, query_camera = extract_feature(self.net, self.query_dataset)

            calculate_result_rerank(test_feature, test_label, test_camera, query_feature, query_label, query_camera)

            t2 = time.time()
            print("Eval in training Time consume: ", t2 - t1, "\n")
