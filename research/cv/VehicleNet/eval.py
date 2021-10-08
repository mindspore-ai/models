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
################################eval vehiclenet################################
python eval.py
"""
import ast
import os
import time
import argparse
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from src.config import common_config, VeRi_test
from src.VehicleNet_resnet50 import VehicleNet
from src.dataset import data_to_mindrecord, create_vehiclenet_dataset
from src.re_ranking import re_ranking

set_seed(1)

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

def calculate_result_rerank(test_feature_, test_label_, test_camera_, query_feature_, query_label_, query_camera_, k1=100, k2=15, lambda_value=0):
    """accuracy calculation
    """
    CMC = np.zeros((len(test_label_)), dtype=float)
    AP = 0.0

    since = time.time()
    q_t_dist = np.matmul(query_feature_, test_feature_.transpose((1, 0)))
    q_q_dist = np.matmul(query_feature_, query_feature_.transpose((1, 0)))
    t_t_dist = np.matmul(test_feature_, test_feature_.transpose((1, 0)))

    re_rank = re_ranking(q_t_dist, q_q_dist, t_t_dist, k1, k2, lambda_value)
    time_elapsed = time.time() - since
    print('Reranking complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    for i in range(len(query_label_)):
        AP_tmp, CMC_tmp = evaluate(re_rank[i, :], query_label_[i], query_camera_[i], test_label_, test_camera_)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        AP += AP_tmp

    CMC = CMC / len(query_label_)
    str_result = 'Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f\n' % (CMC[0], CMC[4], CMC[9], AP / len(query_label_))
    print(str_result)

def evaluate(score, query_label_, query_camera_, test_label_, test_camera_):
    """evaluate
    """
    index = np.argsort(score)

    query_index = np.argwhere(test_label_ == query_label_)
    camera_index = np.argwhere(test_camera_ == query_camera_)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(test_label_ == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)

    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    """compute mAP
    """
    ap = 0
    cmc = np.zeros((len(index)), dtype=int)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vehiclenet eval')
    parser.add_argument('--device_id', type=int, default=None, help='device id of GPU or Ascend. (Default: None)')
    parser.add_argument('--ckpt_url', type=str, default=None, help='Checkpoint file path')
    parser.add_argument('--is_modelarts', type=ast.literal_eval, default=False, help='Train in Modelarts.')
    parser.add_argument('--multiple_scale', type=str, default='1', help='mutiple scale')
    parser.add_argument('--data_url', default=None, help='Location of data.')
    parser.add_argument('--train_url', default=None, help='Location of training outputs.')
    args_opt = parser.parse_args()

    cfg = common_config
    VeRi_cfg = VeRi_test

    device_target = cfg.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    if device_target == "Ascend":
        if args_opt.device_id is not None:
            context.set_context(device_id=args_opt.device_id)
        else:
            context.set_context(device_id=cfg.device_id)
    else:
        raise ValueError("Unsupported platform.")

    eval_dataset_path = cfg.dataset_path
    if args_opt.is_modelarts:
        import moxing as mox
        mox.file.copy_parallel(src_url=args_opt.data_url,
                               dst_url='/cache/dataset_train/device_' + os.getenv('DEVICE_ID'))
        zip_command = "unzip -o /cache/dataset_train/device_" + os.getenv('DEVICE_ID') \
                       + "/VehicleNet_mindrecord.zip -d /cache/dataset_train/device_" + os.getenv('DEVICE_ID')
        os.system(zip_command)
        eval_dataset_path = '/cache/dataset_train/device_' + os.getenv('DEVICE_ID') + '/VehicleNet/'

    mindrecord_dir = cfg.mindrecord_dir
    prefix = "test_VehicleNet.mindrecord"
    test_mindrecord_file = os.path.join(mindrecord_dir, prefix)
    if not os.path.exists(test_mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        print("Create mindrecord for test.")
        data_to_mindrecord(eval_dataset_path, False, False, True, test_mindrecord_file)
        print("Create mindrecord done, at {}".format(mindrecord_dir))
    while not os.path.exists(test_mindrecord_file + ".db"):
        time.sleep(5)

    prefix = "query_VehicleNet.mindrecord"
    query_mindrecord_file = os.path.join(mindrecord_dir, prefix)
    if not os.path.exists(query_mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        print("Create mindrecord for query.")
        data_to_mindrecord(eval_dataset_path, False, False, False, query_mindrecord_file)
        print("Create mindrecord done, at {}".format(mindrecord_dir))
    while not os.path.exists(query_mindrecord_file + ".db"):
        time.sleep(5)

    test_dataset = create_vehiclenet_dataset(test_mindrecord_file, batch_size=1, device_num=1, is_training=False)
    query_dataset = create_vehiclenet_dataset(query_mindrecord_file, batch_size=1, device_num=1, is_training=False)
    test_data_num = test_dataset.get_dataset_size()
    query_data_num = query_dataset.get_dataset_size()

    net = VehicleNet(class_num=VeRi_cfg.num_classes)
    net.classifier.classifier = nn.SequentialCell()

    param_dict = load_checkpoint(args_opt.ckpt_url)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    test_feature, test_label, test_camera = extract_feature(net, test_dataset)
    query_feature, query_label, query_camera = extract_feature(net, query_dataset)

    calculate_result_rerank(test_feature, test_label, test_camera, query_feature, query_label, query_camera)
