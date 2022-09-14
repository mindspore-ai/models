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
"""
######################## eval net ########################
"""
import os
import numpy as np
from mindspore import Model
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from sklearn.metrics import cohen_kappa_score
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.data_preprocess import create_dataset
from src.models_lw_3D import dict_lwnet
from src.loss import NllLoss


def modelarts_process():
    config.ckpt_path = config.ckpt_file


def set_parameter():
    """set_parameter"""
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)


def predict(model, val_loader):
    pre_label = np.array([])
    tar_label = np.array([])
    for data, label in val_loader:
        # predict model
        output = model.predict(data)
        pred = np.argmax(output.asnumpy(), axis=1)  # get the index of the max log-probability
        pre_label = np.append(pre_label, pred)
        tar_label = np.append(tar_label, label.asnumpy())
    return pre_label, tar_label


def oa_aa_k_cal(pre_label, tar_label):
    """
    OA, AA, K calculation
    """
    acc = []
    samples_num = len(tar_label)
    category_num = tar_label.max() + 1
    category_num = category_num.astype(int)
    for i in range(category_num):
        loc_i = np.where(tar_label == i)
        oa_i = np.array(pre_label[loc_i] == tar_label[loc_i], np.float32).sum() / len(loc_i[0])
        acc.append(oa_i)
    oa = np.array(pre_label == tar_label, np.float32).sum() / samples_num
    aa = np.average(np.array(acc))
    # c_matrix = confusion_matrix(tar_label, pre_label)
    # K = (samples_num*c_matrix.diagonal().sum())/(samples_num*samples_num - np.dot(sum(c_matrix,0), sum(c_matrix,1)))
    k = cohen_kappa_score(tar_label, pre_label)
    acc.append(oa)
    acc.append(aa)
    acc.append(k)
    return np.array(acc)


@moxing_wrapper(pre_process=modelarts_process)
def eval_net():
    """eval net"""
    set_parameter()

    data_dir = './data_list/{}_test.txt'.format(config.dataset_name)
    dataset = create_dataset(config, data_dir, do_train=False)

    model = dict_lwnet()[config.model_name](num_classes=config.class_num, dropout_keep_prob=0)

    loss = NllLoss(reduction="mean", num_classes=config.class_num)

    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(model, param_dict)

    net = Model(model, loss_fn=loss, metrics={'top_1_accuracy'})
    pre_gt, tar_gt = predict(net, dataset)

    acc = oa_aa_k_cal(pre_gt, tar_gt)

    for i in range(config.class_num+1):
        if i == config.class_num:
            print('OA: {:.4f}, AA: {:.4f}, K: {:.4f}'.format(acc[i], acc[i+1], acc[i+2]))
        else:
            print('类别{}: {:.4f}'.format(i+1, acc[i]))


if __name__ == "__main__":
    eval_net()
