# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""evaluate osnet."""

import time
import os
import numpy as np


import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import set_seed
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net


from src.dataset import dataset_creator
from src.osnet import create_osnet
from model_utils.config import config
from model_utils.metrics import distance, rank
from model_utils.device_adapter import get_device_id, get_device_num
from model_utils.moxing_adapter import moxing_wrapper


set_seed(1)

class CustomWithEvalCell(nn.Cell):
    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self._network = network

    def construct(self, data):
        outputs = self._network(data)
        return outputs


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.target)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.target + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))


@moxing_wrapper(pre_process=modelarts_pre_process)
def eval_net(net=None):
    '''evaluate osnet.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)

    num_train_classes, query_dataset = dataset_creator(root=config.data_path, height=config.height, width=config.width,
                                                       dataset=config.target, norm_mean=config.norm_mean,
                                                       norm_std=config.norm_std, batch_size_test=config.batch_size_test,
                                                       workers=config.workers, cuhk03_labeled=config.cuhk03_labeled,
                                                       cuhk03_classic_split=config.cuhk03_classic_split, mode='query')
    num_train_classes, gallery_dataset = dataset_creator(root=config.data_path, height=config.height,
                                                         width=config.width, dataset=config.target,
                                                         norm_mean=config.norm_mean, norm_std=config.norm_std,
                                                         batch_size_test=config.batch_size_test, workers=config.workers,
                                                         cuhk03_labeled=config.cuhk03_labeled,
                                                         cuhk03_classic_split=config.cuhk03_classic_split,
                                                         mode='gallery')
    if net is None:
        net = create_osnet(num_train_classes)
        param_dict = load_checkpoint(config.checkpoint_file_path, filter_prefix='epoch_num')
        load_param_into_net(net, param_dict)

    net.set_train(False)
    net_eval = CustomWithEvalCell(net)

    def feature_extraction(eval_dataset):
        f_, pids_, camids_ = [], [], []
        for data in eval_dataset.create_dict_iterator():
            imgs, pids, camids = data['img'], data['pid'], data['camid']
            features = net_eval(imgs)
            f_.append(features)
            pids_.extend(pids.asnumpy())
            camids_.extend(camids.asnumpy())
        concat = ops.Concat(axis=0)
        f_ = concat(f_)
        pids_ = np.asarray(pids_)
        camids_ = np.asarray(camids_)
        return f_, pids_, camids_

    print('Extracting features from query set ...')
    qf, q_pids, q_camids = feature_extraction(query_dataset)
    print('Done, obtained {}-by-{} matrix'.format(qf.shape[0], qf.shape[1]))

    print('Extracting features from gallery set ...')
    gf, g_pids, g_camids = feature_extraction(gallery_dataset)
    print('Done, obtained {}-by-{} matrix'.format(gf.shape[0], gf.shape[1]))

    if config.normalize_feature:
        l2_normalize = ops.L2Normalize(axis=1)
        qf = l2_normalize(qf)
        gf = l2_normalize(gf)

    print('Computing distance matrix with metric={} ...'.format(config.dist_metric))
    distmat = distance.compute_distance_matrix(qf, gf, config.dist_metric)
    distmat = distmat.asnumpy()

    print('Computing CMC and mAP ...')
    cmc, mAP = rank.evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        use_metric_cuhk03=config.use_metric_cuhk03
    )

    print('** Results **')
    print('ckpt={}'.format(config.checkpoint_file_path))
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    ranks = [1, 5, 10, 20]
    i = 0
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[i]))
        i += 1

if __name__ == '__main__':
    if config.target == 'msmt17' or config.target == 'cuhk03 ':
        config.dist_metric = 'cosine'
    else:
        config.dist_metric = 'euclidean'
    print(config.dist_metric)
    eval_net()
