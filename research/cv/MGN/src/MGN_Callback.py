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
"""This is callback program"""
import os
from src.dataset import create_dataset
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import numpy as mnp
from mindspore.common import set_seed
from mindspore.train.callback import Callback
from mindspore.train.serialization import save_checkpoint
from scipy.spatial.distance import cdist
from metric_utils.functions import cmc, mean_ap
from metric_utils.re_ranking import re_ranking
from model_utils.config import get_config
from model_utils.device_adapter import get_device_id

set_seed(1)
config = get_config()


def extract_feature(model, dataset):
    """ Extract dataset features from model """
    def fliphor(tensor):
        """ Flip tensor """
        return tensor[..., ::-1].copy()

    data_loader = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)

    features = []

    for data in data_loader:
        images_ = data["image"]

        ff = mnp.zeros((images_.shape[0], 2048))
        for i in range(2):
            if i == 1:
                images_ = fliphor(images_)
            images = Tensor.from_numpy(images_)
            outputs = model(images)
            f = outputs[0]
            ff = ff + f

        fnorm = mnp.sqrt((ff ** 2).sum(axis=1, keepdims=True))
        ff = ff / fnorm.expand_as(ff)

        features.append(ff.asnumpy())

    return np.concatenate(features, axis=0)


class mgn_callback(Callback):
    def __init__(self, network):
        self.net = network
        self.rank1 = 0.1
        self.best_map = 0
        config.image_size = list(map(int, config.image_size.split(',')))
        config.image_mean = list(map(float, config.image_mean.split(',')))
        config.image_std = list(map(float, config.image_std.split(',')))

        _enable_graph_kernel = False
        context.set_context(
            mode=context.GRAPH_MODE,
            enable_graph_kernel=_enable_graph_kernel,
            device_target=config.device_target,
        )

        config.rank = 0
        config.device_id = get_device_id()
        config.group_size = 1

        _enable_graph_kernel = False
        context.set_context(
            mode=context.GRAPH_MODE,
            enable_graph_kernel=_enable_graph_kernel,
            device_target=config.device_target,
        )

        config.rank = 0
        config.device_id = get_device_id()
        config.group_size = 1

        self.t_dataset, self.t_cams, self.t_ids = create_dataset(
            config.data_dir,
            ims_per_id=4,
            ids_per_batch=12,
            mean=config.image_mean,
            std=config.image_std,
            resize_h_w=config.image_size,
            batch_size=config.per_batch_size,
            rank=config.rank,
            group_size=config.group_size,
            data_part='test'
        )

        self.q_dataset, self.q_cams, self.q_ids = create_dataset(
            config.data_dir,
            ims_per_id=4,
            ids_per_batch=12,
            mean=config.image_mean,
            std=config.image_std,
            resize_h_w=config.image_size,
            batch_size=config.per_batch_size,
            rank=config.rank,
            group_size=config.group_size,
            data_part='query'
        )

    def epoch_end(self, run_context):
        # print(self.net.trainable_params()[0].data.asnumpy()[0][0])

        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % 10 == 0 or cur_epoch > config.max_epoch*0.8:
            self.net.set_train(False)

            re_rank = True
            network = self.net
            gf = extract_feature(network, self.t_dataset)
            print('Got gallery features')
            qf = extract_feature(network, self.q_dataset)
            print('Got query features')

            if re_rank:
                q_g_dist = np.dot(qf, np.transpose(gf))
                q_q_dist = np.dot(qf, np.transpose(qf))
                g_g_dist = np.dot(gf, np.transpose(gf))
                dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
            else:
                dist = cdist(qf, gf)
            r = cmc(dist, self.q_ids, self.t_ids, self.q_cams, self.t_cams,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.q_ids, self.t_ids, self.q_cams, self.t_cams)
            map_score = np.float32(m_ap)

            print(
                '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'.format(
                    m_ap,
                    r[0], r[2], r[4], r[9],
                )
            )

            if self.best_map < map_score:
                save_checkpoint(self.net, os.path.join(config.ckpt_path, 'best.ckpt'))
                self.best_map = map_score
            self.net.set_train(True)
        if cur_epoch == config.max_epoch:
            print("best score: mAP:", self.best_map)
