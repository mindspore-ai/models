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
""" Evaluation script """

import os
import time

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from metric_utils.distance import compute_dist
from metric_utils.metric import cmc, mean_ap
from metric_utils.re_ranking import re_ranking
from model_utils.config import get_config
from model_utils.device_adapter import get_device_id, get_device_num
from model_utils.moxing_adapter import moxing_wrapper
from src.aligned_reid import AlignedReID
from src.dataset import create_dataset

set_seed(1)
config = get_config()


def eval_map_cmc(
        q_g_dist,
        q_ids=None,
        g_ids=None,
        q_cams=None,
        g_cams=None,
        separate_camera_set=None,
        single_gallery_shot=None,
        first_match_break=None,
        topk=None,
):
    """ Compute CMC and mAP

    Args:
        q_g_dist: numpy array with shape [num_query, num_gallery], the
        pairwise distance between query and gallery samples
        q_ids: query ids
        g_ids: gallery ids
        q_cams: query cams
        g_cams: gallery cams
        separate_camera_set: should separate cameras
        single_gallery_shot: choose one instance for each id
        first_match_break: use only first match
        topk: search in top k results

    Returns:
      mAP: numpy array with shape [num_query], the AP averaged across query
        samples
      cmc_scores: numpy array with shape [topk], the cmc curve
        averaged across query samples

    """
    # Compute mean AP
    mAP = mean_ap(
        distmat=q_g_dist,
        query_ids=q_ids, gallery_ids=g_ids,
        query_cams=q_cams, gallery_cams=g_cams)

    # Compute CMC scores
    cmc_scores = cmc(
        distmat=q_g_dist,
        query_ids=q_ids, gallery_ids=g_ids,
        query_cams=q_cams, gallery_cams=g_cams,
        separate_camera_set=separate_camera_set,
        single_gallery_shot=single_gallery_shot,
        first_match_break=first_match_break,
        topk=topk)
    print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'
          .format(mAP, *cmc_scores[[0, 4, 9]]))
    return mAP, cmc_scores


def modelarts_pre_process():
    """ Modelarts pre process function """
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
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
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if config.device_target == "GPU":
            init()
            device_id = get_rank()
            device_num = get_group_size()
        elif config.device_target == "Ascend":
            device_id = get_device_id()
            device_num = get_device_num()
        else:
            raise ValueError("Not support device_target.")

        # Each server contains 8 devices as most.
        if device_id % min(device_num, 8) == 0 and not os.path.exists(sync_lock):
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

        print("Device: {}, Finish sync unzip data from {} to {}.".format(device_id, zip_file_1, save_dir_1))

    config.log_path = os.path.join(config.output_path, config.log_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    """ Run evaluation """
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

    dataset, n_classes, _ = create_dataset(
        config.data_dir,
        config.partitions_file,
        ims_per_id=config.ims_per_id,
        ids_per_batch=config.ids_per_batch,
        batch_size=config.per_batch_size,
        mean=config.image_mean,
        std=config.image_std,
        resize_h_w=config.image_size,
        istrain=False,
        return_len=True,
    )

    network = AlignedReID(num_classes=n_classes)

    # pre_trained
    if config.eval_model:
        print('Load model from', config.eval_model)
        ret = load_param_into_net(network, load_checkpoint(config.eval_model))
        print(ret)
    else:
        print('PRETRAINED MODEL NOT SELECTED!!!')

    data_loader = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)

    id_id = 0
    cam_id = 1
    mark_id = 2

    marks_allowed = (0, 1)

    lst_labels = []
    lst_g_feats = []

    for data in data_loader:
        images_ = data["image"]
        labels = data['label']
        ixs_marks = np.in1d(labels[:, mark_id], marks_allowed)

        if ixs_marks.sum() == 0:
            continue

        images_ = images_[ixs_marks]
        labels = labels[ixs_marks]
        images = Tensor.from_numpy(images_)

        global_feat, _, _ = network(images)

        lst_labels.append(labels)
        lst_g_feats.append(global_feat.asnumpy())

    global_feats = np.concatenate(lst_g_feats, axis=0)
    labels = np.concatenate(lst_labels, axis=0)

    q_inds = labels[:, mark_id] == 0
    g_inds = labels[:, mark_id] == 1

    global_q_g_dist = compute_dist(global_feats[q_inds], global_feats[g_inds], metric='euclidean')
    ids = labels[:, id_id]
    cams = labels[:, cam_id]

    eval_map_cmc(
        q_g_dist=global_q_g_dist,
        q_ids=ids[q_inds], g_ids=ids[g_inds],
        q_cams=cams[q_inds], g_cams=cams[g_inds],
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=True,
        topk=10,
    )

    # Re rank
    global_q_q_dist = compute_dist(
        global_feats[q_inds], global_feats[q_inds], metric='euclidean')

    # gallery-gallery distance using global distance
    global_g_g_dist = compute_dist(
        global_feats[g_inds], global_feats[g_inds], metric='euclidean')

    re_r_global_q_g_dist = re_ranking(global_q_g_dist, global_q_q_dist, global_g_g_dist)

    print('After Re-ranking')
    eval_map_cmc(
        q_g_dist=re_r_global_q_g_dist,
        q_ids=ids[q_inds], g_ids=ids[g_inds],
        q_cams=cams[q_inds], g_cams=cams[g_inds],
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=True,
        topk=10,
    )


if __name__ == '__main__':
    run_eval()
