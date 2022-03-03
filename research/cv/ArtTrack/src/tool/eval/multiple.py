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
import os
import json
from json import encoder

import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np
from scipy import io as sio

from src.dataset.coco import MSCOCO
from src.dataset.pose import Batch
from src.log import log
from src.model.pose import PoseNet, PoseNetTest
from src.model.predict import argmax_arrows_predict, argmax_pose_predict, extract_cnn_output, pose_predict_with_gt_segm
from src.multiperson import visualize
from src.multiperson.detections import extract_detections
from src.multiperson.predict import eval_graph, get_person_conf_multicut, SpatialModel
from src.multiperson.visualize import PersonDraw
from src.tool.decorator import process_cfg

encoder.FLOAT_REPR = lambda o: format(o, '.2f')


def test_one(cfg, test_net, sample, score_maps_cached, cache_score_maps, visual, dataset, sm, graph=False):
    """
    predict one sample
    Args:
        cfg: config
        test_net: eval net
        sample: sample
        cache_score_maps: if True, cache score maps to scoremap_dir in cfg
        visual: if True, visualize prediction
        score_maps_cached: if True, load score from cache
        graph: if True, calculate graph
        dataset: dataset object
        sm: spatial model
    """
    coco_results = []
    draw_multi = PersonDraw()
    cache_name = "{}.mat".format(sample[Batch.data_item].coco_id)
    if not score_maps_cached:
        outputs_np, pairwise_pred, locref = test_net(
            ms.Tensor(np.expand_dims(sample[Batch.inputs], axis=0),
                      dtype=ms.dtype.float32))
        scmap, locref, pairwise_diff = extract_cnn_output(outputs_np.transpose([0, 2, 3, 1]).asnumpy(),
                                                          locref.transpose([0, 2, 3, 1]).asnumpy(),
                                                          pairwise_pred.transpose([0, 2, 3, 1]).asnumpy(),
                                                          cfg, dataset.pairwise_stats)

        if cache_score_maps:
            out_fn = os.path.join(cfg.scoremap_dir, cache_name)
            d = {'scoremaps': scmap.astype('float32'),
                 'locreg_pred': locref.astype('float32'),
                 'pairwise_diff': pairwise_diff.astype('float32')}
            sio.savemat(out_fn, mdict=d)
    else:
        # cache_name = '1.mat'
        full_fn = os.path.join(cfg.cached_scoremaps, cache_name)
        mlab = sio.loadmat(full_fn)
        scmap = mlab["scoremaps"]
        locref = mlab["locreg_pred"]
        pairwise_diff = mlab["pairwise_diff"]

    person_conf_multi = None
    if graph:
        detections = extract_detections(cfg, scmap, locref, pairwise_diff)
        unLab, pos_array, unary_array, _, _ = eval_graph(sm, detections)
        person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)

    coco_img_results = None
    if cfg.use_gt_segm:
        coco_img_results = pose_predict_with_gt_segm(scmap, locref, cfg.stride, sample[Batch.data_item].gt_segm,
                                                     sample[Batch.data_item].coco_id)
        coco_results += coco_img_results

    if visual:
        img = np.transpose(np.squeeze(sample[Batch.inputs]).astype('uint8'), [1, 2, 0])
        pose = argmax_pose_predict(scmap, locref, cfg.stride)
        arrows = argmax_arrows_predict(scmap, locref, pairwise_diff, cfg.stride)
        visualize.show_arrows(cfg, img, pose, arrows)
        visualize.waitforbuttonpress()
        # visualize.show_heatmaps(cfg, img, scmap, pose)

        # visualize part detections after NMS
        # visim_dets = visualize_detections(cfg, img, detections)
        # plt.imshow(visim_dets)
        # plt.show()
        # visualize.waitforbuttonpress()

        if person_conf_multi is not None and graph:
            visim_multi = img.copy()
            draw_multi.draw(visim_multi, dataset, person_conf_multi)
            plt.imshow(visim_multi)
            plt.show()

        if coco_img_results is not None and coco_img_results:
            dataset.visualize_coco(coco_img_results, sample[Batch.data_item].visibilities)
        visualize.waitforbuttonpress()
    return coco_results


def test_list(cfg, test_net, idx_list, dataset, score_maps_cached, cache_score_maps, visual, sm, graph):
    """
     predict multiple sample
     Args:
         cfg: config
         test_net: eval net
         idx_list: sample indices
         cache_score_maps: if True, cache score maps to scoremap_dir in cfg
         visual: if True, visualize prediction
         score_maps_cached: if True, load score from cache
         graph: if True, calculate graph
         dataset: dataset object
         sm: spatial model
     """
    coco_results = []
    count = 0
    total = len(idx_list)
    for k in idx_list:
        count += 1
        log.info('processing image id: %s %s/%s', k, count, total)

        batch = dataset.get_item(k)
        result = test_one(cfg, test_net, batch, score_maps_cached, cache_score_maps, visual, dataset, sm, graph)
        if result is not None:
            coco_results.extend(result)
    return coco_results


@process_cfg
def test(cfg, cache_score_maps=False, visual=False, development=False, score_maps_cached=False, graph=False,
         output=None, range_num=None, range_index=None):
    """
    entry for predicting multiple coco
    Args:
        cfg: config
        cache_score_maps: if True, cache score maps to scoremap_dir in cfg
        visual: if True, visualize prediction
        development: development mode. only predict head 10 samples
        score_maps_cached: if True, load score from cache
        graph: if True, calculate graph
        output: path to output
        range_num: split eval dataset to multiple range. must work with range_index
        range_index: only predict specified range. start is 0
    """
    cfg.train = False
    # noinspection PyUnresolvedReferences
    ms.context.set_context(**cfg.context)
    dataset = MSCOCO(cfg)

    sm = SpatialModel(cfg)
    sm.load()

    net = PoseNet(cfg=cfg)
    test_net = PoseNetTest(net, cfg)
    if hasattr(cfg, 'load_ckpt') and os.path.exists(cfg.load_ckpt):
        ms.load_checkpoint(cfg.load_ckpt, net=test_net)

    if cache_score_maps:
        out_dir = cfg.scoremap_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    num_images = len(dataset) if not development else min(10, dataset.num_images)
    coco_results = []
    if range_num is None or range_num == 1 or range_index is None:
        coco_results.extend(
            test_list(cfg, test_net, range(num_images), dataset, score_maps_cached, cache_score_maps, visual,
                      sm, graph))
    else:
        lists = np.array_split(range(num_images), range_num)
        coco_results.extend(
            test_list(cfg, test_net, lists[range_index], dataset, score_maps_cached, cache_score_maps, visual,
                      sm, graph))
    if cfg.use_gt_segm:
        with open(output or cfg.gt_segm_output, 'w') as outfile:
            json.dump(coco_results, outfile)
