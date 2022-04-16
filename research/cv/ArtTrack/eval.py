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
import sys

import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import context as ctx
import numpy as np
from scipy import io as sio
from tqdm import tqdm
import cv2

from config import check_config
from src.args_util import command, create_arg_parser, TARGET_COCO_MULTI, TARGET_MPII_SINGLE
from src.dataset.mpii import MPII
from src.dataset.pose import Batch
from src.log import log
from src.model.pose import PoseNet, PoseNetTest
from src.model.predict import argmax_pose_predict, extract_cnn_output
from src.multiperson.visualize import show_heatmaps
from src.tool.decorator import process_cfg
from src.tool.eval.pck import enclosing_rect, print_results, rect_size


@command
def test(parser, args, cfg):
    if args.target == TARGET_MPII_SINGLE:
        if args.accuracy:
            eval_mpii(cfg, args.prediction or args.output)
            return
        predict_mpii(cfg, args.visual, args.cache, args.output)
    elif args.target == TARGET_COCO_MULTI:
        if args.accuracy:
            from src.tool.eval.coco import eval_coco
            eval_coco(cfg, args.prediction)
            return
        from src.tool.eval.multiple import test as multiple_test
        multiple_test(cfg, args.cache, args.visual, args.dev,
                      args.score_maps_cached, args.graph, args.output, args.range_num, args.range_index)
    else:
        parser.print_help()

def reshape_image(cfg=None, batch=None):
    """
    reshape image
    """
    test_shape = (cfg.image_width, cfg.image_height)
    img = batch[Batch.inputs].transpose([1, 2, 0])
    img_shape = img.shape
    ratio = (test_shape[0] / img_shape[1], test_shape[1] / img_shape[0])
    img = cv2.resize(img, test_shape, interpolation=cv2.INTER_CUBIC)
    return np.expand_dims(img.transpose([2, 0, 1]), axis=0), ratio

def test_ascend(test_net, cfg=None):
    """
    entry for predicting single mpii
    Args:
        cfg: config
        visual: if True, visualize prediction
        cache: if True, cache score map
        output: path to output
    """
    dataset = MPII(cfg)
    dataset.set_mirror(False)

    num_images = len(dataset)
    predictions = np.zeros((num_images,), dtype=np.object)

    for i in tqdm(range(num_images)):
        batch = dataset.get_item(i)
        img, ratio = reshape_image(cfg, batch)
        o = test_net(ms.Tensor(img, dtype=ms.dtype.float32))
        out = o[0].transpose([0, 2, 3, 1]).asnumpy()
        locref = o[2].transpose([0, 2, 3, 1]).asnumpy() if (len(o) >= 3 and o[2] is not None) else None
        pairwise_pred = o[1].transpose([0, 2, 3, 1]).asnumpy() if (len(o) >= 2 and o[1] is not None) else None
        scmap, locref, _ = extract_cnn_output(out, locref, pairwise_pred, cfg)
        pose = argmax_pose_predict(scmap, locref, cfg.stride)

        pose_refscale = np.copy(pose)
        pose_refscale[:, 0:2] /= cfg.global_scale
        pose_refscale[:, 0] /= ratio[0]
        pose_refscale[:, 1] /= ratio[1]
        predictions[i] = pose_refscale

    return predictions

@process_cfg
def predict_mpii(cfg=None, visual=False, cache=False, output=None):
    """
    entry for predicting single mpii
    Args:
        cfg: config
        visual: if True, visualize prediction
        cache: if True, cache score map
        output: path to output
    """
    cfg.train = False
    ctx.set_context(**cfg.context)
    out_dir = cfg.scoremap_dir
    if cache:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    dataset = MPII(cfg)
    dataset.set_mirror(False)

    net = PoseNet(cfg=cfg)
    test_net = PoseNetTest(net, cfg)

    if hasattr(cfg, 'load_ckpt') and os.path.exists(cfg.load_ckpt):
        ms.load_checkpoint(cfg.load_ckpt, net=test_net)

    num_images = len(dataset)
    predictions = np.zeros((num_images,), dtype=np.object)

    for i in range(num_images):
        log.info('processing image %s/%s', i, num_images - 1)
        batch = dataset.get_item(i)
        o = test_net(
            ms.Tensor(np.expand_dims(batch[Batch.inputs], axis=0),
                      dtype=ms.dtype.float32),
        )
        out = o[0].transpose([0, 2, 3, 1]).asnumpy()
        locref = o[1].transpose([0, 2, 3, 1]).asnumpy() if o[1] is not None else None
        pairwise_pred = o[2].transpose([0, 2, 3, 1]).asnumpy() if o[2] is not None else None
        scmap, locref, _ = extract_cnn_output(out, locref, pairwise_pred,
                                              cfg)
        pose = argmax_pose_predict(scmap, locref, cfg.stride)

        pose_refscale = np.copy(pose)
        pose_refscale[:, 0:2] /= cfg.global_scale
        predictions[i] = pose_refscale

        if visual:
            img = np.transpose(np.squeeze(batch[Batch.inputs]).astype('uint8'), [1, 2, 0])
            show_heatmaps(cfg, img, scmap, pose)
            plt.waitforbuttonpress(timeout=1)
            plt.close()

        if cache:
            base = os.path.basename(batch[Batch.data_item].im_path).decode()
            raw_name = os.path.splitext(base)[0]
            out_fn = os.path.join(out_dir, raw_name + '.mat')
            sio.savemat(out_fn, mdict={'scoremaps': scmap.astype('float32')})

            out_fn = os.path.join(out_dir, raw_name + '_locreg' + '.mat')
            if cfg.location_refinement:
                sio.savemat(out_fn, mdict={'locreg_pred': locref.astype('float32')})

    sio.savemat(output or cfg.output, mdict={'joints': predictions})


@process_cfg
def eval_mpii(cfg=None, prediction=None):
    """
    eval mpii entry
    """
    dataset = MPII(cfg)
    if prediction is None or isinstance(prediction, str):
        filename = prediction or cfg.output
        pred = sio.loadmat(filename)

        joints = pred['joints']
    else:
        joints = np.array([prediction])
    pck_ratio_thresh = cfg.pck_threshold

    num_joints = cfg.num_joints
    num_images = joints.shape[1]

    pred_joints = np.zeros((num_images, num_joints, 2))
    gt_joints = np.zeros((num_images, num_joints, 2))
    pck_thresh = np.zeros((num_images, 1))
    gt_present_joints = np.zeros((num_images, num_joints))

    for k in range(num_images):
        pred = joints[0, k]
        gt = dataset.data[k].joints[0]
        if gt.shape[0] == 0:
            continue
        gt_joint_ids = gt[:, 0].astype('int32')
        rect = enclosing_rect(gt[:, 1:3])
        pck_thresh[k] = pck_ratio_thresh * np.amax(rect_size(rect))

        gt_present_joints[k, gt_joint_ids] = 1
        gt_joints[k, gt_joint_ids, :] = gt[:, 1:3]
        pred_joints[k, :, :] = pred[:, 0:2]

    dists = np.sqrt(np.sum((pred_joints - gt_joints) ** 2, axis=2))
    correct = dists <= pck_thresh

    num_all = np.sum(gt_present_joints, axis=0)

    num_correct = np.zeros((num_joints,))
    for j_id in range(num_joints):
        num_correct[j_id] = np.sum(correct[gt_present_joints[:, j_id] == 1, j_id], axis=0)

    pck = num_correct / num_all * 100.0

    print_results(pck, cfg)


def main():
    parser = create_arg_parser()['eval']
    args = parser.parse_args(sys.argv[1:])
    if args.device_target == 'Ascend':
        cfg = check_config(args.config, args)
        cfg.model_arts.IS_MODEL_ARTS = False
        ms.context.set_context(**cfg.context)
        net = PoseNet(cfg=cfg)
        test_net = PoseNetTest(net, cfg)
        ms.load_checkpoint(args.option[0], net=test_net)
        print("Loading", args.option[0], "succeeded!")

        cfg.train = False
        predictions = test_ascend(test_net, cfg)
        eval_mpii(cfg, predictions)
    else:
        test(parser, args)

if __name__ == '__main__':
    main()
