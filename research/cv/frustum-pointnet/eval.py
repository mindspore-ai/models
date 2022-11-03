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
import argparse

import pickle
import json

import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import load_checkpoint
from mindspore.nn import Metric
from mindspore.ops import functional as mF
from mindspore import nn, context
import src.frustum_pointnets_v1 as MODEL
from frustum_pointnets_v1 import FrustumPointNetLoss
from train.datautil import get_test_data
import train.provider as provider


def checksummary(data, name: str = None):
    # return
    if name:
        print(name)
    print("mean \t \t var")
    # print(f"{ms.numpy.mean(data)}\t{ms.numpy.var(data)}")
    print(f"{np.mean(data)}\t{np.var(data)}")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'src'))



parser = argparse.ArgumentParser()
parser.add_argument('--device_target', type=str, default='Ascend',
                    help='[Ascend, GPU]')
parser.add_argument('--device_id', type=int, default=0,
                    help='default 0')
parser.add_argument('--num_point', type=int, default=1024,
                    help='Point Number [default: 1024]')
parser.add_argument('--model', default='frustum_pointnets_v1',
                    help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--model_path',
                    default='temp_ckpt/fpoint_v1-1_1849_converted.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
# ex. log/20200121-decay_rate=0.7-decay_step=20_caronly/20200121-decay_rate=0.7-decay_step=20_caronly-acc0.777317-epoch130.pth
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for inference [default: 32]')
parser.add_argument('--output', default='test_results',
                    help='output file/folder name [default: test_results]')
parser.add_argument('--data_path', default='kitti/frustum_caronly_val.pickle',
                    help='frustum dataset pickle filepath [default: None]')
# ex. nuscenes2kitti/frustum_caronly_CAM_FRONT_val.pickle
parser.add_argument('--val_sets', type=str, default='val')
parser.add_argument('--from_rgb_detection', action='store_true',
                    help='test from dataset files from rgb detection.')
parser.add_argument('--idx_path', default='kitti/image_sets/val.txt',
                    help='filename of txt where each line is a data idx, \
                        used for rgb detection -- write <id>.txt for all frames. [default: None]')
# ex.nuscenes2kitti/image_sets/val.txt
parser.add_argument('--dump_result', action='store_true',
                    help='If true, also dump results to .pickle file')
parser.add_argument('--return_all_loss', default=False,
                    action='store_true', help='only return total loss default')
parser.add_argument('--objtype', type=str, default='caronly',
                    help='caronly or carpedcyc')
parser.add_argument('--sensor', type=str, default='CAM_FRONT',
                    help='only consider CAM_FRONT')
parser.add_argument('--dataset', type=str, default='kitti',
                    help='kitti or nuscenes or nuscenes2kitti')
parser.add_argument('--split', type=str, default='val',
                    help='v1.0-mini or val')
parser.add_argument('--debug', default=False,
                    action='store_true', help='debug mode')
FLAGS = parser.parse_args()


context.set_context(mode=context.PYNATIVE_MODE,
                    device_target=FLAGS.device_target, device_id=FLAGS.device_id)

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
NUM_POINT = FLAGS.num_point

NUM_CLASSES = 2
NUM_CHANNEL = 4
if FLAGS.objtype == 'carpedcyc':
    n_classes = 3
elif FLAGS.objtype == 'caronly':
    n_classes = 1

# Loss
Loss = FrustumPointNetLoss()

# Load Frustum Datasets.
if FLAGS.dataset == 'kitti':
    if FLAGS.data_path is None:
        overwritten_data_path = 'kitti/frustum_' + \
            FLAGS.objtype + '_' + FLAGS.val_sets + '.pickle'
    else:
        overwritten_data_path = FLAGS.data_path
    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
                                           rotate_to_center=True, one_hot=True,
                                           overwritten_data_path=overwritten_data_path,
                                           from_rgb_detection=FLAGS.from_rgb_detection)
else:
    print('Unknown dataset: %s' % (FLAGS.dataset))
    raise NotImplementedError()

test_dataloader = get_test_data(TEST_DATASET)


# output file dir and name
output_filename = FLAGS.output + '.pickle'
result_dir = FLAGS.output


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def write_detection_results(_result_dir, id_list, type_list, box2d_list, center_list,
                            heading_cls_list, heading_res_list,
                            size_cls_list, size_res_list,
                            rot_angle_list, score_list):
    ''' Write frustum pointnets results to KITTI format label files. '''


    if _result_dir is None:
        return
    # map from idx to list of strings, each string is a line (without \n)
    results = {}
    for i in range(len(center_list)):
        idx = id_list[i]

        output_str = type_list[i] + " -1 -1 -10 "
        box2d = box2d_list[i]
        output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
        h, w, l, tx, ty, tz, ry = provider.from_prediction_to_label_format(center_list[i],
                                                                           heading_cls_list[i], heading_res_list[i],
                                                                           size_cls_list[i], size_res_list[i],
                                                                           rot_angle_list[i])
        score = score_list[i]
        output_str += "%f %f %f %f %f %f %f %f" % (
            h, w, l, tx, ty, tz, ry, score)
        if idx not in results:
            results[idx] = []
        results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(_result_dir):
        os.makedirs(_result_dir)
    output_dir = os.path.join(_result_dir, 'data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx in tqdm(results):
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()


def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()


class MyMAE(Metric):

    def __init__(self):
        super(MyMAE, self).__init__()
        self.clear()
        self.iou2ds = 0
        self.iou3ds = 0
        self.iou3d_acc = 0
        self.argmax_op = ms.ops.Argmax(axis=2)
        self.batch_num = 0
        self.acc = 0
        self.best_iou3d = 0
        self.best_iou3d_epoch = 0
        self.epoch = 0
        self.test_acc = 0
        self.test_n_samples = 0

        self.ps_list = []
        self.seg_list = []
        self.segp_list = []
        self.center_list = []
        self.heading_cls_list = []
        self.heading_res_list = []
        self.size_cls_list = []
        self.size_res_list = []
        self.rot_angle_list = []
        self.score_list = []

        self.pos_cnt = 0.0
        self.pos_pred_cnt = 0.0
        self.all_cnt = 0.0
        self.max_info = np.zeros(3)
        self.min_info = np.zeros(3)
        self.mean_info = np.zeros(3)
        self.test_acc = 0

        self.data_list = []

    def clear(self):
        """Clears the internal evaluation result."""
        self.iou2ds = 0
        self.iou3ds = 0
        self.iou3d_acc = 0
        self.batch_num = 0
        self.acc = 0
        self.best_iou3d = 0
        self.best_iou3d_epoch = 0
        self.test_acc = 0.0

    def update(self, output, label):
        self.batch_num += 1
        (batch_data, batch_label, batch_center, batch_hclass, batch_hres,
         batch_sclass, batch_sres, batch_rot_angle) = label

        self.test_n_samples += batch_data.shape[0]
        logits, mask, _, center_boxnet, \
            heading_scores, _, heading_residuals, \
            size_scores, _, size_residuals, center = output

        # compute seg acc, IoU and acc(IoU)
        t1 = self.argmax_op(logits)
        t2 = batch_label.astype(ms.dtype.int32)
        correct = mF.equal(t1, t2)
        temp = correct.asnumpy()
        accuracy = np.sum(temp) / float(NUM_POINT)
        self.test_acc += accuracy

        logits = logits.asnumpy()
        mask = mask.asnumpy()
        center_boxnet = center_boxnet.asnumpy()
        center = center.asnumpy()
        heading_scores = heading_scores.asnumpy()
        heading_residuals = heading_residuals.asnumpy()
        size_scores = size_scores.asnumpy()
        size_residuals = size_residuals.asnumpy()

        batch_data = batch_data.asnumpy()
        batch_data = np.swapaxes(batch_data, 1, 2)

        batch_label = batch_label.asnumpy()
        batch_center = batch_center.asnumpy()
        batch_hclass = batch_hclass.asnumpy()
        batch_hres = batch_hres.asnumpy()
        batch_sclass = batch_sclass.asnumpy()
        batch_sres = batch_sres.asnumpy()
        batch_rot_angle = batch_rot_angle.asnumpy()

        iou2ds, iou3ds = provider.compute_box3d_iou(
            center,
            heading_scores,
            heading_residuals,
            size_scores,
            size_residuals,
            batch_center,
            batch_hclass,
            batch_hres,
            batch_sclass,
            batch_sres)
        self.iou2ds += np.sum(iou2ds)
        self.iou3ds += np.sum(iou3ds)
        self.iou3d_acc += np.sum(iou3ds >= 0.7)

        # 5. Compute and write all Results
        batch_output = np.argmax(logits, 2)  # mask#[32, 1024]
        batch_center_pred = center  # _boxnet#[32, 3]
        batch_hclass_pred = np.argmax(heading_scores, 1)  # (32,)

        batch_hres_pred = np.array([heading_residuals[j, batch_hclass_pred[j]]
                                    for j in range(batch_data.shape[0])])  # (32,)
        # batch_size_cls,batch_size_res
        batch_sclass_pred = np.argmax(size_scores, 1)  # (32,)
        batch_sres_pred = np.vstack([size_residuals[j, batch_sclass_pred[j], :]
                                     for j in range(batch_data.shape[0])])  # (32,3)

        # batch_scores
        batch_seg_prob = softmax(logits)[:, :, 1]  # (32, 1024, 2) ->(32, 1024)
        batch_seg_mask = np.argmax(logits, 2)  # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1)  # B,

        mask_mean_prob = mask_mean_prob / \
            (np.sum(batch_seg_mask, 1) + 1e-7)  # B,

        _ = np.max(softmax(heading_scores), 1)  # B
        _ = np.max(softmax(size_scores), 1)  # B,
        mask_max_prob = np.max(batch_seg_prob * batch_seg_mask, 1)
        batch_scores = mask_max_prob

        for j in range(batch_output.shape[0]):
            self.ps_list.append(batch_data[j, ...])
            self.seg_list.append(batch_label[j, ...])
            self.segp_list.append(batch_output[j, ...])
            self.center_list.append(batch_center_pred[j, :])
            self.heading_cls_list.append(batch_hclass_pred[j])
            self.heading_res_list.append(batch_hres_pred[j])
            self.size_cls_list.append(batch_sclass_pred[j])
            self.size_res_list.append(batch_sres_pred[j, :])
            self.rot_angle_list.append(batch_rot_angle[j])
            if batch_scores[j] < -1000000000:
                batch_scores[j] = -1000000000
            self.score_list.append(batch_scores[j])
            self.pos_cnt += np.sum(batch_label[j, :])
            self.pos_pred_cnt += np.sum(batch_output[j, :])
            pts_np = batch_data[j, :3, :]  # (3,1024)
            max_xyz = np.max(pts_np, axis=1)
            self.max_info = np.maximum(self.max_info, max_xyz)
            min_xyz = np.min(pts_np, axis=1)
            self.min_info = np.minimum(self.min_info, min_xyz)
            self.mean_info += np.sum(pts_np, axis=1)

    def eval(self):
        self.epoch += 1

        if FLAGS.dump_result:
            print('dumping...')
            with open(output_filename, 'wb') as fp:
                pickle.dump(self.ps_list, fp)
                pickle.dump(self.seg_list, fp)
                pickle.dump(self.segp_list, fp)
                pickle.dump(self.center_list, fp)
                pickle.dump(self.heading_cls_list, fp)
                pickle.dump(self.heading_res_list, fp)
                pickle.dump(self.size_cls_list, fp)
                pickle.dump(self.size_res_list, fp)
                pickle.dump(self.rot_angle_list, fp)
                pickle.dump(self.score_list, fp)

        # Write detection results for KITTI evaluation
        print('Number of point clouds: %d' % (len(self.ps_list)))

        write_detection_results(result_dir, TEST_DATASET.id_list,
                                TEST_DATASET.type_list, TEST_DATASET.box2d_list,
                                self.center_list, self.heading_cls_list, self.heading_res_list,
                                self.size_cls_list, self.size_res_list, self.rot_angle_list, self.score_list)

        # Make sure for each frame (no matter if we have measurement for that frame),
        # there is a TXT file
        output_dir = os.path.join(result_dir, 'data')
        if FLAGS.idx_path is not None:
            to_fill_filename_list = [line.rstrip() + '.txt'
                                     for line in open(FLAGS.idx_path)]
            fill_files(output_dir, to_fill_filename_list)

        all_cnt = FLAGS.num_point * len(self.ps_list)

        print(f"segmentation accuracy {self.test_acc/self.test_n_samples}")
        print(f"box IoU(ground) {self.iou2ds/self.test_n_samples}")
        print(f"box IoU(3D) {self.iou3ds/self.test_n_samples}")
        print(
            f"box estimation accuracy (IoU=0.7) {self.iou3d_acc/self.test_n_samples}")

        print('Average pos ratio: %f' % (self.pos_cnt / float(all_cnt)))
        print('Average pos prediction ratio: %f' %
              (self.pos_pred_cnt / float(all_cnt)))
        print('Average npoints: %f' % (float(all_cnt) / len(self.ps_list)))
        self.mean_info = self.mean_info / len(self.ps_list) / FLAGS.num_point
        print('Mean points: x%f y%f z%f' %
              (self.mean_info[0], self.mean_info[1], self.mean_info[2]))
        print('Max points: x%f y%f z%f' %
              (self.max_info[0], self.max_info[1], self.max_info[2]))
        print('Min points: x%f y%f z%f' %
              (self.min_info[0], self.min_info[1], self.min_info[2]))

        ans = {
            "Average pos ratio:": self.pos_cnt / float(all_cnt),
            "Average pos prediction ratio:": self.pos_pred_cnt / float(all_cnt),
            "Average npoints:": float(all_cnt) / len(self.ps_list),
            "Mean points:": self.mean_info,
            "Max points:": self.max_info,
            "Min points:": self.min_info,
        }
        print(ans)

        ans2 = {
            "test_acc": self.test_acc/self.test_n_samples,
            "test_iou2ds": self.iou2ds/self.test_n_samples,
            "test_iou3ds": self.iou3ds/self.test_n_samples,
            "test_iou3d_acc": self.iou3d_acc/self.test_n_samples,
        }
        return json.dumps(ans2)


def eval_kitti():
    net: nn.Cell = MODEL.FrustumPointNetv1()
    lossfn = MODEL.FrustumPointNetLoss(enable_summery=True)

    load_checkpoint(FLAGS.model_path, net)
    net_eval = MODEL.FpointWithEval(net, lossfn=lossfn)
    net_with_criterion: nn.Cell = MODEL.FpointWithLoss_old(net, lossfn)
    optimizer = nn.Adam(net.trainable_params())
    model = ms.Model(net_with_criterion,
                     loss_fn=None,
                     eval_network=net_eval,
                     optimizer=optimizer,
                     metrics={"MyMAE": MyMAE()})

    model.eval(test_dataloader, dataset_sink_mode=False)


if __name__ == '__main__':
    eval_kitti()
