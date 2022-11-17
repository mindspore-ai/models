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
"""eval"""
import os
import ast
import codecs
import argparse
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import ops

from src.dataset import createDataset
from src.stpm import STPM

parser = argparse.ArgumentParser(description='test')

parser.add_argument('--device_target', type=str, choices=['Ascend', 'GPU', 'CPU'], default='Ascend')
parser.add_argument('--category', type=str, default='screw')
parser.add_argument('--device_id', type=int, default=0, help='Device id')
parser.add_argument('--dataset_path', type=str, default="/")
parser.add_argument('--save_sample', type=ast.literal_eval, default=False, help='Whether to save the infer image')
parser.add_argument('--save_sample_path', type=str, default="", help='The path to save infer image')
parser.add_argument('--ckpt_path', type=str, default='./', help="The path to save checkpoint")
parser.add_argument('--num_class', type=int, default=1000, help="The num of class")
parser.add_argument('--out_size', type=int, default=256, help="out size")

args = parser.parse_args()


class SaveImageTool:
    def __init__(self, save_sample_path):
        self.save_sample_path = save_sample_path

    def cvt2heatmap(self, gray):
        heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
        return heatmap

    def heatmap_on_image(self, heatmap, image):
        out = np.float32(heatmap) / 255 + np.float32(image) / 255
        out = out / np.max(out)
        return np.uint8(255 * out)

    def min_max_norm(self, image):
        a_min, a_max = image.min(), image.max()
        return (image - a_min) / (a_max - a_min)

    def save_anomaly_map(self, anomaly_map, a_maps, input_img, gt_img, file_name, category):
        anomaly_map_norm = self.min_max_norm(anomaly_map)
        anomaly_map_norm_hm = self.cvt2heatmap(anomaly_map_norm * 255)
        # 64x64 map
        am64 = self.min_max_norm(a_maps[0])
        am64 = self.cvt2heatmap(am64 * 255)
        # 32x32 map
        am32 = self.min_max_norm(a_maps[1])
        am32 = self.cvt2heatmap(am32 * 255)
        # 16x16 map
        am16 = self.min_max_norm(a_maps[2])
        am16 = self.cvt2heatmap(am16 * 255)
        # anomaly map on image
        heatmap = self.cvt2heatmap(anomaly_map_norm * 255)
        hm_on_img = self.heatmap_on_image(heatmap, input_img)

        # save images
        save_path = os.path.join(self.save_sample_path, f'{category}_{file_name}')
        cv2.imwrite(os.path.join(save_path + '.jpg'), input_img)
        cv2.imwrite(os.path.join(save_path + '_am64.jpg'), am64)
        cv2.imwrite(os.path.join(save_path + '_am32.jpg'), am32)
        cv2.imwrite(os.path.join(save_path + '_am16.jpg'), am16)
        cv2.imwrite(os.path.join(save_path + '_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(save_path + '_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(save_path + '_gt.jpg'), gt_img)

    def normalize(self, in_x):
        n, c, _, _ = in_x.shape
        if n != 1:
            raise ValueError(f"Only currently support batch size=1 in saving infer image. But got {n}.")
        if c != 3:
            raise ValueError(f"Only currently support that the channel of the input image is 3. But got {c}.")
        mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255]
        std = [1 / 0.229, 1 / 0.224, 1 / 0.255]
        for i in range(c):
            in_x[:, i, :, :] = (in_x[:, i, :, :] - mean[i]) / std[i]
        return in_x


def cal_anomaly_map(fs_list, ft_list, out_size=224):
    """cal_anomaly_map"""
    unsqueeze = ops.ExpandDims()
    Sum = ops.ReduceSum(keep_dims=False)
    Norm = ops.L2Normalize(axis=1)
    amap_mode = 'mul'
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = Norm(fs)
        ft_norm = Norm(ft)
        num = fs_norm * ft_norm
        cos = Sum(num, 1)
        a_map = 1 - cos
        a_map = unsqueeze(a_map, 1)
        a_map = a_map[0, 0, :, :].asnumpy()
        a_map = cv2.resize(a_map, (out_size, out_size))
        map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, map_list


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target,
                        save_graphs=False,
                        device_id=args.device_id)

    _, ds_test = createDataset(args.dataset_path, args.category, save_sample=args.save_sample, out_size=args.out_size)

    net = STPM(args, is_train=False)
    param = load_checkpoint(args.ckpt_path)
    load_param_into_net(net, param)
    net.set_train(False)

    gt_list_px_lvl = []
    pred_list_px_lvl = []
    gt_list_img_lvl = []
    pred_list_img_lvl = []

    if args.save_sample:
        if args.save_sample_path == "":
            current_path = os.path.abspath(os.path.dirname(__file__))
            args.save_sample_path = os.path.join(current_path, f'scripts/eval_{args.category}/sample')
        print(f"The image generated by inference will be saved in this path: {args.save_sample_path}")
        os.makedirs(args.save_sample_path, exist_ok=True)
    for data in ds_test.create_dict_iterator():
        gt = data['gt']
        label = data['label']
        x = data['img']
        features_s, features_t = net(x)
        amap, a_map_list = cal_anomaly_map(features_s, features_t, out_size=args.out_size)
        gt_np = gt.asnumpy()[0, 0].astype(int)

        gt_list_px_lvl.extend(gt_np.ravel())
        pred_list_px_lvl.extend(amap.ravel())
        gt_list_img_lvl.append(label.asnumpy()[0])
        pred_list_img_lvl.append(amap.max())
        if args.save_sample:
            filename = data['filename']
            filename = str(codecs.decode(filename.asnumpy().tostring()).strip(b'\x00'.decode()))
            x = x.asnumpy()
            img_tool = SaveImageTool(args.save_sample_path)
            input_x = img_tool.normalize(x)
            input_x = np.transpose(input_x, (0, 2, 3, 1))
            input_x = cv2.cvtColor(input_x[0] * 255, cv2.COLOR_BGR2RGB)
            img_tool.save_anomaly_map(amap, a_map_list, input_x, gt_np * 255, filename, args.category)

    pixel_auc = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
    img_auc = roc_auc_score(gt_list_img_lvl, pred_list_img_lvl)

    print("category: ", args.category)
    print("Total pixel-level auc-roc score : ", pixel_auc)
    print("Total image-level auc-roc score :", img_auc)
