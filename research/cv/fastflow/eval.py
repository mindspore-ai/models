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

import ast
import os
import json
from pathlib import Path
import cv2
import numpy as np

from sklearn.metrics import roc_auc_score

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import get_arguments
from src.dataset import createDataset, createDatasetJson
from src.resnet import wide_resnet50_2
from src.loss import NetWithLossCell, FastflowLoss
from src.fastflow import build_model
from src.anomaly_map import anomaly_map_generator
from src.operator import  normalize, save_anomaly_map, prep_dirs

def preLauch():
    """parse the console argument"""
    parser = get_arguments()

    # Eval Device.
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument("--distribute", type=ast.literal_eval, default=False, help="Run distribute, default is false.")
    parser.add_argument('--device_target', type=str, default='Ascend')
    parser.add_argument('--device_num', type=int, default=1, help='device num, default is 1.')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend (Default: 0)')

    return parser.parse_args()

if __name__ == '__main__':
    args = preLauch()
    context.set_context(device_target=args.device_target, mode=context.GRAPH_MODE, save_graphs=False)
    context.set_context(device_id=args.device_id)

    # dataset
    _, test_dataset = createDataset(
        dataset_path=args.dataset_path,
        category=args.category,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        im_resize=args.im_resize
        )

    data_iter = test_dataset.create_dict_iterator()
    step_size = test_dataset.get_dataset_size()

    # network
    feature_extractor = wide_resnet50_2()
    network = build_model(
        backbone=feature_extractor,
        flow_step=args.flow_step,
        im_resize=args.im_resize,
        conv3x3_only=args.conv3x3_only,
        hidden_ratio=args.hidden_ratio
        )

    # keep network auto_prefix is same as ckpt
    loss = FastflowLoss()
    _ = NetWithLossCell(network, loss)

    # load param into net
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = os.path.join(args.ckpt_dir, "fastflow_%s.ckpt" % (args.category))
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)
    network.set_train(False)

    gt_list_px_lvl = []
    pred_list_px_lvl = []
    img_path_list = []
    print("***************start eval***************")
    for step, data in enumerate(data_iter):
        hidden_variables, _ = network(data['img'])
        anomaly_map = anomaly_map_generator(hidden_variables, output_size=args.im_resize)
        gt_np = data['gt'].asnumpy().astype(int)
        gt = gt_np[:, 0]
        gt_list_px_lvl.extend(gt.ravel())
        pred_list_px_lvl.extend(anomaly_map.ravel())

        # save visible imgs
        if (args.save_imgs) and (args.test_batch_size == 1):
            current_path = os.path.abspath(os.path.dirname(__file__))
            _, sample_path = prep_dirs(current_path, args.category)

            test_json_path = createDatasetJson(
                dataset_path=args.dataset_path,
                category=args.category,
                test_data=test_dataset,
                phase='test'
                )
            json_path = Path(test_json_path)
            with json_path.open('r') as label_file:
                label = json.load(label_file)
            step_label = label['{}'.format(data['idx'][0])]
            file_name = step_label['name']
            x_type = step_label['img_type']
            img_path_list.extend(file_name)
            img = normalize(data['img'].asnumpy().astype(np.float32), args.mean, args.std)
            input_img = cv2.cvtColor(np.transpose(img, (0, 2, 3, 1))[0] * 255, cv2.COLOR_BGR2RGB)
            anomaly_map = anomaly_map.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)
            save_anomaly_map(sample_path, anomaly_map, input_img, gt * 255, file_name, x_type)

    pixel_auc = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
    print("***************eval end***************")
    print('category is {}'.format(args.category))
    print("pixel_auc: {}".format(pixel_auc))
