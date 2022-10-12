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
"""postprocess"""
import os
import json
from pathlib import Path
import cv2
import numpy as np

from sklearn.metrics import roc_auc_score

from mindspore.common import set_seed

from src.config import get_arguments
from src.anomaly_map import anomaly_map_generator
from src.operator import normalize, save_anomaly_map, prep_dirs

set_seed(1)

def preLauch():
    """parse the console argument"""
    parser = get_arguments()

    # PostProcess
    parser.add_argument('--mode', type=str, default='postprocess for inference 310')
    parser.add_argument('--img_dir', type=str, default='./preprocess_result/img',
                        help="path to save binary files and json files")
    parser.add_argument('--label_dir', type=str, default='./preprocess_result/label',
                        help="path to label files")
    parser.add_argument('--result_dir', type=str, default='./postprocess_result',
                        help="path to result files")

    return parser.parse_args()

if __name__ == '__main__':
    args = preLauch()
    test_label_path = Path(os.path.join(args.label_dir, "test_label.json"))
    test_result_path = args.result_dir

    with test_label_path.open('r') as dst_file:
        test_label = json.load(dst_file)

    gt_list_px_lvl = []
    pred_list_px_lvl = []
    img_path_list = []

    print("***************start eval***************")
    for i in range(int(len(os.listdir(test_result_path)) / 6)):
        # model outputs
        hidden_variables = []
        features_layer1_path = os.path.join(test_result_path, "data_img_{}_0.bin".format(str(i).zfill(3)))
        hidden_variables.append(np.fromfile(features_layer1_path, dtype=np.float32).reshape(-1, 256, 64, 64))
        features_layer2_path = os.path.join(test_result_path, "data_img_{}_1.bin".format(str(i).zfill(3)))
        hidden_variables.append(np.fromfile(features_layer2_path, dtype=np.float32).reshape(-1, 512, 32, 32))
        features_layer3_path = os.path.join(test_result_path, "data_img_{}_2.bin".format(str(i).zfill(3)))
        hidden_variables.append(np.fromfile(features_layer3_path, dtype=np.float32).reshape(-1, 1024, 16, 16))

        # generate anomaly map
        anomaly_map = anomaly_map_generator(hidden_variables, output_size=args.im_resize)
        test_single_label = test_label['{}'.format(i)]
        gt_np = np.array(test_single_label['gt']).astype(int)
        gt = gt_np[:, 0]
        gt_list_px_lvl.extend(gt.ravel())
        pred_list_px_lvl.extend(anomaly_map.ravel())

        # save visible imgs
        if (args.save_imgs) and (args.test_batch_size == 1):
            _, sample_path = prep_dirs('./', args.category)
            test_json_path = test_label['test_json_path']

            json_path = Path(test_json_path)
            with json_path.open('r') as label_file:
                test_label_string = json.load(label_file)
            idx = test_single_label['idx']
            test_single_label_string = test_label_string['{}'.format(idx[0])]
            file_name = test_single_label_string['name']
            x_type = test_single_label_string['img_type']
            img_path = os.path.join(args.img_dir, "data_img_{}.bin".format(str(i).zfill(3)))

            img = np.fromfile(img_path, dtype=np.float32).reshape(1, 3, 256, 256)
            img = normalize(img, args.mean, args.std)
            input_img = cv2.cvtColor(np.transpose(img, (0, 2, 3, 1))[0] * 255, cv2.COLOR_BGR2RGB)
            anomaly_map = anomaly_map.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)
            save_anomaly_map(sample_path, anomaly_map, input_img, gt * 255, file_name, x_type)

    pixel_acc = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
    print("***************eval end***************")
    print('category is {}'.format(args.category))
    print("pixel_acc: {}".format(pixel_acc))
