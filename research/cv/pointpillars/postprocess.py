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
"""post process for 310 inference"""
import argparse
import os
import warnings
import pickle
import numpy as np

from src.utils import get_config
from src.core.eval_utils import get_official_eval_result #, get_coco_eval_result
from src.predict import predict_kitti_to_anno
from src.predict import predict

warnings.filterwarnings('ignore')


def get_result(cfg_path, result_path, gt_kitti_infos_path, input_data_path):
    cfg = get_config(cfg_path)
    eval_input_cfg = cfg['eval_input_reader']
    model_cfg = cfg['model']
    center_limit_range = model_cfg['post_center_limit_range']
    class_names = list(eval_input_cfg['class_names'])
    dt_annos = []
    root = ""
    files = ""
    for root, _, files in os.walk(result_path):
        pass
    for i in range(int(len(files)/3)):
        # 1. read bin files
        path = root + '/kittiVal1_'+ str(i) +'_voxels_'
        boxPredPath = path + '0.bin'
        clsPredPath = path + '1.bin'
        dClsPredPath = path + '2.bin'

        example_root_path = input_data_path
        anchors_path = example_root_path + '/anchors_data/'  + "kittiVal1_" + str(i) + "_anchors.bin"
        rect_path = example_root_path + '/rect_data/'  + "kittiVal1_" + str(i) + "_rect.bin"
        Trv2c_path = example_root_path + '/Trv2c_data/' + "kittiVal1_" + str(i) + "_Trv2c.bin"
        P2_path = example_root_path + '/P2_data/'  + "kittiVal1_" + str(i) + "_P2.bin"
        anchors_mask_path = example_root_path + '/anchors_mask_data/'  + "kittiVal1_" + str(i) + "_anchors_mask.bin"
        image_idx_path = example_root_path + '/image_idx_data/'  + "kittiVal1_" + str(i) + "_image_idx.bin"
        imgshape_path = example_root_path + '/imgshape_data/'  + "kittiVal1_" + str(i) + "_imgshape.bin"

        netOutputBoxPred = np.fromfile(boxPredPath, dtype=np.float32)
        netOutputclsPred = np.fromfile(clsPredPath, dtype=np.float32)
        netOutputdClsPred = np.fromfile(dClsPredPath, dtype=np.float32)
        rect = np.fromfile(rect_path, dtype=np.float32)
        Trv2c = np.fromfile(Trv2c_path, dtype=np.float32)
        P2 = np.fromfile(P2_path, dtype=np.float32)
        anchors_mask = np.fromfile(anchors_mask_path, dtype=np.uint8)
        image_idx = np.fromfile(image_idx_path, dtype=np.int64)
        anchors = np.fromfile(anchors_path, dtype=np.float32)
        imgshape = np.fromfile(imgshape_path, dtype=np.int32)

        # 2. data is one-dim, need to reshape
        if class_names[0] == 'Car':
            netOutputBoxPred = netOutputBoxPred.reshape(1, 248, 216, 14)
            netOutputclsPred = netOutputclsPred.reshape(1, 248, 216, 2)
            netOutputdClsPred = netOutputdClsPred.reshape(1, 248, 216, 4)
            anchors = anchors.reshape(1, 107136, 7)
            anchors_mask = anchors_mask.reshape(1, 107136)

        elif class_names[0] == 'Cyclist':
            netOutputBoxPred = netOutputBoxPred.reshape(1, 248, 296, 28)
            netOutputclsPred = netOutputclsPred.reshape(1, 248, 296, 8)
            netOutputdClsPred = netOutputdClsPred.reshape(1, 248, 296, 8)
            anchors = anchors.reshape(1, 293632, 7)
            anchors_mask = anchors_mask.reshape(1, 293632)

        rect = rect.reshape(1, 4, 4)
        Trv2c = Trv2c.reshape(1, 4, 4)
        P2 = P2.reshape(1, 4, 4)
        image_idx = image_idx.reshape(1)
        imgshape = imgshape.reshape(1, 2)

        preds = {'box_preds': netOutputBoxPred, 'cls_preds': netOutputclsPred, 'dir_cls_preds': netOutputdClsPred}
        data = {'image_shape': imgshape, 'anchors': anchors, 'rect': rect, 'Trv2c': Trv2c,
                'P2': P2, 'anchors_mask': anchors_mask, 'image_idx': image_idx}
        # get box_coder
        from src.builder import box_coder_builder
        box_coder_cfg = model_cfg['box_coder']
        box_coder = box_coder_builder.build(box_coder_cfg)
        preds = predict(data, preds, model_cfg, box_coder)
        dt_annos += predict_kitti_to_anno(preds,
                                          data,
                                          class_names,
                                          center_limit_range)

    with open(gt_kitti_infos_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    gt_annos = [info["annos"] for info in kitti_infos]
    print("==============================================================")
    result = get_official_eval_result(
        gt_annos,
        dt_annos,
        class_names,
    )
    print(result)
    print("==============================================================")
    with open("result.pkl", 'wb') as f:
        pickle.dump(dt_annos, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="postprocess")

    parser.add_argument("--cfg_path", type=str, default="", help="config path")
    parser.add_argument("--label_file", type=str, default="", help="label data dir")
    parser.add_argument("--result_dir", type=str, default="./result_Files", help="infer result Files")
    parser.add_argument("--input_data_path", type=str, default="", help="input data path")

    args, _ = parser.parse_known_args()
    get_result(cfg_path=args.cfg_path, result_path=args.result_dir, gt_kitti_infos_path=args.label_file,
               input_data_path=args.input_data_path)
