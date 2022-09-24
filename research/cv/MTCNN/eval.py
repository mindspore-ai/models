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
import argparse

from src.evaluate import evaluation
from src.models.mtcnn_detector import MtcnnDetector
from src.models.mtcnn import PNet, RNet, ONet

from mindspore import load_checkpoint, load_param_into_net, context
import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MTCNN on FDDB dataset")
    parser.add_argument('--pnet_ckpt', '-p', required=True, help="checkpoint of PNet")
    parser.add_argument('--rnet_ckpt', '-r', required=True, help="checkpoint of RNet")
    parser.add_argument('--onet_ckpt', '-o', required=True, help="checkpoint of ONet")

    args = parser.parse_args()
    return args

def main(args):
    context.set_context(device_target='GPU')

    pnet = PNet()
    pnet_params = load_checkpoint(args.pnet_ckpt)
    load_param_into_net(pnet, pnet_params)
    pnet.set_train(False)

    rnet = RNet()
    rnet_params = load_checkpoint(args.rnet_ckpt)
    load_param_into_net(rnet, rnet_params)
    rnet.set_train(False)

    onet = ONet()
    onet_params = load_checkpoint(args.onet_ckpt)
    load_param_into_net(onet, onet_params)
    onet.set_train(False)

    mtcnn_detector = MtcnnDetector(pnet, rnet, onet)

    FDDB_out_dir = os.path.join(cfg.DATASET_DIR, 'FDDB_out')
    if not os.path.exists(FDDB_out_dir):
        os.mkdir(FDDB_out_dir)

    print("Start detecting FDDB images")
    for i in range(1, 11):
        if not os.path.exists(os.path.join(FDDB_out_dir, str(i))):
            os.mkdir(os.path.join(FDDB_out_dir, str(i)))
        file_path = os.path.join(cfg.FDDB_DIR, 'FDDB-folds', 'FDDB-fold-%02d.txt' % i)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                image_path = os.path.join(cfg.FDDB_DIR, line) + '.jpg'
                line = line.replace('/', '_')
                with open(os.path.join(FDDB_out_dir, str(i), line + '.txt'), 'w') as w:
                    w.write(line)
                    w.write('\n')
                    boxes_c, _ = mtcnn_detector.detect_face(image_path)
                    if boxes_c is not None:
                        w.write(str(boxes_c.shape[0]))
                        w.write('\n')
                        for box in boxes_c:
                            w.write(f'{int(box[0])} {int(box[1])} {int(box[2]-box[0])} {int(box[3]-box[1])} {box[4]}\n')
    print("Detection Done!")
    print("Start evluation!")
    evaluation(FDDB_out_dir, os.path.join(cfg.FDDB_DIR, 'FDDB-folds'))


if __name__ == '__main__':
    main(parse_args())
