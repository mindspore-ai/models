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
"""Lite-HRNet test."""

import os
import time
import argparse

from mindspore.numpy import flip
from mindspore import ops, context, load_checkpoint

from src.model import get_posenet_model
from src.test_utils import flip_back, decode
from src.mmpose.topdown_coco_dataset import get_keypoints_coco_dataset
from src.config import experiment_cfg, model_cfg


parser = argparse.ArgumentParser()
parser.add_argument('--device_id', help="DEVICE_ID", type=int, default=0)
parser.add_argument("--train_url", type=str, default='./checkpoints/', help="Storage path of training results.")
args = parser.parse_args()


def main():
    local_train_url = args.train_url
    model_name = experiment_cfg['model_config']
    last_checkpoint = os.path.join(local_train_url, experiment_cfg['experiment_tag'], f"{model_name}-final.ckpt")
    save_dir = os.path.join(local_train_url, experiment_cfg['experiment_tag'])

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args.device_id)
    _, generator = get_keypoints_coco_dataset(model_cfg.data['test'], 1, train=False)
    model = get_posenet_model(model_cfg.model)
    print(f'Loading checkpoint from {last_checkpoint}')
    load_checkpoint(last_checkpoint, net=model)
    model.set_train(False)

    expand = ops.ExpandDims()
    outputs = []
    start_time = time.time()
    for i, item in enumerate(generator):
        img, img_metas = item
        img = expand(img, 0)
        img_metas = [img_metas]
        output_heatmap = model(img).asnumpy()
        if model_cfg.model['test_cfg']['flip_test']:
            output_flipped_heatmap = model(flip(img, 3)).asnumpy()
            output_flipped_heatmap = flip_back(output_flipped_heatmap, img_metas[0]['flip_pairs'])
            if model_cfg.model['test_cfg']['shift_heatmap']:
                output_flipped_heatmap[:, :, :, 1:] = output_flipped_heatmap[:, :, :, :-1]
            output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5
        keypoint_result = decode(img_metas, output_heatmap, model_cfg.model['test_cfg'])
        outputs.append(keypoint_result)
        if (i + 1) % 100 == 0:
            speed = 100 / (time.time() - start_time)
            print(f'{(i+1):d} images processed. Evaluation speed: {speed:.1f} images/s')
            start_time = time.time()
    generator.evaluate(outputs, res_folder=save_dir)

if __name__ == "__main__":
    main()
