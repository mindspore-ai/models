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
"""Run evaluation for a model exported to ONNX"""
import time
import argparse
import os
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from src.config import config
from src.dataset import keypoint_dataset
from src.dataset import flip_pairs
from src.utils.coco import evaluate
from src.utils.transforms import flip_back
from src.utils.inference import get_final_preds

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate onnx')
    parser.add_argument('--ckpt_path', type=str, default="E:/simple_baselines/simple_baselines.onnx", help='onnx ckpt')
    parser.add_argument('--target_device', type=str, default="GPU", help='select[GPU,CPU]')
    args = parser.parse_args()
    return args


def create_session(checkpoint_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name

def validate(cfg, val_dataset, output_dir, ann_path, ckpt_path, target_device):
    num_samples = val_dataset.get_dataset_size() * cfg.TEST.BATCH_SIZE
    all_preds = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 2))
    image_id = []
    idx = 0

    start = time.time()
    session, input_name = create_session(ckpt_path, target_device)
    for item in tqdm(val_dataset.create_dict_iterator()):
        inputs = item['image'].asnumpy()
        output = session.run(None, {input_name: inputs})[0]
        if cfg.TEST.FLIP_TEST:
            inputs_flipped = inputs[:, :, :, ::-1]
            output_flipped = session.run(None, {input_name: inputs_flipped})[0]
            output_flipped = flip_back(output_flipped, flip_pairs)
            if cfg.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.copy()[:, :, :, 0:-1]
            output = (output + output_flipped) * 0.5

        c = item['center'].asnumpy()
        s = item['scale'].asnumpy()
        score = item['score'].asnumpy()
        file_id = list(item['id'].asnumpy())
        preds, maxvals = get_final_preds(cfg, output.copy(), c, s)
        num_images, _ = preds.shape[:2]
        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        all_boxes[idx:idx + num_images, 0] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 1] = score
        image_id.extend(file_id)
        idx += num_images
        if idx % 1024 == 0:
            print('{} samples validated in {} seconds'.format(idx, time.time() - start))
            start = time.time()


    print(all_preds[:idx].shape, all_boxes[:idx].shape, len(image_id))
    _, perf_indicator = evaluate(cfg, all_preds[:idx], output_dir, all_boxes[:idx], image_id, ann_path)
    print("AP:", perf_indicator)
    return perf_indicator

def main():
    args = parse_args()
    valid_dataset, _ = keypoint_dataset(
        config=config,
        train_mode=False,
        num_parallel_workers=config.TEST.NUM_PARALLEL_WORKERS,
    )
    ckpt_name = args.ckpt_path.split('/')
    ckpt_name = ckpt_name[len(ckpt_name) - 1]
    ckpt_name = ckpt_name.split('.')[0]

    output_dir = os.path.join(config.DATASET.ROOT, 'result/')
    ann_path = config.DATASET.ROOT
    output_dir = output_dir + ckpt_name
    ann_path = ann_path + config.DATASET.TEST_JSON
    print("ann_path is :", ann_path)
    validate(config, valid_dataset, output_dir, ann_path, args.ckpt_path, args.target_device)

if __name__ == '__main__':
    main()
