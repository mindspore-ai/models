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
""" M2Det evaluation """

import argparse
import os
import pickle

from mindspore import context
from mindspore import load_checkpoint
from tqdm import tqdm

from src import config as cfg
from src.dataset import BaseTransform
from src.dataset import get_dataset
from src.detector import Detect
from src.model import get_model
from src.priors import PriorBox
from src.priors import anchors
from src.utils import Timer
from src.utils import image_forward
from src.utils import nms_process

parser = argparse.ArgumentParser()
parser.add_argument('--device_id', help="DEVICE_ID", type=int, default=0)
parser.add_argument("--train_url", type=str, default='./checkpoints/', help="Storage path of training results.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to evaluate")
parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset root folder")
args = parser.parse_args()


def _print_results(images_num, total_detect_time, total_nms_time):
    print(f'Detect time per image: {total_detect_time / (images_num - 1):.3f}s')
    print(f'Nms time per image: {total_nms_time / (images_num - 1):.3f}s')
    print(f'Total time per image: {(total_detect_time + total_nms_time) / (images_num - 1):.3f}s')
    print(f'FPS: {(images_num - 1) / (total_detect_time + total_nms_time):.3f} fps')


def test_network(
        save_folder,
        network,
        detector,
        test_dataset,
        transform,
        priors,
        max_per_image=300,
        threshold=0.005,
):
    images_number = len(test_dataset)
    print(f'=> Total {images_number} images to test.')

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    classes_number = cfg.model['m2det_config']['num_classes']
    all_boxes = [[[] for _ in range(images_number)] for _ in range(classes_number)]

    in_detect_timer = Timer()
    misc_timer = Timer()
    det_file = os.path.join(save_folder, 'detections.pkl')
    tot_detect_time, tot_nms_time = 0, 0
    print('Begin evaluating')
    print(images_number)

    for image_index in tqdm(range(images_number)):
        image = test_dataset.pull_image(image_index)
        # 1: detection
        in_detect_timer.tic()
        boxes, scores = image_forward(image, network, priors, detector, transform)
        detect_time = in_detect_timer.toc()
        # 2: Post-processing
        misc_timer.tic()
        nms_process(classes_number, image_index, scores, boxes, cfg, threshold, all_boxes, max_per_image)
        nms_time = misc_timer.toc()

        tot_detect_time += detect_time if image_index > 0 else 0
        tot_nms_time += nms_time if image_index > 0 else 0

    with open(det_file, 'wb') as file:
        pickle.dump(all_boxes, file, pickle.HIGHEST_PROTOCOL)

    print('===> Evaluating detections')
    test_dataset.evaluate_detections(all_boxes, save_folder)
    print('Done')
    _print_results(images_number, tot_detect_time, tot_nms_time)


def main():
    local_train_url = args.train_url

    if args.checkpoint_path:
        last_checkpoint = args.checkpoint_path
    else:
        model_name = cfg.model['m2det_config']['backbone'] + '_' + str(cfg.model['input_size'])
        last_checkpoint = os.path.join(local_train_url, cfg.experiment_tag, f"{model_name}-final.ckpt")

    if args.dataset_path:
        cfg.COCOroot = args.dataset_path

    save_dir = os.path.join(local_train_url, cfg.experiment_tag)

    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device, device_id=args.device_id)
    cfg.model['m2det_config']['checkpoint_path'] = None
    net = get_model(cfg.model['m2det_config'], cfg.model['input_size'], test=True)
    print(f'Loading checkpoint from {last_checkpoint}')
    load_checkpoint(last_checkpoint, net=net)
    net.set_train(False)

    priorbox = PriorBox(cfg)
    priors = priorbox.forward()
    _, generator = get_dataset(
        cfg,
        'COCO',
        priors.asnumpy(),
        'eval_sets',
    )

    detector = Detect(cfg.model['m2det_config']['num_classes'], anchors(cfg))
    _preprocess = BaseTransform(cfg.model['input_size'], cfg.model['rgb_means'], (2, 0, 1))
    test_network(
        save_dir,
        net,
        detector,
        generator,
        transform=_preprocess,
        priors=priors,
        max_per_image=cfg.test_cfg['topk'],
        threshold=cfg.test_cfg['score_threshold'],
    )


if __name__ == "__main__":
    main()
