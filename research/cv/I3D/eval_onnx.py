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
"""
Evaluate I3D and get accuracy.
"""

import argparse
import random
import time

import mindspore
import mindspore.dataset as ds
import numpy as np
import onnxruntime as ort

from src.get_dataset import get_loader
from src.transforms.spatial_transforms import CenterCrop, Compose
from src.transforms.target_transforms import ClassLabel
from src.transforms.temporal_transforms import TemporalCenterCrop


class Joint_dataset:
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]

        clipA, targetA = xA
        clipB, targetB = xB

        return clipA, targetA, clipB, targetB

    def __len__(self):
        return len(self.datasetA)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_val_samples', type=int, default=1, help='Number of validation samples for each activity')
    parser.add_argument('--test_sample_duration', default=64, type=int,
                        help='Temporal duration of inputs during testing')
    parser.add_argument('--spatial_size', default=224, type=int, help='Height and width of inputs')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of threads for multi-thread loading')

    parser.add_argument('--test_mode', type=str, required=True, help='rgb, flow, joint')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size, Preferably the same as during training')
    parser.add_argument('--device_target', default='GPU', help='Device string')
    parser.add_argument('--device_id', default=0, type=int, help='ID of the target device')

    parser.add_argument('--rgb_path', help='model to save/load', default='./ckpt_url')
    parser.add_argument('--flow_path', help='model to save/load', default='./ckpt_url')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset string (ucf101 | hmdb51)')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to location of dataset videos (if joint mode : rgb file)')
    parser.add_argument('--annotation_path', type=str, required=True,
                        help='Path to location of dataset annotation file (if joint mode : rgb file)')
    parser.add_argument('--video_path_joint_flow', type=str,
                        help='Path to location of joint mode flow dataset videos (joint mode only : flow file)')
    parser.add_argument('--annotation_path_joint_flow', type=str,
                        help='Path to location of joint mode flow dataset annotation file (joint mode only: flow file)')

    config = parser.parse_args()

    if config.test_mode == 'rgb' or config.test_mode == 'flow':
        config.mode = config.test_mode
    elif config.test_mode == 'joint':
        config.mode_flow = 'flow'
        config.mode_rgb = 'rgb'

    return config

def create_session(checkpoint_path, target_device, device_id):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(checkpoint_path, providers=providers, provider_options=[{'device_id': device_id}])
    input_name = session.get_inputs()[0].name
    return session, input_name

def get_single_dataset(config, validation_transforms):
    if config.dataset == 'ucf101' or config.dataset == 'hmdb51':
        dataset = get_loader(
            config.video_path,
            config.annotation_path,
            'validation',
            config.mode,
            config.num_val_samples,
            spatial_transform=validation_transforms['spatial'],
            temporal_transform=validation_transforms['temporal'],
            target_transform=validation_transforms['target'],
            sample_duration=config.test_sample_duration)

    print('Found {} validation examples'.format(len(dataset)))
    validation = ds.GeneratorDataset(source=dataset, column_names=["clip", "target"],
                                     shuffle=True, num_parallel_workers=config.num_workers, max_rowsize=18)
    validation = validation.batch(batch_size=config.batch_size, drop_remainder=True,
                                  num_parallel_workers=config.num_workers)

    return validation


def get_joint_dataset(config, validation_transforms):
    if config.dataset == 'ucf101':
        dataset_flow = get_loader(
            config.video_path_joint_flow,
            config.annotation_path_joint_flow,
            'validation',
            config.mode_flow,
            config.num_val_samples,
            spatial_transform=validation_transforms['spatial'],
            temporal_transform=validation_transforms['temporal'],
            target_transform=validation_transforms['target'],
            sample_duration=config.test_sample_duration)
        dataset_rgb = get_loader(
            config.video_path,
            config.annotation_path,
            'validation',
            config.mode_rgb,
            config.num_val_samples,
            spatial_transform=validation_transforms['spatial'],
            temporal_transform=validation_transforms['temporal'],
            target_transform=validation_transforms['target'],
            sample_duration=config.test_sample_duration)

    elif config.dataset == 'hmdb51':
        dataset_flow = get_loader(
            config.video_path_joint_flow,
            config.annotation_path_joint_flow,
            'validation',
            config.mode_flow,
            config.num_val_samples,
            spatial_transform=validation_transforms['spatial'],
            temporal_transform=validation_transforms['temporal'],
            target_transform=validation_transforms['target'],
            sample_duration=config.test_sample_duration)
        dataset_rgb = get_loader(
            config.video_path,
            config.annotation_path,
            'validation',
            config.mode_rgb,
            config.num_val_samples,
            spatial_transform=validation_transforms['spatial'],
            temporal_transform=validation_transforms['temporal'],
            target_transform=validation_transforms['target'],
            sample_duration=config.test_sample_duration)

    print('Found {} validation_rgb examples'.format(len(dataset_rgb)))
    print('Found {} validation_flow examples'.format(len(dataset_flow)))
    joint_dataset = Joint_dataset(dataset_rgb, dataset_flow)
    validation_joint = ds.GeneratorDataset(source=joint_dataset,
                                           column_names=["clip_rgb", "target_rgb", "clip_flow", "target_flow"],
                                           shuffle=True, num_parallel_workers=config.num_workers,
                                           max_rowsize=18)
    validation_joint = validation_joint.batch(batch_size=config.batch_size, drop_remainder=True,
                                              num_parallel_workers=config.num_workers)

    return validation_joint


def eval_rgb(config, dataset_validation):
    session, input_name = create_session(config.rgb_path, config.device_target, config.device_id)

    total_steps = dataset_validation.get_dataset_size()
    accuracies = np.zeros(total_steps, np.float32)
    print('total steps:', total_steps)
    print('running evaluation')
    step = 0

    iterator = dataset_validation.create_dict_iterator()
    for columns in iterator:

        targets = columns["target"]
        targets = targets.asnumpy()
        clips = columns["clip"].asnumpy()
        logits = session.run(None, {input_name: clips})[0]
        preds = np.argmax(logits, axis=1)

        correct = 0
        for i in range(len(preds)):
            if preds[i].item() == targets[i].item():
                correct = correct + 1
        accuracy = correct / config.batch_size
        print('step:', step, '  acc:', accuracy, ' class:')
        accuracies[step] = accuracy
        step = step + 1

    epoch_avg_acc = np.mean(accuracies)
    print('rgb accuracy:', epoch_avg_acc)

def eval_flow(config, dataset_validation):
    session, input_name = create_session(config.flow_path, config.device_target, config.device_id)

    total_steps = dataset_validation.get_dataset_size()
    accuracies = np.zeros(total_steps, np.float32)
    print('total steps:', total_steps)
    print('running evaluation')
    step = 0

    iterator = dataset_validation.create_dict_iterator()
    for columns in iterator:

        targets = columns["target"].asnumpy()
        clips = columns["clip"].asnumpy()
        logits = session.run(None, {input_name: clips})[0]
        preds = np.argmax(logits, axis=1)

        correct = 0
        for i in range(len(preds)):
            if preds[i].item() == targets[i].item():
                correct = correct + 1
        accuracy = correct / config.batch_size
        print('step:', step, '  acc:', accuracy)
        accuracies[step] = accuracy
        step = step + 1

    epoch_avg_acc = np.mean(accuracies)
    print('flow accuracy:', epoch_avg_acc)

def eval_joint(config, dataset_validation_joint):
    sess_rgb, input_name_rgb = create_session(config.rgb_path, config.device_target, config.device_id)
    sess_flow, input_name_flow = create_session(config.flow_path, config.device_target, config.device_id)

    total_steps = dataset_validation_joint.get_dataset_size()
    accuracies_rgb = np.zeros(total_steps, np.float32)
    accuracies_flow = np.zeros(total_steps, np.float32)
    accuracies_joint = np.zeros(total_steps, np.float32)
    print('total steps:', total_steps)
    print('running evaluation')
    step = 0

    iterator = dataset_validation_joint.create_dict_iterator()
    for columns in iterator:

        targets_rgb = columns["target_rgb"]
        targets_rgb = targets_rgb.asnumpy()
        targets_flow = columns["target_flow"]
        targets_flow = targets_flow.asnumpy()
        clips_rgb = columns["clip_rgb"].asnumpy()
        logits_rgb = sess_rgb.run(None, {input_name_rgb: clips_rgb})[0]

        clips_flow = columns["clip_flow"].asnumpy()
        logits_flow = sess_flow.run(None, {input_name_flow: clips_flow})[0]

        logits = logits_flow + logits_rgb
        preds_rgb = np.argmax(logits_rgb, axis=1)
        preds_flow = np.argmax(logits_flow, axis=1)
        preds = np.argmax(logits, axis=1)

        correct_rgb = 0
        correct_flow = 0
        correct_joint = 0
        for i in range(len(preds)):
            if preds_rgb[i].item() == targets_rgb[i].item():
                correct_rgb = correct_rgb + 1
            if preds_flow[i].item() == targets_flow[i].item():
                correct_flow = correct_flow + 1
            if preds[i].item() == targets_flow[i].item():
                correct_joint = correct_joint + 1
        accuracy_rgb = correct_rgb / config.batch_size
        accuracy_flow = correct_flow / config.batch_size
        accuracy_joint = correct_joint / config.batch_size
        print('(rgb) step:', step, '  acc:', accuracy_rgb)
        print('(flow) step:', step, '  acc:', accuracy_flow)
        accuracies_rgb[step] = accuracy_rgb
        accuracies_flow[step] = accuracy_flow
        accuracies_joint[step] = accuracy_joint
        step = step + 1

    epoch_avg_acc_rgb = np.mean(accuracies_rgb)
    epoch_avg_acc_flow = np.mean(accuracies_flow)
    epoch_avg_acc_joint = np.mean(accuracies_joint)
    print('accuracy of rgb:', epoch_avg_acc_rgb, '   accuracy of flow:',
          epoch_avg_acc_flow, 'accuracy of joint:', epoch_avg_acc_joint)


def main():
    tic = time.time()
    config = parse_args()
    mindspore.dataset.config.set_seed(2022)
    mindspore.set_seed(2022)
    np.random.seed(2022)
    random.seed(2022)

    validation_transforms = {
        'spatial': Compose([CenterCrop(config.spatial_size)]),
        'temporal': TemporalCenterCrop(config.test_sample_duration),
        'target': ClassLabel()
    }

    if config.test_mode == 'rgb' or config.test_mode == 'flow':
        dataset_validation = get_single_dataset(config, validation_transforms)

    elif config.test_mode == 'joint':
        dataset_validation_joint = get_joint_dataset(config, validation_transforms)

    # do eval
    if config.test_mode == 'rgb':
        eval_rgb(config, dataset_validation)

    elif config.test_mode == 'flow':
        eval_flow(config, dataset_validation)

    elif config.test_mode == 'joint':
        eval_joint(config, dataset_validation_joint)

    toc = time.time()
    total_time = toc - tic
    print('total_time:', total_time)

if __name__ == "__main__":
    main()
