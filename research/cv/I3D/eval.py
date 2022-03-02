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
import os

import mindspore
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import context
import numpy as np

from src.get_dataset import get_loader
from src.i3d import InceptionI3D
from src.transforms.spatial_transforms import Compose, CenterCrop
from src.transforms.target_transforms import ClassLabel
from src.transforms.temporal_transforms import TemporalRandomCrop


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
    parser.add_argument('--video_path', type=str,
                        help='Path to location of dataset videos (if joint mode : rgb file)')
    parser.add_argument('--annotation_path', type=str,
                        help='Path to location of dataset annotation file (if joint mode : rgb file)')
    parser.add_argument('--video_path_joint_flow', type=str,
                        help='Path to location of joint mode flow dataset videos (joint mode only : flow file)')
    parser.add_argument('--annotation_path_joint_flow', type=str,
                        help='Path to location of joint mode flow dataset annotation file (joint mode only: flow file)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset string (ucf101 | hmdb51)')
    parser.add_argument('--test_mode', type=str, required=True, help='rgb, flow, joint')
    parser.add_argument('--flow_path', default='', type=str, help='path to flow pretrained model')
    parser.add_argument('--rgb_path', default='', type=str, help='path to rgb pretrained model')
    parser.add_argument('--num_classes', default=51, type=int,
                        help='Number of classes (ucf101: 101, hmdb51: 51)')
    parser.add_argument('--num_val_samples', type=int, default=1, help='Number of validation samples for each activity')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size, Preferably the same as during training')
    parser.add_argument('--test_sample_duration', default=256, type=int,
                        help='Temporal duration of inputs during testing')
    parser.add_argument('--spatial_size', default=224, type=int, help='Height and width of inputs')
    parser.add_argument('--context', default='gr', help='py(pynative) or gr(graph)')
    parser.add_argument('--device_target', default='Ascend', help='Device string')
    parser.add_argument('--device_id', default=0, type=int, help='ID of the target device')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--dropout_keep_prob', default=0.5, type=float, help='Dropout keep probability')

    parser.add_argument('--data_url', help='path to training/inference dataset folder', default='./data')
    parser.add_argument('--ckpt_url', help='model to save/load', default='./ckpt_url')
    parser.add_argument('--result_url', help='result folder to save/load', default='./result')
    parser.add_argument('--openI', default=False, type=bool, help='Train model on openI')
    config = parser.parse_args()

    if config.test_mode == 'rgb' or config.test_mode == 'flow':
        config.mode = config.test_mode
        print('checkpoint:', config.flow_path+config.rgb_path)
    elif config.test_mode == 'joint':
        config.mode_flow = 'flow'
        config.mode_rgb = 'rgb'

    if config.dataset == 'ucf101':
        config.num_classes = 101

    return config


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
    argmax = ops.ArgMaxWithValue(axis=1)
    model = InceptionI3D(
        is_train=False,
        amp_level='O0',
        num_classes=config.num_classes,
        train_spatial_squeeze=False,
        final_endpoint='logits',
        in_channels=3,
        dropout_keep_prob=config.dropout_keep_prob,
        sample_duration=config.test_sample_duration
    )

    model.set_train(False)
    pretrained_weights = mindspore.load_checkpoint(config.rgb_path)
    mindspore.load_param_into_net(model, pretrained_weights)
    total_steps = dataset_validation.get_dataset_size()
    accuracies = np.zeros(total_steps, np.float32)
    print('total steps:', total_steps)
    print('running evaluation')
    step = 0

    iterator = dataset_validation.create_dict_iterator()
    for columns in iterator:

        targets = columns["target"]
        targets = targets.asnumpy()

        clips = columns["clip"]
        logits = model(clips)
        preds, _ = argmax(logits)

        correct = 0
        for i in range(len(preds)):
            if preds[i].item() == targets[i].item():
                correct = correct + 1
        accuracy = correct / config.batch_size

        if step % 1 == 0:
            print('step:', step, '  acc:', accuracy)
        accuracies[step] = accuracy
        step = step + 1

    epoch_avg_acc = np.mean(accuracies)
    print('rgb accuracy:', epoch_avg_acc)
    if config.openI:
        log_file = os.path.join(config.result_url, 'log.txt')
        with open(log_file, "w") as f_w:
            f_w.write('rgb accuracy:' + str(epoch_avg_acc))


def eval_flow(config, dataset_validation):
    argmax = ops.ArgMaxWithValue(axis=1)
    model = InceptionI3D(
        is_train=False,
        amp_level='O0',
        num_classes=config.num_classes,
        train_spatial_squeeze=False,
        final_endpoint='logits',
        in_channels=2,
        dropout_keep_prob=config.dropout_keep_prob,
        sample_duration=config.test_sample_duration
    )

    model.set_train(False)
    pretrained_weights = mindspore.load_checkpoint(config.flow_path)
    mindspore.load_param_into_net(model, pretrained_weights)
    total_steps = dataset_validation.get_dataset_size()
    accuracies = np.zeros(total_steps, np.float32)
    print('total steps:', total_steps)
    print('running evaluation')
    step = 0

    iterator = dataset_validation.create_dict_iterator()
    for columns in iterator:

        targets = columns["target"]
        targets = targets.asnumpy()

        clips = columns["clip"]
        logits = model(clips)
        preds, _ = argmax(logits)

        correct = 0
        for i in range(len(preds)):
            if preds[i].item() == targets[i].item():
                correct = correct + 1
        accuracy = correct / config.batch_size

        if step % 1 == 0:
            print('step:', step, '  acc:', accuracy)
        accuracies[step] = accuracy
        step = step + 1

    epoch_avg_acc = np.mean(accuracies)
    print('flow accuracy:', epoch_avg_acc)
    if config.openI:
        log_file = os.path.join(config.result_url, 'log.txt')
        with open(log_file, "w") as f_w:
            f_w.write('flow accuracy:' + str(epoch_avg_acc))


def eval_joint(config, dataset_validation_joint):
    argmax = ops.ArgMaxWithValue(axis=1)
    rgb_model = InceptionI3D(
        is_train=False,
        amp_level='O0',
        num_classes=config.num_classes,
        train_spatial_squeeze=False,
        final_endpoint='logits',
        in_channels=3,
        dropout_keep_prob=config.dropout_keep_prob,
        sample_duration=config.test_sample_duration
    )
    rgb_model.set_train(False)

    pretrained_weights = mindspore.load_checkpoint(config.rgb_path)
    mindspore.load_param_into_net(rgb_model, pretrained_weights)

    flow_model = InceptionI3D(
        is_train=False,
        amp_level='O0',
        num_classes=config.num_classes,
        train_spatial_squeeze=False,
        final_endpoint='logits',
        in_channels=2,
        dropout_keep_prob=config.dropout_keep_prob,
        sample_duration=config.test_sample_duration
    )
    flow_model.set_train(False)

    pretrained_weights = mindspore.load_checkpoint(config.flow_path)
    mindspore.load_param_into_net(flow_model, pretrained_weights)
    total_steps = dataset_validation_joint.get_dataset_size()
    accuracies_rgb = np.zeros(total_steps, np.float32)
    accuracies_flow = np.zeros(total_steps, np.float32)
    print('total steps:', total_steps)
    print('running evaluation')
    step = 0

    iterator = dataset_validation_joint.create_dict_iterator()
    for columns in iterator:

        targets_rgb = columns["target_rgb"]
        targets_rgb = targets_rgb.asnumpy()
        targets_flow = columns["target_flow"]
        targets_flow = targets_flow.asnumpy()

        clips_rgb = columns["clip_rgb"]
        logits_rgb = rgb_model(clips_rgb)
        clips_flow = columns["clip_flow"]
        logits_flow = flow_model(clips_flow)
        logits = logits_flow + logits_rgb
        preds, _ = argmax(logits)

        correct_rgb = 0
        correct_flow = 0
        for i in range(len(preds)):
            if preds[i].item() == targets_rgb[i].item():
                correct_rgb = correct_rgb + 1
            if preds[i].item() == targets_flow[i].item():
                correct_flow = correct_flow + 1

        accuracy_rgb = correct_rgb / config.batch_size
        accuracy_flow = correct_flow / config.batch_size
        if step % 1 == 0:
            print('(use rgb target) step:', step, '  acc:', accuracy_rgb)
            print('(use flow target) step:', step, '  acc:', accuracy_flow)
        accuracies_rgb[step] = accuracy_rgb
        accuracies_flow[step] = accuracy_flow
        step = step + 1

    epoch_avg_acc_rgb = np.mean(accuracies_rgb)
    epoch_avg_acc_flow = np.mean(accuracies_flow)
    print('accuracy of use rgb targerts:', epoch_avg_acc_rgb, '   accuracy of use flow targerts:',
          epoch_avg_acc_flow)


def main():
    tic = time.time()
    config = parse_args()
    mindspore.dataset.config.set_seed(2022)
    mindspore.set_seed(2022)
    np.random.seed(2022)
    random.seed(2022)

    if config.openI:
        import moxing as mox
        obs_data_url = config.data_url
        config.data_url = 'cache/user-job-dir/data/'
        if not os.path.exists(config.data_url):
            os.makedirs(config.data_url)
        mox.file.copy_parallel(obs_data_url, config.data_url)
        print("Successfully Download {} to {}".format(obs_data_url, config.data_url))

        obs_ckpt_url = config.ckpt_url
        if config.test_mode == 'flow':
            config.ckpt_url = os.path.join('cache/user-job-dir', config.flow_path)
            config.flow_path = config.ckpt_url
        if config.test_mode == 'rgb':
            config.ckpt_url = os.path.join('cache/user-job-dir', config.rgb_path)
            config.rgb_path = config.ckpt_url
        mox.file.copy(obs_ckpt_url, config.ckpt_url)
        print("Successfully Download {} to {}".format(obs_ckpt_url, config.ckpt_url))

        obs_result_url = config.result_url
        config.result_url = 'cache/user-job-dir/result/'
        if not os.path.exists(config.result_url):
            os.makedirs(config.result_url)

        config.video_path = os.path.join(config.data_url, config.dataset, 'jpg')
        if config.dataset == 'ucf101':
            config.annotation_path = os.path.join(config.data_url, config.dataset, 'annotation/ucf101_01.json')
        if config.dataset == 'hmdb51':
            config.annotation_path = os.path.join(config.data_url, config.dataset, 'annotation/hmdb51_1.json')

    assert config.context in ['py', 'gr']
    if config.context == 'py':
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target, device_id=config.device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)

    validation_transforms = {
        'spatial': Compose([CenterCrop(config.spatial_size)]),
        'temporal': TemporalRandomCrop(config.test_sample_duration),
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

    if config.openI:
        mox.file.copy_parallel(config.result_url, obs_result_url)
        print("Successfully Upload {} to {}".format(config.result_url, obs_result_url))



if __name__ == "__main__":
    main()
