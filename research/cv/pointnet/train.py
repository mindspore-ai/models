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
"""train model"""
import argparse
import os
import random
import time
import math
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds
from mindspore import save_checkpoint
import mindspore.ops as ops
from mindspore.communication.management import init, get_rank
from src.dataset import ShapeNetDataset
from src.network import PointNetDenseCls
from src.loss import PointnetLoss

manualSeed = 2
random.seed(manualSeed)
np.random.seed(manualSeed)
mindspore.set_seed(manualSeed)

parser = argparse.ArgumentParser(description='MindSpore Pointnet Segmentation')
parser.add_argument(
    '--batchSize', type=int, default=64, help='input batch size')
parser.add_argument(
    '--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--device_id', type=int, default=5, help='device id')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='device id')
parser.add_argument('--device_target', default='Ascend', help='device target')
parser.add_argument('--data_url', type=str, default='../shapenetcore_partanno_segmentation_benchmark_v0'
                    , help="dataset path")
parser.add_argument('--train_url', type=str, default='./ckpts'
                    , help="ckpts path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--enable_modelarts', default=False, help="use feature transform")

args = parser.parse_args()

reshape = ops.Reshape()
print(args)


def train_model(_net_train, network, _dataset, _test_dataset, _num_classes, rank_id=0):
    """train_model"""
    print('loading data')
    print(time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))

    steps_per_epoch = _dataset.get_dataset_size() - 1
    print((time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime())), 'dataset output shape', _dataset.output_shapes())
    print("============== Starting Training ==============")
    best_accuracy = 0
    save_time = 0

    for epoch in range(1, args.nepoch + 1):
        test_dataset_iterator = _test_dataset.create_dict_iterator()
        next(test_dataset_iterator)
        valid_data = next(test_dataset_iterator)
        for batch_id, data in enumerate(_dataset.create_dict_iterator()):
            t_0 = time.time()
            points = data['data']
            label = data['label']
            loss = _net_train(points, label)
            print('Epoch : %d/%d  episode : %d/%d   Loss : %.4f  step_time: %.4f' %
                  (epoch, args.nepoch, batch_id, steps_per_epoch, np.mean(loss.asnumpy())
                   , (time.time() - t_0)))
            if batch_id % 9 == 0:
                data = valid_data
                points, label = data['point'], data['label']
                network.set_train(False)
                pred = network(points)

                pred = reshape(pred, (-1, _num_classes))
                pred_choice = ops.Argmax(axis=1, output_type=mindspore.int32)(pred)
                pred_np = pred_choice.asnumpy()
                target = reshape(label, (-1, 1))
                target = target[:, 0] - 1
                target_np = target.asnumpy()
                loss = net_loss(pred, label)
                correct = np.equal(pred_np, target_np).sum()
                accuracy = correct.item() / float(args.batchSize * 2500)
                print('[%d: %d/%d] %s  loss: %f accuracy: %.4f  best_accuracy: %f' %
                      (epoch, batch_id, steps_per_epoch, blue('test'), np.mean(loss.asnumpy())
                       , accuracy, best_accuracy))
                if rank_id == 0 and accuracy > best_accuracy:
                    save_time += 1
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                    save_checkpoint(network, os.path.join(local_train_url
                                                          , f"pointnet_network_epoch_{save_time}.ckpt"))

                    if args.enable_modelarts:
                        mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)
                    print(blue('save best model for epoch %d  accuracy : %f' % (epoch, accuracy)))


if __name__ == "__main__":
    blue = lambda x: '\033[94m' + x + '\033[0m'
    local_data_url = args.data_url
    local_train_url = args.train_url
    device_num = int(os.getenv("RANK_SIZE", "1"))
    shard_id = 0
    num_shards = device_num
    if args.enable_modelarts:
        device_id = int(os.getenv("DEVICE_ID"))
        import moxing as mox

        local_data_url = './cache/data'
        local_train_url = './cache/ckpt'
        device_target = args.device_target
        num_shards = int(os.getenv("RANK_SIZE"))
        shard_id = int(os.getenv("DEVICE_ID"))
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        context.set_context(save_graphs=False)
        if device_target == "Ascend":
            context.set_context(device_id=device_id)
            if device_num > 1:
                args.learning_rate *= 2
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
                init()
                local_data_url = os.path.join(local_data_url, str(device_id))
                local_train_url = os.path.join(local_train_url, "_" + str(get_rank()))
        else:
            raise ValueError("Unsupported platform.")
        import moxing as mox

        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
    else:
        # run on the local server
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        context.set_context(save_graphs=False)
        if args.device_target == "GPU":
            context.set_context(enable_graph_kernel=True)
        if device_num > 1:

            args.learning_rate = args.learning_rate * 2
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            shard_id = get_rank()

    if not os.path.exists(local_train_url):
        os.makedirs(local_train_url, exist_ok=True)

    dataset_sink_mode = False

    dataset_generator = ShapeNetDataset(
        root=local_data_url,
        classification=False,
        class_choice=[args.class_choice])
    test_dataset_generator = ShapeNetDataset(
        root=local_data_url,
        classification=False,
        class_choice=[args.class_choice],
        split='test',
        data_augmentation=False)

    dataset = ds.GeneratorDataset(dataset_generator, column_names=["data", "label"]
                                  , shuffle=True, num_shards=num_shards, shard_id=shard_id)
    dataset = dataset.batch(args.batchSize, drop_remainder=True)

    test_dataset = ds.GeneratorDataset(test_dataset_generator, ["point", "label"], shuffle=False,
                                       num_shards=1, shard_id=0)
    test_dataset = test_dataset.batch(args.batchSize, drop_remainder=True)

    num_classes = dataset_generator.num_seg_classes
    classifier = PointNetDenseCls(k=num_classes, feature_transform=args.feature_transform)
    classifier.set_train(True)
    if context.get_context("device_target") == "Ascend":
        classifier.to_float(mindspore.float16)
        for _, cell in classifier.cells_and_names():
            if isinstance(cell, nn.LogSoftmax):
                cell.to_float(mindspore.float32)

    num_batch = math.ceil(len(dataset_generator) / args.batchSize)

    milestone = list(range(80, 20000, 80))
    lr_rate = [args.learning_rate * 0.5 ** x for x in range(249)]
    learning_rates = nn.piecewise_constant_lr(milestone, lr_rate)
    optim = nn.Adam(params=classifier.trainable_params(), learning_rate=learning_rates
                    , beta1=0.9, beta2=0.999, loss_scale=1024)
    net_loss = PointnetLoss(num_class=num_classes, feature_transform=args.feature_transform)
    net_with_loss = nn.WithLossCell(classifier, net_loss)
    net_train = nn.TrainOneStepCell(net_with_loss, optim, sens=1024)

    train_model(_net_train=net_train, network=classifier, _dataset=dataset
                , _test_dataset=test_dataset, _num_classes=num_classes, rank_id=shard_id)
