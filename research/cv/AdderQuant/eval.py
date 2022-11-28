# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""eval resnet."""
import argparse
import multiprocessing
import mindspore as ms
import mindspore.dataset as ds
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init, get_rank, get_group_size
import adder_quant
from res20_adder import resnet20 as resnet


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--device_target', default="GPU", type=str)
    parser.add_argument('--class_num', default=10, type=int)
    parser.add_argument('--checkpoint_file_path',
                        default="/path/to/ckpt", type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--device_num', default=1, type=int)
    parser.add_argument('--eval_dataset_path',
                        default="path/to/cifar-10-verify-bin/",
                        type=str)
    parser.add_argument('--train_dataset_path',
                        default="path/to/cifar-10-batches-bin/",
                        type=str)
    parser.add_argument('--image_size', default=32, type=int)
    return parser


def _get_rank_info(distribute):
    """
    get rank size and rank id
    """
    if distribute:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
    else:
        rank_id = 0
        device_num = 1
    return device_num, rank_id


def get_num_parallel_workers(num_parallel_workers):
    """
    Get num_parallel_workers used in dataset operations.
    If num_parallel_workers > the real CPU cores number, set num_parallel_workers = the real CPU cores number.
    """
    cores = multiprocessing.cpu_count()
    if isinstance(num_parallel_workers, int):
        if cores < num_parallel_workers:
            print("The num_parallel_workers {} is set too large, now set it {}".format(num_parallel_workers, cores))
            num_parallel_workers = cores
    else:
        print("The num_parallel_workers {} is invalid, now set it {}".format(num_parallel_workers, min(cores, 8)))
        num_parallel_workers = min(cores, 8)
    return num_parallel_workers


def create_dataset(dataset_path, do_train, batch_size=32, image_size=224, distribute=False, enable_cache=False,
                   cache_session_id=None):
    """
    create a train or evaluate cifar10 dataset for resnet50
    """
    device_num, rank_id = _get_rank_info(distribute)
    ds.config.set_prefetch_size(64)
    if device_num == 1:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=get_num_parallel_workers(8), shuffle=False)
    else:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=get_num_parallel_workers(8), shuffle=False,
                                     num_shards=device_num, shard_id=rank_id)

    # define map operations
    trans = []
    if do_train:
        trans += [
            ds.vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            ds.vision.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        ds.vision.Resize((image_size, image_size)),
        ds.vision.Rescale(1.0 / 255.0, 0.0),
        ds.vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ds.vision.HWC2CHW()
    ]

    type_cast_op = ds.transforms.transforms.TypeCast(ms.int32)

    data_set = data_set.map(operations=type_cast_op, input_columns="label",
                            num_parallel_workers=get_num_parallel_workers(8))
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=trans, input_columns="image",
                                num_parallel_workers=get_num_parallel_workers(8), cache=eval_cache)
    else:
        data_set = data_set.map(operations=trans, input_columns="image",
                                num_parallel_workers=get_num_parallel_workers(8))

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set


class CrossEntropySmooth(nn.LossBase):
    """CrossEntropy"""

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, ms.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), ms.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss


def count_scale(train_loader, model):
    model.set_train(False)
    for i, (x, _) in enumerate(train_loader):
        input_var = ms.Parameter(x)
        _ = model(input_var)
        if i > 200:
            break


def adjust_BN(train_loader, model):
    model.set_train(True)
    for i, (x, _) in enumerate(train_loader):
        input_var = ms.Parameter(x)
        _ = model(input_var)
        if i > 200:
            break


def eval_net():
    parser = argparse.ArgumentParser('Resnet_Adder evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    ms.set_seed(1)
    target = args.device_target
    # init context
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target, save_graphs=False)
    # create dataset
    dataset_train = create_dataset(dataset_path=args.train_dataset_path, do_train=True, batch_size=args.batch_size,
                                   image_size=args.image_size)
    dataset_val = create_dataset(dataset_path=args.eval_dataset_path, do_train=False, batch_size=args.batch_size,
                                 image_size=args.image_size)
    # define net
    net = resnet(class_num=args.class_num)
    # load checkpoint
    param_dict = ms.load_checkpoint(args.checkpoint_file_path, strict_load=True)
    ms.load_param_into_net(net, param_dict)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    print('count scale')
    count_scale(dataset_train, net)

    print('set w and a para')
    for _, m in net.cells_and_names():
        if isinstance(m, adder_quant.QAdder2dKmeansGroupShare):
            m.cluster_by_kmeans()
            m.set_para()
            m.counting = False

    print('adjust BN')
    adjust_BN(dataset_train, net)

    # define model
    model = ms.Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
    res = model.eval(dataset_val)
    print("result:", res, "ckpt=", args.checkpoint_file_path)


if __name__ == '__main__':
    eval_net()
