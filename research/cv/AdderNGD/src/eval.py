# Copyright 2023 Huawei Technologies Co., Ltd
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
from res20_adder import resnet20 as resnet
import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank, get_group_size


def get_args_parser():
    """
    get input parameters
    """
    parser = argparse.ArgumentParser('adderngd', add_help=False)
    parser.add_argument('--device_target', default="GPU", type=str)
    parser.add_argument('--class_num', default=100, type=int)
    parser.add_argument('--checkpoint_file_path',
                        default="ckpt/adderngd_resnet20_cifar100_7034.ckpt", type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--device_num', default=1, type=int)
    parser.add_argument('--eval_dataset_path',
                        default="data/cifar-100-binary/",
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
    If num_parallel_workers > the real CPU cores number,
    set num_parallel_workers = the real CPU cores number.
    """
    cores = multiprocessing.cpu_count()
    if isinstance(num_parallel_workers, int):
        if cores < num_parallel_workers:
            num_parallel_workers = cores
    else:
        num_parallel_workers = min(cores, 8)
    return num_parallel_workers


def create_dataset(dataset_path, do_train, batch_size=32, image_size=224,
                   distribute=False, enable_cache=False, cache_session_id=None):
    """
    create a train or evaluate cifar10 dataset for resnet50
    """
    device_num, rank_id = _get_rank_info(distribute)
    ds.config.set_prefetch_size(64)
    if do_train:
        usage = 'train'
    else:
        usage = 'test'
    if device_num == 1:
        data_set = ds.Cifar100Dataset(dataset_path, usage=usage,
                                      num_parallel_workers=get_num_parallel_workers(8),
                                      shuffle=False)
    else:
        data_set = ds.Cifar100Dataset(dataset_path, usage=usage,
                                      num_parallel_workers=get_num_parallel_workers(8),
                                      shuffle=False,
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

    data_set = data_set.map(operations=type_cast_op, input_columns="fine_label",
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
    data_set = data_set.batch(batch_size, drop_remainder=False)

    return data_set


def eval_net():
    """
    perform evaluation
    """
    parser = argparse.ArgumentParser('evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    ms.set_seed(1)
    target = args.device_target
    # init context
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target, save_graphs=False)
    # create dataset
    dataset_val = create_dataset(dataset_path=args.eval_dataset_path, do_train=False,
                                 batch_size=args.batch_size, image_size=args.image_size)
    # define net
    net = resnet(class_num=args.class_num)
    # load checkpoint
    param_dict = ms.load_checkpoint(args.checkpoint_file_path, strict_load=True)
    ms.load_param_into_net(net, param_dict)

    topk = nn.Top1CategoricalAccuracy()
    topk.clear()
    for _, (image, _, label) in enumerate(dataset_val):
        output = net(image)
        topk.update(output, label)
    res = topk.eval()
    print("result:", res, "ckpt=", args.checkpoint_file_path)


if __name__ == '__main__':
    eval_net()
