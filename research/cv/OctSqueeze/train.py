# Copyright 2021 Huawei Technologies Co., Ltd
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

import argparse
import os

import mindspore
import mindspore.dataset as ds
from mindspore import Model, context, nn, DynamicLossScaleManager
from mindspore.train.callback import CheckpointConfig, LossMonitor, ModelCheckpoint, TimeMonitor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode

import src.network as network
import src.dataset as dataset

mindspore.set_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", "-N", type=int, default=0, help="batch size")
parser.add_argument(
    "--train", "-f", type=str, default="/home/OctSqueeze/feature/KITTI_1000/", help="folder of training data"
)
parser.add_argument("--max_epochs", "-e", type=int, default=200, help="max epochs")
parser.add_argument(
    "--device_target",
    type=str,
    default="Ascend",
    choices=["Ascend", "GPU", "CPU"],
    help="device where the code will be implemented",
)
parser.add_argument(
    "--checkpoint", "-c", type=str, default="/home/OctSqueeze/ckpt/", help="folder of checkpoint and path"
)
# For Parallel
parser.add_argument(
    "--is_distributed", type=int, default=0, help="Distribute train or not, 1 for yes, 0 for no. Default: 1"
)
parser.add_argument("--rank", type=int, default=0, help="Local rank of distributed. Default: 0")
parser.add_argument("--group_size", type=int, default=1, help="World size of device. Default: 1")
args = parser.parse_args()


def network_init(argvs):
    devid = int(os.getenv("DEVICE_ID", "0"))
    context.set_context(
        mode=context.GRAPH_MODE,
        enable_auto_mixed_precision=True,
        device_target=argvs.device_target,
        save_graphs=False,
        device_id=devid,
    )
    # Init distributed
    if argvs.is_distributed:
        if argvs.device_target == "Ascend":
            init()
        else:
            init("nccl")
        argvs.rank = get_rank()
        argvs.group_size = get_group_size()


def parallel_init(argv):
    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    degree = 1
    if argv.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    context.set_auto_parallel_context(
        parallel_mode=parallel_mode, gradients_mean=True, device_num=degree, parameter_broadcast=True
    )


if __name__ == "__main__":
    if args.is_distributed == 1:
        network_init(args)
        parallel_init(args)
    else:
        # Configure operation information
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=0)

    ## Create networks
    net = network.OctSqueezeNet()

    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(net.trainable_params())

    # Define Loss Scale, optimizer and model
    loss_scale_manager = DynamicLossScaleManager()

    ## Read data
    if args.is_distributed == 1:
        mindspore.dataset.config.set_enable_shared_mem(False)
        data_generator = dataset.NodeInfoFolder(root=args.train)
        dataset = ds.GeneratorDataset(
            data_generator,
            ["feature", "label"],
            shuffle=False,
            python_multiprocessing=True,
            num_parallel_workers=8,
            num_shards=args.group_size,
            shard_id=args.rank,
        )
    else:
        data_generator = dataset.NodeInfoFolder(root=args.train, node_size=args.batch_size)
        dataset = ds.GeneratorDataset(
            data_generator, ["feature", "label"], shuffle=False, num_parallel_workers=8, python_multiprocessing=True
        )

    dataset_sink_mode = not args.device_target == "CPU"

    # Save configuration
    config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
    ckpt_cb = ModelCheckpoint(prefix="octsqueeze_ckp", directory=args.checkpoint, config=config_ck)
    step_per_epoch = dataset.get_dataset_size()
    time_cb = TimeMonitor(data_size=step_per_epoch)

    # Create loss and training network
    metrics = {"accuracy": nn.Accuracy(), "loss": nn.Loss()}
    train_net = Model(net, criterion, optimizer, metrics=metrics, amp_level="O3", loss_scale_manager=loss_scale_manager)

    print("============== Starting Training ==============")
    epoch_max = args.max_epochs
    if (args.is_distributed == 0) or (args.rank == 0):
        train_net.train(epoch_max, dataset, callbacks=[time_cb, LossMonitor(), ckpt_cb], dataset_sink_mode=True)
    else:
        train_net.train(epoch_max, dataset, callbacks=[time_cb, LossMonitor()], dataset_sink_mode=True)
