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

import argparse
import os
import numpy as np

import mindspore
import mindspore.dataset as ds
from mindspore import Model, context, nn, DynamicLossScaleManager
from mindspore.train.callback import CheckpointConfig, LossMonitor, ModelCheckpoint, TimeMonitor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore import Tensor, export, load_checkpoint, load_param_into_net
import mindspore.ops as ops

import src.network as network
import src.dataset as dataset

mindspore.set_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", "-N", type=int, default=0, help="batch size")
parser.add_argument(
    "--data_url", "-f", type=str, default="/home/OctSqueeze/feature/KITTI_1000/", help="folder of training data"
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
    "--train_url", "-c", type=str, default="/home/OctSqueeze/ckpt/", help="folder of checkpoint and path"
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


class OctSqueezeNet(nn.Cell):
    def __init__(self):
        super().__init__()

        self.feature_layers = nn.CellList(network.network_creator([128, 128, 128, 128, 128], 6))
        self.aggregation_layers1 = nn.CellList(network.network_creator([128, 128, 128], 128 * 2))
        self.aggregation_layers2 = nn.CellList(network.network_creator([128, 128, 128], 128 * 2))
        self.softmax = nn.Softmax()
        self.cat = ops.Concat(axis=1)
        self.last_linear = nn.Dense(256, 256, activation=None)

    def construct(self, data):
        cur_node = data[0, 0, :, :6]
        parent_1 = data[0, 0, :, 6:12]
        parent_2 = data[0, 0, :, 12:18]
        parent_3 = data[0, 0, :, 18:24]
        for k in range(len(self.feature_layers)):
            cur_node = self.feature_layers[k](cur_node)
        for k in range(len(self.feature_layers)):
            parent_1 = self.feature_layers[k](parent_1)
        for k in range(len(self.feature_layers)):
            parent_2 = self.feature_layers[k](parent_2)
        for k in range(len(self.feature_layers)):
            parent_3 = self.feature_layers[k](parent_3)

        aggregation_c_p1 = self.cat((cur_node, parent_1))
        aggregation_c_p1 = self.aggregation_layers1[0](aggregation_c_p1)
        aggregation_c_p1 = self.aggregation_layers1[1](aggregation_c_p1)
        for k in range(2, len(self.aggregation_layers1), 2):
            aggregation_c_p1 = aggregation_c_p1 + self.aggregation_layers1[k](aggregation_c_p1)
            aggregation_c_p1 = self.aggregation_layers1[k + 1](aggregation_c_p1)

        aggregation_p1_p2 = self.cat((parent_1, parent_2))
        aggregation_p1_p2 = self.aggregation_layers1[0](aggregation_p1_p2)
        aggregation_p1_p2 = self.aggregation_layers1[1](aggregation_p1_p2)
        for k in range(2, len(self.aggregation_layers1), 2):
            aggregation_p1_p2 = aggregation_p1_p2 + self.aggregation_layers1[k](aggregation_p1_p2)
            aggregation_p1_p2 = self.aggregation_layers1[k + 1](aggregation_p1_p2)

        aggregation_c_p1_p2 = self.cat((aggregation_c_p1, aggregation_p1_p2))
        aggregation_c_p1_p2 = self.aggregation_layers2[0](aggregation_c_p1_p2)
        aggregation_c_p1_p2 = self.aggregation_layers2[1](aggregation_c_p1_p2)
        for k in range(2, len(self.aggregation_layers2), 2):
            aggregation_c_p1_p2 = aggregation_c_p1_p2 + self.aggregation_layers2[k](aggregation_c_p1_p2)
            aggregation_c_p1_p2 = self.aggregation_layers2[k + 1](aggregation_c_p1_p2)

        aggregation_p2_p3 = self.cat((parent_2, parent_3))
        aggregation_p2_p3 = self.aggregation_layers1[0](aggregation_p2_p3)
        aggregation_p2_p3 = self.aggregation_layers1[1](aggregation_p2_p3)
        for k in range(2, len(self.aggregation_layers1), 2):
            aggregation_p2_p3 = aggregation_p2_p3 + self.aggregation_layers1[k](aggregation_p2_p3)
            aggregation_p2_p3 = self.aggregation_layers1[k + 1](aggregation_p2_p3)

        aggregation_p1_p2_p3 = self.cat((aggregation_p1_p2, aggregation_p2_p3))
        aggregation_p1_p2_p3 = self.aggregation_layers2[0](aggregation_p1_p2_p3)
        aggregation_p1_p2_p3 = self.aggregation_layers2[1](aggregation_p1_p2_p3)
        for k in range(2, len(self.aggregation_layers2), 2):
            aggregation_p1_p2_p3 = aggregation_p1_p2_p3 + self.aggregation_layers2[k](aggregation_p1_p2_p3)
            aggregation_p1_p2_p3 = self.aggregation_layers2[k + 1](aggregation_p1_p2_p3)

        aggregation_c_p1_p2_p3 = self.cat((aggregation_c_p1_p2, aggregation_p1_p2_p3))

        feature = aggregation_c_p1_p2_p3.squeeze()
        out = self.softmax(self.last_linear(feature))

        return out


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
        data_generator = dataset.NodeInfoFolder(root=args.data_url)
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
        data_generator = dataset.NodeInfoFolder(root=args.data_url, node_size=args.batch_size)
        dataset = ds.GeneratorDataset(
            data_generator, ["feature", "label"], shuffle=False, num_parallel_workers=8, python_multiprocessing=True
        )

    dataset_sink_mode = not args.device_target == "CPU"

    # Save configuration
    config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
    ckpt_cb = ModelCheckpoint(prefix="octsqueeze_ckpt", directory=args.train_url, config=config_ck)
    step_per_epoch = dataset.get_dataset_size()
    time_cb = TimeMonitor(data_size=step_per_epoch)

    # Create loss and training network
    metrics = {"accuracy": nn.Accuracy(), "loss": nn.Loss()}
    train_net = Model(net, criterion, optimizer, metrics=metrics, amp_level="O3", loss_scale_manager=loss_scale_manager)

    print("============== Starting Training ==============")
    epoch_max = args.max_epochs
    if (args.is_distributed == 0) or (args.rank == 0):
        train_net.train(epoch_max, dataset, callbacks=[time_cb, LossMonitor(), ckpt_cb])
    else:
        train_net.train(epoch_max, dataset, callbacks=[time_cb, LossMonitor()])

    checkpoint_path = sorted([file for file in os.listdir(args.train_url) if ".ckpt" in file])[-1]
    """ export_octsqueeze """
    net = OctSqueezeNet()
    param_dict = load_checkpoint(os.path.join(args.train_url, checkpoint_path))
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.zeros([1, 1, 1000, 24]), mindspore.float32)
    export(net, input_arr, file_name=args.train_url + "/octsqueeze", file_format="AIR")
