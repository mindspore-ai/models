# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

import os
import argparse
import time

import mindspore.numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
import mindspore as ms
from mindspore import nn, Tensor, Model
from mindspore import dtype as mstype
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size

from src.net import CoarseNet
from src.net import FineNet
from src.loss import CustomLoss
from src.data_loader import TrainDatasetGenerator

ms.set_seed(0)


class FineWithLossCell(nn.Cell):
    def __init__(self, network, loss_fn):
        super(FineWithLossCell, self).__init__(auto_prefix=False)
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, rgb, coarse_dept, ground_truth):
        pred_depth = self.network(rgb, coarse_dept)
        return self.loss_fn(pred_depth, ground_truth)

    def backbone_network(self):
        return self.network


def check_folder(input_dir):
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
        print("create ", input_dir, " success")
    else:
        print(input_dir, " already exists, no need to create")


def create_train_data_set(train_data_dir, batch_size, epochs, is_distributed):
    dataset_generator = TrainDatasetGenerator(train_data_dir)

    if is_distributed == 1:
        device_num = get_group_size()
        rank = get_rank()
        print("device_num: ", device_num, " rank: ", rank)
        train_dataset = ds.GeneratorDataset(
            dataset_generator,
            ["rgb", "ground_truth"],
            shuffle=True,
            num_shards=device_num,
            shard_id=rank,
            num_parallel_workers=num_parallel_workers,
        )
    elif is_distributed == 0:
        train_dataset = ds.GeneratorDataset(
            dataset_generator, ["rgb", "ground_truth"], shuffle=True, num_parallel_workers=num_parallel_workers
        )

    type_cast_op_image = C.TypeCast(mstype.float32)
    type_cast_op_depth = C.TypeCast(mstype.float32)

    hwc2chw = CV.HWC2CHW()
    crop = CV.Crop((12, 16), (456, 608))
    rgb_resize = CV.Resize((228, 304))
    depth_resize = CV.Resize((55, 74))

    rgb_data_transform = [
        crop,
        CV.RandomColorAdjust(brightness=(0.8, 1.0)),
        CV.RandomSharpness(degrees=(0.8, 1.2)),
        type_cast_op_image,
        rgb_resize,
        hwc2chw,
    ]

    depth_data_transform = [crop, type_cast_op_depth, depth_resize, hwc2chw]

    rgb_depth_transform = [CV.RandomHorizontalFlip(0.5)]

    train_dataset = train_dataset.map(
        operations=rgb_depth_transform, input_columns=["rgb", "ground_truth"], num_parallel_workers=num_parallel_workers
    )
    train_dataset = train_dataset.map(
        operations=rgb_data_transform, input_columns="rgb", num_parallel_workers=num_parallel_workers
    )
    train_dataset = train_dataset.map(
        operations=depth_data_transform, input_columns="ground_truth", num_parallel_workers=num_parallel_workers
    )

    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size=batch_size)
    train_dataset = train_dataset.repeat(epochs)

    return train_dataset


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
    parser = argparse.ArgumentParser(description="MindSpore Depth Estimation Demo")
    parser.add_argument(
        "--device_target",
        type=str,
        default="Ascend",
        choices=["Ascend", "GPU", "CPU"],
        help="device where the code will be implemented (default: CPU)",
    )
    parser.add_argument(
        "--is_distributed", type=int, default=0, help="Distribute train or not, 1 for yes, 0 for no. Default: 0"
    )
    parser.add_argument("--rank", type=int, default=0, help="Local rank of distributed. Default: 0")
    parser.add_argument("--group_size", type=int, default=1, help="World size of device. Default: 1")
    parser.add_argument("--pre_trained", type=int, default=0, help="Pretrained checkpoint path")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--epochs_coarse", type=int, default=20)
    parser.add_argument("--fine_net_train_step", type=int, default=15830)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate_min", type=float, default=1e-5)
    parser.add_argument("--learning_rate_max", type=float, default=1e-4)
    parser.add_argument("--num_parallel_works", type=int, default=16)
    parser.add_argument("--continue_training", type=int, default=0)
    parser.add_argument("--data_url", type=str, default="./DepthNet_dataset", help="dataset path")
    parser.add_argument("--train_url", type=str, default="./Model", help="train output path")
    args = parser.parse_args()

    _batch_size = args.batch_size
    epochs_coarse = args.epochs_coarse
    num_parallel_workers = args.num_parallel_works
    learning_rate = args.learning_rate_max
    _is_distributed = args.is_distributed

    if args.pre_trained == 1:
        learning_rate = args.learning_rate_min

    if _is_distributed == 0:
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)

    if _is_distributed == 1:
        network_init(args)
        parallel_init(args)
        _batch_size = 4

    train_data_direc = os.path.join(args.data_url, "Train")
    model_dir = args.train_url
    ckpt_dir = model_dir + "/Ckpt"

    check_folder(model_dir)
    check_folder(ckpt_dir)
    train_ds = create_train_data_set(train_data_direc, _batch_size, epochs_coarse, _is_distributed)

    coarse_net = CoarseNet()
    fine_net = FineNet()
    if context.get_context("mode") == context.GRAPH_MODE:
        coarse_net.to_float(ms.float16)
        fine_net.to_float(ms.float16)

    custom_loss = CustomLoss()
    coarse_optim = nn.Adam(params=coarse_net.trainable_params(), learning_rate=learning_rate)

    coarse_net_with_criterion = nn.WithLossCell(coarse_net, custom_loss)
    coarse_train_onestep = nn.TrainOneStepCell(coarse_net_with_criterion, coarse_optim)

    if args.pre_trained == 1:
        print("load pre trained coarse model")
        pre_trained_coarse_model = "./Model/PreTrained/pre_trained.ckpt"
        pre_trained_coarse_param = load_checkpoint(pre_trained_coarse_model)
        load_param_into_net(coarse_net, pre_trained_coarse_param)

    continue_step = -1
    if args.continue_training == 1:
        print("continue training coarse net")
        coarse_net_param_dict = load_checkpoint(ckpt_dir + "/FinalCoarseNet.ckpt")
        load_param_into_net(coarse_net, coarse_net_param_dict)

    step = 0
    for data in train_ds.create_dict_iterator():
        time_start = time.time()
        coarse_train_onestep(data["rgb"], data["ground_truth"])
        loss_val = coarse_net_with_criterion(data["rgb"], data["ground_truth"])
        time_end = time.time()
        time_cost = time_end - time_start

        if step % 10 == 0:
            print(
                "training coarse net, step: ", step + continue_step + 1, "  loss: ", loss_val, " time cost: ", time_cost
            )

        if step % 10000 == 0:
            if _is_distributed == 0:
                ms.save_checkpoint(coarse_net, ckpt_dir + "/CoarseNet_" + str(step + continue_step + 1) + ".ckpt")
            elif _is_distributed == 1:
                ms.save_checkpoint(
                    coarse_net,
                    ckpt_dir + "/CoarseNet_" + str(step + continue_step + 1) + "_rank" + str(get_rank()) + ".ckpt",
                )

        step += 1

    if _is_distributed == 0:
        ms.save_checkpoint(coarse_net, ckpt_dir + "/FinalCoarseNet.ckpt")
    elif _is_distributed == 1:
        ms.save_checkpoint(coarse_net, ckpt_dir + "/FinalCoarseNet_rank" + str(get_rank()) + ".ckpt")

    coarse_net_fix = CoarseNet()
    if _is_distributed == 0:
        coarse_net_param_dict = load_checkpoint(ckpt_dir + "/FinalCoarseNet.ckpt")
    elif _is_distributed == 1:
        coarse_net_param_dict = load_checkpoint(ckpt_dir + "/FinalCoarseNet_rank" + str(get_rank()) + ".ckpt")
    load_param_into_net(coarse_net_fix, coarse_net_param_dict)
    coarse_net_fix.set_train(mode=False)
    fixed_coarse_model = Model(coarse_net_fix)

    fine_optim = nn.Adam(params=fine_net.trainable_params(), learning_rate=args.learning_rate_min)
    fine_net_with_criterion = FineWithLossCell(fine_net, custom_loss)
    fine_train_onestep = nn.TrainOneStepCell(fine_net_with_criterion, fine_optim)

    step = 0
    for data in train_ds.create_dict_iterator():
        if step == args.fine_net_train_step:
            break
        coarse_depth = fixed_coarse_model.predict(Tensor(data["rgb"]))
        coarse_depth = np.clip(coarse_depth, 0.1, 10)
        time_start = time.time()
        fine_train_onestep(data["rgb"], coarse_depth, data["ground_truth"])
        loss_val = fine_net_with_criterion(data["rgb"], coarse_depth, data["ground_truth"])
        time_end = time.time()
        time_cost = time_end - time_start

        if step % 10 == 0:
            print("training fine net, step: ", step, "  loss: ", loss_val, " time cost: ", time_cost)

        if step % 10000 == 0:
            if _is_distributed == 0:
                ms.save_checkpoint(fine_net, ckpt_dir + "/FineNet_" + str(step) + ".ckpt")
            elif _is_distributed == 1:
                ms.save_checkpoint(fine_net, ckpt_dir + "/FineNet_" + str(step) + "_rank" + str(get_rank()) + ".ckpt")

        step += 1

    if _is_distributed == 0:
        ms.save_checkpoint(fine_net, ckpt_dir + "/FinalFineNet.ckpt")
    elif _is_distributed == 1:
        ms.save_checkpoint(fine_net, ckpt_dir + "/FinalFineNet_rank" + str(get_rank()) + ".ckpt")
