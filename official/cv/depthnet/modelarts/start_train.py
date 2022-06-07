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

import os
import argparse
import time
import datetime
import glob
import numpy as np
import moxing as mox

import mindspore.numpy as mnp
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
import mindspore as ms
from mindspore import nn, Tensor, Model
from mindspore import dtype as mstype
from mindspore import context
from mindspore import export, load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size

from src.net import CoarseNet
from src.net import FineNet
from src.loss import CustomLoss
from src.data_loader import TrainDatasetGenerator

ms.set_seed(0)
code_dir = os.path.dirname(__file__)
work_dir = os.getcwd()
print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

def parse_args():
    # train args
    parser = argparse.ArgumentParser(description='MindSpore DepthNet Train Demo')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--is_distributed', type=int, default=0,
                        help='Distribute train or not, 1 for yes, 0 for no. Default: 0')
    parser.add_argument('--rank', type=int, default=0, help='Local rank of distributed. Default: 0')
    parser.add_argument('--group_size', type=int, default=1, help='World size of device. Default: 1')
    parser.add_argument('--pre_trained', type=int, default=0, help='Pretrained checkpoint path')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--epochs_coarse', type=int, default=20)
    parser.add_argument('--fine_net_train_step', type=int, default=15830)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate_min', type=float, default=1e-5)
    parser.add_argument('--learning_rate_max', type=float, default=1e-4)
    parser.add_argument('--num_parallel_works', type=int, default=16)
    parser.add_argument('--continue_training', type=int, default=0)

    # ModelArts args
    parser.add_argument('--coarse_or_fine', type=str, default='coarse',
                        choices=['coarse', 'fine'], help='input coarse or fine, export coarse or fine model')
    parser.add_argument('--data_url', type=str, default='./data/', help='dataset path')
    parser.add_argument('--train_url', type=str, default='./output/', help='train output path')
    parser.add_argument("--training_dataset", type=str, default="/cache/dataset/nyu2_train")
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset/")
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result/")
    parser.add_argument("--ckpt_path", type=str, default="/cache/result/")

    args_opt = parser.parse_args()

    return args_opt


def obs_data2modelarts(FLAGS):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(FLAGS.data_url, FLAGS.modelarts_data_dir))
    mox.file.copy_parallel(src_url=FLAGS.data_url, dst_url=FLAGS.modelarts_data_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(FLAGS.modelarts_data_dir)
    print("===>>>before Files:", files)


def modelarts_result2obs(FLAGS):
    """
    Copy debug data from modelarts to obs.
    According to the switch flags, the debug data may contains auto tune repository,
    dump data for precision comparison, even the computation graph and profiling data.
    """

    mox.file.copy_parallel(src_url=FLAGS.modelarts_result_dir, dst_url=FLAGS.train_url)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(FLAGS.modelarts_result_dir,
                                                                                  FLAGS.train_url))
    files = os.listdir(FLAGS.modelarts_result_dir)
    print("===>>>current Files:", files)
    mox.file.copy(src_url='FinalCoarseNet.air', dst_url=FLAGS.train_url+'FinalCoarseNet.air')
    mox.file.copy(src_url='FinalFineNet.air', dst_url=FLAGS.train_url + 'FinalFineNet.air')


def export_AIR(args_opt):
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    input_rgb = np.random.uniform(0.0, 1.0, size=[1, 3, 228, 304]).astype(np.float32)
    input_depth = np.random.uniform(0.0, 1.0, size=[1, 1, 55, 74]).astype(np.float32)
    ckpt_list = glob.glob(args_opt.modelarts_result_dir + "*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    # export both coarse and fine net
    if args_opt.coarse_or_fine == 'coarse':
        coarse_net = CoarseNet()
        coarse_net_file_name = os.path.join(args_opt.modelarts_result_dir, "FinalCoarseNet.ckpt")
        coarse_param_dict = load_checkpoint(coarse_net_file_name)
        load_param_into_net(coarse_net, coarse_param_dict)
        # coarse_net only one input
        export(coarse_net, Tensor(input_rgb), file_name='FinalCoarseNet', file_format='AIR')

        fine_net = FineNet()
        fine_net_file_name = os.path.join(args_opt.modelarts_result_dir, "FinalFineNet.ckpt")
        fine_param_dict = load_checkpoint(fine_net_file_name)
        load_param_into_net(fine_net, fine_param_dict)
        # fine_net with two
        export(fine_net, Tensor(input_rgb), Tensor(input_depth), file_name='FinalFineNet', file_format='AIR')

    # only export fine net
    else:
        fine_net = FineNet()
        fine_net_file_name = os.path.join(args_opt.modelarts_result_dir, "FinalFineNet.ckpt")
        fine_param_dict = load_checkpoint(fine_net_file_name)
        load_param_into_net(fine_net, fine_param_dict)
        # fine_net with two
        export(fine_net, Tensor(input_rgb), Tensor(input_depth), file_name='FinalFineNet', file_format='AIR')


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
        train_dataset = ds.GeneratorDataset(dataset_generator, ["rgb", "ground_truth"], shuffle=True,
                                            num_shards=device_num, shard_id=rank,
                                            num_parallel_workers=num_parallel_workers)
    elif is_distributed == 0:
        train_dataset = ds.GeneratorDataset(dataset_generator, ["rgb", "ground_truth"], shuffle=True,
                                            num_parallel_workers=num_parallel_workers)

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
        type_cast_op_image, rgb_resize, hwc2chw
    ]

    depth_data_transform = [
        crop,
        type_cast_op_depth, depth_resize, hwc2chw
    ]

    rgb_depth_transform = [
        CV.RandomHorizontalFlip(0.5)
    ]

    train_dataset = train_dataset.map(operations=rgb_depth_transform, input_columns=["rgb", "ground_truth"],
                                      num_parallel_workers=num_parallel_workers)
    train_dataset = train_dataset.map(operations=rgb_data_transform, input_columns="rgb",
                                      num_parallel_workers=num_parallel_workers)
    train_dataset = train_dataset.map(operations=depth_data_transform, input_columns="ground_truth",
                                      num_parallel_workers=num_parallel_workers)

    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size=batch_size)
    train_dataset = train_dataset.repeat(epochs)

    return train_dataset


def network_init(args_opt):
    devid = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                        device_target=args_opt.device_target, save_graphs=False, device_id=devid)
    # Init distributed
    if args_opt.is_distributed:
        if args_opt.device_target == "Ascend":
            init()
        else:
            init("nccl")
        args_opt.rank = get_rank()
        args_opt.group_size = get_group_size()


def parallel_init(args_opt):
    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    degree = 1
    if args_opt.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True,
                                      device_num=degree, parameter_broadcast=True)


def train_prepare(cfg):
    # train_modelarts local variables
    _batch_size = args.batch_size
    epochs_coarse = args.epochs_coarse
    learning_rate = args.learning_rate_max
    _is_distributed = args.is_distributed
    # default no
    if cfg.pre_trained == 1:
        learning_rate = cfg.learning_rate_min

    # default no
    if _is_distributed == 0:
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=cfg.device_id)

    if _is_distributed == 1:
        network_init(cfg)
        parallel_init(cfg)
        _batch_size = 4

    train_data_direc = cfg.training_dataset
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

    # default no
    if cfg.pre_trained == 1:
        print("load pre trained coarse model")
        pre_trained_coarse_model = "./Model/PreTrained/pre_trained.ckpt"
        pre_trained_coarse_param = load_checkpoint(pre_trained_coarse_model)
        load_param_into_net(coarse_net, pre_trained_coarse_param)

    # default no
    if cfg.continue_training == 1:
        print("continue training coarse net")
        coarse_net_param_dict = load_checkpoint(ckpt_dir + "FinalCoarseNet.ckpt")
        load_param_into_net(coarse_net, coarse_net_param_dict)
    return train_ds, coarse_train_onestep, coarse_net_with_criterion, coarse_net, fine_net, custom_loss

def train_modelarts(cfg):
    # train_prepare and args init
    train_ds, coarse_train_onestep, coarse_net_with_criterion, coarse_net, fine_net, custom_loss = train_prepare(cfg)
    _is_distributed = args.is_distributed
    continue_step = -1
    ckpt_dir = cfg.ckpt_path
    check_folder(ckpt_dir)

    # start modelarts train
    step = 0
    for data in train_ds.create_dict_iterator():
        time_start = time.time()
        coarse_train_onestep(data["rgb"], data["ground_truth"])
        loss_val = coarse_net_with_criterion(data["rgb"], data["ground_truth"])
        time_end = time.time()
        time_cost = time_end - time_start

        if step % 10 == 0:
            print("training coarse net, step: ", step + continue_step + 1, "  loss: ", loss_val,
                  " time cost: ", time_cost)

        if step % 10000 == 0:
            if _is_distributed == 0:
                ms.save_checkpoint(coarse_net, ckpt_dir + "CoarseNet_" + str(step + continue_step + 1) + ".ckpt")
            elif _is_distributed == 1:
                ms.save_checkpoint(coarse_net, ckpt_dir + "CoarseNet_" + str(step + continue_step + 1) +
                                   "_rank" + str(get_rank()) + ".ckpt")

        step += 1

    # save final Coarse net
    if _is_distributed == 0:
        ms.save_checkpoint(coarse_net, ckpt_dir + "FinalCoarseNet.ckpt")
    elif _is_distributed == 1:
        ms.save_checkpoint(coarse_net, ckpt_dir + "FinalCoarseNet_rank" + str(get_rank()) + ".ckpt")

    coarse_net_fix = CoarseNet()
    if _is_distributed == 0:
        coarse_net_param_dict = load_checkpoint(ckpt_dir + "FinalCoarseNet.ckpt")
    elif _is_distributed == 1:
        coarse_net_param_dict = load_checkpoint(ckpt_dir + "FinalCoarseNet_rank" + str(get_rank()) + ".ckpt")
    load_param_into_net(coarse_net_fix, coarse_net_param_dict)
    coarse_net_fix.set_train(mode=False)
    fixed_coarse_model = Model(coarse_net_fix)

    fine_optim = nn.Adam(params=fine_net.trainable_params(), learning_rate=cfg.learning_rate_min)
    fine_net_with_criterion = FineWithLossCell(fine_net, custom_loss)
    fine_train_onestep = nn.TrainOneStepCell(fine_net_with_criterion, fine_optim)

    step = 0
    for data in train_ds.create_dict_iterator():
        # default 15830
        if step == cfg.fine_net_train_step:
            break
        coarse_depth = fixed_coarse_model.predict(Tensor(data["rgb"]))
        coarse_depth = mnp.clip(coarse_depth, 0.1, 10)
        time_start = time.time()
        fine_train_onestep(data["rgb"], coarse_depth, data["ground_truth"])
        loss_val = fine_net_with_criterion(data["rgb"], coarse_depth, data["ground_truth"])
        time_end = time.time()
        time_cost = time_end - time_start

        if step % 10 == 0:
            print("training fine net, step: ", step, "  loss: ", loss_val, " time cost: ", time_cost)

        if step % 10000 == 0:
            if _is_distributed == 0:
                ms.save_checkpoint(fine_net, ckpt_dir + "FineNet_" + str(step) + ".ckpt")
            elif _is_distributed == 1:
                ms.save_checkpoint(fine_net, ckpt_dir + "FineNet_" + str(step) + "_rank" + str(get_rank()) + ".ckpt")

        step += 1

    # save final Fine net
    if _is_distributed == 0:
        ms.save_checkpoint(fine_net, ckpt_dir + "FinalFineNet.ckpt")
    elif _is_distributed == 1:
        ms.save_checkpoint(fine_net, ckpt_dir + "FinalFineNet_rank" + str(get_rank()) + ".ckpt")


if __name__ == "__main__":
    args = parse_args()
    # train_modelarts global variable
    num_parallel_workers = args.num_parallel_works

    obs_data2modelarts(args)
    print('train config:\n', args)
    train_modelarts(args)
    export_AIR(args)
    modelarts_result2obs(args)
