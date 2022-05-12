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
"""train"""
import os
import glob
import ast
import datetime
import argparse
import numpy as np
import moxing as mox

from mindspore import Model
from mindspore import Tensor, export, context
from mindspore import nn
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.parallel import set_algo_parameters

from src.args import args
from src.tools.callback import EvaluateCallBack
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, get_model, pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer

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
    print("===>>>Files:", files)


def modelarts_result2obs(FLAGS):
    """
    Copy debug data from modelarts to obs.
    According to the switch flags, the debug data may contains auto tune repository,
    dump data for precision comparison, even the computation graph and profiling data.
    """

    mox.file.copy_parallel(src_url=FLAGS.modelarts_result_dir, dst_url=FLAGS.train_url)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(FLAGS.modelarts_result_dir,
                                                                                  FLAGS.train_url))
    files = os.listdir()
    print("===>>>current Files:", files)
    mox.file.copy(src_url='swim-transformer.air', dst_url=FLAGS.train_url+'/swim-transformer.air')


def export_AIR(args_opt):
    """start modelarts export"""
    ckpt_list = glob.glob(args_opt.modelarts_result_dir + "/swim-transformer*.ckpt")

    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    net_ = get_model(args)
    get_model(args)
    param_dict_ = load_checkpoint(ckpt_model)
    load_param_into_net(net_, param_dict_)

    input_arr = Tensor(np.zeros([1, 3, 224, 224], np.float32))
    export(net_, input_arr, file_name='swim-transformer', file_format='AIR')


def main():
    set_seed(args.seed)
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)

    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    # rank = set_device(args)
    ckpt_save_dir = args.modelarts_result_dir

    if args.run_distribute:
        if args.device_target == "Ascend":
            init()
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            rank = get_rank()

        elif args.device_target == "GPU":
            init('nccl')
            context.reset_auto_parallel_context()
            rank = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)

    # get model and cast amp_level
    net = get_model(args)
    cast_amp(net)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)
    if args.pretrained:
        args.pretrained = load_checkpoint(args.modelarts_data_dir+'/'+args.pre_trained)
        pretrained(args, net)

    data = get_dataset(args)
    batch_num = data.train_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)

    eval_network = nn.WithEvalCell(net, criterion, args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss, metrics={"acc", "loss"},
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)

    config_ck = CheckpointConfig(save_checkpoint_steps=data.train_dataset.get_dataset_size(),
                                 keep_checkpoint_max=args.save_every)
    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())
    ckpoint_cb = ModelCheckpoint(prefix="swim-transformer", directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor()
    eval_cb = EvaluateCallBack(model, eval_dataset=data.val_dataset, src_url=ckpt_save_dir,
                               train_url=os.path.join(args.train_url, "ckpt_" + str(rank)),
                               save_freq=args.save_every)

    print("begin train")
    model.train(int(args.epochs - args.start_epoch), data.train_dataset,
                callbacks=[time_cb, ckpoint_cb, loss_cb, eval_cb],
                dataset_sink_mode=True)
    print("train success")

    ## start export air
    export_AIR(args)
    ## copy result from modelarts to obs
    modelarts_result2obs(args)


if __name__ == '__main__':
    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))
    parser = argparse.ArgumentParser()

    # ===== Modelarts config ======== #
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument("--data_url", type=str, default="./dataset")
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset")
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")

    # Architecture
    parser.add_argument("-a", "--arch", metavar="ARCH", default="swin_tiny_patch4_window7_224",
                        help="model architecture")

    # ===== Dataset ===== #
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--mix_up", default=0., type=float, help="mix up")
    parser.add_argument("--cutmix", default=1.0, type=float, help="cut mix")
    parser.add_argument("--auto_augment", default="rand-m9-mstd0.5-inc1", type=str, help="auto augment type")
    parser.add_argument("--interpolation", default="bicubic", type=str, help="interpolation type")
    parser.add_argument("--re_prob", default=0.25, type=float)
    parser.add_argument("--re_mode", default="pixel", type=str)
    parser.add_argument("--re_count", default=1, type=int)
    parser.add_argument("--mixup_prob", default=1., type=float)
    parser.add_argument("--switch_prob", default=0.5, type=float)
    parser.add_argument("--mixup_mode", default="batch", type=str)

    # ===== Learning Rate Policy ======== #
    parser.add_argument("--optimizer", help="Which optimizer to use", default="adamw")
    parser.add_argument("--base_lr", default=0.0005, type=float)
    parser.add_argument("--warmup_lr", default=5e-7, type=float, help="warm up learning rate")
    parser.add_argument("--min_lr", default=0.000006, type=float)
    parser.add_argument("--lr_scheduler", default="cosine_lr", help="Schedule for the learning rate.")
    parser.add_argument("--warmup_length", default=0, type=int, help="Number of warmup iterations")
    parser.add_argument("--nonlinearity", default="GELU", type=str)

    # ===== Network training config ===== #
    parser.add_argument("--amp_level", default="02", choices=["O0", "O1", "O2", "O3"], help="AMP Level")
    parser.add_argument("--keep_bn_fp32", default="True", type=ast.literal_eval)
    parser.add_argument("--beta", default=[0.9, 0.999], type=lambda x: [float(a) for a in x.split(",")],
                        help="beta for optimizer")
    parser.add_argument("--clip_global_norm_value", default=5., type=float, help="Clip grad value")
    parser.add_argument("--is_dynamic_loss_scale", default=1, type=int, help="is_dynamic_loss_scale ")
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--label_smoothing", type=float, help="Label smoothing to use, default 0.0", default=0.1)
    parser.add_argument("--loss_scale", default=1024, type=int, help="loss_scale")
    parser.add_argument("--wd", "--weight_decay", default=0.05, type=float, metavar="W",
                        help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--batch-size", default=128, type=int)

    # ===== Hardware setup ===== #
    parser.add_argument("-j", "--num_parallel_workers", default=16, type=int, metavar="N",
                        help="number of data loading workers (default: 20)")
    parser.add_argument("--device_target", default="Ascend", choices=["GPU", "Ascend", "CPU"], type=str)

    # ===== Model config ===== #
    parser.add_argument("--drop_path_rate", default=0.2, type=float)
    parser.add_argument("--embed_dim", default=96, type=int)
    parser.add_argument("--depths", default=[2, 2, 6, 2])
    parser.add_argument("--num_heads", default=[3, 6, 12, 24])
    parser.add_argument("--window_size", default=7, type=int)
    parser.add_argument("--image_size", default=224, help="Image Size.", type=int)

    # ===== Other config ===== #
    parser.add_argument("--accumulation_step", default=1, type=int, help="accumulation step")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N",
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("--swin_config", help="Config file to use (see configs dir)", default=None, required=True)
    parser.add_argument("--pretrained", dest="pretrained", default=None, type=str, help="use pre-trained model")
    parser.add_argument("--ape", default=False, type=ast.literal_eval, help="absolute position embedding")
    parser.add_argument("--crop", default=True, type=ast.literal_eval, help="Crop when testing")
    parser.add_argument("--device_id", default=0, type=int, help="Device Id")
    parser.add_argument("--device_num", default=1, type=int, help="device num")
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--file_format", type=str, choices=["AIR", "MINDIR"], default="MINDIR", help="file format")
    parser.add_argument("--in_channel", default=3, type=int)
    parser.add_argument("--keep_checkpoint_max", default=20, type=int, help="keep checkpoint max num")
    parser.add_argument("--graph_mode", default=0, type=int, help="graph mode with 0, python with 1")
    parser.add_argument("--mlp_ratio", help="mlp ", default=4., type=float)
    parser.add_argument("--lr", "--learning_rate", default=5e-4, type=float, help="initial lr", dest="lr")
    parser.add_argument("--lr_adjust", default=30, type=float, help="Interval to drop lr")
    parser.add_argument("--lr_gamma", default=0.97, type=int, help="Multistep multiplier")
    parser.add_argument("--patch_size", type=int, default=4, help="patch_size")
    parser.add_argument("--patch_norm", type=ast.literal_eval, default=True, help="patch_norm")
    parser.add_argument("--seed", default=0, type=int, help="seed for initializing training. ")
    parser.add_argument("--save_every", default=2, type=int, help="Save every ___ epochs(default:2)")
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=True, help='Run distribute')
    parser.add_argument("--run_modelarts", type=ast.literal_eval, default=False, help="Whether run on modelarts")
    args = parser.parse_args()
    set_seed(1)

    ## copy dataset from obs to modelarts
    obs_data2modelarts(args)
    main()
