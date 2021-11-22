# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
This is the boot file for ModelArts platform.
Firstly, the train datasets are copied from obs to ModelArts.
Then, the string of train shell command is concated and using 'os.system()' to execute
"""
import os
import ast
import argparse
import glob
import datetime
import numpy as np
import moxing as mox
from mindspore import Tensor, export, context
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from mindspore.parallel import set_algo_parameters
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth

print(os.system('env'))

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
    mox.file.copy(src_url='SE-net.air', dst_url=FLAGS.train_url+'/SE-net.air')



def export_AIR(args_opt):
    """start modelarts export"""
    ckpt_list = glob.glob(args_opt.modelarts_result_dir + "/resnet*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    if args_opt.device_target == "Ascend":
        context.set_context(device_id=args_opt.device_id)

    net_ = resnet(args_opt.class_num)
    param_dict_ = load_checkpoint(ckpt_model)
    load_param_into_net(net_, param_dict_)

    input_arr = Tensor(np.zeros([1, 3, 224, 224], np.float32))
    export(net, input_arr, file_name='SE-net', file_format='AIR')


if __name__ == '__main__':
    ## Note: the code dir is not the same as work dir on ModelArts Platform!!!
    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument("--data_url", type=str, default="./dataset")
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset")
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")#modelarts train result: /cache/result

    parser.add_argument('--net', type=str, default=None, help='Resnet Model, either resnet50 or resnet101')
    parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset, either cifar10 or imagenet2012')

    parser.add_argument('--epoch_size', type=int, default=1, help='epoch_size')
    parser.add_argument('--pretrain_epoch_size', type=int, default=0, help='pretrain_epoch_size, use with pre_trained')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--class_num', type=int, default=1001, help='class_num')

    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
    parser.add_argument('--device_num', type=int, default=1, help='Device num.')
    parser.add_argument("--device_id", type=int, default=0, help="Device id")
    parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                        help="Device target, support Ascend, GPU and CPU.")
    parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
    parser.add_argument('--parameter_server', type=ast.literal_eval, default=False, help='Run parameter server train')
    args = parser.parse_args()
    set_seed(1)

    if args.net == "se-resnet50":
        from src.resnet import se_resnet50 as resnet
        from src.config import config2 as config
        from src.dataset import create_dataset2 as create_dataset


    ## copy dataset from obs to modelarts
    obs_data2modelarts(args)

    ## start train
    target = args.device_target
    ckpt_save_dir = args.modelarts_result_dir

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if args.parameter_server:
        context.set_ps_context(enable_ps=True)
    if args.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            init()
        elif target == "GPU":
            init('nccl')
            context.reset_auto_parallel_context()
            rank = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        if args.net == "se-resnet50":
            context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        else:
            context.set_auto_parallel_context(all_reduce_fusion_config=[180, 313])

    # create dataset
    dataset = create_dataset(dataset_path=args.modelarts_data_dir, do_train=True, repeat_num=1,
                             batch_size=args.batch_size, target=target, distribute=args.run_distribute)
    step_size = dataset.get_dataset_size()

    # define net
    net = resnet(class_num=args.class_num)
    if args.parameter_server:
        net.set_param_ps()

    # init weight
    if args.pre_trained:
        param_dict = load_checkpoint(args.modelarts_data_dir+'/'+args.pre_trained)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    # init lr
    if args.net == "se-resnet50":
        lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                    warmup_epochs=config.warmup_epochs, total_epochs=args.epoch_size, steps_per_epoch=step_size,
                    lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    # define loss, model
    if target in ["Ascend", "GPU"]:
        if args.dataset == "imagenet2012":
            if not config.use_label_smooth:
                config.label_smooth_factor = 0.0
            loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                      smooth_factor=config.label_smooth_factor, num_classes=args.class_num)
        loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                      amp_level="O2", keep_batchnorm_fp32=False)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(args.epoch_size - args.pretrain_epoch_size, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=True)
    ## start export air
    export_AIR(args)
    ## copy result from modelarts to obs
    modelarts_result2obs(args)
