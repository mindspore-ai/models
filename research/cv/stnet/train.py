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
"""train"""
import os
import argparse
import numpy as np
import mindspore
from mindspore import Model, context, Parameter
from mindspore.nn import Accuracy
from mindspore.parallel import set_algo_parameters
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import set_seed
from src.config import cfg_res50
from src.dataset import create_dataset
from src import Stnet_Res_model
from src.eval_callback import EvalCallBack
from src.model_utils.moxing_adapter import moxing_wrapper
from src.CrossEntropySmooth import CrossEntropySmooth

parser = argparse.ArgumentParser(description='video classification')
parser.add_argument('--device_id', type=int, default=0, help='Device id.')
parser.add_argument('--dataset_path', type=str, default=None, help='label path(not use in model art)')
parser.add_argument('--run_distribute', type=int, default=0, help='0 -- run standalone, 1 -- run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--resume', type=str, default='', help='pre_train_checkpoint file_url')
parser.add_argument('--data_url', type=str, help='model art data url')
parser.add_argument('--train_url', type=str, help='model art train_url ')
args_opt = parser.parse_args()

set_seed(1)


def change_weights(model, state):
    """
        The pretrained params are ResNet50 pretrained on ImageNet.
        However, conv1_weights' shape of StNet is not the same as that in ResNet50 because the input are super-image
        concatanated by a series of images. So it is recommendated to treat conv1_weights specifically.
        The process is as following:
          1, load params from pretrain
          2, get the value of conv1_weights in the state_dict and transform it
          3, set the transformed value to conv1_weights in prog
    """
    pretrained_dict = {}
    for name, _ in state.items():
        if "xception" in name:
            continue
        if "temp1" in name or "temp2" in name:
            continue
        if name.startswith("conv1"):
            state[name] = state[name].mean(axis=1, keep_dims=True) / model.N
            pretrained_dict[name] = np.repeat(state[name], model.N * 3, axis=1)
            pretrained_dict[name] = Parameter(pretrained_dict[name], requires_grad=True)
        else:
            pretrained_dict[name] = state[name]
            pretrained_dict[name] = Parameter(pretrained_dict[name], requires_grad=True)
    return pretrained_dict


def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds)
    return res[metrics_name]


def run_eval(model, ckpt_save_dir, cb):
    """run_eval"""
    config = cfg_res50
    if config['run_eval']:
        eval_dataset = create_dataset(data_dir=args_opt.dataset_path, config=config, shuffle=False, do_trains='val',
                                      num_worker=config['device_num'], list_path=config['local_val_list'])
        eval_param_dict = {"model": model, "dataset": eval_dataset, "metrics_name": "acc"}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config['eval_interval'],
                               eval_start_epoch=config['eval_start_epoch'], save_best_ckpt=config['save_best_ckpt'],
                               ckpt_directory=ckpt_save_dir, besk_ckpt_name="best_acc.ckpt",
                               metrics_name="acc")
        cb += [eval_cb]


def set_save_ckpt_dir():
    """set save ckpt dir"""
    config = cfg_res50
    ckpt_save_dir = config['checkpoint_path']

    if args_opt.run_distribute == 1:
        ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"

    return ckpt_save_dir


def set_parameter(config):
    """set_parameter"""
    # init context
    if args_opt.run_distribute == 1:
        device_num = int(os.getenv('RANK_SIZE'))
        rank_id = int(os.getenv('DEVICE_ID'))

        if config['mode'] == 'GRAPH':
            context.set_context(mode=context.GRAPH_MODE, device_target=config['target'], save_graphs=False,
                                enable_auto_mixed_precision=True,
                                device_id=rank_id)
        else:
            context.set_context(mode=context.PYNATIVE_MODE, device_target=config['target'], save_graphs=False,
                                device_id=rank_id)

        # context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          parameter_broadcast=True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        init()
    else:
        if config['mode'] == 'GRAPH':
            context.set_context(mode=context.GRAPH_MODE, device_target=config['target'], save_graphs=False,
                                device_id=args_opt.device_id)
        else:
            context.set_context(mode=context.PYNATIVE_MODE, device_target=config['target'], save_graphs=False,
                                device_id=args_opt.device_id)


@moxing_wrapper()
def run_train():
    """train function"""
    # load config
    config = cfg_res50
    set_parameter(config)
    dataset_path = args_opt.dataset_path    # label path

    # Load the data
    print('Loading the data...')

    video_datasets_train = create_dataset(data_dir=dataset_path, config=config, do_trains='train',
                                          num_worker=config['device_num'],
                                          list_path=config['local_train_list'])

    print('Starting to training...')
    step_size_train = video_datasets_train.get_dataset_size()
    print('The size of training set is {}'.format(step_size_train))

    # define net
    net = Stnet_Res_model.stnet50(input_channels=3, num_classes=config['class_num'], T=config['T'], N=config['N'])

    # load pretrain_resnet50
    if config['pre_res50'] or config['pre_res50_art_load_path']:
        path = config['pre_res50']
        if config['run_online']:
            path = config['pre_res50_art_load_path']
        if os.path.isfile(path):
            net_parmerters = load_checkpoint(path)
            net_parmerters = change_weights(net, net_parmerters)
            load_param_into_net(net, net_parmerters, strict_load=True)
        else:
            raise RuntimeError('no such file{}'.format(path))

    # load pretrain model
    if args_opt.resume or config['best_acc_art_load_path']:
        resume = os.path.join(args_opt.resume)
        if config['run_online']:
            resume = config['best_acc_art_load_path']
        if os.path.isfile(resume):
            net_parmerters = load_checkpoint(resume)
            load_param_into_net(net, net_parmerters, strict_load=True)
        else:
            raise RuntimeError('no such file{}'.format(resume))

    # define loss function
    loss = CrossEntropySmooth(sparse=True, reduction='mean', num_classes=config['class_num'])

    # lr
    lr = config['lr']
    optimizer_ft = mindspore.nn.Momentum(params=net.trainable_params(), learning_rate=lr,
                                         momentum=config['momentum'],
                                         weight_decay=config['weight_decay'])

    # define model
    model = Model(net, loss_fn=loss, optimizer=optimizer_ft, metrics={'acc': Accuracy()})

    # define callback
    callback = []
    time_cb = TimeMonitor(data_size=step_size_train)
    callback.append(time_cb)
    callback.append(LossMonitor(500))

    ckpt_save_dir = set_save_ckpt_dir()
    if config['save_checkpoint']:
        config_ck = CheckpointConfig(save_checkpoint_steps=config['save_checkpoint_epochs'] * step_size_train,
                                     keep_checkpoint_max=config['keep_checkpoint_max'])
        ckpt_cb = ModelCheckpoint(prefix="stnet", directory=ckpt_save_dir, config=config_ck)
        callback.append(ckpt_cb)

    # Init a SummaryCollector callback instance, and use it in model.train or model.eval
    from mindspore.train.callback import SummaryCollector
    if args_opt.run_distribute == 1:
        summary_dir = config['summary_dir'] + str(get_rank())
        summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=1)
        callback.append(summary_collector)

    run_eval(model, ckpt_save_dir, callback)
    # train
    model.train(config['num_epochs'], video_datasets_train,
                callbacks=callback)


if __name__ == '__main__':
    run_train()
