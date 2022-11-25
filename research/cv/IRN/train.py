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
"""
Model training entrypoint.
"""

import os
import ast
import datetime
import argparse
import moxing as mox

import src.options.options as option
import src.utils.util as util
from src.data import create_dataset
from src.optim import warmup_step_lr, warmup_cosine_annealing_lr
from src.optim.adam_clip import AdamClipped
from src.network import create_model, IRN_loss

from mindspore import context, Tensor

from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_group_size
from mindspore.train.model import ParallelMode
from mindspore import Model, load_checkpoint, load_param_into_net
from mindspore.common import set_seed
set_seed(0)


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
    According to the switch FLAGS, the debug data may contains auto tune repository,
    dump data for precision comparison, even the computation graph and profiling data.
    """

    mox.file.copy_parallel(src_url=FLAGS.modelarts_result_dir, dst_url=FLAGS.train_url)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".
          format(FLAGS.modelarts_result_dir, FLAGS.train_url))


current_path = os.path.abspath(__file__)
root_path = os.path.dirname(current_path)
X2_TRAIN_YAML_FILE = os.path.join(root_path, "src", "options", "train", "train_IRN_x2.yml")
X4_TRAIN_YAML_FILE = os.path.join(root_path, "src", "options", "train", "train_IRN_x4.yml")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRN training args")
    parser.add_argument("--modelarts_FLAG", type=bool, default=True,
                        help="use modelarts or not")
    parser.add_argument('--data_url', type=str, default='./data/',
                        help="local obs data path, used for obs_data2modelarts")
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset/",
                        help="modelarts data path, used for obs_data2modelarts")
    parser.add_argument("--train_url", type=str, default="./output/checkpoint",
                        help="local obs output path, used for modelarts_result2obs")
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/train_output/",
                        help="modelarts output path, used for modelarts_result2obs")
    parser.add_argument("--training_dataset", type=str, default="/cache/dataset/DIV2K/DIV2K_train_HR",
                        help="modelarts dataset path, used for training")

    parser.add_argument('--scale', type=int, default=4, choices=(2, 4),
                        help='Rescaling Parameter.')
    parser.add_argument('--dataset_GT_path', type=str, default='/home/nonroot/DIV2K/DIV2K_train_HR',
                        help='Path to the folder where the intended GT dataset is stored.')
    parser.add_argument('--dataset_LQ_path', type=str, default=None,
                        help='Path to the folder where the intended LQ dataset is stored.')
    parser.add_argument('--resume_state', type=str, default=None,
                        help='Path to the checkpoint.')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=("GPU", "Ascend"),
                        help="Device target, support GPU, Ascend.")
    parser.add_argument('--device_num', type=int,
                        default=1, help='Device num.')
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default: false.")

    args = parser.parse_args()

    if args.modelarts_FLAG:
        obs_data2modelarts(FLAGS=args)
        args.dataset_GT_path = args.training_dataset

    if args.scale == 2:
        opt = option.parse(X2_TRAIN_YAML_FILE, args.dataset_GT_path,
                           args.dataset_LQ_path, is_train=True)
    elif args.scale == 4:
        opt = option.parse(X4_TRAIN_YAML_FILE, args.dataset_GT_path,
                           args.dataset_LQ_path, is_train=True)
    else:
        raise ValueError("Unsupported scale.")

    # initialize context
    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        save_graphs=False)

    # parallel environment setting
    rank = 0
    if args.run_distribute:
        opt['dist'] = True
        if args.device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id,
                                enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(device_num=args.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        elif args.device_target == "GPU":
            context.set_context(device_num=get_group_size(),
                                parallel_mode=ParallelMode.DATA_PARALLEL,
                                gradients_mean=True)
        else:
            raise ValueError("Unsupported device target.")
        init()
    else:
        if args.device_target == "Ascend":
            context.set_context(device_id=int(os.getenv('DEVICE_ID')))
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')

    context.set_context(max_call_depth=4030)

    # loading options for model
    opt = option.dict_to_nonedict(opt)
    train_opt = opt['train']

    # create dataset
    dataset_opt = opt['datasets']['train']
    total_epochs = int(opt['train']['epochs'])
    dataset_opt["data_type"] = "img"
    train_dataset = create_dataset(
        dataset_opt["dataroot_GT"],
        dataset_opt["scale"],
        batch_size=dataset_opt["batch_size"],
        distribute=args.run_distribute,
    )
    step_size = train_dataset.get_dataset_size()
    print("Total epoches:{} , Step size:{}".format(total_epochs, step_size))

    # learning rate
    wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
    if train_opt['lr_scheme'] == 'MultiStepLR':
        lr = warmup_step_lr(train_opt['lr_G']*2,
                            train_opt['lr_steps'],
                            step_size,
                            200,
                            total_epochs,
                            train_opt['lr_gamma'],
                            )
    elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
        lr = warmup_cosine_annealing_lr(train_opt['lr_G'],
                                        train_opt['lr_steps'],
                                        0,
                                        total_epochs,
                                        train_opt['restarts'],
                                        train_opt['eta_min'])
    else:
        raise NotImplementedError(
            'MultiStepLR learning rate scheme is enough.')
    print("Total learning rate:{}".format(lr.shape))

    # define net
    net = create_model(opt)

    # loading resume state if exists
    if args.resume_state is not None:
        param_dict = load_checkpoint(args.resume_state)
        load_param_into_net(net, param_dict)

    # define network with loss
    loss = IRN_loss(net, opt)

    # warp network with optimizer
    optimizer = AdamClipped(
        loss.netG.trainable_params(), learning_rate=Tensor(lr),
        beta1=train_opt['beta1'], beta2=train_opt['beta2'], weight_decay=wd_G)

    # Model
    model = Model(network=loss, optimizer=optimizer, amp_level="O3")

    # define callbacks
    ckpt_save_steps = step_size*100
    callbacks = [LossMonitor(), TimeMonitor(data_size=ckpt_save_steps)]
    config_ck = CheckpointConfig(save_checkpoint_steps=ckpt_save_steps, keep_checkpoint_max=50)
    save_ckpt_path = os.path.join('ckpt_one_step_x4/', util.get_timestamp() + '/')
    if args.modelarts_FLAG:
        save_ckpt_path = os.path.join(args.modelarts_result_dir, save_ckpt_path)
        if not os.path.exists(save_ckpt_path):
            os.makedirs(save_ckpt_path)
    ckpt_cb = ModelCheckpoint(prefix="irn_onestep", directory=save_ckpt_path, config=config_ck)
    callbacks.append(ckpt_cb)

    # training
    model.train(total_epochs, train_dataset, callbacks=callbacks, dataset_sink_mode=True)

    if args.modelarts_FLAG:
        modelarts_result2obs(FLAGS=args)
