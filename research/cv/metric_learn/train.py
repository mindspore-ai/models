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
"""train resnet."""
import argparse
import ast
import os
import time

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import Callback
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.loss import Quadrupletloss
from src.loss import Softmaxloss
from src.loss import Tripletloss
from src.lr_generator import get_lr
from src.resnet import resnet50
from src.utility import GetDatasetGenerator_eval, recall_topk_parallel
from src.utility import GetDatasetGenerator_softmax, GetDatasetGenerator_triplet, GetDatasetGenerator_quadruplet

set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
# modelarts parameter
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--ckpt_url', type=str, default=None, help='Pretrained ckpt path')
parser.add_argument('--checkpoint_name', type=str, default='resnet-120_625.ckpt', help='Checkpoint file')
parser.add_argument('--loss_name', type=str, default='softmax',
                    help='loss name: softmax(pretrained) triplet quadruplet')
# Ascend parameter
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--ckpt_path', type=str, default=None, help='ckpt path name')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--device_id', type=int, default=0, help='Device id')
parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='Run distribute')

parser.add_argument("--device_target", type=str, default='Ascend', choices=['Ascend', 'GPU'],
                    help="Device target, support Ascend and GPU.")
parser.add_argument('--run_eval', type=ast.literal_eval, default=False, help='Run evaluation during training')
args_opt = parser.parse_args()


class EvalCallBack(Callback):
    """EvalCallBack"""

    def __init__(self, eval_function, eval_params):
        self.eval_function = eval_function
        self.eval_params = eval_params

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        recall = self.eval_function(self.eval_params)
        print("Current epoch:", cur_epoch, "eval_recall:", recall, flush=True)


def apply_eval(eval_params):
    """apply evaluation"""
    ckpt_path = eval_params["ckpt_cb"].latest_ckpt_file_name
    eval_ds = eval_params["dataset"]

    eval_net = resnet50(class_num=5184)
    ep_dict = load_checkpoint(ckpt_path)
    load_param_into_net(eval_net.backbone, ep_dict)
    eval_net.set_train(False)
    model_eval = Model(eval_net.backbone)

    f, l = [], []
    for data in eval_ds.create_dict_iterator():
        out = model_eval.predict(data['image'])
        f.append(out.asnumpy())
        l.append(data['label'].asnumpy())
    f = np.vstack(f)
    l = np.hstack(l)
    del eval_net
    del ep_dict
    del model_eval
    return recall_topk_parallel(f, l, k=1)


class Monitor(Callback):
    """Monitor"""

    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()
        dataset_generator.__init__(data_dir=DATA_DIR, train_list=TRAIN_LIST)

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:8.5f}"
              .format(epoch_mseconds, per_step_mseconds, np.mean(self.losses)), flush=True)
        print('batch_size:', config.batch_size, 'epochs_size:', config.epoch_size,
              'lr_model:', config.lr_decay_mode, 'lr:', config.lr_max, 'step_size:', step_size, flush=True)

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        """step_end"""
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs
        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())
        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num
        print("epochs:  [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:8.5f}/{:8.5f}], time:[{:5.3f}], lr:[{:8.5f}]".format(
            cb_params.cur_epoch_num, config.epoch_size, cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]), flush=True)


if __name__ == '__main__':
    print(args_opt, flush=True)

    if args_opt.device_target == 'GPU':
        if args_opt.loss_name == 'softmax':
            from src.config_gpu import config0 as config
            from src.dataset_gpu import create_dataset0 as create_dataset
        elif args_opt.loss_name == 'triplet':
            from src.config_gpu import config1 as config
            from src.dataset_gpu import create_dataset1 as create_dataset
        elif args_opt.loss_name == 'quadruplet':
            from src.config_gpu import config2 as config
            from src.dataset_gpu import create_dataset1 as create_dataset
        else:
            print('loss no')
    else:
        if args_opt.loss_name == 'softmax':
            from src.config import config0 as config
            from src.dataset import create_dataset0 as create_dataset
        elif args_opt.loss_name == 'triplet':
            from src.config import config1 as config
            from src.dataset import create_dataset1 as create_dataset
        elif args_opt.loss_name == 'quadruplet':
            from src.config import config2 as config
            from src.dataset import create_dataset1 as create_dataset
        else:
            print('loss no')

    print(config, flush=True)

    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args_opt.device_target,
        save_graphs=False
    )

    # init distributed
    if args_opt.run_modelarts:
        import moxing as mox

        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=device_id)
        local_data_url = '/cache/data'
        local_ckpt_url = '/cache/ckpt'
        local_train_url = '/cache/train'
        if device_num > 1:
            init()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            local_data_url = os.path.join(local_data_url, str(device_id))
            local_ckpt_url = os.path.join(local_ckpt_url, str(device_id))
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
        mox.file.copy_parallel(args_opt.ckpt_url, local_ckpt_url)
        DATA_DIR = local_data_url + '/'
    else:
        if args_opt.run_distribute:
            if args_opt.device_target == 'Ascend':
                device_id = int(os.getenv('DEVICE_ID'))
                device_num = int(os.getenv('RANK_SIZE'))
                context.set_context(device_id=device_id)
                init()
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num=device_num,
                                                  parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
            else:  # args_opt.device_target == 'GPU'
                init()
                device_id = get_rank()
                device_num = 1
                context.set_context(device_id=device_id)
        else:
            context.set_context(device_id=args_opt.device_id)
            device_num = 1
            device_id = args_opt.device_id
        DATA_DIR = args_opt.dataset_path + '/'

    # create dataset
    TRAIN_LIST = DATA_DIR + 'train.txt'
    if args_opt.loss_name == 'softmax':
        dataset_generator = GetDatasetGenerator_softmax(data_dir=DATA_DIR,
                                                        train_list=TRAIN_LIST)
    elif args_opt.loss_name == 'triplet':
        dataset_generator = GetDatasetGenerator_triplet(data_dir=DATA_DIR,
                                                        train_list=TRAIN_LIST)
    elif args_opt.loss_name == 'quadruplet':
        dataset_generator = GetDatasetGenerator_quadruplet(data_dir=DATA_DIR,
                                                           train_list=TRAIN_LIST)
    else:
        print('loss no')
    dataset = create_dataset(dataset_generator, do_train=True, batch_size=config.batch_size,
                             device_num=device_num, rank_id=device_id)
    step_size = dataset.get_dataset_size()

    # define net
    net = resnet50(class_num=config.class_num)

    # init weight
    if args_opt.run_modelarts:
        checkpoint_path = os.path.join(local_ckpt_url, args_opt.checkpoint_name)
    else:
        checkpoint_path = args_opt.ckpt_path
    if checkpoint_path:
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(net.backbone, param_dict)

    # init lr
    lr = Tensor(
        get_lr(
            lr_init=config.lr_init,
            lr_end=config.lr_end,
            lr_max=config.lr_max,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.epoch_size,
            steps_per_epoch=step_size,
            lr_decay_mode=config.lr_decay_mode
        )
    )

    # define opt
    opt = Momentum(
        params=net.trainable_params(),
        learning_rate=lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        loss_scale=config.loss_scale
    )

    # define loss, model
    if args_opt.loss_name == 'softmax':
        loss = Softmaxloss(sparse=True, smooth_factor=0.1, num_classes=config.class_num)
    elif args_opt.loss_name == 'triplet':
        loss = Tripletloss(margin=0.1)
    elif args_opt.loss_name == 'quadruplet':
        loss = Quadrupletloss(train_batch_size=config.batch_size, samples_each_class=2, margin=0.1)
    else:
        print('loss no')

    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    if args_opt.loss_name == 'softmax':
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=None,
                      amp_level='O3', keep_batchnorm_fp32=False)
    else:
        model = Model(net.backbone, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=None,
                      amp_level='O3', keep_batchnorm_fp32=False)

    # define callback
    cb = []
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)

        check_name = 'ResNet50_' + args_opt.loss_name
        if args_opt.run_modelarts:
            ckpt_cb = ModelCheckpoint(prefix=check_name, directory=local_train_url, config=config_ck)
        else:
            save_ckpt_path = os.path.join(config.save_checkpoint_path, 'model_' + str(device_id) + '/')
            ckpt_cb = ModelCheckpoint(prefix=check_name, directory=save_ckpt_path, config=config_ck)
        cb += [ckpt_cb]
    cb += [Monitor(lr_init=lr.asnumpy())]

    if args_opt.run_eval and config.save_checkpoint and (device_num == 1 or device_id == 0):
        VAL_LIST = DATA_DIR + "/test_half.txt"
        dataset_generator_val = GetDatasetGenerator_eval(DATA_DIR, VAL_LIST)
        eval_dataset = create_dataset(
            dataset_generator_val, do_train=False, batch_size=30, device_num=1, rank_id=device_id
        )
        eval_param_dict = {"dataset": eval_dataset, "ckpt_cb": ckpt_cb}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict)
        cb += [eval_cb]

    # train model
    print(f"Starting training on {args_opt.device_target} {device_id}")
    model.train(config.epoch_size - config.pretrain_epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)

    if args_opt.run_modelarts and config.save_checkpoint and (device_num == 1 or device_id == 0):
        mox.file.copy_parallel(src_url=local_train_url, dst_url=args_opt.train_url)
