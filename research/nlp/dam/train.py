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
# ===========================================================================
"""Train function"""
import os
import time
import numpy as np
import mindspore
from mindspore import nn
from mindspore import dataset as ds
from mindspore import context, Model
from mindspore import ParameterTuple
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from src.net import DAMNet, DAMNetWithLoss, DAMTrainOneStepCell, PredictWithNet
from src.callback import LossCallback, TimeMonitor, EvalCallBack
from src.metric import EvalMetric
from src import dynamic_lr as dl
from src import config as conf


device_num = int(os.getenv('RANK_SIZE'))
device_id = int(os.getenv('DEVICE_ID'))
rank_id = int(os.getenv('RANK_ID'))
print("RANK_SIZE: ", device_num)
print("DEVICE_ID: ", device_id)
print("RANK_ID: ", rank_id)


def prepare_seed(seed):
    """Set Random Seed"""
    print("Random Seed: ", seed)
    mindspore.set_seed(seed)


def write_args(_args, local_eval_file_name):
    """Write parameters to a file"""
    args_dict = _args.__dict__
    with open(local_eval_file_name, 'a+') as out_file:
        out_file.write("--------------- start ---------------\n")
        for eachArg, value in args_dict.items():
            out_file.write(eachArg + ' : ' + str(value) + '\n')
        out_file.write("---------------- end ----------------\n\n")


def mk_dir(path):
    """make dirs"""
    if not os.path.exists(path):
        os.makedirs(path)


def train(config):
    """Training"""
    prepare_seed(config.seed)
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)

    if config.modelArts:
        import moxing as mox
        mox.file.shift('os', 'mox')
        init()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          parameter_broadcast=True, gradients_mean=True)
        shard_id = rank_id
        num_shards = device_num
        root = "/cache/"
        obs_data_path = config.data_url
        if config.model_name == "DAM_ubuntu":
            local_data_path = os.path.join(root, "ubuntu_data")
            local_train_path = os.path.join(root, 'dam/save_checkpoints/ubuntu')
        elif config.model_name == "DAM_douban":
            local_data_path = os.path.join(root, "douban_data")
            local_train_path = os.path.join(root, 'dam/save_checkpoints/douban')
        else:
            raise RuntimeError('{} does not exist'.format(config.model_name))

        local_data_path = os.path.join(local_data_path, str(device_id))
        local_train_path = os.path.join(local_train_path, config.version)
        mox.file.make_dirs(local_train_path)
        print("############## Downloading data from OBS ##############")
        mox.file.copy_parallel(src_url=obs_data_path, dst_url=local_data_path)
    else:
        local_data_path = config.data_root
        local_train_path = config.output_path

        if config.parallel:
            init()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            shard_id = rank_id
            num_shards = device_num
            local_train_path = os.path.join(local_train_path, str(device_id))
        else:
            shard_id = None
            num_shards = None
        mk_dir(local_train_path)
    local_loss_file_name = os.path.join(local_train_path, config.loss_file_name)
    local_eval_file_name = os.path.join(local_train_path, config.eval_file_name)
    print("The path of loss.log:", local_loss_file_name)
    print("The path of eval.log:", local_eval_file_name)

    # loading training data
    train_data_path = os.path.join(local_data_path, config.train_data)
    print("\nStart loading train data: ", train_data_path)
    train_dataset = ds.MindDataset(train_data_path,
                                   columns_list=["turns", "turn_len", "response", "response_len", "label"],
                                   shuffle=True, num_shards=num_shards, shard_id=shard_id)
    train_dataset = train_dataset.batch(config.batch_size, drop_remainder=True)
    train_dataset = train_dataset.repeat(1)
    batch_num = train_dataset.get_dataset_size()
    print("dataset_size: ", batch_num)

    # model init
    if config.emb_init is not None:
        emb_init = os.path.join(local_data_path, config.emb_init)
    else:
        emb_init = None
    dam_net = DAMNet(config, emb_init=emb_init, is_emb_init=config.is_emb_init)

    iter_per_epoch = train_dataset.get_dataset_size()
    total_iters = iter_per_epoch * config.epoch_size
    lr = dl.exponential_decay_lr(learning_rate=config.learning_rate,
                                 decay_rate=config.decay_rate,
                                 decay_steps=config.decay_steps,
                                 max_iteration=total_iters,
                                 is_stair=True)
    lr = mindspore.Tensor(np.array(lr).astype(np.float32))
    train_net = DAMNetWithLoss(dam_net)
    optimizer = nn.Adam(params=ParameterTuple(train_net.trainable_params()), learning_rate=lr)
    train_net = DAMTrainOneStepCell(train_net, optimizer, sens=config.loss_scale)
    eval_net = PredictWithNet(dam_net)
    metric = EvalMetric(config.model_name)
    model = Model(train_net, eval_network=eval_net, metrics={"Accuracy": metric})

    # define callback
    time_cb = TimeMonitor(data_size=batch_num)
    loss_cb = LossCallback(loss_file_path=local_loss_file_name)
    cbs = [time_cb, loss_cb]

    # checkpoint save path
    save_step = int(max(1, batch_num / 10))
    config_ck = CheckpointConfig(save_checkpoint_steps=save_step, keep_checkpoint_max=80)
    save_checkpoint_path = os.path.join(local_train_path, str(device_id))
    ckpoint_cb = ModelCheckpoint(prefix="DAM", directory=save_checkpoint_path, config=config_ck)
    cbs.append(ckpoint_cb)

    if config.do_eval:
        write_args(config, local_eval_file_name)
        eval_data_path = os.path.join(local_data_path, config.eval_data)
        print('\nStart loading eval data: ', eval_data_path)
        eval_dataset = ds.MindDataset(eval_data_path,
                                      columns_list=["turns", "turn_len", "response", "response_len", "label"],
                                      shuffle=False, num_shards=None, shard_id=None)
        eval_dataset = eval_dataset.batch(config.eval_batch_size, drop_remainder=True)
        eval_dataset = eval_dataset.repeat(1)
        print("eval_dataset.size: ", eval_dataset.get_dataset_size())
        print("eval_per_steps: ", save_step)
        eval_callback = EvalCallBack(model, eval_dataset, eval_per_steps=save_step, eval_file_path=local_eval_file_name)
        cbs.append(eval_callback)

    print("############## Start training ##############")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    model.train(epoch=config.epoch_size, train_dataset=train_dataset, callbacks=cbs, dataset_sink_mode=False)

    if config.modelArts:
        obs_train_path = os.path.join(config.train_url, config.version)
        mox.file.copy_parallel(src_url=local_train_path, dst_url=obs_train_path)


if __name__ == '__main__':
    args = conf.parse_args()
    if args.model_name == "DAM_douban":
        args.vocab_size = 172130
        args.channel1_dim = 16
    print("args: ", args)
    train(args)
