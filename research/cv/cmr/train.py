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

from mindspore import nn, Model, load_checkpoint, load_param_into_net
from mindspore import context
from mindspore.train import callback
import mindspore.communication.management as D
from mindspore.communication.management import get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.common import set_seed

from src.utils.options import TrainOptions
from src.dataset.datasets import create_train_dataset
from src.loss.loss import CustomLoss
from src.netCell.netWithLoss import CustomWithLossCell
from src.utils.mesh import MeshCell
from src import config as cfg
from src.models.smpl import SMPL
from src.models.graph_cnn import GraphCNN
from src.models.smpl_param_regressor import SMPLParamRegressor


def train(options):
    """Training CMR"""
    set_seed(66)
    # Initialize training environment
    context.set_context(device_target=options.device_target, mode=context.PYNATIVE_MODE)

    cb = []

    save_ckpt_dir = options.save_checkpoint_dir
    device_id = 0
    device_num = 1

    if options.distribute:
        # Train in distributed mode
        D.init()
        device_id = get_rank()
        # Only save checkpoint when device id = 0
        if device_id == 0:
            save_ckpt_dir = save_ckpt_dir + '/ckpt_rank0/'
            ckpt_config = callback.CheckpointConfig(save_checkpoint_steps=options.checkpoint_steps,
                                                    keep_checkpoint_max=options.keep_checkpoint_max)
            ckpt_cb = callback.ModelCheckpoint(prefix='checkpoint_cmr',
                                               directory=save_ckpt_dir, config=ckpt_config)
            cb.append(ckpt_cb)
        # get device num
        device_num = get_group_size()

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)

    else:
        device_id = int(os.getenv('DEVICE_ID', str(0)))
        save_ckpt_dir = save_ckpt_dir + '/ckpt/'
        ckpt_config = callback.CheckpointConfig(save_checkpoint_steps=options.checkpoint_steps,
                                                keep_checkpoint_max=options.keep_checkpoint_max)
        ckpt_cb = callback.ModelCheckpoint(prefix='checkpoint_cmr',
                                           directory=save_ckpt_dir, config=ckpt_config)
        cb.append(ckpt_cb)
    # Create train dataset
    ds_train = create_train_dataset(options, device_num=device_num, device_id=device_id)

    # Load SMPL model
    smpl = SMPL()
    # Load non-trained Parameters of SMPL model
    smpl_params = load_checkpoint(cfg.SMPL_CKPT_FILE)
    load_param_into_net(smpl, smpl_params)

    # Load Mesh object
    mesh = MeshCell(smpl)
    mesh_params = load_checkpoint(cfg.MESH_CKPT_FILE)
    load_param_into_net(mesh, mesh_params)
    # Update params of mesh
    mesh.update_paramter()

    # Load GraphCNN
    graph_cnn = GraphCNN(mesh.adjmat, mesh.ref_vertices.transpose(),
                         num_channels=options.num_channels,
                         num_layers=options.num_layers)

    # Load SMPL Parameter Regressor
    smpl_param_regressor = SMPLParamRegressor()

    # loss function
    custom_loss = CustomLoss()

    net_with_loss = CustomWithLossCell(smpl, graph_cnn, mesh, smpl_param_regressor, custom_loss)
    net_with_loss.set_train(True)
    # Setup a joint optimizer for the 2 models
    optimizer = nn.Adam(params=net_with_loss.trainable_params(),
                        learning_rate=options.lr,
                        beta1=options.adam_beta1,
                        beta2=0.999,
                        weight_decay=options.wd)

    if options.pretrained_checkpoint != 'None':
        param_dict = load_checkpoint(options.pretrained_checkpoint)
        load_param_into_net(net_with_loss, param_dict)

    manager = nn.FixedLossScaleUpdateCell(loss_scale_value=1024)
    train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer, scale_sense=manager)
    train_net.set_train(True)

    model = Model(train_net)
    cb += [callback.LossMonitor(), callback.TimeMonitor()]
    model.train(train_dataset=ds_train, epoch=options.num_epochs, callbacks=cb, dataset_sink_mode=False)

if __name__ == '__main__':
    options_ = TrainOptions().parse_args()
    train(options_)
