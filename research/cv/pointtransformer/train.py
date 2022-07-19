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

from mindspore import context, nn, ops, load_checkpoint, load_param_into_net, DynamicLossScaleManager
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor


from src.dataset.ModelNet import create_modelnet40_dataset
from src.dataset.ShapeNet import create_shapenet_dataset
from src.model.pointTransfomrer import create_cls_mode, create_seg_mode
from src.utils.common import context_device_init, CustomWithLossCell
from src.config.default import get_config
from src.utils.callback import CallbackSaveByAcc, CheckLoss, CallbackSaveByIoU
from src.utils.local_adapter import get_device_id, moxing_wrapper, get_rank_id
from src.utils.lr_scheduler import MultiStepLR
from src.utils.metric import IoU, WithEvalCell


@moxing_wrapper()
def train(cfg):
    cfg.device_id = get_device_id()
    context_device_init(cfg, context.GRAPH_MODE)

    if get_rank_id() == 0:
        print(f'Load dataset {cfg.dataset_type}...')
    if cfg.dataset_type == 'ModelNet40':
        traindataset = create_modelnet40_dataset('train', cfg)
        eval_dataset = create_modelnet40_dataset('test', cfg)
        metrics = {"acc", "loss"}
    elif cfg.dataset_type == 'ShapeNet':
        traindataset = create_shapenet_dataset('train', cfg)
        eval_dataset = create_shapenet_dataset('test', cfg)
        metrics = {'IoU': IoU()}
    else:
        raise ValueError(f"Not a support data type {cfg.dataset_type}")

    step_size = traindataset.get_dataset_size()
    max_epoch = cfg.epoch_size

    if get_rank_id() == 0:
        print(f'Load model {cfg.model_type} ...')
    if cfg.model_type == 'classification':
        net = create_cls_mode()
        lr = MultiStepLR(cfg.learning_rate,
                         [60, 120, 160],
                         0.1,
                         step_size,
                         cfg.epoch_size).get_lr()
    elif cfg.model_type == 'segmentation':
        net = create_seg_mode()
        lr = MultiStepLR(cfg.learning_rate,
                         [120, 160],
                         0.1,
                         step_size,
                         cfg.epoch_size).get_lr()
    else:
        raise ValueError(f"Not a support model type {cfg.model_type}")

    if cfg.pretrain_ckpt:
        checkpoint = load_checkpoint(cfg.pretrain_ckpt)
        load_param_into_net(net, checkpoint)
        print(f'Use pretrain model {cfg.pretrain_ckpt}')
    else:
        print('No existing model, starting training from scratch...')

    opt = nn.SGD(params=net.trainable_params(),
                 learning_rate=lr,
                 momentum=0.9,
                 weight_decay=cfg.weight_decay)

    scale_factor = 4
    scale_window = 3000
    loss_scaler = DynamicLossScaleManager(scale_factor, scale_window)
    eval_proid = cfg.eval_proid
    eval_start = cfg.eval_start
    save_checkpoint_path = cfg.save_checkpoint_path
    if not os.path.exists(save_checkpoint_path):
        os.makedirs(save_checkpoint_path)

    loss_cb = CheckLoss()
    time_cb = TimeMonitor(step_size)

    if cfg.model_type == 'classification':
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        model = Model(net, loss, opt, amp_level="O2", metrics=metrics, loss_scale_manager=loss_scaler)
        ckpoint_cb = CallbackSaveByAcc(model, eval_dataset, eval_proid, eval_start, save_checkpoint_path)
    elif cfg.model_type == 'segmentation':
        net_with_criterion = CustomWithLossCell(net, ops.NLLLoss())
        eval_network = WithEvalCell(net, True)
        model = Model(net_with_criterion,
                      optimizer=opt,
                      amp_level="O0",
                      eval_network=eval_network,
                      metrics=metrics,
                      loss_scale_manager=loss_scaler)
        ckpoint_cb = CallbackSaveByIoU(model, eval_dataset, eval_proid, eval_start, save_checkpoint_path)

    if get_rank_id() == 0:
        print("============== Starting Training ==============")
    model.train(max_epoch, traindataset, callbacks=[loss_cb, time_cb, ckpoint_cb])


if __name__ == '__main__':
    train(get_config())
