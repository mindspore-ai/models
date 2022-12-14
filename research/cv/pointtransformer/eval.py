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

from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor
from mindspore import context, nn, ops, load_checkpoint, load_param_into_net

from src.dataset.ModelNet import create_modelnet40_dataset
from src.dataset.ShapeNet import create_shapenet_dataset
from src.model.pointTransfomrer import create_cls_mode, create_seg_mode
from src.config.default import get_config
from src.utils.common import context_device_init, CustomWithLossCell
from src.utils.local_adapter import get_device_id
from src.utils.metric import IoU, WithEvalCell

def test(cfg):
    if not cfg.device_id:
        cfg.device_id = get_device_id()
    context_device_init(cfg, context.GRAPH_MODE)

    print(f'Load dataset {cfg.dataset_type}...')
    if cfg.dataset_type == 'ModelNet40':
        eval_dataset = create_modelnet40_dataset('test', cfg)
        metrics = {"acc"}
    elif cfg.dataset_type == 'ShapeNet':
        eval_dataset = create_shapenet_dataset('test', cfg)
        metrics = {'IoU': IoU()}
    else:
        raise ValueError(f"Not a support data type {cfg.dataset_type}")

    print(f'Load model {cfg.model_type} ...')
    if cfg.model_type == 'classification':
        net = create_cls_mode()
    elif cfg.model_type == 'segmentation':
        net = create_seg_mode()
    else:
        raise ValueError(f"Not a support model type {cfg.model_type}")

    print(f"Load checkpoint {cfg.pretrain_ckpt}...")
    checkpoint = load_checkpoint(cfg.pretrain_ckpt)
    load_param_into_net(net, checkpoint, strict_load=True)
    net.set_train(False)

    if cfg.model_type == 'classification':
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        model = Model(net, loss, metrics=metrics)
    elif cfg.model_type == 'segmentation':
        net_with_criterion = CustomWithLossCell(net, ops.NLLLoss())
        eval_network = WithEvalCell(net, False)
        model = Model(net_with_criterion, eval_network=eval_network, metrics=metrics)

    time_cb = TimeMonitor(eval_dataset.get_dataset_size())
    result = model.eval(eval_dataset, dataset_sink_mode=False, callbacks=time_cb)
    print(result)
    if cfg.dataset_type == 'ModelNet40':
        Acc = result["acc"]
        print(f"Accuracy: {(Acc*100):.2f}%")
    elif cfg.dataset_type == 'ShapeNet':
        Ins_mIoU = result["IoU"][1]
        print(f"Instance mIoU: {(Ins_mIoU*100):.2f}%")
    else:
        raise ValueError(f"Not a support data type {cfg.dataset_type}")

if __name__ == '__main__':
    test(get_config())
