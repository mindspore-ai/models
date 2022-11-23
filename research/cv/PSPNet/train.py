# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
""" train PSPNet and get checkpoint files """
import os
import ast
import argparse
import src.utils.functions_args as fa
from src.model import pspnet
from src.model.cell import Aux_CELoss_Cell
from src.dataset import pt_dataset
from src.dataset import pt_transform as transform
from src.utils.lr import poly_lr
from src.utils.metric_and_evalcallback import pspnet_metric
import mindspore
from mindspore import nn
from mindspore import context
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.communication import init
from mindspore.context import ParallelMode
from mindspore.train.callback import Callback
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import mindspore.dataset as ds

set_seed(1234)
rank_id = int(os.getenv('RANK_ID', '0'))
device_id = int(os.getenv('DEVICE_ID', '0'))
device_num = int(os.getenv('RANK_SIZE', '1'))
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
Model_Art = False


def get_parser():
    """
    Read parameter file
        -> for ADE20k: ./config/ade20k_pspnet50.yaml
        -> for voc2012: ./config/voc2012_pspnet50.yaml
    """
    global Model_Art
    parser = argparse.ArgumentParser(description='MindSpore Semantic Segmentation')
    parser.add_argument('--config', type=str, required=True,
                        help='config file')
    parser.add_argument('--model_art', type=ast.literal_eval, default=False,
                        help='train on modelArts or not, default: True')
    parser.add_argument('--data_url', type=str, default='',
                        help='Location of data.')
    parser.add_argument('--train_url', type=str, default='',
                        help='Location of training outputs.')
    parser.add_argument('--dataset_name', type=str, default='',
                        help='aux parameter for ModelArt')
    parser.add_argument('opts', help='see ./config/voc2012_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args_ = parser.parse_args()
    if args_.model_art:
        import moxing as mox
        mox.file.shift('os', 'mox')
        Model_Art = True
        root = "/cache/"
        local_data_path = os.path.join(root, 'data')
        if args_.dataset_name == 'ade':
            obs_data_path = "obs://harbin-engineering-uni/PSPnet/data"
        else:
            obs_data_path = "obs://harbin-engineering-uni/PSPnet/voc"
        print("########### Downloading data from OBS #############")
        mox.file.copy_parallel(src_url=obs_data_path, dst_url=local_data_path)
        print('########### data downloading is completed ############')
    assert args_.config is not None
    cfg = fa.load_cfg_from_cfg_file(args_.config)
    if args_.opts is not None:
        cfg = fa.merge_cfg_from_list(cfg, args_.opts)
    return cfg


class EvalCallBack(Callback):
    """Precision verification using callback function."""

    def __init__(self, models, eval_dataset, eval_per_epochs, epochs_per_eval):
        super(EvalCallBack, self).__init__()
        self.models = models
        self.eval_dataset = eval_dataset
        self.eval_per_epochs = eval_per_epochs
        self.epochs_per_eval = epochs_per_eval

    def epoch_end(self, run_context):
        """ evaluate during training """
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epochs == 0:
            val_loss = self.models.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epochs_per_eval["epoch"].append(cur_epoch)
            self.epochs_per_eval["val_loss"].append(val_loss)
            print(val_loss)

    def get_dict(self):
        """ get eval dict"""
        return self.epochs_per_eval


def create_dataset(purpose, data_root, data_list, batch_size=8):
    """ get dataset """
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    if purpose == 'train':
        cur_transform = transform.Compose([
            transform.RandScale([0.5, 2.0]),
            transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([473, 473], crop_type='rand', padding=mean, ignore_label=255),
            transform.Normalize(mean=mean, std=std, is_train=True)])
        data = pt_dataset.SemData(
            split=purpose, data_root=data_root,
            data_list=data_list,
            transform=cur_transform,
            data_name=args.data_name
        )
        dataset = ds.GeneratorDataset(data, column_names=["data", "label"],
                                      shuffle=True, num_shards=device_num, shard_id=rank_id)
        dataset = dataset.batch(batch_size, drop_remainder=False)
    else:
        cur_transform = transform.Compose([
            transform.Crop([473, 473], crop_type='center', padding=mean, ignore_label=255),
            transform.Normalize(mean=mean, std=std, is_train=True)])
        data = pt_dataset.SemData(
            split=purpose, data_root=data_root,
            data_list=data_list,
            transform=cur_transform,
            data_name=args.data_name
        )

        dataset = ds.GeneratorDataset(data, column_names=["data", "label"],
                                      shuffle=False, num_shards=device_num, shard_id=rank_id)
        dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


def psp_train():
    """ Train process """
    if Model_Art:
        pre_path = args.art_pretrain_path
        data_path = args.art_data_root
        train_list_path = args.art_train_list
        val_list_path = args.art_val_list
    else:
        pre_path = args.pretrain_path
        data_path = args.data_root
        train_list_path = args.train_list
        val_list_path = args.val_list
    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          parameter_broadcast=True, gradients_mean=True)
        init()

        PSPNet = pspnet.PSPNet(
            feature_size=args.feature_size, num_classes=args.classes, backbone=args.backbone, pretrained=True,
            pretrained_path=pre_path, aux_branch=True, deep_base=True,
            BatchNorm_layer=nn.SyncBatchNorm
        )
        train_dataset = create_dataset('train', data_path, train_list_path)
        validation_dataset = create_dataset('val', data_path, val_list_path)
    else:
        PSPNet = pspnet.PSPNet(
            feature_size=args.feature_size, num_classes=args.classes, backbone=args.backbone, pretrained=True,
            pretrained_path=pre_path, aux_branch=True, deep_base=True
        )
        train_dataset = create_dataset('train', data_path, train_list_path)
        validation_dataset = create_dataset('val', data_path, val_list_path)

    # loss
    train_net_loss = Aux_CELoss_Cell(args.classes, ignore_label=255)

    steps_per_epoch = train_dataset.get_dataset_size()  # Return the number of batches in an epoch.
    total_train_steps = steps_per_epoch * args.epochs

    if device_num > 1:
        lr_iter = poly_lr(args.art_base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
        lr_iter_ten = poly_lr(args.art_base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    else:
        lr_iter = poly_lr(args.base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
        lr_iter_ten = poly_lr(args.base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)

    pretrain_params = list(filter(lambda x: 'backbone' in x.name, PSPNet.trainable_params()))
    cls_params = list(filter(lambda x: 'backbone' not in x.name, PSPNet.trainable_params()))
    group_params = [{'params': pretrain_params, 'lr': Tensor(lr_iter, mindspore.float32)},
                    {'params': cls_params, 'lr': Tensor(lr_iter_ten, mindspore.float32)}]
    opt = nn.SGD(
        params=group_params,
        momentum=0.9,
        weight_decay=0.0001,
        loss_scale=1024,
    )
    # loss scale
    manager_loss_scale = FixedLossScaleManager(1024, False)

    m_metric = {'val_loss': pspnet_metric(args.classes, 255)}

    model = Model(
        PSPNet, train_net_loss, optimizer=opt, loss_scale_manager=manager_loss_scale, metrics=m_metric
    )

    time_cb = TimeMonitor(data_size=steps_per_epoch)
    loss_cb = LossMonitor()
    epoch_per_eval = {"epoch": [], "val_loss": []}
    eval_cb = EvalCallBack(model, validation_dataset, 10, epoch_per_eval)
    config_ck = CheckpointConfig(
        save_checkpoint_steps=10 * steps_per_epoch,
        keep_checkpoint_max=12,
    )

    if Model_Art:
        os.path.join('/cache/', 'save')
        ckpoint_cb = ModelCheckpoint(
            prefix=args.prefix, directory='/cache/save/' + str(device_id), config=config_ck
        )
    else:
        ckpoint_cb = ModelCheckpoint(
            prefix=args.prefix, directory=args.save_dir, config=config_ck
        )
    model.train(
        args.epochs, train_dataset, callbacks=[loss_cb, time_cb, ckpoint_cb, eval_cb], dataset_sink_mode=True,
    )
    dict_eval = eval_cb.get_dict()
    val_num_list = dict_eval["epoch"]
    val_value = dict_eval["val_loss"]
    for i in range(len(val_num_list)):
        print(val_num_list[i], " : ", val_value[i])

    if Model_Art:
        print("######### upload to OBS #########")
        import moxing as mox
        mox.file.shift('os', 'mox')
        mox.file.copy_parallel(src_url="/cache/save", dst_url=args.obs_save)


if __name__ == "__main__":
    args = get_parser()
    print(args.obs_save)
    psp_train()
