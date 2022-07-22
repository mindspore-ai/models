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
"""Training Interface"""
import sys
import os
import zipfile
import argparse
import hashlib
from pathlib import Path

from mindspore.communication.management import init, get_rank, get_group_size
from mindspore import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn import SGD, RMSProp, Momentum, Loss, Top1CategoricalAccuracy, \
    Top5CategoricalAccuracy
from mindspore import context, Tensor
from mindspore.common import set_seed

from src.dataset import create_dataset, create_dataset_val
from src.utils import add_weight_decay, str2bool, get_lr_tinynet_c, get_lr
from src.callback import LossMonitor
from src.loss import LabelSmoothingCrossEntropy
from src.eval_callback import EvalCallBack
from src.big_net import GhostNet

local_plog_path = os.path.join(Path.home(), 'ascend/log/')

os.environ["GLOG_v"] = '3'
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = '0'
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = '2'
os.environ["ASCEND_GLOBAL_EVENT_ENABLE"] = '0'

parser = argparse.ArgumentParser(description='Training')

# model architecture parameters
parser.add_argument('--data_path', type=str, default="", metavar="DIR",
                    help='path to dataset')
parser.add_argument('--model', default='tinynet_c', type=str, metavar='MODEL',
                    help='Name of model to train (default: "tinynet_c") ghostnet, big_net')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--channels', type=str, default='16,24,40,80,112,160',
                    help='channel config of model architecure')
parser.add_argument('--layers', type=str, default='1,2,2,4,2,5',
                    help='layer config of model architecure')
parser.add_argument('--large', action='store_true', default=False,
                    help='ghostnet1x or ghostnet larger')
parser.add_argument('--input_size', type=int, default=224,
                    help='input size of model.')

# preprocess parameters
parser.add_argument('--autoaugment', action='store_true', default=False,
                    help='whether use autoaugment for training images')

# training parameters
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('-b', '--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--drop', type=float, default=0.0, metavar='DROP',
                    help='Dropout rate (default: 0.) for big_net, use "1-drop", for others, use "drop"')
parser.add_argument('--drop-path', type=float, default=0.0, metavar='DROP',
                    help='Drop connect rate (default: 0.)')
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--lr_decay_style', type=str, default='cosine',
                    help='learning rate decay method(default: cosine), cosine_step')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr_end', type=float, default=1e-6,
                    help='The end of training learning rate (default: 1e-6)')
parser.add_argument('--lr_max', type=float, default=0.4,
                    help='the max of training learning rate (default: 0.4)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('--ema-decay', type=float, default=0.9999,
                    help='decay factor for model weights moving average \
                    (default: 0.999)')

# training information parameters
parser.add_argument('--amp_level', type=str, default='O0')
parser.add_argument('--per_print_times', type=int, default=1)

# batch norm parameters
parser.add_argument('--sync_bn', action='store_true', default=False,
                    help='Use sync bn in distributed mode. (default: False)')
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that \
                    support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')

# parallel parameters
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--distributed', action='store_true', default=False)
parser.add_argument('--dataset_sink', action='store_false', default=True)
parser.add_argument('--device_num', type=int, default=8, help='Device num.')

# checkpoint config
parser.add_argument('--save_checkpoint', action='store_false', default=True)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--ckpt_save_path', type=str, default='./')
parser.add_argument('--ckpt_save_epoch', type=int, default=5)
parser.add_argument('--loss_scale', type=int,
                    default=128, help='static loss scale')
parser.add_argument('--train', type=str2bool, default=1, help='train or eval')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
# train on cloud
parser.add_argument('--cloud', action='store_true', default=False, help='Whether train on cloud.')
parser.add_argument('--data_url', type=str, default="/home/ma-user/work/data/imagenet", help='path to dataset.')
parser.add_argument('--zip_url', type=str, default="s3://bucket-800/liuchuanjian/data/imagenet_zip/imagenet.zip")
parser.add_argument('--train_url', type=str, default=" ", help='train_dir.')
parser.add_argument('--tmp_data_dir', default='/cache/data/', help='temp data dir')
parser.add_argument('--tmp_save_dir', default='/cache/liuchuanjian/', help='temp save dir')
parser.add_argument('--save_dir', default='/autotest/liuchuanjian/result/big_model', help='temp save dir')

_global_sync_count = 0

def get_file_md5(file_name):
    m = hashlib.md5()
    with open(file_name, 'rb') as fobj:
        while True:
            data = fobj.read(4096)
            if not data:
                break
            m.update(data)

    return m.hexdigest()

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)

def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)

def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file_item in fz.namelist():
            fz.extract(file_item, dst_dir)
    else:
        raise Exception('This is not zip')

def sync_data(args):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    """
    import time
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    if not mox.file.exists(args.tmp_data_dir):
        mox.file.make_dirs(args.tmp_data_dir)
    target_file = os.path.join(args.tmp_data_dir, 'imagenet.zip')
    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("from path: ", args.zip_url)
        print("to path: ", target_file)
        mox.file.copy_parallel(args.zip_url, target_file)
        print('Zip file copy success.')
        os.system('ls /cache/data/')
        print('Computing MD5 of copied file.')
        file_md5 = get_file_md5(target_file)
        print('MD5 is: ', file_md5)
        re_upload_num = 20
        while file_md5 != '674b2e3a185c2c82c8d211a0465c386e' and re_upload_num >= 0:
            mox.file.copy_parallel(args.zip_url, target_file)
            print('Zip file copy success.')
            print('Computing MD5 of copied file.')
            file_md5 = get_file_md5(target_file)
            print('MD5 is: ', file_md5)
            re_upload_num -= 1
            print('reupload num is: ', re_upload_num)

        print('Starting unzip file.')
        unzip_file(target_file, args.tmp_data_dir)
        print('Unzip file success.')
        print("===finish data synchronization===")
        try:
            os.mknod(sync_lock)
        except IOError:
            pass
        print("===save flag===")

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)

    args.data_url = os.path.join(args.tmp_data_dir, 'imagenet')
    args.data_path = args.data_url
    print("Finish sync data from {} to {}.".format(args.zip_url, target_file))

def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds)
    return res[metrics_name]

def main(args):
    """Main entrance for training"""
    if args.channels:
        channel_config = []
        for item in args.channels.split(','):
            channel_config.append(int(item.strip()))
    if args.layers:
        layer_config = []
        for item in args.layers.split(','):
            layer_config.append(int(item.strip()))
    print(sys.argv)
    set_seed(1)
    target = args.device_target

    ckpt_save_dir = args.save_dir
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    device_num = get_device_num()
    if args.distributed:
        devid = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=devid,
                            reserve_class_name_in_scope=True)
        init()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, parameter_broadcast=True)
        print('device_num: ', context.get_auto_parallel_context("device_num"))
        args.rank = get_rank()
        args.group_size = get_group_size()
        print('Rank {}, group_size {}'.format(args.rank, args.group_size))

        # create save dir
        if args.cloud:
            if not mox.file.exists(os.path.join(args.tmp_save_dir, str(args.rank))):
                mox.file.make_dirs(os.path.join(args.tmp_save_dir, str(args.rank)))

            args.save_dir = os.path.join(args.tmp_save_dir, str(args.rank))
        else:
            # Check the save_dir exists or not
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
        print('Imagenet dir: ', args.data_path)

        ckpt_save_dir = os.path.join(args.save_dir, "ckpt_" + str(get_rank()))

    net = GhostNet(layers=layer_config, channels=channel_config, num_classes=args.num_classes,
                   final_drop=args.drop, drop_path_rate=args.drop_path,
                   large=args.large, zero_init_residual=False, sync_bn=args.sync_bn)
    print(net)

    time_cb = TimeMonitor(data_size=batches_per_epoch)

    if args.lr_decay_style == 'cosine_step':
        # original tinynet_c lr_decay method
        lr_array = get_lr_tinynet_c(base_lr=args.lr, total_epochs=args.epochs,
                                    steps_per_epoch=batches_per_epoch, decay_epochs=args.decay_epochs,
                                    decay_rate=args.decay_rate, warmup_epochs=args.warmup_epochs,
                                    warmup_lr_init=args.warmup_lr, global_epoch=0)
    elif args.lr_decay_style == 'cosine':
        # standard cosine lr_decay method, used in official mindspore ghostnet
        lr_array = get_lr(lr_init=args.lr, lr_end=args.lr_end,
                          lr_max=args.lr_max, warmup_epochs=args.warmup_epochs,
                          total_epochs=args.epochs, steps_per_epoch=batches_per_epoch)
    else:
        raise Exception('Unknown lr decay method!!!!!')
    lr = Tensor(lr_array)

    loss_cb = LossMonitor(lr_array, args.epochs, per_print_times=args.per_print_times,
                          start_epoch=0)

    if args.opt == 'sgd':
        param_group = add_weight_decay(net, weight_decay=args.weight_decay)
        optimizer = SGD(param_group, learning_rate=lr,
                        momentum=args.momentum, weight_decay=args.weight_decay,
                        loss_scale=args.loss_scale)

    elif args.opt == 'rmsprop':
        param_group = add_weight_decay(net, weight_decay=args.weight_decay)
        optimizer = RMSProp(param_group, learning_rate=lr,
                            decay=0.9, weight_decay=args.weight_decay,
                            momentum=args.momentum, epsilon=args.opt_eps,
                            loss_scale=args.loss_scale)
    elif args.opt == 'momentum':
        optimizer = Momentum(net.trainable_params(), learning_rate=lr,
                             momentum=args.momentum, loss_scale=args.loss_scale,
                             weight_decay=args.weight_decay)

    if args.smoothing == 0.0:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    else:
        loss = LabelSmoothingCrossEntropy(smooth_factor=args.smoothing, num_classes=args.num_classes)

    loss_scale_manager = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)

    loss.add_flags_recursive(fp32=True, fp16=False)
    eval_metrics = {'Validation-Loss': Loss(),
                    'Top1-Acc': Top1CategoricalAccuracy(),
                    'Top5-Acc': Top5CategoricalAccuracy()}

    if args.ckpt:
        ckpt = load_checkpoint(args.ckpt)
        load_param_into_net(net, ckpt)
        net.set_train(False)

    model = Model(net, loss, optimizer, metrics=eval_metrics,
                  loss_scale_manager=loss_scale_manager,
                  amp_level=args.amp_level)

    eval_param_dict = {"model": model, "dataset": val_dataset, "metrics_name": "Top1-Acc"}
    eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=1,
                           eval_start_epoch=0, save_best_ckpt=False,
                           ckpt_directory=ckpt_save_dir, besk_ckpt_name="best_acc.ckpt",
                           metrics_name="Top1-Acc")

    callbacks = [loss_cb, eval_cb, time_cb]
    if args.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=args.ckpt_save_epoch* batches_per_epoch,
                                     keep_checkpoint_max=5)
        ckpt_cb = ModelCheckpoint(prefix=args.model, directory=ckpt_save_dir, config=config_ck)
        callbacks += [ckpt_cb]

    print('dataset_sink_mode: ', args.dataset_sink)
    model.train(args.epochs, train_dataset, callbacks=callbacks,
                dataset_sink_mode=args.dataset_sink)


if __name__ == '__main__':
    opts, unparsed = parser.parse_known_args()
    # copy data
    if opts.cloud:
        import moxing as mox
        sync_data(opts)

    # input image size of the network
    train_dataset = val_dataset = None
    train_data_url = os.path.join(opts.data_path, 'train')
    val_data_url = os.path.join(opts.data_path, 'val')
    print('train data path: ', train_data_url)
    print('val data path: ', val_data_url)
    val_dataset = create_dataset_val(opts.batch_size,
                                     val_data_url,
                                     workers=opts.workers,
                                     target=opts.device_target,
                                     distributed=False,
                                     input_size=opts.input_size)

    if opts.train:
        train_dataset = create_dataset(opts.batch_size,
                                       train_data_url,
                                       workers=opts.workers,
                                       target=opts.device_target,
                                       distributed=opts.distributed,
                                       input_size=opts.input_size,
                                       autoaugment=opts.autoaugment)
        batches_per_epoch = train_dataset.get_dataset_size()
        print('Number of batches:', batches_per_epoch)
    main(opts)

    if opts.cloud:
        mox.file.copy_parallel(opts.save_dir, opts.train_url)
        if os.path.exists(local_plog_path):
            mox.file.copy_parallel(local_plog_path, opts.train_url)
        else:
            print('{} not exist....'.format(local_plog_path))
