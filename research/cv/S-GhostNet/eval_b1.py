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
"""Inference Interface"""
import sys
import os
import argparse
import zipfile
import time
import moxing as mox
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Loss, Top1CategoricalAccuracy, Top5CategoricalAccuracy
from mindspore import context

from src.dataset import create_dataset_val
from src.utils import count_params
from src.loss import LabelSmoothingCrossEntropy
from src.tinynet import tinynet
from src.ghostnet import ghostnet_1x
from src.big_net import GhostNet

os.environ["GLOG_v"] = '3'
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = '0'
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = '2'
os.environ["ASCEND_GLOBAL_EVENT_ENABLE"] = '0'

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--data_path', type=str, default='/autotest/liuchuanjian/data/imagenet/',
                    metavar='DIR', help='path to dataset')
parser.add_argument('--model', default='tinynet_c', type=str, metavar='MODEL',
                    help='Name of model to train (default: "tinynet_c") ghostnet, big_net')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--channels', type=str, default='24,32,64,112,160,280',
                    help='channel config of model architecure')
parser.add_argument('--layers', type=str, default='2,2,5,10,2,10',
                    help='layer config of model architecure')
parser.add_argument('--large', action='store_true', default=False,
                    help='ghostnet1x or ghostnet larger')
parser.add_argument('--input_size', type=int, default=248,
                    help='input size of model.')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('-b', '--batch-size', type=int, default=125, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--GPU', action='store_true', default=False,
                    help='Use GPU for training (default: False)')
parser.add_argument('--dataset_sink', action='store_false', default=True)
parser.add_argument('--drop', type=float, default=0.2, metavar='DROP',
                    help='Dropout rate (default: 0.) for big_net, use "1-drop", for others, use "drop"')
parser.add_argument('--drop-path', type=float, default=0.0, metavar='DROP',
                    help='Drop connect rate (default: 0.)')
parser.add_argument('--sync_bn', action='store_true', default=False,
                    help='Use sync bn in distributed mode. (default: False)')
parser.add_argument('--test_mode', default=None,
                    help='Use ema saved model to test, "ema_best", "ema_last", ')

# eval on cloud
parser.add_argument('--cloud', action='store_true', default=False, help='Whether train on cloud.')
parser.add_argument('--data_url', type=str, default="/home/ma-user/work/data/imagenet", help='path to dataset.')
parser.add_argument('--zip_url', type=str, default="s3://bucket-800/liuchuanjian/data/imagenet_zip/imagenet.zip")
parser.add_argument('--train_url', type=str, default=" ", help='train_dir.')
parser.add_argument('--tmp_data_dir', default='/cache/data/', help='temp data dir')
parser.add_argument('--trained_model_dir', default='s3://bucket-800/liuchuanjian/results/bignet/1291/',
                    help='temp save dir')
parser.add_argument('--tmp_save_dir', default='/cache/liuchuanjian/', help='temp save dir')

_global_sync_count = 0

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

def sync_data(opts):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    """
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    if not mox.file.exists(opts.tmp_data_dir):
        mox.file.make_dirs(opts.tmp_data_dir)
    target_file = os.path.join(opts.tmp_data_dir, 'imagenet.zip')
    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("from path: ", opts.zip_url)
        print("to path: ", target_file)
        mox.file.copy_parallel(opts.zip_url, target_file)
        print('Zip file copy success.')
        print('Starting unzip file.')
        unzip_file(target_file, opts.tmp_data_dir)
        print('Unzip file success.')
        print("===finish data synchronization===")
        ## ckpt copy
        print('Moving ckpt file')
        if not mox.file.exists(opts.tmp_save_dir):
            mox.file.make_dirs(opts.tmp_save_dir)
        for i in range(8):
            print('copying ckpt_ ', str(i))
            if opts.test_mode == 'ema_best':
                source_ckpt = os.path.join(opts.trained_model_dir, 'ckpt_'+str(i), 'ema_best.ckpt')
                target_ckpt = os.path.join(opts.tmp_save_dir, 'ckpt_'+str(i), 'ema_best.ckpt')
            elif opts.test_mode == 'ema_last':
                source_ckpt = os.path.join(opts.trained_model_dir, 'ckpt_'+str(i), 'ema_last.ckpt')
                target_ckpt = os.path.join(opts.tmp_save_dir, 'ckpt_'+str(i), 'ema_last.ckpt')
            else:
                source_ckpt = os.path.join(opts.trained_model_dir, 'ckpt_'+str(i), 'big_net-500_1251.ckpt')
                target_ckpt = os.path.join(opts.tmp_save_dir, 'ckpt_'+str(i), 'big_net-500_1251.ckpt')
            if mox.file.exists(source_ckpt):
                mox.file.copy(source_ckpt, target_ckpt)
            else:
                print(source_ckpt, 'does not exist.')

        try:
            os.mknod(sync_lock)
        except IOError:
            pass
        print("===save flag===")

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)

    opts.data_url = os.path.join(opts.tmp_data_dir, 'imagenet')
    opts.data_path = opts.data_url
    print("Finish sync data from {} to {}.".format(opts.zip_url, target_file))

def main(opts):
    """Main entrance for training"""
    print(sys.argv)
    if opts.channels:
        channel_config = []
        for item in opts.channels.split(','):
            channel_config.append(int(item.strip()))
    if opts.layers:
        layer_config = []
        for item in opts.layers.split(','):
            layer_config.append(int(item.strip()))
    print(opts)

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    devid = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=devid,
                        reserve_class_name_in_scope=True)
    context.set_auto_parallel_context(device_num=8)
    init()
    opts.rank = get_rank()
    opts.group_size = get_group_size()
    print('Rank {}, group_size {}'.format(opts.rank, opts.group_size))

    val_data_url = os.path.join(opts.data_path, 'val')
    val_dataset = create_dataset_val(opts.batch_size,
                                     val_data_url,
                                     workers=opts.workers,
                                     target='Ascend',
                                     distributed=False,
                                     input_size=opts.input_size)

    # parse model argument
    if opts.model == 'tinynet_c':
        _, sub_name = opts.model.split("_")
        net = tinynet(sub_model=sub_name,
                      num_classes=opts.num_classes,
                      drop_rate=0.0,
                      drop_connect_rate=0.0,
                      global_pool="avg",
                      bn_tf=False,
                      bn_momentum=None,
                      bn_eps=None)
    elif opts.model == 'ghostnet':
        net = ghostnet_1x(num_classes=opts.num_classes)
    else:
        net = GhostNet(layers=layer_config,
                       channels=channel_config,
                       num_classes=opts.num_classes,
                       final_drop=opts.drop,
                       drop_path_rate=opts.drop_path,
                       large=opts.large,
                       zero_init_residual=False,
                       sync_bn=opts.sync_bn)

    print("Total number of parameters:", count_params(net))
    if opts.model == 'tinynet_c':
        opts.input_size = net.default_cfg['input_size'][1]

    loss = LabelSmoothingCrossEntropy(smooth_factor=opts.smoothing,
                                      num_classes=opts.num_classes)

    loss.add_flags_recursive(fp32=True, fp16=False)
    eval_metrics = {'Validation-Loss': Loss(),
                    'Top1-Acc': Top1CategoricalAccuracy(),
                    'Top5-Acc': Top5CategoricalAccuracy()}
    for i in range(8):
        if opts.test_mode == 'ema_best':
            target_ckpt = os.path.join(opts.tmp_save_dir, 'ckpt_'+str(i), 'ema_best.ckpt')
        elif opts.test_mode == 'ema_last':
            target_ckpt = os.path.join(opts.tmp_save_dir, 'ckpt_'+str(i), 'ema_last.ckpt')
        else:
            target_ckpt = os.path.join(opts.tmp_save_dir, 'ckpt_'+str(i), 'big_net-500_1251.ckpt')
        if mox.file.exists(target_ckpt):
            print('Loading checkpoint: ', target_ckpt)
            ckpt = load_checkpoint(target_ckpt)
            load_param_into_net(net, ckpt)
            net.set_train(False)

            model = Model(net, loss, metrics=eval_metrics)

            metrics = model.eval(val_dataset, dataset_sink_mode=False)
            print('ckpt {}, Rank {}, Accuracy {}'.format(i, opts.rank, metrics))


if __name__ == '__main__':
    args, unparsed = parser.parse_known_opts()
    # copy data
    sync_data(args)
    main(args)
