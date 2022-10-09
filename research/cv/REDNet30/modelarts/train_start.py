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
"""train on modelarts"""
import argparse
import os
import subprocess
import moxing as mox
import mindspore.nn as nn
from mindspore import context
from mindspore import dataset as ds
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from src.dataset import Dataset
from src.model import REDNet30

_CACHE_DATA_URL = "/cache/data_url"
_CACHE_TRAIN_URL = "/cache/train_url"

def _parse_args():
    """parse arguments"""
    parser = argparse.ArgumentParser(description='train and export wdsr on modelarts')
    # train output path
    parser.add_argument('--train_url', type=str, default='', help='where training log and ckpts saved')
    # dataset dir
    parser.add_argument('--data_url', type=str, default='', help='where datasets located')
    # train config
    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU'), help='run platform')
    parser.add_argument('--is_distributed', type=bool, default=False, help='distributed training')
    parser.add_argument('--patch_size', type=int, default=50, help='training patch size')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--ckpt_save_path', type=str, default='ckpt', help='path to save ckpt')
    parser.add_argument('--ckpt_save_max', type=int, default=5, help='maximum number of checkpoint files can be saved')
    parser.add_argument('--init_loss_scale', type=float, default=65536., help='initialize loss scale')
    # export config
    parser.add_argument("--export_batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--export_image_height", type=int, default=480, help="height of each input image")
    parser.add_argument("--export_image_width", type=int, default=480, help="width of each input image")
    parser.add_argument("--export_file_name", type=str, default="red30", help="output file name.")
    parser.add_argument("--export_file_format", type=str, default="AIR",
                        choices=['MINDIR', 'AIR', 'ONNX'], help="file format")
    args, _ = parser.parse_known_args()

    return args


def _train(args, data_url):
    """train"""
    device_id = int(os.getenv('DEVICE_ID', '0'))
    rank_id = int(os.getenv('RANK_ID', '0'))
    device_num = int(os.getenv('DEVICE_NUM', '1'))

    if args.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU")
    else:
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                            device_target="Ascend", device_id=device_id)
    # if distribute:
    if args.is_distributed:
        init()
        rank_id = get_rank()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=device_num, gradients_mean=True)
    # dataset
    print("============== Loading Data ==============")
    train_dataset = Dataset(data_url, args.patch_size)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ["input", "label"], num_shards=device_num,
                                           shard_id=rank_id, shuffle=True)
    train_de_dataset = train_de_dataset.batch(args.batch_size, drop_remainder=True)
    step_size = train_de_dataset.get_dataset_size()
    print("============== Loading Model ==============")
    model = REDNet30()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)
    loss = nn.MSELoss()
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=args.init_loss_scale, scale_window=1000)
    model = Model(model, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager, amp_level="O3")
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    config_ck = CheckpointConfig(keep_checkpoint_max=args.ckpt_save_max)
    ckpt_cb = ModelCheckpoint(prefix='RedNet30_{}'.format(rank_id),
                              directory=os.path.join(_CACHE_TRAIN_URL, args.ckpt_save_path),
                              config=config_ck)
    cb += [ckpt_cb]
    print("============== Starting Training ==============")
    model.train(args.num_epochs, train_de_dataset, callbacks=cb, dataset_sink_mode=True)
    print("================== Finished ==================")


def _get_last_ckpt(ckpt_dir):
    """get the last ckpt"""
    file_dict = {}
    lists = os.listdir(ckpt_dir)
    if not lists:
        print("No ckpt file found.")
        return None
    for i in lists:
        # 获取文件创建时间
        ctime = os.stat(os.path.join(ckpt_dir, i)).st_ctime
        file_dict[ctime] = i
    max_ctime = max(file_dict.keys())
    ckpt_file = os.path.join(ckpt_dir, file_dict[max_ctime])

    return ckpt_file


def _export_air(args, ckpt_dir):
    """export"""
    ckpt_file = _get_last_ckpt(ckpt_dir)
    if not ckpt_file:
        return
    # os.path.dirname: 去掉当前文件返回上一级目录的路径
    print("pwd before: ", os.path.realpath(__file__))
    pwd = os.path.dirname(os.path.realpath(__file__))
    print("pwd end: ", pwd)
    export_file = os.path.join(pwd, "export.py")
    cmd = ["python", export_file,
           f"--batch_size={args.export_batch_size}",
           f"--image_height={args.export_image_height}",
           f"--image_width={args.export_image_width}",
           f"--ckpt_path={ckpt_file}",
           f"--file_name={os.path.join(_CACHE_TRAIN_URL, args.export_file_name)}",
           f"--file_format={args.export_file_format}",]
    print(f"Start exporting, cmd = {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()


def main():
    args = _parse_args()
    os.makedirs(_CACHE_TRAIN_URL, exist_ok=True)
    os.makedirs(_CACHE_DATA_URL, exist_ok=True)
    # 拷贝数据集
    mox.file.copy_parallel(args.data_url, _CACHE_DATA_URL)
    data_url = _CACHE_DATA_URL
    _train(args, data_url)
    _export_air(args, os.path.join(_CACHE_TRAIN_URL, args.ckpt_save_path))
    #拷贝ckpt回到obs
    mox.file.copy_parallel(_CACHE_TRAIN_URL, args.train_url)


if __name__ == '__main__':
    main()
