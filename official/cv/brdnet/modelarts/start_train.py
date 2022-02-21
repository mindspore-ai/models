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
'''training script for modelarts'''
import os
import glob
import datetime
import argparse
import moxing as mox
import numpy as np
import PIL.Image as Image

import mindspore
import mindspore.nn as nn
from mindspore import context, export
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.context import ParallelMode
from mindspore.common.tensor import Tensor
from mindspore.train.callback import TimeMonitor, LossMonitor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_rank, get_group_size

from src.logger import get_logger
from src.dataset import create_BRDNetDataset
from src.models import BRDNet, BRDWithLossCell, TrainingWrapper


## Params
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_data', default='../dataset/waterloo5050step40colorimage/'
                    , type=str, help='path of train data')
parser.add_argument('--test_dir', default='./Test/Kodak24/'
                    , type=str, help='directory of test dataset')
parser.add_argument('--sigma', default=15, type=int, help='noise level')
parser.add_argument('--channel', default=3, type=int
                    , help='image channel, 3 for color, 1 for gray')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
parser.add_argument('--resume_path', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument('--resume_name', type=str, default=None,
                    help='resuming file name')
parser.add_argument("--image_height", type=int, default=500, help="Image height for exporting model.")
parser.add_argument("--image_width", type=int, default=500, help="Image width for exporting model.")
parser.add_argument('--train_url', type=str, default='train_url/'
                    , help='needed by modelarts, but we donot use it because the name is ambiguous')
parser.add_argument('--data_url', type=str, default='data_url/'
                    , help='needed by modelarts, but we donot use it because the name is ambiguous')
parser.add_argument('--output_path', type=str, default='./output/'
                    , help='output_path,when use_modelarts is set True, it will be cache/output/')
parser.add_argument('--outer_path', type=str, default='s3://output/'
                    , help='obs path,to store e.g ckpt files ')
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR"\
                    , help="file format")

parser.add_argument('--device_target', type=str, default='Ascend'
                    , help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
parser.add_argument('--is_save_on_master', type=int, default=1, help='save ckpt on master or all rank')
parser.add_argument('--ckpt_save_max', type=int, default=20
                    , help='Maximum number of checkpoint files can be saved. Default: 20.')

set_seed(1)
args = parser.parse_args()
save_dir = os.path.join(args.output_path, 'sigma_' + str(args.sigma) \
           + '_' + datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

def get_lr(steps_per_epoch, max_epoch, init_lr):
    lr_each_step = []
    while max_epoch > 0:
        tem = min(30, max_epoch)
        for _ in range(steps_per_epoch*tem):
            lr_each_step.append(init_lr)
        max_epoch -= tem
        init_lr /= 10
    return lr_each_step

device_id = int(os.getenv('DEVICE_ID', '0'))
context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                    device_target=args.device_target, save_graphs=False)

def copy_data_from_obs():
    args.logger.info("copying train data from obs to cache....")
    mox.file.copy_parallel(args.train_data, 'cache/dataset')
    args.logger.info("copying train data finished....")
    args.train_data = 'cache/dataset/'

    # resume checkpoint if needed
    if args.resume_path:
        args.logger.info("copying resume checkpoint from obs to cache....")
        mox.file.copy_parallel(args.resume_path, 'cache/resume_path')
        args.logger.info("copying resume checkpoint finished....")
        args.resume_path = 'cache/resume_path/'

    args.logger.info("copying test data from obs to cache....")
    mox.file.copy_parallel(args.test_dir, 'cache/test')
    args.logger.info("copying test data finished....")
    args.test_dir = 'cache/test/'

def copy_data_to_obs():
    args.logger.info("copying files from cache to obs....")
    mox.file.copy_parallel(save_dir, args.outer_path)
    args.logger.info("copying finished....")

def check_best_model():
    ckpt_list = glob.glob(os.path.join(save_dir, 'ckpt_' + str(args.rank) + '/*.ckpt'))
    model = BRDNet(args.channel)
    transpose = P.Transpose()
    expand_dims = P.ExpandDims()
    compare_psnr = nn.PSNR()
    compare_ssim = nn.SSIM()
    best_psnr = 0.
    args.best_ckpt = ""
    for ckpt in sorted(ckpt_list):
        args.logger.info("testing ckpt: " + str(ckpt))
        load_param_into_net(model, load_checkpoint(ckpt))
        psnr = []   #after denoise
        ssim = []   #after denoise
        file_list = glob.glob(os.path.join(args.test_dir, "*"))
        model.set_train(False)
        for file in file_list:
            # read image
            if args.channel == 3:
                img_clean = np.array(Image.open(file), dtype='float32') / 255.0
            else:
                img_clean = np.expand_dims(np.array(Image.open(file).convert('L'), \
                                           dtype='float32') / 255.0, axis=2)
            np.random.seed(0) #obtain the same random data when it is in the test phase
            img_test = img_clean + np.random.normal(0, args.sigma/255.0, img_clean.shape)
            img_clean = Tensor(img_clean, mindspore.float32) #HWC
            img_test = Tensor(img_test, mindspore.float32)   #HWC
            # predict
            img_clean = expand_dims(transpose(img_clean, (2, 0, 1)), 0)#NCHW
            img_test = expand_dims(transpose(img_test, (2, 0, 1)), 0)#NCHW
            y_predict = model(img_test)    #NCHW
            # calculate numeric metrics
            img_out = C.clip_by_value(y_predict, 0, 1)
            psnr_denoised = compare_psnr(img_clean, img_out)
            ssim_denoised = compare_ssim(img_clean, img_out)
            psnr.append(psnr_denoised.asnumpy()[0])
            ssim.append(ssim_denoised.asnumpy()[0])
        psnr_avg = sum(psnr)/len(psnr)
        ssim_avg = sum(ssim)/len(ssim)
        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            args.best_ckpt = ckpt
            args.logger.info("new best ckpt: " + str(ckpt) + ", psnr: " +\
                             str(psnr_avg) + ", ssim: " + str(ssim_avg))
def export_models():
    args.logger.info("exporting best model....")
    net = BRDNet(args.channel)
    load_param_into_net(net, load_checkpoint(args.best_ckpt))
    input_arr = Tensor(np.zeros([1, args.channel, \
                                args.image_height, args.image_width]), mindspore.float32)
    export(net, input_arr, file_name=os.path.join(save_dir, "best_ckpt"), \
           file_format=args.file_format)
    args.logger.info("export best model finished....")

def train():

    dataset, args.steps_per_epoch = create_BRDNetDataset(args.train_data, args.sigma, \
                        args.channel, args.batch_size, args.group_size, args.rank, shuffle=True)
    model = BRDNet(args.channel)

    # resume checkpoint if needed
    if args.resume_path:
        args.resume_path = os.path.join(args.resume_path, args.resume_name)
        args.logger.info('loading resume checkpoint {} into network'.format(args.resume_path))
        load_param_into_net(model, load_checkpoint(args.resume_path))
        args.logger.info('loaded resume checkpoint {} into network'.format(args.resume_path))

    model = BRDWithLossCell(model)
    model.set_train()

    lr_list = get_lr(args.steps_per_epoch, args.epoch, args.lr)
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=Tensor(lr_list, mindspore.float32))
    model = TrainingWrapper(model, optimizer)

    model = Model(model)

    # define callbacks
    if args.rank == 0:
        time_cb = TimeMonitor(data_size=args.steps_per_epoch)
        loss_cb = LossMonitor(per_print_times=10)
        callbacks = [time_cb, loss_cb]
    else:
        callbacks = []
    if args.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.steps_per_epoch*args.save_every,
                                       keep_checkpoint_max=args.ckpt_save_max)
        save_ckpt_path = os.path.join(save_dir, 'ckpt_' + str(args.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='channel_'+str(args.channel)+'_sigma_'+str(args.sigma)+'_rank_'+str(args.rank))
        callbacks.append(ckpt_cb)

    model.train(args.epoch, dataset, callbacks=callbacks, dataset_sink_mode=True)

    args.logger.info("training finished....")

if __name__ == '__main__':
    if args.is_distributed:
        assert args.device_target == "Ascend"
        init()
        context.set_context(device_id=device_id)
        args.rank = get_rank()
        args.group_size = get_group_size()
        device_num = args.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        if args.device_target == "Ascend":
            context.set_context(device_id=device_id)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1

    args.logger = get_logger(save_dir, "BRDNet", args.rank)
    args.logger.save_args(args)
    print('Starting training, Total Epochs: %d' % (args.epoch))
    copy_data_from_obs()
    train()
    if args.rank_save_ckpt_flag:
        check_best_model()
        export_models()
    copy_data_to_obs()
    args.logger.info('All task finished!')
