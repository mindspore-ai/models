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

"""train scripts"""

import os
import time
import argparse
import mindspore as ms
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore import context
from mindspore.train.serialization import save_checkpoint
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from src.traindataset import create_traindataset
from src.model import get_model, TrainOnestepStyleTransfer
from src.loss import StyleTransferLoss


def get_args():
    """get args"""
    parser = argparse.ArgumentParser(description="style transfer train")
    # data loader
    parser.add_argument("--train_url", type=str, default='./dataset/train/content')
    parser.add_argument("--data_url", type=str, default='./dataset/train/content')

    parser.add_argument("--content_path", type=str, default='./dataset/train/content')
    parser.add_argument("--style_path", type=str, default='./dataset/train/style')
    parser.add_argument("--ckpt_path", type=str, default='./pretrained_model')
    parser.add_argument("--output_dir", type=str, default='./ckpt')

    parser.add_argument("--style_dim", type=int, default=100,
                        help="Style vector dimension. default: 100")
    parser.add_argument("--reshape_size", type=int, default=286,
                        help="Image size of high resolution image. default: 286")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Image size of high resolution image. default: 256")
    parser.add_argument("--batchsize", default=16, type=int, help="Batch size for training")

    # network training

    parser.add_argument('--learn_rate', type=float, default=1e-4, help='learning rate, default is 1e-4.')

    parser.add_argument('--style_weight', type=int, default=1e3, help='style_weight, default is 1e3.')

    parser.add_argument('--content_layers', type=str, default='vgg_16/conv3',
                        help='the layers of vgg16 for content loss')
    parser.add_argument('--content_each_weight', type=str, default='1',
                        help='the weights of each layer of vgg16 for content loss')
    parser.add_argument('--style_layers', type=str,
                        default='vgg_16/conv1,vgg_16/conv2,vgg_16/conv3,vgg_16/conv4',
                        help='the layers of vgg16 for style loss')
    parser.add_argument('--style_each_weight', type=str, default='0.5e-3,0.5e-3,0.5e-3,0.5e-3',
                        help='the weights of each layer of vgg16 for style loss')

    parser.add_argument("--epochs", default=100, type=int, help="Number of total epochs to run. (default: 100)")
    parser.add_argument('--init_type', type=str, default='normal', choices=("normal", "xavier"), \
                        help='network initialization, default is normal.')
    parser.add_argument('--init_gain', type=float, default=0.02, \
                        help='scaling factor for normal, xavier and orthogonal, default is 0.02.')

    # offline
    parser.add_argument('--run_offline', type=int, default=0, help='offline or not')
    # distribute
    parser.add_argument('--platform', type=str, default='Ascend', help='Ascend or GPU')
    parser.add_argument("--run_distribute", type=int, default=0, help="Run distribute, default: false.")
    parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
    parser.add_argument("--device_num", type=int, default=1, help="number of device, default: 0.")
    parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default: 0.")
    return parser.parse_args()


if __name__ == '__main__':
    local_sty_url = './cache/sty'
    local_con_url = './cache/con'
    local_train_url = './cache/out'
    local_ckpt_url = './cache/ckpt'
    ms.set_seed(42)
    args = get_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, save_graphs=False)

    # distribute
    if args.run_offline:
        local_sty_url = args.style_path
        local_con_url = args.content_path
        local_ckpt_url = args.ckpt_path
        local_train_url = args.output_dir

        if not os.path.exists(local_train_url):
            os.mkdir(local_train_url)
        print('run offline')
        if args.run_distribute:
            print("distribute")
            device_num = args.device_num
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
            if args.platform == 'GPU':
                init('nccl')
            elif args.platform == 'Ascend':
                init()

            rank = get_rank()
    else:
        print('ModelArt')
        import moxing as mox

        if not os.path.exists(local_train_url):
            os.makedirs(local_train_url)
        if args.run_distribute:
            print("distribute")
            device_num = args.device_num
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
            if args.platform == 'GPU':
                init('nccl')
            elif args.platform == 'Ascend':
                init()

            rank = get_rank()
            if not os.path.exists(os.path.join(local_train_url, 'parallel' + str(rank))):
                os.mkdir(os.path.join(local_train_url, 'parallel' + str(rank)))
        mox.file.copy_parallel(src_url=args.style_path, dst_url=local_sty_url)
        mox.file.copy_parallel(src_url=args.content_path, dst_url=local_con_url)
        mox.file.copy_parallel(src_url=args.ckpt_path, dst_url=local_ckpt_url)
    # create dataset
    train_ds = create_traindataset(args, local_con_url, local_sty_url)
    train_data_loader = train_ds.create_dict_iterator()
    # definition of network
    net = get_model(args)
    # network with loss
    if args.platform == 'GPU':
        loss = StyleTransferLoss(args, net, local_ckpt_url)
    elif args.platform == 'Ascend':
        loss = StyleTransferLoss(args, net, local_ckpt_url).to_float(mstype.float16)
    # optimizer
    optimizer = nn.Adam(net.trainable_params(), args.learn_rate)
    # trainonestep
    train_style_transfer = TrainOnestepStyleTransfer(loss, optimizer)
    train_style_transfer.set_train()
    print('start training style transfer:')
    # warm up generator
    for epoch in range(0, args.epochs):
        print("training {:d} epoch:".format(epoch + 1))
        mysince = time.time()
        style_loss = 0
        for i, data in enumerate(train_data_loader):
            content = data['content']
            style = data['style']
            step_loss = train_style_transfer(content, style)
            style_loss += step_loss
            # if i % 10 == 0:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
                  f"epoch {epoch}: {i}/{train_ds.get_dataset_size()}, loss={step_loss}", flush=True)

        steps = train_ds.get_dataset_size()
        time_elapsed = (time.time() - mysince)
        step_time = time_elapsed / steps
        print('per step needs time:{:.0f}ms'.format(step_time * 1000))
        print("per epoch style_loss:{:.8f}".format(style_loss.asnumpy() / steps))

        psnr_list = []
        if (epoch + 1) % 5 == 0:
            if args.run_distribute == 0:
                save_checkpoint(train_style_transfer, \
                                os.path.join(local_train_url, \
                                             'style_transfer_model_%04d.ckpt' % (epoch + 1)))
            else:
                print("===>args.rank_id:{}".format(rank))
                save_checkpoint(train_style_transfer, \
                                os.path.join(local_train_url, \
                                             'style_transfer_rank_%02d_model_%04d.ckpt' % (rank, epoch + 1)))
        print("{:d}/{:d} epoch finished".format(epoch + 1, args.epochs))
