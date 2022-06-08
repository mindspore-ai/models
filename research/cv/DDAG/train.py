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
"""train.py"""

import os
import os.path as osp
import time

import argparse
import psutil
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

from mindspore import context, load_checkpoint, load_param_into_net, save_checkpoint, DatasetHelper, Tensor
from mindspore.context import ParallelMode
from mindspore.communication import init, get_group_size, get_rank
from mindspore.dataset.transforms.transforms import Compose
from mindspore.nn import SGD, Adam


from src.dataset import SYSUDatasetGenerator, RegDBDatasetGenerator, TestData,\
    process_query_sysu, process_gallery_sysu, process_test_regdb
from src.evalfunc import test
from src.models.ddag import DDAG
from src.models.trainingcell import CriterionWithNet, OptimizerWithNetAndCriterion
from src.loss import OriTripletLoss, CenterTripletLoss
from src.utils import IdentitySampler, genidx, AverageMeter, get_param_list,\
    LRScheduler

from PIL import Image
from tqdm import tqdm


def show_memory_info(hint=""):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024./1024
    print(f"{hint} memory used: {memory} MB ")


def get_parser():
    """
    function of get parser
    """
    parser = argparse.ArgumentParser(description="DDAG Code Mindspore Version")
    parser.add_argument('--MSmode', default='GRAPH_MODE',
                        choices=['GRAPH_MODE', 'PYNATIVE_MODE'])

    # dataset settings
    parser.add_argument("--dataset", default='SYSU', choices=['SYSU', 'RegDB'],
                        help='dataset name: RegDB or SYSU')
    parser.add_argument('--data-path', type=str, default='')
    # Only used on Huawei Cloud OBS service,
    # when this is set, --data_path is overridden by --data-url
    parser.add_argument("--data-url", type=str, default=None)
    parser.add_argument('--batch-size', default=8, type=int,
                        metavar='B', help='the number of person IDs in a batch')
    parser.add_argument('--test-batch', default=64, type=int,
                        metavar='tb', help='testing batch size')
    parser.add_argument('--num-pos', default=4, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--trial', default=1, type=int,
                        metavar='t', help='trial (only for RegDB dataset)')

    # image transform
    parser.add_argument('--img-w', default=144, type=int,
                        metavar='imgw', help='img width')
    parser.add_argument('--img-h', default=288, type=int,
                        metavar='imgh', help='img height')

    # model
    parser.add_argument('--low-dim', default=512, type=int,
                        metavar='D', help='feature dimension')
    parser.add_argument('--part', default=0, type=int,
                        metavar='tb', help='part number, either add weighted part attention  module')
    parser.add_argument('--graph', default=False, type=bool,
                        help='either add graph attention or not')

    # loss setting
    parser.add_argument('--loss-func', default='id+tri', type=str, choices=['id', 'tri', 'id+tri'],
                        metavar='m', help='specify loss function type')
    parser.add_argument('--triloss', default='Ori',
                        type=str, choices=['Ori', 'Center'])
    parser.add_argument('--drop', default=0.2, type=float,
                        metavar='drop', help='dropout ratio')
    parser.add_argument('--margin', default=0.3, type=float,
                        metavar='margin', help='triplet loss margin')

    # optimizer and scheduler
    parser.add_argument("--lr", default=0.0035, type=float,
                        help='learning rate, 0.0035 for adam; 0.1 for sgd')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer')
    parser.add_argument("--warmup-steps", default=5,
                        type=int, help='warmup steps')
    parser.add_argument("--start-decay", default=15, type=int)
    parser.add_argument("--end-decay", default=27, type=int)

    # training configs
    parser.add_argument('--epoch', default=40, type=int,
                        metavar='epoch', help='epoch num')
    parser.add_argument('--start-epoch', default=1, type=int)
    parser.add_argument('--device-target', default="CPU",
                        choices=["CPU", "GPU", "Ascend", "Cloud"])
    parser.add_argument('--gpu', default='0', type=str,
                        help='set CUDA_VISIBLE_DEVICES')

    # Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU).
    #  If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be
    # the number set in the environment variable 'CUDA_VISIBLE_DEVICES'.
    #  For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the 'device_id' can be 0,1,2 at the moment,
    # 'device_id' starts from 0, and 'device_id'=0 means using GPU of number 4.
    parser.add_argument('--device-id', default=0, type=int, help='used in Ascend to speicfy device number')

    parser.add_argument('--resume', '-r', default='', type=str,
                        help='resume from checkpoint, no resume:""')
    parser.add_argument('--pretrain', type=str, default="",
                        help='Pretrain resnet-50 checkpoint path, no pretrain: ""')
    parser.add_argument('--run-distribute', action='store_true',
                        help="if set true, this code will be run on distributed architecture with mindspore")
    parser.add_argument('--parameter-server', default=False)
    parser.add_argument('--save-period', default=5, type=int,
                        help=" save checkpoint file every args.save_period epochs")

    # logging configs
    parser.add_argument("--branch-name", default="master",
                        help="Github branch name, for ablation study tagging")
    parser.add_argument('--tag', default='toy', type=str,
                        help='logfile suffix name')

    # testing / evaluation config
    parser.add_argument('--sysu-mode', default='all', type=str,
                        help='all or indoor', choices=['all', 'indoor'])
    parser.add_argument('--regdb-mode', default='v2i',
                        type=str, choices=['v2i', 'i2v'])

    return parser


def print_dataset_info(dataset_type_info, trainset_info, query_label_info, gall_label_info, start_time_info):
    """
    function of print data information
    """
    n_class_info = len(np.unique(trainset_info.train_color_label))
    nquery_info = len(query_label_info)
    ngall_info = len(gall_label_info)
    print('Dataset {} statistics:'.format(dataset_type_info))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(
        n_class_info, len(trainset_info.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(
        n_class_info, len(trainset_info.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(
        len(np.unique(query_label_info)), nquery_info))
    print('  gallery  | {:5d} | {:8d}'.format(
        len(np.unique(gall_label_info)), ngall_info))
    print('  ------------------------------')
    print('Data Loading Time:\t {:.3f}'.format(time.time() - start_time_info))

    print('Dataset {} statistics:'.format(dataset_type_info), file=log_file)
    print('  ------------------------------', file=log_file)
    print('  subset   | # ids | # images', file=log_file)
    print('  ------------------------------', file=log_file)
    print('  visible  | {:5d} | {:8d}'.format(
        n_class_info, len(trainset_info.train_color_label)), file=log_file)
    print('  thermal  | {:5d} | {:8d}'.format(
        n_class_info, len(trainset_info.train_thermal_label)), file=log_file)
    print('  ------------------------------', file=log_file)
    print('  query    | {:5d} | {:8d}'.format(
        len(np.unique(query_label_info)), nquery_info), file=log_file)
    print('  gallery  | {:5d} | {:8d}'.format(
        len(np.unique(gall_label_info)), ngall_info), file=log_file)
    print('  ------------------------------', file=log_file)
    print('Data Loading Time:\t {:.3f}'.format(
        time.time() - start_time_info), file=log_file)


def decode(img):
    return Image.fromarray(img)


def optim(args_, b_lr, h_lr):
    """ Define optimizers """

    if args_.optim == 'sgd':
        ignored_params = list(map(id, net.bottleneck.trainable_params())) \
            + list(map(id, net.classifier.trainable_params())) \
            + list(map(id, net.wpa.trainable_params())) \
            + list(map(id, net.graph_att.trainable_params()))

        base_params = list(
            filter(lambda p: id(p) not in ignored_params, net.net.trainable_params()))

        opt_p = SGD([
            {'params': base_params, 'lr': b_lr},
            {'params': net.bottleneck.trainable_params(), 'lr': h_lr},
            {'params': net.classifier.trainable_params(), 'lr': h_lr},
            {'params': net.wpa.trainable_params(), 'lr': h_lr},
            {'params': net.graph_att.trainable_params(), 'lr': h_lr}
        ],
                    learning_rate=args_.lr, weight_decay=5e-4, nesterov=True, momentum=0.9)

    elif args_.optim == 'adam':
        ignored_params = list(map(id, net.bottleneck.trainable_params())) \
            + list(map(id, net.classifier.trainable_params())) \
            + list(map(id, net.wpa.trainable_params())) \
            + list(map(id, net.graph_att.trainable_params()))

        base_params = list(
            filter(lambda p: id(p) not in ignored_params, net.trainable_params()))

        opt_p = Adam([
            {'params': base_params, 'lr': b_lr},
            {'params': net.bottleneck.trainable_params(), 'lr': h_lr},
            {'params': net.classifier.trainable_params(), 'lr': h_lr},
            {'params': net.wpa.trainable_params(), 'lr': h_lr},
            {'params': net.graph_att.trainable_params(), 'lr': h_lr}
        ],
                     learning_rate=args_.lr, weight_decay=5e-4)

    return opt_p


if __name__ == "__main__":
    parsers = get_parser()
    args = parsers.parse_args()

    if args.device_target == 'GPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ########################################################################
    # Init context
    ########################################################################
    device = args.device_target
    # init context
    if args.MSmode == "GRAPH_MODE":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=device, save_graphs=False)
    else:
        context.set_context(mode=context.PYNATIVE_MODE,
                            device_target=device, save_graphs=False)

    if device == "CPU":
        args.run_distribute = False
    else:
        if device in ["GPU", "Ascend"]:
            context.set_context(device_id=args.device_id)

        if args.parameter_server:
            context.set_ps_context(enable_ps=True)

        # distributed running context setting
        if args.run_distribute:
            # Ascend target
            if device == "Ascend":
                init()
                # assert args.device_num > 1
                context.set_auto_parallel_context(
                    device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                    gradients_mean=True
                )
                # mixed precision setting
                context.set_auto_parallel_context(
                    all_reduce_fusion_config=[85, 160])

            # GPU target
            else:
                init()
                context.set_auto_parallel_context(
                    device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                    gradients_mean=True
                )
                # mixed precision setting
                context.set_auto_parallel_context(
                    all_reduce_fusion_config=[85, 160])

        # Adapt to Huawei Cloud: download data from obs to local location
        if device == "Cloud":
            # Adapt to Cloud: used for downloading data from OBS to docker on the cloud
            import moxing as mox

            # Adapt to Cloud: used for downloading data from OBS to docker on the cloud
            import moxing as mox

            local_data_path = "/cache/data"
            args.data_path = local_data_path
            print("Download data...")
            mox.file.copy_parallel(src_url=args.data_url,
                                   dst_url=local_data_path)
            print("Download complete!(#^.^#)")

            local_pretrainmodel_path = "/cache/pretrain_model"
            pretrain_temp = args.pretrain
            args.pretrain = local_pretrainmodel_path + "/resnet50.ckpt"
            print("Download pretrain model..")
            mox.file.copy_parallel(src_url=pretrain_temp,
                                   dst_url=local_pretrainmodel_path)
            print("Download complete!(#^.^#)")
            # print(os.listdir(local_data_path))

    ########################################################################
    # Logging
    ########################################################################
    loader_batch = args.batch_size * args.num_pos

    if device in ['GPU', 'CPU', 'Ascend']:
        checkpoint_path = os.path.join("logs", args.tag, "training")
        os.makedirs(checkpoint_path, exist_ok=True)

        suffix = str(args.dataset)

        suffix = suffix + \
            '_batch-size_2*{}*{}={}'.format(args.batch_size,
                                            args.num_pos, 2 * loader_batch)
        suffix = suffix + '_{}_lr_{}'.format(args.optim, args.lr)
        suffix = suffix + '_loss-func_{}'.format(args.loss_func)

        if args.part > 0:
            suffix = suffix + '_P_{}'.format(args.part)

        if args.graph:
            suffix = suffix + '_Graph_'

        if args.dataset == 'RegDB':
            suffix = suffix + '_trial_{}'.format(args.trial)

        suffix = suffix + "_" + args.branch_name

        time_msg = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_file = open(osp.join(checkpoint_path,\
                "{}_performance_{}.txt".format(suffix, time_msg)), "w", encoding='utf-8')

        print('Args: {}'.format(args))
        print('Args: {}'.format(args), file=log_file)
        print()
        print(f"Log file is saved in {osp.join(os.getcwd(), checkpoint_path)}")
        print(
            f"Log file is saved in {osp.join(os.getcwd(), checkpoint_path)}", file=log_file)

    ########################################################################
    # Create Dataset
    ########################################################################
    dataset_type = args.dataset

    data_path = args.data_path

    best_acc = 0
    best_acc = 0  # best test accuracy
    start_epoch = 1
    feature_dim = args.low_dim
    start_time = time.time()

    print("==> Loading data")
    print("==> Loading data", file=log_file)
    # Data Loading code

    transform_train_rgb = Compose(
        [
            decode,
            vision.Pad(10),
            vision.RandomCrop((args.img_h, args.img_w)),
            vision.RandomGrayscale(prob=0.5),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False),
            vision.RandomErasing(prob=0.5)
        ]
    )

    transform_train_ir = Compose(
        [
            decode,
            vision.Pad(10),
            vision.RandomCrop((args.img_h, args.img_w)),
            vision.RandomGrayscale(prob=0.5),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False),
            vision.RandomErasing(prob=0.5)
        ]
    )

    transform_test = Compose(
        [
            decode,
            vision.Resize((args.img_h, args.img_w)),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)
        ]
    )

    if dataset_type == "SYSU":
        # train_set
        trainset_generator = SYSUDatasetGenerator(data_dir=data_path)

        color_pos, thermal_pos = genidx(
            trainset_generator.train_color_label, trainset_generator.train_thermal_label)

        # testing set
        query_img, query_label, query_cam = process_query_sysu(
            data_path, mode=args.sysu_mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(
            data_path, mode=args.sysu_mode, random_seed=0)

    elif dataset_type == "RegDB":
        # train_set
        trainset_generator = RegDBDatasetGenerator(
            data_dir=data_path, trial=args.trial)
        color_pos, thermal_pos = genidx(trainset_generator.train_color_label,
                                        trainset_generator.train_thermal_label)

        # testing set
        if args.regdb_mode == "v2i":
            query_img, query_label = process_test_regdb(img_dir=data_path,
                                                        modal="visible", trial=args.trial)
            gall_img, gall_label = process_test_regdb(img_dir=data_path,
                                                      modal="thermal", trial=args.trial)
        elif args.regdb_mode == "i2v":
            query_img, query_label = process_test_regdb(img_dir=data_path,
                                                        modal="thermal", trial=args.trial)
            gall_img, gall_label = process_test_regdb(img_dir=data_path,
                                                      modal="visible", trial=args.trial)

    ########################################################################
    # Create Query && Gallery
    ########################################################################

    gallset_generator = TestData(
        gall_img, gall_label, img_size=(args.img_w, args.img_h))
    queryset_generator = TestData(
        query_img, query_label, img_size=(args.img_w, args.img_h))

    print_dataset_info(dataset_type, trainset_generator,
                       query_label, gall_label, start_time)

    ########################################################################
    # Define net
    ########################################################################

    # pretrain
    if args.pretrain != 0:
        print("Pretrain model: {}".format(args.pretrain))
        print("Pretrain model: {}".format(args.pretrain), file=log_file)

    print('==> Building model..')
    print('==> Building model..', file=log_file)
    n_class = len(np.unique(trainset_generator.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)

    if args.graph:
        net = DDAG(args.low_dim, class_num=n_class, drop=args.drop,\
                   part=args.part, nheads=4, pretrain=args.pretrain)
    else:
        net = DDAG(args.low_dim, class_num=n_class, drop=args.drop,\
                   part=args.part, nheads=0, pretrain=args.pretrain)

    if args.resume != "":
        print("Resume checkpoint:{}". format(args.resume))
        print("Resume checkpoint:{}". format(args.resume), file=log_file)
        param_dict = load_checkpoint(args.resume)
        load_param_into_net(net, param_dict)
        if args.resume.split("/")[-1].split("_")[0] != "best":
            args.resume = int(args.resume.split("/")[-1].split("_")[1])
        print("Start epoch: {}".format(args.resume))
        print("Start epoch: {}".format(args.resume), file=log_file)

    ########################################################################
    # Define loss
    ########################################################################
    CELossNet = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    OriTriLossNet = OriTripletLoss(
        margin=args.margin, batch_size=2 * loader_batch)
    CenterTriLossNet = CenterTripletLoss(
        margin=args.margin, batch_size=2 * loader_batch)

    if args.triloss == "Ori":
        net_with_criterion = CriterionWithNet(
            net, CELossNet, OriTriLossNet, loss_func=args.loss_func)
    else:
        net_with_criterion = CriterionWithNet(
            net, CELossNet, CenterTriLossNet, loss_func=args.loss_func)

    ########################################################################
    # Define LR Schedulers and Optimizers
    ########################################################################

    N = np.maximum(len(trainset_generator.train_color_label),\
                    len(trainset_generator.train_thermal_label))
    total_batch = int(N / loader_batch) + 1

    backbone_lr_scheduler = LRScheduler(0.1 * args.lr, total_batch, args)
    head_lr_scheduler = LRScheduler(args.lr, total_batch, args)

    backbone_lr = backbone_lr_scheduler.getlr()
    head_lr = head_lr_scheduler.getlr()
    optimizer_P = optim(args, backbone_lr, head_lr)
    net_with_optim = OptimizerWithNetAndCriterion(net_with_criterion, optimizer_P)

    ########################################################################
    # Start Training
    ########################################################################

    time_msg = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('==>' + time_msg)
    print('==>' + time_msg, file=log_file)
    print('==> Start Training...')
    print('==> Start Training...', file=log_file)
    log_file.flush()

    best_map = 0.0
    best_r1 = 0.0
    best_epoch = 0
    best_param_list = None
    best_path = None

    for epoch in range(args.start_epoch, args.epoch + 1):
        print('==> Preparing Data Loader...')
        # identity sampler:
        sampler = IdentitySampler(trainset_generator.train_color_label,\
                                  trainset_generator.train_thermal_label,\
                                  color_pos, thermal_pos, args.num_pos, args.batch_size)

        trainset_generator.cindex = sampler.index1  # color index
        trainset_generator.tindex = sampler.index2  # thermal index

        # add sampler
        if args.run_distribute:
            rank_id = get_rank()
            rank_size = get_group_size()
            trainset = ds.GeneratorDataset(trainset_generator,\
                ["color", "thermal", "color_label", "thermal_label"],\
                sampler=sampler, num_parallel_workers=1, num_shards=rank_size, shard_id=rank_id)
        else:
            trainset = ds.GeneratorDataset(trainset_generator,\
                ["color", "thermal", "color_label", "thermal_label"],\
                sampler=sampler, num_parallel_workers=1)

        trainset = trainset.map(
            operations=transform_train_rgb, input_columns=["color"])
        trainset = trainset.map(
            operations=transform_train_ir, input_columns=["thermal"])

        trainset.cindex = sampler.index1  # color index
        trainset.tindex = sampler.index2  # infrared index
        print("Epoch [{}]".format(str(epoch)))
        print("Epoch [{}]".format(str(epoch)), file=log_file)

        trainset = trainset.batch(batch_size=loader_batch, drop_remainder=True)

        dataset_helper = DatasetHelper(trainset, dataset_sink_mode=False)

        batch_idx = 0

        print("The total number of batch is ->", total_batch)
        print("The total number of batch is ->", total_batch, file=log_file)

        # calculate average batch time
        batch_time = AverageMeter()
        end_time = time.time()

        # Calculate Avg loss
        loss_avg = AverageMeter()

        # calculate average accuracy
        acc = AverageMeter()
        net.set_train(mode=True)

        for batch_idx, (img1, img2, label1, label2) in enumerate(tqdm(dataset_helper)):
            # for batch_idx, (img1, img2, label1, label2) in enumerate(trainset):
            label1, label2 = ms.Tensor(label1, dtype=ms.float32), ms.Tensor(
                label2, dtype=ms.float32)
            img1, img2 = ms.Tensor(img1, dtype=ms.float32), ms.Tensor(
                img2, dtype=ms.float32)
            if args.graph:
                adjacency = net.create_graph(label1, label2)
                loss = net_with_optim(img1, img2, label1, label2, adjacency)
            else:
                loss = net_with_optim(img1, img2, label1, label2, None)

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            loss_avg.update(loss.asnumpy())
            net_with_criterion.wg = 1. / \
                (1. + Tensor(np.array(loss_avg.avg), ms.float32))

            if (batch_idx != 0) and (batch_idx % 100 == 0):
                print('Epoch: [{}][{}/{}]   '
                      'LR: {LR:.12f}   '
                      'Loss:{Loss:.4f}   '
                      'Batch Time:{batch_time:.2f}  '
                      .format(epoch, batch_idx, total_batch,
                              LR=float(head_lr[(epoch-1) * total_batch].asnumpy()),
                              Loss=float(loss_avg.avg),
                              batch_time=batch_time.avg,
                              ))
                print('Epoch: [{}][{}/{}]   '
                      'LR: {LR:.12f}   '
                      'Loss:{Loss:.4f}   '
                      'Batch Time:{batch_time:.3f}  '
                      .format(epoch, batch_idx, total_batch,
                              LR=float(head_lr[(epoch-1) * total_batch].asnumpy()),
                              Loss=float(loss.asnumpy()),
                              batch_time=batch_time.avg,
                              ), file=log_file)

        #############################################
        # Only for debug
        show_memory_info()
        #############################################

        if epoch > 0:

            net.set_train(mode=False)
            gallset = ds.GeneratorDataset(gallset_generator, ["img", "label"])
            gallset = gallset.map(
                operations=transform_test, input_columns=["img"])
            gallery_loader = gallset.batch(batch_size=args.test_batch)
            gallery_loader = DatasetHelper(
                gallery_loader, dataset_sink_mode=False)

            queryset = ds.GeneratorDataset(
                queryset_generator, ["img", "label"])
            queryset = queryset.map(
                operations=transform_test, input_columns=["img"])
            query_loader = queryset.batch(batch_size=args.test_batch)
            query_loader = DatasetHelper(query_loader, dataset_sink_mode=False)

            if args.dataset == "SYSU":
                cmc, mAP, cmc_att, mAP_att = test(args, gallery_loader, query_loader, ngall,
                                                  nquery, net, 1, gallery_cam=gall_cam, query_cam=query_cam)

            if args.dataset == "RegDB":
                if args.regdb_mode == "v2i":
                    cmc, mAP, cmc_att, mAP_att = test(args, gallery_loader, query_loader, ngall,
                                                      nquery, net, 2)
                elif args.regdb_mode == "i2v":
                    cmc, mAP, cmc_att, mAP_att = test(args, gallery_loader, query_loader, ngall,
                                                      nquery, net, 1)

            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP))
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=log_file)

            if args.part > 0:
                print('FC_att:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'
                      .format(cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att))
                print('FC_att:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'
                      .format(cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att), file=log_file)

            # Save checkpoint weights every args.save_period Epoch
            save_param_list = get_param_list(net)
            if (epoch >= 2) and (epoch % args.save_period) == 0:
                path = osp.join(checkpoint_path,
                                f"epoch_{epoch:02}_rank1_{cmc[0]*100:.2f}_mAP_{mAP*100:.2f}_{suffix}.ckpt")
                save_checkpoint(save_param_list, path)

            # Record the best performance
            if (mAP > best_map) or (mAP_att > best_map):
                best_map = max(mAP, best_map)

            if (cmc[0] > best_r1) or (cmc_att[0] > best_r1):
                best_param_list = save_param_list
                best_path = osp.join(checkpoint_path,
                                     f"best_epoch_{epoch:02}_rank1_{cmc[0]*100:.2f}_mAP_{mAP*100:.2f}_{suffix}.ckpt")
                best_r1 = max(cmc[0], cmc_att[0])
                best_epoch = epoch

            print(
                "******************************************************************************")
            print("******************************************************************************",
                  file=log_file)

            log_file.flush()

    print("=> Save best parameters...")
    print("=> Save best parameters...", file=log_file)
    save_checkpoint(best_param_list, best_path)

    print("=> Successfully saved")
    print("=> Successfully saved", file=log_file)
    if args.dataset == "SYSU":
        print(f"For SYSU-MM01 {args.sysu_mode} search, the testing result is:")
        print(
            f"For SYSU-MM01 {args.sysu_mode} search, the testing result is:", file=log_file)
    elif args.dataset == "RegDB":
        print(f"For RegDB {args.regdb_mode} search, the testing result is:")
        print(
            f"For RegDB {args.regdb_mode} search, the testing result is:", file=log_file)

    print(f"Best: rank-1: {best_r1:.2%}, mAP: {best_map:.2%}, \
        Best epoch: {best_epoch}(according to Rank-1)")
    print(f"Best: rank-1: {best_r1:.2%}, mAP: {best_map:.2%}, \
        Best epoch: {best_epoch}(according to Rank-1)", file=log_file)

    time_msg = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('==>' + time_msg)
    print('==>' + time_msg, file=log_file)
    print('==> End Training...')
    print('==> End Training...', file=log_file)
    log_file.flush()
    log_file.close()
