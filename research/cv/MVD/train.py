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
import argparse
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

import mindspore as ms
import mindspore.ops as P
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

from mindspore import context, load_checkpoint, \
    load_param_into_net, save_checkpoint, DatasetHelper
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_group_size
from mindspore.dataset.transforms.transforms import Compose
from mindspore.nn import SGD, Adam
from mindspore import nn

from src.dataset import SYSUDatasetGenerator, RegDBDatasetGenerator, \
    TestData, process_gallery_sysu, process_query_sysu, process_test_regdb
from src.evalfunc import test
from src.models.mvd import MVD
from src.models.trainingcell import CriterionWithNet, OptimizerWithNetAndCriterion
from src.loss import OriTripletLoss, CenterTripletLoss
from src.utils import IdentitySampler, genidx, AverageMeter, \
    get_param_list, LRScheduler


def get_parser():
    '''
    return a parser
    '''
    psr = argparse.ArgumentParser(description="DDAG Code Mindspore Version")
    psr.add_argument("--MSmode", default="GRAPH_MODE", choices=["GRAPH_MODE", "PYNATIVE_MODE"])

    # dataset settings
    psr.add_argument("--dataset", default='SYSU', choices=['SYSU', 'RegDB'],
                     help='dataset name: RegDB or SYSU')
    psr.add_argument('--data_path', type=str, default='data')
    # Only used on Huawei Cloud OBS service,
    # when this is set, --data_path is overridden by --data-url
    psr.add_argument("--data_url", type=str, default=None)
    psr.add_argument('--batch_size', default=8, type=int,
                     metavar='B', help='the number of person IDs in a batch')
    psr.add_argument('--test_batch', default=64, type=int,
                     metavar='tb', help='testing batch size')
    psr.add_argument('--num_pos', default=4, type=int,
                     help='num of pos per identity in each modality')
    psr.add_argument('--trial', default=1, type=int,
                     metavar='t', help='trial (only for RegDB dataset)')

    # image transform
    psr.add_argument('--img_w', default=144, type=int,
                     metavar='imgw', help='img width')
    psr.add_argument('--img_h', default=288, type=int,
                     metavar='imgh', help='img height')

    # model
    psr.add_argument('--arch', default='resnet50', type=str,
                     help='network baseline:resnet50')
    psr.add_argument('--z_dim', default=512, type=int,
                     help='information bottleneck z dim')

    # loss setting
    psr.add_argument('--loss_func', default="id+tri", type=str,
                     help='specify loss function type', choices=["id", "id+tri"])
    psr.add_argument('--triloss', default=["OriTri", "CenterTri"])
    psr.add_argument('--drop', default=0.2, type=float,
                     metavar='drop', help='dropout ratio')
    psr.add_argument('--margin', default=0.3, type=float,
                     metavar='margin', help='triplet loss margin')

    # optimizer and scheduler
    psr.add_argument("--lr", default=0.00035, type=float,
                     help='learning rate, 0.0035 for adam; 0.1 for sgd')
    psr.add_argument('--optim', default='adam', type=str, choices=['adam', 'sgd'],
                     help='optimizer')
    psr.add_argument("--warmup_steps", default=5, type=int,
                     help='warmup steps')
    psr.add_argument("--start_decay", default=15, type=int,
                     help='weight decay start epoch(included)')
    psr.add_argument("--end_decay", default=27, type=int,
                     help='weight decay end epoch(included)')

    # training configs
    psr.add_argument('--epoch', default=80, type=int,
                     metavar='epoch', help='epoch num')
    psr.add_argument('--start_epoch', default=1, type=int,
                     help='start training epoch')
    psr.add_argument('--device_target', default="GPU",
                     choices=["CPU", "GPU", "Ascend", "Cloud"])
    psr.add_argument('--is_modelarts', default="False",
                     choices=["True", "False"])
    psr.add_argument('--gpu', default='0', type=str,
                     help='set CUDA_VISIBLE_DEVICES')
    psr.add_argument('--print_per_step', default=100, type=int)

    # Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU).
    #  If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be
    # the number set in the environment variable 'CUDA_VISIBLE_DEVICES'.
    #  For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the 'device_id' can be 0,1,2 at the
    # moment, 'device_id' starts from 0, and 'device_id'=0 means using GPU of number 4.
    psr.add_argument('--device_id', default=0, type=int, help='')

    psr.add_argument('--device_num', default=1, type=int,
                     help='the total number of available gpus')
    psr.add_argument('--resume', '-r', default='', type=str,
                     help='resume from checkpoint, no resume:""')
    psr.add_argument('--pretrain', type=str,
                     default="",
                     help='Pretrain resnet-50 checkpoint path, no pretrain: ""')
    psr.add_argument('--run_distribute', action='store_true',
                     help="if true, will be run on distributed architecture with mindspore")
    psr.add_argument('--save_period', default=10, type=int,
                     help=" save checkpoint file every args.save_period epochs")

    # testing / evaluation config
    psr.add_argument('--sysu_mode', default='all', type=str, choices=["all", "indoor"],
                     help=' test all or indoor search(only for SYSU-MM01)')
    psr.add_argument('--regdb_mode', default='v2i', type=str, choices=["v2i", "i2v"],
                     help='v2i: visible to infrared search; i2v:infrared to visible search.(Only for RegDB)')

    return psr


def print_dataset_info(dtype, trainset_, query_label_, gall_label_, start_time_):
    """
    This method print dataset information.
    """
    n_class_ = len(np.unique(trainset_.train_color_label))
    nquery_ = len(query_label_)
    ngall_ = len(gall_label_)
    print(f'Dataset {dtype} statistics:')
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print(f'  visible  | {n_class_:5d} | {len(trainset_.train_color_label):8d}')
    print(f'  thermal  | {n_class_:5d} | {len(trainset_.train_thermal_label):8d}')
    print('  ------------------------------')
    print(f'  query    | {len(np.unique(query_label_)):5d} | {nquery_:8d}')
    print(f'  gallery  | {len(np.unique(gall_label_)):5d} | {ngall_:8d}')
    print('  ------------------------------')
    print(f'Data Loading Time:\t {time.time() - start_time_:.3f}')


def decode(img):
    '''
    params:
        img: img of Tensor
    Returns:
        PIL image
    '''
    return Image.fromarray(img)


def optim(args_, b_lr, h_lr):
    '''
    return an optimizer of SGD or ADAM
    '''
    ########################################################################
    # Define optimizers
    ########################################################################

    if args_.optim == 'sgd':

        opt_p = SGD([
            {'params': net.rgb_backbone.trainable_params(), 'lr': b_lr},
            {'params': net.ir_backbone.trainable_params(), 'lr': b_lr},
            {'params': net.shared_backbone.trainable_params(), 'lr': b_lr},
            {'params': net.rgb_bottleneck.trainable_params(), 'lr': h_lr},
            {'params': net.ir_bottleneck.trainable_params(), 'lr': h_lr},
            {'params': net.shared_bottleneck.trainable_params(), 'lr': h_lr},
        ], learning_rate=args_.lr, weight_decay=5e-4, nesterov=True, momentum=0.9)

    elif args_.optim == 'adam':

        opt_p = Adam([
            {'params': net.rgb_backbone.trainable_params(), 'lr': b_lr},
            {'params': net.ir_backbone.trainable_params(), 'lr': b_lr},
            {'params': net.shared_backbone.trainable_params(), 'lr': b_lr},
            {'params': net.rgb_bottleneck.trainable_params(), 'lr': h_lr},
            {'params': net.ir_bottleneck.trainable_params(), 'lr': h_lr},
            {'params': net.shared_bottleneck.trainable_params(), 'lr': h_lr},
        ], learning_rate=args_.lr, weight_decay=5e-4)

    return opt_p


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.device_target == 'GPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ########################################################################
    # Init context
    ########################################################################
    device = args.device_target
    # init context
    if args.MSmode == "GRAPH_MODE":
        exec_mode = context.GRAPH_MODE
    else:
        exec_mode = context.PYNATIVE_MODE
    context.set_context(mode=exec_mode, device_target=device, save_graphs=False, max_call_depth=3000)

    if device == "CPU":
        LOCAL_DATA_PATH = args.data_path
        args.run_distribute = False
    else:
        if device in ["GPU", "Ascend"]:
            LOCAL_DATA_PATH = args.data_path
            context.set_context(device_id=args.device_id)

        # distributed running context setting
        if args.run_distribute:
            init()
            # assert args.device_num > 1
            context.set_auto_parallel_context(
                device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True
            )
            # mixed precision setting
            context.set_auto_parallel_context(
                all_reduce_fusion_config=[85, 160])

        # Adapt to Huawei Cloud: download data from obs to local location
        if args.is_modelarts == "True":
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

    ########################################################################
    # Logging
    ########################################################################
    loader_batch = args.batch_size * args.num_pos

    if device in ("GPU", "CPU", "Ascend"):
        log_file = open(args.dataset + "_train_performance.txt", "w", encoding="utf-8")
        print(f'Args: {args}')
        print(f'Args: {args}', file=log_file)

    ########################################################################
    # Create Dataset
    ########################################################################
    dataset_type = args.dataset

    if dataset_type in ("RegDB", "SYSU"):
        data_path = args.data_path

    START_EPOCH = args.start_epoch
    start_time = time.time()

    print("==> Loading data")
    # Data Loading code

    transform_train_rgb = Compose(
        [
            decode,
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
            vision.RandomCrop((args.img_h, args.img_w)),
            # vision.RandomGrayscale(prob=0.5),
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
        color_pos, thermal_pos = genidx(trainset_generator.train_color_label, \
                                        trainset_generator.train_thermal_label)

        # testing set
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.sysu_mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, \
                                                              mode=args.sysu_mode, random_seed=0)

    elif dataset_type == "RegDB":
        # train_set
        trainset_generator = RegDBDatasetGenerator(data_dir=data_path, trial=args.trial)
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

    gallset_generator = TestData(gall_img, gall_label, img_size=(args.img_w, args.img_h))
    queryset_generator = TestData(query_img, query_label, img_size=(args.img_w, args.img_h))

    print_dataset_info(dataset_type, trainset_generator, query_label, gall_label, start_time)

    ########################################################################
    # Define net
    ########################################################################

    # pretrain
    if args.pretrain != "":
        print(f"Pretrain model: {args.pretrain}")
        print(f"Pretrain model: {args.pretrain}", file=log_file)

    print('==> Building model..')
    n_class = len(np.unique(trainset_generator.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)

    net = MVD(num_class=n_class, drop=args.drop, z_dim=args.z_dim,
              pretrain=args.pretrain)

    if args.resume != "":
        print(f"Resume checkpoint:{args.resume}")
        print(f"Resume checkpoint:{args.resume}", file=log_file)
        param_dict = load_checkpoint(args.resume)
        load_param_into_net(net, param_dict)
        if args.resume.split("/")[-1].split("_")[0] != "best":
            args.resume = int(args.resume.split("/")[-1].split("_")[1])
        print(f"Start epoch: {args.resume}")
        print(f"Start epoch: {args.resume}", file=log_file)

    ########################################################################
    # Define loss
    ########################################################################
    CELossNet = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    OriTripLossNet = OriTripletLoss(margin=args.margin, batch_size=2 * loader_batch)
    CenterTripLossNet = CenterTripletLoss(margin=args.margin, batch_size=2 * loader_batch)

    # TripLossNet = TripletLoss(margin=args.margin)

    if args.triloss == "CenterTri":
        net_with_criterion = CriterionWithNet(net, CELossNet, CenterTripLossNet, P.KLDivLoss(),
                                              loss_func=args.loss_func)
    else:
        net_with_criterion = CriterionWithNet(net, CELossNet, OriTripLossNet, P.KLDivLoss(),
                                              loss_func=args.loss_func)

    ########################################################################
    # Define schedulers
    ########################################################################

    assert (args.start_decay > args.warmup_steps) and (args.start_decay < args.end_decay) \
           and (args.end_decay < args.epoch)

    N = np.maximum(len(trainset_generator.train_color_label),
                   len(trainset_generator.train_thermal_label))
    total_batch = int(N / loader_batch) + 1
    print(total_batch)
    backbone_lr_scheduler = LRScheduler(0.1 * args.lr, total_batch, args)
    head_lr_scheduler = LRScheduler(args.lr, total_batch, args)

    backbone_lr = backbone_lr_scheduler.getlr()
    head_lr = head_lr_scheduler.getlr()

    optimizer_P = optim(args, backbone_lr, head_lr)
    net_with_optim = OptimizerWithNetAndCriterion(net_with_criterion, optimizer_P)

    ########################################################################
    # Start Training
    ########################################################################

    print('==> Start Training...')
    BEST_MAP = 0.0
    BEST_R1 = 0.0
    BEST_EPOCH = 0
    best_param_list = None
    best_path = None
    for epoch in range(START_EPOCH, args.epoch + 1):

        print('==> Preparing Data Loader...')
        # identity sampler:
        sampler = IdentitySampler(trainset_generator.train_color_label,
                                  trainset_generator.train_thermal_label,
                                  color_pos, thermal_pos, args.num_pos, args.batch_size)

        trainset_generator.cindex = sampler.index1  # color index
        trainset_generator.tindex = sampler.index2  # thermal index

        # add sampler
        trainset = ds.GeneratorDataset(trainset_generator,
                                       ["color", "thermal", "color_label", "thermal_label"], sampler=sampler)

        trainset = trainset.map(operations=transform_train_rgb, input_columns=["color"])
        trainset = trainset.map(operations=transform_train_ir, input_columns=["thermal"])

        trainset.cIndex = sampler.index1  # color index
        trainset.tIndex = sampler.index2  # infrared index
        print(f"Epoch [{str(epoch)}]")

        trainset = trainset.batch(batch_size=loader_batch, drop_remainder=True)

        dataset_helper = DatasetHelper(trainset, dataset_sink_mode=False)
        # net_with_optim = connect_network_with_dataset(net_with_optim, dataset_helper)

        net.set_train(mode=True)

        BATCH_IDX = 0

        print("The total number of batch is ->", total_batch)

        # calculate average batch time
        batch_time = AverageMeter()
        end_time = time.time()

        # calculate average accuracy
        acc = AverageMeter()

        ########################################################################
        # Batch Training
        ########################################################################
        time_msg = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        print('==>' + time_msg)
        print('==>' + time_msg, file=log_file)
        print('==> Start Training...')
        print('==> Start Training...', file=log_file)
        log_file.flush()

        for BATCH_IDX, (img1, img2, label1, label2) in enumerate(tqdm(dataset_helper)):
            label1 = ms.Tensor(label1, dtype=ms.float32)
            label2 = ms.Tensor(label2, dtype=ms.float32)
            img1, img2 = ms.Tensor(img1, dtype=ms.float32), ms.Tensor(img2, dtype=ms.float32)

            loss = net_with_optim(img1, img2, label1, label2)

            acc.update(net_with_criterion.acc)
            id_loss = net_with_criterion.loss_id
            tri_loss = net_with_criterion.loss_tri

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            if (BATCH_IDX > 0) and (BATCH_IDX % args.print_per_step) == 0:
                print('Epoch: [{}][{}/{}]   '
                      'Convolution LR: {CLR:.7f}   '
                      'IB & Classifier LR: {HLR:.7f}    '
                      'Loss:{Loss:.4f}   '
                      'id:{id:.4f}   '
                      'tri:{tri:.4f}   '
                      'Batch Time:{batch_time:.2f}  '
                      'Accuracy:{acc:.2f}   '
                      .format(epoch, BATCH_IDX, total_batch,
                              CLR=float(head_lr[(epoch - 1) * total_batch].asnumpy()),
                              HLR=float(backbone_lr[(epoch - 1) * total_batch].asnumpy()),
                              Loss=float(loss.asnumpy()),
                              id=float(id_loss.asnumpy()),
                              tri=float(tri_loss.asnumpy()),
                              batch_time=batch_time.avg,
                              acc=float(acc.avg.asnumpy() * 100)
                              ))
                print('Epoch: [{}][{}/{}]   '
                      'Convolution LR: {CLR:.7f}   '
                      'IB & Classifier LR: {HLR:.7f}    '
                      'Loss:{Loss:.4f}   '
                      'id:{id:.4f}   '
                      'tri:{tri:.4f}   '
                      'Batch Time:{batch_time:.2f}  '
                      'Accuracy:{acc:.2f}   '
                      .format(epoch, BATCH_IDX, total_batch,
                              CLR=float(head_lr[(epoch - 1) * total_batch].asnumpy()),
                              HLR=float(backbone_lr[(epoch - 1) * total_batch].asnumpy()),
                              Loss=float(loss.asnumpy()),
                              id=float(id_loss.asnumpy()),
                              tri=float(tri_loss.asnumpy()),
                              batch_time=batch_time.avg,
                              acc=float(acc.avg.asnumpy() * 100)
                              ), file=log_file)
                log_file.flush()

        ########################################################################
        # Epoch Evaluation
        ########################################################################

        if epoch > 0:

            net.set_train(mode=False)
            gallset = ds.GeneratorDataset(gallset_generator, ["img", "label"], shuffle=True)
            gallset = gallset.map(operations=transform_test, input_columns=["img"])
            gallery_loader = gallset.batch(batch_size=args.test_batch)

            queryset = ds.GeneratorDataset(queryset_generator, ["img", "label"], shuffle=True)
            queryset = queryset.map(operations=transform_test, input_columns=["img"])
            query_loader = queryset.batch(batch_size=args.test_batch)

            if args.dataset == "SYSU":
                cmc_ob, map_ob = test(args, gallery_loader, query_loader, ngall,
                                      nquery, net, gallery_cam=gall_cam, query_cam=query_cam)

            if args.dataset == "RegDB":
                cmc_ob, map_ob = test(args, gallery_loader, query_loader, ngall,
                                      nquery, net)

            print('Original Observation:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| '
                  'Rank-20: {:.2%}| mAP: {:.2%}'.format(cmc_ob[0], cmc_ob[4], cmc_ob[9], cmc_ob[19], map_ob))

            print('Original Observation:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| '
                  'Rank-20: {:.2%}| mAP: {:.2%}'.format(cmc_ob[0], cmc_ob[4], cmc_ob[9], cmc_ob[19], map_ob),
                  file=log_file)

            map_ = map_ob
            cmc = cmc_ob

            print(f"rank-1: {cmc[0]:.2%}, mAP: {map_:.2%}")
            print(f"rank-1: {cmc[0]:.2%}, mAP: {map_:.2%}", file=log_file)

            # Save checkpoint weights every args.save_period Epoch
            save_param_list = get_param_list(net)
            if (epoch >= 2) and (epoch % args.save_period) == 0:
                path = f"epoch_{epoch:02}_rank1_{cmc[0] * 100:.2f}_mAP_{map_ * 100:.2f}.ckpt"
                save_checkpoint(save_param_list, path)

            # Record the best performance
            if map_ > BEST_MAP:
                BEST_MAP = map_

            if cmc[0] > BEST_R1:
                print(epoch, "  ", cmc[0], " ", BEST_R1)
                best_param_list = save_param_list
                best_path = f"best_epoch_{epoch:02}_rank1_{cmc[0] * 100:.2f}_mAP_{map_ * 100:.2f}.ckpt"
                BEST_R1 = cmc[0]
                BEST_EPOCH = epoch

            print("******************************************************************************")
            print("******************************************************************************",
                  file=log_file)

            log_file.flush()

    print("=> Save best parameters...")
    print("=> Save best parameters...", file=log_file)
    save_checkpoint(best_param_list, best_path)

    if args.dataset == "SYSU":
        print(f"For SYSU-MM01 {args.sysu_mode} search, the testing result is:")
        print(f"For SYSU-MM01 {args.sysu_mode} search, the testing result is:", file=log_file)
    elif args.dataset == "RegDB":
        print(f"For RegDB {args.regdb_mode} search, the testing result is:")
        print(f"For RegDB {args.regdb_mode} search, the testing result is:", file=log_file)

    print(f"Best: rank-1: {BEST_R1:.2%}, mAP: {BEST_MAP:.2%}, \
        Best epoch: {BEST_EPOCH}(according to Rank-1)")
    print(f"Best: rank-1: {BEST_R1:.2%}, mAP: {BEST_MAP:.2%}, \
        Best epoch: {BEST_EPOCH}(according to Rank-1)", file=log_file)
    log_file.flush()
    log_file.close()
