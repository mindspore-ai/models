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
""" eval.py """

import os
import time

import argparse
import numpy as np
import psutil
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

from mindspore import context, load_checkpoint, load_param_into_net, DatasetHelper
from mindspore.dataset.transforms.transforms import Compose

from PIL import Image

from src.dataset import SYSUDatasetGenerator, RegDBDatasetGenerator, TestData, \
    process_query_sysu, process_gallery_sysu, process_test_regdb
from src.evalfunc import test
from src.models.mvd import MVD
from src.utils import genidx


def show_memory_info(hint=""):
    """
    show memort information
    """
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print(f"{hint} memory used: {memory} MB ")


def get_parser():
    """
    get parser
    """
    parser_ = argparse.ArgumentParser(description="DDAG Code Mindspore Version")

    parser_.add_argument("--MSmode", default="GRAPH_MODE", choices=["GRAPH_MODE", "PYNATIVE_MODE"])
    # dataset settings
    parser_.add_argument("--dataset", default='SYSU', choices=['SYSU', 'RegDB'],
                         help='dataset name: RegDB or SYSU')
    parser_.add_argument('--data_path', type=str, default='data')
    # Only used on Huawei Cloud OBS service,
    # when this is set, --data_path is overridden by --data-url
    parser_.add_argument("--data_url", type=str, default=None)
    parser_.add_argument('--trial', default=1, type=int,
                         metavar='t', help='trial (only for RegDB dataset)')
    parser_.add_argument('--test_batch', default=64, type=int,
                         metavar='tb', help='testing batch size')

    # image transform
    parser_.add_argument('--img_w', default=144, type=int,
                         metavar='imgw', help='img width')
    parser_.add_argument('--img_h', default=288, type=int,
                         metavar='imgh', help='img height')

    # model
    parser_.add_argument('--arch', default='resnet50', type=str,
                         help='network baseline:resnet50')
    parser_.add_argument('--z_dim', default=512, type=int,
                         help='information bottleneck z dim')

    # loss setting
    parser_.add_argument('--epoch', default=80, type=int,
                         metavar='epoch', help='epoch num')
    parser_.add_argument('--drop', default=0.2, type=float,
                         metavar='drop', help='dropout ratio')
    parser_.add_argument('--margin', default=0.3, type=float,
                         metavar='margin', help='triplet loss margin')

    # training configs
    parser_.add_argument('--device_target', default="CPU", choices=["CPU", "GPU", "Ascend", "Cloud"])
    parser_.add_argument('--gpu', default='0', type=str, help='set CUDA_VISIBLE_DEVICES')

    # Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU).
    # If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be
    # the number set in the environment variable 'CUDA_VISIBLE_DEVICES'.
    # For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the 'device_id' can be 0,1,2 at the moment,
    # 'device_id' starts from 0, and 'device_id'=0 means using GPU of number 4.
    parser_.add_argument('--device_id', default=0, type=int, help='')
    parser_.add_argument('--device_num', default=1, type=int,
                         help='the total number of available gpus')
    parser_.add_argument('--resume', '-r', default='', type=str,
                         help='resume from checkpoint, no resume:""')
    parser_.add_argument('--run_distribute', action='store_true',
                         help="if set true, this code will be run on distributed")

    # logging configs
    parser_.add_argument('--tag', default='toy', type=str,
                         help='ckpt suffix name')

    # testing / evaluation config
    parser_.add_argument('--sysu_mode', default='all', type=str, choices=['all', 'indoor'],
                         help='all or indoor')
    parser_.add_argument('--regdb_mode', default='v2i', type=str, choices=["v2i", "i2v"],
                         help='visible to infrared search;infrared to visible search.')

    return parser_


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
    """
    img decode
    """
    return Image.fromarray(img)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.device_target == 'GPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = args.device_target
    # init context
    if args.MSmode == "GRAPH_MODE":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=device, save_graphs=False, max_call_depth=3000)
    else:
        context.set_context(mode=context.PYNATIVE_MODE,
                            device_target=device, save_graphs=False, max_call_depth=3000)

    if device == "CPU":
        local_data_path = args.data_path
        args.run_distribute = False
    else:
        if device in ["GPU", "Ascend"]:
            local_data_path = args.data_path
            context.set_context(device_id=args.device_id)

        # Adapt to Huawei Cloud: download data from obs to local location
        if device == "Cloud":
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

    if device in ["GPU", "CPU", "Ascend"]:
        log_file = open(args.dataset + "_eval_performance.txt", "w", encoding="utf-8")

        print('Args: {}'.format(args))
        print('Args: {}'.format(args), file=log_file)

    ########################################################################
    # Create Test Set
    ########################################################################
    dataset_type = args.dataset

    if dataset_type == "SYSU":
        # infrared to visible(1->2)
        data_path = args.data_path
    elif dataset_type == "RegDB":
        # visible to infrared(2->1)
        data_path = args.data_path

    best_acc = 0  # best test accuracy
    start_epoch = 1
    start_time = time.time()

    print("==> Loading data")

    transform_test = Compose(
        [
            decode,
            vision.Resize((args.img_h, args.img_w)),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)
        ]
    )

    if dataset_type == "SYSU":

        trainset_generator = SYSUDatasetGenerator(data_dir=data_path)
        # testing set
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.sysu_mode)
        gall_img, gall_label, gall_cam = \
            process_gallery_sysu(data_path, mode=args.sysu_mode, random_seed=0)

    elif dataset_type == "RegDB":
        # train_set
        trainset_generator = RegDBDatasetGenerator(data_dir=data_path, trial=args.trial)
        color_pos, thermal_pos = \
            genidx(trainset_generator.train_color_label, trainset_generator.train_thermal_label)

        # testing set
        if args.regdb_mode == "v2i":
            query_img, query_label = \
                process_test_regdb(img_dir=data_path, modal="visible", trial=args.trial)
            gall_img, gall_label = \
                process_test_regdb(img_dir=data_path, modal="thermal", trial=args.trial)
        elif args.regdb_mode == "i2v":
            query_img, query_label = \
                process_test_regdb(img_dir=data_path, modal="thermal", trial=args.trial)
            gall_img, gall_label = \
                process_test_regdb(img_dir=data_path, modal="visible", trial=args.trial)

    ########################################################################
    # Create Query && Gallery
    ########################################################################

    gallset_generator = \
        TestData(gall_img, gall_label, img_size=(args.img_w, args.img_h))
    queryset_generator = \
        TestData(query_img, query_label, img_size=(args.img_w, args.img_h))

    ########################################################################
    # Define net
    ########################################################################

    print('==> Building model..')
    n_class = len(np.unique(trainset_generator.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)

    net = MVD(num_class=n_class, drop=args.drop, z_dim=args.z_dim, pretrain="")

    if args.resume != "":
        print("Resume checkpoint:{}".format(args.resume))
        print("Resume checkpoint:{}".format(args.resume), file=log_file)
        param_dict = load_checkpoint(args.resume)
        load_param_into_net(net, param_dict)
    else:
        print("Please specify your path to checkpoint file in args.resume!")
        exit()

    ########################################################################
    # Start Testing
    ########################################################################

    print('==> Start Testing...')

    net.set_train(mode=False)
    gallset = ds.GeneratorDataset(gallset_generator, ["img", "label"])
    gallset = gallset.map(operations=transform_test, input_columns=["img"])
    gallery_loader = gallset.batch(batch_size=args.test_batch)
    gallery_loader = DatasetHelper(gallery_loader, dataset_sink_mode=False)

    queryset = ds.GeneratorDataset(queryset_generator, ["img", "label"])
    queryset = queryset.map(operations=transform_test, input_columns=["img"])
    query_loader = queryset.batch(batch_size=args.test_batch)
    query_loader = DatasetHelper(query_loader, dataset_sink_mode=False)

    if args.dataset == "SYSU":
        cmc_ob, map_ob = test(args, gallery_loader, query_loader, ngall, \
                              nquery, net, gallery_cam=gall_cam, query_cam=query_cam)

    if args.dataset == "RegDB":
        cmc_ob, map_ob = test(args, gallery_loader, query_loader, ngall, \
                              nquery, net)

    print('Original Observation:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}|\
            Rank-20: {:.2%}| mAP: {:.2%}'.format( \
        cmc_ob[0], cmc_ob[4], cmc_ob[9], cmc_ob[19], map_ob))

    print('Original Observation:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}|\
            Rank-20: {:.2%}| mAP: {:.2%}'.format( \
        cmc_ob[0], cmc_ob[4], cmc_ob[9], cmc_ob[19], map_ob), file=log_file)

    map_ = map_ob
    cmc = cmc_ob

    print("************************************************************************")
    print("************************************************************************", file=log_file)

    log_file.flush()

    if args.dataset == "SYSU":
        print(f"For SYSU-MM01 {args.sysu_mode} search, the testing result is:")
        print(f"For SYSU-MM01 {args.sysu_mode} search, the testing result is:", file=log_file)
    elif args.dataset == "RegDB":
        print(f"For RegDB {args.regdb_mode} search, the testing result is:")
        print(f"For RegDB {args.regdb_mode} search, the testing result is:", file=log_file)

    print(f"Best: rank-1: {cmc[0]:.2%}, mAP: {map_:.2%}")
    print(f"Best: rank-1: {cmc[0]:.2%}, mAP: {map_:.2%}", file=log_file)
    log_file.flush()
    log_file.close()
