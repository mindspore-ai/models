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
"""eval.py"""

import os
import os.path as osp
import time
import argparse
import psutil
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

from mindspore import context, load_checkpoint, load_param_into_net, DatasetHelper
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_group_size
from mindspore.dataset.transforms.transforms import Compose
from src.dataset import SYSUDatasetGenerator, RegDBDatasetGenerator, TestData
from src.dataset import process_gallery_sysu, process_query_sysu, process_test_regdb
from src.evalfunc import test
from src.models.ddag import DDAG

from src.utils import genidx
from PIL import Image


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
    parser.add_argument('--test-batch', default=64, type=int,
                        metavar='tb', help='testing batch size')
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

    # testing configs
    parser.add_argument('--device-target', default="CPU",
                        choices=["CPU", "GPU", "Ascend"])
    parser.add_argument('--gpu', default='0', type=str,
                        help='set CUDA_VISIBLE_DEVICES')

    # Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU).
    #  If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be
    # the number set in the environment variable 'CUDA_VISIBLE_DEVICES'.
    #  For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the 'device_id' can be 0,1,2 at the moment,
    # 'device_id' starts from 0, and 'device_id'=0 means using GPU of number 4.
    parser.add_argument('--device-id', default=0, type=int, help='')

    parser.add_argument('--device-num', default=1, type=int,
                        help='the total number of available gpus')
    parser.add_argument('--resume', '-r', default='', type=str,
                        help='resume from checkpoint, no resume:""')
    parser.add_argument('--run_distribute', action='store_true',
                        help="if set true, this code will be run on distributed architecture with mindspore")

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


def print_dataset_info(dataset_type_info, trainset, query_label_info, gall_label_info, start_time_info):
    """
    function of print dataset information
    """
    n_class_info = len(np.unique(trainset.train_color_label))
    nquery_info = len(query_label_info)
    ngall_info = len(gall_label_info)
    print('Dataset {} statistics:'.format(dataset_type_info))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(
        n_class_info, len(trainset.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(
        n_class_info, len(trainset.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(
        len(np.unique(query_label_info)), nquery_info))
    print('  gallery  | {:5d} | {:8d}'.format(
        len(np.unique(gall_label_info)), ngall_info))
    print('  ------------------------------')
    print('Data Loading Time:\t {:.3f}'.format(time.time() - start_time_info))


def decode(img):
    return Image.fromarray(img)


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
    if args.MSmode == 'GRAPH_MODE':
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=device, save_graphs=False)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=device)

    if device == "CPU":
        local_data_path = args.data_path
        args.run_distribute = False
    else:
        if device in  ["GPU", "Ascend"]:
            local_data_path = args.data_path
            context.set_context(device_id=args.device_id)

        # distributed running context setting
        if args.run_distribute:
            # Ascend target
            if device == "Ascend":
                if args.device_num > 1:
                    # not useful now, because we only have one Ascend Device
                    pass
            # end of if args.device_num > 1:
                init()

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

            local_data_path = "/cache/data"
            args.data_path = local_data_path
            print("Download data...")
            mox.file.copy_parallel(src_url=args.data_url,
                                   dst_url=local_data_path)
            print("Download complete!(#^.^#)")
            # print(os.listdir(local_data_path))

    ########################################################################
    # Logging
    ########################################################################

    if device in ['GPU', 'CPU', 'Ascend']:
        checkpoint_path = os.path.join("logs", args.tag, "testing")
        os.makedirs(checkpoint_path, exist_ok=True)

        suffix = str(args.dataset)

        if args.part > 0:
            suffix = suffix + '_P_{}'.format(args.part)

        if args.dataset == 'RegDB':
            suffix = suffix + '_trial_{}'.format(args.trial)

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

    if dataset_type == "SYSU":
        data_path = args.data_path
    elif dataset_type == "RegDB":
        data_path = args.data_path

    start_epoch = 1
    feature_dim = args.low_dim
    start_time = time.time()

    print("==> Loading data")
    # Data Loading code

    transform_test = Compose(
        [
            decode,
            vision.Resize((args.img_h, args.img_w)),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)
        ]
    )

    ifDebug_dic = {"yes": True, "no": False}
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

    print('==> Building model..')
    n_class = len(np.unique(trainset_generator.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)

    net = DDAG(args.low_dim, class_num=n_class,\
               part=args.part, nheads=0)

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
    # Start Testing
    ########################################################################
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
        cmc, mAP, cmc_att, mAP_att = test(args, gallery_loader, query_loader, ngall,
                                          nquery, net, 1, gallery_cam=gall_cam, query_cam=query_cam)

    if args.dataset == "RegDB":
        if args.regdb_mode == "v2i":
            cmc, mAP, cmc_att, mAP_att = test(args, gallery_loader, query_loader, ngall,
                                              nquery, net, 2)
        elif args.regdb_mode == "i2v":
            cmc, mAP, cmc_att, mAP_att = test(args, gallery_loader, query_loader, ngall,
                                              nquery, net, 1)

    if args.dataset == "SYSU":
        print(f"For SYSU-MM01 {args.sysu_mode} search, the testing result is:")
        print(
            f"For SYSU-MM01 {args.sysu_mode} search, the testing result is:", file=log_file)
    elif args.dataset == "RegDB":
        print(f"For RegDB {args.regdb_mode} search, the testing result is:")
        print(
            f"For RegDB {args.regdb_mode} search, the testing result is:", file=log_file)

    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=log_file)

    if args.part > 0:
        print('FC_att:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att))
        print('FC_att:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att), file=log_file)

    print("******************************************************************************")
    print("******************************************************************************",
          file=log_file)
    log_file.flush()
    log_file.close()
