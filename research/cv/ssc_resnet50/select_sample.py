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
"""Select sample"""
import os
import ast
import sys
import logging
import argparse

from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed

from src.network.model import ModelBaseDis
from src.dataset import create_select_dataset
from src.utils import getMatrix_final, setup_default_logging
from src.model_utils.local_adapter import get_device_id, get_device_num, get_rank_id
from src.model_utils.config import get_config


def run_select(args):
    if args.rank == 0:
        logging.info("start create network!")
    netbase = ModelBaseDis(args, True)

    if args.rank == 0:
        logging.info("args.pre_trained exists, value: %s", args.pre_trained)
    param_dict = load_checkpoint(args.pre_trained)
    param_dict_new = {}
    for param_key, param_values in param_dict.items():
        if param_key.startswith('comatch_network.encode.'):
            param_dict_new[param_key[16:]] = param_values

    para_not_list, _ = load_param_into_net(netbase, param_dict_new, strict_load=True)
    netbase.set_train(False)
    if args.rank == 0:
        logging.info('param not load: %s', str(para_not_list))
        logging.info("load_checkpoint success!!")
    param_dict = None

    if args.rank == 0:
        logging.info("start create dataset!")
    dataset, data_size = create_select_dataset(args)
    args.steps_per_epoch = int(data_size / args.batch_size / args.device_num)
    if args.rank == 0:
        logging.info("step per epoch: %d", args.steps_per_epoch)

    data_loader = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)
    dict_name_val = {}
    save_path = os.path.join(args.exp_dir, str(args.rank) + '.txt')
    test = open(save_path, 'w')
    for i, data in enumerate(data_loader):
        output, _, _, _ = netbase(Tensor.from_numpy(data['img_data']), 1, 1, 1,
                                  Tensor.from_numpy(data['label_target']), True)
        btx = args.batch_size
        for each in range(btx):
            start = each * int(args.aug_num)
            end = (each+1) * int(args.aug_num)
            img_features = output[start: end, :].asnumpy()
            matrix_val = getMatrix_final(img_features, args.preserve)
            dict_name_val[data['label_path'][each]] = round(matrix_val.item(), 3)
            label_path = data["label_path"][each]
            if not isinstance(data["label_path"][each], str):
                label_path = label_path.decode()
            test.write('{} {:.03f} \n'.format(label_path, matrix_val.item()))
            test.flush()
        if args.rank == 0:
            logging.info("process number %d", i * args.batch_size)
    test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate annotation')
    parser.add_argument("--is_distributed", type=ast.literal_eval, default=False,
                        help="Run distribute, default: false.")
    parser.add_argument('--aug_num', type=int, default=12, help='numbers of data aug')
    parser.add_argument('--preserve', type=float, default=0.25, help='numbers of data aug')
    parser.add_argument('--batch_size', type=int, default=50, help='percentage of labeled samples')
    parser.add_argument("--device_target", type=str, default="GPU",
                        help="device where the code will be implemented, default is Ascend")
    parser.add_argument('--annotation', type=str, help='annotation file')

    ## dataset settings
    parser.add_argument("--exp_dir", type=str, help="txt save path")
    parser.add_argument("--pre_trained", type=str, help="The model has been trained in step1.")

    args_input = parser.parse_args()
    sys.argv = sys.argv[:1]
    args_config = get_config()
    for key, value in args_input.__dict__.items():
        if value is not None:
            setattr(args_config, key, value)

    logger = setup_default_logging(args_config)
    logger.info(args_config)

    set_seed(1)
    logger.info("start init dist!")
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_config.device_target,
                        device_id=int(get_device_id()))
    if args_config.is_distributed:
        if args_config.device_target == "Ascend":
            init("hccl")
            args_config.rank = get_rank_id()
            args_config.device_num = get_device_num()
            logger.info("init ascend dist rank: %d, device_num: %d.", args_config.rank, args_config.device_num)
            context.set_auto_parallel_context(device_num=args_config.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)

        else:
            init("nccl")
            context.reset_auto_parallel_context()
            args_config.rank = get_rank()
            args_config.device_num = get_group_size()
            logger.info("init gpu dist rank: %d, device_num: %d.", args_config.rank, args_config.device_num)
            context.set_auto_parallel_context(device_num=args_config.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        args_config.rank = 0
        args_config.device_num = 1
    run_select(args_config)
