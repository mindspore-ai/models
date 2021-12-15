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
"""Eval"""
import os
from mindspore.context import set_context, set_auto_parallel_context, reset_auto_parallel_context, \
    ParallelMode, PYNATIVE_MODE
from mindspore.communication import init
from src.utils.dataset import get_datasets
from src.utils.temporal_shift import make_temporal_pool
from src.utils.dataset_config import return_dataset
from src.model.net import TSM
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id

def main():
    args = config
    num_class, args.train_list, args.val_list, args.root_path, prefix = return_dataset(args.dataset,
                                                                                       args.modality, args.data_path)
    base_model = args.arch

    if get_device_num() > 1:
        set_context(mode=PYNATIVE_MODE, device_target=args.device_target, device_id=get_device_id())
        reset_auto_parallel_context()
        set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                  gradients_mean=True, device_num=get_device_num())
        init()
        rank = get_rank_id()
        args.gpus = get_device_num()

    elif get_device_num() == 1:
        set_context(mode=PYNATIVE_MODE, device_target=args.device_target)
        reset_auto_parallel_context()
        rank = get_rank_id()
        args.gpus = get_device_num()
        set_auto_parallel_context(parallel_mode=ParallelMode.STAND_ALONE, device_num=get_device_num())
    net = TSM(num_class, args.num_segments, args.modality,
              base_model=base_model,
              consensus_type=args.consensus_type,
              dropout=args.dropout,
              img_feature_dim=args.img_feature_dim,
              partial_bn=not args.no_partialbn,
              pretrain=args.pretrain,
              is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
              fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
              temporal_pool=args.temporal_pool,
              non_local=args.non_local)

    crop_size = net.crop_size
    scale_size = net.scale_size
    input_mean = net.input_mean
    input_std = net.input_std
    train_augmentation = net.get_augmentation(flip=False)
    if args.temporal_pool and not args.resume:
        make_temporal_pool(net.module.base_model, args.num_segments)

    _, val_loader = get_datasets(args, rank, input_mean, train_augmentation,
                                 input_std, scale_size, crop_size, prefix)

    for i, data in enumerate(val_loader.create_dict_iterator(output_numpy=True)):
        data['frames'].tofile(os.path.join(args.data_path, "frames/frames{}.bin".format(i)))
        data['label'].tofile(os.path.join(args.data_path, "label/label{}.bin".format(i)))





if __name__ == '__main__':
    main()
