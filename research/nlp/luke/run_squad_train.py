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
"""run squad train"""
import functools
import os
from mindspore.common import set_seed
from mindspore import context
from mindspore.communication import get_group_size
from mindspore.communication import init
from mindspore.context import ParallelMode

from src.luke.config import LukeConfig
from src.model_utils.config_args import args_config as args
from src.model_utils.moxing_adapter import get_device_num, sync_data, get_device_id, get_rank_id
from src.reading_comprehension.dataLoader import load_train
from src.reading_comprehension.model import LukeForReadingComprehension
from src.reading_comprehension.train import do_train
from src.utils.model_utils import ModelArchive

context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
set_seed(args.seed)
if args.duoka:
    context.set_auto_parallel_context(device_num=get_device_num(),
                                      parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)
    context.set_context(save_graphs_path=os.path.join('./save_graphs_path', str(get_rank_id())))
    init()

if args.modelArts:
    args.data = args.local_data_url
    args.output_dir = args.local_train_url
    args.model_file = args.local_checkpoint_url
    context.set_auto_parallel_context(device_num=get_device_num(),
                                      parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)
    init()

if args.distribute:
    init()
    device_num = get_group_size()
    context.set_auto_parallel_context(device_num=device_num,
                                      parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)


config = args


def moxing_wrapper(pre_process=None, post_process=None):
    """
    Moxing wrapper to download dataset and upload outputs.
    """

    def wrapper(run_func):
        @functools.wraps(run_func)
        def wrapped_func(*wrap_args, **kwargs):
            # Download data from data_url
            if config.modelArts:
                if config.data_url:
                    sync_data(config.data_url, config.local_data_url)
                    print("Dataset downloaded: ", os.listdir(config.local_data_url))
                if config.checkpoint_url:
                    sync_data(config.checkpoint_url, config.local_checkpoint_url)
                    print("Preload downloaded: ", os.listdir(config.local_checkpoint_url))
                if config.train_url:
                    sync_data(config.train_url, config.local_train_url)
                    print("Workspace downloaded: ", os.listdir(config.local_train_url))

                context.set_context(save_graphs_path=os.path.join(config.local_train_url, str(get_rank_id())))
                config.device_num = get_device_num()
                config.device_id = get_device_id()
                if not os.path.exists(config.local_train_url):
                    os.makedirs(config.local_train_url)

                if pre_process:
                    pre_process()

            run_func(*wrap_args, **kwargs)

            # Upload data to train_url
            if config.modelArts:
                if post_process:
                    post_process()

                if config.train_url:
                    print("Start to copy output directory")
                    sync_data(config.local_train_url, config.train_url)

        return wrapped_func

    return wrapper


@moxing_wrapper()
def runtrain():
    """run squad train"""
    # load pretrain
    model_archive = ModelArchive.load(args.model_file)
    args.bert_model_name = model_archive.bert_model_name
    args.max_mention_length = model_archive.max_mention_length
    args.model_path = model_archive.model_path
    luke_config = LukeConfig(**model_archive.metadata["model_config"])
    args.model_config = luke_config
    # model art
    network = LukeForReadingComprehension(luke_config)
    network.luke.entity_embeddings.entity_embeddings.embedding_table.requires_grad = False
    dataset = load_train(args)
    do_train(dataset, network, args)


runtrain()
