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
"""eval.py"""

import os
import sys
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net

from src.model_utils.config import config
from src.dataset import create_dataset
from src.pcb import PCB
from src.rpp import RPP
from src.eval_utils import apply_eval
from src.logging import Logger
from src.model_utils.moxing_adapter import moxing_wrapper

def build_model(num_classes):
    model = None
    if config.model_name == "PCB":
        model = PCB(num_classes)
    elif config.model_name == "RPP":
        model = RPP(num_classes)
    return model

@moxing_wrapper()
def eval_net():
    """eval_net"""
    target = config.device_target
    #init context
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
    if target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)
    log_save_dir = os.path.join(config.output_path, config.log_save_path)
    if not os.path.isdir(log_save_dir):
        os.makedirs(log_save_dir)
    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(log_save_dir, 'log.txt'))
    #create_dataset
    _, train_set = \
create_dataset(dataset_name=config.dataset_name, dataset_path=config.dataset_path, \
subset_name="train", batch_size=config.batch_size, num_parallel_workers=config.num_parallel_workers)
    print("Loading query dataset...")
    query_dataset, query_set = \
create_dataset(dataset_name=config.dataset_name, dataset_path=config.dataset_path, \
subset_name="query", batch_size=config.batch_size, num_parallel_workers=config.num_parallel_workers)
    print("Loading gallery dataset...")
    gallery_dataset, gallery_set = \
create_dataset(dataset_name=config.dataset_name, dataset_path=config.dataset_path, \
subset_name="gallery", batch_size=config.batch_size, num_parallel_workers=config.num_parallel_workers)
    #network
    num_classes = train_set.num_ids
    eval_network = build_model(num_classes)
    print("Load Checkpoint!")
    #load checkpoint
    if config.checkpoint_file_path != "":
        param_dict = load_checkpoint(config.checkpoint_file_path)
        load_param_into_net(eval_network, param_dict)
    #apply eval
    print("Processing, please wait a moment.")
    eval_param_dict = {"net": eval_network, "query_dataset": query_dataset, "gallery_dataset": gallery_dataset, \
"query_set": query_set.data, "gallery_set": gallery_set.data}
    mAP, cmc_scores = apply_eval(eval_param_dict)
    print('Mean AP: {:4.1%}'.format(mAP), flush=True)
    print('CMC Scores{:>12}'.format(config.dataset_name), flush=True)
    cmc_topk = (1, 5, 10)
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores[config.dataset_name][k - 1]), flush=True)

if __name__ == "__main__":
    eval_net()
