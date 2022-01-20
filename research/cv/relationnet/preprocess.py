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
'''Preprocess for 310 inference'''
import random
import argparse
import numpy as np
import src.dataset as dt
from src.config import relationnet_cfg as cfg

parser = argparse.ArgumentParser(description="Preprocess for Relationnet")
parser.add_argument("--data_path", default='./omniglot_resized/',
                    help="Path where the dataset is save")
parser.add_argument("-dt", "--device_target", type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument("-di", "--device_id", type=int, default=0, help='device id of GPU or Ascend. (Default: 0)')
args = parser.parse_args()

local_data_url = args.data_path
_, metatest_character_folders = dt.omniglot_character_folders(data_path=local_data_url)
for i in range(cfg.test_episode):
    degrees = random.choice([0, 90, 180, 270])
    flip = random.choice([True, False])
    task = dt.OmniglotTask(metatest_character_folders, cfg.class_num, cfg.sample_num_per_class,
                           cfg.sample_num_per_class)
    sample_dataloader = dt.get_data_loader(task, num_per_class=cfg.sample_num_per_class, split="train",
                                           shuffle=False, rotation=degrees, flip=flip)
    test_dataloader = dt.get_data_loader(task, num_per_class=cfg.sample_num_per_class, split="test",
                                         shuffle=True, rotation=degrees, flip=flip)
    test_samples, _ = sample_dataloader.__iter__().__next__()
    test_batches, test_batch_labels = test_dataloader.__iter__().__next__()
    test_input = np.concatenate((test_samples.asnumpy(), test_batches.asnumpy()), axis=0)# concat samples and batches
    test_input.tofile("./data/a"+str(i)+".bin")
    test_batch_labels.asnumpy().astype(np.int32).tofile("./data/label/b"+str(i)+".bin")
