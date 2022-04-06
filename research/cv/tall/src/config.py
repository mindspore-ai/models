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
'''All hyperparameters and training parameters involved in the model'''
import os
from datetime import datetime
from mindspore import context


class CONFIG:
    ''' Configure training related parameters '''
    def __init__(self, data_dir, log_dir=None):
        super(CONFIG, self).__init__()
        # Random seed
        self.seed = 2021

        # Hardware equipment selection
        self.mode = context.GRAPH_MODE


        # Dataset path
        self.data_path = data_dir
        self.valid_csv_path = os.path.join(self.data_path, "exp_data/TACoS/val_clip-sentvec.pkl")
        self.test_csv_path = os.path.join(self.data_path, "exp_data/TACoS/test_clip-sentvec.pkl")
        self.train_csv_path = os.path.join(self.data_path, "exp_data/TACoS/train_clip-sentvec.pkl")
        self.test_feature_dir = os.path.join(self.data_path, "Interval128_256_overlap0.8_c3d_fc6/")
        self.train_feature_dir = os.path.join(self.data_path, "Interval64_128_256_512_overlap0.8_c3d_fc6/")
        self.movie_length_info_path = "./video_allframes_info_unix.pkl"

        # Data loading
        self.buffer_size = 1024
        self.context_num = 1
        self.context_size = 128
        self.IoU = 0.5
        self.nIoL = 0.15

        # Training strategy
        self.max_epoch = 3
        self.batch_size = 56
        self.test_batch_size = 128
        self.save_checkpoint_steps = 200
        self.keep_checkpoint_max = 100

        # Optimizer hyperparameters
        self.optimizer = 'Adam'
        self.lr = 5e-5
        self.weight_decay = 1e-5
        self.momentum = 0.9

        # Model hyperparameters
        self.visual_dim = 4096 * 3
        self.sentence_embed_dim = 4800
        self.semantic_dim = 1024  # the size of visual and semantic comparison size
        self.middle_layer_dim = 1024
        self.lambda_reg = 0.01
        self.alpha = 1.0 / self.batch_size

        # Training log
        self.save_log = True
        now_time = datetime.now()
        time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
        if log_dir is None:
            self.log_dir = os.path.join("logs", time_str)
        else:
            self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_txt_path = os.path.join(self.log_dir, "log.txt")
        self.test_output_path = os.path.join(self.log_dir, "ctrl_test_results.txt")

    def __str__(self):
        return f"log_dir:{str(self.log_dir)}\noptimizer:{str(self.optimizer)}" \
               f"\nvs_lr:{str(self.lr)}\nbatch_size:{str(self.batch_size)}" \
               f"\nmax_epoch:{str(self.max_epoch)}\nsemantic_dim:{str(self.semantic_dim)}" \
               f"\nmiddle_layer_dim:{str(self.middle_layer_dim)}" \
               f"\nself.train_feature_dir:{self.train_feature_dir}"
