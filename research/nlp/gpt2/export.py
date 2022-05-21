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

"""export checkpoint file into models"""
import argparse
import os
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, load_checkpoint, export

from src.gpt2_for_finetune import GPT2LM
from src.finetune_eval_config import gpt2_net_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune and Evaluate language modelings task")
    parser.add_argument("--load_ckpt_path", type=str, default="",
                        help="Load the checkpoint path.")
    parser.add_argument("--save_air_path", type=str, default="",
                        help="Save the air path.")
    args_opt = parser.parse_args()

    Load_checkpoint_path = os.path.realpath(args_opt.load_ckpt_path)
    save_air_path = os.path.realpath(args_opt.save_air_path)

    net = GPT2LM(config=gpt2_net_cfg,
                 is_training=False,
                 use_one_hot_embeddings=False)

    load_checkpoint(Load_checkpoint_path, net=net)

    net.set_train(False)

    input_ids = Tensor(np.zeros([gpt2_net_cfg.batch_size, gpt2_net_cfg.seq_length]), mstype.int32)
    input_mask = Tensor(np.zeros([gpt2_net_cfg.batch_size, gpt2_net_cfg.seq_length]), mstype.int32)
    label_ids = Tensor(np.zeros([gpt2_net_cfg.batch_size, gpt2_net_cfg.seq_length]), mstype.int32)
    input_data = [input_ids, input_mask, label_ids]
    print("====================    Start exporting   ==================")
    print(" | Ckpt path: {}".format(Load_checkpoint_path))
    print(" | Air path: {}".format(save_air_path))
    export(net, *input_data, file_name=os.path.join(save_air_path, 'gpt2'), file_format="MINDIR")
    print("====================    Exporting finished   ==================")
