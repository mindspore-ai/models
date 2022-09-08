# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from src.models import Monitor

def config_ckpoint(config, lr, step_size, model=None, eval_dataset=None):
    cb = [Monitor(lr_init=lr.asnumpy(), model=model, eval_dataset=eval_dataset)]
    if config.save_checkpoint and config.rank_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_save_dir = config.save_checkpoint_path + "ckpt_" + str(config.rank_id) + "/"
        ckpt_cb = ModelCheckpoint(prefix="mobilenetv2", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]
    return cb
