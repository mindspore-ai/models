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
import os
import shutil
from mindspore import context
from mindspore import load_checkpoint

from model_utils.device_adapter import get_device_id
from model_utils.config import config as cfg
from src.network import AutoEncoder
from src.eval_utils import apply_eval
from src.utils import get_results

context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, device_id=get_device_id())


def get_network():
    current_path = os.path.abspath(os.path.dirname(__file__))
    auto_encoder = AutoEncoder(cfg)
    if cfg.model_arts:
        import moxing as mox

        mox.file.copy_parallel(src_url=cfg.checkpoint_url, dst_url=cfg.cache_ckpt_file)
        ckpt_path = cfg.cache_ckpt_file
    else:
        ckpt_path = cfg.checkpoint_path
    ckpt_path = os.path.join(current_path, ckpt_path)
    load_checkpoint(ckpt_path, net=auto_encoder)
    auto_encoder.set_train(False)
    return auto_encoder


if __name__ == "__main__":
    if os.path.exists(cfg.save_dir):
        shutil.rmtree(cfg.save_dir, True)
    net = get_network()
    get_results(cfg, net)
    print("Generate results at", cfg.save_dir)
    apply_eval(cfg)
