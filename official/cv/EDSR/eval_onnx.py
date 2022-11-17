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
"""
#################evaluate EDSR example on DIV2K########################
"""
import os
import time

import mindspore
from mindspore.common import set_seed
from mindspore import Tensor
import onnxruntime as ort

from src.metric import PSNR, SaveSrHr
from src.utils import init_env, init_dataset
from model_utils.config import config

set_seed(2021)

def create_session(checkpoint_path, target_device):
    """Create ONNX runtime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_names = [x.name for x in session.get_inputs()]
    return session, input_names

def unpadding(img, target_shape):
    h, w = target_shape[2], target_shape[3]
    _, _, img_h, img_w = img.shape
    if img_h > h:
        img = img[:, :, :h, :]
    if img_w > w:
        img = img[:, :, :, :w]
    return img

def do_eval(session, input_names, ds_val, metrics, cur_epoch=None):
    """
    do eval for psnr and save hr, sr
    """
    total_step = ds_val.get_dataset_size()
    setw = len(str(total_step))
    begin = time.time()
    step_begin = time.time()
    rank_id = 0
    for i, (lr, hr) in enumerate(ds_val):
        input_data = [lr.asnumpy()]
        sr = session.run(None, dict(zip(input_names, input_data)))
        sr = Tensor(unpadding(sr[0], hr.shape), mindspore.float32)
        _ = [m.update(sr, hr) for m in metrics.values()]
        result = {k: m.eval(sync=False) for k, m in metrics.items()}
        result["time"] = time.time() - step_begin
        step_begin = time.time()
        print(f"[{i+1:>{setw}}/{total_step:>{setw}}] rank = {rank_id} result = {result}", flush=True)
    result = {k: m.eval(sync=True) for k, m in metrics.items()}
    result["time"] = time.time() - begin
    print(f"evaluation result = {result}", flush=True)
    return result

def run_eval():
    """
    run eval
    """
    print(config, flush=True)
    cfg = config
    cfg.lr_type = "bicubic_AUG_self_ensemble"

    init_env(cfg)
    session, input_names = create_session(cfg.pre_trained, 'GPU')

    if cfg.dataset_name == "DIV2K":
        cfg.batch_size = 1
        cfg.patch_size = -1
        ds_val = init_dataset(cfg, "valid")
        metrics = {
            "psnr": PSNR(rgb_range=cfg.rgb_range, shave=6 + cfg.scale),
        }
        if config.save_sr:
            save_img_dir = os.path.join(cfg.output_path, "HrSr")
            os.makedirs(save_img_dir, exist_ok=True)
            metrics["num_sr"] = SaveSrHr(save_img_dir)
        do_eval(session, input_names, ds_val, metrics)
        print("eval success", flush=True)
    else:
        raise RuntimeError("Unsupported dataset.")

if __name__ == '__main__':
    run_eval()
