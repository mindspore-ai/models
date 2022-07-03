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
"""Run evaluation for a model exported to ONNX"""
import glob
import numpy as np
import onnxruntime as ort
import PIL.Image as pil_image
import mindspore.dataset as ds

from src.model_utils.config import config
from src.metric import SRCNNpsnr
from src.utils import convert_rgb_to_y


class EvalDataset:
    def __init__(self, images_dir, eval_scale, image_width, image_height):
        self.images_dir = images_dir
        scale = eval_scale
        self.lr_group = []
        self.hr_group = []
        for image_path in sorted(glob.glob('{}/*'.format(images_dir))):
            hr = pil_image.open(image_path).convert('RGB')
            hr_width = image_width
            hr_height = image_height
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
            lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            self.lr_group.append(lr)
            self.hr_group.append(hr)

    def __len__(self):
        return len(self.lr_group)

    def __getitem__(self, idx):
        return np.expand_dims(self.lr_group[idx] / 255., 0), np.expand_dims(self.hr_group[idx] / 255., 0)


def create_eval_dataset(images_dir, scale, image_width, image_height, batch_size=1):
    dataset = EvalDataset(images_dir, scale, image_width, image_height)
    data_set = ds.GeneratorDataset(dataset, ["lr", "hr"], shuffle=False)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set


def create_session(checkpoint_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def run_eval():
    cfg = config
    session, input_name = create_session(cfg.onnx_path, cfg.device_target)

    eval_ds = create_eval_dataset(cfg.data_path, cfg.scale, cfg.image_width, cfg.image_height)
    metrics = {
        'PSNR': SRCNNpsnr()
    }

    for batch in eval_ds.create_dict_iterator(num_epochs=1, output_numpy=True):
        y_pred = session.run(None, {input_name: batch['lr']})[0]
        for metric in metrics.values():
            metric.update(y_pred, batch['hr'])

    return {name: metric.eval() for name, metric in metrics.items()}


if __name__ == '__main__':
    results = run_eval()

    for name, value in results.items():
        print(f'{name}: {value:.4f}')
