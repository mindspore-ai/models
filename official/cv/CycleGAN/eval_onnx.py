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

"""Cycle GAN ONNX test."""

import os

import onnxruntime as ort

from src.utils.args import get_args
from src.dataset.cyclegan_dataset import create_dataset
from src.utils.reporter import Reporter
from src.utils.tools import save_image


def create_session(checkpoint_path, target_device):
    """Load ONNX model and create ORT session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_names = [x.name for x in session.get_inputs()]
    return session, input_names


def predict():
    """Predict function."""
    args = get_args("predict")

    file_name, file_extension = os.path.splitext(args.export_file_name)
    gen_a_file_name = f"{file_name}_AtoB{file_extension}"
    gen_b_file_name = f"{file_name}_BtoA{file_extension}"

    gen_a, [gen_a_input_name] = create_session(gen_a_file_name, args.platform)
    gen_b, [gen_b_input_name] = create_session(gen_b_file_name, args.platform)

    imgs_out = os.path.join(args.outputs_dir, "predict")
    if not os.path.exists(imgs_out):
        os.makedirs(imgs_out)
    if not os.path.exists(os.path.join(imgs_out, "fake_A")):
        os.makedirs(os.path.join(imgs_out, "fake_A"))
    if not os.path.exists(os.path.join(imgs_out, "fake_B")):
        os.makedirs(os.path.join(imgs_out, "fake_B"))

    args.data_dir = 'testA'
    ds = create_dataset(args)
    reporter = Reporter(args)
    reporter.start_predict("A to B")
    for data in ds.create_dict_iterator(output_numpy=True):
        img_a = data["image"]
        path_a = data["image_name"][0]
        path_b = path_a[0:-4] + "_fake_B.jpg"
        [fake_b] = gen_a.run(None, {gen_a_input_name: img_a})
        save_image(fake_b, os.path.join(imgs_out, "fake_B", path_b))
        save_image(img_a, os.path.join(imgs_out, "fake_B", path_a))
    reporter.info('save fake_B at %s', os.path.join(imgs_out, "fake_B", path_a))
    reporter.end_predict()

    args.data_dir = 'testB'
    ds = create_dataset(args)
    reporter.dataset_size = args.dataset_size
    reporter.start_predict("B to A")
    for data in ds.create_dict_iterator(output_numpy=True):
        img_b = data["image"]
        path_b = data["image_name"][0]
        path_a = path_b[0:-4] + "_fake_A.jpg"
        [fake_a] = gen_b.run(None, {gen_b_input_name: img_b})
        save_image(fake_a, os.path.join(imgs_out, "fake_A", path_a))
        save_image(img_b, os.path.join(imgs_out, "fake_A", path_b))
    reporter.info('save fake_A at %s', os.path.join(imgs_out, "fake_A", path_b))
    reporter.end_predict()


if __name__ == "__main__":
    predict()
