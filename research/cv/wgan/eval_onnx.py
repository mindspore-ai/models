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

import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.args import get_args


def create_session(checkpoint_path, target_device):
    """Create ONNX runtime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def generate_images(session, input_name, n_images, image_size, nz, nc, input_seed, output_dir):
    """
    Generates images
    Args:
        session (onnxruntime.capi.onnxruntime_inference_collection.InferenceSession): ONNX session
        input_name (str): ONNX session input name
        n_images (int): number of images to generate
        image_size (int): image size
        nz (int): size of the latent z vector
        nc (int): number of input channels
        input_seed (int): seed value for generator
        output_dir (str): directory to store generated images
    """
    randgen = np.random.RandomState(input_seed)
    fixed_noise = randgen.normal(size=[n_images, nz, 1, 1]).astype(np.float32)
    fake = session.run(None, {input_name: fixed_noise})
    fake = np.array(fake, dtype=np.float32).reshape(
        (n_images, nc, image_size, image_size))
    fake = fake * 0.5 * 255 + 0.5 * 255

    for i in range(n_images):
        img_pil = fake[i].reshape((1, nc, image_size, image_size))
        img_pil = img_pil[0].astype(np.uint8).transpose((1, 2, 0))
        img_pil = Image.fromarray(img_pil)
        img_pil.save(os.path.join(output_dir, f"generated_{i:02}.png"))


def main():
    args_opt = get_args('eval_onnx')

    with open(args_opt.config, 'r', encoding='utf-8') as gencfg:
        generator_config = json.loads(gencfg.read())

    image_size = generator_config["imageSize"]
    nz = generator_config["nz"]
    nc = generator_config["nc"]

    session, input_name = create_session(args_opt.file_name, args_opt.device_target)
    generate_images(
        session=session, input_name=input_name, n_images=args_opt.nimages,
        image_size=image_size, nz=nz, nc=nc,
        input_seed=args_opt.input_seed, output_dir=args_opt.output_dir
    )

    print("Generate images success!")


if __name__ == '__main__':
    main()
