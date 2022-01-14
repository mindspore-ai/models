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
""" Model Main """
import os
import time
import argparse
import numpy as np
from api.infer import SdkApi
from config import config as cfg
from src.image_transform import Normalize, TransporeAndMul, Resize
from PIL import Image

def parser_args():
    """ Args Setting """
    parser = argparse.ArgumentParser(description="pgan inference")

    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="../pipeline/pgan.pipeline",
        help="image file path. The default is '../pipeline/pgan.pipeline'. ")

    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="output",
        help=
        "cache dir of inference result. The default is 'output'.")
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=64,
        help=
        "number of inputNoise. The default is 64.")

    args_ = parser.parse_args()
    return args_

def resizeTensor(data, out_size_image):
    """resizeTensor

    Returns:
        output.
    """
    out_data_size = (data.shape[0], data.shape[
        1], out_size_image[0], out_size_image[1])
    outdata = []
    data = np.clip(data, a_min=-1, a_max=1)
    transformList = [Normalize((-1., -1., -1.), (2, 2, 2)), TransporeAndMul(), Resize(out_size_image)]
    for img in range(out_data_size[0]):
        processed = data[img]
        for transform in transformList:
            processed = transform(processed)
        processed = np.array(processed)
        outdata.append(processed)
    return outdata

def image_compose(out_images, size=(8, 8)):
    """image_compose

    Returns:
        output.
    """
    to_image = Image.new('RGB', (size[0] * 128, size[1] * 128))
    for y in range(size[1]):
        for x in range(size[0]):
            from_image = Image.fromarray(out_images[x * size[0] + y])
            to_image.paste(from_image, (x * 128, y * 128))
    return to_image

def size_compute(batch):
    """size_compute

    Returns:
        output.
    """
    size1 = int(np.sqrt(batch))
    size2 = size1
    if size1 * size1 < batch:
        if (batch % size1) == 0:
            size2 = batch // size1
        else:
            i = size1
            while i > 1:
                i = i - 1
                if (batch % i) == 0:
                    size2 = batch // i
                    size1 = i
                    break
    return (size1, size2)

def image_inference(pipeline_path, stream_name, result_dir, batch_size):
    """ Image Inference """
    # init stream manager
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0

    input_dim = 512
    n_image = int(batch_size)
    data = []
    for _ in range(n_image):
        latent_code_eval = np.random.randn(1, input_dim).astype(np.float32)

        img_np = latent_code_eval

        start_time = time.time()

        # set input data
        sdk_api.send_tensor_input(stream_name, img_data_plugin_id, "appsrc0",
                                  img_np.tobytes(), img_np.shape, cfg.TENSOR_DTYPE_FLOAT32)


        result = sdk_api.get_result(stream_name)
        end_time = time.time() - start_time
        print(f"The inference time is {end_time}s")
        data_tmp = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        data_tmp = data_tmp.reshape(3, 128, 128)
        data.append(data_tmp)
    data = np.array(data)

    #save result
    out_image = resizeTensor(data, (128, 128))
    size1, size2 = size_compute(batch_size)
    to_image = image_compose(out_image, (size1, size2))
    to_image.save(os.path.join(result_dir, "result.jpg"))

if __name__ == "__main__":
    args = parser_args()
    args.stream_name = cfg.STREAM_NAME.encode("utf-8")
    image_inference(args.pipeline_path, args.stream_name, args.infer_result_dir, args.batch_size)
