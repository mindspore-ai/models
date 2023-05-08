# Copyright 2023 Huawei Technologies Co., Ltd
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
######################## predict lenet example ########################

Note:
    To run this scripts, 'mindspore' and 'mindspore_lite' must be installed.

Usage:
    python predict.py --config_path [Your config path]
"""
import time

import numpy as np
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from src.model_utils.config import config
from src.lenet import LeNet5


def create_model():
    """
    create model.
    """
    network = LeNet5(config.num_classes, num_channel=3)
    if config.ckpt_file:
        param_dict = load_checkpoint(config.ckpt_file)
        load_param_into_net(network, param_dict)
    ms_model = Model(network)
    return ms_model


def read_image(img_path):
    img = Image.open(img_path).convert("RGB")
    mean = [0.475 * 225, 0.451 * 225, 0.392 * 225]
    std = [0.275 * 225, 0.267 * 225, 0.278 * 225]
    transform_list = [ds.vision.Resize((config.image_width, config.image_height)),
                      ds.vision.Normalize(mean, std),
                      ds.vision.HWC2CHW()]
    for transform in transform_list:
        img = transform(img)
    img = ms.Tensor(np.expand_dims(img, axis=0), ms.float32)
    return img


def predict_backend_lite(data_input):
    """
    model.predict using backend lite.
    """
    # model predict using backend lite
    ms_model = create_model()
    ms.set_context(lite_context=lite_context_config)
    output = ms_model.predict(data_input, backend="lite")
    t_start = time.time()
    for _ in range(100):
        ms_model.predict(data_input, backend="lite")
    avg_time = (time.time() - t_start) / 100
    return output, avg_time


def predict_mindir(data_input):
    """
    predict by MindIR.
    """
    def _predict_core(lite_mode_input):
        """single input."""
        inputs = lite_mode_input.get_inputs()
        if len(inputs) > 1:
            raise RuntimeError("Only support single input in this net.")
        inputs[0].set_data_from_numpy(data_input.asnumpy())
        outputs = lite_mode_input.predict(inputs)
        return ms.Tensor(outputs[0].get_data_to_numpy())

    def _get_lite_context(l_context):
        lite_context_properties = {
            "cpu": ["inter_op_parallel_num", "precision_mode", "thread_num",
                    "thread_affinity_mode", "thread_affinity_core_list"],
            "gpu": ["device_id", "precision_mode"],
            "ascend": ["device_id", "precision_mode", "provider", "rank_id"]
        }
        lite_device_target = config.lite_context_config.get("target")
        l_context.target = [lite_device_target]
        if lite_device_target == 'cpu':
            for single_property in lite_context_properties.get('cpu'):
                context_value = config.lite_context_config.get(single_property)
                if context_value:
                    setattr(l_context.cpu, single_property, context_value)
        elif lite_device_target == 'gpu':
            for single_property in lite_context_properties.get('gpu'):
                context_value = config.lite_context_config.get(single_property)
                if context_value:
                    setattr(l_context.gpu, single_property, context_value)
        elif lite_device_target == 'ascend':
            for single_property in lite_context_properties.get('ascend'):
                context_value = config.lite_context_config.get(single_property)
                if context_value:
                    setattr(l_context.ascend, single_property, context_value)
        else:
            raise RuntimeError(f"For set Lite Context, target should be in ['cpu', 'gpu', 'ascend'], "
                               f"but got {lite_device_target}")
        return l_context

    try:
        import mindspore_lite as mslite
    except ImportError:
        raise ImportError(f"For predict by MindIR, mindspore_lite should be installed.")

    lite_context = mslite.Context()
    lite_context = _get_lite_context(lite_context)

    ms_model = create_model()
    ms.export(ms_model.predict_network, data_input, file_name="net", file_format="MINDIR")

    lite_model = mslite.Model()
    lite_model.build_from_file("net.mindir", mslite.ModelType.MINDIR, lite_context)

    output = _predict_core(lite_model)
    t_start = time.time()
    for _ in range(100):
        _predict_core(lite_model)
    avg_time = (time.time() - t_start) / 100
    return output, avg_time


def predict_lenet(data_input):
    """
    ms.Model.predict
    """
    print('predict with config: ', config)
    ms_model = create_model()
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    # model predict
    output = ms_model.predict(data_input)
    t_start = time.time()
    for _ in range(100):
        ms_model.predict(data_input)
    avg_time = (time.time() - t_start) / 100
    return output, avg_time


def _get_max_index_from_res(data_input):
    data_input = data_input.asnumpy()
    data_input = data_input.flatten()
    res_index = np.where(data_input == np.max(data_input))  # (array([6]), )
    return res_index[0][0]


if __name__ == "__main__":
    image_input = read_image(config.img_path)
    res, avg_t = predict_lenet(image_input)
    print("Prediction res: ", _get_max_index_from_res(res))
    print(f"Prediction avg time: {avg_t * 1000} ms")

    if config.enable_predict_lite_backend:
        res_lite, avg_t_lite = predict_backend_lite(image_input)
        print("Predict using backend lite, res: ", _get_max_index_from_res(res_lite))
        print(f"Predict using backend lite, avg time: {avg_t_lite * 1000} ms")

    if config.enable_predict_lite_mindir:
        res_mindir, avg_t_mindir = predict_mindir(image_input)
        print("Predict by mindir, res: ", _get_max_index_from_res(res_mindir))
        print(f"Predict by mindir, avg time: {avg_t_mindir * 1000} ms")
