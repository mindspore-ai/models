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
"""Predict Retinaface_resnet50_or_mobilenet0.25."""
import argparse
import time

import numpy as np
import cv2

import mindspore as ms
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from src.config import cfg_res50
from src.utils import prior_box
from src.network_with_resnet import RetinaFace, resnet50
from eval import DetectionEngine


def read_image(img_path):
    """Read image."""
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    h_max, w_max = 0, 0
    if img.shape[0] > h_max:
        h_max = img.shape[0]
    if img.shape[1] > w_max:
        w_max = img.shape[1]

    h_max = (int(h_max / 32) + 1) * 32
    w_max = (int(w_max / 32) + 1) * 32

    priors = prior_box(image_sizes=(h_max, w_max),
                       min_sizes=[[16, 32], [64, 128], [256, 512]],
                       steps=[8, 16, 32],
                       clip=False)

    resize = 1
    assert img.shape[0] <= h_max and img.shape[1] <= w_max
    image_t = np.empty((h_max, w_max, 3), dtype=img.dtype)
    image_t[:, :] = (104.0, 117.0, 123.0)
    image_t[0:img.shape[0], 0:img.shape[1]] = img
    img = image_t

    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = Tensor(img)
    return img, (resize, scale, priors)


def detect(boxes, confs, info):
    """Detect after infer."""
    resize, scale, priors = info
    img_name = "test/test.png"
    # init detection engine
    detection = DetectionEngine(cfg_res50)
    start_time = time.time()
    boxes = detection.detect(boxes, confs, resize, scale, img_name, priors)
    print(f'detection time: {time.time() - start_time}')
    return boxes


def create_model():
    """
    create model.
    """
    backbone = resnet50(1001)
    network = RetinaFace(phase='predict', backbone=backbone)
    backbone.set_train(False)
    network.set_train(False)

    # load checkpoint
    network.init_parameters_data()
    if args.ckpt_file:
        param_dict = load_checkpoint(args.ckpt_file)
        print('Load trained model done. {}'.format(args.ckpt_file))
        load_param_into_net(network, param_dict)
    ms_model = Model(network)
    return ms_model


def predict(img_input, info):
    """
    ms.Model.predict
    """
    ms_model = create_model()

    # model predict
    boxes, confs, _ = ms_model.predict(img_input)
    t_start = time.time()
    for _ in range(100):
        ms_model.predict(img_input)
    avg_time = (time.time() - t_start) / 100

    final_boxes = detect(boxes, confs, info)
    return final_boxes, avg_time


def predict_backend_lite(data_input, info):
    """
    model.predict using backend lite.
    """
    # model predict using backend lite
    ms_model = create_model()
    boxes, confs, _ = ms_model.predict(data_input, backend="lite")
    t_start = time.time()
    for _ in range(100):
        ms_model.predict(data_input, backend="lite")
    avg_time = (time.time() - t_start) / 100

    final_boxes = detect(boxes, confs, info)
    return final_boxes, avg_time


def predict_mindir(data_input, info, mindir_path):
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
        outputs = [Tensor(output.get_data_to_numpy()) for output in outputs]
        return outputs

    def _get_lite_context(l_context):
        lite_context_properties = {
            "cpu": ["inter_op_parallel_num", "precision_mode", "thread_num",
                    "thread_affinity_mode", "thread_affinity_core_list"],
            "gpu": ["device_id", "precision_mode"],
            "ascend": ["device_id", "precision_mode", "provider", "rank_id"]
        }
        lite_device_target = ms.get_context('device_target').lower()
        if lite_device_target not in ['cpu', 'gpu', 'ascend']:
            raise RuntimeError(f"Device target should be in ['cpu', 'gpu', 'ascend'], but got {lite_device_target}")
        l_context.target = [lite_device_target]
        l_context_device_dict = {'cpu': l_context.cpu, 'gpu': l_context.gpu, 'ascend': l_context.ascend}
        for single_property in lite_context_properties.get(lite_device_target):
            try:
                context_value = ms.get_context(single_property)
                if context_value:
                    setattr(l_context_device_dict.get(lite_device_target), single_property, context_value)
            except ValueError:
                print(f'For set lite context, fail to get parameter {single_property} from ms.context.'
                      f' Will use default value')
        return l_context

    try:
        import mindspore_lite as mslite
    except ImportError:
        raise ImportError(f"For predict by MindIR, mindspore_lite should be installed.")

    lite_context = mslite.Context()
    lite_context = _get_lite_context(lite_context)

    lite_model = mslite.Model()
    lite_model.build_from_file(mindir_path, mslite.ModelType.MINDIR, lite_context)

    boxes, confs, _ = _predict_core(lite_model)
    t_start = time.time()
    for _ in range(100):
        _predict_core(lite_model)
    avg_time = (time.time() - t_start) / 100

    final_boxes = detect(boxes, confs, info)
    return final_boxes, avg_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--ckpt_file', type=str, default='./train_parallel3/checkpoint/ckpt_3/RetinaFace-56_201.ckpt',
                        help='ckpt location')
    parser.add_argument('--img_path', type=str, default='./test.png', help='image location')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU', save_graphs=False)

    image_input, img_info = read_image(args.img_path)
    res_boxes, res_avg_t = predict(image_input, img_info)
    print("Prediction res: ", res_boxes)
    print(f"Prediction avg time: {res_avg_t * 1000} ms")
