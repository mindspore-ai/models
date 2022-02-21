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
Creates a MindSpore Inception checkpoint for metrics evaluation
"""
import sys

import mindspore as ms
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

tfgan = tf.contrib.gan

session = tf.InteractiveSession()


def inception_activations(images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    for elem in generated_images_list:
        activations = tfgan.eval.run_inception(elem, output_tensor='pool_3:0')
        return activations


def create_inception_checkpoint(ckpt_path):
    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])

    inception_activations(inception_images)

    op_list = []
    for op in tf.get_default_graph().get_operations():
        if op.type == 'Const':
            name = op.name.replace('/', '.')

            replace_dict = {
                'RunClassifier.': '',
                'conv2d_params': 'conv.weight',
                'join': 'concat',
                'tower.conv': 'tower_conv',
                'tower_1.conv': 'tower_1_conv',
                'tower_2.conv': 'tower_2_conv',
                '.mixed.conv': '_mixed_conv'
            }

            for substring in replace_dict:
                name = name.replace(substring, replace_dict[substring])

            if name.find('DecodeJpeg') != -1:
                continue

            np_data = op.outputs[0].eval()

            if name.endswith('weight'):
                np_data = np.transpose(np_data, (3, 2, 0, 1))

            op_list.append({'name': name, 'data': ms.Tensor(np_data)})

    ms.save_checkpoint(op_list, ckpt_path)


if __name__ == '__main__':
    create_inception_checkpoint(sys.argv[1])
