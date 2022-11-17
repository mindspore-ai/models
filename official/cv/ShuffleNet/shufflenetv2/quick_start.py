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

import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.shufflenetv2 import ShuffleNetV2
from src.config import config_cpu
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor
from mindspore import dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C
import mindspore.common.dtype as mstype
from mindspore.train import Model
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image classification transformation quick_start')
    parser.add_argument('--quick_start_ckpt', type=str, default='./save_ckpt/Graph_mode/shufflenetv2-204_22.ckpt',
                        help='the checkpoint of ShuffleNetV2 (Default: None)')
    args_opt = parser.parse_args()
    network = ShuffleNetV2(model_size=config_cpu.model_size, n_class=config_cpu.num_classes)
    if args_opt.quick_start_ckpt:
        ckpt = load_checkpoint(args_opt.quick_start_ckpt)
        load_param_into_net(network, ckpt)
        print(args_opt.quick_start_ckpt, ' is loaded')

    #
    # # define loss
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    #
    # # define opt
    net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)

    model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'accuracy'})

    #load test data
    test_loader = ds.ImageFolderDataset(config_cpu.eval_dataset_path, shuffle=True)
    trans = [
        C.RandomCropDecodeResize(224),
        C.RandomHorizontalFlip(prob=0.5),
        C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
        C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
        C.HWC2CHW()
    ]
    type_cast_op = C2.TypeCast(mstype.int32)
    test_loader = test_loader.map(input_columns="image",
                                  operations=trans,
                                  num_parallel_workers=config_cpu.num_parallel_workers,
                                  python_multiprocessing=True)
    test_loader = test_loader.map(input_columns="label",
                                  operations=type_cast_op,
                                  num_parallel_workers=config_cpu.num_parallel_workers,
                                  python_multiprocessing=True)

    test_loader = test_loader.create_dict_iterator()
    images = []
    labels = []
    plt.figure()
    for i in range(1, 7):
        data = next(test_loader)
        image = data["image"].asnumpy()
        label = data["label"].asnumpy()
        plt.subplot(2, 3, i)
        plt.imshow(image.T)
        images.append(image)
        labels.append(int(label))
    plt.show()


    # # user model.predict to get the classification of the input image
    output = model.predict(Tensor(images))
    predicted = np.argmax(output.asnumpy(), axis=1)
    #
    # # output the actual classification and the predicted classifiaction
    print(f'Predicted: "{predicted}", Actual: "{labels}"')
