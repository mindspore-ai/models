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
"""quick start"""
# ## This paper mainly visualizes the prediction data, uses the model to predict, and visualizes the prediction results.

import argparse
import numpy as np
import matplotlib.pyplot as plt

from mindspore import nn
from mindspore import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import create_dataset
from src.ghostnet import ghostnet_1x, ghostnet_nose_1x
from src.ghostnet600 import ghostnet_600m

parser = argparse.ArgumentParser(description='Quick start')
parser.add_argument('--num_classes', type=int, default=5, help='Classes number')
parser.add_argument('--ckpt_path', type=str, default='./ckpt/ghostnet_1x_2-500_21.ckpt', help='Checkpoint file path')
parser.add_argument('--data_path', type=str, default='./dataset/', help='Dataset path')
args_opt = parser.parse_args()

# class_name corresponds to label,and labels are marked in the order of folders
class_name = {0: "daisy", 1: "dandelion", 2: "roses", 3: "sunflowers", 4: "tulips"}

# define visual prediction data functions：
def visual_input_data(val_dataset):
    data = next(val_dataset.create_dict_iterator())
    images = data["image"]
    labels = data["label"]
    print("Tensor of image", images.shape)
    print("Labels:", labels)
    plt.figure(figsize=(15, 7))
    for i in range(len(labels)):
        # get the image and its corresponding label
        data_image = images[i].asnumpy()
        # data_label = labels[i]
        # process images for display
        data_image = np.transpose(data_image, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        data_image = std * data_image + mean
        data_image = np.clip(data_image, 0, 1)
        # display image
        plt.subplot(4, 8, i+1)
        plt.imshow(data_image)
        plt.title(class_name[int(labels[i].asnumpy())], fontsize=10)
        plt.axis("off")

    plt.show()

# define visualize_model()，visualize model prediction
def visualize_model(ckpt_path, val_ds):
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    if 'ghostnet_1x' in args_opt.ckpt_path:
        net = ghostnet_1x(num_classes=args_opt.num_classes)
    elif 'ghostnet_nose_1x' in args_opt.ckpt_path:
        net = ghostnet_nose_1x(num_classes=args_opt.num_classes)
    elif 'ghostnet600M' in args_opt.ckpt_path:
        net = ghostnet_600m(num_classes=args_opt.num_classes)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
    data = next(val_ds.create_dict_iterator())
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()
    flower_class_name = {0: "daisy", 1: "dandelion", 2: "roses", 3: "sunflowers", 4: "tulips"}
    # prediction image category
    output = model.predict(Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)

    # display the image and the predicted value of the image
    plt.figure(figsize=(15, 7))
    for i in range(len(labels)):
        plt.subplot(4, 8, i + 1)
        # if the prediction is correct, it is displayed in blue; if the prediction is wrong, it is displayed in red
        color = 'blue' if pred[i] == labels[i] else 'red'
        plt.title('predict:{}'.format(flower_class_name[pred[i]]), color=color)
        picture_show = np.transpose(images[i], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        picture_show = std * picture_show + mean
        picture_show = np.clip(picture_show, 0, 1)
        plt.imshow(picture_show)
        plt.axis('off')

    plt.show()

if __name__ == '__main__':
    ds = create_dataset(dataset_path=args_opt.data_path, do_train=False, batch_size=32, num_parallel_workers=None)
    visual_input_data(ds)
    visualize_model(args_opt.ckpt_path, ds)
