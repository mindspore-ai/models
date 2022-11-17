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
"""inference for CPU"""
import matplotlib.pyplot as plt
import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net, nn
from mindspore.train import Model
from fine_tune import import_data
from model_utils.moxing_adapter import config
from src.vgg import Vgg

# class_name for dataset
class_name = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# define head layer
class DenseHead(nn.Cell):
    def __init__(self, input_channel, num_classes):
        super(DenseHead, self).__init__()
        self.dense = nn.Dense(input_channel, num_classes)

    def construct(self, x):
        return self.dense(x)


def visualize_model(best_ckpt_path, val_ds, num_classes):
    """
             visualize model

             Args:
                 val_ds: eval dataset
                 best_ckpt_path(string): the .ckpt file for model to infer
                 num_classes(int): the class num

             Returns:
                 None
        """

    net = Vgg(cfg['16'], num_classes=1000, args=config, batch_norm=True)

    # replace head
    src_head = net.classifier[6]
    in_channels = src_head.in_channels
    head = DenseHead(in_channels, num_classes)
    net.classifier[6] = head

    # load param
    param_dict = load_checkpoint(best_ckpt_path)
    load_param_into_net(net, param_dict)

    net.set_train(False)
    model = Model(net)

    # load some image in eval dataset for prediction
    for i in range(5):
        next(val_ds.create_dict_iterator())
    data = next(val_ds.create_dict_iterator())
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()

    output = model.predict(Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)
    print("\nAccuracy:", (pred == labels).sum() / len(labels))

    # show image
    plt.figure(figsize=(15, 7))
    for i in range(len(labels)):
        plt.subplot(4, 8, i + 1)
        # show blue color if right，otherwise show red color
        color = 'blue' if pred[i] == labels[i] else 'red'
        plt.title('predict:{}'.format(class_name[pred[i]]), color=color)
        picture_show = np.transpose(images[i], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        picture_show = std * picture_show + mean
        picture_show = np.clip(picture_show, 0, 1)
        plt.imshow(picture_show)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    _, dataset_val = import_data(train_dataset_path=config.train_path, eval_dataset_path=config.eval_path)

    visualize_model(config.pre_trained, dataset_val, config.num_classes)
