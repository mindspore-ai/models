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

import mindspore as ms
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
from model_utils.config import get_config
from src.vgg import Vgg
from src.dataset import create_dataset

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU", save_graphs=False)
ms.set_seed(21)


def import_data(train_dataset_path="./datasets/train/", eval_dataset_path="./datasets/test/", batch_size=32):
    """
        Read the dataset

        Args:
            train_dataset_path(string): the path of train dataset.
            eval_dataset_path(string): the path of eval dataset.
            batch_size(int): the batch size of dataset. Default: 32

        Returns:
            dataset_train: the train dataset
            dataset_val:   the  val  dataset
    """

    dataset_train = create_dataset(dataset_path=train_dataset_path, do_train=True,
                                   batch_size=batch_size, train_image_size=224,
                                   eval_image_size=224,
                                   enable_cache=False, cache_session_id=None)
    dataset_val = create_dataset(dataset_path=eval_dataset_path, do_train=False,
                                 batch_size=batch_size, train_image_size=224,
                                 eval_image_size=224,
                                 enable_cache=False, cache_session_id=None)
    # print sample data/label
    data = next(dataset_train.create_dict_iterator())
    images = data["image"]
    labels = data["label"]
    print("Tensor of image", images.shape)  # Tensor of image (18, 3, 224, 224)
    print("Labels:", labels)  # Labels: [1 0 0 0 1 1 1 1 0 0 1 1 1 0 1 0 0 0]

    return dataset_train, dataset_val


# define head layer
class DenseHead(nn.Cell):
    def __init__(self, input_channel, num_classes):
        super(DenseHead, self).__init__()
        self.dense = nn.Dense(input_channel, num_classes)

    def construct(self, x):
        return self.dense(x)


def init_weight(net, param_dict):
    """init_weight"""

    # if config.pre_trained:
    has_trained_epoch = 0
    has_trained_step = 0
    if param_dict:
        if param_dict.get("epoch_num") and param_dict.get("step_num"):
            has_trained_epoch = int(param_dict["epoch_num"].data.asnumpy())
            has_trained_step = int(param_dict["step_num"].data.asnumpy())

        ms.load_param_into_net(net, param_dict)
    print("has_trained_epoch:", has_trained_epoch)
    print("has_trained_step:", has_trained_step)
    return has_trained_epoch, has_trained_step


def eval_net(model_config, checkpoint_path='./vgg16.ckpt',
             train_dataset_path="./datasets/train/",
             eval_dataset_path="./datasets/test/",
             batch_size=32):
    """
      eval the accuracy of vgg16 for flower dataset

      Args:

          model_config(Config in './model_utils/config.py'): vgg16 config
          checkpoint_path(string): model checkout path(end with '.ckpt'). Default: './vgg16.ckpt'
          train_dataset_path: the train dataset path. Default: "./datasets/train/"
          eval_dataset_path:  the eval dataset path.  Default: "./datasets/test/"
          batch_size: the batch size of dataset. Default: 32
      Returns:
          None
      """

    # define val dataset and model
    _, data_val = import_data(train_dataset_path=train_dataset_path,
                              eval_dataset_path=eval_dataset_path, batch_size=batch_size)
    net = Vgg(cfg['16'], num_classes=1000, args=model_config, batch_norm=True)

    # replace head
    src_head = net.classifier[6]
    in_channels = src_head.in_channels
    head = DenseHead(in_channels, 5)
    net.classifier[6] = head

    # load checkpoint
    param_dict = ms.load_checkpoint(checkpoint_path)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss
    from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = ms.Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval step
    res = model.eval(data_val)

    # show accuracy
    print("result:", res)


cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def finetune_train(model_config,
                   finetune_checkpoint_path=
                   './vgg16_bn_ascend_v170_imagenet2012_official_cv_top1acc74.33_top5acc92.1.ckpt',
                   save_checkpoint_path="./vgg16.ckpt",
                   train_dataset_path="./datasets/train/",
                   eval_dataset_path="./datasets/test/",
                   class_num=5,
                   num_epochs=10,
                   learning_rate=0.001,
                   momentum=0.9,
                   batch_size=32
                   ):
    """
         finetune the flower dataset for vgg16

         Args:
             model_config(Config in './model_utils/config.py'): vgg16 config
             class_num(int): the num of class for dataset. Default: 5
             num_epochs(int): the training epoch. Default: 10
             save_checkpoint_path(string): model checkout path for save(end with '.ckpt'). Default: ./vgg16.ckpt
             train_dataset_path(string): the train dataset path. Default: "./datasets/train/"
             eval_dataset_path(string):  the eval dataset path.  Default: "./datasets/test/"
             finetune_checkpoint_path(string): model checkout path for initialize
                       Default: ./vgg16_bn_ascend_v170_imagenet2012_official_cv_top1acc74.33_top5acc92.1.ckpt
             learning_rate: the finetune learning rate
             momentum: the finetune momentum
             batch_size: the batch size of dataset. Default: 32
         Returns:
             None
    """

    # read train/val dataset
    dataset_train, _ = import_data(train_dataset_path=train_dataset_path,
                                   eval_dataset_path=eval_dataset_path,
                                   batch_size=batch_size)

    ckpt_param_dict = ms.load_checkpoint(finetune_checkpoint_path)
    net = Vgg(cfg['16'], num_classes=1000, args=model_config, batch_norm=True)
    init_weight(net=net, param_dict=ckpt_param_dict)
    print("net parameter:")
    for param in net.get_parameters():
        print("param:", param)

    # replace head
    src_head = net.classifier[6]
    print("classifier.6.bias:", net.classifier[6])
    in_channels = src_head.in_channels
    head = DenseHead(in_channels, class_num)
    net.classifier[6] = head

    # freeze the param except last layer
    for param in net.get_parameters():
        if param.name not in ["classifier.6.dense.weight", "classifier.6.dense.bias"]:
            param.requires_grad = False

    # define optimizer and loss
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=learning_rate, momentum=momentum)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = Model(net, loss, opt, metrics={"Accuracy": nn.Accuracy()})

    # define callbacks
    batch_num = dataset_train.get_dataset_size()
    time_cb = TimeMonitor(data_size=batch_num)
    loss_cb = LossMonitor()
    callbacks = [time_cb, loss_cb]

    # do training
    model.train(num_epochs, dataset_train, callbacks=callbacks, dataset_sink_mode=True)
    ms.save_checkpoint(net, save_checkpoint_path)


if __name__ == '__main__':
    config = get_config()
    print("config:", config)
    # finetune
    finetune_train(config,
                   finetune_checkpoint_path=config.ckpt_file,
                   save_checkpoint_path=config.save_file, train_dataset_path=config.train_path,
                   eval_dataset_path=config.eval_path, num_epochs=config.num_epochs, class_num=config.num_classes,
                   learning_rate=config.lr,
                   momentum=config.momentum,
                   batch_size=config.batch_size
                   )

    # eval
    eval_net(config, checkpoint_path=config.save_file, train_dataset_path=config.train_path,
             eval_dataset_path=config.eval_path,
             batch_size=config.batch_size)  # 0.8505434782608695
