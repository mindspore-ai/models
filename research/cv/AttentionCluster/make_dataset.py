# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Train Attention Cluster"""
import os
import numpy as np
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision as c_trans
import mindspore.nn as nn
import mindspore.context as context
import mindspore.common as common
import mindspore.train.callback as callback

from mindspore.train.model import Model
from src.datasets.mnist_noisy import MNISTNoisy
from src.utils.config import config as cfg
from src.datasets.mnist_sampler import dump_pkl
from src.datasets.mnist_flash import MNISTFlash
from src.datasets.mnist_feature import MNISTFeature


class Extractor(nn.Cell):
    """Feature Extractor"""
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, pad_mode='valid')
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, pad_mode='valid')
        self.conv2_drop = nn.Dropout()
        self.fc1 = nn.Dense(320, 50)
        self.relu = mindspore.ops.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = mindspore.nn.Dropout()
        self.log_softmax = mindspore.nn.LogSoftmax()

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.max_pool2d(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.max_pool2d(x)
        x = self.relu(x)

        x = x.reshape(-1, 320)
        x = self.relu(self.fc1(x))
        if self.training:
            x = self.dropout(x)
        return self.log_softmax(x)


class Net1(nn.Cell):
    """Net for training feature extractor"""
    def __init__(self):
        super(Net1, self).__init__()
        self.extractor = Extractor()
        self.fc2 = nn.Dense(50, 11)
        self.log_softmax = mindspore.nn.LogSoftmax()

    def construct(self, x):
        """construct"""
        x = self.extractor(x)
        x = self.fc2(x)
        return self.log_softmax(x)


class Net2(nn.Cell):
    """Net to extract feature"""
    def __init__(self):
        super(Net2, self).__init__()
        self.extractor = Extractor()

    def construct(self, x):
        """construct"""
        batch_size = x.shape[0]
        length = x.shape[1]
        flatx = x.reshape(batch_size * length, x.shape[2], x.shape[3], x.shape[4])
        flatx = self.extractor(flatx)
        x = flatx.reshape(batch_size, length, -1)
        return x


class VideoWrap:
    """transform imgs to video"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, video):
        res = []
        for img in video:
            timg = self.transform(img)
            timg = np.expand_dims(timg, 0)
            res.append(timg)
        res = np.stack(res)
        return res


def extract(net, loader, path):
    """Extract Feature"""
    feats = []
    labels = []
    indexes = []
    length = loader.get_dataset_size()
    for i, item in enumerate(loader.create_dict_iterator()):
        if i % 1 == 0:
            print('%d/%d' % (i, length))

        data, target = item['images'], item['target']
        output = net(data)
        for f, t in zip(output.asnumpy(), target.asnumpy()):
            feats.append(f)
            labels.append(t)
            indexes.append(0)

    dump_pkl((feats, labels, indexes), path)


if __name__ == '__main__':
    # Training settings
    if not os.path.exists(cfg.result_dir):
        os.makedirs(cfg.result_dir)

    # init context
    common.set_seed(cfg.seed)
    np.random.seed(1234)
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=cfg.device,
                        device_id=cfg.device_id)

    # train feature extractor
    # create dataset
    train_dataset_generator = MNISTNoisy(root=cfg.data_dir, train=True, generate=True,
                                         transform=c_trans.Normalize(mean=(0.1307,), std=(0.3081,)))
    train_dataset = ds.GeneratorDataset(source=train_dataset_generator, column_names=["image", "target"], shuffle=True)
    train_dataset = train_dataset.map(operations=[lambda x: np.expand_dims(x, 0)], input_columns=["image"])
    train_dataset = train_dataset.batch(64, drop_remainder=True)

    #define net
    net1 = Net1()

    # define loss
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    # define optimizer
    optimizer = nn.Adam(params=net1.trainable_params(), learning_rate=0.001)

    # define model
    model = Model(network=net1, loss_fn=loss, optimizer=optimizer,
                  metrics={'top_1_accuracy': nn.Top1CategoricalAccuracy(),
                           'top_5_accuracy': nn.Top5CategoricalAccuracy()}
                  )

    # define callback
    step_size = train_dataset.get_dataset_size()
    time_cb = callback.TimeMonitor(data_size=step_size)
    loss_cb = callback.LossMonitor()
    config_ck = callback.CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=2)
    ckpt_cb = callback.ModelCheckpoint(prefix='extractor', directory=cfg.result_dir, config=config_ck)
    cb = [time_cb, loss_cb, ckpt_cb]

    # train
    if not os.path.isfile(os.path.join(cfg.result_dir, 'extractor-10_5156.ckpt')):
        print("===> Training feature extractor")
        model.train(epoch=10, train_dataset=train_dataset, callbacks=cb, dataset_sink_mode=False)
    else:
        print("===> Found existing extractor checkpoint")

    # extractor feature
    # create dataset
    train_dataset_generator = MNISTFlash(cfg.data_dir, train=True, generate=True,
                                         transform=VideoWrap(transform=c_trans.Normalize(mean=(0.1307,), std=(0.3081,)))
                                         )
    train_dataset = ds.GeneratorDataset(source=train_dataset_generator, column_names=["images", "target"],
                                        shuffle=False)
    train_dataset = train_dataset.batch(1024, drop_remainder=True)

    test_dataset_generator = MNISTFlash(cfg.data_dir, train=False,
                                        transform=VideoWrap(transform=c_trans.Normalize(mean=(0.1307,), std=(0.3081,)))
                                        )
    test_dataset = ds.GeneratorDataset(source=test_dataset_generator, column_names=["images", "target"],
                                       shuffle=False)
    test_dataset = test_dataset.batch(1024, drop_remainder=True)

    # define net
    net2 = Net2()

    # load checkpoint
    param_dict = mindspore.load_checkpoint(os.path.join(cfg.result_dir, 'extractor-10_5156.ckpt'))
    mindspore.load_param_into_net(net2.extractor, param_dict)
    net2.set_train(False)

    extract(net2, train_dataset, os.path.join(cfg.data_dir, MNISTFeature.training_file))
    extract(net2, test_dataset, os.path.join(cfg.data_dir, MNISTFeature.test_file))
