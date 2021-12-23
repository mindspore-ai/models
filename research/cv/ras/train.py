"""
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


import sys
import os
import time
import argparse
import numpy as np


import mindspore as ms
import mindspore.nn as nn
from mindspore import save_checkpoint
from mindspore.communication import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig

from src.model import BoneModel
from src.TrainOneStepMyself import TrainOneStep
from src.loss import LossFn, BceIouLoss, BuildTrainNetwork
from src.dataset_train import TrainDataLoader


sys.path.append("../")



# The data_url directory is the directory where the data set is located,
#and there must be two folders, images and labels, under data_url;

# If you are training on modelarts, there are two zip compressed files named after images and labels under data_url,
# and there are only these two files




parser = argparse.ArgumentParser()
parser.add_argument('--is_modelarts', type=str, default='NO')
parser.add_argument('--distribution_flag', type=str, default='YES', help='distirbution or not')
parser.add_argument('--device_target', type=str, default='Ascend', help="device's name, Ascend,GPU,CPU")
parser.add_argument('--device_id', type=int, default=5, help="Number of device")
parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
parser.add_argument('--batchsize', type=int, default=10, help="training batch size")
parser.add_argument('--decay_epoch', type=list, default=[27], help='every n epochs dacay learning rate')
parser.add_argument('--epoch', type=int, default=35, help='epoch number')
parser.add_argument('--print_flag', type=int, default=20, help='determines whether to print loss')
parser.add_argument('--data_url', type=str)
parser.add_argument('--pretrained_model', type=str)
parser.add_argument('--train_url', type=str)
parser.add_argument('--dataset_sink_mode', type=str, default='YES')
par = parser.parse_args()


class RASWhole:
    """
    Finally, the components of each part of the network are integrated, and the train method is added
    """
    def __init__(self, images_path_, labels_path_, pre_trained_model_path_):
        self.images_path = images_path_
        self.labels_path = labels_path_
        self.batchsize = par.batchsize
        self.df = par.distribution_flag
        self.pre_trained_model_path = pre_trained_model_path_

    def count_params_number(self, model_):
        count = 0
        for item in model_.trainable_params():
            shape = item.data.shape
            size = np.prod(shape)
            count += size
        return count

    def train(self):
        """

        Training process of Ras network

        """
        traindataloader = TrainDataLoader(self.images_path, self.labels_path, batch_size=self.batchsize, df=self.df)
        train_time = time.time()
        model = BoneModel(device_target, self.pre_trained_model_path)
        param_number = self.count_params_number(model)
        print("The number of the model parameters:{}".format(param_number))
        print("Data Number {}".format(traindataloader.data_number))
        loss_fn = LossFn()
        bce_iou_loss = BceIouLoss(par.batchsize)
        train_model = BuildTrainNetwork(model, loss_fn)

        if device_target == "GPU":
            if par.distribution_flag == "YES":
                rank_id = get_rank()
                par.decay_epoch = ["".join(par.decay_epoch)]
            else:
                rank_id = 0
            batch_num = traindataloader.dataset.get_dataset_size()
            if rank_id == 0:
                save_per_epoch = 5
                config_ck = CheckpointConfig(save_checkpoint_steps=save_per_epoch*batch_num,
                                             keep_checkpoint_max=30)
                ckpoint = ModelCheckpoint(prefix="RAS", directory=os.path.join(par.train_url, str(rank_id)),
                                          config=config_ck)
                callbacks = [TimeMonitor(), LossMonitor(), ckpoint]
            else:
                callbacks = [TimeMonitor(), LossMonitor()]
            piecewise = [int(x) * batch_num for x in par.decay_epoch]
            piecewise.append(par.epoch * batch_num)
            lr = nn.piecewise_constant_lr(piecewise, [par.lr, par.lr * 0.1])
            opt = nn.Adam(params=model.trainable_params(), learning_rate=lr, loss_scale=1024)
            model_train = Model(model, loss_fn=bce_iou_loss, optimizer=opt)
            model_train.train(epoch=par.epoch, train_dataset=traindataloader.dataset,
                              callbacks=callbacks, dataset_sink_mode=par.dataset_sink_mode == "YES")
        else:
            lr = par.lr
            opt = nn.Adam(params=model.trainable_params(), learning_rate=lr, loss_scale=1024)
            train_net = TrainOneStep(train_model, optimizer=opt)
            train_net.set_train()
            total_step = traindataloader.data_number
            for epoch in range(par.epoch):
                print("-----------------This Training is epoch %d --------------" % (epoch + 1))
                i = 1
                if epoch+1 in par.decay_epoch:
                    lr = lr * 0.1
                    opt = nn.Adam(params=model.trainable_params(), learning_rate=lr, loss_scale=1024)
                    train_net = TrainOneStep(train_model, optimizer=opt)
                    train_net.set_train()

                for data in traindataloader.dataset.create_dict_iterator():
                    image, label = data["data"], data["label"]
                    loss = train_net(image, label, par.batchsize)

                    if par.distribution_flag == "NO":
                        if (epoch > 10) and (loss > 0.5):
                            print("Please try once again")
                            return

                    if i % par.print_flag == 0 or i == total_step:
                        print("epoch:%d, learning_rate:%.8f,iter [%d/%d],Loss    ||  "
                              % ((epoch + 1), lr, i, total_step/self.batchsize), end='')
                        print(loss)
                        present_time = time.time()
                        mean_step_time = (present_time - train_time) / par.print_flag
                        print("The Consumption of per step is %.3f s" % mean_step_time)
                        train_time = present_time
                        print("+++++++++++++++++++++++++++++++++++++++++++++++++")
                    i += 1

                if par.distribution_flag == "YES":
                    rank_id = get_rank()
                    if (epoch + 1) % 10 == 0:
                        if par.is_modelarts == "YES":
                            save_path_all = os.path.join(checkpoint_out, "RAS%d" % (epoch + 1) + str(rank_id) + ".ckpt")
                        else:
                            save_path_all = os.path.join(save_path, "RAS%d" % (epoch + 1) + str(rank_id) + ".ckpt")
                        save_checkpoint(train_net, save_path_all)
                else:
                    if (epoch + 1) % 5 == 0:
                        if par.is_modelarts == "YES":
                            save_path_all = os.path.join(checkpoint_out, "RAS%d.ckpt" % (epoch + 1))
                        else:
                            save_path_all = os.path.join(save_path, "RAS%d.ckpt" % (epoch + 1))
                        save_checkpoint(train_net, save_path_all)

        if par.is_modelarts == "YES":
            mox.file.copy_parallel(src_url=checkpoint_out, dst_url=models_path)

if __name__ == "__main__":
    device_target = par.device_target
    np.random.seed(100)
    if par.distribution_flag == 'YES':
        print("++++++++++++++++++++   Training with distributed style  +++++++++++++++++++")
        if device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target=device_target, device_id=device_id)
            ms.context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                                 gradients_mean=True, device_num=device_num)
            init()
        else:
            init()
            ms.context.reset_auto_parallel_context()
            device_num = int(os.getenv('RANK_SIZE'))
            rank = get_rank()
            ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target=device_target)
            ms.context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                                 gradients_mean=True, device_num=device_num)
    else:
        if par.is_modelarts == "YES":
            device_id = int(os.getenv("DEVICE_ID"))
        else:
            device_id = int(par.device_id)
        ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target=device_target, device_id=device_id)
        if device_target == "GPU":
            ms.context.set_context(enable_graph_kernel=True)


    if par.is_modelarts == "YES":
        data_true_path = par.data_url
        pre_trained_true_path = par.pretrained_model
        models_path = par.train_url
        import moxing as mox

        checkpoint_out = '/cache/train_output/'
        local_data_path = '/cache/train/' + str(os.getenv("DEVICE_ID")) + "/"
        os.system("mkdir {0}".format(checkpoint_out))
        os.system("rm -rf {0}".format(local_data_path))
        os.system("mkdir -p {0}".format(local_data_path))
        image_name = 'images.zip'
        label_name = 'labels.zip'
        mox.file.copy_parallel(src_url=data_true_path, dst_url=local_data_path)
        mox.file.copy_parallel(src_url=pre_trained_true_path, dst_url=local_data_path)
        zip_command1 = "unzip -o -q %s -d %s" % (local_data_path + image_name, local_data_path)
        zip_command2 = "unzip -o -q %s -d %s" % (local_data_path + label_name, local_data_path)
        os.system(zip_command1)
        print("unzip images success!")
        os.system(zip_command2)
        print("unzip labels success!")

        images_path = os.path.join(local_data_path, "images/")
        labels_path = os.path.join(local_data_path, "labels/")
        pre_trained_model_path = os.path.join(local_data_path, pre_trained_true_path.split("/")[-1])
    else:
        images_path = os.path.join(par.data_url, "images/")
        labels_path = os.path.join(par.data_url, "labels/")
        pre_trained_model_path = par.pretrained_model
        save_models_path = par.train_url
        save_path = save_models_path
        if device_target == "GPU" and par.distribution_flag == "YES":
            save_path = os.path.join(save_path, str(rank))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    start_time = time.time()
    model_ras = RASWhole(images_path, labels_path, pre_trained_model_path)
    model_ras.train()
    end_time = time.time()
    total_time = end_time - start_time
    print("Total training time is %.3f s" % total_time)
