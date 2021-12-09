# Copyright 2021 Huawei Technologies Co., Ltd
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
modelarts train
"""
import os
import time
import numpy as np

from mindspore import Tensor
import mindspore.context as context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import save_checkpoint
from mindspore.train.serialization import export, load_checkpoint
from mindspore.common import set_seed

from src.bgcf import BGCF
from src.utils import convert_item_id
from src.callback import TrainBGCF
from src.dataset import load_graph, create_dataset
from src.callback_modelarts import ForwardBGCF

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

last_epoch_name = ""
set_seed(1)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(
                            int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(
            config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(
            get_device_id(), zip_file_1, save_dir_1))

    config.ckptpath = os.path.join(config.output_path, config.ckptpath)
    if not os.path.isdir(config.ckptpath):
        os.makedirs(config.ckptpath)

def run_export():
    '''run export.'''
    global last_epoch_name

    last_epoch_name = os.path.join(config.ckptpath, last_epoch_name)
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=config.device_id)

    num_user, num_item = 7068, 3570

    network = BGCF([config.input_dim, num_user, num_item],
                   config.embedded_dimension,
                   config.activation,
                   [0.0, 0.0, 0.0],
                   num_user,
                   num_item,
                   config.input_dim)
    print(os.listdir(config.ckptpath))
    load_checkpoint(last_epoch_name, net=network)

    forward_net = ForwardBGCF(network)

    users = Tensor(np.zeros([1, num_user]).astype(np.int32))
    items = Tensor(np.zeros([1, num_item]).astype(np.int32))
    neg_items = Tensor(np.zeros([1, num_item]).astype(np.int32))
    u_test_neighs = Tensor(
        np.zeros([1, num_user*config.row_neighs]).astype(np.int32))
    u_test_gnew_neighs = Tensor(
        np.zeros([1, num_user * config.gnew_neighs]).astype(np.int32))
    i_test_neighs = Tensor(
        np.zeros([1, num_item*config.row_neighs]).astype(np.int32))
    i_test_gnew_neighs = Tensor(
        np.zeros([1, num_item * config.gnew_neighs]).astype(np.int32))

    input_data = [users, items, neg_items, u_test_neighs,
                  u_test_gnew_neighs, i_test_neighs, i_test_gnew_neighs]
    export(forward_net, *input_data, file_name=os.path.join(config.ckptpath, config.file_name),
           file_format=config.file_format)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """Train"""
    global last_epoch_name
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target,
                        save_graphs=False)

    if config.device_target == "Ascend":
        context.set_context(device_id=get_device_id())
    if config.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)

    print(os.listdir('/cache/data/'))
    train_graph, _, sampled_graph_list = load_graph(config.datapath)
    train_ds = create_dataset(train_graph, sampled_graph_list, config.workers, batch_size=config.batch_pairs,
                              num_samples=config.raw_neighs, num_bgcn_neigh=config.gnew_neighs, num_neg=config.num_neg)

    num_user = train_graph.graph_info()["node_num"][0]
    num_item = train_graph.graph_info()["node_num"][1]
    num_pairs = train_graph.graph_info()['edge_num'][0]

    bgcfnet = BGCF([config.input_dim, num_user, num_item],
                   config.embedded_dimension,
                   config.activation,
                   config.neighbor_dropout,
                   num_user,
                   num_item,
                   config.input_dim)

    train_net = TrainBGCF(bgcfnet, config.num_neg, config.l2, config.learning_rate,
                          config.epsilon, config.dist_reg)
    train_net.set_train(True)

    itr = train_ds.create_dict_iterator(config.num_epoch, output_numpy=True)
    num_iter = int(num_pairs / config.batch_pairs)

    for _epoch in range(1, config.num_epoch + 1):

        epoch_start = time.time()
        iter_num = 1

        for data in itr:

            u_id = Tensor(data["users"], mstype.int32)
            pos_item_id = Tensor(convert_item_id(
                data["items"], num_user), mstype.int32)
            neg_item_id = Tensor(convert_item_id(
                data["neg_item_id"], num_user), mstype.int32)
            pos_users = Tensor(data["pos_users"], mstype.int32)
            pos_items = Tensor(convert_item_id(
                data["pos_items"], num_user), mstype.int32)

            u_group_nodes = Tensor(data["u_group_nodes"], mstype.int32)
            u_neighs = Tensor(convert_item_id(
                data["u_neighs"], num_user), mstype.int32)
            u_gnew_neighs = Tensor(convert_item_id(
                data["u_gnew_neighs"], num_user), mstype.int32)

            i_group_nodes = Tensor(convert_item_id(
                data["i_group_nodes"], num_user), mstype.int32)
            i_neighs = Tensor(data["i_neighs"], mstype.int32)
            i_gnew_neighs = Tensor(data["i_gnew_neighs"], mstype.int32)

            neg_group_nodes = Tensor(convert_item_id(
                data["neg_group_nodes"], num_user), mstype.int32)
            neg_neighs = Tensor(data["neg_neighs"], mstype.int32)
            neg_gnew_neighs = Tensor(data["neg_gnew_neighs"], mstype.int32)

            train_loss = train_net(u_id,
                                   pos_item_id,
                                   neg_item_id,
                                   pos_users,
                                   pos_items,
                                   u_group_nodes,
                                   u_neighs,
                                   u_gnew_neighs,
                                   i_group_nodes,
                                   i_neighs,
                                   i_gnew_neighs,
                                   neg_group_nodes,
                                   neg_neighs,
                                   neg_gnew_neighs)

            if iter_num == num_iter:
                print('Epoch', '%03d' % _epoch, 'iter', '%02d' % iter_num,
                      'loss',
                      '{}, cost:{:.4f}'.format(train_loss, time.time() - epoch_start))
            iter_num += 1

        if _epoch % config.eval_interval == 0:
            save_checkpoint(bgcfnet, config.ckptpath +
                            "/bgcf_epoch{}.ckpt".format(_epoch))
            last_epoch_name = "bgcf_epoch{}.ckpt".format(_epoch)
    run_export()

if __name__ == "__main__":
    # train and export
    run_train()
    