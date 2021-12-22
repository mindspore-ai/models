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
Use this file for model inference and accuracy evaluation
"""
import os
import sys
import time
import mindspore as ms
from mindspore import context
from mindspore import Model
from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num
from src.econet import ECONet
from src.dataset import create_dataset_val

best_prec1 = 0


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
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
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
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
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

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
    config.resume = os.path.join(dirname, config.resume)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    """run eval"""
    device_id = get_device_id()
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)

    if config.dataset == 'ucf101':
        num_class = 101
        rgb_read_format = "{:06d}.jpg"
    elif config.dataset == 'hmdb51':
        num_class = 51
        rgb_read_format = "{:05d}.jpg"
    elif config.dataset == 'kinetics':
        num_class = 400
        rgb_read_format = "{:04d}.jpg"
    elif config.dataset == 'something':
        num_class = 174
        rgb_read_format = "{:05d}.jpg"
    else:
        raise ValueError('Unknown dataset ' + config.dataset)
    net = ECONet(num_class, config.num_segments, config.modality,
                 base_model=config.arch,
                 consensus_type=config.consensus_type, dropout=config.dropout, partial_bn=not config.no_partialbn)

    model_dict = net.parameters_dict()
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint(fintune) '{}'".format(config.resume)))
            pretrained_dict = load_checkpoint(config.resume)
            new_state_dict = {}
            for k, v in pretrained_dict.items():
                if (k in model_dict) and (v.shape == model_dict[k].shape):
                    new_state_dict[k] = v
            un_init_dict_keys = [k for k in model_dict.keys() if k not in new_state_dict]
            print("un_init_dict_keys:", len(un_init_dict_keys))
            load_param_into_net(net, new_state_dict)

        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))
    val_dataset = create_dataset_val(config, rgb_read_format)
    net.set_train(False)
    loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    metrics = {
        'Loss': nn.Loss(),
        'Top1-Acc': nn.Top1CategoricalAccuracy(),
        'Top5-Acc': nn.Top5CategoricalAccuracy()
    }
    model = Model(net, loss, optimizer=None, metrics=metrics)
    acc = model.eval(val_dataset)
    print(acc)


if __name__ == '__main__':
    run_eval()
