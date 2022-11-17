# Copyright 2020 Huawei Technologies Co., Ltd
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
import time
import os
import glob
import moxing as mox
import numpy as np
from mindspore import context, Tensor
from mindspore import export
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.nn.optim import Adam
from mindspore.common import set_seed
from src.model import get_pose_net
from src.network_define import JointsMSELoss, WithLossCell
from src.dataset import keypoint_dataset
#from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
#from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id

set_seed(1)

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)

def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)

def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)

def get_args():
    parser = argparse.ArgumentParser(description="simple_pose training")
    parser.add_argument("--data_url", type=str, default="", help="")
    parser.add_argument("--train_url", type=str, default="", help="")
    parser.add_argument("--air_path", type=str, default="airpath", help="")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--EPOCH", type=int, default=140, help="")
    parser.add_argument("--EXPORT_FILE_FORMAT", type=str, default="AIR", help="")

    reargs_opt = parser.parse_args()
    return reargs_opt
def get_lr(begin_epoch,
           total_epochs,
           steps_per_epoch,
           lr_init=0.1,
           factor=0.1,
           epoch_number_to_drop=(90, 120)
           ):
    """
    Generate learning rate array.

    Args:
        begin_epoch (int): Initial epoch of training.
        total_epochs (int): Total epoch of training.
        steps_per_epoch (float): Steps of one epoch.
        lr_init (float): Initial learning rate. Default: 0.316.
        factor:Factor of lr to drop.
        epoch_number_to_drop:Learing rate will drop after these epochs.
    Returns:
        np.array, learning rate array.
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    step_number_to_drop = [steps_per_epoch * x for x in epoch_number_to_drop]
    for i in range(int(total_steps)):
        if i in step_number_to_drop:
            lr_init = lr_init * factor
        lr_each_step.append(lr_init)
    current_step = steps_per_epoch * begin_epoch
    lr_each_step = np.array(lr_each_step, dtype=np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate

def simple_pose_export(args_opt):
    # define net
    net = get_pose_net(config, True, ckpt_path=config.MODEL.PRETRAINED)

    # load checkpoint
    prob_ckpt_list = os.path.join(config.ckpt_save_dir, "simplepose*.ckpt")
    ckpt_list = glob.glob(prob_ckpt_list)
    if not ckpt_list:
        print('Freezing model failed!')
        print("can not find ckpt files. ")
        return 0

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_name = ckpt_list[-1]
    print("checkpoint file name", ckpt_name)
    param_dict = load_checkpoint(ckpt_name)
    load_param_into_net(net, param_dict)

    image = Tensor(np.zeros([args_opt.batch_size, 3, config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]], np.float32))
    export_path = os.path.join(config.output_path, args_opt.air_path)
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    file_name = os.path.join(export_path, config.EXPORT.FILE_NAME)
    export(net, image, file_name=file_name, file_format=args_opt.EXPORT_FILE_FORMAT)
    print('Freezing model success!')
    return 0

def modelarts_pre_process(args_opt):
    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path)
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    mox.file.copy_parallel(args_opt.data_url, config.data_path)

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
                #data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for files in fz.namelist():
                    i += 1
                    fz.extract(files, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.modelarts_dataset_unzip_name:
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
    config.ckpt_save_dir = os.path.join(config.output_path, config.ckpt_save_dir)
    config.DATASET.ROOT = config.data_path
    config.MODEL.PRETRAINED = os.path.join(config.data_path, config.MODEL.PRETRAINED)
    #args_opt.MODEL_PRETRAINED = os.path.join(os.path.abspath(os.path.dirname(__file__)), args_opt.MODEL_PRETRAINED)

#@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train(args_opt):
    print('batch size :{}'.format(args_opt.batch_size))
    # distribution and context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        save_graphs=False,
                        device_id=get_device_id())

    if config.run_distribute:
        rank = get_rank_id()
        device_num = get_device_num()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
    else:
        rank = 0
        device_num = 1

    # only rank = 0 can write
    rank_save_flag = False
    if rank == 0 or device_num == 1:
        rank_save_flag = True

        # create dataset
    dataset, _ = keypoint_dataset(config,
                                  rank=rank,
                                  group_size=device_num,
                                  train_mode=True,
                                  num_parallel_workers=8)

    # network
    net = get_pose_net(config, True, ckpt_path=config.MODEL.PRETRAINED)
    loss = JointsMSELoss(use_target_weight=True)
    net_with_loss = WithLossCell(net, loss)

    # lr schedule and optim
    dataset_size = dataset.get_dataset_size()
    lr = Tensor(get_lr(config.TRAIN.BEGIN_EPOCH,
                       config.TRAIN.END_EPOCH,
                       dataset_size,
                       lr_init=config.TRAIN.LR,
                       factor=config.TRAIN.LR_FACTOR,
                       epoch_number_to_drop=config.TRAIN.LR_STEP))
    opt = Adam(net.trainable_params(), learning_rate=lr)

    # callback
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.run_distribute:
        config.ckpt_save_dir = os.path.join(config.ckpt_save_dir, str(get_rank_id()))
    if config.ckpt_save_dir and rank_save_flag:
        config_ck = CheckpointConfig(save_checkpoint_steps=dataset_size, keep_checkpoint_max=5)
        ckpoint_cb = ModelCheckpoint(prefix="simplepose", directory=config.ckpt_save_dir, config=config_ck)
        cb.append(ckpoint_cb)
    # train model
    model = Model(net_with_loss, loss_fn=None, optimizer=opt, amp_level="O2")
    epoch_size = config.TRAIN.END_EPOCH - config.TRAIN.BEGIN_EPOCH
    print('start training, epoch size = %d' % epoch_size)
    model.train(epoch_size, dataset, callbacks=cb)

    #export air
    if args_opt.air_path is not None:
        simple_pose_export(args_opt)
    mox.file.copy_parallel(config.output_path, args_opt.train_url)

if __name__ == '__main__':
    args = get_args()
    config.TRAIN.END_EPOCH = args.EPOCH
    modelarts_pre_process(args)
    run_train(args)
    print("simple training finish!")
