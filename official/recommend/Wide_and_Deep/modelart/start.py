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
"""wideanddeep modelarts"""
import os
from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
import moxing as mox
from src.wide_and_deep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from src.callbacks import LossCallBack
from src.datasets import create_dataset, DataType
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from export import export_widedeep


def get_widedeep_net(configure):
    """
    Get network of wide&deep model.
    """
    wide_deep_net = WideDeepModel(configure)

    loss_net = NetWithLossClass(wide_deep_net, configure)
    train_net = TrainStepWrap(loss_net)
    eval_net = PredictWithSigmoid(wide_deep_net)

    return train_net, eval_net


class ModelBuilder():
    """
    Build the model.
    """
    def get_train_hook(self):
        """ get train hook """
        hooks = []
        callback = LossCallBack()
        hooks.append(callback)
        if int(os.getenv('DEVICE_ID')) == 0:
            pass
        return hooks

    def get_net(self, configure):
        """ get net """
        return get_widedeep_net(configure)


def test_train(configure):
    """
    test_train
    """
    data_path = configure.data_path
    batch_size = configure.batch_size
    epochs = configure.epochs
    if configure.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif configure.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5
    ds_train = create_dataset(data_path,
                              train_mode=True,
                              epochs=1,
                              batch_size=batch_size,
                              data_type=dataset_type)

    net_builder = ModelBuilder()
    train_net, _ = net_builder.get_net(configure)
    train_net.set_train()

    model = Model(train_net)
    callback = LossCallBack(config=configure)
    ckptconfig = CheckpointConfig(
        save_checkpoint_steps=ds_train.get_dataset_size(),
        keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix='widedeep_train',
                                 directory=configure.ckpt_path,
                                 config=ckptconfig)
    model.train(epochs,
                ds_train,
                callbacks=[
                    TimeMonitor(ds_train.get_dataset_size()), callback,
                    ckpoint_cb
                ])


def modelarts_pre_process():
    """modelarts pre processt"""
    config.ckpt_path = config.output_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_wide_and_deep():
    """train wide and deep"""
    enable_graph_kernel_ = config.device_target == "GPU"
    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=enable_graph_kernel_,
                        device_target=config.device_target)
    if enable_graph_kernel_:
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")
    test_train(config)


def freeze_model():
    """export model to air format"""
    print("outputs_dir:" + config.output_path)
    for curdir, dirs, files in os.walk(config.output_path):
        print(dirs)
        for file in files:
            if file.endswith('.ckpt'):
                config.ckpt_file = os.path.join(curdir, file)
                print("get_config.ckpt_file:", config.ckpt_file)
                export_widedeep()
    from_path = os.path.join(
        './', config.file_name + '.' + config.file_format.lower())
    to_path = os.path.join(config.train_url,
                           config.file_name + '.' + config.file_format.lower())
    mox.file.copy(from_path, to_path)


def main():
    """main function"""
    train_wide_and_deep()
    os.system("chmod -R 750 /cache/train")
    freeze_model()


if __name__ == "__main__":
    main()
