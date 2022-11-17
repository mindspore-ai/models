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
""" train and evaluate """

from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor

from src.callbacks import LossCallBack, EvalCallBack
from src.datasets import create_dataset, DataType
from src.metrics import AUCMetric
from src.model_utils.config import config as cfg
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_builder import ModelBuilder

def train_eval_fibinet(config):
    """
    train and eval fibinet
    """
    data_path = config.data_path
    batch_size = config.batch_size
    epochs = config.epochs
    sparse = config.sparse
    if config.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif config.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5
    ds_train = create_dataset(data_path, train_mode=True, line_per_sample=config.line_per_sample,
                              batch_size=batch_size, data_type=dataset_type)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))

    net_builder = ModelBuilder()

    train_net, eval_net = net_builder.get_net(config)
    train_net.set_train()
    model = Model(train_net)
    callback = LossCallBack(config=config)
    ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(), keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix='fibinet_train', directory=config.ckpt_path, config=ckptconfig)


    if config.eval_while_train:
        ds_eval = create_dataset(data_path, train_mode=False, line_per_sample=config.line_per_sample,
                                 batch_size=batch_size, data_type=dataset_type)
        print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))
        auc_metric = AUCMetric()
        model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})
        eval_callback = EvalCallBack(model, ds_eval, auc_metric, config)
        out = model.eval(ds_eval, dataset_sink_mode=(not sparse))
        print("=====" * 5 + "model.eval() initialized: {}".format(out))
        model.train(epochs, ds_train,
                    callbacks=[TimeMonitor(ds_train.get_dataset_size()), eval_callback, callback, ckpoint_cb],
                    dataset_sink_mode=(not sparse))
    else:
        model.train(epochs, ds_train, callbacks=[TimeMonitor(ds_train.get_dataset_size()), callback, ckpoint_cb],
                    dataset_sink_mode=(not sparse))



def modelarts_pre_process():
    cfg.ckpt_path = cfg.output_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_fibinet():
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target,
                        max_call_depth=10000)
    train_eval_fibinet(cfg)
if __name__ == "__main__":
    train_fibinet()
