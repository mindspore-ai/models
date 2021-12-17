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
##############test WideResNet example on cifar10#################
python eval.py
"""
import os
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.summary.summary_record import SummaryRecord

from src.cross_entropy_smooth import CrossEntropySmooth
from src.wide_resnet import wideresnet
from src.dataset import create_dataset
from src.model_utils.config import config as cfg

from src.callbacks import PredictionsCallback

if __name__ == '__main__':

    target = cfg.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=target,
                            save_graphs=False,
                            device_id=int(os.environ["DEVICE_ID"]))
    else:
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=target,
                            save_graphs=False)
    data_path = cfg.data_path

    if cfg.modelart:
        import moxing as mox
        mox.file.copy_parallel(cfg.ckpt_url, dst_url=cfg.checkpoint_file_path)
        param_dict = load_checkpoint('/cache/ckpt_path/WideResNet_best.ckpt')
    else:
        param_dict = load_checkpoint(cfg.checkpoint_file_path)

    ds_eval = create_dataset(dataset_path=data_path,
                             do_train=False,
                             repeat_num=cfg.repeat_num,
                             batch_size=cfg.batch_size,
                             target=target,
                             infer_910=cfg.infer_910)

    net = wideresnet(mode='eval', batch_size=cfg.batch_size)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    if not cfg.use_label_smooth:
        cfg.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True, reduction='mean',
                              smooth_factor=cfg.label_smooth_factor, num_classes=cfg.num_classes)

    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy'})


    output_path = os.path.join(cfg.output_path, "eval_exp_" + cfg.experiment_label)
    summary_save_dir = output_path +  cfg.summary_dir

    cb = []
    with SummaryRecord(summary_save_dir) as summary_record:
        cb += [PredictionsCallback(summary_record=summary_record, summary_freq=cfg.collection_freq)]
        output = model.eval(ds_eval, callbacks=cb)
    print("result:", output)
