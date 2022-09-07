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

# Loading unknown datasets for inference
import ast
import operator
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import train_dataset_creator
from src.PSENET.psenet import PSENet
from src.PSENET.dice_loss import DiceLoss
from src.network_define import WithLossCell, TrainOneStepCell, LossCallBack
from src.lr_schedule import dynamic_lr
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id, get_device_num

binOps = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod
}

def arithmeticeval(s):
    node = ast.parse(s, mode='eval')

    def _eval(node):
        if isinstance(node, ast.BinOp):
            return binOps[type(node.op)](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.Num):
            return node.n

        if isinstance(node, ast.Expression):
            return _eval(node.body)

        raise Exception('unsupported type{}'.format(node))
    return _eval(node.body)
# load env
device_target = config.device_target
context.set_context(mode=context.GRAPH_MODE, device_target=device_target, device_id=get_device_id())

# load DataSet
rank_id = 0
device_num = get_device_num()
ds = train_dataset_creator(rank_id, device_num)
print('Create dataset done!')

# Define the model
config.INFERENCE = False
net = PSENet(config)
net = net.set_train()
if config.pre_trained:
    param_dict = load_checkpoint(config.pre_trained)
    load_param_into_net(net, param_dict, strict_load=True)
    print('Load Pretrained parameters done!')

criterion = DiceLoss(batch_size=config.TRAIN_BATCH_SIZE)
start_lr = 1e-3
lr_scale = 0.1
total_iters = config.TRAIN_TOTAL_ITER
config.BASE_LR = arithmeticeval(config.BASE_LR)
config.WARMUP_RATIO = arithmeticeval(config.WARMUP_RATIO)
lrs = dynamic_lr(config.BASE_LR, config.TRAIN_TOTAL_ITER, config.WARMUP_STEP, config.WARMUP_RATIO)
opt = nn.SGD(params=net.trainable_params(), learning_rate=lrs, momentum=0.99, weight_decay=5e-4)

# Model deformation
net = WithLossCell(net, criterion)
net = TrainOneStepCell(net, opt)

step_size = ds.get_dataset_size()
time_cb = TimeMonitor(data_size=step_size)
loss_cb = LossCallBack(per_print_times=20)

# Set and apply checkpoint parameters
ckpoint_cf = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=2)
ckpoint_cb = ModelCheckpoint(prefix="PSENet", config=ckpoint_cf, directory=config.TRAIN_MODEL_SAVE_PATH)

model = Model(net)
model.train(config.TRAIN_REPEAT_NUM, ds, dataset_sink_mode=False, callbacks=[time_cb, loss_cb, ckpoint_cb])
print("train Done")

# Load the pretrained model
param_dict = load_checkpoint(config.checkpoint_path)
load_param_into_net(net, param_dict)
net.set_train(False)

# Make predictions on unknown datasets
acc = model.eval(dataset)
print("accuracy: ", acc)
