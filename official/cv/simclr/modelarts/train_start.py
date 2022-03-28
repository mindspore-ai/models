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
######################## train SimCLR example ########################
train simclr and get network model files(.ckpt) :
python train.py --train_dataset_path /YourDataPath
"""
import ast
import argparse
import os
import numpy as np

from src.nt_xent import NT_Xent_Loss
from src.optimizer import get_train_optimizer as get_optimizer
from src.dataset import create_dataset
from src.simclr_model import SimCLR, SimCLR_Classifier
from src.resnet import resnet50 as resnet
from src.reporter import Reporter


from mindspore import context, Tensor, nn, load_checkpoint, load_param_into_net, export
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import initializer as weight_init
from mindspore.common import set_seed
from mindspore.common.initializer import TruncatedNormal
from mindspore.common import dtype as mstype
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank

parser = argparse.ArgumentParser(description='MindSpore SimCLR')
parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False)
parser.add_argument('--device_target', type=str, default='Ascend',
                    help='Device target, Currently GPU,Ascend are supported.')
parser.add_argument('--run_cloudbrain', type=ast.literal_eval, default=True,
                    help='Whether it is running on CloudBrain platform.')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=True, help='Run distributed training.')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--device_id', type=int, default=0, help='device id, default is 0.')
parser.add_argument('--dataset_name', type=str, default='cifar10', help='Dataset, Currently only cifar10 is supported.')
parser.add_argument('--train_url', default=None, help='Cloudbrain Location of training outputs.\
                    This parameter needs to be set when running on the cloud brain platform.')
parser.add_argument('--data_url', default=None, help='Cloudbrain Location of data.\
                    This parameter needs to be set when running on the cloud brain platform.')
parser.add_argument('--train_dataset_path', type=str, default='/cache/data',
                    help='Dataset path for training classifier. '
                         'This parameter needs to be set when running on the host.')
parser.add_argument('--train_output_path', type=str, default='/cache/train', help='Location of ckpt and log.\
                    This parameter needs to be set when running on the host.')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size, default is 128.')
parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training, default is 100.')
parser.add_argument('--projection_dimension', type=int, default=128,
                    help='Projection output dimensionality, default is 128.')
parser.add_argument('--width_multiplier', type=int, default=1, help='width_multiplier for ResNet50')
parser.add_argument('--temperature', type=float, default=0.5, help='temperature for loss')
parser.add_argument('--pre_trained_path', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--pretrain_epoch_size', type=int, default=0,
                    help='real_epoch_size = epoch_size - pretrain_epoch_size.')
parser.add_argument('--save_checkpoint_epochs', type=int, default=1, help='Save checkpoint epochs, default is 1.')
parser.add_argument('--save_graphs', type=ast.literal_eval, default=False,
                    help='whether save graphs, default is False.')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer, Currently only Adam is supported.')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--warmup_epochs', type=int, default=15, help='warmup epochs.')
parser.add_argument('--use_crop', type=ast.literal_eval, default=True, help='RandomResizedCrop')
parser.add_argument('--use_flip', type=ast.literal_eval, default=True, help='RandomHorizontalFlip')
parser.add_argument('--use_color_jitter', type=ast.literal_eval, default=True, help='RandomColorAdjust')
parser.add_argument('--use_color_gray', type=ast.literal_eval, default=True, help='RandomGrayscale')
parser.add_argument('--use_blur', type=ast.literal_eval, default=False, help='GaussianBlur')
parser.add_argument('--use_norm', type=ast.literal_eval, default=False, help='Normalize')
parser.add_argument("--file_name", type=str, default="simclr_classifier", help="output file name.")
parser.add_argument('--print_iter', type=int, default=100, help='log print iter, default is 100.')

args = parser.parse_args()
local_data_url = '/cache/data'
local_train_url = '/cache/train'
_local_train_url = local_train_url

if args.device_target != "Ascend" and args.device_target != "GPU":
    raise ValueError("Unsupported device target.")
if args.run_distribute and args.device_target == "Ascend":
    device_id = os.getenv("DEVICE_ID", default=None)
    if device_id is None:
        raise ValueError("Unsupported device id.")
    args.device_id = int(device_id)
    rank_size = os.getenv("RANK_SIZE", default=None)
    if rank_size is None:
        raise ValueError("Unsupported rank size.")
    if args.device_num > int(rank_size) or args.device_num == 1:
        args.device_num = int(rank_size)
    context.set_context(device_id=args.device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=args.save_graphs)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True, device_num=args.device_num)
    init()
    args.rank = get_rank()
    local_data_url = os.path.join(local_data_url, str(args.device_id))
    local_train_url = os.path.join(local_train_url, str(args.device_id))
    args.train_output_path = os.path.join(args.train_output_path, str(args.device_id))
elif args.run_distribute and args.device_target == "GPU":
    # GPU target
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=args.save_graphs)
    init()
    context.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)
    context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
    args.train_output_path = os.path.join(args.train_output_path, str(get_rank()))
else:
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        save_graphs=args.save_graphs, device_id=args.device_id)
    args.rank = 0
    args.device_num = 1


if args.run_cloudbrain:
    import moxing as mox
    args.train_dataset_path = os.path.join(local_data_url, "train")
    args.eval_dataset_path = os.path.join(local_data_url, "val")
    args.train_output_path = local_train_url
    mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)


set_seed(1)

class NetWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(NetWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data_x, data_y, label):
        _, _, x_pred, y_pred = self._backbone(data_x, data_y)
        return self._loss_fn(x_pred, y_pred)


class LogisticRegression(nn.Cell):
    """
    Logistic regression
    """
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()
        self.model = nn.Dense(n_features, n_classes, TruncatedNormal(0.02), TruncatedNormal(0.02))

    def construct(self, x):
        x = self.model(x)
        return x


class Linear_Train(nn.Cell):
    """
    Train linear classifier
    """
    def __init__(self, net, loss, opt):
        super(Linear_Train, self).__init__()
        self.netwithloss = nn.WithLossCell(net, loss)
        self.train_net = nn.TrainOneStepCell(self.netwithloss, opt)
        self.train_net.set_train()
    def construct(self, x, y):
        return self.train_net(x, y)


def linear_eval(encoder_ckpt_file):
    """
        linear evaluate
    """
    class_num = 10

    base_net = resnet(1, args.width_multiplier, cifar_stem=args.dataset_name == "cifar10")
    simclr_model = SimCLR(base_net, args.projection_dimension, base_net.end_point.in_channels)
    simclr_param = load_checkpoint(encoder_ckpt_file)

    load_param_into_net(simclr_model.encoder, simclr_param)

    classifier = LogisticRegression(simclr_model.n_features, class_num)
    dataset = create_dataset(args, dataset_mode="train_classifier")
    optimizer = get_optimizer(classifier, dataset.get_dataset_size(), args)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_Train = Linear_Train(net=classifier, loss=criterion, opt=optimizer)
    reporter = Reporter(args, linear_eval=True)
    reporter.dataset_size = dataset.get_dataset_size()
    reporter.linear_eval = True

    dataset_train = []
    for _, data in enumerate(dataset, start=1):
        _, images, labels = data
        features = simclr_model.inference(images)
        dataset_train.append([features.asnumpy(), labels.asnumpy()])
    reporter.info('==========start training linear classifier===============')
    # Train.
    for _ in range(args.epoch_size):
        reporter.epoch_start()
        for _, data in enumerate(dataset_train, start=1):
            features, labels = data
            out = net_Train(Tensor(features), Tensor(labels))
            reporter.step_end(out)
        reporter.epoch_end(classifier)
    reporter.info('==========end training  linear classifier===============')


def _get_last_ckpt(ckpt_dir):
    """
        get last ckpt file
    """
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt') and ckpt_file.startswith('checkpoint')]

    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def _get_last_linear_ckpt(ckpt_dir):
    """
        get last linear ckpt file
    """
    linear_ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                         if ckpt_file.endswith('.ckpt') and ckpt_file.startswith('linearClassifier')]
    if not linear_ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(linear_ckpt_files)[-1])


def _export_air(ckpt_dir):
    """
        export air
    """
    ckpt_file = _get_last_ckpt(ckpt_dir)
    if not ckpt_file:
        return

    linear_eval(ckpt_file)

    linear_ckpt_file = _get_last_linear_ckpt(ckpt_dir)

    width_multiplier = 1
    cifar_stem = True
    projection_dimension = 128
    class_num = 10
    image_height = 32
    image_width = 32

    encoder = resnet(1, width_multiplier=width_multiplier, cifar_stem=cifar_stem)
    classifier = nn.Dense(encoder.end_point.in_channels, class_num)

    simclr = SimCLR(encoder, projection_dimension, encoder.end_point.in_channels)
    param_simclr = load_checkpoint(ckpt_file)
    load_param_into_net(simclr, param_simclr)

    param_classifier = load_checkpoint(linear_ckpt_file)
    load_param_into_net(classifier, param_classifier)

    # export SimCLR_Classifier network
    simclr_classifier = SimCLR_Classifier(simclr.encoder, classifier)
    input_data = Tensor(np.zeros([args.batch_size, 3, image_height, image_width]), mstype.float32)
    export(simclr_classifier, input_data, file_name=os.path.join(local_train_url, args.file_name), file_format="AIR")


def main():
    dataset = create_dataset(args, dataset_mode="train_endcoder")
    # Net.
    base_net = resnet(1, args.width_multiplier, cifar_stem=args.dataset_name == "cifar10")
    net = SimCLR(base_net, args.projection_dimension, base_net.end_point.in_channels)
    # init weight
    if args.pre_trained_path:
        if args.run_cloudbrain:
            mox.file.copy_parallel(src_url=args.pre_trained_path, dst_url=local_data_url + '/pre_train.ckpt')
            param_dict = load_checkpoint(local_data_url + '/pre_train.ckpt')
        else:
            param_dict = load_checkpoint(args.pre_trained_path)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
    optimizer = get_optimizer(net, dataset.get_dataset_size(), args)
    loss = NT_Xent_Loss(args.batch_size, args.temperature)
    net_loss = NetWithLossCell(net, loss)
    train_net = nn.TrainOneStepCell(net_loss, optimizer)
    model = Model(train_net)
    time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_epochs)
    ckpts_dir = os.path.join(args.train_output_path, "checkpoint")
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_simclr", directory=ckpts_dir, config=config_ck)
    print("============== Starting Training ==============")
    model.train(args.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, LossMonitor()])
    _export_air(ckpts_dir)
    if args.run_cloudbrain and args.device_id == 0:
        mox.file.copy_parallel(src_url=_local_train_url, dst_url=args.train_url)

if __name__ == "__main__":
    main()
