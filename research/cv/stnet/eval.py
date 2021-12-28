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
eval
"""
import argparse
from mindspore import Model, context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.CrossEntropySmooth import CrossEntropySmooth
from src.config import cfg_res50
from src.dataset import create_dataset
from src.Stnet_Res_model import Stnet_Res_model


parser = argparse.ArgumentParser(description='video classification')
parser.add_argument('--device_id', type=int, default=1, help='Device id.')
parser.add_argument('--dataset_path', type=str, default=None, help='label_path')
parser.add_argument('--resume', type=str, default='', help='pre_train_checkpoint file_url')
args_opt = parser.parse_args()

set_seed(1)


def val():
    '''eval function.'''
    config = cfg_res50
    target = config['target']
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=config['target'], save_graphs=False)
    if target == "Ascend":
        context.set_context(device_id=args_opt.device_id)

    # Load the data
    print('Loading the data...')
    video_datasets_val = create_dataset(data_dir=args_opt.dataset_path, config=config, shuffle=False, do_trains='val',
                                        num_worker=config['device_num'])
    print('Starting to valid...')
    step_size_val = video_datasets_val.get_dataset_size()
    print('The size of valid set is {}'.format(step_size_val))

    # define net
    net = Stnet_Res_model.stnet50(input_channels=3, num_classes=config['class_num'], T=config['T'], N=config['N'])

    # load pretrain model
    param_dict = load_checkpoint(args_opt.resume)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss function
    loss = CrossEntropySmooth(sparse=True, reduction='mean', num_classes=config['class_num'])

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy'})

    res = model.eval(video_datasets_val)
    print("result:", res, "ckpt=", args_opt.resume)


if __name__ == '__main__':
    val()
