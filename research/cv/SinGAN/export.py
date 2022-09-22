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
"""Export SinGAN"""
import os

from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
import src.functions as functions
from src.imresize import imresize
from src.model import get_model
from src.config import get_arguments

def preLauch():
    """parse the console argument"""
    parser = get_arguments()

    # Export Device.
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--device_target', type=str, default='Ascend')
    parser.add_argument('--device_id', type=int, default=1, help='device id of Ascend (Default: 0)')

    # Directories.
    parser.add_argument('--input_dir', type=str, default='data', help='input image dir')
    parser.add_argument('--model_dir', type=str, default='TrainedModels', help='input model dir')
    parser.add_argument('--input_name', type=str, default='thunder.jpg', help='input image name')
    parser.add_argument('--output_dir', type=str, default='export_Output', help='output folder')
    parser.add_argument('--file_format', type=str, default='ONNX', help='["AIR", "MINDIR", "ONNX"]')
    opt = parser.parse_args()
    functions.post_config(opt)

    context.set_context(save_graphs=False, device_id=opt.device_id, \
                                device_target=opt.device_target, mode=context.GRAPH_MODE)
    return opt


def main():
    """main_export"""
    opt = preLauch()
    reals = []
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    scale_num = 0
    real_ = functions.read_image(opt)
    real = imresize(real_, opt.scale1, opt)
    reals = functions.creat_reals_pyramid(real, reals, opt)

    while scale_num < opt.stop_scale + 1:
        print("scale_num:  ", scale_num)
        opt.nzx = reals[scale_num].shape[2]
        opt.nzy = reals[scale_num].shape[3]
        G_curr, _ = get_model(scale_num, opt)
        load_param_into_net(G_curr, load_checkpoint('%s/%d/netG.ckpt' % (opt.model_dir, scale_num)))
        G_curr.set_train(False)
        x = Tensor(functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy]))
        y = Tensor(functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy]))
        path = '%s/%d/' % (opt.output_dir, scale_num)
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        export(G_curr, x, y, file_name='%s/%d/SinGAN' % (opt.output_dir, scale_num), file_format=opt.file_format)
        scale_num += 1
    print("==========SinGAN exported==========")

if __name__ == '__main__':
    main()
