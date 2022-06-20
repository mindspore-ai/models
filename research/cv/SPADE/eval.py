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
""" test """

import os
from collections import OrderedDict
from src.options.test_options import TestOptions
from src.util.visualizer import Visualizer
from src.models.netG import SPADEGenerator
from src.util.eval_fid import calculate_fid_given_paths
from src.data import DatasetInit, ade20k_dataset
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

opt = TestOptions().parse()
context.set_context(
    mode=context.GRAPH_MODE,
    device_target="GPU")
instance = ade20k_dataset.ADE20KDataset()
instance.initialize(opt)
dataset_init = DatasetInit(opt)
dataset = dataset_init.create_dataset_not_distribute(instance)
dataset_iterator = dataset.create_dict_iterator(output_numpy=True)


netG = SPADEGenerator(opt)
param_dict = load_checkpoint(opt.ckpt_dir)
load_param_into_net(netG, param_dict)
visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
img_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
img_dir = os.path.join(img_dir, 'images')
for i, data_i in enumerate(dataset_iterator):
    generated = netG(Tensor(data_i['input_semantics']))
    for b in range(generated.shape[0]):
        print('process image... {0}'.format(i * generated.shape[0] + b))
        visuals = OrderedDict([('input_label', Tensor(data_i['label'])),
                               ('synthesized_image', generated[b]), ('input_image', Tensor(data_i['image'][b]))])
        visualizer.save_images(
            img_dir, visuals, '{0}.png'.format(i * generated.shape[0] + b))

path = [os.path.join(img_dir, 'input_image'), os.path.join(img_dir, 'synthesized_image')]
print("path ", path)
fid_value = calculate_fid_given_paths(path,
                                      50,
                                      2048,
                                      opt.fid_eval_ckpt_dir)
print(fid_value)
