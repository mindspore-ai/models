#!/bin/bash
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

# To train images at 2048 x 1024 resolution after training 1024 x 512 resolution models
# First precompute feature maps and save them
python precompute_feature_maps.py --name label2city_512p_feat --batch_size 1 --serial_batches True \
                                  --no_flip True --instance_feat True;
# Adding instances and encoded features
python train.py --name label2city_1024p_feat --netG local --ngf 32 --num_D 3 \
                --load_pretrain checkpoints/label2city_512p_feat/ --niter 50 --niter_decay 50 \
                --niter_fix_global 10 --resize_or_crop none --instance_feat True --load_features True

