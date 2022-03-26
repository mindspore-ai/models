#!/bin/bash
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
cd ..
python eval.py --device_target=Ascend   \
      --test_fold='./result/ECSSD'      \
      --model='./EGNet/run-nnet/models/final_resnet_bone.ckpt'  \
      --sal_mode=e  \
      --base_model=resnet >test_e.log
python eval.py --device_target=Ascend    \
      --test_fold='./result/PASCAL-S'  \
      --model='./EGNet/run-nnet/models/final_resnet_bone.ckpt'  \
      --sal_mode=p  \
      --base_model=resnet >test_p.log
python eval.py --device_target=Ascend      \
      --test_fold='./result/DUT-OMRON'  \
      --model='./EGNet/run-nnet/models/final_resnet_bone.ckpt'  \
      --sal_mode=d  \
      --base_model=resnet >test_d.log
python eval.py --device_target=Ascend   \
      --test_fold='./result/HKU-IS'     \
      --model='./EGNet/run-nnet/models/final_resnet_bone.ckpt'  \
      --sal_mode=h  \
      --base_model=resnet >test_h.log
python eval.py --device_target=Ascend   \
      --test_fold='./result/SOD'           \
      --model='./EGNet/run-nnet/models/final_resnet_bone.ckpt'  \
      --sal_mode=s  \
      --base_model=resnet >test_s.log
python eval.py --device_target=Ascend   \
      --test_fold='./result/DUTS-TE'  \
      --model='./EGNet/run-nnet/models/final_resnet_bone.ckpt'  \
      --sal_mode=t  \
      --base_model=resnet >test_t.log