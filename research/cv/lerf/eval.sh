#!/bin/bash

# Copyright 2023 Huawei Technologies Co., Ltd
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

# lerf-lut
python lerf_sr/eval_lerf_lut.py --expDir ./models/lerf-lut \
                                --testDir ./datasets/Benchmark \
                                --resultRoot ./results \
                                --lutName LUT_ft
                                
# reference result
# Scale           2.0x2.0         3.0x3.0         4.0x4.0
# Set5            35.71/0.9474    32.02/0.8980    30.15/0.8548
                        
# lerf-net
python lerf_sr/eval_lerf_net.py --expDir ./models/lerf-net \
                                --testDir ./datasets/Benchmark \
                                --resultRoot ./results
                                             
# reference result   
# Scale           2.0x2.0         3.0x3.0         4.0x4.0
# Set5            36.03/0.9517    32.17/0.9035    30.26/0.8608