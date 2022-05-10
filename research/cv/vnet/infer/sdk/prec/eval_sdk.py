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
""" eval_sdk.py """
import os
import numpy as np
from src.config import vnet_cfg as cfg
from src.dataset import InferImagelist
from src.utils import evaluation

cfg['dirPredictionImage'] = './result_imgs'
if not os.path.exists('./result_imgs'):
    os.makedirs('./result_imgs')

dataInferlist = InferImagelist(cfg, '../../promise/TestData', '../../val.csv')
dataManagerInfer = dataInferlist.dataManagerInfer



files = os.listdir('./result')
for file in files:
    res = np.fromfile('./result/' + file, dtype=np.float32).reshape(128, 128, 64)
    dataManagerInfer.writeResultsFromNumpyLabel(res, file.split('.')[0], '_test', '.mhd')

evaluation('../../promise/TestData/gt', cfg['dirPredictionImage'])
