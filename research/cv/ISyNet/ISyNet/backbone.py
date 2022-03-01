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
"""ISyNet backbone"""
from collections import OrderedDict
from mindspore import nn

from .layers import Conv2dBatchActivation, ResidualCell


__all__ = ['CustomBackbone']
class CustomBackbone(nn.Cell):
    """IsyNet backbone definition"""
    def __init__(self, json_defenition, layer_index, weight_standardization=0):
        super().__init__()

        self.json_defenition = json_defenition
        self.weight_standardization = weight_standardization
        self.n_stages = self.json_defenition['nStages']
        self.n_edges_block = self.json_defenition['nEdgesInBlock']
        self.blocks = self.json_defenition['Blocks']
        self.layers_list = []
        ind = 0
        for stage in range(self.n_stages):
            for block in range(int(self.blocks[str(stage)]["nBlocks"])):
                block_dict = OrderedDict()
                if block == 0:
                    skip_type = "noSkip"
                else:
                    skip_type = self.blocks[str(stage)]["skipType"]
                block_dict = OrderedDict([
                    (f"{stage}_{block}_{edge}",
                     Conv2dBatchActivation(layer_index[ind + edge][0],
                                           layer_index[ind + edge][1],
                                           layer_index[ind + edge][2],
                                           layer_index[ind + edge][3],
                                           layer_index[ind + edge][4],
                                           layer_index[ind + edge][5],
                                           weight_standardization=self.weight_standardization))
                    for edge in range(self.n_edges_block)])
                block_cell = ResidualCell(nn.SequentialCell(block_dict), skip_type)
                self.layers_list.append(block_cell)
                ind += self.n_edges_block
        self.json_defenition = json_defenition
        self.layer_index = layer_index
        self.all_cells = nn.SequentialCell(self.layers_list)

    def construct(self, *inputs, **_kwargs):
        """ISyNet construct"""
        x = inputs[0]
        y = self.all_cells(x)
        return y
