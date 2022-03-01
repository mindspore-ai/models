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
"""Parser of json file architecture description in ISyNet search space"""

class Block():
    """Stage block description"""
    def __init__(self, json_defenition):
        super().__init__()
        self.exp_factor = json_defenition['expantionFactor']
        self.operations = json_defenition['operations']
        self.last_activation = json_defenition['lastActivation']
        self.skip_type = json_defenition['skipType']
        self.n_blocks = json_defenition['nBlocks']
        if 'cIncrease' in json_defenition:
            self.channel_increase = json_defenition['cIncrease']
        else:
            self.channel_increase = 0

def get_layers(json_arch):
    """Construct layer parameters from json definition"""
    backbone_layers = []
    n_stages = json_arch['nStages']
    n_edges_block = json_arch['nEdgesInBlock']
    blocks = {}
    for i in range(n_stages):
        blocks[i] = Block(json_arch['Blocks'][str(i)])
    for stage in range(n_stages):
        curr_block = blocks[stage]
        for bl in range(int(curr_block.n_blocks)):
            for ind_edge, edge in enumerate(curr_block.operations):
                if ind_edge == 0:
                    if bl == 0 and stage == 0:
                        in_channels = 3
                    elif bl == 0 and stage != 0:
                        in_channels = 16 * int((2**(3+stage) * 2**blocks[stage-1].channel_increase)/16)
                    else:
                        in_channels = 16 * int((2**(4+stage) * 2**curr_block.channel_increase)/16)
                else:
                    in_channels = 16 * int((2**(4+stage) * curr_block.exp_factor * 2**curr_block.channel_increase)/16)
                if ind_edge == n_edges_block - 1 or ((edge['op'] != 'identity') and
                                                     all(curr_block.operations[e]['op'] == 'identity'
                                                         for e in range(ind_edge+1, n_edges_block))):
                    out_channels = 16 * int((2**(4+stage) * 2**curr_block.channel_increase)/16)
                else:
                    out_channels = 16 * int((2**(4+stage) * curr_block.exp_factor * 2**curr_block.channel_increase)/16)
                if bl == 0 and ind_edge == 0:
                    stride = 2
                else:
                    stride = 1
                n_groups = edge['group']
                operation = edge['op']
                if ind_edge == n_edges_block - 1:
                    activation = curr_block.last_activation
                else:
                    activation = 'Relu'
                backbone_layers.append([operation, n_groups, stride, in_channels, out_channels, activation])
    return [backbone_layers, out_channels]
