# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tools for building a model using the user defined parameters."""
from src.trans_x import TransD
from src.trans_x import TransE
from src.trans_x import TransH
from src.trans_x import TransR


def create_model(entities_total_num, relations_total_num, config):
    """Creates a model, specified by name

    Args:
        entities_total_num (int): Total number of unique entities in the dataset.
        relations_total_num (int): Total number of unique relations in the dataset.
        config: Object containing the configuration data.
    """
    model_name = config.model_name
    ent_tot = entities_total_num
    rel_tot = relations_total_num
    dim_e = config.dim_e
    dim_r = config.dim_r

    if model_name == 'TransE':
        if dim_e != dim_r:
            raise ValueError(
                f'TransE model requires that the dimensions of embeddings '
                f'for entities and relations were the same, got {dim_e} and {dim_r}'
            )
        return TransE(ent_tot, rel_tot, dim_e)

    if model_name == 'TransH':
        if dim_e != dim_r:
            raise ValueError(
                f'TransH model requires that the dimensions of embeddings '
                f'for entities and relations were the same, got {dim_e} and {dim_r}'
            )
        return TransH(ent_tot, rel_tot, dim_e)

    if model_name == 'TransR':
        return TransR(ent_tot, rel_tot, dim_e, dim_r)

    if model_name == 'TransD':
        return TransD(ent_tot, rel_tot, dim_e, dim_r)

    raise ValueError(
        f'Unsupported model name "{model_name}". '
        f'Must be one of the following: ["TransE", "TransD", "TransH", "TransR"]'
    )
