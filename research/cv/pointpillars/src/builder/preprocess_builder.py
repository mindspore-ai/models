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
"""preproc builder"""
import src.core.preprocess as prep


def build_db_preprocess(db_prep_config):
    """build db preprocess"""
    prep_type = list(db_prep_config.keys())[0]

    if prep_type == 'filter_by_difficulty':
        cfg = db_prep_config['filter_by_difficulty']
        return prep.DBFilterByDifficulty(list(cfg['removed_difficulties']))
    if prep_type == 'filter_by_min_num_points':
        cfg = db_prep_config['filter_by_min_num_points']
        return prep.DBFilterByMinNumPoint({cfg['min_num_point_pairs']['key']: cfg['min_num_point_pairs']['value']})
    raise ValueError("unknown database prep type")
