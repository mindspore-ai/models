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
"""Export models into different formats"""
import os
from pathlib import Path

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import export
from mindspore.train.serialization import load_checkpoint

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from src.dataset import get_number_of_entities_and_relations
from src.model_builder import create_model


def modelarts_pre_process():
    """modelarts pre process function."""
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """Export model to the format specified by the user."""
    ent_tot, rel_tot = get_number_of_entities_and_relations(
        config.dataset_root,
        config.entities_file_name,
        config.relations_file_name,
    )
    net = create_model(ent_tot, rel_tot, config)

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    load_checkpoint(config.ckpt_file, net=net)
    net.set_train(False)

    hrt_data = Tensor(np.zeros([config.export_batch_size, 3], np.int32))

    export_dir = Path(config.export_output_dir)
    export_file = export_dir / config.model_name

    export(net, hrt_data, file_name=str(export_file), file_format=config.file_format)


if __name__ == '__main__':
    run_export()
