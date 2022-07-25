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
"""Export to MINDIR."""
import argparse
from pathlib import Path

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import load_checkpoint
from mindspore.train.serialization import export

from src import config as default_config
from src.model import get_model


def run_export(config, ckpt_url):
    """
    Export model to MINDIR.

    Args:
        config (any): Config parameters.
        ckpt_url (str): Path to the trained model checkpoint.
    """
    # Init the model
    model = get_model(config.model['m2det_config'], config.model['input_size'], test=True)
    load_checkpoint(ckpt_url, model)
    model.set_train(False)

    # Correctly process input
    model_input = np.ones([3, config.model['input_size'], config.model['input_size']])  # CxWxH
    model_input = Tensor(np.expand_dims(model_input, 0), mstype.float32)

    name = Path(ckpt_url).stem
    path = Path(ckpt_url).resolve().parent
    save_path = str(Path(path, name))

    export(model, model_input, file_name=save_path, file_format='MINDIR')
    print('Model exported successfully!')
    print(f'Path to exported model {save_path}.mindir')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to MINDIR.")
    parser.add_argument('--device_id', help="device_id", type=int, default=0)
    parser.add_argument("--ckpt_url", type=str, default='checkpoints/model.ckpt', help="Trained model ckpt.")
    args = parser.parse_args()

    if not Path(args.ckpt_url).is_file():
        raise FileNotFoundError(f"Can not find checkpoint by --ckpt_url path: {args.ckpt_url}")

    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=default_config.device,
        device_id=args.device_id,
    )

    run_export(default_config, args.ckpt_url)
