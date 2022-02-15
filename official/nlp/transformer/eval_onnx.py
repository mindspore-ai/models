# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Transformer evaluation script."""

import os

import mindspore as ms
import onnxruntime as ort

from eval import load_test_data
from src.model_utils.config import config


def create_session(checkpoint_path, target_device):
    """Create ONNX runtime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_names = [x.name for x in session.get_inputs()]
    return session, input_names


def run_transformer_eval():
    """
    Transformer evaluation.
    """

    dataset = load_test_data(
        batch_size=config.batch_size,
        data_file=os.path.join(config.data_file, config.data_file_name)
    )
    session, (ids_name, mask_name) = create_session(config.file_name, config.device_target)

    predictions = []
    source_sents = []
    target_sents = []
    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        source_sents.append(batch["source_eos_ids"])
        target_sents.append(batch["target_eos_ids"])
        source_ids = batch["source_eos_ids"]
        source_mask = batch["source_eos_mask"]

        inputs = {
            ids_name: source_ids,
            mask_name: source_mask
        }

        predicted_ids = session.run(None, inputs)[0]
        predictions.append(predicted_ids)

    # decode and write to file
    with open(config.output_file, 'w') as f:
        for batch_out in predictions:
            for i in range(config.batch_size):
                if batch_out.ndim == 3:
                    batch_out = batch_out[:, 0]
                token_ids = [str(x) for x in batch_out[i].tolist()]
                f.write(" ".join(token_ids) + "\n")


def main():
    """Main function"""
    config.dtype = ms.float32
    config.compute_type = ms.float16
    config.batch_size = config.batch_size_ev
    config.hidden_dropout_prob = config.hidden_dropout_prob_ev
    config.attention_probs_dropout_prob = config.attention_probs_dropout_prob_ev
    run_transformer_eval()


if __name__ == "__main__":
    main()
