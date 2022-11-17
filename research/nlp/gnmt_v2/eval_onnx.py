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
"""ONNX evaluation"""
import pickle
import time

import numpy as np
import onnxruntime as ort
from src.dataset import load_dataset
from src.gnmt_model.bleu_calculate import bleu_calculate
from src.dataset.tokenizer import Tokenizer
from src.utils.get_config import get_config

from model_utils.config import config as default_config


def create_session(checkpoint_path, target_device):
    """Create ONNX runtime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_names = [x.name for x in session.get_inputs()]
    return session, input_names


def infer(config):
    """Run inference"""
    session, [ids_name, mask_name] = create_session(config.file_name, config.device_target)
    eval_dataset = load_dataset(data_files=config.test_dataset,
                                batch_size=config.batch_size,
                                sink_mode=config.dataset_sink_mode,
                                drop_remainder=False,
                                is_translate=True,
                                shuffle=False)

    predictions = []
    source_sentences = []

    batch_index = 1
    pad_idx = 0
    sos_idx = 2
    eos_idx = 3
    source_ids_pad = np.tile(np.array([[sos_idx, eos_idx] + [pad_idx] * (config.seq_length - 2)], np.int32),
                             [config.batch_size, 1])
    source_mask_pad = np.tile(np.array([[1, 1] + [0] * (config.seq_length - 2)], np.int32),
                              [config.batch_size, 1])
    for batch in eval_dataset.create_dict_iterator(output_numpy=True):
        source_sentences.append(batch["source_eos_ids"])
        source_ids = batch["source_eos_ids"]
        source_mask = batch["source_eos_mask"]

        active_num = source_ids.shape[0]
        if active_num < config.batch_size:
            source_ids = np.concatenate((source_ids, source_ids_pad[active_num:, :]))
            source_mask = np.concatenate((source_mask, source_mask_pad[active_num:, :]))

        start_time = time.time()
        [predicted_ids] = session.run(None, {ids_name: source_ids, mask_name: source_mask})

        print(f" | BatchIndex = {batch_index}, Batch size: {config.batch_size}, active_num={active_num}, "
              f"Time cost: {time.time() - start_time}.")
        if active_num < config.batch_size:
            predicted_ids = predicted_ids[:active_num, :]
        batch_index = batch_index + 1
        predictions.append(predicted_ids)

    output = []
    for inputs, batch_out in zip(source_sentences, predictions):
        for i, _ in enumerate(batch_out):
            if batch_out.ndim == 3:
                batch_out = batch_out[:, 0]

            example = {
                "source": inputs[i].tolist(),
                "prediction": batch_out[i].tolist()
            }
            output.append(example)

    return output


def run_onnx_eval():
    """ONNX eval"""
    config = get_config(default_config)
    result = infer(config)

    with open(config.output, "wb") as f:
        pickle.dump(result, f, 1)

    result_npy_addr = config.output
    vocab = config.vocab
    bpe_codes = config.bpe_codes
    test_tgt = config.test_tgt
    tokenizer = Tokenizer(vocab, bpe_codes, 'en', 'de')
    scores = bleu_calculate(tokenizer, result_npy_addr, test_tgt)
    print(f"BLEU scores is :{scores}")


if __name__ == '__main__':
    run_onnx_eval()
