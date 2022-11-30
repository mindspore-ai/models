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
"""Infer api."""
import time

import numpy as np

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore import context

from src.dataset import load_dataset

import onnx
import onnxruntime

context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=False,
    device_target="CPU",
    reserve_class_name_in_scope=False)


def seq2seq_infer_onnx(config, dataset):
    """
    Run infer with Seq2seqModel.

    Args:
        config (Seq2seqConfig): Config.
        dataset (Dataset): Dataset.

    Returns:
        List[Dict], prediction, each example has 4 keys, "source",
        "target", "prediction" and "prediction_prob".
    """

    predictions = []
    source_sentences = []

    shape = P.Shape()
    batch_index = 1
    pad_idx = 0
    sos_idx = 2
    eos_idx = 3
    source_ids_pad = np.tile(np.array([[sos_idx, eos_idx] + [pad_idx] * (config.seq_length - 2)]),
                             [config.batch_size, 1])
    source_mask_pad = np.tile(np.array([[1, 1] + [0] * (config.seq_length - 2)]), [config.batch_size, 1])

    original_model = onnx.load(config.onnx_file, load_external_data=False)
    onnx.checker.check_model(original_model)
    m = original_model
    model_payload = m.SerializeToString()
    session = onnxruntime.InferenceSession(model_payload, providers=['CPUExecutionProvider']) #xxx.onnx is exported filename

    for batch in dataset.create_dict_iterator():
        source_sentences.append(batch["source_eos_ids"].asnumpy())
        source_ids = Tensor(batch["source_eos_ids"], mstype.int32)
        active_num = shape(source_ids)[0]
        source_ids = batch["source_eos_ids"].asnumpy()
        source_ids[source_ids >= config.vocab_size] = config.vocab_size-1
        source_mask = batch["source_eos_mask"].asnumpy()

        if active_num < config.batch_size:
            source_ids = np.concatenate([source_ids, source_ids_pad[active_num:, :]], axis=0).astype(np.int32)
            source_mask = np.concatenate([source_mask, source_mask_pad[active_num:, :]], axis=0).astype(np.int32)
        print("source_mask.shape = ", source_mask.shape)
        inputs = {session.get_inputs()[0].name: source_ids, session.get_inputs()[1].name: source_mask}

        start_time = time.time()
        predicted_ids = session.run(None, inputs)
        predicted_ids = np.array(predicted_ids)
        predicted_ids = np.squeeze(predicted_ids, axis=0)

        print(f" | BatchIndex = {batch_index}, Batch size: {config.batch_size}, active_num={active_num}, "
              f"Time cost: {time.time() - start_time}.")

        if active_num < config.batch_size:
            predicted_ids = predicted_ids[:active_num, :]
        batch_index = batch_index + 1
        print(predicted_ids)
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


def infer_onnx(config):
    """
    Seq2seqModel infer api.

    Args:
        config (GNMTConfig): Config.

    Returns:
        list, result with
    """
    eval_dataset = load_dataset(data_files=config.test_dataset,
                                batch_size=config.batch_size,
                                sink_mode=config.dataset_sink_mode,
                                drop_remainder=False,
                                is_translate=True,
                                shuffle=False) if config.test_dataset else None
    prediction = seq2seq_infer_onnx(config, eval_dataset)
    return prediction
