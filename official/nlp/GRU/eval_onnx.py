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
"""ONNX evaluation script."""
import os
import onnxruntime as ort
from src.dataset import create_gru_dataset

from model_utils.config import config
from model_utils.device_adapter import get_device_num


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


def run_onnx_eval():
    """
    ONNX evaluation.
    """
    mindrecord_file = config.dataset_path
    if not os.path.exists(mindrecord_file):
        print("dataset file {} not exists, please check!".format(mindrecord_file))
        raise ValueError(mindrecord_file)
    dataset = create_gru_dataset(epoch_count=config.num_epochs, batch_size=config.eval_batch_size,
                                 dataset_path=mindrecord_file, rank_size=get_device_num(), rank_id=0,
                                 do_shuffle=False, is_training=False)
    dataset_size = dataset.get_dataset_size()
    print("dataset size is {}".format(dataset_size))

    session, [ids_name, target_name] = create_session(config.ckpt_file, config.device_target)

    predictions = []
    source_sents = []
    target_sents = []
    eval_text_len = 0
    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        source_sents.append(batch["source_ids"])
        target_sents.append(batch["target_ids"])
        source_ids = batch["source_ids"]
        target_ids = batch["target_ids"]
        [predicted_ids] = session.run(None, {ids_name: source_ids, target_name: target_ids})
        print("predicts is ", predicted_ids)
        print("target_ids is ", target_ids)
        predictions.append(predicted_ids)
        eval_text_len = eval_text_len + 1

    f_output = open(config.output_file, 'w')
    f_target = open(config.target_file, "w")
    for batch_out, true_sentence in zip(predictions, target_sents):
        for i in range(config.eval_batch_size):
            target_ids = [str(x) for x in true_sentence[i].tolist()]
            f_target.write(" ".join(target_ids) + "\n")
            token_ids = [str(x) for x in batch_out[i].tolist()]
            f_output.write(" ".join(token_ids) + "\n")
    f_output.close()
    f_target.close()

if __name__ == "__main__":
    run_onnx_eval()
