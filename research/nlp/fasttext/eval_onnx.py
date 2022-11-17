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
"""FastText for Evaluation"""
import numpy as np
import onnxruntime as ort

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as deC

from model_utils.config import config

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


def load_infer_dataset(batch_size, datafile, bucket):
    """data loader for infer"""
    def batch_per_bucket(bucket_length, input_file):
        input_file = input_file + '/test_dataset_bs_' + str(bucket_length) + '.mindrecord'
        if not input_file:
            raise FileNotFoundError("input file parameter must not be empty.")

        data_set = ds.MindDataset(input_file,
                                  columns_list=['src_tokens', 'src_tokens_length', 'label_idx'])
        type_cast_op = deC.TypeCast(mstype.int32)
        data_set = data_set.map(operations=type_cast_op, input_columns="src_tokens")
        data_set = data_set.map(operations=type_cast_op, input_columns="src_tokens_length")
        data_set = data_set.map(operations=type_cast_op, input_columns="label_idx")

        data_set = data_set.batch(batch_size, drop_remainder=False)
        return data_set
    for i, _ in enumerate(bucket):
        bucket_len = bucket[i]
        ds_per = batch_per_bucket(bucket_len, datafile)
        if i == 0:
            data_set = ds_per
        else:
            data_set = data_set + ds_per

    return data_set


def run_fasttext_onnx_infer(target_label1):
    """run infer with FastText"""
    dataset = load_infer_dataset(batch_size=config.batch_size, datafile=config.dataset_path, bucket=config.test_buckets)



    predictions = []
    target_sens = []
    onnx_path = config.onnx_path
    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        target_sens.append(batch['label_idx'])
        src_tokens = batch['src_tokens']
        src_tokens_length = batch['src_tokens_length']
        src_tokens_shape = batch['src_tokens'].shape
        onnx_file = onnx_path + '/fasttext_' + str(src_tokens_shape[0]) + '_' + str(src_tokens_shape[1]) + '_' \
                    + config.data_name + '.onnx'
        session, [src_t, src_t_len] = create_session(onnx_file, config.device_target)
        [predicted_idx] = session.run(None, {src_t: src_tokens, src_t_len: src_tokens_length})
        predictions.append(predicted_idx)




    from sklearn.metrics import accuracy_score, classification_report
    target_sens = np.array(target_sens).flatten()
    merge_target_sens = []
    for target_sen in target_sens:
        merge_target_sens.extend(target_sen)
    target_sens = merge_target_sens
    predictions = np.array(predictions).flatten()
    merge_predictions = []
    for prediction in predictions:
        merge_predictions.extend(prediction)
    predictions = merge_predictions
    acc = accuracy_score(target_sens, predictions)
    result_report = classification_report(target_sens, predictions, target_names=target_label1)
    print("********Accuracy: ", acc)
    print(result_report)


def main():
    if config.data_name == "ag":
        target_label1 = ['0', '1', '2', '3']
    elif config.data_name == 'dbpedia':
        target_label1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    elif config.data_name == 'yelp_p':
        target_label1 = ['0', '1']
    print(target_label1)
    run_fasttext_onnx_infer(target_label1)

if __name__ == '__main__':
    main()
