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

"""
sample script of CLUE infer using SDK run in docker
"""
import string
import re
import json
import sys
import math
import pickle
import collections

import argparse
import glob
import os
import time

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
    MxProtobufIn, StringVector

sys.path.append("../utils")


class SquadExample:
    """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


TP = 0
FP = 0
FN = 0


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="albert process")
    parser.add_argument("--pipeline", type=str, default="",
                        help="SDK infer pipeline")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Dataset contain input_ids, input_mask, segment_ids, label_ids")
    parser.add_argument("--eval_json_path", type=str,
                        default="", help="label ids to name")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    args_opt = parser.parse_args()
    return args_opt


def send_source_data(appsrc_id, filename, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensor = np.fromfile(filename, dtype=np.int32)
    tensor = np.expand_dims(tensor, 0)
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    data_input = MxDataInput()
    data_input.data = array_bytes
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = data_input.data
    tensor_vec.tensorDataSize = len(array_bytes)

    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if ret < 0:
        print("Failed to send data to stream.")
        return False
    return True


def send_appsrc_data(args, file_name, stream_name, stream_manager):
    """
    send three stream to infer model, include input ids, input mask and token type_id.

    Returns:
        bool: send data success or not
    """
    input_ids = os.path.realpath(os.path.join(
        args.data_dir, "00_data", file_name))
    if not send_source_data(0, input_ids, stream_name, stream_manager):
        return False
    input_mask = os.path.realpath(os.path.join(
        args.data_dir, "01_data", file_name))
    if not send_source_data(1, input_mask, stream_name, stream_manager):
        return False
    token_type_id = os.path.realpath(
        os.path.join(args.data_dir, "02_data", file_name))
    if not send_source_data(2, token_type_id, stream_name, stream_manager):
        return False
    return True


def f1_score(prediction, ground_truth):
    """calculate f1 score"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(
        prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def post_process(dataset_file, all_predictions, output_metrics="output.json"):
    """
    process the result of infer tensor to Visualization results.
    Args:
        args: param of config.
        file_name: label file name.
        infer_result: get logit from infer result
        max_seq_length: sentence input length default is 128.
    """
    # print the infer result
    with open(dataset_file) as ds:
        print('==========')
        dataset_json = json.load(ds)
        dataset = dataset_json['data']
        # print(dataset)
    print('success')
    re_json = evaluate(dataset, all_predictions)
    print(json.dumps(re_json))
    with open(output_metrics, 'w') as wr:
        wr.write(json.dumps(re_json))


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    """do evaluation"""
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                if not ground_truths:
                    continue
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    print(exact_match)
    print(f1)
    return {'exact_match': exact_match, 'f1': f1}


def get_infer_logits(args, file_name, infer_result, max_seq_length=384, num_class=2):
    """
    get the result of model output.
    Args:
        infer_result: get logit from infer result
        max_seq_length: sentence input length default is 384.
    """
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)

    res = np.frombuffer(
        result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')

    input_mask_file = os.path.realpath(
        os.path.join(args.data_dir, "01_data", file_name))
    input_mask = np.fromfile(
        input_mask_file, np.float32).reshape(max_seq_length)

    res = res.reshape(max_seq_length, num_class)
    #print("output tensor is: ", res.shape)
    start_logits = np.squeeze(res[:, 0:1], axis=-1)
    start_logits = start_logits + 100 * input_mask
    end_logits = np.squeeze(res[:, 1:2], axis=-1)
    end_logits = end_logits + 100 * input_mask

    start_logits = [float(x) for x in start_logits.flat]
    end_logits = [float(x) for x in end_logits.flat]

    return start_logits, end_logits


_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
     "start_log_prob", "end_log_prob"])

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id",
                                    "start_log_prob",
                                    "end_log_prob"])


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def write_predictions_v1(result_dict, all_examples, all_features,
                         all_results, n_best_size, max_answer_length):
    """Write final predictions to the json file and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        for (feature_index, feature) in enumerate(features):
            for ((start_idx, end_idx), logprobs) in \
                    result_dict[example_index][feature.unique_id].items():
                start_log_prob = 0
                end_log_prob = 0
                for logprob in logprobs:
                    start_log_prob += logprob[0]
                    end_log_prob += logprob[1]
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_idx,
                        end_index=end_idx,
                        start_log_prob=start_log_prob / len(logprobs),
                        end_log_prob=end_log_prob / len(logprobs)))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index >= 0:  # this is a non-null prediction
                tok_start_to_orig_index = feature.tok_start_to_orig_index
                tok_end_to_orig_index = feature.tok_end_to_orig_index
                start_orig_pos = tok_start_to_orig_index[pred.start_index]
                end_orig_pos = tok_end_to_orig_index[pred.end_index]

                paragraph_text = example.paragraph_text
                final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_log_prob=0.0, end_log_prob=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def accumulate_predictions_v1(result_dict, all_examples, all_features,
                              all_results, n_best_size, max_answer_length):
    """accumulate predictions for each positions in a dictionary."""
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    for (example_index, _) in enumerate(all_examples):
        if example_index not in result_dict:
            result_dict[example_index] = {}
        features = example_index_to_features[example_index]

        for (_, feature) in enumerate(features):
            if feature.unique_id not in result_dict[example_index]:
                result_dict[example_index][feature.unique_id] = {}
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(
                result.start_log_prob, n_best_size)
            end_indexes = _get_best_indexes(result.end_log_prob, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    doc_offset = feature.tokens.index("[SEP]") + 1
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index - doc_offset >= len(feature.tok_start_to_orig_index):
                        continue
                    if end_index - doc_offset >= len(feature.tok_end_to_orig_index):
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    start_log_prob = result.start_log_prob[start_index]
                    end_log_prob = result.end_log_prob[end_index]
                    start_idx = start_index - doc_offset
                    end_idx = end_index - doc_offset
                    if (start_idx, end_idx) not in result_dict[example_index][feature.unique_id]:
                        result_dict[example_index][feature.unique_id][(
                            start_idx, end_idx)] = []
                    result_dict[example_index][feature.unique_id][(start_idx, end_idx)].append(
                        (start_log_prob, end_log_prob))
    return result_dict


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                orig_answer_text = None
                is_impossible = False

                if is_training:
                    is_impossible = qa.get("is_impossible", False)
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        start_position = answer["answer_start"]
                    else:
                        start_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    paragraph_text=paragraph_text,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    is_impossible=is_impossible)
                examples.append(example)

    return examples


def get_result(result, eval_examples, eval_features):
    """Evaluate the checkpoint on SQuAD 1.0."""

    result_dict = {}
    accumulate_predictions_v1(
        result_dict, eval_examples, eval_features,
        result, 20, 30)
    all_predictions, all_nbest_json = write_predictions_v1(
        result_dict, eval_examples, eval_features, result, 20, 30)
    return all_predictions, all_nbest_json


def takeSecond(elem):
    return elem[1]


def run():
    """
    read pipeline and do infer
    """
    args = parse_args()
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(os.path.realpath(args.pipeline), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'im_albertbase'
    infer_total_time = 0
    # input_ids file list, every file content a tensor[1,128]
    file_list = glob.glob(os.path.join(
        os.path.realpath(args.data_dir), "00_data", "*.bin"))
    cwq_lists = []
    for i in range(len(file_list)):
        b = os.path.split(file_list[i])
        cwq_lists.append(b)

    cwq_lists.sort(key=takeSecond)
    yms_lists = []
    for i in range(len(cwq_lists)):
        c = cwq_lists[i][0]+'/'+cwq_lists[i][1]
        yms_lists.append(c)
    file_list = yms_lists

    eval_examples = read_squad_examples(
        args.eval_json_path, False)
    with open(args.eval_data_file_path, "rb") as fin:
        eval_features = pickle.load(fin)

    outputs = []
    for input_ids in file_list:
        file_name = input_ids.split('/')[-1]
        if not send_appsrc_data(args, file_name, stream_name, stream_manager_api):
            return
        # Obtain the inference result by specifying streamName and uniqueId.
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        start_time = time.time()
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        infer_total_time += time.time() - start_time
        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" %
                  (infer_result[0].errorCode))
            return
        start_log_prob, end_log_prob = get_infer_logits(
            args, file_name, infer_result)

        unique_id_name = os.path.join(args.data_dir, "03_data", file_name)
        unique_id = np.fromfile(unique_id_name, np.int32)
        unique_id = int(unique_id[0])
        outputs.append(RawResult(
            unique_id=unique_id,
            start_log_prob=start_log_prob,
            end_log_prob=end_log_prob))

    all_predictions, _ = get_result(outputs, eval_examples, eval_features)

    js = json.dumps(all_predictions)
    file = open('infer_result.txt', 'w')
    file.write(js)
    file.close()
    print(all_predictions)
    print('done')
    post_process(args.eval_json_path, all_predictions,
                 output_metrics="output.json")


if __name__ == '__main__':
    run()
