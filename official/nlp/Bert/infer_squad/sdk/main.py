# Copyright 2021 Huawei Technologies Co., Ltd
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
import unicodedata
import collections

import argparse
import glob
import os
import time
import six

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
    MxProtobufIn, StringVector

TP = 0
FP = 0
FN = 0


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="bert process")
    parser.add_argument("--pipeline", type=str, default="", help="SDK infer pipeline")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Dataset contain input_ids, input_mask, segment_ids, label_ids")
    parser.add_argument("--eval_json_path", type=str, default="", help="label ids to name")
    parser.add_argument("--output_file", type=str, default="", help="save result to file")
    parser.add_argument("--f1_method", type=str, default="BF1", help="calc F1 use the number label,(BF1, MF1)")
    parser.add_argument("--do_eval", type=bool, default=False, help="eval the accuracy of model ")
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
    input_ids = os.path.realpath(os.path.join(args.data_dir, "00_data", file_name))
    if not send_source_data(0, input_ids, stream_name, stream_manager):
        return False
    input_mask = os.path.realpath(os.path.join(args.data_dir, "01_data", file_name))
    if not send_source_data(1, input_mask, stream_name, stream_manager):
        return False
    token_type_id = os.path.realpath(os.path.join(args.data_dir, "02_data", file_name))
    if not send_source_data(2, token_type_id, stream_name, stream_manager):
        return False
    return True


def f1_score(prediction, ground_truth):
    """calculate f1 score"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
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
        print(dataset)
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


def get_infer_logits(infer_result, max_seq_length=384):
    """
    get the result of model output.
    Args:
        infer_result: get logit from infer result
        max_seq_length: sentence input length default is 384.
    """
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    res = res.reshape(max_seq_length, -1)
    #print("output tensor is: ", res.shape)
    start_logits = [float(x) for x in res[:, 0].flat]
    end_logits = [float(x) for x in res[:, 1].flat]

    return start_logits, end_logits


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for (i, score) in enumerate(index_and_score):
        if i >= n_best_size:
            break
        best_indexes.append(score[0])
    return best_indexes


def get_prelim_predictions(args, file_name, start_logits, end_logits, n_best_size, max_answer_length):
    """
    process the logits of infer tensor to Visualization results based on n_best_size and max_answer_length
    Args:
        args: param of config.
        file_name: used feature file name.
        start_logits: get start logit from infer result
        end_logits: get end logit from infer result
        n_best_size: best of n answer (start point and end point).
        max_answer_length: maximum answer length
    """
    feature_tokens_file = os.path.realpath(os.path.join(args.data_dir, "04_data", file_name))
    feature_token_to_orig_map_file = os.path.realpath(os.path.join(args.data_dir, "05_data", file_name))
    feature_token_is_max_context_file = os.path.realpath(os.path.join(args.data_dir, "06_data", file_name))
    example_index_file = os.path.realpath(os.path.join(args.data_dir, "09_data", file_name))
    tokens = pickle.load(open(feature_tokens_file, 'rb+'))
    token_to_orig_map = pickle.load(open(feature_token_to_orig_map_file, 'rb+'))
    token_is_max_context = pickle.load(open(feature_token_is_max_context_file, 'rb+'))
    example_index = np.fromfile(example_index_file, np.int32)[0]
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"])
    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    start_indexes = _get_best_indexes(start_logits, n_best_size)
    end_indexes = _get_best_indexes(end_logits, n_best_size)
    # if we could have irrelevant answers, get the min score of irrelevant
    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= len(tokens):
                continue
            if end_index >= len(tokens):
                continue
            if start_index not in token_to_orig_map:
                continue
            if end_index not in token_to_orig_map:
                continue
            if not token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logits[start_index],
                    end_logit=end_logits[end_index]))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)
    return prelim_predictions, (tokens, token_to_orig_map, token_is_max_context, example_index)


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def get_nbest(args, prelim_predictions, features, n_best_size, do_lower_case):
    """get nbest predictions"""
    _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])
    seen_predictions = {}
    nbest = []
    (tokens, token_to_orig_map, token_is_max_context, example_index) = features
    token_is_max_context = token_is_max_context
    doc_tokens_file = os.path.realpath(os.path.join(args.data_dir, "07_data", 'squad_bs_'+str(example_index)+'.bin'))
    qas_id_file = os.path.realpath(os.path.join(args.data_dir, "08_data", 'squad_bs_'+str(example_index)+'.bin'))
    doc_tokens = pickle.load(open(doc_tokens_file, 'rb+'))
    qas_id = pickle.load(open(qas_id_file, 'rb+'))
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = token_to_orig_map[pred.start_index]
            orig_doc_end = token_to_orig_map[pred.end_index]
            orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)
            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ""
            seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1
    return nbest, qas_id


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


def get_one_prediction(nbest):
    '''get one prediction'''
    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        nbest_json.append(output)

    assert len(nbest_json) >= 1
    return nbest_json


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((33 <= cp <= 47) or (58 <= cp <= 64) or
            (91 <= cp <= 96) or (123 <= cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    control_char = ["\t", "\n", "\r"]
    if char in control_char:
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    whitespace_char = [" ", "\t", "\n", "\r"]
    if char in whitespace_char:
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


class BasicTokenizer():
    """
    Basic tokenizer
    """
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """
        Do basic tokenization.
        Args:
            text: text in unicode.

        Returns:
            a list of tokens split from text
        """
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            aaa = self._run_split_on_punc(token)
            split_tokens.extend(aaa)

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        i = 0
        start_new_word = True
        output = []
        for char in text:
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((0x4E00 <= cp <= 0x9FFF) or
                (0x3400 <= cp <= 0x4DBF) or
                (0x20000 <= cp <= 0x2A6DF) or
                (0x2A700 <= cp <= 0x2B73F) or
                (0x2B740 <= cp <= 0x2B81F) or
                (0x2B820 <= cp <= 0x2CEAF) or
                (0xF900 <= cp <= 0xFAFF) or
                (0x2F800 <= cp <= 0x2FA1F)):
            return True

        return False


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

    stream_name = b'im_bertbase'
    infer_total_time = 0
    # input_ids file list, every file content a tensor[1,128]
    file_list = glob.glob(os.path.join(os.path.realpath(args.data_dir), "00_data", "*.bin"))
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
    all_predictions = collections.OrderedDict()
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
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            return
        start_logits, end_logits = get_infer_logits(infer_result)
        prelim_predictions, features = get_prelim_predictions(args, file_name, start_logits=start_logits,
                                                              end_logits=end_logits, n_best_size=20,
                                                              max_answer_length=30)
        nbest, qas_id = get_nbest(args, prelim_predictions, features, n_best_size=20, do_lower_case=True)
        nbest_json = get_one_prediction(nbest)
        all_predictions[qas_id] = nbest_json[0]["text"]
    file = open('infer_result.txt', 'w')
    file.write(str(all_predictions))
    file.close()
    print(all_predictions)
    print('done')
    post_process(args.eval_json_path, all_predictions, output_metrics="output.json")


if __name__ == '__main__':
    run()
