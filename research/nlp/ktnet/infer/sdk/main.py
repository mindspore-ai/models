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

import argparse
import collections
import glob
import os
import time
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
    MxProtobufIn, StringVector
from src.reader.squad_twomemory import DataProcessor as SquadDataProcessor
from src.reader.squad_twomemory import write_predictions as write_predictions_squad

parser = argparse.ArgumentParser(description="bert process")
parser.add_argument("--pipeline", type=str, default="", help="SDK infer pipeline")
parser.add_argument("--data_dir", type=str, default="",
                    help="Dataset contain input_ids, input_mask, segment_ids, label_ids")
parser.add_argument("--output_file", type=str, default="", help="save result to file")
parser.add_argument("--task_name", type=str, default="squad", help="(squad, record)")
parser.add_argument("--do_eval", type=str, default="True", help="eval the accuracy of model")
parser.add_argument("--checkpoints", type=str, default="./squad", help="Path to save checkpoints")
parser.add_argument("--data_url", type=str, default="../data/rawdata", help="Path to save data")

args, _ = parser.parse_known_args()

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def send_source_data(appsrc_id, filename, stream_name, stream_manager, data_name):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    if data_name == "input_mask":
        tensor = np.fromfile(filename, dtype=np.float32)
    else:
        tensor = np.fromfile(filename, dtype=np.int64)
    tensor = np.expand_dims(tensor, 0)
    if data_name == "wn_concept_ids":
        tensor = tensor.reshape(1, 384, 49, 1)
    elif data_name == "nell_concept_ids":
        tensor = tensor.reshape(1, 384, 27, 1)
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


def send_appsrc_data(args_opt, file_name, stream_name, stream_manager):
    """
    send three stream to infer model, include input ids, input mask and token type_id.

    Returns:
        bool: send data success or not
    """
    input_mask_path = os.path.realpath(os.path.join(args_opt.data_dir, "00_data", file_name))
    if not send_source_data(0, input_mask_path, stream_name, stream_manager, "input_mask"):
        return False
    src_ids_path = os.path.realpath(os.path.join(args_opt.data_dir, "01_data", file_name))
    if not send_source_data(1, src_ids_path, stream_name, stream_manager, "src_ids"):
        return False
    pos_ids_path = os.path.realpath(os.path.join(args_opt.data_dir, "02_data", file_name))
    if not send_source_data(2, pos_ids_path, stream_name, stream_manager, "pos_ids"):
        return False
    sent_ids_path = os.path.realpath(os.path.join(args_opt.data_dir, "03_data", file_name))
    if not send_source_data(3, sent_ids_path, stream_name, stream_manager, "sent_ids"):
        return False
    wn_concept_ids_path = os.path.realpath(os.path.join(args_opt.data_dir, "04_data", file_name))
    if not send_source_data(4, wn_concept_ids_path, stream_name, stream_manager, "wn_concept_ids"):
        return False
    nell_concept_ids_path = os.path.realpath(os.path.join(args_opt.data_dir, "05_data", file_name))
    if not send_source_data(5, nell_concept_ids_path, stream_name, stream_manager, "nell_concept_ids"):
        return False
    unique_id_path = os.path.realpath(os.path.join(args_opt.data_dir, "06_data", file_name))
    if not send_source_data(6, unique_id_path, stream_name, stream_manager, "unique_id"):
        return False
    return True


def read_concept_embedding(embedding_path):
    """read concept embedding"""
    fin = open(embedding_path, encoding='utf-8')
    info = [line.strip() for line in fin]
    dim = len(info[0].split(' ')[1:])
    embedding_mat = []
    id2concept, concept2id = [], {}
    # add padding concept into vocab
    id2concept.append('<pad_concept>')
    concept2id['<pad_concept>'] = 0
    embedding_mat.append([0.0 for _ in range(dim)])
    for line in info:
        concept_name = line.split(' ')[0]
        embedding = [float(value_str) for value_str in line.split(' ')[1:]]
        assert len(embedding) == dim and not np.any(np.isnan(embedding))
        embedding_mat.append(embedding)
        concept2id[concept_name] = len(id2concept)
        id2concept.append(concept_name)
    return concept2id


def post_process(args_opt, file_name, infer_result):
    """
    process the result of infer tensor to Visualization results.
    Args:
        args_opt: param of config.
        file_name: label file name.
        infer_result: get logit from infer result
    """
    # print the infer result
    print("==============================================================")
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    logits = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    print("output tensor is: ", logits.shape)

    # output to file
    result_label = str(logits)
    with open(args_opt.output_file, "a") as output_file:
        output_file.write("{}: {}\n".format(file_name, str(result_label)))
    return logits


def run():
    """
    read pipeline and do infer
    """
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

    stream_name = b'im_ktnet'
    infer_total_time = 0
    # input_ids file list
    file_list = glob.glob(os.path.join(os.path.realpath(args.data_dir), "00_data", "*.bin"))
    data_prefix_len = len(args.task_name) + 1
    for i in range(len(file_list)):
        file_list[i] = file_list[i].split('/')[-1]
    file_list = sorted(file_list, key=lambda name: int(name[data_prefix_len:-4]))
    all_results = []
    for file_name in file_list:
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

        logits = post_process(args, file_name, infer_result)
        if args.do_eval.lower() == 'true':
            logits = logits.reshape((2, 1, 384))

            start_logits, end_logits = np.split(logits, 2, 0)

            label_file = os.path.realpath(os.path.join(args.data_dir, "06_data", file_name))
            unique_ids = np.fromfile(label_file, np.int64)

            np_unique_ids = unique_ids[0].reshape(1, 1)
            np_start_logits = np.squeeze(start_logits, axis=0)
            np_end_logits = np.squeeze(end_logits, axis=0)

            for idx in range(np_unique_ids.shape[0]):
                if len(all_results) % 1000 == 0:
                    print("Processing example: %d" % len(all_results))
                unique_id = int(np_unique_ids[idx])
                start_logits = [float(x) for x in np_start_logits[idx].flat]
                end_logits = [float(x) for x in np_end_logits[idx].flat]

                all_results.append(RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    output_prediction_file = os.path.join(args.checkpoints, "predictions.json")
    output_nbest_file = os.path.join(args.checkpoints, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.checkpoints, "null_odds.json")
    output_evaluation_result_file = os.path.join(args.checkpoints, "eval_result.json")

    wn_concept2id = read_concept_embedding(args.data_url + "/KB_embeddings/wn_concept2vec.txt")
    nell_concept2id = read_concept_embedding(args.data_url + "/KB_embeddings/nell_concept2vec.txt")

    processor = SquadDataProcessor(
        vocab_path=args.data_url + "/cased_L-24_H-1024_A-16/vocab.txt",
        do_lower_case=False,
        max_seq_length=384,
        in_tokens=False,
        doc_stride=128,
        max_query_length=64)

    eval_concept_settings = {
        'tokenization_path': args.data_url + '/tokenization_squad/tokens/dev.tokenization.cased.data',
        'wn_concept2id': wn_concept2id,
        'nell_concept2id': nell_concept2id,
        'use_wordnet': True,
        'retrieved_synset_path': args.data_url + "/retrieve_wordnet/output_squad/retrived_synsets.data",
        'use_nell': True,
        'retrieved_nell_concept_path': args.data_url + "/retrieve_nell/output_squad/dev.retrieved_nell_concepts.data",
    }
    processor.data_generator(
        data_path=args.data_url + "/SQuAD/dev-v1.1.json",
        batch_size=1,
        phase='predict',
        shuffle=False,
        dev_count=1,
        epoch=1,
        **eval_concept_settings)

    features = processor.get_features(
        processor.predict_examples, is_training=False, **eval_concept_settings)

    eval_result = write_predictions_squad(processor.predict_examples, features, all_results,
                                          20, 30, False, output_prediction_file,
                                          output_nbest_file, output_null_log_odds_file,
                                          False, 0.0, False, args.data_url + '/SQuAD/dev-v1.1.json',
                                          output_evaluation_result_file)

    if args.do_eval.lower() == 'true':
        print("==============================================================")
        print(eval_result)
        print("infer_total_time:", infer_total_time)
        print("==============================================================")
    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
