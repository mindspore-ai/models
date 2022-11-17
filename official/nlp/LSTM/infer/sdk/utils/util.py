# coding = utf-8
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License  (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://opensource.org/licenses/BSD-3-Clause

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

import numpy as np

from .word_vector_parser import WordVectorParser


def convert_sentence_to_word_vector(sentence_dir, vocab_path):
    """
    generate word vector from sentence txt file
    :param sentence_dir: dir where sentences store
    :param vocab_path: path where vocab table store
    :return: word vectors and input files
    """
    parser = WordVectorParser(sentence_dir, vocab_path)
    parser.parse()

    return parser.get_datas()


def load_infer_input(input_sentences_dir, max_load_num=None):
    """
    load preprocessed data as model input
    :param input_sentences_dir: path where preprocessed file store
    :param max_load_num: the max number need to load
    :return: sentence word vector and input file names
    """
    input_sentences_files = os.listdir(input_sentences_dir)
    input_sentences_files.sort(key=lambda x: int(x[15:-4]))

    batch_size = 64
    sentence_word_vector_size = 500
    max_load_num = len(input_sentences_files) if max_load_num is None else max_load_num

    # load preprocessed data
    sentence_word_vectors = np.empty(shape=(0, sentence_word_vector_size))
    print('need load test sentences num: {}'.format(max_load_num * batch_size))
    for i in range(0, max_load_num):
        feature_path = os.path.join(input_sentences_dir, input_sentences_files[i])
        print('load ', feature_path)
        data = np.fromfile(feature_path, dtype=np.int32).reshape(batch_size, sentence_word_vector_size)
        sentence_word_vectors = np.vstack([sentence_word_vectors, data])

    return sentence_word_vectors, input_sentences_files[:max_load_num]


def load_preprocess_test_set(test_set_feature_path, test_set_label_path, max_load_num=None):
    """
    load preprocessed test set (feature: .bin label: .npy)
    :param test_set_feature_path: path where feature store
    :param test_set_label_path: path where label store
    :param max_load_num: the max number need to load
    :return: sentence word vector, corresponding label and test set file names
    """
    test_set_feature_files = os.listdir(test_set_feature_path)
    test_set_feature_files.sort(key=lambda x: int(x[15:-4]))

    batch_size = 64
    sentence_word_vector_size = 500
    max_load_num = len(test_set_feature_files) if max_load_num is None else max_load_num

    # load features
    sentence_word_vectors = np.empty(shape=(0, sentence_word_vector_size))
    print('need load test data num: {}'.format(max_load_num * batch_size))
    for i in range(0, max_load_num):
        feature_path = os.path.join(test_set_feature_path, test_set_feature_files[i])
        print('load ', feature_path)
        data = np.fromfile(feature_path, dtype=np.int32).reshape(batch_size, sentence_word_vector_size)
        sentence_word_vectors = np.vstack([sentence_word_vectors, data])

    # load labels
    sentence_labels = np.load(test_set_label_path)
    sentence_labels = sentence_labels[:max_load_num]
    sentence_labels = sentence_labels.reshape(-1, 1).astype(np.int32)

    return sentence_word_vectors, sentence_labels, test_set_feature_files[:max_load_num]


def calculate_metrics(ground_truth, prediction):
    """
    calculate metrics including accuracy, precision of positive and negative,
    recall of positive and negative, F1-score of positive and negative
    :param ground_truth: actual labels
    :param prediction: model inference result
    :return: calculated metrics
    """
    confusion_matrix = np.empty(shape=(2, 2))
    for i in range(0, 2):
        for j in range(0, 2):
            confusion_matrix[i, j] = np.sum(prediction[ground_truth == i] == j)

    total = ground_truth.shape[0]
    negative_gt_total = ground_truth[ground_truth == 0].shape[0]
    positive_gt_total = ground_truth[ground_truth == 1].shape[0]
    negative_pred_total = prediction[prediction == 0].shape[0]
    positive_pred_total = prediction[prediction == 1].shape[0]

    correct = prediction[prediction == ground_truth].shape[0]
    negative_correct = confusion_matrix[0, 0]
    positive_correct = confusion_matrix[1, 1]

    total_accuracy = correct / total
    negative_precision = negative_correct / negative_pred_total
    positive_precision = positive_correct / positive_pred_total

    negative_recall = negative_correct / negative_gt_total
    positive_recall = positive_correct / positive_gt_total

    negative_F1_score = 2 * negative_precision * negative_recall / (negative_precision + negative_recall)
    positive_F1_score = 2 * positive_precision * positive_recall / (positive_precision + positive_recall)

    print('confusion matrix:\n', confusion_matrix)
    print('total accuracy: {}/{} ({})'.format(correct, total, total_accuracy))
    print('negative precision: {}/{} ({})'.format(negative_correct, negative_pred_total, negative_precision))
    print('positive precision: {}/{} ({})'.format(positive_correct, positive_pred_total, positive_precision))
    print('negative recall: {}/{} ({})'.format(negative_correct, negative_gt_total, negative_recall))
    print('positive recall: {}/{} ({})'.format(positive_correct, positive_gt_total, positive_recall))
    print('negative_F1_score: {}'.format(negative_F1_score))
    print('positive_F1_score: {}'.format(positive_F1_score))

    metrics = {'confusion_matrix': confusion_matrix,
               'total_accuracy': total_accuracy,
               'negative_precision': negative_precision,
               'positive_precision': positive_precision,
               'negative_recall': negative_recall,
               'positive_recall': positive_recall,
               'negative_F1_score': negative_F1_score,
               'positive_F1_score': positive_F1_score}

    return metrics


def write_infer_result_to_file(file_path, file_name, infer_files, results, is_raw_data):
    """
    save infer result to file
    :param file_path: result dir
    :param file_name: result file name
    :param infer_files: infer file names
    :param results: infer results
    :param is_raw_data: whether preprocess files
    :return: no return
    """
    if os.path.exists(file_path) != 1:
        os.makedirs(file_path)

    with open(os.path.join(file_path, file_name), 'w') as f:
        if is_raw_data:
            for i in range(0, len(infer_files)):
                f.writelines('file: {}, result: {}\n'.format(infer_files[i], 'positive' if results[i] else 'negative'))
        else:
            f.writelines('files: {}, total reviews num: {}\n'.format(infer_files, results.shape[0]))
            for i in range(0, results.shape[0]):
                f.writelines('{}-th review: {}\n'.format(i + 1, 'positive' if results[i] else 'negative'))


def write_eval_result_to_file(file_path, file_name, eval_result):
    """
    save eval result to file
    :param file_path: result dir
    :param file_name: result file name
    :param eval_result: eval result
    :return: no return
    """
    if os.path.exists(file_path) != 1:
        os.makedirs(file_path)

    with open(os.path.join(file_path, file_name), 'w') as f:
        f.writelines('confusion matrix:\n {}\n'.format(eval_result.get('confusion_matrix')))
        f.writelines('total accuracy: {}\n'.format(eval_result.get('total_accuracy')))
        f.writelines('negative precision: {}\n'.format(eval_result.get('negative_precision')))
        f.writelines('positive precision: {}\n'.format(eval_result.get('positive_precision')))
        f.writelines('negative recall: {}\n'.format(eval_result.get('negative_recall')))
        f.writelines('positive recall: {}\n'.format(eval_result.get('positive_recall')))
        f.writelines('negative F1-score: {}\n'.format(eval_result.get('negative_F1_score')))
        f.writelines('positive F1-score: {}\n'.format(eval_result.get('positive_F1_score')))
