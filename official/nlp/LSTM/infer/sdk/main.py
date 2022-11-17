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

import argparse

from sentiment import sentiment_analysis
from utils import util


def parse_args():
    """
    set and check parameters
    """
    parser = argparse.ArgumentParser(description="lstm process")
    parser.add_argument("--pipeline", type=str, default='pipeline/lstm.pipeline',
                        help="SDK infer pipeline")

    parser.add_argument("--input_sentences_dir", type=str, default="../dataset/data/input",
                        help="input sentences dir")
    parser.add_argument("--max_load_num", type=str, default='', help="max num of infer files")

    parser.add_argument("--parse_word_vector", type=bool, default=False,
                        help="whether need to parse review to word vector")
    parser.add_argument("--vocab_path", type=str, default="../dataset/data/imdb.vocab",
                        help="vocabulary table path")

    parser.add_argument("--do_eval", type=bool, default=False,
                        help="whether evaluate sentiment model accuracy with test_set")
    parser.add_argument("--test_set_feature_path", type=str, default="../dataset/aclImdb/preprocess/00_data",
                        help="test dataset feature path, effective when enable do_eval")
    parser.add_argument("--test_set_label_path", type=str, default="../dataset/aclImdb/preprocess/label_ids.npy",
                        help="test dataset label path, effective when enable do_eval")

    parser.add_argument("--result_dir", type=str, default="result", help="save result path")
    parser.add_argument("--infer_result_file", type=str, default="infer.txt", help="infer result file name")
    parser.add_argument("--eval_result_file", type=str, default="eval.txt", help="evaluate result file name")

    args_opt = parser.parse_args()
    return args_opt


def run():
    """
    do infer and evaluate
    """
    config = parse_args()
    print(config)

    batch_size = 64
    load_preprocess_num = None if config.max_load_num == '' else int(config.max_load_num)

    if config.do_eval:
        # load test set
        sequences_word_vector, sequences_label, eval_files = util.load_preprocess_test_set(config.test_set_feature_path,
                                                                                           config.test_set_label_path,
                                                                                           load_preprocess_num)

        # model inference
        infer_result = sentiment_analysis.senti_analysis(sequences_word_vector, config.pipeline, is_batch=True,
                                                         batch_size=batch_size)
        if infer_result is None or infer_result.shape[0] < sequences_label.shape[0]:
            print('Sentiment model infer error.')
            return

        # calculate evaluation metrics
        metrics = util.calculate_metrics(sequences_label, infer_result)
        # write infer result to file
        util.write_infer_result_to_file(config.result_dir, config.infer_result_file, eval_files, infer_result, False)
        # write evaluation result to file
        util.write_eval_result_to_file(config.result_dir, config.eval_result_file, metrics)
    else:
        # load model input
        if config.parse_word_vector:
            sentence_word_vector, infer_files = util.convert_sentence_to_word_vector(config.input_sentences_dir,
                                                                                     config.vocab_path)
        else:
            sentence_word_vector, infer_files = util.load_infer_input(config.input_sentences_dir, load_preprocess_num)

        # model inference
        if sentence_word_vector.shape[0] > 1:
            infer_result = sentiment_analysis.senti_analysis(sentence_word_vector,
                                                             config.pipeline, is_batch=True, batch_size=batch_size)
        else:
            infer_result = sentiment_analysis.senti_analysis(sentence_word_vector, config.pipeline, is_batch=False)

        if infer_result is None:
            print('Sentiment model infer error.')
        else:
            print('sentiment model infer result: ', infer_result.reshape(1, -1).squeeze())
            # write infer result to file
            util.write_infer_result_to_file(config.result_dir, config.infer_result_file, infer_files, infer_result,
                                            config.parse_word_vector)

    return


if __name__ == '__main__':
    run()
