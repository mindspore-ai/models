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
# ===========================================================================
"""Precision calculation function of Douban data"""


def mean_average_precision(sort_data):
    """
    Evaluate mean average precision
    """
    count_1 = 0
    sum_precision = 0
    for index in range(len(sort_data)):
        if sort_data[index][1] == 1:
            count_1 += 1
            sum_precision += 1.0 * count_1 / (index + 1)
    return sum_precision / count_1


def mean_reciprocal_rank(sort_data):
    """
    Evaluate MRR
    """
    sort_lable = [s_d[1] for s_d in sort_data]
    assert 1 in sort_lable
    return 1.0 / (1 + sort_lable.index(1))


def precision_at_position_1(sort_data):
    """
    Evaluate precision
    """
    score = 0
    if sort_data[0][1] == 1:
        score = 1
    return score


def recall_at_position_k_in_10(sort_data, k):
    """"
    Evaluate recall
    """
    sort_lable = [s_d[1] for s_d in sort_data]
    select_lable = sort_lable[:k]
    return 1.0 * select_lable.count(1) / sort_lable.count(1)


def evaluation_one_session(data):
    """
    Evaluate one session
    """
    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    m_a_p = mean_average_precision(sort_data)
    m_r_r = mean_reciprocal_rank(sort_data)
    p_1 = precision_at_position_1(sort_data)
    r_1 = recall_at_position_k_in_10(sort_data, 1)
    r_2 = recall_at_position_k_in_10(sort_data, 2)
    r_5 = recall_at_position_k_in_10(sort_data, 5)
    return m_a_p, m_r_r, p_1, r_1, r_2, r_5


def evaluate(file_path):
    """
    Evaluation is done through a score file.
    :param file_path: Score file path.
    :return: A tuple of accuracy
    """
    sum_m_a_p = 0
    sum_m_r_r = 0
    sum_p_1 = 0
    sum_r_1 = 0
    sum_r_2 = 0
    sum_r_5 = 0

    i = 0
    total_num = 0
    with open(file_path, 'r') as infile:
        for line in infile:
            if i % 10 == 0:
                data = []

            tokens = line.strip().split('\t')
            data.append((float(tokens[0]), int(tokens[1])))

            if i % 10 == 9:
                total_num += 1
                m_a_p, m_r_r, p_1, r_1, r_2, r_5 = evaluation_one_session(data)
                sum_m_a_p += m_a_p
                sum_m_r_r += m_r_r
                sum_p_1 += p_1
                sum_r_1 += r_1
                sum_r_2 += r_2
                sum_r_5 += r_5

            i += 1

    return (1.0 * sum_m_a_p / total_num, 1.0 * sum_m_r_r / total_num, 1.0 * sum_p_1 / total_num,
            1.0 * sum_r_1 / total_num, 1.0 * sum_r_2 / total_num, 1.0 * sum_r_5 / total_num)


def evaluate_m(logits, labels):
    """
    Evaluate through network output.

    :param logits: Network score.
    :param labels: Actual label
    :return: A tuple of accuracy
    """
    sum_m_a_p = 0
    sum_m_r_r = 0
    sum_p_1 = 0
    sum_r_1 = 0
    sum_r_2 = 0
    sum_r_5 = 0

    data = []
    total_num = 0
    for i in range(len(logits)):
        if i % 10 == 0:
            data = []
        data.append((float(logits[i]), int(labels[i])))
        if i % 10 == 9:
            total_num += 1
            m_a_p, m_r_r, p_1, r_1, r_2, r_5 = evaluation_one_session(data)
            sum_m_a_p += m_a_p
            sum_m_r_r += m_r_r
            sum_p_1 += p_1
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5

    return (1.0 * sum_m_a_p / total_num, 1.0 * sum_m_r_r / total_num, 1.0 * sum_p_1 / total_num,
            1.0 * sum_r_1 / total_num, 1.0 * sum_r_2 / total_num, 1.0 * sum_r_5 / total_num)
