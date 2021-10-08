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
preprocess.It reads the rgb.txt file and the depth.txt file, and joins them by finding the best matches.
"""

import argparse


def read_file_list(filename):
    """
    Reads a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitrary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples

    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    lists = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
             len(line) > 0 and line[0] != "#"]
    lists = [(float(l[0]), l[1:]) for l in lists if len(l) > 1]
    return dict(lists)


def associate(first_list1, second_list1, offset, max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = list(first_list1.keys())

    second_keys = list(second_list1.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches1 = []
    for _, a1, b1 in potential_matches:
        if a1 in first_keys and b1 in second_keys:
            first_keys.remove(a1)
            second_keys.remove(b1)
            matches1.append((a1, b1))

    matches1.sort()
    return matches1


if __name__ == '__main__':

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script takes two data files with timestamps and associates them   
    ''')
    parser.add_argument('first_file', default='', help='first text file (format: timestamp data)')
    parser.add_argument('second_file', default='', help='second text file (format: timestamp data)')
    parser.add_argument('--first_only', help='only output associated lines from first file', action='store_true')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.0)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.02)', default=0.02)
    args = parser.parse_args()

    first_list = read_file_list(args.first_file)
    print(first_list)
    second_list = read_file_list(args.second_file)

    matches = associate(first_list, second_list, float(args.offset), float(args.max_difference))
    with open("associate.txt", "w") as f:

        if args.first_only:
            for a, b in matches:

                print("%f %s" % (a, " ".join(first_list[a])))
        else:
            for a, b in matches:

                f.write(' '.join(first_list[a]) + " " + ' '.join(second_list[b]) + '\n')
                print("%f %s %f %s" % (a, " ".join(first_list[a]), b - float(args.offset), " ".join(second_list[b])))
