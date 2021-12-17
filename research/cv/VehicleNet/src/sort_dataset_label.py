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
"""dataset process"""
if __name__ == '__main__':
    # process dataset: VeCc
    VeCc_file = '../dataset/dataset_txt/VeCc_name_train.txt'
    VeCc_fid = open('../dataset/VehicleNet/VeCc/name_train.txt', 'w')
    write_lines = []
    with open(VeCc_file, 'r') as f:
        new_label = 0
        line_0 = f.readline().split('\n')[0]
        old_label = line_0.split('\n')[0][0:4]
        write_lines.append(line_0.split('\n')[0] + ' ' + str(new_label))

        for line in f.readlines():
            label = line.split('\n')[0][0:4]
            if label != old_label:
                new_label += 1
                old_label = label
            write_lines.append(line.split('\n')[0] + ' ' + str(new_label))

    for each in write_lines:
        VeCc_fid.write(each + '\n')
    VeCc_fid.close()
    print("VeCc SUCCESS")

    # process dataset: VeRi train
    VeRi_file = '../dataset/dataset_txt/VeRi_name_train.txt'
    VeRi_fid = open('../dataset/VehicleNet/VeRi/name_train_second.txt', 'w')
    write_lines = []
    with open(VeRi_file, 'r') as f:
        new_label = 0
        line_0 = f.readline().split('\n')[0]
        old_label = line_0.split('\n')[0][0:4]
        write_lines.append(line_0.split('\n')[0] + ' ' + str(new_label))

        for line in f.readlines():
            label = line.split('\n')[0][0:4]
            if label != old_label:
                new_label += 1
                old_label = label
            write_lines.append(line.split('\n')[0] + ' ' + str(new_label))

    for each in write_lines:
        VeRi_fid.write(each + '\n')
    VeRi_fid.close()
    print("VeRi train SUCCESS")

    # process dataset: VeRi test
    VeRi_test_file = '../dataset/dataset_txt/VeRi_name_test.txt'
    VeRi_test_fid = open('../dataset/VehicleNet/VeRi/name_test.txt', 'w')
    write_lines = []
    with open(VeRi_test__file, 'r') as f:
        new_label = 0
        line_0 = f.readline().split('\n')[0]
        old_label = line_0.split('\n')[0][0:4]
        write_lines.append(line_0.split('\n')[0] + ' ' + str(new_label) + ' ' + str(line_0.split('\n')[0][6:9]))

        for line in f.readlines():
            label = line.split('\n')[0][0:4]
            if label != old_label:
                new_label += 1
                old_label = label
            write_lines.append(line.split('\n')[0] + ' ' + str(new_label) + ' ' + str(line.split('\n')[0][6:9]))

    for each in write_lines:
        VeRi_test_fid.write(each + '\n')
    VeRi_test_fid.close()
    print("VeRi test SUCCESS")

    # process dataset: VeRi query
    VeRi_query_file = '../dataset/dataset_txt/VeRi_name_query.txt'
    VeRi_query_fid = open('../dataset/VehicleNet/VeRi/name_query.txt', 'w')
    write_lines = []
    with open(VeRi_query_file, 'r') as f:
        new_label = 0
        line_0 = f.readline().split('\n')[0]
        old_label = line_0.split('\n')[0][0:4]
        write_lines.append(line_0.split('\n')[0] + ' ' + str(new_label) + ' ' + str(line_0.split('\n')[0][6:9]))

        for line in f.readlines():
            label = line.split('\n')[0][0:4]
            if label != old_label:
                new_label += 1
                old_label = label
            write_lines.append(line.split('\n')[0] + ' ' + str(new_label) + ' ' + str(line.split('\n')[0][6:9]))

    for each in write_lines:
        VeRi_query_fid.write(each + '\n')
    VeRi_query_fid.close()
    print("VeRi query SUCCESS")

    # process dataset: VeId
    VeId_fid1 = open('../dataset/dataset_test/VeId_sort_new_name_train.txt', 'w')
    write_lines = []
    with open(VeId_file, 'r') as f:
        for line in f.readlines():
            name, label = line.split()
            write_lines.append(name + '.jpg ' + str(label).zfill(6))

    def sort_helper(oldline):
        """sort_helper
        """
        newlabel = oldline.split()[1]
        return newlabel

    write_lines_sort = sorted(write_lines, key=sort_helper)
    for each in write_lines_sort:
        VeId_fid1.write(each + '\n')
    VeId_fid1.close()
    print("VeId sort SUCCESS")

    VeId_file = '../dataset/dataset_txt/VeId_sort_new_name_train.txt'
    VeId_fid = open('../dataset/VehicleNet/VeId/name_train.txt', 'w')
    write_lines = []
    with open(VeId_file, 'r') as f:
        new_label = 5021
        line_0 = f.readline()
        fname, old_label = line_0.split()
        write_lines.append(line_0.split()[0] + ' ' + str(old_label) + ' ' + str(new_label))
        for line in f.readlines():
            name, label = line.split()
            if label != old_label:
                new_label += 1
                old_label = label
            write_lines.append(line.split()[0] + ' ' + str(label) + ' ' + str(new_label))

    for each in write_lines:
        VeId_fid.write(each + '\n')
    VeId_fid.close()
    print("VeId SUCCESS")

    # process dataset: VeCf
    VeCf_file = '../dataset/VeCf/name_train.txt'
    VeCf_fid = open('../dataset/VehicleNet/VeCf/name_train.txt', 'w')
    write_lines = []
    with open(VeCf_file, 'r') as f:
        new_label = 31349
        line_0 = f.readline().split('\n')[0]
        old_label = line_0.split('\n')[0][0:4]
        write_lines.append(line_0.split('\n')[0] + ' ' + str(new_label))
        for line in f.readlines():
            label = line.split('\n')[0][0:4]
            if label != old_label:
                new_label += 1
                old_label = label
            write_lines.append(line.split('\n')[0] + ' ' + str(new_label))

    for each in write_lines:
        VeCf_fid.write(each + '\n')
    VeCf_fid.close()
    print("VeCf SUCCESS")
