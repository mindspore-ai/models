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
""" define dataset class """
import numpy as np
import mindspore.dataset as ds


class MyDataset:
    ''' define dataset '''
    def __init__(self, data, label):

        self.data = data.astype(np.float32)
        self.label = label.astype(np.int32)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def create_dataset(data, label, params, is_batch=True):
    dataset_generator = MyDataset(data, label)
    dataset = ds.GeneratorDataset(
        dataset_generator, ["data", "label"], shuffle=False)
    if is_batch:
        dataset = dataset.batch(params.batch_size)
    else:
        # take the whole data as one batch
        dataset = dataset.batch(data.shape[0])
    return dataset


if __name__ == "__main__":

    x_train, y_train, x_val, y_val, x_test, y_test = LoadTabularData(params)
    dataset_ = create_dataset(x_train, y_train)
    for data_, label_ in dataset_:
        data1 = data_['data'].asnumpy()
        label1 = data_['label'].asnumpy()
        print(
            f'data:[{data1[0]:7.5f}, {data1[1]:7.5f}], label:[{label1[0]:7.5f}]')
    print("data size:", dataset_.get_dataset_size())
