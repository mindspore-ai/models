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
"""postprocess for 310 inference"""
import numpy as np
import mindspore
from mindspore import context
from src.model_utils.device_adapter import get_device_id
from src.model_utils.config import config

def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        ktest = base[int(i*n/n_folds):int((i+1)*n/n_folds)]
        train = list(set(base)-set(ktest))
        folds.append([train, ktest])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true == y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

class get_predict():
    def __init__(self):
        super(get_predict, self).__init__()
        normnet = mindspore.nn.Norm(axis=1)
        self.res_src = config.eval_data_dir
        self.predict = []
        for i in range(6000):
            trueflag = (i//300+1)%2
            res1name = self.res_src + '/' + '{}_{}_{}_{}.bin'.format(i, 1, trueflag, 0)
            res2name = self.res_src + '/' + '{}_{}_{}_{}.bin'.format(i, 2, trueflag, 0)
            pred1 = np.fromfile(res1name, np.float32)
            output1 = pred1.reshape(1, 512)
            pred2 = np.fromfile(res2name, np.float32)
            output2 = pred2.reshape(1, 512)
            cosdistance = np.sum(output1[0]*output2[0])
            output1 = mindspore.Tensor(output1)
            output2 = mindspore.Tensor(output2)
            cosdistance = mindspore.Tensor(cosdistance)
            norm1 = normnet(output1)
            norm2 = normnet(output2)
            cosdistance = cosdistance / (norm1 * norm2 + 1e-5)
            cosdistance = float(cosdistance.asnumpy())
            self.predict.append('{}\t{}\t{}\t{}\n'.format(res1name, res2name, cosdistance, trueflag))

    def getresult(self):
        return self.predict

def test():
    config.image_size = list(map(int, config.image_size.split(',')))
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                        save_graphs=False)
    if config.device_target == 'Ascend':
        devid = get_device_id()
        context.set_context(device_id=devid)
    ImgOut = get_predict()
    predicts = ImgOut.getresult()
    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    ans = lambda line: line.strip('\n').split()
    predicts = np.array([ans(x) for x in predicts])
    for idx, (train, ktest) in enumerate(folds):
        print("now the idx is %d", idx)
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[ktest]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

if __name__ == '__main__':
    test()
