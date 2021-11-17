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

import os
import glob
import numpy as np
from PIL import Image
import mindspore
from mindspore import context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication.management import init, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from src.model_utils.config import config
from src.model_utils.device_adapter import get_rank_id, get_device_id
from src.model_utils.matlab_cp2tform import get_similarity_transform_for_cv2
import cv2

def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        ktest = base[int(i*n/n_folds):int((i+1)*n/n_folds)]
        train = list(set(base)-set(ktest))
        folds.append([train, ktest])
    return folds

def alignment(src_img, src_pts):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    crop_size = (96, 112)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

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

class ParameterReduce(nn.Cell):
    """
    reduce parameter
    """
    def __init__(self):
        super(ParameterReduce, self).__init__()
        self.cast = P.Cast()
        self.reduce = P.AllReduce()

    def construct(self, x):
        one = self.cast(F.scalar_to_array(1.0), mstype.float32)
        out = x * one
        ret = self.reduce(out)
        return ret

class getImg():
    def __init__(self, datatxt_src, pairtxt_src, datasrc, net):
        super(getImg, self).__init__()
        self.landmark = {}
        self.predict = []
        with open(datatxt_src) as f:
            landmark_lines = f.readlines()
        for line in landmark_lines:
            l = line.replace('\n', '').split('\t')
            ascend = []
            for i in range(5):
                ascend.append([int(l[2 * i + 1]), int(l[2 * i + 2])])
            self.landmark[datasrc + l[0]] = ascend
        with open(pairtxt_src) as f:
            pairs_lines = f.readlines()[1:]
        self.batchsize = 60
        for i in range(int(6000 / self.batchsize)):
            imglist = []
            samelist = []
            name1list = []
            name2list = []
            reshape = mindspore.ops.Reshape()
            normnet = mindspore.nn.Norm(axis=1)
            for j in range(self.batchsize):
                p = pairs_lines[i * self.batchsize + j].replace('\n', '').split('\t')
                if len(p) == 3:
                    samelist.append(1)
                    name1 = datasrc + p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                    name2 = datasrc + p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))

                if len(p) == 4:
                    samelist.append(0)
                    name1 = datasrc + p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                    name2 = datasrc + p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
                name1list.append(name1)
                name2list.append(name2)
                img1 = np.array(Image.open(name1).convert('RGB'), np.float32)
                img1 = img1[:, :, ::-1]
                img1 = alignment(img1, self.landmark[name1])
                img2 = np.array(Image.open(name2).convert('RGB'), np.float32)
                img2 = img2[:, :, ::-1]
                img2 = alignment(img2, self.landmark[name2])
                imglist.append(img1)
                imglist.append(img2)
            for k in range(len(imglist)):
                imglist[k] = imglist[k].transpose(2, 0, 1)
                imglist[k] = imglist[k].reshape(1, 3, 112, 96)
                imglist[k] = (imglist[k] - 127.5) / 128.0
            for k in range(len(imglist)):
                if k == 0:
                    img = imglist[k]
                else:
                    img = np.concatenate((img, imglist[k]), axis=0)
            output = net.construct(Tensor(img, mindspore.float32))
            for k in range(int(len(imglist) / 2)):
                output1 = output[k * 2]
                output2 = output[k * 2 + 1]
                output1 = reshape(output1, (1, -1))
                output2 = reshape(output2, (1, -1))
                cosdistance = mindspore.ops.tensor_dot(output1, output2, (1, 1))
                norm1 = normnet(output1)
                norm2 = normnet(output2)
                cosdistance = cosdistance / (norm1 * norm2 + 1e-5)
                cosdistance = float(cosdistance.asnumpy())
                self.predict.append('{}\t{}\t{}\t{}\n'.format(name1list[k], name2list[k], cosdistance, samelist[k]))

    def getresult(self):
        return self.predict

def test():
    config.image_size = list(map(int, config.image_size.split(',')))
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                        save_graphs=False)
    if config.device_target == 'Ascend':
        devid = get_device_id()
        context.set_context(device_id=devid)
    # init distributed
    if config.is_distributed:
        init()
        config.rank = get_rank_id()
        config.group_size = get_group_size()
    if config.enable_modelarts:
        import moxing as mox
        config.ckpt_files = '/cache/dataset/device' + os.getenv("DEVICE_ID")
        os.system('mkdir %s' % config.ckpt_files)
        mox.file.copy_parallel(src_url=config.data_url, dst_url='/cache/dataset/device' + os.getenv('DEVICE_ID'))
    if os.path.isdir(config.ckpt_files):
        models = list(glob.glob(os.path.join(config.ckpt_files, '*.ckpt')))
        print(models)
        f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('-')[-1].split('_')[0])

        config.models = sorted(models, key=f)
    else:
        config.models = [config.ckpt_files]

    if  config.net == "sphereface20a":
        from src.network.spherenet import sphere20a as spherenet

    for model in config.models:
        network = spherenet(config.num_classes, True)
        param_dict = load_checkpoint(model)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        print('load model %s success', str(model))

        if config.device_target == 'Ascend':
            network.add_flags_recursive(fp16=True)
        datatxt_src = config.datatxt_src
        pairtxt_src = config.pairtxt_src
        datasrc = config.datasrc
        if config.enable_modelarts:
            images_dir = '/cache/dataset/device' + os.getenv("DEVICE_ID")
            os.system('cd %s ; tar -xvf lfw.tar' % (images_dir))
            datasrc = images_dir +'/'
            pairtxt_src = datasrc + 'pairs.txt'
            datatxt_src = datasrc + 'lfw_landmark.txt'
        ImgOut = getImg(datatxt_src, pairtxt_src, datasrc, network)
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
