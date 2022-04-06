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
import operator
import pickle
import os
import numpy as np

class TestingDataSet:
    '''TestingDataSet'''
    def __init__(self, img_dir, csv_path, batch_size):

        self.batch_size = batch_size
        self.image_dir = img_dir
        self.semantic_size = 4800
        csv = pickle.load(open(csv_path, 'rb'), encoding='iso-8859-1')
        self.clip_sentence_pairs = []
        for l in csv:
            clip_name = l[0]
            sent_vecs = l[1]
            for sent_vec in sent_vecs:
                self.clip_sentence_pairs.append((clip_name, sent_vec))
        movie_names_set = set()
        self.movie_clip_names = {}
        for k in range(len(self.clip_sentence_pairs)):
            clip_name = self.clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(k)
        self.movie_names = list(movie_names_set)
        self.movie_names.sort()
        self.clip_num_per_movie_max = 0
        for movie_name in self.movie_clip_names:
            if len(self.movie_clip_names[movie_name]) > self.clip_num_per_movie_max:
                self.clip_num_per_movie_max = len(self.movie_clip_names[movie_name])

        self.sliding_clip_path = img_dir
        sliding_clips_tmp = os.listdir(self.sliding_clip_path)
        self.sliding_clip_names = []
        for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[2] == "npy":
                movie_name = clip_name.split("_")[0]
                if movie_name in self.movie_clip_names:
                    self.sliding_clip_names.append(clip_name.split(".")[0]+"."+clip_name.split(".")[1])
        self.num_samples = len(self.clip_sentence_pairs)
        assert self.batch_size <= self.num_samples

    def get_clip_sample(self, sample_num, movie_name, clip_name):
        '''Get a clip'''
        length = len(os.listdir(self.image_dir+movie_name+"/"+clip_name))
        sample_step = 1.0*length/sample_num
        sample_pos = np.floor(sample_step*np.array(range(sample_num)))
        sample_pos_str = []
        img_names = os.listdir(self.image_dir+movie_name+"/"+clip_name)
        # sort is very important! to get a correct sequence order
        img_names.sort()
        for pos in sample_pos:
            sample_pos_str.append(self.image_dir+movie_name+"/"+clip_name+"/"+img_names[int(pos)])
        return sample_pos_str

    def get_context_window(self, clip_name, win_length):
        '''Get the context window of the fragment'''
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        clip_length = 128#end-start
        left_context_feats = np.zeros([win_length, 4096], dtype=np.float32)
        right_context_feats = np.zeros([win_length, 4096], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path+clip_name)
        last_right_feat = np.load(self.sliding_clip_path+clip_name)
        for k in range(win_length):
            left_context_start = start-clip_length*(k+1)
            left_context_end = start-clip_length*k
            right_context_start = end+clip_length*k
            right_context_end = end+clip_length*(k+1)
            left_context_name = movie_name+"_"+str(left_context_start)+"_"+str(left_context_end)+".npy"
            right_context_name = movie_name+"_"+str(right_context_start)+"_"+str(right_context_end)+".npy"
            if os.path.exists(self.sliding_clip_path+left_context_name):
                left_context_feat = np.load(self.sliding_clip_path+left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if os.path.exists(self.sliding_clip_path+right_context_name):
                right_context_feat = np.load(self.sliding_clip_path+right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat

        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)

    def load_movie_byclip(self, movie_name, sample_num):
        '''Read visual features through clip'''
        movie_clip_sentences = []
        movie_clip_featmap = []
        clip_set = set()
        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append(
                    (self.clip_sentence_pairs[k][0], self.clip_sentence_pairs[k][1][:self.semantic_size]))
                if not self.clip_sentence_pairs[k][0] in clip_set:
                    clip_set.add(self.clip_sentence_pairs[k][0])
                    visual_feature_path = self.image_dir+self.clip_sentence_pairs[k][0]+".npy"
                    feature_data = np.load(visual_feature_path)
                    movie_clip_featmap.append((self.clip_sentence_pairs[k][0], feature_data))
        return movie_clip_featmap, movie_clip_sentences

    def load_movie_slidingclip(self, movie_name, sample_num):
        '''Read visual features through slidingclip'''
        movie_clip_sentences = []
        movie_clip_featmap = []
        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append(
                    (self.clip_sentence_pairs[k][0], self.clip_sentence_pairs[k][1][:self.semantic_size]))
        for k in range(len(self.sliding_clip_names)):
            if movie_name in self.sliding_clip_names[k]:
                visual_feature_path = self.sliding_clip_path+self.sliding_clip_names[k]+".npy"
                left_context_feat, right_context_feat = self.get_context_window(self.sliding_clip_names[k]+".npy", 1)
                feature_data = np.load(visual_feature_path)
                comb_feat = np.hstack((left_context_feat, feature_data, right_context_feat))
                movie_clip_featmap.append((self.sliding_clip_names[k], comb_feat))
        return movie_clip_featmap, movie_clip_sentences


def compute_IoU_recall_top_n_forreg(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips):
    '''
    compute recall at certain IoU
    '''
    correct_num = 0.0
    for k in range(sentence_image_mat.shape[0]):
        gt = sclips[k]
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])
        sim_v = [v for v in sentence_image_mat[k]]
        starts = [s for s in sentence_image_reg_mat[k, :, 0]]
        ends = [e for e in sentence_image_reg_mat[k, :, 1]]
        picks = nms_temporal(starts, ends, sim_v, iou_thresh-0.05)
        if top_n < len(picks):
            picks = picks[0:top_n]
        for index in picks:
            pred_start = sentence_image_reg_mat[k, index, 0]
            pred_end = sentence_image_reg_mat[k, index, 1]
            iou = calculate_IoU((gt_start, gt_end), (pred_start, pred_end))
            if iou >= iou_thresh:
                correct_num += 1
                break
    return correct_num

def calculate_IoU(i0, i1):
    '''calculate_IoU'''
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

def nms_temporal(x1, x2, s, overlap):
    '''nms_temporal'''
    pick = []
    assert len(x1) == len(s)
    assert len(x2) == len(s)
    if not x1:
        return pick

    union = list(map(operator.sub, x2, x1)) # union = x2-x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x: x[1])] # sort and get index

    while I:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i], x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i], x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <= overlap:
                I_new.append(I[j])
        I = I_new
    return pick
