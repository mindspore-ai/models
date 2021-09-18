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
'''Training set and test set loader'''
import os
import pickle
import numpy as np


class TrainDataset:
    '''Training data loader'''
    def __init__(self,
                 sliding_dir,
                 train_pkl_path,
                 valid_pkl_path,
                 visual_dim,
                 sentence_embed_dim,
                 IoU=0.5,
                 nIoU=0.15,
                 context_num=1,
                 context_size=128
                 ):
        self.sliding_dir = sliding_dir
        self.train_pkl_path = train_pkl_path
        self.valid_pkl_path = valid_pkl_path
        self.visual_dim = visual_dim
        self.sentence_embed_dim = sentence_embed_dim
        self.IoU = IoU
        self.nIoU = nIoU
        self.context_num = context_num
        self.context_size = context_size

        self.load_data()

    def load_data(self):
        '''load_data'''
        train_csv = pickle.load(open(self.train_pkl_path, 'rb'), encoding='iso-8859-1')
        self.clip_sentence_pairs = []
        for l in train_csv:
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
        self.num_samples = len(self.clip_sentence_pairs)

        # read sliding windows, and match them with the groundtruths to make training samples
        sliding_clips_tmp = os.listdir(self.sliding_dir)
        sliding_clips_tmp.sort()
        self.clip_sentence_pairs_iou = []
        movie_names = set()
        for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[2] == "npy":
                movie_name = clip_name.split("_")[0]
                movie_names.add(movie_name)
        movie_names = list(movie_names)
        movie_names.sort()
        for movie_name in self.movie_names:
            start_ends = []
            clip_names = []
            for clip_name in sliding_clips_tmp:
                if clip_name.split(".")[2] == "npy":
                    if clip_name.split("_")[0] == movie_name:
                        start = int(clip_name.split("_")[1])
                        end = int(clip_name.split("_")[2].split(".")[0])
                        start_ends.append((start, end))
                        clip_names.append(clip_name)
            table = {}
            for clip_sentence in self.clip_sentence_pairs:
                o_start_ends = []
                original_clip_name = clip_sentence[0]
                original_movie_name = original_clip_name.split("_")[0]
                if original_movie_name == movie_name:
                    o_start = int(original_clip_name.split("_")[1])
                    o_end = int(original_clip_name.split("_")[2].split(".")[0])
                    if (o_start, o_end) in table.keys():
                        match_indexs = table[(o_start, o_end)]
                        for j in match_indexs:
                            start, end = start_ends[j]
                            clip_name = clip_names[j]
                            start_offset = o_start - start
                            end_offset = o_end - end
                            self.clip_sentence_pairs_iou.append(
                                (clip_sentence[0], clip_sentence[1], clip_name, start_offset, end_offset))
                    else:
                        o_start_ends.append((o_start, o_end))
                        start_ends = np.array(start_ends)
                        o_start_ends = np.array(list(set(o_start_ends)))
                        if o_start_ends.shape[0] == 0:
                            continue
                        ious = self.calc_IoU(start_ends, o_start_ends)
                        nIoLs = self.calc_nIoL(o_start_ends, start_ends)
                        match_indexs = (nIoLs < self.nIoU)[0] & (ious > self.IoU)[:, 0]
                        match_indexs = np.where(match_indexs)[0]

                        table[(o_start, o_end)] = match_indexs
                        for k in match_indexs:
                            start, end = start_ends[k]
                            clip_name = clip_names[k]
                            start_offset = o_start - start
                            end_offset = o_end - end
                            self.clip_sentence_pairs_iou.append(
                                (clip_sentence[0], clip_sentence[1], clip_name, start_offset, end_offset))

        self.num_samples_iou = len(self.clip_sentence_pairs_iou)

    def calc_nIoL(self, base, sliding_clip):
        '''Calculate the nIoL of two fragments'''
        A = base.shape[0]
        inter = self.calc_inter(base, sliding_clip)
        sliding_clip = np.expand_dims(sliding_clip, 0).repeat(A, axis=0)
        length = sliding_clip[:, :, 1] - sliding_clip[:, :, 0]
        nIoL = 1 - inter / length

        return nIoL

    def calc_IoU(self, clips_a, clips_b):
        '''Calculate the IoU of two fragments'''
        inter = self.calc_inter(clips_a, clips_b)
        union = self.calc_union(clips_a, clips_b)
        return inter / union

    def calc_inter(self, clips_a, clips_b):
        '''Calculate the intersection of two fragments'''
        A = clips_a.shape[0]
        B = clips_b.shape[0]
        clips_a = np.expand_dims(clips_a, 1).repeat(B, axis=1)
        clips_b = np.expand_dims(clips_b, 0).repeat(A, axis=0)
        max_min = np.maximum(clips_a[:, :, 0], clips_b[:, :, 0])
        min_max = np.minimum(clips_a[:, :, 1], clips_b[:, :, 1])
        return np.maximum(min_max - max_min, 0)

    def calc_union(self, clips_a, clips_b):
        '''Calculate the union of two fragments'''
        A = clips_a.shape[0]
        B = clips_b.shape[0]
        clips_a = np.expand_dims(clips_a, 1).repeat(B, axis=1)
        clips_b = np.expand_dims(clips_b, 0).repeat(A, axis=0)
        min_min = np.minimum(clips_a[:, :, 0], clips_b[:, :, 0])
        max_max = np.maximum(clips_a[:, :, 1], clips_b[:, :, 1])
        return max_max - min_min

    def get_context_window(self, clip_name):
        '''Get the context window of the fragment'''
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        self.context_size = end - start
        left_context_feats = np.zeros([self.context_num, self.visual_dim // 3], dtype=np.float32)
        right_context_feats = np.zeros([self.context_num, self.visual_dim // 3], dtype=np.float32)
        last_left_feat = np.load(os.path.join(self.sliding_dir, clip_name))
        last_right_feat = np.load(os.path.join(self.sliding_dir, clip_name))
        for k in range(self.context_num):
            left_context_start = start - self.context_size * (k + 1)
            left_context_end = start - self.context_size * k
            right_context_start = end + self.context_size * k
            right_context_end = end + self.context_size * (k + 1)
            left_context_name = movie_name + "_" + str(left_context_start) + "_" + str(left_context_end) + ".npy"
            right_context_name = movie_name + "_" + str(right_context_start) + "_" + str(right_context_end) + ".npy"

            left_context_path = os.path.join(self.sliding_dir, left_context_name)
            if os.path.exists(left_context_path):
                left_context_feat = np.load(left_context_path)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat

            right_context_path = os.path.join(self.sliding_dir, right_context_name)
            if os.path.exists(right_context_path):
                right_context_feat = np.load(right_context_path)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat

            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat
        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)

    def __getitem__(self, index):
        '''Return a data'''
        left_context_feat, right_context_feat = self.get_context_window(self.clip_sentence_pairs_iou[index][2])
        feat_path = os.path.join(self.sliding_dir, self.clip_sentence_pairs_iou[index][2])
        featmap = np.load(feat_path)
        vis = np.hstack((left_context_feat, featmap, right_context_feat))

        sent = self.clip_sentence_pairs_iou[index][1][:self.sentence_embed_dim]

        p_offset = self.clip_sentence_pairs_iou[index][3]
        l_offset = self.clip_sentence_pairs_iou[index][4]
        offset = np.array([p_offset, l_offset], dtype=np.float32)
        return np.concatenate((vis, sent)), offset

    def __len__(self):
        '''Return the length of the data set'''
        return self.num_samples_iou


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
