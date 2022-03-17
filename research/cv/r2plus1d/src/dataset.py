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
"""Prepare dataset"""
import os
import random
import numbers
import warnings
import cv2
import numpy as np
from PIL import Image

import mindspore.dataset as ds

def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow

class VideoClsDataset:
    """Load your own video classification dataset."""

    def __init__(self, dataset_root_path, dataset_name="ucf101", mode='train', clip_len=16,
                 frame_sample_rate=2, crop_size=112, short_side_size=128,
                 new_height=128, new_width=171, keep_aspect_ratio=False,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3):
        self.dataset_root_path = dataset_root_path
        self.dataset_name = dataset_name
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.dataset_samples, self.label_array = self.get_data_and_labels()
        if mode == 'test':
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def random_horizontal_flip(self, clip):
        '''
        horizontal flip the image randomly, rate: 0.5
        '''
        if random.random() < 0.5:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            if isinstance(clip[0], Image.Image):
                return [
                    img.transpose(Image.FLIP_LEFT_RIGHT) for img in clip
                ]
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(clip[0])))
        return clip

    def center_crop(self, clip, size):
        """
        center_crop
        """
        if isinstance(size, numbers.Number):
            size = (size, size)
        h, w = size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, _ = clip[0].shape
        elif isinstance(clip[0], Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        if isinstance(clip[0], np.ndarray):
            cropped = [img[y1:y1 + h, x1:x1 + w, :] for img in clip]

        elif isinstance(clip[0], Image.Image):
            cropped = [
                img.crop((x1, y1, x1 + w, y1 + h)) for img in clip
            ]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return cropped

    def random_crop(self, clip, size):
        """
        random_crop
        """
        if isinstance(size, numbers.Number):
            size = (size, size)
        h, w = size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, _ = clip[0].shape
        elif isinstance(clip[0], Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)

        if isinstance(clip[0], np.ndarray):
            cropped = [img[y1:y1 + h, x1:x1 + w, :] for img in clip]

        elif isinstance(clip[0], Image.Image):
            cropped = [
                img.crop((x1, y1, x1 + w, y1 + h)) for img in clip
            ]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return cropped

    def random_resize_clip(self, clip, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        '''
        random_resize_clip
        '''
        scaling_factor = random.uniform(ratio[0], ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, _ = clip[0].shape
        elif isinstance(clip[0], Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)

        return self.resize_clip(clip, new_size, interpolation)

    def resize_clip(self, clip, size, interpolation='bilinear'):
        '''
        resize the clip
        '''
        if isinstance(clip[0], np.ndarray):
            if isinstance(size, numbers.Number):
                im_h, im_w, _ = clip[0].shape
                # Min spatial dim already matches minimal size
                if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                       and im_h == size):
                    return clip
                new_h, new_w = get_resize_sizes(im_h, im_w, size)
                size = (new_w, new_h)
            else:
                size = size[0], size[1]
            if interpolation == 'bilinear':
                np_inter = cv2.INTER_LINEAR
            else:
                np_inter = cv2.INTER_NEAREST
            scaled = [
                cv2.resize(img, size, interpolation=np_inter) for img in clip
            ]
        elif isinstance(clip[0], Image.Image):
            if isinstance(size, numbers.Number):
                im_w, im_h = clip[0].size
                # Min spatial dim already matches minimal size
                if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                       and im_h == size):
                    return clip
                new_h, new_w = get_resize_sizes(im_h, im_w, size)
                size = (new_w, new_h)
            else:
                size = size[1], size[0]
            if interpolation == 'bilinear':
                pil_inter = Image.BILINEAR
            else:
                pil_inter = Image.NEAREST
            scaled = [img.resize(size, pil_inter) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return scaled

    def __getitem__(self, index):
        if self.mode == 'train':
            scale_t = 1

            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)
            if not buffer.any():
                while not buffer.any():
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)
            buffer = self.resize_clip(buffer, self.short_side_size, "bilinear")
            buffer = self.random_resize_clip(buffer, ratio=(1, 1.25), interpolation='bilinear')
            buffer = self.random_crop(buffer, (int(self.crop_size), int(self.crop_size)))
            buffer = self.random_horizontal_flip(buffer)
            buffer = self.clipToTensor(np.array(buffer))
            buffer = self.normalize(buffer, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            return buffer.astype(np.float32), self.label_array[index]

        if self.mode == 'val':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if not buffer.any():
                while not buffer.any():
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.resize_clip(buffer, self.short_side_size, "bilinear")
            buffer = self.center_crop(buffer, size=(self.crop_size, self.crop_size))
            buffer = self.clipToTensor(np.array(buffer))
            buffer = self.normalize(buffer, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            return buffer.astype(np.float32), self.label_array[index]

        if self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)
            #print()
            while not buffer.any():
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)

            buffer = self.resize_clip(buffer, self.short_side_size, "bilinear")
            buffer = self.center_crop(buffer, size=(self.crop_size, self.crop_size))
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                 / (self.test_num_crop - 1)
            temporal_step = max(1.0 * (buffer.shape[0] - self.clip_len) \
                                / (self.test_num_segment - 1), 0)
            temporal_start = int(chunk_nb * temporal_step)
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.clipToTensor(buffer)
            buffer = self.normalize(buffer, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            return buffer.astype(np.float32), self.test_label_array[index]
        raise NameError('mode {} unkown'.format(self.mode))

    def normalize(self, buffer, mean, std):
        for i in range(3):
            buffer[i] = (buffer[i] - mean[i]) / std[i]
        return buffer

    def clipToTensor(self, buffer):
        #m (H x W x C) --> #(C x m x H x W)
        return buffer.transpose((3, 0, 1, 2)) / 255.0

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """
        load images by cv2
        if mode == 'test', we return a whole list, it will be selected by chunk_nb, split_nb in __getitem__;
        otherwise, we select 'num_segment' list segment, other images in list will be discarded
        """
        frames = sorted([os.path.join(sample, img) for img in os.listdir(sample)])
        frame_count = len(frames)
        frame_list = np.empty((frame_count, self.new_height, self.new_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            frame_list[i] = frame

        if self.mode == 'test':
            all_index = [x for x in range(0, frame_count, self.frame_sample_rate)]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])
            buffer = frame_list[all_index]
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = frame_count // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        buffer = frame_list[all_index]
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        return len(self.test_dataset)

    def get_data_and_labels(self):
        myclass = self.classes
        dataset_samples, label_array = [], []
        split_class = "train" if self.mode == "train" else "val"
        for index, category in enumerate(sorted(os.listdir(os.path.join(self.dataset_root_path, split_class)))):
            assert category in myclass, str(category)+ " not belong to " + str(self.dataset_name)
            for video in sorted(os.listdir(os.path.join(self.dataset_root_path, split_class, category))):
                dataset_samples.append(os.path.join(self.dataset_root_path, split_class, category, video))
                label_array.append(index)
        return dataset_samples, label_array

    @property
    def classes(self):
        """Category names."""
        assert self.dataset_name in ["kinetics400", "ucf101"], \
               "The 'dataset_name' should be either 'ucf101' or 'kinetics400' "
        if self.dataset_name == "ucf101":
            return ucf101_label_names
        return kinetics400_label_names

def create_VideoDataset(dataset_root_path, dataset_name="ucf101", mode='train', clip_len=16,\
                        batch_size=8, device_num=1, rank=0, shuffle=True):
    '''create_VideoDataset
    dataset_root_path: string, root path of dataset
    dataset_name: string, either 'ucf101' or 'kinetics400'
    mode: string, one of ["train", 'val', 'test']
    '''
    dataset = VideoClsDataset(dataset_root_path=dataset_root_path,
                              dataset_name=dataset_name,
                              mode=mode,
                              clip_len=clip_len)
    data_set = ds.GeneratorDataset(dataset, column_names=["image", "label"], num_parallel_workers=4, \
                                   shuffle=shuffle, num_shards=device_num, shard_id=rank)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set, data_set.get_dataset_size()

ucf101_label_names = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', \
                      'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', \
                      'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', \
                      'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', \
                      'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', \
                      'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', \
                      'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'HammerThrow', \
                      'Hammering', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', \
                      'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', \
                      'JugglingBalls', 'JumpRope', 'JumpingJack', 'Kayaking', 'Knitting', \
                      'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', \
                      'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', \
                      'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', \
                      'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', \
                      'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', \
                      'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', \
                      'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', \
                      'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', \
                      'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', \
                      'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', \
                      'YoYo']

kinetics400_label_names = ["abseiling", "air drumming", "answering questions", \
                           "applauding", "applying cream", "archery", "arm wrestling", \
                           "arranging flowers", "assembling computer", "auctioning", "baby waking up", \
                           "baking cookies", "balloon blowing", "bandaging", "barbequing", "bartending", \
                           "beatboxing", "bee keeping", "belly dancing", "bench pressing", \
                           "bending back", "bending metal", "biking through snow", "blasting sand", \
                           "blowing glass", "blowing leaves", "blowing nose", "blowing out candles", \
                           "bobsledding", "bookbinding", "bouncing on trampoline", "bowling", \
                           "braiding hair", "breading or breadcrumbing", "breakdancing", \
                           "brush painting", "brushing hair", "brushing teeth", "building cabinet", \
                           "building shed", "bungee jumping", "busking", "canoeing or kayaking", \
                           "capoeira", "carrying baby", "cartwheeling", "carving pumpkin", \
                           "catching fish", "catching or throwing baseball", \
                           "catching or throwing frisbee", "catching or throwing softball", "celebrating", \
                           "changing oil", "changing wheel", "checking tires", "cheerleading", \
                           "chopping wood", "clapping", "clay pottery making", "clean and jerk", \
                           "cleaning floor", "cleaning gutters", "cleaning pool", "cleaning shoes", \
                           "cleaning toilet", "cleaning windows", "climbing a rope", "climbing ladder", \
                           "climbing tree", "contact juggling", "cooking chicken", \
                           "cooking egg", "cooking on campfire", "cooking sausages", \
                           "counting money", "country line dancing", "cracking neck", "crawling baby", \
                           "crossing river", "crying", "curling hair", "cutting nails", \
                           "cutting pineapple", "cutting watermelon", "dancing ballet", \
                           "dancing charleston", "dancing gangnam style", "dancing macarena", \
                           "deadlifting", "decorating the christmas tree", "digging", "dining", \
                           "disc golfing", "diving cliff", "dodgeball", "doing aerobics", "doing laundry", \
                           "doing nails", "drawing", "dribbling basketball", "drinking", \
                           "drinking beer", "drinking shots", "driving car", "driving tractor", \
                           "drop kicking", "drumming fingers", "dunking basketball", "dying hair", \
                           "eating burger", "eating cake", "eating carrots", "eating chips", \
                           "eating doughnuts", "eating hotdog", "eating ice cream", \
                           "eating spaghetti", "eating watermelon", "egg hunting", \
                           "exercising arm", "exercising with an exercise ball", \
                           "extinguishing fire", "faceplanting", "feeding birds", "feeding fish", \
                           "feeding goats", "filling eyebrows", "finger snapping", "fixing hair", \
                           "flipping pancake", "flying kite", "folding clothes", "folding napkins", \
                           "folding paper", "front raises", "frying vegetables", "garbage collecting", \
                           "gargling", "getting a haircut", "getting a tattoo", \
                           "giving or receiving award", "golf chipping", "golf driving", "golf putting", \
                           "grinding meat", "grooming dog", "grooming horse", "gymnastics tumbling", \
                           "hammer throw", "headbanging", "headbutting", "high jump", "high kick", \
                           "hitting baseball", "hockey stop", "holding snake", "hopscotch", "hoverboarding", \
                           "hugging", "hula hooping", "hurdling", "hurling (sport)", "ice climbing", \
                           "ice fishing", "ice skating", "ironing", "javelin throw", "jetskiing", \
                           "jogging", "juggling balls", "juggling fire", \
                           "juggling soccer ball", "jumping into pool", "jumpstyle dancing", \
                           "kicking field goal", "kicking soccer ball", "kissing", "kitesurfing", "knitting", \
                           "krumping", "laughing", "laying bricks", "long jump", "lunge", \
                           "making a cake", "making a sandwich", "making bed", "making jewelry", \
                           "making pizza", "making snowman", "making sushi", "making tea", "marching", \
                           "massaging back", "massaging feet", "massaging legs", \
                           "massaging person's head", "milking cow", "mopping floor", "motorcycling", \
                           "moving furniture", "mowing lawn", "news anchoring", "opening bottle", \
                           "opening present", "paragliding", "parasailing", "parkour", \
                           "passing American football (in game)", "passing American football (not in game)", \
                           "peeling apples", "peeling potatoes", "petting animal (not cat)", \
                           "petting cat", "picking fruit", "planting trees", "plastering", \
                           "playing accordion", "playing badminton", "playing bagpipes", \
                           "playing basketball", "playing bass guitar", "playing cards", "playing cello", \
                           "playing chess", "playing clarinet", "playing controller", \
                           "playing cricket", "playing cymbals", "playing didgeridoo", "playing drums", \
                           "playing flute", "playing guitar", "playing harmonica", \
                           "playing harp", "playing ice hockey", "playing keyboard", \
                           "playing kickball", "playing monopoly", "playing organ", "playing paintball", \
                           "playing piano", "playing poker", "playing recorder", \
                           "playing saxophone", "playing squash or racquetball", "playing tennis", \
                           "playing trombone", "playing trumpet", "playing ukulele", \
                           "playing violin", "playing volleyball", "playing xylophone", \
                           "pole vault", "presenting weather forecast", "pull ups", "pumping fist", \
                           "pumping gas", "punching bag", "punching person (boxing)", "push up", \
                           "pushing car", "pushing cart", "pushing wheelchair", "reading book", \
                           "reading newspaper", "recording music", "riding a bike", "riding camel", \
                           "riding elephant", "riding mechanical bull", "riding mountain bike", \
                           "riding mule", "riding or walking with horse", "riding scooter", \
                           "riding unicycle", "ripping paper", "robot dancing", "rock climbing", \
                           "rock scissors paper", "roller skating", "running on treadmill", "sailing", \
                           "salsa dancing", "sanding floor", "scrambling eggs", "scuba diving", \
                           "setting table", "shaking hands", "shaking head", "sharpening knives", \
                           "sharpening pencil", "shaving head", "shaving legs", "shearing sheep", \
                           "shining shoes", "shooting basketball", "shooting goal (soccer)", \
                           "shot put", "shoveling snow", "shredding paper", "shuffling cards", \
                           "side kick", "sign language interpreting", "singing", "situp", \
                           "skateboarding", "ski jumping", "skiing (not slalom or crosscountry)", \
                           "skiing crosscountry", "skiing slalom", "skipping rope", \
                           "skydiving", "slacklining", "slapping", "sled dog racing", "smoking", \
                           "smoking hookah", "snatch weight lifting", "sneezing", "sniffing", "snorkeling", \
                           "snowboarding", "snowkiting", "snowmobiling", "somersaulting", \
                           "spinning poi", "spray painting", "spraying", "springboard diving", \
                           "squat", "sticking tongue out", "stomping grapes", \
                           "stretching arm", "stretching leg", "strumming guitar", "surfing crowd", \
                           "surfing water", "sweeping floor", "swimming backstroke", \
                           "swimming breast stroke", "swimming butterfly stroke", "swing dancing", \
                           "swinging legs", "swinging on something", "sword fighting", "tai chi", \
                           "taking a shower", "tango dancing", "tap dancing", "tapping guitar", \
                           "tapping pen", "tasting beer", "tasting food", "testifying", "texting", \
                           "throwing axe", "throwing ball", "throwing discus", "tickling", "tobogganing", \
                           "tossing coin", "tossing salad", "training dog", "trapezing", \
                           "trimming or shaving beard", "trimming trees", "triple jump", \
                           "tying bow tie", "tying knot (not on a tie)", "tying tie", "unboxing", \
                           "unloading truck", "using computer", \
                           "using remote controller (not gaming)", "using segway", "vault", "waiting in line", \
                           "walking the dog", "washing dishes", "washing feet", "washing hair", \
                           "washing hands", "water skiing", "water sliding", "watering plants", \
                           "waxing back", "waxing chest", "waxing eyebrows", "waxing legs", "weaving basket", \
                           "welding", "whistling", "windsurfing", "wrapping present", "wrestling", "writing", \
                           "yawning", "yoga", "zumba"]
