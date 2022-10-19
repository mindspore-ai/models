# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import math
import json
from pathlib import Path
import librosa
import numpy as np
import mindspore.dataset.engine as de

from src.audio import AudioSegment, SpeedPerturbation
from src.text import _clean_text, punctuation_map

TRAIN_INPUT_PAD_LENGTH = 1500
TRAIN_LABEL_PAD_LENGTH = 360
TEST_INPUT_PAD_LENGTH = 3500


class BaseFeatures():
    """Base class for GPU accelerated audio preprocessing."""
    __constants__ = ["pad_align", "pad_to_max_duration", "max_len"]

    def __init__(self, pad_align, pad_to_max_duration, max_duration,
                 sample_rate, window_size, window_stride):
        super(BaseFeatures, self).__init__()

        self.pad_align = pad_align
        self.pad_to_max_duration = pad_to_max_duration
        self.win_length = int(sample_rate * window_size)  # frame size
        self.hop_length = int(sample_rate * window_stride)

        # Calculate maximum sequence length (# frames)
        if pad_to_max_duration:
            self.max_len = 1 + math.ceil(
                (max_duration * sample_rate - self.win_length) / self.hop_length
            )

    def calculate_features(self, audio, audio_lens):
        return audio, audio_lens

    def __call__(self, audio, audio_lens):
        dtype = audio.dtype
        audio = audio
        feat, feat_lens = self.calculate_features(audio, audio_lens)
        feat = self.apply_padding(feat)
        feat = feat.astype(dtype)
        return feat, feat_lens

    def apply_padding(self, x):
        if self.pad_to_max_duration:
            x_size = max(x.shape[-1], self.max_len)
        else:
            x_size = x.shape[-1]
        if self.pad_align > 0:
            pad_amt = x_size % self.pad_align
        else:
            pad_amt = 0

        padded_len = x_size + (self.pad_align - pad_amt if pad_amt > 0 else 0)
        return np.pad(x, ((0, 0), (0, 0), (0, padded_len - x.shape[-1])))


def normalize_string(s, labels, punct_map):
    """Normalizes string.

    Example:
        'call me at 8:00 pm!' -> 'call me at eight zero pm'
    """
    labels = set(labels)
    try:
        text = _clean_text(s, ["english_cleaners"], punct_map).strip()
        return ''.join([tok for tok in text if all(t in labels for t in tok)])
    except ValueError:
        print(f"WARNING: Normalizing failed: {s}")
        return None


class SpecAugment():
    """Spec augment. refer to https://arxiv.org/abs/1904.08779
    """

    def __init__(self, freq_masks=2, min_freq=0, max_freq=20, time_masks=2,
                 min_time=0, max_time=75):
        super(SpecAugment, self).__init__()
        assert 0 <= min_freq <= max_freq
        assert 0 <= min_time <= max_time

        self.freq_masks = freq_masks
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.time_masks = time_masks
        self.min_time = min_time
        self.max_time = max_time

    def run(self, x):
        sh = x.shape
        mask = np.ones(x.shape, dtype=np.bool)

        for idx in range(sh[0]):
            for _ in range(self.freq_masks):
                w = np.random.randint(
                    self.min_freq, self.max_freq + 1, size=(1,)).item()
                f0 = np.random.randint(0, max(1, sh[1] - w), size=(1,)).item()
                mask[idx, f0:f0 + w] = 0

            for _ in range(self.time_masks):
                w = np.random.randint(
                    self.min_time, self.max_time + 1, size=(1,)).item()
                t0 = np.random.randint(0, max(1, sh[2] - w), size=(1,)).item()
                mask[idx, :, t0:t0 + w] = 0
        x = x * mask
        return x


def normalize_batch(x):
    x_mean = np.zeros((x.shape[0], x.shape[1]), dtype=x.dtype)
    x_std = np.zeros((x.shape[0], x.shape[1]), dtype=x.dtype)
    for i in range(x.shape[0]):
        x_mean[i, :] = x[i, :, :].mean(axis=1)
        x_std[i, :] = x[i, :, :].std(axis=1)
    # make sure x_std is not zero
    x_std += 1e-5
    return (x - np.expand_dims(x_mean, 2)) / np.expand_dims(x_std, 2)


class FilterbankFeatures(BaseFeatures):
    """
    parse audio and transcript
    """

    def __init__(self, sample_rate=16000, window_size=0.02, window_stride=0.01,
                 window="hann", normalize="per_feature", n_fft=512,
                 preemph=0.97, n_filt=64, lowfreq=0, highfreq=None, log=True,
                 dither=1e-5, pad_align=16, pad_to_max_duration=False,
                 max_duration=16.7, frame_splicing=1):
        super(FilterbankFeatures, self).__init__(
            pad_align=pad_align, pad_to_max_duration=pad_to_max_duration,
            max_duration=max_duration, sample_rate=sample_rate,
            window_size=window_size, window_stride=window_stride)

        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.sample_rate = sample_rate
        self.window = window
        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.n_filt = n_filt
        self.preemph = preemph
        highfreq = highfreq or sample_rate / 2
        self.lowfreq = lowfreq
        self.highfreq = highfreq

    def get_seq_len(self, seq_len):
        return np.ceil(seq_len / self.hop_length)

    def calculate_features(self, x):
        dtype = x.dtype
        seq_len = self.get_seq_len(np.asarray([x.shape[0]]))
        # dither
        if self.dither > 0:
            x += self.dither * np.random.randn(x.shape[0])

        # do preemphasis
        if self.preemph is not None:
            x = np.concatenate(
                (np.expand_dims(x[0], 0), x[1:] - self.preemph * x[:-1]), axis=0)

        tmp = librosa.stft(y=x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                           window=self.window, pad_mode='reflect', dtype=np.complex64)

        rel = np.real(tmp)
        img = np.imag(tmp)

        x = np.power(rel, 2) + np.power(img, 2)

        filterbank = np.array(
            librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_filt, fmin=self.lowfreq,
                                fmax=self.highfreq), dtype=np.float32)
        filterbanks = np.expand_dims(filterbank, 0)
        x = np.matmul(filterbanks, x)
        # frame splicing if required
        if self.frame_splicing > 1:
            raise ValueError('Frame splicing not supported')

        x = np.log(x + 1e-20)
        x = normalize_batch(x)
        max_len = x.shape[-1]
        mask = np.arange(max_len, dtype=np.int32)
        mask = np.expand_dims(mask, 0)
        mask = mask < np.expand_dims(seq_len, 1)

        x = x * np.expand_dims(mask, 1)
        x = self.apply_padding(x)
        return x.astype(dtype)


class ASRDataset():
    """
    create ASRDataset
    Args:
        data_dir: Dataset path
        manifest_filepath (str): manifest_file path.
        labels (list): List containing all the possible characters to map to
        normalize: Apply standard mean and deviation Normalization to audio tensor
        batch_size (int): Dataset batch size (default=32)
    """

    def __init__(self, data_dir, manifest_fpaths, labels, batch_size=64, train_mode=True,
                 sample_rate=16000, min_duration=0.1, max_duration=16.7,
                 pad_to_max_duration=False, max_utts=0, normalize_transcripts=True,
                 sort_by_duration=False, trim_silence=True,
                 ignore_offline_speed_perturbation=True):
        self.data_dir = data_dir
        self.labels = labels
        self.labels_map = {labels[i]: i for i in range(len(labels))}
        self.punctuation_map = punctuation_map(labels)
        self.blank_index = len(labels) - 1
        self.pad_to_max_duration = pad_to_max_duration
        self.sort_by_duration = sort_by_duration
        self.max_utts = max_utts
        self.normalize_transcripts = normalize_transcripts
        self.ignore_offline_speed_perturbation = ignore_offline_speed_perturbation
        self.min_duration = min_duration
        self.max_duration = max_duration
        if not train_mode:
            self.max_duration = float("inf")
            self.ignore_offline_speed_perturbation = False
        else:
            batch_size = 1
        self.trim_silence = trim_silence
        self.sample_rate = sample_rate
        perturbations = []
        perturbations.append(SpeedPerturbation())
        self.perturbations = perturbations
        self.max_duration = max_duration

        self.samples = []
        self.duration = 0.0
        self.duration_filtered = 0.0

        for fpath in manifest_fpaths:
            self._load_json_manifest(fpath)
        if sort_by_duration:
            self.samples = sorted(self.samples, key=lambda s: s['duration'])

        ids = self.samples
        self.bins = [ids[i:i + batch_size]
                     for i in range(0, len(ids), batch_size)]
        if len(ids) % batch_size != 0:
            self.bins = self.bins[:-1]
            self.bins.append(ids[-batch_size:])
        self.size = len(self.bins)
        self.batch_size = batch_size
        self.train_feat_proc = FilterbankFeatures()
        self.mask_length = 0
        self.train_mode = train_mode

    def __getitem__(self, index):
        batch_idx = self.bins[index]
        if self.train_mode:
            s = batch_idx[0]
            rn_indx = np.random.randint(len(s['audio_filepath']))
            duration = s['audio_duration'][rn_indx] if 'audio_duration' in s else 0
            offset = s.get('offset', 0)
            segment = AudioSegment(
                s['audio_filepath'][rn_indx], target_sr=self.sample_rate,
                offset=offset, duration=duration, trim=self.trim_silence)
            for p in self.perturbations:
                p.maybe_apply(segment, self.sample_rate)
            segment = segment.samples
            inputs = self.train_feat_proc.calculate_features(segment)
            transcript = np.array(s["transcript"], np.int32)
            return np.array(inputs, np.float32), transcript
        batch_spect = []
        batch_script = []
        for data in batch_idx:
            s = data
            rn_indx = np.random.randint(len(s['audio_filepath']))
            duration = s['audio_duration'][rn_indx] if 'audio_duration' in s else 0
            offset = s.get('offset', 0)
            segment = AudioSegment(
                s['audio_filepath'][rn_indx], target_sr=self.sample_rate,
                offset=offset, duration=duration, trim=self.trim_silence)
            segment = segment.samples
            inputs = self.train_feat_proc.calculate_features(segment)
            inputs = np.squeeze(inputs, 0)
            batch_spect.append(inputs)
            batch_script.append(np.array(s["transcript"], np.int32))
        batch_size = len(batch_idx)
        input_length = np.zeros(batch_size, np.float32)
        target_indices = []
        frez = inputs.shape[0]
        inputs = np.zeros(
            (batch_size, frez, TEST_INPUT_PAD_LENGTH), dtype=np.float32)
        targets = []
        for k, spect_, scripts_ in zip(range(batch_size), batch_spect, batch_script):
            seq_length = np.shape(spect_)[1]
            input_length[k] = seq_length
            targets.extend(scripts_)
            for m in range(len(scripts_)):
                target_indices.append([k, m])
            inputs[k, :, 0:seq_length] = spect_
        return inputs, input_length, np.array(target_indices, dtype=np.int64), np.array(targets, dtype=np.int32)

    def __len__(self):
        return self.size

    def _load_json_manifest(self, fpath):
        for s in json.load(open(fpath, "r", encoding="utf-8")):

            if self.pad_to_max_duration and not self.ignore_offline_speed_perturbation:
                # require all perturbed samples to be < self.max_duration
                s_max_duration = max(f['duration'] for f in s['files'])
            else:
                # otherwise we allow perturbances to be > self.max_duration
                s_max_duration = s['original_duration']

            s['duration'] = s.pop('original_duration')
            if not self.min_duration <= s_max_duration <= self.max_duration:
                self.duration_filtered += s['duration']
                continue

            # Prune and normalize according to transcript
            tr = (s.get('transcript', None)
                  or self.load_transcript(s['text_filepath']))

            if not isinstance(tr, str):
                print(f'WARNING: Skipped sample (transcript not a str): {tr}.')
                self.duration_filtered += s['duration']
                continue

            if self.normalize_transcripts:
                tr = normalize_string(tr, self.labels, self.punctuation_map)
            s["transcript"] = self.to_vocab_inds(tr)

            files = s.pop('files')
            if self.ignore_offline_speed_perturbation:
                files = [f for f in files if f['speed'] == 1.0]

            s['audio_duration'] = [f['duration'] for f in files]
            s['audio_filepath'] = [str(Path(self.data_dir, f['fname']))
                                   for f in files]
            self.samples.append(s)
            self.duration += s['duration']

            if self.max_utts > 0 and len(self.samples) >= self.max_utts:
                print(
                    f'Reached max_utts={self.max_utts}. Finished parsing {fpath}.')
                break

    def load_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding="utf-8") as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        return transcript

    def to_vocab_inds(self, transcript):
        chars = [self.labels_map.get(x, self.blank_index)
                 for x in list(transcript)]
        transcript = list(filter(lambda x: x != self.blank_index, chars))
        return transcript


def preprocess(batch_spect, batch_script, blank_index):
    specAugment = SpecAugment()
    x = specAugment.run(batch_spect)
    batch_spect = np.squeeze(x, 0)
    frez = batch_spect.shape[0]
    # 1501 is the max length in train dataset(LibriSpeech).
    # The length is fixed to this value because Mindspore does not support dynamic shape currently
    inputs = np.zeros((frez, TRAIN_INPUT_PAD_LENGTH), dtype=np.float32)
    # The target length is fixed to this value because Mindspore does not support dynamic shape currently
    # 350 may be greater than the max length of labels in train dataset(LibriSpeech).
    targets = np.ones((TRAIN_LABEL_PAD_LENGTH), dtype=np.int32) * blank_index
    seq_length = batch_spect.shape[1]
    script_length = batch_script.shape[0]
    targets[:script_length] = batch_script
    if seq_length <= TRAIN_INPUT_PAD_LENGTH:
        input_length = seq_length
        inputs[:, :seq_length] = batch_spect
    else:
        maxstart = seq_length - TRAIN_INPUT_PAD_LENGTH
        start = np.random.randint(maxstart)
        input_length = TRAIN_INPUT_PAD_LENGTH
        inputs[:, :] = batch_spect[:, start:start + TRAIN_INPUT_PAD_LENGTH]
    return inputs, np.array(input_length, dtype=np.float32), np.array(targets, dtype=np.int32)


def postprocess(inputs, input_length, targets):
    batch_size = inputs.shape[0]
    target_indices = []
    for b in range(batch_size):
        for m in range(TRAIN_LABEL_PAD_LENGTH):
            target_indices.append([b, m])
    targets = np.reshape(targets, (-1,))
    return inputs, input_length, np.array(target_indices, dtype=np.int64), targets


def create_train_dataset(mindrecord_files, labels, batch_size, train_mode, rank=None, group_size=None):
    """
    create train dataset

    Args:
        mindrecord_files (list): A list of mindrecord files
        labels (list): list containing all the possible characters to map to
        batch_size (int): Dataset batch size
        train_mode (bool): Whether dataset is use for train or eval (default=True).
        rank (int): The shard ID within num_shards (default=None).
        group_size (int): Number of shards that the dataset should be divided into (default=None).

    Returns:
        Dataset.
    """

    output_columns = ["batch_spect", "batch_script"]
    ds = de.MindDataset(mindrecord_files, columns_list=output_columns, num_shards=group_size, shard_id=rank,
                        num_parallel_workers=4, shuffle=train_mode)

    compose_map_func = (lambda batch_spect, batch_script: preprocess(
        batch_spect, batch_script, len(labels) - 1))
    ds = ds.map(operations=compose_map_func, input_columns=["batch_spect", "batch_script"],
                output_columns=["inputs", "input_length", "targets"],
                num_parallel_workers=8)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(operations=postprocess, input_columns=["inputs", "input_length", "targets"],
                output_columns=["inputs", "input_length", "target_indices", "targets"])
    return ds


def create_eval_dataset(data_dir, manifest_filepath, labels, batch_size, train_mode):
    """
    create train dataset

    Args:
        data_dir (str): Dataset path
        manifest_filepath (str): manifest_file path.
        labels (list): list containing all the possible characters to map to
        batch_size (int): Dataset batch size
        train_mode (bool): Whether dataset is use for train or eval (default=True).
        rank (int): The shard ID within num_shards (default=None).
        group_size (int): Number of shards that the dataset should be divided into (default=None).

    Returns:
        Dataset.
    """
    dataset = ASRDataset(data_dir=data_dir, manifest_fpaths=manifest_filepath, labels=labels,
                         batch_size=batch_size, train_mode=train_mode)
    ds = de.GeneratorDataset(
        dataset, ["inputs", "input_length", "target_indices", "targets"])
    return ds
