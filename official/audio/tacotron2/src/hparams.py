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
''' configs file '''
from src.text import symbols


class hparams:
    ''' configs '''
    text_cleaners = ['english_cleaners']
    pad_sides = 1
    num_mels = 80  # Number of mel-spectrogram channels and local conditioning dimensionality
    rescale = False  # Whether to rescale audio prior to preprocessing
    rescaling_max = 0.8  # Rescaling value
    num_freq = 1025  # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
    n_fft = 2048 # Extra window size is filled with 0 paddings to match this parameter
    hop_size = 200  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size = 800  # For 16000Hz, 1100 ~= 50 ms (0.05 * sample_rate) (If None, win_size = n_fft)
    sample_rate = 16000  # 16000 Hz (sox --i <filename>)
    magnitude_power = 1.  # The power of the spectrogram magnitude (1. for energy, 2. for power)
    constant_values = 0.

    # M-AILABS (and other datasets) trim params
    # (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
    trim_silence = False  # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    trim_fft_size = 512  # Trimming window size
    trim_hop_size = 128  # Trimmin hop length
    trim_top_db = 40  # Trimming db difference from reference db (smaller==harder trim.)

    # Mel and Linear spectrograms normalization/scaling and clipping
    mel_normalization = True
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization = True  # Only relevant if mel_normalization = True
    symmetric_mels = True
    # Whether to scale the data to be symmetric around 0.
    # (Also multiplies the output range by 2, faster and cleaner convergence)
    max_abs_value = 4.
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max]
    # (Must not be too big to avoid gradient explosion, not too small for fast convergence)

    # Contribution by @begeekmyfriend
    preemphasize = True  # whether to apply filter
    preemphasis = 0.85  # filter coefficient.

    # Limits
    min_level_db = -115
    ref_level_db = 20
    fmin = 0
    fmax = 8000  # To be increased/reduced depending on data.

    # Preprocessing parameters
    frame_length_ms = 50
    frame_shift_ms = 12.5
    preemphasis = 0.85
    min_level_db = -100
    ref_level_db = 20
    power = 1.5
    gl_iters = 100
    hop_length = 256
    win_length = 1024

    # Model Parameters

    symbols_embedding_dim = 512

    # Encoder parameters
    encoder_kernel_size = 5
    encoder_n_convolutions = 3
    encoder_embedding_dim = 512

    # Decoder parameters
    n_frames_per_step = 3
    decoder_rnn_dim = 1024
    prenet_dim = 256
    max_decoder_steps = 1000
    gate_threshold = 0.5
    p_attention_dropout = 0.1
    p_decoder_dropout = 0.1

    # Attention parameters
    attention_rnn_dim = 1024
    attention_dim = 256

    # Location Layer parameters
    attention_location_n_filters = 32
    attention_location_kernel_size = 31

    # Mel-post processing network parameters
    postnet_embedding_dim = 512
    postnet_kernel_size = 5
    postnet_n_convolutions = 5

    lr = 0.002
    epoch_num = 2000
    batch_size = 16
    test_batch_size = 1
    mask_padding = True
    p = 10  # mel spec loss penalty

    max_text_len = 189
    n_symbols = len(symbols)
    fp16_flag = True
