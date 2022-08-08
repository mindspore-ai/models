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

import string
import src.cleaners as cleaners


def _clean_text(text, cleaner_names, *args):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text, *args)
    return text


def punctuation_map(labels):
    # Punctuation to remove
    punctuation = string.punctuation
    punctuation = punctuation.replace("+", "")
    punctuation = punctuation.replace("&", "")
    # TODO We might also want to consider:
    # @ -> at
    # # -> number, pound, hashtag
    # ~ -> tilde
    # _ -> underscore
    # % -> percent
    # If a punctuation symbol is inside our vocab, we do not remove from text
    for l in labels:
        punctuation = punctuation.replace(l, "")
    # Turn all punctuation to whitespace
    table = str.maketrans(punctuation, " " * len(punctuation))
    return table
