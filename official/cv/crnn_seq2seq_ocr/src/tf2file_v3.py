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
"""tfrecord to file."""
import os
import shutil
import tensorflow as tf

def tf_reader(tfrecoed_file, save_image_dir, count=0):

    print('Decoding file: ', tfrecoed_file)

    annots_list = []

    for (_, example) in enumerate(tf.python_io.tf_record_iterator(tfrecoed_file)):

        out = tf.train.Example.FromString(example)

        image = out.features.feature["image/encoded"].bytes_list.value[0]
        annot = out.features.feature["image/text"].bytes_list.value[0].decode("utf-8")

        with open(os.path.join(save_image_dir, str(count) + '.png'), "wb") as img:
            img.write(image)

        annots_list.append(os.path.join(save_image_dir, str(count) + '.png') + '\t' + annot + '\n')

        count += 1

    return annots_list, count


if __name__ == '__main__':

    phase = 'test'

    save_img_dir = '/data/FSNS/data/'
    save_annot_dir = '/data/FSNS/data/'
    tfrecord_dir = '/data/FSNS/tfrecord/'

    save_img_dir = os.path.join(save_img_dir, phase)
    tfrecord_dir = os.path.join(tfrecord_dir, phase)
    tfrecord_files = os.listdir(tfrecord_dir)
    print("TFRecord Dir: ", tfrecord_dir)
    tfrecord_files = [os.path.join(tfrecord_dir, x) for x in tfrecord_files]

    if os.path.exists(save_img_dir):
        shutil.rmtree(save_img_dir)
    os.makedirs(save_img_dir, exist_ok=True)

    cnt = 1

    with open(os.path.join(save_annot_dir, phase + ".txt"), "w") as txt:

        for tfrecord_file in tfrecord_files:

            annots, cnt = tf_reader(tfrecord_file, save_img_dir, cnt)

            for ann in annots:
                txt.write(ann)
