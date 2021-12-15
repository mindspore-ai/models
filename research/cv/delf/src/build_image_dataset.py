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
"""convert dataset to mindrecord"""
from io import BytesIO
import os
import csv
import argparse
import glob
from PIL import Image
import numpy as np
import pandas as pd

from mindspore.mindrecord import FileWriter

parser = argparse.ArgumentParser(description='MindSpore delf eval')


parser.add_argument('--train_directory', type=str,
                    default="/tmp/", help='Training data directory.')

parser.add_argument('--output_directory', type=str,
                    default="/tmp/", help='Output data directory.')

parser.add_argument('--train_csv_path', type=str,
                    default="/tmp/train.csv", help='Training data csv file path.')

parser.add_argument('--train_clean_csv_path', type=str, default=None,
                    help='(Optional) Clean training data csv file path. ')

parser.add_argument('--num_shards', type=int, default=128,
                    help='Number of shards in output data.')

parser.add_argument(
    '--generate_train_validation_splits', type=bool, default=True, help='(Optional) Whether to split the train dataset')
parser.add_argument(
    '--validation_split_size', type=float, default=0.2, help='(Optional)The size of the VALIDATION split as a fraction')
parser.add_argument('--seed', type=int, default=0,
                    help='(Optional) The seed to be used while shuffling the train')


FLAGS = parser.parse_known_args()[0]

_FILE_IDS_KEY = 'file_ids'
_IMAGE_PATHS_KEY = 'image_paths'
_LABELS_KEY = 'labels'
_TEST_SPLIT = 'test'
_TRAIN_SPLIT = 'train'
_VALIDATION_SPLIT = 'validation'


def _get_all_image_files_and_labels(name, csv_path, image_dir):
    """Process input and get the image file paths, image ids and the labels.

    Args:
        name: 'train' or 'test'.
        csv_path: path to the Google-landmark Dataset csv Data Sources files.
        image_dir: directory that stores downloaded images.
    Returns:
        image_paths: the paths to all images in the image_dir.
        file_ids: the unique ids of images.
        labels: the landmark id of all images. When name='test', the returned labels
            will be an empty list.
    Raises:
        ValueError: if input name is not supported.
    """
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    file_ids = [os.path.basename(os.path.normpath(f))[:-4]
                for f in image_paths]
    if name == _TRAIN_SPLIT:
        csv_file = open(csv_path, 'rb')
        df = pd.read_csv(csv_file)
        df = df.set_index('id')
        labels = [int(df.loc[fid]['landmark_id']) for fid in file_ids]
    elif name == _TEST_SPLIT:
        labels = []
    else:
        raise ValueError('Unsupported dataset split name: %s' % name)
    return image_paths, file_ids, labels


def _get_clean_train_image_files_and_labels(csv_path, image_dir):
    """Get image file paths, image ids and labels for the clean training split.

    Args:
        csv_path: path to the Google-landmark Dataset v2 CSV Data Sources files
                            of the clean train dataset. Assumes CSV header landmark_id;images.
        image_dir: directory that stores downloaded images.

    Returns:
        image_paths: the paths to all images in the image_dir.
        file_ids: the unique ids of images.
        labels: the landmark id of all images.
        relabeling: relabeling rules created to replace actual labels with
        a continuous set of labels.
    """
    # Load the content of the CSV file (landmark_id/label -> images).
    csv_file = open(csv_path, 'rb')
    df = pd.read_csv(csv_file)

    # Create the dictionary (key = file_id, value = {label, file_id}).
    images = {}
    for _, row in df.iterrows():
        label = row['landmark_id']
        for file_id in row['images'].split(' '):
            images[file_id] = {}
            images[file_id]['label'] = label
            images[file_id]['file_id'] = file_id

    # Add the full image path to the dictionary of images.
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    for image_path in image_paths:
        file_id = os.path.basename(os.path.normpath(image_path))[:-4]
        if file_id in images:
            images[file_id]['image_path'] = image_path

    # Explode the dictionary into lists (1 per image attribute).
    image_paths = []
    file_ids = []
    labels = []
    for _, value in images.items():
        image_paths.append(value['image_path'])
        file_ids.append(value['file_id'])
        labels.append(value['label'])

    # Relabel image labels to contiguous values.
    unique_labels = sorted(set(labels))
    relabeling = {label: index for index, label in enumerate(unique_labels)}
    new_labels = [relabeling[label] for label in labels]
    return image_paths, file_ids, new_labels, relabeling


def _process_image(filename):
    """Process a single image file.

    Args:
        filename: string, path to an image file e.g., '/path/to/example.jpg'.

    Returns:
        image_buffer: string, JPEG encoding of RGB image.
    Raises:
        ValueError: if parsed image has wrong number of dimensions or channels.
    """
    white_io = BytesIO()
    Image.open(filename).save(white_io, 'JPEG')
    image_data = white_io.getvalue()

    os.remove(filename)
    return image_data


def _write_mindrecord(output_prefix, image_paths, file_ids, labels):
    """Read image files and write image and label data into MindRecord files.

Args:
    output_prefix: string, the prefix of output files, e.g. 'train'.
    image_paths: list of strings, the paths to images to be converted.
    file_ids: list of strings, the image unique ids.
    labels: list of integers, the landmark ids of images. It is an empty list
        when output_prefix='test'.

Raises:
    ValueError: if the length of input images, ids and labels don't match
"""
    if output_prefix == _TEST_SPLIT:
        labels = [None] * len(image_paths)
    if not len(image_paths) == len(file_ids) == len(labels):
        raise ValueError('length of image_paths, file_ids, labels should be the' +
                         ' same. But they are %d, %d, %d, respectively' %
                         (len(image_paths), len(file_ids), len(labels)))

    output_file = os.path.join(
        FLAGS.output_directory, '%s.mindrecord' % (output_prefix))
    writer = FileWriter(file_name=output_file, shard_num=FLAGS.num_shards)

    cv_schema = {"file_id": {"type": "string"}, "label": {
        "type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema, "GLDv2")
    writer.add_index(["file_id", "label"])

    data = []
    for i in range(len(image_paths)):

        sample = {}

        image_bytes = _process_image(image_paths[i])
        sample['file_id'] = file_ids[i]
        sample['label'] = labels[i]
        sample['data'] = image_bytes

        data.append(sample)
        if i % 10 == 0:
            writer.write_raw_data(data)
            data = []

    if data:
        writer.write_raw_data(data)

    print(writer.commit())


def _write_relabeling_rules(relabeling_rules):
    """Write to a file the relabeling rules when the clean train dataset is used.

    Args:
        relabeling_rules: dictionary of relabeling rules applied when the clean
            train dataset is used (key = old_label, value = new_label).
    """
    relabeling_file_name = os.path.join(
        FLAGS.output_directory, 'relabeling.csv')

    relabeling_file = open(relabeling_file_name, 'w')
    csv_writer = csv.writer(relabeling_file, delimiter=',')
    csv_writer.writerow(['new_label', 'old_label'])
    for old_label, new_label in relabeling_rules.items():
        csv_writer.writerow([new_label, old_label])


def _shuffle_by_columns(np_array, random_state):
    """Shuffle the columns of a 2D numpy array.

    Args:
        np_array: array to shuffle.
        random_state: numpy RandomState to be used for shuffling.
    Returns:
        The shuffled array.
    """
    columns = np_array.shape[1]
    columns_indices = np.arange(columns)
    random_state.shuffle(columns_indices)
    return np_array[:, columns_indices]


def _build_train_and_validation_splits(image_paths, file_ids, labels, validation_split_size, seed):
    """Create TRAIN and VALIDATION splits containing all labels in equal proportion.

    Args:
        image_paths: list of paths to the image files in the train dataset.
        file_ids: list of image file ids in the train dataset.
        labels: list of image labels in the train dataset.
        validation_split_size: size of the VALIDATION split as a ratio of the train
            dataset.
        seed: seed to use for shuffling the dataset for reproducibility purposes.

    Returns:
        splits : tuple containing the TRAIN and VALIDATION splits.
    Raises:
        ValueError: if the image attributes arrays don't all have the same length,
                                which makes the shuffling impossible.
    """
    # Ensure all image attribute arrays have the same length.
    total_images = len(file_ids)
    if not (len(image_paths) == total_images and len(labels) == total_images):
        raise ValueError('Inconsistencies between number of file_ids (%d), number '
                         'of image_paths (%d) and number of labels (%d). Cannot'
                         'shuffle the train dataset.' % (total_images, len(image_paths), len(labels)))

    # Stack all image attributes arrays in a single 2D array of dimensions
    # (3, number of images) and group by label the indices of datapoins in the
    # image attributes arrays. Explicitly convert label types from 'int' to 'str'
    # to avoid implicit conversion during stacking with image_paths and file_ids
    # which are 'str'.
    labels_str = [str(label) for label in labels]
    image_attrs = np.stack((image_paths, file_ids, labels_str))
    image_attrs_idx_by_label = {}
    for index, label in enumerate(labels):
        if label not in image_attrs_idx_by_label:
            image_attrs_idx_by_label[label] = []
        image_attrs_idx_by_label[label].append(index)

    # Create subsets of image attributes by label, shuffle them separately and
    # split each subset into TRAIN and VALIDATION splits based on the size of the
    # validation split.
    splits = {
        _VALIDATION_SPLIT: [],
        _TRAIN_SPLIT: []
    }
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
    for label, indexes in image_attrs_idx_by_label.items():
        # Create the subset for the current label.
        image_attrs_label = image_attrs[:, indexes]
        # Shuffle the current label subset.
        image_attrs_label = _shuffle_by_columns(image_attrs_label, rs)
        # Split the current label subset into TRAIN and VALIDATION splits and add
        # each split to the list of all splits.
        images_per_label = image_attrs_label.shape[1]
        cutoff_idx = max(1, int(validation_split_size * images_per_label))
        splits[_VALIDATION_SPLIT].append(image_attrs_label[:, 0: cutoff_idx])
        splits[_TRAIN_SPLIT].append(image_attrs_label[:, cutoff_idx:])

    # Concatenate all subsets of image attributes into TRAIN and VALIDATION splits
    # and reshuffle them again to ensure variance of labels across batches.
    validation_split = _shuffle_by_columns(
        np.concatenate(splits[_VALIDATION_SPLIT], axis=1), rs)
    train_split = _shuffle_by_columns(
        np.concatenate(splits[_TRAIN_SPLIT], axis=1), rs)

    # Unstack the image attribute arrays in the TRAIN and VALIDATION splits and
    # convert them back to lists. Convert labels back to 'int' from 'str'
    # following the explicit type change from 'str' to 'int' for stacking.
    return (
        {
            _IMAGE_PATHS_KEY: validation_split[0, :].tolist(),
            _FILE_IDS_KEY: validation_split[1, :].tolist(),
            _LABELS_KEY: [int(label)
                          for label in validation_split[2, :].tolist()]
        }, {
            _IMAGE_PATHS_KEY: train_split[0, :].tolist(),
            _FILE_IDS_KEY: train_split[1, :].tolist(),
            _LABELS_KEY: [int(label)
                          for label in train_split[2, :].tolist()]
        })


def _build_train_mindrecord_dataset(
        csv_path, clean_csv_path, image_dir, generate_train_validation_splits, validation_split_size, seed):
    """Build a MindRecord dataset for the train split.

    Args:
        csv_path: path to the train Google-landmark Dataset csv Data Sources files.
        clean_csv_path: path to the Google-landmark Dataset v2 CSV Data Sources
        files of the clean train dataset.
        image_dir: directory that stores downloaded images.
        generate_train_validation_splits: whether to split the test dataset into
            TRAIN and VALIDATION splits.
        validation_split_size: size of the VALIDATION split as a ratio of the train
            dataset. Only used if 'generate_train_validation_splits' is True.
        seed: seed to use for shuffling the dataset for reproducibility purposes.
            Only used if 'generate_train_validation_splits' is True.

    Returns:
        Nothing. After the function call, sharded MindRecord files are materialized.
    Raises:
        ValueError: if the size of the VALIDATION split is outside (0,1) when TRAIN
        and VALIDATION splits need to be generated.
    """
    # Make sure the size of the VALIDATION split is inside (0, 1) if we need to
    # generate the TRAIN and VALIDATION splits.
    if generate_train_validation_splits:
        if validation_split_size <= 0 or validation_split_size >= 1:
            raise ValueError('Invalid VALIDATION split size. Expected inside (0,1)'
                             'but received %f.' % validation_split_size)

    if clean_csv_path:
        # Load clean train images and labels and write the relabeling rules.
        (image_paths, file_ids, labels,
         relabeling_rules) = _get_clean_train_image_files_and_labels(clean_csv_path, image_dir)
        _write_relabeling_rules(relabeling_rules)
    else:
        # Load all train images.
        image_paths, file_ids, labels = _get_all_image_files_and_labels(
            _TRAIN_SPLIT, csv_path, image_dir)

    if generate_train_validation_splits:
        # Generate the TRAIN and VALIDATION splits and write them to MindRecord.
        validation_split, train_split = _build_train_and_validation_splits(
            image_paths, file_ids, labels, validation_split_size, seed)
        _write_mindrecord(
            _VALIDATION_SPLIT, validation_split[_IMAGE_PATHS_KEY],
            validation_split[_FILE_IDS_KEY], validation_split[_LABELS_KEY])

        _write_mindrecord(
            _TRAIN_SPLIT, train_split[_IMAGE_PATHS_KEY],
            train_split[_FILE_IDS_KEY], train_split[_LABELS_KEY])

    else:
        # Write to MindRecord a single split, TRAIN.
        _write_mindrecord(_TRAIN_SPLIT, image_paths, file_ids, labels)


if __name__ == '__main__':
    _build_train_mindrecord_dataset(
        FLAGS.train_csv_path, FLAGS.train_clean_csv_path, FLAGS.train_directory,
        FLAGS.generate_train_validation_splits, FLAGS.validation_split_size, FLAGS.seed)
