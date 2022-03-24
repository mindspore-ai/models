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
"""
dataset processing.
"""
import ctypes
from pathlib import Path
from typing import Optional

import mindspore.dataset as ds
import numpy as np

_LIB_PATH = Path(__file__).resolve().parent / 'dataset_lib' / 'train_dataset_lib.so'


def _check_dataset_dir(dataset_root: Path):
    """Check that the specified dataset root is an existing directory."""
    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f'The specified dataset root at "{dataset_root}" '
            f'does not exist or is not a directory.'
        )


def _check_dataset_file(dataset_file_path, name):
    """Check that the specified exists"""
    if not dataset_file_path.is_file():
        raise FileNotFoundError(
            f'Cannot find {name} data file "{dataset_file_path}"'
        )


class TrainDatasetLibWrapper:
    """Class for generating corrupted triplets using C++ backend

    Args:
        triplets (numpy.ndarray): Contiguous Numpy array containing triplets (h/r/t) definitions.
        entities_number (int): Number of entities in the dataset.
        relations_number (int): Number of relations in the dataset.
        threads_number (int): Number of threads a C++ wrapper will use for data sampling. Default: 8.
        seed (int): Random seed. Default: 1.
    """

    def __init__(
            self,
            triplets: np.ndarray,
            entities_number: int,
            relations_number: int,
            threads_number: int = 8,
            seed: int = 1,
    ):
        self.lib = ctypes.cdll.LoadLibrary(_LIB_PATH.as_posix())
        self._threads_number = threads_number
        self._entities_number = entities_number
        self._relations_number = relations_number

        self.lib.getBatch.argtypes = [
            ctypes.c_void_p,  # triplets buffer
            ctypes.c_void_p,  # indices
            ctypes.c_int64,  # batch size
            ctypes.c_int64,  # negative sampling rate
            ctypes.c_int64,  # number of threads
        ]

        self.lib.setSeed.argtypes = [ctypes.c_uint32]
        self.lib.loadDataset.argtypes = [
            ctypes.c_int64,  # relations number
            ctypes.c_int64,  # entities number
            ctypes.c_int64,  # triplets number
            ctypes.c_void_p,  # pointer to a source triplets buffer
        ]

        # Initialize the dataset wrapper
        self._load_dataset(triplets, entities_number, relations_number)
        self.set_seed(seed)

    def _load_dataset(
            self,
            triplets: np.ndarray,
            entities_number: int,
            relations_number: int,
    ):
        """Load the dataset into the library buffer"""
        triplets_num = len(triplets)
        triplets = np.ascontiguousarray(triplets).astype(np.int64)
        self.lib.loadDataset(
            relations_number,
            entities_number,
            triplets_num,
            triplets.__array_interface__["data"][0],
        )

    def set_seed(self, seed: int):
        """Set seed of the data generator"""
        self.lib.setSeed(seed)

    def get_batch(
            self,
            batch_size,
            indices_array_pointer,
            neg_sampling_rate,
            dest_triplets_array_pointer,
    ):
        """Load triplets into the numpy array, using the __array_interface__

        Args:
            batch_size (int): Number of triplets indices
            indices_array_pointer: Pointer to the int64 array buffer containing the batch triplets indices.
            neg_sampling_rate (int): Number of corrupted triplets to generate per each selected triplet.
            dest_triplets_array_pointer: Pointer to the int64 array buffer where the triplets will be stored.
        """
        self.lib.getBatch(
            dest_triplets_array_pointer,
            indices_array_pointer,
            batch_size,
            neg_sampling_rate,
            self._threads_number,
        )


class BaseTripletsDataset:
    """Base class for the TripletDataset

    Args:
        dataset_root (str or Path): Dataset root directory
        triplet_file_name (str or Path): Name of the file, containing triplets list
        entities_number (int): Number of entities in the dataset.
        relations_number (int): Number of relations in the dataset.

    Returns:
        Triplets dataset.
    """

    def __init__(
            self,
            dataset_root,
            triplet_file_name,
            entities_number,
            relations_number,
    ):
        self.dataset_root = Path(dataset_root)
        self.triplet_file_path = self.dataset_root / triplet_file_name

        _check_dataset_dir(self.dataset_root)
        _check_dataset_file(self.triplet_file_path, 'triplets')

        self._entities_number = entities_number
        self._relations_number = relations_number

        self._triplets_number, self._triplets = _read_triplets(self.triplet_file_path)

    @property
    def entities_number(self):
        """The number of entities of the dataset"""
        return self._entities_number

    @property
    def relations_number(self):
        """The number of relations of the dataset"""
        return self._relations_number

    @property
    def triplets_number(self):
        """The number of triplets in the dataset"""
        return self._triplets_number

    def __getitem__(self, index):
        """gets the item of the dataset"""
        # Returning a tuple containing a single element (a triplet)
        return (self._triplets[index],)

    def __len__(self):
        """Returns the number of items in the dataset"""
        return self._triplets_number


class TrainTripletsDataset(BaseTripletsDataset):
    """Class for the TripletDataset adapted for training.

    This class additionally provides the negative sampling mechanism.

    Args:
        dataset_root (str or Path): Dataset root directory
        triplet_file_name (str or Path): Name of the file, containing triplets list
        entities_number (int): Number of entities in the dataset.
        relations_number (int): Number of relations in the dataset.
        negative_sampling_rate (int): Negative sampling rate. Default: 0.
        batch_size (int): Number of original triplets in a batch, Default: 1.
            The actual batch size will be batch_size * (1 + negative_sampling_rate)
        group_num (int): Number of grops the dataset is split to. Default: 1
        group_id (int): Dataset group ID, used for selecting the part of the dataset. Default: 0.
        seed (int): Random seed. Default: 1.
        threads_number (int): Number of threads a C++ wrapper will use for data sampling. Default: 8.

    Returns:
        Triplets dataset.
    """

    def __init__(
            self,
            dataset_root,
            triplet_file_name,
            entities_number,
            relations_number,
            negative_sampling_rate=0,
            batch_size=1,
            group_num=1,
            group_id=0,
            seed=1,
            threads_number=8,
    ):
        # Triplets are loaded by the parent class into an array of shape [triplets_num, 3]
        # (The structure of triplets is [[h1, r1, t1], [h2, r2, t2], ...])
        super().__init__(dataset_root, triplet_file_name, entities_number, relations_number)
        self._negative_sampling_rate = negative_sampling_rate

        # This class will manage sampling on its own.
        # We use the additional flag to indicate, that the sampler is initialized.
        self._check_sampling_parameters(batch_size, group_num, group_id, seed, self._triplets_number)
        self._sampling_initialized = False
        self._group_id = group_id
        self._group_num = group_num
        self._batch_size = batch_size
        self._actual_batch_size = self._batch_size * (1 + self._negative_sampling_rate)
        self._number_of_batches_per_group = self._triplets_number // self._group_num // self._batch_size

        # Index, which will be used to shuffle the data by shuffling the index (shape: [triplets_number])
        self._index = np.arange(self._triplets_number, dtype=np.int64)

        # Pre-allocated data containers for faster triplets selection.
        # Shape of Batch container: [batch_size * (1 * neg_rate), 3]
        # Shape of corrupted triplets container: [triplets_number, neg_rate, 3]
        self._batch_container = np.zeros([self._actual_batch_size, 3], np.int64)

        # Random state generator
        self._random_state = np.random.RandomState(seed)

        # Creating a C++ wrapper for faster dataset processing
        self._dataset_wrapper = TrainDatasetLibWrapper(
            self._triplets,
            self._entities_number,
            self._relations_number,
            threads_number=threads_number,
            seed=seed,
        )

    def __getitem__(self, index):
        """Returns the specified triplet of the dataset with the corrupted triplets"""

        if index == 0:
            # Shuffling the index
            self._random_state.shuffle(self._index)

        if index >= self._number_of_batches_per_group:
            raise IndexError

        return self._get_batch(index)

    def __len__(self):
        """Returns the number of items in the dataset"""
        return self._number_of_batches_per_group

    def _get_batch(self, index):
        """Get batch of triplets"""

        index = index + self._number_of_batches_per_group * self._group_id
        index_slice = slice(index * self._batch_size, (index + 1) * self._batch_size)
        batch_indices = self._index[index_slice]

        self._dataset_wrapper.get_batch(
            self._batch_size,
            batch_indices.__array_interface__["data"][0],
            self._negative_sampling_rate,
            self._batch_container.__array_interface__["data"][0]
        )

        # Returning a tuple, containing a single element (array with valid and corrupted triplets)
        return (self._batch_container.astype(np.int32),)

    @staticmethod
    def _check_sampling_parameters(
            batch_size,
            group_num,
            group_id,
            seed,
            dataset_size,
    ):
        """Checks sampling parameters"""
        if not isinstance(batch_size, int):
            raise TypeError(f'Batch size must be an integer value, got {type(batch_size)}.')

        if not isinstance(group_num, int):
            raise TypeError(f'Group number must be an integer value, got {type(group_num)}.')

        if not isinstance(group_id, int):
            raise TypeError(f'Group ID must be an integer value, got {type(group_id)}.')

        if not isinstance(seed, int):
            raise TypeError(f'Seed must be an integer value, got {type(seed)}.')

        if group_num <= 0:
            raise ValueError(f'Group number must be greater than zero, got {group_num}.')

        if group_id < 0 or group_id >= group_num:
            raise ValueError(f'Group ID must be in range [0, {group_num - 1}], got {group_id}.')

        if batch_size <= 0:
            raise ValueError(f'Batch size must be greater than zero, got {batch_size}')

        if batch_size > dataset_size // group_num:
            raise ValueError(
                f'The specified batch size ({batch_size}) is larger than '
                f'the available number of triplets per group ({dataset_size // group_num})'
            )


class TripletsFilter:
    """Triplets filter.

    Allows to exclude false negative results.

    Args:
        dataset_root (str or Path): Dataset root directory
        triplets_files_names (list of str or list of Path): List of the files containing the triplets.
    """

    def __init__(self, dataset_root, triplets_files_names):
        dataset_root = Path(dataset_root)
        _check_dataset_dir(dataset_root)

        if not isinstance(triplets_files_names, list):
            raise TypeError(
                f'Triplets files names must be provided via '
                f'a list, got {type(triplets_files_names)}.'
            )

        triplets_files_paths = [dataset_root / x for x in triplets_files_names]
        for t_file_path in triplets_files_paths:
            _check_dataset_file(t_file_path, 'triplets')

        # Loading the triplets
        hr_t_map = {}
        tr_h_map = {}

        for t_file_path in triplets_files_paths:
            _, triplets = _read_triplets(t_file_path)

            for triplet in triplets:
                h, r, t = map(int, triplet)

                hr_t_map.setdefault(h, {}).setdefault(r, []).append(t)
                tr_h_map.setdefault(t, {}).setdefault(r, []).append(h)

        self._hr_t_map = hr_t_map
        self._tr_h_map = tr_h_map

    def get_related_tails(self, head, relation):
        """Returns the set of tails indices, related to the specified head and relation.

        Args:
            head (int): head index
            relation (int): relation index

        Returns:
            (list) Indices of the related tails
        """
        head = int(head)
        relation = int(relation)
        if head not in self._hr_t_map:
            return []

        return self._hr_t_map[head].get(relation, [])

    def get_related_heads(self, tail, relation):
        """Returns the set of heads indices, related to the specified tail and relation.

        Args:
            tail (int): tail index
            relation (int): relation index

        Returns:
            (list) Indices of the related heads
        """
        tail = int(tail)
        relation = int(relation)
        if tail not in self._tr_h_map:
            return []

        return self._tr_h_map[tail].get(relation, [])


class ValidationTripletsDataset(BaseTripletsDataset):
    """Class for the TripletDataset adapted for validation.

    This class additionally provides the negative sampling mechanism.

    Args:
        dataset_root (str or Path): Dataset root directory
        triplet_file_name (str or Path): Name of the file, containing triplets list
        entities_number (int): Number of entities in the dataset.
        relations_number (int): Number of relations in the dataset.
        triplets_filter (TripletsFilter, optional): Filter for the triplets,
            which will be used to determine if the triplet is corrupted. Default: None.
    """

    def __init__(
            self,
            dataset_root,
            triplet_file_name,
            entities_number,
            relations_number,
            triplets_filter=None,
    ):
        super().__init__(
            dataset_root,
            triplet_file_name,
            entities_number,
            relations_number,
        )

        self._triplets_filter: Optional[TripletsFilter] = triplets_filter
        self._head_batch_container = np.zeros((self._entities_number + 1, 3), dtype=np.int32)
        self._tail_batch_container = np.zeros((self._entities_number + 1, 3), dtype=np.int32)
        self._entities_range = np.arange(self._entities_number, dtype=np.int32)

        self._head_batch_container[self._entities_range, 0] = self._entities_range
        self._tail_batch_container[self._entities_range, 2] = self._entities_range

    def __getitem__(self, index):
        """gets the item of the dataset"""
        triplet = self._triplets[index]

        # The first part of the batch data will contain triplets with corrupted heads.
        # The second part of the batch data will contain triplets with corrupted tails.
        head_batch_data = self._head_batch_container
        tail_batch_data = self._tail_batch_container

        head_batch_data[-1, 0] = triplet[0]
        head_batch_data[:, 1].fill(triplet[1])
        head_batch_data[:, 2].fill(triplet[2])

        tail_batch_data[:, 0].fill(triplet[0])
        tail_batch_data[:, 1].fill(triplet[1])
        tail_batch_data[-1, 2] = triplet[2]

        # # Fill triplets with fake indices

        # Create corruption mask
        corrupted_head_mask = np.ones(self._entities_number + 1, dtype=bool)
        corrupted_tail_mask = np.ones(self._entities_number + 1, dtype=bool)

        if self._triplets_filter is not None:
            not_corrupted_tails = self._triplets_filter.get_related_tails(triplet[0], triplet[1])
            not_corrupted_heads = self._triplets_filter.get_related_heads(triplet[2], triplet[1])

            corrupted_head_mask[not_corrupted_heads] = 0
            corrupted_tail_mask[not_corrupted_tails] = 0

        return head_batch_data, corrupted_head_mask, tail_batch_data, corrupted_tail_mask


def create_train_dataset(
        dataset_root,
        triplet_file_name,
        entities_file_name='entity2id.txt',
        relations_file_name='relation2id.txt',
        negative_sampling_rate=1,
        batch_size=1,
        group_size=1,
        rank=0,
        seed=1,
):
    """Prepare the Mindspore dataset for generating triplets data
    with head, relations, and tails data in placed into separate columns.


    Args:
        dataset_root (str or Path): Dataset root directory
        triplet_file_name (str or Path): Name of the file, containing triplets list
        entities_file_name (str or Path): Name of the file, containing entities definitions.
        relations_file_name (str or Path): Name of the file, containing relations definitions.
        negative_sampling_rate (int): Negative sampling rate. Default 1.
        batch_size (int): Number of original triplets in a batch
            (differs from the actual tensor size). Default 1.
        group_size (int): number of devices (for distributed training). Default 1.
        rank (int): Rank of the process. Default 0.
        seed (int): Random seed value. Default: 1.

    Returns:
        Triplets dataset, number of entities and number of relations.
    """

    entities_number, relations_number = get_number_of_entities_and_relations(
        dataset_root,
        entities_file_name,
        relations_file_name,
    )

    triplet_data_generator = TrainTripletsDataset(
        dataset_root=dataset_root,
        triplet_file_name=triplet_file_name,
        entities_number=entities_number,
        relations_number=relations_number,
        negative_sampling_rate=negative_sampling_rate,
        batch_size=batch_size,
        group_num=group_size,
        group_id=rank,
        seed=seed,
    )

    ms_dataset = ds.GeneratorDataset(
        triplet_data_generator,
        ['batched_triplets'],
        shuffle=False,
        num_parallel_workers=1,
        python_multiprocessing=False,
    )

    return ms_dataset, triplet_data_generator.entities_number, triplet_data_generator.relations_number


def create_evaluation_generator(
        dataset_root,
        triplet_file_name,
        entities_file_name='entity2id.txt',
        relations_file_name='relation2id.txt',
        triplets_filter_files=tuple(),
):
    """Prepare a custom dataset, which produces triplets batches with corrupted heads and tails
    which are suitable for measuring hit@10 metrics.

    Args:
        dataset_root (str or Path): Dataset root directory
        triplet_file_name (str or Path): Name of the file, containing triplets list
        entities_file_name (str or Path): Name of the file, containing entities definitions.
        relations_file_name (str or Path): Name of the file, containing relations definitions.
        triplets_filter_files (tuple): Names of files, which will be used to mark the corrupted triplets only.

    Returns:
        Triplets generator, number of entities and number of relations.
    """

    entities_number, relations_number = get_number_of_entities_and_relations(
        dataset_root,
        entities_file_name,
        relations_file_name,
    )

    if not triplets_filter_files:
        triplets_filter = None
    else:
        triplets_filter = TripletsFilter(
            dataset_root,
            list(triplets_filter_files),
        )

    validation_data_generator = ValidationTripletsDataset(
        dataset_root=dataset_root,
        triplet_file_name=triplet_file_name,
        entities_number=entities_number,
        relations_number=relations_number,
        triplets_filter=triplets_filter,
    )

    return validation_data_generator


def get_number_of_entities_and_relations(
        dataset_root,
        entities_file_name='entity2id.txt',
        relations_file_name='relation2id.txt',
):
    """Get total number of entities from the specified file.

    Args:
        dataset_root (str or Path): Dataset root directory
        entities_file_name (str or Path): Name of the file, containing entities definitions.
        relations_file_name (str or Path): Name of the file, containing relations definitions.

    Returns:
        (tuple of int) Total number of entities and total number of relations.
    """
    dataset_root = Path(dataset_root)
    _check_dataset_dir(dataset_root)

    entities_file_path = dataset_root / entities_file_name
    relations_file_path = dataset_root / relations_file_name

    _check_dataset_file(entities_file_path, 'entities')
    _check_dataset_file(relations_file_path, 'relations')

    try:
        with entities_file_path.open('r') as e_file:
            entities_number = int(e_file.readline())
    except Exception as err:
        raise RuntimeError(f'Unable to read the number of entities from the first line.') from err

    try:
        with relations_file_path.open('r') as r_file:
            relations_number = int(r_file.readline())
    except Exception as err:
        raise RuntimeError(f'Unable to read the number of relations from the first line.') from err

    return entities_number, relations_number


def _read_triplets(triplets_file_path):
    """Read triplets from the specified path to the Numpy array

    Args:
        triplets_file_path (Path): Path to the file containing triplets definitions.

    Returns:
        (tuple) Number of triplets and a Numpy array, containing the read triplets data.
    """
    with triplets_file_path.open('r') as t_file:

        triplets_number = int(t_file.readline())

        htr_triplets = np.fromfile(t_file, dtype=np.int32, sep=' ')  # type: np.ndarray

        if len(htr_triplets) % 3 != 0:
            raise ValueError(
                f'Total number of elements must be a multiple of 3, '
                f'got {len(htr_triplets)}.'
            )
        htr_triplets = htr_triplets.reshape(-1, 3)

        if len(htr_triplets) != triplets_number:
            raise ValueError(
                f'The number of triplets ({len(htr_triplets)}) '
                f'is different from the number specified in the file '
                f'({triplets_number}).'
            )

        if htr_triplets.shape != (triplets_number, 3):
            raise ValueError(
                f'Triplets array must be of shape (triplets_num, 3), '
                f'got {htr_triplets.shape}'
            )

        triplets = htr_triplets[:, [0, 2, 1]]

        return triplets_number, triplets
