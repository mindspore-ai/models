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
"""
Preprocess the data.
"""
import os
import pickle
import argparse
import numpy as np
import constants as rconst

arg_parser = argparse.ArgumentParser(description='preprocess dataset')
arg_parser.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "ml-20m"])
args, _ = arg_parser.parse_known_args()

_EXPECTED_CACHE_KEYS = (
    rconst.TRAIN_USER_KEY, rconst.TRAIN_ITEM_KEY, rconst.EVAL_USER_KEY,
    rconst.EVAL_ITEM_KEY, rconst.USER_MAP, rconst.ITEM_MAP)

DATASET_TO_NUM_USERS_AND_ITEMS = {
    "ml-1m": (6040, 3706),
    "ml-20m": (138493, 26744)
}

def very_slightly_biased_randint(max_val_vector):
    """very_slightly_biased_randint function"""
    sample_dtype = np.uint64
    out_dtype = max_val_vector.dtype
    samples = np.random.randint(low=0, high=np.iinfo(sample_dtype).max,
                                size=max_val_vector.shape, dtype=sample_dtype)
    return np.mod(samples, max_val_vector.astype(sample_dtype)).astype(out_dtype)

def mask_duplicates(x, axis=1):  # type: (np.ndarray, int) -> np.ndarray
    """Identify duplicates from sampling with replacement.

    Args:
      x: A 2D NumPy array of samples
      axis: The axis along which to de-dupe.

    Returns:
      A NumPy array with the same shape as x with one if an element appeared
      previously along axis 1, else zero.
    """
    if axis != 1:
        raise NotImplementedError

    x_sort_ind = np.argsort(x, axis=1, kind="mergesort")
    sorted_x = x[np.arange(x.shape[0])[:, np.newaxis], x_sort_ind]

    # compute the indices needed to map values back to their original position.
    inv_x_sort_ind = np.argsort(x_sort_ind, axis=1, kind="mergesort")

    # Compute the difference of adjacent sorted elements.
    diffs = sorted_x[:, :-1] - sorted_x[:, 1:]

    # We are only interested in whether an element is zero. Therefore left padding
    # with ones to restore the original shape is sufficient.
    diffs = np.concatenate(
        [np.ones((diffs.shape[0], 1), dtype=diffs.dtype), diffs], axis=1)

    # Duplicate values will have a difference of zero. By definition the first
    # element is never a duplicate.
    return np.where(diffs[np.arange(x.shape[0])[:, np.newaxis],
                          inv_x_sort_ind], 0, 1)

def construct_lookup_variables(pos_users, pos_items, num_users):
    """Lookup variables"""
    index_bounds = None
    sorted_pos_items = None

    def index_segment(user):
        lower, upper = index_bounds[user:user + 2]
        items = sorted_pos_items[lower:upper]

        negatives_since_last_positive = np.concatenate(
            [items[0][np.newaxis], items[1:] - items[:-1] - 1])

        return np.cumsum(negatives_since_last_positive)

    inner_bounds = np.argwhere(pos_users[1:] -
                               pos_users[:-1])[:, 0] + 1
    (upper_bound,) = pos_users.shape
    index_bounds = np.array([0] + inner_bounds.tolist() + [upper_bound])

    # Later logic will assume that the users are in sequential ascending order.
    assert np.array_equal(pos_users[index_bounds[:-1]], np.arange(num_users))

    sorted_pos_items = pos_items.copy()

    for i in range(num_users):
        lower, upper = index_bounds[i:i + 2]
        sorted_pos_items[lower:upper].sort()

    total_negatives = np.concatenate([
        index_segment(i) for i in range(num_users)])

    return total_negatives, index_bounds, sorted_pos_items

def lookup_negative_items(negative_users, total_negatives, index_bounds,
                          sorted_pos_items, num_items):
    """Lookup negative items"""
    output = np.zeros(shape=negative_users.shape, dtype=rconst.ITEM_DTYPE) - 1

    left_index = index_bounds[negative_users]
    right_index = index_bounds[negative_users + 1] - 1

    num_positives = right_index - left_index + 1
    num_negatives = num_items - num_positives
    neg_item_choice = very_slightly_biased_randint(num_negatives)

    # Shortcuts:
    # For points where the negative is greater than or equal to the tally before
    # the last positive point there is no need to bisect. Instead the item id
    # corresponding to the negative item choice is simply:
    #   last_postive_index + 1 + (neg_choice - last_negative_tally)
    # Similarly, if the selection is less than the tally at the first positive
    # then the item_id is simply the selection.
    #
    # Because MovieLens organizes popular movies into low integers (which is
    # preserved through the preprocessing), the first shortcut is very
    # efficient, allowing ~60% of samples to bypass the bisection. For the same
    # reason, the second shortcut is rarely triggered (<0.02%) and is therefore
    # not worth implementing.
    use_shortcut = neg_item_choice >= total_negatives[right_index]
    output[use_shortcut] = (
        sorted_pos_items[right_index] + 1 +
        (neg_item_choice - total_negatives[right_index])
        )[use_shortcut]

    if np.all(use_shortcut):
        # The bisection code is ill-posed when there are no elements.
        return output

    not_use_shortcut = np.logical_not(use_shortcut)
    left_index = left_index[not_use_shortcut]
    right_index = right_index[not_use_shortcut]
    neg_item_choice = neg_item_choice[not_use_shortcut]

    num_loops = np.max(
        np.ceil(np.log2(num_positives[not_use_shortcut])).astype(np.int32))

    for _ in range(num_loops):
        mid_index = (left_index + right_index) // 2
        right_criteria = total_negatives[mid_index] > neg_item_choice
        left_criteria = np.logical_not(right_criteria)

        right_index[right_criteria] = mid_index[right_criteria]
        left_index[left_criteria] = mid_index[left_criteria]

    # Expected state after bisection pass:
    #   The right index is the smallest index whose tally is greater than the
    #   negative item choice index.

    assert np.all((right_index - left_index) <= 1)

    output[not_use_shortcut] = (
        sorted_pos_items[right_index] - (total_negatives[right_index] - neg_item_choice)
        )

    assert np.all(output >= 0)

    return output

def _assemble_eval_batch(users, positive_items, negative_items,
                         users_per_batch):
    """Construct duplicate_mask and structure data accordingly.

    The positive items should be last so that they lose ties. However, they
    should not be masked out if the true eval positive happens to be
    selected as a negative. So instead, the positive is placed in the first
    position, and then switched with the last element after the duplicate
    mask has been computed.

    Args:
      users: An array of users in a batch. (should be identical along axis 1)
      positive_items: An array (batch_size x 1) of positive item indices.
      negative_items: An array of negative item indices.
      users_per_batch: How many users should be in the batch. This is passed
        as an argument so that ncf_test.py can use this method.

    Returns:
      User, item, and duplicate_mask arrays.
    """
    items = np.concatenate([positive_items, negative_items], axis=1)

    # We pad the users and items here so that the duplicate mask calculation
    # will include padding. The metric function relies on all padded elements
    # except the positive being marked as duplicate to mask out padded points.
    if users.shape[0] < users_per_batch:
        pad_rows = users_per_batch - users.shape[0]
        padding = np.zeros(shape=(pad_rows, users.shape[1]), dtype=np.int32)
        users = np.concatenate([users, padding.astype(users.dtype)], axis=0)
        items = np.concatenate([items, padding.astype(items.dtype)], axis=0)

    duplicate_mask = mask_duplicates(items, axis=1).astype(np.float32)

    items[:, (0, -1)] = items[:, (-1, 0)]
    duplicate_mask[:, (0, -1)] = duplicate_mask[:, (-1, 0)]

    assert users.shape == items.shape == duplicate_mask.shape
    return users, items, duplicate_mask

if __name__ == '__main__':

    cache_path = "../data/input/raw_data_cache_py3.pickle"

    # Read data cache input
    valid_cache = os.path.exists(cache_path)
    if valid_cache:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)

        for key in _EXPECTED_CACHE_KEYS:
            if key not in cached_data:
                valid_cache = False

        if not valid_cache:
            print("Invalid raw data cache file.")

    if valid_cache:
        data = cached_data
    # Construct the input of the stream
    train_pos_users = data[rconst.TRAIN_USER_KEY]
    train_pos_items = data[rconst.TRAIN_ITEM_KEY]
    eval_pos_users = data[rconst.EVAL_USER_KEY]
    eval_pos_items = data[rconst.EVAL_ITEM_KEY]

    users_num, items_num = DATASET_TO_NUM_USERS_AND_ITEMS[args.dataset]
    accum_negatives, index_boundary, sorted_train_pos_items = \
        construct_lookup_variables(train_pos_users, train_pos_items, users_num)

    eval_users_per_batch = int(
        rconst.BATCH_SIZE // (1 + rconst.NUM_EVAL_NEGATIVES))

    users_path = "../data/input/tensor_0"
    items_path = "../data/input/tensor_1"
    masks_path = "../data/input/tensor_2"

    if not os.path.exists(users_path):
        os.mkdir(users_path)
    if not os.path.exists(items_path):
        os.mkdir(items_path)
    if not os.path.exists(masks_path):
        os.mkdir(masks_path)

    for i_batch in np.arange(users_num // eval_users_per_batch + 1):
        low_index = i_batch * eval_users_per_batch
        if i_batch == users_num // eval_users_per_batch:
            high_index = users_num
        else:
            high_index = (i_batch + 1) * eval_users_per_batch

        users_tensor = np.repeat(eval_pos_users[low_index:high_index, np.newaxis], \
                          1 + rconst.NUM_EVAL_NEGATIVES, axis=1)
        eval_positive_items = eval_pos_items[low_index:high_index, np.newaxis]
        eval_negative_items = (lookup_negative_items(users_tensor[:, :-1], accum_negatives, \
                          index_boundary, sorted_train_pos_items, items_num) \
                          .reshape(-1, rconst.NUM_EVAL_NEGATIVES))

        users_tensor, items_tensor, masks_tensor = _assemble_eval_batch(
            users_tensor, eval_positive_items, eval_negative_items, eval_users_per_batch)

        users_tensor = np.reshape(users_tensor.flatten(), \
            (1, rconst.BATCH_SIZE)).astype(np.int32)  # (1, self._batch_size), int32
        items_tensor = np.reshape(items_tensor.flatten(), \
            (1, rconst.BATCH_SIZE)).astype(np.int32)  # (1, self._batch_size), int32
        masks_tensor = np.reshape(masks_tensor.flatten(), \
            (1, rconst.BATCH_SIZE)).astype(np.float32)  # (1, self._batch_size), float32

        # save inputs
        users_file = os.path.join(users_path, "batch_{}.txt".format(i_batch))
        items_file = os.path.join(items_path, "batch_{}.txt".format(i_batch))
        masks_file = os.path.join(masks_path, "batch_{}.txt".format(i_batch))

        users_tensor.tofile(users_file)
        items_tensor.tofile(items_file)
        masks_tensor.tofile(masks_file)
