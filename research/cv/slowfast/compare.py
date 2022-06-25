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
Common comparisons.
"""
import argparse

import numpy as np

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)


class Comparison:
    '''Comparison'''
    def __init__(self, expect, pred):
        assert pred.shape == expect.shape, f'expect.expect={expect.shape} is not equal to pred.shape={pred.shape}'
        if pred.dtype == expect.dtype:
            pred = pred.astype(expect.dtype)
        if expect.dtype == bool:
            pred = pred.astype(np.int32)
            expect = expect.astype(np.int32)
        self.pred = pred
        self.expect = expect
        self.shape = expect.shape
        self.size = expect.size
        maximum = np.maximum(np.abs(pred), np.abs(expect))
        self.abs_err = np.abs(pred - expect)
        self.rel_err = np.abs(pred - expect) / np.maximum(np.abs(pred), np.abs(expect))
        self.rel_err[maximum == 0] = 0
        print(f'[__init__] \n'
              f'shape={expect.shape}, size={expect.size}')

    def print_nan(self, details=True, count=32):
        """print nan"""
        rel_err_nan = np.isnan(self.rel_err)
        num_nan = np.count_nonzero(rel_err_nan)
        print(f'[print_nan]:\n'
              f'num_nan={num_nan}, nan_pct={num_nan / self.size}')
        if details:
            cnt = 0
            for i, v in enumerate(rel_err_nan.flatten()):
                if v:
                    print(f'idx={self._get_unflatten_index(i, self.shape)}, '
                          f'val={self.rel_err.flatten()[i]}')
                    cnt += 1
                    if cnt > count:
                        break

    def print_inf(self, details=True, count=32):
        """print inf"""
        rel_err_inf = np.isinf(self.rel_err)
        num_inf = np.count_nonzero(rel_err_inf)
        print(f'[print_inf]:\n'
              f'num_inf={num_inf}, inf_pct={num_inf / self.size}')
        if details:
            cnt = 0
            for i, v in enumerate(rel_err_inf.flatten()):
                if v:
                    print(f'idx={self._get_unflatten_index(i, self.shape)}, val={v}')
                    cnt += 1
                    if cnt > count:
                        break

    def print_unequal_count(self, rtol=0.001, atol=0.001):
        is_np_all_close = np.allclose(self.pred, self.expect, rtol=rtol, atol=atol)
        np_unequal_count = np.count_nonzero(~np.isclose(self.pred, self.expect, rtol=rtol, atol=atol))
        my_unequal_count = np.count_nonzero(np.abs(self.pred - self.expect) > atol + rtol * np.abs(self.expect))
        print(f'[print_unequal_count]: size={self.size}, atol={rtol}, atol={atol}\n'
              f'np.allclose={is_np_all_close}\n'
              f'my_unequal_count={my_unequal_count}, '
              f'my_unequal_pct= {my_unequal_count / self.size}\n'
              f'np_unequal_count={np_unequal_count}, '
              f'np_unequal_pct= {np_unequal_count / self.size}')

    @staticmethod
    def _get_unflatten_index(idx, shape):
        """Get unflatten index"""
        ret = []
        idx_base = np.prod(shape)
        for i in shape:
            idx_base = idx_base // i
            idx_cur = idx // idx_base
            ret.append(idx_cur)
            idx = idx % idx_base
        return ret

    def print_mean_relative_error(self):
        rel_err = self.rel_err
        # filter out nan and info result
        rel_err = rel_err[~np.isnan(rel_err)]
        rel_err = rel_err[~np.isinf(rel_err)]
        print(f'[print_mean_relative_error] mean_relative_error={np.mean(rel_err)}')

    def print_sorted_relative_error(self, count=32, rtol=0.001):
        """print descending relative error w/o nan and inf"""
        rel_err = self.rel_err
        rel_err[np.isnan(rel_err)] = 0
        rel_err[np.isinf(rel_err)] = 0

        print(f'[print_sorted_relative_error] rtol={rtol}, count={count}')
        # default argsort is ascending
        idx = np.argsort(rel_err.flatten() * -1)
        val = rel_err.flatten()[idx]
        cnt = 0
        for i, v in zip(idx, val):
            if v > rtol:
                unflatten_idx = self._get_unflatten_index(i, self.shape)
                print(f'idx={unflatten_idx}, data1={self.pred.flatten()[i]}, '
                      f'data2={self.expect.flatten()[i]}, rel_err={v}')
                cnt += 1
                if cnt >= count:
                    break

    def print_sequential_relative_error(self, rtol=0.001, count=32):
        """Print sequential relative error"""
        rel_err = self.rel_err
        rel_err[np.isnan(rel_err)] = 0
        rel_err[np.isinf(rel_err)] = 0

        print(f'[print_sequential_relative_error] rtol={rtol}, count={count}')
        cnt = 0
        for i, v in enumerate(rel_err.flatten()):
            if v > rtol:
                unflatten_idx = self._get_unflatten_index(i, self.shape)
                print(f'idx={unflatten_idx}, data1={self.pred.flatten()[i]}, '
                      f'data2={self.expect.flatten()[i]}, rel_err={v}')
                cnt += 1
                if cnt >= count:
                    break


def np_compare(expect, pred):
    """NP compare."""
    assert expect.shape == pred.shape, 'the shape of expect and pred should be same '
    # if size of expect is larger than pred, shrink size of expect into the same as pred
    com = Comparison(expect=expect, pred=pred)
    com.print_mean_relative_error()
    com.print_unequal_count()
    com.print_inf()
    com.print_nan()
    com.print_sorted_relative_error()
    com.print_sequential_relative_error()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data compare')
    parser.add_argument('--path1', type=str, required=False, help='file path of data 1,'
                                                                  'the file should be saved by np.save or np.load')
    parser.add_argument('--path2', type=str, required=False, help='file path of data 2,'
                                                                  'the file should be saved by np.save or np.load')
    args = parser.parse_args()

    print("data1 file path: ", args.path1)
    print("data2 file path: ", args.path2)
    data1 = np.random.random((2, 3, 4, 5, 6))
    data2 = np.random.random((2, 3, 4, 5, 6))
    np_compare(data1, data2)
