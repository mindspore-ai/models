import os
import argparse
import multiprocessing as mp
import numpy as np

parser = argparse.ArgumentParser(description="metric_learn inference")
parser.add_argument("--data_dir", type=str, required=True, help="data files directory.")
parser.add_argument("--result_dir", type=str, required=True, help="result files directory.")
args = parser.parse_args()

def functtt(param):
    """ fun """
    sharedlist, s, e = param
    fea, a, b = sharedlist
    ab = np.dot(fea[s:e], fea.T)
    d = a[s:e] + b - 2 * ab
    for i in range(e - s):
        d[i][s + i] += 1e8
    sorted_index = np.argsort(d, 1)[:, :10]
    return sorted_index


def recall_topk_parallel(fea, lab, k):
    """ recall_topk_parallel """
    fea = np.array(fea)
    fea = fea.reshape(fea.shape[0], -1)
    n = np.sqrt(np.sum(fea ** 2, 1)).reshape(-1, 1)
    fea = fea / n
    a = np.sum(fea ** 2, 1).reshape(-1, 1)
    b = a.T
    sharedlist = mp.Manager().list()
    sharedlist.append(fea)
    sharedlist.append(a)
    sharedlist.append(b)
    N = 100
    L = fea.shape[0] / N
    params = []
    for i in range(N):
        if i == N - 1:
            s, e = int(i * L), int(fea.shape[0])
        else:
            s, e = int(i * L), int((i + 1) * L)
        params.append([sharedlist, s, e])
    pool = mp.Pool(processes=4)
    sorted_index_list = pool.map(functtt, params)
    pool.close()
    pool.join()
    sorted_index = np.vstack(sorted_index_list)
    res = 0
    for i in range(len(fea)):
        for j in range(k):
            pred = lab[sorted_index[i][j]]
            if lab[i] == pred:
                res += 1.0
                break
    res = res / len(fea)
    return res


def eval_mxbase(data_dir, result_dir):
    print("\nBegin to eval \n")
    TRAIN_LIST = os.path.join(data_dir, "test_half.txt")
    TRAIN_LISTS = open(TRAIN_LIST, "r").readlines()

    # cal_acc
    result_shape = (1, 2048)
    f, l = [], []
    for _, item in enumerate(TRAIN_LISTS):
        items = item.strip().split()
        path = items[0]
        result_bin_path = os.path.join(result_dir, "{}.bin".format(path.split(".")[0]))
        result = np.fromfile(result_bin_path, dtype=np.float32).reshape(result_shape)
        gt = int(items[1]) - 1
        f.append(result)
        l.append(gt)
    f = np.vstack(f)
    l = np.hstack(l)
    recall = recall_topk_parallel(f, l, k=1)
    print("eval_recall:", recall)

if __name__ == '__main__':
    eval_mxbase(args.data_dir, args.result_dir)
    