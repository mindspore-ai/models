import os
import argparse
import json
from pathlib import Path
import numpy as np

FRAME_SIZE = 160
NB_FEATURES = 36
NB_USED_FEATURES = 20
ORDER = 16
RNN_UNITS1 = 384
RNN_UNITS2 = 16

def parse_args(parsers):
    """
    Parse commandline arguments.
    """
    parsers.add_argument('--feature_path', type=Path, default="../data/eval-data", help='input data path')
    parser.add_argument("--output_path", type=Path, default="../data/testing-data", help="output path")
    return parsers


parser = argparse.ArgumentParser(description='lpcnet data process')
parser = parse_args(parser)
args, _ = parser.parse_known_args()
tst_dir = args.feature_path
out_dir = args.output_path
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

def save_data(features_, out_file):
    c = 0
    nb_frames = 1
    feature_chunk_size_ = 500
    features_ = np.reshape(features_, (nb_frames, feature_chunk_size_, NB_FEATURES))
    periods = (.1 + 50 * features_[:, :, 18:19] + 100).astype('int32')
    enc_input1 = features_[c:c + 1, :, :NB_USED_FEATURES].astype("float32").reshape(-1)
    enc_input2 = periods[c:c + 1, :, :].reshape(-1)
    path1 = out_file + "_cfeat.txt"
    path2 = out_file + "_period.txt"
    path3 = out_file + "_feature.txt"
    feat = features_.reshape(-1)
    np.savetxt(path1, enc_input1)
    np.savetxt(path2, enc_input2)
    np.savetxt(path3, feat)

if __name__ == "__main__":
    dct = {}
    loop = 0
    for _f in tst_dir.glob('*.f32'):
        _feature_file = tst_dir / (_f.stem + '.f32')
        features = np.fromfile(_feature_file, dtype='float32')
        features = np.reshape(features, (-1, NB_FEATURES))
        feature_chunk_size = features.shape[0]
        if feature_chunk_size < 500:
            zeros = np.zeros((500 - feature_chunk_size, 36))
            features = np.concatenate((features, zeros), 0)
        else:
            features = features[:500, :]
        dct[loop] = _f.stem
        out_path = str(out_dir) + "/" + str(loop)
        save_data(features, out_path)
        loop += 1
    info_json = json.dumps(dct, sort_keys=False, indent=4, separators=(',', ': '))
    # 显示数据类型
    print(type(info_json))
    f = open('info.json', 'w')
    f.write(info_json)

    # JSON到字典转化
    f2 = open('info.json', 'r')
    info_data = json.load(f2)
    print(info_data)
    # 显示数据类型
    print(type(info_data))
    print("loop = ", loop)
