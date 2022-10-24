import os
from argparse import ArgumentParser
from src.eval_edge import eval_edge
from src.nms_process import nms_process



def main(args):
    alg = [args.alg]  # algorithms for plotting
    model_name_list = [args.model_name_list]  # model name
    result_dir = os.path.abspath(args.result_dir)  # forward result directory
    save_dir = os.path.abspath(args.save_dir)  # nms result directory
    gt_dir = os.path.abspath(args.gt_dir)  # ground truth directory
    key = args.key  # x = scipy.io.loadmat(filename)[key]
    file_format = args.file_format  # ".mat" or ".npy"
    workers = args.workers  # number workers
    nms_process(model_name_list, result_dir, save_dir, key, file_format)
    eval_edge(alg, model_name_list, save_dir, gt_dir, workers)


if __name__ == '__main__':
    parser = ArgumentParser("edge eval")
    parser.add_argument("--alg", type=str, default="HED", help="algorithms for plotting.")
    parser.add_argument("--model_name_list", type=str, default="hed", help="model name")
    parser.add_argument("--result_dir", type=str, default="./result/hed_result", help="results directory")
    parser.add_argument("--save_dir", type=str, default="./result/hed_eval_result", help="nms result directory")
    parser.add_argument("--gt_dir", type=str, default="../data/BSR/BSDS500/data/groundTruth/test",
                        help="ground truth directory")
    parser.add_argument("--key", type=str, default="result", help="key")
    parser.add_argument("--file_format", type=str, default=".mat", help=".mat or .npy")
    parser.add_argument("--workers", type=int, default="-1", help="number workers, -1 for all workers")
    argps = parser.parse_args()
    main(argps)
