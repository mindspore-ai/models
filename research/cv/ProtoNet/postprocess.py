'''
calculate the accuracy using the infer result which are binary files
'''
import os
import argparse
import numpy as np
from loss_for_infer import calculate_loss

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', default=None, help='Location of result.')
parser.add_argument('--label_classes_path', default=None, help='Location of label and classes.')
parser.add_argument('--classes_per_it_val', type=int,
                    help='number of random classes per episode for validation, default=5', default=5)
parser.add_argument('--num_support_val', type=int,
                    help='number of samples per class to use as support for validation, default=5', default=5)
parser.add_argument('--num_query_val', type=int,
                    help='number of samples per class to use as query for validation, default=15', default=15)

def get_result(options):
    '''
    calculate the acc
    '''
    files = os.listdir(options.result_path)
    acc = list()
    loss = list()
    for file in files:
        result_file_name = file
        num = result_file_name.split('_')[1]

        result_file_path = options.result_path + os.sep + result_file_name
        label_file_path = options.label_classes_path + os.sep + 'label_' + str(num)
        classes_file_path = options.label_classes_path + os.sep + 'classes_' + str(num)

        output = np.fromfile(result_file_path, dtype=np.float32)
        label = np.fromfile(label_file_path, dtype=np.int32)
        classes = np.fromfile(classes_file_path, dtype=np.int32)
        batch_size = (options.num_support_val + options.num_query_val) * options.classes_per_it_val
        # 64 is the fixed output dimension of the model
        output = np.reshape(output, (batch_size, 64))

        acc_val, loss_val = calculate_loss(output, label, classes, options.num_support_val,
                                           options.num_query_val, options.classes_per_it_val, is_train=False)
        acc.append(acc_val)
        loss.append(loss_val)
    mean_acc = sum(acc) / len(acc)
    mean_loss = sum(loss) / len(loss)
    print("accuracy: {}  loss:{}".format(mean_acc, mean_loss))

if __name__ == '__main__':
    options_ = parser.parse_args()
    get_result(options_)
