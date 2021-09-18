'''
preprocess the source data and generate the result data with binary file
'''
import os
import argparse
from model_init import init_dataloader
from mindspore import dataset as ds


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default=None, help='Location of data.')
parser.add_argument('--data_output_path', default=None, help='Location of converted data.')
parser.add_argument('--label_classses_output_path', default=None,
                    help='Location of converted label and classes.')
parser.add_argument('-its', '--iterations', type=int, help='number of episodes per epoch, default=100',
                    default=100)
parser.add_argument('-cTr', '--classes_per_it_tr', type=int,
                    help='number of random classes per episode for training, default=60', default=20)
parser.add_argument('-nsTr', '--num_support_tr', type=int,
                    help='number of samples per class to use as support for training, default=5', default=5)
parser.add_argument('-nqTr', '--num_query_tr', type=int,
                    help='number of samples per class to use as query for training, default=5', default=5)
parser.add_argument('-cVa', '--classes_per_it_val', type=int,
                    help='number of random classes per episode for validation, default=5', default=5)
parser.add_argument('-nsVa', '--num_support_val', type=int,
                    help='number of samples per class to use as support for validation, default=5', default=5)
parser.add_argument('-nqVa', '--num_query_val', type=int,
                    help='number of samples per class to use as query for validation, default=15', default=15)

def convert_img_to_bin(options_, root, output_path, label_classses_path):
    '''
    convert the image to binary file
    '''
    val_dataloader = init_dataloader(options_, 'val', root)
    inp = ds.GeneratorDataset(val_dataloader, column_names=['data', 'label', 'classes'])
    i = 1
    for batch in inp.create_dict_iterator():
        x = batch['data']
        y = batch['label']
        classes = batch['classes']
        x_array = x.asnumpy()
        y_array = y.asnumpy()
        classes_array = classes.asnumpy()
        x_array.tofile(output_path + os.sep +"data_" + str(i) + ".bin")
        y_array.tofile(label_classses_path + os.sep +"label_" + str(i) + ".bin")
        classes_array.tofile(label_classses_path + os.sep +"classes_" + str(i) + ".bin")
        i = i + 1
if __name__ == '__main__':
    options = parser.parse_args()
    convert_img_to_bin(options, options.dataset_path, options.data_output_path, options.label_classses_output_path)
