import os
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='tool that takes tfrecord files and extracts all images + labels from it')
parser.add_argument('input_path', default='../data/fsns-tfrecord', help='path to directory containing tfrecord files')
parser.add_argument('output_path', default='../data/fsns', help='path to dir where resulting images shall be saved')

args = parser.parse_args()

def parseAll(input_pattern, ttype):
    output_dir = os.path.join(args.output_path, ttype)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + "-anno"):
        os.makedirs(output_dir + "-anno")
    output_txt = output_dir + "-anno/image2text.txt"
    filenames = os.listdir(os.path.join(input_pattern, ttype))
    dirs = [os.path.join(input_pattern, ttype, i) for i in filenames]
    text = open(output_txt, 'w+')
    idx = 0
    for filename in dirs:
        for serialized_example in tf.python_io.tf_record_iterator(path=filename):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            image_raw = example.features.feature['image/encoded'].bytes_list.value[0]
            output_image = str(idx) + ".png"
            output_imagePath = os.path.join(output_dir, output_image)
            with open(output_imagePath, "wb") as png:
                png.write(image_raw)
            text.write(output_image + '\t' + example.features.feature['image/text'].bytes_list.value[0].decode() + '\n')
            idx = idx + 1


if __name__ == "__main__":
    parseAll(args.input_path, "test")
    parseAll(args.input_path, "train")
