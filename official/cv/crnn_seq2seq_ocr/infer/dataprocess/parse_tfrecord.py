import os
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='tool that takes tfrecord files and extracts all images + labels from it')
parser.add_argument('input_path', default='../data/fsns-tfrecord', help='path to directory containing tfrecord files')
parser.add_argument('output_path', default='../data/fsns', help='path to dir where resulting images shall be saved')

args = parser.parse_args()

feature_description = {'image/encoded': tf.io.FixedLenFeature(
    [], dtype=tf.string, default_value=''),
                       'image/text': tf.io.FixedLenFeature(
                           [1], dtype=tf.string, default_value=''),
                       'image/class': tf.io.VarLenFeature(dtype=tf.int64),
                       'image/unpadded_class': tf.io.VarLenFeature(dtype=tf.int64),
                       'image/height': tf.io.FixedLenFeature(
                           [1], dtype=tf.int64, default_value=1),
                       'image/width': tf.io.FixedLenFeature(
                           [1], dtype=tf.int64, default_value=1)}


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def parseAll(input_pattern, ttype):
    output_dir = os.path.join(args.output_path, ttype)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + "-anno"):
        os.makedirs(output_dir + "-anno")
    output_txt = output_dir + "-anno/image2text.txt"
    filenames = os.listdir(os.path.join(input_pattern, ttype))
    dirs = [os.path.join(input_pattern, ttype, i) for i in filenames]
    raw_dataset = tf.data.TFRecordDataset(dirs)
    parsed_dataset = raw_dataset.map(_parse_function)
    text = open(output_txt, 'w+')
    idx = 0
    for image_features in parsed_dataset:
        image_raw = image_features['image/encoded'].numpy()
        output_image = str(idx) + ".png"
        output_imagePath = os.path.join(output_dir, output_image)
        with open(output_imagePath, "wb") as png:
            png.write(image_raw)
        text.write(output_image + '\t' + str(image_features['image/text'].numpy()[0], encoding='utf-8') + '\n')
        idx = idx + 1


if __name__ == "__main__":
    parseAll(args.input_path, "test")
    parseAll(args.input_path, "train")
