import tensorflow as tf
import numpy as np
import random

__author__ = "Sun Jie"

tf.app.flags.DEFINE_string('jpeg_file_path', './*',
							'Jpeg file path')

tf.app.flags.DEFINE_string('tfrecord_name', './Records/data.tfrecords',
							'TFRecord name')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
	
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _process_image_files(filenames, labels):

	writer = tf.python_io.TFRecordWriter(FLAGS.tfrecord_name)
	
	for i in range(len(filenames)):
		
		with tf.gfile.FastGFile(filenames[i], 'rb') as f:
			
			image_data = f.read()

		example = tf.train.Example(features=tf.train.Features(feature={
			'image': _bytes_feature(image_data),
			'label': _int64_feature(labels[i])
		}))
		
		writer.write(example.SerializeToString())
	
	writer.close()


def  _find_image_files():
	
	labels = []
	
	filenames = tf.gfile.Glob(FLAGS.jpeg_file_path)
	
	for filename in filenames:
		
		label = 1 if 'cat' in filename else 0
		
		labels.append(label)

	shuffled_index = list(range(len(filenames)))
	
	random.seed(12345)
	
	random.shuffle(shuffled_index)

	filenames = [filenames[i] for i in shuffled_index]
	
	labels = [labels[i] for i in shuffled_index]

	return filenames, labels


def _process_dataset():
	
	filenames, labels = _find_image_files()
	
	_process_image_files(filenames, labels)


def main(_):
	
	_process_dataset()


if __name__ == '__main__':
	
	tf.app.run()
