
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Sun Jie'
'''
Tensorflow Implementation of MobileNets
More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).
'''


tf.app.flags.DEFINE_string('data_path', '/home/sunjieeee/new_project/valid.*',
	'Data directory')

tf.app.flags.DEFINE_integer('batch_size', 256,
	"""Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('image_size', 224,
	"""Provide square images of this size.""")

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
	"""Number of preprocessing threads per tower. """)

tf.app.flags.DEFINE_bool('is_training', True,
	'''Is trainning''')

FLAGS = tf.app.flags.FLAGS

def decode_jpeg(image):

	image = tf.image.decode_jpeg(image, channels=3)

	image = tf.image.convert_image_dtype(image, dtype=tf.float32)

	return image


def distort_color(image, color_ordering=0):
	
	if color_ordering == 0:
		
		image = tf.image.random_brightness(image, max_delta=32./255.)
		
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		
		image = tf.image.random_hue(image, max_delta=0.2)
		
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
	
	else:
		
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		
		image = tf.image.random_brightness(image, max_delta=32./255.)
		
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		
		image = tf.image.random_hue(image, max_delta=0.2)

	return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image, height, width, bbox):
	
	if bbox is None:
		
		bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        
	bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
		tf.shape(image), bounding_boxes=bbox, min_object_covered=0.8)
	
	distorted_image = tf.slice(image, bbox_begin, bbox_size)

	distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))
	
	distorted_image = tf.image.random_flip_left_right(distorted_image)
	
	distorted_image = distort_color(distorted_image, np.random.randint(2))
	
	return distorted_image
		

def _read_input(filename_queue):
	
	examples_per_shard = 1024
	
	min_queue_examples = examples_per_shard * 16;

	if is_training:
		
		examples_queue = tf.RandomShuffleQueue(
			capacity=min_queue_examples + 3 * batch_size,
			min_after_dequeue=min_queue_examples,
			dtypes=[tf.string])
	
	else:
		
		examples_queue = tf.FIFOQueue(
			capacity=examples_per_shard + 3 * batch_size,
			dtypes=[tf.string])

	enqueue_ops = []
	
	for _ in range(num_readers):
		
		reader = tf.TFRecordReader()
		
		_, value = reader.read(filename_queue)
		
		enqueue_ops.append(examples_queue.enqueue([value]))

	tf.train.queue_runner.add_queue_runner(
		tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
	
	serialized_example = examples_queue.dequeue()
	
	images_and_labels = []
	
	for thread_id in range(FLAGS.num_preprocess_threads):

		features = tf.parse_single_example(
			serialized_example,
			features={
				'image':tf.FixedLenFeature([],tf.string),
				'label':tf.FixedLenFeature([],tf.int64)
			})
		
		decoded_image = decode_jpeg(features['image'])

		if FLAGS.is_training:
			
			distorted_image = preprocess_for_train(decoded_image, FLAGS.image_size, FLAGS.image_size, None)
		
		else:
			
			distorted_image = tf.image.resize_images(decoded_image, [FLAGS.image_size, FLAGSimage_size], 
				method=np.random.randint(4))
			
		distorted_image = tf.subtract(distorted_image, 0.5)
		
		distorted_image = tf.multiply(distorted_image, 2.0)

		label = tf.cast(features['label'], tf.int32)
		
		images_and_labels.append([distorted_image, label])

	return images_and_labels
		


def _read_input_queue():
	
	with tf.name_scope('input_data'):

		files = tf.train.match_filenames_once(FLAGS.data_path)

		filename_queue = tf.train.string_input_producer(files, shuffle=False)

		images_and_labels = _read_input(filename_queue)

		capacity = 2 * FLAGS.num_preprocess_threads * FLAGS.batch_size
		
		image_batch, label_batch = tf.train.batch_join(images_and_labels, 
			batch_size=FLAGS.batch_size,
			capacity=capacity)
	
	return image_batch, label_batch
