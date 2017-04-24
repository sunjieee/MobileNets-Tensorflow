import tensorflow as tf
import numpy as np


class Preprocessing(object):
	
	def __init__(self, data_path, image_size=224, num_preprocess_threads=1, min_after_dequeue=10000, is_training=True):

		self.data_path = data_path
		self.image_size = image_size
		self.num_preprocess_threads = num_preprocess_threads
		self.min_after_dequeue = min_after_dequeue
		self.is_training = is_training


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
		if image.dtype != tf.float32:
			image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	        
		bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
			tf.shape(image), bounding_boxes=bbox)
		bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
			tf.shape(image), bounding_boxes=bbox)
		distorted_image = tf.slice(image, bbox_begin, bbox_size)


		distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))
		distorted_image = tf.image.random_flip_left_right(distorted_image)
		distorted_image = distort_color(distorted_image, np.random.randint(2))
		return distorted_image
		

	def _read_input(self, filename_queue):
		reader = tf.TFRecordReader()

		_,serialized_example = reader.read(filename_queue)

		features = tf.parse_single_example(
			serialized_example,
			features={
				'image':tf.FixedLenFeature([],tf.string),
				'label':tf.FixedLenFeature([],tf.int64)
			})
		decoded_image = tf.decode_jpeg(features['image'], channels=3)
		reshaped_image = tf.reshape(decoded_image, [self.image_size, self.image_size, 3])
		if is_training:
			distorted_image = preprocess_for_train(reshaped_image)
		else:
			distorted_image = tf.image.convert_image_dtype(reshaped_image, dtype=tf.float32)

		distorted_image = tf.subtract(distorted_image, 0.5)
		distorted_image = tf.multiply(distorted_image, 2.0)
		#tf.summary.image('final_distorted_image', tf.expand_dims(distorted_image, 0))

		label = tf.cast(features['label'], tf.int32)

		return distorted_image, label
		


	def _read_input_queue(self):
		files = tf.train.match_filenames_once(data_path)
		filename_queue = tf.train.string_input_producer(files, shuffle=True)

		image, label = self._read_input(filename_queue)

		capacity = min_after_dequeue + 3 * self.batch_size
		image_batch, label_batch = tf.train.shuffle_batch([image, label], 
														batch_size=self,batch_size,
														num_threads=self.num_preprocess_threads,
														capacity=self.capacity,
														min_after_dequeue=min_after_dequeue)
		return image_batch, label_batch
