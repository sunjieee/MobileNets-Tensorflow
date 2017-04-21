import tensorflow as tf
import numpy as np

__author__ = 'Sun Jie'
'''
Tensorflow Implementation of MobileNets
More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).
'''

class MobileNets(object):

	def __init__(self, images, is_training=True, spatial_squeeze=True):
		self.images = images
		self.is_training = is_training
		self.spatial_squeeze = spatial_squeeze
		self.end_points = {}

	def get_tensor_name(self, tensor):
		return tensor.op.name

	def get_tensor_size(self, tensor):
		return tensor.get_shape().as_list()

	def weight_variable_xavier_initialized(self, shape, constant=1, name=None):
		stddev = constant * np.sqrt(2.0 / (shape[2] + shape[3]))
		return self.weight_variable(shape, stddev=stddev, name=name)

	def variance_scaling_initializer(self, shape, constant=1, name=None):
		stddev = constant * np.sqrt(2.0 / shape[2])
		return self.weight_variable(shape, stddev=stddev, name=name)

	def weight_variable(self, shape, stddev=0.02, name=None):
		initial = tf.truncated_normal(shape, stddev=stddev)
		if name is None:
			return tf.Variable(initial)
		else:
			return tf.get_variable(name, initializer=initial)

	def bias_variable(self, shape, name=None):
		initial = tf.constant(0.0, shape=shape)
		if name is None:
			return tf.Variable(initial)
		else:
			return tf.get_variable(name, initializer=initial)

	def conv2d_strided(self, x, W, b, s):
		conv = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding="SAME")
		return tf.nn.bias_add(conv, b)

	def depthwise_conv2d_strided(self, x, W, b, s):
		conv = tf.nn.depthwise_conv2d(x, W, strides=[1, s, s, 1], padding="SAME")
		return tf.nn.bias_add(conv, b)

	def add_to_regularization_and_summary(var):
		if var is not None:
			tf.summary.histogram(var.op.name, var)
			tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))

	def add_activation_summary(self, var):
		tf.summary.histogram(var.op.name + "/activation", var)
		tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


	def add_gradient_summary(self, grad, var):
		if grad is not None:
			tf.summary.histogram(var.op.name + "/gradient", grad)


	def conv(self, input_tensor, depth, filter, stride, scope, bn=tf.layers.batch_normalization, act=tf.nn.relu):
		
		with tf.variable_scope(scope):
			
			dim = self.get_tensor_size(input_tensor)[3]
			
			W = self.variance_scaling_initializer([filter[0], filter[1], dim, depth], name='weight')
			b = self.bias_variable([depth], name='bias')
			h_conv = self.conv2d_strided(input_tensor, W, b, stride)

			h_conv = bn(h_conv, training=self.is_training)

			h_conv = act(h_conv)

			self.end_points[scope] = h_conv
			self.add_activation_summary(h_conv)

			return h_conv


	def conv_dw(self, input_tensor, filter, stride, scope, bn=tf.layers.batch_normalization, act=tf.nn.relu):
		
		with tf.variable_scope(scope):

			dim = self.get_tensor_size(input_tensor)[3]

			W = self.variance_scaling_initializer([filter[0], filter[1], dim, 1], name='weight')
			b = self.bias_variable([dim], name='bias')
			h_conv = self.depthwise_conv2d_strided(input_tensor, W, b, stride)

			h_conv = bn(h_conv, training=self.is_training)
			
			h_conv = act(h_conv)

			self.end_points[scope] = h_conv
			self.add_activation_summary(h_conv)

			return h_conv


	def global_avg_pool(self, x, scope):

		k = self.get_tensor_size(x)
		return tf.nn.avg_pool(x, ksize=[1, k[1], k[2], 1], strides=[1, 1, 1, 1], padding="VALID")

	def squeeze(self, x):
		if self.spatial_squeeze:
			return tf.squeeze(x, [1, 2], name='spatial_squeeze')
		return x
	
	def inference(self):
		net = self.conv(self.images, 32, [3, 3], 2, scope='conv1')

		net = self.conv_dw(net, [3, 3], 1, scope='conv2_dw')
		net = self.conv(net, 64, [1, 1], 1, scope='conv2_pw')

		net = self.conv_dw(net, [3, 3], 2, scope='conv3_dw')
		net = self.conv(net, 128, [1, 1], 1, scope='conv3_pw')

		net = self.conv_dw(net, [3, 3], 1, scope='conv4_dw')
		net = self.conv(net, 128, [1, 1], 1, scope='conv4_pw')

		net = self.conv_dw(net, [3, 3], 2, scope='conv5_dw')
		net = self.conv(net, 256, [1, 1], 1, scope='conv5_pw')

		net = self.conv_dw(net, [3, 3], 1, scope='conv6_dw')
		net = self.conv(net, 256, [1, 1], 1, scope='conv6_pw')

		net = self.conv_dw(net, [3, 3], 2, scope='conv7_dw')
		net = self.conv(net, 512, [1, 1], 1, scope='conv7_pw')

		net = self.conv_dw(net, [3, 3], 1, scope='conv8_dw')
		net = self.conv(net, 512, [1, 1], 1, scope='conv8_pw')

		net = self.conv_dw(net, [3, 3], 1, scope='conv9_dw')
		net = self.conv(net, 512, [1, 1], 1, scope='conv9_pw')

		net = self.conv_dw(net, [3, 3], 1, scope='conv10_dw')
		net = self.conv(net, 512, [1, 1], 1, scope='conv10_pw')

		net = self.conv_dw(net, [3, 3], 1, scope='conv11_dw')
		net = self.conv(net, 512, [1, 1], 1, scope='conv11_pw')

		net = self.conv_dw(net, [3, 3], 1, scope='conv12_dw')
		net = self.conv(net, 512, [1, 1], 1, scope='conv12_pw')

		net = self.conv_dw(net, [3, 3], 2, scope='conv13_dw')
		net = self.conv(net, 1024, [1, 1], 1, scope='conv13_pw')

		net = self.conv_dw(net, [3, 3], 1, scope='conv14_dw')
		net = self.conv(net, 1024, [1, 1], 1, scope='conv14_pw')

		net = self.global_avg_pool(net, scope='avg_pool15')
		
		net = self.conv(net, 1000, [1, 1], 1, bn=None, act=None, scope='fc16')
		#print self.get_tensor_name(net) ,self.get_tensor_size(net)
		self.logits = self.squeeze(net)

		self.end_points['logits'] = self.logits
		self.end_points['predictions'] = tf.nn.softmax(self.logits)
		
		return self.logits, self.end_points


