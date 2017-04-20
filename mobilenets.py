import tensorflow

__author__ = "Sun Jie"
"""
Tensorflow implementation of MobileNets
"""

class MobileNets(object):
	"""docstring for MobileNets"""
	def __init__(self, images, is_training=True, end_point=[]):
		self.images = images
		self.is_training = is_training
		self.end_point = end_point

	def get_tensor_size(self, tensor):
		return tensor.get_shape().as_list()

	def weight_variable_xavier_initialized(self, shape, constant=1, name=None):
		stddev = constant * np.sqrt(2.0 / (shape[2] + shape[3]))
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

	def batch_norm(self, x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5, stddev=0.02):
		"""
		Code taken from http://stackoverflow.com/a/34634291/2267819
		"""
		with tf.variable_scope(scope):
			
			beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
	                               , trainable=True)
			gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev),
	                                trainable=True)
			batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
			ema = tf.train.ExponentialMovingAverage(decay=decay)

			def mean_var_with_update():
				ema_apply_op = ema.apply([batch_mean, batch_var])
				with tf.control_dependencies([ema_apply_op]):
					return tf.identity(batch_mean), tf.identity(batch_var)
			with tf.variable_scope(tf.get_variable_scope(), reuse=False):
				mean, var = tf.cond(phase_train,
									mean_var_with_update,
									lambda: (ema.average(batch_mean), ema.average(batch_var)))
			normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
		return normed


	def conv(self, input_tensor, depth, filter, stride=1, bn=batch_norm, act=tf.nn.relu, scope):
		
		with tf.variable_scope(scope):
			
			dim = self.get_tensor_size(input_tensor)[3]
			
			W = self.weight_variable_xavier_initialized([filter[0], filter[1], dim, depth], name='weight')
			b = self.bias_variable([depth], name='bias')
			h_conv = self.conv2d_strided(input_tensor, W, b, stride)

			if bn != None:
				h_bn = self.batch_norm(h_conv, depth, self.is_training)

			h_act = act(h_bn)
			end_point.append(h_act)
			self.add_activation_summary(h_act)

			return h_act


	def conv_dw(self, input_tensor, filter, stride=1, bn=batch_norm, act=tf.nn.relu, scope):
		
		with tf.variable_scope(scope):

			dim = self.get_tensor_size(input_tensor)[3]

			W = self.weight_variable_xavier_initialized([filter[0], filter[1], dim, 1], name='weight')
			b = self.bias_variable([depth], name='bias')
			h_conv = self.depthwise_conv2d_strided(input_tensor, W, b, stride)

			if bn != None:
				h_bn = self.batch_norm(h_conv, depth, self.is_training)

			h_act = act(h_bn)
			end_point.append(h_act)
			self.add_activation_summary(h_act)

			return h_act


	def global_avg_pool(self, x):

		k = self.get_tensor_size(x)
		return tf.nn.avg_pool(x, ksize=[1, k[1], k[2], 1], strides=[1, 1, 1, 1], padding="SAME")


	def inference():
		net = self.conv(self.images, 32, [3, 3], 2, scope='conv1')
		
		net = self.conv_dw(net, [3, 3], scope='conv2_dw')
		net = self.conv(net, 64, [1, 1], scope='conv2_pw')
		
		net = self.conv_dw(net, [3, 3], 2, scope='conv3_dw')
		net = self.conv(net, 128, [1, 1], scope='conv3_pw')
		
		net = self.conv_dw(net, [3, 3], scope='conv4_dw')
		net = self.conv(net, 128, [1, 1], scope='conv4_pw')

		net = self.conv_dw(net, [3, 3], 2, scope='conv5_dw')
		net = self.conv(net, 256, [1, 1], scope='conv5_pw')

		net = self.conv_dw(net, [3, 3], scope='conv6_dw')
		net = self.conv(net, 256, [1, 1], scope='conv6_pw')

		net = self.conv_dw(net, [3, 3], 2, scope='conv7_dw')
		net = self.conv(net, 512, [1, 1], scope='conv7_pw')

		net = self.conv_dw(net, [3, 3], scope='conv8_dw')
		net = self.conv(net, 512, [1, 1], scope='conv8_pw')

		net = self.conv_dw(net, [3, 3], scope='conv9_dw')
		net = self.conv(net, 512, [1, 1], scope='conv9_pw')

		net = self.conv_dw(net, [3, 3], scope='conv10_dw')
		net = self.conv(net, 512, [1, 1], scope='conv10_pw')

		net = self.conv_dw(net, [3, 3], scope='conv11_dw')
		net = self.conv(net, 512, [1, 1], scope='conv11_pw')

		net = self.conv_dw(net, [3, 3], scope='conv12_dw')
		net = self.conv(net, 512, [1, 1], scope='conv12_pw')

		net = self.conv_dw(net, [3, 3], 2, scope='conv13_dw')
		net = self.conv(net, 1024, [1, 1], scope='conv13_pw')

		net = self.conv_dw(net, [3, 3], 2, scope='conv14_dw')
		net = self.conv(net, 1024, [1, 1], scope='conv14_pw')

		net = self.global_avg_pool(net, scope='avg_pool15')

		self.net = self.conv(net, 1000, [1, 1], bn=None, act=tf.nn.softmax, scope='fc16')

		return net


