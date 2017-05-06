
import tensorflow as tf
import numpy as np
from model import *
from input_data import *

__author__ = 'Sun Jie'
'''
Tensorflow Implementation of MobileNets
More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).
'''

tf.app.flags.DEFINE_string('data_path', '',
	"""Directory where is data""")

tf.app.flags.DEFINE_integer('num_gpus', 4,
	'Num gpus')

tf.app.flags.DEFINE_float('moving_average_decay', 0.99,
	'Moving average decay')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
	'Learning rate decay factor')

tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
	'Initial learning rate')

tf.app.flags.DEFINE_string('model_save_path', '',
	'Model save path')

tf.app.flags.DEFINE_string('model_name', 'model.ckpt',
	'Model name')

FLAGS = tf.app.flags.FLAGS



def get_loss(logits, labels):
	
	models = MobileNets(images)
		
	logits, end_points = models.inference()

	labels_onehot = tf.one_hot(labels, depth=1000)
	
	cross_entropy = tf.losses.softmax_cross_entropy(logits, labels_onehot)
	
	regularization_loss = tf.add_n(tf.get_collection('losses', scope))
	
	loss = cross_entropy +　regularization_loss

	tf.summary.scalar('loss', loss)
	
	return loss
	
def average_gradients(tower_grads):
	
	average_grads = []

	for grad_and_vars in zip(*tower_grads):

		grads = []
		
		for g, _ in grad_and_vars:
			
			expanded_g = tf.expand_dims(g, 0)
			
			grads.append(expanded_g)
		
		grad = tf.concat(grads, 0)
		
		grad = tf.reduce_mean(grad, 0)

		v = grad_and_vars[0][1]
		
		grad_and_var = (grad, v)

		average_grads.append(grad_and_var)

	return average_grads

def training(iamges, labels):
	
	global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
		trainable=False)

	learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
		global_step,
		decay_steps,
		FLAGS.learning_rate_decay_factor,
		staircase=True)

	opt = tf.train.RMSPropOptimizer(learning_rate)

	images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=images)
    
	labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=labels)
	
	tower_grads = []
	
	reuse_variables = False
	
	for i in range(N_GPU):

		with tf.device('/gpu:%d' % i):
			
			with tf.name_scope('GPU_%d' % i) as scope:
				
				cur_loss = get_loss(x, y_, scope, reuse_variables)
				
				reuse_variables = True
				
				grads = opt.compute_gradients(cur_loss)
				
				tower_grads.append(grads)
	
	grads = average_gradients(tower_grads)
	
	for grad, var in grads:
		
		if grad is not None:
			
			tf.histogram_summary('gradients_on_average/%s' % var.op.name, grad)
			
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	
	for var in tf.trainable_variables():
		
		tf.histogram_summary(var.op.name, var)
		
	variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
	
	variables_to_average = (tf.trainable_variables() +tf.moving_average_variables())
	
	variables_averages_op = variable_averages.apply(variables_to_average)
	
	train_op = tf.group(apply_gradient_op, variables_averages_op)

	return train_op


def main():
	
	with tf.Graph().as_default(), tf.device('/cpu:0'):
		
		images, labels = _read_input_queue()
		
		train_op = training(loss)

		summary_op = tf.summary.merge_all()
		
		saver = tf.train.Saver()
		
		init = tf.global_variables_initializer()

		with tf.Session(config=tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=False)) as sess:

			sess.run(init)
			
			coord = tf.train.Coordinator()
			
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)
			
			summary_writer = tf.summary.FileWriter(FLAGS.model_save_path, sess.graph)

			for step in range(TRAINING_STEPS):
				
				_, loss_value = sess.run([train_op, loss])

				if step % 10 == 0:
					
					print('setp %d: loss = %.4f'  % (step, loss_value))
					
					summary = sess.run(summary_op)
					
					summary_writer.add_summary(summary, step)

				if step % 1000 == 0:
					
					checkpoint_path = os.path.join(
						FLAGS.model_save_path, FLAGS.model_name)
					
					saver.save(sess, checkpoint_path, global_step=step)

			coord.request_stop()
			
			coord.join(threads)


if __name__ == '__main__':
	
	tf.app.run()
