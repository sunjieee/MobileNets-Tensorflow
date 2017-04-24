import tensorflow as tf
import numpy as np
from model import *
from input_data import *

__author__ = 'Sun Jie'
'''
Tensorflow Implementation of MobileNets
More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).
'''

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_path', '',
                           """Directory where is data"""
                           )


def get_loss(logits, labels):
	labels_onehot = tf.one_hot(labels, depth=1000)
	cross_entropy = tf.losses.softmax_cross_entropy(logits, labels_onehot)
	return cross_entropy


def training(loss):
	tf.summary.scalar('loss', loss)
	global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
			trainable=False)

	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
		global_step,
		60000 / BATCH_SIZE,
		LEARING_RATE_DECAY,
		staircase=True)

	optimizer = tf.train.RMSPropOptimizer(learning_rate)

	train_op = optimizer.minimize(loss, global_step=global_step)

	return train_op


def main():
	with tf.Graph().as_default(), tf.device('/cpu:0'):
		input = Preprocessing(data_path)
		images, labels = input._read_input_queue()
		with tf.device('gpu:1'):
			models = MobileNets(images)
			logits, end_points = models.inference()
			loss = get_loss(labels, logits)
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
			
			summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)

			for step in range(TRAINING_STEPS):
				
				_, loss_value = sess.run([train_op, loss])

				if step % 10 == 0:
					
					print('setp %d: loss = %.4f'  % (step, loss_value))
					
					summary = sess.run(summary_op)
					
					summary_writer.add_summary(summary, step)

				if step % 1000 == 0:
					
					checkpoint_path = os.path.join(
						MODEL_SAVE_PATH, MODEL_NAME)
					
					saver.save(sess, checkpoint_path, global_step=step)



			coord.request_stop()
			
			coord.join(threads)



if __name__ == '__main__':
	tf.app.run()
