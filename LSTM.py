"""
Code to define LSTM model

Author: Shakthi Duraimurugan

"""
import tensorflow as tf

class lstm_model(object):
	def __init__(self, batch_size, sequence_length, num_hidden, num_classes):
		lstm = tf.contrib.rnn.LSTMCell((num_hidden))

		w = tf.get_variable(name = "lstm_weights", shape = [num_hidden, num_classes], 
			initializer = tf.contrib.layers.xavier_initializer(), trainable = True)
		b = tf.get_variable(name = "lstm_bias", shape = [num_classes], 
			initializer = tf.contrib.layers.xavier_initializer(), trainable = True)

		seq_out, final_out = tf.nn.dynamic_rnn(cell = lstm, sequence_length = sequence_length, inputs = x)
		out = tf.nn.xw_plus_b(tf.reshape(seq_out, [-1, num_hidden]), w, b)
		self.out = tf.reshape(out, [tf.shape(x)[0], tf.shape(x)[1], -1])


if __name__ == "__main__":

	batch_size = 16
	sequence_length = 16
	num_hidden = 1000
	num_classes = 24

	model = lstm_model(batch_size, sequence_length, num_hidden, num_classes)
	