# Temporal block
import numpy as np
import tensorflow as tf

class TemporalBlock(tf.keras.Model):
	def __init__(self, dilation_rate, nb_filters, kernel_size, 
				       padding, dropout_rate=0.0): 
		super(TemporalBlock, self).__init__()
		init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
		assert padding in ['causal', 'same']

		# block1
		self.conv1 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
				                   dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
		self.batch1 = tf.keras.layers.BatchNormalization(axis=-1)
		self.ac1 = tf.keras.layers.Activation('relu')
		self.drop1 = tf.keras.layers.Dropout(rate=dropout_rate)
		
		# block2
		self.conv2 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
						           dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
		self.batch2 = tf.keras.layers.BatchNormalization(axis=-1)		
		self.ac2 = tf.keras.layers.Activation('relu')
		self.drop2 = tf.keras.layers.Dropout(rate=dropout_rate)

		self.downsample = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1, 
									    padding='same', kernel_initializer=init)
		self.ac3 = tf.keras.layers.Activation('relu')


	def call(self, x, training):
		prev_x = x
		x = self.conv1(x)
		x = self.batch1(x)
		x = self.ac1(x)
		x = self.drop1(x) if training else x

		x = self.conv2(x)
		x = self.batch2(x)
		x = self.ac2(x)
		x = self.drop2(x) if training else x

		if prev_x.shape[-1] != x.shape[-1]:    # match the dimention
			prev_x = self.downsample(prev_x)
		assert prev_x.shape == x.shape

		return self.ac3(prev_x + x)            # skip connection

class TemporalConvNet(tf.keras.Model):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2):
    	# num_channels is a list contains hidden sizes of Conv1D
        super(TemporalConvNet, self).__init__()
        assert isinstance(num_channels, list)

        model = tf.keras.Sequential()

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i                  # exponential growth
            model.add(TemporalBlock(dilation_rate, num_channels[i], kernel_size, 
                      padding='causal', dropout_rate=dropout))
        model.add(tf.keras.layers.LSTM(60, return_sequences=True))
        model.add(tf.keras.layers.LSTM(60, return_sequences=True))
        model.add(tf.keras.layers.LSTM(30, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.5))
        self.network = model

    def call(self, x, training):
        return self.network(x, training=training)


class TCN_LSTM(tf.keras.Model):
    def __init__(self, output_size, num_channels, kernel_size, dropout):
        super(TCN_LSTM, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

        self.temporalCN = TemporalConvNet(num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = tf.keras.layers.Dense(output_size, activation='softmax')

    def call(self, x, training=True):
        X = tf.keras.layers.Reshape((1, 800))(x)
        y = self.temporalCN(X, training=training)
        y = self.linear(y)
        return y    # use the last element to output the result