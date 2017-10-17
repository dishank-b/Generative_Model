import tensorflow as tf
import numpy as np
import glob
	

def Conv2d(input, output_dim=64, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='conv_2d'):

    with tf.variable_scope(name):
    	# print input.get_shape()[-1]
        W = tf.get_variable('Conv2dW', [kernel[0], kernel[1], input.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Conv2db', [output_dim], initializer=tf.zeros_initializer())
        
        return tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding='SAME') + b

def Deconv2d(input, output_dim, batch_size, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='deconv_2d'):
    
    with tf.variable_scope(name):
        W = tf.get_variable('Deconv2dW', [kernel[0], kernel[1], output_dim, input.get_shape()[-1]],
                             initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Deconv2db', [output_dim], initializer=tf.zeros_initializer())

        input_shape = input.get_shape().as_list()
        output_shape = [batch_size,
                        int(input_shape[1] * strides[0]),
                        int(input_shape[2] * strides[1]),
                        output_dim]

        deconv = tf.nn.conv2d_transpose(input, W, output_shape=output_shape,
                                        strides=[1, strides[0], strides[1], 1])
    
        return deconv + b

def Dense(input, output_dim, stddev=0.02, name='dense'):
    
    with tf.variable_scope(name):
        shape = input.get_shape()
        W = tf.get_variable('Weight', [shape[1], output_dim], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('Bias', [output_dim],
                            initializer=tf.zeros_initializer())
        return tf.nn.relu(tf.matmul(input, W) + b)

def BatchNormalization(input, name='bn'):
    
    with tf.variable_scope(name):
    
        output_dim = input.get_shape()[-1]
        beta = tf.get_variable('BnBeta', [output_dim],
                            initializer=tf.zeros_initializer())
        gamma = tf.get_variable('BnGamma', [output_dim],
                            initializer=tf.ones_initializer())
    
        if len(input.get_shape()) == 2:
            mean, var = tf.nn.moments(input, [0])
        else:
            mean, var = tf.nn.moments(input, [0, 1, 2])
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-5)
    
def LeakyReLU(input, leak=0.2, name='lrelu'):
    
    return tf.maximum(input, leak*input)

class GAN(object):
	"""docstring for GAN"""
	def __init__(self):
		self.weights_bias=[]

	def generator(self, z, batch_size):
		with tf.name_scope("generator") as sc:
			with tf.variable_scope("gen") as var_scope:
				G_1 = Dense(z, output_dim=1024, name='G_hidden1') # [-1, 1024]
				G_bn1 = BatchNormalization(G_1, name='G_hidden1')
				G_h1 = tf.nn.relu(G_bn1)
				G_2 = Dense(G_h1, output_dim=7*7*128, name='G_hidden2') # [-1, 7*7*128]
				G_bn2 = BatchNormalization(G_2, name='G_hidden2')        
				G_h2 = tf.nn.relu(G_bn2)
				G_r2 = tf.reshape(G_h2, [-1, 7, 7, 128])
				G_conv3 = Deconv2d(G_r2, output_dim=64, batch_size=batch_size, name='G_hidden3')
				G_bn3 = BatchNormalization(G_conv3, name='G_hidden3')        
				G_h3 = tf.nn.relu(G_bn3)
				G_conv4 = Deconv2d(G_h3, output_dim=1, batch_size=batch_size, name='G_hidden4')
				G_r4 = tf.reshape(G_conv4, [-1, 784])
				return G_conv4, tf.nn.sigmoid(G_r4)

	def sampler(self, z, sample_size):
		with tf.name_scope("generator") as sc:
			with tf.variable_scope("gen", reuse=True) as var_scope:
				G_1 = Dense(z, output_dim=1024, name='G_hidden1') # [-1, 1024]
				G_bn1 = BatchNormalization(G_1, name='G_hidden1')
				G_h1 = tf.nn.relu(G_bn1)
				G_2 = Dense(G_h1, output_dim=7*7*128, name='G_hidden2') # [-1, 7*7*128]
				G_bn2 = BatchNormalization(G_2, name='G_hidden2')        
				G_h2 = tf.nn.relu(G_bn2)
				G_r2 = tf.reshape(G_h2, [-1, 7, 7, 128])
				G_conv3 = Deconv2d(G_r2, output_dim=64, batch_size=sample_size, name='G_hidden3')
				G_bn3 = BatchNormalization(G_conv3, name='G_hidden3')        
				G_h3 = tf.nn.relu(G_bn3)
				G_conv4 = Deconv2d(G_h3, output_dim=1, batch_size=sample_size, name='G_hidden4')
				return G_conv4

	def discriminator(self, x, reuse=False):
		with tf.name_scope("discriminator") as sc:
			with tf.variable_scope("dis", reuse=reuse) as var_scope:
				if len(x.get_shape()) > 2:
            		# X: -1, 28, 28, 1
					D_conv1 = Conv2d(x, output_dim=64, name='D_hidden1')
				else:
					D_reshaped = tf.reshape(x, [-1, 28, 28, 1])
					D_conv1 = Conv2d(D_reshaped, output_dim=64, name='D_hidden1')
				D_h1 = LeakyReLU(D_conv1) # [-1, 28, 28, 64]
				D_conv2 = Conv2d(D_h1, output_dim=128, name='D_hidden2')
				D_h2 = LeakyReLU(D_conv2) # [-1, 28, 28, 128]
				D_r2 = tf.reshape(D_h2, [-1, 256])
				D_h3 = LeakyReLU(D_r2) # [-1, 256]
				D_h4 = tf.nn.dropout(D_h3, 0.5)
				D_h5 = Dense(D_h4, output_dim=1, name='D_hidden3') # [-1, 1]
				return tf.nn.sigmoid(D_h5)

	def build_model(self, batch_size):
		with tf.name_scope("Inputs") as scope:
			self.z = tf.placeholder(tf.float32, shape=[None, 100])
			self.z_summ = tf.summary.histogram("Noise", self.z)
			self.x = tf.placeholder(tf.float32, shape=[None, 28,28,1])
		
		with tf.name_scope("Model") as scope:
			self.gen_img, self.gen_out = self.generator(self.z, batch_size)
			self.dis_real = self.discriminator(self.x, False)
			self.dis_fake = self.discriminator(self.gen_out, reuse=True)
			self.gen_out_summ = tf.summary.image("Generator Image", self.gen_img)
			self.dis_real_summ = tf.summary.histogram("Discriminator Real", self.dis_real)
			self.dis_fake_summ = tf.summary.histogram("Discriminator Fake", self.dis_fake)

		with tf.name_scope("Loss") as scope:
			self.dis_loss = -tf.reduce_mean(tf.log(self.dis_real)) - tf.reduce_mean(tf.log(1-self.dis_fake))
			self.gen_loss = -tf.reduce_mean(tf.log(self.dis_fake))
			self.total_loss = self.dis_loss + self.gen_loss

			self.dis_loss_summ = tf.summary.scalar("Discriminator Loss", self.dis_loss)
			self.gen_loss_summ = tf.summary.scalar("Generator Loss", self.gen_loss)
			self.total_loss.summ = tf.summary.scalar("Total Loss", self.total_loss)
			train_vars = tf.trainable_variables()
			# print train_vars
			self.d_vars = [var for var in train_vars if 'D_' in var.name]
			self.g_vars = [var for var in train_vars if 'G_' in var.name]
			# print self.d_vars, self.g_vars

	
	def train_model(self, images, learning_rate, epoch_size, batch_size, z_length, sample_size):

		D_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.1).minimize(self.dis_loss, var_list=self.d_vars)
		G_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.3).minimize(self.gen_loss, var_list=self.g_vars)


		self.G_summ = tf.summary.merge([self.z_summ, self.gen_out_summ, self.gen_loss_summ, self.dis_fake_summ])
		self.D_summ = tf.summary.merge([self.z_summ, self.dis_loss_summ, self.dis_real_summ])

		sample_z = np.random.uniform(-1,1, size=(sample_size, z_length))

		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		self.writer = tf.summary.FileWriter("./logs/graphs")
		self.writer.add_graph(self.sess.graph)

		for epoch in range(epoch_size):
			for itr in xrange(0, len(images), batch_size):
				batch_images = images[itr:itr+batch_size]
				batch_z = np.random.uniform(-1,1,size=(batch_size, z_length))

				_, D_summary, D_loss = self.sess.run([D_solver, self.D_summ, self.dis_loss], {self.z: batch_z, self.x:batch_images})
				self.writer.add_summary(D_summary, itr)

				batch_z = np.random.uniform(-1,1, size=(batch_size, z_length))

				_, G_summary, G_loss = self.sess.run([G_solver, self.G_summ, self.gen_loss], {self.z : batch_z, self.x:batch_images})
				self.writer.add_summary(G_summary, itr)

				print "Epoch: ", epoch, "Iteration: ", itr
				print "Discriminator Loss: ", D_loss
				print "Generator Loss: ", G_loss 

			if epoch%5==0:
				self.saver.save(self.sess, "logs/model")
				print "Checkpoint saved"

				sample_images = sampler(batch_z, batch_size)
				sample_img_summ = tf.summary.image("Generated Images", sample_images, max_outputs=5)
				self.writer.add_summary(self.sess.run([sample_images_summ],{self.z: batch_z, self.x:batch_images}))


images = np.load("./dataset/trainingSet/mnist_data_28*28*1_0to1values.npy")
print "Data Loaded"
print images.shape
gan_model = GAN()
gan_model.build_model(batch_size=64)
gan_model.train_model(images = images,learning_rate=0.0002, epoch_size=100, batch_size=64, z_length=100, sample_size=10)