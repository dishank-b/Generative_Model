# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import glob
import cv2
	
def Conv2d(input, output_dim=64, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='conv_2d'):

    with tf.variable_scope(name):
    	# print input.get_shape()[-1]
        W = tf.get_variable('Conv2dW', [kernel[0], kernel[1], input.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Conv2db', [output_dim], initializer=tf.constant_initializer(0.0001))
        
        return tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding='SAME') + b

def Deconv2d(input, output_dim, batch_size, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='deconv_2d'):
    
    with tf.variable_scope(name):
        W = tf.get_variable('Deconv2dW', [kernel[0], kernel[1], output_dim, input.get_shape()[-1]],
                             initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Deconv2db', [output_dim], initializer=tf.constant_initializer(0.0001))

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

def BatchNormalization(input, name='bn', train=True):
    
    with tf.variable_scope(name):
    
        # output_dim = input.get_shape()[-1]
        # beta = tf.get_variable('BnBeta', [output_dim],
        #                     initializer=tf.zeros_initializer())
        # gamma = tf.get_variable('BnGamma', [output_dim],
        #                     initializer=tf.ones_initializer())
    
        # if len(input.get_shape()) == 2:
        #     mean, var = tf.nn.moments(input, [0])
        # else:
        #     mean, var = tf.nn.moments(input, [0, 1, 2])
        # return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-5)

        return tf.contrib.layers.batch_norm(input,decay=0.9, updates_collections=None, epsilon=1e-5,scale=True,is_training=train, scope=name)

    
def LeakyReLU(input, leak=0.2, name='lrelu'):
    
    return tf.maximum(input, leak*input)

class GAN(object):
	"""docstring for GAN"""
	def __init__(self):
		self.weights_bias=[]

	def generator(self, z, batch_size, is_train=True):
		with tf.name_scope("generator") as sc:
			with tf.variable_scope("gen") as var_scope:
				G_1 = Dense(z, output_dim=4*4*512, name='G_hidden1') # [-1, 7*7*128]
				G_h1 = tf.nn.relu(G_1)
				G_r1 = tf.reshape(G_h1, [-1, 4, 4, 512])
				G_conv2 = Deconv2d(G_r1, output_dim=128, batch_size=batch_size, name='G_hidden2')
				G_bn2 = BatchNormalization(G_conv2, name='G_hidden2', train=is_train)        
				G_h2 = tf.nn.relu(G_bn2)
				G_conv3 = Deconv2d(G_h2, output_dim=64, batch_size=batch_size, name='G_hidden3')
				G_bn3 = BatchNormalization(G_conv3, name='G_hidden3',train= is_train)        
				G_h3 = tf.nn.relu(G_bn3)
				G_conv4 = Deconv2d(G_h3, output_dim=3, batch_size=batch_size, name='G_hidden4')
				G_out = tf.nn.sigmoid(G_conv4)
				return G_out

	def discriminator(self, x, reuse=False, is_train=True):
		with tf.name_scope("discriminator") as sc:
			with tf.variable_scope("dis", reuse=reuse) as var_scope:
				D_conv1 = Conv2d(x, output_dim=64, name='D_hidden1')
				D_bn1 = BatchNormalization(D_conv1, name="D_hidden1",train= is_train)
				D_h1 = LeakyReLU(D_bn1) # [-1, 16, 16, 16]
				D_conv2 = Conv2d(D_h1, output_dim=128, name='D_hidden2')
				D_bn2 = BatchNormalization(D_conv2, name="D_hidden2", train=is_train)
				D_h2 = LeakyReLU(D_bn2) # [-1, 8, 8, 32]
				D_conv3 = Conv2d(D_h2, output_dim=512, name='D_hidden3')
				D_bn3 = BatchNormalization(D_conv3, name="D_hidden3",train= is_train)
				D_h3 = LeakyReLU(D_bn3) # [-1, 4, 4, 64]
				D_r3 = tf.reshape(D_h3, [-1, int(np.prod(D_h3.get_shape()[1:]))])
				D_h4 = Dense(D_r3, output_dim=1, name='D_hidden4') # [-1, 1]
				return tf.nn.sigmoid(D_h4)

	def build_model(self, batch_size):
		with tf.name_scope("Inputs") as scope:
			self.is_train = tf.placeholder(tf.bool, name="is_train")
			self.z = tf.placeholder(tf.float32, shape=[None, 100])
			self.z_summ = tf.summary.histogram("Noise", self.z)
			self.x = tf.placeholder(tf.float32, shape=[None, 32,32,3]) # [batch_size, 32*32*3]
		
		with tf.name_scope("Model") as scope:
			self.gen_out = self.generator(self.z, batch_size, is_train= self.is_train)
			self.dis_real = self.discriminator(self.x, False,is_train= self.is_train)
			self.dis_fake = self.discriminator(self.gen_out, reuse=True,is_train= self.is_train)
			# print self.dis_real.get_shape(), self.dis_fake.get_shape()
			self.gen_out_summ = tf.summary.image("Generator Image", self.gen_out, max_outputs=4)
			self.dis_real_summ = tf.summary.histogram("Discriminator Real", self.dis_real)
			self.dis_fake_summ = tf.summary.histogram("Discriminator Fake", self.dis_fake)

		with tf.name_scope("Loss") as scope:
			self.dis_loss = tf.reduce_mean(-tf.log(self.dis_real) - tf.log(1-self.dis_fake))
			self.gen_loss = tf.reduce_mean(-tf.log(self.dis_fake))

			self.dis_loss_summ = tf.summary.scalar("Discriminator Loss", self.dis_loss)
			self.gen_loss_summ = tf.summary.scalar("Generator Loss", self.gen_loss)
			train_vars = tf.trainable_variables()
			# print train_vars
			self.d_vars = [var for var in train_vars if 'D_' in var.name]
			self.g_vars = [var for var in train_vars if 'G_' in var.name]
			# print self.d_vars, self.g_vars

	
	def train_model(self, images, learning_rate, epoch_size, batch_size, z_length):

		D_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.1).minimize(self.dis_loss, var_list=self.d_vars)
		D_gradients = tf.gradients(self.dis_loss, [var for var in self.d_vars if 'D_hidden4' in var.name])
		D_grad_summ = tf.summary.histogram("Dis Gradients", D_gradients[0])
		G_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.3).minimize(self.gen_loss, var_list=self.g_vars)
		G_gradients = tf.gradients(self.gen_loss, self.g_vars)
		G_grad_summ = tf.summary.histogram("Gen Gradients", G_gradients[0])


		self.G_summ = tf.summary.merge([self.z_summ, self.gen_out_summ, self.gen_loss_summ, self.dis_fake_summ])
		self.D_summ = tf.summary.merge([self.z_summ, self.dis_loss_summ, self.dis_real_summ])

		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		self.writer = tf.summary.FileWriter("./logs/graphs")
		self.writer.add_graph(self.sess.graph)

		for epoch in range(epoch_size):
			for itr in xrange(0, len(images)-batch_size, batch_size):
				batch_images = images[itr:itr+batch_size]
				batch_z = np.random.uniform(0.0,1.0,size=(batch_size, z_length))

				D_inputs = [D_solver, self.D_summ, self.dis_loss, D_gradients]
				D_outputs = self.sess.run(D_inputs, {self.z: batch_z, self.x:batch_images, self.is_train:True})
				self.writer.add_summary(D_outputs[1], itr)

				G_inputs = [G_solver, self.G_summ, self.gen_loss, G_gradients]
				G_outputs = self.sess.run(G_inputs, {self.z : batch_z, self.x:batch_images, self.is_train:True})
				self.writer.add_summary(G_outputs[1], itr)

				print "Epoch: ", epoch, "Iteration: ", itr
				print "Discriminator Loss: ", D_outputs[2]
				print "Generator Loss: ", G_outputs[2] 
				# print "Dis Gradients", D_grads
				# print "Dis Output", self.dis_fake.eval({self.z:batch_z, self.x:batch_images},session=self.sess)
				# print "Gen Gradients", G_grads

			if epoch%5==0:
				self.saver.save(self.sess, "logs/model")
				print "Checkpoint saved"

				sample_z = np.random.uniform(-1,1,size=(batch_size, z_length))
				generated_images = self.sess.run([self.gen_out], {self.z : sample_z, self.is_train:False})
				all_images = np.array(generated_images[0])
				
				for i in range(8):
					image_grid_horizontal = 255.0*all_images[i*24]
					for j in range(7):
						image = 255.0*all_images[i*24+(j+1)*3]
						image_grid_horizontal = np.hstack((image_grid_horizontal, image))
					if i==0:
						image_grid_vertical = image_grid_horizontal
					else:
						image_grid_vertical = np.vstack((image_grid_vertical, image_grid_horizontal))

				cv2.imwrite("./logs/gen_images/img_"+str(epoch)+".jpg", image_grid_vertical)




images = np.load("./cifar10_data/cifar_training.npy")
print "Data Loaded"

images = 1/255.0*images
reshape_images = images.reshape((images.shape[0],3,32,32)).transpose(0,2,3,1)
print reshape_images.shape
# print images[0].shape

gan_model = GAN()
gan_model.build_model(batch_size=256)
gan_model.train_model(images = reshape_images,learning_rate=0.0002, epoch_size=200, batch_size=256, z_length=100)