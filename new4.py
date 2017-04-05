import tensorflow as tf
import numpy as np
import scipy.io as sio
from six.moves import xrange
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data(path, name, samples, dimension, result_max, result_min):
    data = sio.loadmat(path)
    data = data[name]
    
    x = np.zeros((samples,dimension)).astype('float32')
    
    for i in xrange(0,samples):
        x[i] = data[i]
    
    x_min = x.min(axis = 0)
    x_min = x_min - 1e-6
    x_max = x.max(axis = 0)
    x_max = x_max + 1e-6

    x = (result_max-result_min)*(x-x_min)/(x_max - x_min) + result_min

    return x, x_min, x_max

data, data_min, data_max = load_data('mesh2.mat', 'mesh', 71, 12996, 0.9, -0.9)

def recover_data(x, x_min, x_max, result_min, result_max):
    
    x = (x_max - x_min)*(x - result_min)/(result_max-result_min) + x_min
    
    return x

def linear(input_, input_size, output_size, name='Linear', stddev=0.02, bias_start=0.0, wreturn = False):
    with tf.variable_scope(name) as scope:
        matrix = tf.get_variable("weights", [input_size, output_size], tf.float32,
                tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], tf.float32, 
          initializer=tf.constant_initializer(bias_start))

        if wreturn:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def deconv2d(input_, output_shape, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
      w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
      try:
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
        strides=[1, d_h, d_w, 1])

      # Support for verisons of TensorFlow before 0.7.0
      except AttributeError:
        deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

      biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
      deconv = tf.nn.bias_add(deconv, biases)

      if with_w:
        return deconv, w, biases
      else:
        return deconv

def leaky_relu(input, alpha = 0.2):
    return tf.maximum(input, alpha*input)

def batch_norm_wrapper(inputs, name = 'batch_norm',is_training = False, decay = 0.9, epsilon = 1e-5):  # need to improve
    with tf.variable_scope(name) as scope:
        if is_training == True:
            scale = tf.get_variable('scale', dtype=tf.float32, trainable=True, initializer=tf.ones([inputs.get_shape()[-1]],dtype=tf.float32))  
            beta = tf.get_variable('beta', dtype=tf.float32, trainable=True, initializer=tf.zeros([inputs.get_shape()[-1]],dtype=tf.float32))
            #scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=True, name = 'scale')
            #beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=True, name = 'beta')
            pop_mean = tf.get_variable('overallmean',  dtype=tf.float32,trainable=False, initializer=tf.zeros([inputs.get_shape()[-1]],dtype=tf.float32))
            pop_var = tf.get_variable('overallvar',  dtype=tf.float64, trainable=False, initializer=tf.ones([inputs.get_shape()[-1]],dtype=tf.float64))
        else:
            scope.reuse_variables()
            scale = tf.get_variable('scale', dtype=tf.float64, trainable=True)
            beta = tf.get_variable('beta', dtype=tf.float64, trainable=True)
            pop_mean = tf.get_variable('overallmean', dtype=tf.float64, trainable=False)
            pop_var = tf.get_variable('overallvar', dtype=tf.float64, trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, epsilon)

class meshGAN(object):
    def __init__(self, sess):
        self.sess = sess
        
        self.build_model()
    
    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [None,12996], name = 'real_mesh')
        
        #inputs = self.inputs
        #sample_inputs = self.sample_inputs
        
        self.batch_size = 64
        
        self.z = tf.placeholder(tf.float32, [None, 100], name = 'z')
        
        self.z_size = tf.shape(self.z)[0]
        
        self.cg = [1024, 512, 256, 128, 1]
        self.cd = [0, 64, 128, 256, 512, 1024, 1]
        self.G = self.generator(self.z)
        self.Sampler = self.sampler(self.z)
        
        self.D1, self.D1_ = self.discriminator(self.inputs)
        self.D2, self.D2_ = self.discriminator(self.G, reuse=True)
        
        self.D1_loss = 1000 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_, labels=tf.ones_like(self.D1_,dtype=tf.float32)))
        self.D2_loss = 1000 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_, labels=tf.zeros_like(self.D2_,dtype=tf.float32)))
        self.G_loss = 10000 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_, labels=tf.ones_like(self.D2_,dtype=tf.float32)))
        
        self.D_loss = self.D1_loss + self.D2_loss
        
        t_vars = tf.trainable_variables()
        
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        
        for var in self.g_vars:
            print('g ', var.name)
            
            
        for var in self.d_vars:
            print('d ', var.name)
            
        self.D_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, name='D_opt').minimize(self.D_loss, var_list = self.d_vars)
        self.G_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, name='G_opt').minimize(self.G_loss, var_list = self.g_vars)
        
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

        
    def train_model(self):
        
        sample_z = np.random.uniform(-5, 5, size = (64, 100))
        samples = sess.run([self.Sampler], feed_dict={self.z: sample_z})[0]
        samples = recover_data(samples, data_min, data_max, -0.9, 0.9)
        sio.savemat('start.mat', {'sample_mesh': samples})
        
        for epoch in xrange(0, 5000):
            tidx = np.random.permutation(71)
            
            for bidx in xrange(0, 2):
                data_batch = [data[i] for i in tidx[bidx*32:min(71, bidx*32+31)]]
                z_batch = np.random.uniform(-5, 5, size = (64, 100))
                
                for times in xrange(0,1):
                    self.sess.run([self.D_opt], feed_dict = { self.inputs: data_batch, self.z: z_batch})
                
                for times in xrange(0,8):
                    z_batch = np.random.uniform(-5, 5, size = (64, 100))
                    self.sess.run([self.G_opt], feed_dict = { self.z : z_batch})
                    
                cost_D1 = self.D1_loss.eval({ self.inputs: data_batch })
                cost_D2 = self.D2_loss.eval({ self.z: z_batch })
                cost_G = self.G_loss.eval({self.z: z_batch})
                print("Epoch: [%2d] [%4d/4] d_loss: %.8f, g_loss: %.8f" % (epoch+1, bidx+1, cost_D1+cost_D2, cost_G))
                 
            
            if np.mod(epoch, 20) == 1:
                sample_z = np.random.uniform(-5, 5, size = (64, 100))
                samples = sess.run([self.Sampler], feed_dict={self.z: sample_z})[0]
                samples = recover_data(samples, data_min, data_max, -0.9, 0.9)
                sio.savemat('sample'+str(epoch)+'.mat', {'sample_mesh': samples})
        return
        
        
    def generator(self, z):
	#z = tf.convert_to_tensor(z)
        with tf.variable_scope("generator") as scope:
            self.rsp1 = tf.layers.dense(z, self.cg[0] * 8 *8)
            self.rsp2 = tf.reshape(self.rsp1, [-1, 8, 8, self.cg[0]])
            self.gbn0 = tf.layers.batch_normalization(self.rsp2, training = True, name = 'gbn0')
            self.ga0 = leaky_relu(self.gbn0)   #leaky_relu
            
            self.h1, self.h1_w, self.h1_b = deconv2d(
                self.ga0, [self.batch_size, 15, 15, self.cg[1]], name='g_h1', with_w=True )
            self.gbn1 = tf.layers.batch_normalization(self.h1, training = True,name = 'gbn1')
            self.ga1 = leaky_relu(self.gbn1)
                
            
            self.h2, self.h2_w, self.h2_b = deconv2d(
		self.h1, [self.batch_size, 29, 29, self.cg[2]], name='g_h2', with_w=True)
            self.gbn2 = tf.layers.batch_normalization(self.h2, training = True,name = 'gbn2')
            self.ga2 = leaky_relu(self.gbn2)
            
            self.h3, self.h3_w, self.h3_b = deconv2d(
		self.h2, [self.batch_size, 57, 57, self.cg[3]], name='g_h3', with_w=True)
            self.gbn3 = tf.layers.batch_normalization(self.h3, training = True,name = 'gbn3')
            self.ga3 = leaky_relu(self.gbn3)

            self.h4, self.h4_w, self.h4_b = deconv2d(
		self.h3, [self.batch_size, 114, 114, self.cg[4]], name='g_h4', with_w=True)
            self.gbn4 = tf.layers.batch_normalization(self.h4, training = True,name = 'gbn4')
            self.ga4 = leaky_relu(self.gbn4)
            
            self.rsp3 = tf.reshape(self.ga4, [-1, 12996])
            
        return tf.nn.tanh(self.rsp3)
        
    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            
            self.rsp1 = tf.layers.dense(z, self.cg[0] * 8 * 8)
            self.rsp2 = tf.reshape(self.rsp1, [-1, 8, 8, self.cg[0]])
            self.gbn0 = tf.layers.batch_normalization(self.rsp2, training = False, name = 'gbn0')
            self.ga0 = leaky_relu(self.gbn0)   #leaky_relu
            
            #self.g1 = tf.layers.conv2d_transpose(self.ga0, self.cg[1], [5, 3], strides=(1, 1), padding='SAME', name='deconv1')
            self.h1, self.h1_w, self.h1_b = deconv2d(
		self.ga0, [self.batch_size, 15, 15, self.cg[1]], name='g_h1', with_w=True)
            self.gbn1 = tf.layers.batch_normalization(self.h1, training = False,name = 'gbn1')
            self.ga1 = leaky_relu(self.gbn1)
            

            self.h2, self.h2_w, self.h2_b = deconv2d(
		self.h1, [self.batch_size, 29, 29, self.cg[2]], name='g_h2', with_w=True)
            self.gbn2 = tf.layers.batch_normalization(self.h2, training = False,name = 'gbn2')
            self.ga2 = leaky_relu(self.gbn2)

            self.h3, self.h3_w, self.h3_b = deconv2d(
		self.h2, [self.batch_size, 57, 57, self.cg[3]], name='g_h3', with_w=True)
            self.gbn3 = tf.layers.batch_normalization(self.h3, training = False,name = 'gbn3')
            self.ga3 = leaky_relu(self.gbn3)

            self.h4, self.h4_w, self.h4_b = deconv2d(
		self.h3, [self.batch_size, 114, 114, self.cg[4]], name='g_h4', with_w=True)
            self.ga4 = leaky_relu(self.gbn4)
            
            self.rsp3 = tf.reshape(self.ga4, [-1, 12996])
            
            return tf.nn.tanh(self.rsp3)
            
        
    def discriminator(self, inputmesh, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            
            outputs = tf.reshape(inputmesh, [-1, 114, 114, 1])                                          #114 * 114
            with tf.variable_scope("conv1"):
                outputs = tf.layers.conv2d(outputs, self.cd[1], [4, 4], strides=[2, 2], padding='SAME', name="conv") #56 * 56
            with tf.variable_scope("conv2"):
                outputs = tf.layers.conv2d(outputs, self.cd[2], [4, 4], strides=[2, 2], padding='SAME', name="conv") #27 * 27
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=True))
            with tf.variable_scope("conv3"):
                outputs = tf.layers.conv2d(outputs, self.cd[3], [4, 4], strides=[2, 2], padding='SAME', name="conv") #13 * 13
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=True))
            with tf.variable_scope("conv4"):
                outputs = tf.layers.conv2d(outputs, self.cd[4], [4, 4], strides=[2, 2], padding='SAME', name="conv") #6 * 6
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=True))
            with tf.variable_scope("conv5"):
                outputs = tf.layers.conv2d(outputs, self.cd[5], [4, 4], strides=[2, 2], padding='SAME', name="conv") #2 * 2
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=True))
            with tf.variable_scope("conv6"):
                outputs = tf.layers.conv2d(outputs, self.cd[6], [2, 2], strides=[1, 1], padding='SAME', name="conv") #1 * 1
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=True))
            outputs = tf.reshape(outputs, [-1, 1])
           
            return tf.nn.sigmoid(outputs), outputs
            

sess = tf.InteractiveSession()

meshgan = meshGAN(sess)

meshgan.train_model()

meshgan.saver.save(meshgan.sess, 'meshgan.model', global_step = 126)

meshgan.saver.save(meshgan.sess, 'meshgan.model', global_step = 450)

