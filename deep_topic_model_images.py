import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
import time

tf.keras.backend.set_floatx('float64')

class InferenceNetwork(tf.keras.layers.Layer):
    def __init__(self, data_dim, num_topics, mu2, var2):
        super(InferenceNetwork, self).__init__()
        self.n_input = data_dim
        self.num_topics = num_topics
        self.encoder_layer1 = Dense( 100, activation = 'selu', use_bias = True )
        self.encoder_batchnorm1 = BatchNormalization( center = False, scale = True )
        #self.encoder_layer2 = Dense( 100, activation = 'selu', use_bias = True )
        self.z_mean_layer = Dense( self.num_topics, activation = 'linear', use_bias = True )
        self.z_logvar_layer = Dense( self.num_topics, activation = 'linear', use_bias = True )
        self.h_dim = int(self.num_topics)
        self.mu2 = mu2
        self.var2 = var2
    def sampling(self, inputs):
        z_mean, z_logvar = inputs
        batch_size = tf.shape( z_mean )[0]
        dim = tf.shape( z_mean )[1]
        eps = tf.keras.backend.random_normal( shape=(batch_size,dim) )
        return z_mean + tf.math.exp( 0.5*z_logvar )*eps
    def call(self, inputs):
        hs_e = []
        hs_e.append( self.encoder_layer1( inputs ) )
        #hs_e.append( self.encoder_layer2( hs_e[-1] ) )
        hs_e.append( self.encoder_batchnorm1( hs_e[-1] ) )
        z_mean = self.z_mean_layer( hs_e[-1] )
        z_logvar = self.z_logvar_layer( hs_e[-1] )
        return z_mean, z_logvar, self.sampling( (z_mean,z_logvar) )
    def reset_weights(self):
        self.encoder_layer1 = Dense( 100, activation = 'selu', use_bias = True )
        #self.encoder_layer2 = Dense( 100, activation = 'selu', use_bias = True )
        self.z_mean_layer = Dense( self.num_topics, activation = 'linear', use_bias = True )
        self.z_logvar_layer = Dense( self.num_topics, activation = 'linear', use_bias = True )

class TopicsNetwork(tf.keras.layers.Layer):
    def __init__(self, data_dim, num_topics):
        super(TopicsNetwork, self).__init__()
        self.num_topics = num_topics
        self.data_dim = data_dim
        self.topics_layer = Dense( self.data_dim, activation = 'linear', use_bias = False )
        #self.topics_batchnorm = BatchNormalization( center = True, scale = False )
    def call(self, z):
        z_norm = tf.nn.softmax( z )
        hs_d = []
        hs_d.append( self.topics_layer( z_norm ) )
        #hs_d.append( self.topics_batchnorm( hs_d[-1] ) )
        return tf.nn.softmax( hs_d[-1] )
    def get_topics(self):
        arr = np.diag( np.ones( self.num_topics ) )
        topics = []
        for i in range(self.num_topics):
            topic_probs = tf.nn.softmax( self.topics_layer( np.asarray([arr[i]]) ) )[0]
            topics.append( topic_probs )
        return np.array( topics )
    def likelihood_ratio(self, event, likelihood_topics):
        arr = np.diag( np.ones( self.num_topics ) )
        arr0 = arr[ likelihood_topics[0] ]
        arr1 = arr[ likelihood_topics[1] ]
        topic_probs_0 = tf.nn.softmax( self.topics_layer( np.asarray([arr0]) ) )[0]
        topic_probs_1 = tf.nn.softmax( self.topics_layer( np.asarray([arr1]) ) )[0]
        numerator_vector = event * topic_probs_0
        numerator_vector = numerator_vector[ event!=0 ]
        denominator_vector = event * topic_probs_1
        denominator_vector = denominator_vector[ event!=0 ]
        lr_vector = numerator_vector / denominator_vector
        lr = np.prod( lr_vector )
        return lr
    def likelihood_ratios(self, events, likelihood_topics):
        lrs = []
        for event in events:
            lrs.append( self.likelihood_ratio(event, likelihood_topics) )
        return lrs

class VAE(tf.keras.Model):
    def __init__(self, data_dim, num_topics, mu2, var2):
        super(VAE, self).__init__()
        self.data_dim = data_dim
        self.num_topics = num_topics
        self.mu2 = mu2
        self.var2 = var2
        self.encoder = InferenceNetwork( data_dim, num_topics, mu2, var2 )
        self.decoder = TopicsNetwork( self.data_dim, self.num_topics )
    def call(self, inputs):
        z_mean, z_logvar, z = self.encoder( inputs )
        reco = self.decoder( z )
        return z_mean, z_logvar, reco

