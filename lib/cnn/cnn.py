 
import tensorflow as tf
import numpy as np
from math import ceil

def init_weights(shape, dist='random_normal', normalized=True):
    """Initializes network weights.
    
    Args:
        shape: A tensor. Shape of the weights.
        dist: A str. Distribution at initialization, one of 'random_normal' or 
            'truncated_normal'.
        normalized: A boolean. Whether weights should be normalized.
        
    Returns:
        A tf.variable.
    """
    # Normalized if normalized set to True
    if normalized == True:
        denom = np.prod(shape[:-1])
        std = 1 / denom
    else:
        std = .1
    
    # Draw from random or truncated normal
    if dist == 'random_normal':
        weights = tf.random_normal(shape, stddev=std)
    elif dist == 'truncated_normal':
        weights = tf.truncated_normal(shape, stddev=0.1)
    
    return tf.Variable(weights)


def init_biases(shape):
    """Initialize biases. """
    biases = tf.constant(0., shape=shape)
    return tf.Variable(biases)

def l2_loss(weights, l2_regularization_penalty, y_deconv, y_, name):
    """Implements L2 loss for an arbitrary number of weights.
    
    Args:
        weights: A dict. One key/value pair per layer in the network.
        l2_regularization_penalty: An int. Scales the l2 loss arbitrarily.
        y_:
        y_conv:
        name: 
            
    Returns:
        L2 loss.        
    """
    weights_loss = {}
    for key, value in weights.items():
        weights_loss[key] = tf.nn.l2_loss(value)
    
    l2_loss = l2_regularization_penalty * sum(weights_loss.values())
    
    unregularized_loss = tf.reduce_mean(tf.square(y_deconv - y_) ) # MSE
    return tf.add(unregularized_loss, l2_loss, name=name)


def create_network(x_in,
                   x_stim,
                   classes,
                    n_layers,
                   channels_in,
                   channels_out,
                   fc_conv, 
                   fc_central,
                   fc_stim,
                   fc_deconv,
                   patch_dim,
                   pool_dim,
                   training,
                   keep_prob,
                   n_chans2,
                   samples_per_trial2,                   
                   weights_dist='random_normal',
                   normalized_weights = True,
                   nonlin = 'leaky_relu',
                   bn =  False,
                   kpt = False, 
                   DECAY = 0.9):
    
    if nonlin == 'elu':
        activation = tf.nn.elu
    elif nonlin == 'leaky_relu':
        activation = tf.nn.leaky_relu
    elif nonlin == 'sigmoid':
        activation = tf.nn.sigmoid
    else :
        print('nonlin ,', nonlin, 'not supported')
    
    ###################
    ### CONV LAYERS ###
    ###################
    
    weights = {}

    for i in range(n_layers):
        W_conv = init_weights([patch_dim[0], patch_dim[1], channels_in[i], channels_out[i]],
                            dist=weights_dist,
                            normalized=normalized_weights)
        weights[i] = W_conv # for deconv or l2
        b_conv = init_biases([channels_out[i]])
        if i==0:
            h_conv = tf.nn.conv2d(x_in, W_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv
        else:
            h_conv = tf.nn.conv2d(out, W_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv
        #h_relu = tf.nn.leaky_relu(h_conv)
        h_relu = activation(h_conv)
        
        if bn == True:
            h_relu_bn = tf.contrib.layers.batch_norm(
                h_relu,
                data_format='NHWC',
                center=True,
                scale=True,
                is_training=training,
                decay=DECAY,
                renorm=True)

            out = tf.nn.max_pool(h_relu_bn, 
                                    ksize=[1, pool_dim[0], pool_dim[1], 1], 
                                    strides=[1, pool_dim[0], pool_dim[1], 1], 
                                    padding='SAME')
        else:
            out = tf.nn.max_pool(h_relu, 
                            ksize=[1, pool_dim[0], pool_dim[1], 1], 
                            strides=[1, pool_dim[0], pool_dim[1], 1], 
                            padding='SAME')
            

    #################
    ### FC LAYERS ###
    #################        
        
    ## FLATTEN OUTPUT 
    size = tf.shape(out)[0] # batch size

    shape_in = out.get_shape().as_list()
    dim1 = 1
    for i in range(1, len(shape_in)):
        dim1 = dim1 * shape_in[i]

    flat = tf.reshape(out, [-1, dim1])

    ## COUCHE 1 : output conv to fc_conv (FC + nonlin)
    # La couche fc_in "melange" les differentes series temporelles

    weights_fc_in = init_weights([dim1, fc_conv],
                        dist=weights_dist,
                        normalized=normalized_weights)
    h_fc_in = tf.matmul(flat, weights_fc_in)
    
    #out_fc_in = tf.nn.leaky_relu(h_fc_in)
    out_fc_in = activation(h_fc_in)
    
    #dropout
    if kpt == True :
        out_fc_in = tf.nn.dropout(out_fc_in, keep_prob)

    ## COUCHE 2 : fc_conv to fc_central (FC without nonlin)
    # la couche fc_central "compresse" fc_in (couche linéaire)

    weights_fc_central = init_weights([fc_conv, fc_central],
                                    dist=weights_dist,
                                    normalized=normalized_weights)

    h_fc_central = tf.matmul(out_fc_in, weights_fc_central)
    #out_fc_central = h_fc_central
    
    #### ADD STIMULUS INFORMATION
    
    weights_fc_stimulus = init_weights([classes, fc_stim],
                                    dist=weights_dist,
                                    normalized=normalized_weights)

    h_fc_stim = tf.matmul(x_stim, weights_fc_stimulus)
    
    # Concat
    out_fc_central = tf.concat([h_fc_central, h_fc_stim], 1)
    
    

    # COUCHE 3 : fc_central to fc_deconv (FC + nonlin)
    # la couche fc_out "décompresse" le code

    weights_fc_out = init_weights([fc_central+fc_stim, fc_deconv],
                                    dist=weights_dist,
                                    normalized=normalized_weights)
    h_fc_out = tf.matmul(out_fc_central, weights_fc_out)
    #out_fc_out = tf.nn.leaky_relu(h_fc_out)
    out_fc_out = activation(h_fc_out)
    
    # COUCHE 4 : fc_deconv to a flatten, then reshape it for deconv layers
    # cette couche reforme le signal

    dim2 = n_chans2 * ceil(samples_per_trial2/(2**n_layers)) * channels_out[-1]
    weights_fc_out = init_weights([fc_deconv, dim2],
                                    dist=weights_dist,
                                    normalized=normalized_weights)
    h_fc_out = tf.matmul(out_fc_out, weights_fc_out)
    #out_fc_out = tf.nn.leaky_relu(h_fc_out)
    out_fc_out = activation(h_fc_out)

    out_fc = tf.reshape(out_fc_out, (size, n_chans2, ceil(samples_per_trial2/(2**n_layers)), channels_out[-1]))


    #####################
    ### DECONV LAYERS ###
    #####################

    #size = tf.shape(out)[0] # batch_size
    for i in range(n_layers):
        
        W_deconv = init_weights([patch_dim[0], patch_dim[1], channels_in[-1-i], channels_out[-1-i]])
        #                        )
        #W_deconv = weights[n_layers-1-i]
        
        weights[n_layers+3+i] = W_deconv
        if i==0:
            h_deconv = tf.nn.conv2d_transpose(
                out_fc,
                W_deconv,
                (size, n_chans2, ceil(samples_per_trial2/(2**(n_layers-1))), channels_in[-1-i]),
                [1, 1 , 2, 1],
                padding='SAME',
                data_format='NHWC',
                name=None)
        else:
            h_deconv = tf.nn.conv2d_transpose(
                h_deconv,
                W_deconv,
                (size, n_chans2, ceil(samples_per_trial2/(2**(n_layers-1-i))), channels_in[-1-i]),
                [1, 1 , 2, 1],
                padding='SAME',
                data_format='NHWC',
                name=None)
        if i != n_layers-1 : # no bn or nonlin on the last layer
            #h_deconv = tf.nn.leaky_relu(h_deconv)
            h_deconv = activation(h_deconv)
            
            if bn == True:
                h_deconv = tf.contrib.layers.batch_norm(
                    h_deconv,
                    data_format='NHWC',
                    center=True,
                    scale=True,
                    is_training=training,
                    decay=DECAY,
                    renorm=True)
                
    return( h_deconv, weights)
    
                   
                   
                   