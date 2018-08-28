# Imports
from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        var_image = tf.reshape(var, (tf.shape(var)[0], 1, tf.shape(var)[1], 1))
        tf.summary.image('value', var_image, max_outputs=100)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

def input_embedding(input_sentence, emsize, n_characters, counters, reuse=False, pretrain_vec=None):
    """Wrapper for input embedding
    # Arguments
        input_sentence: Tensor of shape [N, L]
        counters: for variable reuse and references
        reuse: to reuse or not
    # Returns
        Tensor of shape [N, L, emsize]
    """
    if reuse:
        counters['Embedding'] = 0
    name = get_name('Embedding', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # use pretrain word vec (fasttext) or not
        if pretrain_vec is not None:
            emb = tf.get_variable('emb', shape=[n_characters, emsize], dtype=tf.float32,
                                  initializer=tf.constant_initializer(pretrain_vec))
        else:
            emb = tf.get_variable('emb', shape=[n_characters, emsize], dtype=tf.float32,
                                  initializer=None)

        lookup_result = tf.nn.embedding_lookup(emb, input_sentence)
        # masking the first dimension
        masked_emb = tf.concat([tf.zeros([1, 1]),
                                tf.ones([n_characters - 1, 1])], 0)
        mask_lookup_result = tf.nn.embedding_lookup(masked_emb, input_sentence)
        # broadcast
        embedded_input = tf.multiply(lookup_result, mask_lookup_result)
    return embedded_input

def temporal_padding(x, padding=(1, 1)):
    """Pads the middle dimension of a 3D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.
    # Returns
        A padded 3D tensor.
    """
    assert len(padding) == 2
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.pad(x, pattern)

def han_attention(inputs, query):
    """HAN-style Attention block
    # Arguments
        inputs: Tensor of shape [N, L, Cin]
        query:  Tensor of shape [Cin]
    # Returns
        Tensor of shape [N, Cin]
        The weighted sum of inputs, where weights are calculated by
        inputs and query
    """
    vector_attn = tf.reduce_sum(tf.multiply(inputs, query), axis=2, keepdims=True)
    attention_weights = tf.nn.softmax(vector_attn, axis=1)
    variable_summaries(attention_weights)
    weighted_inputs = tf.multiply(inputs, attention_weights)
    outputs = tf.reduce_sum(weighted_inputs, axis=1)
    return outputs

def multi_head_attention(x, counters, dropout, num_head=8, reuse=False):
    """multi head attention block
    # Arguments
        x: Tensor of shape [N, L, Cin]
    """
    x_size = x.get_shape()[-1].value
    k_size = x_size // num_head
    v_size = x_size // num_head

    name = get_name('multihead_attention', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope(name)
        heads = []
        for _ in range(num_head):
            heads.append(self_attention_block(x, k_size, v_size, counters, dropout, reuse))
        outputs = tf.concat(heads, axis=2)
        outputs = tf.layers.dense(outputs, units=x_size, use_bias=False)

    return outputs

def self_attention_block(x, counters, k_size=-1, v_size=-1, dropout=0.0, reuse=False):
    """self attention block
    # Arguments
        x: Tensor of shape [N, L, Cin]
    """

    if k_size == -1:
        k_size = x.get_shape()[-1].value
    if v_size == -1:
        v_size = x.get_shape()[-1].value

    name = get_name('attention_block', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # [N, L, k_size]
        key = tf.layers.dense(x, units=k_size, use_bias=False,
                              kernel_initializer=tf.random_normal_initializer(0, 0.01))
        key = tf.nn.dropout(key, 1.0 - dropout)
        # [N, L, k_size]
        query = tf.layers.dense(x, units=k_size, use_bias=False,
                                kernel_initializer=tf.random_normal_initializer(0, 0.01))
        query = tf.nn.dropout(query, 1.0 - dropout)
        value = tf.layers.dense(x, units=v_size, use_bias=False,
                                kernel_initializer=tf.random_normal_initializer(0, 0.01))
        value = tf.nn.dropout(value, 1.0 - dropout)

        logits = tf.matmul(query, key, transpose_b=True)
        logits = logits / np.sqrt(k_size)
        weights = tf.nn.softmax(logits, name="attention_weights")
        output = tf.matmul(weights, value)

    return output + x

def batch_norm_conv1d(x, is_training, num_filters, dilation_rate, filter_size=3, stride=[1],
                      pad='VALID', gated=False, counters={}, reuse=False):
    name = get_name('batch_norm_conv1d', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # Gating mechanism (Dauphin 2016 LM with Gated Conv. Nets)
        if gated:
            num_filters = num_filters * 2

        W = tf.get_variable('W', [filter_size, int(x.get_shape()[-1]), num_filters],
                            tf.float32, initializer=tf.random_normal_initializer(0, 0.01),
                            trainable=True)

        b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                            initializer=None, trainable=True)

        # pad x
        #left_pad = dilation_rate * (filter_size - 1)
        #x = temporal_padding(x, (left_pad, 0))

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.convolution(x, W, 'SAME', stride, [dilation_rate]), b)

        # batch normalization
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                         is_training=is_training, scope='bn')

        # apply nonlinearity
        # GLU
        if gated:
            split0, split1 = tf.split(x, num_or_size_splits=2, axis=2)
            split1 = tf.sigmoid(split1)
            x = tf.multiply(split0, split1)
        # ReLU
        else:
            x = tf.nn.relu(x)

        return x

@add_arg_scope
def weightNormConvolution1d(x, num_filters, dilation_rate, filter_size=3, stride=[1],
                            pad='VALID', init_scale=1., init=False, gated=False,
                            counters={}, reuse=False):
    """a dilated convolution with weight normalization
    # Arguments
        x: A tensor of shape [N, L, Cin]
        num_filters: number of convolution filters
        dilation_rate: dilation rate / holes
        filter_size: window / kernel width of each filter
        stride: stride in convolution
    # Returns
        A tensor of shape [N, L, num_filters]
    """
    name = get_name('weight_norm_conv1d', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # currently this part is never used
        if init:
            print("initializing weight norm")
            # data based initialization of parameters
            V = tf.get_variable('V', [filter_size, x.get_shape()[-1], num_filters],
                                tf.float32, tf.random_normal_initializer(0, 0.01),
                                trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1])

            # pad x
            left_pad = dilation_rate * (filter_size - 1)
            x = temporal_padding(x, (left_pad, 0))
            x_init = tf.nn.convolution(x, V_norm, pad, stride, [dilation_rate])
            #x_init = tf.nn.conv2d(x, V_norm, [1]+stride+[1], pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1])
            scale_init = init_scale/tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init,
                                trainable=True)
            b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init*scale_init,
                                trainable=True)
            x_init = tf.reshape(scale_init, [1, 1, num_filters]) \
                                * (x_init - tf.reshape(m_init, [1, 1, num_filters]))
            # apply nonlinearity
            x_init = tf.nn.relu(x_init)
            return x_init

        else:
            # Gating mechanism (Dauphin 2016 LM with Gated Conv. Nets)
            if gated:
                num_filters = num_filters * 2

            # size of V is L, Cin, Cout
            V = tf.get_variable('V', [filter_size, x.get_shape()[-1], num_filters],
                                tf.float32, initializer=tf.random_normal_initializer(0, 0.01),
                                trainable=True)
            g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.), trainable=True)
            b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                                initializer=None, trainable=True)

            # size of input x is N, L, Cin

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1])

            # pad x for causal convolution
            left_pad = dilation_rate * (filter_size  - 1)
            x = temporal_padding(x, (left_pad, 0))

            # calculate convolutional layer output
            x = tf.nn.bias_add(tf.nn.convolution(x, W, pad, stride, [dilation_rate]), b)

            # GLU
            if gated:
                split0, split1 = tf.split(x, num_or_size_splits=2, axis=2)
                split1 = tf.sigmoid(split1)
                x = tf.multiply(split0, split1)
            # ReLU
            else:
                # apply nonlinearity
                x = tf.nn.relu(x)

            print(x.get_shape())

            return x

def TemporalBlock(input_layer, out_channel, filter_size, stride, dilation_rate, counters,
                  dropout, use_highway=False, gated=False, reuse=False,
                  normalization='weight_norm', is_training=False, init=False):
    """A temporal block consists of one dilated CNN and residual connection
    # Arguments
        input_layer: A tensor of shape [N, L, Cin]
        out_channel: output size / # of filter in CNN layer
        filter_size: the window of convolution filter
        stride: stride in convolution
        dilation_rate: dilation rate in dilated convolution
    # Returns
        A tensor of shape [N, L, out_channel]
    """

    keep_prob = 1.0 - dropout

    in_channel = input_layer.get_shape()[-1]
    name = get_name('temporal_block', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # num_filters is the hidden units in TCN
        # which is the number of out channels
        if normalization == 'weight_norm':
            conv1 = weightNormConvolution1d(input_layer, out_channel, dilation_rate,
                                            filter_size, [stride], counters=counters,
                                            init=init, gated=gated, reuse=reuse)
        elif normalization == 'batch_norm':
            conv1 = batch_norm_conv1d(input_layer, is_training, out_channel, dilation_rate,
                                      filter_size, [stride], counters=counters,
                                      gated=gated, reuse=reuse)
        else:
            print('Error: no such normalization : ' + normalization)
            conv1 = weightNormConvolution1d(input_layer, out_channel, dilation_rate,
                                            filter_size, [stride], counters=counters,
                                            init=init, gated=gated, reuse=reuse)

        # set noise shape for spatial dropout
        # refer to https://colab.research.google.com/drive/
        # 1la33lW7FQV1RicpfzyLq9H0SH1VSD4LE#scrollTo=TcFQu3F0y-fy
        # shape should be [N, 1, C]
        noise_shape = (tf.shape(conv1)[0], tf.constant(1), tf.shape(conv1)[2])
        dropout1 = tf.nn.dropout(conv1, keep_prob, noise_shape)

        # highway connetions or residual connection
        residual = None
        if use_highway:
            W_h = tf.get_variable('W_h', [1, int(input_layer.get_shape()[-1]), out_channel],
                                  tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_h = tf.get_variable('b_h', shape=[out_channel], dtype=tf.float32,
                                  initializer=None, trainable=True)
            H = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)

            W_t = tf.get_variable('W_t', [1, int(input_layer.get_shape()[-1]), out_channel],
                                  tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_t = tf.get_variable('b_t', shape=[out_channel], dtype=tf.float32,
                                  initializer=None, trainable=True)
            T = tf.nn.bias_add(tf.nn.convolution(input_layer, W_t, 'SAME'), b_t)
            T = tf.nn.sigmoid(T)
            residual = H*T + input_layer * (1.0 - T)
        elif in_channel != out_channel:
            W_h = tf.get_variable('W_h', [1, int(input_layer.get_shape()[-1]), out_channel],
                                  tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_h = tf.get_variable('b_h', shape=[out_channel], dtype=tf.float32,
                                  initializer=None, trainable=True)
            residual = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)
        else:
            print("no residual convolution")

        res = input_layer if residual is None else residual

        return tf.nn.relu(dropout1 + res)

def TemporalConvNet(input_layer, num_channels, kernel_sizes,
                    dropout=tf.constant(0.0, dtype=tf.float32),
                    init=False, is_training=False, normalization='batch_norm',
                    self_atten=False, use_highway=False, use_gated=False,
                    counters={}, reuse=False):
    """A stacked dilated CNN architecture
    # Arguments
        input_layer: Tensor of shape [N, L, Cin]
        num_channels: # of filters for each CNN layer
        kernel_size: kernel for every CNN layer
        dropout: channel dropout after CNN
    # Returns
        A tensor of shape [N, L, num_channels[-1]]
    """
    num_levels = len(num_channels)
    sub_counters = {}
    name = get_name('temporal_conv_net', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        for i in range(num_levels):
            print(i)
            dilation_size = 2 ** i
            out_channel = num_channels[i]
            kernel_size = kernel_sizes[i]
            input_layer = TemporalBlock(input_layer, out_channel, kernel_size, stride=1,
                                        dilation_rate=dilation_size, dropout=dropout,
                                        counters=sub_counters, reuse=reuse, gated=use_gated,
                                        normalization=normalization,
                                        init=init, is_training=is_training)
            if self_atten:
                input_layer = self_attention_block(input_layer, dropout=dropout,
                                                   reuse=reuse, counters=sub_counters)

    return input_layer

def StackDilatedCNN(input_layer, num_channels, kernel_sizes, num_stacks=1,
                    dropout=tf.constant(0.0, dtype=tf.float32),
                    init=False, is_training=False, normalization='weight_norm',
                    self_atten=False, use_highway=False, use_gated=False,
                    use_multi_scale=False,
                    counters={}, reuse=False):
    """A stacked dilated CNN architecture customize
    # Arguments
        input_layer: Tensor of shape [N, L, Cin]
        num_channels: # of filters for each CNN layer
        kernel_size: kernel for every CNN layer
        dropout: channel dropout after CNN
    # Returns
        A tensor of shape [N, L, num_channels[-1]]
    """
    num_levels = len(num_channels)
    sub_counters = {}
    name = get_name('stack_dilated_cnn', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        x = input_layer

        # how many stacks
        for _ in range(num_stacks):
            # first stack
            layers = []
            for i in range(num_levels):
                print(i)
                dilation_size = 2 ** i
                out_channel = num_channels[i]
                kernel_size = kernel_sizes[i]
                x = TemporalBlock(x, out_channel, kernel_size, stride=1,
                                  dilation_rate=dilation_size, dropout=dropout,
                                  counters=sub_counters, reuse=reuse, gated=use_gated,
                                  normalization=normalization,
                                  init=init, is_training=is_training)
                layers.append(x)

            if use_multi_scale:
                # feature ensemble and multi-scale attention
                stack_layers = tf.stack(layers, axis=3) # N, L, nhid, num_levels
                filter_ensem = tf.reduce_sum(stack_layers, axis=2) # (N, L, num_levels)
                atten_weight = tf.contrib.layers.fully_connected(filter_ensem, num_levels, activation_fn=tf.nn.relu)
                atten_weight = tf.expand_dims(tf.nn.softmax(atten_weight, axis=2), axis=2) # (N, L, 1, num_levels)
                weighted_sum = tf.multiply(atten_weight, stack_layers) # (N, L, nhid, num_levels)
                x = tf.reduce_sum(weighted_sum, axis=3) # (N, L, nhid)

            if self_atten:
                x = self_attention_block(x, dropout=dropout,
                                         reuse=reuse, counters=sub_counters)
        # last stack
        for i in range(num_levels):
            print(i)
            dilation_size = 2 ** i
            out_channel = num_channels[i]
            kernel_size = kernel_sizes[i]
            x = TemporalBlock(x, out_channel, kernel_size, stride=1,
                              dilation_rate=dilation_size, dropout=dropout,
                              counters=sub_counters, reuse=reuse, gated=use_gated,
                              normalization=normalization,
                              init=init, is_training=is_training)

    return x

def HADCN(X, num_classes, doc_length, sent_length, kwarg_sent, kwarg_doc,
          is_training=False, init=False, is_stacked=False):
    """Return a HADCN model
    # Arguments
        X: A tensor of shape [batch_size, doc_size, sen_size, emsize]
        num_classes: # of output / class
        doc_length: max # of sentences among all documents
        sent_length: max # of words among all sentence
        kwarg_sent: arguments for sentence TCN encoder
        kwarg_doc: arguments for document TCN encoder
    """
    batch_size = tf.shape(X)[0]
    document_size = tf.shape(X)[1]
    sentence_size = tf.shape(X)[2]
    emsize = X.get_shape()[3]
    counters = {}

    with tf.variable_scope('HADCN'):
        # sentence encoder
        X_r = tf.reshape(X, (batch_size * document_size, sentence_size, emsize))
        # [batch * doc, sent, num_channels[-1]]
        if is_stacked:
            sent_out = StackDilatedCNN(X_r, counters=counters, reuse=False, **kwarg_sent)
        else:
            sent_out = TemporalConvNet(X_r, counters=counters, reuse=False, **kwarg_sent)
        sent_out_size = sent_out.get_shape()[-1]
        atten_sent_query = tf.get_variable(name='atten_sent_query', dtype=tf.float32,
                                           shape=(sent_out_size), initializer=None)
        with tf.variable_scope('sent_atten'):
            # [batch * doc, num_channels[-1]]
            sent_encode = han_attention(sent_out, atten_sent_query)

        # document encoding
        sent_encode_r = tf.reshape(sent_encode, (batch_size, document_size, sent_out_size))
        if is_stacked:
            doc_out = StackDilatedCNN(sent_encode_r, counters=counters, reuse=False, **kwarg_doc)
        else:
            doc_out = TemporalConvNet(sent_encode_r, counters=counters, reuse=False, **kwarg_doc)
        doc_out_size = doc_out.get_shape()[-1]
        atten_doc_query = tf.get_variable(name='atten_doc_query', dtype=tf.float32,
                                          shape=(doc_out_size), initializer=None)
        with tf.variable_scope('doc_atten'):
            doc_encode = han_attention(doc_out, atten_doc_query)

        score = tf.contrib.layers.fully_connected(doc_encode, num_classes, activation_fn=None)
        predict = tf.cast(tf.argmax(score, axis=1), dtype=tf.int32)
    return score, predict

def SDCN(X, num_classes, kwarg,
         is_training=False, init=False, is_stacked=False):
    """Return a SDCN model
    # Arguments
        X: A tensor of shape [batch_size, doc_size, emsize]
        num_classes: # of output / class
        kwarg: arguments for document encoder
    """
    batch_size = tf.shape(X)[0]
    document_size = tf.shape(X)[1]
    emsize = X.get_shape()[2]
    counters = {}

    with tf.variable_scope('SDCN'):
        # [batch, doc, num_channels[-1]]
        if is_stacked:
            doc_out = StackDilatedCNN(X, counters=counters, reuse=False, **kwarg)
        else:
            doc_out = TemporalConvNet(X, counters=counters, reuse=False, **kwarg)
        doc_out_size = doc_out.get_shape()[-1]
        atten_doc_query = tf.get_variable(name='atten_doc_query', dtype=tf.float32,
                                          shape=(doc_out_size), initializer=None)
        with tf.variable_scope('doc_atten'):
            # [batch, num_channels[-1]]
            doc_encode = han_attention(doc_out, atten_doc_query)

        score = tf.contrib.layers.fully_connected(doc_encode, num_classes, activation_fn=None)
        predict = tf.cast(tf.argmax(score, axis=1), dtype=tf.int32)
    return score, predict
