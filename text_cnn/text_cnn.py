import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, num_reels=2):

        # Placeholders for input, output and dropout
        self.input_x           = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y           = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_reels       = tf.placeholder(tf.float32, [None, 2], name="input_reels")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_reg_lambda     = l2_reg_lambda


        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="embedding")
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        self.embedded_chars_expanded = tf.nn.dropout(self.embedded_chars_expanded, self.dropout_keep_prob)


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, (filter_size, num_filter) in enumerate(zip(filter_sizes, num_filters)):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total=0
        for i, (filter_size, num_filter) in enumerate(zip(filter_sizes, num_filters)):
            num_filters_total += num_filter

        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="pre_fc")

        # Add dropout
        with tf.name_scope("dropout"):
            self.pre_FC = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob, name="drop")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):

            self.W1 = tf.get_variable(
                "W1",
                shape=[num_filters_total, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.Variable(tf.constant(0.1, shape=[2]), name="b1")


            # self.W2 = tf.get_variable(
            #     "W2",
            #     shape=[128, num_reels],
            #     initializer=tf.contrib.layers.xavier_initializer())
            # self.b2 = tf.Variable(tf.constant(0.1, shape=[num_reels]), name="b2")


            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(self.W1)
            l2_loss += tf.nn.l2_loss(self.b1)
            # l2_loss += tf.nn.l2_loss(self.W2)
            # l2_loss += tf.nn.l2_loss(self.b2)

            self.scores    = tf.nn.xw_plus_b(self.pre_FC, self.W1, self.b1, name="pre")
            # self.scores = tf.nn.xw_plus_b(self.pre, self.W2, self.b2, name="scores")


        with tf.name_scope("loss"):
            loss = tf.losses.mean_squared_error(self.scores, self.input_reels)
            self.loss = loss + self.l2_reg_lambda * l2_loss

