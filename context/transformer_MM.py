import tensorflow as tf
import numpy as np
import sys
import tensorflow as tf
from count_sketch import count_sketch, bilinear_pool



'''
Transformer modules
'''

def add_and_norm(x, sub_x, dropout_keep):
  with tf.variable_scope('add_and_norm'):
    sub_x = tf.nn.dropout(sub_x, dropout_keep)
    return tf.contrib.layers.layer_norm(x + sub_x)

def feed_forward(x, d_ff):
  output_dim = x.get_shape()[-1]
  with tf.variable_scope('feed_forward'):
    x = tf.layers.dense(x, d_ff, activation=tf.nn.relu)
    x = tf.layers.dense(x, output_dim)
    return x

def multihead_attention_block(vk_input, q_input,
    batch_size, pad_length, d_model, d_k, d_v, masked=False):

  with tf.variable_scope('multihead_attention'):
    K = tf.layers.dense(vk_input, d_k, name='K', activation=tf.nn.relu)
    V = tf.layers.dense(vk_input, d_v, name='V', activation=tf.nn.relu)
    Q = tf.layers.dense(q_input, d_k, name='Q', activation=tf.nn.relu)

    '''
    Scaled Dot-Product Attention
    '''
    # Mask (pad_length x pad_length)
    mask = tf.ones([pad_length, pad_length])
    if masked == True:
      #mask = tf.linalg.LinearOperatorLowerTriangular(mask, f.float32).to_dense()
      mask = tf.contrib.linalg.LinearOperatorTriL(mask, tf.float32).to_dense()
    mask = tf.reshape(tf.tile(mask, [batch_size, 1]),
        [batch_size, pad_length, pad_length])

    # Attention(Q,K,V)
    attn = tf.nn.softmax(
        mask * (Q @ tf.transpose(K, [0, 2, 1])) / tf.sqrt(tf.to_float(d_k))) @ V

    return attn

def multihead_attention(vk_input, q_input, pad_length, num_features,d_k, d_v, h, masked=False):
  outputs = []

  batch_size = tf.shape(vk_input)[0]

  for i in range(h):
    outputs.append(
        multihead_attention_block(vk_input, q_input,
          batch_size, pad_length, num_features, d_k, d_v, masked=masked))
  outputs = tf.concat(outputs, axis=2)
  outputs = tf.layers.dense(outputs, num_features)
  return outputs

'''
Transformer Encoder block
'''
def encoder_block(inputs, dropout_keep, pad_length, num_features,d_k, d_v, h, d_ff):
  # load hyper parameters

  with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
    flow = multihead_attention(inputs, inputs, pad_length, num_features,d_k, d_v, h)
    flow = add_and_norm(inputs, flow, dropout_keep)
    flow = add_and_norm(flow, feed_forward(flow, d_ff), dropout_keep)
    return flow

'''
Positional Encoding
'''
def positional_encoding(x, pad_length, num_features):

  def sincos(x, i):
    if i%2 == 0:
      return np.sin(x)
    return np.cos(x)

  with tf.variable_scope('positional_encoding'):
    pe = tf.convert_to_tensor([sincos(pos/(10000**(2*i/num_features)), i)
      for pos in range(1, pad_length+1) for i in range(1, num_features+1)])
    pe = tf.reshape(pe, [-1, pad_length, num_features])
    return tf.add(x, pe)

'''
Transformer class
'''
class Transformer_MM(object):
  def __init__(self, num_features=None,
               splits=None,
               batch_size = None,
               stack_num=None,
               d_k=None,
               d_v=None,
               h=None,
               d_ff=None,
               pad_length=None,
               use_cbp=None,
               d=512
               ):
    self.pad_length = pad_length
    self.splits = splits

    self.inputs = tf.placeholder(tf.float32, shape=[batch_size, self.pad_length, num_features], name="input_x")
    self.inputs_split = self.inputs

    if use_cbp:
      inputs_resh = tf.reshape(self.inputs, [-1, num_features])
      inputs_split = tf.split(inputs_resh, self.splits, 1, name="split")
      bp = bilinear_pool(inputs_split, d)
      self.inputs_split = tf.reshape(bp, [batch_size, self.pad_length, d])
      num_features = d

    self.inputs_lengths    = tf.placeholder(tf.int32,   shape=[batch_size], name="input_length_x")
    self.outputs           = tf.placeholder(tf.float32, shape=[batch_size, self.pad_length, 2], name="input_y")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    self.stack_num = stack_num
    self.d_k       = d_k
    self.d_v       = d_v
    self.h         = h
    self.d_ff      = d_ff


    with tf.variable_scope('transformer'):

      inputs = positional_encoding(self.inputs_split, self.pad_length, num_features)

      for i in range(self.stack_num):
        with tf.variable_scope('enc_b_' + str(i)):

          inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)
          inputs = encoder_block(inputs, self.dropout_keep_prob, self.pad_length, num_features, self.d_k, self.d_v, self.h, self.d_ff)


      with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
        pre_fc = tf.reshape(inputs, [-1, self.pad_length, num_features], name="pre_fc")
        pre_FC_resh = tf.reshape(pre_fc, [-1, num_features])


        # W2 = tf.get_variable(
        #   "W2",
        #   shape=[num_features, 512],
        #   initializer=tf.contrib.layers.xavier_initializer())
        # b2 = tf.Variable(tf.constant(0.1, shape=[512]), name="b2")
        #
        #
        # wot = (tf.nn.xw_plus_b(pre_FC_resh, W2, b2, name="scorxes"))

        # wot = tf.nn.dropout(wot, self.dropout_keep_prob)

        W1 = tf.get_variable(
          "W1",
          shape=[num_features, 2],
          initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.constant(0.1, shape=[2]), name="b1")

        scores = tf.sigmoid(tf.nn.xw_plus_b(pre_FC_resh, W1, b1, name="scores"))


        # scores = tf.layers.dense(pre_FC_resh, 2, activation=tf.sigmoid)
        self.scores = tf.reshape(scores, [-1, self.pad_length, 2])

      with tf.variable_scope('masking', reuse=tf.AUTO_REUSE):

        mask = tf.sequence_mask([self.inputs_lengths,self.inputs_lengths], self.pad_length, dtype=tf.int32)
        mask = tf.transpose(mask, perm=[1, 2, 0])
        # https://stackoverflow.com/questions/39065517/how-to-mask-vectors-in-reduce-xxx-tensorflow-operations
        _, score_ar =  tf.dynamic_partition(self.scores[:, :, 0], mask[:, :, 0], 2, name="score_ar")
        _, score_val = tf.dynamic_partition(self.scores[:, :, 1], mask[:, :, 1], 2, name="score_val")
        self.masked_scores = [score_ar, score_val]

        #we also mask the features, for extraction
        _, self.pre_fc =  tf.dynamic_partition(pre_fc, mask[:, :, 0], 2, name="pre_fc_features")

      with tf.name_scope("loss"):
        l2_loss = tf.constant(0.0)
        l2_loss += tf.nn.l2_loss(W1)
        l2_loss += tf.nn.l2_loss(b1)

        if(True):
            loss_mse = tf.losses.mean_squared_error(
              self.scores,
              self.outputs,
            weights=mask)


            with tf.name_scope("masking_GT"):
              # doing the same for ground truth
              _, GT_ar = tf.dynamic_partition(self.outputs[:, :, 0], mask[:, :, 0], 2)
              _, GT_val = tf.dynamic_partition(self.outputs[:, :, 1], mask[:, :, 1], 2)

            with tf.name_scope("loss"):
              def concordance_cc(predictions, ground_truth):
                pred_mean, pred_var = tf.nn.moments(predictions, [0])
                gt_mean, gt_var = tf.nn.moments(ground_truth, [0])
                # mean_cent_prod = tf.reduce_mean((predictions - pred_mean) * (ground_truth - gt_mean))
                mean_cent_prod = tf.reduce_mean((tf.subtract(predictions,pred_mean)) * (tf.subtract(ground_truth,gt_mean)))
                return 1 - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))


              loss_ccc = concordance_cc(score_ar, GT_ar) + concordance_cc(score_val, GT_val)


            self.loss = loss_mse  + 0.15*l2_loss + 0.15*loss_ccc

