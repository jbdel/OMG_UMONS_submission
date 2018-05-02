#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import sys

# Parameters
# ==================================================



# python eval.py \
#     --test_data_path ./data/train.txt \
#     --validationCSV ./data/omg_TrainVideos.csv
#
#
# python eval.py \
#     --test_data_path ./data/validation.txt \
#     --validationCSV ./data/omg_ValidationVideos.csv
#
#
# python eval.py \
#     --test_data_path ./data/test.txt \
#     --compute_ccc False


# Eval Parameters
tf.flags.DEFINE_string("test_data_path", "./data/train.txt", "Data path to evaluation")
tf.flags.DEFINE_string("checkpoint_dir", "/media/jb/DATA/OMGEmotionChallenge/text_cnn/runs/1525115737/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_string("calculateEvaluationCCC", "./data/calculateEvaluationCCC.py", "path to ccc script")
tf.flags.DEFINE_string("validationCSV", "./data/omg_TrainVideos.csv", "path to ccc script")
tf.flags.DEFINE_boolean("compute_ccc", True, "compute_ccc ?")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

x_test, y_test, arousals, valences, video_ids, utterances, vocabulary, vocabulary_inv, onehot_label, max_sequence_length = data_helpers.load_data(FLAGS.test_data_path, FLAGS.checkpoint_dir)


# Evaluation
# ==================================================

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        print ("FLAGS.checkpoint_dir %s" % FLAGS.checkpoint_dir)
        print ("checkpoint_file %s" % checkpoint_file)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        scores = graph.get_operation_by_name("output/pre").outputs[0]
        pre = graph.get_operation_by_name("pre_fc").outputs[0]

        pre, scores_ = sess.run([pre,scores], {input_x: x_test, dropout_keep_prob: 1.0})

        #saving represenation
        fname = os.path.basename(FLAGS.test_data_path)
        print("Features shape :", pre.shape)
        print("Saving Features at :",os.path.join(FLAGS.checkpoint_dir, fname) +".npy")
        np.save(os.path.join(FLAGS.checkpoint_dir, fname),pre)

        if FLAGS.compute_ccc:
            mean_ccc = data_helpers.get_CCC_score(FLAGS, FLAGS.checkpoint_dir,
                                                  scores_, video_ids, utterances)
