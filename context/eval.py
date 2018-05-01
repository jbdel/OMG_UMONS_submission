#! /usr/bin/env python

import tensorflow as tf
import os
import sys
import numpy as np
import utils
from transformer import Transformer

#
# python eval.py \
#     --test_data_path ./data/video_validation.npy \
#     --mode Validation \
#     --checkpoint_dir ./runs/v \
#     --checkpoint_file checkpoint-1850 \
#     --out_name context_video_validation \
#     --fileCSV ./data/omg_ValidationVideos.csv \
#     --calculateEvaluationCCC ./data/calculateEvaluationCCC.py \
#     --compute_ccc True \
#     --stack_num 2 \
#     --d_k 64 \
#     --d_v 64 \
#     --h 8 \
#     --d_ff 256
#



# python eval.py \
#     --test_data_path ./data/video_train.npy \
#     --mode Train \
#     --checkpoint_dir ./runs/v \
#     --checkpoint_file checkpoint-1850 \
#     --out_name context_video_train \
#     --fileCSV ./data/omg_TrainVideos.csv \
#     --calculateEvaluationCCC ./data/calculateEvaluationCCC.py \
#     --compute_ccc True \
#     --stack_num 2 \
#     --d_k 64 \
#     --d_v 64 \
#     --h 8 \
#     --d_ff 256



#
# python eval.py \
#     --test_data_path ./data/train.txt.npy \
#     --mode Train \
#     --checkpoint_dir ./runs/t \
#     --checkpoint_file checkpoint-450 \
#     --out_name context_text_train \
#     --fileCSV ./data/omg_TrainVideos.csv \
#     --calculateEvaluationCCC ./data/calculateEvaluationCCC.py \
#     --compute_ccc True \
#     --stack_num 2 \
#     --d_k 64 \
#     --d_v 64 \
#     --h 8 \
#     --d_ff 256



# python eval.py \
#     --test_data_path ./data/validation.txt.npy \
#     --mode Validation \
#     --checkpoint_dir ./runs/t \
#     --checkpoint_file checkpoint-450 \
#     --out_name context_text_validation \
#     --fileCSV ./data/omg_ValidationVideos.csv \
#     --calculateEvaluationCCC ./data/calculateEvaluationCCC.py \
#     --compute_ccc True \
#     --stack_num 2 \
#     --d_k 64 \
#     --d_v 64 \
#     --h 8 \
#     --d_ff 256



#
# python eval.py \
#     --test_data_path ./data/test.txt.npy \
#     --mode Test \
#     --checkpoint_dir ./runs/t \
#     --checkpoint_file checkpoint-450 \
#     --out_name context_text_test \
#     --compute_ccc False \
#     --stack_num 2 \
#     --d_k 64 \
#     --d_v 64 \
#     --h 8 \
#     --d_ff 256




#
# python eval.py \
#     --test_data_path ./data/audio_train.npy \
#     --mode Train \
#     --checkpoint_dir ./runs/a \
#     --checkpoint_file checkpoint-2650 \
#     --out_name context_audio_train \
#     --fileCSV ./data/omg_TrainVideos.csv \
#     --calculateEvaluationCCC ./data/calculateEvaluationCCC.py \
#     --compute_ccc True \
#     --stack_num 2 \
#     --d_k 64 \
#     --d_v 64 \
#     --h 8 \
#     --d_ff 256



# python eval.py \
#     --test_data_path ./data/audio_validation.npy \
#     --mode Validation \
#     --checkpoint_dir ./runs/a \
#     --checkpoint_file checkpoint-2650 \
#     --out_name context_audio_validation \
#     --fileCSV ./data/omg_ValidationVideos.csv \
#     --calculateEvaluationCCC ./data/calculateEvaluationCCC.py \
#     --compute_ccc True \
#     --stack_num 2 \
#     --d_k 64 \
#     --d_v 64 \
#     --h 8 \
#     --d_ff 256
#
#




# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("test_data_path", "./data/validation.txt.npy", "Data path to evaluation")
tf.flags.DEFINE_string("mode", "Validation", "[Train, Validation, Test]")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1524730055/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("checkpoint_file", "checkpoint-650", "If not specified, take last checkpoint")
tf.flags.DEFINE_string("calculateEvaluationCCC", "./data/calculateEvaluationCCC.py", "path to ccc script")
tf.flags.DEFINE_string("fileCSV", "./data/omg_ValidationVideos.csv", "path to ccc script")
tf.flags.DEFINE_string("out_name", "default", "filename for features")

tf.flags.DEFINE_boolean("compute_ccc", True, "compute_ccc ?")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# model parameter
tf.flags.DEFINE_integer("stack_num", 3, 'stack num')
tf.flags.DEFINE_integer("d_k", 64, 'key dim')
tf.flags.DEFINE_integer("d_v", 64, 'value dim')
tf.flags.DEFINE_integer("h", 10, 'stack of multihead attention')
tf.flags.DEFINE_integer("d_ff", 1024, 'feed forward dim')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()



X = np.load(FLAGS.test_data_path)
X_sorted, Y_sorted, X_seq_length, X_pad_length = utils.get_sorted_data(X, FLAGS.mode, os.path.dirname(FLAGS.test_data_path))


# Evaluation
# ==================================================

if(FLAGS.checkpoint_file == ""):
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
else:
    checkpoint_file = os.path.join(FLAGS.checkpoint_dir, FLAGS.checkpoint_file)

graph = tf.Graph()


with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        print ("FLAGS.checkpoint_dir %s" % FLAGS.checkpoint_dir)
        print ("checkpoint_file %s" % checkpoint_file)

        model_dev = Transformer(num_features=X_sorted.shape[2],
                                batch_size=X_sorted.shape[0],
                                stack_num=FLAGS.stack_num,
                                d_k=FLAGS.d_k,
                                d_v=FLAGS.d_v,
                                h=FLAGS.h,
                                d_ff=FLAGS.d_ff,
                                pad_length=X_pad_length
                                )

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)

        feed_dict = {
            model_dev.inputs: X_sorted,
            model_dev.outputs: Y_sorted,
            model_dev.inputs_lengths: X_seq_length,
            model_dev.dropout_keep_prob: 1.0
        }

        pre, scores = sess.run(
            [model_dev.pre_fc, model_dev.masked_scores],
            feed_dict)

        scores = np.array(scores).T

        data_path = os.path.dirname(FLAGS.test_data_path)
        # saving represenation
        print("Features shape :", pre.shape)
        print("Saving Features at :", os.path.join(data_path, FLAGS.out_name)+".npy")


        np.save(os.path.join(data_path, FLAGS.out_name), pre)

        if FLAGS.compute_ccc:
            mean_ccc = utils.write_ccc_csv(scores, FLAGS.calculateEvaluationCCC, FLAGS.fileCSV, FLAGS.checkpoint_dir, verbose=False)
