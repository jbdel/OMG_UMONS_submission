import numpy as np
import tensorflow as tf
import os
import time
import datetime
import sys
import collections
import utils
from transformer import Transformer
from transformer_MM import Transformer_MM



# Data
tf.flags.DEFINE_string("data_path", "./data/", "Data path")
tf.flags.DEFINE_string("train", "context_text_train.npy,context_video_train.npy,context_audio_train.npy", "Data path to training")
tf.flags.DEFINE_string("validation", "context_text_validation.npy,context_video_validation.npy,context_audio_validation.npy", "Data path to dev")
tf.flags.DEFINE_string("calculateEvaluationCCC", "./data/calculateEvaluationCCC.py", "CCC eval script")
tf.flags.DEFINE_string("fileCSV", "./data/omg_ValidationVideos.csv", "validation file")

tf.flags.DEFINE_string("checkpoint_path", "./runs/", "Data path to dev")

# Mode
tf.flags.DEFINE_boolean("eval", False, "Use test set")

# model parameter
tf.flags.DEFINE_integer("stack_num", 4, 'stack num')
tf.flags.DEFINE_integer("d_k", 64, 'key dim')
tf.flags.DEFINE_integer("d_v", 64, 'value dim')
tf.flags.DEFINE_integer("h", 10, 'stack of multihead attention')
tf.flags.DEFINE_integer("d_ff", 1024, 'feed forward dim')
tf.flags.DEFINE_float('dropout_keep', 0.8, 'dropout keep rate')
tf.flags.DEFINE_boolean("use_cbp", False, "cbp?")

#hyper parameters
tf.flags.DEFINE_integer("batch_size", 2, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("early_stop", 10, "Stop if no improvement after x epoch")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()



datasets = []
splits = []
for d in [FLAGS.train, FLAGS.validation]:
    data = []
    split = []
    files = d.split(',')
    for f in files:
        x = np.load(os.path.join(FLAGS.data_path,f))
        print("File",f, "with shape",x.shape)
        data.append(x)
        split.append(x.shape[1])

    datasets.append(np.concatenate(data,1))
    splits.append(np.array(split))

print("Training size : ",datasets[0].shape)
print("Validation size : ",datasets[1].shape)
print("Training split : ",splits[0])
print("Validation split : ",splits[0])

X = datasets[0]
X_dev = datasets[1]



dev_mode = "Test" if FLAGS.eval else "Validation"

#stack data per videos : size [videos, uterances, feature_size]
X_sorted, Y_sorted, X_seq_length, X_pad_length = utils.get_sorted_data(X, "Train", FLAGS.data_path)
X_dev_sorted, Y_dev_sorted, X_dev_seq_length, X_dev_pad_length = utils.get_sorted_data(X_dev, dev_mode, FLAGS.data_path)


#creating checkpoint folder
out_dir = os.path.join(FLAGS.checkpoint_path, str(int(time.time())))

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        model = Transformer_MM(num_features=X_sorted.shape[2],
                                splits=splits[0],
                                batch_size=FLAGS.batch_size,
                                stack_num=FLAGS.stack_num,
                                d_k=FLAGS.d_k,
                                d_v=FLAGS.d_v,
                                h=FLAGS.h,
                                d_ff=FLAGS.d_ff,
                                pad_length=X_pad_length,
                                use_cbp=FLAGS.use_cbp
                                )

        model_dev = Transformer_MM(num_features=X_dev_sorted.shape[2],
                                splits=splits[1],
                                batch_size=X_dev_sorted.shape[0],
                                stack_num=FLAGS.stack_num,
                                d_k=FLAGS.d_k,
                                d_v=FLAGS.d_v,
                                h=FLAGS.h,
                                d_ff=FLAGS.d_ff,
                                pad_length=X_dev_pad_length,
                                use_cbp=FLAGS.use_cbp
                                )

        # Define Training procedure
        train_op = tf.train.AdamOptimizer(5e-5).minimize(model.loss)
        global_step = 0

        saver = tf.train.Saver(tf.all_variables())
        sess.run(tf.initialize_all_variables())


        def train_step(X_batch, Y_batch, X_batch_seq_length):
            """
            A single training step
            """
            feed_dict = {
              model.inputs: X_batch,
              model.outputs: Y_batch,
              model.inputs_lengths: X_batch_seq_length,
              model.dropout_keep_prob: FLAGS.dropout_keep
            }

            _, loss = sess.run(
                 [train_op, model.loss],
                 feed_dict)

        def dev_step(X_batch, Y_batch, X_batch_seq_length):
            """
            A single training step
            """
            feed_dict = {
                model_dev.inputs: X_batch,
                model_dev.outputs: Y_batch,
                model_dev.inputs_lengths: X_batch_seq_length,
                model_dev.dropout_keep_prob: 1.0
            }

            loss, scores = sess.run(
                [model_dev.loss, model_dev.masked_scores],
                feed_dict)
            scores = np.array(scores).T

            mean_ccc = utils.write_ccc_csv(scores, FLAGS.calculateEvaluationCCC, FLAGS.fileCSV, out_dir, verbose=False)

            #0 is mean, 1 is ar, 2 is val
            if mean_ccc[0] > dev_step.best_loss[0]:
                dev_step.best_loss = mean_ccc

                path = saver.save(sess, os.path.join(out_dir, "checkpoint"), global_step=global_step)

                best_scores_path = os.path.join(out_dir, "best_scores.txt")
                with open(best_scores_path , "a+") as f:
                    f.write("{} \t {} \n".format(global_step, dev_step.best_loss[0]))

                dev_step.early_stop = 0


            time_str = datetime.datetime.now().isoformat()
            print("dev {}: step {}, loss {}".format(time_str, global_step, loss))


        dev_step.best_loss = [0.0, 0.0, 0.0]
        dev_step.early_stop = 0
        # Generate batches
        batches = utils.batch_iter(X_sorted, Y_sorted, X_seq_length, FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            global_step += 1

            X_batch, Y_batch, X_batch_seq_length = batch
            train_step(X_batch, Y_batch, X_batch_seq_length)
            if global_step % FLAGS.evaluate_every == 0:
                dev_step.early_stop += 1
                print("Evaluation:")
                dev_step(X_dev_sorted, Y_dev_sorted, X_dev_seq_length)
                print("")

            how_many_eval = int((X_sorted.shape[0]/FLAGS.batch_size/FLAGS.evaluate_every)*FLAGS.early_stop)
            if dev_step.early_stop == how_many_eval:
                with open(os.path.join("runs", "best_scores_overall.txt"), "a+") as f:
                    f.write("{}({}/{}) \t {} \n".format(dev_step.best_loss[0], dev_step.best_loss[1], dev_step.best_loss[2], out_dir))
                sys.exit()
