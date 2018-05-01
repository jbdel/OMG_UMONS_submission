#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import datetime
import data_helpers
from text_cnn import TextCNN
import sys
import codecs
from gensim.models import KeyedVectors


# Model Hyperparameters
tf.flags.DEFINE_string("train_data_path", "./data/train.txt", "Data path to training")
tf.flags.DEFINE_string("dev_data_path", "./data/validation.txt", "Data path to dev")
tf.flags.DEFINE_string("w2v_data_path", "./data/w2v.npy", "Data path to dev")
tf.flags.DEFINE_string("word2vec", "./data/GoogleNews-vectors-negative300.bin", "Word2vec file with pre-trained embeddings")
tf.flags.DEFINE_string("calculateEvaluationCCC", "./data/calculateEvaluationCCC.py", "path to ccc script")
tf.flags.DEFINE_string("validationCSV", "./data/omg_ValidationVideos.csv", "path to csv file")

tf.flags.DEFINE_boolean("use_ccc", True, "Use ccc score to pick best model ?")
tf.flags.DEFINE_boolean("use_word2vec", True, "Use word2vec?")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("batch_size", 8, "Batch Size (default: 64)")
tf.flags.DEFINE_string("filter_sizes", "3,4,2", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("num_filters", "30,30,60", "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.15, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


#creating directories
out_dir, checkpoint_dir, checkpoint_prefix = data_helpers.create_directories()


# Load data
print("Loading data...")

x_train, y_train, arousals_train, valences_train, \
video_ids, utterances, \
vocabulary, vocabulary_inv, \
onehot_label, \
max_sequence_length = data_helpers.load_data(FLAGS.train_data_path)

# Save additional model info
codecs.open(os.path.join(checkpoint_dir, "max_sent_len"), "w", encoding='utf8').write(str(max_sequence_length))
codecs.open(os.path.join(checkpoint_dir, "vocab"), "w", encoding='utf8').write('\n'.join(vocabulary_inv))
codecs.open(os.path.join(checkpoint_dir, "label"), "w", encoding='utf8').write('\n'.join(onehot_label.values()))


x_dev, y_dev, arousals_dev, valences_dev, \
video_ids_, utterances_, \
vocabulary_, vocabulary_inv_, \
onehot_label_, \
max_sequence_length_ = data_helpers.load_data(FLAGS.dev_data_path, checkpoint_dir=checkpoint_dir)

#the labels arent used here
print("Labels: %d: %s" % ( len(onehot_label), ','.join( onehot_label.values() ) ) )
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

if FLAGS.use_word2vec:
    FLAGS.embedding_dim = 300

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=len(onehot_label),
            vocab_size=len(vocabulary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=list(map(int, FLAGS.num_filters.split(","))),
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = 0
        train_op = tf.train.AdamOptimizer(0.001).minimize(cnn.loss)

        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        if FLAGS.use_word2vec:
            # initial matrix with random uniform
            if os.path.isfile(FLAGS.w2v_data_path):
                initE = np.load(FLAGS.w2v_data_path)
            else:
                initE = np.random.uniform(-1.0,1.0, (len(vocabulary), FLAGS.embedding_dim))

                # load any vectors from the word2vec
                print("Loading word2vec file {}\n".format(FLAGS.word2vec))
                word_vectors = KeyedVectors.load_word2vec_format(FLAGS.word2vec, binary=True)
                print("Loaded")
                print("Creating embedding data file for vocabulary")
                for i,w in enumerate(vocabulary_inv):
                    try:
                        initE[i] = word_vectors[w]
                    except KeyError:
                        print("No word2vec for word",w)

                np.save(FLAGS.w2v_data_path, initE)

            sess.run(cnn.embedding.assign(initE))


        def train_step(x_batch, y_batch, arousals_batch, valences_batch):

            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.input_reels: np.array([arousals_batch, valences_batch]).T,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _, loss, scores = sess.run(
                 [train_op, cnn.loss, cnn.scores],
                 feed_dict)

            time_str = datetime.datetime.now().isoformat()
            if global_step % 50 == 0:
                print("{}: step {}, loss {}".format(time_str, global_step, loss))



        def dev_step(x_batch, y_batch, arousals_batch, valences_batch):

            dev_step.early_stop += 1

            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.input_reels: np.array([arousals_batch, valences_batch]).T,
              cnn.dropout_keep_prob: 1.0
            }

            loss, scores, W1 = sess.run(
                 [cnn.loss, cnn.scores, cnn.W1],
                 feed_dict)

            time_str = datetime.datetime.now().isoformat()

            #compute ccc scores
            mean_ccc = data_helpers.get_CCC_score(FLAGS, checkpoint_prefix,
                                                  scores, video_ids_, utterances_)

            #logging best dev
            if mean_ccc[0] > dev_step.best_loss[0]:
                print("Best dev beaten. Before ",dev_step.best_loss[0], "Now", mean_ccc)
                dev_step.best_loss=mean_ccc
                path = saver.save(sess, checkpoint_prefix, global_step=global_step)
                data_helpers.log_score(checkpoint_prefix, global_step,
                                       dev_step.best_loss[0],
                                       path, out_dir, W1)

                dev_step.early_stop = 0

            print("{}: step {}, loss {}".format(time_str, global_step, loss))

        # Generate batches
        dev_step.best_loss = [0.0,0.0,0.0]
        dev_step.early_stop = 0

        stack = list(zip(x_train, y_train, arousals_train, valences_train))
        batches = data_helpers.batch_iter(stack, FLAGS.batch_size)


        # Training loop. For each batch...
        for batch in batches:
            global_step+=1
            x_batch, y_batch, arousals_batch, valences_batch = zip(*batch)
            train_step(x_batch, y_batch, arousals_batch, valences_batch)
            if global_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, arousals_dev, valences_dev)
                if dev_step.early_stop == 10:
                    with open(os.path.join("runs", "best_scores_overall.txt"), "a+") as f:
                        f.write("{}({}/{}) \t {} \n".format(dev_step.best_loss[0], dev_step.best_loss[1],
                                                            dev_step.best_loss[2], out_dir))
                    sys.exit()

