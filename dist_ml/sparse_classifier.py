# Most of the tensorflow code is adapted from Tensorflow's tutorial on using
# CNNs to train MNIST
# https://www.tensorflow.org/get_started/mnist/pros#build-a-multilayer-convolutional-network.  # noqa: E501

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import datetime
import logging
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)

# from . import sparse_model
# import sparse_model

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def parse_tfrecords_function(example_proto):
    """
      Decode TFRecords for Dataset.

      Args:
        example_proto: TensorFlow ExampleProto object.

      Return:
        The op of features and labels
      """
    features = {
        "ids": tf.VarLenFeature(tf.int64),
        "values": tf.VarLenFeature(tf.float32),
        "label": tf.FixedLenFeature([], tf.int64, default_value=0)
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    labels = parsed_features["label"]
    ids = parsed_features["ids"]
    values = parsed_features["values"]

    return labels, ids, values


class SparseClassifier(object):
    # init func: hyperparameters
    def __init__(self, learning_rate=1e-3, epoch_num=100,
                 train_batch_size=256, train_files=None, num_train_file=4, num_worker=4, worker_id=1,
                 test_batch_size=256, test_files=None, num_test_file=1):

        train_buffer_size = train_batch_size * 3
        test_buffer_size = test_batch_size * 3

        with tf.Graph().as_default():
            # Loading training data from TFRecords
            train_filename_list = self.get_train_files(train_files, num_test_file, num_worker, worker_id)
            train_filename_placeholder = tf.placeholder(tf.string, shape=[None])
            train_dataset = tf.data.TFRecordDataset(train_filename_placeholder)
            train_dataset = train_dataset.map(parse_tfrecords_function).repeat(epoch_num).batch(train_batch_size).shuffle(buffer_size=train_buffer_size)
            train_dataset_iterator = train_dataset.make_initializable_iterator()
            batch_labels, batch_ids, batch_values = train_dataset_iterator.get_next()

            # Loading testing data from TFRecords
            test_filename_list = self.get_train_files(test_files, num_train_file, num_worker, worker_id)
            test_filename_placeholder = tf.placeholder(tf.string, shape=[None])
            test_dataset = tf.data.TFRecordDataset(test_filename_placeholder)
            test_dataset = test_dataset.map(parse_tfrecords_function).repeat().batch(
                test_batch_size).shuffle(buffer_size=test_buffer_size)
            test_dataset_iterator = test_dataset.make_initializable_iterator()
            test_batch_labels, test_batch_ids, test_batch_values = test_dataset_iterator.get_next()

            # Model definition
            logits = lr_inference(batch_ids, batch_values, 124, 2)
            test_logits = lr_inference(test_batch_ids, test_batch_values, 124, 2)
            # test_logits =
            batch_labels = tf.to_int64(batch_labels)
            test_batch_labels = tf.to_int64(test_batch_labels)

            # self.x = (batch_ids, batch_values)
            # self.y_ = batch_labels
            # self.y = logits

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            with tf.name_scope('loss'):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=batch_labels)
                test_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=test_logits, labels=test_batch_labels)
            self.cross_entropy = tf.reduce_mean(cross_entropy)
            self.test_cross_entropy = tf.reduce_mean(test_cross_entropy)

            with tf.name_scope('optimizer'):
                # self.optimizer = tf.train.AdamOptimizer(learning_rate)
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                self.train_step = self.optimizer.minimize(self.cross_entropy, global_step=self.global_step)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(logits, 1), batch_labels)
                correct_prediction = tf.cast(correct_prediction, tf.float32)
                test_correct_prediction = tf.equal(tf.argmax(test_logits, 1), test_batch_labels)
                test_correct_prediction = tf.cast(test_correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)
            self.test_accuracy = tf.reduce_mean(test_correct_prediction)

            # summaries
            tf.summary.scalar("loss", self.cross_entropy)
            tf.summary.scalar("train_accuracy", self.accuracy)
            tf.summary.scalar("test_accuracy", self.test_accuracy)
            self.summary_op = tf.summary.merge_all()

            # Create session to run
            self.sess = tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1
            ))
            init_op = [tf.global_variables_initializer(),
                       tf.local_variables_initializer()]
            self.sess.run(init_op)
            self.sess.run(
                train_dataset_iterator.initializer,
                feed_dict={train_filename_placeholder: train_filename_list}
            )
            self.sess.run(
                test_dataset_iterator.initializer,
                feed_dict={test_filename_placeholder: test_filename_list}
            )

            # Helper values
            self.variables = ray.experimental.TensorFlowVariables(
                self.cross_entropy, self.sess
            )

            self.grads = self.optimizer.compute_gradients(
                self.cross_entropy
            )

            self.grads_placeholder = [
                (tf.placeholder("float", shape=grad[1].get_shape()), grad[1])
                for grad in self.grads
            ]
            self.apply_grads_placeholder = self.optimizer.apply_gradients(
                self.grads_placeholder
            )

    def test(self):
        loss_val, train_acc, test_acc, summaries = self.sess.run(
             [self.cross_entropy, self.accuracy, self.test_accuracy, self.summary_op])
        return loss_val, train_acc, test_acc, summaries

    def compute_grad_next_batch(self):
        return self.sess.run([grad[0] for grad in self.grads])

    def compute_update_next_batch(self):
        weights = self.get_weights()[1]
        self.sess.run(self.train_step)
        new_weights = self.get_weights()[1]
        return [x - y for x, y in zip(new_weights, weights)]

    def apply_gradients(self, gradients):
        feed_dict = {}
        for i in range(len(self.grads_placeholder)):
            feed_dict[self.grads_placeholder[i][0]] = gradients[i]
        # TODO: check apply_grads_placeholder
        self.sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

    def compute_accuracy(self):
        # TODO: compute_accuracy(x, y)
        return self.sess.run(self.accuracy)

    def compute_accuracy_test(self):
        return self.sess.run(self.test_accuracy)

    def set_weights(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))

    def get_weights(self):
        weights = self.variables.get_weights()
        return list(weights.keys()), list(weights.values())

    def get_train_files(self, train_file_prefix, num_train_file, num_worker, worker_id):
        fids = [i*num_worker+worker_id for i in range(int(np.ceil(num_train_file / num_worker)))]
        if fids[-1] > num_train_file:
            fids.pop()

        train_files = []
        for fid in fids:
            filename = '%s-part-%03d.libsvm.tfrecords' % (train_file_prefix, fid)
            train_files.append(filename)
        return train_files


def lr_inference(sparse_ids, sparse_values, input_units, label_size):
    with tf.variable_scope("logistic_regression", reuse=tf.AUTO_REUSE):
        layer = sparse_full_connect(sparse_ids, sparse_values,
                                [input_units, label_size],
                                [label_size])
    return layer


def sparse_full_connect(sparse_ids,
                        sparse_values,
                        weights_shape,
                        biases_shape):

    weights = tf.get_variable(
        "weights", weights_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable(
        "biases", biases_shape, initializer=tf.random_normal_initializer())
    return tf.nn.embedding_lookup_sparse(
        weights, sparse_ids, sparse_values, combiner="sum") + biases


def nn_pca(sparse_ids, sparse_values, input_units, label_size):
    with tf.variable_scope("nn_pca", reuse=tf.AUTO_REUSE):
        layer = sparse_full_connect(sparse_ids, sparse_values,
                                    [input_units, label_size],
                                    [label_size])
        loss = tf.reduce_mean(tf.square(layer)) * -1.0
    return loss