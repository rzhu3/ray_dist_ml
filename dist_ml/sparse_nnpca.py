# Most of the tensorflow code is adapted from Tensorflow's tutorial on using
# CNNs to train MNIST
# https://www.tensorflow.org/get_started/mnist/pros#build-a-multilayer-convolutional-network.  # noqa: E501

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import datetime
import logging
import os
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.python.ops.init_ops import RandomNormal
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)

# from . import sparse_model
# import sparse_model

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

dataset_configurations = {
    'a8a': {
        'train_files': 'a8a_data/a8a_train',
        'test_files': 'a8a_data/a8a_test',
        'feature_dim': 123,
        'label_size': 2
    },
    'a9a': {
        'train_files': 'a9a_data/a9a_train',
        'test_files': 'a9a_data/a9a_test',
        'feature_dim': 123,
        'label_size': 2
    },
    'rcv1': {
        'train_files': 'rcv1_data/rcv1_train',
        'test_files': 'rcv1_data/rcv1_test',
        'feature_dim': 47236,
        'label_size': 2
    },
    'covtype': {
        'train_files': 'covtype_data/covtype_train',
        'test_files': 'covtype_data/covtype_train',
        'feature_dim': 54,
        'label_size': 7
    },
    'mnist': {
        'train_files': 'mnist_data/mnist_train',
        'test_files': 'mnist_data/mnist_test',
        'feature_dim': 780,
        'label_size': 10
    },
    'aloi': {
        'train_files': 'aloi_data/aloi_train',
        'test_files': 'aloi_data/aloi_test',
        'feature_dim': 128,
        'label_size': 1000
    }
}

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


class NNL2BallInitializer(RandomNormal):
    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        tmp = random_ops.random_normal(
            shape, self.mean, self.stddev, dtype, seed=self.seed)
        tmp = tf.nn.l2_normalize(tf.nn.relu(tmp))
        return tmp


class SparsePCA(object):
    # init func: hyperparameters
    def __init__(self, dataset, learning_rate=1e-3, epoch_num=100,
                 train_batch_size=256, num_train_file=4, num_worker=4, worker_id=1,
                 test_batch_size=256, num_test_file=1):

        train_buffer_size = train_batch_size * 3
        test_buffer_size = test_batch_size * 3
        dataset_config = dataset_configurations[dataset]
        train_files = os.path.join('dataset', dataset_config['train_files'])
        test_files = os.path.join('dataset', dataset_config['test_files'])
        self.dataset = dataset

        dataset_path = os.path.join('dataset', '%s_data' % dataset)

        with tf.Graph().as_default():
            # Loading training data from TFRecords
            # train_filename_list = self.get_train_files(train_files, num_test_file, num_worker, worker_id)
            train_filename_list = self.get_train_files(dataset_path)
            train_filename_placeholder = tf.placeholder(tf.string, shape=[None])
            train_dataset = tf.data.TFRecordDataset(train_filename_placeholder)
            train_dataset = train_dataset.map(parse_tfrecords_function).shuffle(buffer_size=train_buffer_size).repeat(epoch_num).batch(train_batch_size)
            train_dataset_iterator = train_dataset.make_initializable_iterator()
            _, batch_ids, batch_values = train_dataset_iterator.get_next()

            # Loading testing data from TFRecords
            # test_filename_list = self.get_train_files(test_files, num_test_file, num_worker, worker_id)
            # test_filename_list = ['%s.libsvm.tfrecords' % train_files]
            test_filename_list = self.get_train_files(dataset_path)
            test_filename_placeholder = tf.placeholder(tf.string, shape=[None])
            test_dataset = tf.data.TFRecordDataset(test_filename_placeholder)
            # test_dataset = test_dataset.map(parse_tfrecords_function).repeat().batch(
                # test_batch_size).shuffle(buffer_size=test_buffer_size)
            test_dataset = test_dataset.map(parse_tfrecords_function).repeat().batch(
                test_batch_size)
            test_dataset_iterator = test_dataset.make_initializable_iterator()
            _, test_batch_ids, test_batch_values = test_dataset_iterator.get_next()

            # Model definition
            feature_dim = dataset_config['feature_dim'] + 1
            label_size = 1 # dataset_config['label_size']
            logits = inference(batch_ids, batch_values, feature_dim, label_size)
            test_logits = inference(test_batch_ids, test_batch_values, feature_dim, label_size)

            # self.x = (batch_ids, batch_values)
            # self.y_ = batch_labels
            # self.y = logits

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            with tf.name_scope('loss'):
                square = tf.negative(tf.square(logits))
                test_square = tf.negative(tf.square(test_logits))
            self.objective = tf.reduce_mean(square)
            self.test_objective = tf.reduce_mean(test_square)
            # self.objective = tf.add(tf.reduce_mean(square), 0.22655645184748496 * 2.0)
            # self.test_objective = tf.add(tf.reduce_mean(test_square), 0.22655645184748496*2.0)

            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                self.train_step = self.optimizer.minimize(self.objective, global_step=self.global_step)

            # summaries
            tf.summary.scalar("objective%d" % worker_id, self.objective)
            tf.summary.scalar("test_objective%d" % worker_id, self.test_objective)
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
                self.objective, self.sess
            )

            self.grads = self.optimizer.compute_gradients(
                self.objective
            )

            self.grads_placeholder = [
                (tf.placeholder("float", shape=grad[1].get_shape()), grad[1])
                for grad in self.grads
            ]
            self.apply_grads_placeholder = self.optimizer.apply_gradients(
                self.grads_placeholder
            )

    def test(self):
        loss_val, summaries, global_step = self.sess.run(
             [self.test_objective, self.summary_op, self.global_step])
        return loss_val, summaries, global_step

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

    def set_weights(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))

    def get_weights(self):
        weights = self.variables.get_weights()
        return list(weights.keys()), list(weights.values())

    def get_train_files(self, dataset_path):
        dataset_files = os.listdir(dataset_path)
        train_files = []
        for filename in dataset_files:
            if ('train' in filename) and ('tfrecords' in filename):
                if self.dataset == 'rcv1': continue
                filepath = os.path.join(dataset_path, filename)
                train_files.append(filepath)
            if 'rcv1_test.libsvm.tfrecords' in filename:
                filepath = os.path.join(dataset_path, filename)
                train_files.append(filepath)
        return train_files

    def get_individual_train_files(self, train_file_prefix, num_train_file, num_worker, worker_id):
        fids = [i*num_worker+worker_id for i in range(int(np.ceil(num_train_file / num_worker)))]
        if fids[-1] > num_train_file:
            fids.pop()

        train_files = []
        for fid in fids:
            filename = '%s-part-%03d.libsvm.tfrecords' % (train_file_prefix, fid)
            train_files.append(filename)
        return train_files


def inference(sparse_ids, sparse_values, input_units, label_size):
    with tf.variable_scope("sparse_pca", reuse=tf.AUTO_REUSE):
        layer = sparse_full_connect(sparse_ids, sparse_values,
                                [input_units, label_size],
                                [label_size])
    return layer


def sparse_full_connect(sparse_ids,
                        sparse_values,
                        weights_shape,
                        biases_shape):

    weights = tf.get_variable(
        "weights", weights_shape,
        initializer=NNL2BallInitializer())
        # "weights", weights_shape, initializer=tf.random_normal_initializer())
    # weights = tf.nn.relu(weights)
    # weights = tf.nn.l2_normalize(weights)
    # biases = tf.get_variable(
    #     "biases", biases_shape, initializer=tf.random_normal_initializer())
    return tf.nn.embedding_lookup_sparse(
        weights, sparse_ids, sparse_values, combiner="sum") # + biases