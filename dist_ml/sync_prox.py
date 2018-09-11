from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import datetime

import ray
import tensorflow as tf
from tensorflow.python.framework.ops import IndexedSlicesValue
import numpy as np

from dist_ml.sparse_nnpca import SparseClassifier

@ray.remote
class ParameterServer(object):
    def __init__(self, learning_rate):
        self.net = SparseClassifier(learning_rate=learning_rate)

    def _average_gradients(self, *gradients):
        return gradients

    def apply_gradient(self, *gradients):
        self.net.apply_gradients(self._average_gradients(gradients))
        return self.net.variables.get_flat()

    def get_weights(self):
        return self.net.variables.get_flat()

@ray.remote
class Worker(object):
    def __init__(self, worker_index, train_files, test_files):
        self.worker_index = worker_index
        self.net = SparseClassifier(worker_id=worker_index)

    def compute_gradients(self, weights):
        self.net.variables.set_flat(weights)
        return self.net.compute_grad_next_batch()

@ray.remote
def worker_task(ps, worker_index):
    # Configure dataset
    train_files = 'a8a_data/a8a_train'
    test_files = 'a8a_data/a8a_test'
    # Initialize the model.
    net = SparseClassifier(train_files=train_files, test_files=test_files,
                           worker_id=worker_index)
    keys = net.get_weights()[0]

    # while True:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=net.sess)

    try:
        while not coord.should_stop():
            weights = ray.get(ps.pull.remote(keys))
            net.set_weights(keys, weights)

            # compute an update and push it to the parameter server.
            # gradients = net.compute_update_next_batch()
            gradients = net.compute_grad_next_batch()
            ps.push.remote(keys, gradients)

    except tf.errors.OutOfRangeError:
        print("Training Finished.")
    finally:
        coord.request_stop()
    coord.join(threads)

    # for ii in range(10):
    #     # Get the current weights from the parameter server.
    #     weights = ray.get(ps.pull.remote(keys))
    #     net.set_weights(keys, weights)

        # Compute an update and push it to the parameter server.
        # xs, ys = mnist.train.next_batch(batch_size)
        # gradients = net.compute_update(xs, ys)
        # ps.push.remote(keys, gradients)


if __name__ == "__main__":

    ray.init(redis_address="localhost:6379")

    # Create a parameter server with some random weights.
    # net = model.SimpleCNN()
    train_files = 'a8a_data/a8a_train' #  ['a8a_data/a8a_train-part-%03d' % i for i in range(1, 5)]
    test_files = 'a8a_data/a8a_test' # ['a8a_data/a8a_test-part-001']
    net = SparseClassifier(train_files=train_files, test_files=test_files)
    all_keys, all_values = net.get_weights()
    ps = ParameterServer.remote(all_keys, all_values)

    # Start some training tasks.
    worker_tasks = [worker_task.remote(ps, i+1) for i in range(2)]

    i = 0
    # while True:
    for iteration in range(10):
        # Get and evaluate the current model.
        current_weights = ray.get(ps.pull.remote(all_keys))
        net.set_weights(all_keys, current_weights)
        accuracy = net.compute_accuracy_test()
        print("Iteration {}: accuracy is {}".format(i, accuracy))
        i += 1
        time.sleep(1)
