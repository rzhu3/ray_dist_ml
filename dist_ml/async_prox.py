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

flags = tf.app.flags
# The following arguments are cluster related ones.
flags.DEFINE_integer("num_workers", 4,
                     "The number of workers to use.")
flags.DEFINE_string("redis_address", "",
                    "The Redis address of the cluster.")
# The following arguments are dataset configurations
flags.DEFINE_string("dataset", 'a8a',
                    "The experiment dataset for current execution.")
# Hyperparameter configuration
flags.DEFINE_float("learning_rate", 1e-4,
                   "Learning rate for proximal gradient.")
flags.DEFINE_integer("batch_size", 256,
                     "Batch size for training.")
flags.DEFINE_integer("test_batch_size", 256,
                     "Batch size for testing.")

# parser.add_argument("--batch-size", default=256, type=int,
#                     help="Batch size for training")
# parser.add_argument("--test-batch-size", default=256, type=int,
#                     help="Batch size for testing")

# parser = argparse.ArgumentParser(description="Run the asynchronous parameter "
#                                              "server example.")
# # The following arguments are cluster related ones.
# parser.add_argument("--num-workers", default=4, type=int,
#                     help="The number of workers to use.")
# parser.add_argument("--redis-address", default=None, type=str,
#                     help="The Redis address of the cluster.")
# # The following arguments are dataset configurations
# parser.add_argument("--dataset", default='a8a', type=str,
#                     help="The experiment dataset for current execuetion.")
# # Hyperparameter configuration
# parser.add_argument("--learning-rate", default=1e-4, type=float,
#                     help="Learning rate for proximal gradient")
# parser.add_argument("--batch-size", default=256, type=int,
#                     help="Batch size for training")
# parser.add_argument("--test-batch-size", default=256, type=int,
#                     help="Batch size for testing")


@ray.remote
class ParameterServer(object):
    def __init__(self, learning_rate, keys, values):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.
        values = [value.copy() for value in values]
        self.weights = dict(zip(keys, values))
        self.step = 0
        self.learning_rate = learning_rate

    def update(self, key, new_value, learning_rate):
        # local_weight = self.weights[key]
        if isinstance(new_value, np.ndarray):
            self.weights[key] -= learning_rate*new_value
        elif isinstance(new_value, IndexedSlicesValue):
            new_indices = new_value.indices
            new_values = new_value.values
            self.weights[key][new_indices] -= learning_rate*new_values

    def push(self, keys, values):
        learning_rate = self.learning_rate # / np.sqrt(self.step+1)
        for key, value in zip(keys, values):
            self.update(key, value, learning_rate)
        self.step += 1

    def pull(self, keys):
        return [self.weights[key] for key in keys]

    def pull_steps(self, keys):
        return [self.weights[key] for key in keys], self.step


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


if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(redis_address=args.redis_address)

    # Create a parameter server with some random weights.
    # net = model.SimpleCNN()
    train_files = 'a8a_data/a8a_train' #  ['a8a_data/a8a_train-part-%03d' % i for i in range(1, 5)]
    test_files = 'a8a_data/a8a_test' # ['a8a_data/a8a_test-part-001']
    net = SparseClassifier(train_files=train_files, test_files=test_files)
    all_keys, all_values = net.get_weights()
    ps = ParameterServer.remote(1e-3/2, all_keys, all_values)

    # Start some training tasks.
    worker_tasks = [worker_task.remote(ps, i+1) for i in range(2)]

    i = 0
    # while True:
    for iteration in range(10):
        # Get and evaluate the current model.
        # current_weights = ray.get(ps.pull.remote(all_keys))
        current_weights, current_step = ray.get(ps.pull_steps.remote(all_keys))
        net.set_weights(all_keys, current_weights)
        accuracy = net.compute_accuracy_test()
        print("Iteration {}: accuracy is {}, global_step is {}".format(i, accuracy, current_step))
        i += 1
        time.sleep(1)
