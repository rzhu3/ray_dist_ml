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

from dist_ml.sparse_nnpca import SparsePCA
from dist_ml.util import *


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
flags.DEFINE_boolean("enable_lr_decay", False,
                     "Enable learning rate decay.")
flags.DEFINE_float("decay_rate", 0.96,
                   "LR decay rate.")
flags.DEFINE_float("decay_step", 500,
                   "LR decay step.")
flags.DEFINE_integer("epochs", 10000,
                     "Number of epochs for training.")
flags.DEFINE_integer("iterations", 300,
                     "Number of iterations for testing.")

#
# parser = argparse.ArgumentParser(description="Run the asynchronous parameter "
#                                              "server example.")
# # The following arguments are cluster related ones.
# parser.add_argument("--num-workers", default=1, type=int,
#                     help="The number of workers to use.")
# parser.add_argument("--redis-address", default=None, type=str,
#                     help="The Redis address of the cluster.")
# # The following arguments are dataset configurations
# parser.add_argument("--dataset", default='a8a', type=str,
#                     help="The experiment dataset for current execuetion.")
# # Hyperparameter configuration
# parser.add_argument("--learning-rate", default=1e-3, type=float,
#                     help="Learning rate for proximal gradient")
# parser.add_argument("--enable-lr-decay", action="store_true",
#                     help="Enable learning rate decay")
# parser.add_argument("--decay-rate", default=0.96, type=float,
#                     help="Learning rate decay rate")
# parser.add_argument("--decay-step", default=500, type=int,
#                     help="Learning rate decay rate steps")
# parser.add_argument("--batch-size", default=1024, type=int,
#                     help="Batch size for training")
# parser.add_argument("--test-batch-size", default=256, type=int,
#                     help="Batch size for testing")
# parser.add_argument("--epochs", default=10000, type=int,
#                     help="Number of epochs for training")
# parser.add_argument("--iterations", default=300, type=int,
#                     help="Number of iterations for testing")

@ray.remote
class ParameterServer(object):
    def __init__(self, args, keys, values):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.
        values = [value.copy() for value in values]
        states = [np.zeros(value.shape) for value in values]
        self.weights = dict(zip(keys, values))
        self.states = dict(zip(keys, states))
        self.step = 0
        self.learning_rate = args.learning_rate
        self.enable_lr_decay = args.enable_lr_decay
        self.num_workers = args.num_workers
        if args.enable_lr_decay:
            self.decay_rate = args.decay_rate
            self.decay_step = args.decay_step

    def update(self, key, new_value, learning_rate):
        # local_weight = self.weights[key]
        if isinstance(new_value, np.ndarray):
            # self.weights[key] -= learning_rate*new_value
            self.states[key] += new_value
        elif isinstance(new_value, IndexedSlicesValue):
            new_indices = new_value.indices
            new_values = new_value.values
            self.states[key][new_indices] += new_values

        if self.step % self.num_workers == self.num_workers-1:
            new_value = self.states[key]
            self.states[key] = np.zeros(self.states[key].shape)
            self.weights[key] = proxgrad_posl2ball(self.weights[key], new_value, learning_rate)
        else:
            time.sleep(0.2)

    def update_old(self, key, new_value, learning_rate):
        # local_weight = self.weights[key]
        if isinstance(new_value, np.ndarray):
            self.weights[key] = proxgrad_posl2ball(self.weights[key], new_value, learning_rate)

        elif isinstance(new_value, IndexedSlicesValue):
            new_indices = new_value.indices
            new_values = new_value.values
            self.states[key][new_indices] = proxgrad_posl2ball(self.states[key][new_indices], new_values, learning_rate)

    def learning_rate_scheduler(self):
        # return self.learning_rate / np.sqrt(1.0 + self.step // 1000)
        if self.enable_lr_decay:
            # return self.learning_rate * (self.decay_rate ** (self.step // self.decay_step))
            return self.learning_rate / np.sqrt(1.0 + self.decay_rate * self.step // self.decay_step)
        else:
            return self.learning_rate

    def push(self, keys, values):
        learning_rate = self.learning_rate_scheduler()
        for key, value in zip(keys, values):
            self.update(key, value, learning_rate)
            # self.update_old(key, value, learning_rate)
        self.step += 1

    def pull(self, keys):
        return [self.weights[key] for key in keys]

    def pull_steps(self, keys):
        return [self.weights[key] for key in keys], self.step


@ray.remote
def worker_task(ps, worker_index, args):
    # Configure dataset
    # Initialize the model.
    net = SparsePCA(dataset=args.dataset,
                    worker_id=worker_index,
                    epoch_num=args.epochs,
                    num_worker=args.num_workers,
                    train_batch_size=args.batch_size,
                    test_batch_size=args.test_batch_size)
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


def ckpt_filename(dataset, num_worker, timestamp):
    import os
    foldername = os.path.join('ckpt', dataset+'_data')
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    filename = 'ckpt_worker%d_%d.pl' % (num_worker, timestamp)
    return os.path.join(foldername, filename)


if __name__ == "__main__":
    # args = parser.parse_args()
    
    ray.init(redis_address=args.redis_address)

    # Create a parameter server with some random weights.
    # net = model.SimpleCNN()

    net = SparsePCA(dataset=args.dataset,
                    num_worker=args.num_workers,
                    epoch_num=args.epochs,
                    train_batch_size=args.batch_size,
                    test_batch_size=args.test_batch_size)
    all_keys, all_values = net.get_weights()
    ps = ParameterServer.remote(args, all_keys, all_values)

    # Start some training tasks.
    worker_tasks = [worker_task.remote(ps, i+1, args) for i in range(args.num_workers)]

    writer = tf.summary.FileWriter('summary', net.sess.graph)

    start_time = datetime.datetime.now()
    i = 0
    # while True:
    results = {'time': [], 'step': [], 'weight': []}

    for iteration in range(args.iterations):
        # Get and evaluate the current model.
        # current_weights = ray.get(ps.pull.remote(all_keys))
        current_weights, current_step = ray.get(ps.pull_steps.remote(all_keys))
        end_time = datetime.datetime.now()
        time_elapsed = end_time-start_time
        results['time'].append(time_elapsed.total_seconds())
        results['step'].append(current_step)
        results['weight'].append(current_weights[0])
        net.set_weights(all_keys, current_weights)
        test_obj, summary_value, global_step = net.test()
        # test_obj -= -0.22655645184748496
        print("Iteration {}: test_obj is {}, global_step is {}".format(i, test_obj, current_step))

        writer.add_summary(summary_value, current_step)
        i += 1
        time.sleep(1)

    import pickle
    timestamp = datetime.datetime.now().timestamp()
    ckpt_file = ckpt_filename(args.dataset, args.num_workers, timestamp)
    print('Saving weights on %s during training...' % ckpt_file)
    with open(ckpt_file, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saving finished.')