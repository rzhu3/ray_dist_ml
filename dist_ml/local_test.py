from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import datetime
import logging

import tensorflow as tf
import numpy as np

from dist_ml.sparse_nnpca import SparsePCA
# from dist_ml.sparse_classifier import SparseClassifier


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    train_files = 'a8a_data/a8a_train'
    test_files = 'a8a_data/a8a_test'

    tf.reset_default_graph()
    # net = SparseClassifier(train_files=train_files, test_files=test_files)
    net = SparsePCA(dataset='a8a')
    weights = net.get_weights()

    # while True:
    writer = tf.summary.FileWriter('summary', net.sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=net.sess)
    start_time = datetime.datetime.now()

    try:
        while not coord.should_stop():
            # compute an update and push it to the parameter server.
            # gradients = net.compute_grad_next_batch()
            _, step, logits = net.sess.run([net.train_step, net.global_step, net.logits])
            # grad = net.compute_grad_next_batch()

            if step % 100 == 0:
                # loss_value, train_acc_value, test_acc_value, summary_value = net.test()
                loss_value, summary_value, global_steps = net.test()
                weights = net.get_weights()
                np.save('%s_data/ckpt/local_weights_%03d.npy' % ('a8a', global_steps),
                        weights[1][0])
                end_time = datetime.datetime.now()
                # logging.info(
                #     "[{}] Step: {}, loss: {}, train_acc: {}, valid_acc: {}".
                #         format(end_time - start_time, step, loss_value,
                #                train_acc_value, test_acc_value))
                logging.info(
                    "[{}] Step: {}, loss: {}".
                        format(end_time - start_time, step, loss_value))

                writer.add_summary(summary_value, step)
                # saver.save(sess, checkpoint_file_path, global_step=step)
                start_time = end_time
            tf.get_variable_scope().reuse_variables()

    except tf.errors.OutOfRangeError:
        print("Training Finished.")
    finally:
        coord.request_stop()
    coord.join(threads)
