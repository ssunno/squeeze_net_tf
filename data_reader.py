# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random


def load_cifar10():
    return tf.keras.datasets.cifar10.load_data()


class BatchSet:

    def __init__(self, batch_size, load_target=None):
        load_data = load_target
        self.batch_size = batch_size
        self.data_train, self.data_test = load_data()
        self.image_shape = np.shape(self.data_train[0])
        self.num_classes = max(self.data_train[1]) + 1

    def batches(self):
        data_size = len(self.data_train[0])
        rand_idx = [i for i in range(data_size)]
        random.shuffle(rand_idx)
        for i in range(0, data_size, self.batch_size):
            yield [self.data_train[0][idx] for idx in rand_idx[i:i + self.batch_size]],\
                  [self.data_train[1][idx] for idx in rand_idx[i:i + self.batch_size]]

    def test_batches(self):
        data_size = len(self.data_test[0])
        rand_idx = [i for i in range(data_size)]
        random.shuffle(rand_idx)
        for i in range(0, data_size, self.batch_size):
            yield [self.data_test[0][idx] for idx in rand_idx[i:i + self.batch_size]], \
                  [self.data_test[1][idx] for idx in rand_idx[i:i + self.batch_size]]
