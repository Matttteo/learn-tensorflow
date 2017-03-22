#!/usr/bin/env python
# coding=utf-8

"""A very simple MNIST classifier change from tf's tutorial

Use the uncompressed mnist data as input

"""
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.contrib.learn.python.learn.datasets import base
#from mnist import DataSet
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
import numpy
def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels)* num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_images(filename):
    with open(filename, 'rb') as f:
        magic = _read32(f)
        if magic != 2051:
             raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                            (magic, filename))
        num_images = _read32(f)
        rows = _read32(f)
        cols = _read32(f)
        buf = f.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

def extract_labels(filename, one_hot=False, num_classes=10):
    with open(filename, 'rb') as f:
        magic = _read32(f)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST image file %s' %
                            (magic, filename))
        num_items =_read32(f)
        buf = f.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            labels =dense_to_one_hot(labels, num_classes)
        return labels

def read_data_sets(train_dir,
                   one_hot=False,
                   dtype=dtypes.float32):

    TRAIN_IMAGES = 'train-images-idx3-ubyte'
    TRAIN_LABELS = 'train-labels-idx1-ubyte'
    TEST_IMAGES = 't10k-images-idx3-ubyte'
    TEST_LABELS = 't10k-labels-idx1-ubyte'

    train_images = extract_images(train_dir + TRAIN_IMAGES)
    train_labels = extract_labels(train_dir + TRAIN_LABELS, one_hot)
    test_images = extract_images(train_dir + TEST_IMAGES)
    test_labels = extract_labels(train_dir + TEST_LABELS, one_hot)

    VALIDATION_SIZE = 50000

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]

    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    train = mnist.DataSet(train_images, train_labels, dtype=dtype)
    validation = mnist.DataSet(validation_images, validation_labels, dtype=dtype)
    test = mnist.DataSet(test_images, test_labels, dtype=dtype)

    return base.Datasets(train=train, validation=validation, test=test)
