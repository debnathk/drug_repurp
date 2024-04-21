# Load datasets
import tensorflow as tf
import numpy as np
import h5py

root = '/lustre/home/debnathk/dleps/code/DLEPS/reference_drug/'

# Load the SSP datasets
num_samples_train = 44988
num_samples_test = 5000

# shape = (207, 3072)

h5f = h5py.File(root + 'ssp_data_train.h5', 'r')
data_train = h5f['data'][:]
h5f2 = h5py.File(root + 'ssp_data_test.h5', 'r')
data_test = h5f['data'][:]

# Function to convert data to tf.Example format
def _int64_feature(value):
    """Returns an int64_list from a int / bool."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))

def serialize_example(feature0):
    feature = {
        'feature0': _int64_feature(feature0),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Write train data to tfrecord file
with tf.io.TFRecordWriter(root + 'data_train_45k.tfrecords') as writer:
    for i in range(num_samples_train):
        example = serialize_example(data_train[i])
        writer.write(example)

# Write test data to tfrecord file
with tf.io.TFRecordWriter(root + 'data_test_5k.tfrecords') as writer:
    for i in range(num_samples_test):
        example = serialize_example(data_test[i])
        writer.write(example)

# Read data from tfrecord file
def _parse_function(example_proto):
    feature_description = {
        'feature0': tf.io.FixedLenFeature([np.prod(shape)], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    feature0 = tf.reshape(example['feature0'], shape)
    return feature0

# Use multiple threads to read and preprocess the data
dataset_train = tf.data.TFRecordDataset(root + 'data_train_45k.tfrecords')
dataset_train = dataset_train.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset_test = tf.data.TFRecordDataset(root + 'data_test_5k.tfrecords')
dataset_test = dataset_test.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)