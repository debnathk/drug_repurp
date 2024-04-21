# Load datasets
import tensorflow as tf
import numpy as np

root = '/lustre/home/debnathk/dleps/code/DLEPS/reference_drug/'

# Create dummy data
num_samples_train = 100
num_samples_test = 20

shape = (10, 10, 10)

data_train = np.random.randint(low=0, high=100, size=(num_samples_train, *shape), dtype=np.int64)
data_test = np.random.randint(low=0, high=100, size=(num_samples_test, *shape), dtype=np.int64)

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
with tf.io.TFRecordWriter(root + 'data_train.tfrecords') as writer:
    for i in range(num_samples_train):
        example = serialize_example(data_train[i])
        writer.write(example)

# Write test data to tfrecord file
with tf.io.TFRecordWriter(root + 'data_test.tfrecords') as writer:
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
dataset_train = tf.data.TFRecordDataset(root + 'data_train.tfrecords')
dataset_train = dataset_train.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset_test = tf.data.TFRecordDataset(root + 'data_test.tfrecords')
dataset_test = dataset_test.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)