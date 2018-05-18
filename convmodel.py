# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def custom_one_hot(x):
    """
    :param x: label (int)
    :return: one hot code
    """
    if x == 0:
        return [1., 0., 0.]
    if x == 1:
        return [0., 1., 0.]
    return [0., 0., 1.]


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3
batch_size = 5


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), num_classes)
        # image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [128, 128, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)#126,126,32
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu) #63,63,16
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)#61,61,64
        # o5 = tf.layers.conv2d(inputs=o4, filters=32, kernel_size=3, activation=tf.nn.relu)#30,30,32
        # o6 = tf.layers.max_pooling2d(inputs=o5, pool_size=2, strides=2)#28,28,32
        # 14,14,16

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 30 * 30 * 64]), units=15, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(
    ["data3/0_train/*.jpg", "data3/1_train/*.jpg", "data3/2_train/*.jpg"]
    , batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["data3/0_validation/*.jpg", "data3/1_validation/*.jpg",
                                                     "data3/2_validation/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["data3/0_test/*.jpg", "data3/1_test/*.jpg",
                                                   "data3/2_test/*.jpg"], batch_size=batch_size)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_valid, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, dtype=tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_train, dtype=tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    training_error_values = []
    validation_error_error_values = []
    validation_error=100
    i = 0
    while True:
        sess.run(optimizer)
        if i % 20 == 0:
            print("Iter:", i, "---------------------------------------------")
            print(sess.run(label_batch_valid))
            print(sess.run(example_batch_valid_predicted))
            validation_error = sess.run(cost_valid)
            print("Validation error:", validation_error)
            validation_error_error_values.append(validation_error)
            training_error = sess.run(cost)
            print("Training error:", training_error)
            training_error_values.append(training_error)
        if validation_error < 0.001:
            break
        i = i + 1

    save_path = saver.save(sess, "./tmp/model15.ckpt")
    print("Model saved in file: %s" % save_path)

    failCounter = 0
    for i in range(3):
        output = sess.run(example_batch_valid_predicted)
        label = sess.run(label_batch_test)
        for o, l in zip(output, label):
            if round(o[0]) != l[0] or round(o[1]) != l[1] or round(o[2]) != l[2]:
                failCounter += 1

    failPercentage = 100*failCounter/15
    print("Fail percentage: " + str(failPercentage))
    # print("length: " + str(len(label)))
    # print("output: " + str(len(output)))

    coord.request_stop()
    coord.join(threads)

    plt.plot(validation_error_error_values)
    plt.plot(training_error_values)
    plt.show()
