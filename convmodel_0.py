# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

filenames0 = tf.train.match_filenames_once("data3/0_train/*.jpg")
filenames1 = tf.train.match_filenames_once("data3/1_train/*.jpg")
filenames2 = tf.train.match_filenames_once("data3/2_train/*.jpg")


filename_queue0 = tf.train.string_input_producer(filenames0, shuffle=False)
filename_queue1 = tf.train.string_input_producer(filenames1, shuffle=False)
filename_queue2 = tf.train.string_input_producer(filenames2, shuffle=False)


reader0 = tf.WholeFileReader()
reader1 = tf.WholeFileReader()
reader2 = tf.WholeFileReader()

key0, file_image0 = reader0.read(filename_queue0)
key1, file_image1 = reader1.read(filename_queue1)
key2, file_image2 = reader2.read(filename_queue2)

image0, label0 = tf.image.decode_jpeg(file_image0), [0.]  # key0
image0 = tf.reshape(image0, [80, 140, 1])

image1, label1 = tf.image.decode_jpeg(file_image1), [1.]  # key1
image1 = tf.reshape(image1, [80, 140, 1])

image2, label2 = tf.image.decode_jpeg(file_image2), [2.]  # key2
image2 = tf.reshape(image2, [80, 140, 1])

image0 = tf.to_float(image0) / 256. - 0.5
image1 = tf.to_float(image1) / 256. - 0.5
image2 = tf.to_float(image2) / 256. - 0.5

batch_size = 4
min_after_dequeue = 10  # 10000
capacity = min_after_dequeue + 3 * batch_size

example_batch0, label_batch0 = tf.train.shuffle_batch([image0, label0], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

example_batch1, label_batch1 = tf.train.shuffle_batch([image1, label1], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

example_batch2, label_batch2 = tf.train.shuffle_batch([image2, label2], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)


example_batch = tf.concat(values=[example_batch0, example_batch1, example_batch2], axis=0)
label_batch = tf.concat(values=[label_batch0, label_batch1, label2], axis=0)

# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

o1 = tf.layers.conv2d(inputs=example_batch, filters=32, kernel_size=3, activation=tf.nn.relu)
o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 2, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
y = tf.layers.dense(inputs=h, units=1, activation=tf.nn.sigmoid)

cost = tf.reduce_sum(tf.square(y - label_batch))
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
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())


    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for _ in range(200):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(y))
            print(sess.run(label_batch))
            print("Error:", sess.run(cost))

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
            
    coord.request_stop()
    coord.join(threads)
