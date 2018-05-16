import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print(tf.__version__)

tensorboard_dir = "tensorboard/"


def fc_layer(input, input_size, output_size):
    with tf.name_scope('fully_connected') as scope:
        W_fc1 = weight_variable([input_size, output_size])
        b_fc1 = bias_variable([output_size])
        return tf.nn.relu(tf.matmul(input, W_fc1) + b_fc1)


def conv_layer(input, kernel_size, input_ch, output_ch, stride=1):
    with tf.name_scope('convolutional') as scope:
        W_conv1 = weight_variable(
            [kernel_size, kernel_size, input_ch, output_ch])  # with, height, channel, output_channels
        b_conv1 = bias_variable([output_ch])
        return tf.nn.relu(conv2d(input, W_conv1, stride) + b_conv1)


def dw_conv_layer(input, kernel_size, input_ch, output_ch, stride=1):
    W_conv1 = weight_variable([kernel_size, kernel_size, input_ch, 1])  # with, height, channel, output_channels
    W_conv2 = weight_variable([1, 1, input_ch, output_ch])  # with, height, channel, output_channels
    b_conv2 = bias_variable([output_ch])
    return tf.nn.relu(tf.nn.separable_conv2d(input, W_conv1, W_conv2, [1, stride, stride, 1], 'SAME') + b_conv2)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    conv1 = conv_layer(x, 3, 1, 32, 2)  # 14
    conv2 = conv_layer(conv1, 3, 32, 32, 1)  # 14
    conv3 = conv_layer(conv2, 3, 32, 64, 2)  # 7
    conv4 = conv_layer(conv3, 3, 64, 64, 1)  # 7
    flat_size = 7 * 7 * 64
    flat = tf.reshape(conv4, [-1, flat_size])
    fc = fc_layer(flat, flat_size, 1024)
    droped = tf.nn.dropout(fc, keep_prob)
    y = fc_layer(droped, 1024, 10)
    with tf.name_scope('loss_calculation') as scope:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope('check_accuracy') as scope:
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    writer = tf.summary.FileWriter(tensorboard_dir)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch = mnist.train.next_batch(1000)
            images = batch[0].reshape((-1, 28, 28, 1))
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: images, y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: images, y_: batch[1], keep_prob: 0.5})
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images.reshape((-1, 28, 28, 1)), y_: mnist.test.labels, keep_prob: 1.0}))

    writer.add_graph(sess.graph)
    writer.close()


if __name__ == "__main__":
    main()
