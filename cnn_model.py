import tensorflow as tf


def fc_layer(input_layer, output_size):
    return tf.layers.dense(inputs=input_layer, units=output_size, activation=tf.nn.relu)


def conv_layer(input_layer, output_ch, strides=1, kernel_size=5):
    return tf.layers.conv2d(
        inputs=input_layer,
        filters=output_ch,
        kernel_size=[kernel_size, kernel_size],
        strides=(strides, strides),
        padding="same",
        activation=tf.nn.relu)


def dropout(input_layer, mode, rate=0.4):
    return tf.layers.dropout(inputs=input_layer, rate=rate, training=mode == tf.estimator.ModeKeys.TRAIN)


def max_pool_2x2(input_layer, pool_size=2):
    return tf.layers.max_pooling2d(inputs=input_layer, pool_size=[pool_size, pool_size], strides=pool_size)


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    conv1 = conv_layer(input_layer, 32, 2)  # 14
    conv2 = conv_layer(conv1, 32, 1)  # 14
    conv3 = conv_layer(conv2, 64, 2)  # 7
    conv4 = conv_layer(conv3, 64, 1)  # 7
    flat_size = 7 * 7 * 64
    flat = tf.reshape(conv4, [-1, flat_size])
    fc = fc_layer(flat, 1024)
    droped = dropout(fc, mode)
    logits = fc_layer(droped, 10)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)