import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from cnn_model import cnn_model_fn

model_dir = "mnist_convnet_model/"
mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir)

img = mpimg.imread("9.png")
img = np.mean(img, 2)
img = img.reshape((1, -1))

predictions = mnist_classifier.predict(input_fn=lambda: {"x": img})
template = '\nPrediction is "{}" ({:.1f}%)'

for pred_dict in predictions:
    class_id = pred_dict['classes']
    probability = pred_dict['probabilities'][class_id]

    print(template.format(class_id,
                          100 * probability))
    break
