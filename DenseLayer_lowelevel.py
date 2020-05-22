from builtins import float

import tensorflow as tf
import os
import logging

# setting logging error configs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Problem statement ---
# Reference picture  - <<<../one_uni_hidden_layer.png>>>
# Let us say that we have been given one data-point and three features
# HOW to build manually without tf.keras utility this single unit layer ?


features = tf.constant([[2.0, 0.3, 0.45]], dtype=float)
print(features.shape)
# 3 FEATURES
# 1 HIDDEN layer unit so 3x1 weight matrix
weights = tf.Variable(tf.ones((3, 1)))

bias = tf.Variable(1.0)
#  --- DENSE LAYER Building ----- 1 Unit
# matrix multiply of features with weight
prod = tf.matmul(features, weights)

# Dense Hidden layer
dense1 = tf.keras.activations.sigmoid(prod+bias)

print(dense1.numpy())
