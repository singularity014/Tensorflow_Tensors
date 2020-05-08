'''
Author: Prafull SHARMA
'''
# Low Level approach with multiple examples
# In this code, we'll create Neural Net Reference : 'two_unit_hidden_layer_NN.png'
# Neural Network, we low level coding in TF2.0

import tensorflow as tf
import numpy as np

# -------------------- 1.) FEATURE MATRIX SHAPE --------------------- #
borrower_features = np.array([#F1, F2, F3
                              [3.0, 3.0, 23.0],
                              [2.0, 1.0, 24.0],
                              [1.0, 1.0, 49.0],
                              [1.0, 1.0, 49.0],
                              [2.0, 1.0, 29.0]
                             ]
                             )
borrower_features = tf.constant(borrower_features, dtype=float)
# where there are three features F1, F2, F3 , there are 5 data-points (or examples for each feature)

# -------------------- 2.) WEIGHT MATRIX SHAPE --------------------- #
# Now the shape of weight depends on the architecture of Neural Nets
# its a matrix, with  A x B dimension
# A - dimension is the number of features (3 in this case)
# B - dimension is the number of units in the hidden layer
# Lets say we want to have two units in the hidden dense layer(dense 1)
# ------------------------------------------------------------------- #

weights_np = np.array([[-0.6 ,  0.6 ],
                        [ 0.8 , -0.3 ],
                        [-0.09, -0.08]])
weights1 = tf.Variable(weights_np, dtype=float)
bias1 = tf.constant(3.0)

# -------------------------  HIDDEN LAYER (2 Units) -------------------------- #
# Compute the product of borrower_features and weights1
products1 = tf.matmul(borrower_features, weights1)

# Apply a sigmoid activation function to products1 + bias1
dense1 = tf.keras.activations.sigmoid(products1 + bias1)

# Print the shapes of borrower_features, weights1, bias1, and dense1
print('\n shape of borrower_features: ', borrower_features.shape)
print('\n shape of weights1: ', weights1.shape)
print('\n shape of bias1: ', bias1.shape)
print('\n shape of dense1: ', dense1.shape)

# Now interesting thing happens in a single unit...
# Each single unit will be be 5 x 1 array
# so shape of dense layer will be 5 x 2

