#%%TensorFlow Basics
#This is very low level implementations in TF 1.0, in other sections codes are written in TF 2.0
#This gives Python access to all of TensorFlow's classes, methods, and symbols Getting Started With TensorFlow
import tensorflow as tf
import numpy as np

#tf.enable_eager_execution() #Eager execution enables a more interactive front end

# Some constants and 
# casting in TF compatible formats
node1 = tf.constant(1.0, dtype=tf.float32)
node2 = tf.constant(2.0) # also tf.float32 implicitly
print(node1, node2)
# checking node type

#The following code creates a Session object and then invokes its run method to run enough of
#the computational graph to evaluate node1 and node2. By running the computational graph in
#a session as follows:
sess = tf.Session()
print(sess.run([node1, node2]))
sess.run(node1)

#Some computations by combining Tensor nodes
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))
print("sess.run(+): ",sess.run(node1 + node2))

#A placeholder is a promise to provide a value later.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
print(sess.run(adder_node, {a: 1, b:2}))
print(sess.run(adder_node, {a: [1,2], b: [3, 4]}))

#more example
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b:4.5}))

#%% Rank of tensors
scalar = tf.constant(100) # 0 dimension
print(scalar.get_shape())

vector = tf.constant([1,2,3,4,5]) # 1 dimension
print(vector.get_shape())

matrix = tf.constant([[1,2,3],[4,5,6]]) # 2 dimension
print(matrix.get_shape())

cube_matrix = tf.constant([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]]) # 3 dimension
print(cube_matrix.get_shape())

#%% Simple Equations
#Variables allow us to add trainable parameters to a graph.
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

#To initialize all the variables in a TensorFlow program
init = tf.global_variables_initializer()
#sess.run(W)
sess.run(init)
sess.run(W)

# It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables. Until we call sess.run, the
# variables are uninitialized.

#Since x is a placeholder, we can evaluate linear_model for several values of x simultaneously as follows:
print(sess.run(linear_model, {x:[1,2,3,4]}))

#%% Various Reduction's operation
#https://www.tensorflow.org/api_docs/python/tf

x = tf.constant([[1,3,5],[2,3,4]], dtype = tf.int32)
sess.run(x)
sess.run(tf.reduce_max(x, axis = 0)) # Row wise vertical downwards
sess.run(tf.reduce_max(x, axis = 1)) # Col wise horizontal
sess.run(tf.reduce_max(x))

# CW: practice for tf.reduce_min and tf.reduce_mean

#%% Few important operations
# Few more: tf.cast
sess.run(tf.cast([1.2,2.8], tf.int32))
sess.run(tf.cast([1.2,2.8], tf.float32))

# Few more: tf.reshape
y = tf.reshape(x, [-1]) # Reshape to a vector. -1 means all. Like np.ravel
sess.run(y)
z = tf.reshape(x, [3,2])
sess.run(z)

sess.close() # Close as in next session , we will have different way to create session
#%% Another type of session. tf.InteractiveSession() is just convenient
#for keeping a default session open in ipython
sess = tf.InteractiveSession()

a = tf.zeros((2,2)); b = tf.ones((2,2))
a.eval()
b.eval()

tf.reduce_sum(b, reduction_indices=1).eval()

a.get_shape()

tf.reshape(a, (1, 4)).eval()

sess.close() # Close as in next session , we will have different way to create session
#%%TensorFlow Variables
w1 = tf.ones((2,2))
w2 = tf.Variable(tf.zeros((2,2)), name="w2")
with tf.Session() as sess:
    print(sess.run(w1))
    # must be initialized before they have values!
    sess.run(tf.global_variables_initializer())
    print(sess.run(w2))

#Updating Variable State
# CW: What does it prints
state = tf.Variable(0, name="counter")
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state)) # # HW: What does it prints
    for _ in range(3):
        sess.run(update)
        print(sess.run(state)) # HW: What does it prints

#convert to tensor from python data
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(sess.run(ta))

#%%Placeholders and Feed Dictionaries
#placeholder is dummy nodes that provide entry points for data to computational graph
#A feed_dict is a python dictionary mapping from tf and used to pass variables
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
#%%Tensor segmentation
sess = tf.InteractiveSession()
seg_ids = tf.constant([0,1,1,2,2]); # Group indexes : 0|1,2|3,4
x = tf.constant([[2, 5, 3, -5],[0, 3,-2, 5],[4, 3, 5, 3],[6, 1,4, 0],[6, 1, 4, 0]]) # A sample constant matrix
tf.segment_sum(x, seg_ids).eval()
#array([[ 2, 5, 3, -5],
#[ 4, 6, 3, 8],
#[12, 2, 8, 0]])

tf.segment_prod(x, seg_ids).eval()
#array([[ 2, 5, 3, -5],
#[ 0, 9, -10, 15],
#[ 36, 1, 16, 0]])

#CW: Run and see the output
tf.segment_min(x, seg_ids).eval()
tf.segment_max(x, seg_ids).eval()
tf.segment_mean(x, seg_ids).eval()
sess.close()
#%% CW: Self practice of Sequences: Various sequence utilities

sess = tf.InteractiveSession()
x = tf.constant([[2, 5, 3, -5],[0, 3,-2, 5],[4, 3, 5, 3],[6, 1, 4, 0]])

# argmin shows the index of minimum value of a dimension
tf.argmin(x, 1).eval()

# argmax shows the index of maximum value of a dimension
tf.argmax(x, 1).eval()

#where : Conditional operation on tensor
boolx = tf.constant([True,False]); a = tf.constant([2,7]); b = tf.constant([4,5])
tf.where(boolx, a+b, a-b).eval()

#listdiff (showing the complement of the intersection between lists)
listx = tf.constant([1,2,3,4,5,6,7,8, 8])
listy = tf.constant([4,5,8,9])
tf.setdiff1d(listx, listy)[0].eval()

#unique (showing unique values on a list).
tf.unique(listx)[0].eval()

sess.close()
#%% CW Practice:Tensor slicing and joining
#In order to extract and merge useful information from big datasets, the slicing and joining
#methods allow you to consolidate the required column information without having to
#occupy memory space with nonspecific information.

sess = tf.InteractiveSession()
t_matrix = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
t_array = tf.constant([1,2,3,4,9,8,6,5])
t_array2= tf.constant([2,3,4,5,6,7,8,9])

#It extracts a slice of size size from a tensor input starting at the location specified by begin.
tf.slice(input_ =  t_matrix, begin = [1, 1], size = [2,2]).eval()

#Splits a tensor into sub tensors
split0, split1 = tf.split(value=t_array, num_or_size_splits=2, axis=0)

# View the splitted part's shape
tf.shape(split0)
tf.shape(split1)

# View the splitted part's content
split0.eval()
split1.eval()

# creates a new tensor by replicating input multiples times
tf.tile(input = [1,2], multiples = [3]).eval()

#Packs the list of tensors in values into a tensor with rank one higher than each tensor in values,
#by packing them along the axis dimension.
#Given a list of length N of tensors of shape (A, B, C); if axis == 0 then the output tensor will
#have the shape (N, A, B, C). if axis == 1 then the output tensor will have the shape (A, N, B, C).

# Simple example
tf.stack(values = [t_array, t_array2], axis=0).eval()

# Detail example
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.stack([x, y, z], axis=0).eval()  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
tf.stack([x, y, z], axis=1).eval()  # [[1, 2, 3], [4, 5, 6]]
# Same as np.stack([x, y, z])

# unstack is just opposite of stack, explained above
sess.run(tf.unstack(t_matrix))

# Concatenates the list of tensors values along dimension axis
tf.concat(values=[t_array, t_array2], axis=0,).eval()

# One more detail example of concat
t1 = tf.constant([[1, 2, 3], [4, 5, 6]])
t2 = tf.constant([[7, 8, 9], [10, 11, 12]])
tf.concat([t1, t2], 0).eval()  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1).eval()  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

sess.close()

#%% Eager execution: Go to the top and start afresh
