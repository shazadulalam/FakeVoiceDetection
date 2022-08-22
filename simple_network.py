from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

learning_rate = 0.1
num_steps = 5000
batch_size = 128
display_steep = 200

n_hidden_1 = 256
n_hidden_2 = 256

num_input = 784
num_classes = 10

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

#print(X, Y)

#store layer weight and biases
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([num_classes]))
}

#model creation
def neural_net(x):
    #hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer

#construct models
logits = neural_net(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits = logits, labels = Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#initialize the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1,num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        #run optimization op backpropagation

        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if (step % display_steep == 0 or step == 1):
            #calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})

            print("step" + str(step) + ", Minibatch loss = " + \
                "{:.4f}".format(loss) + ", Training accuracy = " \
                "{:.3f}".format(acc))

    print("Optimization Finished")

    #calculate accuracy for mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))

