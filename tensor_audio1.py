import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def load_sound_files(file_paths):
    raw_sounds = []
    labels = []
    for fp in  file_paths:
        X, sample_rate = librosa.load(fp, res_type='kaiser_fast', duration=2.97)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
        # new_shape = librosa.feature.melspectrogram(y = X, sr = sample_rate)
        label = fp.split('.wav')

        raw_sounds.append(mfccs)
        # raw_sounds.append(new_shape)
        # labels.append(label)
        # raw_sounds = np.reshape(mfccs, (-1,2))
    raw_sounds = np.asarray(raw_sounds)
    # labels = np.asarray(labels)
    # raw_sounds = raw_sounds.reshape(len(raw_sounds), 40)
    return raw_sounds


sound_file_paths = glob.glob("/home/shul/audio/originalData/*.wav")

raw_sounds = load_sound_files(sound_file_paths[:8000])
raw_sounds = np.reshape(raw_sounds, (40, 40*40*5))
print(len(raw_sounds[1]))
print("##################################################################", raw_sounds.shape, raw_sounds[4].shape)
# print(labels[1:10])

learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 10

num_input = raw_sounds[1]
num_classes = 16
dropout = 0.75
iteration = 2000

def weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))

def biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def conv_layer(input, num_input_channels, filter_size, num_filters):

    weight = weights(shape=[filter_size, filter_size, num_input_channels, num_filters])

    bias = biases(num_filters)

    layer = tf.nn.conv2d(input = input, filter = weight, strides = [1,1,1,1], padding = 'SAME')
    layer = tf.nn.max_pool(value = layer, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    layer = tf.nn.relu(layer)

    return layer

def flatten_layer(layer):

    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
 
    return layer

def fc_layer(input, num_inputs, num_outputs, use_relu=True):

    weight = weights(shape=[num_inputs, num_outputs])
    bias = biases(num_outputs)

    layer = tf.matmul(input, weight) + bias
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, 40, 40, 5], name='x')
x = tf.cast(x, tf.float32)
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')

y = tf.argmax(y, dimension=1)
y = tf.cast(y, tf.float32)

filter_size_conv1 = 5 
num_filters_conv1 = 32
fc_layer_size = 128

convLayer = conv_layer(input=x, num_input_channels=5,  filter_size=filter_size_conv1, num_filters=num_filters_conv1)
flattenLayer = flatten_layer(convLayer)

fc_layer = fc_layer(input=flattenLayer,
                     num_inputs=flattenLayer.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

y_pred = tf.nn.softmax(fc_layer, name='y_pred')
y_pred = tf.cast(y_pred, tf.float32)
pred_class = tf.argmax(y_pred, dimension=1)
pred_class = tf.cast(pred_class, tf.float32)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fc_layer, labels = y)

loss = tf.reduce_mean(cross_entropy)

print(loss)


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

correct_pred = tf.equal(y_pred, pred_class)
correct_pred = tf.cast(correct_pred, tf.float32)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

samples = tf.train.slice_input_producer([raw_sounds], num_epochs = None)
batch = tf.train.batch(samples, 10)

with session:

    session.run(init)
    for step in range(iteration):

        logs = session.run(train_op)

        if step % display_step == 0 or step == 1:
            tain_loss, train_acc = session.run(loss, accuracy)

            print("Step" + str(step) + ", Minibatch Loss = " + "{:.4f}".format(tain_loss) + ", Training Accuracy = " + "{:.3f}".format(train_acc))

    print("Finished")

    print("Testing Accuracy: ", session.run(accuracy))
