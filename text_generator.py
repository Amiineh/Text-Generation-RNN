import tensorflow as tf
from tensorflow.contrib import rnn
from random import *
import numpy as np

# Read data:
text = open("nietzsche.txt", 'r').read().lower()
chars = sorted(list(set(text)))
char_to_idx = dict((ch,i) for i, ch in enumerate(chars))
idx_to_char = dict((i,ch) for i, ch in enumerate(chars))
dict_size = len(chars)
text_len = len(text)

# Set hyperparameters:
learning_rate = 0.01
seq_len = 40
stride = 37
hidden_size = 128
epoch_size = 30
batch_size = 100
test_size = 5

# Prepare training data:
train_data = []
target_data = []
for i in range(0, int((text_len - seq_len)/stride) + 1, seq_len - stride):
    train_data.append([char_to_idx[ch] for ch in text[i:i+seq_len]])
    target_data.append(char_to_idx[text[i+seq_len]])

x_train = np.zeros((len(train_data), seq_len, dict_size))
y_train = np.zeros((len(train_data), dict_size))

for seq_index, seq in enumerate(train_data):
    y_train[seq_index, target_data[seq_index]] = 1
    for i in range(seq_len):
        x_train[seq_index, i, seq[i]] = 1

# Model the network:
x = tf.placeholder(tf.float32, (None, seq_len, dict_size))
x_flat = tf.reshape(tf.transpose(x, [1, 0, 2]), [-1, dict_size])
# x_flat = tf.split(x, seq_len, 0)
y = tf.placeholder(tf.float32, (None, dict_size))

cell = rnn.GRUCell(hidden_size)
initial_state = cell.zero_state(batch_size, dtype=tf.float32)
# outputs, states = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state,dtype=tf.float32)
outputs, states = rnn.static_rnn(cell, tf.unstack(tf.transpose(x, [1, 0, 2])), dtype=tf.float32)

softmax_w = tf.Variable(tf.truncated_normal([hidden_size, dict_size]))
softmax_b = tf.Variable(tf.zeros(dict_size))

# predictions = tf.matmul(outputs[:,-1,:], softmax_w) + softmax_b
predictions = tf.matmul(outputs[-1], softmax_w) + softmax_b

# w = tf.ones((batch_size, seq_len))
# loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=y, weights= w))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions))
train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Start training:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

for epoch in range(epoch_size):
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        sess.run(train_op, feed_dict={x:batch_x, y: batch_y})

    print ("Epoch #%d\t Loss: %f" % (epoch+1, sess.run(loss, feed_dict={x:x_train, y:y_train})))

# Test the model:
# for i in range(test_size):
    print ("-------------------------------------------------------- Test #%d --------------------------------------------------------"%(epoch+1))
    print ("Input: ")
    test_input_chars = ''
    rand_idx = randint(0, len(x_train))
    x_test = x_train[rand_idx]
    x_test_idx = [np.where(w==1)[0][0] for w in x_test]
    for w in train_data[rand_idx]:
        test_input_chars += idx_to_char[w]

    print ("\"%s\"" %(test_input_chars))

    # Predict next 400 characters:
    res_chars = ''
    for i in range(400):
        if (i != 0):
            x_test[0, 0:-2] = x_test[0, 1:-1]
            x_test[0, -1] = predicted
        x_test = np.reshape(x_test, [1, seq_len, dict_size])
        predicted = sess.run(predictions, feed_dict={x: x_test})
        predicted_char = idx_to_char[np.argmax(predicted, 1)[0]]

        res_chars += predicted_char
        # x_test_idx = []
        # for w in x_test[0]:
        #     x_test_idx.append(np.argmax(w))

    print ("\nGenerated text: ")
    print ("\"%s\"\n\n" %(res_chars))

