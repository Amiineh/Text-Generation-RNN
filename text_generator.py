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
learning_rate = 0.001
seq_len = 40
stride = 3
hidden_size = 128
epoch_size = 50
batch_size = 100
test_size = 5

# Prepare training data:
train_data = []
target_data = []
for i in range(0, int((text_len - seq_len)/stride) + 1, seq_len - stride):
    train_data.append([char_to_idx[ch] for ch in text[i:i+seq_len]])
    target_data.append(char_to_idx[text[i+seq_len]])

x_train = np.zeros((len(train_data), seq_len, dict_size))
y_train = np.zeros((len(train_data), seq_len, dict_size))

for seq_index, seq in enumerate(train_data):
    for i in range(seq_len):
        x_train[seq_index, i, seq[i]] = 1
        if i <seq_len-1:
            y_train[seq_index, i, seq[i+1]] = 1

# Model the network:
x = tf.placeholder(tf.float32, (None, seq_len, dict_size))
y = tf.placeholder(tf.float32, (None, seq_len, dict_size))

cell = rnn.GRUCell(hidden_size)
outputs, states = tf.nn.dynamic_rnn(cell, x,dtype=tf.float32)

outputs_flat = tf.reshape(outputs, [-1, hidden_size])

softmax_w = tf.Variable(tf.truncated_normal([hidden_size, dict_size], mean=0, stddev=0.1))
softmax_b = tf.Variable(tf.zeros(dict_size))
tf.summary.histogram("w", softmax_w)
tf.summary.histogram("b", softmax_b)

predictions = tf.matmul(outputs_flat, softmax_w) + softmax_b
tf.summary.histogram("predictions", predictions)

y_flat = tf.reshape(y, [-1, dict_size])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_flat, logits=predictions))
train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

prediction = tf.nn.softmax(predictions[-1])

# Start training:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

# Show summaries:
writer = tf.summary.FileWriter('./log/', sess.graph)
merge = tf.summary.merge_all()

# Save the model:
# saver = tf.train.Saver()
# saver.restore(sess, "./save/")

for epoch in range(epoch_size):
    # saver.save(sess, "./save/model", global_step=1000, write_meta_graph=False)

    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        sess.run(train_op, feed_dict={x:batch_x, y: batch_y})

    report_loss, merge_smry = sess.run([loss, merge], feed_dict={x:x_train, y:y_train})
    print ("Epoch #%d\t Loss: %f" % (epoch+1, report_loss))
    # tf.summary.scalar("loss", report_loss)
    smry = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=report_loss)])
    writer.add_summary(smry, epoch)
    writer.add_summary(merge_smry, epoch)

# Test the model:
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
            x_test = np.append(np.delete(x_test, 0, axis=1), np.reshape(predicted, [1, 1, dict_size]))
        x_test = np.reshape(x_test, [1, seq_len, dict_size])
        predicted = sess.run(prediction, feed_dict={x: x_test})
        predicted_char = idx_to_char[np.argmax(predicted, 0)]
        res_chars += predicted_char

        # x_test_idx = []
        # for w in x_test[0]:
        #     x_test_idx.append(np.argmax(w))
        # print (x_test_idx)

    print ("\nGenerated text: ")
    print ("\"%s\"\n\n" %(res_chars))

