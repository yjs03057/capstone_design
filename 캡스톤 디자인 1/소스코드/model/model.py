
import tensorflow as tf
from preprocess import read_label_file, _one_hot_encoder, read_test_label_file
import os
import numpy as np
from random import shuffle
from math import ceil

os.environ['KMP_DUPLICATE_LIB_OK']='True'
tf.set_random_seed(777)  # reproducibility


data0 = np.load('class_1_1.npy')
data1 = np.load('class_1_2.npy')
data2 = np.load('class_2_1.npy')
data3 = np.load('class_2_2.npy')
data4 = np.load('class_3_1.npy')
data5 = np.load('class_3_2.npy')
data6 = np.load('class_4_1.npy')
data7 = np.load('class_4_2.npy')

data = np.r_[data0, data1, data2, data3, data4, data5, data6, data7]
print(data.shape)
label = _one_hot_encoder(read_label_file())

# hyper parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 400
nb_classes = 4

def shuffle_data(xFeed, yFeed):
    # 셔플
    xFeed_shuf = []
    yFeed_shuf = []
    index_shuf = list(range(len(xFeed)))
    shuffle(index_shuf)
    # one more shuffle
    shuffle(index_shuf)
    for i in index_shuf:
        xFeed_shuf.append(xFeed[i])
        yFeed_shuf.append(yFeed[i])

    return xFeed_shuf, yFeed_shuf

data, label = shuffle_data(data, label)
train_data = data[:6001]
train_label = label[:6001]

vali_data = data[6001:]
vali_label = label[6001:]

test_data = np.load('test.npy')
test_label = _one_hot_encoder(read_test_label_file())

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 40, 150])
            X_img = tf.reshape(self.X, [-1, 40, 150, 1])
            self.Y = tf.placeholder(tf.float32, [None, 4])

            conv1 = tf.layers.conv2d(inputs=X_img, filters=16, kernel_size=[1,2], padding="SAME",
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1,2], padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(pool1, self.keep_prob)


            conv2 = tf.layers.conv2d(inputs=dropout1, filters=32, kernel_size=[1,2], padding="SAME",
                                     activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1,2], padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(pool2, self.keep_prob)


            conv3 = tf.layers.conv2d(inputs=dropout2, filters=16, kernel_size=[1,2], padding="SAME",
                                     activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1,2], padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(pool3, self.keep_prob)

            flat = tf.reshape(dropout3, [-1, 5 * 19 * 16])
            self.logits = tf.layers.dense(flat, nb_classes, activation=None)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.2):
        # print(y_data)
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})
# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning Started!')
# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(train_data) / batch_size)

    for i in range(int(ceil(len(train_data)/batch_size))):
        batch_xs = train_data[batch_size * i : batch_size * (i+1)]
        batch_ys = train_label[batch_size * i : batch_size * (i+1)]
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Validation Accuracy : ', m1.get_accuracy(vali_data, vali_label))

print('Learning Finished!')
print('Accuracy:', m1.get_accuracy(test_data, test_label))


#
# for step in range(45):
#     c, _ = m1.train(train_data, train_label)
#     print('Step:', '%04d' % (step + 1), 'cost =', '{:.9f}'.format(c))
#     print('Accuracy: ', m1.get_accuracy(vali_data, vali_label))
# # Test model and check accuracy
# print('Learning Finished!')
