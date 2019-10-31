import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob

# PIL는 이미지를 load 할 때, numpy는 array
from PIL import Image
from numpy import array
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
batch_size = 50  
data_num = 0
num_epochs = 30
# convolutional network layer 1
def conv1(input_data):
    # layer 1 (convolutional layer)
    with tf.name_scope('conv_1'):
        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 128], stddev=1e-2))
        b1 = tf.Variable(tf.truncated_normal([128], stddev=1e-2))
        h_conv1 = tf.nn.conv2d(input_data, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1_relu = tf.nn.relu(tf.add(h_conv1, b1))
        h_conv1_maxpool = tf.nn.max_pool(h_conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv1_maxpool

# convolutional network layer 2
def conv2(input_data):

    with tf.name_scope('conv_2'):
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 128, 256], stddev=1e-2))
        b2 = tf.Variable(tf.truncated_normal([256], stddev=1e-2))
        h_conv2 = tf.nn.conv2d(input_data, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2_relu = tf.nn.relu(tf.add(h_conv2, b2))
        h_conv2_maxpool = tf.nn.max_pool(h_conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv2_maxpool

# convolutional network layer 3
def conv3(input_data):
    with tf.name_scope('conv_3'):
        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=1e-2))
        b3 = tf.Variable(tf.truncated_normal([512], stddev=1e-2))
        h_conv3 = tf.nn.conv2d(input_data, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3_relu = tf.nn.relu(tf.add(h_conv3, b3))
        h_conv3_maxpool = tf.nn.max_pool(h_conv3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv3_maxpool

# convolutional network layer 3
def conv4(input_data):

    with tf.name_scope('conv_4'):
        W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=1e-2))
        b4 = tf.Variable(tf.truncated_normal([512], stddev=1e-2))
        h_conv4 = tf.nn.conv2d(input_data, W_conv4, strides=[1, 1, 1, 1], padding='SAME')
        h_conv4_relu = tf.nn.relu(tf.add(h_conv4, b4))
        h_conv4_maxpool = tf.nn.max_pool(h_conv4_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv4_maxpool
# fully connected layer 1
def fc1(input_data):
    input_layer_size = 13 * 13 * 512

    with tf.name_scope('fc_1'):
        # 앞에서 입력받은 다차원 텐서를 fcc에 넣기 위해서 1차원으로 피는 작업
        input_data_reshape = tf.reshape(input_data, [-1, input_layer_size])
        W_fc1 = tf.Variable(tf.truncated_normal([input_layer_size, 512], stddev=1e-2))
        b_fc1 = tf.Variable(tf.truncated_normal([512], stddev=1e-2))
        h_fc1 = tf.add(tf.matmul(input_data_reshape, W_fc1), b_fc1)  # h_fc1 = input_data*W_fc1 + b_fc1
        h_fc1_relu = tf.nn.relu(h_fc1)

    return h_fc1_relu


# fully connected layer 2
def fc2(input_data):

    with tf.name_scope('fc_2'):
        W_fc2 = tf.Variable(tf.truncated_normal([512, 256], stddev=1e-2))
        b_fc2 = tf.Variable(tf.truncated_normal([256], stddev=1e-2))
        h_fc2 = tf.add(tf.matmul(input_data, W_fc2), b_fc2)  # h_fc1 = input_data*W_fc1 + b_fc1
        h_fc2_relu = tf.nn.relu(h_fc2)

    return h_fc2_relu


# final layer
def final_out(input_data):
    with tf.name_scope('final_out'):
        W_fo = tf.Variable(tf.truncated_normal([256, 4], stddev=1e-2))
        b_fo = tf.Variable(tf.truncated_normal([4], stddev=1e-2))
        h_fo = tf.add(tf.matmul(input_data, W_fo), b_fo)  # h_fc1 = input_data*W_fc1 + b_fc1

    # 최종 레이어에 softmax 함수는 적용하지 않았다.
    return h_fo

# build cnn_graph
def build_model(images, keep_prob):
    # define CNN network graph
    # output shape will be (*,48,48,16)
    r_cnn1 = conv1(images)  # convolutional layer 1

    # output shape will be (*,24,24,32)
    r_cnn2 = conv2(r_cnn1)  # convolutional layer 2

    # output shape will be (*,12,12,64)
    r_cnn3 = conv3(r_cnn2)  # convolutional layer 3
    # output shape will be (*,6,6,128)
    r_cnn4 = conv4(r_cnn3)  # convolutional layer 4
    # fully connected layer 1
    r_fc1 = fc1(r_cnn4)
    # fully connected layer2
    r_fc2 = fc2(r_fc1)
    # final layer
    r_out = final_out(r_fc2)

    return r_out

X = tf.placeholder(tf.float32, shape=[None, 200, 200, 1])
keep_prob = tf.placeholder(tf.float32)

prediction = tf.nn.softmax(build_model(X, keep_prob))
#prediction = build_model(X, keep_prob)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, 'E:\\LEGO\\lego2019_10_20')
#summary = tf.summary.merge_all()

image = np.array(Image.open('C:\\Users\\Aluminum\\PycharmProjects\\lego\\test\\2357_30.png').convert('L'))
image = image.reshape([-1, image.shape[0], image.shape[1], 1])
image = image.astype(np.int32)
image = tf.image.resize(image, [200, 200])
image = sess.run(image)
p_val = sess.run(prediction, feed_dict={X: image, keep_prob:1})
#summary_writer = tf.summary.FileWriter('E:\\LEGO\\tensor_board_lego_val', sess.graph)

name_labels = ['2357', '3003', '3004', '3005']
i = 0

for x in p_val[0]:
    print('%s              %f' % (name_labels[i], float(x)))
    i += 1
