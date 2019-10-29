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
tf.debugging.set_log_device_placement(True)
learning_rate = 0.001  # 학습 주기
batch_size = 50  # 학습당 학습량
data_num = 0
num_epochs = 30
# convolutional network layer 1
def conv1(X):
    with tf.name_scope('conv1'):  # 200,200
        W1 = tf.Variable(tf.random_normal(shape=[5, 5, 1, 128], stddev=0.01))
        L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')

        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return L1


# convolutional network layer 2
def conv2(input_data):
    with tf.name_scope('conv2'):  # 100, 100
        W2 = tf.Variable(tf.random_normal(shape=[3, 3, 128, 256], stddev=0.01))
        L2 = tf.nn.conv2d(input_data, W2, strides=[1, 1, 1, 1], padding='SAME')

        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return L2


# convolutional network layer 3
def conv3(input_data):
    with tf.name_scope('conv3'):  # 50,50
        W3 = tf.Variable(tf.random_normal(shape=[3, 3, 256, 128], stddev=0.01))
        L3 = tf.nn.conv2d(input_data, W3, strides=[1, 1, 1, 1], padding='SAME')

        L3 = tf.nn.relu(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return L3

# fully connected layer 1
def fc1(input_data):
    with tf.name_scope('fc_1'):
        L3_flat = tf.reshape(input_data, [-1, 25 * 25 * 128])
        W3 = tf.get_variable('W3', shape=[25 * 25 * 128, 256])
        b = tf.Variable(tf.random_normal([256]))
        h_fc1 = tf.add(tf.matmul(L3_flat, W3), b)  # h_fc1 = input_data*W_fc1 + b_fc1
        h_fc1_relu = tf.nn.relu(h_fc1)

    return h_fc1_relu

# final layer
def final_out(input_data):
    with tf.name_scope('final_out'):
        W_fo = tf.Variable(tf.truncated_normal([256, 3], stddev=0.1))
        b_fo = tf.Variable(tf.truncated_normal([3], stddev=0.1))
        h_fo = tf.add(tf.matmul(input_data, W_fo), b_fo)  # h_fc1 = input_data*W_fc1 + b_fc1

    return h_fo


# build cnn_graph
def build_model(images, keep_prob):
    # define CNN network graph
    # output shape will be (*,48,48,16)
    r_cnn1 = conv1(images)  # convolutional layer 1
    print("shape after cnn1 ", r_cnn1.get_shape())

    # output shape will be (*,24,24,32)
    r_cnn2 = conv2(r_cnn1)  # convolutional layer 2
    print("shape after cnn2 :", r_cnn2.get_shape())

    # output shape will be (*,12,12,64)
    r_cnn3 = conv3(r_cnn2)  # convolutional layer 3
    print("shape after cnn3 :", r_cnn3.get_shape())

    # fully connected layer 1
    r_fc1 = fc1(r_cnn3)
    print("shape after fc1 :", r_fc1.get_shape())

    ## drop out
    # 참고 http://stackoverflow.com/questions/34597316/why-input-is-scaled-in-tf-nn-dropout-in-tensorflow
    # 트레이닝시에는 keep_prob < 1.0 , Test 시에는 1.0으로 한다.
    r_dropout = tf.nn.dropout(r_fc1, keep_prob)
    print("shape after dropout :", r_dropout.get_shape())
    # final layer
    r_out = final_out(r_dropout)
    print("shape after final layer :", r_out.get_shape())

    return r_out

X = tf.placeholder(tf.float32, shape=[None, 200, 200, 1])
#X_img = tf.reshape(X, [-1, 200, 200, 1])
Y = tf.placeholder(tf.float32, shape=[None, 3])
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('conv1'):     #200,200
    W1 = tf.Variable(tf.random_normal(shape=[5, 5, 1, 128], stddev=0.01))
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')

    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('conv2'):     # 100, 100
    W2 = tf.Variable(tf.random_normal(shape=[3, 3, 128, 256], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')

    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('conv3'):     # 50,50
    W3 = tf.Variable(tf.random_normal(shape=[3, 3, 256, 128], stddev=0.01))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')

    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

L3_flat = tf.reshape(L3, [-1, 25 * 25 * 128])

W3 = tf.get_variable('W3', shape=[25 * 25 * 128, 3])
b = tf.Variable(tf.random_normal([3]))
H = tf.nn.dropout(L3_flat, keep_prob)
H = tf.matmul(H, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=Y))

train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

prediction = tf.nn.softmax(H)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)


with tf.Session() as sess:
    startTime = datetime.now()
    saver = tf.train.Saver()  # create saver to store training model into file
    saver.restore(sess, 'E:/LEGO/lego2')
    #summary_writer = tf.summary.FileWriter('E:/LEGO/', sess.graph)
    img = cv2.imread('3003_1.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(200, 200))
    image = np.array(Image.open('3003_1.png').convert('L'))
    # Channel 1을 살려주기 위해 reshape 해줌
    img = np.reshape(img, [-1, 200, 200, 1])
    sess.run(tf.global_variables_initializer())
    #image = np.array(Image.open('C:\\Users\\BIT\\PycharmProjects\\untitled2\\3003.png').convert('L'))
    image = image.reshape([-1, image.shape[0], image.shape[1], 1])
    image = image.astype(np.int32)
    image = tf.image.resize_images(image, [200, 200])
    image = sess.run(image)

    now = datetime.now() - startTime
    p_val = sess.run(prediction, feed_dict={X: image, keep_prob:1})
    #saver.save(sess, 'E:/LEGO/lego')  # save session

    name_labels = ['3003', '3004', '3005']
    i = 0

    for x in p_val[0]:

        print('%s              %f'%(name_labels[i], float(x)))
        i += 1