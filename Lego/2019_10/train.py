# 필요한 패키지들
import numpy as np
import tensorflow as tf
from data_read import data_read

from datetime import datetime

# PIL는 이미지를 load 할 때, numpy는 array

tf.debugging.set_log_device_placement(True)

learning_rate = 1e-5  # 학습 주기
batch_size = 50  # 학습당 학습량
test_size = 1
num_epochs = 100
data_read = data_read(batch_size)
train_list = glob('E:\\LEGO\\train\\*\\*.png')
test_list = glob('E:\\LEGO\\test\\*\\*.png')
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
        W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=1e-2))
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
        # 앞에서 입력받은 :다차:wq원 텐서를 fcc에 넣기 위해서 1차원으로 피는 작업
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
    r_cnn1 = conv1(images)  # convolutional layer 1
    r_cnn2 = conv2(r_cnn1)  # convolutional layer 2

    # output shape will be (*,12,12,64)
    r_cnn3 = conv3(r_cnn2)  # convolutional layer 3

    # output shape will be (*,6,6,128)
    r_cnn4 = conv4(r_cnn3)  # convolutional layer 4

    # fully connected layer 1
    r_fc1 = fc1(r_cnn4)
    r_dropout1 = tf.nn.dropout(r_fc1, keep_prob)
    # fully connected layer2
    r_fc2 = fc2(r_dropout1)

    ## drop out
    r_dropout2 = tf.nn.dropout(r_fc2, keep_prob)

    # final layer
    r_out = final_out(r_dropout2)

    return r_out

X = tf.placeholder(tf.float32, shape=[None, 200, 200, 1])
Y = tf.placeholder(tf.float32, shape=[None, 4])
keep_prob = tf.placeholder(tf.float32)
iterator = data_read.made_batch(train_list, batch_size).make_initializable_iterator()
test_iterator = data_read.made_batch(test_list, test_size).make_initializable_iterator()

prediction = build_model(X, keep_prob)
# define loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))

train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
label_max = tf.argmax(Y, 1)
pre_max = tf.argmax(prediction, 1)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
startTime = datetime.now()
train_next = iterator.get_next()
test_next = test_iterator.get_next()
tf.summary.scalar('cost', cost)
tf.summary.scalar('accuracy', accuracy)
summary = tf.summary.merge_all()
with tf.Session() as sess:

    sess.run(iterator.initializer)
    sess.run(test_iterator.initializer)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()  # create saver to store training model into file
    summary_writer = tf.summary.FileWriter('tensor_board_lego', sess.graph)
    for step in range(num_epochs):
        avg_cost = 0
        for x in range(int(data_read.data_size/batch_size)):
            x_data, y_data = sess.run(train_next)
            cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})
            avg_cost += cost_val / int(data_read.data_size/batch_size)
            print(x)
            if(x%4 == 0 ):
                validate_images_, validate_labels_ = sess.run(test_next)
                rv = sess.run([label_max, pre_max, cost, accuracy], feed_dict={X: validate_images_
                    , Y: validate_labels_
                    , keep_prob: 1.0})
                print('Validation cost:', rv[2], ' accuracy:', rv[3])

        now = datetime.now() - startTime

        print('step: ', step, 'cost_val : ', avg_cost, 'time', now)

    summary_str = sess.run(summary, feed_dict={X: validate_images_
        , Y: validate_labels_
        , keep_prob: 1.0})
    summary_writer.flush()
    saver.save(sess, 'lego2019_10_31.ckpt')  # save session
