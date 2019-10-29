import os
import cv2
import numpy as np
import tensorflow as tf

from numpy import array
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

tf.debugging.set_log_device_placement(True)
learning_rate = 0.001  # 학습 주기
batch_size = 100  # 학습당 학습량
log_dir = 'E:/LEGO/'

def get_input_queue():
    train_images = []
    train_labels = []
    TRAIN_DIR = 'E:/LEGO/train/'
    train_folder_list = array(os.listdir(TRAIN_DIR))
    for index in range(len(train_folder_list)):
        path = TRAIN_DIR+ train_folder_list[index]
        path = path + '/'
        img_list = os.listdir(path)
        for list in range(len(img_list)):
            img_path = path+ img_list[list]
            train_images.append(img_path)
            train_labels.append(int(index))
    print('train_image',train_images, len(train_images), train_labels,  len(train_labels))
    input_queue = tf.train.slice_input_producer([train_images, train_labels], shuffle=True)
    #input_queue = tf.data.Dataset.from_tensor_slices([train_images, train_labels])
    print('input__', input_queue[0])
    return input_queue


def read_data(input_queue):
    image_file = input_queue[0]
    label = input_queue[1]

    image = tf.image.decode_png(tf.read_file(image_file), channels=1)
    print('image',image, 'read',tf.read_file(image_file), input_queue[0])
    return image, label, image_file


def read_data_batch(batch_size=batch_size):
    input_queue = get_input_queue()
    image, label, file_name = read_data(input_queue)
    image = tf.reshape(image, [200, 200, 1])

    # random image
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, lower=0.2, upper=2.0)
    image = tf.image.random_hue(image, max_delta=0.08)
    image = tf.image.random_saturation(image, lower=0.2, upper=2.0)

    batch_image, batch_label, batch_file = tf.train.batch([image, label, file_name], batch_size=batch_size)
    # ,enqueue_many=True)
    batch_file = tf.reshape(batch_file, [batch_size, 1])

    batch_label_on_hot = tf.one_hot(tf.to_int64(batch_label), 3, on_value=1.0, off_value=0.0)
    return batch_image, batch_label_on_hot, batch_file


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


def main(argv=None):
    # define placeholders for image data & label for traning dataset

    images = tf.placeholder(tf.float32, [None, 200, 200, 1])
    labels = tf.placeholder(tf.int32, [None, 3])
    image_batch, label_batch, file_batch = read_data_batch()

    keep_prob = tf.placeholder(tf.float32)  # dropout ratio
    prediction = build_model(images, keep_prob)
    # define loss function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

    tf.summary.scalar('loss', cost)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    # for validation
    # with tf.name_scope("prediction"):

    #validate_image_batch, validate_label_batch, validate_file_batch = read_data_batch()
    #label_max = tf.argmax(labels, 1)
    #pre_max = tf.argmax(prediction, 1)
    #correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #tf.summary.scalar('accuracy', accuracy)

    startTime = datetime.now()

    # build the summary tensor based on the tF collection of Summaries
    summary = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        saver = tf.train.Saver()  # create saver to store training model into file
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        init_op = tf.global_variables_initializer()  # use this for tensorflow 0.12rc0
        sess.run(init_op)
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(10000):
            images_, labels_ = sess.run([image_batch, label_batch])
            # sess.run(train_step,feed_dict={images:images_,labels:labels_,keep_prob:0.8})
            cost_val, _ = sess.run([cost, train], feed_dict={images: images_, labels: labels_, keep_prob: 0.7})

            if i %100:
                print('cost', cost)
            '''if i % 10 == 0:
                now = datetime.now() - startTime
                print('## time:', now, ' steps:', i)

                # print out training status
                rt = sess.run([label_max, pre_max, loss, accuracy], feed_dict={images: images_
                    , labels: labels_
                    , keep_prob: 1.0})
                print('Prediction loss:', rt[2], ' accuracy:', rt[3])
                # validation steps
                validate_images_, validate_labels_ = sess.run([validate_image_batch, validate_label_batch])
                rv = sess.run([label_max, pre_max, loss, accuracy], feed_dict={images: validate_images_
                    , labels: validate_labels_
                    , keep_prob: 1.0})
                print('Validation loss:', rv[2], ' accuracy:', rv[3])
                if (rv[3] > 0.9):
                    break
                # validation accuracy
                summary_str = sess.run(summary, feed_dict={images: validate_images_
                    , labels: validate_labels_
                    , keep_prob: 1.0})
                summary_writer.add_summary(summary_str, i)
                summary_writer.flush()'''

        saver.save(sess, 'E:/LEGO/lego_recog')  # save session
        coord.request_stop()
        coord.join(threads)
        print('finish')


main()