# 필요한 패키지들
import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from datetime import datetime

# PIL는 이미지를 load 할 때, numpy는 array
from PIL import Image
import numpy as np

tf.debugging.set_log_device_placement(True)

batch_size = 100
learning_rate = 0.001
data_num = 0
num_epochs = 1000
data_list = glob('E:\\LEGO\\train\\*\\*.png')

def get_label_from_path(path):
    return path.split('\\')[-2]

def read_image(path):
    image = np.array(Image.open(path).convert('L'))
    # Channel 1을 살려주기 위해 reshape 해줌
    return image.reshape(image.shape[0], image.shape[1], 1)

# 앞서 만들었던 get_label_from_path 함수를 통해 data_list에 있는 label 이름들을 list에 다 묶어준다
# 더 쉬운 방법이 있지만, 굳이 함수를 통해 label 들을 얻는 것은 함수도 잘 작동하는지 확인함을 목적을 가지고 있다.

label_name_list = []
for path in data_list:
    label_name_list.append(get_label_from_path(path))

unique_label_names = np.unique(label_name_list)

def onehot_encode_label(path):
    onehot_label = unique_label_names == get_label_from_path(path)
    onehot_label = onehot_label.astype(np.uint8)
    return onehot_label

def _read_py_function(path, label):
    image = read_image(path)
    label = np.array(label, dtype=np.uint8)
    return image.astype(np.int32), label

def _resize_function(image_decoded, label):
    image_decoded.set_shape([None, None, None])
    image = tf.image.resize_images(image_decoded, [200, 200])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image, label

# label을 array 통채로 넣는게 아니고, list 화 시켜서 하나씩 넣기 위해 list로 바꿔주었다.
label_list = [onehot_encode_label(path).tolist() for path in data_list]

dataset = tf.data.Dataset.from_tensor_slices((data_list, label_list))
dataset = dataset.map(
    lambda data_list, label_list: tuple(tf.py_func(_read_py_function, [data_list, label_list], [tf.int32, tf.uint8])))

dataset = dataset.map(_resize_function)
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size=(int(len(data_list) * 0.4) + 3 * batch_size))
dataset = dataset.batch(batch_size)

# convolutional network layer 1
def conv1(input_data):
    # layer 1 (convolutional layer)
    conv1_filter_size = 3
    conv1_layer_size = 16
    stride1 = 1

    with tf.name_scope('conv_1'):
        W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1))
        b1 = tf.Variable(tf.truncated_normal([16], stddev=0.1))
        h_conv1 = tf.nn.conv2d(input_data, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1_relu = tf.nn.relu(tf.add(h_conv1, b1))
        h_conv1_maxpool = tf.nn.max_pool(h_conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv1_maxpool


# convolutional network layer 2
def conv2(input_data):
    conv2_filter_size = 3
    conv2_layer_size = 32
    stride2 = 1

    with tf.name_scope('conv_2'):
        W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1))
        b2 = tf.Variable(tf.truncated_normal([32], stddev=0.1))
        h_conv2 = tf.nn.conv2d(input_data, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2_relu = tf.nn.relu(tf.add(h_conv2, b2))
        h_conv2_maxpool = tf.nn.max_pool(h_conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv2_maxpool


# convolutional network layer 3
def conv3(input_data):
    conv3_filter_size = 3
    conv3_layer_size = 64
    stride3 = 1

    print('## stride1 ', 1)
    with tf.name_scope('conv_3'):
        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
        b3 = tf.Variable(tf.truncated_normal([64], stddev=0.1))
        h_conv3 = tf.nn.conv2d(input_data, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3_relu = tf.nn.relu(tf.add(h_conv3, b3))
        h_conv3_maxpool = tf.nn.max_pool(h_conv3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv3_maxpool


# convolutional network layer 3
def conv4(input_data):
    conv4_filter_size = 5
    conv4_layer_size = 128
    stride4 = 1

    with tf.name_scope('conv_4'):
        W_conv4 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1))
        b4 = tf.Variable(tf.truncated_normal([128], stddev=0.1))
        h_conv4 = tf.nn.conv2d(input_data, W_conv4, strides=[1, 1, 1, 1], padding='SAME')
        h_conv4_relu = tf.nn.relu(tf.add(h_conv4, b4))
        h_conv4_maxpool = tf.nn.max_pool(h_conv4_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv4_maxpool


# fully connected layer 1
def fc1(input_data):
    print('fc1', input_data)
    input_layer_size = 13 * 13 * 128
    fc1_layer_size = 512

    with tf.name_scope('fc_1'):
        # 앞에서 입력받은 다차원 텐서를 fcc에 넣기 위해서 1차원으로 피는 작업
        input_data_reshape = tf.reshape(input_data, [-1, 13 * 13 * 128])
        W_fc1 = tf.Variable(tf.truncated_normal([13 * 13 * 128, 512], stddev=0.1))
        b_fc1 = tf.Variable(tf.truncated_normal([512], stddev=0.1))
        h_fc1 = tf.add(tf.matmul(input_data_reshape, W_fc1), b_fc1)  # h_fc1 = input_data*W_fc1 + b_fc1
        h_fc1_relu = tf.nn.relu(h_fc1)

    return h_fc1_relu


# fully connected layer 2
def fc2(input_data):
    fc2_layer_size = 256

    with tf.name_scope('fc_2'):
        W_fc2 = tf.Variable(tf.truncated_normal([512, 256], stddev=0.1))
        b_fc2 = tf.Variable(tf.truncated_normal([256], stddev=0.1))
        h_fc2 = tf.add(tf.matmul(input_data, W_fc2), b_fc2)  # h_fc1 = input_data*W_fc1 + b_fc1
        h_fc2_relu = tf.nn.relu(h_fc2)

    return h_fc2_relu


# final layer
def final_out(input_data):
    with tf.name_scope('final_out'):
        W_fo = tf.Variable(tf.truncated_normal([256, 3], stddev=0.1))
        b_fo = tf.Variable(tf.truncated_normal([3], stddev=0.1))
        h_fo = tf.add(tf.matmul(input_data, W_fo), b_fo)  # h_fc1 = input_data*W_fc1 + b_fc1

    # 최종 레이어에 softmax 함수는 적용하지 않았다.

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

    # output shape will be (*,6,6,128)
    r_cnn4 = conv4(r_cnn3)  # convolutional layer 4
    print("shape after cnn4 :", r_cnn4.get_shape())

    # fully connected layer 1
    r_fc1 = fc1(r_cnn4)
    print("shape after fc1 :", r_fc1.get_shape())

    # fully connected layer2
    r_fc2 = fc2(r_fc1)
    print("shape after fc2 :", r_fc2.get_shape())

    ## drop out
    # 참고 http://stackoverflow.com/questions/34597316/why-input-is-scaled-in-tf-nn-dropout-in-tensorflow
    # 트레이닝시에는 keep_prob < 1.0 , Test 시에는 1.0으로 한다.
    r_dropout = tf.nn.dropout(r_fc2, keep_prob)
    print("shape after dropout :", r_dropout.get_shape())

    # final layer
    r_out = final_out(r_dropout)
    print("shape after final layer :", r_out.get_shape())

    return r_out

images = tf.placeholder(tf.float32, [None, 200, 200, 1])
labels = tf.placeholder(tf.int32, [None, 3])
iterator = dataset.make_initializable_iterator()
image_stacked, label_stacked = iterator.get_next()
next_element = iterator.get_next()

keep_prob = tf.placeholder(tf.float32)  # dropout ratio
prediction = build_model(images, keep_prob)
# define loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

tf.summary.scalar('loss', loss)

# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# for validation
# with tf.name_scope("prediction"):

#validate_image_batch, validate_label_batch = read_data_batch(VALIDATION_FILE)
label_max = tf.argmax(labels, 1)
pre_max = tf.argmax(prediction, 1)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy', accuracy)

startTime = datetime.now()

# build the summary tensor based on the tF collection of Summaries
summary = tf.summary.merge_all()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    saver = tf.train.Saver()  # create saver to store training model into file
    #summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    sess.run(iterator.initializer)
    init_op = tf.global_variables_initializer()  # use this for tensorflow 0.12rc0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init_op)

    for i in range(1000):
        images_, labels_ = sess.run([image_stacked, label_stacked])
        # sess.run(train_step,feed_dict={images:images_,labels:labels_,keep_prob:0.8})
        sess.run(train, feed_dict={images: images_, labels: labels_, keep_prob: 0.7})

        if i % 10 == 0:
            now = datetime.now() - startTime
            print('## time:', now, ' steps:', i)

            # print out training status
            rt = sess.run([label_max, pre_max, loss, accuracy], feed_dict={images: images_
                , labels: labels_
                , keep_prob: 1.0})
            print('Prediction loss:', rt[2], ' accuracy:', rt[3])
            # validation steps
            rv = sess.run([label_max, pre_max, loss, accuracy], feed_dict={images: images_
                , labels: labels_
                , keep_prob: 1.0})
            print('Validation loss:', rv[2], ' accuracy:', rv[3])

            # validation accuracy
            summary_str = sess.run(summary, feed_dict={images: images_
                , labels: labels_
                , keep_prob: 1.0})
            #summary_writer.add_summary(summary_str, i)
            #summary_writer.flush()

    saver.save(sess, 'face_recog')  # save session
    coord.request_stop()
    coord.join(threads)
    print('finish')



