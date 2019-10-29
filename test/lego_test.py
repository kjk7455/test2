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
num_epochs = 1000

def next_batch(num):
    TRAIN_DIR = 'E:/LEGO/train/'
    train_folder_list = array(os.listdir(TRAIN_DIR))

    train_input = []
    train_label = []

    label_encoder = LabelEncoder()  # LabelEncoder Class 호출
    integer_encoded = label_encoder.fit_transform(train_folder_list)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    for i in range(num):
        index = np.random.randint(0, 3)
        path = os.path.join(TRAIN_DIR, train_folder_list[index])
        path = path + '/'
        img_list = os.listdir(path)

        img_path = os.path.join(path, img_list[np.random.randint(len(img_list))-1])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = tf.image.decode_png(tf.io.read_file(img_path), channels=1)
        #print(i, img_path)
        print(tf.shape(tf.io.read_file(img_path)))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image,max_delta=0.5)
        image = tf.image.random_contrast(image,lower=0.2,upper=2.0)
        image = tf.image.random_hue(image,max_delta=0.08)
        image = tf.image.random_saturation(image,lower=0.2,upper=2.0)
        image = tf.reshape(image, [-1, 200, 200, 1])
        print(image)
        train_input.append(img)
        train_label.append([np.array(onehot_encoded[index])])


    #train_input = np.reshape(train_input, (-1, 200, 200, 1))
    train_label = np.reshape(train_label, (-1, 3))

    return np.asarray(train_input), np.asarray(train_label)

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
Y = tf.placeholder(tf.int32, shape=[None, 3])
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

correct_prediction = tf.equal(tf.argmax(H, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

data_list = glob('E:\\LEGO\\train\\*\\*.png')
data_num = len(data_list)

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

import tensorflow as tf

def _read_py_function(path, label):
    image = read_image(path)
    label = np.array(label, dtype=np.uint8)

    return image.astype(np.int32), label

def _resize_function(image_decoded, label):

    image_decoded.set_shape([None, None, None])
    image = tf.image.resize_images(image_decoded, [200, 200])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, lower=0.2, upper=2.0)
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
iterator = dataset.make_initializable_iterator()
image_stacked, label_stacked = iterator.get_next()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())
    startTime = datetime.now()
    saver = tf.train.Saver()  # create saver to store training model into file
    summary_writer = tf.summary.FileWriter('E:/LEGO/', sess.graph)

    for step in range(num_epochs):
        avg_cost = 0
        for x in range(int(data_num/batch_size)):
            x_data, y_data = sess.run([image_stacked, label_stacked])
            cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})
            avg_cost += cost_val / int(data_num/batch_size)


        now = datetime.now() - startTime
        print('step: ', step, 'cost_val : ', avg_cost, 'time', now)

    '''prediction = tf.equal(tf.math.argmax(H, 1), tf.math.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    #print('accuracy : ', accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))'''
    saver.save(sess, 'E:/LEGO/lego2')  # save session