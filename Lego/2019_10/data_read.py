
import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image

class data_read:
    def __init__(self, batch_size):
        self.data_size = len(self.train_list)
        self.batch_size = batch_size

    def get_label_from_path(self, path):
        return path.split('\\')[-2]

    def read_image(self, path):
        self.image = np.array(Image.open(path).convert('L'))
        return self.image.reshape([self.image.shape[0], self.image.shape[1], 1])

    def onehot_encode_label(self, path, unique_label_names):
        self.onehot_label = unique_label_names == self.get_label_from_path(path)
        self.onehot_label = self.onehot_label.astype(np.uint8)
        return self.onehot_label

    def _read_py_function(self, path, label):
        self.image = self.read_image(path)
        self.label = np.array(label, dtype=np.uint8)
        return self.image.astype(np.int32), self.label

    def _random_function(self, image_decoded, label):
        image_decoded.set_shape([None, None, None])
        self.image = tf.image.resize_images(image_decoded, [200, 200])
        self.image = tf.image.random_flip_left_right(self.image)
        self.image = tf.image.random_flip_up_down(self.image)
        self.image = tf.image.random_brightness(self.image, max_delta=0.5)
        self.image = tf.image.random_contrast(self.image, lower=0.2, upper=2.0)
        #self.image = tf.image.random_crop(self.image, [100, 100, 1])
        return self.image, label

    def made_batch(self, link, size):
        self.label_name_list = []
        for path in link:
            self.label_name_list.append(self.get_label_from_path(path))

        self.unique_label_names = np.unique(self.label_name_list)

        self.label_list = [self.onehot_encode_label(path, self.unique_label_names).tolist() for path in link]

        self.dataset = tf.data.Dataset.from_tensor_slices((link, self.label_list))
        self.dataset = self.dataset.map(
            lambda data_list, label_list: tuple(
                tf.py_func(self._read_py_function, [data_list, label_list], [tf.int32, tf.uint8])))

        self.dataset = self.dataset.map(self._random_function)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.shuffle(buffer_size=(int(len(link) * 0.4) + 3 * self.batch_size))
        self.dataset = self.dataset.batch(size)

        return self.dataset

if __name__ == '__main__':
  made_batch(
  train_list = glob('E:\\LEGO\\train\\*\\*.png'), 2)
