import cv2

from glob import glob
from datetime import datetime
# PIL는 이미지를 load 할 때, numpy는 array
from PIL import Image
import numpy as np

import tensorflow as tf

train_list = glob('C:\\Users\\Aluminum\\Desktop\\lrgoooo\\*\\*.jpg')
print(len(train_list))
for x in range(len(train_list)):
    img = cv2.imread(train_list[x])
    img_name = train_list[x].split('\\')[-2]
    img_name += '_'+str(x)
    img_name += '.png'

    print(img_name)
    cv2.imwrite(img_name, img)