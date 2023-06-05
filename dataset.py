from keras.utils import image_utils
import random


# 分离路径与标签
def prepare_image(file_path):
    X_train = []
    y_train = []

    with open(file_path) as f:
        context = f.readlines()
    random.shuffle(context)

    for str in context:
        str = str.strip('\n').split('\t')

        X_train.append('./data/' + str[0])
        y_train.append(str[1])

    return X_train, y_train


# 数据归一化
def preprocess_image(img):
    img = image_utils.load_img(img, target_size=(224, 224))
    img = image_utils.img_to_array(img)
    img = img / 255.0
    return img
