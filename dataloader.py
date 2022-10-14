import tensorflow as tf
import numpy as np
import os
import cv2



def load_list(list_path='./train.txt', root_path='./train/all/'):
    images = []
    labels = []
    outimg = []
    outlabel = []
    with open(list_path, 'r') as f:
        for line in f:
            label, name = line.split('\t')
            label = int(label)
            images.append(root_path + name[:-1])
            labels.append(label)
    f.close()
    for i in range(len(images)):
        if i %10000 ==0:
            print(i)
        data = cv2.imread(images[i], 0)
        outimg.append(data)
        label_one_hot = np.zeros(6)
        label_one_hot[labels[i]] = 1.0
        outlabel.append(label_one_hot)
    outimg = np.array(outimg)
    outlabel = np.array(outlabel)
    return outimg, outlabel


def load_list2(list_path='./test.txt',root_path='./train/all/'):
# def load_list2(list_path='./testorg.txt', root_path='./test/org/'):
    images = []
    labels = []
    outimg = []
    outlabel = []
    with open(list_path, 'r') as f:
        for line in f:
            label, name = line.split('\t')
            label = int(label)
            images.append(root_path + name[:-1])
            labels.append(label)
    f.close()
    for i in range(len(images)):
        if i %10000 ==0:
            print(i)
        data = cv2.imread(images[i], 0)
        outimg.append(data)
        label_one_hot = np.zeros(6)
        label_one_hot[labels[i]] = 1.0
        outlabel.append(label_one_hot)
    outimg = np.array(outimg)
    outlabel = np.array(outlabel)
    return outimg, outlabel


def load_image(image_path, label):

    # print(image_path.numpy().decode())
    image = image_path.numpy()
    image = image.reshape((40,9,1))/255

    return image, label




def train_iterator():
    images, labels = load_list()
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(len(images))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y], Tout=[tf.float32, tf.float64]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)#.cache('/mnt/ssd0.5T/train/face68.TF-data')
    dataset = dataset.repeat()
    dataset = dataset.batch(32).prefetch(1)
    it = dataset.__iter__()
    return it

def test_iterator():
    images, labels = load_list2()
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y], Tout=[tf.float32, tf.float64]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    dataset = dataset.repeat()
    dataset = dataset.batch(1).prefetch(1)
    it = dataset.__iter__()
    return it


if __name__ == '__main__':
    it = train_iterator()
    images, labels = it.next()
    print(labels[0])

    # wav,lable = load_list()
    # print(lable)




