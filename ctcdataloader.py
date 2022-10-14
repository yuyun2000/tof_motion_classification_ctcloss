import tensorflow as tf
import numpy as np
import os
import cv2



def load_list(list_path='./train.txt', root_path='./train/ctc/'):
    images = []
    labels = []
    seqlen = []
    labelslen = []
    with open(list_path, 'r') as f:
        for line in f:
            times, name = line.split('\t')
            act = int(name[2])
            label = np.ones(int(times))*act
            label = np.pad(label,(0,16-len(label)))
            # print(name,label)
            seqlen.append(16)
            labelslen.append(int(times))
            labels.append(label)

            data = cv2.imread(root_path + name[:-1], 0)
            images.append(data)
    f.close()
    labels = np.array(labels)
    labels = tf.constant(labels, dtype=tf.int32)
    labelslen = tf.constant(labelslen, dtype=tf.int32)
    seqlen = tf.constant(seqlen, dtype=tf.int32)
    return images, labels,labelslen,seqlen


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


def load_image(image_path, label,labellen,seqlen):

    # print(image_path.numpy().decode())
    image = image_path.numpy()
    # for i in range(4):
    #     x = np.random.randint(0, 63)
    #     # y = np.random.randint(0, 8)
    #     xoft = np.random.randint(0, 63 - x)
    #     # yoft = np.random.randint(0, 9 - y)
    #     b = np.random.randint(20, 40)
    #     image[x:x + xoft, :] += b

    image = image.reshape((64,9,1))/255

    return image, label,labellen,seqlen




def train_iterator():
    images, labels,labelslen,seqlen  = load_list()
    dataset = tf.data.Dataset.from_tensor_slices((images, labels,labelslen,seqlen)).shuffle(len(images))
    dataset = dataset.map(lambda x, y,z1,z2: tf.py_function(load_image, inp=[x, y,z1,z2], Tout=[tf.float32, tf.int32, tf.int32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
    # it = train_iterator()
    # images, labels = it.next()
    # print(labels[0])
    #
    wav,lable,_,_ = load_list()
    # print(lable)




