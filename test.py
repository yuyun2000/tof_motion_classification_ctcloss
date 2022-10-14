import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from dataloader import test_iterator
from utils import  correct_num_batch, l2_loss
from model import model
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@tf.function
def test_step(model, images, labels):
    prediction = model(images, training=False)
    cross_entropy = tf.keras.losses.categorical_crossentropy(labels, prediction)

    cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy, prediction

def test(model, log_file):
    data_iterator = test_iterator()

    sum_ce = 0
    sum_correct_num = 0

    for i in tqdm(range(int(156/1))):
        images, labels = data_iterator.next()
        ce, prediction = test_step(model, images, labels)
        correct_num = correct_num_batch(labels, prediction)

        sum_ce += ce * 1
        sum_correct_num += correct_num
        # print('ce: {:.4f}, accuracy: {:.4f}'.format(ce, correct_num / 1))

    log_file.write('test: cross entropy loss: {:.4f}, l2 loss: {:.4f}, accuracy: {:.4f}\n'.format(sum_ce / 156,
                                                                                                  l2_loss(model),
                                                                                                  sum_correct_num / 156))

if __name__ == '__main__':
    # gpu config
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # get model

    # model = kwsmodel(dim0=4)  # 随即杀死神经元的概率
    # model.build(input_shape=(None,) + (256,40,1))
    # model.load_weights('./h5/kws0930.h5')

    model = tf.keras.models.load_model('./h5/tof-7x3-190.h5')
    # model = tf.keras.models.load_model('./h5/kws-l2-mixup-14.h5')
    # model.summary()

    testin = cv2.imread('./train/ctc/1-1-352-h3.jpg',0)
    testin = np.array(testin).reshape((1,64,9,1))/255
    out = model(testin)
    out = np.array(out).reshape((16,4))
    print(out)
    print(np.argmax(out,axis=-1))
    print(out[6][1])

    # test
    # with open('test_log.txt', 'a') as f:
    #     test(model, f)
