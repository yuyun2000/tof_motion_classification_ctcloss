import os
from copy import deepcopy
from random import randint
import cv2
def shuffle(lst):
  temp_lst = deepcopy(lst)
  m = len(temp_lst)
  while (m):
    m -= 1
    i = randint(0, m)
    temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
  return temp_lst


file = open('train.txt', mode='a+')
list = os.listdir('./train/ctc/')
list = shuffle(list)

num = 0
for i in range(len(list)):
    # img = cv2.imread('./train/%s'%(list[i]),0)
    label = list[i][0]
    # if label =='2' and num < 35000:
    #     file.write(label + '\t' + '%s\n' % (list[i]))
    #     num +=1
    # elif label != '2':
    file.write(label + '\t' + '%s\n' % (list[i]))

file.close()


