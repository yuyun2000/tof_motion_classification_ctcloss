'''
读取生成的npy文件并生成jpg以及标签，其中：
1右挥手2左挥手3摆手4连续右挥手5连续左挥手0无动作
'''
import numpy as np
import cv2
data = np.load('./npy/12-2.npy')
print(data.shape)
oft = 0
while True:
    img = data[oft:oft+64]
    # img = cv2.resize(img, (img.shape[1] * 20, img.shape[0] * 10))
    # p = sum(sum(img))
    # cv2.putText(img,'%s'%p,(10,20),1,1,(0,0,255),1,1)
    # cv2.imshow('1', img)
    # cv2.waitKey(1000)
    cv2.imwrite('./train/temp/1-2-%s-h12.jpg'%oft,img)#第一个数字是次数第二个数字是动作第三个数字是同组不重名第四个数字是不同组不重名

    if oft+64 < data.shape[0]:
        oft += 16
    else:
        break