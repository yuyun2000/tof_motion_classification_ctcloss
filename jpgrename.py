'''
随机切割图片，需要把背景提取出来并把其标签改成0
ascii码 0-48 1-49 2-50 3-51 4-52 5-53
1右挥手2左挥手3摆手4连续右挥手5连续左挥手0无动作
'''
import os
import cv2
import numpy as np

filelist = os.listdir('./train/temp/')
print(filelist)
for i in filelist:
    img = cv2.imread('./train/temp/'+i,0)
    img = cv2.resize(img,(200,600))

    mri_max = np.amax(img)
    mri_min = np.amin(img)
    img = ((img - mri_min) / (mri_max - mri_min)) * 255
    img = img.astype('uint8')

    r, c= img.shape
    for k in range(1):
        temp = img[:, :]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(temp)

    img = img.T
    img2 = cv2.applyColorMap(img,cv2.COLORMAP_JET)
    cv2.putText(img2, '%s' % i[0], (20, 50), 1, 3, (0, 0, 255), 3)
    cv2.imshow('1',img2)
    x = cv2.waitKey(0)
    print(x)
    if x == 48:
        os.rename('./train/temp/' + i, './train/temp/0' + i[1:])
    if x == 49:
        os.rename('./train/temp/' + i, './train/temp/1' + i[1:])
    if x == 50:
        os.rename('./train/temp/' + i, './train/temp/2' + i[1:])
    if x == 51:
        os.rename('./train/temp/' + i, './train/temp/3' + i[1:])
    if x == 52:
        os.rename('./train/temp/' + i, './train/temp/4' + i[1:])
    if x == 53:
        os.rename('./train/temp/' + i, './train/temp/5' + i[1:])
    if x == 54:
        os.rename('./train/temp/' + i, './train/temp/6' + i[1:])
    if x == 55:
        os.rename('./train/temp/' + i, './train/temp/7' + i[1:])
    if x == 56:
        os.rename('./train/temp/' + i, './train/temp/8' + i[1:])
    if x == 27:
        os.rename('./train/temp/' + i, './train/temp/000' + i[1:])
    # label = input()
    #