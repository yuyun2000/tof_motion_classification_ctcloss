import tensorflow as tf
import cv2
import librosa
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("./h5/tof-190.h5")

testin = cv2.imread('./train/test/4-1-1840-4.jpg',0)
testinf = testin.reshape((1,64,9,1))/255

out = model(testinf)

out = np.argmax(out,axis=-1).reshape(16)
print(out)
out = list(out)
for i in range(len(out)):
    if out[i]==1:
        out[i] = '右'
    elif out[i]==2:
        out[i] = '左'
    elif out[i]==3:
        out[i] = '摆'
    else:
        out[i] = ' '


from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

fig = plt.figure()
fig.patch.set_facecolor('red')
plt.imshow(testin.T,aspect=2)
plt.xticks(ticks=[0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60],labels=out,fontproperties=font,size=24)
plt.show()