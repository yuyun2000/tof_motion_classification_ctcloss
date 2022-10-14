文件说明：
-
- ctcdataloader,ctcmodel,ctctrain,ctctest为采用了ctcloss的dataset，模型，训练和测试脚本
- dataloader,model,train为直接分类的模型的相关文件，目前没有用作为备份
- npy2img,jpgrename先把npy按照长度和步长进行截取保存为图片，然后修改图片名称作为其标签（注意每次只做相同的动作，这样打标签时只数数就可以了）
- writelist是把训练集中的文件都写入txt以在dataset中读取
- utils里面定义了l2loss和正确率计算（普通分类用，ctcloss没写正确率计算的脚本）
- 文件夹：
  - h5保存模型
  - npy保存npy
  - train/1 2 3为坐姿采集的样本的图片
  - train/h1 h2 h3 为站姿采集的样本图片
  - train/ctc 为训练集样本就是把前面的都拉进去
  - train/temp 暂时将npy的图片放进来，再读取修改名称
  - train/file.py 统计训练集的一些代码

模型说明：
-
- 前半部分为卷积，经过GAP和dropout后接全连接进行分类，输入为(64,9,1) 输出为(16,4)大概就是原图每小格进行一次四分类
- 训练的时候不加softmax，之前训练加softmax不收敛，但是推理时可以加上，方便阈值判断
- dataset结构为先读取所有的图片入内存，然后通过fom_tensor_slices创建dataset，之后用map方法对每个样本进行单独的处理（归一化、增强等）
- 训练时在计算ctcloss前把标签转换为稀疏张量，训练会快很多倍，dataset里长度不够的用0补齐，计算loss时再用tf.spares.from_dense转化为稀疏变量
