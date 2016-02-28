# 人脸识别

使用李子青团队的webface人脸数据集，根据汤晓欧团队的DeepID网络，通过Caffe训练出模型参数，经过LFW二分类得到人脸识别准确率。

## 人脸数据集
李子青团队的webface
- 下载网址：http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html
- 10,575个人，494,414幅图像
- 需要申请，个人申请无效，需要学校部门的领导或者代表。

## 图像预处理
对包含人脸的图像进行人脸框识别，人脸对齐和人脸剪裁。

### 原理
- 人脸框识别
- 人脸对齐
- 人脸剪裁

### 实现
- 下载webface
- 根据预处理工具进行人脸框检测，人脸对齐。
    - 预处理工具是其他人写的，地址：https://github.com/RiweiChen/FaceTools
    - 根据香港中文大学提供的人脸框检测和人脸特征点检测的windows二进制程序实现。
        - http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm

## caffe训练 
根据DeepID的网络使用caffe训练得到模型参数。

### 原理
- 对原始数据分离训练集和测试集
- 转换为caffe可以处理的lmdb格式
- 根据设定的Net网络和Solver配置文件进行训练
- 得到训练的模型

### 实现

- 修改DeepID.py中demo(num)方法中的人脸对齐后的文件夹以及最后一行的中训练的人数（1-10575）。

- 执行以下代码
```
python DeepID.py
```

## 人脸识别检测
检验训练好的模型,得到LFW的人脸准确率。

### 原理
- 根据http://vis-www.cs.umass.edu/lfw/提供的数据集以及pairs.txt得到需要检验的6000对图像（3000对相同人脸，3000对不同人脸）
- 分别将对应的两个图像分别作为训练好的模型输入，得到两个160维的特征向量。6000对图像依次进行操作，共得到6000对160维特征向量。
- 计算对应的特征向量的余弦距离（或欧式距离等其他距离），6000对图像依次进行该操作，得到6000个余弦距离（或欧式距离等其他距离），通过选择阈值得到人脸识别的准确率。

### 实现

- 下载lfwcrop_color.zip
    - http://conradsanderson.id.au/lfwcrop/lfwcrop_color.zip
- 修改DeepIDTest.py中demo_test方法中caffepath,lfwpath等相关路径,以及最后一行的中训练的人数与所用模型的迭代次数。

- 在caffe_path（一般为～/caffe-master）下，执行以下代码
```
python DeepIDTest.py
```
## 参考论文

1. [deepID](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)
《Deep learning face representation from predicting 10,000 classes》

2. [deepID2](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)
《Deep Learning Face Representation by Joint Identification-Verification》
3. [deepID2+](http://www.ee.cuhk.edu.hk/~xgwang/papers/shaoKLWcvpr15.pdf)
《Deeply Learned Attributes for Crowded Scene Understanding", IEEE Conf. on Computer Vision and Pattern Recognition, June 2015 (Oral)》
4. [deepID3](http://arxiv.org/pdf/1502.00873.pdf)
《Face Recognition with Very Deep Neural Networks》
