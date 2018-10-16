# Deep-image-analogy-TensorFlow
Implementation of the paper : Deep image analogy

# Introduction
This code mainly implements the interesting paper [Visual Attribute Transfer through Deep Image Analogy](http://cn.arxiv.org/pdf/1705.01088), details of the algorithm please see the paper. 
![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/suanfa.jpg)

# How to use 
------------------------------------------------------
Step 1. Download the vgg19.npy(BaiduDrive), and put it into the folder 'vgg19'

Step 2. Modify the image's path in Main.py

Step 3. excute Main.py
--------------------------------------------------------

# Python packages

===================

1. python 3.5
2. tensorflow1.4.0
3. numpy
4. scipy
5. pillow

===================
# Results
|Input|:|Output|::|Output|:|Input|
|-|-|-|-|-|-|-|
|![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/girl_A.jpg)|:|![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/girl_A_prime.jpg)|::|![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/girl_B.jpg)|:|![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/girl_B_prime.jpg)|
|![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/wlh.jpg)|:|![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/wlh_prime.jpg)|::|![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/catong_prime.jpg)|:|![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/catong.jpg)|
|![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/sumiao.jpg)|:|![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/lanse_prime.jpg)|::|![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/sumiao_prime.jpg)|:|![](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow/blob/master/IMAGES/lanse.jpg)|

#### Time: 3 minutes for 224x224 image size
These results are not satisfactory compare to paper's attractive results
# Acknowledgement
Thanks for the PatchMatch code from [harveyslash](https://github.com/harveyslash/PatchMatch)
