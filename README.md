# CaffeToTensorRT_PVA_interface

本机环境 windows7 cuda10.1 cudnn7.5.0 cublas10.2.0

faster RCNN 系列 之 PVA 的前向接口

有些代码不想整理了有点乱，备注一下几点：

（1）里面涉及了2中图像预处理方式，使用opencv库进行减均值，channel转换速度较慢，反而直接用for循环会更快些

（2）后续会做一个版本，直接将opencv读取的图片传到GPU中，并增加2个层来进行需处理操作：

--<1>使用TensorRT的Permute层 把 n h w c  结构  转化为 nchw
--<2>使用TensorRT的scale层对每个channel上的值减均值

(3)备注个偶尔会出现的BUG，即handle初始化会出现空的情况

有问题请指正
