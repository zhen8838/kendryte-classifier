Tensorflow Workspace for K210
======

## 介绍

此仓库修改了官方`demo`部分代码,增加了更多的操作空间,并且可以载入冻结的`bp`文件进行继续训练.

## Kendryte Tensorflow 分类器示例
1. 下载分类数据集，解压之后将其放置为一个文件夹包含一类图像数据.文件夹名与`mobilenetv1/data/label.txt`一一对应,`mobilenetv1/data/names.list`为类别的别名,也一一对应.

2. 使用`mobilenetv1/train.sh`进行训练,具体参数见脚本文件


3. 使用`mobilenetv1/freeze.sh`冻结图生成`pb`文件,具体参数见脚本文件


4. 使用`mobilenetv1/predict.sh`进行预测单张图像,具体参数见脚本文件,预测时候将所有需要预测的图像放置到一个文件夹中,具体参数见脚本文件
   
## Reference

[勘智官方demo](https://github.com/kendryte/tensorflow-workspace)