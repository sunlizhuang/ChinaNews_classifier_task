# ChinaNews_classifier_task
 using RNN LSTM BERT to classify postive or negtive by news about china from global website（datasets check another repo）
 
 RNN LSTM base on Keras， 
 BERT base on Tensorflow（jupyter,colab）
 
 分类数据集的是涉华的外网舆情，主要新闻来源为世界上各个国家关于中国的报道
 
 ### Results
 | Type      |    acc| 
 | :-------- | --------:| 
 | RNN | 0.836341666|
 | LSTM| 0.898583333| 
 | BERT | 0.92264503| 

RNN和LSTM下载Keras直接就能跑
BERT那个要下载后，上传至谷歌colab，在上面选择实时运行方式改为Python3+TPU，再点全部运行，colab要分配30GB RAM才能
跑，不足可能会爆掉，重新分配就好了
