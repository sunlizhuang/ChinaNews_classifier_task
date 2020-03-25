# ChinaNews_classifier_task
 using RNN LSTM BERT to classify postive or negtive by news about china from global website（datasets check another repo）
 
 RNN LSTM base on Keras， 
 BERT base on Tensorflow（jupyter,colab）
 
 ### Results
 | Type      |    acc| 
 | :-------- | --------:| 
 | RNN | 0.81653231|
 | LSTM| 0.89008333| 
 | BERT | 0.92264503| 

RNN和LSTM下载Keras直接就能跑
BERT那个要下载后，上传至谷歌colab，在上面选择实时运行方式改为Python3+TPU，再点全部运行
