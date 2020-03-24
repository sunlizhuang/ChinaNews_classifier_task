import urllib.request
import os
import tarfile
os.environ['KERAS_BACKEND']='tensorflow'
#coding=utf8
# -*- coding: utf-8 -*-

#下载数据集，创建data/chinaNews文件夹
# url="https://raw.githubusercontent.com/sunlizhuang/GDELT_chinaNews/master/chinaNews.zip"
# filepath="data/chinaNews.zip"
# if not os.path.isfile(filepath):
#     result=urllib.request.urlretrieve(url,filepath)
#     print('downloaded:',result)
#
# if not os.path.exists("data/chinaNews.zip"):
#     tfile = tarfile.open("data/chinaNews.zip", 'r:zip')
#     result=tfile.extractall('data/')
#


import os
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import re
re_tag = re.compile(r'<[^>]+>')
def rm_tags(text):
    return re_tag.sub('', text)


import os
def read_files(filetype):
    path = "data/chinaNews/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print('read', filetype, 'files:', len(file_list))

    all_labels = ([1] * 6000 + [0] * 6000)

    all_texts = []

    for fi in file_list:
        with open(fi,encoding='utf-8',errors='ignore') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
    return all_labels, all_texts

# 读文件
y_train,train_text=read_files("train")
y_test,test_text=read_files("test")
token = Tokenizer(num_words=4000)
token.fit_on_texts(train_text)
print(token.word_index)
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
print(train_text[0])
print(x_train_seq[0])
x_train=sequence.pad_sequences(x_train_seq,maxlen=400)
x_test= sequence.pad_sequences(x_test_seq,maxlen=400)

print('before pad_sequences length=',len(x_train_seq[0]))
print(x_train_seq[0])

print('after pad_sequences length=',len(x_train[0]))
print(x_train[0])

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN,LSTM
import tensorflow
from tensorflow import keras
model = Sequential()
model.add(Embedding(output_dim=32,
                    input_dim=4000,
                    input_length=400))
model.add(Dropout(0.25))
model.add(SimpleRNN(units=16)) 
model.add(Dense(units=256,activation='relu' ))
model.add(Dropout(0.25))
model.add(Dense(units=1,activation='sigmoid' ))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x_train,y_train,batch_size=100,epochs=10,verbose=2,validation_split=0.2)


scores= model.evaluate(x_test,y_test,verbose=1)
print(scores[1])

predict=model.predict_classes(x_test)
predict_classes = predict.reshape(-1)
print(predict_classes)

SentimentDict={1:'pos',0:'neg'}
def display_test_Sentiment(i):
   # print(test_text[i])
    print('label：',SentimentDict[y_test[i]],'predict_result:',SentimentDict[predict_classes[i]])

for i in range(0,20):
    print(display_test_Sentiment(i))


def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_inpt_seq = sequence.pad_sequences(input_seq, maxlen=400)
    predict_result = model.predict_classes(pad_inpt_seq)
    print('predict_result:'+SentimentDict[predict_result[0][0]])


predict_review('''MANILA, Philippines—All foreign residents in the country departing for vacation ahead of the Chinese New Year are advised to process reentry fees before leaving for abroad, the Bureau of Immigration (BI) said.
    
    BI Port Operations Division Chief Grifton Medina issued the appeal to reduce the volume of travelers lining up to pay their reentry fees at Immigration cashiers of the Ninoy Aquino International Airport (NAIA).
    
    “We are expecting a surge of Chinese residents in the Philippines who wish to spend the Chinese New Year abroad,” said Medina.
    
    “It may result in heavy congestion of our airports, which could be avoided if they process and pay their fees before heading to the airport,” he added.
    
    Medina said aliens may proceed to BI satellite offices to pay before they proceed to the airport.
    
    This would also lessen worries of not catching their flights as they could directly proceed to visa counters.
    
    The Philippine Immigration law requires foreign nationals who are holders of valid immigrant and non-immigrant visas to pay exit and re-entry permits every time they leave the country.
    
    Official receipt of payment must be presented at Immigration counters before they can be cleared for departure.
    
    “The BI has almost 60 offices nationwide that may cater to this need. It’s a very quick process, which will only take a few minutes,” Medina explained.
    
    “We also have offices located in malls and other convenient locations,” he said. “Coming to the airport with the receipt at hand makes processing faster, allowing departing aliens to avoid the rush and relax before their flight,” he added.
    
    The Philippines is home to thousands of Chinese immigrants and non-immigrants.
    
    According to Chinese traditions, they should at least visit their homeland for the celebration of Chinese New Year, which is on January 25.
    
    The post Aliens in PH urged to pre-pay reentry fees at BI offices appeared first on UNTV News.''')
