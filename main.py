import tensorflow as tf
import pandas as pd
import string
def clean_text(text):
    wordss=text.split()
    table = str.maketrans('','', string.punctuation)
    wordss=[word.translate(table) for word in wordss]
    ss=[word.lower() for word in wordss if word.isalpha()]
    return ss
data=pd.read_csv("spam.csv",encoding="latin")
data=data[["v1","v2"]]
y_train=pd.get_dummies(data.v1)
y_train=y_train.to_numpy()
print(y_train)
data=data.v2.to_numpy()
xtrain=[]
for i in data:
    xtrain.append(clean_text(i))
MAX=171
tok=tf.keras.preprocessing.text.Tokenizer(60000)
tok.fit_on_texts(xtrain)
xtrain_procesed=tok.texts_to_sequences(xtrain)
print(xtrain_procesed)
xtrain_procesed=tf.keras.utils.pad_sequences(xtrain_procesed,MAX)
xtrain_procesed=tf.constant(xtrain_procesed)
y_train=tf.constant(y_train)
print(xtrain_procesed.shape)
model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(60000,50,input_length=MAX))
model.add(tf.keras.layers.LSTM(128,return_sequences=True))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2,activation="sigmoid"))
model.compile("Adam",loss="CategoricalCrossentropy",metrics=["Accuracy"])
model.fit(xtrain_procesed,y_train,256,100)
