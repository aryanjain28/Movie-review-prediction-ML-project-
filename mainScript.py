import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, render_template, url_for, redirect

data = None
xTrain = None
yTrain = None
xTest = None
yTest = None
wordIndex = {}
reversedWordIndex = {}
model = None

def loadData():
    global xTrain, data, yTrain, xTest, yTest, wordIndex, reversedWordIndex

    data = keras.datasets.imdb
    (xTrain, yTrain), (xTest, yTest) = data.load_data(path='imbd.npz', num_words=88000)

    wordIndex = data.get_word_index()
    wordIndex = {k:v+3 for k,v in wordIndex.items() }
    wordIndex['<PAD>'] = 0
    wordIndex['<START>'] = 1
    wordIndex['<UNK>'] = 2
    wordIndex['<UNUSED>'] = 3

    xTrain = keras.preprocessing.sequence.pad_sequences(xTrain, maxlen=250, padding='post', value=wordIndex['<PAD>'])
    xTest = keras.preprocessing.sequence.pad_sequences(xTest, maxlen=250, padding='post', value=wordIndex['<PAD>'])

    reversedWordIndex = {v:k for k,v in wordIndex.items() }

def getText(index):
    global reversedWordIndex
    return ' '.join([reversedWordIndex[i] for i in index])

def trainModel():
    #model
    global model, xTrain, yTrain, xTest, yTest
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(88000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    xValidate = xTrain[:10000]
    yValidate = yTrain[:10000]

    xTrain = xTrain[10000:]
    yTrain = yTrain[10000:]

    model.fit(xTrain, yTrain, batch_size=512, epochs=40, verbose=1, validation_data=(xValidate, yValidate), shuffle=True)

    loss, accuracy = model.evaluate(xTest, yTest)
    print('Accuracy : ', accuracy)

    model.save('./MovieReviewPrediction/myMovieModel.h5')

def loadModel():
    global model, xTest, yTest
    model = keras.models.load_model('/home/aryan/Desktop/Machine Learning (Coursera)/Python Practice/MovieReviewPrediction/myMovieModel.h5')
    loss, accuracy = model.evaluate(xTest, yTest)
    print('Accuracy : ', accuracy)

def test():
    with open('/home/aryan/Desktop/Machine Learning (Coursera)/Python Practice/MovieReviewPrediction/test.txt') as f:
        for line in f.readlines():
            print(line)

def getResults(review):
    global model
    prediction = model.predict([review])
    return prediction[0][0]

def getIndex(r):

    # words = []
    # with open('/home/aryan/Desktop/Machine Learning (Coursera)/Python Practice/MovieReviewPrediction/test.txt') as f:
    #     for line in f.readlines():
    #         words = line.replace(':', '').replace(';', '').replace('(', '').replace(')', '').replace(',', '').replace('.', '').replace('?', '').replace('"', '').replace("'", '').strip().split()

    words = []
    words = r.replace(':', '').replace(';', '').replace('(', '').replace(')', '').replace(',', '').replace('.', '').replace('?', '').replace('"', '').replace("'", '').strip().split()

    global wordIndex
    index = [1]
    for w in words:
        if w.lower() in wordIndex:
            index.append(wordIndex[w.lower()])

    index = keras.preprocessing.sequence.pad_sequences([index], maxlen=250, padding='post', value=wordIndex['<PAD>'])
    return index


app = Flask(__name__)


@app.route('/getResults', methods=['GET', 'POST'])
def review():
    if request.method == 'POST':
        review = request.form['review']
        return redirect(url_for('sendResults', r=review))
    else:
        return render_template('enterReview.html')

@app.route('/<r>')
def sendResults(r):
    loadData()
    loadModel()
    res = getResults(getIndex(r))
    print(res)

    if res*100 > 50.0:
        s2 = "~Positive review~"
    else:
        s2 = "~Negative review~"

    s1 = f"<body  style='background-color:#3b5998;'><div style='margin-top:15%; font-size: 20px; text-shadow: black 10px 10px 50px; color: blanchedalmond;' ><h1><center>{s2}<br/><br/>According to our machine's prediction, it looks like this review has a<br/><br/> <i>{res*100}%</i> chance of being +ve.</center></h1></div></body>"
    return s1

if __name__ == '__main__':
    app.run(debug=True)
