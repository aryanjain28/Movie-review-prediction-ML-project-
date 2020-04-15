# from warnings import filterwarnings
# filterwarnings('ignore')

# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tensorflow import keras


# data = keras.datasets.imdb
# (xTrain, yTrain), (xTest, yTest) = data.load_data(num_words=88000, path='imdb.npz')

# wordIndex = data.get_word_index()
# wordIndex = {k:v+3 for k,v in wordIndex.items()}
# wordIndex['<PAD>'] = 0
# wordIndex['<START>'] = 1
# wordIndex['<UNL>'] = 2
# wordIndex['<UNUSED>'] = 3

# reverseWordIndex = {v:k for k,v in wordIndex.items()}
# print(str(dict(list(reverseWordIndex.items())[-5:])))

# xTrain = keras.preprocessing.sequence.pad_sequences(xTrain, maxlen=250, padding='post', value=wordIndex['<PAD>'])
# xTest = keras.preprocessing.sequence.pad_sequences(xTest, maxlen=250, padding='post', value=wordIndex['<PAD>'])

# def getReadableText(index):
#     return " ".join([reverseWordIndex.get(i, '?') for i in index])

# #model
# # model = keras.models.Sequential()
# # model.add(keras.layers.Embedding(88000, 16))
# # model.add(keras.layers.GlobalAveragePooling1D())
# # model.add(keras.layers.Dense(16, activation='relu'))
# # model.add(keras.layers.Dense(1, activation='sigmoid'))

# # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # xValidate = xTrain[:10000]
# # yValidate = yTrain[:10000]

# # xTrain = xTrain[10000:]
# # yTrain = yTrain[10000:]

# # fitModel = model.fit(xTrain, yTrain, batch_size=256, epochs=20, verbose=1, validation_data=(xValidate, yValidate))
# # model.save('./MovieReviewPrediction/movieReviewModel.h5')



# # best = 0
# # for _ in range(10):
# #     fitModel = model.fit(xTrain, yTrain, batch_size=256, epochs=40, verbose=1, validation_data=(xValidate, yValidate))
# #     loss, accuracy = model.evaluate(xTest, yTest)
# #     if accuracy > best:
# #         print('Accuracy : ', accuracy)
# #         model.save('./movieReviewModel.h5')
# #         best = accuracy


# model = keras.models.load_model('/home/aryan/Desktop/Machine Learning (Coursera)/Python Practice/MovieReviewPrediction/myMovieModel.h5')


# loss, accuracy = model.evaluate(xTest, yTest)
# print('Model accuracy : ', accuracy)


# def getIndex(words):
#     index = [1]
#     for word in words:
#         if word.lower() in wordIndex:
#             index.append(wordIndex[word.lower()])
#     return index

# with open('/home/aryan/Desktop/Machine Learning (Coursera)/Python Practice/MovieReviewPrediction/test.txt') as f:
#     for line in f.readlines():
#         LINE = line.replace(':', '').replace(';', '').replace('.', '').replace('(', '').replace(')', '').replace('"', '').replace(',', '').strip().split(' ')
#         index = getIndex(LINE)
#         index = keras.preprocessing.sequence.pad_sequences([index], value=wordIndex['<PAD>'], padding='post', maxlen=250)
#         print(index)
#         prediction = model.predict(index)
#         # print(getReadableText(index[0]))
#         print('Hmm.. Looks like the review has a ', *prediction[0]*100 ,' chance that it is a good review.')
#         if prediction[0][0]*100 > 50.0:
#             print('Answer : +ve REVIEW')
#         else:
#             print('Answer : -ve REVIEW')

# prediction = model.predict([xTest[0]])
# print('Prediction : ', str(prediction[0]))
# print('Actual : ', str(yTest[0]))
# print(getReadableText(xTest[0]))