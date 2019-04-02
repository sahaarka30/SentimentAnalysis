import pickle
import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def ingest(path,columnsdrop,col1,col2):
    # data = pd.read_csv('./tweets.csv')
    data = pd.read_csv(path)
    data.drop(columnsdrop, axis=1, inplace=True)
    # data.drop(['ItemID','Date','Blank','SentimentSource'], axis=1, inplace=True)
    data = data[data.Sentiment.isnull() == False]
    # data['Sentiment'] = data['Sentiment'].map(int)
    data[col1] = data[col1].map({4: 1, 0: 0})
    data = data[data[col2].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print 'dataset loaded with shape', data.shape
    return data

data = ingest('./tweets.csv',['ItemID','Date','Blank','SentimentSource'],'Sentiment','SentimentText')
data1=ingest('./Negative.csv',['Source'],'Sentiment','SentimentText')
data2=ingest('./Positive.csv',['Source'],'Sentiment','SentimentText')
data=data.append(data1,ignore_index=True)
data=data.append(data2,ignore_index=True)
data.head(5)
print(data.shape)

def tokenize(tweet):
    try:
        tweet = unicode(tweet.decode('utf-8').lower())
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    except:
        return 'NC'


def postprocess(data, n=1000000):
    data = data.head(n)
    data['tokens'] = data['SentimentText'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(data)

with open('data.pkl','wb') as f:
    pickle.dump(data,f)

n=1000000
x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens),
                                                    np.array(data.head(n).Sentiment), test_size=0.2)

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

n_dim=200
tweet_w2v = Word2Vec(size=n_dim, min_count=5)
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
# tweet_w2v.train([x.words for x in tqdm(x_train)])
tweet_w2v.train([x.words for x in tqdm(x_train)],total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

tweet_w2v.save('tweet_w2v')

print 'building tf-idf matrix ...'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=5)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print 'vocab size :', len(tfidf)

with open('tfidf.pkl','wb') as f:
    pickle.dump(tfidf,f)

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)

with open('test_vecs_w2v.pkl','wb') as f:
    pickle.dump(test_vecs_w2v,f)

from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout,Flatten
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, epochs=10, batch_size=32, verbose=2)

score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print score[1]



model_json = model.to_json()
with open("model_sentiment_w2v.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_sentiment_w2v.h5")
print("Saved model to disk")

