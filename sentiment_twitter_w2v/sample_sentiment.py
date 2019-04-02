from keras.models import model_from_json
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec # the word2vec
from sklearn.model_selection import train_test_split
tokenizer = TweetTokenizer()

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


with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

n=1000000
x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens),
                                                    np.array(data.head(n).Sentiment), test_size=0.2)
n_dim=200

# tweet_w2v = Word2Vec(size=n_dim, min_count=10)
# tweet_w2v.build_vocab([x for x in tqdm(x_train)])
# tweet_w2v.train([x for x in x_train],total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

tweet_w2v = Word2Vec.load('tweet_w2v')

print 'building tf-idf matrix ...'

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

print 'vocab size :', len(tfidf)

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


json_file = open('model_sentiment_w2v.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_sentiment_w2v.h5")
print("Loaded model from disk")
with open('test_vecs_w2v.pkl', 'rb') as f:
    test_vecs_w2v = pickle.load(f)

# score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
# print score[1]

max_review_length = 200

word_to_id = imdb.get_word_index()
word_to_id = {k:v for k,v in word_to_id.items()}

# sent = ["This movie sounds good"]
# sent_indx = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x, sent))])
# prediction = model.predict(sent_indx)
while(1):
    sent=raw_input("Enter sentence : ")
    sent=[str(sent)]
    sent_indx = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x, sent))])
    prediction = model.predict(sent_indx)
    print "prediction: ", sent," : ",prediction


# sent_list = ["This movie does not sound good", "This movie sounds good"]
# for sent in sent_list:
#     sent = [sent]
#     sent_indx = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x, sent))])
#     prediction = model.predict(sent_indx)
#
#     print "prediction: ", prediction
# sent=["This program feels absolutely great"]
# sent_indx = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x, sent))])
# prediction = model.predict(sent_indx)
# print "prediction: ", sent," : ",prediction
