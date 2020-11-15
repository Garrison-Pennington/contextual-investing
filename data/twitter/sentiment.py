import math
import time
from joblib import dump, load

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as hub
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tqdm import tqdm

from data import TWITTER_DIR

start = time.time()

print("Loading dataset")
data = shuffle(pd.read_csv(TWITTER_DIR.joinpath("tweets_with_sentiments.csv"), encoding='ISO-8859-1', header=None).set_index(2))
data[0] = data[0].mask(lambda v: v != 0, 1)

loaded = time.time()
print(loaded - start, "Splitting data set. Max sentiment label is", data[0].max())
train, test = data.values[:int(len(data)*.8)], data.values[int(len(data)*.8):]

train_sent, train_tweet = train[:, 0], train[:, 4]
test_sent, test_tweet = test[:, 0], test[:, 4]

prep = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')
bert = hub.load("https://tfhub.dev/google/experts/bert/wiki_books/sst2/2")

models_loaded = time.time()
print(models_loaded - loaded, "Encoding training tweets with BERT")
pooled = list()
seq = list()
for i in tqdm(range(math.ceil(len(train_tweet)/100))):
    b_start = time.time()
    batch = train_tweet[i*100:(i+1)*100]
    bert_inp = prep(batch)
    bert_out = bert(bert_inp)
    pooled.append(bert_out["pooled_output"])
    seq += [*bert_out['sequence_output']]
    print(f"batch #{i} took {time.time() - b_start}")

train_encoded = time.time()
print(train_encoded, "Standardizing for SKLearn")
scaler = StandardScaler()
x_std = scaler.fit_transform(seq)

standardized = time.time()
print(standardized - train_encoded, "Fitting logistic regression")
reg = LogisticRegression(solver='sag')
reg.fit(pooled, train_sent)

model_trained = time.time()
print(model_trained - standardized, "Saving model")
dump(reg, "/home/noisette/tweet_sent_v0.joblib")

print(train_encoded - models_loaded, "Pre-Processing test tweets for BERT")
bert_test = prep(test_tweet)

test_prepped = time.time()
print(test_prepped - train_encoded, "Encoding test tweets with BERT")
bert_test_out = bert(bert_test)
test_pooled = bert_test_out["pooled_output"]
test_seq = bert_test_out['sequence_output']

model_saved = time.time()
print(model_saved - model_trained, "Evaluating model")
pred = reg.predict(test_pooled)
label = test_sent
avg_diff = np.mean(np.abs(np.array(pred) - np.array(test_sent)))

model_evaluated = time.time()
print(model_evaluated - model_saved, "Average difference in prediction", avg_diff)

print("total time =", time.time() - start)

