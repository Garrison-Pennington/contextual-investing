import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as hub
from official.nlp import optimization
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import TWITTER_DIR, MODEL_DIR

SENT_BERT_URL = 'https://tfhub.dev/google/experts/bert/wiki_books/sst2/2'
PREP_BERT_URL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
SHUFFLE_SEED = 20210429


def load_tweets():
    return pd.read_csv(
        TWITTER_DIR.joinpath('tweets_with_sentiments.csv'),
        encoding="ISO-8859-1",
        names=['sentiment', 'id', 'date', 'query', 'username', 'content']
    )


def shuffle_and_split(df):
    return train_test_split(df.sample(frac=1, random_state=SHUFFLE_SEED), test_size=0.2)


def split_by_class(df):
    pos = df[df['sentiment'] != 0]
    neg = df[df['sentiment'] == 0]
    return pos, neg


def save_tweets_to_directory(df, base_dir):
    for _, row in tqdm(list(df.iterrows()), desc=f'saving tweets to {str(base_dir)}'):
        tweet_id, content = row['id'], row['content']
        with open(base_dir.joinpath(f'{tweet_id}.txt'), 'w+') as f:
            f.write(content)


def create_directory_datasets():
    tweets = load_tweets()
    train, test = shuffle_and_split(tweets)
    train_pos, train_neg = split_by_class(train)
    test_pos, test_neg = split_by_class(test)
    save_tweets_to_directory(train_pos, TWITTER_DIR.joinpath('train', 'positive'))
    save_tweets_to_directory(train_neg, TWITTER_DIR.joinpath('train', 'negative'))
    save_tweets_to_directory(test_pos, TWITTER_DIR.joinpath('test', 'positive'))
    save_tweets_to_directory(test_neg, TWITTER_DIR.joinpath('test', 'negative'))


def load_bert():
    prep = hub.KerasLayer(PREP_BERT_URL)
    bert = hub.KerasLayer(SENT_BERT_URL)
    return prep, bert


AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    TWITTER_DIR.joinpath('train'),
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    TWITTER_DIR.joinpath('train'),
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    TWITTER_DIR.joinpath('test'),
    batch_size=batch_size)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

for text_batch, label_batch in train_ds.take(1):
    for i in range(3):
        print(f'Tweet: {text_batch.numpy()[i]}')
        label = label_batch.numpy()[i]
        print(f'Label : {label} ({class_names[label]})')

bert_preprocess_model, bert_model = load_bert()

text_test = ['this is such an amazing movie!']
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {SENT_BERT_URL}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')


def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(PREP_BERT_URL, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(SENT_BERT_URL, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)


classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

print(f'Training model with {SENT_BERT_URL}')
history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs)

loss, accuracy = classifier_model.evaluate(test_ds)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

dataset_name = 'tweets'
saved_model_path = str(MODEL_DIR.joinpath('/{}_bert'.format(dataset_name.replace('/', '_'))))

classifier_model.save(saved_model_path, include_optimizer=False)
