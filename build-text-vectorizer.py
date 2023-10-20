import tensorflow as tf
from tensorflow import keras
from keras.layers import TextVectorization
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_preprocess_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    texts, labels = [], []
    for line in lines:
        line = line.split('\n')[0]
        text, label = line.split(';')
        texts.append(text)
        labels.append(label)
    return texts, labels

train_texts, train_labels = load_preprocess_data('./DATA/train.txt')
train_df = pd.DataFrame({'texts': train_texts, 'target': train_labels})
lr = LabelEncoder()
train_df['target'] = lr.fit_transform(train_df['target'])
X_train = train_df['texts'].to_numpy()
y_train = train_df['target'].to_numpy()


# Building Tensorflow TextVectorization Layer
max_vocab_size = 30000
max_length = 45

text_vectorizer = TextVectorization(max_tokens=max_vocab_size, # how many words in the vocabulary
                                    output_sequence_length=max_length,
                                    output_mode='int')

# Fit the text_vectorizer to train_data
text_vectorizer.adapt(X_train)

weights = text_vectorizer.get_weights()
np.save("text_vectorization_weights.npy", weights)