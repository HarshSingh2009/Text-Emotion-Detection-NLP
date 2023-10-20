import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.layers import TextVectorization
import numpy as np


class NlpPredictionPipeline():
    def __init__(self) -> None:
        self.CLASS_NAMES = ['Anger', 'Fear', 'JOY', 'Love', 'Sadness', 'Surprise']

        # Loading model
        self.model = load_model('./Deep Learning Model/Text-Emotion_detection_BiLSTM_model.h5')
        
        # Building Tensorflow TextVectorization Layer
        max_vocab_size = 30000
        max_length = 45

        self.text_vectorization_layer = TextVectorization(max_tokens=max_vocab_size, # how many words in the vocabulary
                                            output_sequence_length=max_length,
                                            output_mode='int')
        self.text_vectorization_weights = np.load("text_vectorization_weights.npy", allow_pickle=True)
        # Set the loaded weights to the new TextVectorization layer
        self.text_vectorization_layer.set_weights(self.text_vectorization_weights)
    
    def predict_text_emotion(self, text):
        input_tensor = self.text_vectorization_layer([text])
        y_probs = self.model.predict(input_tensor)
        return tf.argmax(y_probs, axis=1), y_probs
        
