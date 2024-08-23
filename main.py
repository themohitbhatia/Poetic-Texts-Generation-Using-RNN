import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop


filepath = tf.keras.utils.get_file('shakespear_text.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()


# STEP 1: SET TEXT IN NUMERICAL FORMAT

text = text[300000:800000]

characters_set = sorted(set(text))


char_to_index = dict((c,i) for i,c in enumerate(characters_set))
index_to_char = dict((i,c) for i,c in enumerate(characters_set))


# STEP 2: PREDICT NEXT CHARACTERS

SEQ_LEN = 40
STEP_SIZE = 3


sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LEN, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LEN])
    next_characters.append(text[i+SEQ_LEN])


# STEP 3: CREATING NUMPY ARRAY
x = np.zeros((len(sentences), SEQ_LEN, len(characters_set)), dtype=bool)
y = np.zeros((len(sentences), len(characters_set)), dtype=bool)


for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1


# STEP 4: Bulding the Neural Network

model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LEN, len(characters_set))))
model.add(Dense(len(characters_set)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

model.fit(x, y, batch_size = 256, epochs=4)

model.save('textgenerator.model')


# STEP 5: PREDICTION

def sample(preds, temperature=0.1):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(length, temprature):
    start_index = random.randint(0, len(text) - SEQ_LEN - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LEN]
    generated += sentence

    for i in range(length):
        x = np.zeros((1,SEQ_LEN, len(characters_set)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1
        
        predictions = model.predict(x, verbose=0)[0]

        next_index = sample(predictions, temprature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    
    return generated


print('---------- 0.2 ----------')
print(generate_text(300, 0.2))

print('---------- 0.4 ----------')
print(generate_text(300, 0.2))

print('---------- 0.6 ----------')
print(generate_text(300, 0.2))

print('---------- 0.8 ----------')
print(generate_text(300, 0.2))

print('---------- 1 ----------')
print(generate_text(300, 0.2))